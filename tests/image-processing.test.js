import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Writable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import {
    acceptanceThreshold, selectDetection, createImageProcessor,
    DISEASE_NAMES, UNKNOWN_MESSAGE, SERVICE_ERROR_MESSAGE
} from '../image-processing.js'

async function readForm(form) {
    const chunks = []
    await pipeline(form, new Writable({ write(chunk, encoding, callback) {
        chunks.push(Buffer.from(chunk))
        callback()
    } }))
    return Buffer.concat(chunks).toString()
}

function fixture(t, { predictions = [{ class: 'Rice_Blast', confidence: 0.9 }], data,
    downloadFailure = false, detectorFailure = false, earlyDetectorFailure = false, persistenceFailure = false,
    deliveryFailure = false, threshold = 0.25 } = {}) {
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rice-test-'))
    t.after(() => fs.rmSync(tempDir, { recursive: true, force: true }))
    const calls = [], messages = [], records = [], logs = []
    const processImage = createImageProcessor({
        lineToken: 'test-token', yoloApiUrl: 'https://detector.test/predict', tempDir, threshold,
        saveAnalysis: async (user, record) => {
            calls.push('persist')
            records.push(record)
            if (persistenceFailure) throw new Error('secret-user-id')
        },
        pushMessage: async (user, value) => {
            calls.push('push')
            messages.push(value)
            // Files are already gone and the result does not await persistence.
            assert.deepEqual(fs.readdirSync(tempDir), [])
            if (deliveryFailure) throw new Error('secret-token')
        },
        logger: { error: message => logs.push(message) },
        httpClient: {
            get: async (url, options) => {
                calls.push('download')
                assert.equal(url, 'https://api-data.line.me/v2/bot/message/message-id/content')
                assert.equal(options.headers.Authorization, 'Bearer test-token')
                if (downloadFailure) throw new Error('secret-user-id')
                return { data: Buffer.from('mock image contents') }
            },
            post: async (url, form) => {
                calls.push('detect')
                // Any AI, RAG, or storage request fails this assertion.
                assert.equal(url, 'https://detector.test/predict')
                if (earlyDetectorFailure) throw new Error('secret-token')
                const body = await readForm(form)
                assert.match(body, /name="include_annotated_image"\r\n\r\nfalse/)
                assert.ok(body.includes(`name="confidence_threshold"\r\n\r\n${threshold}`))
                if (detectorFailure) throw new Error('secret-token')
                return { data: data ?? { success: true, predictions, annotated_image: 'must-never-upload' } }
            }
        }
    })
    return { processImage, tempDir, calls, messages, records, logs }
}

test('supported result is one exact Thai text; no AI, RAG, upload or image delivery', async t => {
    const f = fixture(t)
    await f.processImage('message-id', 'user-id')
    assert.deepEqual(f.messages, [[{ type: 'text', text: '🌾 ผลตรวจเบื้องต้น: โรคไหม้' }]])
    assert.deepEqual(f.calls, ['download', 'detect', 'push', 'persist'])
    assert.deepEqual(f.records, [{ disease: 'โรคไหม้', confidence: 0.9, severity: null, advice: null, imageUrl: null }])
    assert.deepEqual(fs.readdirSync(f.tempDir), [])
})

for (const [name, predictions] of [
    ['empty', []], ['low confidence', [{ class: 'Rice_Blast', confidence: 0.2499 }]],
    ['unknown', [{ class: 'new_class', confidence: 0.99 }]],
    ['healthy', [{ class: 'healthy', confidence: 0.99 }]],
    ['invalid confidence', [{ class: 'Rice_Blast', confidence: '0.9' }]],
    ['missing class', [{ confidence: 0.99 }]],
    ['prototype key', [{ class: 'constructor', confidence: 0.99 }]]
]) {
    test(`${name} produces the exact fallback`, async t => {
        const f = fixture(t, { predictions })
        await f.processImage('message-id', 'user-id')
        assert.deepEqual(f.messages, [[{ type: 'text', text: UNKNOWN_MESSAGE }]])
        assert.equal(f.records.length, 0)
        assert.deepEqual(fs.readdirSync(f.tempDir), [])
    })
}

for (const options of [{ detectorFailure: true }, { earlyDetectorFailure: true }, { downloadFailure: true },
    { data: { success: false, predictions: [] } }, { data: { error: 'unavailable' } }, { data: {} }]) {
    test(`service/retrieval failure: ${JSON.stringify(options)}`, async t => {
        const f = fixture(t, options)
        await f.processImage('message-id', 'user-id')
        assert.deepEqual(f.messages, [[{ type: 'text', text: SERVICE_ERROR_MESSAGE }]])
        assert.deepEqual(fs.readdirSync(f.tempDir), [])
        assert.ok(!f.logs.join(' ').includes('secret-'))
    })
}

test('persistence failure cannot block delivery or cause a duplicate', async t => {
    const f = fixture(t, { persistenceFailure: true })
    await f.processImage('message-id', 'user-id')
    assert.equal(f.messages.length, 1)
    assert.ok(f.calls.indexOf('push') < f.calls.indexOf('persist'))
    assert.ok(!f.logs.join(' ').includes('secret-'))
})

test('delivery failure makes one attempt and still saves history', async t => {
    const f = fixture(t, { deliveryFailure: true })
    await f.processImage('message-id', 'user-id')
    assert.equal(f.messages.length, 1)
    assert.equal(f.records.length, 1)
    assert.deepEqual(fs.readdirSync(f.tempDir), [])
})

test('partial temporary file is removed when writing a downloaded image fails', async t => {
    const f = fixture(t)
    const writeFileSync = fs.writeFileSync
    t.mock.method(fs, 'writeFileSync', (filename, data) => {
        writeFileSync(filename, data.subarray(0, 2))
        throw new Error('disk write failure')
    })
    await f.processImage('message-id', 'user-id')
    assert.deepEqual(fs.readdirSync(f.tempDir), [])
    assert.deepEqual(f.messages, [[{ type: 'text', text: SERVICE_ERROR_MESSAGE }]])
})

test('highest-confidence supported class wins even if response is unsorted or unknown is first', () => {
    assert.deepEqual(selectDetection({ predictions: [
        { class: 'unknown', confidence: 1 }, { class: 'Brown_Spot', confidence: 0.4 },
        { class: 'Rice_Blast', confidence: 0.85 }, { class: 'Stem_Rot', confidence: 0.6 }
    ] }, 0.25), { status: 'detected', disease: 'โรคไหม้', confidence: 0.85 })
})

test('all supported names map explicitly to Thai', () => {
    for (const [name, disease] of Object.entries(DISEASE_NAMES)) {
        assert.match(disease, /^โรค[\u0E00-\u0E7F]+$/)
        assert.equal(selectDetection({ predictions: [{ class: name.toUpperCase(), confidence: 0.9 }] }, 0.25).disease, disease)
    }
})

test('default, configurable boundary and invalid thresholds', async t => {
    assert.equal(acceptanceThreshold(), 0.25)
    for (const value of ['', 0, -1, 1.01, 'NaN', 'Infinity']) assert.throws(() => acceptanceThreshold(value))
    const f = fixture(t, { threshold: 0.4, predictions: [{ class: 'Rice_Blast', confidence: 0.4 }] })
    await f.processImage('message-id', 'user-id')
    assert.equal(f.messages[0][0].text, '🌾 ผลตรวจเบื้องต้น: โรคไหม้')
})
