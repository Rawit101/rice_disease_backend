import test from 'node:test'
import assert from 'node:assert/strict'
import { Writable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { loadModule } from './module-fixture.js'

const instruction = 'กรุณาส่งรูปใบข้าวเพื่อตรวจโรค ระบบนี้รองรับเฉพาะรูปภาพ'

async function boot({ persistenceFailure = false, legacyConfig = false } = {}) {
    const routes = new Map(), calls = [], replies = [], pushes = [], records = [], unexpected = []
    let started = false
    const app = {
        use() {}, get() {}, post(route, handler) { routes.set(route, handler) },
        listen(port, callback) { started = true; callback() }
    }
    const express = Object.assign(() => app, { json: () => () => {} })
    let complete
    const done = new Promise(resolve => { complete = resolve })
    const axios = {
        async get(url) {
            if (url.includes('/profile/')) { calls.push('profile'); return { data: { displayName: 'test' } } }
            if (!url.startsWith('https://api-data.line.me/')) { unexpected.push(url); throw new Error('Unexpected request') }
            calls.push('download')
            return { data: Buffer.from('mock image') }
        },
        async post(url, body) {
            if (url.endsWith('/message/reply')) { calls.push('reply'); replies.push(body); return { data: {} } }
            if (url.endsWith('/message/push')) { calls.push('push'); pushes.push(body); return { data: {} } }
            if (url === 'https://detector.test/predict') {
                calls.push('detect')
                await pipeline(body, new Writable({ write(chunk, encoding, cb) { cb() } }))
                return { data: { success: true, predictions: [{ class: 'Brown_Spot', confidence: 0.8 }], annotated_image: 'ignored' } }
            }
            unexpected.push(url)
            throw new Error('Unexpected external request')
        }
    }
    const forbiddenMemory = () => { calls.push('forbidden-memory'); throw new Error('Chat memory must not be used') }
    await loadModule('index.js', {
        express: { default: express }, axios: { default: axios },
        dotenv: { default: { config() {} } }
    }, {
        process: {
            env: { LINE_TOKEN: 'fake-token', YOLO_API_URL: 'https://detector.test/predict',
                ...(legacyConfig ? { MAXPLUS_API_KEY: 'unused', GEMINI_API_KEY: 'unused',
                    MAXPLUS_BASE_URL: 'https://ai.test', RAG_API_URL: 'https://rag.test/search' } : {}) },
            on() {}, exit() { throw new Error('Unexpected exit') }
        },
        setInterval() {},
        // Simulate stale state from the previous implementation. No path may read it.
        userSessions: new Map([['test-user', { disease: 'old diagnosis', advice: 'old advice', chatHistory: ['old chat'] }]]),
        testDatabase: {
            async initDatabase() { if (persistenceFailure) throw new Error('db unavailable') },
            async upsertUser() { if (persistenceFailure) throw new Error('db unavailable') },
            async saveAnalysis(user, result) {
                records.push(result)
                complete()
                if (persistenceFailure) throw new Error('db unavailable')
            },
            saveChatMessage: forbiddenMemory, getChatHistory: forbiddenMemory,
            async getDashboardStats() {}, getUserStats: forbiddenMemory
        }
    })
    async function send(event) {
        await routes.get('/webhook')({
            headers: { host: 'localhost' }, ip: 'test-ip', protocol: 'http',
            body: { events: [event] }
        }, { sendStatus(status) { calls.push(`http-${status}`) } })
        assert.deepEqual(unexpected, [], 'No AI, RAG, embedding, storage or other unexpected requests')
        assert.equal(calls.includes('forbidden-memory'), false)
    }
    return { send, done, calls, replies, pushes, records, started }
}

for (const persistenceFailure of [false, true]) {
    test(`startup and image detection without AI keys/RAG (DB failure=${persistenceFailure})`, async () => {
        const f = await boot({ persistenceFailure })
        assert.equal(f.started, true)
        await f.send({ type: 'message', replyToken: 'reply', source: { userId: 'test-user' }, message: { type: 'image', id: 'test-message' } })
        await f.done
        assert.equal(f.calls[0], 'http-200')
        assert.ok(f.calls.indexOf('reply') < f.calls.indexOf('detect'))
        assert.equal(f.replies.length, 1)
        assert.equal(f.replies[0].messages[0].text, '📷 ได้รับรูปแล้ว กำลังวิเคราะห์โรคข้าว')
        assert.equal(f.pushes.length, 1)
        assert.equal(f.pushes[0].messages.length, 1)
        assert.equal(f.pushes[0].messages[0].text, '🌾 ผลตรวจเบื้องต้น: โรคใบจุดสีน้ำตาล')
        assert.equal(f.records[0].advice, null)
    })
}

const event = message => ({ type: 'message', replyToken: 'reply', source: { userId: 'test-user' }, message })
for (const text of ['สวัสดี', 'วิธีรักษาโรคไหม้', 'ใช้ยาอะไร', 'ขอบคุณ', 'พยากรณ์อากาศ', '📷 อัปโหลดรูปภาพโรคข้าว', 'help']) {
    test(`text always receives image-only instruction: ${text}`, async () => {
        const f = await boot({ legacyConfig: true })
        await f.send(event({ type: 'text', text }))
        assert.deepEqual(f.calls, ['http-200', 'reply'])
        assert.equal(f.replies.length, 1)
        assert.equal(f.replies[0].messages.length, 1)
        assert.equal(f.replies[0].messages[0].text, instruction)
        assert.equal(f.pushes.length, 0)
    })
}

for (const type of ['video', 'audio', 'file', 'sticker', 'location', 'future-type']) {
    test(`unsupported ${type} receives image-only instruction`, async () => {
        const f = await boot()
        await f.send(event({ type }))
        assert.deepEqual(f.calls, ['http-200', 'reply'])
        assert.equal(f.replies[0].messages[0].text, instruction)
    })
}

for (const data of ['action=diseases_northeast', 'action=farmer_tips', 'action=weather', 'action=chat', 'action=upload', 'unknown']) {
    test(`legacy postback cannot trigger removed features: ${data}`, async () => {
        const f = await boot({ legacyConfig: true })
        await f.send({ type: 'postback', replyToken: 'reply', postback: { data } })
        assert.deepEqual(f.calls, ['http-200', 'reply'])
        assert.equal(f.replies[0].messages[0].text, instruction)
    })
}

for (const type of ['follow', 'join']) {
    test(`${type} welcome remains image-only`, async () => {
        const f = await boot()
        await f.send({ type, replyToken: 'reply' })
        assert.equal(f.replies[0].messages[0].text, instruction)
    })
}

test('follow-up after detection ignores both current and old diagnosis/history', async () => {
    const f = await boot({ legacyConfig: true })
    await f.send(event({ type: 'image', id: 'test-message' }))
    await f.done
    await f.send(event({ type: 'text', text: 'ต้องรักษาอย่างไร' }))
    assert.equal(f.replies.at(-1).messages[0].text, instruction)
    assert.equal(f.pushes.length, 1)
    assert.equal(f.records.length, 1)
})

test('events without a reply token are safely acknowledged without outgoing requests', async () => {
    const f = await boot()
    await f.send({ type: 'unfollow' })
    assert.deepEqual(f.calls, ['http-200'])
})
