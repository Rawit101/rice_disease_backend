import axios from 'axios'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import FormData from 'form-data'

// Names in best.pt, plus existing aliases used by detector API consumers.
export const DISEASE_NAMES = Object.freeze({
    bacterial_blight: 'โรคขอบใบแห้ง',
    bacterial_leaf_blight: 'โรคขอบใบแห้ง',
    brown_spot: 'โรคใบจุดสีน้ำตาล',
    rice_blast: 'โรคไหม้',
    leaf_blast: 'โรคใบไหม้',
    narrow_brown_spot: 'โรคใบขีดสีน้ำตาล',
    narrow_brown_leaf_spot: 'โรคใบขีดสีน้ำตาล',
    false_smut: 'โรคดอกกระถิน',
    dirty_seed: 'โรคเมล็ดด่าง',
    sheath_rot: 'โรคกาบใบเน่า',
    stem_rot: 'โรคลำต้นเน่า',
    red_stripe: 'โรคใบแถบแดง'
})

export const UNKNOWN_MESSAGE = 'ยังระบุโรคไม่ได้ กรุณาถ่ายภาพใบข้าวให้ชัดแล้วส่งอีกครั้ง'
export const SERVICE_ERROR_MESSAGE = 'ระบบตรวจภาพไม่พร้อมใช้งาน กรุณาลองใหม่ภายหลัง'

export function acceptanceThreshold(value = '0.25') {
    const threshold = Number(value)
    if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
        throw new Error('YOLO_CONFIDENCE_THRESHOLD must be greater than 0 and at most 1')
    }
    return threshold
}

export function selectDetection(payload, threshold) {
    if (!payload || payload.error || payload.success === false || !Array.isArray(payload.predictions)) {
        throw new Error('Invalid detector response')
    }

    let top = null
    for (const prediction of payload.predictions) {
        const key = typeof prediction?.class === 'string' ? prediction.class.toLowerCase() : ''
        const confidence = prediction?.confidence
        if (!Object.hasOwn(DISEASE_NAMES, key) || typeof confidence !== 'number'
            || !Number.isFinite(confidence) || confidence < threshold || confidence > 1) continue
        if (!top || confidence > top.confidence) {
            top = { disease: DISEASE_NAMES[key], confidence }
        }
    }
    return top ? { status: 'detected', ...top } : { status: 'unknown' }
}

export async function analyzeRiceDiseaseWithYOLO(imagePath, {
    httpClient = axios,
    yoloApiUrl = process.env.YOLO_API_URL || 'http://localhost:5002/predict',
    threshold = acceptanceThreshold(process.env.YOLO_CONFIDENCE_THRESHOLD)
} = {}) {
    const stream = fs.createReadStream(imagePath)
    try {
        const formData = new FormData()
        formData.append('image', stream, { filename: 'rice.jpg' })
        formData.append('include_annotated_image', 'false')
        // The inference filter and LINE acceptance check use the same threshold.
        formData.append('confidence_threshold', String(threshold))
        const response = await httpClient.post(yoloApiUrl, formData, {
            headers: formData.getHeaders(),
            timeout: 90000, // Preserve support for detector cold starts.
            maxContentLength: 50 * 1024 * 1024,
            maxBodyLength: 50 * 1024 * 1024
        })
        return selectDetection(response.data, threshold)
    } finally {
        stream.destroy()
    }
}

export async function getImageFromLine(messageId, { httpClient = axios, lineToken, tempDir = os.tmpdir() }) {
    const filePath = path.join(tempDir, `line_${randomUUID()}.jpg`)
    try {
        const response = await httpClient.get(`https://api-data.line.me/v2/bot/message/${messageId}/content`, {
            headers: { Authorization: `Bearer ${lineToken}` },
            responseType: 'arraybuffer',
            timeout: 15000
        })
        fs.writeFileSync(filePath, response.data)
        return filePath
    } catch (error) {
        fs.rmSync(filePath, { force: true })
        throw error
    }
}

// Explicit dependencies keep this flow independent of AI/RAG and storage uploads.
export function createImageProcessor({
    lineToken, yoloApiUrl, saveAnalysis, pushMessage,
    httpClient = axios, tempDir = os.tmpdir(),
    threshold = acceptanceThreshold(process.env.YOLO_CONFIDENCE_THRESHOLD),
    logger = console
}) {
    return async function processImage(messageId, userId) {
        let imagePath = null
        let result = { status: 'error' }
        try {
            imagePath = await getImageFromLine(messageId, { httpClient, lineToken, tempDir })
            result = await analyzeRiceDiseaseWithYOLO(imagePath, { httpClient, yoloApiUrl, threshold })
        } catch {
            // Never log HTTP error objects: they can contain tokens or image bytes.
            logger.error('Image retrieval or detection failed')
        } finally {
            if (imagePath) {
                try {
                    fs.rmSync(imagePath, { force: true })
                } catch {
                    logger.error('Temporary image cleanup failed')
                }
            }
        }

        const text = result.status === 'detected'
            ? `🌾 ผลตรวจเบื้องต้น: ${result.disease}`
            : result.status === 'unknown' ? UNKNOWN_MESSAGE : SERVICE_ERROR_MESSAGE

        // Exactly one delivery attempt, before persistence; never send a second error result.
        try {
            await pushMessage(userId, [{ type: 'text', text }])
        } catch {
            logger.error('Image result delivery failed')
        }

        if (result.status === 'detected') {
            try {
                await saveAnalysis(userId, {
                    disease: result.disease,
                    confidence: result.confidence,
                    severity: null,
                    advice: null,
                    imageUrl: null
                })
            } catch {
                logger.error('Analysis history persistence failed')
            }
        }
    }
}
