import dotenv from "dotenv"
dotenv.config({ override: true })
import express from "express"
import axios from "axios"
import crypto from "crypto"
import {
    initDatabase,
    upsertUser, saveAnalysis,
    getDashboardStats, getUserStats
} from './database.js'
import { createImageProcessor } from './image-processing.js'
import { IMAGE_ONLY_INSTRUCTION } from './line-interface.js'

const app = express()
app.use(express.json({
    limit: '10mb',
    verify: (req, res, buf) => {
        req.rawBody = buf  // เก็บ raw body สำหรับ LINE signature verification
    }
}))

// =====================
// Environment Variables (ไม่ hardcode API keys)
// =====================
const LINE_TOKEN = process.env.LINE_TOKEN || process.env.CHANNEL_ACCESS_TOKEN
const CHANNEL_SECRET = process.env.CHANNEL_SECRET
const YOLO_API_URL = process.env.YOLO_API_URL || "http://localhost:5002/predict"

if (!LINE_TOKEN) {
    console.error('❌ LINE_TOKEN or CHANNEL_ACCESS_TOKEN not found in .env')
    process.exit(1)
}

if (!CHANNEL_SECRET) {
    console.warn('⚠️  CHANNEL_SECRET not found - webhook signature verification disabled')
}

// =====================
// Simple Logger (timestamp + level)
// =====================
function log(level, ...args) {
    const timestamp = new Date().toISOString()
    const prefix = `[${timestamp}] [${level}]`
    if (level === 'ERROR') {
        console.error(prefix, ...args)
    } else if (level === 'WARN') {
        console.warn(prefix, ...args)
    } else {
        console.log(prefix, ...args)
    }
}

// =====================
// LINE Webhook Signature Verification
// =====================
function verifySignature(rawBody, signature) {
    if (!CHANNEL_SECRET) return true  // ข้ามถ้ายังไม่ได้ตั้ง CHANNEL_SECRET
    if (!signature || !rawBody) return false
    try {
        const hash = crypto
            .createHmac('sha256', CHANNEL_SECRET)
            .update(rawBody)
            .digest('base64')
        const hashBuffer = Buffer.from(hash)
        const signatureBuffer = Buffer.from(signature)
        if (hashBuffer.length !== signatureBuffer.length) return false
        return crypto.timingSafeEqual(hashBuffer, signatureBuffer)
    } catch {
        return false
    }
}

// =====================
// Rate Limiter (in-memory, ไม่ต้องติดตั้ง package เพิ่ม)
// =====================
const rateLimitMap = new Map()
const RATE_LIMIT_WINDOW = 60 * 1000  // 1 นาที
const RATE_LIMIT_MAX = 60  // สูงสุด 60 requests ต่อ IP ต่อนาที

function checkRateLimit(ip) {
    const now = Date.now()
    const entry = rateLimitMap.get(ip)

    if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW) {
        rateLimitMap.set(ip, { windowStart: now, count: 1 })
        return true
    }

    entry.count++
    return entry.count <= RATE_LIMIT_MAX
}

// ล้าง rate limit entries เก่าทุก 1 นาที
setInterval(() => {
    const now = Date.now()
    for (const [ip, entry] of rateLimitMap) {
        if (now - entry.windowStart > RATE_LIMIT_WINDOW) {
            rateLimitMap.delete(ip)
        }
    }
}, RATE_LIMIT_WINDOW)

// =====================
// LINE reply
// =====================
async function replyMessage(replyToken, messages) {
    try {
        await axios.post(
            "https://api.line.me/v2/bot/message/reply",
            { replyToken, messages },
            {
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${LINE_TOKEN}`,
                },
                timeout: 10000
            }
        )
    } catch (error) {
        console.error("Error replying to LINE")
    }
}

// =====================
// LINE push message
// =====================
async function pushMessage(userId, messages) {
    try {
        await axios.post(
            "https://api.line.me/v2/bot/message/push",
            { to: userId, messages },
            {
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${LINE_TOKEN}`,
                },
                timeout: 10000
            }
        )
    } catch (error) {
        console.error("Error pushing to LINE")
    }
}

// Every replyable non-image event uses the same instruction, including old postbacks.
async function handleEvent(event) {
    if (!event.replyToken) return
    const userId = event.source?.userId

    try {
        if (event.type !== 'message' || event.message?.type !== 'image') {
            await replyMessage(event.replyToken, [{ type: 'text', text: IMAGE_ONLY_INSTRUCTION }])
            return
        }

        // Keep user records for analysis-history foreign keys; never load chat memory.
        if (userId) {
            try {
                const profile = await axios.get(`https://api.line.me/v2/bot/profile/${userId}`, {
                    headers: { Authorization: `Bearer ${LINE_TOKEN}` },
                    timeout: 5000
                })
                await upsertUser(userId, profile.data?.displayName || null)
            } catch {
                try {
                    await upsertUser(userId)
                } catch {
                    console.error('User profile persistence failed')
                }
            }
        }

        await replyMessage(event.replyToken, [
            { type: 'text', text: '📷 ได้รับรูปแล้ว กำลังวิเคราะห์โรคข้าว' }
        ])
        processImage(event.message.id, userId).catch(() => {
            console.error('Image processing error')
        })
    } catch {
        console.error('Event handling error')
    }
}

// =====================
// Webhook (รองรับหลาย events)
// =====================
app.post("/webhook", async (req, res) => {
    // Rate Limiting
    const clientIp = req.headers['x-forwarded-for'] || req.ip
    if (!checkRateLimit(clientIp)) {
        log('WARN', '⚠️ Rate limit exceeded')
        return res.sendStatus(429)
    }

    // LINE Signature Verification
    const signature = req.headers['x-line-signature']
    if (!verifySignature(req.rawBody, signature)) {
        log('WARN', '⚠️ Invalid webhook signature')
        return res.sendStatus(403)
    }

    log('INFO', '📩 Webhook received')

    // ส่ง 200 ทันทีเพื่อไม่ให้ LINE timeout
    res.sendStatus(200)

    const events = req.body.events
    if (!events || events.length === 0) return

    // ประมวลผลทุก event พร้อมกัน
    await Promise.all(events.map(handleEvent))
})

// =====================
// ประมวลผลรูปภาพแยกต่างหาก
// =====================
const processImage = createImageProcessor({
    lineToken: LINE_TOKEN,
    yoloApiUrl: YOLO_API_URL,
    saveAnalysis,
    pushMessage
})

// =====================
// Health check
// =====================
app.get("/", (req, res) => {
    res.send("Rice image detector is running! ✅")
})

// =====================
// API: สถิติรวมของระบบ
// =====================
app.get("/api/stats", async (req, res) => {
    const stats = await getDashboardStats()
    if (!stats) {
        return res.status(500).json({ error: 'Database not available' })
    }
    res.json(stats)
})

// =====================
// API: ประวัติการใช้งานของผู้ใช้
// =====================
app.get("/api/users/:userId/history", async (req, res) => {
    const { userId } = req.params
    const userStats = await getUserStats(userId)
    if (!userStats) {
        return res.status(404).json({ error: 'User not found' })
    }
    res.json(userStats)
})

// (ลบ /models และ /test-yolo endpoints ออกเพื่อความปลอดภัย — ใช้ตอน dev เท่านั้น)

// =====================
// เริ่มต้น Database + Server
// =====================
;(async () => {
    try {
        await initDatabase()
        log('INFO', '✅ Database ready')
    } catch (error) {
        log('ERROR', '⚠️ Database failed to initialize — server will run without DB:', error.message)
    }

    const PORT = process.env.PORT || 3000
    app.listen(PORT, () => {
        log('INFO', `🚀 Webhook running on port ${PORT}`)
        log('INFO', `📍 Webhook URL: http://localhost:${PORT}/webhook`)
        log('INFO', '🤖 YOLO API configured')
        log('INFO', `🔐 Signature verification: ${CHANNEL_SECRET ? 'ENABLED' : 'DISABLED'}`)
        log('INFO', `📊 Rate limit: ${RATE_LIMIT_MAX} req/${RATE_LIMIT_WINDOW / 1000}s per IP`)
        log('INFO', `🗄️ Database: Supabase (PostgreSQL)`)
    })
})()

// =====================
// Graceful Shutdown
// =====================
process.on('SIGINT', () => {
    log('INFO', '🛑 Shutting down...')
    process.exit(0)
})

process.on('SIGTERM', () => {
    log('INFO', '🛑 Shutting down...')
    process.exit(0)
})

