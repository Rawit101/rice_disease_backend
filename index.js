import "dotenv/config"
import express from "express"
import axios from "axios"
import fs from "fs"
import os from "os"
import path from "path"
import crypto from "crypto"
import FormData from "form-data"
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
app.use(express.json({
    limit: '10mb',
    verify: (req, res, buf) => {
        req.rawBody = buf  // เก็บ raw body สำหรับ LINE signature verification
    }
}))

// =====================
// โฟลเดอร์เก็บรูปผลลัพธ์ (serve เป็น static files)
// =====================
const RESULTS_DIR = path.join(__dirname, 'public', 'results')
if (!fs.existsSync(RESULTS_DIR)) {
    fs.mkdirSync(RESULTS_DIR, { recursive: true })
    console.log('📁 Created results directory:', RESULTS_DIR)
}
app.use('/public', express.static(path.join(__dirname, 'public')))

// =====================
// Environment Variables (ไม่ hardcode API keys)
// =====================
const LINE_TOKEN = process.env.LINE_TOKEN || process.env.CHANNEL_ACCESS_TOKEN
const CHANNEL_SECRET = process.env.CHANNEL_SECRET
const GEMINI_API_KEY = process.env.GEMINI_API_KEY
const YOLO_API_URL = process.env.YOLO_API_URL || "http://localhost:5000/predict"
let BASE_URL = process.env.BASE_URL || ""  // auto-detect จาก ngrok webhook request

if (!LINE_TOKEN) {
    console.error('❌ LINE_TOKEN or CHANNEL_ACCESS_TOKEN not found in .env')
    process.exit(1)
}

if (!CHANNEL_SECRET) {
    console.warn('⚠️  CHANNEL_SECRET not found - webhook signature verification disabled')
}

if (!GEMINI_API_KEY) {
    console.warn('⚠️  GEMINI_API_KEY not found - will use fallback advice only')
} else {
    console.log('✅ GEMINI_API_KEY loaded')
}

// =====================
// Local Knowledge Base (scraped from Rice Knowledge Bank)
// =====================
const KNOWLEDGE_FILE = path.join(__dirname, 'data', 'rice_knowledge.json')
const MAX_KNOWLEDGE_ENTRIES = 3
const MAX_KNOWLEDGE_CHARS_PER_ENTRY = 2200

function normalizeText(text = '') {
    return String(text)
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .trim()
}

function loadRiceKnowledge() {
    try {
        if (!fs.existsSync(KNOWLEDGE_FILE)) {
            log('WARN', `Local knowledge file not found: ${KNOWLEDGE_FILE}`)
            return []
        }
        const payload = JSON.parse(fs.readFileSync(KNOWLEDGE_FILE, 'utf8'))
        const entries = Array.isArray(payload.entries) ? payload.entries : []
        log('INFO', `Local rice knowledge loaded: ${entries.length} entries`)
        return entries
            .filter(entry => entry?.title && entry?.content)
            .map(entry => ({
                title: entry.title,
                source: entry.source,
                content: entry.content,
                searchable: normalizeText(`${entry.title} ${entry.content}`)
            }))
    } catch (error) {
        log('WARN', 'Failed to load local rice knowledge:', error.message)
        return []
    }
}

const riceKnowledge = loadRiceKnowledge()

function extractSearchTerms(text = '') {
    return normalizeText(text)
        .split(/[^\p{L}\p{N}]+/u)
        .map(term => term.trim())
        .filter(term => term.length >= 2)
        .filter(term => !['ครับ', 'ค่ะ', 'คะ', 'หน่อย', 'ยังไง', 'อะไร', 'วิธี', 'แก้'].includes(term))
}

function findKnowledgeContext(question, diseaseHint = '') {
    if (riceKnowledge.length === 0) return ''

    const query = normalizeText(`${question} ${diseaseHint}`)
    const terms = extractSearchTerms(query)

    const ranked = riceKnowledge
        .map(entry => {
            const title = normalizeText(entry.title)
            const titleWithoutPrefix = title.replace(/^โรค/, '')
            let score = 0

            if (query.includes(title)) score += 120
            if (titleWithoutPrefix && query.includes(titleWithoutPrefix)) score += 90

            for (const term of terms) {
                if (title.includes(term)) score += 25
                if (entry.searchable.includes(term)) score += Math.min(term.length, 12)
            }

            return { ...entry, score }
        })
        .filter(entry => entry.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, MAX_KNOWLEDGE_ENTRIES)

    if (ranked.length === 0) return ''

    return ranked
        .map(entry => {
            const content = entry.content.length > MAX_KNOWLEDGE_CHARS_PER_ENTRY
                ? `${entry.content.slice(0, MAX_KNOWLEDGE_CHARS_PER_ENTRY)}...`
                : entry.content
            return `หัวข้อ: ${entry.title}\nแหล่งข้อมูล: ${entry.source}\nเนื้อหา:\n${content}`
        })
        .join('\n\n---\n\n')
}

function buildLocalKnowledgeFallback(question, diseaseHint = '') {
    const context = findKnowledgeContext(question, diseaseHint)
    if (!context) {
        return "ขออภัยครับ ตอนนี้ Gemini มีผู้ใช้งานหนาแน่นมาก เลยตอบแบบ AI ไม่ได้ชั่วคราว\nลองถามชื่อโรคข้าวให้ชัดขึ้น เช่น โรคไหม้ โรคขอบใบแห้ง หรือโรคกาบใบแห้ง แล้วผมจะดึงข้อมูลจากไฟล์กรมการข้าวในเครื่องมาตอบให้ครับ"
    }

    const firstBlock = context.split('\n\n---\n\n')[0]
    const title = firstBlock.match(/^หัวข้อ: (.+)$/m)?.[1] || 'ข้อมูลโรคข้าว'
    const source = firstBlock.match(/^แหล่งข้อมูล: (.+)$/m)?.[1] || 'Rice Knowledge Bank'
    const content = firstBlock
        .replace(/^หัวข้อ: .+$/m, '')
        .replace(/^แหล่งข้อมูล: .+$/m, '')
        .replace(/^เนื้อหา:\s*/m, '')
        .replace(/\n{3,}/g, '\n\n')
        .trim()

    const shortContent = content.length > 900 ? `${content.slice(0, 900)}...` : content
    return `ตอนนี้ Gemini มีผู้ใช้งานหนาแน่นมาก ผมตอบจากไฟล์ข้อมูลกรมการข้าวในเครื่องให้ก่อนนะครับ\n\n${title}\n\n${shortContent}\n\nแหล่งข้อมูล: ${source}`
}

// =====================
// Session เก็บข้อมูลผลวิเคราะห์ล่าสุด + ประวัติการสนทนาของแต่ละ user
// =====================
const userSessions = new Map()
const SESSION_TIMEOUT = 30 * 60 * 1000  // 30 นาที
const MAX_SESSIONS = 10000  // จำกัดจำนวน session ป้องกัน memory overflow
const MAX_HISTORY = 6  // เก็บประวัติสนทนาล่าสุด 6 รอบ (3 คู่ user-bot)

function getSession(userId) {
    const session = userSessions.get(userId)
    if (!session) return null
    // เช็คว่า session หมดอายุหรือยัง
    if (Date.now() - session.timestamp > SESSION_TIMEOUT) {
        userSessions.delete(userId)
        return null
    }
    return session
}

function saveSession(userId, data) {
    // ป้องกัน memory overflow: ถ้า session เกิน MAX_SESSIONS ลบตัวเก่าสุด
    if (userSessions.size >= MAX_SESSIONS && !userSessions.has(userId)) {
        const oldestKey = userSessions.keys().next().value
        userSessions.delete(oldestKey)
    }
    userSessions.set(userId, {
        ...data,
        timestamp: Date.now()
    })
}

// =====================
// ลบ Markdown formatting (LINE แสดง markdown ไม่ได้)
// =====================
function stripMarkdown(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, '$1')   // **bold** → bold
        .replace(/\*(.+?)\*/g, '$1')        // *italic* → italic
        .replace(/```(?:\w*\n)?([\s\S]*?)```/g, '$1')  // code block → เก็บเนื้อหา ลบแค่ ```
        .replace(/`(.+?)`/g, '$1')          // `code` → code
        .replace(/^#{1,6}\s+/gm, '')        // # headers → ลบ #
        .replace(/^[-*]\s+/gm, '- ')        // normalize bullet points
        .trim()
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

// ล้าง session หมดอายุทุก 5 นาที
setInterval(() => {
    const now = Date.now()
    let cleaned = 0
    for (const [userId, session] of userSessions) {
        if (now - session.timestamp > SESSION_TIMEOUT) {
            userSessions.delete(userId)
            cleaned++
        }
    }
    if (cleaned > 0) {
        log('INFO', `🗑️ Cleaned ${cleaned} expired sessions (remaining: ${userSessions.size})`)
    }
}, 5 * 60 * 1000)

// =====================
// Gemini API Helper (ลด code ซ้ำ)
// =====================
const GEMINI_MODELS = [
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash'
]

async function callGemini(prompt, options = {}) {
    const { timeout = 30000 } = options
    let rateLimitCount = 0
    const MAX_RATE_LIMIT_RETRIES = 2  // จำกัดการ retry เมื่อโดน rate limit

    for (const modelName of GEMINI_MODELS) {
        try {
            const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${GEMINI_API_KEY}`

            const body = {
                contents: [
                    {
                        parts: [{ text: prompt }]
                    }
                ]
            }

            const res = await axios.post(url, body, {
                headers: { "Content-Type": "application/json" },
                timeout
            })

            log('INFO', `✅ Using model: ${modelName}`)

            // ป้องกัน Gemini ตอบว่างหรือถูก safety filter block
            const candidates = res.data?.candidates
            if (!candidates || candidates.length === 0) {
                log('WARN', `⚠️ Model ${modelName} returned empty candidates`)
                continue
            }

            const rawText = candidates[0]?.content?.parts?.[0]?.text
            if (!rawText) {
                log('WARN', `⚠️ Model ${modelName} returned empty text (possibly blocked by safety filter)`)
                continue
            }

            return stripMarkdown(rawText)
        } catch (error) {
            log('ERROR', `❌ Model ${modelName} failed:`, error.response?.data?.error?.message || error.message)

            // ถ้าเป็น 429 (rate limit) — จำกัดจำนวน retry
            if (error.response?.status === 429) {
                rateLimitCount++
                if (rateLimitCount >= MAX_RATE_LIMIT_RETRIES) {
                    log('ERROR', `❌ Rate limited ${rateLimitCount} ครั้ง — หยุดเพื่อประหยัด quota`)
                    break
                }
                const waitTime = 2 ** rateLimitCount  // exponential backoff: 2, 4 วินาที
                log('WARN', `⏳ Rate limited, waiting ${waitTime}s... (${rateLimitCount}/${MAX_RATE_LIMIT_RETRIES})`)
                await new Promise(resolve => setTimeout(resolve, waitTime * 1000))
            }
            continue
        }
    }

    throw new Error('All Gemini models failed')
}

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
        console.error("Error replying:", error.response?.data || error.message)
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
        console.error("Error pushing:", error.response?.data || error.message)
    }
}

// =====================
// ตรวจจับทักทาย/กล่าวลา (ไม่ต้องเรียก Gemini — ประหยัด API)
// =====================
const GREETING_PATTERNS = /^(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ|ดีจ้า|ดี$|hello|hi|hey|สวัสดีครับ|สวัสดีค่ะ|หวัดดีครับ|หวัดดีค่ะ)/i
const FAREWELL_PATTERNS = /^(บาย|ลาก่อน|ไว้เจอกัน|bye|ลาก่อนครับ|ลาก่อนค่ะ|ไปก่อน)/i
const THANKS_PATTERNS = /^(ขอบคุณ|ขอบใจ|thanks|thank you|ขอบคุณครับ|ขอบคุณค่ะ|ขอบคุณมาก)/i

function getLocalReply(text) {
    const trimmed = text.trim()
    if (GREETING_PATTERNS.test(trimmed)) {
        return "🌾 สวัสดีครับ! ผม \"ไอนาย\" ผู้เชี่ยวชาญเรื่องข้าว\n\n📸 ส่งรูปใบข้าวมาวิเคราะห์โรคได้เลย\n💬 หรือถามเรื่องข้าวมาได้เลยครับ!"
    }
    if (FAREWELL_PATTERNS.test(trimmed)) {
        return "👋 ลาก่อนครับ! ถ้ามีปัญหาเรื่องข้าว กลับมาถามได้ตลอดนะครับ 🌾"
    }
    if (THANKS_PATTERNS.test(trimmed)) {
        return "😊 ยินดีครับ! ถ้ามีอะไรอยากถามเพิ่มเติมเรื่องข้าว บอกได้เลยนะครับ 🌾"
    }
    return null  // ไม่ match → ส่งต่อให้ Gemini
}

// =====================
// Chat เฉพาะเรื่องโรคข้าว (มี context จากผลวิเคราะห์ + ประวัติสนทนา)
// =====================
async function askChatbot(text, userId) {
    try {
        // ตรวจจับทักทาย/กล่าวลา/ขอบคุณ ตอบทันทีไม่ต้องเรียก AI
        const localReply = getLocalReply(text)
        if (localReply) {
            // บันทึกลง history เพื่อให้ bot จำได้
            if (userId) {
                const session = getSession(userId) || {}
                const history = session.chatHistory || []
                history.push({ role: 'user', text })
                history.push({ role: 'bot', text: localReply })
                saveSession(userId, { ...session, chatHistory: history.slice(-MAX_HISTORY * 2) })
            }
            return localReply
        }

        // ดึง session ของ user เพื่อใช้เป็น context
        const session = userId ? getSession(userId) : null
        const knowledgeContext = findKnowledgeContext(text, session?.disease || '')

        let contextInfo = ''
        if (session) {
            // ข้อมูลผลวิเคราะห์โรค
            if (session.disease) {
                contextInfo += `\n\nข้อมูลจากการวิเคราะห์ล่าสุดของ user นี้:\n- โรคที่พบ: ${session.disease}\n- ความมั่นใจ: ${session.confidence}%\n- ระดับความรุนแรง: ${session.severity}\n- คำแนะนำที่ให้ไปแล้ว (ห้ามตอบซ้ำเด็ดขาด): ${session.advice}\n\nกฎสำคัญ:\n- ถ้า user ขอวิธีแก้เพิ่มเติม ขอคำแนะนำเพิ่ม หรือถามซ้ำ → ต้องตอบด้วยข้อมูลใหม่ที่ยังไม่เคยให้ เช่น ยาตัวอื่น สารชีวภัณฑ์ วิธีธรรมชาติ การป้องกันระยะยาว พันธุ์ข้าวต้านทาน การจัดการดิน/น้ำ ฯลฯ\n- ห้ามพูดซ้ำคำแนะนำข้างต้นเด็ดขาด ต้องเป็นข้อมูลใหม่ทั้งหมด\n- ถ้า user ถามรายละเอียดเรื่องยา/สาร ให้ลงลึกมากขึ้น เช่น ยี่ห้อ วิธีผสม ช่วงเวลาพ่น ข้อควรระวัง`
            }

            // ประวัติสนทนา
            if (session.chatHistory && session.chatHistory.length > 0) {
                const historyText = session.chatHistory
                    .map(h => h.role === 'user' ? `User: ${h.text}` : `ไอนาย: ${h.text}`)
                    .join('\n')
                contextInfo += `\n\nประวัติสนทนาล่าสุด:\n${historyText}\n\nกฎ: ใช้ประวัตินี้เป็น context ในการตอบ อ้างอิงสิ่งที่คุยกันไปแล้ว ไม่ต้องทักทายซ้ำถ้าเคยทักทายไปแล้ว`
            }
        }

        const prompt = `คุณคือผู้เชี่ยวชาญด้านโรคข้าวและการดูแลข้าวโดยเฉพาะ ชื่อ "ไอนาย" ทำหน้าที่ให้คำปรึกษาเกษตรกรชาวนาไทย อ้างอิงองค์ความรู้จากกรมการข้าว (rkb.ricethailand.go.th) ห้ามตอบคำถามที่ไม่เกี่ยวข้องกับข้าวเด็ดขาด

ข้อมูลอ้างอิงจากไฟล์ local ที่ scrape จาก Rice Knowledge Bank:
${knowledgeContext || 'ไม่พบข้อมูลอ้างอิงที่ตรงกับคำถามนี้ในไฟล์ local ให้ตอบเฉพาะความรู้ทั่วไปเรื่องข้าวอย่างระมัดระวัง และอย่าอ้างว่าเป็นข้อมูลจากเว็บถ้าไม่มีในข้อความอ้างอิง'}

กฎการตอบ:
1. ตอบเฉพาะเรื่องโรคข้าว การปลูกข้าว ศัตรูพืช ปุ๋ย ยา การดูแลข้าวเท่านั้น
2. ถ้าไม่เกี่ยวกับข้าว → ตอบว่า "ขออภัยครับ ผมตอบได้เฉพาะเรื่องข้าวเท่านั้นนะครับ ลองถามเรื่องข้าวมาได้เลย 🌾"
3. ห้ามใช้ ** หรือ markdown
4. ถ้า user ถามสั้นๆ เช่น "วิธีแก้" "ทำยังไง" "ใช้ยาอะไร" ให้ตอบโดยอ้างอิงโรคที่วิเคราะห์ล่าสุด

รูปแบบการตอบ:
- ใช้ emoji นำหน้าหัวข้อ เช่น 🌾 📌 💊 ⚠️
- เว้นบรรทัดระหว่างหัวข้อ
- ใช้ - นำหน้า bullet point
- สั้นกระชับ ไม่เกิน 120 คำ
- ภาษาง่ายๆ เป็นกันเอง
- ระบุชื่อสาร อัตราส่วน วิธีใช้ ให้ชาวนาใช้ได้จริง${contextInfo}

คำถาม: ${text}`

        const fullText = await callGemini(prompt)

        // บันทึกประวัติสนทนา
        if (userId) {
            const currentSession = getSession(userId) || {}
            const history = currentSession.chatHistory || []
            history.push({ role: 'user', text })
            history.push({ role: 'bot', text: fullText })
            // เก็บแค่ล่าสุด MAX_HISTORY คู่
            saveSession(userId, { ...currentSession, chatHistory: history.slice(-MAX_HISTORY * 2) })
        }

        // จำกัดความยาวไม่เกิน 4500 ตัวอักษร (LINE รองรับ 5000 เผื่อ buffer)
        if (fullText.length > 4500) {
            // ตัดที่จุดจบประโยคที่ใกล้สุด
            const cutText = fullText.substring(0, 4500)
            const lastBreak = Math.max(cutText.lastIndexOf('\n'), cutText.lastIndexOf(' '))
            return cutText.substring(0, lastBreak > 0 ? lastBreak : 4500) + '...'
        }

        return fullText
    } catch (error) {
        log('ERROR', 'Chatbot error:', error.message)
        if (error.message === 'All Gemini models failed') {
            return buildLocalKnowledgeFallback(text, userId ? getSession(userId)?.disease : '')
        }
        return "❌ ขออภัยครับ เกิดข้อผิดพลาด\nกรุณาลองใหม่อีกครั้งนะครับ"
    }
}

// =====================
// แนะนำโรคข้าว (ใช้ helper function)
// =====================
async function askDiseaseAdvice(disease, severity) {
    const fallbackAdvice = {
        // Key ใช้ทั้งชื่อเต็ม (จาก diseaseMap) และชื่อสั้น เพื่อให้ match ได้เสมอ
        'โรคใบขีดสีน้ำตาล (Narrow Brown Leaf Spot)': `📌 อาการ
แผลขีดสีน้ำตาลแคบๆ ยาวตามเส้นใบ ใบเหลืองแห้ง

💊 วิธีแก้
- ใช้พันธุ์ต้านทาน เช่น กข6 หรือ กข15 ที่ทนโรคได้ดี
- พ่นสาร Propiconazole อัตรา 15-20 ซีซี ต่อน้ำ 20 ลิตร พ่นทุก 7-10 วัน
- ลดปุ๋ยไนโตรเจน ใส่ไม่เกิน 10 กก./ไร่ เพราะไนโตรเจนสูงทำให้โรคลุกลาม

⚠️ ระวัง: ระบาดหนักช่วงข้าวออกรวง ควรพ่นป้องกันก่อนออกรวง`,

        'โรคใบไหม้ (Leaf Blast)': `📌 อาการ
จุดสีน้ำตาลรูปเพชรบนใบ คอรวงหัก รวงเมล็ดลีบ

💊 วิธีแก้
- พ่น Tricyclazole อัตรา 10-15 กรัม ต่อน้ำ 20 ลิตร พ่นทุก 7 วัน
- ตัดใบที่เป็นโรคทิ้ง แล้วเผาทำลายนอกแปลง ป้องกันเชื้อแพร่
- ระบายน้ำออก อย่าให้น้ำขังนาน เพราะความชื้นสูงทำให้โรคลุกลามเร็ว

⚠️ ระวัง: แพร่เร็วมากตอนอากาศชื้น มีหมอก ควรรีบรักษาทันทีที่พบ`,

        'โรคไหม้ (Rice Blast)': `📌 อาการ
จุดสีน้ำตาลรูปเพชรบนใบ คอรวงหัก รวงเมล็ดลีบ

💊 วิธีแก้
- พ่น Tricyclazole อัตรา 10-15 กรัม ต่อน้ำ 20 ลิตร พ่นทุก 7 วัน
- ตัดใบที่เป็นโรคทิ้ง แล้วเผาทำลายนอกแปลง ป้องกันเชื้อแพร่
- ระบายน้ำออก อย่าให้น้ำขังนาน เพราะความชื้นสูงทำให้โรคลุกลามเร็ว

⚠️ ระวัง: แพร่เร็วมากตอนอากาศชื้น มีหมอก ควรรีบรักษาทันทีที่พบ`,

        'โรคใบจุดสีน้ำตาล (Brown Spot)': `📌 อาการ
จุดน้ำตาลกลมรีกระจายบนใบ ใบแห้ง เมล็ดลีบ ผลผลิตลด

💊 วิธีแก้
- พ่นสาร Mancozeb อัตรา 40-50 กรัม ต่อน้ำ 20 ลิตร พ่นทุก 7-10 วัน
- ใส่ปุ๋ยให้สมดุล โดยเฉพาะโพแทสเซียม (K) ช่วยให้ข้าวแข็งแรงต้านโรค
- ระบายน้ำดี อย่าปล่อยให้น้ำขังในแปลงนานเกินไป

⚠️ ระวัง: มักเกิดในนาที่ดินขาดธาตุอาหาร ควรตรวจดินก่อนปลูก`,

        'โรคขอบใบแห้ง (Bacterial Leaf Blight)': `📌 อาการ
ขอบใบเหลืองแห้งลุกลามเข้าใน มีเมือกสีขาวบนใบตอนเช้า

💊 วิธีแก้
- ใช้สารป้องกันแบคทีเรีย เช่น ทองแดงไฮดรอกไซด์ อัตรา 30-40 กรัม ต่อน้ำ 20 ลิตร
- ลดปุ๋ยไนโตรเจน ใส่ไม่เกิน 8 กก./ไร่ เพราะไนโตรเจนสูงทำให้ข้าวอ่อนแอ
- ถอนต้นที่เป็นโรครุนแรงทิ้ง แล้วเผาทำลาย ป้องกันแพร่ไปต้นอื่น

⚠️ ระวัง: แพร่ทางน้ำและลม ห้ามเดินผ่านแปลงเปียกเพราะจะพาเชื้อไปด้วย`,

        'ไม่พบโรค (Healthy)': `✅ ข้าวแข็งแรงดี!

📋 คำแนะนำ
- ดูแลตามปกติ รดน้ำให้สม่ำเสมอ
- ตรวจใบข้าวทุก 3-5 วัน เพื่อจับอาการแต่เนิ่นๆ
- ใส่ปุ๋ยตามระยะ อย่าใส่มากเกินไป

📸 พบอาการผิดปกติ ส่งรูปมาใหม่ได้เลยนะครับ`
    }

    // Fallback lookup: ลอง match ชื่อเต็มก่อน ถ้าไม่เจอ ลอง match บางส่วน
    function findFallback(diseaseName) {
        // 1. ลอง exact match
        if (fallbackAdvice[diseaseName]) return fallbackAdvice[diseaseName]
        // 2. ลอง match บางส่วน (เช่น ชื่อโรคมีอยู่ใน key)
        for (const key of Object.keys(fallbackAdvice)) {
            if (key.includes(diseaseName) || diseaseName.includes(key)) {
                return fallbackAdvice[key]
            }
        }
        // 3. ไม่เจอ → ใช้ default
        return fallbackAdvice['ไม่พบโรค (Healthy)']
    }

    const prompt = `คุณคือผู้เชี่ยวชาญด้านโรคข้าว
ตรวจพบ: ${disease}
ระดับ: ${severity}

ตอบตาม format นี้เท่านั้น ห้ามใช้ ** หรือ markdown:

📌 อาการ
(อธิบายสั้นๆ 1-2 บรรทัด)

💊 วิธีแก้
- ข้อ 1 (ระบุชื่อสาร/วิธีการ + อัตราส่วนหรือวิธีใช้สั้นๆ)
- ข้อ 2 (ระบุชื่อสาร/วิธีการ + อัตราส่วนหรือวิธีใช้สั้นๆ)
- ข้อ 3 (ระบุชื่อสาร/วิธีการ + อัตราส่วนหรือวิธีใช้สั้นๆ)

⚠️ ระวัง: (1-2 ประโยค อธิบายให้ชัดเจน)

รวมไม่เกิน 120 คำ ภาษาง่ายๆ เป็นกันเอง ให้ข้อมูลที่ชาวนาใช้ได้จริง`

    try {
        return await callGemini(prompt)
    } catch (error) {
        log('ERROR', 'Disease advice error:', error.message)
        log('INFO', '📝 Using fallback advice')
        return findFallback(disease)
    }
}

// =====================
// วิเคราะห์โรคด้วย YOLO
// =====================
async function analyzeRiceDiseaseWithYOLO(imagePath) {
    try {
        console.log('Analyzing image with YOLO:', imagePath)

        const formData = new FormData()
        formData.append('image', fs.createReadStream(imagePath))

        const response = await axios.post(YOLO_API_URL, formData, {
            headers: {
                ...formData.getHeaders()
            },
            timeout: 30000,
            maxContentLength: 50 * 1024 * 1024,
            maxBodyLength: 50 * 1024 * 1024
        })

        console.log('YOLO Response predictions:', response.data.predictions?.length || 0)

        if (response.data && response.data.predictions && response.data.predictions.length > 0) {
            const topPrediction = response.data.predictions[0]

            const diseaseMap = {
                'brown_spot': 'โรคใบจุดสีน้ำตาล (Brown Spot)',
                'leaf_blast': 'โรคใบไหม้ (Leaf Blast)',
                'rice_blast': 'โรคไหม้ (Rice Blast)',
                'bacterial_leaf_blight': 'โรคขอบใบแห้ง (Bacterial Leaf Blight)',
                'narrow_brown_leaf_spot': 'โรคใบขีดสีน้ำตาล (Narrow Brown Leaf Spot)',
                'healthy': 'ไม่พบโรค (Healthy)',
                // === สำหรับ model อนาคต (ยังไม่มีใน best.pt ปัจจุบัน) ===
                'hispa': 'โรคหนอนชอนใบ (Hispa)',
                'dead_heart': 'โรคหนอนกอ (Dead Heart)',
                'tungro': 'โรคใบสีส้ม (Tungro)'
            }

            // ใช้ lowercase matching เพื่อรองรับทุกรูปแบบชื่อ class เช่น Rice_Blast, rice_blast, RICE_BLAST
            const classKey = topPrediction.class.toLowerCase()
            const diseaseName = diseaseMap[classKey] || topPrediction.class
            const confidence = topPrediction.confidence || 0

            let severity = 'ต่ำ'
            if (confidence > 0.7) severity = 'สูง'
            else if (confidence > 0.5) severity = 'ปานกลาง'

            return {
                disease: diseaseName,
                confidence: confidence,
                severity: severity,
                raw: topPrediction,
                annotated_image: response.data.annotated_image || null
            }
        }

        return {
            disease: "ไม่พบโรค",
            confidence: 0,
            severity: "ไม่ทราบ",
            annotated_image: null
        }

    } catch (error) {
        console.error("YOLO analysis error:", error.response?.data || error.message)

        return {
            disease: "ไม่สามารถวิเคราะห์ได้",
            confidence: 0,
            severity: "ไม่ทราบ",
            error: error.message,
            annotated_image: null
        }
    }
}

// =====================
// ดึงรูปจาก LINE
// =====================
async function getImageFromLine(messageId) {
    const url = `https://api-data.line.me/v2/bot/message/${messageId}/content`

    try {
        const response = await axios.get(url, {
            headers: {
                Authorization: `Bearer ${LINE_TOKEN}`,
            },
            responseType: "arraybuffer",
            timeout: 15000
        })

        const tempDir = os.tmpdir()
        const filePath = path.join(tempDir, `line_${Date.now()}_${messageId}.jpg`)

        fs.writeFileSync(filePath, response.data)
        console.log('Image saved:', filePath)
        return filePath
    } catch (error) {
        console.error("Error downloading image:", error.message)
        throw error
    }
}

// =====================
// ประมวลผล event เดี่ยว
// =====================
// =====================
// จัดการ Postback จาก Rich Menu
// =====================
async function handlePostback(replyToken, data, userId) {
    const params = new URLSearchParams(data)
    const action = params.get('action')

    switch (action) {
        // ===== ปุ่ม: โรคข้าวทั้งหมดในภาคตะวันออกเฉียงเหนือ =====
        case 'diseases_northeast':
            await replyMessage(replyToken, [{
                type: "text",
                text: `🗺️ โรคข้าวที่พบบ่อยในภาคตะวันออกเฉียงเหนือ
━━━━━━━━━━━━━━

🔴 1. โรคไหม้ (Rice Blast)
- จุดสีน้ำตาลรูปเพชรบนใบ คอรวงหัก
- พบมากช่วงฝนชุก อากาศเย็น

🟤 2. โรคใบจุดสีน้ำตาล (Brown Spot)
- จุดน้ำตาลกลมรีกระจายบนใบ
- พบในดินขาดธาตุอาหาร

🟡 3. โรคใบขีดสีน้ำตาล (Narrow Brown Leaf Spot)
- แผลขีดสีน้ำตาลแคบยาวตามเส้นใบ
- ระบาดช่วงข้าวออกรวง

🟠 4. โรคขอบใบแห้ง (Bacterial Leaf Blight)
- ขอบใบเหลืองแห้ง มีเมือกบนใบ
- แพร่ทางน้ำและลม

🟢 5. โรคใบสีส้ม (Tungro)
- ใบเปลี่ยนสีเหลืองส้ม ต้นเตี้ยแคระ
- เกิดจากเพลี้ยจักจั่นสีเขียว

📸 ส่งรูปใบข้าวมาวิเคราะห์โรคได้เลยครับ!`
            }])
            break


        // (องค์ความรู้เรื่องข้าว → เปิดเว็บ https://rkb.ricethailand.go.th โดยตรง ไม่ผ่าน postback)


        // ===== ปุ่ม: ความรู้คู่ชาวนา =====
        case 'farmer_tips':
            await replyMessage(replyToken, [{
                type: "text",
                text: `🌾 ความรู้คู่ชาวนา
━━━━━━━━━━━━━━

🛡️ การป้องกันโรค
- เลือกพันธุ์ต้านทานโรค เช่น กข6, กข15
- ไม่ปลูกข้าวแน่นเกินไป เว้นระยะให้อากาศถ่ายเท
- ตรวจแปลงทุก 3-5 วัน เพื่อจับโรคแต่เนิ่นๆ

🐛 การป้องกันแมลงศัตรูพืช
- ใช้กับดักแสงไฟ ดักแมลงตอนกลางคืน
- ปล่อยแมลงศัตรูธรรมชาติ เช่น แมลงปอ มวนเขียว
- พ่นสารชีวภัณฑ์ เช่น บิวเวอเรีย เมตาไรเซียม

📅 ปฏิทินชาวนา
- พ.ค.-มิ.ย.: เตรียมดิน หว่านกล้า
- ก.ค.-ส.ค.: ปักดำ ใส่ปุ๋ย
- ก.ย.-ต.ค.: ดูแลระยะออกรวง
- พ.ย.-ธ.ค.: เก็บเกี่ยว

📸 ส่งรูปใบข้าวมาวิเคราะห์โรคได้เลย!`
            }])
            break

        default:
            await replyMessage(replyToken, [{
                type: "text",
                text: "🌾 สวัสดีครับ! ผม \"ไอนาย\" ผู้เชี่ยวชาญเรื่องข้าว\n📸 ส่งรูปใบข้าวมาวิเคราะห์โรคได้เลยครับ!"
            }])
    }
}

async function handleEvent(event) {
    if (!event.replyToken) return

    const replyToken = event.replyToken

    try {
        // ===== Postback จาก Rich Menu =====
        if (event.type === "postback") {
            const userId = event.source.userId
            console.log(`📩 Postback received: ${event.postback.data}`)
            await handlePostback(replyToken, event.postback.data, userId)
            return
        }

        // ===== ข้อความ =====
        if (event.message?.type === "text") {
            const userId = event.source.userId
            const text = event.message.text

            // จัดการข้อความจากปุ่ม Rich Menu "อัปโหลดรูปภาพโรคข้าว"
            if (text === "📷 อัปโหลดรูปภาพโรคข้าว") {
                await replyMessage(replyToken, [{
                    type: "text",
                    text: "📷 ส่งรูปใบข้าวมาได้เลยครับ!\n\n💡 เคล็ดลับถ่ายรูปให้ได้ผลดี:\n- ถ่ายใกล้ๆ ใบที่มีอาการ\n- ใช้แสงธรรมชาติ ไม่มืดเกินไป\n- ให้ภาพชัด ไม่เบลอ\n- ถ่ายตรงๆ ไม่เอียงมาก\n\n🤖 AI จะวิเคราะห์โรคให้ทันทีครับ!"
                }])
                return
            }

            const reply = await askChatbot(text, userId)

            await replyMessage(replyToken, [
                {
                    type: "text",
                    text: reply,
                },
            ])
        }

        // ===== รูปภาพ =====
        if (event.message?.type === "image") {
            const userId = event.source.userId

            await replyMessage(replyToken, [
                { type: "text", text: "📷 ได้รับรูปแล้ว กำลังวิเคราะห์โรคข้าวด้วย AI..." },
            ])

            processImage(event.message.id, userId).catch(err => {
                console.error('Image processing error:', err)
            })
        }
    } catch (error) {
        console.error("Event handling error:", error)
    }
}

// =====================
// Webhook (รองรับหลาย events)
// =====================
app.post("/webhook", async (req, res) => {
    // Rate Limiting
    const clientIp = req.headers['x-forwarded-for'] || req.ip
    if (!checkRateLimit(clientIp)) {
        log('WARN', `⚠️ Rate limit exceeded for IP: ${clientIp}`)
        return res.sendStatus(429)
    }

    // LINE Signature Verification
    const signature = req.headers['x-line-signature']
    if (!verifySignature(req.rawBody, signature)) {
        log('WARN', `⚠️ Invalid signature from IP: ${clientIp}`)
        return res.sendStatus(403)
    }

    log('INFO', `📩 Webhook received from IP: ${clientIp}`)

    // Auto-detect BASE_URL จาก ngrok request header (ทำครั้งเดียว)
    if (!BASE_URL) {
        const proto = req.headers['x-forwarded-proto'] || req.protocol
        const host = req.headers['x-forwarded-host'] || req.headers.host
        if (host) {
            BASE_URL = `${proto}://${host}`
            log('INFO', `🌐 Auto-detected BASE_URL: ${BASE_URL}`)
        }
    }

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
async function processImage(messageId, userId) {
    let imagePath = null
    let annotatedPath = null

    try {
        imagePath = await getImageFromLine(messageId)
        console.log("Image saved:", imagePath)

        const result = await analyzeRiceDiseaseWithYOLO(imagePath)
        console.log("YOLO result:", result.disease, result.confidence)

        if (result.error) {
            await pushMessage(userId, [
                { type: "text", text: "❌ ขออภัย YOLO API ไม่สามารถทำงานได้ในขณะนี้\nกรุณาตรวจสอบว่า YOLO server กำลังทำงานอยู่" }
            ])
            return
        }

        if (result.confidence < 0.3) {
            await pushMessage(userId, [
                {
                    type: "text",
                    text: `⚠️ วิเคราะห์ไม่ชัดเจน\n\n🔍 พบ: ${result.disease}\n📊 ความมั่นใจ: ${Math.round(result.confidence * 100)}%\n\n📸 ลองถ่ายรูปใหม่นะครับ:\n- ถ่ายใกล้ขึ้น\n- ใช้แสงให้ดี\n- ให้ภาพชัด\n- ลองอีกมุม`
                }
            ])
            return
        }

        const advice = await askDiseaseAdvice(result.disease, result.severity)
        console.log("Advice generated")

        // บันทึกผลวิเคราะห์ลง session เพื่อให้ user ถามต่อได้
        saveSession(userId, {
            disease: result.disease,
            confidence: Math.round(result.confidence * 100),
            severity: result.severity,
            advice: advice
        })
        console.log(`💾 Session saved for user: ${userId}`)

        let confidenceEmoji = '✅'
        if (result.confidence < 0.5) confidenceEmoji = '⚠️'

        let message = `🌾 ผลวิเคราะห์โรคข้าว\n━━━━━━━━━━━━━━\n\n🔍 พบ: ${result.disease}\n📊 ความมั่นใจ: ${Math.round(result.confidence * 100)}% ${confidenceEmoji}\n🔴 ความรุนแรง: ${result.severity}\n\n━━━━━━━━━━━━━━\n${advice}`

        if (message.length > 2000) {
            message = message.substring(0, 1997) + '...'
        }

        // ส่งรูปที่วาด bounding box กลับไปให้ user
        const messages = []

        if (result.annotated_image && BASE_URL) {
            try {
                // บันทึกรูป annotated เป็นไฟล์
                const filename = `result_${Date.now()}_${crypto.randomBytes(4).toString('hex')}.jpg`
                annotatedPath = path.join(RESULTS_DIR, filename)
                const imageBuffer = Buffer.from(result.annotated_image, 'base64')
                fs.writeFileSync(annotatedPath, imageBuffer)
                console.log('🖼️ Annotated image saved:', annotatedPath)

                const imageUrl = `${BASE_URL}/public/results/${filename}`

                messages.push({
                    type: "image",
                    originalContentUrl: imageUrl,
                    previewImageUrl: imageUrl
                })
                console.log('📤 Image URL:', imageUrl)

                // ลบรูปเก่าที่เกิน 30 นาที
                cleanupOldResults()
            } catch (imgError) {
                console.error('Error saving annotated image:', imgError.message)
            }
        }

        messages.push({
            type: "text",
            text: message,
        })

        await pushMessage(userId, messages)
    } catch (error) {
        console.error('Process image error:', error)

        await pushMessage(userId, [
            { type: "text", text: "❌ ขออภัย เกิดข้อผิดพลาดในการวิเคราะห์รูปภาพ กรุณาลองใหม่อีกครั้ง" }
        ])
    } finally {
        // ลบไฟล์ชั่วคราวจาก LINE
        if (imagePath) {
            try {
                fs.unlinkSync(imagePath)
                console.log("Temp file deleted")
            } catch (e) {
                console.log("Cannot delete temp file:", e.message)
            }
        }
    }
}

// =====================
// ลบรูปผลลัพธ์เก่าที่เกิน 30 นาที
// =====================
function cleanupOldResults() {
    try {
        const files = fs.readdirSync(RESULTS_DIR)
        const now = Date.now()
        let deleted = 0

        for (const file of files) {
            const filePath = path.join(RESULTS_DIR, file)
            const stats = fs.statSync(filePath)
            // ลบไฟล์ที่เก่ากว่า 30 นาที
            if (now - stats.mtimeMs > 30 * 60 * 1000) {
                fs.unlinkSync(filePath)
                deleted++
            }
        }

        if (deleted > 0) {
            console.log(`🗑️ Cleaned up ${deleted} old result images`)
        }
    } catch (e) {
        console.log('Cleanup error:', e.message)
    }
}

// =====================
// Health check
// =====================
app.get("/", (req, res) => {
    res.send("LINE Bot is running! ✅")
})

// (ลบ /models และ /test-yolo endpoints ออกเพื่อความปลอดภัย — ใช้ตอน dev เท่านั้น)

// =====================
const PORT = process.env.PORT || 3000
app.listen(PORT, () => {
    log('INFO', `🚀 Webhook running on port ${PORT}`)
    log('INFO', `📍 Webhook URL: http://localhost:${PORT}/webhook`)
    log('INFO', `🤖 YOLO API URL: ${YOLO_API_URL}`)
    log('INFO', `🔐 Signature verification: ${CHANNEL_SECRET ? 'ENABLED' : 'DISABLED'}`)
    log('INFO', `📊 Rate limit: ${RATE_LIMIT_MAX} req/${RATE_LIMIT_WINDOW / 1000}s per IP`)
})
