// =====================================================
// database.js — SQLite Database Module
// จัดการข้อมูลผู้ใช้, ผลวิเคราะห์โรค, ประวัติแชท
// =====================================================

import Database from 'better-sqlite3'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// เก็บ DB ไว้ในโฟลเดอร์ data/
const DB_PATH = path.join(__dirname, 'data', 'rice_farmer.db')

let db = null

// =====================
// เริ่มต้น Database + สร้าง Tables
// =====================
export function initDatabase() {
    try {
        db = new Database(DB_PATH)

        // เปิด WAL mode เพื่อประสิทธิภาพที่ดีขึ้น (อ่าน-เขียนพร้อมกันได้)
        db.pragma('journal_mode = WAL')
        db.pragma('foreign_keys = ON')

        // สร้างตารางผู้ใช้ LINE
        db.exec(`
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_user_id TEXT UNIQUE NOT NULL,
                display_name TEXT,
                first_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_analyses INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0
            )
        `)

        // สร้างตารางผลการวิเคราะห์โรคข้าว
        db.exec(`
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_user_id TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT,
                advice TEXT,
                image_url TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (line_user_id) REFERENCES users(line_user_id)
            )
        `)

        // สร้างตารางประวัติการสนทนา
        db.exec(`
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                line_user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (line_user_id) REFERENCES users(line_user_id)
            )
        `)

        // สร้าง Index เพื่อความเร็วในการค้นหา
        db.exec(`
            CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(line_user_id);
            CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
            CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(line_user_id);
            CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_history(created_at);
        `)

        console.log('✅ Database initialized:', DB_PATH)
        return db
    } catch (error) {
        console.error('❌ Database initialization failed:', error.message)
        throw error
    }
}

// =====================
// ผู้ใช้ (Users)
// =====================

/**
 * สร้างหรืออัปเดตข้อมูลผู้ใช้
 * @param {string} lineUserId - LINE userId
 * @param {string|null} displayName - ชื่อผู้ใช้จาก LINE profile
 */
export function upsertUser(lineUserId, displayName = null) {
    if (!db) return null
    try {
        const stmt = db.prepare(`
            INSERT INTO users (line_user_id, display_name)
            VALUES (?, ?)
            ON CONFLICT(line_user_id) DO UPDATE SET
                display_name = COALESCE(?, display_name),
                last_active_at = CURRENT_TIMESTAMP
        `)
        stmt.run(lineUserId, displayName, displayName)
        return true
    } catch (error) {
        console.error('❌ upsertUser error:', error.message)
        return false
    }
}

/**
 * เพิ่มจำนวนข้อความของผู้ใช้
 */
export function incrementMessageCount(lineUserId) {
    if (!db) return
    try {
        db.prepare(`
            UPDATE users SET total_messages = total_messages + 1, last_active_at = CURRENT_TIMESTAMP
            WHERE line_user_id = ?
        `).run(lineUserId)
    } catch (error) {
        console.error('❌ incrementMessageCount error:', error.message)
    }
}

/**
 * เพิ่มจำนวนการวิเคราะห์ของผู้ใช้
 */
export function incrementAnalysisCount(lineUserId) {
    if (!db) return
    try {
        db.prepare(`
            UPDATE users SET total_analyses = total_analyses + 1, last_active_at = CURRENT_TIMESTAMP
            WHERE line_user_id = ?
        `).run(lineUserId)
    } catch (error) {
        console.error('❌ incrementAnalysisCount error:', error.message)
    }
}

// =====================
// ผลวิเคราะห์โรค (Analyses)
// =====================

/**
 * บันทึกผลการวิเคราะห์โรคข้าว
 * @param {string} lineUserId - LINE userId
 * @param {object} result - ผลวิเคราะห์ { disease, confidence, severity, advice, imageUrl }
 */
export function saveAnalysis(lineUserId, result) {
    if (!db) return null
    try {
        const stmt = db.prepare(`
            INSERT INTO analyses (line_user_id, disease, confidence, severity, advice, image_url)
            VALUES (?, ?, ?, ?, ?, ?)
        `)
        const info = stmt.run(
            lineUserId,
            result.disease,
            result.confidence,
            result.severity || null,
            result.advice || null,
            result.imageUrl || null
        )
        incrementAnalysisCount(lineUserId)
        return info.lastInsertRowid
    } catch (error) {
        console.error('❌ saveAnalysis error:', error.message)
        return null
    }
}

/**
 * ดึงประวัติการวิเคราะห์โรคของผู้ใช้
 * @param {string} lineUserId - LINE userId
 * @param {number} limit - จำนวนรายการที่ต้องการ (default: 10)
 */
export function getAnalysisHistory(lineUserId, limit = 10) {
    if (!db) return []
    try {
        return db.prepare(`
            SELECT id, disease, confidence, severity, advice, image_url, created_at
            FROM analyses
            WHERE line_user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        `).all(lineUserId, limit)
    } catch (error) {
        console.error('❌ getAnalysisHistory error:', error.message)
        return []
    }
}

// =====================
// ประวัติแชท (Chat History)
// =====================

/**
 * บันทึกข้อความแชท
 * @param {string} lineUserId - LINE userId
 * @param {string} role - 'user' หรือ 'bot'
 * @param {string} message - ข้อความ
 */
export function saveChatMessage(lineUserId, role, message) {
    if (!db) return null
    try {
        const stmt = db.prepare(`
            INSERT INTO chat_history (line_user_id, role, message)
            VALUES (?, ?, ?)
        `)
        const info = stmt.run(lineUserId, role, message)

        if (role === 'user') {
            incrementMessageCount(lineUserId)
        }

        return info.lastInsertRowid
    } catch (error) {
        console.error('❌ saveChatMessage error:', error.message)
        return null
    }
}

/**
 * ดึงประวัติแชทของผู้ใช้
 * @param {string} lineUserId - LINE userId
 * @param {number} limit - จำนวนรายการที่ต้องการ (default: 20)
 */
export function getChatHistory(lineUserId, limit = 20) {
    if (!db) return []
    try {
        return db.prepare(`
            SELECT id, role, message, created_at
            FROM chat_history
            WHERE line_user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        `).all(lineUserId, limit)
    } catch (error) {
        console.error('❌ getChatHistory error:', error.message)
        return []
    }
}

// =====================
// สถิติรวม (Dashboard Stats)
// =====================

/**
 * ดึงสถิติรวมของระบบ
 */
export function getDashboardStats() {
    if (!db) return null
    try {
        const totalUsers = db.prepare('SELECT COUNT(*) as count FROM users').get().count
        const totalAnalyses = db.prepare('SELECT COUNT(*) as count FROM analyses').get().count
        const totalMessages = db.prepare('SELECT COUNT(*) as count FROM chat_history WHERE role = ?').get('user').count

        // โรคที่พบบ่อยที่สุด 5 อันดับ
        const topDiseases = db.prepare(`
            SELECT disease, COUNT(*) as count, ROUND(AVG(confidence), 2) as avg_confidence
            FROM analyses
            WHERE disease != 'ไม่พบโรค (Healthy)' AND disease != 'ไม่พบโรค'
            GROUP BY disease
            ORDER BY count DESC
            LIMIT 5
        `).all()

        // ผู้ใช้ที่ active ล่าสุด 5 คน
        const recentUsers = db.prepare(`
            SELECT line_user_id, display_name, total_analyses, total_messages, last_active_at
            FROM users
            ORDER BY last_active_at DESC
            LIMIT 5
        `).all()

        // วิเคราะห์ล่าสุด 10 ครั้ง
        const recentAnalyses = db.prepare(`
            SELECT a.disease, a.confidence, a.severity, a.created_at, u.display_name
            FROM analyses a
            LEFT JOIN users u ON a.line_user_id = u.line_user_id
            ORDER BY a.created_at DESC
            LIMIT 10
        `).all()

        return {
            total_users: totalUsers,
            total_analyses: totalAnalyses,
            total_messages: totalMessages,
            top_diseases: topDiseases,
            recent_users: recentUsers,
            recent_analyses: recentAnalyses
        }
    } catch (error) {
        console.error('❌ getDashboardStats error:', error.message)
        return null
    }
}

/**
 * ดึงสถิติของผู้ใช้คนเดียว
 * @param {string} lineUserId - LINE userId
 */
export function getUserStats(lineUserId) {
    if (!db) return null
    try {
        const user = db.prepare(`
            SELECT * FROM users WHERE line_user_id = ?
        `).get(lineUserId)

        if (!user) return null

        const analyses = getAnalysisHistory(lineUserId, 10)
        const recentChats = getChatHistory(lineUserId, 20)

        return {
            user,
            analyses,
            recent_chats: recentChats
        }
    } catch (error) {
        console.error('❌ getUserStats error:', error.message)
        return null
    }
}

// =====================
// ปิด Database (สำหรับ graceful shutdown)
// =====================
export function closeDatabase() {
    if (db) {
        db.close()
        console.log('🔒 Database closed')
    }
}
