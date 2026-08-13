// =====================================================
// database.js — Supabase (PostgreSQL) Database Module
// จัดการข้อมูลผู้ใช้, ผลวิเคราะห์โรค, ประวัติแชท
// =====================================================

import { createClient } from '@supabase/supabase-js'

let supabase = null

// =====================
// เริ่มต้น Database (เชื่อมต่อ Supabase)
// =====================
export async function initDatabase() {
    try {
        const supabaseUrl = process.env.SUPABASE_URL
        const supabaseKey = process.env.SUPABASE_ANON_KEY

        if (!supabaseUrl || !supabaseKey) {
            throw new Error('SUPABASE_URL and SUPABASE_ANON_KEY are required in .env')
        }

        supabase = createClient(supabaseUrl, supabaseKey)

        // ทดสอบ connection โดย query ตาราง users
        const { error } = await supabase.from('users').select('id').limit(1)
        if (error) throw error

        console.log('✅ Supabase connected:', supabaseUrl)
        return supabase
    } catch (error) {
        console.error('❌ Supabase initialization failed:', error.message)
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
export async function upsertUser(lineUserId, displayName = null) {
    if (!supabase) return null
    try {
        const { error } = await supabase
            .from('users')
            .upsert(
                {
                    line_user_id: lineUserId,
                    display_name: displayName,
                    last_active_at: new Date().toISOString()
                },
                {
                    onConflict: 'line_user_id',
                    ignoreDuplicates: false
                }
            )

        if (error) throw error
        return true
    } catch (error) {
        console.error('❌ upsertUser error:', error.message)
        return false
    }
}

/**
 * เพิ่มจำนวนข้อความของผู้ใช้
 */
async function incrementMessageCount(lineUserId) {
    if (!supabase) return
    try {
        const { error } = await supabase.rpc('increment_message_count', {
            user_line_id: lineUserId
        })
        if (error) throw error
    } catch (error) {
        console.error('❌ incrementMessageCount error:', error.message)
    }
}

/**
 * เพิ่มจำนวนการวิเคราะห์ของผู้ใช้
 */
async function incrementAnalysisCount(lineUserId) {
    if (!supabase) return
    try {
        const { error } = await supabase.rpc('increment_analysis_count', {
            user_line_id: lineUserId
        })
        if (error) throw error
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
export async function saveAnalysis(lineUserId, result) {
    if (!supabase) return null
    try {
        const { data, error } = await supabase
            .from('analyses')
            .insert({
                line_user_id: lineUserId,
                disease: result.disease,
                confidence: result.confidence,
                severity: result.severity || null,
                advice: result.advice || null,
                image_url: result.imageUrl || null
            })
            .select('id')
            .single()

        if (error) throw error

        // เพิ่มจำนวนการวิเคราะห์ (ไม่ block ถ้า fail)
        incrementAnalysisCount(lineUserId)

        return data.id
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
export async function getAnalysisHistory(lineUserId, limit = 10) {
    if (!supabase) return []
    try {
        const { data, error } = await supabase
            .from('analyses')
            .select('id, disease, confidence, severity, advice, image_url, created_at')
            .eq('line_user_id', lineUserId)
            .order('created_at', { ascending: false })
            .limit(limit)

        if (error) throw error
        return data || []
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
export async function saveChatMessage(lineUserId, role, message) {
    if (!supabase) return null
    try {
        const { data, error } = await supabase
            .from('chat_history')
            .insert({
                line_user_id: lineUserId,
                role,
                message
            })
            .select('id')
            .single()

        if (error) throw error

        // เพิ่มจำนวนข้อความ (ไม่ block ถ้า fail)
        if (role === 'user') {
            incrementMessageCount(lineUserId)
        }

        return data.id
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
export async function getChatHistory(lineUserId, limit = 20) {
    if (!supabase) return []
    try {
        const { data, error } = await supabase
            .from('chat_history')
            .select('id, role, message, created_at')
            .eq('line_user_id', lineUserId)
            .order('created_at', { ascending: false })
            .limit(limit)

        if (error) throw error
        return data || []
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
export async function getDashboardStats() {
    if (!supabase) return null
    try {
        const { data, error } = await supabase.rpc('get_dashboard_stats')
        if (error) throw error
        return data
    } catch (error) {
        console.error('❌ getDashboardStats error:', error.message)
        return null
    }
}

/**
 * ดึงสถิติของผู้ใช้คนเดียว
 * @param {string} lineUserId - LINE userId
 */
export async function getUserStats(lineUserId) {
    if (!supabase) return null
    try {
        const { data: user, error: userError } = await supabase
            .from('users')
            .select('*')
            .eq('line_user_id', lineUserId)
            .single()

        if (userError || !user) return null

        const analyses = await getAnalysisHistory(lineUserId, 10)
        const recentChats = await getChatHistory(lineUserId, 20)

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
// อัปโหลดรูปภาพขึ้น Supabase Storage
// =====================

/**
 * อัปโหลดรูปผลวิเคราะห์ขึ้น Supabase Storage
 * @param {Buffer} imageBuffer - ไฟล์รูปภาพ (Buffer)
 * @param {string} filename - ชื่อไฟล์ เช่น result_xxx.jpg
 * @returns {string|null} Public URL ของรูป หรือ null ถ้า error
 */
export async function uploadImage(imageBuffer, filename) {
    if (!supabase) return null
    try {
        const filePath = `results/${filename}`

        const { error: uploadError } = await supabase.storage
            .from('rice-disease-analysis-results')
            .upload(filePath, imageBuffer, {
                contentType: 'image/jpeg',
                upsert: false
            })

        if (uploadError) throw uploadError

        // สร้าง Signed URL (ใช้ได้ 1 ปี) — ทำงานได้แน่นอนไม่ว่า bucket จะ public หรือไม่
        const { data, error: signError } = await supabase.storage
            .from('rice-disease-analysis-results')
            .createSignedUrl(filePath, 60 * 60 * 24 * 365) // 1 ปี

        if (signError) throw signError

        console.log('🖼️ Image uploaded to Supabase Storage:', data.signedUrl)
        return data.signedUrl
    } catch (error) {
        console.error('❌ uploadImage error:', error.message)
        return null
    }
}

// =====================
// ปิด Database (ไม่จำเป็นสำหรับ Supabase — HTTP-based)
// =====================
export function closeDatabase() {
    // Supabase ใช้ HTTP requests ไม่มี persistent connection ที่ต้อง close
    console.log('🔒 Supabase client released')
}
