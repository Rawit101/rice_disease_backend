/**
 * สคริปต์สร้าง Rich Menu สำหรับ LINE Bot ไอนาย
 * 
 * วิธีใช้:
 *   1. วางรูป Rich Menu ไว้ที่ d:\backend-farmer\richmenu.jpg (หรือ .png)
 *   2. รัน: node setup-richmenu.js
 * 
 * รูปต้องมีขนาด 2500x1686 px, ไม่เกิน 1MB, เป็น JPEG หรือ PNG
 */

import "dotenv/config"
import axios from "axios"
import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const LINE_TOKEN = process.env.LINE_TOKEN || process.env.CHANNEL_ACCESS_TOKEN

if (!LINE_TOKEN) {
    console.error('❌ LINE_TOKEN not found in .env')
    process.exit(1)
}

const headers = {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${LINE_TOKEN}`
}

// =====================
// Rich Menu Structure (2500x1686)
// Layout: (3 แถวแนวตั้ง)
//   ┌─────────────────────────────┐
//   │      องค์ความรู้เรื่องข้าว        │  y: 0-562
//   ├─────────────────────────────┤
//   │           เลือกรูปภาพ           │  y: 562-1124
//   ├─────────────────────────────┤
//   │            โรคข้าว            │  y: 1124-1686
//   └─────────────────────────────┘
// =====================
const richMenuBody = {
    size: {
        width: 2500,
        height: 1686
    },
    selected: true,
    name: "ไอนาย - เมนูหลัก (3 แถว)",
    chatBarText: "เมนู",
    areas: [
        // ===== ปุ่ม 1: องค์ความรู้เรื่องข้าว (บน) =====
        {
            bounds: { x: 0, y: 0, width: 2500, height: 562 },
            action: {
                type: "uri",
                label: "องค์ความรู้เรื่องข้าว",
                uri: "https://rkb.ricethailand.go.th/web/index.php"
            }
        },
        // ===== ปุ่ม 2: เลือกรูปภาพ (กลาง) =====
        {
            bounds: { x: 0, y: 562, width: 2500, height: 562 },
            action: {
                type: "uri",
                label: "เลือกรูปภาพ",
                uri: "line://nv/cameraRoll/single"
            }
        },
        // ===== ปุ่ม 3: โรคข้าว (ล่าง) =====
        {
            bounds: { x: 0, y: 1124, width: 2500, height: 562 },
            action: {
                type: "uri",
                label: "โรคข้าว",
                uri: "https://web-rice-diseases.vercel.app/"
            }
        }
    ]
}


// =====================
// Main Functions
// =====================

async function listExistingRichMenus() {
    try {
        const res = await axios.get('https://api.line.me/v2/bot/richmenu/list', { headers })
        return res.data.richmenus || []
    } catch (error) {
        console.error('Error listing rich menus:', error.response?.data || error.message)
        return []
    }
}

async function deleteRichMenu(richMenuId) {
    try {
        await axios.delete(`https://api.line.me/v2/bot/richmenu/${richMenuId}`, { headers })
        console.log(`🗑️  Deleted rich menu: ${richMenuId}`)
    } catch (error) {
        console.error('Error deleting rich menu:', error.response?.data || error.message)
    }
}

async function createRichMenu() {
    try {
        const res = await axios.post('https://api.line.me/v2/bot/richmenu', richMenuBody, { headers })
        const richMenuId = res.data.richMenuId
        console.log(`✅ Rich menu created: ${richMenuId}`)
        return richMenuId
    } catch (error) {
        console.error('❌ Error creating rich menu:', error.response?.data || error.message)
        throw error
    }
}

async function uploadRichMenuImage(richMenuId, imagePath) {
    try {
        const imageBuffer = fs.readFileSync(imagePath)
        const ext = path.extname(imagePath).toLowerCase()
        const contentType = ext === '.png' ? 'image/png' : 'image/jpeg'

        await axios.post(
            `https://api-data.line.me/v2/bot/richmenu/${richMenuId}/content`,
            imageBuffer,
            {
                headers: {
                    'Authorization': `Bearer ${LINE_TOKEN}`,
                    'Content-Type': contentType
                },
                maxContentLength: 10 * 1024 * 1024,
                maxBodyLength: 10 * 1024 * 1024
            }
        )
        console.log(`🖼️  Image uploaded successfully`)
    } catch (error) {
        console.error('❌ Error uploading image:', error.response?.data || error.message)
        throw error
    }
}

async function setDefaultRichMenu(richMenuId) {
    try {
        await axios.post(
            `https://api.line.me/v2/bot/user/all/richmenu/${richMenuId}`,
            {},
            { headers }
        )
        console.log(`📌 Set as default rich menu for all users`)
    } catch (error) {
        console.error('❌ Error setting default:', error.response?.data || error.message)
        throw error
    }
}

// =====================
// Run Setup
// =====================
async function main() {
    console.log('🚀 Starting Rich Menu Setup...\n')

    // 1. ค้นหาไฟล์รูป
    const possibleImages = ['richmenu.jpg', 'richmenu.jpeg', 'richmenu.png']
    let imagePath = null
    for (const filename of possibleImages) {
        const fullPath = path.join(__dirname, filename)
        if (fs.existsSync(fullPath)) {
            imagePath = fullPath
            break
        }
    }

    if (!imagePath) {
        console.error('❌ ไม่พบรูป Rich Menu!')
        console.error('   กรุณาวางรูปชื่อ richmenu.jpg หรือ richmenu.png ไว้ที่:')
        console.error(`   ${__dirname}`)
        console.error('\n   ขนาดรูปต้องเป็น 2500x1686 px, ไม่เกิน 1MB')
        process.exit(1)
    }

    // ตรวจสอบขนาดไฟล์
    const stats = fs.statSync(imagePath)
    const sizeMB = (stats.size / (1024 * 1024)).toFixed(2)
    console.log(`📄 Found image: ${path.basename(imagePath)} (${sizeMB} MB)`)

    if (stats.size > 1024 * 1024) {
        console.warn(`⚠️  ไฟล์ใหญ่เกิน 1MB (${sizeMB} MB) - LINE อาจไม่รับ`)
        console.warn('   ลองลดขนาดรูปก่อน\n')
    }

    // 2. ลบ rich menu เดิมทั้งหมด (ถ้ามี)
    console.log('\n📋 ตรวจสอบ Rich Menu เดิม...')
    const existing = await listExistingRichMenus()
    if (existing.length > 0) {
        console.log(`   พบ ${existing.length} rich menu เดิม - กำลังลบ...`)
        for (const menu of existing) {
            await deleteRichMenu(menu.richMenuId)
        }
    } else {
        console.log('   ไม่มี rich menu เดิม')
    }

    // 3. สร้าง Rich Menu ใหม่
    console.log('\n📝 สร้าง Rich Menu ใหม่...')
    const richMenuId = await createRichMenu()

    // 4. อัปโหลดรูป
    console.log('\n🖼️  อัปโหลดรูป...')
    await uploadRichMenuImage(richMenuId, imagePath)

    // 5. ตั้งเป็น default
    console.log('\n📌 ตั้งเป็น Default Rich Menu...')
    await setDefaultRichMenu(richMenuId)

    console.log('\n' + '='.repeat(50))
    console.log('🎉 Rich Menu พร้อมใช้งานแล้ว!')
    console.log(`   Rich Menu ID: ${richMenuId}`)
    console.log('='.repeat(50))
}

main().catch(err => {
    console.error('\n❌ Setup failed:', err.message)
    process.exit(1)
})
