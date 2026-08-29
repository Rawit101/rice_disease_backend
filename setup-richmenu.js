/**
 * สคริปต์สร้าง Rich Menu สำหรับ LINE Bot น้องข้าวสวย
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
// Rich Menu Helper & Structure (2 ช่อง: ซ้าย-ขวา)
// Layout:
//   ┌───────────────────┬───────────────────┐
//   │   A: เลือกรูปภาพ    │ B: เว็บไซต์กรมการข้าว  │
//   └───────────────────┴───────────────────┘
// =====================

function getImageDimensions(buffer) {
    try {
        // PNG
        if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47) {
            return {
                width: buffer.readUInt32BE(16),
                height: buffer.readUInt32BE(20)
            }
        }
        // JPEG
        if (buffer[0] === 0xFF && buffer[1] === 0xD8) {
            let offset = 2
            while (offset < buffer.length) {
                const marker = buffer.readUInt16BE(offset)
                offset += 2
                if (marker === 0xFFC0 || marker === 0xFFC2) {
                    const height = buffer.readUInt16BE(offset + 3)
                    const width = buffer.readUInt16BE(offset + 5)
                    return { width, height }
                }
                const length = buffer.readUInt16BE(offset)
                offset += length
            }
        }
    } catch (e) {
        console.warn('⚠️ Could not parse image header, using default 2500x843')
    }
    return { width: 2500, height: 843 }
}

function buildRichMenuBody(width = 2500, height = 843) {
    const col1Width = Math.round(width * (833 / 2500)) // 833 for 2500px width
    const col2Width = width - col1Width // 1667 for 2500px width
    return {
        size: {
            width: width,
            height: height
        },
        selected: true,
        name: "น้องข้าวสวย - เมนูหลัก (2 ช่อง 1:2)",
        chatBarText: "เมนู",
        areas: [
            // ===== ช่องที่ 1 (ซ้ายมือ - A): เลือกรูปภาพในโทรศัพท์ (กว้าง 833px / 1 ใน 3) =====
            {
                bounds: { x: 0, y: 0, width: col1Width, height: height },
                action: {
                    type: "uri",
                    label: "เลือกรูปภาพ",
                    uri: "line://nv/cameraRoll/single"
                }
            },
            // ===== ช่องที่ 2 (ขวามือ - B): ลิ้งก์ไปเว็บไซต์กรมการข้าว (กว้าง 1667px / 2 ใน 3) =====
            {
                bounds: { x: col1Width, y: 0, width: col2Width, height: height },
                action: {
                    type: "uri",
                    label: "กรมการข้าว",
                    uri: "https://www.ricethailand.go.th/"
                }
            }
        ]
    }
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

async function createRichMenu(richMenuObject) {
    try {
        const res = await axios.post('https://api.line.me/v2/bot/richmenu', richMenuObject, { headers })
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
        console.error('\n   ขนาดรูปต้องเป็น 2500x843 px หรือ 2500x1686 px, ไม่เกิน 1MB')
        process.exit(1)
    }

    // ตรวจสอบขนาดไฟล์และขนาดภาพ
    const imageBuffer = fs.readFileSync(imagePath)
    const dimensions = getImageDimensions(imageBuffer)
    const stats = fs.statSync(imagePath)
    const sizeMB = (stats.size / (1024 * 1024)).toFixed(2)
    console.log(`📄 Found image: ${path.basename(imagePath)} (${dimensions.width}x${dimensions.height} px, ${sizeMB} MB)`)

    if (stats.size > 1024 * 1024) {
        console.warn(`⚠️  ไฟล์ใหญ่เกิน 1MB (${sizeMB} MB) - LINE อาจไม่รับ`)
        console.warn('   ลองลดขนาดรูปก่อน\n')
    }

    const richMenuObject = buildRichMenuBody(dimensions.width, dimensions.height)

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
    console.log('\n📝 สร้าง Rich Menu ใหม่ (2 ช่อง ซ้าย-ขวา)...')
    const richMenuId = await createRichMenu(richMenuObject)

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
