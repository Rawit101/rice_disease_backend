/** Preview by default. Only --publish changes LINE OA settings. Never deletes menus. */
import 'dotenv/config'
import axios from 'axios'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { buildRichMenuBody } from './line-interface.js'

const directory = path.dirname(fileURLToPath(import.meta.url))
const imagePath = path.join(directory, 'image-only-richmenu.png')
const body = buildRichMenuBody()

async function main() {
    const publish = process.argv.includes('--publish')
    if (!publish) {
        console.log(JSON.stringify({ image: imagePath, menu: body }, null, 2))
        console.log('Preview only. Use --publish to create, upload and set the default menu in LINE.')
        return
    }
    const token = process.env.LINE_TOKEN || process.env.CHANNEL_ACCESS_TOKEN
    if (!token) throw new Error('LINE_TOKEN or CHANNEL_ACCESS_TOKEN is required to publish')
    const image = fs.readFileSync(imagePath)
    if (image.length > 1024 * 1024) throw new Error('Menu image must be at most 1MB')
    const headers = { Authorization: `Bearer ${token}` }
    const response = await axios.post('https://api.line.me/v2/bot/richmenu', body, { headers })
    const id = response.data.richMenuId
    await axios.post(`https://api-data.line.me/v2/bot/richmenu/${id}/content`, image, {
        headers: { ...headers, 'Content-Type': 'image/png' }
    })
    await axios.post(`https://api.line.me/v2/bot/user/all/richmenu/${id}`, {}, { headers })
    console.log('Image-only default menu published. Existing menus were retained.')
}

main().catch(() => {
    // HTTP errors may contain LINE tokens; do not print raw error objects.
    console.error('Rich menu setup failed')
    process.exitCode = 1
})
