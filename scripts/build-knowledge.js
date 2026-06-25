import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const ROOT_DIR = path.resolve(__dirname, '..')
const OUT_DIR = path.join(ROOT_DIR, 'data')
const OUT_FILE = path.join(OUT_DIR, 'rice_knowledge.json')

const LIST_URL = 'https://rkb.ricethailand.go.th/web/redirect_action.php?code=GRE6SHELSB'
const BASE_URL = 'https://rkb.ricethailand.go.th/web/'

function decodeEntities(text) {
    return text
        .replace(/&nbsp;/g, ' ')
        .replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'")
        .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
}

function cleanHtml(html) {
    return decodeEntities(html)
        .replace(/<script[\s\S]*?<\/script>/gi, ' ')
        .replace(/<style[\s\S]*?<\/style>/gi, ' ')
        .replace(/<nav[\s\S]*?<\/nav>/gi, ' ')
        .replace(/<footer[\s\S]*?<\/footer>/gi, ' ')
        .replace(/<br\s*\/?>/gi, '\n')
        .replace(/<\/(p|div|li|h[1-6]|tr)>/gi, '\n')
        .replace(/<[^>]+>/g, ' ')
        .replace(/\r/g, '')
        .replace(/[ \t]+/g, ' ')
        .replace(/\n[ \t]+/g, '\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim()
}

function extractContent(html) {
    const text = cleanHtml(html)
    const start = text.search(/โรค|สาเหตุ|อาการ/)
    const end = text.search(/ติดต่อเรา|กรมการข้าว เลขที่/)
    return text
        .slice(start >= 0 ? start : 0, end > 0 ? end : undefined)
        .replace(/^Print\s*/i, '')
        .trim()
}

function extractDiseaseLinks(html) {
    const links = [...html.matchAll(/href="([^"]*content_page\.php\?code=[^"]+)"[^>]*>([^<]+)/g)]
        .map(match => ({
            url: new URL(match[1], BASE_URL).href,
            title: decodeEntities(match[2]).replace(/\s+/g, ' ').trim()
        }))
        .filter(link => link.title.startsWith('โรค'))

    const unique = []
    const seen = new Set()
    for (const link of links) {
        if (seen.has(link.url)) continue
        seen.add(link.url)
        unique.push(link)
    }
    return unique
}

async function fetchText(url) {
    const response = await fetch(url)
    if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`)
    }
    return response.text()
}

async function main() {
    fs.mkdirSync(OUT_DIR, { recursive: true })

    console.log(`Fetching disease list: ${LIST_URL}`)
    const listHtml = await fetchText(LIST_URL)
    const links = extractDiseaseLinks(listHtml)
    console.log(`Found ${links.length} disease pages`)

    const entries = []
    for (const [index, link] of links.entries()) {
        console.log(`[${index + 1}/${links.length}] ${link.title}`)
        const html = await fetchText(link.url)
        const content = extractContent(html)
        entries.push({
            title: link.title,
            source: link.url,
            content
        })
    }

    const payload = {
        source: 'Rice Knowledge Bank, กรมการข้าว',
        sourceUrl: LIST_URL,
        updatedAt: new Date().toISOString(),
        entries
    }

    fs.writeFileSync(OUT_FILE, JSON.stringify(payload, null, 2), 'utf8')
    console.log(`Saved ${entries.length} entries to ${OUT_FILE}`)
}

main().catch(error => {
    console.error('Failed to build knowledge file:', error)
    process.exit(1)
})
