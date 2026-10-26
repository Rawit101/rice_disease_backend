import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { loadModule } from './module-fixture.js'

test('Supabase insert and history support null advice, severity and image URL', async () => {
    let inserted
    const client = {
        from(table) {
            if (table === 'users') return { select: () => ({ limit: async () => ({ error: null }) }) }
            assert.equal(table, 'analyses')
            return {
                insert(record) {
                    inserted = record
                    return { select: () => ({ single: async () => ({ data: { id: 1 }, error: null }) }) }
                },
                select(columns) {
                    assert.ok(columns.includes('advice, image_url'))
                    return { eq: () => ({ order: () => ({ limit: async () => ({ data: [{ id: 1, ...inserted }], error: null }) }) }) }
                }
            }
        },
        rpc: async () => ({ error: null })
    }
    const db = await loadModule('database.js', {
        '@supabase/supabase-js': { createClient: () => client }
    }, { process: { env: { SUPABASE_URL: 'https://db.test', SUPABASE_ANON_KEY: 'fake' } } })
    await db.initDatabase()
    assert.equal(await db.saveAnalysis('test-user', { disease: 'โรคไหม้', confidence: 0.9 }), 1)
    assert.equal(inserted.advice, null)
    assert.equal(inserted.image_url, null)
    assert.equal(inserted.severity, null)
    const history = await db.getAnalysisHistory('test-user')
    assert.equal(history[0].advice, null)
    assert.equal(history[0].image_url, null)
    const schema = fs.readFileSync('scripts/supabase-init.sql', 'utf8')
    assert.match(schema, /severity TEXT,\s+advice TEXT,\s+image_url TEXT,/)
})
