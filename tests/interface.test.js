import test from 'node:test'
import assert from 'node:assert/strict'
import { buildRichMenuBody } from '../line-interface.js'
import { loadModule } from './module-fixture.js'

test('rich menu has one photo-selection action covering the entire image', () => {
    const menu = buildRichMenuBody()
    assert.equal(menu.areas.length, 1)
    assert.deepEqual(menu.areas[0].bounds, { x: 0, y: 0, ...menu.size })
    assert.deepEqual(menu.areas[0].action, {
        type: 'uri', label: 'เลือกรูปใบข้าว', uri: 'line://nv/cameraRoll/single'
    })
})

test('rich menu setup previews without credentials or any LINE request by default', async () => {
    const output = []
    const forbidden = () => assert.fail('Preview must not call LINE')
    await loadModule('setup-richmenu.js', {
        'dotenv/config': {},
        axios: { default: { get: forbidden, post: forbidden, delete: forbidden } }
    }, {
        process: { argv: ['node', 'setup-richmenu.js'], env: {} },
        console: { log: message => output.push(message), error: forbidden }
    })
    const preview = JSON.parse(output[0])
    assert.ok(preview.image.endsWith('image-only-richmenu.png'))
    assert.equal(preview.menu.areas.length, 1)
})
