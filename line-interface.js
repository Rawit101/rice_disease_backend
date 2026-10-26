export const IMAGE_ONLY_INSTRUCTION = 'กรุณาส่งรูปใบข้าวเพื่อตรวจโรค ระบบนี้รองรับเฉพาะรูปภาพ'

// One full-width action, matching image-only-richmenu.png. No legacy feature links.
export function buildRichMenuBody() {
    return {
        size: { width: 2500, height: 843 },
        selected: true,
        name: 'ตรวจโรคข้าวจากรูปใบข้าว',
        chatBarText: 'ส่งรูปใบข้าว',
        areas: [{
            bounds: { x: 0, y: 0, width: 2500, height: 843 },
            action: {
                type: 'uri',
                label: 'เลือกรูปใบข้าว',
                uri: 'line://nv/cameraRoll/single'
            }
        }]
    }
}
