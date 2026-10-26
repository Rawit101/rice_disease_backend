// Native text layout preserves Thai shaping. Run through combine_richmenu.py on macOS.
import AppKit

let width = 2500, height = 843
let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
    bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
    colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
let context = NSGraphicsContext.current!.cgContext
context.translateBy(x: 0, y: CGFloat(height))
context.scaleBy(x: 1, y: -1)

func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) -> NSColor {
    NSColor(srgbRed: r / 255, green: g / 255, blue: b / 255, alpha: 1)
}
let green = color(20, 92, 60), light = color(217, 235, 207)
func rect(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat, _ radius: CGFloat, _ fill: NSColor) {
    fill.setFill()
    NSBezierPath(roundedRect: NSRect(x: x, y: y, width: w, height: h), xRadius: radius, yRadius: radius).fill()
}
func text(_ value: String, _ x: CGFloat, _ y: CGFloat, _ size: CGFloat, _ fill: NSColor) {
    // Use a flipped text context so glyphs are upright in the top-left coordinate system.
    NSGraphicsContext.saveGraphicsState()
    context.translateBy(x: x, y: y)
    context.scaleBy(x: 1, y: -1)
    let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont(name: "Thonburi", size: size) ?? NSFont.systemFont(ofSize: size),
        .foregroundColor: fill
    ]
    (value as NSString).draw(at: NSPoint(x: 0, y: 0), withAttributes: attributes)
    NSGraphicsContext.restoreGraphicsState()
}
rect(0, 0, 2500, 843, 0, color(244, 248, 240))
rect(48, 48, 2404, 747, 54, green)
light.setStroke()
let camera = NSBezierPath(roundedRect: NSRect(x: 205, y: 310, width: 330, height: 240), xRadius: 34, yRadius: 34)
camera.lineWidth = 15
camera.stroke()
let lens = NSBezierPath(ovalIn: NSRect(x: 304, y: 364, width: 132, height: 132))
lens.lineWidth = 15
lens.stroke()
rect(295, 265, 150, 50, 15, light)
text("ส่งรูปใบข้าวเพื่อตรวจโรค", 700, 365, 115, .white)
text("เลือกรูปใบข้าว แล้วรับชื่อโรคเบื้องต้น", 704, 502, 70, light)
rect(704, 563, 490, 114, 57, color(217, 239, 163))
text("เลือกรูปภาพ", 775, 653, 62, green)
NSGraphicsContext.restoreGraphicsState()
let destination = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "image-only-richmenu.png"
try bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: destination))
