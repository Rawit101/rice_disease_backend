/* Names describe the annotation class, never a diagnosis inferred from a folder. */
(function (global) {
  'use strict';
  const thai = Object.freeze({
    Bacterial_Blight: 'โรคขอบใบแห้ง', Brown_Spot: 'โรคใบจุดสีน้ำตาล',
    Rice_Blast: 'โรคไหม้', Narrow_Brown_Spot: 'โรคใบขีดสีน้ำตาล',
    False_Smut: 'โรคดอกกระถิน', Dirty_Seed: 'โรคเมล็ดด่าง',
    Sheath_Rot: 'โรคกาบใบเน่า', Stem_Rot: 'โรคลำต้นเน่า',
    Red_Stripe: 'โรคใบแถบแดง', Tungro: 'โรคใบสีส้ม'
  });
  const colors = ['#37e5a3','#ffcd43','#78c4ff','#ff95cc','#c6abff',
    '#ffab70','#bce77b','#86e4e4','#ff8282','#ffe56b'];
  function name(id, names) {
    const en = names[id];
    return en ? `${thai[en] || en} · ${en} · class ${id}` : `class ${id} · ยังไม่มีชื่อในชุดนี้`;
  }
  function color(id) { return colors[((Number(id) % colors.length) + colors.length) % colors.length]; }
  function summary(objects, names) {
    return [...new Set(objects.map(o => Number(o.class_id)))].sort((a,b)=>a-b)
      .map(id => name(id, names)).join(' / ');
  }
  function draw(canvas, img, objects, names, options = {}) {
    // A consistent drawing size keeps names readable even for 224px source images.
    const width = 1200, height = Math.round(width * img.naturalHeight / img.naturalWidth);
    const ids = [...new Set(objects.map(o => Number(o.class_id)))].sort((a,b)=>a-b);
    const legendHeight = ids.length ? 18 + ids.length * 50 : (options.emptyText ? 63 : 0);
    canvas.width = width; canvas.height = height + legendHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, width, height);
    const groups = new Map();
    for (const o of objects) {
      const points = o.polygon;
      if (!points || points.length < 3) continue;
      const id = Number(o.class_id), col = color(id);
      ctx.beginPath(); points.forEach(([x,y],i)=>i ? ctx.lineTo(x*width,y*height) : ctx.moveTo(x*width,y*height));
      ctx.closePath(); ctx.fillStyle = col + '35'; ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.stroke();
      const xs = points.map(p=>p[0]), ys = points.map(p=>p[1]);
      const box = {x: Math.min(...xs)*width, y: Math.min(...ys)*height,
        right: Math.max(...xs)*width, bottom: Math.max(...ys)*height};
      box.area = (box.right-box.x)*(box.bottom-box.y);
      if (!groups.has(id) || box.area > groups.get(id).area) groups.set(id, box);
    }
    // One full name per class avoids burying small lesions under repeated text.
    // Every region of that class shares the same color and the legend counts all regions.
    const occupied = [];
    ctx.font = 'bold 35px system-ui, sans-serif'; ctx.textBaseline = 'middle';
    for (const [id, box] of groups) {
      const text = name(id, names), pad = 12, h = 59;
      const w = Math.min(width-16, ctx.measureText(text).width+pad*2);
      const x = Math.min(width-w-8, Math.max(8, box.x));
      let y = Math.max(8, Math.min(height-h-8, box.y-h-8));
      for (let tries=0; tries<ids.length; tries++) {
        if (!occupied.some(r=>x<r.x+r.w && x+w>r.x && y<r.y+r.h && y+h>r.y)) break;
        y = Math.min(height-h-8, y+h+6);
      }
      occupied.push({x,y,w,h});
      ctx.fillStyle='#10281fec'; ctx.fillRect(x,y,w,h);
      ctx.strokeStyle=color(id); ctx.lineWidth=3; ctx.strokeRect(x,y,w,h);
      ctx.fillStyle=color(id); ctx.fillText(text,x+pad,y+h/2,w-pad*2);
      ctx.beginPath(); ctx.moveTo(x+Math.min(w/2,40),y+h);
      ctx.lineTo(Math.max(box.x,Math.min(box.right,x+40)),Math.max(box.y,Math.min(box.bottom,y+h)));
      ctx.stroke();
    }
    if (legendHeight) {
      ctx.fillStyle='#10281f'; ctx.fillRect(0,height,width,legendHeight);
      ctx.font='30px system-ui, sans-serif'; ctx.textBaseline='middle';
      ids.forEach((id,i)=>{
        const count=objects.filter(o=>Number(o.class_id)===id).length;
        ctx.fillStyle=color(id); ctx.fillRect(16,height+20+i*50,14,23);
        ctx.fillText(`${name(id,names)}  ·  ${count} ขอบเขต`,43,height+31+i*50,width-60);
      });
      if (!ids.length) { ctx.fillStyle='#d7e7dd'; ctx.fillText(options.emptyText,20,height+31,width-40); }
    }
    canvas.setAttribute('aria-label', summary(objects,names) || options.emptyText || 'ภาพต้นฉบับ');
  }
  global.RiceClassNames = Object.freeze({thai, name, color, summary, draw});
})(globalThis);
