-- =====================================================
-- Supabase Init SQL — สร้างตารางสำหรับ Rice Disease Backend
-- วิธีใช้: Copy ทั้งหมด → Supabase Dashboard → SQL Editor → Run
-- =====================================================

-- 1) ตารางผู้ใช้ LINE
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    line_user_id TEXT UNIQUE NOT NULL,
    display_name TEXT,
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_active_at TIMESTAMPTZ DEFAULT NOW(),
    total_analyses INTEGER DEFAULT 0,
    total_messages INTEGER DEFAULT 0
);

-- 2) ตารางผลการวิเคราะห์โรคข้าว
CREATE TABLE IF NOT EXISTS analyses (
    id BIGSERIAL PRIMARY KEY,
    line_user_id TEXT NOT NULL REFERENCES users(line_user_id),
    disease TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    severity TEXT,
    advice TEXT,
    image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3) ตารางประวัติการสนทนา
CREATE TABLE IF NOT EXISTS chat_history (
    id BIGSERIAL PRIMARY KEY,
    line_user_id TEXT NOT NULL REFERENCES users(line_user_id),
    role TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4) Indexes เพื่อความเร็วในการค้นหา
CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(line_user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(line_user_id);
CREATE INDEX IF NOT EXISTS idx_chat_created ON chat_history(created_at);

-- 5) RPC function: เพิ่มจำนวนการวิเคราะห์ของผู้ใช้
CREATE OR REPLACE FUNCTION increment_analysis_count(user_line_id TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE users
    SET total_analyses = total_analyses + 1,
        last_active_at = NOW()
    WHERE line_user_id = user_line_id;
END;
$$ LANGUAGE plpgsql;

-- 6) RPC function: เพิ่มจำนวนข้อความของผู้ใช้
CREATE OR REPLACE FUNCTION increment_message_count(user_line_id TEXT)
RETURNS VOID AS $$
BEGIN
    UPDATE users
    SET total_messages = total_messages + 1,
        last_active_at = NOW()
    WHERE line_user_id = user_line_id;
END;
$$ LANGUAGE plpgsql;

-- 7) RPC function: ดึงสถิติรวม dashboard
CREATE OR REPLACE FUNCTION get_dashboard_stats()
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'total_users', (SELECT COUNT(*) FROM users),
        'total_analyses', (SELECT COUNT(*) FROM analyses),
        'total_messages', (SELECT COUNT(*) FROM chat_history WHERE role = 'user'),
        'top_diseases', (
            SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
            FROM (
                SELECT disease, COUNT(*) as count, ROUND(AVG(confidence)::numeric, 2) as avg_confidence
                FROM analyses
                WHERE disease NOT IN ('ไม่พบโรค (Healthy)', 'ไม่พบโรค')
                GROUP BY disease
                ORDER BY count DESC
                LIMIT 5
            ) t
        ),
        'recent_users', (
            SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
            FROM (
                SELECT line_user_id, display_name, total_analyses, total_messages, last_active_at
                FROM users
                ORDER BY last_active_at DESC
                LIMIT 5
            ) t
        ),
        'recent_analyses', (
            SELECT COALESCE(json_agg(row_to_json(t)), '[]'::json)
            FROM (
                SELECT a.disease, a.confidence, a.severity, a.created_at, u.display_name
                FROM analyses a
                LEFT JOIN users u ON a.line_user_id = u.line_user_id
                ORDER BY a.created_at DESC
                LIMIT 10
            ) t
        )
    ) INTO result;
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- ✅ เสร็จ! ตาราง + indexes + RPC functions พร้อมใช้งาน
