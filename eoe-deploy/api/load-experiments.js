export default async function handler(req, res) {
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

  const { key } = req.query;
  if (!key) return res.status(400).json({ error: 'Missing key' });

  const url = `${process.env.KV_REST_API_URL}/lrange/${key}/0/-1`;
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}` },
  });

  const data = await response.json();
  const experiments = (data.result || []).map(e => {
    try { return typeof e === 'string' ? JSON.parse(e) : e; } catch { return null; }
  }).filter(Boolean);
  return res.status(200).json({ experiments });
}
