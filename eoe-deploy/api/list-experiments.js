export default async function handler(req, res) {
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });

  const { secret } = req.query;
  if (secret !== 'eoe2026') return res.status(403).json({ error: 'Forbidden' });

  const base = process.env.KV_REST_API_URL;
  const auth = { Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}` };

  // Get all experiment keys from index
  const idxResp = await fetch(`${base}/smembers/eoe_exp_index`, { headers: auth });
  const idxData = await idxResp.json();
  const keys = idxData.result || [];

  if (keys.length === 0) return res.status(200).json({ experiments: [] });

  // Load all experiments for each key
  const results = await Promise.all(keys.map(async key => {
    const r = await fetch(`${base}/lrange/${key}/0/-1`, { headers: auth });
    const d = await r.json();
    const runs = (d.result || []).map(e => {
      try { return typeof e === 'string' ? JSON.parse(e) : e; } catch { return null; }
    }).filter(Boolean);
    return { key, runs };
  }));

  return res.status(200).json({ experiments: results });
}
