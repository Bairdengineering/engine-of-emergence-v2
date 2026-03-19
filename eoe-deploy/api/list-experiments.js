export default async function handler(req, res) {
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' });
  const { secret } = req.query;
  if (secret !== 'eoe2026') return res.status(403).json({ error: 'Forbidden' });
  const base = process.env.KV_REST_API_URL;
  const auth = { Authorization: 'Bearer ' + process.env.KV_REST_API_TOKEN };
  const idxResp = await fetch(base + '/smembers/eoe_exp_index', { headers: auth });
  const idxData = await idxResp.json();
  let keys = (idxData.result || []).map(k => {
    try { const p = JSON.parse(k); return Array.isArray(p) ? p[0] : k; } catch { return k; }
  }).filter(k => k && k.startsWith('eoe_exp_'));
  if (keys.length === 0) return res.status(200).json({ experiments: [] });
  const results = await Promise.all(keys.map(async key => {
    const r = await fetch(base + '/lrange/' + key + '/0/-1', { headers: auth });
    const d = await r.json();
    const runs = (d.result || []).map(e => {
      let val = e;
      while (Array.isArray(val) && val.length > 0) val = val[0];
      try {
        const run = typeof val === 'string' ? JSON.parse(val) : val;
        if (!run || typeof run !== 'object') return null;
        if (run.analysis && !run.analysis.chart_data && run.analysis.narrative) {
          try {
            const s = run.analysis.narrative;
            const i = s.indexOf('"chart_data"');
            if (i > 0) {
              const start = s.lastIndexOf('{', i);
              const end = s.indexOf('}]', i) + 2;
              if (start >= 0 && end > start) {
                const parsed = JSON.parse(s.slice(start, end) + '}');
                if (parsed.chart_data) run.analysis.chart_data = parsed.chart_data;
              }
            }
          } catch(e2) {}
        }
        return run;
      } catch { return null; }
    }).filter(Boolean);
    return { key, runs };
  }));
  return res.status(200).json({ experiments: results });
}
