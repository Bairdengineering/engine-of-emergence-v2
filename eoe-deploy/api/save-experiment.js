export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const body = await new Promise(resolve => {
    let data = '';
    req.on('data', chunk => data += chunk);
    req.on('end', () => resolve(data));
  });

  if (body.length > 500000) return res.status(413).json({ error: 'Experiment too large' });

  let parsed;
  try { parsed = JSON.parse(body); }
  catch(e) { return res.status(400).json({ error: 'Invalid JSON' }); }

  const { key, experiment, secret } = parsed;
  if (!key || !experiment) return res.status(400).json({ error: 'Missing key or experiment' });
  if (secret !== 'eoe2026') return res.status(403).json({ error: 'Forbidden' });

  const base = process.env.KV_REST_API_URL;
  const auth = { Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}` };

  // Save experiment to its key
  await fetch(`${base}/lpush/${key}`, {
    method: 'POST',
    headers: { ...auth, 'Content-Type': 'application/json' },
    body: JSON.stringify([JSON.stringify(experiment)]),
  });

  // Add key to index (dedupe with srem+sadd)
  await fetch(`${base}/sadd/eoe_exp_index`, {
    method: 'POST',
    headers: { ...auth, 'Content-Type': 'application/json' },
    body: JSON.stringify([key]),
  });

  return res.status(200).json({ ok: true });
}
