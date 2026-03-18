export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const body = await new Promise(resolve => {
    let data = '';
    req.on('data', chunk => data += chunk);
    req.on('end', () => resolve(data));
  });

  if (body.length > 100000) return res.status(413).json({ error: 'Experiment too large' });

  const { key, experiment, secret } = JSON.parse(body);
  if (!key || !experiment) return res.status(400).json({ error: 'Missing key or experiment' });
  if (secret !== 'eoe2026') return res.status(403).json({ error: 'Forbidden' });

  const url = `${process.env.KV_REST_API_URL}/lpush/${key}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${process.env.KV_REST_API_TOKEN}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify([JSON.stringify(experiment)]),
  });

  const data = await response.json();
  return res.status(200).json({ ok: true, length: data.result });
}
