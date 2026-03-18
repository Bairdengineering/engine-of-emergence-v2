const LIMIT = 20;
const WINDOW_MS = 60 * 60 * 1000;
const rateLimitMap = new Map();

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const ip = (req.headers['x-forwarded-for'] || 'unknown').split(',')[0].trim();
  const now = Date.now();
  const entry = rateLimitMap.get(ip);

  if (!entry || now - entry.windowStart > WINDOW_MS) {
    rateLimitMap.set(ip, { count: 1, windowStart: now });
  } else if (entry.count >= LIMIT) {
    return res.status(429).json({ error: 'Rate limit exceeded. Max 20 requests/hour.' });
  } else {
    entry.count++;
  }

  const body = await new Promise(resolve => {
    let data = '';
    req.on('data', chunk => data += chunk);
    req.on('end', () => resolve(data));
  });

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_KEY || '',
      'anthropic-version': '2023-06-01',
    },
    body,
  });

  const data = await response.json();
  return res.status(response.status).json(data);
}
