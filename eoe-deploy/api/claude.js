const LIMIT = 20;
const WINDOW_MS = 60 * 60 * 1000;
const rateLimitMap = new Map();

const ALLOWED_ORIGIN = 'https://engine-of-emergence-v2.vercel.app';

export default async function handler(req, res) {
  // CORS
  const origin = req.headers['origin'] || '';
  if (origin && origin !== ALLOWED_ORIGIN && !origin.includes('localhost')) {
    return res.status(403).json({ error: 'Forbidden' });
  }
  res.setHeader('Access-Control-Allow-Origin', origin || ALLOWED_ORIGIN);
  res.setHeader('Access-Control-Allow-Methods', 'POST');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  // Rate limiting
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

  // Size limit — 50KB max
  const body = await new Promise(resolve => {
    let data = '';
    req.on('data', chunk => data += chunk);
    req.on('end', () => resolve(data));
  });
  if (body.length > 50000) return res.status(413).json({ error: 'Request too large' });

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_KEY || '',
