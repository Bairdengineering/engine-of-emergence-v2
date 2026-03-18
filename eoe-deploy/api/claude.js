export const config = { runtime: 'edge' };

const rateLimitMap = new Map();
const LIMIT = 20;
const WINDOW_MS = 60 * 60 * 1000;

export default async function handler(req) {
  if (req.method !== 'POST') {
    return new Response('Method not allowed', { status: 405 });
  }

  const ip = req.headers.get('x-forwarded-for')?.split(',')[0]?.trim() || 'unknown';
  const now = Date.now();
  const entry = rateLimitMap.get(ip);

  if (!entry || now - entry.windowStart > WINDOW_MS) {
    rateLimitMap.set(ip, { count: 1, windowStart: now });
  } else if (entry.count >= LIMIT) {
    return new Response(JSON.stringify({ error: 'Rate limit exceeded. Max 20 requests/hour.' }), {
      status: 429,
      headers: { 'Content-Type': 'application/json' },
    });
  } else {
    entry.count++;
  }

  const body = await req.text();

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_KEY || '',
      'anthropic-version': '2023-06-01',
    },
    body,
  });

  const data = await response.text();
  return new Response(data, {
    status: response.status,
    headers: { 'Content-Type': 'application/json' },
  });
}
