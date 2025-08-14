import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import fetch from 'node-fetch';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '1mb' }));

// --- Config ---
const PORT = process.env.PORT || 3000;
const TIMEOUT_MS = 25000; // per provider attempt

// Order matters: first available provider here will be tried first
const PROVIDER_ORDER = [
  'openai',
  'anthropic',
  'gemini',
  'mistral',
  'groq'
];

// --- Utilities ---
const withTimeout = (promise, ms, tag) => Promise.race([
  promise,
  new Promise((_, reject) => setTimeout(() => reject(new Error(`${tag} timeout after ${ms}ms`)), ms))
]);

function isNonEmptyString(s) {
  return typeof s === 'string' && s.trim().length > 0;
}

function pickModel(provider) {
  switch (provider) {
    case 'openai':
      return 'gpt-4o-mini'; // or 'gpt-4.1', 'gpt-5' if available
    case 'anthropic':
      return 'claude-3-5-sonnet-latest';
    case 'gemini':
      return 'gemini-1.5-flash'; // or 'gemini-1.5-pro'
    case 'mistral':
      return 'mistral-large-latest';
    case 'groq':
      return 'llama-3.1-70b-versatile';
    default:
      return null;
  }
}

// --- Provider callers ---
async function callOpenAI(prompt, systemPrompt) {
  const key = process.env.OPENAI_API_KEY;
  if (!key) throw new Error('OPENAI_API_KEY missing');

  const model = pickModel('openai');
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${key}`
    },
    body: JSON.stringify({
      model,
      messages: [
        systemPrompt ? { role: 'system', content: systemPrompt } : null,
        { role: 'user', content: prompt }
      ].filter(Boolean),
      temperature: 0.7
    })
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`OpenAI error ${res.status}: ${t}`);
  }
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content ?? '';
  if (!isNonEmptyString(text)) throw new Error('OpenAI returned empty');
  return { provider: 'openai', model, text };
}

async function callAnthropic(prompt, systemPrompt) {
  const key = process.env.ANTHROPIC_API_KEY;
  if (!key) throw new Error('ANTHROPIC_API_KEY missing');
  const model = pickModel('anthropic');

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': key,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model,
      max_tokens: 1024,
      system: systemPrompt || undefined,
      messages: [{ role: 'user', content: prompt }]
    })
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Anthropic error ${res.status}: ${t}`);
  }
  const data = await res.json();
  const text = data?.content?.[0]?.text ?? '';
  if (!isNonEmptyString(text)) throw new Error('Anthropic returned empty');
  return { provider: 'anthropic', model, text };
}

async function callGemini(prompt, systemPrompt) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) throw new Error('GEMINI_API_KEY missing');
  const model = pickModel('gemini');

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`;
  const parts = [];
  if (systemPrompt) parts.push({ text: `System: ${systemPrompt}` });
  parts.push({ text: prompt });

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents: [{ parts }] })
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Gemini error ${res.status}: ${t}`);
  }
  const data = await res.json();
  const text = data?.candidates?.[0]?.content?.parts?.map(p => p.text).join('') ?? '';
  if (!isNonEmptyString(text)) throw new Error('Gemini returned empty');
  return { provider: 'gemini', model, text };
}

async function callMistral(prompt, systemPrompt) {
  const key = process.env.MISTRAL_API_KEY;
  if (!key) throw new Error('MISTRAL_API_KEY missing');
  const model = pickModel('mistral');

  const res = await fetch('https://api.mistral.ai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${key}`
    },
    body: JSON.stringify({
      model,
      messages: [
        systemPrompt ? { role: 'system', content: systemPrompt } : null,
        { role: 'user', content: prompt }
      ].filter(Boolean)
    })
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Mistral error ${res.status}: ${t}`);
  }
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content ?? '';
  if (!isNonEmptyString(text)) throw new Error('Mistral returned empty');
  return { provider: 'mistral', model, text };
}

async function callGroq(prompt, systemPrompt) {
  const key = process.env.GROQ_API_KEY;
  if (!key) throw new Error('GROQ_API_KEY missing');
  const model = pickModel('groq');

  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${key}`
    },
    body: JSON.stringify({
      model,
      messages: [
        systemPrompt ? { role: 'system', content: systemPrompt } : null,
        { role: 'user', content: prompt }
      ].filter(Boolean)
    })
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Groq error ${res.status}: ${t}`);
  }
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content ?? '';
  if (!isNonEmptyString(text)) throw new Error('Groq returned empty');
  return { provider: 'groq', model, text };
}

// Map provider keys to functions
const PROVIDERS = {
  openai: callOpenAI,
  anthropic: callAnthropic,
  gemini: callGemini,
  mistral: callMistral,
  groq: callGroq
};

// --- Fallback chain ---
async function askTramPLAR({ prompt, systemPrompt }) {
  const errors = [];

  for (const name of PROVIDER_ORDER) {
    const fn = PROVIDERS[name];

    // Skip providers with missing keys silently (by attempting and catching)
    try {
      const result = await withTimeout(fn(prompt, systemPrompt), TIMEOUT_MS, name);
      if (result && isNonEmptyString(result.text)) {
        return { ...result, tried: [...PROVIDER_ORDER], errors };
      }
    } catch (err) {
      errors.push({ provider: name, error: String(err?.message || err) });
      continue;
    }
  }

  // If we got here, nothing worked
  return {
    provider: null,
    model: null,
    text: 'কোনো প্রোভাইডার উত্তর দিতে পারেনি। পরে আবার চেষ্টা করুন।',
    tried: [...PROVIDER_ORDER],
    errors
  };
}

// --- API route ---
app.post('/api/chat', async (req, res) => {
  try {
    const { prompt, system } = req.body || {};
    if (!isNonEmptyString(prompt)) {
      return res.status(400).json({ error: 'prompt is required' });
    }

    const result = await askTramPLAR({ prompt, systemPrompt: system });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: String(e?.message || e) });
  }
});

// --- Minimal UI ---
app.get('/', (_req, res) => {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.end(`<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>TramPLAR — Multi‑AI Fallback</title>
<style>
  :root { --bg:#0b0f17; --fg:#dfe7ff; --card:#141a22; --acc:#3aa0ff; }
  html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);font-family:Inter,system-ui,Segoe UI,Arial,sans-serif}
  .wrap{max-width:900px;margin:40px auto;padding:0 16px}
  .title{font-size:28px;font-weight:800;letter-spacing:.3px;display:flex;gap:10px;align-items:center}
  .glow{color:#a9d6ff;text-shadow:0 0 12px rgba(58,160,255,.7)}
  .card{background:linear-gradient(180deg,#121821,#0d121a);border:1px solid #1f2a38;border-radius:20px;box-shadow:0 10px 30px rgba(0,0,0,.35);padding:16px}
  .row{display:flex;gap:8px;flex-wrap:wrap}
  input,textarea,button{border-radius:14px;border:1px solid #253247;background:#10161f;color:var(--fg);padding:12px 14px;font-size:14px}
  textarea{width:100%;min-height:120px;resize:vertical}
  button{cursor:pointer;background:linear-gradient(180deg,#15243a,#0f1b2c);border:1px solid #28415d}
  button:hover{filter:brightness(1.1)}
  .muted{opacity:.8;font-size:12px}
  .badge{display:inline-flex;gap:6px;align-items:center;padding:6px 10px;border-radius:999px;border:1px solid #2a3a54;background:#101827}
  .resp{white-space:pre-wrap;line-height:1.5}
  .grid{display:grid;grid-template-columns:1fr;gap:12px}
  @media (min-width:800px){ .grid{grid-template-columns:2fr 1fr} }
</style>
</head>
<body>
  <div class="wrap">
    <div class="title">🚀 <span class="glow">TramPLAR</span> — Multi‑AI Fallback</div>
    <p class="muted">একটা প্রশ্ন করো — TramPLAR প্রথমে OpenAI, ব্যর্থ হলে Claude, তারপর Gemini, Mistral, Groq ট্রাই করবে। যে আগে উত্তর দেবে, সেটাই দেখাবে।</p>

    <div class="grid">
      <div class="card">
        <textarea id="prompt" placeholder="তোমার প্রশ্ন লিখো..."></textarea>
        <div class="row" style="margin-top:8px">
          <input id="system" placeholder="(ঐচ্ছিক) সিস্টেম নির্দেশনা / টোন" style="flex:1" />
          <button id="ask">Ask TramPLAR</button>
        </div>
      </div>
      <div class="card">
        <div class="badge">⚙️ Provider order: <code style="padding-left:6px">openai → anthropic → gemini → mistral → groq</code></div>
        <div style="margin-top:10px" class="muted">API keys শুধু সার্ভারেই থাকে। ব্রাউজারে কখনো পাঠানো হয় না।</div>
      </div>
    </div>

    <div class="card" style="margin-top:16px">
      <div id="meta" class="muted">Ready.</div>
      <div id="answer" class="resp" style="margin-top:8px"></div>
      <details style="margin-top:10px"><summary>Debug (providers tried)</summary>
        <pre id="debug"></pre>
      </details>
    </div>
  </div>

<script>
  const btn = document.getElementById('ask');
  const promptEl = document.getElementById('prompt');
  const systemEl = document.getElementById('system');
  const meta = document.getElementById('meta');
  const answer = document.getElementById('answer');
  const debug = document.getElementById('debug');

  btn.addEventListener('click', async () => {
    const prompt = promptEl.value.trim();
    const system = systemEl.value.trim();
    if (!prompt) return alert('প্রশ্ন লিখে নাও!');

    btn.disabled = true; meta.textContent = 'Thinking...'; answer.textContent = ''; debug.textContent = '';

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, system })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      const tag = data.provider ? `${data.provider} • ${data.model || ''}` : 'No provider';
      meta.textContent = `Answered by: ${tag}`;
      answer.textContent = data.text || '';
      debug.textContent = JSON.stringify({ tried: data.tried, errors: data.errors }, null, 2);
    } catch (e) {
      meta.textContent = 'Error';
      answer.textContent = String(e.message || e);
    } finally {
      btn.disabled = false;
    }
  });
</script>
</body>
</html>`);
});

app.listen(PORT, () => console.log(`TramPLAR running on http://localhost:${PORT}`));
