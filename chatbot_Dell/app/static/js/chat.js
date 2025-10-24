// static/js/chat.js
const chat = document.getElementById('chat');
const form = document.getElementById('form');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statusPill = document.getElementById('status');
const clearBtn = document.getElementById('clear');

const FALLBACK_GREETING =
  "My name is Kevin. I’m the virtual assistant for our company’s products. What can I help you with?";

let pendingConfirm = null; // remember follow-up to send after Yes

function setBusy(b) {
  if (sendBtn) sendBtn.disabled = b;
  if (statusPill) statusPill.textContent = b ? 'Thinking…' : 'Ready';
}

function append(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = `<span class="meta">${role}</span><br>${renderText(text)}`;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function renderText(s) {
  const esc = (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  return esc.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br>');
}

async function greet() {
  try {
    const r = await fetch('/greet');
    const data = r.ok ? await r.json() : null;
    append('assistant', data?.answer || FALLBACK_GREETING);
  } catch {
    append('assistant', FALLBACK_GREETING);
  }
}

// Enter = send; Shift+Enter = newline
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});

clearBtn?.addEventListener('click', () => {
  chat.innerHTML = '';
  greet();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  let q = (input.value || '').trim();
  if (!q) return;

  // consume Yes/No confirmations
  if (pendingConfirm) {
    const t = q.toLowerCase();
    if (['yes','y','ok','okay','sure'].includes(t)) {
      q = pendingConfirm;            // send prepared follow-up
    } else if (['no','n'].includes(t)) {
      pendingConfirm = null;
      append('assistant', 'No problem — please tell me the exact model (e.g., "Dell Vostro 3420").');
      input.value = '';
      return;
    }
    pendingConfirm = null;
  }

  append('user', q);
  input.value = '';
  setBusy(true);

  try {
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q })
    });
    const raw = await res.text();
    let data; try { data = JSON.parse(raw); } catch {}

    if (!res.ok) {
      append('assistant', `Error ${res.status}: ${raw || res.statusText}`);
      return;
    }

    if (data?.mode === 'CONFIRM' && data?.confirm_next) {
      pendingConfirm = data.confirm_next; // arm confirmation
    }
    append('assistant', data?.answer || '(no answer)');
  } catch (err) {
    append('assistant', 'Network error: ' + err.message);
  } finally {
    setBusy(false);
  }
});

document.addEventListener('DOMContentLoaded', greet);
