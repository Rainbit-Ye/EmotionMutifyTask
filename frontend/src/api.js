// API 客户端：统一带 X-Session-Id 头（每浏览器一个会话，状态隔离）。

const SID_KEY = 'aac_sid';
let sid = localStorage.getItem(SID_KEY);
if (!sid) {
  sid = (window.crypto && crypto.randomUUID)
    ? crypto.randomUUID()
    : 's' + Math.random().toString(36).slice(2);
  localStorage.setItem(SID_KEY, sid);
}

function heads() {
  return { 'Content-Type': 'application/json', 'X-Session-Id': sid };
}

export const iconUrl = (id) => `/api/icon/${encodeURIComponent(id)}`;

export async function getCatalog() {
  const r = await fetch('/api/catalog');
  if (!r.ok) throw new Error('catalog HTTP ' + r.status);
  return await r.json();
}

async function post(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: heads(),
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
  return data;
}

export const addIcon = (id) => post('/api/add', { icon_id: id });
export const commit = () => post('/api/commit', {});
export const undo = () => post('/api/undo', {});
export const reset = () => post('/api/reset', {});

// 后台异步翻译：点击已即时返回预测，稍后轮询取高质量翻译补上。
export async function getTranslation() {
  const r = await fetch('/api/translation', { headers: heads() });
  if (!r.ok) throw new Error('translation HTTP ' + r.status);
  return await r.json();
}

export { sid };
