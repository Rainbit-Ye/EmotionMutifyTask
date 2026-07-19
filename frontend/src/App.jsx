import React from 'react'
import Toolbar from './components/Toolbar'
import SequenceBar from './components/SequenceBar'
import Suggestions from './components/Suggestions'
import IconPalette from './components/IconPalette'
import Shortcuts from './components/Shortcuts'
import EmotionBadge from './components/EmotionBadge'
import {
  getCatalog, addIcon, commit, undo, reset, getTranslation, sid,
} from './api'

const THEME_KEY = 'aac-theme'

// 高频人称词常驻快捷按钮（I=我 / U=你），从目录取 label/has_image。
const SHORTCUT_IDS = ['I', 'U']

function flattenSuggestions(pred) {
  if (!pred) return []
  const cats = ['actions', 'entities', 'emotions', 'others']
  const out = []
  for (const c of cats) for (const it of (pred[c] || [])) out.push(it)
  out.sort((a, b) => (b.final_score || 0) - (a.final_score || 0))
  return out.slice(0, 12)
}

export default function App() {
  const [theme, setTheme] = React.useState(
    () => localStorage.getItem(THEME_KEY) || 'light'
  )
  const [catalog, setCatalog] = React.useState([])
  const [seq, setSeq] = React.useState([])        // [{id,label,hasImage}]
  const [translation, setTranslation] = React.useState('')
  const [suggest, setSuggest] = React.useState([])
  const [emotion, setEmotion] = React.useState(null)
  const [loading, setLoading] = React.useState(false)
  const [committing, setCommitting] = React.useState(false)
  const [error, setError] = React.useState('')
  const [toast, setToast] = React.useState('')

  const catMap = React.useMemo(() => {
    const m = new Map()
    for (const c of catalog) m.set(c.icon_id, c)
    return m
  }, [catalog])

  // 高频人称词常驻快捷按钮（图标可能不在目录，回退用 id+含义）
  const shortcuts = React.useMemo(() => SHORTCUT_IDS.map(id => {
    const c = catMap.get(id)
    return { id, label: (c && c.label) || id, hasImage: c ? c.has_image : undefined }
  }), [catMap])

  React.useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  React.useEffect(() => {
    getCatalog().then(setCatalog).catch(e => setError('加载图标目录失败：' + e.message))
  }, [])

  const toggleTheme = () => {
    const n = theme === 'dark' ? 'light' : 'dark'
    setTheme(n)
    localStorage.setItem(THEME_KEY, n)
  }

  const applyResp = (res) => {
    // 仅在确有【中文】缓存时才覆盖翻译，避免回退成英文图标 id / 闪烁成空白；
    // 异步翻译到达前，保持上一句的中文（或首次为空时由气泡显示"翻译中…"）。
    if (res.partial_translation) setTranslation(res.partial_translation)
    setSuggest(flattenSuggestions(res.next_icon_predictions))
    setEmotion(res.emotion || null)
    const seq = (res.current_sequence || []).map(id => {
      const c = catMap.get(id)
      return { id, label: (c && c.label) || id, hasImage: c ? c.has_image : undefined }
    })
    setSeq(seq)
  }

  // 后台异步翻译轮询：点击已即时返回预测，稍后取到高质量翻译补上。
  // 用 pollRef 防止连续点击时旧轮询覆盖新结果。
  const pollRef = React.useRef(0)
  const pollTranslation = React.useCallback(async () => {
    const myPoll = ++pollRef.current
    for (let i = 0; i < 30; i++) {           // 30 × 400ms ≈ 12s，稳妥覆盖 ~4.6s 的 8B 翻译
      if (pollRef.current !== myPoll) return     // 用户又点了，本次作废
      await new Promise(r => setTimeout(r, 400))
      try {
        const t = await getTranslation()
        const tx = (t && t.partial_translation) || ''
        if (tx) { setTranslation(tx); return }
      } catch (e) { /* 忽略瞬时错误 */ }
    }
  }, [])

  const handlePick = async (id) => {
    setError(''); setLoading(true)
    try {
      const res = await addIcon(id)
      if (res.error) setError(res.error)
      else {
        applyResp(res)
        if (res.translation_pending) pollTranslation()
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleUndo = async () => {
    setError(''); setLoading(true)
    try {
      const res = await undo()
      if (res.error) setError(res.error)
      else applyResp(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleCommit = async () => {
    setError(''); setCommitting(true)
    try {
      const res = await commit()
      if (res.error) setError(res.error)
      else {
        const full = (res.translation && res.translation.sentence) || translation
        setToast('整句：' + full + (res.trend ? '　·　趋势 ' + res.trend.trend : ''))
        setSeq([]); setSuggest([]); setTranslation('')
        setTimeout(() => setToast(''), 4000)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setCommitting(false)
    }
  }

  const handleReset = async () => {
    setError('')
    try { await reset() } catch (e) { setError(e.message) }
    setSeq([]); setSuggest([]); setTranslation(''); setEmotion(null)
  }

  return (
    <div className="app">
      <Toolbar
        theme={theme}
        onToggleTheme={toggleTheme}
        onReset={handleReset}
        sid={sid}
      />

      <Shortcuts items={shortcuts} onPick={handlePick} disabled={loading} />

      <SequenceBar
        sequence={seq}
        translation={translation}
        onUndo={handleUndo}
        onCommit={handleCommit}
        canUndo={seq.length > 0}
        committing={committing}
      />

      <EmotionBadge emotion={emotion} />

      {error && <div className="err">⚠ {error}</div>}

      <Suggestions items={suggest} onPick={handlePick} loading={loading} />

      <IconPalette catalog={catalog} onPick={handlePick} />

      {toast && <div className="toast">{toast}</div>}
    </div>
  )
}
