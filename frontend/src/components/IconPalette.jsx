import React from 'react'
import IconTile from './IconTile'

// 全部图标调色板：搜索 + 分类过滤，点击即选入当前序列。
export default function IconPalette({ catalog, onPick }) {
  const [q, setQ] = React.useState('')
  const [cat, setCat] = React.useState('all')

  const cats = React.useMemo(() => {
    const s = new Set()
    for (const c of catalog || []) if (c.semantic_type) s.add(c.semantic_type)
    return ['all', ...Array.from(s).sort()]
  }, [catalog])

  const filtered = React.useMemo(() => {
    const kw = q.trim().toLowerCase()
    return (catalog || []).filter(c => {
      if (cat !== 'all' && c.semantic_type !== cat) return false
      if (kw && !(c.label || '').toLowerCase().includes(kw) && !(c.icon_id || '').toLowerCase().includes(kw)) return false
      return true
    })
  }, [catalog, q, cat])

  return (
    <div className="card">
      <h2>全部图标 · 点击选入</h2>
      <div className="palette-controls">
        <input
          className="input"
          placeholder="搜索标签 / id…"
          value={q}
          onChange={e => setQ(e.target.value)}
        />
        <select className="select" value={cat} onChange={e => setCat(e.target.value)}>
          {cats.map(c => (
            <option key={c} value={c}>{c === 'all' ? '全部分类' : c}</option>
          ))}
        </select>
        <span className="sub" style={{ color: 'var(--text-soft)', fontSize: 12 }}>
          {filtered.length} 个
        </span>
      </div>
      <div className="grid palette">
        {filtered.map(c => (
          <IconTile
            key={c.icon_id}
            id={c.icon_id}
            label={c.label}
            hasImage={c.has_image}
            onClick={() => onPick(c.icon_id)}
          />
        ))}
      </div>
    </div>
  )
}
