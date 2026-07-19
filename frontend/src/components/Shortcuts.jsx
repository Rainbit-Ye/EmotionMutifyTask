import React from 'react'

// 高频人称词常驻快捷按钮（I=我 / U=你），点一下即加入当前序列。
// 与从调色板选图标走同一 handlePick 流程。
const MEANING = { I: '我', U: '你' }

export default function Shortcuts({ items, onPick, disabled }) {
  if (!items || items.length === 0) return null
  return (
    <div className="card shortcuts">
      <h2>高频人称 · 一点即选</h2>
      <div className="sc-row">
        {items.map(it => (
          <button
            key={it.id}
            className="sc-btn"
            disabled={disabled}
            title={`${it.id}（${MEANING[it.id] || it.label || it.id}）`}
            onClick={() => onPick(it.id)}
          >
            <span className="sc-sym">{it.id}</span>
            <span className="sc-label">{MEANING[it.id] || it.label || it.id}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
