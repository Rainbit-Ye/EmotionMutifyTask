import React from 'react'
import IconTile from './IconTile'

// 当前序列：翻译气泡 + 已选图标 chips + Undo/Commit 操作。
export default function SequenceBar({ sequence, translation, onUndo, onCommit, canUndo, committing }) {
  const hasItems = sequence && sequence.length > 0
  return (
    <div className="card">
      <h2>当前句子</h2>
      <div className="bubble">{translation || (hasItems ? <span style={{ color: 'var(--text-soft)' }}>翻译中…</span> : <span style={{ color: 'var(--text-soft)' }}>点击下方图标开始…</span>)}</div>
      {hasItems && (
        <div className="chips">
          {sequence.map((it, i) => (
            <span className="chip" key={i}>
              <IconTile id={it.id} label={it.label} hasImage={it.hasImage} compact onClick={() => {}} />
              <span>{it.label || it.id}</span>
            </span>
          ))}
        </div>
      )}
      <div className="actions-row">
        <button className="btn" disabled={!canUndo} onClick={onUndo}>↶ 撤销</button>
        <button className="btn btn-primary" disabled={!hasItems || committing} onClick={onCommit}>
          {committing ? '生成中…' : '✓ 完成整句'}
        </button>
      </div>
    </div>
  )
}
