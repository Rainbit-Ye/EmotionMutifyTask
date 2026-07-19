import React from 'react'
import IconTile from './IconTile'

// 模型推荐（下一个图标）：来自 SASRec 预测的 top-k，带分数条，可点选。
export default function Suggestions({ items, onPick, loading }) {
  return (
    <div className="card">
      <h2>模型推荐 · 下一个图标</h2>
      {loading && <div className="loading">预测中…</div>}
      {!loading && (!items || items.length === 0) && (
        <div className="loading">先选一个图标，这里会给出建议。</div>
      )}
      {!loading && items && items.length > 0 && (
        <div className="grid suggest">
          {items.map((it, i) => (
            <IconTile
              key={it.icon_id + ':' + i}
              id={it.icon_id}
              label={it.label}
              score={it.final_score}
              hasImage={it.has_image}
              onClick={() => onPick(it.icon_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
