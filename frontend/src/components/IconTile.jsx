import React from 'react'
import { iconUrl } from '../api'

// 单个图标瓦片：优先显示真实 PNG，加载失败或未提供图片时回退文字。
export default function IconTile({ id, label, score = null, hasImage, onClick, compact = false }) {
  const [err, setErr] = React.useState(false)
  const showImg = hasImage !== false && !err && id
  return (
    <button className="tile" onClick={onClick} title={id}>
      <div className="tile-img">
        {showImg ? (
          <img
            src={iconUrl(id)}
            alt={label || id}
            loading="lazy"
            onError={() => setErr(true)}
          />
        ) : (
          <span className="tile-label">{(label || id || '?').toString().slice(0, 14)}</span>
        )}
      </div>
      {!compact && <div className="tile-name">{label || id}</div>}
      {score != null && (
        <div className="score-bar">
          <div
            className="score-fill"
            style={{ width: Math.max(4, Math.min(100, score * 100)) + '%' }}
          />
        </div>
      )}
    </button>
  )
}
