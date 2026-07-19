import React from 'react'

// 情绪徽标：当前情绪 + 模型预测下一情绪。
export default function EmotionBadge({ emotion }) {
  if (!emotion) return null
  return (
    <div className="emotion-row">
      <span className="badge">
        <span className="dot" />
        当前：{emotion.current || emotion.single || '—'}
      </span>
      {emotion.next && (
        <span className="badge next">
          <span className="dot" />
          预测下一：{emotion.next}
        </span>
      )}
    </div>
  )
}
