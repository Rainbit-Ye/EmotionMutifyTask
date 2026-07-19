import React from 'react'

// 顶栏：标题 + 主题切换 + 重置 + 会话 ID。
export default function Toolbar({ theme, onToggleTheme, onReset, onResetDone, sid }) {
  return (
    <div className="toolbar">
      <h1>AAC 沟通板</h1>
      <span className="sub">选词预测 + 自然语言翻译</span>
      <span className="spacer" />
      <span className="sid" title="本次会话 ID（状态隔离）">sid: {sid ? sid.slice(0, 8) : '—'}</span>
      <button className="btn btn-ghost" onClick={onReset} title="清空当前序列">重置</button>
      <button className="btn" onClick={onToggleTheme}>
        {theme === 'dark' ? '☀ 浅色' : '🌙 深色'}
      </button>
    </div>
  )
}
