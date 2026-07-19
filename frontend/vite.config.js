import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 构建产物由 FastAPI 在根路径 "/" 托管（base 默认 "/" 即可）
export default defineConfig({
  plugins: [react()],
  server: { host: true },
  build: { outDir: 'dist', emptyOutDir: true },
})
