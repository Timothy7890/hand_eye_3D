import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const BACKEND = 'http://localhost:8132'

export default defineConfig({
  root: fileURLToPath(new URL('./mount-diagnostics', import.meta.url)),
  cacheDir: fileURLToPath(new URL('./node_modules/.vite-mount-diagnostics', import.meta.url)),
  plugins: [vue()],
  server: {
    host: '0.0.0.0',
    port: 7015,
    strictPort: true,
    proxy: {
      '/api': {
        target: BACKEND,
        timeout: 120000,
        proxyTimeout: 120000,
      },
      '/assets': {
        target: BACKEND,
        timeout: 120000,
        proxyTimeout: 120000,
      },
    },
  },
  build: {
    outDir: fileURLToPath(new URL('./dist-mount-diagnostics', import.meta.url)),
    emptyOutDir: true,
  },
})
