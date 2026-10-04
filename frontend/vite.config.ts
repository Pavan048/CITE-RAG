import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev-only proxy: forwards API calls to the FastAPI backend (app/main.py, run separately via
// `uvicorn app.main:app --port 8010`) so `npm run dev` gets instant HMR against the real running
// backend without a build step. The production build (`npm run build` -> dist/) is served
// directly by that same FastAPI app, which needs no proxy since it IS the origin at that point.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/v1': 'http://localhost:8010',
    },
  },
})
