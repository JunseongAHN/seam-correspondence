import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
// GitHub Pages serves a project site from /<repo>/, so a production build has to be
// based there or every asset 404s.  Dev stays at / so the local URL is unchanged.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/seam-correspondence/' : '/',
  plugins: [react()],
}))
