import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // Dev-only convenience: proxy /api to your backend on localhost
    proxy: { "/api": { target: "http://localhost:8080", changeOrigin: true } }
  }
});
