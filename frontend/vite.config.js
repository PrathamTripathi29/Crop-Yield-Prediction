import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, every request to /api is forwarded to the FastAPI backend,
// so the React code can call "/api/..." without worrying about ports or CORS.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
});
