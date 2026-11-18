import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const { TARZ_BACKEND_URL = "http://localhost:8000" } = loadEnv(mode, process.cwd(), "");
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": { target: TARZ_BACKEND_URL, changeOrigin: true },
        "/uploads": { target: TARZ_BACKEND_URL, changeOrigin: true },
      },
    },
  };
});
