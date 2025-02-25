import { defineConfig } from "vite";

const API_URL = "http://localhost:5000"; // changer en fonction de l'API

export default defineConfig({
  server: {
    proxy: {
      "/output_feed": API_URL,
      "/input_feed": API_URL,
      "/prompt": {
        target: API_URL,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
