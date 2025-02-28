import { defineConfig } from "vite";

const API_URL = "http://127.0.0.1:5000"; // changer en fonction de l'API

export default defineConfig({
  server: {
    proxy: {
      "/output_feed": API_URL,
      "/input_feed": API_URL,
      "/set_params": {
        target: API_URL,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
