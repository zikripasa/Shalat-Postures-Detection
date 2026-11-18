import { defineConfig } from "@playwright/test";

// End-to-end tests against the REAL backend (model bundle loaded) and the built frontend.
// Build first with:  VITE_API_BASE_URL=http://127.0.0.1:8000 npm run build
export default defineConfig({
  testDir: "./e2e",
  timeout: 240_000,
  expect: { timeout: 20_000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: "http://127.0.0.1:4173", trace: "off", viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 },
  webServer: [
    {
      command: "cd .. && .venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000",
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: true,
      timeout: 120_000,
    },
    { command: "npx vite preview --host 127.0.0.1 --port 4173 --strictPort", url: "http://127.0.0.1:4173", reuseExistingServer: true, timeout: 60_000 },
  ],
  projects: [{ name: "chromium", use: { browserName: "chromium" } }],
});
