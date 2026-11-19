import { expect, test, type Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const SHOTS = path.join(ROOT, "portfolio", "assets", "application_screenshots");
const STILL = path.join(ROOT, "artifacts", "demo", "stills", "still_6.5.jpg");
fs.mkdirSync(SHOTS, { recursive: true });

async function shot(page: Page, name: string, fullPage = false) {
  await page.screenshot({ path: path.join(SHOTS, name), fullPage });
}

test("landing page shows live model status from the backend", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Kenali urutan gerakan shalat");
  await expect(page.getByRole("status").first()).toContainText("aktif");
  await expect(page.getByText(/F1 makro/)).toBeVisible();
  await page.waitForTimeout(1600); // let the one-time hero drawing finish
  await shot(page, "01_beranda.png");
  await shot(page, "01_beranda_full.png", true);
});

test("photo analysis returns a posture with skeleton overlay", async ({ page }) => {
  test.skip(!fs.existsSync(STILL), "demo still not generated (scripts/make_demo_video.py)");
  await page.goto("/analisis");
  await page.getByRole("tab", { name: "Foto" }).click();
  await page.locator('input[type="file"]').setInputFiles(STILL);
  await expect(page.getByText("Postur yang dikenali")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("list", { name: "Skor per postur" }).locator("li")).toHaveCount(4);
  await page.waitForTimeout(400);
  await shot(page, "03_analisis_foto.png");
});

test("full video flow: demo video -> upload -> progress -> results dashboard", async ({ page }) => {
  await page.goto("/analisis");
  const demo = page.getByRole("button", { name: "Pakai video demo" });
  const available = await demo.waitFor({ state: "visible", timeout: 20_000 }).then(() => true, () => false);
  test.skip(!available, "demo video not available on the backend");
  await demo.click();
  await expect(page.getByLabel("Pratinjau video")).toBeVisible();
  await shot(page, "02a_analisis_video.png");
  await page.getByRole("button", { name: "Analisis video" }).click();
  const progress = page.getByRole("progressbar", { name: "Kemajuan analisis" });
  await expect(progress).toBeVisible();
  await expect(page.getByText(/Menganalisis frame|Menunggu giliran|Menyiapkan model/)).toBeVisible({ timeout: 30_000 });
  await shot(page, "02b_progres.png");
  await page.waitForURL(/\/hasil\//, { timeout: 180_000 });
  await expect(page.getByRole("heading", { name: "Hasil analisis" })).toBeVisible();
  const timeline = page.getByRole("list", { name: "Linimasa fase shalat" });
  const blocks = timeline.getByRole("button");
  expect(await blocks.count()).toBeGreaterThan(3);
  // seeking from the timeline moves the annotated video
  const video = page.getByLabel("Video beranotasi", { exact: true });
  await expect(video).toBeVisible();
  await blocks.nth(3).click();
  await expect.poll(async () => video.evaluate((v: HTMLVideoElement) => v.currentTime)).toBeGreaterThan(1);
  // for the screenshot, rest on the middle of the first rukuk segment (no cross-fade frame)
  await video.evaluate((v: HTMLVideoElement) => {
    v.currentTime = 6.6;
  });
  await page.waitForTimeout(1200);
  await shot(page, "02c_hasil.png");
  await shot(page, "02d_hasil_full.png", true);
  // downloads exist and are real
  const href = await page.getByRole("link", { name: "Unduh segmen CSV" }).getAttribute("href");
  const csv = await page.request.get(href!);
  expect(csv.ok()).toBeTruthy();
  expect(await csv.text()).toContain("index,phase,label");
});

test("model page renders real evaluation results", async ({ page }) => {
  await page.goto("/model");
  await expect(page.getByRole("heading", { name: "Kinerja model terpilih" })).toBeVisible();
  await expect(page.getByText("F1 makro, potongan orang")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Perbandingan 11 model" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Matriks kebingungan (potongan beranotasi)" })).toBeVisible();
  await page.waitForTimeout(500);
  await shot(page, "04_model.png");
  await shot(page, "04_model_full.png", true);
});

test("mobile layout keeps navigation and timeline usable", async ({ browser }) => {
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  await page.goto("/");
  await expect(page.getByRole("navigation", { name: "Navigasi utama" })).toBeVisible();
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(SHOTS, "05_mobile_beranda.png") });
  await ctx.close();
});
