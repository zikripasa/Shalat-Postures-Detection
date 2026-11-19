// Render SVG diagrams to PNG with the pre-installed Chromium (Playwright).
// Usage: NODE_PATH=$(npm root -g) node scripts/render_svg.cjs portfolio/assets/diagrams/*.svg
const { chromium } = require("playwright");
const fs = require("node:fs");

(async () => {
  const files = process.argv.slice(2);
  const browser = await chromium.launch();
  for (const f of files) {
    const svg = fs.readFileSync(f, "utf8");
    const w = Number(/width="(\d+)"/.exec(svg)[1]);
    const h = Number(/height="(\d+)"/.exec(svg)[1]);
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: 2 });
    await page.setContent(`<html><body style="margin:0;background:#fff">${svg}</body></html>`);
    await page.evaluate(() => document.fonts.ready);
    const out = f.replace(/\.svg$/, ".png");
    await page.screenshot({ path: out, clip: { x: 0, y: 0, width: w, height: h } });
    console.log("rendered", out);
    await page.close();
  }
  await browser.close();
})();
