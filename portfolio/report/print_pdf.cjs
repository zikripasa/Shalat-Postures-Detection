// Print an HTML file to PDF with Chromium (Playwright).
// Usage: NODE_PATH=$(npm root -g) node print_pdf.cjs input.html output.pdf [cover|body]
const { chromium } = require("playwright");
const path = require("node:path");

(async () => {
  const [input, output, kind = "body"] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("file://" + path.resolve(input), { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  const footer = `<div style="width:100%;font-family:'Plus Jakarta Sans',sans-serif;font-size:7.5px;color:#6b7280;padding:0 18mm;display:flex;justify-content:space-between;">
      <span>Salat Vision — Technical Report</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>`;
  await page.pdf({
    path: output,
    format: "A4",
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: kind === "body",
    headerTemplate: "<div></div>",
    footerTemplate: footer,
  });
  await browser.close();
  console.log("printed", output);
})();
