// Build portfolio/presentation/Salat_Vision_Portfolio_Deck.pptx from the project's real data.
//   cd portfolio/presentation && npm ci && node build_deck.cjs
// Every chart, table and timeline below is driven by portfolio/assets/data/deck_data.json,
// which scripts export from artifacts/metrics (no hand-typed results).
const path = require("node:path");
const fs = require("node:fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const A = (...p) => path.join(ROOT, "portfolio", "assets", ...p);
const D = JSON.parse(fs.readFileSync(A("data", "deck_data.json"), "utf8"));
const CFG = JSON.parse(fs.readFileSync(path.join(ROOT, "portfolio", "report", "report_config.json"), "utf8"));
const OUT = path.join(__dirname, "Salat_Vision_Portfolio_Deck.pptx");
const SKILL_THEME = process.env.PPTX_APPLY_THEME; // optional path to apply_theme.js

const THEME = {
  name: "Salat Vision",
  headFontFace: "Plus Jakarta Sans",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0E1B33", lt1: "FFFFFF", dk2: "4A5565", lt2: "F3F5F4",
    accent1: "0F766E", accent2: "B8892C", accent3: "2A78D6", accent4: "EB6834", accent5: "1BAF7A", accent6: "E87BA4",
    hlink: "0F766E", folHlink: "6B7280",
  },
};
// data colours (validated categorical palette; fixed, not theme-driven)
const PH = { berdiri: "EDA100", rukuk: "1BAF7A", iktidal: "EB6834", sujud: "2A78D6", duduk_antara_sujud: "008300", tasyahud: "E87BA4" };
const PH_NAME = { berdiri: "Berdiri", rukuk: "Rukuk", iktidal: "Iktidal", sujud: "Sujud", duduk_antara_sujud: "Duduk antara sujud", tasyahud: "Tasyahud" };
const UNC = "CFCCC2";
const GRID = "E1E0D9";
const DEEMPH = "C9CCD1";
const BLUE = "2A78D6";
const SEQ = ["F4F8FE", "CDE2FB", "9EC5F4", "6DA7EC", "3987E5", "256ABF", "184F95", "0D366B"];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = CFG.author;
pres.title = "Salat Vision — portfolio deck";
pres.subject = "Prayer-posture recognition: data, models, temporal decoding, product";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333;

// ------------------------------------------------------------------ layouts
const footer = (dark) => [
  { text: { text: "Salat Vision", options: { x: 0.6, y: 7.02, w: 4, h: 0.3, fontSize: 10, color: dark ? "AEB7C2" : "6B7280", isTextBox: true } } },
];
pres.defineSlideMaster({
  title: "COVER",
  background: { color: "0E1B33" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 6.2, h: 1.2, fontSize: 60, bold: true, color: "FFFFFF", fontFace: THEME.headFontFace, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.95, w: 6.0, h: 1.4, fontSize: 20, color: "D8DEE6", valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: "0E1B33", fontFace: THEME.headFontFace, valign: "top", align: "left", margin: 0 }, text: "" } },
    ...footer(false),
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: "6B7280", align: "right" },
});
pres.defineSlideMaster({
  title: "CLOSING",
  background: { color: "0E1B33" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.85, fontSize: 32, bold: true, color: "FFFFFF", fontFace: THEME.headFontFace, valign: "top", align: "left", margin: 0 }, text: "" } },
    ...footer(true),
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: "AEB7C2", align: "right" },
});

// ------------------------------------------------------------------ helpers
let n = 0;
const id = (s) => `${s}_${++n}`;
function txt(slide, text, o) {
  slide.addText(text, { isTextBox: true, margin: 0, fontFace: THEME.bodyFontFace, color: "4A5565", fontSize: 14, valign: "top", objectName: id("text"), ...o });
}
function box(slide, o) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { rectRadius: 0.08, line: { color: "DFE3E1", width: 1 }, fill: { color: "FFFFFF" }, objectName: id("box"), ...o });
}
function arrow(slide, x1, y1, x2, y2, color = "0E1B33") {
  slide.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color, width: 1.5, endArrowType: "triangle" }, objectName: id("arrow") });
}
function stat(slide, x, y, w, value, label, dark = false) {
  txt(slide, value, { x, y, w, h: 0.75, fontSize: 40, bold: true, color: dark ? "FFFFFF" : "0E1B33", fontFace: THEME.headFontFace, valign: "bottom" });
  txt(slide, label, { x, y: y + 0.8, w, h: 0.55, fontSize: 13, color: dark ? "C9D1D9" : "4A5565" });
}
function mark(slide, x, y, s, color = "B8892C") {
  // the product mark: two interlaced squares (khatam)
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: s, h: s, fill: { type: "none" }, line: { color, width: 1.75 }, objectName: id("mark") });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: s, h: s, rotate: 45, fill: { type: "none" }, line: { color, width: 1.75 }, objectName: id("mark") });
}
function phaseBand(slide, x, y, w, h, labels = true, dark = false) {
  const order = ["berdiri", "rukuk", "iktidal", "sujud", "duduk_antara_sujud", "sujud", "tasyahud"];
  const weights = [1.1, 0.9, 0.9, 0.9, 1.5, 0.9, 1.1];
  const tot = weights.reduce((a, b) => a + b, 0);
  let cx = x;
  order.forEach((p, i) => {
    const bw = (weights[i] / tot) * w - 0.04;
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: cx, y, w: bw, h, rectRadius: 0.04, fill: { color: PH[p] }, line: { type: "none" }, objectName: id("phase") });
    if (labels) txt(slide, PH_NAME[p], { x: cx + 0.06, y: y + h + 0.06, w: bw, h: 0.3, fontSize: 11, color: dark ? "C9D1D9" : "4A5565" });
    cx += bw + 0.04;
  });
}
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const f2 = (x) => x.toFixed(2);
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mj-lt" };
const chartQuiet = {
  ...chartText,
  catAxisLabelColor: "4A5565", valAxisLabelColor: "6B7280", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
  valGridLine: { color: GRID, size: 0.75 }, catGridLine: { style: "none" }, valAxisLineShow: false,
  dataLabelColor: "4A5565", dataLabelFontSize: 11, legendFontSize: 12, legendColor: "4A5565",
};

const sm = D.selected_metrics;

// ================================================================== 1. cover
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Introduction" });
  mark(s, 0.62, 0.62, 0.32);
  txt(s, "Technical portfolio project", { x: 1.15, y: 0.6, w: 5, h: 0.4, fontSize: 14, color: "D8B76A", bold: true, valign: "middle" });
  s.addText("Salat Vision", { placeholder: "title" });
  s.addText("Recognising the visible postures of Islamic prayer in video and turning them into an honest, time-stamped timeline of prayer phases.", { placeholder: "body" });
  s.addShape(pres.shapes.RECTANGLE, { x: 7.0, y: 1.0, w: 5.75, h: 3.6, fill: { color: "FFFFFF" }, line: { color: "3B4A66", width: 1 }, objectName: "hero_frame" });
  s.addImage({ path: A("application_screenshots", "02c_hasil.png"), x: 7.03, y: 1.03, w: 5.69, h: 3.54, sizing: { type: "cover", w: 5.69, h: 3.54 }, altText: "Salat Vision results dashboard with phase timeline", objectName: "hero_shot" });
  phaseBand(s, 0.6, 5.05, 12.13, 0.32, true, true);
  txt(s, [{ text: "Author  ", options: { bold: true, color: "FFFFFF" } }, { text: `${CFG.author}   ${CFG.role}`, options: { color: "C9D1D9" } }], { x: 0.6, y: 6.25, w: 8, h: 0.35, fontSize: 13 });
  txt(s, "Data audit · model research · temporal decoding · FastAPI · React", { x: 0.6, y: 6.65, w: 8, h: 0.35, fontSize: 12, color: "AEB7C2" });
  s.addNotes("Salat Vision turns a recording of a person praying into a timeline of prayer phases. This deck walks through the data problem, the modelling decisions, the evaluation and the product. Replace the author fields with your details.");
}

// ================================================================== 2. problem
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("A prayer is a sequence, not a single pose", { placeholder: "title" });
  txt(s, "One rakaat has six phases, but a single frame can only show four distinct postures.", { x: 0.6, y: 1.25, w: 11.5, h: 0.45, fontSize: 16 });
  const phases = ["berdiri", "rukuk", "iktidal", "sujud", "duduk_antara_sujud", "tasyahud"];
  const seen = { berdiri: "Seen; named by context", rukuk: "Seen directly", iktidal: "Looks like standing: inferred", sujud: "Seen directly", duduk_antara_sujud: "Generic sitting: inferred", tasyahud: "Generic sitting: inferred" };
  phases.forEach((p, i) => {
    const x = 0.6 + i * 2.04;
    box(s, { x, y: 1.95, w: 1.88, h: 2.1, fill: { color: "F3F5F4" }, line: { type: "none" } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 2.15, w: 0.5, h: 0.5, fill: { color: PH[p] }, line: { type: "none" }, objectName: id("dot") });
    txt(s, `${i + 1}`, { x: x + 0.2, y: 2.15, w: 0.5, h: 0.5, fontSize: 16, bold: true, align: "center", valign: "middle", color: ["sujud", "duduk_antara_sujud"].includes(p) ? "FFFFFF" : "0B0B0B" });
    txt(s, PH_NAME[p], { x: x + 0.2, y: 2.78, w: 1.6, h: 0.62, fontSize: 15, bold: true, color: "0E1B33", fontFace: THEME.headFontFace, valign: "top" });
    txt(s, seen[p], { x: x + 0.2, y: 3.45, w: 1.6, h: 0.55, fontSize: 12 });
  });
  stat(s, 0.6, 4.55, 3.6, "4 → 6", "postures annotated → phases the product must report");
  stat(s, 4.6, 4.55, 3.9, `${D.temporal.segment.raw.seg_ratio.toFixed(1)}×`, "too many segments when every frame is labelled on its own");
  stat(s, 8.9, 4.55, 3.8, "0", "religious judgements: the system names postures, it does not rate prayers");
  s.addNotes("Two pairs of phases look identical in one frame: standing vs. iktidal, and the two kinds of sitting. Frame-by-frame labels also flicker: on our conservative benchmark raw labels produce over four times as many segments as there really are. So the system is split into a posture classifier and a temporal decoder.");
}

// ================================================================== 3. product
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
  s.addText("A private, honest timeline of prayer movements", { placeholder: "title" });
  const cols = [
    ["Who it is for", ["Learners practising the order of movements", "Teachers of children's prayer classes", "Researchers in activity recognition"], "1BAF7A"],
    ["What it does", ["Upload or record a video; get phases with times and durations", "Uncertain segments flagged, not hidden", "Photo check, JSON/CSV report, annotated video"], "2A78D6"],
    ["Principles", ["Runs locally; uploads deleted after analysis", "Says what is seen and what is inferred", "Never judges whether a prayer is valid"], "EDA100"],
  ];
  cols.forEach(([h, items, col], i) => {
    const x = 0.6 + i * 2.75;
    s.addShape(pres.shapes.OVAL, { x, y: 1.5, w: 0.32, h: 0.32, fill: { color: col }, line: { type: "none" }, objectName: id("dot") });
    txt(s, h, { x: x + 0.45, y: 1.48, w: 2.2, h: 0.4, fontSize: 18, bold: true, color: "0E1B33", fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })), { x, y: 2.05, w: 2.5, h: 3.2, fontSize: 14, paraSpaceAfter: 8 });
  });
  s.addImage({ path: A("application_screenshots", "02b_progres.png"), x: 8.95, y: 1.45, w: 3.78, h: 2.36, altText: "Analysis workspace with upload progress", objectName: "shot_progress" });
  s.addImage({ path: A("application_screenshots", "03_analisis_foto.png"), x: 8.95, y: 4.0, w: 3.78, h: 2.36, altText: "Photo analysis with skeleton overlay", objectName: "shot_photo" });
  txt(s, "Indonesian interface: Analisis (video, camera, photo) and the progress the server really reports.", { x: 0.6, y: 5.6, w: 7.9, h: 0.6, fontSize: 12, color: "6B7280" });
  s.addNotes("The audience is people learning or teaching the sequence of movements. Three product principles shaped every screen: local processing, visible uncertainty, and a clear boundary between posture recognition and religious judgement.");
}

// ================================================================== 4. dataset
pres.addSection({ title: "Data and method" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText(`${D.dataset.images} images, ${D.dataset.instances} people, ${D.dataset.groups} independent groups`, { placeholder: "title" });
  stat(s, 0.6, 1.3, 2.6, `${D.dataset.instances}`, "annotated people (4 classes)");
  stat(s, 3.3, 1.3, 2.6, `${D.dataset.groups}`, "independent capture groups after leakage review");
  const classes = ["qiyam", "ruku", "sujud", "julus"];
  const cname = { qiyam: "Standing", ruku: "Bowing", sujud: "Prostration", julus: "Sitting" };
  s.addChart(pres.charts.BAR, [
    { name: "Annotated people", labels: classes.map((c) => cname[c]), values: classes.map((c) => D.dataset.per_class[c]) },
    { name: "Independent groups", labels: classes.map((c) => cname[c]), values: classes.map((c) => D.dataset.groups_per_class[c]) },
  ], {
    x: 6.3, y: 1.25, w: 6.45, h: 3.0, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [BLUE, "EB6834"], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    showLegend: true, legendPos: "b", valAxisMinVal: 0, valAxisMaxVal: 14, valAxisMajorUnit: 2,
    showTitle: true, title: "Class balance: people vs. independent groups", titleFontSize: 14, titleColor: "0E1B33", ...chartQuiet,
  });
  const findings = [
    ["Hidden duplicates", "A web image is the same video frame as a lab file; another comes from the same video as two sitting frames."],
    ["Source confounds class", "Sitting comes almost only from the authors' own captures: a shortcut a pixel model can learn."],
    ["Parser traps", "Spelling variants, a polygon-shaped box, 2 orphan XML files and 4 unannotated images."],
  ];
  findings.forEach(([h, t], i) => {
    const x = 0.6 + i * 4.1;
    box(s, { x, y: 4.6, w: 3.9, h: 1.9, fill: { color: "F3F5F4" }, line: { type: "none" } });
    txt(s, h, { x: x + 0.25, y: 4.8, w: 3.4, h: 0.4, fontSize: 16, bold: true, color: "0E1B33", fontFace: THEME.headFontFace });
    txt(s, t, { x: x + 0.25, y: 5.25, w: 3.45, h: 1.15, fontSize: 13 });
  });
  s.addNotes("The provided archive is the public sample of the Salat Postures Dataset. The audit found cross-source duplicates and a strong source-class confound, so every split in the project is a stratified group split over 29 groups, not over images.");
}

// ================================================================== 5. architecture
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText("End-to-end architecture", { placeholder: "title" });
  const cols = [
    ["Data", ["Safe ZIP extraction", "Audit + leakage groups", "Group k-fold splits"]],
    ["Experiments", ["Pose + CNN feature cache", "11 models, nested CV", "Bootstrap CIs"]],
    ["Model bundle", ["ONNX embedder", "scikit-learn heads", "Manifest + SHA-256"]],
    ["FastAPI", ["Async video jobs", "Validation + cleanup", "503 if model missing"]],
    ["React app", ["Upload, camera, photo", "Phase timeline", "Model transparency"]],
  ];
  const cw = 2.2, gap = 0.27, y0 = 1.55;
  cols.forEach(([h, items], i) => {
    const x = 0.6 + i * (cw + gap);
    const key = i === 2;
    box(s, { x, y: y0, w: cw, h: 3.4, fill: { color: key ? "E3F1EF" : "F3F5F4" }, line: key ? { color: "0F766E", width: 1.5 } : { type: "none" } });
    txt(s, `${i + 1}`, { x: x + 0.2, y: y0 + 0.2, w: 0.5, h: 0.35, fontSize: 13, bold: true, color: "B8892C" });
    txt(s, h, { x: x + 0.2, y: y0 + 0.55, w: cw - 0.3, h: 0.45, fontSize: 17, bold: true, color: "0E1B33", fontFace: THEME.headFontFace });
    items.forEach((t, k) => {
      box(s, { x: x + 0.15, y: y0 + 1.2 + k * 0.7, w: cw - 0.3, h: 0.56 });
      txt(s, t, { x: x + 0.27, y: y0 + 1.2 + k * 0.7, w: cw - 0.5, h: 0.56, fontSize: 13, color: "0E1B33", valign: "middle" });
    });
    if (i < cols.length - 1) arrow(s, x + cw + 0.02, y0 + 1.7, x + cw + gap - 0.02, y0 + 1.7);
  });
  txt(s, [
    { text: "One configuration file and one model bundle are the contract between training and serving. ", options: { color: "0E1B33", bold: true } },
    { text: "Serving needs no PyTorch: ONNX Runtime + MediaPipe + scikit-learn, with every file checksummed before it is loaded.", options: {} },
  ], { x: 0.6, y: 5.3, w: 12.1, h: 0.8, fontSize: 15 });
  s.addNotes("Five layers. The experiments produce a model bundle with a manifest; the API verifies each file's SHA-256 before deserialising it, and refuses to predict without a valid bundle instead of returning placeholder results.");
}

// ================================================================== 6. method
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data and method" });
  s.addText("Two complementary views of every person", { placeholder: "title" });
  const step = (x, y, w, h, t, sub, style = {}) => {
    box(s, { x, y, w, h, ...style });
    txt(s, t, { x: x + 0.15, y: y + 0.1, w: w - 0.25, h: 0.35, fontSize: 14, bold: true, color: "0E1B33" });
    if (sub) txt(s, sub, { x: x + 0.15, y: y + 0.45, w: w - 0.25, h: h - 0.5, fontSize: 12 });
  };
  step(0.6, 2.05, 1.75, 1.0, "Video frame", "5 fps via FFmpeg");
  step(2.65, 2.05, 1.95, 1.0, "Find person", "EfficientDet-Lite0");
  step(4.9, 2.05, 1.6, 1.0, "Crop", "box + 15%");
  arrow(s, 2.37, 2.55, 2.63, 2.55); arrow(s, 4.62, 2.55, 4.88, 2.55);
  const tealLine = { line: { color: "0F766E", width: 1.5 } };
  const blueLine = { line: { color: "2A78D6", width: 1.5 } };
  step(6.95, 1.35, 2.05, 1.0, "MediaPipe Pose", "33 landmarks", tealLine);
  step(9.25, 1.35, 1.9, 1.0, "19 features", "angles, heights", tealLine);
  step(6.95, 2.75, 2.05, 1.0, "MobileNetV3", "frozen, ONNX", blueLine);
  step(9.25, 2.75, 1.9, 1.0, "960-d embedding", "letterboxed", blueLine);
  arrow(s, 6.52, 2.55, 6.93, 1.85); arrow(s, 6.52, 2.55, 6.93, 3.25);
  arrow(s, 9.02, 1.85, 9.23, 1.85); arrow(s, 9.02, 3.25, 9.23, 3.25);
  step(11.45, 2.05, 1.3, 1.0, "Fusion", "mean of 2 LR", { fill: { color: "E3F1EF" }, line: { color: "0F766E", width: 1.5 } });
  arrow(s, 11.17, 1.85, 11.43, 2.4); arrow(s, 11.17, 3.25, 11.43, 2.7);
  // sequence row
  step(0.6, 4.3, 3.0, 1.05, "Scores over time", "4 postures per frame");
  step(3.95, 4.3, 3.3, 1.05, "HMM segmentation", "7 states, prayer grammar", { fill: { color: "E3F1EF" }, line: { color: "0F766E", width: 1.5 } });
  step(7.6, 4.3, 2.8, 1.05, "Local naming", "Iktidal, Tasyahud by context", { fill: { color: "E3F1EF" }, line: { color: "0F766E", width: 1.5 } });
  step(10.75, 4.3, 2.0, 1.05, "6 phases", "+ uncertainty flag");
  arrow(s, 3.62, 4.82, 3.93, 4.82); arrow(s, 7.27, 4.82, 7.58, 4.82); arrow(s, 10.42, 4.82, 10.73, 4.82);
  txt(s, [
    { text: "Why pose: ", options: { bold: true, color: "0E1B33" } }, { text: "background-free geometry resists the source shortcut.  ", options: {} },
    { text: "Why frozen CNN: ", options: { bold: true, color: "0E1B33" } }, { text: "29 groups are too few to fine-tune.  ", options: {} },
    { text: "Why a decoder: ", options: { bold: true, color: "0E1B33" } }, { text: "order resolves what pixels cannot.", options: {} },
  ], { x: 0.6, y: 5.75, w: 12.1, h: 0.8, fontSize: 14 });
  s.addNotes("Per frame: find the person, crop with the same padding as training, then two branches. Pose landmarks give 19 view-tolerant features; a frozen MobileNetV3 gives an appearance embedding. Each feeds a logistic regression and the scores are averaged. Over time: a 7-state HMM segments the sequence and local rules name iktidal and tasyahud.");
}

// ================================================================== 7. comparison
pres.addSection({ title: "Results" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Eleven models, one leakage-aware protocol", { placeholder: "title" });
  const ms = D.models;
  s.addChart(pres.charts.BAR, [{ name: "Macro-F1 (annotated crops)", labels: ms.map((m) => m.name), values: ms.map((m) => +m.crop.toFixed(2)) }], {
    x: 0.5, y: 1.2, w: 7.6, h: 5.55, barDir: "bar", catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.2,
    valAxisLabelFormatCode: "0.0", chartColors: ms.map((m) => (m.key === D.selected ? BLUE : DEEMPH)), barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: false,
    showTitle: true, title: "Macro-F1 on held-out people (mean of 3 repeats)", titleFontSize: 14, titleColor: "0E1B33", ...chartQuiet,
  });
  box(s, { x: 8.45, y: 1.3, w: 4.3, h: 2.15, fill: { color: "F3F5F4" }, line: { type: "none" } });
  txt(s, "Protocol", { x: 8.7, y: 1.45, w: 3.9, h: 0.4, fontSize: 16, bold: true, color: "0E1B33", fontFace: THEME.headFontFace });
  txt(s, ["3 × stratified group 5-fold CV", "Hyper-parameters by inner CV", "95% CI: bootstrap over groups", "Fusion fixed before results"].map((t, k, a) => ({ text: t, options: { bullet: true, breakLine: k < a.length - 1 } })),
    { x: 8.7, y: 1.9, w: 3.9, h: 1.5, fontSize: 13, paraSpaceAfter: 4 });
  const top = ms.slice(0, 4);
  const rows = [[{ text: "Model", options: { bold: true, color: "0E1B33" } }, { text: "F1", options: { bold: true, color: "0E1B33" } }, { text: "95% CI", options: { bold: true, color: "0E1B33" } }]]
    .concat(top.map((m) => [
      { text: m.name, options: { bold: m.key === D.selected, color: "0E1B33" } },
      { text: f2(m.crop), options: { align: "right" } },
      { text: `${f2(m.crop_lo)}–${f2(m.crop_hi)}`, options: { align: "right" } },
    ]));
  s.addTable(rows, { x: 8.45, y: 3.7, w: 4.3, colW: [2.3, 0.65, 1.35], fontSize: 12, fontFace: THEME.bodyFontFace, color: "4A5565", border: { type: "solid", color: "DFE3E1", pt: 0.75 }, rowH: 0.36, objectName: "top_models" });
  txt(s, "Intervals overlap: the ranking is suggestive, not conclusive.", { x: 8.45, y: 5.75, w: 4.3, h: 0.6, fontSize: 12, color: "6B7280" });
  s.addNotes("All eleven models were trained and tested under the same group-aware cross-validation. Pose geometry alone, with no training, already reaches 0.55. The pre-registered fusion is best at 0.80, and fine-tuned CNNs were not better than frozen features while taking about a hundred times longer to train.");
}

// ================================================================== 8. evaluation
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
  s.addText("Strong on standing and prostration, weaker on bowing", { placeholder: "title" });
  const cls = ["qiyam", "ruku", "sujud", "julus"];
  const short = { qiyam: "Standing", ruku: "Bowing", sujud: "Prostration", julus: "Sitting" };
  const cm = D.confusion_crop;
  const head = [{ text: "True \\ predicted", options: { bold: true, color: "0E1B33", fontSize: 11 } }].concat(cls.map((c) => ({ text: short[c], options: { bold: true, color: "0E1B33", align: "center", fontSize: 11 } })));
  const body = cm.map((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0) || 1;
    return [{ text: short[cls[i]], options: { bold: true, color: "0E1B33" } }].concat(row.map((v) => {
      const share = v / tot;
      const fill = SEQ[Math.min(SEQ.length - 1, Math.round(share * (SEQ.length - 1)))];
      return { text: `${v}`, options: { align: "center", valign: "middle", fill: { color: fill }, color: share > 0.55 ? "FFFFFF" : "0E1B33", bold: true } };
    }));
  });
  s.addTable([head, ...body], { x: 0.6, y: 1.5, w: 5.4, colW: [1.6, 0.95, 0.95, 0.95, 0.95], rowH: 0.62, fontSize: 14, fontFace: THEME.bodyFontFace, color: "4A5565", border: { type: "solid", color: "FFFFFF", pt: 2 }, objectName: "confusion_matrix" });
  txt(s, "Out-of-fold predictions on annotated crops, summed over 3 repeats; colour = share of the row.", { x: 0.6, y: 4.75, w: 5.4, h: 0.55, fontSize: 11, color: "6B7280" });
  s.addChart(pres.charts.BAR, [{ name: "F1", labels: cls.map((c) => short[c]), values: cls.map((c) => +D.per_class[c].f1.toFixed(2)) }], {
    x: 6.4, y: 1.3, w: 3.4, h: 3.6, barDir: "col", chartColors: [BLUE], valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: false, barGapWidthPct: 70,
    showTitle: true, title: "F1 per posture", titleFontSize: 14, titleColor: "0E1B33", ...chartQuiet,
  });
  stat(s, 10.2, 1.35, 2.6, f2(sm.crop_f1), `macro-F1 on annotated people (95% CI ${f2(sm.crop_ci[0])}–${f2(sm.crop_ci[1])})`);
  stat(s, 10.2, 3.05, 2.6, f2(sm.image_f1), "macro-F1 on whole images, automatic localisation");
  stat(s, 10.2, 4.75, 2.6, f2(sm.ece), "calibration error: scores are under-confident");
  txt(s, "Bowing has only 6 examples and is confused with sitting when the camera looks down; end-to-end, detector crops turn some sittings into prostrations.", { x: 0.6, y: 5.55, w: 9.2, h: 0.9, fontSize: 14 });
  s.addNotes("Standing and prostration are almost always right. Bowing is the weakest class. The gap between the crop view and the whole-image view is mostly sitting mistaken for prostration on detector crops. Scores are under-confident, so the uncertain flag is conservative.");
}

// ================================================================== 9. app
pres.addSection({ title: "Product" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Product" });
  s.addText("An Indonesian-first app built around the timeline", { placeholder: "title" });
  s.addImage({ path: A("application_screenshots", "02c_hasil.png"), x: 0.6, y: 1.3, w: 7.6, h: 4.75, altText: "Results dashboard", objectName: "shot_results" });
  s.addImage({ path: A("application_screenshots", "01_beranda.png"), x: 8.55, y: 1.3, w: 4.2, h: 2.62, altText: "Landing page", objectName: "shot_landing" });
  const pts = [
    ["Timeline seeks the video", "Click a phase block to jump there; the playhead follows playback."],
    ["Uncertainty is visible", "Grey blocks, a count of uncertain seconds, raw-vs-decoded comparison."],
    ["Model page shows real metrics", "Intervals, comparison, confusion matrix and limitations."],
  ];
  pts.forEach(([h, t], i) => {
    const y = 4.15 + i * 0.68;
    txt(s, h, { x: 8.55, y, w: 4.2, h: 0.3, fontSize: 14, bold: true, color: "0E1B33" });
    txt(s, t, { x: 8.55, y: y + 0.3, w: 4.2, h: 0.36, fontSize: 12 });
  });
  txt(s, "React 19, TypeScript, Vite, Tailwind, Recharts. Screens captured from the running app with the real model.", { x: 0.6, y: 6.2, w: 7.6, h: 0.4, fontSize: 12, color: "6B7280" });
  s.addNotes("The signature component is the phase timeline under the annotated video. It is keyboard accessible, and every chart has a table view. These screenshots come from the end-to-end tests running against the real backend.");
}

// ================================================================== 10. temporal
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Product" });
  s.addText("From flickering frames to clean phases", { placeholder: "title" });
  const total = D.demo.duration;
  const x0 = 2.4, wTl = 10.35;
  const rows = [["Clip plan", D.demo.plan], ["Raw frames", D.demo.raw], ["Decoded", D.demo.decoded]];
  rows.forEach(([name, segs], r) => {
    const y = 1.45 + r * 0.55;
    txt(s, `${name}${r === 1 ? ` (${segs.length})` : r === 2 ? ` (${segs.length})` : ""}`, { x: 0.6, y, w: 1.7, h: 0.4, fontSize: 13, color: "0E1B33", bold: r === 2, valign: "middle" });
    segs.forEach((g) => {
      const x = x0 + (g.start / total) * wTl;
      const w = Math.max(0.03, ((g.end - g.start) / total) * wTl - 0.025);
      s.addShape(pres.shapes.RECTANGLE, { x, y, w, h: 0.4, fill: { color: g.uncertain ? UNC : PH[g.phase] }, line: { type: "none" }, objectName: id("seg") });
    });
  });
  const legend = ["berdiri", "rukuk", "iktidal", "sujud", "duduk_antara_sujud", "tasyahud"];
  legend.forEach((p, i) => {
    const x = 2.4 + i * 1.62;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 3.17, w: 0.2, h: 0.2, fill: { color: PH[p] }, line: { type: "none" }, objectName: id("leg") });
    txt(s, PH_NAME[p], { x: x + 0.27, y: 3.1, w: 1.4, h: 0.32, fontSize: 11, valign: "middle" });
  });
  txt(s, "Synthetic 43 s demo clip from training images (faces blurred): checks the mechanics, not accuracy.", { x: 2.4, y: 3.45, w: 10.3, h: 0.3, fontSize: 11, color: "6B7280" });
  const decs = ["raw", "smoothing", "hmm", "hmm_local"];
  const dn = { raw: "Raw frames", smoothing: "Smoothing", hmm: "Global HMM", hmm_local: "HMM + local (used)" };
  const T = D.temporal.segment;
  s.addChart(pres.charts.BAR, [
    { name: "Frame phase accuracy (%)", labels: decs.map((d) => dn[d]), values: decs.map((d) => +(T[d].phase_acc * 100).toFixed(0)) },
    { name: "Segment F1@50", labels: decs.map((d) => dn[d]), values: decs.map((d) => +T[d].f1_50.toFixed(0)) },
  ], {
    x: 0.5, y: 3.9, w: 8.0, h: 2.9, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55, chartColors: [BLUE, "EB6834"],
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    showLegend: true, legendPos: "b", showTitle: true, title: "300 semi-synthetic sequences, conservative errors", titleFontSize: 13, titleColor: "0E1B33", ...chartQuiet,
  });
  box(s, { x: 8.85, y: 4.0, w: 3.9, h: 2.7, fill: { color: "E3F1EF" }, line: { type: "none" } });
  txt(s, "Lesson", { x: 9.1, y: 4.15, w: 3.4, h: 0.4, fontSize: 16, bold: true, color: "0E1B33", fontFace: THEME.headFontFace });
  txt(s, "A global grammar is the best segmenter, but one misread segment can shift every later phase name. Keeping HMM segmentation and naming locally won on both metrics, chosen on a separate validation seed.", { x: 9.1, y: 4.6, w: 3.45, h: 2.0, fontSize: 13 });
  s.addNotes(`Raw frame labels produce ${D.demo.raw.length} segments on the demo clip; decoding gives ${D.demo.decoded.length}, the planned two rakaat. On 300 semi-synthetic sequences whose scores are real held-out predictions, the hybrid decoder reaches ${pct(T.hmm_local.phase_acc)} frame phase accuracy and F1@50 of ${T.hmm_local.f1_50.toFixed(0)}, versus ${T.raw.f1_50.toFixed(0)} for raw labels.`);
}

// ================================================================== 11. decisions & limits
pres.addSection({ title: "Engineering" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Engineering" });
  s.addText("Decisions, trade-offs and honest limits", { placeholder: "title" });
  const hd = (t) => ({ text: t, options: { bold: true, color: "0E1B33" } });
  const rows = [
    [hd("Decision"), hd("Why"), hd("Rejected")],
    ["Group-aware CV + cluster bootstrap", "duplicates span file families", "random image split"],
    ["Pose + frozen-CNN fusion", "best, stable, cheap", "fine-tuning (memorises)"],
    ["4 visual classes + temporal naming", "data cannot support 6", "inventing labels"],
    ["MediaPipe Pose full", `${pct(D.pose_variants.full.rate)} found at ${D.pose_variants.full.ms.toFixed(0)} ms`, "heavy (2× slower); YOLO-Pose (AGPL)"],
    ["ONNX Runtime serving", `${(D.runtime.torch / D.runtime.ort).toFixed(1)}× faster embedder`, "PyTorch in production"],
  ].map((r) => r.map((c) => (typeof c === "string" ? { text: c } : c)));
  s.addTable(rows, { x: 0.6, y: 1.35, w: 7.6, colW: [2.75, 2.45, 2.4], fontSize: 13, fontFace: THEME.bodyFontFace, color: "4A5565", border: { type: "solid", color: "DFE3E1", pt: 0.75 }, rowH: 0.5, objectName: "decisions" });
  box(s, { x: 8.55, y: 1.35, w: 4.2, h: 5.1, fill: { color: "F3F5F4" }, line: { type: "none" } });
  txt(s, "Limitations", { x: 8.8, y: 1.55, w: 3.8, h: 0.4, fontSize: 16, bold: true, color: "0E1B33", fontFace: THEME.headFontFace });
  const lim = [`Only ${D.dataset.instances} people, ${D.dataset.groups} groups: wide intervals`, "No real prayer videos: temporal results are semi-synthetic", "Iktidal and both sittings are inferred from order",
    `Pose found on ${pct(D.pose_variants.full.sujud)} of prostrations`, "Scores under-confident, not calibrated", "Viewpoints, clothing, children, groups untested"];
  txt(s, lim.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < lim.length - 1 } })), { x: 8.8, y: 2.05, w: 3.75, h: 4.2, fontSize: 13, paraSpaceAfter: 7 });
  txt(s, "Each limitation is stated in the app, the model card and the report, not only here.", { x: 0.6, y: 4.6, w: 7.6, h: 0.5, fontSize: 13, color: "6B7280" });
  s.addNotes("The main trade-off was capacity versus data: with 29 independent groups, strong priors beat end-to-end training. The limitations on the right are shown to users inside the product as well.");
}

// ================================================================== 12. deployment & testing
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Engineering" });
  s.addText("Fast on a CPU, reproducible, tested", { placeholder: "title" });
  stat(s, 0.6, 1.3, 2.9, `${D.runtime.total.toFixed(0)} ms`, `per frame on ${D.runtime.cores} vCPU, no GPU (${D.runtime.fps.toFixed(1)} fps)`);
  stat(s, 3.65, 1.3, 2.9, `${(D.runtime.torch / D.runtime.ort).toFixed(1)}×`, "faster embedder with ONNX Runtime than PyTorch");
  stat(s, 6.7, 1.3, 2.9, "100%", "identical files when the bundle is re-exported (SHA-256)");
  stat(s, 9.75, 1.3, 3.0, `${D.tests.python + D.tests.frontend_unit + D.tests.e2e}`, `automated tests: ${D.tests.python} Python, ${D.tests.frontend_unit} unit, ${D.tests.e2e} end-to-end`);
  const st = [["localize", "Person localisation"], ["pose", "Pose landmarks"], ["image_model", "Image model (ONNX)"], ["pose_classifier", "Pose classifier"]];
  s.addChart(pres.charts.BAR, [{ name: "ms per frame", labels: st.map((x) => x[1]), values: st.map((x) => +D.runtime[x[0]].toFixed(1)) }], {
    x: 0.5, y: 3.35, w: 6.6, h: 3.3, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [BLUE], valAxisMinVal: 0,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: false, barGapWidthPct: 60,
    showTitle: true, title: "Where the time goes (mean ms per frame)", titleFontSize: 13, titleColor: "0E1B33", ...chartQuiet,
  });
  const items = ["Pinned dependencies, fixed seeds, saved folds, full environment lock", "Notebook re-run reproduced the scripted CV numbers exactly",
    "Tests cover ZIP attacks, leakage, decoder, real-model inference, tampered bundles, API jobs, CORS", "Uploads streamed with size limits, deleted after analysis; results expire after 30 min",
    "Dockerfiles and Compose provided (syntax-checked; not built in the authoring environment)"];
  txt(s, items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })), { x: 7.45, y: 3.45, w: 5.3, h: 3.2, fontSize: 13, paraSpaceAfter: 6 });
  s.addNotes("Person localisation and pose estimation dominate the per-frame cost; the two classifiers are almost free. Reproducibility was checked, not assumed: re-exporting the bundle gives byte-identical files.");
}

// ================================================================== 13. learnings & roadmap
pres.addSection({ title: "Close" });
{
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Close" });
  s.addText("What I learned, and what comes next", { placeholder: "title" });
  const learn = [
    ["Audit before modelling", "The archive was a 41-image sample with hidden duplicates; finding that first changed the protocol."],
    ["Priors beat capacity at small n", "Pose geometry and frozen features generalised; fine-tuning memorised."],
    ["Say what is inferred", "Naming phases by order is useful only if users know it is an inference."],
  ];
  learn.forEach(([h, t], i) => {
    const y = 1.4 + i * 1.6;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.42, h: 0.42, fill: { color: "0F766E" }, line: { type: "none" }, objectName: id("dot") });
    txt(s, h, { x: 1.2, y: y - 0.02, w: 4.9, h: 0.45, fontSize: 17, bold: true, color: "0E1B33", fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, t, { x: 1.2, y: y + 0.48, w: 4.9, h: 0.9, fontSize: 14 });
  });
  const road = ["Re-run on the full 1,904-image dataset", "Record consented, phase-labelled prayer videos", "Calibrate scores; learn phase durations", "Run in the browser so no video is uploaded"];
  txt(s, "Roadmap", { x: 6.9, y: 1.35, w: 5.8, h: 0.45, fontSize: 17, bold: true, color: "0E1B33", fontFace: THEME.headFontFace });
  s.addShape(pres.shapes.LINE, { x: 7.15, y: 2.05, w: 0, h: 4.25, line: { color: "C9CCD1", width: 2 }, objectName: "road_line" });
  road.forEach((t, i) => {
    const y = 1.95 + i * 1.1;
    s.addShape(pres.shapes.OVAL, { x: 6.95, y, w: 0.4, h: 0.4, fill: { color: i === 0 ? "B8892C" : "FFFFFF" }, line: { color: "B8892C", width: 2 }, objectName: id("step") });
    txt(s, `${i + 1}`, { x: 6.95, y, w: 0.4, h: 0.4, fontSize: 12, bold: true, align: "center", valign: "middle", color: i === 0 ? "FFFFFF" : "B8892C" });
    txt(s, t, { x: 7.6, y: y - 0.02, w: 5.1, h: 0.45, fontSize: 15, color: "0E1B33", valign: "middle" });
  });
  s.addNotes("Three lessons that transfer to any small-data computer-vision project, and a roadmap ordered by what would most improve confidence in the results.");
}

// ================================================================== 14. summary
{
  const s = pres.addSlide({ masterName: "CLOSING", sectionTitle: "Close" });
  s.addText("Salat Vision in one slide", { placeholder: "title" });
  stat(s, 0.6, 1.3, 3.0, f2(sm.crop_f1), "macro-F1 on held-out people", true);
  stat(s, 3.9, 1.3, 3.0, `${D.temporal.segment.hmm_local.f1_50.toFixed(0)}`, "segment F1@50 for video phases", true);
  stat(s, 7.2, 1.3, 2.6, `${D.runtime.total.toFixed(0)} ms`, "per frame, CPU only", true);
  stat(s, 10.1, 1.3, 2.6, `${D.tests.python + D.tests.frontend_unit + D.tests.e2e}`, "automated tests passing", true);
  const stack = ["Python", "MediaPipe Pose", "PyTorch → ONNX", "scikit-learn", "HMM / Viterbi", "FastAPI", "React + TypeScript", "Tailwind", "Playwright"];
  let x = 0.6;
  let y = 3.2;
  stack.forEach((t) => {
    const w = 0.4 + t.length * 0.11;
    if (x + w > 12.75) {
      x = 0.6;
      y += 0.55;
    }
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.42, rectRadius: 0.2, fill: { type: "none" }, line: { color: "56657E", width: 1 }, objectName: id("chip") });
    txt(s, t, { x, y, w, h: 0.42, fontSize: 12, color: "E8ECEF", align: "center", valign: "middle" });
    x += w + 0.15;
  });
  const refs = [
    "Koubâa et al. (2020). Activity Monitoring of Islamic Prayer (Salat) Postures using Deep Learning. CDMA, IEEE. arXiv:1911.04102",
    "Bazarevsky et al. (2020). BlazePose: On-device Real-time Body Pose Tracking. arXiv:2006.10204",
    "Howard et al. (2019). Searching for MobileNetV3. ICCV",
    "Lea et al. (2017). Temporal Convolutional Networks for Action Segmentation. CVPR (edit score, F1@k)",
  ];
  txt(s, "References", { x: 0.6, y: 4.45, w: 6, h: 0.4, fontSize: 15, bold: true, color: "FFFFFF", fontFace: THEME.headFontFace });
  txt(s, refs.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < refs.length - 1 } })), { x: 0.6, y: 4.9, w: 8.4, h: 1.8, fontSize: 11, color: "C9D1D9", paraSpaceAfter: 4 });
  txt(s, [{ text: `${CFG.author}\n`, options: { bold: true, color: "FFFFFF" } }, { text: `${CFG.role}\n${CFG.contact}\n${CFG.repository}`, options: { color: "C9D1D9" } }], { x: 9.4, y: 4.9, w: 3.35, h: 1.6, fontSize: 13 });
  mark(s, 12.3, 0.55, 0.3);
  s.addNotes("Summary: a leakage-aware model comparison, a temporal decoder that turns four postures into six phases, and a tested product that is honest about its limits. Replace the contact fields with your own.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL_THEME) {
    const { applyTheme } = require(SKILL_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})();
