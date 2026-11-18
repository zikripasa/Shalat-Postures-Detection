// Mirrors src/salat_vision/viz.py — validated categorical palette (CVD ΔE ≥ 8.4,
// normal-vision ΔE ≥ 19 on every pair of phases that can touch in a timeline).
import type { Phase, VisualClass } from "./types";

export const PHASE_COLORS: Record<Phase, string> = {
  berdiri: "#eda100",
  rukuk: "#1baf7a",
  iktidal: "#eb6834",
  sujud: "#2a78d6",
  duduk_antara_sujud: "#008300",
  tasyahud: "#e87ba4",
};

export const VISUAL_COLORS: Record<VisualClass, string> = {
  qiyam: "#eda100",
  ruku: "#1baf7a",
  sujud: "#2a78d6",
  julus: "#008300",
};

export const UNCERTAIN_COLOR = "#cfccc2";

/** Ink to use for a label printed *inside* a coloured fill (APCA-chosen per palette doc). */
export const INK_ON: Record<string, string> = {
  "#eda100": "#0b0b0b",
  "#1baf7a": "#0b0b0b",
  "#eb6834": "#0b0b0b",
  "#2a78d6": "#ffffff",
  "#008300": "#ffffff",
  "#e87ba4": "#0b0b0b",
  "#cfccc2": "#0b0b0b",
};

export const CHART = {
  surface: "#fcfcfb",
  grid: "#e1e0d9",
  axis: "#c3c2b7",
  ink: "#0e1b33",
  ink2: "#4a5565",
  muted: "#6b7280",
  series1: "#2a78d6",
  series2: "#eb6834",
  deemph: "#c9ccd1",
  seq: ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
};
