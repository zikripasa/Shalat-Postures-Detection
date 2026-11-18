/** 75.3 -> "1:15.3" ; 4.25 -> "0:04.3" */
export function formatTime(seconds: number, decimals = 1): string {
  if (!Number.isFinite(seconds)) return "–";
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  const sec = rest.toFixed(decimals).padStart(decimals > 0 ? 3 + decimals : 2, "0");
  return `${m}:${sec}`;
}

export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "–";
  if (seconds < 60) return `${seconds.toFixed(1).replace(".", ",")} dtk`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  return `${m} mnt ${s} dtk`;
}

export function formatPercent(x: number | null | undefined, digits = 0): string {
  if (x == null || !Number.isFinite(x)) return "–";
  return `${(x * 100).toFixed(digits).replace(".", ",")}%`;
}

export function formatNumber(x: number | null | undefined, digits = 2): string {
  if (x == null || !Number.isFinite(x)) return "–";
  return x.toFixed(digits).replace(".", ",");
}

export const EVIDENCE_LABEL: Record<string, string> = {
  visual: "Terlihat langsung",
  "visual + context": "Terlihat + urutan gerakan",
};

export const LOCALIZATION_LABEL: Record<string, string> = {
  detector: "Detektor orang",
  pose: "Kerangka pose",
  full_image: "Seluruh gambar",
};

export const MODEL_LABEL: Record<string, string> = {
  baseline_majority: "Baseline kelas mayoritas",
  pose_rules: "Aturan geometri pose (tanpa pelatihan)",
  pose_logreg: "Fitur pose + regresi logistik",
  pose_svm_rbf: "Fitur pose + SVM RBF",
  pose_random_forest: "Fitur pose + random forest",
  pose_mlp: "Fitur pose + MLP",
  cnn_probe_mobilenet_v3_large: "MobileNetV3 beku + regresi logistik",
  cnn_probe_resnet18: "ResNet18 beku + regresi logistik",
  cnn_ft_mobilenet_v3_large: "MobileNetV3 fine-tuning",
  cnn_ft_resnet18: "ResNet18 fine-tuning",
  fusion_pose_cnn: "Fusi pose + MobileNetV3 (terpilih)",
};
