export type VisualClass = "qiyam" | "ruku" | "sujud" | "julus";
export type Phase = "berdiri" | "rukuk" | "iktidal" | "sujud" | "duduk_antara_sujud" | "tasyahud";

export interface Health {
  status: "ok";
  model_loaded: boolean;
  model_version: string | null;
  model_error: string | null;
  uptime_seconds: number;
  package_version: string;
}

export interface ImagePrediction {
  visual_class: VisualClass | null;
  label: string | null;
  confidence: number | null;
  uncertain: boolean;
  probabilities: Record<VisualClass, number> | null;
  probabilities_pose_model: Record<VisualClass, number> | null;
  probabilities_image_model: Record<VisualClass, number> | null;
  localization: "detector" | "pose" | "full_image";
  person_box: [number, number, number, number] | null;
  pose_detected: boolean;
  landmarks: number[][] | null;
  timings_ms: Record<string, number>;
  image: { format: string; width: number; height: number };
  note: string;
}

export interface Segment {
  index: number;
  phase: Phase;
  label: string;
  display_label: string;
  visual_class: VisualClass;
  evidence: string;
  start: number;
  end: number;
  duration: number;
  n_frames: number;
  confidence: number;
  frames_without_person: number;
  uncertain: boolean;
  state?: string | null;
}

export interface FrameRecord {
  t: number;
  visual_class: VisualClass | null;
  confidence: number | null;
  probabilities: Record<VisualClass, number> | null;
  smoothed_probabilities?: Record<VisualClass, number>;
  phase: Phase;
  localization: string;
  pose_detected: boolean;
  person_box: number[] | null;
}

export interface VideoAnalysis {
  video: { duration_seconds: number; width: number; height: number; source_fps: number; sample_fps: number };
  decoder: string;
  segments: Segment[];
  raw_segments: Segment[];
  summary: {
    phase_seconds: Record<Phase, number>;
    phase_share: Record<Phase, number>;
    n_segments: number;
    n_uncertain_segments: number;
    uncertain_seconds: number;
    sequence: Phase[];
    rukuk_count: number;
    sujud_count: number;
    cycles_observed: number | null;
  };
  frames: FrameRecord[] | null;
  performance: {
    frames_analyzed: number;
    analysis_seconds: number;
    analysis_fps: number | null;
    realtime_factor: number | null;
    mean_ms_per_frame: Record<string, number>;
    p95_ms_per_frame: Record<string, number>;
    annotation_seconds?: number;
  };
  phase_labels: Record<Phase, string>;
  phases: Phase[];
  notes: string[];
}

export interface JobStatus {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  progress: { stage?: string; fraction?: number; frames_done?: number; frames_expected?: number };
  error: string | null;
  created: number;
  finished: number | null;
  annotated_video_available: boolean;
  result: VideoAnalysis | null;
}

export interface ModelComparisonRow {
  model: string;
  family: string;
  crop_macro_f1: number;
  crop_macro_f1_sd: number;
  crop_macro_f1_ci_lo: number;
  crop_macro_f1_ci_hi: number;
  crop_accuracy: number;
  image_macro_f1: number;
  image_macro_f1_ci_lo: number;
  image_macro_f1_ci_hi: number;
  image_accuracy: number;
  abstain_rate: number;
  ece: number | null;
}

export interface CvView {
  macro_f1_mean: number;
  macro_f1_std: number;
  macro_f1_ci95: [number, number];
  accuracy_mean: number;
  accuracy_ci95: [number, number];
}

export interface ModelInfo {
  available: boolean;
  reason: string | null;
  version?: string;
  created_utc?: string;
  classes: { id: VisualClass; label: string; label_en: string }[];
  phases: { id: Phase; label: string; label_en: string; visual_class: VisualClass; evidence: string }[];
  pipeline?: Record<string, unknown> & { fusion?: { w_pose: number; rule: string }; uncertainty_threshold?: number };
  training?: Record<string, unknown> & { dataset?: { instances: number; images: number; groups: number; per_class: Record<string, number> } };
  cv_metrics?: {
    model_key: string;
    protocol: string;
    crop: CvView;
    image: CvView;
    per_class_crop: Record<VisualClass, { precision: number; recall: number; f1: number; support: number }>;
    confusion_matrix_crop: number[][];
    confusion_matrix_image: number[][];
    ece_crop: number | null;
  } | null;
  model_comparison: ModelComparisonRow[] | null;
  temporal_benchmark: Record<string, unknown> | null;
  runtime_benchmark: Record<string, unknown> | null;
  dataset: Record<string, unknown> | null;
  intended_use?: string;
  out_of_scope?: string[];
  limitations?: string[];
  third_party?: { role: string; name: string; license: string }[];
  environment?: Record<string, string>;
  demo_video_available: boolean;
}
