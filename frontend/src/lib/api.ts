import type { Health, ImagePrediction, JobStatus, ModelInfo } from "./types";

export const API_BASE: string = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parseError(res: Response): Promise<ApiError> {
  let detail = res.statusText || "Permintaan gagal";
  try {
    const body = await res.json();
    if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* not JSON */
  }
  return new ApiError(res.status, detail);
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, { signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "Server tidak dapat dihubungi");
  }
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

export const getHealth = (signal?: AbortSignal) => getJson<Health>("/health", signal);
export const getModelInfo = (signal?: AbortSignal) => getJson<ModelInfo>("/model-info", signal);
export const getJob = (id: string, signal?: AbortSignal) => getJson<JobStatus>(`/jobs/${encodeURIComponent(id)}`, signal);

export async function predictImage(file: File | Blob): Promise<ImagePrediction> {
  const form = new FormData();
  form.append("file", file, (file as File).name || "image.jpg");
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/predict/image`, { method: "POST", body: form });
  } catch {
    throw new ApiError(0, "Server tidak dapat dihubungi");
  }
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as ImagePrediction;
}

export interface UploadOptions {
  annotate: boolean;
  sampleFps: number;
  onUploadProgress?: (fraction: number) => void;
}

/** Uses XMLHttpRequest so the upload itself can report real progress. */
export function createVideoJob(file: File | Blob, opts: UploadOptions): Promise<{ job_id: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const q = new URLSearchParams({ annotate: String(opts.annotate), sample_fps: String(opts.sampleFps) });
    xhr.open("POST", `${API_BASE}/jobs/video?${q}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) opts.onUploadProgress?.(e.loaded / e.total);
    };
    xhr.onload = () => {
      let body: { job_id?: string; detail?: string } = {};
      try {
        body = JSON.parse(xhr.responseText || "{}");
      } catch {
        /* ignore */
      }
      if (xhr.status === 202 && body.job_id) resolve({ job_id: body.job_id });
      else reject(new ApiError(xhr.status, body.detail || `Unggahan gagal (${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Server tidak dapat dihubungi"));
    const form = new FormData();
    form.append("file", file, (file as File).name || "rekaman.webm");
    xhr.send(form);
  });
}

export async function deleteJob(id: string): Promise<void> {
  await fetch(`${API_BASE}/jobs/${encodeURIComponent(id)}`, { method: "DELETE" }).catch(() => undefined);
}

export const jobVideoUrl = (id: string) => `${API_BASE}/jobs/${encodeURIComponent(id)}/video`;
export const jobReportUrl = (id: string, format: "json" | "csv") => `${API_BASE}/jobs/${encodeURIComponent(id)}/report?format=${format}`;

export async function fetchDemoVideo(): Promise<File> {
  const res = await fetch(`${API_BASE}/demo/video`);
  if (!res.ok) throw await parseError(res);
  const blob = await res.blob();
  return new File([blob], "salat_vision_demo.mp4", { type: "video/mp4" });
}
