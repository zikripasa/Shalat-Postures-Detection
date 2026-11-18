import type { JobStatus } from "../../lib/types";

interface Props {
  stage: "uploading" | "processing";
  uploadFraction: number;
  job: JobStatus | null;
}

const STAGE_TEXT: Record<string, string> = {
  queued: "Menunggu giliran di server",
  starting: "Menyiapkan model",
  analyzing: "Menganalisis frame",
  rendering: "Membuat video beranotasi",
  done: "Selesai",
};

export function ProgressPanel({ stage, uploadFraction, job }: Props) {
  let fraction = 0;
  let text = "";
  if (stage === "uploading") {
    fraction = uploadFraction * 0.15;
    text = `Mengunggah video ${Math.round(uploadFraction * 100)}%`;
  } else {
    const p = job?.progress ?? {};
    const st = p.stage ?? job?.status ?? "queued";
    const serverFrac = typeof p.fraction === "number" ? p.fraction : 0;
    fraction = 0.15 + 0.85 * serverFrac;
    text = STAGE_TEXT[st] ?? "Memproses";
    if (st === "analyzing" && p.frames_done) {
      text += p.frames_expected ? ` ${p.frames_done} dari ±${p.frames_expected}` : ` (${p.frames_done} frame)`;
    }
  }
  const pct = Math.max(2, Math.min(100, Math.round(fraction * 100)));
  return (
    <div className="rounded-lg border border-hairline bg-white px-4 py-4">
      <div className="flex items-baseline justify-between gap-4 text-[14px]">
        <span className="font-semibold text-night" aria-live="polite">
          {text}
        </span>
        <span className="text-muted tabular">{pct}%</span>
      </div>
      <div
        className="mt-3 h-2 overflow-hidden rounded-full bg-[#e6ebe9]"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-label="Kemajuan analisis"
      >
        <div className="h-full rounded-full bg-faience transition-[width] duration-500" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-[13px] text-muted">Video diproses di server Salat Vision dan dihapus setelah analisis.</p>
    </div>
  );
}
