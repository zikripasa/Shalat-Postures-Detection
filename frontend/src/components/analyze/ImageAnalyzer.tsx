import { useEffect, useRef, useState } from "react";
import { ApiError, predictImage } from "../../lib/api";
import { formatPercent, LOCALIZATION_LABEL } from "../../lib/format";
import { VISUAL_COLORS } from "../../lib/palette";
import type { ImagePrediction, VisualClass } from "../../lib/types";
import { Notice, Spinner } from "../ui";
import { Dropzone } from "./Dropzone";

const EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32]];
const CLASS_LABEL: Record<VisualClass, string> = { qiyam: "Berdiri", ruku: "Rukuk", sujud: "Sujud", julus: "Duduk" };
const MAX_MB = 10;

function Overlay({ src, pred }: { src: string; pred: ImagePrediction | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  useEffect(() => {
    const img = imgRef.current;
    const cv = canvasRef.current;
    if (!img || !cv) return;
    const draw = () => {
      cv.width = img.clientWidth;
      cv.height = img.clientHeight;
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, cv.width, cv.height);
      if (!pred) return;
      const W = cv.width;
      const H = cv.height;
      if (pred.person_box) {
        const [x0, y0, x1, y1] = pred.person_box;
        ctx.strokeStyle = pred.visual_class ? VISUAL_COLORS[pred.visual_class] : "#ffffff";
        ctx.lineWidth = 3;
        ctx.strokeRect(x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H);
      }
      if (pred.landmarks) {
        const lm = pred.landmarks;
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#ffffff";
        ctx.lineCap = "round";
        for (const [a, b] of EDGES) {
          if (lm[a][2] > 0.35 && lm[b][2] > 0.35) {
            ctx.beginPath();
            ctx.moveTo(lm[a][0] * W, lm[a][1] * H);
            ctx.lineTo(lm[b][0] * W, lm[b][1] * H);
            ctx.stroke();
          }
        }
        ctx.fillStyle = "#2dd4bf";
        for (const p of lm.slice(11)) {
          if (p[2] > 0.35) {
            ctx.beginPath();
            ctx.arc(p[0] * W, p[1] * H, 4, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
    };
    if (img.complete) draw();
    img.addEventListener("load", draw);
    window.addEventListener("resize", draw);
    return () => {
      img.removeEventListener("load", draw);
      window.removeEventListener("resize", draw);
    };
  }, [src, pred]);
  return (
    <div className="relative overflow-hidden rounded-lg bg-night">
      <img ref={imgRef} src={src} alt="Foto yang dianalisis" className="mx-auto max-h-[460px] w-auto" />
      <canvas ref={canvasRef} className="pointer-events-none absolute left-1/2 top-0 -translate-x-1/2" aria-hidden="true" />
    </div>
  );
}

export function ImageAnalyzer({ modelReady }: { modelReady: boolean }) {
  const [src, setSrc] = useState<string | null>(null);
  const [pred, setPred] = useState<ImagePrediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  useEffect(() => () => void (src && URL.revokeObjectURL(src)), [src]);

  async function run(f: File) {
    setFile(f);
    setPred(null);
    setError(null);
    setSrc((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
    setBusy(true);
    try {
      setPred(await predictImage(f));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Analisis foto gagal.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <Dropzone
          accept="image/jpeg,image/png,image/webp"
          label="Pilih atau seret foto"
          hint={`JPEG, PNG, atau WebP hingga ${MAX_MB} MB, satu orang terlihat utuh`}
          validate={(f) => (f.size > MAX_MB * 1024 * 1024 ? `Ukuran foto melebihi ${MAX_MB} MB.` : !/^image\//.test(f.type) ? "Berkas bukan foto." : null)}
          onFile={(f) => modelReady && run(f)}
        />
        {!modelReady && <Notice tone="warning">Model belum tersedia, sehingga foto belum bisa dianalisis.</Notice>}
        {src && <Overlay src={src} pred={pred} />}
      </div>
      <div className="space-y-4" aria-live="polite">
        {busy && <Spinner label="Menganalisis foto…" />}
        {error && (
          <Notice tone="error" title="Analisis foto gagal">
            {error}{" "}
            {file && (
              <button type="button" className="font-semibold text-faience underline" onClick={() => run(file)}>
                Coba lagi
              </button>
            )}
          </Notice>
        )}
        {pred && (
          <div className="rounded-xl border border-hairline bg-paper p-5">
            <p className="text-[14px] text-slate">Postur yang dikenali</p>
            <p className="mt-1 text-[30px] font-extrabold tracking-[-0.02em] text-night">
              {pred.uncertain ? `Tidak pasti (${pred.label ?? "–"})` : pred.label}
            </p>
            <p className="text-[14px] text-slate">Keyakinan model {formatPercent(pred.confidence)}</p>
            {pred.probabilities && (
              <ul className="mt-5 space-y-2.5" aria-label="Skor per postur">
                {(Object.keys(CLASS_LABEL) as VisualClass[]).map((c) => (
                  <li key={c} className="grid grid-cols-[84px_1fr_48px] items-center gap-3 text-[14px]">
                    <span className="text-night">{CLASS_LABEL[c]}</span>
                    <span className="h-3 overflow-hidden rounded-r-[4px] bg-[#edf0ef]">
                      <span className="block h-full rounded-r-[4px]" style={{ width: `${pred.probabilities![c] * 100}%`, background: VISUAL_COLORS[c] }} />
                    </span>
                    <span className="text-right text-slate tabular">{formatPercent(pred.probabilities![c])}</span>
                  </li>
                ))}
              </ul>
            )}
            <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-hairline pt-4 text-[13.5px]">
              <dt className="text-muted">Lokasi orang dari</dt>
              <dd className="text-night">{LOCALIZATION_LABEL[pred.localization] ?? pred.localization}</dd>
              <dt className="text-muted">Kerangka pose</dt>
              <dd className="text-night">{pred.pose_detected ? "Terdeteksi" : "Tidak terdeteksi (hanya model visual)"}</dd>
              <dt className="text-muted">Waktu proses</dt>
              <dd className="text-night tabular">{Math.round(pred.timings_ms.total ?? 0)} ms</dd>
            </dl>
            <p className="mt-4 text-[13px] leading-relaxed text-muted">{pred.note}</p>
          </div>
        )}
      </div>
    </div>
  );
}
