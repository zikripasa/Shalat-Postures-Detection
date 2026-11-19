import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { CameraRecorder, cameraSupported } from "../components/analyze/CameraRecorder";
import { Dropzone } from "../components/analyze/Dropzone";
import { ImageAnalyzer } from "../components/analyze/ImageAnalyzer";
import { ProgressPanel } from "../components/analyze/ProgressPanel";
import { Notice, PageHeader, Panel, PrimaryButton, SecondaryButton } from "../components/ui";
import { ApiError, createVideoJob, fetchDemoVideo, getJob } from "../lib/api";
import { useAnalysis } from "../lib/analysisStore";
import type { JobStatus } from "../lib/types";
import { useModelInfo } from "../lib/useModelInfo";

type Mode = "video" | "foto";
type Source = "file" | "camera";
const MAX_VIDEO_MB = 150;
const FPS_OPTIONS = [
  { value: 2, label: "2 frame/detik", hint: "paling cepat" },
  { value: 5, label: "5 frame/detik", hint: "disarankan" },
  { value: 10, label: "10 frame/detik", hint: "paling rinci" },
];

export function AnalyzePage() {
  const mi = useModelInfo();
  const navigate = useNavigate();
  const { setLocalVideo } = useAnalysis();
  const [mode, setMode] = useState<Mode>("video");
  const [source, setSource] = useState<Source>("file");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [fps, setFps] = useState(5);
  const [annotate, setAnnotate] = useState(true);
  const [stage, setStage] = useState<"idle" | "uploading" | "processing">("idle");
  const [uploadFrac, setUploadFrac] = useState(0);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const pollRef = useRef<number | null>(null);

  const ready = mi.status === "ready" && mi.info.available;
  const busy = stage !== "idle";

  useEffect(() => () => {
    if (pollRef.current) window.clearTimeout(pollRef.current);
  }, []);
  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function choose(f: File) {
    setFile(f);
    setError(null);
    setJob(null);
  }

  async function useDemo() {
    setDemoLoading(true);
    setError(null);
    try {
      choose(await fetchDemoVideo());
      setSource("file");
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Video demo tidak dapat dimuat.");
    } finally {
      setDemoLoading(false);
    }
  }

  async function start() {
    if (!file) return;
    setError(null);
    setStage("uploading");
    setUploadFrac(0);
    try {
      const { job_id } = await createVideoJob(file, { annotate, sampleFps: fps, onUploadProgress: setUploadFrac });
      setLocalVideo(file, file.name);
      setStage("processing");
      const poll = async () => {
        try {
          const st = await getJob(job_id);
          setJob(st);
          if (st.status === "done") {
            navigate(`/hasil/${job_id}`);
            return;
          }
          if (st.status === "error") {
            setStage("idle");
            setError(st.error ?? "Analisis gagal.");
            return;
          }
        } catch (e) {
          setStage("idle");
          setError(e instanceof ApiError ? e.message : "Status analisis tidak dapat dibaca.");
          return;
        }
        pollRef.current = window.setTimeout(poll, 700);
      };
      poll();
    } catch (e) {
      setStage("idle");
      setError(e instanceof ApiError ? e.message : "Unggahan gagal.");
    }
  }

  const tabCls = (on: boolean) =>
    `rounded-md px-4 py-2 text-[15px] font-semibold transition-colors ${on ? "bg-night text-white" : "text-slate hover:text-night"}`;

  return (
    <div className="mx-auto max-w-6xl px-5 py-10 sm:px-8">
      <PageHeader
        title="Analisis"
        lead="Unggah atau rekam video shalat untuk mendapatkan linimasa gerakan. Foto tunggal juga bisa dicek untuk melihat postur yang dikenali."
      />
      {mi.status === "error" && (
        <div className="mt-6">
          <Notice tone="error" title="Server analisis tidak terhubung">
            Jalankan backend (uvicorn) dan pastikan alamatnya sama dengan VITE_API_BASE_URL.{" "}
            <button type="button" className="font-semibold text-faience underline" onClick={mi.reload}>
              Periksa lagi
            </button>
          </Notice>
        </div>
      )}
      {mi.status === "ready" && !mi.info.available && (
        <div className="mt-6">
          <Notice tone="warning" title="Model belum tersedia">
            {mi.info.reason}
          </Notice>
        </div>
      )}

      <div className="mt-6 inline-flex gap-1 rounded-lg border border-hairline bg-white p-1" role="tablist" aria-label="Jenis masukan">
        <button role="tab" aria-selected={mode === "video"} className={tabCls(mode === "video")} onClick={() => setMode("video")}>
          Video
        </button>
        <button role="tab" aria-selected={mode === "foto"} className={tabCls(mode === "foto")} onClick={() => setMode("foto")}>
          Foto
        </button>
      </div>

      {mode === "foto" ? (
        <div className="mt-6">
          <ImageAnalyzer modelReady={ready} />
        </div>
      ) : (
        <div className="mt-6 grid gap-6 lg:grid-cols-12">
          <Panel className="lg:col-span-7">
            <div className="flex flex-wrap items-center gap-2">
              <SecondaryButton onClick={() => setSource("file")} className={source === "file" ? "!border-night" : ""} ariaLabel="Unggah berkas video">
                Unggah berkas
              </SecondaryButton>
              {cameraSupported() && (
                <SecondaryButton onClick={() => setSource("camera")} className={source === "camera" ? "!border-night" : ""}>
                  Rekam dengan kamera
                </SecondaryButton>
              )}
              {mi.status === "ready" && mi.info.demo_video_available && (
                <SecondaryButton onClick={useDemo} disabled={demoLoading || busy}>
                  {demoLoading ? "Memuat demo…" : "Pakai video demo"}
                </SecondaryButton>
              )}
            </div>
            <div className="mt-5">
              {source === "camera" ? (
                <CameraRecorder onRecorded={(f) => {
                  choose(f);
                  setSource("file");
                }} />
              ) : preview ? (
                <div>
                  <video src={preview} controls className="aspect-video w-full rounded-lg bg-night" aria-label="Pratinjau video" />
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-[14px] text-slate">
                    <span className="truncate">
                      {file?.name} ({((file?.size ?? 0) / 1e6).toFixed(1).replace(".", ",")} MB)
                    </span>
                    <button type="button" className="font-semibold text-faience hover:underline" onClick={() => setFile(null)} disabled={busy}>
                      Ganti video
                    </button>
                  </div>
                </div>
              ) : (
                <Dropzone
                  accept="video/mp4,video/webm,video/quicktime,video/x-matroska,.mp4,.webm,.mov,.mkv"
                  label="Pilih atau seret video"
                  hint={`MP4, MOV, atau WebM hingga ${MAX_VIDEO_MB} MB dan 10 menit`}
                  validate={(f) => (f.size > MAX_VIDEO_MB * 1024 * 1024 ? `Ukuran video melebihi ${MAX_VIDEO_MB} MB.` : null)}
                  onFile={choose}
                />
              )}
              {mi.status === "ready" && mi.info.demo_video_available && (
                <p className="mt-3 text-[13px] leading-relaxed text-muted">
                  Video demo disusun dari foto dataset (wajah disamarkan) untuk mencoba alur aplikasi. Model pernah melihat foto-foto
                  tersebut, jadi hasilnya bukan ukuran akurasi.
                </p>
              )}
            </div>
          </Panel>

          <Panel className="lg:col-span-5">
            <fieldset disabled={busy}>
              <legend className="text-[17px] font-bold text-night">Pengaturan</legend>
              <p className="mt-1 text-[14px] text-slate">Frame yang dianalisis per detik video.</p>
              <div className="mt-3 grid gap-2">
                {FPS_OPTIONS.map((o) => (
                  <label key={o.value} className={`flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 text-[14.5px] ${fps === o.value ? "border-faience bg-faience-soft" : "border-hairline bg-white"}`}>
                    <input type="radio" name="fps" value={o.value} checked={fps === o.value} onChange={() => setFps(o.value)} className="accent-[#0f766e]" />
                    <span className="font-semibold text-night">{o.label}</span>
                    <span className="text-muted">{o.hint}</span>
                  </label>
                ))}
              </div>
              <label className="mt-4 flex cursor-pointer items-start gap-3 text-[14.5px]">
                <input type="checkbox" checked={annotate} onChange={(e) => setAnnotate(e.target.checked)} className="mt-1 accent-[#0f766e]" />
                <span>
                  <span className="font-semibold text-night">Buat video beranotasi</span>
                  <span className="block text-muted">Kerangka tubuh, label fase, dan linimasa digambar di atas video.</span>
                </span>
              </label>
            </fieldset>
            <div className="mt-6 space-y-4">
              <PrimaryButton onClick={start} disabled={!file || !ready || busy} className="w-full">
                {busy ? "Sedang menganalisis…" : "Analisis video"}
              </PrimaryButton>
              {!file && !busy && <p className="text-center text-[13px] text-muted">Pilih video terlebih dahulu.</p>}
              {busy && <ProgressPanel stage={stage === "uploading" ? "uploading" : "processing"} uploadFraction={uploadFrac} job={job} />}
              {error && (
                <Notice tone="error" title="Analisis tidak berhasil">
                  {error}{" "}
                  {file && (
                    <button type="button" className="font-semibold text-faience underline" onClick={start}>
                      Coba lagi
                    </button>
                  )}
                </Notice>
              )}
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}
