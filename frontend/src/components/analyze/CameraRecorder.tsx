import { useEffect, useRef, useState } from "react";
import { Notice, PrimaryButton, SecondaryButton } from "../ui";

const MAX_SECONDS = 180;

function pickMime(): string | undefined {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  if (typeof MediaRecorder === "undefined") return undefined;
  return candidates.find((m) => MediaRecorder.isTypeSupported?.(m));
}

export const cameraSupported = () =>
  typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder !== "undefined";

export function CameraRecorder({ onRecorded }: { onRecorded: (file: File) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const [state, setState] = useState<"off" | "preview" | "recording">("off");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => () => stopStream(), []);
  useEffect(() => {
    if (state !== "recording") return;
    const id = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [state]);
  useEffect(() => {
    if (state === "recording" && seconds >= MAX_SECONDS) stopRecording();
  });

  function stopStream() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }

  async function startCamera() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setState("preview");
    } catch (e) {
      const name = (e as DOMException).name;
      setError(
        name === "NotAllowedError"
          ? "Akses kamera ditolak. Izinkan kamera di pengaturan browser, lalu coba lagi."
          : name === "NotFoundError"
            ? "Kamera tidak ditemukan di perangkat ini."
            : "Kamera tidak dapat dibuka. Pastikan halaman dibuka melalui HTTPS atau localhost.",
      );
    }
  }

  function startRecording() {
    if (!streamRef.current) return;
    const mime = pickMime();
    const rec = new MediaRecorder(streamRef.current, mime ? { mimeType: mime } : undefined);
    chunks.current = [];
    rec.ondataavailable = (ev) => ev.data.size && chunks.current.push(ev.data);
    rec.onstop = () => {
      const type = rec.mimeType || mime || "video/webm";
      const ext = type.includes("mp4") ? "mp4" : "webm";
      const file = new File(chunks.current, `rekaman-kamera.${ext}`, { type: type.split(";")[0] });
      stopStream();
      setState("off");
      onRecorded(file);
    };
    rec.start(1000);
    recRef.current = rec;
    setSeconds(0);
    setState("recording");
  }

  function stopRecording() {
    if (recRef.current && recRef.current.state !== "inactive") recRef.current.stop();
  }

  return (
    <div className="space-y-3">
      <div className="relative aspect-video overflow-hidden rounded-lg bg-night">
        <video ref={videoRef} className="h-full w-full object-contain" muted playsInline aria-label="Pratinjau kamera" />
        {state === "off" && (
          <div className="absolute inset-0 flex items-center justify-center p-6 text-center text-[14px] text-white/80">
            Letakkan kamera menyamping agar seluruh tubuh terlihat dari kepala hingga kaki.
          </div>
        )}
        {state === "recording" && (
          <span className="absolute left-3 top-3 inline-flex items-center gap-2 rounded-full bg-black/60 px-3 py-1 text-[13px] text-white tabular">
            <span className="h-2 w-2 animate-pulse rounded-full bg-[#d03b3b]" aria-hidden="true" />
            Merekam {Math.floor(seconds / 60)}:{String(seconds % 60).padStart(2, "0")} / 3:00
          </span>
        )}
      </div>
      {error && <Notice tone="error">{error}</Notice>}
      <div className="flex flex-wrap gap-2">
        {state === "off" && <SecondaryButton onClick={startCamera}>Nyalakan kamera</SecondaryButton>}
        {state === "preview" && (
          <>
            <PrimaryButton onClick={startRecording}>Mulai merekam</PrimaryButton>
            <SecondaryButton
              onClick={() => {
                stopStream();
                setState("off");
              }}
            >
              Matikan kamera
            </SecondaryButton>
          </>
        )}
        {state === "recording" && <PrimaryButton onClick={stopRecording}>Selesai merekam</PrimaryButton>}
      </div>
    </div>
  );
}
