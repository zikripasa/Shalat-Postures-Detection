import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { PhaseTimeline } from "../components/PhaseTimeline";
import { ConfidenceChart, DurationChart, StageLatencyChart } from "../components/results/charts";
import { SegmentTable } from "../components/results/SegmentTable";
import { LinkButton, Notice, PageHeader, Section, Spinner, StatTile } from "../components/ui";
import { ApiError, getJob, jobReportUrl, jobVideoUrl } from "../lib/api";
import { useAnalysis } from "../lib/analysisStore";
import { formatDuration, formatNumber } from "../lib/format";
import type { JobStatus } from "../lib/types";
import { useModelInfo } from "../lib/useModelInfo";

export function ResultsPage() {
  const { jobId = "" } = useParams();
  const { localVideoUrl } = useAnalysis();
  const mi = useModelInfo();
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [time, setTime] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const load = async () => {
      try {
        const st = await getJob(jobId);
        if (!alive) return;
        setJob(st);
        if (st.status === "queued" || st.status === "running") timer = window.setTimeout(load, 800);
      } catch (e) {
        if (alive) setError(e instanceof ApiError && e.status === 404 ? "Hasil tidak ditemukan atau sudah kedaluwarsa (disimpan 30 menit)." : (e as Error).message);
      }
    };
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [jobId]);

  const result = job?.result ?? null;
  const activeIndex = useMemo(() => {
    if (!result) return null;
    const s = result.segments.find((x) => time >= x.start && time < x.end);
    return s ? s.index : null;
  }, [result, time]);

  const seek = (t: number) => {
    const v = videoRef.current;
    if (v) {
      v.currentTime = t + 0.01;
      setTime(t);
    }
  };

  if (error)
    return (
      <div className="mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <Notice tone="error" title="Hasil tidak tersedia">
          {error} <Link to="/analisis" className="font-semibold text-faience underline">Analisis video baru</Link>
        </Notice>
      </div>
    );
  if (!job || !result)
    return (
      <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8">
        {job?.status === "error" ? (
          <Notice tone="error" title="Analisis gagal">{job.error}</Notice>
        ) : (
          <Spinner label={job ? "Analisis masih berjalan…" : "Memuat hasil…"} />
        )}
      </div>
    );

  const s = result.summary;
  const threshold = mi.status === "ready" ? (mi.info.pipeline?.uncertainty_threshold as number | undefined) ?? 0.55 : 0.55;
  const videoSrc = job.annotated_video_available ? jobVideoUrl(job.job_id) : localVideoUrl;

  return (
    <div className="mx-auto max-w-6xl px-5 py-10 sm:px-8">
      <PageHeader
        title="Hasil analisis"
        lead={`${s.n_segments} segmen gerakan ditemukan dalam video berdurasi ${formatDuration(result.video.duration_seconds)}. Klik segmen pada linimasa untuk melompat ke bagian video tersebut.`}
      />


      <div className="mt-5 flex flex-wrap items-center gap-2 text-[14px]" aria-label="Unduhan">
        <span className="mr-1 text-slate">Unduh:</span>
        <a className="rounded-lg border border-[#c7cecb] bg-white px-3 py-1.5 font-semibold text-night hover:border-night" href={jobReportUrl(job.job_id, "json")} aria-label="Unduh laporan JSON">
          Laporan JSON
        </a>
        <a className="rounded-lg border border-[#c7cecb] bg-white px-3 py-1.5 font-semibold text-night hover:border-night" href={jobReportUrl(job.job_id, "csv")} aria-label="Unduh segmen CSV">
          Segmen CSV
        </a>
        {job.annotated_video_available && (
          <a className="rounded-lg border border-[#c7cecb] bg-white px-3 py-1.5 font-semibold text-night hover:border-night" href={jobVideoUrl(job.job_id)} download aria-label="Unduh video beranotasi">
            Video beranotasi
          </a>
        )}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-12">
        <div className="lg:col-span-8">
          {videoSrc ? (
            <video
              ref={videoRef}
              src={videoSrc}
              controls
              onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
              className="aspect-video w-full rounded-xl bg-night"
              aria-label={job.annotated_video_available ? "Video beranotasi" : "Video yang dianalisis"}
            />
          ) : (
            <div className="flex aspect-video items-center justify-center rounded-xl bg-night p-6 text-center text-white/80">
              Video tidak tersedia di sesi ini. Linimasa dan tabel tetap dapat dibaca.
            </div>
          )}
        </div>
        <div className="grid content-start gap-3 sm:grid-cols-2 lg:col-span-4 lg:grid-cols-1">
          <StatTile label="Siklus rakaat teramati" value={s.cycles_observed ?? "–"} detail="Dihitung dari sujud kedua yang terlihat" />
          <StatTile label="Segmen tidak pasti" value={`${s.n_uncertain_segments} dari ${s.n_segments}`} detail={`${s.uncertain_seconds.toFixed(1).replace(".", ",")} detik di bawah ambang keyakinan`} />
          <StatTile label="Kecepatan analisis" value={`${formatNumber(result.performance.analysis_fps, 1)} fps`} detail={`${result.performance.frames_analyzed} frame dalam ${result.performance.analysis_seconds.toFixed(1).replace(".", ",")} detik`} />
        </div>
      </div>

      <Section id="linimasa" title="Linimasa fase" lead="Urutan fase setelah pemrosesan urutan (HMM). Warna abu-abu berarti keyakinan model rendah.">
        <div className="rounded-xl border border-hairline bg-paper p-5 sm:p-6">
          <PhaseTimeline segments={result.segments} duration={result.video.duration_seconds} phaseLabels={result.phase_labels} currentTime={time} activeIndex={activeIndex} onSeek={seek} />
        </div>
        <details className="mt-3 rounded-xl border border-hairline bg-paper p-5 text-[14px] sm:p-6">
          <summary className="cursor-pointer font-semibold text-night">Bandingkan dengan prediksi mentah per frame</summary>
          <p className="mt-2 max-w-[72ch] text-slate">
            Prediksi mentah memakai label tertinggi setiap frame tanpa pemrosesan urutan. Perubahan yang sangat sering di sini adalah
            alasan dekoder urutan digunakan.
          </p>
          <div className="mt-4">
            <PhaseTimeline segments={result.raw_segments} duration={result.video.duration_seconds} phaseLabels={result.phase_labels} height={36} ariaLabel="Linimasa prediksi mentah per frame" />
          </div>
          <p className="mt-2 text-muted">
            Mentah: {result.raw_segments.length} segmen. Setelah pemrosesan: {result.segments.length} segmen.
          </p>
        </details>
      </Section>

      <Section id="segmen" title="Daftar segmen">
        <SegmentTable segments={result.segments} activeIndex={activeIndex} onSeek={seek} />
      </Section>

      <Section id="grafik" title="Analisis durasi dan keyakinan">
        <div className="grid gap-6 lg:grid-cols-2">
          <DurationChart result={result} />
          <StageLatencyChart result={result} />
        </div>
        <div className="mt-6">
          <ConfidenceChart result={result} threshold={threshold} />
        </div>
      </Section>

      <div className="mt-10 space-y-3">
        {result.notes.map((n) => (
          <Notice key={n} tone="info">{n}</Notice>
        ))}
        <div className="pt-2">
          <LinkButton to="/analisis" variant="secondary">Analisis video lain</LinkButton>
        </div>
      </div>
    </div>
  );
}
