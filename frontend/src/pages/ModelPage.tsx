import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ChartFrame, Notice, PageHeader, Panel, Section, Spinner, StatTile } from "../components/ui";
import { formatNumber, formatPercent, MODEL_LABEL } from "../lib/format";
import { CHART } from "../lib/palette";
import type { ModelInfo, VisualClass } from "../lib/types";
import { useModelInfo } from "../lib/useModelInfo";

const tick = { fill: CHART.ink2, fontSize: 12, fontFamily: "Plus Jakarta Sans" };
const CLASS_ID: Record<VisualClass, string> = { qiyam: "Berdiri", ruku: "Rukuk", sujud: "Sujud", julus: "Duduk" };
const CLASSES: VisualClass[] = ["qiyam", "ruku", "sujud", "julus"];

/** "3x repeated stratified group 5-fold CV" -> Indonesian sentence with the same numbers. */
function protocolId(protocol: string): string {
  const m = /(\d+)x repeated .*?(\d+)-fold/.exec(protocol);
  return m ? `Validasi silang berkelompok ${m[2]}-fold yang diulang ${m[1]} kali, tanpa kebocoran sesi antar-fold.` : protocol;
}

function ComparisonChart({ info, view }: { info: ModelInfo; view: "crop" | "image" }) {
  const sel = info.cv_metrics?.model_key ?? "fusion_pose_cnn";
  const rows = (info.model_comparison ?? [])
    .map((r) => {
      const mean = view === "crop" ? r.crop_macro_f1 : r.image_macro_f1;
      const lo = view === "crop" ? r.crop_macro_f1_ci_lo : r.image_macro_f1_ci_lo;
      const hi = view === "crop" ? r.crop_macro_f1_ci_hi : r.image_macro_f1_ci_hi;
      return { model: r.model, label: MODEL_LABEL[r.model] ?? r.model, mean, err: [mean - lo, hi - mean] as [number, number], lo, hi };
    })
    .sort((a, b) => b.mean - a.mean);
  return (
    <ChartFrame
      title={view === "crop" ? "F1 makro pada potongan orang beranotasi" : "F1 makro dari gambar utuh (lokalisasi otomatis)"}
      description="Rata-rata 3 pengulangan validasi silang 5-fold berkelompok. Garis tipis adalah interval kepercayaan 95% (bootstrap per kelompok). Model terpilih berwarna biru."
      table={{ caption: "Perbandingan model", columns: ["Model", "F1 makro", "IK 95% bawah", "IK 95% atas"], rows: rows.map((r) => [r.label, formatNumber(r.mean), formatNumber(r.lo), formatNumber(r.hi)]) }}
    >
      <div style={{ height: rows.length * 34 + 48 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 48, bottom: 8, left: 8 }} barCategoryGap={8}>
            <CartesianGrid horizontal={false} stroke={CHART.grid} />
            <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => formatNumber(v, 1)} tick={tick} axisLine={{ stroke: CHART.axis }} tickLine={false} />
            <YAxis type="category" dataKey="label" width={250} tick={tick} axisLine={false} tickLine={false} />
            <Tooltip cursor={{ fill: "#f1f3f2" }} content={({ active, payload }) =>
              active && payload?.length ? (
                <div className="rounded-md border border-hairline bg-white px-3 py-2 text-[13px] shadow-sm">
                  <strong>{payload[0].payload.label}</strong>
                  <br />F1 makro {formatNumber(payload[0].payload.mean)} (IK 95% {formatNumber(payload[0].payload.lo)}–{formatNumber(payload[0].payload.hi)})
                </div>
              ) : null} />
            <Bar dataKey="mean" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false}>
              {rows.map((r) => (
                <Cell key={r.model} fill={r.model === sel ? CHART.series1 : CHART.deemph} />
              ))}
              <ErrorBar dataKey="err" direction="x" width={6} stroke={CHART.ink} strokeWidth={1} />
              <LabelList
                dataKey="mean"
                content={(props) => {
                  const { x, y, width, height, index } = props as { x: number; y: number; width: number; height: number; index: number };
                  const r = rows[index];
                  if (!r || r.model !== sel || !r.mean) return null;
                  const px = x + width * (r.hi / r.mean) + 8; // just beyond the upper CI whisker
                  return (
                    <text x={px} y={y + height / 2} dy={4} fill={CHART.ink} fontSize={12} fontWeight={600}>
                      {formatNumber(r.mean)}
                    </text>
                  );
                }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

function ConfusionMatrix({ cm, title }: { cm: number[][]; title: string }) {
  const rows = cm.map((r) => r.slice(0, 4));
  const abstain = cm.map((r) => r[4] ?? 0);
  const hasAbstain = abstain.some((v) => v > 0);
  const max = Math.max(...rows.flat(), 1);
  const color = (v: number) => {
    const i = Math.min(CHART.seq.length - 1, Math.round((v / max) * (CHART.seq.length - 1)));
    return CHART.seq[i];
  };
  return (
    <ChartFrame
      title={title}
      description="Baris = postur sebenarnya, kolom = prediksi. Angka adalah jumlah prediksi out-of-fold dari ketiga pengulangan (setiap contoh muncul tiga kali)."
      table={{ caption: title, columns: ["Sebenarnya \\ Prediksi", ...CLASSES.map((c) => CLASS_ID[c]), ...(hasAbstain ? ["Abstain"] : [])], rows: rows.map((r, i) => [CLASS_ID[CLASSES[i]], ...r, ...(hasAbstain ? [abstain[i]] : [])]) }}
    >
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-[3px] text-[14px] tabular">
          <thead>
            <tr>
              <th className="p-1 text-left text-[12px] font-normal text-muted">Sebenarnya ↓ / Prediksi →</th>
              {CLASSES.map((c) => (
                <th key={c} scope="col" className="w-[72px] p-1 text-center text-[13px] font-semibold text-slate">{CLASS_ID[c]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <th scope="row" className="pr-3 text-left text-[13px] font-semibold text-slate">{CLASS_ID[CLASSES[i]]}</th>
                {r.map((v, j) => (
                  <td key={j} className="h-12 w-[72px] rounded-[4px] text-center font-semibold" style={{ background: color(v), color: v / max > 0.55 ? "#ffffff" : CHART.ink }}>
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartFrame>
  );
}

export function ModelPage() {
  const mi = useModelInfo();
  const [view, setView] = useState<"crop" | "image">("crop");
  if (mi.status === "loading") return <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8"><Spinner label="Memuat informasi model…" /></div>;
  if (mi.status === "error")
    return (
      <div className="mx-auto max-w-6xl px-5 py-10 sm:px-8">
        <Notice tone="error" title="Server analisis tidak terhubung">{mi.message}</Notice>
      </div>
    );
  const info = mi.info;
  const cv = info.cv_metrics;
  const tb = info.temporal_benchmark as { results?: Record<string, Record<string, Record<string, { mean: number }>>>; configured_decoder?: string } | null;
  const rt = info.runtime_benchmark as { hardware?: Record<string, unknown>; stages?: Record<string, { mean_ms: number; p95_ms: number }>; throughput_fps?: number; embedder?: Record<string, { mean_ms: number }> } | null;
  const ds = info.dataset as { n_labeled_instances?: number; n_labeled_images?: number; n_groups?: number; instances_per_class?: Record<string, number>; n_image_files?: number } | null;
  const train = info.training as { pose_classifier?: { best_params?: Record<string, unknown> }; image_classifier?: { best_params?: Record<string, unknown> } } | undefined;
  return (
    <div className="mx-auto max-w-6xl px-5 py-10 sm:px-8">
      <PageHeader
        title="Model"
        lead={info.available ? `Bundel model ${info.version} aktif, dibuat ${info.created_utc?.slice(0, 10)}. Semua angka di halaman ini berasal dari eksperimen yang benar-benar dijalankan.` : `Model belum tersedia: ${info.reason}`}
      />
      {!info.available && <div className="mt-6"><Notice tone="warning">Jalankan scripts/train_final.py untuk membuat bundel model.</Notice></div>}

      {cv && (
        <Section id="ringkasan" title="Kinerja model terpilih" lead={`${protocolId(cv.protocol)} Interval kepercayaan 95% dihitung dengan bootstrap per kelompok sesi rekaman.`}>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatTile label="F1 makro, potongan orang" value={formatNumber(cv.crop.macro_f1_mean)} detail={`IK 95% ${formatNumber(cv.crop.macro_f1_ci95[0])}–${formatNumber(cv.crop.macro_f1_ci95[1])}`} />
            <StatTile label="Akurasi, potongan orang" value={formatPercent(cv.crop.accuracy_mean)} detail={`IK 95% ${formatPercent(cv.crop.accuracy_ci95[0])}–${formatPercent(cv.crop.accuracy_ci95[1])}`} />
            <StatTile label="F1 makro, gambar utuh" value={formatNumber(cv.image.macro_f1_mean)} detail={`IK 95% ${formatNumber(cv.image.macro_f1_ci95[0])}–${formatNumber(cv.image.macro_f1_ci95[1])}`} />
            <StatTile label="Galat kalibrasi (ECE)" value={formatNumber(cv.ece_crop)} detail="Skor belum dikalibrasi" />
          </div>
        </Section>
      )}

      <Section id="alur" title="Alur model" lead="Satu frame melewati empat tahap. Dua model postur dipadukan dengan bobot sama.">
        <Panel>
          <ol className="grid gap-4 text-[14px] md:grid-cols-4">
            {[
              ["Lokalisasi orang", "EfficientDet-Lite0 (COCO, kelas orang). Cadangan: kotak dari kerangka pose, lalu seluruh frame."],
              ["Dua model postur", `${(info.pipeline?.pose_features as string[] | undefined)?.length ?? "–"} fitur sudut, posisi, dan visibilitas dari MediaPipe Pose ke regresi logistik (C=${String(train?.pose_classifier?.best_params?.clf__C ?? "–")}); embedding MobileNetV3-Large beku ke regresi logistik (C=${String(train?.image_classifier?.best_params?.clf__C ?? "–")}).`],
              ["Fusi", `Rata-rata skor dengan bobot pose ${info.pipeline?.fusion?.w_pose ?? 0.5}. Jika pose tidak terdeteksi, hanya model visual yang dipakai.`],
              ["Dekoder urutan", `HMM 7 keadaan mengikuti tata urutan shalat, lalu aturan konteks lokal menamai fase (${tb?.configured_decoder ?? "hmm_local"}).`],
            ].map(([t, d], i) => (
              <li key={t} className="border-l-2 border-faience pl-4">
                <span className="text-[13px] font-semibold text-brass">Tahap {i + 1}</span>
                <p className="mt-0.5 font-bold text-night">{t}</p>
                <p className="mt-1 leading-relaxed text-slate">{d}</p>
              </li>
            ))}
          </ol>
        </Panel>
      </Section>

      {info.model_comparison && (
        <Section id="perbandingan" title="Perbandingan 11 model" lead="Semua model dievaluasi dengan protokol yang sama. Selisih antar model teratas masih berada dalam interval kepercayaan yang lebar.">
          <div className="mb-4 inline-flex gap-1 rounded-lg border border-hairline bg-white p-1" role="tablist" aria-label="Jenis evaluasi">
            {(["crop", "image"] as const).map((v) => (
              <button key={v} role="tab" aria-selected={view === v} onClick={() => setView(v)} className={`rounded-md px-4 py-2 text-[14px] font-semibold ${view === v ? "bg-night text-white" : "text-slate hover:text-night"}`}>
                {v === "crop" ? "Potongan beranotasi" : "Gambar utuh"}
              </button>
            ))}
          </div>
          <ComparisonChart info={info} view={view} />
        </Section>
      )}

      {cv && (
        <Section id="per-kelas" title="Kinerja per postur">
          <div className="grid gap-6 lg:grid-cols-2">
            <ConfusionMatrix cm={cv.confusion_matrix_crop} title="Matriks kebingungan (potongan beranotasi)" />
            <ChartFrame title="Presisi, recall, dan F1 per postur" description="Dari prediksi out-of-fold model terpilih pada potongan beranotasi.">
              <table className="w-full border-collapse text-left text-[14px] tabular">
                <thead className="text-slate">
                  <tr className="border-b border-hairline">
                    <th scope="col" className="py-2 pr-3 font-semibold">Postur</th>
                    <th scope="col" className="py-2 pr-3 font-semibold">Presisi</th>
                    <th scope="col" className="py-2 pr-3 font-semibold">Recall</th>
                    <th scope="col" className="py-2 pr-3 font-semibold">F1</th>
                    <th scope="col" className="py-2 font-semibold">Contoh</th>
                  </tr>
                </thead>
                <tbody>
                  {CLASSES.map((c) => {
                    const m = cv.per_class_crop[c];
                    return (
                      <tr key={c} className="border-b border-hairline/70 last:border-0">
                        <td className="py-2.5 pr-3 font-semibold text-night">{CLASS_ID[c]}</td>
                        <td className="py-2.5 pr-3">{formatNumber(m.precision)}</td>
                        <td className="py-2.5 pr-3">{formatNumber(m.recall)}</td>
                        <td className="py-2.5 pr-3">{formatNumber(m.f1)}</td>
                        <td className="py-2.5 text-slate">{m.support / 3} × 3</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </ChartFrame>
          </div>
        </Section>
      )}

      {tb?.results && (
        <Section id="urutan" title="Evaluasi dekoder urutan" lead="Belum ada video shalat berlabel. Urutan disimulasikan dari skor out-of-fold nyata model: 300 rangkaian 2–4 rakaat per skenario.">
          <div className="overflow-x-auto rounded-xl border border-hairline bg-paper">
            <table className="w-full min-w-[720px] border-collapse text-left text-[14px] tabular">
              <thead className="text-slate">
                <tr className="border-b border-hairline">
                  <th scope="col" className="px-4 py-3 font-semibold">Skenario galat</th>
                  <th scope="col" className="px-4 py-3 font-semibold">Dekoder</th>
                  <th scope="col" className="px-4 py-3 font-semibold">Akurasi fase per frame</th>
                  <th scope="col" className="px-4 py-3 font-semibold">Skor edit</th>
                  <th scope="col" className="px-4 py-3 font-semibold">F1@50</th>
                </tr>
              </thead>
              <tbody>
                {(["segment", "iid"] as const).flatMap((mode) =>
                  ["raw", "smoothing", "hmm", "hmm_local"].map((d) => {
                    const r = tb.results?.[mode]?.[d];
                    if (!r) return null;
                    const chosen = d === (tb.configured_decoder ?? "hmm_local");
                    return (
                      <tr key={`${mode}-${d}`} className={`border-b border-hairline/70 last:border-0 ${chosen ? "bg-faience-soft" : ""}`}>
                        <td className="px-4 py-2.5 text-slate">{mode === "segment" ? "Galat per segmen (konservatif)" : "Galat per frame (optimistis)"}</td>
                        <td className="px-4 py-2.5 font-semibold text-night">
                          {{ raw: "Mentah per frame", smoothing: "Perataan + aturan", hmm: "HMM global", hmm_local: "HMM + aturan lokal (dipakai)" }[d]}
                        </td>
                        <td className="px-4 py-2.5">{formatPercent(r.phase_acc.mean, 1)}</td>
                        <td className="px-4 py-2.5">{formatNumber(r.edit.mean, 1)}</td>
                        <td className="px-4 py-2.5">{formatNumber(r.f1_50.mean, 1)}</td>
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      <div className="mt-12 grid gap-6 lg:grid-cols-2">
        {ds && (
          <Panel>
            <h2 className="text-[19px] font-bold text-night">Data pelatihan</h2>
            <p className="mt-2 text-[14.5px] leading-relaxed text-slate">
              {ds.n_labeled_instances} orang beranotasi pada {ds.n_labeled_images} gambar dari {ds.n_image_files} berkas gambar di sampel publik
              Salat Postures Dataset. Setelah pengelompokan duplikat dan sesi rekaman, hanya {ds.n_groups} kelompok independen yang tersisa.
            </p>
            <ul className="mt-3 grid grid-cols-2 gap-2 text-[14px]">
              {CLASSES.map((c) => (
                <li key={c} className="flex justify-between rounded-md bg-limestone px-3 py-2">
                  <span>{CLASS_ID[c]}</span>
                  <span className="font-semibold tabular">{ds.instances_per_class?.[c] ?? "–"}</span>
                </li>
              ))}
            </ul>
          </Panel>
        )}
        {rt?.stages && (
          <Panel>
            <h2 className="text-[19px] font-bold text-night">Kecepatan pada server ini</h2>
            <p className="mt-2 text-[14.5px] leading-relaxed text-slate">
              {String(rt.hardware?.cpu ?? "CPU")}, {String(rt.hardware?.logical_cores ?? "–")} inti logis, tanpa GPU. Rata-rata {formatNumber(rt.stages.total?.mean_ms, 0)} ms per frame ({formatNumber(rt.throughput_fps, 1)} frame per detik).
              Embedding ONNX Runtime {formatNumber(rt.embedder?.onnxruntime_ms?.mean_ms, 1)} ms dibanding PyTorch {formatNumber(rt.embedder?.pytorch_ms?.mean_ms, 1)} ms.
            </p>
          </Panel>
        )}
      </div>

      <Section id="batasan" title="Penggunaan dan batasan">
        <div className="grid gap-3">
          {info.intended_use && <Notice tone="info" title="Tujuan penggunaan">Demonstrasi riset dan edukasi untuk mengenali postur shalat yang terlihat pada gambar dan video pendek.</Notice>}
          <Notice tone="warning" title="Di luar cakupan">Menilai sah atau tidaknya shalat, mengenali identitas orang, atau pengawasan.</Notice>
          {info.limitations && (
            <Panel>
              <ul className="list-disc space-y-1.5 pl-5 text-[14.5px] leading-relaxed text-slate">
                <li>
                  Dilatih pada {ds?.n_labeled_instances ?? "–"} orang beranotasi ({ds?.n_groups ?? "–"} kelompok independen); estimasi kinerja
                  memiliki ketidakpastian besar.
                </li>
                <li>Iktidal, duduk di antara dua sujud, dan tasyahud disimpulkan dari urutan postur, bukan dilihat langsung.</li>
                <li>Deteksi pose lebih sering gagal saat sujud (kepala menunduk, tubuh terlipat).</li>
                <li>Belum divalidasi pada video shalat nyata yang berkesinambungan.</li>
              </ul>
            </Panel>
          )}
          {info.third_party && (
            <p className="text-[13px] text-muted">
              Model pihak ketiga: {info.third_party.map((t) => `${t.name} (${t.license})`).join(", ")}; bobot ImageNet torchvision (BSD-3-Clause).
            </p>
          )}
        </div>
      </Section>
    </div>
  );
}
