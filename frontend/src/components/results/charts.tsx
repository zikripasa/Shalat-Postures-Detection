import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, LabelList, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from "recharts";
import { formatPercent, formatTime } from "../../lib/format";
import { CHART, PHASE_COLORS } from "../../lib/palette";
import type { FrameRecord, Phase, VideoAnalysis, VisualClass } from "../../lib/types";
import { PHASE_ORDER } from "../PhaseTimeline";
import { ChartFrame } from "../ui";

const tick = { fill: CHART.ink2, fontSize: 12, fontFamily: "Plus Jakarta Sans" };

function TipBox({ children }: { children: React.ReactNode }) {
  return <div className="rounded-md border border-hairline bg-white px-3 py-2 text-[13px] text-night shadow-sm">{children}</div>;
}

export function DurationChart({ result }: { result: VideoAnalysis }) {
  const data = PHASE_ORDER.map((p) => ({ phase: p, label: result.phase_labels[p], seconds: result.summary.phase_seconds[p] ?? 0 })).filter(
    (d) => d.seconds > 0,
  );
  const max = Math.max(...data.map((d) => d.seconds), 1);
  return (
    <ChartFrame
      title="Durasi per fase"
      description="Total waktu setiap fase di seluruh video, termasuk segmen yang ditandai tidak pasti."
      table={{ caption: "Durasi per fase", columns: ["Fase", "Detik", "Porsi"], rows: data.map((d) => [d.label, d.seconds.toFixed(1), formatPercent(result.summary.phase_share[d.phase as Phase], 1)]) }}
    >
      <div style={{ height: 44 * data.length + 40 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }} barCategoryGap={10}>
            <CartesianGrid horizontal={false} stroke={CHART.grid} />
            <XAxis type="number" domain={[0, Math.ceil(max * 1.1)]} tick={tick} axisLine={{ stroke: CHART.axis }} tickLine={false} unit=" d" />
            <YAxis type="category" dataKey="label" width={176} tick={tick} axisLine={false} tickLine={false} />
            <Tooltip cursor={{ fill: "#f1f3f2" }} content={({ active, payload }) =>
              active && payload?.length ? (
                <TipBox>
                  <strong>{payload[0].payload.label}</strong>: {Number(payload[0].value).toFixed(1)} detik
                </TipBox>
              ) : null} />
            <Bar dataKey="seconds" radius={[0, 4, 4, 0]} maxBarSize={22} isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.phase} fill={PHASE_COLORS[d.phase as Phase]} />
              ))}
              <LabelList dataKey="seconds" position="right" formatter={(v: unknown) => `${Number(v).toFixed(1)} d`} style={{ fill: CHART.ink2, fontSize: 12 }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}

function decodedConfidence(f: FrameRecord): number | null {
  const vis: Record<Phase, VisualClass> = { berdiri: "qiyam", rukuk: "ruku", iktidal: "qiyam", sujud: "sujud", duduk_antara_sujud: "julus", tasyahud: "julus" };
  const p = f.probabilities?.[vis[f.phase]];
  return p == null ? null : p;
}

export function ConfidenceChart({ result, threshold }: { result: VideoAnalysis; threshold: number }) {
  const frames = result.frames ?? [];
  const data = frames.map((f) => ({ t: f.t, conf: decodedConfidence(f), top: f.confidence }));
  return (
    <ChartFrame
      title="Keyakinan model per frame"
      description="Skor model untuk postur fase yang dipilih pada setiap frame yang dianalisis. Di bawah garis ambang, segmen ditandai tidak pasti. Skor ini belum dikalibrasi."
      table={{
        caption: "Keyakinan per frame",
        columns: ["Waktu", "Fase", "Skor fase terpilih", "Skor tertinggi"],
        rows: frames.map((f) => [formatTime(f.t), result.phase_labels[f.phase], formatPercent(decodedConfidence(f)), formatPercent(f.confidence)]),
      }}
    >
      <div className="h-[240px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid vertical={false} stroke={CHART.grid} />
            <XAxis dataKey="t" type="number" domain={[0, "dataMax"]} tickFormatter={(v) => formatTime(v, 0)} tick={tick} axisLine={{ stroke: CHART.axis }} tickLine={false} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} tick={tick} axisLine={false} tickLine={false} width={44} />
            <ReferenceLine y={threshold} stroke={CHART.ink2} strokeWidth={1} label={{ value: "Ambang tidak pasti", position: "insideTopRight", fill: CHART.ink2, fontSize: 12 }} />
            <Tooltip content={({ active, payload }) =>
              active && payload?.length ? (
                <TipBox>
                  {formatTime(Number(payload[0].payload.t))}: skor fase {formatPercent(payload[0].payload.conf)}
                </TipBox>
              ) : null} />
            <Scatter dataKey="top" fill={CHART.deemph} isAnimationActive={false} shape="circle" legendType="none" />
            <Line dataKey="conf" stroke={CHART.series1} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <ul className="mt-2 flex flex-wrap gap-5 text-[13px] text-slate">
        <li className="flex items-center gap-2">
          <span className="h-[2px] w-5 rounded" style={{ background: CHART.series1 }} aria-hidden="true" />
          Skor postur fase terpilih
        </li>
        <li className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: CHART.deemph }} aria-hidden="true" />
          Skor tertinggi mentah per frame
        </li>
      </ul>
    </ChartFrame>
  );
}

export function StageLatencyChart({ result }: { result: VideoAnalysis }) {
  const names: Record<string, string> = { localize: "Lokalisasi orang", pose: "Pose (MediaPipe)", pose_classifier: "Klasifikasi pose", image_model: "Model visual (ONNX)" };
  const data = Object.entries(result.performance.mean_ms_per_frame)
    .filter(([k]) => k in names)
    .map(([k, v]) => ({ stage: names[k], ms: v }));
  return (
    <ChartFrame
      title="Waktu proses per frame"
      description={`Rata-rata ${Math.round(result.performance.mean_ms_per_frame.total ?? 0)} ms per frame di CPU server, ${result.performance.analysis_fps ?? "–"} frame per detik.`}
      table={{ caption: "Waktu per tahap", columns: ["Tahap", "Rata-rata (ms)", "P95 (ms)"], rows: data.map((d) => {
        const key = Object.keys(names).find((k) => names[k] === d.stage) ?? "";
        return [d.stage, d.ms.toFixed(1), (result.performance.p95_ms_per_frame[key] ?? 0).toFixed(1)];
      }) }}
    >
      <div style={{ height: 40 * data.length + 36 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }}>
            <CartesianGrid horizontal={false} stroke={CHART.grid} />
            <XAxis type="number" tick={tick} axisLine={{ stroke: CHART.axis }} tickLine={false} unit=" ms" />
            <YAxis type="category" dataKey="stage" width={150} tick={tick} axisLine={false} tickLine={false} />
            <Bar dataKey="ms" fill={CHART.series1} radius={[0, 4, 4, 0]} maxBarSize={20} isAnimationActive={false}>
              <LabelList dataKey="ms" position="right" formatter={(v: unknown) => `${Number(v).toFixed(0)} ms`} style={{ fill: CHART.ink2, fontSize: 12 }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartFrame>
  );
}
