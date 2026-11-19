import { EVIDENCE_LABEL, formatPercent, formatTime } from "../../lib/format";
import { PHASE_COLORS, UNCERTAIN_COLOR } from "../../lib/palette";
import type { Segment } from "../../lib/types";

export function SegmentTable({ segments, activeIndex, onSeek }: { segments: Segment[]; activeIndex: number | null; onSeek: (t: number) => void }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-hairline bg-paper">
      <table className="w-full min-w-[760px] border-collapse text-left text-[14px]">
        <caption className="sr-only">Daftar segmen fase dengan waktu dan keyakinan</caption>
        <thead className="text-slate">
          <tr className="border-b border-hairline">
            <th scope="col" className="px-4 py-3 font-semibold">No.</th>
            <th scope="col" className="px-4 py-3 font-semibold">Fase</th>
            <th scope="col" className="px-4 py-3 font-semibold">Mulai</th>
            <th scope="col" className="px-4 py-3 font-semibold">Selesai</th>
            <th scope="col" className="px-4 py-3 font-semibold">Durasi</th>
            <th scope="col" className="px-4 py-3 font-semibold">Keyakinan</th>
            <th scope="col" className="px-4 py-3 font-semibold">Dasar penentuan</th>
          </tr>
        </thead>
        <tbody className="tabular">
          {segments.map((s) => (
            <tr key={s.index} className={`border-b border-hairline/70 last:border-0 ${activeIndex === s.index ? "bg-faience-soft" : ""}`}>
              <td className="px-4 py-2.5 text-muted">{s.index + 1}</td>
              <td className="px-4 py-2.5">
                <button type="button" onClick={() => onSeek(s.start)} className="flex items-center gap-2 text-left font-semibold text-night hover:underline">
                  <span className="h-3 w-3 shrink-0 rounded-[3px]" style={{ background: s.uncertain ? UNCERTAIN_COLOR : PHASE_COLORS[s.phase] }} aria-hidden="true" />
                  {s.label}
                  {s.uncertain && <span className="rounded-full bg-[#fff3d6] px-2 py-0.5 text-[12px] font-semibold text-[#8a5a00]">Tidak pasti</span>}
                </button>
              </td>
              <td className="px-4 py-2.5">{formatTime(s.start)}</td>
              <td className="px-4 py-2.5">{formatTime(s.end)}</td>
              <td className="px-4 py-2.5">{s.duration.toFixed(1).replace(".", ",")} dtk</td>
              <td className="px-4 py-2.5">
                <span className="flex items-center gap-2">
                  <span className="h-1.5 w-16 overflow-hidden rounded-full bg-[#e6ebe9]" aria-hidden="true">
                    <span className="block h-full rounded-full bg-night" style={{ width: `${Math.round(s.confidence * 100)}%` }} />
                  </span>
                  {formatPercent(s.confidence)}
                </span>
              </td>
              <td className="px-4 py-2.5 text-slate">{EVIDENCE_LABEL[s.evidence] ?? s.evidence}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
