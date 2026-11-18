import { INK_ON, PHASE_COLORS, UNCERTAIN_COLOR } from "../lib/palette";
import { formatTime, formatPercent } from "../lib/format";
import type { Phase, Segment } from "../lib/types";

export const PHASE_ORDER: Phase[] = ["berdiri", "rukuk", "iktidal", "sujud", "duduk_antara_sujud", "tasyahud"];

interface Props {
  segments: Segment[];
  duration: number;
  phaseLabels: Record<Phase, string>;
  currentTime?: number;
  activeIndex?: number | null;
  onSeek?: (t: number, segment: Segment) => void;
  height?: number;
}

/**
 * The signature component: a woven band of prayer phases. Each block is a button
 * (keyboard reachable) that seeks the video to the segment start. Uncertain segments
 * are neutral grey with their phase still named in the label and tooltip.
 */
export function PhaseTimeline({ segments, duration, phaseLabels, currentTime, activeIndex, onSeek, height = 56 }: Props) {
  const total = Math.max(duration, segments.length ? segments[segments.length - 1].end : 0, 1e-6);
  const usedPhases = PHASE_ORDER.filter((p) => segments.some((s) => s.phase === p));
  const hasUncertain = segments.some((s) => s.uncertain);
  return (
    <div>
      <div className="relative" style={{ height }}>
        <ol className="absolute inset-0 flex" aria-label="Linimasa fase shalat">
          {segments.map((s) => {
            const fill = s.uncertain ? UNCERTAIN_COLOR : PHASE_COLORS[s.phase];
            const widthPct = ((s.end - s.start) / total) * 100;
            const isActive = activeIndex === s.index;
            const showText = widthPct > 9;
            return (
              <li key={s.index} style={{ width: `${widthPct}%` }} className="h-full pr-[2px] last:pr-0">
                <button
                  type="button"
                  onClick={() => onSeek?.(s.start, s)}
                  title={`${s.label}${s.uncertain ? " (tidak pasti)" : ""}: ${formatTime(s.start)}–${formatTime(s.end)}, keyakinan ${formatPercent(s.confidence)}`}
                  aria-label={`${s.label}${s.uncertain ? ", tidak pasti" : ""}, ${formatTime(s.start)} sampai ${formatTime(s.end)}, keyakinan ${formatPercent(s.confidence)}`}
                  aria-current={isActive ? "true" : undefined}
                  className={`group flex h-full w-full items-end overflow-hidden rounded-[4px] px-1.5 pb-1.5 text-left transition-[filter] hover:brightness-95 ${isActive ? "ring-2 ring-night ring-offset-2 ring-offset-paper" : ""}`}
                  style={{ background: fill, color: INK_ON[fill] ?? "#0b0b0b" }}
                >
                  {showText && <span className="truncate text-[12px] font-semibold leading-none">{s.uncertain ? "Tidak pasti" : s.label}</span>}
                </button>
              </li>
            );
          })}
        </ol>
        {currentTime != null && (
          <div
            className="pointer-events-none absolute -top-1.5 -bottom-1.5 w-[2px] rounded bg-night"
            style={{ left: `calc(${Math.min(100, (currentTime / total) * 100)}% - 1px)` }}
            aria-hidden="true"
          />
        )}
      </div>
      <div className="mt-2 flex justify-between text-[12px] text-muted tabular" aria-hidden="true">
        <span>0:00</span>
        <span>{formatTime(total / 2, 0)}</span>
        <span>{formatTime(total, 0)}</span>
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-[13px] text-slate" aria-label="Keterangan warna">
        {usedPhases.map((p) => (
          <li key={p} className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-[3px]" style={{ background: PHASE_COLORS[p] }} aria-hidden="true" />
            {phaseLabels[p]}
          </li>
        ))}
        {hasUncertain && (
          <li className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-[3px]" style={{ background: UNCERTAIN_COLOR }} aria-hidden="true" />
            Tidak pasti
          </li>
        )}
      </ul>
    </div>
  );
}
