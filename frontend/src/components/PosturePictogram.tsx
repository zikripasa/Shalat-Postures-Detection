import type { Phase } from "../lib/types";
import { PHASE_COLORS } from "../lib/palette";

type Pt = [number, number];
interface Figure {
  head: Pt;
  bones: [Pt, Pt][];
  joints: Pt[];
  extra?: [Pt, Pt][];
}

// Side-view skeletons (viewBox 120x120, floor at y=108), drawn like pose landmarks.
const FIGURES: Record<Phase, Figure> = {
  berdiri: {
    head: [60, 21],
    bones: [[[60, 31], [60, 64]], [[60, 64], [60, 86]], [[60, 86], [60, 107]], [[60, 107], [69, 107]], [[60, 37], [68, 49]], [[68, 49], [59, 52]]],
    joints: [[60, 37], [60, 64], [60, 86], [68, 49]],
  },
  rukuk: {
    head: [95, 57],
    bones: [[[47, 62], [84, 60]], [[47, 62], [49, 85]], [[49, 85], [47, 107]], [[47, 107], [56, 107]], [[80, 61], [64, 72]], [[64, 72], [52, 82]]],
    joints: [[80, 61], [47, 62], [49, 85], [64, 72]],
  },
  iktidal: {
    head: [60, 21],
    bones: [[[60, 31], [60, 64]], [[60, 64], [60, 86]], [[60, 86], [60, 107]], [[60, 107], [69, 107]], [[60, 37], [62, 51]], [[62, 51], [62, 64]]],
    joints: [[60, 37], [60, 64], [60, 86], [62, 51]],
  },
  sujud: {
    head: [88, 99],
    bones: [[[44, 79], [76, 92]], [[44, 79], [51, 105]], [[51, 105], [27, 107]], [[76, 92], [81, 99]], [[81, 99], [84, 106]]],
    joints: [[44, 79], [51, 105], [76, 92], [81, 99]],
  },
  duduk_antara_sujud: {
    head: [57, 35],
    bones: [[[56, 45], [53, 84]], [[53, 84], [84, 101]], [[84, 101], [49, 107]], [[56, 50], [66, 70]], [[66, 70], [80, 92]]],
    joints: [[56, 50], [53, 84], [84, 101], [66, 70]],
  },
  tasyahud: {
    head: [57, 35],
    bones: [[[56, 45], [53, 88]], [[53, 88], [84, 101]], [[84, 101], [44, 107]], [[56, 50], [66, 70]], [[66, 70], [80, 91]]],
    joints: [[56, 50], [53, 88], [84, 101], [66, 70]],
    extra: [[[80, 91], [88, 86]]], // raised index finger
  },
};

export function PosturePictogram({ phase, size = 96, title }: { phase: Phase; size?: number; title?: string }) {
  const f = FIGURES[phase];
  const color = PHASE_COLORS[phase];
  return (
    <svg width={size} height={size} viewBox="0 0 120 120" role={title ? "img" : undefined} aria-label={title} aria-hidden={title ? undefined : true}>
      <line x1="14" y1="110" x2="106" y2="110" stroke="#c9ccd1" strokeWidth="1.5" strokeLinecap="round" />
      <g stroke="currentColor" strokeWidth="5.5" strokeLinecap="round" strokeLinejoin="round" fill="none">
        {f.bones.map(([a, b], i) => (
          <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />
        ))}
        {f.extra?.map(([a, b], i) => (
          <line key={`e${i}`} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} strokeWidth="3" />
        ))}
        <circle cx={f.head[0]} cy={f.head[1]} r="8.5" />
      </g>
      {/* only hip and knee joints: the two angles the model actually relies on */}
      {f.joints.slice(1, 3).map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="5" fill={color} stroke="#fcfcfb" strokeWidth="2.5" />
      ))}
    </svg>
  );
}
