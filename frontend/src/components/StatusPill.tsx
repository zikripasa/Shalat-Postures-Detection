import { useEffect, useState } from "react";
import { getHealth } from "../lib/api";
import type { Health } from "../lib/types";

type State = { kind: "loading" } | { kind: "ok"; health: Health } | { kind: "offline" };

export function useHealth(pollMs = 30000): State {
  const [state, setState] = useState<State>({ kind: "loading" });
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    const tick = () =>
      getHealth(ctrl.signal)
        .then((h) => alive && setState({ kind: "ok", health: h }))
        .catch(() => alive && setState({ kind: "offline" }));
    tick();
    const id = window.setInterval(tick, pollMs);
    return () => {
      alive = false;
      ctrl.abort();
      window.clearInterval(id);
    };
  }, [pollMs]);
  return state;
}

export function StatusPill({ tone = "dark" }: { tone?: "dark" | "light" }) {
  const s = useHealth();
  let dot = "#9ca3af";
  let text = "Memeriksa server…";
  if (s.kind === "offline") {
    dot = "#d03b3b";
    text = "Server tidak terhubung";
  } else if (s.kind === "ok" && s.health.model_loaded) {
    dot = "#0ca30c";
    text = `Model ${s.health.model_version ?? ""} aktif`;
  } else if (s.kind === "ok") {
    dot = "#fab219";
    text = "Model belum tersedia";
  }
  const cls = tone === "dark" ? "bg-white/10 text-white/90 ring-white/15" : "bg-white text-night ring-hairline";
  return (
    <span role="status" aria-live="polite" className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-sm ring-1 ${cls}`}>
      <span className="h-2 w-2 rounded-full" style={{ background: dot }} aria-hidden="true" />
      {text}
    </span>
  );
}
