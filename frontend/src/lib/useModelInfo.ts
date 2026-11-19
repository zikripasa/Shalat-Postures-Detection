import { useEffect, useState } from "react";
import { getModelInfo } from "./api";
import type { ModelInfo } from "./types";

export type ModelInfoState = { status: "loading" } | { status: "ready"; info: ModelInfo } | { status: "error"; message: string };

export function useModelInfo(): ModelInfoState & { reload: () => void } {
  const [state, setState] = useState<ModelInfoState>({ status: "loading" });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    const ctrl = new AbortController();
    setState({ status: "loading" });
    getModelInfo(ctrl.signal)
      .then((info) => setState({ status: "ready", info }))
      .catch((e: Error) => {
        if (e.name !== "AbortError") setState({ status: "error", message: e.message });
      });
    return () => ctrl.abort();
  }, [nonce]);
  return { ...state, reload: () => setNonce((n) => n + 1) };
}
