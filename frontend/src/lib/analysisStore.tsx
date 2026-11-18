import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

/** Keeps the local preview of the analysed video while the user moves between pages. */
interface AnalysisState {
  localVideoUrl: string | null;
  localVideoName: string | null;
  setLocalVideo: (file: File | Blob | null, name?: string) => void;
}

const Ctx = createContext<AnalysisState | null>(null);

export function AnalysisProvider({ children }: { children: ReactNode }) {
  const [localVideoUrl, setUrl] = useState<string | null>(null);
  const [localVideoName, setName] = useState<string | null>(null);
  const setLocalVideo = useCallback((file: File | Blob | null, name?: string) => {
    setUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return file ? URL.createObjectURL(file) : null;
    });
    setName(file ? name || (file as File).name || "video" : null);
  }, []);
  const value = useMemo(() => ({ localVideoUrl, localVideoName, setLocalVideo }), [localVideoUrl, localVideoName, setLocalVideo]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAnalysis(): AnalysisState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAnalysis must be used inside AnalysisProvider");
  return v;
}
