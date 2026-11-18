import { useId, useRef, useState, type DragEvent } from "react";

interface Props {
  accept: string;
  hint: string;
  label: string;
  onFile: (f: File) => void;
  validate?: (f: File) => string | null;
}

export function Dropzone({ accept, hint, label, onFile, validate }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const id = useId();
  const [over, setOver] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  function take(f: File | undefined) {
    if (!f) return;
    const problem = validate?.(f) ?? null;
    setErr(problem);
    if (!problem) onFile(f);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setOver(false);
    take(e.dataTransfer.files?.[0]);
  }

  return (
    <div>
      <label
        htmlFor={id}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors ${
          over ? "border-faience bg-faience-soft" : "border-[#c7cecb] bg-white hover:border-faience"
        }`}
      >
        <svg width="34" height="34" viewBox="0 0 24 24" aria-hidden="true" className="text-faience">
          <path d="M12 16V4m0 0-4 4m4-4 4 4M5 15v3a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-3" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="mt-3 text-[15px] font-semibold text-night">{label}</span>
        <span className="mt-1 text-[13px] text-muted">{hint}</span>
        <input
          ref={inputRef}
          id={id}
          type="file"
          accept={accept}
          className="sr-only"
          onChange={(e) => {
            take(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </label>
      {err && (
        <p role="alert" className="mt-2 text-[14px] text-[#b42318]">
          {err}
        </p>
      )}
    </div>
  );
}
