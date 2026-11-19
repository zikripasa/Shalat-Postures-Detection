import type { ReactNode } from "react";
import { Link } from "react-router-dom";

export function PageHeader({ title, lead, actions }: { title: string; lead?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex flex-col gap-4 border-b border-hairline pb-6 md:flex-row md:items-end md:justify-between">
      <div className="max-w-3xl">
        <h1 className="text-[32px] font-extrabold leading-[1.1] tracking-[-0.02em] text-night sm:text-[38px]">{title}</h1>
        {lead && <p className="mt-3 max-w-[68ch] text-[16px] leading-relaxed text-slate">{lead}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function Section({ title, lead, children, id }: { title: string; lead?: ReactNode; children: ReactNode; id?: string }) {
  return (
    <section id={id} className="mt-12" aria-labelledby={id ? `${id}-title` : undefined}>
      <h2 id={id ? `${id}-title` : undefined} className="text-[22px] font-bold tracking-[-0.01em] text-night">
        {title}
      </h2>
      {lead && <p className="mt-1.5 max-w-[72ch] text-[15px] leading-relaxed text-slate">{lead}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

export function Panel({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-xl border border-hairline bg-paper p-5 sm:p-6 ${className}`}>{children}</div>;
}

type BtnProps = { children: ReactNode; onClick?: () => void; disabled?: boolean; type?: "button" | "submit"; className?: string; ariaLabel?: string };

export function PrimaryButton({ children, onClick, disabled, type = "button", className = "", ariaLabel }: BtnProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className={`inline-flex items-center justify-center gap-2 rounded-lg bg-faience px-5 py-2.5 text-[15px] font-semibold text-white transition-colors hover:bg-faience-2 disabled:cursor-not-allowed disabled:bg-[#9db9b6] ${className}`}
    >
      {children}
    </button>
  );
}

export function SecondaryButton({ children, onClick, disabled, type = "button", className = "", ariaLabel }: BtnProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className={`inline-flex items-center justify-center gap-2 rounded-lg border border-[#c7cecb] bg-white px-4 py-2.5 text-[15px] font-semibold text-night transition-colors hover:border-night disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
    >
      {children}
    </button>
  );
}

export function LinkButton({ to, children, variant = "primary" }: { to: string; children: ReactNode; variant?: "primary" | "secondary" }) {
  const cls =
    variant === "primary"
      ? "bg-faience text-white hover:bg-faience-2"
      : "border border-[#c7cecb] bg-white text-night hover:border-night";
  return (
    <Link to={to} className={`inline-flex items-center justify-center rounded-lg px-5 py-2.5 text-[15px] font-semibold transition-colors ${cls}`}>
      {children}
    </Link>
  );
}

export function StatTile({ label, value, detail }: { label: string; value: ReactNode; detail?: ReactNode }) {
  return (
    <div className="rounded-xl border border-hairline bg-paper px-5 py-4">
      <div className="text-[14px] text-slate">{label}</div>
      <div className="mt-1 text-[28px] font-bold leading-tight tracking-[-0.01em] text-night">{value}</div>
      {detail && <div className="mt-1 text-[13px] text-muted">{detail}</div>}
    </div>
  );
}

export function Notice({ tone = "info", title, children }: { tone?: "info" | "warning" | "error"; title?: string; children: ReactNode }) {
  const styles = {
    info: "border-[#b9d9d4] bg-faience-soft text-night",
    warning: "border-[#f1d38a] bg-[#fff8e6] text-night",
    error: "border-[#f0b4b4] bg-[#fdf1f1] text-night",
  }[tone];
  const icon = { info: "i", warning: "!", error: "×" }[tone];
  const iconBg = { info: "#0f766e", warning: "#b7791f", error: "#d03b3b" }[tone];
  return (
    <div role={tone === "error" ? "alert" : "note"} className={`flex gap-3 rounded-lg border px-4 py-3 text-[14px] leading-relaxed ${styles}`}>
      <span aria-hidden="true" className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[12px] font-bold text-white" style={{ background: iconBg }}>
        {icon}
      </span>
      <div>
        {title && <p className="font-semibold">{title}</p>}
        <div>{children}</div>
      </div>
    </div>
  );
}

export interface TableSpec {
  caption: string;
  columns: string[];
  rows: (string | number)[][];
}

/** Every chart ships with a table twin (accessibility / exact values). */
export function ChartFrame({ title, description, children, table }: { title: string; description?: ReactNode; children: ReactNode; table?: TableSpec }) {
  return (
    <figure className="rounded-xl border border-hairline bg-paper p-5 sm:p-6">
      <figcaption>
        <h3 className="text-[17px] font-bold text-night">{title}</h3>
        {description && <p className="mt-1 text-[14px] leading-relaxed text-slate">{description}</p>}
      </figcaption>
      <div className="mt-4">{children}</div>
      {table && (
        <details className="mt-3 text-[14px]">
          <summary className="cursor-pointer select-none text-faience hover:underline">Lihat data sebagai tabel</summary>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full border-collapse text-left tabular">
              <caption className="sr-only">{table.caption}</caption>
              <thead>
                <tr>
                  {table.columns.map((c) => (
                    <th key={c} scope="col" className="border-b border-hairline py-1.5 pr-4 font-semibold text-slate">
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((r, i) => (
                  <tr key={i}>
                    {r.map((v, j) => (
                      <td key={j} className="border-b border-hairline/60 py-1.5 pr-4">
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </figure>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-slate" role="status">
      <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="12" r="9" stroke="#cbd3d0" strokeWidth="3" fill="none" />
        <path d="M21 12a9 9 0 0 0-9-9" stroke="#0f766e" strokeWidth="3" fill="none" strokeLinecap="round" />
      </svg>
      {label}
    </span>
  );
}
