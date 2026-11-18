/** Two interlaced squares (the 8-point khatam star) — the product mark. */
export function BrandMark({ size = 28, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className={className} aria-hidden="true">
      <g fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round">
        <rect x="9" y="9" width="14" height="14" />
        <rect x="9" y="9" width="14" height="14" transform="rotate(45 16 16)" />
      </g>
      <circle cx="16" cy="16" r="2.4" fill="#2dd4bf" />
    </svg>
  );
}

/** Very faint 8-fold lattice used behind the hero only. */
export function Lattice({ className = "" }: { className?: string }) {
  return (
    <svg className={className} aria-hidden="true" width="100%" height="100%">
      <defs>
        <pattern id="khatam" width="64" height="64" patternUnits="userSpaceOnUse">
          <g fill="none" stroke="currentColor" strokeWidth="1">
            <rect x="16" y="16" width="32" height="32" />
            <rect x="16" y="16" width="32" height="32" transform="rotate(45 32 32)" />
            <path d="M0 32h9M55 32h9M32 0v9M32 55v9" />
          </g>
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#khatam)" />
    </svg>
  );
}
