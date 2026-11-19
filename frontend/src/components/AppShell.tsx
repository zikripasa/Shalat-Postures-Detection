import { NavLink, Outlet } from "react-router-dom";
import { BrandMark } from "./BrandMark";
import { StatusPill } from "./StatusPill";

const NAV = [
  { to: "/", label: "Beranda", end: true },
  { to: "/analisis", label: "Analisis" },
  { to: "/model", label: "Model" },
];

export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col">
      <a href="#konten" className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded-md focus:bg-white focus:px-3 focus:py-2">
        Lewati ke konten
      </a>
      <header className="bg-night text-white">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-8 gap-y-3 px-5 py-3.5 sm:px-8">
          <NavLink to="/" className="flex items-center gap-2.5 text-white">
            <BrandMark className="text-brass" />
            <span className="text-[17px] font-bold tracking-tight">Salat Vision</span>
          </NavLink>
          <nav aria-label="Navigasi utama" className="order-3 w-full sm:order-none sm:w-auto">
            <ul className="flex gap-1">
              {NAV.map((n) => (
                <li key={n.to}>
                  <NavLink
                    to={n.to}
                    end={n.end}
                    className={({ isActive }) =>
                      `rounded-md px-3 py-1.5 text-[15px] transition-colors ${isActive ? "bg-white/12 text-white" : "text-white/70 hover:text-white"}`
                    }
                  >
                    {n.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
          <div className="ml-auto">
            <StatusPill />
          </div>
        </div>
      </header>
      <main id="konten" className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-hairline bg-paper">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-5 py-6 text-sm text-slate sm:px-8 md:flex-row md:justify-between">
          <p className="max-w-2xl">
            Salat Vision mengenali postur yang terlihat pada gambar. Sistem ini tidak menilai sah atau tidaknya shalat, dan video
            Anda dihapus dari server setelah dianalisis.
          </p>
          <p className="text-muted">Proyek riset dan portofolio. Analisis berjalan lokal, tanpa API berbayar.</p>
        </div>
      </footer>
    </div>
  );
}
