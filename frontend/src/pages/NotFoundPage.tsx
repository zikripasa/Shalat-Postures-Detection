import { LinkButton } from "../components/ui";

export function NotFoundPage() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-20 sm:px-8">
      <h1 className="text-[32px] font-extrabold tracking-[-0.02em] text-night">Halaman tidak ditemukan</h1>
      <p className="mt-3 text-slate">Alamat ini tidak ada di Salat Vision.</p>
      <div className="mt-6">
        <LinkButton to="/">Kembali ke beranda</LinkButton>
      </div>
    </div>
  );
}
