import { Lattice } from "../components/BrandMark";
import { PHASE_ORDER } from "../components/PhaseTimeline";
import { PosturePictogram } from "../components/PosturePictogram";
import { LinkButton, Notice, Section } from "../components/ui";
import { formatNumber } from "../lib/format";
import { PHASE_COLORS } from "../lib/palette";
import type { Phase } from "../lib/types";
import { useModelInfo } from "../lib/useModelInfo";

const PHASE_NAMES: Record<Phase, string> = {
  berdiri: "Berdiri",
  rukuk: "Rukuk",
  iktidal: "Iktidal",
  sujud: "Sujud",
  duduk_antara_sujud: "Duduk antara sujud",
  tasyahud: "Tasyahud",
};

const SEEN_DIRECTLY: Phase[] = ["rukuk", "sujud"];

const STEPS = [
  {
    title: "Rekam atau unggah video",
    body: "Gunakan kamera perangkat atau berkas MP4/WebM. Video hanya diproses di server Salat Vision dan dihapus setelah analisis selesai.",
  },
  {
    title: "Temukan orang dan kerangka tubuhnya",
    body: "Detektor orang memotong area tubuh, lalu MediaPipe Pose menandai 33 titik tubuh untuk mengukur sudut punggung, pinggul, dan lutut.",
  },
  {
    title: "Kenali postur pada setiap frame",
    body: "Dua model dipadukan: regresi logistik atas sudut tubuh dan klasifikasi tampilan visual dari MobileNetV3. Hasilnya empat postur: berdiri, rukuk, sujud, duduk.",
  },
  {
    title: "Susun menjadi urutan gerakan",
    body: "Model urutan menghaluskan perubahan antar-frame, lalu menamai Iktidal, Duduk di antara dua sujud, dan Tasyahud dari posisinya dalam urutan.",
  },
];

export function HomePage() {
  const mi = useModelInfo();
  const cv = mi.status === "ready" ? mi.info.cv_metrics : null;
  return (
    <>
      <section className="relative overflow-hidden border-b border-hairline bg-paper">
        <Lattice className="pointer-events-none absolute inset-0 text-[#e5ebe9]" />
        <div className="relative mx-auto grid max-w-6xl gap-10 px-5 py-14 sm:px-8 md:py-20 lg:grid-cols-12 lg:items-center">
          <div className="lg:col-span-6">
            <h1 className="text-[40px] font-extrabold leading-[1.04] tracking-[-0.03em] text-night sm:text-[54px]">
              Kenali urutan gerakan shalat dari sebuah video.
            </h1>
            <p className="mt-5 max-w-[56ch] text-[17px] leading-relaxed text-slate">
              Salat Vision mengenali postur yang terlihat pada setiap frame, lalu menyusunnya menjadi linimasa gerakan lengkap dengan
              waktu, durasi, dan tingkat keyakinan model.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <LinkButton to="/analisis">Analisis video</LinkButton>
              <LinkButton to="/model" variant="secondary">
                Lihat kinerja model
              </LinkButton>
            </div>
            <p className="mt-6 max-w-[56ch] text-[14px] leading-relaxed text-muted" aria-live="polite">
              {mi.status === "loading" && "Memuat status model…"}
              {mi.status === "error" && "Server analisis belum terhubung. Jalankan backend untuk mulai menganalisis."}
              {mi.status === "ready" && !mi.info.available && `Model belum tersedia: ${mi.info.reason ?? "bundel model tidak ditemukan"}.`}
              {mi.status === "ready" && mi.info.available && cv &&
                `Model ${mi.info.version} aktif. Pada validasi silang berkelompok, F1 makro ${formatNumber(cv.crop.macro_f1_mean)} (IK 95% ${formatNumber(cv.crop.macro_f1_ci95[0])}–${formatNumber(cv.crop.macro_f1_ci95[1])}) dari ${mi.info.training?.dataset?.instances ?? "–"} contoh berlabel.`}
            </p>
          </div>
          <div className="lg:col-span-6">
            <figure className="rounded-2xl border border-[#d6c7a3] bg-paper/95 p-2 shadow-[0_1px_0_#d6c7a3]">
              <div className="rounded-xl border border-[#e7dcc3] px-4 pb-5 pt-4">
                <ol className="draw-once grid grid-cols-3 items-start gap-x-1 gap-y-5 text-night sm:grid-cols-6" aria-label="Urutan gerakan dalam satu rakaat">
                  {PHASE_ORDER.map((p, i) => (
                    <li key={p} className="flex flex-col items-center text-center">
                      <PosturePictogram phase={p} size={104} />
                      <span className="mt-1 h-1 w-8 rounded-full" style={{ background: PHASE_COLORS[p] }} aria-hidden="true" />
                      <span className="mt-2 text-[13px] font-semibold leading-tight">
                        <span className="sr-only">Langkah {i + 1}: </span>
                        {PHASE_NAMES[p]}
                      </span>
                      <span className="mt-0.5 text-[11.5px] leading-tight text-muted">
                        {SEEN_DIRECTLY.includes(p) ? "dari gambar" : "gambar + urutan"}
                      </span>
                    </li>
                  ))}
                </ol>
              </div>
              <figcaption className="px-3 pb-1 pt-3 text-[13px] text-slate">
                Satu rakaat: enam fase, empat postur visual. Iktidal tampak seperti berdiri, dan kedua jenis duduk tampak serupa.
                Perbedaannya dibaca dari urutan.
              </figcaption>
            </figure>
          </div>
        </div>
      </section>

      <div className="mx-auto max-w-6xl px-5 pb-16 sm:px-8">
        <Section id="cara-kerja" title="Cara kerja" lead="Empat tahap yang dijalankan untuk setiap video, semuanya di server Anda sendiri.">
          <ol className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {STEPS.map((s, i) => (
              <li key={s.title} className="border-t-2 border-night pt-4">
                <span className="text-[13px] font-semibold text-brass tabular">Tahap {i + 1}</span>
                <h3 className="mt-1 text-[17px] font-bold leading-snug text-night">{s.title}</h3>
                <p className="mt-2 text-[14.5px] leading-relaxed text-slate">{s.body}</p>
              </li>
            ))}
          </ol>
        </Section>

        <Section
          id="dikenali"
          title="Apa yang dikenali, dan bagaimana"
          lead="Dataset pelatihan hanya memberi label empat postur. Tiga fase lain diturunkan dari konteks urutan, bukan dari gambar."
        >
          <div className="overflow-x-auto rounded-xl border border-hairline bg-paper">
            <table className="w-full min-w-[640px] border-collapse text-left text-[14.5px]">
              <thead className="text-slate">
                <tr className="border-b border-hairline">
                  <th scope="col" className="px-5 py-3 font-semibold">Fase</th>
                  <th scope="col" className="px-5 py-3 font-semibold">Postur visual</th>
                  <th scope="col" className="px-5 py-3 font-semibold">Cara ditentukan</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["berdiri", "Berdiri", "Postur berdiri yang tidak didahului rukuk"],
                  ["rukuk", "Rukuk", "Langsung dari postur"],
                  ["iktidal", "Berdiri", "Postur berdiri tepat setelah rukuk"],
                  ["sujud", "Sujud", "Langsung dari postur"],
                  ["duduk_antara_sujud", "Duduk", "Postur duduk yang diapit dua sujud"],
                  ["tasyahud", "Duduk", "Postur duduk setelah sujud kedua"],
                ].map(([p, vis, how]) => (
                  <tr key={p} className="border-b border-hairline/70 last:border-0">
                    <td className="px-5 py-3 font-semibold text-night">
                      <span className="mr-2 inline-block h-2.5 w-2.5 rounded-[3px] align-middle" style={{ background: PHASE_COLORS[p as Phase] }} aria-hidden="true" />
                      {PHASE_NAMES[p as Phase]}
                    </td>
                    <td className="px-5 py-3 text-slate">{vis}</td>
                    <td className="px-5 py-3 text-slate">{how}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>

        <Section id="batasan" title="Batasan yang perlu diketahui">
          <div className="grid gap-3 lg:grid-cols-2">
            <Notice tone="info" title="Bukan penilaian ibadah">
              Sistem mengenali postur tubuh yang terlihat kamera. Sistem tidak menilai sah atau tidaknya shalat, kekhusyukan, maupun
              bacaan.
            </Notice>
            <Notice tone="warning" title="Data pelatihan masih sangat kecil">
              Model dilatih pada 40 orang beranotasi dari sampel publik Salat Postures Dataset. Kinerja pada kamera, pakaian, dan
              ruangan lain belum teruji, sehingga segmen berkeyakinan rendah ditandai sebagai tidak pasti.
            </Notice>
          </div>
        </Section>
      </div>
    </>
  );
}
