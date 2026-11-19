import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { Spinner } from "./components/ui";
import { AnalysisProvider } from "./lib/analysisStore";
import { AnalyzePage } from "./pages/AnalyzePage";
import { HomePage } from "./pages/HomePage";
import { NotFoundPage } from "./pages/NotFoundPage";

// Chart-heavy pages (Recharts) load on demand to keep the first page light.
const ResultsPage = lazy(() => import("./pages/ResultsPage").then((m) => ({ default: m.ResultsPage })));
const ModelPage = lazy(() => import("./pages/ModelPage").then((m) => ({ default: m.ModelPage })));

const fallback = (
  <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8">
    <Spinner label="Memuat halaman…" />
  </div>
);

export function AppRoutes() {
  return (
    <Suspense fallback={fallback}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<HomePage />} />
          <Route path="analisis" element={<AnalyzePage />} />
          <Route path="hasil/:jobId" element={<ResultsPage />} />
          <Route path="model" element={<ModelPage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </Suspense>
  );
}

export function App() {
  return (
    <AnalysisProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AnalysisProvider>
  );
}
