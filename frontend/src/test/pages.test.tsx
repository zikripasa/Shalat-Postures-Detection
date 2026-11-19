import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { AppRoutes } from "../App";
import { ApiError, getHealth } from "../lib/api";
import { AnalysisProvider } from "../lib/analysisStore";
import fixture from "./fixtures/job_demo.json";

const modelInfo = {
  available: true,
  reason: null,
  version: "1.0.0",
  created_utc: "2026-11-18T16:31:00+00:00",
  classes: [],
  phases: [],
  pipeline: { uncertainty_threshold: 0.55, fusion: { w_pose: 0.5, rule: "" }, pose_features: new Array(19).fill("f") },
  training: { dataset: { instances: 40, images: 37, groups: 29, per_class: {} } },
  cv_metrics: null,
  model_comparison: null,
  temporal_benchmark: null,
  runtime_benchmark: null,
  dataset: null,
  demo_video_available: false,
};

function mockFetch(routes: Record<string, unknown>) {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const key = Object.keys(routes).find((k) => url.includes(k));
    if (!key) return new Response(JSON.stringify({ detail: "not found" }), { status: 404 });
    return new Response(JSON.stringify(routes[key]), { status: 200, headers: { "Content-Type": "application/json" } });
  });
}

function renderAt(path: string) {
  return render(
    <AnalysisProvider>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </AnalysisProvider>,
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("pages", () => {
  it("results page renders the real analysis: timeline, table and stats", async () => {
    vi.stubGlobal("fetch", mockFetch({ "/jobs/fixture-demo": fixture, "/model-info": modelInfo, "/health": { status: "ok", model_loaded: true, model_version: "1.0.0" } }));
    renderAt("/hasil/fixture-demo");
    await screen.findByRole("heading", { name: "Hasil analisis" }, { timeout: 20000 });
    const timeline = screen.getByRole("list", { name: "Linimasa fase shalat" });
    expect(timeline.querySelectorAll("button")).toHaveLength(fixture.result.segments.length);
    expect(screen.getByText("Siklus rakaat teramati")).toBeInTheDocument();
    expect(screen.getByRole("table", { name: /Daftar segmen/ })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Unduh laporan JSON" }).getAttribute("href")).toContain("/jobs/fixture-demo/report?format=json");
  });

  it("results page explains an expired job", async () => {
    vi.stubGlobal("fetch", mockFetch({ "/model-info": modelInfo, "/health": { status: "ok", model_loaded: true } }));
    renderAt("/hasil/unknown");
    expect(await screen.findByText(/kedaluwarsa/, undefined, { timeout: 20000 })).toBeInTheDocument();
  });

  it("home page shows the six phases and live model status", async () => {
    vi.stubGlobal("fetch", mockFetch({ "/model-info": modelInfo, "/health": { status: "ok", model_loaded: true, model_version: "1.0.0" } }));
    renderAt("/");
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Kenali urutan gerakan shalat");
    expect(screen.getByRole("list", { name: "Urutan gerakan dalam satu rakaat" }).querySelectorAll("li")).toHaveLength(6);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("Model 1.0.0 aktif"));
  });

  it("analysis page warns when the server is offline", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new TypeError("network");
    }));
    renderAt("/analisis");
    expect(await screen.findByText("Server analisis tidak terhubung")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analisis video" })).toBeDisabled();
  });
});

describe("api client", () => {
  it("turns error responses into ApiError with the server detail", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Model unavailable: missing" }), { status: 503 })));
    await expect(getHealth()).rejects.toEqual(expect.objectContaining({ status: 503, message: "Model unavailable: missing" }));
    await expect(getHealth()).rejects.toBeInstanceOf(ApiError);
  });
});
