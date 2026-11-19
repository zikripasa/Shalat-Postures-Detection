import { fireEvent, render, screen } from "@testing-library/react";
import { PhaseTimeline } from "../components/PhaseTimeline";
import type { Phase, Segment } from "../lib/types";
import fixture from "./fixtures/job_demo.json";

const result = fixture.result as unknown as { segments: Segment[]; video: { duration_seconds: number }; phase_labels: Record<Phase, string> };

describe("PhaseTimeline", () => {
  it("renders one button per segment with an accessible description", () => {
    render(<PhaseTimeline segments={result.segments} duration={result.video.duration_seconds} phaseLabels={result.phase_labels} />);
    const list = screen.getByRole("list", { name: "Linimasa fase shalat" });
    const buttons = list.querySelectorAll("button");
    expect(buttons).toHaveLength(result.segments.length);
    expect(buttons[0].getAttribute("aria-label")).toMatch(/^Berdiri \(Qiyam\), 0:00\.0 sampai/);
  });

  it("sizes blocks proportionally to segment duration", () => {
    render(<PhaseTimeline segments={result.segments} duration={result.video.duration_seconds} phaseLabels={result.phase_labels} />);
    const items = screen.getByRole("list", { name: "Linimasa fase shalat" }).querySelectorAll("li");
    const total = result.video.duration_seconds;
    const s0 = result.segments[0];
    expect(parseFloat((items[0] as HTMLElement).style.width)).toBeCloseTo(((s0.end - s0.start) / total) * 100, 3);
  });

  it("seeks to the segment start when a block is activated", () => {
    const onSeek = vi.fn();
    render(<PhaseTimeline segments={result.segments} duration={result.video.duration_seconds} phaseLabels={result.phase_labels} onSeek={onSeek} />);
    const buttons = screen.getByRole("list", { name: "Linimasa fase shalat" }).querySelectorAll("button");
    fireEvent.click(buttons[2]);
    expect(onSeek).toHaveBeenCalledWith(result.segments[2].start, result.segments[2]);
  });

  it("marks uncertain segments in their accessible name", () => {
    render(<PhaseTimeline segments={result.segments} duration={result.video.duration_seconds} phaseLabels={result.phase_labels} />);
    const uncertain = result.segments.filter((s) => s.uncertain);
    expect(uncertain.length).toBeGreaterThan(0);
    expect(screen.getAllByRole("button", { name: /tidak pasti/ })).toHaveLength(uncertain.length);
  });
});
