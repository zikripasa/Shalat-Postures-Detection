import { formatDuration, formatPercent, formatTime } from "../lib/format";

describe("format helpers", () => {
  it("formats timestamps as m:ss.s", () => {
    expect(formatTime(0)).toBe("0:00.0");
    expect(formatTime(4.25)).toBe("0:04.3");
    expect(formatTime(75.3)).toBe("1:15.3");
    expect(formatTime(75.3, 0)).toBe("1:15");
  });
  it("formats durations in Indonesian", () => {
    expect(formatDuration(12.34)).toBe("12,3 dtk");
    expect(formatDuration(125)).toBe("2 mnt 5 dtk");
  });
  it("formats percentages with a comma decimal separator", () => {
    expect(formatPercent(0.553)).toBe("55%");
    expect(formatPercent(0.553, 1)).toBe("55,3%");
    expect(formatPercent(null)).toBe("–");
  });
});
