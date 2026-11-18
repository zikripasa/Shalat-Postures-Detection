"""Shared visual system for every chart in the project (figures, report, deck, web app).

Colours are taken from a validated categorical palette (OKLab CVD separation >= 8.4 and
normal-vision separation >= 19 on every pair that can touch, in light and dark mode).
The same hex values are mirrored in ``frontend/src/lib/palette.ts``.

Rules applied by the helpers below: thin bars (<= 24 px) with a rounded data-end and a
square baseline, 2 px surface gaps between touching marks, solid hairline grids, text in
ink tokens (never in the series colour), and no dual axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

# ---- surfaces & ink ------------------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#d9d7cf"  # "uncertain" / not-a-category fill

# ---- brand (UI chrome only, never data) -----------------------------------------------
BRAND_NAVY = "#0f1f3d"
BRAND_TEAL = "#0e7c74"
BRAND_GOLD = "#b8892c"
BRAND_IVORY = "#f6f2e8"

# ---- categorical data colours -------------------------------------------------------
PHASE_COLORS = {
    "berdiri": "#eda100",
    "rukuk": "#1baf7a",
    "iktidal": "#eb6834",
    "sujud": "#2a78d6",
    "duduk_antara_sujud": "#008300",
    "tasyahud": "#e87ba4",
}
PHASE_COLORS_DARK = {
    "berdiri": "#c98500",
    "rukuk": "#199e70",
    "iktidal": "#d95926",
    "sujud": "#3987e5",
    "duduk_antara_sujud": "#0ca30c",
    "tasyahud": "#d55181",
}
#: A visual class wears the colour of the phase it looks like.
VISUAL_COLORS = {"qiyam": "#eda100", "ruku": "#1baf7a", "sujud": "#2a78d6", "julus": "#008300"}

#: Generic series slots (model comparisons etc.), fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

#: Sequential blue ramp (light -> dark) for magnitudes such as confusion matrices.
BLUE_RAMP = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("salat_blue", BLUE_RAMP)


_FONTS_REGISTERED = False


def _register_brand_fonts() -> None:
    """Register Plus Jakarta Sans (OFL, shipped in portfolio/assets/fonts) for charts."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    from matplotlib import font_manager

    from salat_vision.config import find_project_root

    for ttf in (find_project_root() / "portfolio" / "assets" / "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    _FONTS_REGISTERED = True


def apply_style() -> None:
    """Matplotlib rcParams for the project's chart look (brand UI sans, Inter as fallback)."""
    _register_brand_fonts()
    plt.rcParams.update({
        "font.family": ["Plus Jakarta Sans", "Inter", "DejaVu Sans"],
        "font.size": 10,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def _px_to_data(ax, px: float, axis: str) -> float:
    """Convert a length in display pixels to data units along ``axis``."""
    fig = ax.figure
    fig.canvas.draw_idle()
    bbox = ax.get_window_extent()
    if axis == "x":
        lo, hi = ax.get_xlim()
        return px * (hi - lo) / max(bbox.width, 1)
    lo, hi = ax.get_ylim()
    return px * (hi - lo) / max(bbox.height, 1)


_K = 0.5522847498  # cubic Bezier constant for a quarter circle


def _bar_path(base: float, end: float, lo: float, hi: float, r_len: float, r_th: float, horizontal: bool) -> MplPath:
    """Closed path for one bar: square at ``base``, quarter-rounded corners at ``end``.

    ``base``/``end`` run along the value axis, ``lo``/``hi`` across the bar. Radii are
    given separately per axis (data units) so corners look circular on screen.
    """
    s = 1.0 if end >= base else -1.0
    e_in = end - s * r_len
    verts = [
        (base, lo), (e_in, lo),
        (e_in + s * _K * r_len, lo), (end, lo + r_th - _K * r_th), (end, lo + r_th),
        (end, hi - r_th),
        (end, hi - r_th + _K * r_th), (e_in + s * _K * r_len, hi), (e_in, hi),
        (base, hi), (base, lo),
    ]
    codes = [MplPath.MOVETO, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CLOSEPOLY]
    if not horizontal:
        verts = [(y, x) for x, y in verts]
    return MplPath(verts, codes)


def rounded_bars(ax, positions, values, *, color, width: float = 0.6, horizontal: bool = False,
                 radius_px: float = 4.0, max_thickness_px: float = 24.0, base: float = 0.0, zorder: int = 3) -> None:
    """Bars with a rounded data-end and a square baseline (set axis limits *before* calling)."""
    colors = color if isinstance(color, (list, tuple)) else [color] * len(values)
    len_axis, th_axis = ("x", "y") if horizontal else ("y", "x")
    for pos, val, c in zip(positions, values, colors):
        if val is None or val == base:
            continue
        thick = min(width, _px_to_data(ax, max_thickness_px, th_axis))
        r_len = min(_px_to_data(ax, radius_px, len_axis), abs(val - base) / 2)
        r_th = min(_px_to_data(ax, radius_px, th_axis), thick / 2)
        path = _bar_path(base, val, pos - thick / 2, pos + thick / 2, r_len, r_th, horizontal)
        ax.add_patch(PathPatch(path, facecolor=c, linewidth=0, zorder=zorder))


def hairline_grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)


def save(fig, path: str | Path, *, also_svg: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if also_svg:
        fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path
