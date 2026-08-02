#!/usr/bin/env python3
"""Renders the measured numbers as SVG, one light file and one dark file each.

Every performance figure in this repository was a table of ASCII, which is fine
for someone reading carefully and useless for someone deciding in forty seconds
whether to keep reading.

Written by hand rather than with matplotlib, for three reasons that all point
the same way: matplotlib is a 50 MB dependency to draw four bar charts; its SVG
output bakes in a white background, which looks broken in GitHub's dark theme;
and the specs below (24px bar cap, 4px rounded data-end, 2px surface gaps,
hairline grid) are easier to hit directly than to talk a plotting library into.
The whole renderer is about 150 lines and needs nothing but the standard
library.

Dark mode is a *selected* second palette, not an inverted first one -- the same
three hues stepped for the dark surface. README.md picks between them with a
<picture> element, which is how GitHub does theme-aware images.

The palette is the reference categorical order (blue, orange, aqua), validated
with the data-viz validator in both modes: all-pairs CVD deltaE 9.2 light /
9.4 dark against a floor of 8, normal-vision 24.0 / 20.9 against a floor of 15.
Aqua sits at 2.74:1 on the light surface, below the 3:1 line, which obliges
visible labels -- every bar carries its value, and docs/PERFORMANCE.md carries
the table.

Usage:
    python tools/plot_benchmarks.py          # writes docs/img/*.svg
"""

import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "img")

# --- the two selected palettes ------------------------------------------------

LIGHT = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": ["#2a78d6", "#eb6834", "#1baf7a"],
}
DARK = {
    "surface": "#1a1a19",
    "primary": "#ffffff",
    "secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": ["#3987e5", "#d95926", "#199e70"],
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
BAR = 22       # <= 24px, so the band keeps some air
GAP = 2        # the surface gap, one width everywhere
RADIUS = 4     # rounded data-end, square at the baseline


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def hbar(x, y, w, h, fill, r=RADIUS):
    """A horizontal bar: square at the baseline, rounded at the data end."""
    if w <= r:
        return f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.5):.1f}" height="{h}" fill="{fill}"/>'
    return (f'<path d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} '
            f'v{h - 2 * r} a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.1f} z" fill="{fill}"/>')


def text(x, y, s, fill, size=12, anchor="start", weight="400", tabular=False):
    extra = ' style="font-variant-numeric:tabular-nums"' if tabular else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{extra}>{esc(s)}</text>')


def legend(x, y, names, p):
    """A swatch beside text -- identity comes from the mark, never coloured text."""
    out, cx = [], x
    for i, name in enumerate(names):
        out.append(f'<rect x="{cx}" y="{y - 8}" width="10" height="10" rx="2" '
                   f'fill="{p["series"][i]}"/>')
        out.append(text(cx + 15, y, name, p["secondary"], 12))
        cx += 15 + len(name) * 6.6 + 22
    return "".join(out)


# One vertical rhythm for all four charts, because the first version drew the
# legend at y=52 straight through the subtitle's baseline at y=45.
TITLE_Y, SUBTITLE_Y, LEGEND_Y, PLOT_TOP = 26, 46, 70, 88


def frame(width, height, body, p, title, subtitle=None):
    head = [f'<rect width="{width}" height="{height}" fill="{p["surface"]}"/>',
            text(20, TITLE_Y, title, p["primary"], 15, weight="600")]
    if subtitle:
        head.append(text(20, SUBTITLE_Y, subtitle, p["muted"], 12))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">'
            + "".join(head) + body + "</svg>")


def gridlines(x0, plot_w, y0, y1, vmax, ticks, p, unit=""):
    out = []
    for t in ticks:
        gx = x0 + plot_w * t / vmax
        out.append(f'<line x1="{gx:.1f}" y1="{y0}" x2="{gx:.1f}" y2="{y1}" '
                   f'stroke="{p["grid"]}" stroke-width="1"/>')
        out.append(text(gx, y1 + 16, f"{t:g}{unit}", p["muted"], 11, "middle", tabular=True))
    out.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" '
               f'stroke="{p["axis"]}" stroke-width="1"/>')
    return "".join(out)


# --- the four charts ----------------------------------------------------------
#
# Every number below was measured on an RTX 3060 Ti and is recorded, with how it
# was taken, in docs/PERFORMANCE.md. They are repeated here rather than parsed
# from the benchmarks because plotting must not need a GPU; the commit that adds
# a run-and-verify job is what will make them self-checking.

def chart_vs_pytorch(p):
    """Grouped bars: two implementations, two devices. The headline comparison."""
    rows = [("CPU  (12 cores / 6 threads)", 24.1, 5.30),
            ("CUDA (RTX 3060 Ti)", 4.70, 2.10)]
    w, x0, plot_w = 760, 210, 460
    y = PLOT_TOP + 14
    body = [gridlines(x0, plot_w, PLOT_TOP, y + len(rows) * 74 - 10, 25, [0, 5, 10, 15, 20, 25], p, " s")]
    for label, engine, torch in rows:
        body.append(text(20, y + 16, label, p["secondary"], 12))
        for i, v in enumerate((engine, torch)):
            by = y + i * (BAR + GAP)
            bw = plot_w * v / 25
            body.append(hbar(x0, by, bw, BAR, p["series"][i]))
            body.append(text(x0 + bw + 8, by + 15, f"{v:.2f} s", p["secondary"], 12,
                             weight="600" if i == 0 else "400", tabular=True))
        y += 74
    body.append(legend(x0, LEGEND_Y, ["this engine", "PyTorch 2.11 + cuDNN"], p))
    return frame(w, y + 20, "".join(body), p,
                 "MNIST, same model and data, same machine",
                 "2 000-image subset, 12 epochs, 206 922 parameters, fp32 both sides. Lower is better.")


def chart_step(p):
    """Stacked bars: where a training step goes, and what each fix removed."""
    rows = [("CPU", 23.82, 43.41, 2.64),
            ("CUDA, before", 8.32, 29.10, 2.74),
            ("+ three kernel fixes", 8.94, 20.31, 2.95),
            ("+ lazy host mirror", 4.54, 3.64, 1.81)]
    w, x0, plot_w = 760, 210, 440
    vmax = 72
    y = PLOT_TOP + 12
    body = [gridlines(x0, plot_w, PLOT_TOP, y + len(rows) * 42 - 8, vmax, [0, 20, 40, 60], p, " ms")]
    for label, fwd, bwd, other in rows:
        total = fwd + bwd + other
        cx = x0
        for i, v in enumerate((fwd, bwd, other)):
            seg = plot_w * v / vmax
            last = i == 2
            body.append(hbar(cx, y, max(seg - GAP, 1), BAR, p["series"][i],
                             RADIUS if last else 0))
            cx += seg
        body.append(text(20, y + 15, label, p["secondary"], 12))
        body.append(text(x0 + plot_w * total / vmax + 8, y + 15, f"{total:.1f} ms",
                         p["primary"], 12, weight="600", tabular=True))
        y += 42
    body.append(legend(x0, LEGEND_Y, ["forward", "backward", "everything else"], p))
    return frame(w, y + 24, "".join(body), p,
                 "One training step, batch 64",
                 "The transfers were never the bottleneck: 95% of the step was the host.")


def chart_matmul(p):
    """One measure, six entities. Ours in slot 1, the reference in slot 2."""
    rows = [("naive", 898, 0), ("tiled", 1178, 0), ("tensorcore (tf32)", 5200, 0),
            ("register", 6871, 0), ("vectorized", 7660, 0), ("cuBLAS", 9258, 1)]
    peak = 16489
    w, x0, plot_w = 760, 210, 430
    y = PLOT_TOP + 14
    body = [gridlines(x0, plot_w, PLOT_TOP, y + len(rows) * 34 - 6, peak,
                      [0, 4000, 8000, 12000, 16000], p)]
    px = x0 + plot_w
    body.append(f'<line x1="{px}" y1="{PLOT_TOP}" x2="{px}" y2="{y + len(rows) * 34 - 6}" '
                f'stroke="{p["axis"]}" stroke-width="1"/>')
    body.append(text(px, PLOT_TOP - 6, "fp32 peak 16 489", p["muted"], 11, "end", tabular=True))
    for label, v, slot in rows:
        bw = plot_w * v / peak
        body.append(hbar(x0, y, bw, BAR, p["series"][slot]))
        body.append(text(20, y + 15, label, p["secondary"], 12))
        body.append(text(x0 + bw + 8, y + 15, f"{v:,} ({v / peak * 100:.0f}%)".replace(",", " "),
                         p["secondary"], 12, tabular=True))
        y += 34
    body.append(legend(x0, LEGEND_Y, ["hand-written here", "cuBLAS, reference only"], p))
    return frame(w, y + 22, "".join(body), p,
                 "matmul 4096³, GFLOP/s against the card's fp32 peak",
                 "One process per kernel, operands already resident. Higher is better.")


def chart_kernel_fixes(p):
    """Before and after, per kernel. The bars for 'after' are meant to vanish."""
    rows = [("sum_over_axis", 534.3, 5.9, "90×"),
            ("matmul_tiled", 487.9, 62.3, "7.9×"),
            ("permute_gather", 155.7, 9.8, "15.7×")]
    w, x0, plot_w = 760, 210, 400
    vmax = 560
    y = PLOT_TOP + 14
    body = [gridlines(x0, plot_w, PLOT_TOP, y + len(rows) * 74 - 10, vmax,
                      [0, 200, 400], p, " ms")]
    for label, before, after, factor in rows:
        for i, v in enumerate((before, after)):
            by = y + i * (BAR + GAP)
            bw = plot_w * v / vmax
            body.append(hbar(x0, by, bw, BAR, p["series"][i]))
            body.append(text(x0 + bw + 8, by + 15, f"{v:.1f} ms", p["secondary"], 12,
                             tabular=True))
        body.append(text(20, y + 15, label, p["secondary"], 12))
        body.append(text(20, y + 32, factor + " faster", p["muted"], 11))
        y += 74
    body.append(legend(x0, LEGEND_Y, ["before", "after"], p))
    return frame(w, y + 20, "".join(body), p,
                 "Three kernels, one mistake, 100 MNIST steps",
                 "Each drew its parallelism from the size of the output, not the amount of work.")


CHARTS = {
    "mnist-vs-pytorch": chart_vs_pytorch,
    "step-breakdown": chart_step,
    "matmul-kernels": chart_matmul,
    "kernel-fixes": chart_kernel_fixes,
}

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in CHARTS.items():
        for suffix, palette in (("", LIGHT), ("-dark", DARK)):
            path = os.path.join(OUT, f"{name}{suffix}.svg")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(fn(palette))
            print(f"  {os.path.relpath(path, REPO)}")
    print(f"\n{len(CHARTS) * 2} files written to docs/img/.")
