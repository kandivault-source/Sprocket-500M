"""Emit standalone SVG figures of the training run, for the README.

scripts/plot_run.py builds an interactive HTML report. GitHub will not run its
JavaScript and will not resolve its CSS custom properties, so this writes the
same charts as self-contained SVG files with literal colours instead.

    py scripts/plot_figures.py            # -> docs/figures/*.svg

Colours are chosen to stay legible against both the light and dark GitHub
backgrounds rather than switching between them, because an SVG referenced from
markdown is loaded as an image and cannot reliably see the page's theme.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_run import line_chart, main_segment, parse, sft_segment, thin  # noqa: E402

# Readable on #ffffff and on #0d1117 alike.
TRAIN = "#3b82f6"
VAL = "#f97316"
ACCENT = "#10b981"

# plot_run emits CSS custom properties for the chrome, which resolve to nothing
# in a standalone file. Map them to fixed mid-tones that survive either theme.
SUBS = {
    "var(--grid)": "#8b949e",
    "var(--text-muted)": "#8b949e",
    "var(--text-secondary)": "#8b949e",
    "var(--surface-2)": "none",
}


def standalone(svg: str) -> str:
    """Turn a plot_run chart fragment into a self-contained SVG document."""
    # The hover payload is dead weight without the report's JavaScript, and it
    # is by far the largest part of the file.
    svg = re.sub(r" data-series='[^']*'", "", svg)
    # Same for the crosshair and hover dots, which are invisible by design.
    svg = re.sub(r'<line class="crosshair".*?/>', "", svg, flags=re.S)
    svg = re.sub(r'<circle class="hdot".*?/>', "", svg, flags=re.S)
    for k, v in SUBS.items():
        svg = svg.replace(k, v)
    # Gridlines at full strength dominate a static image.
    svg = svg.replace('stroke="#8b949e" stroke-width="1"/>',
                      'stroke="#8b949e" stroke-width="1" opacity="0.28"/>')
    return svg.replace(
        "<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)


def write(path: str, svg: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(standalone(svg))
    print(f"  wrote {path}  ({os.path.getsize(path):,} bytes)")


def main() -> int:
    log = sys.argv[1] if len(sys.argv) > 1 else "logs/train.log"
    if not os.path.exists(log):
        sys.exit(f"{log} not found")

    iters, steps, shards = parse(log)
    run = main_segment(iters)
    if not run:
        sys.exit("no pretrain iterations found in the log")

    out = os.path.join("docs", "figures")
    os.makedirs(out, exist_ok=True)
    tokfmt = lambda v: f"{v:.1f}B"                                  # noqa: E731

    # The full range is dominated by the collapse out of random init, so the
    # headline figure is the part where training is actually legible.
    cut = run[-1]["tok"] * 0.05
    zoom = thin([r for r in run if r["tok"] >= cut])
    write(os.path.join(out, "pretrain-loss.svg"), line_chart(
        [dict(name="train", color=TRAIN, pts=[(r["tok"], r["train"]) for r in zoom]),
         dict(name="val", color=VAL, pts=[(r["tok"], r["val"]) for r in zoom])],
        h=300, xlabel="tokens seen", ylabel="cross-entropy loss", xfmt=tokfmt))

    plot = thin(run)
    write(os.path.join(out, "pretrain-loss-full.svg"), line_chart(
        [dict(name="train", color=TRAIN, pts=[(r["tok"], r["train"]) for r in plot]),
         dict(name="val", color=VAL, pts=[(r["tok"], r["val"]) for r in plot])],
        h=300, xlabel="tokens seen", ylabel="cross-entropy loss", xfmt=tokfmt))

    write(os.path.join(out, "throughput.svg"), line_chart(
        [dict(name="tok/s", color=TRAIN, pts=[(r["tok"], r["tps"]) for r in plot])],
        h=250, xlabel="tokens seen", ylabel="tokens / second",
        xfmt=tokfmt, yfmt=lambda v: f"{v/1000:.0f}k", y0=0))

    write(os.path.join(out, "lr-schedule.svg"), line_chart(
        [dict(name="lr", color=ACCENT, pts=[(r["tok"], r["lr"] * 1e4) for r in plot])],
        h=220, xlabel="tokens seen", ylabel="learning rate (x1e-4)",
        xfmt=tokfmt, yfmt=lambda v: f"{v:.1f}", y0=0))

    sf = sft_segment(steps) if steps else []
    if sf:
        write(os.path.join(out, "sft-loss.svg"), line_chart(
            [dict(name="train", color=TRAIN, pts=[(r["step"], r["train"]) for r in sf]),
             dict(name="val", color=VAL, pts=[(r["step"], r["val"]) for r in sf])],
            h=250, xlabel="step", ylabel="loss", xfmt=lambda v: f"{v:.0f}"))

    last = run[-1]
    print(f"\n  final: iter {last['it']:,}  train {last['train']:.3f}  "
          f"val {last['val']:.3f}  {last['tok']:.2f}B tok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
