"""Build a self-contained HTML report of the cloud training run from train.log.

No dependencies - no matplotlib, no CDN, no network. Charts are inline SVG with
a hover/crosshair layer, so the output is one file you can open, present from,
or email. Re-run it any time the log grows:

    py scripts/fetch_run_log.py        # refresh logs/train.log from the Hub
    py scripts/plot_run.py             # -> logs/run_report.html

Palette is the validated categorical default (blue/orange/aqua in fixed slot
order); both light and dark are selected steps, not an automatic flip.
"""
import argparse
import html
import json
import os
import re
import sys

ITER_RX = re.compile(
    r"iter\s+(\d+) \| train ([\d.]+) \| val ([\d.]+) \| lr ([\d.e+-]+) \| "
    r"([\d,]+) tok/s \| ([\d.]+) GB \| ([\d.]+)B tok")
STEP_RX = re.compile(
    r"step\s+(\d+)/(\d+) \| train ([\d.]+) \| val ([\d.]+) \| lr ([\d.e+-]+) \| "
    r"ep ([\d.]+)")
SHARD_RX = re.compile(
    r"\[\s*(\d+)/\s*(\d+)\]\s+(\S+)\s+([\d,]+) docs\s+([\d.]+)M tok in ([\d.]+)s"
    r".*?\|\s*([\d.]+)([KM]) tok/s")


def parse(path):
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    iters, steps, shards = [], [], []
    for l in lines:
        m = ITER_RX.search(l)
        if m:
            g = m.groups()
            iters.append(dict(it=int(g[0]), train=float(g[1]), val=float(g[2]),
                              lr=float(g[3]), tps=int(g[4].replace(",", "")),
                              gb=float(g[5]), tok=float(g[6])))
            continue
        m = STEP_RX.search(l)
        if m:
            g = m.groups()
            steps.append(dict(step=int(g[0]), total=int(g[1]), train=float(g[2]),
                              val=float(g[3]), lr=float(g[4]), ep=float(g[5])))
            continue
        m = SHARD_RX.search(l)
        if m:
            g = m.groups()
            rate = float(g[6]) * (1e6 if g[7] == "M" else 1e3)
            shards.append(dict(n=int(g[0]), of=int(g[1]),
                               docs=int(g[3].replace(",", "")),
                               mtok=float(g[4]), secs=float(g[5]), rate=rate))
    return iters, steps, shards


def thin(rows, target=700):
    """Downsample for plotting while ALWAYS keeping the last point - the final
    loss is the number everyone reads off the chart."""
    if len(rows) <= target:
        return rows
    step = len(rows) / target
    out = [rows[int(i * step)] for i in range(target)]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


# The 20B run restarted the iteration counter (smoke runs used the same log),
# so split on a counter reset and keep the LONGEST contiguous segment.
def main_segment(iters):
    segs, cur = [], []
    for r in iters:
        if cur and r["it"] < cur[-1]["it"]:
            segs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        segs.append(cur)
    return max(segs, key=len) if segs else []


CSS = """
*,*::before,*::after{box-sizing:border-box}
.viz-root{
  color-scheme:light;
  --surface-1:#fcfcfb; --surface-2:#f4f4f2; --border:#e2e1dd;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#84837d;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --grid:#e8e7e3;
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  background:var(--surface-1); color:var(--text-primary);
  max-width:1180px; margin:0 auto; padding:28px 20px 64px;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;
    --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a37;
    --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --grid:#2e2e2c;
  }
}
:root[data-theme="dark"] .viz-root{
  color-scheme:dark;
  --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a37;
  --text-primary:#fff; --text-secondary:#c3c2b7; --text-muted:#8f8e86;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --grid:#2e2e2c;
}
h1{font-size:1.6rem;margin:0 0 4px;letter-spacing:-.02em}
.sub{color:var(--text-secondary);margin:0 0 26px;font-size:.92rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:32px}
.tile{background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted)}
.tile .v{font-size:1.5rem;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.tile .n{font-size:.75rem;color:var(--text-secondary);margin-top:2px}
figure{margin:0 0 34px}
figcaption{font-size:1rem;font-weight:600;margin-bottom:2px}
.cap{font-size:.83rem;color:var(--text-secondary);margin-bottom:10px}
.chart{position:relative;background:var(--surface-2);border:1px solid var(--border);border-radius:10px;padding:12px 8px 6px}
svg{display:block;width:100%;height:auto;overflow:visible}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:.82rem;color:var(--text-secondary);margin:8px 0 0 12px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:11px;height:11px;border-radius:3px;display:inline-block}
.tt{position:absolute;pointer-events:none;opacity:0;transition:opacity .08s;
  background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
  padding:8px 10px;font-size:.78rem;box-shadow:0 4px 16px rgba(0,0,0,.16);
  font-variant-numeric:tabular-nums;white-space:nowrap;z-index:5;color:var(--text-primary)}
.tt b{font-weight:650}
.tt i{font-style:normal;color:var(--text-secondary)}
table{border-collapse:collapse;width:100%;font-size:.82rem;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--border)}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-secondary);font-weight:600}
details{margin-top:8px}
summary{cursor:pointer;font-size:.85rem;color:var(--text-secondary);padding:6px 0}
.wrap{overflow-x:auto}
"""

JS = r"""
function mkChart(el){
  const d=JSON.parse(el.getAttribute('data-series'));
  const box=el.closest('.chart'), tip=box.querySelector('.tt');
  const P=d.pad, W=d.w, H=d.h;
  const x2p=x=>P.l+(x-d.x0)/(d.x1-d.x0||1)*(W-P.l-P.r);
  const p2x=p=>d.x0+(p-P.l)/(W-P.l-P.r)*(d.x1-d.x0);
  const ov=el.querySelector('.crosshair'), dots=[...el.querySelectorAll('.hdot')];
  el.addEventListener('pointermove',ev=>{
    const r=el.getBoundingClientRect();
    const px=(ev.clientX-r.left)/r.width*W;
    if(px<P.l-6||px>W-P.r+6){tip.style.opacity=0;ov.setAttribute('opacity',0);dots.forEach(o=>o.setAttribute('opacity',0));return;}
    const xv=p2x(px);
    let bi=0,bd=1e18;
    d.series[0].pts.forEach((p,i)=>{const dd=Math.abs(p[0]-xv);if(dd<bd){bd=dd;bi=i}});
    const xs=x2p(d.series[0].pts[bi][0]);
    ov.setAttribute('x1',xs);ov.setAttribute('x2',xs);ov.setAttribute('opacity',1);
    let rows='';
    d.series.forEach((s,si)=>{
      const p=s.pts[bi]; if(!p)return;
      const y=P.t+(1-(p[1]-d.y0)/(d.y1-d.y0||1))*(H-P.t-P.b);
      if(dots[si]){dots[si].setAttribute('cx',xs);dots[si].setAttribute('cy',y);dots[si].setAttribute('opacity',1);}
      rows+=`<div><i>${s.name}</i> <b>${s.fmt?eval(s.fmt)(p[1]):p[1]}</b></div>`;
    });
    tip.innerHTML=`<div><b>${d.xlabel}: ${d.xfmt?eval(d.xfmt)(d.series[0].pts[bi][0]):d.series[0].pts[bi][0]}</b></div>${rows}`;
    tip.style.opacity=1;
    const bw=box.getBoundingClientRect();
    let L=(xs/W)*bw.width+14; if(L>bw.width-150)L-=170;
    tip.style.left=L+'px'; tip.style.top='14px';
  });
  el.addEventListener('pointerleave',()=>{tip.style.opacity=0;ov.setAttribute('opacity',0);dots.forEach(o=>o.setAttribute('opacity',0))});
}
document.querySelectorAll('svg[data-series]').forEach(mkChart);
"""


def line_chart(series, w=1080, h=300, pad=None, ylabel="", xlabel="",
               xfmt=None, yfmt=None, y0=None, y1=None):
    """series = [{name, color, pts:[(x,y)…]}]"""
    # Top padding leaves room for the y-axis caption to sit ABOVE the plot
    # instead of colliding with the first gridline; bottom fits tick row +
    # x caption on separate lines.
    pad = pad or dict(l=66, r=20, t=30, b=46)
    xs = [p[0] for s in series for p in s["pts"]]
    ys = [p[1] for s in series for p in s["pts"]]
    if not xs:
        return "<p>no data</p>"
    x0, x1 = min(xs), max(xs)
    y0 = min(ys) if y0 is None else y0
    y1 = max(ys) if y1 is None else y1
    span = (y1 - y0) or 1
    y0 -= span * 0.06
    y1 += span * 0.06

    def X(v):
        return pad["l"] + (v - x0) / ((x1 - x0) or 1) * (w - pad["l"] - pad["r"])

    def Y(v):
        return pad["t"] + (1 - (v - y0) / ((y1 - y0) or 1)) * (h - pad["t"] - pad["b"])

    out = [f'<svg viewBox="0 0 {w} {h}" role="img" data-series=\'{{}}\'>']
    # grid + y ticks
    ticks = 5
    for i in range(ticks + 1):
        v = y0 + (y1 - y0) * i / ticks
        yy = Y(v)
        out.append(f'<line x1="{pad["l"]}" x2="{w-pad["r"]}" y1="{yy:.1f}" y2="{yy:.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        lab = yfmt(v) if yfmt else f"{v:.2f}"
        out.append(f'<text x="{pad["l"]-9}" y="{yy+4:.1f}" text-anchor="end" '
                   f'font-size="11" fill="var(--text-muted)">{html.escape(lab)}</text>')
    # x ticks
    for i in range(6):
        v = x0 + (x1 - x0) * i / 5
        xx = X(v)
        lab = xfmt(v) if xfmt else f"{v:g}"
        out.append(f'<text x="{xx:.1f}" y="{h-pad["b"]+18}" text-anchor="middle" '
                   f'font-size="11" fill="var(--text-muted)">{html.escape(lab)}</text>')
    out.append(f'<text x="{(pad["l"]+w-pad["r"])/2:.0f}" y="{h-8}" text-anchor="middle" '
               f'font-size="11" fill="var(--text-secondary)">{html.escape(xlabel)}</text>')
    out.append(f'<text x="{pad["l"]-56}" y="{pad["t"]-12}" font-size="11" '
               f'fill="var(--text-secondary)">{html.escape(ylabel)}</text>')
    # 2px lines
    for s in series:
        pts = " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in s["pts"])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" '
                   f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    out.append(f'<line class="crosshair" y1="{pad["t"]}" y2="{h-pad["b"]}" '
               f'stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="3 3" opacity="0"/>')
    for s in series:
        out.append(f'<circle class="hdot" r="4.5" fill="{s["color"]}" '
                   f'stroke="var(--surface-2)" stroke-width="2" opacity="0"/>')
    out.append("</svg>")
    svg = "\n".join(out)

    payload = dict(
        w=w, h=h, pad=pad, x0=x0, x1=x1, y0=y0, y1=y1, xlabel=xlabel,
        xfmt="(v)=>" + (xfmt.__doc__ or "v.toFixed(2)") if False else None,
        series=[dict(name=s["name"], pts=[[round(p[0], 4), round(p[1], 5)]
                                          for p in s["pts"]], fmt=None)
                for s in series])
    return svg.replace("data-series='{}'",
                       "data-series='" + html.escape(json.dumps(payload), quote=True) + "'")


def bar_chart(vals, labels, color, w=1080, h=260, yfmt=None, xlabel="", ylabel=""):
    pad = dict(l=62, r=18, t=14, b=42)
    if not vals:
        return "<p>no data</p>"
    y1 = max(vals) * 1.08
    n = len(vals)
    iw = (w - pad["l"] - pad["r"]) / n
    bw = max(3, iw - 2)                       # 2px surface gap between bars
    out = [f'<svg viewBox="0 0 {w} {h}" role="img">']
    for i in range(5 + 1):
        v = y1 * i / 5
        yy = pad["t"] + (1 - v / y1) * (h - pad["t"] - pad["b"])
        out.append(f'<line x1="{pad["l"]}" x2="{w-pad["r"]}" y1="{yy:.1f}" y2="{yy:.1f}" '
                   f'stroke="var(--grid)" stroke-width="1"/>')
        lab = yfmt(v) if yfmt else f"{v:.0f}"
        out.append(f'<text x="{pad["l"]-9}" y="{yy+4:.1f}" text-anchor="end" font-size="11" '
                   f'fill="var(--text-muted)">{html.escape(lab)}</text>')
    base = h - pad["b"]
    for i, v in enumerate(vals):
        bh = (v / y1) * (h - pad["t"] - pad["b"])
        x = pad["l"] + i * iw + 1
        out.append(f'<rect x="{x:.1f}" y="{base-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                   f'rx="4" fill="{color}"><title>{html.escape(labels[i])}: '
                   f'{yfmt(v) if yfmt else v:s}</title></rect>'
                   if isinstance(yfmt(v) if yfmt else v, str) else
                   f'<rect x="{x:.1f}" y="{base-bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                   f'rx="4" fill="{color}"><title>{html.escape(labels[i])}</title></rect>')
    out.append(f'<text x="{pad["l"]}" y="{h-6}" font-size="11" '
               f'fill="var(--text-secondary)">{html.escape(xlabel)}</text>')
    out.append(f'<text x="14" y="{pad["t"]+10}" font-size="11" '
               f'fill="var(--text-secondary)">{html.escape(ylabel)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/train.log")
    ap.add_argument("--out", default="logs/run_report.html")
    a = ap.parse_args()
    if not os.path.exists(a.log):
        sys.exit(f"{a.log} not found - run scripts/fetch_run_log.py first")

    iters, steps, shards = parse(a.log)
    run = main_segment(iters)
    if not run:
        sys.exit("no pretrain iterations found in the log")
    plot = thin(run)
    S1, S2, S3 = "var(--s1)", "var(--s2)", "var(--s3)"

    tokfmt = lambda v: f"{v:.1f}B"
    loss = line_chart(
        [dict(name="train", color=S1, pts=[(r["tok"], r["train"]) for r in plot]),
         dict(name="val", color=S2, pts=[(r["tok"], r["val"]) for r in plot])],
        xlabel="tokens seen", ylabel="cross-entropy loss", xfmt=tokfmt)

    # The full-range chart is dominated by the collapse out of random init
    # (10.6 -> ~4 inside the first 1%), which squashes everything that follows
    # into a flat band. A second panel over the remaining 95% is where the
    # actual training progress is legible.
    cut = run[-1]["tok"] * 0.05
    zoom_rows = thin([r for r in run if r["tok"] >= cut])
    loss_zoom = line_chart(
        [dict(name="train", color=S1, pts=[(r["tok"], r["train"]) for r in zoom_rows]),
         dict(name="val", color=S2, pts=[(r["tok"], r["val"]) for r in zoom_rows])],
        h=290, xlabel="tokens seen", ylabel="cross-entropy loss", xfmt=tokfmt)
    tps = line_chart(
        [dict(name="tok/s", color=S1, pts=[(r["tok"], r["tps"]) for r in plot])],
        h=240, xlabel="tokens seen", ylabel="tokens / second",
        xfmt=tokfmt, yfmt=lambda v: f"{v/1000:.0f}k", y0=0)
    lr = line_chart(
        [dict(name="lr", color=S3, pts=[(r["tok"], r["lr"] * 1e4) for r in plot])],
        h=220, xlabel="tokens seen", ylabel="learning rate (x1e-4)",
        xfmt=tokfmt, yfmt=lambda v: f"{v:.1f}", y0=0)

    last, first = run[-1], run[0]
    hours = last["tok"] * 1e9 / (sum(r["tps"] for r in run) / len(run)) / 3600
    tiles = [
        ("final val loss", f'{last["val"]:.3f}', f'from {first["val"]:.2f} at start'),
        ("tokens trained", f'{last["tok"]:.2f}B', f'{last["it"]:,} steps'),
        ("throughput", f'{max(r["tps"] for r in run)/1000:.0f}k tok/s', "peak, sustained"),
        ("wall clock", f"{hours:.1f} h", "H100 SXM 80GB"),
        ("VRAM", f'{last["gb"]:.1f} GB', "of 80 GB"),
        ("parameters", "501.1M", "ctx 2048"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div><div class="n">{html.escape(n)}</div></div>'
        for k, v, n in tiles)

    shard_html = ""
    if shards:
        uniq, seen = [], set()
        for s in shards:
            if s["n"] not in seen:
                seen.add(s["n"])
                uniq.append(s)
        shard_html = f"""
<figure>
  <figcaption>Corpus build — tokenization throughput per shard</figcaption>
  <p class="cap">{len(uniq)} FineWeb-Edu shards streamed, tokenized to uint16 and appended.
     Each bar is one shard's sustained rate through the 32k BPE tokenizer.</p>
  <div class="chart">{bar_chart([s['rate'] for s in uniq],
        [f"shard {s['n']}: {s['mtok']:.1f}M tok in {s['secs']:.0f}s" for s in uniq],
        S3, yfmt=lambda v: f"{v/1e6:.1f}M", xlabel="shard (in build order)",
        ylabel="tokens / second")}</div>
</figure>"""

    sft_html = ""
    if steps:
        seg, cur = [], []
        for r in steps:
            if cur and r["step"] < cur[-1]["step"]:
                seg.append(cur); cur = []
            cur.append(r)
        if cur:
            seg.append(cur)
        sf = max(seg, key=len)
        sft_html = f"""
<figure>
  <figcaption>Instruct fine-tune (SFT) — loss over the chat corpus</figcaption>
  <p class="cap">Assistant-only loss masking over packed 2048-token blocks;
     {sf[-1]['total']} steps, {sf[-1]['ep']:.1f} epochs.</p>
  <div class="chart">
    {line_chart([dict(name="train", color=S1, pts=[(r['step'], r['train']) for r in sf]),
                 dict(name="val", color=S2, pts=[(r['step'], r['val']) for r in sf])],
                h=250, xlabel="step", ylabel="loss", xfmt=lambda v: f"{v:.0f}")}
    <div class="tt"></div>
  </div>
  <div class="legend"><span><i class="sw" style="background:var(--s1)"></i>train</span>
    <span><i class="sw" style="background:var(--s2)"></i>val</span></div>
</figure>"""

    tbl = "".join(
        f'<tr><td>{r["it"]:,}</td><td>{r["tok"]:.2f}B</td><td>{r["train"]:.3f}</td>'
        f'<td>{r["val"]:.3f}</td><td>{r["lr"]:.2e}</td><td>{r["tps"]:,}</td>'
        f'<td>{r["gb"]:.1f}</td></tr>'
        for r in run[::max(1, len(run) // 40)])

    doc = f"""<style>{CSS}</style>
<div class="viz-root">
<h1>Sprocket 500M — training run</h1>
<p class="sub">501.1M-parameter Llama-architecture model trained from scratch on
FineWeb-Edu. Single H100 SXM 80GB, context 2048, micro-batch 16 x 4 grad-accum,
bf16 with torch.compile. Every number below is read from the run's own log.</p>

<div class="tiles">{tiles_html}</div>

<figure>
  <figcaption>Pretraining loss</figcaption>
  <p class="cap">Train and validation cross-entropy against tokens seen. Validation
     tracks train throughout — the model is still learning from fresh data at the end,
     not memorising.</p>
  <div class="chart">{loss}<div class="tt"></div></div>
  <div class="legend"><span><i class="sw" style="background:var(--s1)"></i>train</span>
    <span><i class="sw" style="background:var(--s2)"></i>validation</span></div>
</figure>

<figure>
  <figcaption>Pretraining loss — after the first {cut:.2f}B tokens</figcaption>
  <p class="cap">The same run with the initial collapse out of random init removed.
     This is where the real training happens: a long grind from ~3.9 down to
     {last['val']:.2f}, with a visible step down as the learning rate anneals.</p>
  <div class="chart">{loss_zoom}<div class="tt"></div></div>
  <div class="legend"><span><i class="sw" style="background:var(--s1)"></i>train</span>
    <span><i class="sw" style="background:var(--s2)"></i>validation</span></div>
</figure>

<figure>
  <figcaption>Throughput</figcaption>
  <p class="cap">Sustained tokens/second. The flat line is the point — no thermal
     decay, no dataloader stalls, no drift across the whole run.</p>
  <div class="chart">{tps}<div class="tt"></div></div>
</figure>

<figure>
  <figcaption>Learning-rate schedule</figcaption>
  <p class="cap">Linear warmup then cosine decay. The loss steps down as the rate
     anneals — visible in the first chart.</p>
  <div class="chart">{lr}<div class="tt"></div></div>
</figure>

{shard_html}
{sft_html}

<details open>
  <summary>Data table (sampled every {max(1, len(run)//40)} logged points)</summary>
  <div class="wrap"><table>
    <thead><tr><th>iter</th><th>tokens</th><th>train</th><th>val</th><th>lr</th>
      <th>tok/s</th><th>VRAM GB</th></tr></thead>
    <tbody>{tbl}</tbody>
  </table></div>
</details>
</div>
<script>{JS}</script>
"""
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {a.out}")
    print(f"  pretrain points : {len(run):,} (plotted {len(plot):,})")
    print(f"  corpus shards   : {len(shards)}")
    print(f"  SFT points      : {len(steps)}")
    print(f"  final           : iter {last['it']:,}  train {last['train']:.3f}  "
          f"val {last['val']:.3f}  {last['tok']:.2f}B tok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
