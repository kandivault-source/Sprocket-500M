/**
 * A small line chart with a hover readout.
 *
 * The README ships baked SVG files, which cannot be interrogated. On this page
 * the point is to let someone drag along the curve and read the actual numbers
 * at any moment of the run, so the charts are drawn here from the same data the
 * training log produced.
 *
 * Deliberately not a charting library: one chart type, no dependencies, and
 * nothing to keep up to date.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.MiniChart = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";

  function el(name, attrs) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) if (attrs.hasOwnProperty(k)) n.setAttribute(k, attrs[k]);
    return n;
  }

  /**
   * opts: { series:[{name,color,x:[],y:[]}], xlabel, ylabel, xfmt, yfmt,
   *         y0, y1, height, logy }
   */
  function draw(host, opts) {
    host.textContent = "";

    var series = opts.series.filter(function (s) { return s.x.length; });
    if (!series.length) return;

    var W = 720, H = opts.height || 260;
    var pad = { l: 58, r: 14, t: 16, b: 42 };

    var xs = [], ys = [];
    series.forEach(function (s) {
      xs = xs.concat(s.x[0], s.x[s.x.length - 1]);
      ys = ys.concat(s.y);
    });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = opts.y0 !== undefined ? opts.y0 : Math.min.apply(null, ys);
    var y1 = opts.y1 !== undefined ? opts.y1 : Math.max.apply(null, ys);
    var span = (y1 - y0) || 1;
    if (opts.y0 === undefined) y0 -= span * 0.08;
    if (opts.y1 === undefined) y1 += span * 0.08;

    var X = function (v) { return pad.l + (v - x0) / ((x1 - x0) || 1) * (W - pad.l - pad.r); };
    var Y = function (v) { return pad.t + (1 - (v - y0) / ((y1 - y0) || 1)) * (H - pad.t - pad.b); };

    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": opts.ylabel + " against " + opts.xlabel
    });
    svg.style.width = "100%";
    svg.style.height = "auto";
    svg.style.display = "block";
    svg.style.touchAction = "pan-y";

    var i, v;
    for (i = 0; i <= 4; i++) {
      v = y0 + (y1 - y0) * i / 4;
      svg.appendChild(el("line", {
        x1: pad.l, x2: W - pad.r, y1: Y(v).toFixed(1), y2: Y(v).toFixed(1),
        stroke: "currentColor", "stroke-width": 1, opacity: 0.14
      }));
      var yt = el("text", {
        x: pad.l - 8, y: (Y(v) + 4).toFixed(1), "text-anchor": "end",
        "font-size": 11, fill: "currentColor", opacity: 0.55
      });
      yt.textContent = opts.yfmt ? opts.yfmt(v) : v.toFixed(2);
      svg.appendChild(yt);
    }

    for (i = 0; i <= 4; i++) {
      v = x0 + (x1 - x0) * i / 4;
      var xt = el("text", {
        x: X(v).toFixed(1), y: H - pad.b + 18, "text-anchor": "middle",
        "font-size": 11, fill: "currentColor", opacity: 0.55
      });
      xt.textContent = opts.xfmt ? opts.xfmt(v) : String(Math.round(v));
      svg.appendChild(xt);
    }

    var xl = el("text", {
      x: ((pad.l + W - pad.r) / 2).toFixed(0), y: H - 6, "text-anchor": "middle",
      "font-size": 11, fill: "currentColor", opacity: 0.6
    });
    xl.textContent = opts.xlabel || "";
    svg.appendChild(xl);

    var yl = el("text", {
      x: 4, y: pad.t - 4, "font-size": 11, fill: "currentColor", opacity: 0.6
    });
    yl.textContent = opts.ylabel || "";
    svg.appendChild(yl);

    series.forEach(function (s) {
      var d = "";
      for (var j = 0; j < s.y.length; j++) {
        d += (j ? "L" : "M") + X(s.x[j]).toFixed(1) + "," + Y(s.y[j]).toFixed(1);
      }
      svg.appendChild(el("path", {
        d: d, fill: "none", stroke: s.color, "stroke-width": 1.9,
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
    });

    var cross = el("line", {
      y1: pad.t, y2: H - pad.b, stroke: "currentColor", "stroke-width": 1,
      "stroke-dasharray": "3 3", opacity: 0
    });
    svg.appendChild(cross);

    var dots = series.map(function (s) {
      var c = el("circle", { r: 4, fill: s.color, stroke: "var(--panel)", "stroke-width": 2, opacity: 0 });
      svg.appendChild(c);
      return c;
    });

    host.appendChild(svg);

    var tip = document.createElement("div");
    tip.className = "chart-tip";
    host.appendChild(tip);

    function nearest(s, xv) {
      var lo = 0, hi = s.x.length - 1;
      while (lo < hi) {
        var mid = (lo + hi) >> 1;
        if (s.x[mid] < xv) lo = mid + 1; else hi = mid;
      }
      if (lo > 0 && Math.abs(s.x[lo - 1] - xv) < Math.abs(s.x[lo] - xv)) lo--;
      return lo;
    }

    function move(ev) {
      var r = svg.getBoundingClientRect();
      var px = (ev.clientX - r.left) / r.width * W;
      if (px < pad.l - 8 || px > W - pad.r + 8) { leave(); return; }
      var xv = x0 + (px - pad.l) / (W - pad.l - pad.r) * (x1 - x0);

      var rows = "";
      series.forEach(function (s, si) {
        var k = nearest(s, xv);
        dots[si].setAttribute("cx", X(s.x[k]).toFixed(1));
        dots[si].setAttribute("cy", Y(s.y[k]).toFixed(1));
        dots[si].setAttribute("opacity", 1);
        rows += '<div><span style="color:' + s.color + '">●</span> ' +
          s.name + " <b>" + (opts.yfmt ? opts.yfmt(s.y[k]) : s.y[k].toFixed(3)) + "</b></div>";
      });
      var k0 = nearest(series[0], xv);
      cross.setAttribute("x1", X(series[0].x[k0]).toFixed(1));
      cross.setAttribute("x2", X(series[0].x[k0]).toFixed(1));
      cross.setAttribute("opacity", 0.5);

      tip.innerHTML = "<div class='k'>" +
        (opts.xfmt ? opts.xfmt(series[0].x[k0]) : series[0].x[k0]) + "</div>" + rows;
      tip.style.opacity = 1;
      var frac = X(series[0].x[k0]) / W;
      tip.style.left = (frac > 0.6 ? frac * 100 - 3 : frac * 100 + 3) + "%";
      tip.style.transform = frac > 0.6 ? "translateX(-100%)" : "none";
    }

    function leave() {
      tip.style.opacity = 0;
      cross.setAttribute("opacity", 0);
      dots.forEach(function (d) { d.setAttribute("opacity", 0); });
    }

    svg.addEventListener("pointermove", move);
    svg.addEventListener("pointerleave", leave);
  }

  return { draw: draw };
});
