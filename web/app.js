"use strict";

/* ------------------------------------------------------------------ helpers */

const $ = (sel, root = document) => root.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const ALGO_VAR = { expected: "--series-1", keep: "--series-2", brute: "--series-3",
                   "greedy-density": "--series-4", "expected-uniform": "--series-5" };
// drone colours for peak groups (categorical order); scavengers use the muted ink
const GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                      "#6250d6", "#e34948"];
const algoColor = (key) => css(ALGO_VAR[key] ||
  (key.startsWith("peaks-fixed") ? "--series-6" : key.startsWith("peaks-mass") ? "--series-7"
    : key.startsWith("peaks-reserve") ? "--series-8" : "--text-secondary"));
const DEFAULT_ALGOS = new Set(["expected", "greedy-density", "peaks-mass-group"]);

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else node.setAttribute(k, v);
  }
  if (text !== undefined) node.textContent = text;
  return node;
}

function clock(min) {
  if (min == null || !isFinite(min)) return "–";
  const h = Math.floor(min / 60);
  const m = min - 60 * h;
  return `${h}h ${m.toFixed(1).padStart(4, "0")}m`;
}
const fmt = (v, d = 1) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));
const pm = (v, ci, d = 1, scale = 1, signed = false) =>
  v == null ? "–"
    : `${signed && v > 0 ? "+" : ""}${(v / scale).toFixed(d)} ± ${ci == null ? "–" : (ci / scale).toFixed(d)}`;
const algoRank = (key) => Object.keys(state.options.algorithms).indexOf(key);

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}

function upperBound(arr, t, key = (x) => x) {
  // number of elements with key <= t, for arr sorted by key
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (key(arr[mid]) <= t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/* ------------------------------------------------------------------- state */

const state = {
  options: null,
  run: null,         // last /api/run response
  order: [],         // algorithm keys in display order
  t: 0,              // playhead (minutes)
  tMax: 0,
  playing: false,
  lastFrame: 0,
  charts: {},
  replays: {},       // key -> { canvas, ctx, grid, info, data }
  bench: null,
};

/* -------------------------------------------------------------------- init */

async function init() {
  state.options = await api("/api/options");
  for (const select of document.querySelectorAll("select[name=depot]")) {
    for (const [key, km] of Object.entries(state.options.depots)) {
      select.append(el("option", { value: key }, `${key[0].toUpperCase()}${key.slice(1)} (${km} km)`));
    }
  }
  for (const fs of document.querySelectorAll("[data-role=algos]")) {
    for (const [key, a] of Object.entries(state.options.algorithms)) {
      const label = el("label", { title: a.rule });
      const box = el("input", { type: "checkbox", name: "algos", value: key });
      box.checked = DEFAULT_ALGOS.has(key);
      const sw = el("span", { class: "swatch" });
      sw.style.background = algoColor(key);
      label.append(box, sw, document.createTextNode(a.name));
      fs.append(label);
    }
  }
  const baseline = $("select[name=baseline]");
  for (const [key, a] of Object.entries(state.options.algorithms)) {
    const opt = el("option", { value: key }, a.name);
    if (key === "brute") opt.selected = true;
    baseline.append(opt);
  }

  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".tabpanel").forEach((p) => { p.hidden = p.id !== `tab-${tab.dataset.tab}`; });
    if (tab.dataset.tab === "scenario") resizeReplays();
  }));

  $("#run-form").addEventListener("submit", onRun);
  $("#bench-form").addEventListener("submit", onBench);
  $("#play").addEventListener("click", togglePlay);
  $("#scrub").addEventListener("input", (e) => {
    setTime((e.target.value / 1000) * state.tMax);
  });
  $("#show-unfound").addEventListener("change", drawReplays);
  $("#bench-people").addEventListener("change", renderBenchCharts);
  window.addEventListener("resize", () => { resizeReplays(); });
  buildLegend();
}

function formValues(form) {
  const data = new FormData(form);
  const out = {};
  for (const [k, v] of data.entries()) {
    if (k === "algos") (out.algos ||= []).push(v);
    else out[k] = v;
  }
  out.check = data.has("check");
  return out;
}

/* ---------------------------------------------------------------- scenario */

async function onRun(event) {
  event.preventDefault();
  const params = formValues(event.target);
  if (!params.algos) return setStatus("#run-status", "Pick at least one algorithm.", true);
  const btn = event.target.querySelector("button[type=submit]");
  btn.disabled = true;
  setStatus("#run-status", "Running…");
  $("#run-output").style.opacity = 0.5;
  try {
    const t0 = performance.now();
    const data = await api("/api/run", {
      drones: +params.drones, people: +params.people, depot: params.depot, campus: params.campus,
      cell_side: +params.cell_side, seed: +params.seed, algos: params.algos, check: params.check,
    });
    state.run = data;
    state.order = Object.keys(data.results);
    renderRun();
    setStatus("#run-status", `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s` +
      (params.check ? " — built-in checks passed." : "."));
  } catch (err) {
    setStatus("#run-status", err.message, true);
  } finally {
    btn.disabled = false;
    $("#run-output").style.opacity = 1;
  }
}

function setStatus(sel, text, error = false) {
  const node = $(sel);
  node.textContent = text;
  node.classList.toggle("error", error);
}

function renderRun() {
  const { scenario: sc, results } = state.run;
  const m = sc.map;
  $("#map-note").textContent = m
    ? `Real UW campus: ${m.n_buildings} buildings, ${Math.round(m.indoor_fraction * 100)}% of people indoors ` +
      `(placed by floor area). Map data ${m.source.replace(" - https://www.openstreetmap.org/copyright", "")}, ` +
      `fetched ${m.fetched}.`
    : "Square approximation of the campus. Run `python campus_map.py fetch` to use the real UW map.";
  $("#scenario-info").textContent =
    `${sc.k}×${sc.k} grid of ${sc.cell_side_m.toFixed(1)} m cells, ${sc.area_km2.toFixed(2)} km² searched · ` +
    `${(sc.search_time_min * 60).toFixed(1)} s to search a cell · ` +
    `${sc.trip_time_min.toFixed(1)} min depot round trip from the campus edge · ` +
    `${state.run.drones} drones, ${state.run.people} people, seed ${state.run.seed}`;
  $("#run-output").hidden = false;
  state.tMax = Math.max(...state.order.map((k) => results[k].metrics.completion), 1e-6);
  renderMetricsTable();
  renderTimeTable();
  renderCurves();
  buildReplays();
  setTime(0);
}

const METRIC_ROWS = [
  ["completion", "Everyone served", clock],
  ["all_found", "Everyone found", clock],
  ["mean_supplied_at", "Average time to be served", clock],
  ["mean_wait", "Mean wait, found → served (min)", (v) => fmt(v)],
  ["p90_wait", "90th percentile wait (min)", (v) => fmt(v)],
  ["max_wait", "Longest wait (min)", (v) => fmt(v)],
  ["trips", "Depot round trips", (v) => fmt(v, 0)],
  ["unused_kits", "Kits fetched but never used", (v) => fmt(v, 0)],
];

function renderMetricsTable() {
  const { results } = state.run;
  const table = $("#metrics-table");
  table.replaceChildren();
  const head = el("tr");
  head.append(el("th", {}, "Metric"));
  for (const key of state.order) {
    const th = el("th");
    const sw = el("span", { class: "swatch" });
    sw.style.background = algoColor(key);
    th.append(sw, document.createTextNode(` ${results[key].name}`));
    head.append(th);
  }
  table.appendChild(el("thead")).append(head);
  const body = table.appendChild(el("tbody"));
  for (const [metric, label, format] of METRIC_ROWS) {
    const tr = el("tr");
    const td = el("td", {}, label);
    td.append(el("span", { class: "desc" }, state.options.metrics[metric] || ""));
    tr.append(td);
    const values = state.order.map((k) => results[k].metrics[metric]);
    const best = Math.min(...values);
    values.forEach((v) => {
      const cell = el("td", {}, format(v));
      if (state.order.length > 1 && Math.abs(v - best) < 1e-9) cell.classList.add("best");
      tr.append(cell);
    });
    body.append(tr);
  }
}

const STATUS_LABEL = {
  idle: "Idle", transit: "Flying to a cell", search: "Searching", deliver: "Flying to serve",
  exit: "Flying to campus edge", return: "Depot round trip",
};

function renderTimeTable() {
  const { results } = state.run;
  const table = $("#time-table");
  table.replaceChildren();
  const head = el("tr");
  head.append(el("th", {}, "Activity"));
  for (const key of state.order) head.append(el("th", {}, results[key].name));
  table.appendChild(el("thead")).append(head);
  const body = table.appendChild(el("tbody"));
  for (const status of state.options.statuses) {
    const tr = el("tr");
    tr.append(el("td", {}, STATUS_LABEL[status] || status));
    for (const key of state.order) {
      const share = results[key].metrics[`share_${status}`] || 0;
      const td = el("td", {}, `${(share * 100).toFixed(1)}%`);
      const bar = el("span", { class: "bar" });
      bar.style.width = `${Math.round(share * 120)}px`;
      bar.style.background = algoColor(key);
      td.append(bar);
      tr.append(td);
    }
    body.append(tr);
  }
}

/* ------------------------------------------------------------------ charts */

const playheadPlugin = {
  id: "playhead",
  afterDatasetsDraw(chart) {
    if (!state.run) return;
    const x = chart.scales.x.getPixelForValue(state.t / 60);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = css("--text-muted");
    ctx.setLineDash([4, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.restore();
  },
};

function stepSeries(times, tEnd) {
  const pts = [{ x: 0, y: 0 }];
  times.forEach((t, i) => pts.push({ x: t / 60, y: i + 1 }));
  pts.push({ x: tEnd / 60, y: times.length });
  return pts;
}

function countAt(times, tMin) { return upperBound(times, tMin); }

function baseChartOptions(yTitle, xTitle = "Hours since the earthquake") {
  const ink = css("--text-secondary");
  const grid = css("--grid");
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    parsing: false, normalized: true,
    interaction: { mode: "nearest", axis: "x", intersect: false },
    scales: {
      x: { type: "linear", title: { display: true, text: xTitle, color: ink },
           ticks: { color: ink }, grid: { color: grid }, border: { color: grid } },
      y: { beginAtZero: true, title: { display: true, text: yTitle, color: ink },
           ticks: { color: ink, precision: 0 }, grid: { color: grid }, border: { color: grid } },
    },
    plugins: {
      legend: { labels: { color: ink, boxWidth: 14, boxHeight: 2, usePointStyle: false } },
    },
  };
}

function renderCurves() {
  const { results } = state.run;
  for (const [id, field, label] of [["chart-found", "found", "People found"],
                                     ["chart-served", "supplied", "People served"]]) {
    state.charts[id]?.destroy();
    const opts = baseChartOptions(label);
    opts.scales.x.min = 0;
    opts.scales.x.max = state.tMax / 60;
    opts.plugins.tooltip = {
      callbacks: {
        title: (items) => `t = ${clock(items[0].parsed.x * 60)}`,
        label: () => null,
        afterBody: (items) => {
          const t = items[0].parsed.x * 60;
          return state.order.map((k) =>
            `${countAt(results[k].curves[field], t)} / ${state.run.people}   ${results[k].name}`);
        },
      },
    };
    opts.onClick = (evt, _els, chart) => {
      const x = chart.scales.x.getValueForPixel(evt.x);
      setTime(Math.max(0, Math.min(state.tMax, x * 60)));
    };
    state.charts[id] = new Chart($(`#${id}`), {
      type: "line",
      data: {
        datasets: state.order.map((k) => ({
          label: results[k].name,
          data: stepSeries(results[k].curves[field], state.tMax),
          borderColor: algoColor(k), backgroundColor: algoColor(k),
          borderWidth: 2, pointRadius: 0, stepped: "after",
        })),
      },
      options: opts,
      plugins: [playheadPlugin],
    });
  }
}

/* ------------------------------------------------------------------ replay */

function buildReplays() {
  const wrap = $("#replays");
  wrap.replaceChildren();
  state.replays = {};
  for (const key of state.order) {
    const r = state.run.results[key];
    const box = el("div", { class: "replay" });
    const title = el("h3");
    const sw = el("span", { class: "swatch" });
    sw.style.background = algoColor(key);
    const name = el("span", {}, r.name);
    const progress = el("span", { class: "done" });
    title.append(sw, name, progress);
    const canvas = el("canvas");
    const info = el("pre");
    box.append(title, canvas, info);
    wrap.append(box);

    const rp = r.replay;
    const segs = Object.values(rp.segments);
    state.replays[key] = {
      canvas, ctx: canvas.getContext("2d"), info, progress, data: r,
      cells: rp.cells, people: rp.people, segs,
      decisions: rp.decisions,
      grid: document.createElement("canvas"),
    };
  }
  resizeReplays();
}

function resizeReplays() {
  const dpr = window.devicePixelRatio || 1;
  for (const rep of Object.values(state.replays)) {
    const w = rep.canvas.parentElement.clientWidth;
    if (!w) continue;
    rep.W = w;
    rep.strip = 56;
    rep.canvas.width = Math.round(w * dpr);
    rep.canvas.height = Math.round((w + rep.strip + 8) * dpr);
    rep.canvas.style.height = `${w + rep.strip + 8}px`;
    rep.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  drawReplays();
}

function setTime(t) {
  state.t = Math.max(0, Math.min(state.tMax, t));
  $("#scrub").value = Math.round((state.t / state.tMax) * 1000) || 0;
  $("#clock").textContent = clock(state.t);
  drawReplays();
  for (const chart of Object.values(state.charts)) {
    if (chart.canvas.id.startsWith("chart-bench")) continue;
    chart.draw();
  }
}

function togglePlay() {
  state.playing = !state.playing;
  $("#play").textContent = state.playing ? "Pause" : "Play";
  if (state.playing) {
    if (state.t >= state.tMax) setTime(0);
    state.lastFrame = performance.now();
    requestAnimationFrame(tick);
  }
}

function tick(now) {
  if (!state.playing) return;
  const dt = (now - state.lastFrame) / 1000;
  state.lastFrame = now;
  setTime(state.t + dt * Number($("#speed").value));
  if (state.t >= state.tMax) { togglePlay(); return; }
  requestAnimationFrame(tick);
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function drawReplays() {
  if (!state.run) return;
  const sc = state.run.scenario;
  const k = sc.k;
  const t = state.t;
  const colors = {
    un: hexToRgb(css("--cell-unsearched")), se: hexToRgb(css("--cell-searched")),
    off: hexToRgb(css("--cell-offcampus")), building: css("--building-line"),
    ink: css("--text-primary"), muted: css("--text-muted"), strip: css("--depot-strip"),
    waiting: css("--critical"), enroute: css("--warning"), served: css("--good"),
    secondary: css("--text-secondary"),
  };
  const showUnfound = $("#show-unfound").checked;

  for (const [key, rep] of Object.entries(state.replays)) {
    if (!rep.W) continue;
    const { ctx, W } = rep;
    const S = W / k;                       // pixels per cell
    const X = (x) => x * S;
    const Y = (y) => W - y * S;

    // --- grid: searched fraction per cell
    const g = rep.grid;
    if (g.width !== k) { g.width = k; g.height = k; }
    const gctx = g.getContext("2d");
    const img = gctx.createImageData(k, k);
    const px = img.data;
    const mask = sc.map ? sc.map.mask : null;
    for (let i = 0; i < k * k; i++) {
      const off = mask && mask[k - 1 - Math.floor(i / k)][i % k] === "0";
      const c = off ? colors.off : colors.un;
      px[4 * i] = c[0]; px[4 * i + 1] = c[1]; px[4 * i + 2] = c[2]; px[4 * i + 3] = 255;
    }
    for (const [cx, cy, a, b] of rep.cells) {
      if (t <= a) continue;
      const f = b > a ? Math.min(1, (t - a) / (b - a)) : 1;
      const i = (k - 1 - cy) * k + cx;
      for (let c = 0; c < 3; c++) px[4 * i + c] = colors.un[c] + f * (colors.se[c] - colors.un[c]);
    }
    gctx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, W, W + rep.strip + 8);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(g, 0, 0, W, W);
    if (sc.map) drawOutlines(ctx, sc.map, X, Y, colors);

    // depot exit point
    const [dx, dy] = sc.depot_point;
    ctx.fillStyle = colors.secondary;
    ctx.fillRect(X(dx) - 5, Y(dy) - 5, 10, 10);

    // --- people
    const pr = Math.max(2, Math.min(4, S * 0.35));
    let found = 0, served = 0;
    for (const [x, y, tf, ts, tv] of rep.people) {
      let color = null;
      if (tf != null && t >= tf) {
        found++;
        if (tv != null && t >= tv) { color = colors.served; served++; }
        else if (ts != null && t >= ts) color = colors.enroute;
        else color = colors.waiting;
      } else if (showUnfound) {
        ctx.strokeStyle = colors.muted;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(X(x), Y(y), pr, 0, 2 * Math.PI); ctx.stroke();
        continue;
      }
      if (!color) continue;
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(X(x), Y(y), pr, 0, 2 * Math.PI); ctx.fill();
    }

    // --- drones
    const ds = Math.max(5, Math.min(9, S * 0.7));
    const stripTop = W + 8;
    const counts = { search: 0, transit: 0, deliver: 0, idle: 0, exit: 0, return: 0 };
    rep.segs.forEach((segs, idx) => {
      const n = upperBound(segs, t, (s) => s[0]);
      const seg = segs[Math.max(0, n - 1)];
      if (!seg) return;
      const [t0, t1, status, loaded, x0, y0, x1, y1, group] = seg;
      // peak-group models: colour by group, scavengers muted; other models: ink
      const ink = group == null ? colors.ink
        : group < 0 ? colors.muted : GROUP_COLORS[group % GROUP_COLORS.length];
      const f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 1;
      counts[status] = (counts[status] || 0) + 1;
      if (status === "return") return;  // drawn in the depot strip below
      const x = x0 + f * (x1 - x0);
      const y = y0 + f * (y1 - y0);
      if (status === "deliver") {
        ctx.strokeStyle = colors.secondary;
        ctx.setLineDash([3, 3]);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(X(x), Y(y)); ctx.lineTo(X(x1), Y(y1)); ctx.stroke();
        ctx.setLineDash([]);
        drawDiamond(ctx, X(x), Y(y), ds, ink);
      } else if (status === "exit") {
        drawArrow(ctx, X(x), Y(y), ds, Math.sign(x1 - x0) || 1, false, ink);
      } else if (status === "idle") {
        ctx.strokeStyle = ink;
        ctx.fillStyle = ink;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(X(x), Y(y), ds * 0.5, 0, 2 * Math.PI);
        loaded ? ctx.fill() : ctx.stroke();
      } else {
        drawTriangle(ctx, X(x), Y(y), ds, !!loaded, ink);
      }
    });

    // depot strip
    ctx.fillStyle = colors.strip;
    ctx.fillRect(0, stripTop, W, rep.strip);
    ctx.fillStyle = colors.secondary;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "bottom";
    ctx.fillText("campus edge", 4, stripTop + rep.strip - 3);
    const depotLabel = `${sc.depot[0].toUpperCase()}${sc.depot.slice(1)} depot · ${sc.distance_km} km`;
    ctx.fillText(depotLabel, W - ctx.measureText(depotLabel).width - 4, stripTop + rep.strip - 3);
    // redraw strip drones on top of the strip background
    rep.segs.forEach((segs, idx) => {
      const n = upperBound(segs, t, (s) => s[0]);
      const seg = segs[Math.max(0, n - 1)];
      if (!seg || seg[2] !== "return") return;
      const [t0, t1] = seg;
      const f = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 1;
      const out = f < 0.5;
      const sx = 14 + (W - 28) * (out ? 2 * f : 2 * (1 - f));
      const sy = stripTop + 12 + (idx % 4) * 8;
      drawArrow(ctx, sx, sy, ds, out ? 1 : -1, !out, colors.ink);
    });

    // header + decision readout
    const total = rep.people.length;
    const done = t >= rep.data.metrics.completion - 1e-9;
    rep.progress.textContent = done ? `done at ${clock(rep.data.metrics.completion)}`
                                    : `found ${found}/${total} · served ${served}/${total}`;
    const di = upperBound(rep.decisions, t, (d) => d[0]) - 1;
    const d = rep.decisions[di];
    const lines = [
      `searching ${counts.search + counts.transit} · serving ${counts.deliver} · ` +
      `to/at depot ${counts.exit + counts.return} · idle ${counts.idle}`,
    ];
    if (d) {
      const [dt, mode, H, F, U, Sx, r, E, n, R] = d;
      lines.push(`last decision ${clock(dt)} (${mode}): H ${H} + F ${F} + r ${r} = ${H + F + r}` +
                 `  vs  U ${U} + E ${E.toFixed(2)} = ${(U + E).toFixed(2)}   [S ${Sx}, n ${n}]`);
    }
    rep.info.textContent = lines.join("\n");
  }
}

function drawOutlines(ctx, map, X, Y, colors) {
  const path = (ring) => {
    ctx.beginPath();
    ring.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.closePath();
  };
  ctx.lineWidth = 0.8;
  ctx.strokeStyle = colors.building;
  for (const ring of map.buildings) { path(ring); ctx.stroke(); }
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = colors.secondary;
  for (const ring of map.campus) { path(ring); ctx.stroke(); }
}

function drawTriangle(ctx, x, y, s, filled, ink) {
  ctx.beginPath();
  ctx.moveTo(x, y - s * 0.6);
  ctx.lineTo(x + s * 0.55, y + s * 0.45);
  ctx.lineTo(x - s * 0.55, y + s * 0.45);
  ctx.closePath();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = ink;
  ctx.fillStyle = ink;
  filled ? ctx.fill() : ctx.stroke();
}

function drawDiamond(ctx, x, y, s, ink) {
  ctx.beginPath();
  ctx.moveTo(x, y - s * 0.6); ctx.lineTo(x + s * 0.5, y);
  ctx.lineTo(x, y + s * 0.6); ctx.lineTo(x - s * 0.5, y);
  ctx.closePath();
  ctx.fillStyle = ink;
  ctx.fill();
}

function drawArrow(ctx, x, y, s, dir, filled, ink) {
  ctx.beginPath();
  ctx.moveTo(x + dir * s * 0.6, y);
  ctx.lineTo(x - dir * s * 0.5, y - s * 0.45);
  ctx.lineTo(x - dir * s * 0.5, y + s * 0.45);
  ctx.closePath();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = ink;
  ctx.fillStyle = ink;
  filled ? ctx.fill() : ctx.stroke();
}

function buildLegend() {
  const ink = css("--text-primary");
  const items = [
    ["Searching, carrying a kit", (c) => drawTriangle(c, 8, 9, 11, true, ink)],
    ["Searching, empty", (c) => drawTriangle(c, 8, 9, 11, false, ink)],
    ["Flying to serve someone", (c) => drawDiamond(c, 8, 8, 11, ink)],
    ["Idle (filled = carrying a kit)", (c) => { c.fillStyle = ink; c.beginPath(); c.arc(8, 8, 4.5, 0, 7); c.fill(); }],
    ["To depot, empty (on campus and in the strip)", (c) => drawArrow(c, 8, 8, 11, 1, false, ink)],
    ["Back from depot, carrying a kit", (c) => drawArrow(c, 8, 8, 11, -1, true, ink)],
    ["Found, waiting for a drone", (c) => dot(c, css("--critical"))],
    ["Found, drone on the way", (c) => dot(c, css("--warning"))],
    ["Served", (c) => dot(c, css("--good"))],
    ["Campus exit toward the depot", (c) => { c.fillStyle = css("--text-secondary"); c.fillRect(3, 3, 10, 10); }],
    ["Peak-group models: drone colour = group, grey = scavenger", (c) => {
      GROUP_COLORS.slice(0, 3).forEach((g, i) => { c.fillStyle = g; c.fillRect(1 + 5 * i, 4, 4, 8); }); }],
    ["Building outline (real map)", (c) => { c.strokeStyle = css("--building-line"); c.lineWidth = 1; c.strokeRect(3, 3, 10, 10); }],
  ];
  function dot(c, color) { c.fillStyle = color; c.beginPath(); c.arc(8, 8, 4, 0, 7); c.fill(); }
  const ul = $("#replay-legend");
  ul.replaceChildren();
  const dpr = window.devicePixelRatio || 1;
  for (const [label, draw] of items) {
    const li = el("li");
    const c = el("canvas");
    c.width = 16 * dpr; c.height = 16 * dpr;
    const cx = c.getContext("2d");
    cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw(cx);
    li.append(c, document.createTextNode(label));
    ul.append(li);
  }
}

/* --------------------------------------------------------------- benchmark */

function parseList(text) {
  return String(text).split(/[\s,]+/).filter(Boolean).map(Number).filter((v) => Number.isFinite(v) && v >= 0);
}

async function onBench(event) {
  event.preventDefault();
  const params = formValues(event.target);
  if (!params.algos) return setStatus("#bench-status", "Pick at least one algorithm.", true);
  const drones = parseList(params.drones), people = parseList(params.people);
  if (!drones.length || !people.length) return setStatus("#bench-status", "Enter drone and people counts.", true);
  const btn = event.target.querySelector("button[type=submit]");
  btn.disabled = true;
  setStatus("#bench-status", "");
  const bar = $("#bench-progress");
  bar.hidden = false;
  try {
    const { job } = await api("/api/benchmark", {
      drones, people, runs: +params.runs, depot: params.depot, cell_side: +params.cell_side,
      campus: params.campus,
      algos: params.algos, baseline: params.baseline, check: params.check,
    });
    const started = performance.now();
    for (;;) {
      const status = await api(`/api/benchmark/${job}`);
      const frac = status.total ? status.done / status.total : 0;
      bar.firstElementChild.style.width = `${(frac * 100).toFixed(1)}%`;
      bar.lastElementChild.textContent = `${status.done} / ${status.total} runs`;
      if (status.finished) {
        if (status.error) throw new Error(status.error);
        state.bench = status;
        $("#bench-csv").href = `/api/benchmark/${job}.csv`;
        renderBench();
        setStatus("#bench-status",
          `${status.total} runs in ${((performance.now() - started) / 1000).toFixed(0)} s` +
          (params.check ? " — built-in checks passed." : "."));
        break;
      }
      await new Promise((r) => setTimeout(r, 600));
    }
  } catch (err) {
    setStatus("#bench-status", err.message, true);
  } finally {
    btn.disabled = false;
  }
}

function renderBench() {
  $("#bench-output").hidden = false;
  const { config } = state.bench;
  const summary = state.bench.summary = [...state.bench.summary].sort((a, b) =>
    a.drones - b.drones || a.people - b.people || algoRank(a.algo) - algoRank(b.algo));
  const names = state.options.algorithms;
  const table = $("#bench-table");
  table.replaceChildren();
  const head = el("tr");
  for (const h of ["Algorithm", "Everyone served (h)", "Mean wait (min)", "Longest wait (min)",
                   "Everyone found (h)", "Unused kits",
                   `Δ served vs ${names[config.baseline].name} (min)`, "Δ mean wait (min)", "Faster in"]) {
    head.append(el("th", {}, h));
  }
  table.appendChild(el("thead")).append(head);
  const body = table.appendChild(el("tbody"));
  let group = null;
  for (const s of summary) {
    const g = `${s.drones} drones · ${s.people} people · ${s.runs} seeds`;
    if (g !== group) {
      group = g;
      const tr = el("tr", { class: "group" });
      tr.append(el("td", { colspan: 9 }, g));
      body.append(tr);
    }
    const tr = el("tr");
    const name = el("td");
    const sw = el("span", { class: "swatch" });
    sw.style.background = algoColor(s.algo);
    name.append(sw, document.createTextNode(` ${names[s.algo].name}`));
    tr.append(
      name,
      el("td", {}, pm(s.completion, s.completion_ci, 2, 60)),
      el("td", {}, pm(s.mean_wait, s.mean_wait_ci, 1)),
      el("td", {}, fmt(s.max_wait)),
      el("td", {}, fmt(s.all_found / 60, 2)),
      el("td", {}, fmt(s.unused_kits)),
      el("td", {}, s.completion_diff == null ? "baseline" : pm(s.completion_diff, s.completion_diff_ci, 1, 1, true)),
      el("td", {}, s.mean_wait_diff == null ? "" : pm(s.mean_wait_diff, s.mean_wait_diff_ci, 1, 1, true)),
      el("td", {}, s.faster == null ? "" : `${s.faster}/${s.runs}`),
    );
    body.append(tr);
  }
  const sel = $("#bench-people");
  const people = [...new Set(summary.map((s) => s.people))].sort((a, b) => a - b);
  sel.replaceChildren(...people.map((p) => el("option", { value: p }, String(p))));
  renderBenchCharts();
}

function renderBenchCharts() {
  if (!state.bench) return;
  const p = Number($("#bench-people").value);
  const rows = state.bench.summary.filter((s) => s.people === p);
  const algos = [...new Set(rows.map((s) => s.algo))];
  const names = state.options.algorithms;
  for (const [id, metric, label, scale] of [
    ["chart-bench-completion", "completion", "Everyone served (hours)", 60],
    ["chart-bench-wait", "mean_wait", "Mean wait (minutes)", 1],
  ]) {
    state.charts[id]?.destroy();
    const opts = baseChartOptions(label, "Drones");
    opts.interaction = { mode: "index", intersect: false };
    opts.scales.y.beginAtZero = true;
    opts.plugins.tooltip = {
      callbacks: {
        title: (items) => `${items[0].parsed.x} drones, ${p} people`,
        label: (item) => {
          const s = rows.find((r) => r.algo === algos[item.datasetIndex] && r.drones === item.parsed.x);
          return `${names[s.algo].name}: ${pm(s[metric], s[`${metric}_ci`], scale === 60 ? 2 : 1, scale)}`;
        },
      },
    };
    state.charts[id] = new Chart($(`#${id}`), {
      type: "line",
      data: {
        datasets: algos.map((a) => ({
          label: names[a].name,
          data: rows.filter((s) => s.algo === a).sort((x, y) => x.drones - y.drones)
                    .map((s) => ({ x: s.drones, y: s[metric] / scale })),
          borderColor: algoColor(a), backgroundColor: algoColor(a),
          borderWidth: 2, pointRadius: 4, pointHoverRadius: 6,
        })),
      },
      options: opts,
    });
  }
}

init().catch((err) => setStatus("#run-status", `Could not reach the simulator: ${err.message}`, true));
