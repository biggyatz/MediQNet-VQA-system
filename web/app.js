"use strict";
const el = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Official VQA-Med 2019 leaderboard (Ben Abacha et al. 2019, Table 3). Per-category values in the
// paper are contributions to overall accuracy (out of 0.25), converted here to accuracy (x4).
const BEST = { team: "Hanlin", modality: 0.808, plane: 0.768, organ: 0.736, abnormality: 0.184, overall: 0.624 };
const LEADERBOARD = [0.624, 0.620, 0.616, 0.606, 0.566, 0.564, 0.558, 0.556, 0.536, 0.534, 0.488, 0.462, 0.366, 0.294, 0.210, 0.160, 0.088];
const SUGGEST = {
  modality: "what modality is shown?",
  plane: "in what plane is this image oriented?",
  organ: "what organ system is shown in this image?",
  abnormality: "what is the primary abnormality in this image?",
  normal: "is this image abnormal?",
};
const CAT_LABEL = { modality: "Modality", plane: "Plane", organ: "Organ system", abnormality: "Abnormality" };

let session, V, METRICS, SAMPLES, current = null;
const tip = el("tooltip");
function bindTips(svg, lookup) {
  svg.querySelectorAll("[data-tip]").forEach(n => {
    n.addEventListener("pointermove", e => {
      const t = lookup(n.dataset.tip);
      tip.innerHTML = `<b>${esc(t.title)}</b>` + t.rows.map(([k, v]) => `<div class="row"><span>${esc(k)}</span><span>${esc(v)}</span></div>`).join("");
      tip.hidden = false;
      const w = tip.offsetWidth, h = tip.offsetHeight;
      let x = e.clientX + 14, y = e.clientY + 14;
      if (x + w > innerWidth - 8) x = e.clientX - w - 14;
      if (y + h > innerHeight - 8) y = e.clientY - h - 14;
      tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
    });
    n.addEventListener("pointerleave", () => { tip.hidden = true; });
  });
}
function hbar(x, y, w, h, r = 4) {
  w = Math.max(w, 1); r = Math.min(r, w, h / 2);
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}

Promise.all([fetch("vocab.json").then(r => r.json()), fetch("metrics.json").then(r => r.json()), fetch("samples.json").then(r => r.json())])
  .then(([v, m, s]) => { V = v; METRICS = m; SAMPLES = s; init(); });

function init() {
  renderKPIs(); renderCapabilities();
  el("gallery").innerHTML = SAMPLES.map((s, i) => `<button class="thumb" data-i="${i}" aria-pressed="false" title="${esc(CAT_LABEL[s.category])} question"><img src="samples/${esc(s.id)}.jpg" alt="Test image ${esc(s.id)}" loading="lazy"></button>`).join("");
  el("gallery").addEventListener("click", e => { const b = e.target.closest(".thumb"); if (b) pickSample(+b.dataset.i); });
  el("suggest").innerHTML = Object.entries(SUGGEST).map(([k, q]) => `<button class="chip" type="button" data-q="${esc(q)}">${k === "normal" ? "Normal?" : CAT_LABEL[k]}</button>`).join("");
  el("suggest").addEventListener("click", e => { const b = e.target.closest(".chip"); if (!b) return; el("question").value = b.dataset.q; ask(); });
  el("qform").addEventListener("submit", e => { e.preventDefault(); ask(); });
  el("upload").addEventListener("change", e => {
    const f = e.target.files[0]; if (!f) return;
    current = { sample: null, url: URL.createObjectURL(f) };
    document.querySelectorAll(".thumb").forEach(t => t.setAttribute("aria-pressed", "false"));
    el("viewer").src = current.url; el("answer").textContent = "–"; el("status").textContent = "Your image. Ask a question."; el("top-chart").innerHTML = "";
  });
  document.querySelectorAll("[data-table-toggle]").forEach(btn => btn.addEventListener("click", () => {
    const t = el(btn.dataset.tableToggle + "-table"); t.hidden = !t.hidden; btn.textContent = t.hidden ? "Show table" : "Hide table";
  }));
  pickSample(0, false);
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
  ort.InferenceSession.create("mediqnet.onnx", { executionProviders: ["wasm"] }).then(s => {
    session = s; el("ask").disabled = false; el("ask").textContent = "Ask";
    ask();
  }).catch(err => { el("ask").textContent = "Model failed to load"; el("status").textContent = String(err); });
  let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(renderCapabilities, 120); });
}

function pickSample(i, run = true) {
  const s = SAMPLES[i];
  current = { sample: s, url: `samples/${s.id}.jpg` };
  document.querySelectorAll(".thumb").forEach(t => t.setAttribute("aria-pressed", String(+t.dataset.i === i)));
  el("viewer").src = current.url;
  el("question").value = s.question;
  if (run && session) ask();
}

function renderKPIs() {
  const t = METRICS.test, rank = LEADERBOARD.filter(x => x > t.overall).length + 1;
  const tiles = [
    ["Test accuracy", `${(t.overall * 100).toFixed(1)}%`, "exact match, 500 official test questions"],
    ["Best category", `${(Math.max(t.modality, t.plane, t.organ) * 100).toFixed(0)}%`, ["modality", "plane", "organ"].sort((a, b) => t[b] - t[a])[0] + " questions"],
    ["vs. 2019 challenge", `#${rank} of ${LEADERBOARD.length + 1}`, `best team: ${(BEST.overall * 100).toFixed(1)}%`],
    ["Answers it knows", V.answers.length.toLocaleString("en-GB"), "runs on-device, no server"],
  ];
  el("kpis").innerHTML = tiles.map(([l, v, n]) => `<div class="kpi"><div class="label">${l}</div><div class="value">${v}</div><div class="note">${esc(n)}</div></div>`).join("");
}

// ---------- inference ----------
const tok = q => (q.toLowerCase().match(/[a-z0-9]+/g) || []);
function bow(q) {
  const v = new Float32Array(V.vocab.length), idx = Object.fromEntries(V.vocab.map((w, i) => [w, i]));
  for (const w of tok(q)) if (w in idx) v[idx[w]] = 1;
  return v;
}
function category(qv) {
  let best = 0, bestZ = -Infinity;
  V.qcls_coef.forEach((row, c) => { let z = V.qcls_intercept[c]; for (let i = 0; i < row.length; i++) z += row[i] * qv[i]; if (z > bestZ) { bestZ = z; best = c; } });
  return best;
}
function imageTensor(img) {
  const c = document.createElement("canvas"); c.width = c.height = 224;
  const ctx = c.getContext("2d"); ctx.imageSmoothingQuality = "high"; ctx.drawImage(img, 0, 0, 224, 224);
  const d = ctx.getImageData(0, 0, 224, 224).data, out = new Float32Array(3 * 224 * 224);
  for (let p = 0; p < 224 * 224; p++) for (let ch = 0; ch < 3; ch++) out[ch * 50176 + p] = (d[p * 4 + ch] / 255 - V.mean[ch]) / V.std[ch];
  return new ort.Tensor("float32", out, [1, 3, 224, 224]);
}
function loadImage(url) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url; }); }

async function ask() {
  const q = el("question").value.trim();
  if (!session || !q || !current) return;
  el("ask").disabled = true;
  try {
    const qv = bow(q);
    if (qv.every(x => x === 0)) { el("answer").textContent = "–"; el("status").textContent = "I don't recognise any words in that question. Try one of the suggested questions."; el("top-chart").innerHTML = ""; return; }
    const cat = category(qv), img = await loadImage(current.url);
    const t0 = performance.now();
    const out = await session.run({ image: imageTensor(img), question: new ort.Tensor("float32", qv, [1, qv.length]) });
    const ms = performance.now() - t0;
    const logits = out.logits.data, mask = V.cat_mask[cat];
    let mx = -Infinity; for (let i = 0; i < logits.length; i++) if (mask[i] && logits[i] > mx) mx = logits[i];
    let sum = 0; const probs = []; for (let i = 0; i < logits.length; i++) if (mask[i]) { const p = Math.exp(logits[i] - mx); probs.push([i, p]); sum += p; }
    probs.forEach(p => { p[1] /= sum; }); probs.sort((a, b) => b[1] - a[1]);
    const top = probs.slice(0, 5).map(([i, p]) => ({ answer: V.answers[i], p }));
    el("answer").textContent = top[0].answer;
    let status = `Detected a <strong>${CAT_LABEL[V.categories[cat]].toLowerCase()}</strong> question · ${Math.round(ms)} ms on your device`;
    const s = current.sample;
    if (s && q.toLowerCase() === s.question.toLowerCase()) {
      const ok = top[0].answer === s.answer;
      status += ` · Reference answer: <strong>${esc(s.answer)}</strong> <span class="${ok ? "ok" : "miss"}">${ok ? "✓ correct" : "✗ different"}</span>`;
    }
    el("status").innerHTML = status;
    renderTop(top);
  } finally { el("ask").disabled = false; }
}

function renderTop(top) {
  const host = el("top-chart"), W = Math.max(280, host.clientWidth), labelW = Math.min(220, W * 0.5), rowH = 26, barH = 14, H = top.length * rowH + 6;
  const x = v => labelW + v * (W - labelW - 50);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Top answers with confidence">`;
  top.forEach((t, i) => {
    const y = i * rowH + 5, label = t.answer.length > 32 ? t.answer.slice(0, 31) + "…" : t.answer;
    s += `<text class="${i ? "lbl" : "lbl-strong"}" x="${labelW - 8}" y="${y + barH / 2 + 4}" text-anchor="end">${esc(label)}</text>`;
    s += `<path d="${hbar(labelW, y, x(t.p) - labelW, barH)}" fill="var(--accent)" opacity="${i ? 0.55 : 1}"/>`;
    s += `<text class="val" x="${x(t.p) + 6}" y="${y + barH / 2 + 4}">${(t.p * 100).toFixed(0)}%</text>`;
    s += `<rect class="hit" data-tip="${i}" x="0" y="${y - 6}" width="${W}" height="${rowH}"/>`;
  });
  host.innerHTML = s + "</svg>";
  bindTips(host.querySelector("svg"), i => ({ title: top[i].answer, rows: [["Confidence", (top[i].p * 100).toFixed(1) + "%"]] }));
}

// ---------- capabilities chart ----------
function renderCapabilities() {
  const t = METRICS.test, cats = ["modality", "plane", "organ", "abnormality", "overall"];
  const series = [{ key: "ours", label: "MediQNet (this demo)", color: "var(--a)", v: c => t[c] }, { key: "best", label: `Best 2019 team (${BEST.team})`, color: "var(--b)", v: c => BEST[c] }];
  el("cap-legend").innerHTML = series.map(s => `<span class="key"><span class="sw" style="background:${s.color}"></span>${esc(s.label)}</span>`).join("");
  const host = el("cap-chart"), W = Math.max(300, host.clientWidth), labelW = 110, barH = 12, gap = 2, groupH = 2 * barH + gap, rowH = groupH + 18, H = cats.length * rowH + 24;
  const x = v => labelW + v * (W - labelW - 50);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy by question category">`;
  for (const g of [0, 0.25, 0.5, 0.75, 1]) s += `<line class="gridline" x1="${x(g)}" x2="${x(g)}" y1="0" y2="${H - 20}"/><text class="tick" x="${x(g)}" y="${H - 6}" text-anchor="middle">${g * 100}%</text>`;
  const cells = [];
  cats.forEach((c, i) => {
    const y = i * rowH + 6;
    s += `<text class="${c === "overall" ? "lbl-strong" : "lbl"}" x="${labelW - 10}" y="${y + groupH / 2 + 4}" text-anchor="end">${c === "overall" ? "Overall" : CAT_LABEL[c]}</text>`;
    series.forEach((se, j) => {
      const v = se.v(c), by = y + j * (barH + gap);
      cells.push({ c, se, v });
      s += `<path d="${hbar(labelW, by, x(v) - labelW, barH)}" fill="${se.color}"/><text class="val" x="${x(v) + 6}" y="${by + barH - 1}">${(v * 100).toFixed(1)}%</text>`;
      s += `<rect class="hit" data-tip="${cells.length - 1}" x="${labelW}" y="${by - 1}" width="${W - labelW}" height="${barH + 2}"/>`;
    });
  });
  host.innerHTML = s + `<line class="baseline" x1="${labelW}" x2="${labelW}" y1="0" y2="${H - 20}"/></svg>`;
  bindTips(host.querySelector("svg"), i => ({ title: cells[i].se.label, rows: [[cells[i].c === "overall" ? "Overall" : CAT_LABEL[cells[i].c], (cells[i].v * 100).toFixed(1) + "%"]] }));
  el("cap-table").innerHTML = `<table><thead><tr><th>Category</th><th class="n">MediQNet</th><th class="n">Best 2019 team</th></tr></thead><tbody>` +
    cats.map(c => `<tr><td>${c === "overall" ? "Overall" : CAT_LABEL[c]}</td><td class="n">${(t[c] * 100).toFixed(1)}%</td><td class="n">${(BEST[c] * 100).toFixed(1)}%</td></tr>`).join("") + "</tbody></table>";
  const rank = LEADERBOARD.filter(x => x > t.overall).length + 1;
  el("cap-insight").innerHTML = `Overall ${(t.overall * 100).toFixed(1)}% would place <strong>#${rank} of ${LEADERBOARD.length + 1}</strong> on the 2019 challenge leaderboard, from a model small enough to run in a browser. ` +
    `Modality, plane and organ questions have a fixed set of answers and are answered well. Naming the specific abnormality is open-ended (over 1,000 possible diagnoses) and remains hard for every system: the best 2019 team scored ${(BEST.abnormality * 100).toFixed(1)}%.`;
}
