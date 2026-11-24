const $ = (id) => document.getElementById(id);

function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }

async function api(url, options) {
  showError("");
  let res;
  try { res = await fetch(url, options); } catch (err) { showError("Could not reach the server."); throw err; }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { showError(data.detail || ("Request failed (" + res.status + ")")); throw new Error(data.detail || res.status); }
  return data;
}

// ---------- tabs ----------
let loaded = {};
document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === btn.dataset.tab));
  showError("");
  if (!loaded[btn.dataset.tab]) { loaded[btn.dataset.tab] = true; initTab(btn.dataset.tab); }
}));

function initTab(name) {
  if (name === "mae") shuffleTiles();
  if (name === "vae") { loadClasses().then(() => newRandom()); }
  if (name === "diffusion") loadClasses();
}

// ---------- MAE ----------
let maeId = null, maeSeed = 0;
async function shuffleTiles() {
  $("mae-shuffle").disabled = true;
  try {
    const tiles = await api("/api/mae/tiles?n=24&seed=" + Math.floor(Math.random() * 1e6));
    const box = $("mae-tiles"); box.innerHTML = "";
    tiles.forEach((t) => {
      const f = document.createElement("figure"); f.dataset.id = t.id;
      const img = document.createElement("img"); img.src = t.img; img.alt = t.label;
      const cap = document.createElement("div"); cap.textContent = t.label;
      f.append(img, cap);
      f.addEventListener("click", () => selectTile(t.id, f));
      box.append(f);
    });
  } finally { $("mae-shuffle").disabled = false; }
}
async function selectTile(id, fig) {
  document.querySelectorAll("#mae-tiles figure").forEach((f) => f.classList.toggle("sel", f === fig));
  maeId = id; maeSeed = Math.floor(Math.random() * 1e6); await runMae();
}
async function runMae() {
  if (maeId === null) return;
  const r = await api("/api/mae/run?index=" + maeId + "&seed=" + maeSeed);
  $("mae-masked").src = r.masked; $("mae-recon").src = r.reconstruction; $("mae-orig").src = r.original;
  $("mae-pred").textContent = "Predicted land cover: " + r.predicted + "   (true label: " + r.true_label + ")";
  const box = $("mae-top3"); box.innerHTML = "";
  r.top3.forEach((t) => {
    const d = document.createElement("div"); d.className = "bar";
    const name = document.createElement("span"); name.style.width = "170px"; name.textContent = t.label;
    const bar = document.createElement("div"); bar.className = "b"; bar.style.width = Math.max(2, t.prob * 260) + "px";
    const pct = document.createElement("span"); pct.textContent = (t.prob * 100).toFixed(1) + "%";
    d.append(name, bar, pct); box.append(d);
  });
  $("mae-result").hidden = false; $("mae-newmask").disabled = false;
}
$("mae-shuffle").addEventListener("click", shuffleTiles);
$("mae-newmask").addEventListener("click", () => { maeSeed = Math.floor(Math.random() * 1e6); runMae(); });

// ---------- VAE ----------
let frames = [], ts = [];
async function loadClasses() {
  if ($("vae-a").options.length) return;
  const names = await api("/api/vae/classes");
  ["vae-a", "vae-b", "diff-char"].forEach((id, k) => {
    const sel = $(id); names.forEach((n) => { const o = document.createElement("option"); o.value = o.textContent = n; sel.append(o); });
    if (id === "vae-b") sel.selectedIndex = Math.min(5, names.length - 1);
  });
}
async function newRandom() {
  $("vae-random").disabled = true;
  try {
    const imgs = await api("/api/vae/random?n=16&seed=" + Math.floor(Math.random() * 1e6));
    const box = $("vae-grid"); box.innerHTML = "";
    imgs.forEach((src) => { const i = document.createElement("img"); i.src = src; box.append(i); });
  } finally { $("vae-random").disabled = false; }
}
async function pickPair() {
  $("vae-pair").disabled = true;
  try {
    const r = await api("/api/vae/pair?a=" + encodeURIComponent($("vae-a").value) + "&b=" + encodeURIComponent($("vae-b").value) +
                        "&seed=" + Math.floor(Math.random() * 1e6));
    frames = r.frames; ts = r.ts;
    $("vae-img-a").src = r.a; $("vae-img-b").src = r.b;
    const s = $("vae-slider"); s.max = frames.length - 1; s.value = 0; s.hidden = false; $("vae-interp").hidden = false;
    showFrame();
  } finally { $("vae-pair").disabled = false; }
}
function showFrame() {
  const k = parseInt($("vae-slider").value, 10);
  $("vae-frame").src = frames[k]; $("vae-cap").textContent = "t = " + ts[k].toFixed(2);
}
$("vae-random").addEventListener("click", newRandom);
$("vae-pair").addEventListener("click", pickPair);
$("vae-slider").addEventListener("input", showFrame);

// ---------- Diffusion ----------
$("diff-go").addEventListener("click", async () => {
  const btn = $("diff-go"); btn.disabled = true; $("diff-status").textContent = "generating (1000 denoising steps)...";
  const t0 = performance.now();
  try {
    const imgs = await api("/api/diffusion/sample", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ character: $("diff-char").value, n: 8, seed: Math.floor(Math.random() * 1e6) }) });
    const box = $("diff-grid"); box.innerHTML = "";
    imgs.forEach((src) => { const i = document.createElement("img"); i.src = src; box.append(i); });
    $("diff-status").textContent = "done in " + ((performance.now() - t0) / 1000).toFixed(0) + " s";
  } catch (e) { $("diff-status").textContent = ""; } finally { btn.disabled = false; }
});

// ---------- RAG ----------
function passageCard(p) {
  const d = document.createElement("div"); d.className = "card";
  const meta = document.createElement("div"); meta.className = "meta";
  meta.textContent = "[" + p.n + "]  PubMed abstract id " + p.abstract_id + "  |  similarity " + p.score.toFixed(3);
  const body = document.createElement("div"); body.textContent = p.text;
  d.append(meta, body); return d;
}
$("rag-go").addEventListener("click", async () => {
  const q = $("rag-q").value.trim(); if (!q) { showError("Write a question first."); return; }
  const btn = $("rag-go"); btn.disabled = true; $("rag-status").textContent = "retrieving and generating (slow on a CPU)...";
  try {
    const r = await api("/api/rag", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: q }) });
    $("rag-notice").textContent = r.notice;
    $("rag-answer").textContent = r.answer; $("rag-idk").hidden = !r.idk;
    const cited = $("rag-cited"), other = $("rag-other"); cited.innerHTML = ""; other.innerHTML = "";
    const c = r.passages.filter((p) => p.cited);
    if (!c.length) { const m = document.createElement("p"); m.textContent = "The answer cites no passage."; cited.append(m); }
    c.forEach((p) => cited.append(passageCard(p)));
    r.passages.filter((p) => !p.cited).forEach((p) => other.append(passageCard(p)));
    $("rag-out").hidden = false; $("rag-status").textContent = "";
  } catch (e) { $("rag-status").textContent = ""; } finally { btn.disabled = false; }
});

loaded.mae = true; initTab("mae");
