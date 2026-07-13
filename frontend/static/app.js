/* LeadScore — frontend vanilla. Rutas /api relativas (nginx hace proxy). */

const API = "/api";

// ---------- utilidades compartidas ----------
const BAND_ICON = { caliente: "🔥", tibio: "🌡️", frío: "❄️" };

function mix(a, b, t) {
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}
/** Color térmico frío→tibio→caliente según p ∈ [0,1]. */
function thermal(p) {
  const cold = [47, 111, 176], warm = [217, 138, 36], hot = [224, 71, 76];
  const rgb = p < 0.5 ? mix(cold, warm, p / 0.5) : mix(warm, hot, (p - 0.5) / 0.5);
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

async function postJSON(path, body) {
  const res = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

// =====================================================================
// SIMULADOR (index.html)
// =====================================================================
function initSimulator() {
  const timeline = document.getElementById("timeline");
  if (!timeline) return;

  const state = { events: [] };
  const $ = (id) => document.getElementById(id);

  function addEvent(type) {
    state.events.push({
      type,
      item_category: $("category").value,
      seconds_since_prev: state.events.length === 0 ? 0 : Number($("seconds").value),
    });
    render();
    score();
  }

  function render() {
    timeline.innerHTML = "";
    state.events.forEach((e, i) => {
      const li = document.createElement("li");
      li.className = `tl-item ${e.type}`;
      const name = e.type === "view" ? "Vio producto" : "Agregó al carrito";
      li.innerHTML =
        `<span class="dot"></span>` +
        `<span class="label">${name} · ${e.item_category}</span>` +
        `<span class="meta">+${e.seconds_since_prev}s</span>`;
      timeline.appendChild(li);
    });
  }

  function setGauge(p) {
    const arc = $("gaugeArc");
    arc.style.strokeDasharray = `${(p * 100).toFixed(2)} 100`;
    arc.style.stroke = thermal(p);
    $("scoreNum").textContent = Math.round(p * 100);
  }

  function resetDisplay() {
    setGauge(0);
    $("scoreNum").textContent = "0";
    const band = $("band");
    band.dataset.band = "—";
    band.textContent = "— sin datos";
    $("actionText").textContent = "Agrega al menos un evento para obtener un score.";
    $("segment").textContent = "—";
    $("threshold").textContent = "—";
    document.querySelectorAll("#breakdown [data-k]").forEach((el) => (el.style.width = "0"));
    document.querySelectorAll("#breakdown [data-v]").forEach((el) => (el.textContent = "—"));
  }

  async function score() {
    if (state.events.length === 0) return resetDisplay();
    try {
      const r = await postJSON("/score", {
        events: state.events,
        hour_of_day: Number($("hour").value),
        day_of_week: Number($("dow").value),
      });
      setGauge(r.conversion_probability);
      const band = $("band");
      band.dataset.band = r.label;
      band.textContent = `${BAND_ICON[r.label] || ""} ${r.label}`;
      $("actionText").textContent = r.recommended_action;
      $("segment").textContent = r.segment;
      $("threshold").textContent = r.threshold;
      for (const [k, v] of Object.entries(r.model_breakdown)) {
        const bar = document.querySelector(`#breakdown [data-k="${k}"]`);
        const val = document.querySelector(`#breakdown [data-v="${k}"]`);
        if (bar) bar.style.width = `${v * 100}%`;
        if (val) val.textContent = v.toFixed(2);
      }
    } catch (err) {
      $("actionText").textContent = "No se pudo contactar la API. ¿Está el backend arriba?";
      console.error(err);
    }
  }

  document.querySelectorAll(".event-buttons [data-type]").forEach((b) =>
    b.addEventListener("click", () => addEvent(b.dataset.type))
  );
  $("undo").addEventListener("click", () => { state.events.pop(); render(); score(); });
  $("reset").addEventListener("click", () => { state.events = []; render(); resetDisplay(); });
  ["hour", "dow"].forEach((id) => $(id).addEventListener("change", score));

  resetDisplay();
}

// =====================================================================
// DASHBOARD (dashboard.html)
// =====================================================================
function initDashboard() {
  const dropzone = document.getElementById("dropzone");
  if (!dropzone) return;

  const $ = (id) => document.getElementById(id);
  const fileInput = $("fileInput");
  let rows = [];

  const SAMPLE =
    "lead_id,n_views,n_addtocart,n_unique_items,duration_sec,hour_of_day,day_of_week\n" +
    "lead-001,14,4,8,420,21,4\nlead-002,3,0,3,25,3,1\nlead-003,9,2,5,210,18,2\n" +
    "lead-004,22,1,12,650,12,5\nlead-005,2,0,1,8,2,6\nlead-006,7,3,4,180,20,3\n" +
    "lead-007,5,0,4,60,9,0\nlead-008,18,5,9,540,22,4\n";

  async function scoreCSV(text, filename) {
    const form = new FormData();
    form.append("file", new Blob([text], { type: "text/csv" }), filename || "leads.csv");
    const res = await fetch(API + "/score/batch", { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    rows = await res.json();
    populateSegments();
    renderTable();
    $("toolbar").hidden = false;
    $("tableWrap").hidden = false;
  }

  function populateSegments() {
    const segs = [...new Set(rows.map((r) => r.segment))].sort();
    const sel = $("segFilter");
    sel.innerHTML = '<option value="">todos</option>' +
      segs.map((s) => `<option>${s}</option>`).join("");
  }

  function renderTable() {
    const filter = $("segFilter").value;
    const view = filter ? rows.filter((r) => r.segment === filter) : rows;
    const tbody = $("tbody");
    tbody.innerHTML = "";
    view.forEach((r) => {
      const icon = BAND_ICON[r.label] || "";
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td class="semaphore">${icon}</td>` +
        `<td>${r.lead_id}</td>` +
        `<td class="prob"><span class="mini-bar"><span style="width:${r.probability * 100}%;background:${thermal(r.probability)}"></span></span> ${r.probability.toFixed(2)}</td>` +
        `<td>${r.label}</td>` +
        `<td><span class="seg-tag">${r.segment}</span></td>` +
        `<td>${r.recommended_action}</td>`;
      tbody.appendChild(tr);
    });
    $("count").textContent = `${view.length} de ${rows.length} leads`;
  }

  function downloadCSV() {
    const header = "lead_id,probability,label,segment,recommended_action\n";
    const body = rows
      .map((r) => `${r.lead_id},${r.probability},${r.label},${r.segment},"${r.recommended_action}"`)
      .join("\n");
    const url = URL.createObjectURL(new Blob([header + body], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url; a.download = "leads_scored.csv"; a.click();
    URL.revokeObjectURL(url);
  }

  function handleFile(file) {
    const reader = new FileReader();
    reader.onload = () => scoreCSV(reader.result, file.name).catch((e) => alert("Error: " + e.message));
    reader.readAsText(file);
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter") fileInput.click(); });
  fileInput.addEventListener("change", () => fileInput.files[0] && handleFile(fileInput.files[0]));
  ["dragover", "dragenter"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
  );
  dropzone.addEventListener("drop", (e) => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));
  $("segFilter").addEventListener("change", renderTable);
  $("download").addEventListener("click", downloadCSV);
  $("sample").addEventListener("click", () => scoreCSV(SAMPLE, "ejemplo.csv").catch((e) => alert(e.message)));
}

initSimulator();
initDashboard();
