let chart = null;          // Chart.js object (kept so we can destroy it before redrawing)
let history = [];          // every complaint routed in this browser session
const $ = id => document.getElementById(id);   // short helper for getElementById

// load sample complaints from the server
fetch("/api/samples").then(r => r.json()).then(d => {
  $("samples").innerHTML = d.samples.map(s => `<span>${s}</span>`).join("");
  document.querySelectorAll("#samples span").forEach(el =>
    el.onclick = () => { $("text").value = el.textContent; send(); });
});

async function send() {
  const text = $("text").value.trim();
  $("error").style.display = "none";
  if (!text) { showError("Please type a complaint."); return; }
  $("loading").style.display = "block";
  try {
    const res = await fetch("/predict", {method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify({text})});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    show(text, data);
  } catch (e) { showError(e.message); }
  $("loading").style.display = "none";
}

function showError(msg) { $("error").textContent = msg; $("error").style.display = "block"; }

function escapeHtml(s) { return s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

function show(text, d) {
  $("dept").textContent = d.department;
  $("conf").textContent = d.confidence;
  const u = $("urg"); u.textContent = d.urgency; u.className = "badge " + d.urgency;
  $("warn").style.display = d.low_confidence ? "block" : "none";

  // highlight urgent words inside the original text
  let html = escapeHtml(text);
  d.urgent_words.forEach(w => { html = html.replace(new RegExp("\\b(" + w + ")\\b", "gi"), "<mark>$1</mark>"); });
  $("highlight").innerHTML = html;

  // keywords that influenced the model
  $("keywords").innerHTML = d.keywords.length
    ? d.keywords.map(k => `<span class="kw">${k}</span>`).join("") : "<i>None found</i>";

  // bar chart of department scores (skipped if Chart.js could not load, e.g. offline)
  if (typeof Chart !== "undefined") {
    if (chart) chart.destroy();
    chart = new Chart($("chart"), {type: "bar",
      data: {labels: d.scores.map(s => s.dept),
             datasets: [{data: d.scores.map(s => s.pct), backgroundColor: "#1d4ed8"}]},
      options: {plugins: {legend: {display: false}}, scales: {y: {beginAtZero: true, max: 100}}}});
  }
  $("result").style.display = "block";

  // history table
  history.push({text, dept: d.department, urgency: d.urgency});
  $("history").querySelector("tbody").innerHTML = history.map((h, i) =>
    `<tr><td>${i + 1}</td><td>${escapeHtml(h.text)}</td><td>${h.dept}</td><td>${h.urgency}</td></tr>`).join("");
  $("historyCard").style.display = "block";
}

function downloadCsv() {   // export history as a CSV file
  const rows = ["complaint,department,urgency"].concat(
    history.map(h => `"${h.text.replace(/"/g, '""')}",${h.dept},${h.urgency}`));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.join("\n")], {type: "text/csv"}));
  a.download = "routed_complaints.csv"; a.click();
}

$("goBtn").onclick = send;
$("clearBtn").onclick = () => { $("text").value = ""; $("result").style.display = "none"; $("error").style.display = "none"; };
$("csvBtn").onclick = downloadCsv;
