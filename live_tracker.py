#!/usr/bin/env python3
"""
Live Training Tracker — polls HF Jobs logs and serves a real-time dashboard.

Usage:
    python live_tracker.py [JOB_ID]

Opens http://localhost:8787 in your browser with auto-updating charts.
If JOB_ID is omitted, picks the most recent RUNNING job.
"""
import json, os, re, subprocess, sys, threading, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

POLL_INTERVAL = 30  # seconds between log fetches
PORT = 8787

# ── State ────────────────────────────────────────────────────────────────
state = {
    "job_id": None,
    "job_name": None,
    "batches": [],      # [{batch_idx, y, epoch, batch, batch_total}]
    "epochs": [],       # [{batch_idx, epoch, train, val, lr, elapsed_s}]
    "status": "unknown",
    "last_update": 0,
    "error": None,
}
lock = threading.Lock()


def find_running_job():
    """Find the most recent RUNNING job via CLI."""
    try:
        out = subprocess.check_output(
            ["hf", "jobs", "list", "--all"], text=True, timeout=15)
        for line in out.splitlines():
            if "RUNNING" in line:
                parts = line.split()
                return parts[0]  # job_id is first column
    except Exception as e:
        print(f"  [warn] Could not list jobs: {e}")
    return None


def fetch_logs(job_id: str) -> str:
    """Fetch full logs for a job."""
    try:
        out = subprocess.check_output(
            ["hf", "jobs", "logs", f"tisaconundrum/{job_id}"],
            text=True, timeout=30, stderr=subprocess.STDOUT)
        return out
    except subprocess.CalledProcessError as e:
        return e.output or ""
    except Exception as e:
        return f"ERROR: {e}"


def parse_logs(raw: str):
    """Parse batch losses and epoch summaries from log text."""
    batches = []
    epochs = []
    batch_idx = 0
    current_epoch = 0
    job_name = None
    gpu_info = None
    model_info = {}

    for line in raw.splitlines():
        # Job header info
        if "GPU:" in line and "Tesla" in line:
            gpu_info = line.strip().split("GPU:")[1].strip() if "GPU:" in line else None
        if "Parameters:" in line:
            m = re.search(r'Parameters:\s+([\d,]+)', line)
            if m:
                model_info["params"] = m.group(1)
        if "Mode:" in line:
            model_info["mode"] = line.strip()

        # Batch loss
        m = re.search(r'batch\s+(\d+)/(\d+)\s+loss:\s+([\d.]+)', line)
        if m:
            batch_num = int(m.group(1))
            batch_total = int(m.group(2))
            loss = float(m.group(3))
            batches.append({
                "batch_idx": batch_idx,
                "y": loss,
                "epoch": current_epoch,
                "batch": batch_num,
                "batch_total": batch_total,
            })
            batch_idx += 1
            continue

        # Epoch summary
        m = re.search(
            r'Epoch\s+(\d+)/(\d+)\s+\|\s+train:\s+([\d.]+)\s+val:\s+([\d.]+)'
            r'\s+lr:\s+([\d.e+-]+)\s+\(([\d.]+)s',
            line)
        if m:
            epoch = int(m.group(1))
            total_epochs = int(m.group(2))
            current_epoch = epoch + 1
            epochs.append({
                "batch_idx": batch_idx - 1,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "train": float(m.group(3)),
                "val": float(m.group(4)),
                "lr": float(m.group(5)),
                "elapsed_s": float(m.group(6)),
            })
            continue

        # Checkpoint upload
        if "Checkpoint uploaded" in line:
            if epochs:
                epochs[-1]["checkpoint"] = True

        # Resumed info
        m = re.search(r'Resumed from checkpoint: epoch (\d+)', line)
        if m:
            model_info["resumed_from"] = int(m.group(1))

    return batches, epochs, model_info


def poll_loop():
    """Background thread that fetches logs periodically."""
    while True:
        job_id = state["job_id"]
        if not job_id:
            time.sleep(5)
            continue

        try:
            raw = fetch_logs(job_id)
            batches, epochs, model_info = parse_logs(raw)

            with lock:
                state["batches"] = batches
                state["epochs"] = epochs
                state["model_info"] = model_info
                state["last_update"] = time.time()
                state["error"] = None

                # Detect if job finished
                if "COMPLETED" in raw or "Application is shutting down" in raw:
                    state["status"] = "completed"
                elif "ERROR" in raw and "exit code" in raw:
                    state["status"] = "error"
                else:
                    state["status"] = "running"

        except Exception as e:
            with lock:
                state["error"] = str(e)

        time.sleep(POLL_INTERVAL)


# ── HTTP Server ──────────────────────────────────────────────────────────
DASHBOARD_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Live Training Tracker</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f0f23; color: #e0e0e0; margin: 0; padding: 20px; }
  h1 { text-align: center; color: #00d4ff; margin-bottom: 2px; font-size: 26px; }
  .subtitle { text-align: center; color: #666; margin-bottom: 20px; font-size: 13px; }
  .live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #66bb6a; margin-right: 6px; animation: pulse 2s infinite; }
  .live-dot.error { background: #ef5350; }
  .live-dot.completed { background: #ffd93d; animation: none; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .chart-container { max-width: 1300px; margin: 0 auto 25px; background: #1a1a2e; border-radius: 12px; padding: 20px; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
  .stats { display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; margin: 0 auto 25px; max-width: 1200px; }
  .stat { text-align: center; background: #1a1a2e; padding: 10px 18px; border-radius: 8px; min-width: 120px; }
  .stat-value { font-size: 20px; font-weight: bold; }
  .stat-label { font-size: 11px; color: #666; margin-top: 3px; }
  h2 { color: #ccc; font-size: 15px; margin: 20px auto 8px; max-width: 1300px; }
  .update-info { text-align: center; color: #555; font-size: 11px; margin-top: 10px; }
  canvas { max-height: 400px; }
</style>
</head>
<body>

<h1><span class="live-dot" id="liveDot"></span> Live Training Tracker</h1>
<div style="text-align:center;margin-bottom:12px">
  <input id="jobInput" type="text" placeholder="Job ID" style="padding:6px 12px;border-radius:6px;border:1px solid #444;background:#1a1a2e;color:#e0e0e0;font-size:13px;width:320px;font-family:monospace">
  <button onclick="switchJob()" style="padding:6px 14px;border-radius:6px;border:1px solid #444;background:#2d3748;color:#e0e0e0;cursor:pointer;font-size:13px;margin-left:4px">Switch</button>
</div>
<div class="subtitle" id="subtitle">Connecting...</div>

<div class="stats" id="statsRow"></div>

<h2>Batch Loss (log scale)</h2>
<div class="chart-container"><canvas id="batchChart"></canvas></div>

<h2>Batch Loss (linear scale)</h2>
<div class="chart-container"><canvas id="batchChartLinear"></canvas></div>

<h2>Epoch Summary (log scale)</h2>
<div class="chart-container"><canvas id="epochChart"></canvas></div>

<h2>Epoch Summary (linear scale)</h2>
<div class="chart-container"><canvas id="epochChartLinear"></canvas></div>

<div class="update-info" id="updateInfo"></div>

<script>
let batchChart = null;
let batchChartLinear = null;
let epochChart = null;
let epochChartLinear = null;
const POLL_MS = 5000;

function downsample(arr, maxPts) {
  const step = Math.max(1, Math.floor(arr.length / maxPts));
  return arr.filter((_, i) => i % step === 0 || i === arr.length - 1);
}

function ema(arr, alpha) {
  if (!arr.length) return [];
  let val = arr[0].y;
  return arr.map(d => {
    val = alpha * val + (1 - alpha) * d.y;
    return { x: d.batch_idx, y: val };
  });
}

function formatTime(s) {
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(1) + 'min';
  return (s/3600).toFixed(1) + 'h';
}

function updateDashboard(data) {
  const { batches, epochs, status, last_update, error, job_id, model_info } = data;

  // Status dot
  const dot = document.getElementById('liveDot');
  dot.className = 'live-dot' + (status === 'error' ? ' error' : status === 'completed' ? ' completed' : '');

  // Subtitle
  const sub = document.getElementById('subtitle');
  sub.textContent = `Job: ${job_id || '?'} | Status: ${status} | ${batches.length} batches logged`;

  // Stats
  const statsEl = document.getElementById('statsRow');
  const lastBatch = batches[batches.length - 1];
  const lastEpoch = epochs[epochs.length - 1];
  const progress = lastBatch ? `${lastBatch.batch}/${lastBatch.batch_total}` : '—';
  const currentEpoch = lastEpoch ? `${lastEpoch.epoch}/${lastEpoch.total_epochs}` : (lastBatch ? `${lastBatch.epoch}` : '—');
  const batchLoss = lastBatch ? lastBatch.y.toFixed(4) : '—';
  const trainLoss = lastEpoch ? lastEpoch.train.toFixed(4) : '—';
  const valLoss = lastEpoch ? lastEpoch.val.toFixed(4) : '—';
  const lr = lastEpoch ? lastEpoch.lr.toExponential(2) : '—';
  const epochTime = lastEpoch ? formatTime(lastEpoch.elapsed_s) : '—';
  const totalEpochs = epochs.length;

  // ETA
  let eta = '—';
  if (lastEpoch && lastBatch) {
    const remainEpochs = lastEpoch.total_epochs - lastEpoch.epoch;
    // Remaining batches in current epoch
    const batchesLeft = lastBatch.batch_total - lastBatch.batch;
    const secPerBatch = lastEpoch.elapsed_s / lastBatch.batch_total;
    const remainSec = (remainEpochs * lastEpoch.elapsed_s) + (batchesLeft * secPerBatch);
    eta = formatTime(remainSec);
  }

  statsEl.innerHTML = `
    <div class="stat"><div class="stat-value" style="color:#4fc3f7">${currentEpoch}</div><div class="stat-label">Epoch</div></div>
    <div class="stat"><div class="stat-value" style="color:#81c784">${progress}</div><div class="stat-label">Batch</div></div>
    <div class="stat"><div class="stat-value" style="color:#ce93d8">${batchLoss}</div><div class="stat-label">Batch Loss</div></div>
    <div class="stat"><div class="stat-value" style="color:#66bb6a">${trainLoss}</div><div class="stat-label">Train Loss</div></div>
    <div class="stat"><div class="stat-value" style="color:#ff6b6b">${valLoss}</div><div class="stat-label">Val Loss</div></div>
    <div class="stat"><div class="stat-value" style="color:#ffd93d">${lr}</div><div class="stat-label">LR</div></div>
    <div class="stat"><div class="stat-value" style="color:#90a4ae">${epochTime}</div><div class="stat-label">Epoch Time</div></div>
    <div class="stat"><div class="stat-value" style="color:#90a4ae">${eta}</div><div class="stat-label">ETA</div></div>
  `;

  // Prepare data
  const down = downsample(batches, 2000);
  const smooth = ema(down, 0.96);
  const valMarkers = epochs.map(e => ({ x: e.batch_idx, y: e.val }));
  const rawPts = down.map(d => ({x: d.batch_idx, y: d.y}));

  const epochAnns = {};
  epochs.forEach((e, i) => {
    epochAnns['ep' + i] = {
      type: 'line', xMin: e.batch_idx, xMax: e.batch_idx,
      borderColor: e.checkpoint ? 'rgba(102,187,106,0.3)' : 'rgba(255,255,255,0.08)',
      borderWidth: e.checkpoint ? 2 : 1, borderDash: [3, 3],
      label: e.checkpoint ? {
        display: true, content: '✓ E' + e.epoch,
        position: 'start', backgroundColor: 'rgba(102,187,106,0.7)',
        color: '#fff', font: { size: 10 }
      } : undefined
    };
  });

  const tooltipCb = { label: ctx => ctx.dataset.label + ': ' + ctx.raw.y.toFixed(4) };
  const batchScalesBase = {
    x: { title: { display: true, text: 'Batch', color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } },
  };

  // Helper: create or update a batch-style chart
  function upsertBatchChart(chartVar, canvasId, yType) {
    if (chartVar) {
      chartVar.data.datasets[0].data = rawPts;
      chartVar.data.datasets[1].data = smooth;
      chartVar.data.datasets[2].data = valMarkers;
      chartVar.options.plugins.annotation.annotations = epochAnns;
      chartVar.update('none');
      return chartVar;
    }
    return new Chart(document.getElementById(canvasId).getContext('2d'), {
      type: 'scatter',
      data: {
        datasets: [
          { label: 'Batch Loss (raw)', data: rawPts, pointRadius: 0.6, pointBackgroundColor: 'rgba(206,147,216,0.12)', showLine: false, order: 3 },
          { label: 'Batch Loss (smoothed)', data: smooth, pointRadius: 0, showLine: true, borderColor: '#ce93d8', borderWidth: 2.5, tension: 0.1, order: 1 },
          { label: 'Validation', data: valMarkers, pointRadius: 7, pointBackgroundColor: '#ff6b6b', pointBorderColor: '#fff', pointBorderWidth: 2, pointStyle: 'triangle', showLine: true, borderColor: 'rgba(255,107,107,0.4)', borderWidth: 2, borderDash: [5,3], order: 0 }
        ]
      },
      options: {
        responsive: true, animation: false,
        interaction: { mode: 'nearest', intersect: false, axis: 'x' },
        plugins: {
          legend: { labels: { color: '#ccc', usePointStyle: true, font: { size: 11 } }, position: 'bottom' },
          tooltip: { callbacks: tooltipCb },
          annotation: { annotations: epochAnns }
        },
        scales: {
          ...batchScalesBase,
          y: { title: { display: true, text: 'Loss', color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' }, ...(yType === 'log' ? {type:'logarithmic'} : {min:0}) }
        }
      }
    });
  }

  batchChart = upsertBatchChart(batchChart, 'batchChart', 'log');
  batchChartLinear = upsertBatchChart(batchChartLinear, 'batchChartLinear', 'linear');

  // Epoch charts (log + linear)
  if (epochs.length > 0) {
    const trainPts = epochs.map(e => ({x: e.epoch, y: e.train}));
    const valPts = epochs.map(e => ({x: e.epoch, y: e.val}));
    const lrPts = epochs.map(e => ({x: e.epoch, y: e.lr * 1000}));

    function upsertEpochChart(chartVar, canvasId, yType) {
      if (chartVar) {
        chartVar.data.datasets[0].data = trainPts;
        chartVar.data.datasets[1].data = valPts;
        chartVar.data.datasets[2].data = lrPts;
        chartVar.update('none');
        return chartVar;
      }
      return new Chart(document.getElementById(canvasId).getContext('2d'), {
        type: 'scatter',
        data: {
          datasets: [
            { label: 'Train Loss', data: trainPts, borderColor: '#ce93d8', backgroundColor: '#ce93d8', borderWidth: 2, pointRadius: 6, showLine: true, tension: 0.2 },
            { label: 'Val Loss', data: valPts, borderColor: '#ff6b6b', backgroundColor: '#ff6b6b', borderWidth: 2, borderDash: [6,4], pointRadius: 6, pointStyle: 'triangle', showLine: true, tension: 0.2 },
            { label: 'Learning Rate (×1000)', data: lrPts, borderColor: '#ffd93d', backgroundColor: '#ffd93d', borderWidth: 1.5, borderDash: [3,3], pointRadius: 3, showLine: true, tension: 0.2, yAxisID: 'y2' }
          ]
        },
        options: {
          responsive: true, animation: false,
          plugins: { legend: { labels: { color: '#ccc', usePointStyle: true }, position: 'bottom' } },
          scales: {
            x: { title: { display: true, text: 'Epoch', color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' } },
            y: { title: { display: true, text: 'Loss', color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#888' }, position: 'left', ...(yType === 'log' ? {type:'logarithmic'} : {min:0}) },
            y2: { title: { display: true, text: 'LR ×1000', color: '#ffd93d' }, grid: { drawOnChartArea: false }, ticks: { color: '#ffd93d' }, position: 'right' }
          }
        }
      });
    }

    epochChart = upsertEpochChart(epochChart, 'epochChart', 'log');
    epochChartLinear = upsertEpochChart(epochChartLinear, 'epochChartLinear', 'linear');
  }

  // Update info (no DOM reflow — just textContent)
  const ago = last_update ? Math.round((Date.now()/1000 - last_update)) + 's ago' : 'never';
  document.getElementById('updateInfo').textContent =
    `Last poll: ${ago} | Polling every POLL_INTERVAL_PLACEHOLDERs | ${error ? 'Error: ' + error : 'OK'}`;
}

async function poll() {
  try {
    const resp = await fetch('/api/data');
    const data = await resp.json();
    updateDashboard(data);
  } catch (e) {
    document.getElementById('updateInfo').textContent = 'Fetch error: ' + e;
  }
}

// Load job ID from URL ?job= param
const params = new URLSearchParams(window.location.search);
if (params.has('job')) {
  fetch('/api/set_job?job=' + encodeURIComponent(params.get('job')))
    .then(() => poll());
} else {
  poll();
}
setInterval(poll, POLL_MS);

function switchJob() {
  const id = document.getElementById('jobInput').value.trim();
  if (!id) return;
  const url = new URL(window.location);
  url.searchParams.set('job', id);
  window.history.replaceState({}, '', url);
  fetch('/api/set_job?job=' + encodeURIComponent(id))
    .then(() => { batchChart=null; batchChartLinear=null; epochChart=null; epochChartLinear=null;
      document.getElementById('batchChart').getContext('2d').clearRect(0,0,9999,9999);
      document.getElementById('batchChartLinear').getContext('2d').clearRect(0,0,9999,9999);
      document.getElementById('epochChart').getContext('2d').clearRect(0,0,9999,9999);
      document.getElementById('epochChartLinear').getContext('2d').clearRect(0,0,9999,9999);
      poll();
    });
}

// Pre-fill input from current state
setTimeout(async () => {
  const r = await fetch('/api/data');
  const d = await r.json();
  if (d.job_id) document.getElementById('jobInput').value = d.job_id;
}, 500);
</script>
</body>
</html>'''


class TrackerHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            html = DASHBOARD_HTML.replace("POLL_INTERVAL_PLACEHOLDER", str(POLL_INTERVAL))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
            return

        if parsed.path == "/api/data":
            with lock:
                payload = json.dumps(state)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload.encode())
            return

        if parsed.path == "/api/set_job":
            params = parse_qs(parsed.query)
            new_id = params.get("job", [None])[0]
            if new_id:
                with lock:
                    state["job_id"] = new_id
                    state["batches"] = []
                    state["epochs"] = []
                    state["model_info"] = {}
                    state["status"] = "switching"
                    state["error"] = None
                print(f"  Switched to job: {new_id}", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "job_id": new_id}).encode())
            return

        self.send_error(404)

    def log_message(self, format, *args):
        pass  # silence request logs


def main():
    job_id = sys.argv[1] if len(sys.argv) > 1 else None

    if not job_id:
        print("  No job ID provided, looking for running jobs...")
        job_id = find_running_job()

    if not job_id:
        print("  No job ID provided. Use the browser UI to set one, or pass as argument.")
        print("  Starting server anyway...")

    state["job_id"] = job_id
    print(f"  Tracking job: {job_id}")
    print(f"  Polling every {POLL_INTERVAL}s")

    # Initial fetch
    if job_id:
        print("  Fetching initial logs...")
        raw = fetch_logs(job_id)
        batches, epochs, model_info = parse_logs(raw)
        state["batches"] = batches
        state["epochs"] = epochs
        state["model_info"] = model_info
        state["last_update"] = time.time()
        state["status"] = "running"
        print(f"  Loaded {len(batches)} batches, {len(epochs)} epochs")

    # Start background poller
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()

    # Start HTTP server (SO_REUSEADDR avoids "Address already in use" after restart)
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("127.0.0.1", PORT), TrackerHandler)
    print(f"\n  Dashboard: http://localhost:{PORT}")
    print(f"  Press Ctrl+C to stop\n")

    # Open browser
    import webbrowser
    webbrowser.open(f"http://localhost:{PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
