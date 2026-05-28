const state = {
  datasetId: null,
  summary: null,
  episodes: [],
  history: [],
  suggestions: [],
  suggestionTimer: null,
  episode: null,
  elapsed: [],
  series: [],
  selectedSeries: [],
  currentElapsed: 0,
  duration: 0,
  chartStart: 0,
  chartEnd: 0,
  panMode: false,
  playing: false,
  primaryVideo: null,
  videos: [],
  raf: null,
  currentView: "overviewView",
  editOperations: [],
  trimDraftStart: null,
  trimDraftEnd: null,
};

const els = {
  datasetPath: document.getElementById("datasetPath"),
  loadDataset: document.getElementById("loadDataset"),
  loadError: document.getElementById("loadError"),
  pathSuggestions: document.getElementById("pathSuggestions"),
  historyList: document.getElementById("historyList"),
  envInfo: document.getElementById("envInfo"),
  installRequirements: document.getElementById("installRequirements"),
  installOutput: document.getElementById("installOutput"),
  episodeList: document.getElementById("episodeList"),
  datasetSummary: document.getElementById("datasetSummary"),
  featureList: document.getElementById("featureList"),
  taskList: document.getElementById("taskList"),
  pageTitle: document.getElementById("pageTitle"),
  statusPill: document.getElementById("statusPill"),
  episodeTitle: document.getElementById("episodeTitle"),
  episodeMeta: document.getElementById("episodeMeta"),
  videoGrid: document.getElementById("videoGrid"),
  videoCount: document.getElementById("videoCount"),
  playPause: document.getElementById("playPause"),
  speed: document.getElementById("speed"),
  timeSlider: document.getElementById("timeSlider"),
  currentTime: document.getElementById("currentTime"),
  duration: document.getElementById("duration"),
  chart: document.getElementById("chart"),
  chartWindowLabel: document.getElementById("chartWindowLabel"),
  zoomIn: document.getElementById("zoomIn"),
  zoomOut: document.getElementById("zoomOut"),
  resetZoom: document.getElementById("resetZoom"),
  panMode: document.getElementById("panMode"),
  seriesDropdown: document.getElementById("seriesDropdown"),
  seriesToggle: document.getElementById("seriesToggle"),
  seriesSummary: document.getElementById("seriesSummary"),
  seriesMenu: document.getElementById("seriesMenu"),
  seriesOptions: document.getElementById("seriesOptions"),
  episodeInfo: document.getElementById("episodeInfo"),
  currentValues: document.getElementById("currentValues"),
  editEpisodeTitle: document.getElementById("editEpisodeTitle"),
  editEpisodeMeta: document.getElementById("editEpisodeMeta"),
  markDeleteEpisode: document.getElementById("markDeleteEpisode"),
  setTrimStart: document.getElementById("setTrimStart"),
  setTrimEnd: document.getElementById("setTrimEnd"),
  markTrimEpisode: document.getElementById("markTrimEpisode"),
  trimDraft: document.getElementById("trimDraft"),
  editOperationList: document.getElementById("editOperationList"),
  runEditDryRun: document.getElementById("runEditDryRun"),
  applyEditPlan: document.getElementById("applyEditPlan"),
  editOutputPath: document.getElementById("editOutputPath"),
  editOverwrite: document.getElementById("editOverwrite"),
  editDryRunOutput: document.getElementById("editDryRunOutput"),
};

const palette = ["#087f8c", "#b76e00", "#2f6fbb", "#7a5195", "#2f9e44", "#c92a2a", "#5f6c72", "#805ad5"];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const body = await response.json();
      message = body.detail || message;
      if (typeof message !== "string") message = JSON.stringify(message, null, 2);
    } catch (_) {
      message = await response.text();
    }
    throw new Error(message);
  }
  return response.json();
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  if (typeof value === "number") return value.toFixed(digits);
  return String(value);
}

function minChartWindow() {
  if (!state.duration) return 0.001;
  return Math.max(0.05, state.duration / 500);
}

function resetChartWindow() {
  state.chartStart = 0;
  state.chartEnd = Math.max(state.duration, 0);
  updateChartWindowLabel();
}

function chartSpan() {
  return Math.max(state.chartEnd - state.chartStart, minChartWindow());
}

function clampChartWindow() {
  const duration = Math.max(state.duration, 0);
  if (!duration) {
    state.chartStart = 0;
    state.chartEnd = 0;
    updateChartWindowLabel();
    return;
  }

  let span = Math.min(Math.max(chartSpan(), minChartWindow()), duration);
  if (state.chartStart < 0) state.chartStart = 0;
  if (state.chartStart + span > duration) state.chartStart = duration - span;
  state.chartEnd = state.chartStart + span;
  updateChartWindowLabel();
}

function updateChartWindowLabel() {
  if (!els.chartWindowLabel) return;
  els.chartWindowLabel.textContent = `${fmt(state.chartStart)}s - ${fmt(state.chartEnd)}s`;
}

function zoomChart(factor, anchorElapsed = null) {
  if (!state.duration) return;
  const currentSpan = chartSpan();
  const nextSpan = Math.min(Math.max(currentSpan * factor, minChartWindow()), state.duration);
  const anchor = anchorElapsed ?? (state.chartStart + currentSpan / 2);
  const ratio = currentSpan > 0 ? (anchor - state.chartStart) / currentSpan : 0.5;
  state.chartStart = anchor - nextSpan * Math.min(Math.max(ratio, 0), 1);
  state.chartEnd = state.chartStart + nextSpan;
  clampChartWindow();
  drawChart();
}

function panChart(deltaSeconds) {
  if (!state.duration) return;
  state.chartStart += deltaSeconds;
  state.chartEnd += deltaSeconds;
  clampChartWindow();
  drawChart();
}

function chartElapsedFromEvent(event) {
  const rect = els.chart.getBoundingClientRect();
  const pad = { left: 52, right: 18 };
  const x = Math.max(pad.left, Math.min(event.clientX - rect.left, rect.width - pad.right));
  const ratio = (x - pad.left) / Math.max(rect.width - pad.left - pad.right, 1);
  return state.chartStart + ratio * chartSpan();
}

function setView(viewId) {
  state.currentView = viewId;
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === viewId);
  });
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === viewId);
  });
  if (viewId === "episodeView") requestAnimationFrame(drawChart);
}

function currentEpisodeIndex() {
  return state.episode ? Number(state.episode.episode_index) : null;
}

async function loadEnv() {
  const env = await api("/api/env");
  const rows = [
    ["Python", env.python],
    ["venv", env.venv ? "是" : "否"],
    ["prefix", env.prefix],
    ["requirements", env.requirements],
  ];
  for (const [name, version] of Object.entries(env.packages)) {
    rows.push([name, version || "未安装"]);
  }
  els.envInfo.innerHTML = rows.map(([k, v]) => `
    <div class="env-row">
      <strong>${escapeHtml(k)}</strong><br>
      <span>${escapeHtml(v)}</span>
    </div>
  `).join("");
}

async function loadHistory() {
  state.history = await api("/api/history");
  renderHistory();
}

async function openDataset() {
  els.loadError.textContent = "";
  const path = els.datasetPath.value.trim();
  if (!path) {
    els.loadError.textContent = "请输入 dataset 根目录";
    return;
  }
  try {
    const summary = await api("/api/datasets/open", {
      method: "POST",
      body: JSON.stringify({ path }),
    });
    state.datasetId = summary.id;
    state.summary = summary;
    state.episodes = await api(`/api/datasets/${state.datasetId}/episodes`);
    state.episode = null;
    clearEditPlan();
    renderSummary();
    renderFeatures();
    renderTasks();
    renderEpisodes();
    resetEpisodeView();
    await loadHistory();
    setView("overviewView");
  } catch (error) {
    els.loadError.textContent = error.message;
  }
}

async function openPath(path) {
  els.datasetPath.value = path;
  hideSuggestions();
  await openDataset();
}

function renderHistory() {
  if (!state.history.length) {
    els.historyList.classList.add("empty");
    els.historyList.innerHTML = "暂无历史记录";
    return;
  }
  els.historyList.classList.remove("empty");
  els.historyList.innerHTML = state.history.map((item) => {
    const videos = Array.isArray(item.video_keys) ? item.video_keys.length : 0;
    return `
      <button class="history-item" data-path="${escapeAttr(item.path)}">
        <strong>${escapeHtml(item.name || item.path)}</strong>
        <small>${escapeHtml(item.path)}</small>
        <span>${item.total_episodes ?? "-"} episodes · ${videos} views · ${escapeHtml(item.opened_at || "")}</span>
      </button>
    `;
  }).join("");
  for (const item of els.historyList.querySelectorAll(".history-item")) {
    item.addEventListener("click", () => openPath(item.dataset.path));
  }
}

function schedulePathSuggestions() {
  clearTimeout(state.suggestionTimer);
  state.suggestionTimer = setTimeout(loadPathSuggestions, 160);
}

async function loadPathSuggestions() {
  const value = els.datasetPath.value.trim();
  try {
    const result = await api(`/api/path/suggest?path=${encodeURIComponent(value)}`);
    state.suggestions = result.items || [];
    renderPathSuggestions();
  } catch (_) {
    state.suggestions = [];
    hideSuggestions();
  }
}

function renderPathSuggestions() {
  if (!state.suggestions.length) {
    hideSuggestions();
    return;
  }
  els.pathSuggestions.innerHTML = state.suggestions.map((item) => `
    <button class="suggestion-item" type="button" data-path="${escapeAttr(item.path)}">
      <span>${escapeHtml(item.name)}</span>
      ${item.has_dataset_marker ? "<strong>dataset</strong>" : ""}
    </button>
  `).join("");
  els.pathSuggestions.classList.add("active");
  for (const item of els.pathSuggestions.querySelectorAll(".suggestion-item")) {
    item.addEventListener("mousedown", (event) => {
      event.preventDefault();
      els.datasetPath.value = item.dataset.path;
      hideSuggestions();
      if (item.querySelector("strong")) openDataset();
    });
  }
}

function hideSuggestions() {
  els.pathSuggestions.classList.remove("active");
  els.pathSuggestions.innerHTML = "";
}

function renderSummary() {
  const summary = state.summary;
  els.pageTitle.textContent = summary.root;
  els.statusPill.textContent = `${summary.total_episodes} episodes`;
  const metrics = [
    ["FPS", summary.fps],
    ["Episodes", summary.total_episodes],
    ["Frames", summary.total_frames],
    ["Tasks", summary.total_tasks],
    ["Video Keys", summary.video_keys.length],
    ["Numeric Keys", summary.numeric_keys.length],
    ["Version", summary.codebase_version || "v3.0"],
    ["Robot", summary.robot_type || "-"],
    ["data_path", summary.data_path],
  ];
  els.datasetSummary.innerHTML = metrics.map(([label, value]) => `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(value)}</div>
    </div>
  `).join("");
}

function renderFeatures() {
  const features = state.summary?.features || {};
  const videoKeys = new Set(state.summary?.video_keys || []);
  const numericKeys = new Set(state.summary?.numeric_keys || []);
  const entries = Object.entries(features);
  if (!entries.length) {
    els.featureList.innerHTML = "暂无特征信息";
    return;
  }
  els.featureList.innerHTML = entries.map(([key, feature]) => {
    const cls = videoKeys.has(key) ? "video" : numericKeys.has(key) ? "numeric" : "";
    const shape = feature.shape ? ` · [${feature.shape.join(", ")}]` : "";
    return `<div class="feature-chip ${cls}">${escapeHtml(key)} · ${escapeHtml(feature.dtype)}${escapeHtml(shape)}</div>`;
  }).join("");
}

function renderTasks() {
  const tasks = state.summary?.tasks || [];
  if (!tasks.length) {
    els.taskList.innerHTML = "暂无任务信息";
    return;
  }
  els.taskList.innerHTML = tasks.map((task) => {
    const taskIndex = task.task_index ?? task.index ?? "-";
    const text = task.task ?? task.name ?? JSON.stringify(task);
    return `<div class="task-row"><strong>${escapeHtml(taskIndex)}</strong> ${escapeHtml(text)}</div>`;
  }).join("");
}

function renderEpisodes() {
  if (!state.episodes.length) {
    els.episodeList.classList.add("empty");
    els.episodeList.innerHTML = "没有 episode";
    return;
  }
  els.episodeList.classList.remove("empty");
  els.episodeList.innerHTML = state.episodes.map((episode) => {
    const index = episode.episode_index;
    const length = episode.length ?? "-";
    const task = Array.isArray(episode.tasks) ? episode.tasks.join(", ") : "";
    const videoCount = Array.isArray(episode.videos) ? episode.videos.length : 0;
    return `
      <button class="episode-item" data-episode="${index}">
        <strong>Episode ${index}</strong>
        <small>${length} frames · ${videoCount} views${task ? ` · ${escapeHtml(task)}` : ""}</small>
      </button>
    `;
  }).join("");
  for (const item of els.episodeList.querySelectorAll(".episode-item")) {
    item.addEventListener("click", () => loadEpisode(Number(item.dataset.episode)));
  }
}

function resetEpisodeView() {
  pause();
  state.elapsed = [];
  state.series = [];
  state.selectedSeries = [];
  state.currentElapsed = 0;
  state.duration = 0;
  resetChartWindow();
  state.videos = [];
  state.primaryVideo = null;
  els.episodeTitle.textContent = "未选择 episode";
  els.episodeMeta.textContent = "从左侧 episode 列表选择一条记录。";
  els.videoGrid.innerHTML = "<div class=\"empty-state\">选择 episode 后，这里会显示该 episode 的所有视频视角。</div>";
  els.videoCount.textContent = "0 个视角";
  els.timeSlider.max = "0";
  els.timeSlider.value = "0";
  els.currentTime.textContent = "0.000s";
  els.duration.textContent = "0.000s";
  els.seriesOptions.innerHTML = "";
  els.seriesSummary.textContent = "选择要绘制的数据对象";
  els.episodeInfo.innerHTML = "选择 episode 后显示 episode 信息。";
  els.episodeInfo.classList.add("empty");
  els.currentValues.innerHTML = "选择 episode 后显示当前 state/action。";
  drawChart();
}

async function loadEpisode(index) {
  if (!state.datasetId) return;
  pause();
  const detail = await api(`/api/datasets/${state.datasetId}/episodes/${index}`);
  state.episode = detail.episode;
  state.elapsed = detail.timeline.elapsed || [];
  state.series = detail.series || [];
  state.selectedSeries = preferredSeries(state.series);
  const timelineDuration = state.elapsed.length ? state.elapsed[state.elapsed.length - 1] : 0;
  const videoDuration = Math.max(0, ...((detail.videos || []).map((item) => {
    if (item.from_timestamp === null || item.to_timestamp === null) return 0;
    return item.to_timestamp - item.from_timestamp;
  })));
  state.duration = Math.max(timelineDuration, videoDuration);
  state.currentElapsed = 0;
  resetChartWindow();

  for (const item of els.episodeList.querySelectorAll(".episode-item")) {
    item.classList.toggle("active", Number(item.dataset.episode) === index);
  }

  els.episodeTitle.textContent = `Episode ${index}`;
  els.episodeMeta.textContent = `${state.episode.length || state.elapsed.length} frames · ${detail.data_file}`;
  els.timeSlider.max = String(state.duration);
  els.timeSlider.value = "0";
  els.duration.textContent = `${fmt(state.duration)}s`;

  renderVideos(detail.videos || []);
  renderSeriesPicker();
  renderEpisodeInfo(detail);
  renderEditPanel();
  setView("episodeView");
  drawChart();
  updateCurrentValues();
}

function preferredSeries(series) {
  const priority = series.filter((item) => {
    const name = item.name.toLowerCase();
    return name.includes("action") || name.includes("state");
  });
  const selected = (priority.length ? priority : series).slice(0, Math.min(12, series.length));
  return selected.map((item) => item.name);
}

function renderVideos(videos) {
  els.videoGrid.innerHTML = "";
  els.videoCount.textContent = `${videos.length} 个视角`;
  state.videos = [];
  state.primaryVideo = null;
  for (const segment of videos) {
    const card = document.createElement("div");
    card.className = "video-card";
    const url = `/api/datasets/${state.datasetId}/video?episode_index=${state.episode.episode_index}&video_key=${encodeURIComponent(segment.key)}`;
    card.innerHTML = `
      <video preload="metadata" src="${url}"></video>
      <div class="video-label">
        <strong>${escapeHtml(segment.key)}</strong>
        <span>${fmt(segment.from_timestamp)}s - ${fmt(segment.to_timestamp)}s</span>
      </div>
    `;
    const video = card.querySelector("video");
    video.dataset.from = segment.from_timestamp || 0;
    video.dataset.to = segment.to_timestamp || 0;
    video.addEventListener("loadedmetadata", () => seekTo(state.currentElapsed));
    video.addEventListener("timeupdate", () => {
      if (video === state.primaryVideo && state.playing) {
        const elapsed = Math.max(0, video.currentTime - Number(video.dataset.from || 0));
        setElapsed(elapsed, false);
        if (elapsed >= state.duration - 0.005) pause();
      }
    });
    state.videos.push(video);
    if (!state.primaryVideo) state.primaryVideo = video;
    els.videoGrid.appendChild(card);
  }
  if (!videos.length) {
    els.videoGrid.innerHTML = "<div class=\"empty-state\">这个 episode 没有可用视频视角，但仍可查看数值型时序数据。</div>";
  }
}

function renderSeriesPicker() {
  els.seriesOptions.innerHTML = state.series.map((item) => {
    const checked = state.selectedSeries.includes(item.name) ? "checked" : "";
    return `
      <label class="series-option">
        <input type="checkbox" value="${escapeAttr(item.name)}" ${checked}>
        <span>${escapeHtml(item.name)}</span>
      </label>
    `;
  }).join("");
  for (const checkbox of els.seriesOptions.querySelectorAll("input[type='checkbox']")) {
    checkbox.addEventListener("change", () => {
      state.selectedSeries = Array.from(els.seriesOptions.querySelectorAll("input[type='checkbox']:checked"))
        .map((item) => item.value);
      updateSeriesSummary();
      updateCurrentValues();
      drawChart();
    });
  }
  updateSeriesSummary();
}

function updateSeriesSummary() {
  const count = state.selectedSeries.length;
  if (!state.series.length) {
    els.seriesSummary.textContent = "无可绘制数值字段";
  } else if (!count) {
    els.seriesSummary.textContent = "未选择数据对象";
  } else if (count === 1) {
    els.seriesSummary.textContent = state.selectedSeries[0];
  } else {
    els.seriesSummary.textContent = `已选择 ${count} / ${state.series.length} 个数据对象`;
  }
}

function setAllSeries(selected) {
  state.selectedSeries = selected ? state.series.map((item) => item.name) : [];
  for (const checkbox of els.seriesOptions.querySelectorAll("input[type='checkbox']")) {
    checkbox.checked = selected;
  }
  updateSeriesSummary();
  updateCurrentValues();
  drawChart();
}

function selectedSeries() {
  const selected = new Set(state.selectedSeries);
  return state.series.filter((item) => state.selectedSeries.includes(item.name));
}

function renderEpisodeInfo(detail) {
  const episode = detail.episode || {};
  const tasks = Array.isArray(episode.tasks) ? episode.tasks.join(", ") : episode.tasks;
  const videos = detail.videos || [];
  const rows = [
    ["episode", episode.episode_index],
    ["frames", episode.length || state.elapsed.length],
    ["duration", `${fmt(state.duration)}s`],
    ["task", tasks || "-"],
    ["data file", detail.data_file || "-"],
    ["dataset index", `${episode.dataset_from_index ?? "-"} → ${episode.dataset_to_index ?? "-"}`],
    ["data shard", `chunk ${episode["data/chunk_index"] ?? "-"} / file ${episode["data/file_index"] ?? "-"}`],
    ["video views", videos.length],
  ];
  for (const video of videos) {
    rows.push([video.key, `${fmt(video.from_timestamp)}s → ${fmt(video.to_timestamp)}s`]);
  }
  els.episodeInfo.classList.remove("empty");
  els.episodeInfo.innerHTML = rows.map(([label, value]) => `
    <div class="episode-info-row">
      <div class="episode-info-label">${escapeHtml(label)}</div>
      <div class="episode-info-value">${escapeHtml(value)}</div>
    </div>
  `).join("");
}

function drawChart() {
  const canvas = els.chart;
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(640, Math.floor(rect.width * dpr));
  canvas.height = Math.max(320, Math.floor(rect.height * dpr));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  const pad = { left: 52, right: 18, top: 20, bottom: 36 };
  ctx.clearRect(0, 0, width, height);

  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  ctx.fillStyle = "#fbfcfd";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#dde4eb";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + plotH * i / 5;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
  }

  clampChartWindow();
  const visibleStart = state.chartStart;
  const visibleEnd = state.chartEnd || Math.max(state.duration, 0.001);
  const visibleSpan = Math.max(visibleEnd - visibleStart, 0.001);
  const series = selectedSeries();

  if (!state.elapsed.length || !series.length) {
    ctx.fillStyle = "#64717f";
    ctx.font = "14px Segoe UI";
    ctx.fillText("选择 episode 和数值字段后显示时序图", pad.left, pad.top + 28);
  }

  series.forEach((item, index) => {
    const valid = item.values.filter((value) => value !== null && Number.isFinite(value));
    if (!valid.length) return;
    const min = Math.min(...valid);
    const max = Math.max(...valid);
    const span = Math.max(max - min, 1e-9);
    ctx.strokeStyle = palette[index % palette.length];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < state.elapsed.length; i++) {
      const elapsed = state.elapsed[i];
      if (elapsed < visibleStart || elapsed > visibleEnd) {
        started = false;
        continue;
      }
      const value = item.values[i];
      if (value === null || !Number.isFinite(value)) {
        started = false;
        continue;
      }
      const x = pad.left + ((elapsed - visibleStart) / visibleSpan) * plotW;
      const y = pad.top + (1 - (value - min) / span) * plotH;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();
  });

  if (state.currentElapsed >= visibleStart && state.currentElapsed <= visibleEnd) {
    const playheadX = pad.left + ((state.currentElapsed - visibleStart) / visibleSpan) * plotW;
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playheadX, pad.top);
    ctx.lineTo(playheadX, height - pad.bottom);
    ctx.stroke();
  }

  ctx.fillStyle = "#64717f";
  ctx.font = "12px Segoe UI";
  ctx.fillText(`${fmt(visibleStart)}s`, pad.left, height - 13);
  ctx.fillText(`${fmt(visibleEnd)}s`, Math.max(pad.left + 24, width - pad.right - 72), height - 13);
}

function currentFrameIndex() {
  if (!state.elapsed.length) return 0;
  let best = 0;
  let bestDiff = Infinity;
  for (let i = 0; i < state.elapsed.length; i++) {
    const diff = Math.abs(state.elapsed[i] - state.currentElapsed);
    if (diff < bestDiff) {
      best = i;
      bestDiff = diff;
    }
  }
  return best;
}

function updateCurrentValues() {
  const index = currentFrameIndex();
  const rows = selectedSeries().map((item) => {
    const value = item.values[index];
    return `
      <div class="value-row">
        <div class="value-name">${escapeHtml(item.name)}</div>
        <div class="value-number">${fmt(value, 5)}</div>
      </div>
    `;
  });
  els.currentValues.innerHTML = rows.join("") || "<div class=\"empty\">未选择数值字段。</div>";
  els.currentTime.textContent = `${fmt(state.currentElapsed)}s`;
  els.timeSlider.value = String(Math.min(state.currentElapsed, state.duration));
  updateTrimDraftLabel();
}

function clearEditPlan() {
  state.editOperations = [];
  state.trimDraftStart = null;
  state.trimDraftEnd = null;
  renderEditPanel();
  if (els.editDryRunOutput) els.editDryRunOutput.textContent = "尚未运行预估。";
}

function renderEditPanel() {
  if (!els.editEpisodeTitle) return;
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) {
    els.editEpisodeTitle.textContent = "未选择 episode";
    els.editEpisodeMeta.textContent = "在 Episode 播放页选择一条记录后，可以在这里添加编辑操作。";
  } else {
    els.editEpisodeTitle.textContent = `Episode ${episodeIndex}`;
    els.editEpisodeMeta.textContent = `${state.episode.length || state.elapsed.length} frames · 当前 ${fmt(state.currentElapsed)}s / ${fmt(state.duration)}s`;
  }
  updateTrimDraftLabel();
  renderEditOperations();
}

function updateTrimDraftLabel() {
  if (!els.trimDraft) return;
  const start = state.trimDraftStart;
  const end = state.trimDraftEnd;
  if (start === null && end === null) {
    els.trimDraft.textContent = "裁剪区间未设置";
  } else {
    els.trimDraft.textContent = `保留区间: ${start === null ? "未设置" : `${fmt(start)}s`} - ${end === null ? "未设置" : `${fmt(end)}s`}`;
  }
  if (els.editEpisodeMeta && state.episode) {
    els.editEpisodeMeta.textContent = `${state.episode.length || state.elapsed.length} frames · 当前 ${fmt(state.currentElapsed)}s / ${fmt(state.duration)}s`;
  }
}

function operationKey(operation) {
  return `${operation.type}:${operation.episode_index}`;
}

function upsertEditOperation(operation) {
  state.editOperations = state.editOperations.filter((item) => item.episode_index !== operation.episode_index);
  state.editOperations.push(operation);
  renderEditOperations();
  if (els.editDryRunOutput) els.editDryRunOutput.textContent = "编辑计划已变化，请重新运行预估。";
}

function renderEditOperations() {
  if (!els.editOperationList) return;
  if (!state.editOperations.length) {
    els.editOperationList.classList.add("empty");
    els.editOperationList.innerHTML = "暂无待应用修改";
    return;
  }
  els.editOperationList.classList.remove("empty");
  els.editOperationList.innerHTML = state.editOperations.map((operation) => {
    const title = operation.type === "delete_episode"
      ? `删除 Episode ${operation.episode_index}`
      : `裁剪 Episode ${operation.episode_index}`;
    const detail = operation.type === "delete_episode"
      ? "应用后该 episode 会被移除，后续 episode 重新编号。"
      : `保留 ${fmt(operation.start_time)}s - ${fmt(operation.end_time)}s。`;
    return `
      <div class="edit-operation-item" data-operation-key="${escapeAttr(operationKey(operation))}">
        <div>
          <strong>${escapeHtml(title)}</strong>
          <span>${escapeHtml(detail)}</span>
        </div>
        <button type="button" data-remove-operation="${escapeAttr(operationKey(operation))}">撤销</button>
      </div>
    `;
  }).join("");
  for (const button of els.editOperationList.querySelectorAll("[data-remove-operation]")) {
    button.addEventListener("click", () => {
      const key = button.dataset.removeOperation;
      state.editOperations = state.editOperations.filter((item) => operationKey(item) !== key);
      renderEditOperations();
      if (els.editDryRunOutput) els.editDryRunOutput.textContent = "编辑计划已变化，请重新运行预估。";
    });
  }
}

function markDeleteCurrentEpisode() {
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) return;
  upsertEditOperation({ type: "delete_episode", episode_index: episodeIndex });
}

function setTrimPoint(kind) {
  if (!state.episode) return;
  if (kind === "start") state.trimDraftStart = state.currentElapsed;
  if (kind === "end") state.trimDraftEnd = state.currentElapsed;
  updateTrimDraftLabel();
}

function markTrimCurrentEpisode() {
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) return;
  const start = state.trimDraftStart;
  const end = state.trimDraftEnd;
  if (start === null || end === null || end <= start) {
    els.editDryRunOutput.textContent = "裁剪区间无效：请先设置起点和终点，且终点必须大于起点。";
    return;
  }
  upsertEditOperation({
    type: "trim_episode",
    episode_index: episodeIndex,
    start_time: start,
    end_time: end,
  });
}

async function runEditDryRun() {
  if (!state.summary) {
    els.editDryRunOutput.textContent = "请先加载数据集。";
    return;
  }
  els.editDryRunOutput.textContent = "正在预估...";
  try {
    const result = await api("/api/edit/dry-run", {
      method: "POST",
      body: JSON.stringify({
        path: state.summary.root,
        operations: state.editOperations,
      }),
    });
    els.editDryRunOutput.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    els.editDryRunOutput.textContent = error.message;
  }
}

async function applyEditPlan() {
  if (!state.summary) {
    els.editDryRunOutput.textContent = "请先加载数据集。";
    return;
  }
  if (!state.editOperations.length) {
    els.editDryRunOutput.textContent = "没有待应用的编辑操作。";
    return;
  }
  const outputPath = els.editOutputPath.value.trim();
  if (!outputPath) {
    els.editDryRunOutput.textContent = "请填写输出目录。";
    return;
  }
  els.editDryRunOutput.textContent = "正在生成新数据集...";
  try {
    const result = await api("/api/edit/apply", {
      method: "POST",
      body: JSON.stringify({
        path: state.summary.root,
        output_path: outputPath,
        overwrite: els.editOverwrite.checked,
        operations: state.editOperations,
      }),
    });
    els.editDryRunOutput.textContent = JSON.stringify(result, null, 2);
  } catch (error) {
    els.editDryRunOutput.textContent = error.message;
  }
}

function setElapsed(elapsed, seekVideos = true) {
  state.currentElapsed = Math.max(0, Math.min(elapsed, state.duration));
  if (seekVideos) seekTo(state.currentElapsed);
  updateCurrentValues();
  drawChart();
}

function seekTo(elapsed) {
  for (const video of state.videos) {
    const start = Number(video.dataset.from || 0);
    const target = start + elapsed;
    if (Number.isFinite(video.duration) && Math.abs(video.currentTime - target) > 0.05) {
      video.currentTime = target;
    }
  }
}

async function play() {
  if (!state.videos.length) return;
  state.playing = true;
  els.playPause.textContent = "暂停";
  seekTo(state.currentElapsed);
  const rate = Number(els.speed.value);
  for (const video of state.videos) video.playbackRate = rate;
  await Promise.allSettled(state.videos.map((video) => video.play()));
  tickSync();
}

function pause() {
  state.playing = false;
  els.playPause.textContent = "播放";
  for (const video of state.videos) video.pause();
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = null;
}

function tickSync() {
  if (!state.playing) return;
  const primary = state.primaryVideo;
  if (primary) {
    const elapsed = Math.max(0, primary.currentTime - Number(primary.dataset.from || 0));
    for (const video of state.videos) {
      if (video === primary) continue;
      const target = Number(video.dataset.from || 0) + elapsed;
      if (Math.abs(video.currentTime - target) > 0.12) video.currentTime = target;
    }
  }
  state.raf = requestAnimationFrame(tickSync);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;",
  }[char]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/"/g, "&quot;");
}

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

els.loadDataset.addEventListener("click", openDataset);
els.datasetPath.addEventListener("keydown", (event) => {
  if (event.key === "Enter") openDataset();
});
els.datasetPath.addEventListener("input", schedulePathSuggestions);
els.datasetPath.addEventListener("focus", schedulePathSuggestions);
els.datasetPath.addEventListener("blur", () => setTimeout(hideSuggestions, 120));
els.installRequirements.addEventListener("click", async () => {
  els.installOutput.style.display = "block";
  els.installOutput.textContent = "正在安装...";
  try {
    const result = await api("/api/env/install-requirements", { method: "POST", body: "{}" });
    els.installOutput.textContent = `returncode=${result.returncode}\n\n${result.stdout}\n${result.stderr}`;
    await loadEnv();
  } catch (error) {
    els.installOutput.textContent = error.message;
  }
});
els.playPause.addEventListener("click", () => state.playing ? pause() : play());
els.speed.addEventListener("change", () => {
  for (const video of state.videos) video.playbackRate = Number(els.speed.value);
});
els.timeSlider.addEventListener("input", () => setElapsed(Number(els.timeSlider.value)));
els.markDeleteEpisode.addEventListener("click", markDeleteCurrentEpisode);
els.setTrimStart.addEventListener("click", () => setTrimPoint("start"));
els.setTrimEnd.addEventListener("click", () => setTrimPoint("end"));
els.markTrimEpisode.addEventListener("click", markTrimCurrentEpisode);
els.runEditDryRun.addEventListener("click", runEditDryRun);
els.applyEditPlan.addEventListener("click", applyEditPlan);
els.zoomIn.addEventListener("click", () => zoomChart(0.5));
els.zoomOut.addEventListener("click", () => zoomChart(2));
els.resetZoom.addEventListener("click", () => {
  resetChartWindow();
  drawChart();
});
els.panMode.addEventListener("click", () => {
  state.panMode = !state.panMode;
  els.panMode.classList.toggle("active", state.panMode);
  els.panMode.setAttribute("aria-pressed", String(state.panMode));
  els.chart.classList.toggle("pan-active", state.panMode);
});
els.seriesToggle.addEventListener("click", () => {
  els.seriesDropdown.classList.toggle("open");
});
els.seriesMenu.addEventListener("click", (event) => {
  event.stopPropagation();
  const action = event.target?.dataset?.seriesAction;
  if (action === "all") setAllSeries(true);
  if (action === "none") setAllSeries(false);
});
document.addEventListener("click", (event) => {
  if (!els.seriesDropdown.contains(event.target)) {
    els.seriesDropdown.classList.remove("open");
  }
});

let chartDragging = false;
let chartLastX = 0;
function chartSeek(event) {
  if (!state.duration) return;
  setElapsed(chartElapsedFromEvent(event));
}
els.chart.addEventListener("mousedown", (event) => {
  chartDragging = true;
  chartLastX = event.clientX;
  if (state.panMode || event.shiftKey) {
    event.preventDefault();
    return;
  }
  chartSeek(event);
});
window.addEventListener("mousemove", (event) => {
  if (!chartDragging) return;
  if (state.panMode || event.shiftKey) {
    const rect = els.chart.getBoundingClientRect();
    const deltaPixels = event.clientX - chartLastX;
    chartLastX = event.clientX;
    const secondsPerPixel = chartSpan() / Math.max(rect.width - 70, 1);
    panChart(-deltaPixels * secondsPerPixel);
  } else {
    chartSeek(event);
  }
});
window.addEventListener("mouseup", () => {
  chartDragging = false;
});
els.chart.addEventListener("wheel", (event) => {
  if (!state.duration) return;
  event.preventDefault();
  const factor = event.deltaY < 0 ? 0.8 : 1.25;
  zoomChart(factor, chartElapsedFromEvent(event));
}, { passive: false });
window.addEventListener("resize", drawChart);

loadEnv().catch((error) => {
  els.envInfo.textContent = error.message;
});
loadHistory().catch(() => {});
