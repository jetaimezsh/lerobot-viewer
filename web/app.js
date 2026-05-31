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
  editMode: "edit",
  mergePaths: [],   // source of truth for merge path list
  editOperations: [],
  trimDraftStart: null,
  trimDraftEnd: null,
  models: [],
  modelEnv: null,
  backtestResult: null,
  visibleBacktestModels: new Set(),
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
  modelOverview: document.getElementById("modelOverview"),
  pageTitle: document.getElementById("pageTitle"),
  statusPill: document.getElementById("statusPill"),
  episodeTitle: document.getElementById("episodeTitle"),
  episodeMeta: document.getElementById("episodeMeta"),
  videoGrid: document.getElementById("videoGrid"),
  videoCount: document.getElementById("videoCount"),
  prevEpisode: document.getElementById("prevEpisode"),
  nextEpisode: document.getElementById("nextEpisode"),
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
  modeEdit: document.getElementById("modeEdit"),
  modeExport: document.getElementById("modeExport"),
  modeEdit2: document.getElementById("modeEdit2"),
  modeExport2: document.getElementById("modeExport2"),
  setTrimStart: document.getElementById("setTrimStart"),
  setTrimEnd: document.getElementById("setTrimEnd"),
  markEpisode: document.getElementById("markEpisode"),
  markRange: document.getElementById("markRange"),
  sendEpisodeToBacktest: document.getElementById("sendEpisodeToBacktest"),
  trimDraft: document.getElementById("trimDraft"),
  editOperationList: document.getElementById("editOperationList"),
  editOperationBadge: document.getElementById("editOperationBadge"),
  checkEditTools: document.getElementById("checkEditTools"),
  strictValidateDataset: document.getElementById("strictValidateDataset"),
  fullSweep: document.getElementById("fullSweep"),
  runEditDryRun: document.getElementById("runEditDryRun"),
  applyEditPlan: document.getElementById("applyEditPlan"),
  editOutputPath: document.getElementById("editOutputPath"),
  editOverwrite: document.getElementById("editOverwrite"),
  editDryRunOutput: document.getElementById("editDryRunOutput"),
  toolStatusReport: document.getElementById("toolStatusReport"),
  addCurrentDatasetToMerge: document.getElementById("addCurrentDatasetToMerge"),
  clearMergeList: document.getElementById("clearMergeList"),
  validateMerge: document.getElementById("validateMerge"),
  applyMerge: document.getElementById("applyMerge"),
  mergePaths: document.getElementById("mergePaths"),
  mergeOutputPath: document.getElementById("mergeOutputPath"),
  mergeOverwrite: document.getElementById("mergeOverwrite"),
  mergeResult: document.getElementById("mergeResult"),
  mergePathTable: document.getElementById("mergePathTable"),
  addMergePathBtn: document.getElementById("addMergePathBtn"),
  folderBrowser: document.getElementById("folderBrowser"),
  folderBrowserClose: document.getElementById("folderBrowserClose"),
  folderBrowserUp: document.getElementById("folderBrowserUp"),
  folderBrowserPath: document.getElementById("folderBrowserPath"),
  folderBrowserList: document.getElementById("folderBrowserList"),
  folderBrowserSelect: document.getElementById("folderBrowserSelect"),
  folderBrowserCurrent: document.getElementById("folderBrowserCurrent"),
  checkModelEnv: document.getElementById("checkModelEnv"),
  refreshModels: document.getElementById("refreshModels"),
  modelName: document.getElementById("modelName"),
  checkpointPath: document.getElementById("checkpointPath"),
  browseCheckpoint: document.getElementById("browseCheckpoint"),
  modelAdapterType: document.getElementById("modelAdapterType"),
  modelScriptPath: document.getElementById("modelScriptPath"),
  modelDevice: document.getElementById("modelDevice"),
  registerModel: document.getElementById("registerModel"),
  modelEnvReport: document.getElementById("modelEnvReport"),
  modelList: document.getElementById("modelList"),
  backtestEpisodes: document.getElementById("backtestEpisodes"),
  limitBacktestFrames: document.getElementById("limitBacktestFrames"),
  backtestModelChoices: document.getElementById("backtestModelChoices"),
  runBacktest: document.getElementById("runBacktest"),
  clearBacktest: document.getElementById("clearBacktest"),
  backtestResult: document.getElementById("backtestResult"),
  backtestEpisodeSelect: document.getElementById("backtestEpisodeSelect"),
  backtestDimSelect: document.getElementById("backtestDimSelect"),
  showGroundTruth: document.getElementById("showGroundTruth"),
  showBacktestError: document.getElementById("showBacktestError"),
  backtestSeriesToggles: document.getElementById("backtestSeriesToggles"),
  backtestChart: document.getElementById("backtestChart"),
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
  if (viewId === "modelManagerView") {
    loadModelEnv();
    loadModels();
  }
  if (viewId === "modelBacktestView") {
    loadModels();
    requestAnimationFrame(drawBacktestChart);
  }
}

function goToDatasetEditor() {
  setView("datasetEditView");
}

function goToModelBacktest() {
  setView("modelBacktestView");
}

function currentEpisodeIndex() {
  return state.episode ? Number(state.episode.episode_index) : null;
}

function currentEpisodePosition() {
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) return -1;
  return state.episodes.findIndex((episode) => Number(episode.episode_index) === episodeIndex);
}

async function loadEnv() {
  const env = await api("/api/env");
  const rows = [
    ["Python", env.python],
    ["venv", env.venv ? "是" : "否"],
    ["prefix", env.prefix],
    ["conda", env.conda?.active ? "是" : "否"],
    ["conda env", env.conda?.env_name || "-"],
    ["conda prefix", env.conda?.prefix || "-"],
    ["conda command", env.conda?.command || env.conda?.exe || (env.conda?.available ? "可用" : "未检测到")],
    ["conda version", env.conda?.version || "-"],
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
    renderModelOverview();
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
      <div class="history-item-row" data-path="${escapeAttr(item.path)}">
        <button class="history-item">
          <strong>${escapeHtml(item.name || item.path)}</strong>
          <small>${escapeHtml(item.path)}</small>
          <span>${item.total_episodes ?? "-"} episodes · ${videos} views · ${escapeHtml(item.opened_at || "")}</span>
        </button>
        <button class="history-delete" data-path="${escapeAttr(item.path)}" title="删除记录">✕</button>
      </div>
    `;
  }).join("");
  for (const btn of els.historyList.querySelectorAll(".history-item")) {
    btn.addEventListener("click", () => openPath(btn.dataset.path));
  }
  for (const btn of els.historyList.querySelectorAll(".history-delete")) {
    btn.addEventListener("click", async (event) => {
      event.stopPropagation();
      try {
        await api("/api/history/delete", {
          method: "POST",
          body: JSON.stringify({ path: btn.dataset.path }),
        });
        await loadHistory();
      } catch (error) {
        console.error("删除历史记录失败:", error);
      }
    });
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

function renderModelOverview() {
  if (!els.modelOverview) return;
  if (!state.models.length) {
    els.modelOverview.classList.add("empty");
    els.modelOverview.innerHTML = "尚未注册模型。";
    return;
  }
  els.modelOverview.classList.remove("empty");
  const loaded = state.models.filter((model) => model.loaded).length;
  els.modelOverview.innerHTML = `
    <div class="model-overview-row">
      <strong>${state.models.length}</strong>
      <span>已注册模型</span>
    </div>
    <div class="model-overview-row">
      <strong>${loaded}</strong>
      <span>已加载模型</span>
    </div>
    ${state.models.slice(0, 3).map((model) => `
      <div class="model-overview-item">
        <strong>${escapeHtml(model.name)}</strong>
        <span>${escapeHtml(model.inspection?.policy_type || model.adapter_type)} · ${escapeHtml(model.status)}</span>
      </div>
    `).join("")}
  `;
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
  renderEpisodeNavigation();
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
  renderEpisodeNavigation();
  drawChart();
}

function renderEpisodeNavigation() {
  if (!els.prevEpisode || !els.nextEpisode) return;
  const position = currentEpisodePosition();
  const hasEpisode = position >= 0;
  els.prevEpisode.disabled = !hasEpisode || position === 0;
  els.nextEpisode.disabled = !hasEpisode || position >= state.episodes.length - 1;
}

async function loadAdjacentEpisode(delta) {
  const position = currentEpisodePosition();
  if (position < 0) return;
  const next = state.episodes[position + delta];
  if (!next) return;
  await loadEpisode(Number(next.episode_index));
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
  renderEpisodeNavigation();
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
  if (els.toolStatusReport) {
    els.toolStatusReport.classList.add("empty");
    els.toolStatusReport.textContent = "尚未检测数据修改工具。";
  }
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
  updateEditOperationBadge();
  refreshMarkButtons();
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

function currentEditMode() {
  return state.editMode;
}

function setEditMode(mode) {
  if (state.editMode === mode) return;
  state.editMode = mode;
  state.editOperations = [];
  renderEditOperations();
  updateEditOperationBadge();
  if (els.editDryRunOutput) els.editDryRunOutput.textContent = "模式已切换，已标记操作已清空。";
  refreshModeButtons();
  refreshMarkButtons();
}

function refreshModeButtons() {
  const active = state.editMode;
  for (const btn of [els.modeEdit, els.modeEdit2]) {
    if (btn) btn.classList.toggle("active", active === "edit");
  }
  for (const btn of [els.modeExport, els.modeExport2]) {
    if (btn) btn.classList.toggle("active", active === "export");
  }
}

function upsertEditOperation(operation) {
  state.editOperations = state.editOperations.filter((item) => item.episode_index !== operation.episode_index);
  state.editOperations.push(operation);
  renderEditOperations();
  if (els.editDryRunOutput) els.editDryRunOutput.textContent = "标记已变化，请重新运行预估。";
}

function renderEditOperations() {
  if (!els.editOperationList) return;
  if (!state.editOperations.length) {
    els.editOperationList.classList.add("empty");
    els.editOperationList.innerHTML = "暂无待应用修改";
    updateEditOperationBadge();
    return;
  }
  els.editOperationList.classList.remove("empty");
  els.editOperationList.innerHTML = state.editOperations.map((operation) => {
    let title, detail;
    switch (operation.type) {
      case "delete_episode":
        title = `删除 Episode ${operation.episode_index}`;
        detail = "应用后该 episode 会被移除，后续 episode 重新编号。";
        break;
      case "trim_episode":
        title = `裁剪 Episode ${operation.episode_index}`;
        detail = `保留 ${fmt(operation.start_time)}s - ${fmt(operation.end_time)}s。`;
        break;
      case "select_episode":
        title = `选择导出 Episode ${operation.episode_index}`;
        detail = "完整导出该 episode。";
        break;
      case "select_episode_range":
        title = `选择导出区间 Episode ${operation.episode_index}`;
        detail = `导出区间 ${fmt(operation.start_time)}s - ${fmt(operation.end_time)}s。`;
        break;
      default:
        title = `未知操作 ${operation.type} Episode ${operation.episode_index}`;
        detail = "";
    }
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
      refreshMarkButtons();
    });
  }
  updateEditOperationBadge();
}

function updateEditOperationBadge() {
  if (!els.editOperationBadge) return;
  const count = state.editOperations.length;
  const modeText = state.editMode === "export" ? "导出模式" : "删除模式";
  els.editOperationBadge.textContent = `${count} 个已标记${count > 0 ? " (" + modeText + ")" : ""}`;
}

function getEpisodeOpType() {
  return state.editMode === "export" ? "select_episode" : "delete_episode";
}

function getRangeOpType() {
  return state.editMode === "export" ? "select_episode_range" : "trim_episode";
}

function findEpisodeMark(episodeIndex) {
  return state.editOperations.find((op) => op.episode_index === episodeIndex);
}

function removeEpisodeMark(episodeIndex) {
  state.editOperations = state.editOperations.filter((op) => op.episode_index !== episodeIndex);
  renderEditOperations();
  if (els.editDryRunOutput) els.editDryRunOutput.textContent = "标记已变化，请重新运行预估。";
}

function markCurrentEpisode() {
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) return;
  const targetType = getEpisodeOpType();
  const existing = findEpisodeMark(episodeIndex);
  if (existing && existing.type === targetType) {
    removeEpisodeMark(episodeIndex);
    els.editEpisodeMeta.textContent = `已取消 Episode ${episodeIndex} 的标记。`;
    refreshMarkButtons();
    return;
  }
  upsertEditOperation({ type: targetType, episode_index: episodeIndex });
  const label = state.editMode === "export" ? "导出" : "删除";
  els.editEpisodeMeta.textContent = `已标记 Episode ${episodeIndex}（${label}模式）。`;
  refreshMarkButtons();
}

function setTrimPoint(kind) {
  if (!state.episode) return;
  if (kind === "start") state.trimDraftStart = state.currentElapsed;
  if (kind === "end") state.trimDraftEnd = state.currentElapsed;
  updateTrimDraftLabel();
}

function markCurrentRange() {
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) return;
  const start = state.trimDraftStart;
  const end = state.trimDraftEnd;
  if (start === null || end === null || end <= start) {
    els.editDryRunOutput.textContent = "区间无效：请先设置起点和终点，且终点必须大于起点。";
    return;
  }
  const targetType = getRangeOpType();
  const existing = findEpisodeMark(episodeIndex);
  const sameInterval = existing && existing.type === targetType
    && Math.abs((existing.start_time || 0) - start) < 1e-6
    && Math.abs((existing.end_time || 0) - end) < 1e-6;
  if (sameInterval) {
    removeEpisodeMark(episodeIndex);
    els.editEpisodeMeta.textContent = `已取消 Episode ${episodeIndex} 的区间标记。`;
    refreshMarkButtons();
    return;
  }
  upsertEditOperation({ type: targetType, episode_index: episodeIndex, start_time: start, end_time: end });
  const label = state.editMode === "export" ? "导出" : "裁剪";
  els.editEpisodeMeta.textContent = `已标记 Episode ${episodeIndex} 的区间（${label}模式）。`;
  refreshMarkButtons();
}

function refreshMarkButtons() {
  if (!els.markEpisode || !els.markRange) return;
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) {
    els.markEpisode.classList.remove("active", "mark-delete", "mark-export");
    els.markRange.classList.remove("active", "mark-delete", "mark-export");
    return;
  }
  const existing = findEpisodeMark(episodeIndex);
  const episodeType = getEpisodeOpType();
  const rangeType = getRangeOpType();
  const colorClass = state.editMode === "export" ? "mark-export" : "mark-delete";

  const epActive = !!(existing && existing.type === episodeType);
  els.markEpisode.classList.toggle("active", epActive);
  els.markEpisode.classList.toggle("mark-delete", epActive && colorClass === "mark-delete");
  els.markEpisode.classList.toggle("mark-export", epActive && colorClass === "mark-export");

  const rangeActive = !!(existing && existing.type === rangeType);
  els.markRange.classList.toggle("active", rangeActive);
  els.markRange.classList.toggle("mark-delete", rangeActive && colorClass === "mark-delete");
  els.markRange.classList.toggle("mark-export", rangeActive && colorClass === "mark-export");
}

function sendCurrentEpisodeToBacktest() {
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) return;
  els.backtestEpisodes.value = String(episodeIndex);
  goToModelBacktest();
}

async function runEditDryRun() {
  if (!state.summary) {
    els.editDryRunOutput.textContent = "请先加载数据集。";
    els.editDryRunOutput.classList.add("empty");
    return;
  }
  setEditOutputLoading("正在预估修改结果...");
  try {
    const result = await api("/api/edit/dry-run", {
      method: "POST",
      body: JSON.stringify({
        path: state.summary.root,
        operations: state.editOperations,
      }),
    });
    renderEditResult(result, "dry-run");
  } catch (error) {
    renderEditError(error.message);
  }
}

async function strictValidateCurrentDataset() {
  if (!state.summary) {
    els.editDryRunOutput.textContent = "请先加载数据集。";
    els.editDryRunOutput.classList.add("empty");
    return;
  }
  const fullSweep = els.fullSweep?.checked || false;
  const message = fullSweep ? "正在进行严格校验（含全量遍历）..." : "正在进行严格校验...";
  setEditOutputLoading(message);
  try {
    const result = await api("/api/datasets/strict-validate", {
      method: "POST",
      body: JSON.stringify({ path: state.summary.root, full_sweep: fullSweep }),
    });
    renderEditResult(result, "strict-validation");
  } catch (error) {
    renderEditError(error.message);
  }
}

async function checkEditTools() {
  els.toolStatusReport.textContent = "正在检测...";
  els.toolStatusReport.classList.remove("empty");
  try {
    const options = { method: "POST", body: "{}" };
    if (state.summary) {
      options.body = JSON.stringify({ path: state.summary.root });
    }
    const result = await api("/api/edit/tool-status", options);
    els.toolStatusReport.innerHTML = formatToolStatus(result);
  } catch (error) {
    els.toolStatusReport.textContent = error.message;
  }
}

function formatToolStatus(result) {
  const statusRows = [
    ["无视频数据编辑", result.ready_for_no_video_edits ? "可用" : "不可用"],
    ["含视频数据编辑", result.ready_for_video_edits ? "可用" : "不可用"],
  ];
  if (result.dataset) {
    statusRows.push(["当前数据集", result.dataset.path]);
    statusRows.push(["视频字段", result.dataset.has_video ? result.dataset.video_keys.join(", ") : "无"]);
    statusRows.push(["当前数据集落盘编辑", result.dataset.can_apply_now ? "可用" : "暂不可用"]);
    statusRows.push(["原因", result.dataset.reason]);
  }

  const missing = result.missing || [];
  const checks = result.checks || [];
  const capabilities = result.capabilities || [];
  const recommendations = result.recommendations || [];
  return `
    <div class="tool-report-section">
      <h4>总体状态</h4>
      ${statusRows.map(([label, value]) => `
        <div class="tool-report-row">
          <strong>${escapeHtml(label)}</strong>
          <span>${escapeHtml(value)}</span>
        </div>
      `).join("")}
    </div>
    <div class="tool-report-section">
      <h4>缺失项</h4>
      ${missing.length ? missing.map((item) => `
        <div class="tool-missing-item">
          <strong>${escapeHtml(item.name)}</strong>
          <span>影响：${escapeHtml(item.impact)}</span>
          <span>原因：${escapeHtml(item.reason)}</span>
          <span>解决：${escapeHtml(item.fix)}</span>
          <small>${escapeHtml(item.raw_detail)}</small>
        </div>
      `).join("") : "<div class=\"tool-ok\">没有缺失项。</div>"}
    </div>
    <div class="tool-report-section">
      <h4>已通过检查</h4>
      ${checks.filter((check) => check.ok).map((check) => `
        <div class="tool-check ok">
          <strong>${escapeHtml(check.label)}</strong>
          <span>${escapeHtml(check.detail)}</span>
        </div>
      `).join("") || "<div class=\"empty\">暂无通过项。</div>"}
    </div>
    <div class="tool-report-section">
      <h4>功能可用性</h4>
      ${capabilities.map((item) => `
        <div class="tool-check ${item.available ? "ok" : "fail"}">
          <strong>${escapeHtml(item.name)}</strong>
          <span>${item.available ? "可用" : `不可用，缺少：${escapeHtml((item.blocked_by || []).join(", "))}`}</span>
        </div>
      `).join("")}
    </div>
    <div class="tool-report-section">
      <h4>建议</h4>
      ${recommendations.map((item) => `<div class="tool-recommendation">${escapeHtml(item)}</div>`).join("")}
    </div>
  `;
}

function setEditOutputLoading(message) {
  els.editDryRunOutput.classList.remove("empty");
  els.editDryRunOutput.innerHTML = `<div class="result-loading">${escapeHtml(message)}</div>`;
}

function renderEditError(message) {
  els.editDryRunOutput.classList.remove("empty");
  els.editDryRunOutput.innerHTML = `
    <div class="result-section result-error">
      <h4>执行失败</h4>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function renderEditResult(result, mode) {
  els.editDryRunOutput.classList.remove("empty");
  if (mode === "strict-validation") {
    els.editDryRunOutput.innerHTML = formatValidationResult(result, "当前数据集严格校验");
    return;
  }
  const plan = result.dry_run || result;
  const title = mode === "apply" ? "生成新数据集结果" : "修改预估结果";
  const statusText = mode === "apply"
    ? (result.ok ? "成功" : "失败")
    : (plan.valid ? "可执行" : "不可执行");
  const statusClass = (mode === "apply" ? result.ok : plan.valid) ? "ok" : "fail";
  const outputPath = result.output_path ? `
    <div class="result-section">
      <h4>输出目录</h4>
      <div class="result-path">${escapeHtml(result.output_path)}</div>
    </div>
  ` : "";
  const summary = result.summary ? formatSummaryCards(result.summary) : "";
  const validation = result.validation ? formatValidationResult(result.validation, "输出数据集校验") : "";
  els.editDryRunOutput.innerHTML = `
    <div class="result-header">
      <div>
        <h4>${escapeHtml(title)}</h4>
        <p>${escapeHtml(result.path || state.summary?.root || "")}</p>
      </div>
      <span class="result-status ${statusClass}">${escapeHtml(statusText)}</span>
    </div>
    ${formatIssueList("错误", result.errors || plan.errors || [], "error")}
    ${formatIssueList("警告", result.warnings || plan.warnings || [], "warning")}
    ${formatPlanSummary(plan)}
    ${formatOperationSummary(plan.operations || [])}
    ${outputPath}
    ${summary}
    ${validation}
  `;
}

function formatPlanSummary(plan) {
  const original = plan.original || {};
  const predicted = plan.predicted || {};
  const rows = [
    ["原始 episodes", original.episodes],
    ["原始 frames", original.frames],
    ["预计 episodes", predicted.episodes],
    ["预计 frames", predicted.frames],
  ];
  if (predicted.selected_episodes !== undefined) {
    rows.push(["选择导出 episodes", predicted.selected_episodes]);
    rows.push(["选择导出区间", predicted.selected_ranges]);
  } else {
    rows.push(["删除 episodes", predicted.deleted_episodes]);
    rows.push(["裁剪 episodes", predicted.trimmed_episodes]);
  }
  rows.push(["需要处理视频", plan.requires_video_processing ? "是" : "否"]);
  return `
    <div class="result-section">
      <h4>影响范围</h4>
      ${formatKeyValueGrid(rows)}
    </div>
  `;
}

function formatOperationSummary(operations) {
  if (!operations.length) {
    return `
      <div class="result-section">
        <h4>操作明细</h4>
        <div class="result-empty">没有待应用操作。</div>
      </div>
    `;
  }
  return `
    <div class="result-section">
      <h4>操作明细</h4>
      <div class="result-operation-list">
        ${operations.map((operation) => {
          let title, detail;
          switch (operation.type) {
            case "delete_episode":
              title = `删除 Episode ${operation.episode_index}`;
              detail = "该 episode 会被移除，后续 episode 会重新编号。";
              break;
            case "trim_episode":
              title = `裁剪 Episode ${operation.episode_index}`;
              detail = `保留 ${fmt(operation.start_time)}s - ${fmt(operation.end_time)}s，共 ${operation.new_length ?? "-"} 帧。`;
              break;
            case "select_episode":
              title = `选择导出 Episode ${operation.episode_index}`;
              detail = "完整导出该 episode。";
              break;
            case "select_episode_range":
              title = `选择导出区间 Episode ${operation.episode_index}`;
              detail = `导出区间 ${fmt(operation.start_time)}s - ${fmt(operation.end_time)}s，共 ${operation.new_length ?? "-"} 帧。`;
              break;
            default:
              title = `${operation.type} Episode ${operation.episode_index}`;
              detail = "";
          }
          return `
            <div class="result-operation-item">
              <strong>${escapeHtml(title)}</strong>
              <span>${escapeHtml(detail)}</span>
            </div>
          `;
        }).join("")}
      </div>
    </div>
  `;
}

function formatValidationResult(result, title) {
  const statusClass = result.valid ? "ok" : "fail";
  const official = result.official || {};
  const officialText = official.status
    ? `${official.status}${official.reason ? `：${official.reason}` : ""}${official.error ? `：${official.error}` : ""}`
    : "未返回官方校验结果";
  const sweep = official.full_sweep || {};
  const sweepText = sweep.scanned !== undefined
    ? `全量遍历：${sweep.scanned} 帧 · ${sweep.elapsed_s}s ${sweep.errors?.length ? "· 发现 " + sweep.errors.length + " 处错误" : "· ✓"}`
    : "";
  return `
    <div class="result-header">
      <div>
        <h4>${escapeHtml(title)}</h4>
        <p>${escapeHtml(result.root || result.output_path || "")}</p>
      </div>
      <span class="result-status ${statusClass}">${result.valid ? "通过" : "失败"}</span>
    </div>
    ${formatIssueList("错误", result.errors || [], "error")}
    ${formatIssueList("警告", result.warnings || [], "warning")}
    ${sweepText ? `<div class="result-section"><h4>LeRobot 全量遍历</h4><div class="result-path">${escapeHtml(sweepText)}</div>${formatIssueList("遍历错误", sweep.errors || [], "error")}</div>` : ""}
    ${result.summary ? formatSummaryCards(result.summary) : ""}
    <div class="result-section">
      <h4>官方 LeRobot 校验</h4>
      <div class="result-path">${escapeHtml(officialText)}</div>
    </div>
  `;
}

function formatSummaryCards(summary) {
  const rows = Object.entries(summary).map(([key, value]) => [key, value]);
  return `
    <div class="result-section">
      <h4>摘要</h4>
      ${formatKeyValueGrid(rows)}
    </div>
  `;
}

function formatIssueList(title, items, type) {
  if (!items.length) return "";
  return `
    <div class="result-section ${type === "error" ? "result-error" : "result-warning"}">
      <h4>${escapeHtml(title)}</h4>
      <ul class="result-list">
        ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function formatKeyValueGrid(rows) {
  return `
    <div class="result-grid">
      ${rows
        .filter(([, value]) => value !== undefined && value !== null)
        .map(([key, value]) => `
          <div class="result-metric">
            <span>${escapeHtml(key)}</span>
            <strong>${escapeHtml(formatResultValue(value))}</strong>
          </div>
        `).join("")}
    </div>
  `;
}

function formatResultValue(value) {
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return Object.entries(value).map(([key, item]) => `${key}: ${item}`).join(", ");
  return value;
}

async function loadModelEnv() {
  if (!els.modelEnvReport) return;
  els.modelEnvReport.classList.remove("empty");
  els.modelEnvReport.textContent = "正在检测模型环境...";
  try {
    state.modelEnv = await api("/api/models/env");
    renderModelEnv();
  } catch (error) {
    els.modelEnvReport.textContent = error.message;
  }
}

function renderModelEnv() {
  const env = state.modelEnv;
  if (!env) return;
  const rows = [
    ["系统", env.os],
    ["Linux 推理", env.is_linux ? "支持" : "当前不支持"],
    ["LeRobot 回测", env.ready_for_lerobot_backtest ? "可运行" : "不可运行"],
    ["CUDA", env.cuda?.available ? `${env.cuda.device_count} 个设备` : "不可用"],
    ["缺失项", env.missing?.length ? env.missing.join(", ") : "无"],
  ];
  els.modelEnvReport.innerHTML = `
    <div class="model-env-grid">
      ${rows.map(([label, value]) => `
        <div class="model-env-cell">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `).join("")}
    </div>
    <div class="model-env-checks">
      ${(env.checks || []).map((check) => `
        <div class="tool-check ${check.ok ? "ok" : "fail"}">
          <strong>${escapeHtml(check.label)}</strong>
          <span>${escapeHtml(check.detail)}</span>
        </div>
      `).join("")}
    </div>
  `;
}

async function loadModels() {
  try {
    state.models = await api("/api/models");
    renderModels();
    renderModelOverview();
    renderBacktestModelChoices();
  } catch (error) {
    if (els.modelList) els.modelList.textContent = error.message;
  }
}

async function registerCurrentModel() {
  const checkpointPath = els.checkpointPath.value.trim();
  if (!checkpointPath) {
    els.modelList.textContent = "请填写 checkpoint 路径。";
    return;
  }
  try {
    await api("/api/models/register", {
      method: "POST",
      body: JSON.stringify({
        name: els.modelName.value.trim() || null,
        checkpoint_path: checkpointPath,
        adapter_type: els.modelAdapterType.value,
        device: els.modelDevice.value,
        script_path: els.modelScriptPath.value.trim() || null,
      }),
    });
    els.modelName.value = "";
    els.checkpointPath.value = "";
    els.modelScriptPath.value = "";
    await loadModels();
  } catch (error) {
    els.modelList.textContent = error.message;
  }
}

function renderModels() {
  if (!els.modelList) return;
  if (!state.models.length) {
    els.modelList.classList.add("empty");
    els.modelList.innerHTML = "尚未注册模型。";
    return;
  }
  els.modelList.classList.remove("empty");
  els.modelList.innerHTML = state.models.map((model) => {
    const inspection = model.inspection || {};
    const errors = inspection.errors || [];
    const warnings = inspection.warnings || [];
    return `
      <div class="model-card" data-model-id="${escapeAttr(model.id)}">
        <div class="model-card-head">
          <div>
            <strong>${escapeHtml(model.name)}</strong>
            <span>${escapeHtml(model.adapter_type)} · ${escapeHtml(model.device)} · ${escapeHtml(model.status)}</span>
          </div>
          <span class="result-status ${model.loaded ? "ok" : errors.length ? "fail" : "neutral"}">${model.loaded ? "已加载" : errors.length ? "无效" : "已注册"}</span>
        </div>
        <div class="model-card-path">${escapeHtml(model.checkpoint_path)}</div>
        <div class="model-card-meta">
          <span>policy: ${escapeHtml(inspection.policy_type || "-")}</span>
          <span>size: ${escapeHtml(inspection.size_mb ?? 0)} MB</span>
          <span>files: ${escapeHtml(inspection.file_count ?? 0)}</span>
          ${inspection.parameter_count ? `<span>params: ${escapeHtml(inspection.parameter_count)}</span>` : ""}
        </div>
        ${errors.length ? `<div class="model-card-issues error">${errors.map(escapeHtml).join("<br>")}</div>` : ""}
        ${warnings.length ? `<div class="model-card-issues warning">${warnings.map(escapeHtml).join("<br>")}</div>` : ""}
        <div class="model-card-actions">
          <button type="button" data-model-action="inspect">检查</button>
          <button type="button" data-model-action="load">加载</button>
          <button type="button" data-model-action="unload">卸载</button>
          <button type="button" data-model-action="delete">删除</button>
        </div>
      </div>
    `;
  }).join("");
}

async function handleModelAction(event) {
  const action = event.target?.dataset?.modelAction;
  if (!action) return;
  const card = event.target.closest(".model-card");
  const modelId = card?.dataset?.modelId;
  if (!modelId) return;
  const endpoint = {
    inspect: "/api/models/inspect",
    load: "/api/models/load",
    unload: "/api/models/unload",
    delete: "/api/models/delete",
  }[action];
  try {
    await api(endpoint, {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    });
    await loadModels();
  } catch (error) {
    card.insertAdjacentHTML("beforeend", `<div class="model-card-issues error">${escapeHtml(error.message)}</div>`);
  }
}

function renderBacktestModelChoices() {
  if (!els.backtestModelChoices) return;
  if (!state.models.length) {
    els.backtestModelChoices.classList.add("empty");
    els.backtestModelChoices.innerHTML = "请先在模型管理中注册模型。";
    return;
  }
  els.backtestModelChoices.classList.remove("empty");
  els.backtestModelChoices.innerHTML = state.models.map((model) => `
    <label class="model-choice">
      <input type="checkbox" value="${escapeAttr(model.id)}">
      <span>
        <strong>${escapeHtml(model.name)}</strong>
        <small>${escapeHtml(model.inspection?.policy_type || model.adapter_type)} · ${escapeHtml(model.status)}</small>
      </span>
    </label>
  `).join("");
}

function selectedBacktestModelIds() {
  return Array.from(els.backtestModelChoices.querySelectorAll("input:checked")).map((input) => input.value);
}

function parseEpisodeSelection(raw) {
  const result = new Set();
  for (const part of raw.split(/[\s,]+/).map((item) => item.trim()).filter(Boolean)) {
    if (part.includes("-")) {
      const [start, end] = part.split("-").map((item) => Number(item.trim()));
      if (!Number.isInteger(start) || !Number.isInteger(end) || end < start) throw new Error(`episode 范围无效: ${part}`);
      for (let index = start; index <= end; index += 1) result.add(index);
    } else {
      const value = Number(part);
      if (!Number.isInteger(value)) throw new Error(`episode 无效: ${part}`);
      result.add(value);
    }
  }
  return Array.from(result).sort((a, b) => a - b);
}

async function runSelectedBacktest() {
  if (!state.summary) {
    els.backtestResult.textContent = "请先加载数据集。";
    return;
  }
  const modelIds = selectedBacktestModelIds();
  if (!modelIds.length) {
    els.backtestResult.textContent = "请至少选择一个模型。";
    return;
  }
  let episodeIndexes;
  try {
    episodeIndexes = parseEpisodeSelection(els.backtestEpisodes.value.trim());
  } catch (error) {
    els.backtestResult.textContent = error.message;
    return;
  }
  if (!episodeIndexes.length) {
    els.backtestResult.textContent = "请填写要回测的 episode。";
    return;
  }
  els.backtestResult.classList.remove("empty");
  els.backtestResult.textContent = "正在运行回测...";
  try {
    state.backtestResult = await api("/api/backtests/run", {
      method: "POST",
      body: JSON.stringify({
        dataset_path: state.summary.root,
        model_ids: modelIds,
        episode_indexes: episodeIndexes,
        max_frames: els.limitBacktestFrames.checked ? 20 : null,
      }),
    });
    state.visibleBacktestModels = new Set(modelIds);
    renderBacktestResult();
  } catch (error) {
    els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>回测失败</h4><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function clearBacktestResult() {
  state.backtestResult = null;
  state.visibleBacktestModels = new Set();
  els.backtestResult.classList.add("empty");
  els.backtestResult.innerHTML = "尚未运行回测。";
  els.backtestEpisodeSelect.innerHTML = "";
  els.backtestDimSelect.innerHTML = "";
  els.backtestSeriesToggles.innerHTML = "";
  drawBacktestChart();
}

function renderBacktestResult() {
  const run = state.backtestResult;
  if (!run) return;
  const summary = run.summary || {};
  const rows = [
    ["组合数", summary.total],
    ["完成", summary.done],
    ["失败", summary.failed],
    ["平均 MAE", summary.mean_mae],
    ["平均 RMSE", summary.mean_rmse],
    ["最大误差", summary.max_error],
  ];
  els.backtestResult.classList.remove("empty");
  els.backtestResult.innerHTML = `
    <div class="result-header">
      <div>
        <h4>回测任务 ${escapeHtml(run.run_id)}</h4>
        <p>${escapeHtml(run.dataset_path)}</p>
      </div>
      <span class="result-status ${summary.failed ? "fail" : "ok"}">${summary.failed ? "部分失败" : "完成"}</span>
    </div>
    ${formatKeyValueGrid(rows)}
    <div class="backtest-matrix">
      ${(run.results || []).map((item) => `
        <div class="backtest-cell ${item.status === "done" ? "ok" : "fail"}">
          <strong>${escapeHtml(modelName(item.model_id))}</strong>
          <span>Episode ${escapeHtml(item.episode_index)} · ${escapeHtml(item.status)}</span>
          ${item.metrics ? `<small>MAE ${escapeHtml(item.metrics.mae)} · RMSE ${escapeHtml(item.metrics.rmse)}</small>` : `<small>${escapeHtml(item.error || "")}</small>`}
        </div>
      `).join("")}
    </div>
  `;
  populateBacktestChartControls();
  drawBacktestChart();
}

function populateBacktestChartControls() {
  const done = doneBacktestResults();
  const episodes = Array.from(new Set(done.map((item) => item.episode_index))).sort((a, b) => a - b);
  els.backtestEpisodeSelect.innerHTML = episodes.map((index) => `<option value="${index}">Episode ${index}</option>`).join("");
  const first = done[0];
  const dims = first?.series?.length || 0;
  els.backtestDimSelect.innerHTML = Array.from({ length: dims }, (_, index) => `<option value="${index}">action[${index}]</option>`).join("");
  els.backtestSeriesToggles.innerHTML = state.models.map((model) => `
    <label class="series-option">
      <input type="checkbox" value="${escapeAttr(model.id)}" ${state.visibleBacktestModels.has(model.id) ? "checked" : ""}>
      <span>${escapeHtml(model.name)}</span>
    </label>
  `).join("");
}

function doneBacktestResults() {
  return (state.backtestResult?.results || []).filter((item) => item.status === "done");
}

function selectedChartResults() {
  const episode = Number(els.backtestEpisodeSelect.value);
  const visible = new Set(Array.from(els.backtestSeriesToggles.querySelectorAll("input:checked")).map((input) => input.value));
  state.visibleBacktestModels = visible;
  return doneBacktestResults().filter((item) => item.episode_index === episode && visible.has(item.model_id));
}

function drawBacktestChart() {
  if (!els.backtestChart) return;
  const canvas = els.backtestChart;
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth || canvas.width;
  const height = canvas.clientHeight || canvas.height;
  canvas.width = width;
  canvas.height = height;
  ctx.clearRect(0, 0, width, height);
  const results = selectedChartResults();
  const dim = Number(els.backtestDimSelect.value || 0);
  if (!results.length || Number.isNaN(dim)) {
    ctx.fillStyle = "#5f6c72";
    ctx.font = "14px sans-serif";
    ctx.fillText("运行回测后选择 episode、action 维度和模型。", 24, 34);
    return;
  }
  const lines = [];
  if (els.showGroundTruth.checked && results[0].series?.[dim]) {
    lines.push({ name: "ground truth", values: results[0].series[dim].ground_truth, color: "#111827" });
  }
  const colors = ["#087f8c", "#b76e00", "#2f6fbb", "#7a5195", "#c92a2a", "#2f9e44"];
  results.forEach((item, index) => {
    const series = item.series?.[dim];
    if (!series) return;
    lines.push({ name: `${modelName(item.model_id)} predicted`, values: series.predicted, color: colors[index % colors.length] });
    if (els.showBacktestError.checked) {
      lines.push({ name: `${modelName(item.model_id)} error`, values: series.error, color: colors[(index + 2) % colors.length], dashed: true });
    }
  });
  drawLineChart(ctx, width, height, lines);
}

function drawLineChart(ctx, width, height, lines) {
  const pad = { left: 48, right: 18, top: 24, bottom: 38 };
  const values = lines.flatMap((line) => line.values.filter((value) => value !== null && value !== undefined));
  if (!values.length) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, width - pad.left - pad.right, height - pad.top - pad.bottom);
  for (const line of lines) {
    ctx.beginPath();
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2;
    ctx.setLineDash(line.dashed ? [6, 4] : []);
    line.values.forEach((value, index) => {
      const x = pad.left + (index / Math.max(line.values.length - 1, 1)) * (width - pad.left - pad.right);
      const y = pad.top + (1 - ((value - min) / span)) * (height - pad.top - pad.bottom);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = "#334155";
  ctx.font = "12px sans-serif";
  lines.slice(0, 8).forEach((line, index) => {
    ctx.fillStyle = line.color;
    ctx.fillText(line.name, pad.left + index * 130, height - 12);
  });
}

function modelName(modelId) {
  return state.models.find((model) => model.id === modelId)?.name || modelId;
}

async function applyEditPlan() {
  if (!state.summary) {
    els.editDryRunOutput.textContent = "请先加载数据集。";
    els.editDryRunOutput.classList.add("empty");
    return;
  }
  if (!state.editOperations.length) {
    els.editDryRunOutput.textContent = "没有待应用的编辑操作。";
    els.editDryRunOutput.classList.add("empty");
    return;
  }
  const outputPath = els.editOutputPath.value.trim();
  if (!outputPath) {
    els.editDryRunOutput.textContent = "请填写输出目录。";
    els.editDryRunOutput.classList.add("empty");
    return;
  }
  setEditOutputLoading("正在生成新数据集...");
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
    renderEditResult(result, "apply");
  } catch (error) {
    renderEditError(error.message);
  }
}

// state.mergePaths is the single source of truth.
// els.mergePaths (textarea) is for input only — never read as the
// authoritative list, always written to reflect state.mergePaths.

function mergePathList() {
  return state.mergePaths.slice();
}

function setMergePathList(paths) {
  state.mergePaths = Array.from(new Set(paths));
  renderMergePathTable();
  updateMergePathCount();
}

function addMergePath(path) {
  if (!path || !path.trim()) return 0;
  const incoming = path.split(/\r?\n/).map((p) => p.trim()).filter(Boolean);
  if (!incoming.length) return 0;
  let added = 0;
  const skipped = [];
  for (const p of incoming) {
    if (!state.mergePaths.includes(p)) {
      state.mergePaths.push(p);
      added++;
    } else {
      skipped.push(p);
    }
  }
  if (added) {
    renderMergePathTable();
    updateMergePathCount();
  }
  // Brief feedback in result area.
  const parts = [];
  if (added) parts.push(`已添加 ${added} 个路径`);
  if (skipped.length) parts.push(`跳过 ${skipped.length} 个重复`);
  if (parts.length && els.mergeResult) {
    els.mergeResult.classList.remove("empty");
    els.mergeResult.innerHTML = `<div class="result-loading">${parts.join("，")}</div>`;
  }
  return added;
}

function removeMergePath(path) {
  state.mergePaths = state.mergePaths.filter((p) => p !== path);
  renderMergePathTable();
  updateMergePathCount();
}

function updateMergePathCount() {
  const countEl = document.getElementById("mergePathCount");
  if (countEl) countEl.textContent = state.mergePaths.length ? `${state.mergePaths.length} 个数据集` : "";
}

function renderMergePathTable() {
  if (!els.mergePathTable) return;
  if (!state.mergePaths.length) {
    els.mergePathTable.classList.add("empty");
    els.mergePathTable.innerHTML = `<span class="merge-empty-hint">用上方按钮或下方输入框添加数据集路径</span>`;
    updateMergePathCount();
    return;
  }
  els.mergePathTable.classList.remove("empty");
  els.mergePathTable.innerHTML = state.mergePaths.map((p, i) => `
    <div class="merge-path-row" data-path="${escapeAttr(p)}">
      <span class="merge-path-index">${i + 1}</span>
      <span class="merge-path-text" title="${escapeAttr(p)}">${escapeHtml(p)}</span>
      <span class="merge-path-status" data-path="${escapeAttr(p)}">-</span>
      <button type="button" class="merge-path-remove" data-path="${escapeAttr(p)}">✕</button>
    </div>
  `).join("");
  for (const btn of els.mergePathTable.querySelectorAll(".merge-path-remove")) {
    btn.addEventListener("click", () => {
      removeMergePath(btn.dataset.path);
    });
  }
  updateMergePathCount();
}

function addCurrentDatasetToMerge() {
  if (!state.summary) {
    if (els.mergeResult) els.mergeResult.textContent = "请先加载数据集。";
    return;
  }
  addMergePath(state.summary.root);
}

function clearMergeList() {
  state.mergePaths = [];
  renderMergePathTable();
  resetMergeStatus();
}

// ── Folder Browser for merge path input ────────────────────────────────
function openFolderBrowser(targetInput, onSelect) {
  if (!els.folderBrowser) return;
  els.folderBrowser.style.display = "flex";
  // Store callback to use when user clicks "选择此目录".
  state._fbOnSelect = onSelect;
  state._fbTarget = targetInput;
  const startDir = (targetInput && targetInput.value) ? targetInput.value.trim() : "";
  navigateFolderBrowser(startDir);
}

function closeFolderBrowser() {
  if (!els.folderBrowser) return;
  els.folderBrowser.style.display = "none";
}

async function navigateFolderBrowser(dir) {
  if (!els.folderBrowserList || !els.folderBrowserCurrent) return;
  try {
    const result = await api(`/api/path/suggest?path=${encodeURIComponent(dir)}`);
    state._fbBase = result.base || dir;
    state._fbItems = result.items || [];
    els.folderBrowserCurrent.textContent = result.base || dir || "/";
    els.folderBrowserPath.value = "";
    if (!state._fbItems.length) {
      els.folderBrowserList.innerHTML = "<span class=\"fb-empty\">此目录下没有子文件夹</span>";
      return;
    }
    els.folderBrowserList.innerHTML = state._fbItems.map((item) => `
      <button class="fb-item" type="button" data-path="${escapeAttr(item.path)}">
        <span>📁 ${escapeHtml(item.name)}</span>
        ${item.has_dataset_marker ? "<strong>dataset</strong>" : ""}
      </button>
    `).join("");
    for (const btn of els.folderBrowserList.querySelectorAll(".fb-item")) {
      btn.addEventListener("click", () => navigateFolderBrowser(btn.dataset.path));
    }
  } catch (error) {
    els.folderBrowserList.innerHTML = `<span class="fb-empty">${escapeHtml(error.message)}</span>`;
  }
}

function resetMergeStatus() {
  for (const el of (els.mergePathTable?.querySelectorAll(".merge-path-status") || [])) {
    el.textContent = "-";
    el.className = "merge-path-status";
  }
  if (els.mergeResult) {
    els.mergeResult.classList.add("empty");
    els.mergeResult.innerHTML = "尚未检查合并。";
  }
}

function renderMergeResult(result) {
  if (!els.mergeResult) return;
  els.mergeResult.classList.remove("empty");
  const ok = result.ok;
  const allValid = result.dataset_validations
    ? result.dataset_validations.every((v) => v.valid)
    : true;
  const errors = result.errors || [];
  const warnings = result.warnings || [];
  els.mergeResult.innerHTML = `
    <div class="result-header">
      <div><h4>合并检查结果</h4></div>
      <span class="result-status ${ok && allValid ? "ok" : "fail"}">${ok && allValid ? "通过" : "失败"}</span>
    </div>
    ${formatIssueList("合并兼容性错误", errors, "error")}
    ${formatIssueList("合并兼容性警告", warnings, "warning")}
    ${result.dataset_validations ? renderDatasetValidationTable(result.dataset_validations) : ""}
    ${result.predicted ? formatMergePredicted(result.predicted) : ""}
    ${result.output_path ? `<div class="result-section"><h4>输出目录</h4><div class="result-path">${escapeHtml(result.output_path)}</div></div>` : ""}
    ${result.summary ? formatSummaryCards(result.summary) : ""}
    ${result.validation ? formatValidationResult(result.validation, "输出数据集校验") : ""}
  `;
}

function renderDatasetValidationTable(validations) {
  if (!validations || !validations.length) return "";
  return `
    <div class="result-section">
      <h4>各数据集严格校验</h4>
      <div class="merge-validation-table">
        ${validations.map((v) => `
          <div class="merge-validation-row ${v.valid ? "valid" : "invalid"}">
            <span class="merge-validation-path" title="${escapeAttr(v.path)}">${escapeHtml(v.path.split("/").pop() || v.path)}</span>
            <span class="merge-validation-status">${v.valid ? "✓" : "✗"}</span>
            <span class="merge-validation-detail">${v.valid ? `${v.summary?.total_episodes || 0} episodes` : (v.errors || [])[0] || "校验失败"}</span>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function formatMergePredicted(predicted) {
  return `
    <div class="result-section">
      <h4>合并预测</h4>
      <div class="result-grid">
        <div class="result-metric"><span>合并后 episodes</span><strong>${predicted.episodes}</strong></div>
        <div class="result-metric"><span>合并后 frames</span><strong>${predicted.frames}</strong></div>
        <div class="result-metric"><span>参与数据集</span><strong>${predicted.dataset_count}</strong></div>
      </div>
    </div>
  `;
}

function addMergeToResult(message) {
  if (!els.mergeResult) return;
  els.mergeResult.classList.remove("empty");
  els.mergeResult.innerHTML = `<div class="result-loading">${escapeHtml(message)}</div>`;
}

async function validateMergePlan() {
  const paths = mergePathList();
  if (paths.length < 2) {
    if (els.mergeResult) els.mergeResult.textContent = "至少需要 2 个数据集路径。";
    return;
  }

  addMergeToResult("正在逐数据集严格校验...");
  const datasetValidations = [];
  for (const path of paths) {
    setMergeStatus(path, "checking");
    try {
      const v = await api("/api/datasets/strict-validate", {
        method: "POST",
        body: JSON.stringify({ path }),
      });
      datasetValidations.push({ path, ...v });
      setMergeStatus(path, v.valid ? "valid" : "invalid", v);
    } catch (error) {
      datasetValidations.push({ path, valid: false, errors: [error.message] });
      setMergeStatus(path, "invalid");
    }
  }

  // Check if all individual validations passed
  const allValid = datasetValidations.every((v) => v.valid);
  if (!allValid) {
    els.mergeResult.innerHTML = renderMergeResult({
      ok: false,
      errors: ["部分数据集未通过严格校验，无法继续合并检查。"],
      dataset_validations: datasetValidations,
    });
    return;
  }

  // Now run merge compatibility check
  addMergeToResult("正在检查合并兼容性...");
  try {
    const result = await api("/api/merge/validate", {
      method: "POST",
      body: JSON.stringify({ paths }),
    });
    result.dataset_validations = datasetValidations;
    renderMergeResult(result);
    // Update status badges with episode counts
    for (const v of datasetValidations) {
      setMergeStatus(v.path, "valid", v);
    }
  } catch (error) {
    els.mergeResult.innerHTML = renderMergeResult({
      ok: false,
      errors: [error.message],
      dataset_validations: datasetValidations,
    });
  }
}

function setMergeStatus(path, status, validation) {
  if (!els.mergePathTable) return;
  const el = els.mergePathTable.querySelector(`.merge-path-status[data-path="${escapeAttr(path)}"]`);
  if (!el) return;
  el.className = `merge-path-status status-${status}`;
  if (status === "checking") {
    el.textContent = "...";
  } else if (status === "valid") {
    el.textContent = "✓ " + (validation?.summary?.total_episodes ?? "?") + " eps";
  } else {
    el.textContent = "✗";
  }
}

async function applyMergePlan() {
  const paths = mergePathList();
  if (paths.length < 2) {
    if (els.mergeResult) els.mergeResult.textContent = "至少需要 2 个数据集路径。";
    return;
  }
  const outputPath = (els.mergeOutputPath?.value || "").trim();
  if (!outputPath) {
    if (els.mergeResult) els.mergeResult.textContent = "请填写输出目录。";
    return;
  }
  addMergeToResult("正在生成合并数据集...");
  try {
    const result = await api("/api/merge/apply", {
      method: "POST",
      body: JSON.stringify({
        paths,
        output_path: outputPath,
        overwrite: els.mergeOverwrite?.checked || false,
      }),
    });
    renderMergeResult(result);
  } catch (error) {
    addMergeToResult(error.message);
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
els.prevEpisode.addEventListener("click", () => loadAdjacentEpisode(-1));
els.nextEpisode.addEventListener("click", () => loadAdjacentEpisode(1));
els.speed.addEventListener("change", () => {
  for (const video of state.videos) video.playbackRate = Number(els.speed.value);
});
els.timeSlider.addEventListener("input", () => setElapsed(Number(els.timeSlider.value)));
els.markEpisode.addEventListener("click", markCurrentEpisode);
els.markRange.addEventListener("click", markCurrentRange);
els.setTrimStart.addEventListener("click", () => setTrimPoint("start"));
els.setTrimEnd.addEventListener("click", () => setTrimPoint("end"));
els.sendEpisodeToBacktest.addEventListener("click", sendCurrentEpisodeToBacktest);
if (els.modeEdit) els.modeEdit.addEventListener("click", () => setEditMode("edit"));
if (els.modeExport) els.modeExport.addEventListener("click", () => setEditMode("export"));
if (els.modeEdit2) els.modeEdit2.addEventListener("click", () => setEditMode("edit"));
if (els.modeExport2) els.modeExport2.addEventListener("click", () => setEditMode("export"));
els.checkEditTools.addEventListener("click", checkEditTools);
els.strictValidateDataset.addEventListener("click", strictValidateCurrentDataset);
els.runEditDryRun.addEventListener("click", runEditDryRun);
els.applyEditPlan.addEventListener("click", applyEditPlan);
els.addCurrentDatasetToMerge.addEventListener("click", addCurrentDatasetToMerge);
if (els.addMergePathBtn) els.addMergePathBtn.addEventListener("click", () => {
  openFolderBrowser(els.mergePaths, (dir) => { addMergePath(dir); els.mergePaths.value = ""; });
});
els.clearMergeList.addEventListener("click", clearMergeList);
els.validateMerge.addEventListener("click", validateMergePlan);
els.applyMerge.addEventListener("click", applyMergePlan);

// Folder browser events
if (els.folderBrowserClose) els.folderBrowserClose.addEventListener("click", closeFolderBrowser);
if (els.folderBrowserSelect) els.folderBrowserSelect.addEventListener("click", () => {
  const dir = (state._fbBase || "").trim();
  if (dir) {
    if (state._fbOnSelect) {
      state._fbOnSelect(dir);
    } else {
      addMergePath(dir);
    }
  }
  closeFolderBrowser();
});
if (els.folderBrowserUp) els.folderBrowserUp.addEventListener("click", () => {
  let current = (state._fbBase || "").trim()
    .replace(/[\\/]+$/, "");
  let parent = "";
  if (current.length >= 2 && current[1] === ":" && current.length <= 3) {
    // Windows root, e.g. "D:" — keep it.
    // Up from "D:/foo" → "D:/".
  }
  const lastForward = current.lastIndexOf("/");
  const lastBack = current.lastIndexOf("\\");
  const idx = Math.max(lastForward, lastBack);
  if (idx > 0) {
    parent = current.substring(0, idx);
  } else if (idx === 0 && current.length > 1) {
    parent = "/"; // Unix root
  } else {
    parent = current; // already at root, stay
  }
  if (!parent && current.length >= 2 && current[1] === ":") {
    parent = current[0] + ":/";
  }
  navigateFolderBrowser(parent || current || "/");
});
if (els.folderBrowserPath) els.folderBrowserPath.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    navigateFolderBrowser(els.folderBrowserPath.value.trim());
  }
});
if (els.folderBrowser) els.folderBrowser.addEventListener("click", (event) => {
  if (event.target === els.folderBrowser) closeFolderBrowser();
});

// Enter key in merge textarea adds path, then clears input
if (els.mergePaths) els.mergePaths.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    const text = els.mergePaths.value.trim();
    if (text) addMergePath(text);
    els.mergePaths.value = "";
  }
});
els.checkModelEnv.addEventListener("click", loadModelEnv);
els.refreshModels.addEventListener("click", loadModels);
els.registerModel.addEventListener("click", registerCurrentModel);

// Checkpoint path: folder browser button
if (els.browseCheckpoint) els.browseCheckpoint.addEventListener("click", () => {
  openFolderBrowser(els.checkpointPath, (dir) => { els.checkpointPath.value = dir; });
});

// Checkpoint path: autocomplete via /api/path/suggest
if (els.checkpointPath) els.checkpointPath.addEventListener("input", async () => {
  clearTimeout(state._cpTimer);
  state._cpTimer = setTimeout(async () => {
    const val = els.checkpointPath.value.trim();
    if (!val) return;
    try {
      const result = await api(`/api/path/suggest?path=${encodeURIComponent(val)}`);
      if (result && result.base === val && result.items && result.items.length > 0) {
        // Show first match as a single suggestion below the input
        // (simple approach: just highlight, no dropdown for now)
      }
    } catch (_) {}
  }, 200);
});
els.modelList.addEventListener("click", handleModelAction);
els.runBacktest.addEventListener("click", runSelectedBacktest);
els.clearBacktest.addEventListener("click", clearBacktestResult);
els.backtestEpisodeSelect.addEventListener("change", drawBacktestChart);
els.backtestDimSelect.addEventListener("change", drawBacktestChart);
els.showGroundTruth.addEventListener("change", drawBacktestChart);
els.showBacktestError.addEventListener("change", drawBacktestChart);
els.backtestSeriesToggles.addEventListener("change", drawBacktestChart);
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
window.addEventListener("resize", drawBacktestChart);

loadEnv().catch((error) => {
  els.envInfo.textContent = error.message;
});
loadHistory().catch(() => {});
loadModels().catch(() => {});
