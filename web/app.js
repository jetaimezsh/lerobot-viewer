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
  currentRoot: "data",
  currentView: "overviewView",
  editMode: "edit",
  mergePaths: [],   // source of truth for merge path list
  editOperations: [],
  trimDraftStart: null,
  trimDraftEnd: null,
  models: [],
  profileTemplates: [],
  selectedProfileId: null,
  modelEnv: null,
  backtestEpisodes: [],
  backtestResult: null,
  visibleBacktestModels: new Set(),
  visibleBacktestDims: new Set(),
  backtestDimsInitialized: false,
  pollingBacktestJobs: new Set(),
  displayedBacktestJobId: null,
  backtestChartStart: 0,
  backtestChartEnd: null,
  trainingFrameworks: [],
  trainingTemplates: [],
  trainingRecipes: [],
  selectedTrainingRecipeId: null,
  trainingJobs: [],
  trainingPipelines: [],
};

const ACTIVE_BACKTEST_JOB_KEY = "lerobotViewer.activeBacktestJobId";
const BACKTEST_ENV_VARS_KEY = "lerobotViewer.backtestEnvVars";

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
  episodeSideSection: document.getElementById("episodeSideSection"),
  datasetSummary: document.getElementById("datasetSummary"),
  featureList: document.getElementById("featureList"),
  taskList: document.getElementById("taskList"),
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
  loadOperationLogs: document.getElementById("loadOperationLogs"),
  strictValidateDataset: document.getElementById("strictValidateDataset"),
  fullSweep: document.getElementById("fullSweep"),
  runEditDryRun: document.getElementById("runEditDryRun"),
  applyEditPlan: document.getElementById("applyEditPlan"),
  editOutputPath: document.getElementById("editOutputPath"),
  editOverwrite: document.getElementById("editOverwrite"),
  editDryRunOutput: document.getElementById("editDryRunOutput"),
  toolStatusReport: document.getElementById("toolStatusReport"),
  operationLogReport: document.getElementById("operationLogReport"),
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
  browseRoot: document.getElementById("browseRoot"),
  folderBrowser: document.getElementById("folderBrowser"),
  folderBrowserClose: document.getElementById("folderBrowserClose"),
  folderBrowserUp: document.getElementById("folderBrowserUp"),
  folderBrowserPath: document.getElementById("folderBrowserPath"),
  folderBrowserList: document.getElementById("folderBrowserList"),
  folderBrowserSelect: document.getElementById("folderBrowserSelect"),
  folderBrowserCurrent: document.getElementById("folderBrowserCurrent"),
  checkModelEnv: document.getElementById("checkModelEnv"),
  refreshModels: document.getElementById("refreshModels"),
  profileId: document.getElementById("profileId"),
  profileTemplate: document.getElementById("profileTemplate"),
  profileDescription: document.getElementById("profileDescription"),
  profileRuntimeParams: document.getElementById("profileRuntimeParams"),
  profileExtraParams: document.getElementById("profileExtraParams"),
  addRuntimeParam: document.getElementById("addRuntimeParam"),
  addExtraParam: document.getElementById("addExtraParam"),
  profileSave: document.getElementById("profileSave"),
  profileInspect: document.getElementById("profileInspect"),
  profileLoad: document.getElementById("profileLoad"),
  profileUnload: document.getElementById("profileUnload"),
  profileTest: document.getElementById("profileTest"),
  profileDelete: document.getElementById("profileDelete"),
  modelName: document.getElementById("modelName"),
  checkpointPath: document.getElementById("checkpointPath"),
  browseCheckpoint: document.getElementById("browseCheckpoint"),
  modelAdapterType: document.getElementById("modelAdapterType"),
  modelDevice: document.getElementById("modelDevice"),
  registerModel: document.getElementById("registerModel"),
  modelEnvReport: document.getElementById("modelEnvReport"),
  modelList: document.getElementById("modelList"),
  backtestSelectionTable: document.getElementById("backtestSelectionTable"),
  clearBacktestSelection: document.getElementById("clearBacktestSelection"),
  limitBacktestFrames: document.getElementById("limitBacktestFrames"),
  backtestEnvVars: document.getElementById("backtestEnvVars"),
  addBacktestEnvVar: document.getElementById("addBacktestEnvVar"),
  backtestModelChoices: document.getElementById("backtestModelChoices"),
  runBacktest: document.getElementById("runBacktest"),
  refreshBacktestJobs: document.getElementById("refreshBacktestJobs"),
  clearBacktest: document.getElementById("clearBacktest"),
  refreshBacktestHistory: document.getElementById("refreshBacktestHistory"),
  backtestExportActions: document.getElementById("backtestExportActions"),
  backtestResult: document.getElementById("backtestResult"),
  backtestJobQueue: document.getElementById("backtestJobQueue"),
  backtestHistory: document.getElementById("backtestHistory"),
  backtestEpisodeSelect: document.getElementById("backtestEpisodeSelect"),
  backtestDimSelect: document.getElementById("backtestDimSelect"),
  selectAllBacktestDims: document.getElementById("selectAllBacktestDims"),
  clearBacktestDims: document.getElementById("clearBacktestDims"),
  showGroundTruth: document.getElementById("showGroundTruth"),
  showBacktestError: document.getElementById("showBacktestError"),
  backtestSeriesToggles: document.getElementById("backtestSeriesToggles"),
  backtestChart: document.getElementById("backtestChart"),
  zoomBacktestIn: document.getElementById("zoomBacktestIn"),
  zoomBacktestOut: document.getElementById("zoomBacktestOut"),
  resetBacktestZoom: document.getElementById("resetBacktestZoom"),
  refreshTrainingRecipes: document.getElementById("refreshTrainingRecipes"),
  checkTrainingEnv: document.getElementById("checkTrainingEnv"),
  trainingRecipeId: document.getElementById("trainingRecipeId"),
  trainingTemplate: document.getElementById("trainingTemplate"),
  trainingRecipeName: document.getElementById("trainingRecipeName"),
  trainingRecipeDescription: document.getElementById("trainingRecipeDescription"),
  trainingFramework: document.getElementById("trainingFramework"),
  trainingDevice: document.getElementById("trainingDevice"),
  trainingLauncher: document.getElementById("trainingLauncher"),
  trainingGpuDevices: document.getElementById("trainingGpuDevices"),
  trainingNumProcesses: document.getElementById("trainingNumProcesses"),
  trainingDatasetPath: document.getElementById("trainingDatasetPath"),
  trainingEpisodeFilter: document.getElementById("trainingEpisodeFilter"),
  trainingOutputDir: document.getElementById("trainingOutputDir"),
  trainingAutoProfile: document.getElementById("trainingAutoProfile"),
  trainingProfileName: document.getElementById("trainingProfileName"),
  trainingProfileAdapter: document.getElementById("trainingProfileAdapter"),
  trainingHyperparams: document.getElementById("trainingHyperparams"),
  addTrainingHyperparam: document.getElementById("addTrainingHyperparam"),
  trainingEnvVars: document.getElementById("trainingEnvVars"),
  addTrainingEnvVar: document.getElementById("addTrainingEnvVar"),
  trainingExtraParams: document.getElementById("trainingExtraParams"),
  addTrainingExtraParam: document.getElementById("addTrainingExtraParam"),
  createTrainingRecipe: document.getElementById("createTrainingRecipe"),
  saveTrainingRecipe: document.getElementById("saveTrainingRecipe"),
  inspectTrainingRecipe: document.getElementById("inspectTrainingRecipe"),
  submitTrainingJob: document.getElementById("submitTrainingJob"),
  deleteTrainingRecipe: document.getElementById("deleteTrainingRecipe"),
  trainingEnvReport: document.getElementById("trainingEnvReport"),
  trainingRecipeList: document.getElementById("trainingRecipeList"),
  refreshTrainingJobs: document.getElementById("refreshTrainingJobs"),
  trainingJobList: document.getElementById("trainingJobList"),
  trainingJobLog: document.getElementById("trainingJobLog"),
  createTrainingPipeline: document.getElementById("createTrainingPipeline"),
  refreshTrainingPipelines: document.getElementById("refreshTrainingPipelines"),
  pipelineRecipe: document.getElementById("pipelineRecipe"),
  pipelineProfileChoices: document.getElementById("pipelineProfileChoices"),
  pipelineEpisodeList: document.getElementById("pipelineEpisodeList"),
  trainingPipelineList: document.getElementById("trainingPipelineList"),
};

// Expose folder-browser entry points on window so onclick handlers
// work even if JS event listeners silently fail to attach.
window._openMergeBrowser = function () {
  try {
    if (typeof openFolderBrowser !== "function") return;
    openFolderBrowser(els.mergePaths, function (dir) {
      addMergePath(dir);
      if (els.mergePaths) els.mergePaths.value = "";
    });
  } catch (_) {}
};
window._openCheckpointBrowser = function () {
  try {
    if (typeof openFolderBrowser !== "function") return;
    openFolderBrowser(els.checkpointPath, function (dir) {
      if (els.checkpointPath) els.checkpointPath.value = dir;
    });
  } catch (_) {}
};
window._openRootBrowser = function () {
  try {
    if (typeof openFolderBrowser !== "function") return;
    openFolderBrowser(els.datasetPath, function (dir) {
      if (els.datasetPath) els.datasetPath.value = dir;
    });
  } catch (_) {}
};
window._openTrainingDatasetBrowser = function () {
  try {
    if (typeof openFolderBrowser !== "function") return;
    openFolderBrowser(els.trainingDatasetPath, function (dir) {
      if (els.trainingDatasetPath) els.trainingDatasetPath.value = dir;
    });
  } catch (_) {}
};
window._openTrainingOutputBrowser = function () {
  try {
    if (typeof openFolderBrowser !== "function") return;
    openFolderBrowser(els.trainingOutputDir, function (dir) {
      if (els.trainingOutputDir) els.trainingOutputDir.value = dir;
    });
  } catch (_) {}
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

function rootForView(viewId) {
  const button = document.querySelector(`.nav-button[data-view="${viewId}"]`);
  return button?.dataset?.root || "data";
}

function setRoot(root, switchView = true) {
  state.currentRoot = root;
  document.querySelectorAll(".root-nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.root === root);
  });
  document.querySelectorAll("[data-root-nav]").forEach((nav) => {
    nav.classList.toggle("active", nav.dataset.rootNav === root);
  });
  if (els.episodeSideSection) {
    els.episodeSideSection.classList.toggle("hidden", root !== "data");
  }
  if (switchView) {
    const currentRoot = rootForView(state.currentView);
    if (currentRoot !== root) {
      setView(root === "model" ? "modelManagerView" : "overviewView");
    }
  }
}

function setView(viewId) {
  const root = rootForView(viewId);
  if (root !== state.currentRoot) setRoot(root, false);
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
    loadProfileTemplates();
    loadModels();
  }
  if (viewId === "modelBacktestView") {
    loadModels();
    loadBacktestJobs();
    loadBacktestHistory();
    renderBacktestSelectionTable();
    requestAnimationFrame(drawBacktestChart);
  }
  if (viewId === "trainingRecipeView") {
    loadTrainingFrameworks();
    loadTrainingTemplates();
    loadTrainingRecipes();
  }
  if (viewId === "trainingQueueView") {
    loadTrainingJobs();
  }
  if (viewId === "trainingPipelineView") {
    loadTrainingRecipes();
    loadModels();
    loadTrainingPipelines();
    renderPipelineSetup();
  }
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
      <div class="history-item-row">
        <button class="history-item" data-path="${escapeAttr(item.path)}">
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
  state.trimDraftStart = 0;
  state.trimDraftEnd = state.duration;
  // 如果当前 episode 已有区间标记，恢复为标记的实际范围
  const existingMark = findEpisodeMark(index);
  if (existingMark
      && (existingMark.type === "trim_episode" || existingMark.type === "select_episode_range")
      && existingMark.start_time != null && existingMark.end_time != null) {
    state.trimDraftStart = existingMark.start_time;
    state.trimDraftEnd = existingMark.end_time;
  }
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
  const episodeIndex = currentEpisodeIndex();
  const mark = episodeIndex !== null ? findEpisodeMark(episodeIndex) : null;
  const hasRangeMark = mark
    && (mark.type === "trim_episode" || mark.type === "select_episode_range")
    && mark.start_time != null && mark.end_time != null;
  // draft 与已提交标记一致时才显示"已标记"，不一致则显示 draft（用户正在重新调整）
  const draftMatchesMark = hasRangeMark
    && Math.abs((state.trimDraftStart ?? 0) - mark.start_time) < 1e-6
    && Math.abs((state.trimDraftEnd ?? 0) - mark.end_time) < 1e-6;

  if (hasRangeMark && draftMatchesMark) {
    const modeLabel = state.editMode === "export" ? "导出" : "裁剪";
    els.trimDraft.textContent = `已标记保留区间（${modeLabel}）: ${fmt(mark.start_time)}s - ${fmt(mark.end_time)}s`;
  } else {
    const start = state.trimDraftStart;
    const end = state.trimDraftEnd;
    if (start === null && end === null) {
      els.trimDraft.textContent = "裁剪区间未设置";
    } else {
      els.trimDraft.textContent = `保留区间: ${start === null ? "未设置" : `${fmt(start)}s`} - ${end === null ? "未设置" : `${fmt(end)}s`}`;
    }
  }
  if (els.editEpisodeMeta && state.episode) {
    els.editEpisodeMeta.textContent = `${state.episode.length || state.elapsed.length} frames · 当前 ${fmt(state.currentElapsed)}s / ${fmt(state.duration)}s`;
  }
}

function operationKey(operation) {
  return `${operation.type}:${operation.episode_index}`;
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
      updateTrimDraftLabel();
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
  updateTrimDraftLabel();
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
    updateTrimDraftLabel();
    return;
  }
  upsertEditOperation({ type: targetType, episode_index: episodeIndex });
  const label = state.editMode === "export" ? "导出" : "删除";
  els.editEpisodeMeta.textContent = `已标记 Episode ${episodeIndex}（${label}模式）。`;
  refreshMarkButtons();
  updateTrimDraftLabel();
}

function setTrimPoint(kind) {
  if (!state.episode) return;
  if (kind === "start") state.trimDraftStart = state.currentElapsed;
  if (kind === "end") state.trimDraftEnd = state.currentElapsed;
  updateTrimDraftLabel();
  const btn = kind === "start" ? els.setTrimStart : els.setTrimEnd;
  btn.classList.add("trim-set-flash");
  setTimeout(() => btn.classList.remove("trim-set-flash"), 350);
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
    updateTrimDraftLabel();
    return;
  }
  upsertEditOperation({ type: targetType, episode_index: episodeIndex, start_time: start, end_time: end });
  const label = state.editMode === "export" ? "导出" : "裁剪";
  els.editEpisodeMeta.textContent = `已标记 Episode ${episodeIndex} 的区间（${label}模式）。`;
  refreshMarkButtons();
  updateTrimDraftLabel();
}

function refreshMarkButtons() {
  if (!els.markEpisode || !els.markRange) return;
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null) {
    els.markEpisode.classList.remove("active", "mark-delete", "mark-export");
    els.markRange.classList.remove("active", "mark-delete", "mark-export");
    refreshBacktestSampleButton();
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
  refreshBacktestSampleButton();
}

function sendCurrentEpisodeToBacktest() {
  toggleCurrentEpisodeBacktest();
}

function backtestEpisodeKey(item) {
  return `${item.dataset_path}::${item.episode_index}`;
}

function currentBacktestEpisodeItem() {
  const episodeIndex = currentEpisodeIndex();
  if (episodeIndex === null || !state.summary || !state.episode) return null;
  const length = Number(state.episode.length || state.elapsed.length || 0);
  const fps = Number(state.summary.fps || 0);
  return {
    dataset_id: state.summary.id,
    dataset_path: state.summary.root,
    dataset_name: datasetName(state.summary.root),
    episode_index: episodeIndex,
    length,
    duration: fps > 0 ? length / fps : state.duration,
    fps,
    tasks: Array.isArray(state.episode.tasks) ? state.episode.tasks : [],
    video_keys: state.summary.video_keys || [],
  };
}

function currentBacktestEpisodePoolIndex() {
  const item = currentBacktestEpisodeItem();
  if (!item) return -1;
  const key = backtestEpisodeKey(item);
  return state.backtestEpisodes.findIndex((existing) => backtestEpisodeKey(existing) === key);
}

function refreshBacktestSampleButton() {
  if (!els.sendEpisodeToBacktest) return;
  const active = currentBacktestEpisodePoolIndex() >= 0;
  els.sendEpisodeToBacktest.classList.toggle("active", active);
  els.sendEpisodeToBacktest.classList.toggle("mark-backtest", active);
  els.sendEpisodeToBacktest.textContent = active ? "已加入回测样本池" : "加入回测样本池";
}

function toggleCurrentEpisodeBacktest() {
  const item = currentBacktestEpisodeItem();
  if (!item) return;
  const existingIndex = currentBacktestEpisodePoolIndex();
  if (existingIndex >= 0) {
    state.backtestEpisodes.splice(existingIndex, 1);
    if (els.editEpisodeMeta) {
      els.editEpisodeMeta.textContent = `已从回测样本池移除：${item.dataset_name} / Episode ${item.episode_index}。`;
    }
  } else {
    state.backtestEpisodes.push(item);
    if (els.editEpisodeMeta) {
      els.editEpisodeMeta.textContent = `已加入回测样本池：${item.dataset_name} / Episode ${item.episode_index}。`;
    }
  }
  renderBacktestSelectionTable();
  renderPipelineSetup();
  refreshBacktestSampleButton();
}

function addCurrentEpisodeToBacktest() {
  const item = currentBacktestEpisodeItem();
  if (!item) return;
  if (!state.backtestEpisodes.some((existing) => backtestEpisodeKey(existing) === backtestEpisodeKey(item))) {
    state.backtestEpisodes.push(item);
  }
  renderBacktestSelectionTable();
  renderPipelineSetup();
  if (els.editEpisodeMeta) {
    els.editEpisodeMeta.textContent = `已加入回测样本池：${item.dataset_name} / Episode ${item.episode_index}。`;
  }
  refreshBacktestSampleButton();
}

function datasetName(path) {
  const text = String(path || "").replace(/[\\/]+$/, "");
  const parts = text.split(/[\\/]/);
  return parts[parts.length - 1] || text || "-";
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

async function loadOperationLogs() {
  if (!els.operationLogReport) return;
  els.operationLogReport.classList.remove("empty");
  els.operationLogReport.textContent = "正在加载操作日志...";
  try {
    const logs = await api("/api/operations/logs?limit=200");
    renderOperationLogs(logs);
  } catch (error) {
    els.operationLogReport.textContent = error.message;
  }
}

function renderOperationLogs(logs) {
  if (!logs.length) {
    els.operationLogReport.classList.add("empty");
    els.operationLogReport.textContent = "暂无操作日志。";
    return;
  }
  els.operationLogReport.classList.remove("empty");
  els.operationLogReport.innerHTML = logs.slice().reverse().map((item) => `
    <div class="operation-log-row ${escapeAttr(item.status || "unknown")}">
      <div>
        <strong>${escapeHtml(item.action || "-")}</strong>
        <span>${escapeHtml(item.target || "")}</span>
      </div>
      <div class="operation-log-meta">
        <span>${escapeHtml(item.status || "-")}</span>
        <time>${escapeHtml(formatTimestamp(item.timestamp))}</time>
      </div>
      ${item.error ? `<small>${escapeHtml(item.error)}</small>` : ""}
    </div>
  `).join("");
}

function formatTimestamp(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
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
    state.modelEnv = await api("/api/profiles/env");
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
    ["已加载档案", env.profiles_loaded?.length ? env.profiles_loaded.join(", ") : "无"],
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

async function loadProfileTemplates() {
  if (!els.profileTemplate) return;
  try {
    state.profileTemplates = await api("/api/profiles/templates");
    els.profileTemplate.innerHTML = state.profileTemplates.map((template) => `
      <option value="${escapeAttr(template.id)}">${escapeHtml(template.label || template.id)}</option>
    `).join("");
    applySelectedTemplate(false);
  } catch (error) {
    if (els.modelEnvReport) els.modelEnvReport.textContent = error.message;
  }
}

function applySelectedTemplate(overwrite = true) {
  if (!els.profileTemplate || !state.profileTemplates.length) return;
  const template = state.profileTemplates.find((item) => item.id === els.profileTemplate.value) || state.profileTemplates[0];
  if (template.adapter && els.modelAdapterType) els.modelAdapterType.value = template.adapter;
  if (overwrite || !paramRows(els.profileRuntimeParams).length) {
    renderParamTable(els.profileRuntimeParams, template.runtime_params || {});
  }
}

function renderParamTable(container, params = {}) {
  if (!container) return;
  const entries = Object.entries(params || {});
  container.classList.toggle("empty", entries.length === 0);
  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>参数名</th>
          <th>类型</th>
          <th>值</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        ${entries.map(([key, value]) => paramRowHtml(key, value)).join("")}
      </tbody>
    </table>
    ${entries.length ? "" : `<div class="param-empty">暂无参数，点击添加行。</div>`}
  `;
}

function paramRowHtml(key = "", value = "") {
  const type = inferParamType(value);
  return `
    <tr>
      <td><input class="param-key" type="text" value="${escapeAttr(key)}" placeholder="key"></td>
      <td>
        <select class="param-type">
          ${["string", "number", "boolean", "null", "json"].map((item) => `<option value="${item}" ${item === type ? "selected" : ""}>${item}</option>`).join("")}
        </select>
      </td>
      <td><input class="param-value" type="text" value="${escapeAttr(paramValueText(value, type))}" placeholder="value"></td>
      <td><button type="button" data-param-remove="1">删除</button></td>
    </tr>
  `;
}

function inferParamType(value) {
  if (value === null) return "null";
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number") return "number";
  if (typeof value === "string") return "string";
  return "json";
}

function paramValueText(value, type = inferParamType(value)) {
  if (type === "null") return "";
  if (type === "json") return JSON.stringify(value);
  return String(value ?? "");
}

function paramRows(container) {
  return Array.from(container?.querySelectorAll("tbody tr") || []);
}

function addParamRow(container, key = "", value = "") {
  if (!container) return;
  if (!container.querySelector("table")) renderParamTable(container, {});
  const tbody = container.querySelector("tbody");
  tbody.insertAdjacentHTML("beforeend", paramRowHtml(key, value));
  container.classList.remove("empty");
  const empty = container.querySelector(".param-empty");
  if (empty) empty.remove();
}

function readParamTable(container, label) {
  const result = {};
  for (const row of paramRows(container)) {
    const key = row.querySelector(".param-key")?.value.trim();
    if (!key) continue;
    if (Object.prototype.hasOwnProperty.call(result, key)) {
      throw new Error(`${label} 存在重复参数名：${key}`);
    }
    const type = row.querySelector(".param-type")?.value || "string";
    const raw = row.querySelector(".param-value")?.value ?? "";
    result[key] = parseParamValue(raw, type, `${label}.${key}`);
  }
  return result;
}

function readBacktestEnvVars() {
  return readScalarEnvVars(els.backtestEnvVars, "回测环境变量");
}

function readTrainingEnvVars() {
  return readScalarEnvVars(els.trainingEnvVars, "训练环境变量");
}

function readScalarEnvVars(container, label) {
  const params = readParamTable(container, label);
  const envVars = {};
  for (const [key, value] of Object.entries(params)) {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key)) {
      throw new Error(`环境变量名不合法：${key}`);
    }
    if (value === null || value === undefined) {
      envVars[key] = "";
    } else if (["string", "number", "boolean"].includes(typeof value)) {
      envVars[key] = String(value);
    } else {
      throw new Error(`环境变量 ${key} 的值必须是字符串、数字或布尔值`);
    }
  }
  return envVars;
}

function saveBacktestEnvVars() {
  if (!els.backtestEnvVars) return;
  try {
    localStorage.setItem(BACKTEST_ENV_VARS_KEY, JSON.stringify(readBacktestEnvVars()));
  } catch (_) {
    // Ignore incomplete rows while the user is editing.
  }
}

function initBacktestEnvVars() {
  if (!els.backtestEnvVars) return;
  let envVars = {};
  try {
    envVars = JSON.parse(localStorage.getItem(BACKTEST_ENV_VARS_KEY) || "{}") || {};
  } catch (_) {
    envVars = {};
  }
  renderParamTable(els.backtestEnvVars, envVars);
}

function addBacktestEnvVar() {
  if (!els.backtestEnvVars) return;
  if (!els.backtestEnvVars.querySelector("table")) {
    renderParamTable(els.backtestEnvVars, {});
  }
  const hasCudaRow = paramRows(els.backtestEnvVars).some((row) => {
    return row.querySelector(".param-key")?.value.trim() === "CUDA_VISIBLE_DEVICES";
  });
  addParamRow(els.backtestEnvVars, hasCudaRow ? "" : "CUDA_VISIBLE_DEVICES", "");
  saveBacktestEnvVars();
}

window._addBacktestEnvVar = addBacktestEnvVar;

function addTrainingEnvVar() {
  if (!els.trainingEnvVars) return;
  if (!els.trainingEnvVars.querySelector("table")) {
    renderParamTable(els.trainingEnvVars, {});
  }
  addParamRow(els.trainingEnvVars);
}

function formatEnvVars(envVars) {
  const entries = Object.entries(envVars || {});
  if (!entries.length) return "-";
  return entries.map(([key, value]) => `${key}=${value}`).join(", ");
}

function parseParamValue(raw, type, label) {
  if (type === "string") return raw;
  if (type === "number") {
    const value = Number(raw);
    if (Number.isNaN(value)) throw new Error(`${label} 必须是数字`);
    return value;
  }
  if (type === "boolean") {
    const normalized = raw.trim().toLowerCase();
    if (["true", "1", "yes", "是"].includes(normalized)) return true;
    if (["false", "0", "no", "否"].includes(normalized)) return false;
    throw new Error(`${label} 必须是 boolean，可填 true/false`);
  }
  if (type === "null") return null;
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`${label} JSON 格式错误：${error.message}`);
  }
}

function handleParamTableClick(event) {
  const button = event.target?.closest?.("[data-param-remove]");
  if (!button) return;
  const container = button.closest(".param-table");
  button.closest("tr")?.remove();
  if (container && !paramRows(container).length) renderParamTable(container, {});
}

async function loadModels() {
  try {
    state.models = await api("/api/profiles");
    renderModels();
    renderBacktestModelChoices();
    renderPipelineSetup();
  } catch (error) {
    if (els.modelList) els.modelList.textContent = error.message;
  }
}

async function registerCurrentModel() {
  const profileId = els.profileId.value.trim();
  if (!profileId) {
    els.modelList.textContent = "请填写 profile id。";
    return;
  }
  try {
    await api("/api/profiles", {
      method: "POST",
      body: JSON.stringify({
        id: profileId,
        template_id: els.profileTemplate?.value || "blank",
        name: els.modelName.value.trim() || null,
        description: els.profileDescription.value.trim() || null,
        checkpoint_path: els.checkpointPath.value.trim() || "",
        adapter: els.modelAdapterType.value,
        device: els.modelDevice.value,
        runtime_params: readParamTable(els.profileRuntimeParams, "运行期参数"),
        extra_params: readParamTable(els.profileExtraParams, "自定义参数"),
      }),
    });
    state.selectedProfileId = null;
    if (els.profileId) els.profileId.disabled = false;
    await loadModels();
  } catch (error) {
    els.modelList.textContent = error.message;
  }
}

function renderModels() {
  if (!els.modelList) return;
  if (!state.models.length) {
    els.modelList.classList.add("empty");
    els.modelList.innerHTML = "尚未创建模型档案。";
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
            <span>${escapeHtml(model.adapter)} · ${escapeHtml(model.device)} · ${escapeHtml(model.status)}</span>
          </div>
          <span class="result-status ${model.loaded ? "ok" : errors.length ? "fail" : "neutral"}">${model.loaded ? "已加载" : errors.length ? "无效" : "已保存"}</span>
        </div>
        <div class="model-card-path">${escapeHtml(model.checkpoint_path)}</div>
        <div class="model-card-meta">
          <span>policy: ${escapeHtml(inspection.policy_type || "-")}</span>
          <span>size: ${escapeHtml(inspection.size_mb ?? 0)} MB</span>
          <span>files: ${escapeHtml(inspection.file_count ?? 0)}</span>
          ${inspection.parameter_count ? `<span>params: ${escapeHtml(inspection.parameter_count)}</span>` : ""}
        </div>
        <div class="model-card-path">runtime: ${escapeHtml(JSON.stringify(model.runtime_params || {}))}</div>
        ${errors.length ? `<div class="model-card-issues error">${errors.map(escapeHtml).join("<br>")}</div>` : ""}
        ${warnings.length ? `<div class="model-card-issues warning">${warnings.map(escapeHtml).join("<br>")}</div>` : ""}
        <div class="model-card-actions">
          <button type="button" data-model-action="select">编辑</button>
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
  try {
    if (action === "select") {
      fillProfileEditor(state.models.find((model) => model.id === modelId) || await api(`/api/profiles/${encodeURIComponent(modelId)}`));
      return;
    }
    if (action === "delete") {
      await api(`/api/profiles/${encodeURIComponent(modelId)}`, { method: "DELETE" });
      if (state.selectedProfileId === modelId) clearProfileEditor();
    } else {
      await api(`/api/profiles/${encodeURIComponent(modelId)}/${action}`, { method: "POST" });
    }
    await loadModels();
  } catch (error) {
    card.insertAdjacentHTML("beforeend", `<div class="model-card-issues error">${escapeHtml(error.message)}</div>`);
  }
}

function fillProfileEditor(profile) {
  state.selectedProfileId = profile.id;
  els.profileId.value = profile.id || "";
  els.profileId.disabled = true;
  els.modelName.value = profile.name || "";
  els.profileDescription.value = profile.description || "";
  els.checkpointPath.value = profile.checkpoint_path || "";
  els.modelAdapterType.value = profile.adapter || "lerobot_official";
  els.modelDevice.value = profile.device || "cuda";
  renderParamTable(els.profileRuntimeParams, profile.runtime_params || {});
  renderParamTable(els.profileExtraParams, profile.extra_params || {});
}

function clearProfileEditor() {
  state.selectedProfileId = null;
  if (els.profileId) {
    els.profileId.value = "";
    els.profileId.disabled = false;
  }
  if (els.modelName) els.modelName.value = "";
  if (els.profileDescription) els.profileDescription.value = "";
  if (els.checkpointPath) els.checkpointPath.value = "";
  if (els.modelDevice) els.modelDevice.value = "cuda";
  applySelectedTemplate(true);
  renderParamTable(els.profileExtraParams, {});
}

async function saveCurrentProfile() {
  if (!state.selectedProfileId) {
    await registerCurrentModel();
    return;
  }
  try {
    await api(`/api/profiles/${encodeURIComponent(state.selectedProfileId)}`, {
      method: "PUT",
      body: JSON.stringify({
        name: els.modelName.value.trim() || state.selectedProfileId,
        description: els.profileDescription.value.trim() || "",
        checkpoint_path: els.checkpointPath.value.trim() || "",
        adapter: els.modelAdapterType.value,
        device: els.modelDevice.value,
        runtime_params: readParamTable(els.profileRuntimeParams, "运行期参数"),
        extra_params: readParamTable(els.profileExtraParams, "自定义参数"),
      }),
    });
    await loadModels();
  } catch (error) {
    els.modelList.textContent = error.message;
  }
}

async function runProfileEditorAction(action) {
  const profileId = state.selectedProfileId || els.profileId?.value?.trim();
  if (!profileId) {
    els.modelList.textContent = "请先选择或创建 profile。";
    return;
  }
  try {
    if (action === "delete") {
      await api(`/api/profiles/${encodeURIComponent(profileId)}`, { method: "DELETE" });
      clearProfileEditor();
    } else if (action === "test") {
      if (!state.summary || currentEpisodeIndex() === null) {
        throw new Error("请先在数据查看中加载数据集并选择一个 episode，再做真实帧测试。");
      }
      const result = await api(`/api/profiles/${encodeURIComponent(profileId)}/test`, {
        method: "POST",
        body: JSON.stringify({
          dataset_path: state.summary.root,
          episode_index: currentEpisodeIndex(),
          frame_index: currentFrameIndex(),
        }),
      });
      els.modelEnvReport.classList.remove("empty");
      els.modelEnvReport.innerHTML = formatKeyValueGrid([
        ["Profile", result.profile?.id],
        ["Episode", result.episode_index],
        ["Frame", result.frame_index],
        ["Action shape", result.action_shape],
        ["Elapsed", `${result.elapsed_ms} ms`],
        ["Action preview", result.action_preview],
      ]);
    } else {
      await api(`/api/profiles/${encodeURIComponent(profileId)}/${action}`, { method: "POST" });
    }
    await loadModels();
  } catch (error) {
    els.modelEnvReport.classList.remove("empty");
    els.modelEnvReport.textContent = error.message;
  }
}

async function loadTrainingFrameworks() {
  if (!els.trainingFramework) return;
  try {
    state.trainingFrameworks = await api("/api/train/frameworks");
    els.trainingFramework.innerHTML = state.trainingFrameworks.map((framework) => `
      <option value="${escapeAttr(framework.id)}">${escapeHtml(framework.label || framework.id)}</option>
    `).join("");
  } catch (error) {
    showTrainingMessage(error.message);
  }
}

async function loadTrainingTemplates() {
  if (!els.trainingTemplate) return;
  try {
    state.trainingTemplates = await api("/api/train/templates");
    els.trainingTemplate.innerHTML = state.trainingTemplates.map((template) => `
      <option value="${escapeAttr(template.id)}">${escapeHtml(template.label || template.id)}</option>
    `).join("");
    applyTrainingTemplate(false);
  } catch (error) {
    showTrainingMessage(error.message);
  }
}

function applyTrainingTemplate(overwrite = true) {
  if (!els.trainingTemplate || !state.trainingTemplates.length) return;
  const template = state.trainingTemplates.find((item) => item.id === els.trainingTemplate.value) || state.trainingTemplates[0];
  if (template.framework && els.trainingFramework) els.trainingFramework.value = template.framework;
  if (overwrite || !paramRows(els.trainingHyperparams).length) {
    renderParamTable(els.trainingHyperparams, template.hyperparams || {});
  }
}

async function loadTrainingEnv() {
  if (!els.trainingEnvReport) return;
  try {
    const env = await api("/api/train/env");
    els.trainingEnvReport.classList.remove("empty");
    els.trainingEnvReport.innerHTML = `
      <div class="model-env-grid">
        <div class="model-env-cell"><span>Worker</span><strong>${escapeHtml(env.worker?.worker_alive ? "运行中" : "空闲")}</strong></div>
        <div class="model-env-cell"><span>Queued</span><strong>${escapeHtml(env.worker?.queued ?? 0)}</strong></div>
        <div class="model-env-cell"><span>Running</span><strong>${escapeHtml(env.worker?.running || "无")}</strong></div>
      </div>
      <div class="model-env-checks">
        ${(env.frameworks || []).map((item) => `
          <div class="tool-check ok"><strong>${escapeHtml(item.label || item.id)}</strong><span>${escapeHtml(item.id)}</span></div>
        `).join("")}
      </div>
    `;
  } catch (error) {
    showTrainingMessage(error.message);
  }
}

async function loadTrainingRecipes() {
  try {
    state.trainingRecipes = await api("/api/train/recipes");
    renderTrainingRecipes();
    renderPipelineSetup();
  } catch (error) {
    if (els.trainingRecipeList) els.trainingRecipeList.textContent = error.message;
  }
}

function renderTrainingRecipes() {
  if (!els.trainingRecipeList) return;
  if (!state.trainingRecipes.length) {
    els.trainingRecipeList.classList.add("empty");
    els.trainingRecipeList.innerHTML = "尚未创建训练配方。";
    return;
  }
  els.trainingRecipeList.classList.remove("empty");
  els.trainingRecipeList.innerHTML = state.trainingRecipes.map((recipe) => {
    const hp = recipe.hyperparams || {};
    const inspection = recipe.inspection || {};
    const errors = inspection.errors || [];
    return `
      <div class="model-card" data-training-recipe-id="${escapeAttr(recipe.id)}">
        <div class="model-card-head">
          <div>
            <strong>${escapeHtml(recipe.name || recipe.id)}</strong>
            <span>${escapeHtml(recipe.framework)} · ${escapeHtml(recipe.device)} · ${escapeHtml(recipe.status)}</span>
          </div>
          <span class="result-status ${errors.length ? "fail" : "neutral"}">${errors.length ? "无效" : "已保存"}</span>
        </div>
        <div class="model-card-path">dataset: ${escapeHtml(recipe.dataset_path || "-")}</div>
        <div class="model-card-path">output: ${escapeHtml(recipe.output_dir || "-")}</div>
        <div class="model-card-meta">
          <span>policy: ${escapeHtml(hp.policy_type || "-")}</span>
          <span>batch: ${escapeHtml(hp.batch_size ?? "-")}</span>
          <span>epochs: ${escapeHtml(hp.epochs ?? "-")}</span>
          <span>lr: ${escapeHtml(hp.learning_rate ?? "-")}</span>
          <span>launcher: ${escapeHtml(recipe.launcher || "direct")}</span>
          <span>gpu: ${escapeHtml(recipe.gpu_devices || "-")}</span>
          <span>proc: ${escapeHtml(recipe.num_processes || 1)}</span>
        </div>
        ${errors.length ? `<div class="model-card-issues error">${errors.map(escapeHtml).join("<br>")}</div>` : ""}
        <div class="model-card-actions">
          <button type="button" data-training-recipe-action="select">编辑</button>
          <button type="button" data-training-recipe-action="inspect">检查</button>
          <button type="button" data-training-recipe-action="submit">提交训练</button>
          <button type="button" data-training-recipe-action="delete">删除</button>
        </div>
      </div>
    `;
  }).join("");
}

function readEpisodeFilter() {
  const text = (els.trainingEpisodeFilter?.value || "").trim();
  if (!text) return null;
  return text.split(/[,\s]+/).filter(Boolean).map((item) => {
    const value = Number(item);
    if (!Number.isInteger(value) || value < 0) throw new Error(`episode 过滤值无效：${item}`);
    return value;
  });
}

function trainingRecipePayload() {
  const numProcesses = Number(els.trainingNumProcesses?.value || 1);
  if (!Number.isInteger(numProcesses) || numProcesses < 1) {
    throw new Error("训练进程数必须是大于等于 1 的整数。");
  }
  return {
    name: els.trainingRecipeName?.value.trim() || null,
    description: els.trainingRecipeDescription?.value.trim() || null,
    framework: els.trainingFramework?.value || "lerobot_train",
    dataset_path: els.trainingDatasetPath?.value.trim() || "",
    episode_filter: readEpisodeFilter(),
    output_dir: els.trainingOutputDir?.value.trim() || "",
    hyperparams: readParamTable(els.trainingHyperparams, "训练超参数"),
    device: els.trainingDevice?.value || "cuda",
    launcher: els.trainingLauncher?.value || "direct",
    num_processes: numProcesses,
    gpu_devices: els.trainingGpuDevices?.value.trim() || "",
    env_vars: readTrainingEnvVars(),
    extra_params: readParamTable(els.trainingExtraParams, "训练自定义参数"),
    auto_profile_on_complete: Boolean(els.trainingAutoProfile?.checked),
    profile_name: els.trainingProfileName?.value.trim() || null,
    profile_adapter: els.trainingProfileAdapter?.value || "lerobot_official",
  };
}

async function createTrainingRecipe() {
  const recipeId = els.trainingRecipeId?.value.trim();
  if (!recipeId) {
    showTrainingMessage("请填写 recipe id。");
    return;
  }
  try {
    await api("/api/train/recipes", {
      method: "POST",
      body: JSON.stringify({
        id: recipeId,
        template_id: els.trainingTemplate?.value || "blank_train",
        ...trainingRecipePayload(),
      }),
    });
    state.selectedTrainingRecipeId = recipeId;
    if (els.trainingRecipeId) els.trainingRecipeId.disabled = true;
    await loadTrainingRecipes();
  } catch (error) {
    showTrainingMessage(error.message);
  }
}

async function saveTrainingRecipe() {
  if (!state.selectedTrainingRecipeId) {
    await createTrainingRecipe();
    return;
  }
  try {
    await api(`/api/train/recipes/${encodeURIComponent(state.selectedTrainingRecipeId)}`, {
      method: "PUT",
      body: JSON.stringify(trainingRecipePayload()),
    });
    await loadTrainingRecipes();
  } catch (error) {
    showTrainingMessage(error.message);
  }
}

async function runTrainingRecipeAction(action, recipeId = null) {
  const id = recipeId || state.selectedTrainingRecipeId || els.trainingRecipeId?.value.trim();
  if (!id) {
    showTrainingMessage("请先选择或创建训练配方。");
    return;
  }
  try {
    if (action === "delete") {
      await api(`/api/train/recipes/${encodeURIComponent(id)}`, { method: "DELETE" });
      clearTrainingRecipeEditor();
    } else if (action === "submit") {
      const job = await api("/api/train/jobs", {
        method: "POST",
        body: JSON.stringify({ recipe_id: id }),
      });
      showTrainingMessage(`已提交训练作业：${job.job_id}`);
      await loadTrainingJobs();
    } else {
      const result = await api(`/api/train/recipes/${encodeURIComponent(id)}/${action}`, { method: "POST" });
      fillTrainingRecipeEditor(result);
    }
    await loadTrainingRecipes();
  } catch (error) {
    showTrainingMessage(error.message);
  }
}

function fillTrainingRecipeEditor(recipe) {
  state.selectedTrainingRecipeId = recipe.id;
  if (els.trainingRecipeId) {
    els.trainingRecipeId.value = recipe.id || "";
    els.trainingRecipeId.disabled = true;
  }
  if (els.trainingRecipeName) els.trainingRecipeName.value = recipe.name || "";
  if (els.trainingRecipeDescription) els.trainingRecipeDescription.value = recipe.description || "";
  if (els.trainingFramework) els.trainingFramework.value = recipe.framework || "lerobot_train";
  if (els.trainingDevice) els.trainingDevice.value = recipe.device || "cuda";
  if (els.trainingLauncher) els.trainingLauncher.value = recipe.launcher || "direct";
  if (els.trainingGpuDevices) els.trainingGpuDevices.value = recipe.gpu_devices || "";
  if (els.trainingNumProcesses) els.trainingNumProcesses.value = recipe.num_processes || 1;
  if (els.trainingDatasetPath) els.trainingDatasetPath.value = recipe.dataset_path || "";
  if (els.trainingEpisodeFilter) els.trainingEpisodeFilter.value = Array.isArray(recipe.episode_filter) ? recipe.episode_filter.join(",") : "";
  if (els.trainingOutputDir) els.trainingOutputDir.value = recipe.output_dir || "";
  if (els.trainingAutoProfile) els.trainingAutoProfile.checked = recipe.auto_profile_on_complete !== false;
  if (els.trainingProfileName) els.trainingProfileName.value = recipe.profile_name || "";
  if (els.trainingProfileAdapter) els.trainingProfileAdapter.value = recipe.profile_adapter || "lerobot_official";
  renderParamTable(els.trainingHyperparams, recipe.hyperparams || {});
  renderParamTable(els.trainingEnvVars, recipe.env_vars || {});
  renderParamTable(els.trainingExtraParams, recipe.extra_params || {});
}

function clearTrainingRecipeEditor() {
  state.selectedTrainingRecipeId = null;
  if (els.trainingRecipeId) {
    els.trainingRecipeId.value = "";
    els.trainingRecipeId.disabled = false;
  }
  if (els.trainingRecipeName) els.trainingRecipeName.value = "";
  if (els.trainingRecipeDescription) els.trainingRecipeDescription.value = "";
  if (els.trainingLauncher) els.trainingLauncher.value = "direct";
  if (els.trainingGpuDevices) els.trainingGpuDevices.value = "";
  if (els.trainingNumProcesses) els.trainingNumProcesses.value = 1;
  if (els.trainingDatasetPath) els.trainingDatasetPath.value = "";
  if (els.trainingEpisodeFilter) els.trainingEpisodeFilter.value = "";
  if (els.trainingOutputDir) els.trainingOutputDir.value = "";
  if (els.trainingProfileName) els.trainingProfileName.value = "";
  if (els.trainingAutoProfile) els.trainingAutoProfile.checked = true;
  applyTrainingTemplate(true);
  renderParamTable(els.trainingEnvVars, {});
  renderParamTable(els.trainingExtraParams, {});
}

async function handleTrainingRecipeAction(event) {
  const action = event.target?.dataset?.trainingRecipeAction;
  if (!action) return;
  const card = event.target.closest(".model-card");
  const recipeId = card?.dataset?.trainingRecipeId;
  if (!recipeId) return;
  if (action === "select") {
    const recipe = await api(`/api/train/recipes/${encodeURIComponent(recipeId)}`);
    fillTrainingRecipeEditor(recipe);
    return;
  }
  await runTrainingRecipeAction(action, recipeId);
}

async function loadTrainingJobs() {
  if (!els.trainingJobList) return;
  try {
    state.trainingJobs = await api("/api/train/jobs");
    renderTrainingJobs();
  } catch (error) {
    els.trainingJobList.textContent = error.message;
  }
}

function renderTrainingJobs() {
  if (!els.trainingJobList) return;
  if (!state.trainingJobs.length) {
    els.trainingJobList.classList.add("empty");
    els.trainingJobList.innerHTML = "暂无训练作业。";
    return;
  }
  els.trainingJobList.classList.remove("empty");
  els.trainingJobList.innerHTML = `
    <table>
      <thead><tr><th>Job</th><th>Recipe</th><th>Status</th><th>Progress</th><th>Profile</th><th>操作</th></tr></thead>
      <tbody>
        ${state.trainingJobs.map((job) => {
          const progress = job.progress || {};
          const progressText = progress.epoch ? `epoch ${progress.epoch}/${progress.total_epochs || "?"} · loss ${fmt(progress.loss)}` : "-";
          return `
            <tr data-training-job-id="${escapeAttr(job.job_id)}">
              <td>${escapeHtml(job.job_id)}<br><small>${escapeHtml(job.created_at || "")}</small></td>
              <td>${escapeHtml(job.recipe_name || job.recipe_id)}</td>
              <td>${escapeHtml(job.status)}</td>
              <td>${escapeHtml(progressText)}</td>
              <td>${escapeHtml(job.auto_generated_profile_id || "-")}</td>
              <td>
                <button type="button" data-training-job-action="log">日志</button>
                <button type="button" data-training-job-action="cancel">取消</button>
                <button type="button" data-training-job-action="requeue">重排</button>
                <button type="button" data-training-job-action="delete">删除</button>
              </td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

async function handleTrainingJobAction(event) {
  const action = event.target?.dataset?.trainingJobAction;
  if (!action) return;
  const row = event.target.closest("[data-training-job-id]");
  const jobId = row?.dataset?.trainingJobId;
  if (!jobId) return;
  try {
    if (action === "log") {
      const log = await api(`/api/train/jobs/${encodeURIComponent(jobId)}/log?tail=300`);
      if (els.trainingJobLog) els.trainingJobLog.textContent = log.text || "暂无日志。";
      return;
    }
    if (action === "delete") {
      await api(`/api/train/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
    } else {
      await api(`/api/train/jobs/${encodeURIComponent(jobId)}/${action}`, { method: "POST" });
    }
    await loadTrainingJobs();
  } catch (error) {
    if (els.trainingJobLog) els.trainingJobLog.textContent = error.message;
  }
}

function showTrainingMessage(message) {
  if (!els.trainingEnvReport) return;
  els.trainingEnvReport.classList.remove("empty");
  els.trainingEnvReport.textContent = message;
}

function renderPipelineSetup() {
  if (els.pipelineRecipe) {
    els.pipelineRecipe.innerHTML = state.trainingRecipes.map((recipe) => `
      <option value="${escapeAttr(recipe.id)}">${escapeHtml(recipe.name || recipe.id)}</option>
    `).join("");
  }
  if (els.pipelineProfileChoices) {
    if (!state.models.length) {
      els.pipelineProfileChoices.classList.add("empty");
      els.pipelineProfileChoices.innerHTML = "暂无可对比 profile。";
    } else {
      els.pipelineProfileChoices.classList.remove("empty");
      els.pipelineProfileChoices.innerHTML = state.models.map((model) => `
        <label><input type="checkbox" value="${escapeAttr(model.id)}"> ${escapeHtml(model.name || model.id)}</label>
      `).join("");
    }
  }
  if (els.pipelineEpisodeList) {
    if (!state.backtestEpisodes.length) {
      els.pipelineEpisodeList.classList.add("empty");
      els.pipelineEpisodeList.innerHTML = "从 Episode 播放页加入回测样本后会显示在这里。";
    } else {
      els.pipelineEpisodeList.classList.remove("empty");
      els.pipelineEpisodeList.innerHTML = `<table><tbody>${state.backtestEpisodes.map((item) => `
        <tr><td>${escapeHtml(item.dataset_name || item.dataset_path)}</td><td>ep ${escapeHtml(item.episode_index)}</td><td>${escapeHtml(item.length ?? "-")} frames</td></tr>
      `).join("")}</tbody></table>`;
    }
  }
}

async function createTrainingPipeline() {
  const recipeId = els.pipelineRecipe?.value;
  if (!recipeId) {
    showTrainingMessage("请先创建训练配方。");
    return;
  }
  const comparisonProfileIds = Array.from(els.pipelineProfileChoices?.querySelectorAll("input:checked") || []).map((input) => input.value);
  try {
    await api("/api/train/pipelines", {
      method: "POST",
      body: JSON.stringify({
        recipe_id: recipeId,
        comparison_profile_ids: comparisonProfileIds,
        episodes: state.backtestEpisodes,
      }),
    });
    await loadTrainingPipelines();
    await loadTrainingJobs();
  } catch (error) {
    if (els.trainingPipelineList) els.trainingPipelineList.textContent = error.message;
  }
}

async function loadTrainingPipelines() {
  if (!els.trainingPipelineList) return;
  try {
    state.trainingPipelines = await api("/api/train/pipelines");
    renderTrainingPipelines();
  } catch (error) {
    els.trainingPipelineList.textContent = error.message;
  }
}

function renderTrainingPipelines() {
  if (!els.trainingPipelineList) return;
  if (!state.trainingPipelines.length) {
    els.trainingPipelineList.classList.add("empty");
    els.trainingPipelineList.innerHTML = "暂无流水线。";
    return;
  }
  els.trainingPipelineList.classList.remove("empty");
  els.trainingPipelineList.innerHTML = `
    <table>
      <thead><tr><th>Pipeline</th><th>Recipe</th><th>Training Job</th><th>Status</th><th>Episodes</th></tr></thead>
      <tbody>
        ${state.trainingPipelines.map((pipe) => `
          <tr>
            <td>${escapeHtml(pipe.pipeline_id)}</td>
            <td>${escapeHtml(pipe.recipe_id)}</td>
            <td>${escapeHtml(pipe.training_job_id)}</td>
            <td>${escapeHtml(pipe.status)}</td>
            <td>${escapeHtml((pipe.episodes || []).length)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function parseJsonObject(raw, label) {
  const text = (raw || "").trim();
  if (!text) return {};
  let value;
  try {
    value = JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} JSON 格式错误：${error.message}`);
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} 必须是 JSON object`);
  }
  return value;
}

function renderBacktestModelChoices() {
  if (!els.backtestModelChoices) return;
  if (!state.models.length) {
    els.backtestModelChoices.classList.add("empty");
    els.backtestModelChoices.innerHTML = "请先在模型档案中创建 profile。";
    return;
  }
  els.backtestModelChoices.classList.remove("empty");
  els.backtestModelChoices.innerHTML = state.models.map((model) => `
    <label class="model-choice">
      <input type="checkbox" value="${escapeAttr(model.id)}">
      <span>
        <strong>${escapeHtml(model.name)}</strong>
        <small>${escapeHtml(model.inspection?.policy_type || model.adapter)} · ${escapeHtml(model.status)}</small>
      </span>
    </label>
  `).join("");
}

function selectedBacktestProfileIds() {
  return Array.from(els.backtestModelChoices.querySelectorAll("input:checked")).map((input) => input.value);
}

function renderBacktestSelectionTable() {
  if (!els.backtestSelectionTable) return;
  if (!state.backtestEpisodes.length) {
    els.backtestSelectionTable.classList.add("empty");
    els.backtestSelectionTable.innerHTML = "尚未选择 episode。请先到 LeRobot 数据 / Episode 播放页加入样本。";
    return;
  }
  els.backtestSelectionTable.classList.remove("empty");
  els.backtestSelectionTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>数据集</th>
          <th>路径</th>
          <th>Episode</th>
          <th>帧数</th>
          <th>时长</th>
          <th>FPS</th>
          <th>Task</th>
          <th>视频</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        ${state.backtestEpisodes.map((item) => `
          <tr data-backtest-episode="${escapeAttr(backtestEpisodeKey(item))}">
            <td><strong>${escapeHtml(item.dataset_name || datasetName(item.dataset_path))}</strong></td>
            <td class="path-cell">${escapeHtml(item.dataset_path)}</td>
            <td>${escapeHtml(item.episode_index)}</td>
            <td>${escapeHtml(item.length ?? "-")}</td>
            <td>${item.duration !== null && item.duration !== undefined ? `${fmt(Number(item.duration))}s` : "-"}</td>
            <td>${escapeHtml(item.fps ?? "-")}</td>
            <td>${escapeHtml(Array.isArray(item.tasks) ? item.tasks.join(", ") : item.tasks || "-")}</td>
            <td>${escapeHtml(Array.isArray(item.video_keys) ? item.video_keys.length : 0)} 路</td>
            <td><button type="button" data-backtest-remove="${escapeAttr(backtestEpisodeKey(item))}">移除</button></td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function removeBacktestEpisode(key) {
  state.backtestEpisodes = state.backtestEpisodes.filter((item) => backtestEpisodeKey(item) !== key);
  renderBacktestSelectionTable();
  renderPipelineSetup();
  refreshBacktestSampleButton();
}

function clearBacktestSelection() {
  state.backtestEpisodes = [];
  renderBacktestSelectionTable();
  renderPipelineSetup();
  refreshBacktestSampleButton();
}

async function runSelectedBacktest() {
  const profileIds = selectedBacktestProfileIds();
  if (!profileIds.length) {
    els.backtestResult.textContent = "请至少选择一个模型档案。";
    return;
  }
  if (!state.backtestEpisodes.length) {
    els.backtestResult.textContent = "请先从 Episode 播放页加入回测样本。";
    return;
  }
  els.backtestResult.classList.remove("empty");
  els.backtestResult.textContent = "正在提交后台回测任务...";
  renderBacktestExportActions(null);
  try {
    const envVars = readBacktestEnvVars();
    saveBacktestEnvVars();
    const job = await api("/api/backtests/jobs", {
      method: "POST",
      body: JSON.stringify({
        profile_ids: profileIds,
        episodes: state.backtestEpisodes.map((item) => ({
          dataset_path: item.dataset_path,
          episode_index: Number(item.episode_index),
        })),
        max_frames: els.limitBacktestFrames.checked ? 20 : null,
        env_vars: envVars,
      }),
    });
    state.visibleBacktestModels = new Set(profileIds);
    rememberActiveBacktestJob(job.job_id);
    showBacktestJobStatus(job);
    await loadBacktestJobs();
    pollBacktestJob(job.job_id);
  } catch (error) {
    els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>回测失败</h4><p>${escapeHtml(error.message)}</p></div>`;
  }
}

async function pollBacktestJob(jobId) {
  if (!jobId || state.pollingBacktestJobs.has(jobId)) return;
  state.pollingBacktestJobs.add(jobId);
  try {
    while (true) {
      const job = await api(`/api/backtests/jobs/${encodeURIComponent(jobId)}`);
      if (isDisplayingBacktestJob(jobId)) renderBacktestJobStatus(job);
      await loadBacktestJobs();
      if (job.status === "done") {
        if (isDisplayingBacktestJob(jobId)) {
          state.backtestResult = job.result || await api(`/api/backtests/runs/${encodeURIComponent(job.run_id)}`);
          state.visibleBacktestModels = new Set(state.backtestResult.profile_ids || state.backtestResult.model_ids || []);
          renderBacktestResult();
        }
        forgetActiveBacktestJob(jobId);
        await loadBacktestJobs();
        await loadBacktestHistory();
        return;
      }
      if (job.status === "failed") {
        if (isDisplayingBacktestJob(jobId)) {
          els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>回测失败</h4><p>${escapeHtml(job.error || "后台任务失败")}</p></div>`;
          state.displayedBacktestJobId = null;
        }
        forgetActiveBacktestJob(jobId);
        await loadBacktestJobs();
        await loadBacktestHistory();
        return;
      }
      if (job.status === "interrupted") {
        if (isDisplayingBacktestJob(jobId)) {
          els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>回测任务已中断</h4><p>${escapeHtml(job.error || "服务重启后无法继续这个后台任务")}</p></div>`;
          state.displayedBacktestJobId = null;
        }
        forgetActiveBacktestJob(jobId);
        await loadBacktestJobs();
        return;
      }
      await delay(900);
    }
  } catch (error) {
    forgetActiveBacktestJob(jobId);
    if (isDisplayingBacktestJob(jobId)) {
      els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>回测状态读取失败</h4><p>${escapeHtml(error.message)}</p></div>`;
      state.displayedBacktestJobId = null;
    }
  } finally {
    state.pollingBacktestJobs.delete(jobId);
  }
}

function showBacktestJobStatus(job) {
  state.displayedBacktestJobId = job.job_id;
  renderBacktestJobStatus(job);
}

function isDisplayingBacktestJob(jobId) {
  return state.displayedBacktestJobId === jobId;
}

function rememberActiveBacktestJob(jobId) {
  try {
    localStorage.setItem(ACTIVE_BACKTEST_JOB_KEY, jobId);
  } catch (_) {}
}

function forgetActiveBacktestJob(jobId) {
  try {
    if (!jobId || localStorage.getItem(ACTIVE_BACKTEST_JOB_KEY) === jobId) {
      localStorage.removeItem(ACTIVE_BACKTEST_JOB_KEY);
    }
  } catch (_) {}
}

async function restoreActiveBacktestJob() {
  let jobId = "";
  try {
    jobId = localStorage.getItem(ACTIVE_BACKTEST_JOB_KEY) || "";
  } catch (_) {
    jobId = "";
  }
  if (!jobId) return;
  setView("modelBacktestView");
  try {
    const job = await api(`/api/backtests/jobs/${encodeURIComponent(jobId)}`);
    showBacktestJobStatus(job);
    await loadBacktestJobs();
    if (job.status === "done") {
      state.backtestResult = job.result || await api(`/api/backtests/runs/${encodeURIComponent(job.run_id)}`);
      state.visibleBacktestModels = new Set(state.backtestResult.profile_ids || state.backtestResult.model_ids || []);
      renderBacktestResult();
      forgetActiveBacktestJob(jobId);
      await loadBacktestHistory();
      return;
    }
    if (job.status === "failed") {
      els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>回测失败</h4><p>${escapeHtml(job.error || "后台任务失败")}</p></div>`;
      state.displayedBacktestJobId = null;
      forgetActiveBacktestJob(jobId);
      await loadBacktestHistory();
      return;
    }
    if (job.status === "interrupted") {
      els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>回测任务已中断</h4><p>${escapeHtml(job.error || "服务重启后无法继续这个后台任务")}</p></div>`;
      state.displayedBacktestJobId = null;
      forgetActiveBacktestJob(jobId);
      return;
    }
    pollBacktestJob(jobId);
  } catch (error) {
    forgetActiveBacktestJob(jobId);
    state.displayedBacktestJobId = null;
    els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>回测任务未找到</h4><p>${escapeHtml(error.message)}</p></div>`;
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderBacktestJobStatus(job) {
  const summary = job.summary || {};
  els.backtestResult.classList.remove("empty");
  els.backtestResult.innerHTML = `
    <div class="result-header">
      <div>
        <h4>后台回测任务 ${escapeHtml(job.job_id)}</h4>
        <p>${escapeHtml(job.status)}${job.run_id ? ` / run ${job.run_id}` : ""}</p>
      </div>
      <span class="result-status neutral">${escapeHtml(job.status)}</span>
    </div>
    ${formatKeyValueGrid([
      ["创建时间", job.created_at || "-"],
      ["开始时间", job.started_at || "-"],
      ["结束时间", job.finished_at || "-"],
      ["环境变量", formatEnvVars(job.request?.env_vars)],
      ["完成组合", summary.done ?? "-"],
      ["失败组合", summary.failed ?? "-"],
    ])}
  `;
}

function clearBacktestResult() {
  state.backtestResult = null;
  state.displayedBacktestJobId = null;
  state.visibleBacktestModels = new Set();
  els.backtestResult.classList.add("empty");
  els.backtestResult.innerHTML = "尚未运行回测。";
  renderBacktestExportActions(null);
  els.backtestEpisodeSelect.innerHTML = "";
  els.backtestDimSelect.innerHTML = "";
  state.visibleBacktestDims = new Set();
  state.backtestDimsInitialized = false;
  resetBacktestChartZoom();
  els.backtestSeriesToggles.innerHTML = "";
  drawBacktestChart();
}

function renderBacktestResult() {
  const run = state.backtestResult;
  if (!run) return;
  state.displayedBacktestJobId = null;
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
        <p>${escapeHtml((run.dataset_paths || []).join(" · "))}</p>
      </div>
      <span class="result-status ${summary.failed ? "fail" : "ok"}">${summary.failed ? "部分失败" : "完成"}</span>
    </div>
    ${formatKeyValueGrid(rows)}
    <div class="backtest-matrix">
      ${(run.results || []).map((item) => `
        <div class="backtest-cell ${item.status === "done" ? "ok" : "fail"}">
          <strong>${escapeHtml(item.profile_name || modelName(item.profile_id || item.model_id))}</strong>
          <span>${escapeHtml(item.dataset_name || datasetName(item.dataset_path))} / Episode ${escapeHtml(item.episode_index)} · ${escapeHtml(item.status)}</span>
          ${item.metrics ? `<small>MAE ${escapeHtml(item.metrics.mae)} · RMSE ${escapeHtml(item.metrics.rmse)}</small>` : `<small>${escapeHtml(item.error || "")}</small>`}
        </div>
      `).join("")}
    </div>
  `;
  renderBacktestExportActions(run);
  populateBacktestChartControls();
  drawBacktestChart();
}

function renderBacktestExportActions(run) {
  if (!els.backtestExportActions) return;
  if (!run?.run_id) {
    els.backtestExportActions.classList.add("empty");
    els.backtestExportActions.innerHTML = "运行回测后可导出报告。";
    return;
  }
  const base = `/api/backtests/runs/${encodeURIComponent(run.run_id)}/export`;
  const actionBase = `/api/backtests/runs/${encodeURIComponent(run.run_id)}/actions/export`;
  const actionLinks = (run.results || [])
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => item.status === "done" && item.series?.length)
    .map(({ item, index }) => {
      const label = `${item.profile_name || modelName(item.profile_id || item.model_id)} / ${item.dataset_name || datasetName(item.dataset_path)} / Episode ${item.episode_index}`;
      return `<a href="${actionBase}?result_index=${encodeURIComponent(index)}" download>${escapeHtml(label)}</a>`;
    });
  els.backtestExportActions.classList.remove("empty");
  els.backtestExportActions.innerHTML = `
    <span>导出报告</span>
    <a href="${base}?format=html" target="_blank" rel="noopener">HTML</a>
    <a href="${base}?format=csv" target="_blank" rel="noopener">CSV</a>
    <a href="${base}?format=json" target="_blank" rel="noopener">JSON</a>
    ${actionLinks.length ? `
      <span>Action 明细</span>
      <a href="${actionBase}" download>批量 ZIP</a>
      <details class="export-details">
        <summary>单组 CSV</summary>
        <div class="export-link-list">${actionLinks.join("")}</div>
      </details>
    ` : ""}
  `;
}

async function loadBacktestHistory() {
  if (!els.backtestHistory) return;
  try {
    const runs = await api("/api/backtests/runs?limit=50");
    renderBacktestHistory(runs);
  } catch (error) {
    els.backtestHistory.classList.add("empty");
    els.backtestHistory.textContent = `读取回测历史失败：${error.message}`;
  }
}

function renderBacktestHistory(runs) {
  if (!els.backtestHistory) return;
  if (!runs.length) {
    els.backtestHistory.classList.add("empty");
    els.backtestHistory.innerHTML = "暂无回测历史。";
    return;
  }
  els.backtestHistory.classList.remove("empty");
  els.backtestHistory.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>时间</th>
          <th>Run</th>
          <th>数据集</th>
          <th>模型数</th>
          <th>Episode</th>
          <th>完成</th>
          <th>失败</th>
          <th>平均 MAE</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        ${runs.map((run) => {
          const summary = run.summary || {};
          return `
            <tr>
              <td>${escapeHtml(run.created_at || "-")}</td>
              <td><code>${escapeHtml(run.run_id)}</code></td>
              <td>${escapeHtml((run.dataset_paths || []).map(datasetName).join(", ") || "-")}</td>
              <td>${escapeHtml((run.profile_ids || run.model_ids || []).length)}</td>
              <td>${escapeHtml((run.episodes || []).length)}</td>
              <td>${escapeHtml(summary.done ?? "-")}</td>
              <td>${escapeHtml(summary.failed ?? "-")}</td>
              <td>${escapeHtml(summary.mean_mae ?? "-")}</td>
              <td>
                <button type="button" data-backtest-run-id="${escapeAttr(run.run_id)}">查看</button>
              </td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
}

async function loadBacktestRun(runId) {
  const run = await api(`/api/backtests/runs/${encodeURIComponent(runId)}`);
  state.displayedBacktestJobId = null;
  state.backtestResult = run;
  state.visibleBacktestModels = new Set(run.profile_ids || run.model_ids || []);
  renderBacktestResult();
}

function handleBacktestHistoryClick(event) {
  const runId = event.target?.dataset?.backtestRunId;
  if (!runId) return;
  loadBacktestRun(runId).catch((error) => {
    els.backtestResult.classList.remove("empty");
    els.backtestResult.innerHTML = `<div class="result-section result-error"><h4>读取历史结果失败</h4><p>${escapeHtml(error.message)}</p></div>`;
  });
}

function populateBacktestChartControls() {
  const done = doneBacktestResults();
  const episodeMap = new Map();
  for (const item of done) {
    const key = backtestResultEpisodeKey(item);
    if (!episodeMap.has(key)) {
      episodeMap.set(key, `${item.dataset_name || datasetName(item.dataset_path)} / Episode ${item.episode_index}`);
    }
  }
  const previousEpisode = els.backtestEpisodeSelect.value;
  els.backtestEpisodeSelect.innerHTML = Array.from(episodeMap.entries())
    .map(([key, label]) => `<option value="${escapeAttr(key)}">${escapeHtml(label)}</option>`)
    .join("");
  if (previousEpisode && episodeMap.has(previousEpisode)) {
    els.backtestEpisodeSelect.value = previousEpisode;
  }
  renderBacktestDimChoices(chartActionDimCountForEpisode(selectedBacktestEpisodeKey()));
  const modelMap = new Map(state.models.map((model) => [model.id, model.name]));
  for (const item of done) {
    const id = item.profile_id || item.model_id;
    if (!modelMap.has(id)) modelMap.set(id, item.profile_name || id);
  }
  els.backtestSeriesToggles.innerHTML = Array.from(modelMap.entries()).map(([id, name]) => `
    <label class="series-option">
      <input type="checkbox" value="${escapeAttr(id)}" ${state.visibleBacktestModels.has(id) ? "checked" : ""}>
      <span>${escapeHtml(name)}</span>
    </label>
  `).join("");
}

function renderBacktestDimChoices(dims) {
  if (!els.backtestDimSelect) return;
  if (!dims) {
    els.backtestDimSelect.classList.add("empty");
    els.backtestDimSelect.innerHTML = "暂无 action";
    state.visibleBacktestDims = new Set();
    state.backtestDimsInitialized = false;
    return;
  }
  const previous = new Set(Array.from(state.visibleBacktestDims).filter((dim) => dim < dims));
  if (!state.backtestDimsInitialized && !previous.size) {
    for (let index = 0; index < dims; index += 1) previous.add(index);
    state.backtestDimsInitialized = true;
  }
  state.visibleBacktestDims = previous;
  els.backtestDimSelect.classList.remove("empty");
  els.backtestDimSelect.innerHTML = Array.from({ length: dims }, (_, index) => `
    <label class="action-dim-option">
      <input type="checkbox" value="${index}" ${previous.has(index) ? "checked" : ""}>
      <span>action[${index}]</span>
    </label>
  `).join("");
}

function selectedBacktestDims() {
  const checked = Array.from(els.backtestDimSelect?.querySelectorAll("input:checked") || [])
    .map((input) => Number(input.value))
    .filter((value) => Number.isInteger(value));
  state.visibleBacktestDims = new Set(checked);
  state.backtestDimsInitialized = true;
  return checked;
}

function setBacktestDimSelection(selectAll) {
  const boxes = Array.from(els.backtestDimSelect?.querySelectorAll("input[type='checkbox']") || []);
  boxes.forEach((box) => {
    box.checked = selectAll;
  });
  state.visibleBacktestDims = new Set(selectAll ? boxes.map((box) => Number(box.value)) : []);
  state.backtestDimsInitialized = true;
  drawBacktestChart();
}

function doneBacktestResults() {
  return (state.backtestResult?.results || []).filter((item) => item.status === "done");
}

function backtestResultEpisodeKey(item) {
  return item.episode_key || `${item.dataset_path}::${item.episode_index}`;
}

function selectedBacktestEpisodeKey() {
  return els.backtestEpisodeSelect?.value || "";
}

function doneResultsForEpisode(episodeKey) {
  if (!episodeKey) return [];
  return doneBacktestResults().filter((item) => backtestResultEpisodeKey(item) === episodeKey);
}

function chartActionDimCountForEpisode(episodeKey) {
  return Math.max(0, ...doneResultsForEpisode(episodeKey).map((item) => item.series?.length || 0));
}

function selectedChartResults() {
  const episodeKey = selectedBacktestEpisodeKey();
  const visible = new Set(Array.from(els.backtestSeriesToggles.querySelectorAll("input:checked")).map((input) => input.value));
  state.visibleBacktestModels = visible;
  return doneResultsForEpisode(episodeKey).filter((item) => visible.has(item.profile_id || item.model_id));
}

function resetBacktestChartZoom() {
  state.backtestChartStart = 0;
  state.backtestChartEnd = null;
}

function zoomBacktestChart(factor, anchorRatio = 0.5) {
  const results = doneResultsForEpisode(selectedBacktestEpisodeKey());
  const dims = selectedBacktestDims();
  const maxLength = Math.max(
    0,
    ...results.flatMap((item) => dims.map((dim) => item.series?.[dim]?.ground_truth?.length || 0)),
  );
  if (maxLength <= 1) return;
  const current = backtestChartWindow([{ values: Array.from({ length: maxLength }, (_, index) => index) }]);
  const span = Math.max(current.end - current.start, 1);
  const nextSpan = Math.max(2, Math.min(maxLength, Math.round(span * factor)));
  const anchor = current.start + span * Math.min(Math.max(anchorRatio, 0), 1);
  let start = Math.round(anchor - nextSpan * anchorRatio);
  start = Math.max(0, Math.min(start, maxLength - nextSpan));
  state.backtestChartStart = start;
  state.backtestChartEnd = start + nextSpan;
  drawBacktestChart();
}

async function loadBacktestJobs() {
  if (!els.backtestJobQueue) return;
  try {
    const jobs = await api("/api/backtests/jobs");
    renderBacktestJobs(jobs);
  } catch (error) {
    els.backtestJobQueue.classList.add("empty");
    els.backtestJobQueue.textContent = `读取回测队列失败：${error.message}`;
  }
}

function renderBacktestJobs(jobs) {
  if (!els.backtestJobQueue) return;
  if (!jobs.length) {
    els.backtestJobQueue.classList.add("empty");
    els.backtestJobQueue.innerHTML = "暂无排队任务。";
    return;
  }
  const ordered = jobs.slice().sort((a, b) => {
    const rank = { running: 0, queued: 1, failed: 2, done: 3 };
    const diff = (rank[a.status] ?? 9) - (rank[b.status] ?? 9);
    if (diff) return diff;
    return String(b.created_at || "").localeCompare(String(a.created_at || ""));
  });
  els.backtestJobQueue.classList.remove("empty");
  els.backtestJobQueue.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>状态</th>
          <th>Job</th>
          <th>Profiles</th>
          <th>Episodes</th>
          <th>Env</th>
          <th>创建</th>
          <th>开始</th>
          <th>结束</th>
          <th>结果</th>
        </tr>
      </thead>
      <tbody>
        ${ordered.map((job) => {
          const request = job.request || {};
          const profileIds = request.profile_ids || request.model_ids || [];
          const episodes = request.episodes || [];
          return `
            <tr>
              <td><span class="result-status ${job.status === "done" ? "ok" : job.status === "failed" ? "fail" : "neutral"}">${escapeHtml(job.status)}</span></td>
              <td><code>${escapeHtml(job.job_id)}</code></td>
              <td>${escapeHtml(profileIds.join(", ") || "-")}</td>
              <td>${escapeHtml(episodes.length)}</td>
              <td>${escapeHtml(formatEnvVars(request.env_vars))}</td>
              <td>${escapeHtml(job.created_at || "-")}</td>
              <td>${escapeHtml(job.started_at || "-")}</td>
              <td>${escapeHtml(job.finished_at || "-")}</td>
              <td>${job.run_id ? `<button type="button" data-backtest-run-id="${escapeAttr(job.run_id)}">查看</button>` : escapeHtml(job.error || "-")}</td>
            </tr>
          `;
        }).join("")}
      </tbody>
    </table>
  `;
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
  const episodeResults = doneResultsForEpisode(selectedBacktestEpisodeKey());
  const results = selectedChartResults();
  const dims = selectedBacktestDims();
  if (!episodeResults.length || !dims.length) {
    ctx.fillStyle = "#5f6c72";
    ctx.font = "14px sans-serif";
    ctx.fillText("运行回测后选择 episode 和 action 维度。", 24, 34);
    return;
  }
  const lines = [];
  const colors = ["#087f8c", "#b76e00", "#2f6fbb", "#7a5195", "#c92a2a", "#2f9e44"];
  const truthSource = episodeResults.find((item) => item.series?.length);
  dims.forEach((dim, dimIndex) => {
    if (els.showGroundTruth.checked && truthSource?.series?.[dim]) {
      lines.push({
        name: `action[${dim}] truth`,
        values: truthSource.series[dim].ground_truth,
        color: dimColor(dimIndex, 0),
        dim,
      });
    }
    results.forEach((item, resultIndex) => {
      const series = item.series?.[dim];
      if (!series) return;
      const modelLabel = item.profile_name || modelName(item.profile_id || item.model_id);
      lines.push({
        name: `${modelLabel} action[${dim}] pred`,
        values: series.predicted,
        color: colors[(resultIndex + dimIndex) % colors.length],
        dim,
      });
      if (els.showBacktestError.checked) {
        lines.push({
          name: `${modelLabel} action[${dim}] err`,
          values: series.error,
          color: colors[(resultIndex + dimIndex + 2) % colors.length],
          dashed: true,
          dim,
        });
      }
    });
  });
  if (!lines.length) {
    ctx.fillStyle = "#5f6c72";
    ctx.font = "14px sans-serif";
    ctx.fillText("当前筛选没有可绘制的 action 曲线。", 24, 34);
    return;
  }
  drawLineChart(ctx, width, height, lines, backtestChartWindow(lines));
}

function dimColor(index, offset = 0) {
  const colors = ["#111827", "#087f8c", "#b76e00", "#2f6fbb", "#7a5195", "#c92a2a", "#2f9e44"];
  return colors[(index + offset) % colors.length];
}

function backtestChartWindow(lines) {
  const maxLength = Math.max(0, ...lines.map((line) => line.values.length));
  if (!maxLength) return { start: 0, end: 0, maxLength: 0 };
  if (state.backtestChartEnd === null || state.backtestChartEnd > maxLength) {
    state.backtestChartStart = 0;
    state.backtestChartEnd = maxLength;
  }
  const minSpan = Math.min(maxLength, Math.max(2, Math.ceil(maxLength / 100)));
  let start = Math.max(0, Math.min(Math.floor(state.backtestChartStart || 0), maxLength - minSpan));
  let end = Math.max(start + minSpan, Math.min(Math.ceil(state.backtestChartEnd || maxLength), maxLength));
  if (end > maxLength) {
    end = maxLength;
    start = Math.max(0, end - minSpan);
  }
  state.backtestChartStart = start;
  state.backtestChartEnd = end;
  return { start, end, maxLength };
}

function drawLineChart(ctx, width, height, lines, windowSpec = null) {
  const pad = { left: 64, right: 22, top: 24, bottom: 76 };
  const start = windowSpec?.start ?? 0;
  const end = windowSpec?.end ?? Math.max(0, ...lines.map((line) => line.values.length));
  const values = lines.flatMap((line) => line.values.slice(start, end).filter((value) => value !== null && value !== undefined));
  if (!values.length) return;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.left, pad.top, chartWidth, chartHeight);
  drawChartAxes(ctx, width, height, pad, min, max, start, end, windowSpec?.maxLength || end);
  for (const line of lines) {
    ctx.beginPath();
    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2;
    ctx.setLineDash(line.dashed ? [6, 4] : []);
    line.values.slice(start, end).forEach((value, offset) => {
      const index = start + offset;
      const x = pad.left + (offset / Math.max(end - start - 1, 1)) * chartWidth;
      const y = pad.top + (1 - ((value - min) / span)) * chartHeight;
      if (offset === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.setLineDash([]);
  drawChartLegend(ctx, width, height, pad, lines);
}

function drawChartAxes(ctx, width, height, pad, min, max, start, end, maxLength) {
  const chartWidth = width - pad.left - pad.right;
  const chartHeight = height - pad.top - pad.bottom;
  ctx.save();
  ctx.font = "12px sans-serif";
  ctx.fillStyle = "#475569";
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;
  const yTicks = 5;
  for (let i = 0; i <= yTicks; i += 1) {
    const ratio = i / yTicks;
    const y = pad.top + ratio * chartHeight;
    const value = max - ratio * (max - min);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(formatAxisValue(value), 8, y + 4);
  }
  const xTicks = 6;
  for (let i = 0; i <= xTicks; i += 1) {
    const ratio = i / xTicks;
    const x = pad.left + ratio * chartWidth;
    const frame = Math.round(start + ratio * Math.max(end - start - 1, 0));
    ctx.beginPath();
    ctx.moveTo(x, height - pad.bottom);
    ctx.lineTo(x, height - pad.bottom + 5);
    ctx.stroke();
    ctx.fillText(String(frame), x - 10, height - pad.bottom + 20);
  }
  ctx.fillStyle = "#334155";
  ctx.fillText("frame", width - pad.right - 34, height - pad.bottom + 38);
  ctx.save();
  ctx.translate(18, pad.top + 12);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("action value", 0, 0);
  ctx.restore();
  if (maxLength && end - start < maxLength) {
    ctx.fillText(`显示 ${start}-${Math.max(start, end - 1)} / ${maxLength - 1}`, pad.left, pad.top - 8);
  }
  ctx.restore();
}

function formatAxisValue(value) {
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.001)) return value.toExponential(1);
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

function drawChartLegend(ctx, width, height, pad, lines) {
  ctx.font = "12px sans-serif";
  const maxItems = Math.min(lines.length, 12);
  let x = pad.left;
  let y = height - 42;
  for (let index = 0; index < maxItems; index += 1) {
    const line = lines[index];
    const label = line.name.length > 34 ? `${line.name.slice(0, 31)}...` : line.name;
    const itemWidth = Math.min(220, Math.max(120, ctx.measureText(label).width + 26));
    if (x + itemWidth > width - pad.right) {
      x = pad.left;
      y += 18;
    }
    ctx.fillStyle = line.color;
    ctx.fillRect(x, y - 9, 14, 3);
    ctx.fillText(label, x + 20, y - 4);
    x += itemWidth;
  }
  if (lines.length > maxItems) {
    ctx.fillStyle = "#64748b";
    ctx.fillText(`+${lines.length - maxItems} 条曲线`, x, y - 4);
  }
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
  state._fbOnSelect = onSelect;
  state._fbTarget = targetInput;
  const startDir = (targetInput && targetInput.value) ? targetInput.value.trim() : "";
  navigateFolderBrowser(startDir);
}

function closeFolderBrowser() {
  if (els.folderBrowser) els.folderBrowser.style.display = "none";
}

async function navigateFolderBrowser(dir) {
  try {
    if (!els.folderBrowserList || !els.folderBrowserCurrent) return;
    const result = await api("/api/path/suggest?path=" + encodeURIComponent(dir));
    state._fbBase = result.base || dir;
    state._fbItems = result.items || [];
    els.folderBrowserCurrent.textContent = result.base || dir || "/";
    els.folderBrowserPath.value = "";
    if (!state._fbItems.length) {
      els.folderBrowserList.innerHTML = "<span class=\"fb-empty\">此目录下没有子文件夹</span>";
      return;
    }
    const html = state._fbItems.map(function (item) {
      if (item.is_dir) {
        return `<button class="fb-item" type="button" data-path="${escapeAttr(item.path)}">
          <span>📁 ${escapeHtml(item.name)}</span>
          ${item.has_dataset_marker ? "<strong>dataset</strong>" : ""}
        </button>`;
      }
      return `<div class="fb-item fb-file">
        <span>📄 ${escapeHtml(item.name)}</span>
      </div>`;
    }).join("");
    els.folderBrowserList.innerHTML = html;
    const buttons = els.folderBrowserList.querySelectorAll(".fb-item:not(.fb-file)");
    for (let i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", (function (path) {
        return function () { navigateFolderBrowser(path); };
      })(buttons[i].dataset.path));
    }
  } catch (error) {
    els.folderBrowserList.innerHTML = "<span class=\"fb-empty\">" + escapeHtml(error.message) + "</span>";
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
document.querySelectorAll(".root-nav-button").forEach((button) => {
  button.addEventListener("click", () => setRoot(button.dataset.root));
});

els.loadDataset.addEventListener("click", openDataset);
els.datasetPath.addEventListener("keydown", (event) => {
  if (event.key === "Enter") openDataset();
});
els.datasetPath.addEventListener("input", schedulePathSuggestions);
els.datasetPath.addEventListener("focus", schedulePathSuggestions);
els.datasetPath.addEventListener("blur", () => setTimeout(hideSuggestions, 120));
if (els.browseRoot) els.browseRoot.addEventListener("click", () => {
  window._openRootBrowser && window._openRootBrowser();
});
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
if (els.loadOperationLogs) els.loadOperationLogs.addEventListener("click", loadOperationLogs);
els.strictValidateDataset.addEventListener("click", strictValidateCurrentDataset);
els.runEditDryRun.addEventListener("click", runEditDryRun);
els.applyEditPlan.addEventListener("click", applyEditPlan);
els.addCurrentDatasetToMerge.addEventListener("click", addCurrentDatasetToMerge);
// addMergePathBtn uses onclick in HTML for robustness
els.clearMergeList.addEventListener("click", clearMergeList);
els.validateMerge.addEventListener("click", validateMergePlan);
els.applyMerge.addEventListener("click", applyMergePlan);

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
  const current = (state._fbBase || "").trim();
  // Windows 盘符根目录（D:\）→ 退回盘符列表
  if (/^[A-Za-z]:[\\/]$/.test(current)) {
    navigateFolderBrowser("");
    return;
  }
  // Unix 根目录 / → 不再往上
  if (current === "/") {
    navigateFolderBrowser("/");
    return;
  }
  // 先去掉末尾分隔符再找上级
  const stripped = current.replace(/[\\/]+$/, "");
  const idx = Math.max(stripped.lastIndexOf("/"), stripped.lastIndexOf("\\"));
  let parent;
  if (idx > 0) {
    parent = stripped.substring(0, idx);
    // Windows: D: → D:\
    if (/^[A-Za-z]:$/.test(parent)) parent += "\\";
  } else if (idx === 0) {
    // Unix: /home → /
    parent = "/";
  } else {
    // 无分隔符（如裸 D:）→ 退回盘符列表
    if (/^[A-Za-z]:$/.test(stripped)) {
      navigateFolderBrowser("");
      return;
    }
    parent = stripped;
  }
  navigateFolderBrowser(parent);
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
if (els.profileTemplate) els.profileTemplate.addEventListener("change", () => applySelectedTemplate(true));
els.registerModel.addEventListener("click", registerCurrentModel);
if (els.profileSave) els.profileSave.addEventListener("click", saveCurrentProfile);
if (els.profileInspect) els.profileInspect.addEventListener("click", () => runProfileEditorAction("inspect"));
if (els.profileLoad) els.profileLoad.addEventListener("click", () => runProfileEditorAction("load"));
if (els.profileUnload) els.profileUnload.addEventListener("click", () => runProfileEditorAction("unload"));
if (els.profileTest) els.profileTest.addEventListener("click", () => runProfileEditorAction("test"));
if (els.profileDelete) els.profileDelete.addEventListener("click", () => runProfileEditorAction("delete"));

// browseCheckpoint uses onclick in HTML for robustness

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
if (els.backtestSelectionTable) {
  els.backtestSelectionTable.addEventListener("click", (event) => {
    const key = event.target?.dataset?.backtestRemove;
    if (key) removeBacktestEpisode(key);
  });
}
if (els.clearBacktestSelection) els.clearBacktestSelection.addEventListener("click", clearBacktestSelection);
els.runBacktest.addEventListener("click", runSelectedBacktest);
els.clearBacktest.addEventListener("click", clearBacktestResult);
if (els.refreshBacktestJobs) els.refreshBacktestJobs.addEventListener("click", loadBacktestJobs);
if (els.refreshBacktestHistory) els.refreshBacktestHistory.addEventListener("click", loadBacktestHistory);
if (els.backtestHistory) els.backtestHistory.addEventListener("click", handleBacktestHistoryClick);
if (els.backtestJobQueue) els.backtestJobQueue.addEventListener("click", handleBacktestHistoryClick);
if (els.addRuntimeParam) els.addRuntimeParam.addEventListener("click", () => addParamRow(els.profileRuntimeParams));
if (els.addExtraParam) els.addExtraParam.addEventListener("click", () => addParamRow(els.profileExtraParams));
if (els.profileRuntimeParams) els.profileRuntimeParams.addEventListener("click", handleParamTableClick);
if (els.profileExtraParams) els.profileExtraParams.addEventListener("click", handleParamTableClick);
if (els.backtestEnvVars) {
  els.backtestEnvVars.addEventListener("click", (event) => {
    handleParamTableClick(event);
    saveBacktestEnvVars();
  });
  els.backtestEnvVars.addEventListener("input", saveBacktestEnvVars);
  els.backtestEnvVars.addEventListener("change", saveBacktestEnvVars);
}
if (els.refreshTrainingRecipes) els.refreshTrainingRecipes.addEventListener("click", loadTrainingRecipes);
if (els.checkTrainingEnv) els.checkTrainingEnv.addEventListener("click", loadTrainingEnv);
if (els.trainingTemplate) els.trainingTemplate.addEventListener("change", () => applyTrainingTemplate(true));
if (els.addTrainingEnvVar) els.addTrainingEnvVar.addEventListener("click", addTrainingEnvVar);
if (els.createTrainingRecipe) els.createTrainingRecipe.addEventListener("click", createTrainingRecipe);
if (els.saveTrainingRecipe) els.saveTrainingRecipe.addEventListener("click", saveTrainingRecipe);
if (els.inspectTrainingRecipe) els.inspectTrainingRecipe.addEventListener("click", () => runTrainingRecipeAction("inspect"));
if (els.submitTrainingJob) els.submitTrainingJob.addEventListener("click", () => runTrainingRecipeAction("submit"));
if (els.deleteTrainingRecipe) els.deleteTrainingRecipe.addEventListener("click", () => runTrainingRecipeAction("delete"));
if (els.trainingRecipeList) els.trainingRecipeList.addEventListener("click", handleTrainingRecipeAction);
if (els.addTrainingHyperparam) els.addTrainingHyperparam.addEventListener("click", () => addParamRow(els.trainingHyperparams));
if (els.addTrainingExtraParam) els.addTrainingExtraParam.addEventListener("click", () => addParamRow(els.trainingExtraParams));
if (els.trainingHyperparams) els.trainingHyperparams.addEventListener("click", handleParamTableClick);
if (els.trainingEnvVars) els.trainingEnvVars.addEventListener("click", handleParamTableClick);
if (els.trainingExtraParams) els.trainingExtraParams.addEventListener("click", handleParamTableClick);
if (els.refreshTrainingJobs) els.refreshTrainingJobs.addEventListener("click", loadTrainingJobs);
if (els.trainingJobList) els.trainingJobList.addEventListener("click", handleTrainingJobAction);
if (els.createTrainingPipeline) els.createTrainingPipeline.addEventListener("click", createTrainingPipeline);
if (els.refreshTrainingPipelines) els.refreshTrainingPipelines.addEventListener("click", loadTrainingPipelines);
els.backtestEpisodeSelect.addEventListener("change", () => {
  state.visibleBacktestDims = new Set();
  state.backtestDimsInitialized = false;
  renderBacktestDimChoices(chartActionDimCountForEpisode(selectedBacktestEpisodeKey()));
  resetBacktestChartZoom();
  drawBacktestChart();
});
els.backtestDimSelect.addEventListener("change", drawBacktestChart);
if (els.selectAllBacktestDims) els.selectAllBacktestDims.addEventListener("click", () => setBacktestDimSelection(true));
if (els.clearBacktestDims) els.clearBacktestDims.addEventListener("click", () => setBacktestDimSelection(false));
els.showGroundTruth.addEventListener("change", drawBacktestChart);
els.showBacktestError.addEventListener("change", drawBacktestChart);
els.backtestSeriesToggles.addEventListener("change", drawBacktestChart);
if (els.zoomBacktestIn) els.zoomBacktestIn.addEventListener("click", () => zoomBacktestChart(0.5));
if (els.zoomBacktestOut) els.zoomBacktestOut.addEventListener("click", () => zoomBacktestChart(2));
if (els.resetBacktestZoom) els.resetBacktestZoom.addEventListener("click", () => {
  resetBacktestChartZoom();
  drawBacktestChart();
});
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
if (els.backtestChart) els.backtestChart.addEventListener("wheel", (event) => {
  event.preventDefault();
  const rect = els.backtestChart.getBoundingClientRect();
  const ratio = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0.5;
  zoomBacktestChart(event.deltaY < 0 ? 0.8 : 1.25, ratio);
}, { passive: false });
window.addEventListener("resize", drawChart);
window.addEventListener("resize", drawBacktestChart);

initBacktestEnvVars();
loadEnv().catch((error) => {
  els.envInfo.textContent = error.message;
});
loadHistory().catch(() => {});
loadModels().catch(() => {});
restoreActiveBacktestJob().catch(() => {});
