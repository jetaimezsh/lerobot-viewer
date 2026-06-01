# 架构设计文档

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3.10+, FastAPI, Uvicorn |
| 前端 | 单页应用、Vanilla JS、HTML5、CSS3 |
| 数据 | Pandas, PyArrow (Parquet), NumPy |
| 视频 | ffmpeg / ffprobe（可选） |
| 模型推理 | PyTorch, LeRobot official SDK（仅 Linux） |
| 模型训练 | LeRobot CLI（真实训练）/ mock trainer（本地测试） |

## 后端架构

```
app/
├── main.py          FastAPI 路由、请求模型、DatasetCache
├── editing.py       编辑引擎、合并引擎、视频处理
├── validation.py    v3.0 严格校验、官方 LeRobotDataset 加载
├── adapters/        模型回测适配器
├── backtesting.py   profile-based 模型加载、回测、回测队列
├── trainers/        训练框架适配器
├── training_store.py     training recipe/job/pipeline JSON 存储
├── training_executor.py  单 worker 训练队列、子进程、日志、profile 桥接
├── training_pipeline.py  轻量训练流水线记录
└── operation_log.py JSONL 操作日志
```

### 数据流

```
load dataset ──► DatasetCache ──► 浏览 / 播放 API
                    │
                    ▼
          validate_edit_plan()
          validate_merge_compatibility()
                    │
                    ▼
          build_edited_dataset()   /  build_merged_dataset()
                    │
                    ▼
          write_edited_videos()    /  write_merged_videos()
                    │
                    ▼
          write_dataset() ──► frames.parquet + episodes.parquet
                              + info.json + stats.json
                    │
                    ▼
          validate_lerobot_v3_dataset()
          official_lerobot_validation()
```

### 关键模块

#### `app/editing.py` — 数据集修改引擎

核心函数：

| 函数 | 行号 | 用途 |
|------|------|------|
| `_classify_mode()` | ~96 | 根据操作列表判定 edit / select / mixed 模式 |
| `validate_edit_plan()` | ~116 | 校验编辑/选择计划，返回预测结果 |
| `_validate_edit_plan()` | ~260 | 原有编辑模式（delete + trim）校验 |
| `_validate_select_plan()` | ~220 | 选择导出模式（select_episode + select_episode_range）校验 |
| `build_edited_dataset()` | ~640 | 编辑/选择模式：重建 frames 和 episodes |
| `apply_edit_plan()` | ~460 | 完整编辑流程：校验 → 构建 → 视频写入 → 写盘 → 校验 |
| `build_merged_dataset()` | ~536 | 合并多数据集：task 去重、episode 重编号、video_jobs 构造 |
| `apply_merge_plan()` | ~410 | 完整合并流程 |
| `write_edited_videos()` | ~793 | 单源视频重写（segment 提取 + concat） |
| `write_merged_videos()` | ~928 | 多源视频拼接（跨 cache 定位源文件） |
| `write_dataset()` | ~1343 | 统一写盘：Parquet + info.json + stats.json |
| `_enforce_frame_int_dtypes()` | ~1286 | 帧数据 int 列 dtype 强制转换 |
| `_enforce_episode_int_dtypes()` | ~1258 | episode 元数据 int 列 dtype 强制转换 |

操作类型支持：

| Type | 模式 | 含义 |
|------|------|------|
| `delete_episode` | edit | 删除整个 episode |
| `trim_episode` | edit | 保留区间，丢弃其他 |
| `select_episode` | select | 导出整个 episode |
| `select_episode_range` | select | 导出区间 |

两种模式互斥，不能混合。

#### `app/validation.py` — v3.0 合规校验

info.json 字段分级（基于 LeRobot v3.0 官方规范）：

| 级别 | 字段 |
|------|------|
| **ERROR** | `codebase_version` (非 v3.0), `fps`, `features`, `data_path` 缺失或类型错误 |
| **ERROR** | `video_path` 缺失（有 video feature 时） |
| **ERROR** | `fps` 为 float 而非 int |
| **WARNING** | `chunks_size`, `data_files_size_in_mb`, `video_files_size_in_mb` 缺失 |
| **WARNING** | `total_episodes`, `total_frames`, `total_tasks` 缺失 |
| **WARNING** | video feature 缺少 `info` 子字典 |
| **WARNING** | `total_*`, `chunks_size` 为 float |

### API 路由表

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/` | 前端页面 |
| GET | `/api/env` | 后端环境信息 |
| GET | `/api/history` | 最近打开记录 |
| POST | `/api/datasets/open` | 加载数据集 |
| GET | `/api/datasets/{id}` | 数据集摘要 |
| GET | `/api/datasets/{id}/episodes` | episode 列表 |
| GET | `/api/datasets/{id}/episodes/{idx}` | episode 详情 |
| GET | `/api/datasets/{id}/video?ep_index=…` | 视频文件 |
| POST | `/api/datasets/validate` | 基础校验 |
| POST | `/api/datasets/strict-validate` | 严格校验 |
| POST | `/api/edit/tool-status` | 编辑工具检测 |
| POST | `/api/edit/dry-run` | 编辑预估 |
| POST | `/api/edit/apply` | 编辑执行 |
| POST | `/api/merge/validate` | 合并预估 |
| POST | `/api/merge/apply` | 合并执行 |
| GET | `/api/path/suggest` | 路径补全 |
| POST | `/api/env/install-requirements` | 安装依赖 |
| GET | `/api/profiles/env` | 模型回测环境与 worker 状态 |
| GET | `/api/profiles/templates` | 内置 profile 模板 |
| GET | `/api/profiles` | profile 列表 |
| POST | `/api/profiles` | 创建 profile |
| GET | `/api/profiles/{id}` | profile 详情 |
| PUT | `/api/profiles/{id}` | 更新 profile |
| DELETE | `/api/profiles/{id}` | 删除 profile |
| POST | `/api/profiles/{id}/inspect` | 检查 checkpoint 和参数 |
| POST | `/api/profiles/{id}/load` | 加载 profile |
| POST | `/api/profiles/{id}/unload` | 卸载 profile |
| POST | `/api/profiles/{id}/test` | 使用真实 episode 帧做快速推理测试 |
| POST | `/api/backtests/jobs` | 创建 profile-based 后台回测，任务进入单 worker 队列依次执行 |
| GET | `/api/backtests/jobs` | 回测任务队列 |
| GET | `/api/backtests/jobs/{job_id}` | 查询回测任务 |
| GET | `/api/backtests/runs` | 回测历史 |
| GET | `/api/backtests/runs/{run_id}/export` | 导出回测报告 |
| GET | `/api/train/env` | 训练环境和 worker 状态 |
| GET | `/api/train/frameworks` | 可用训练框架 |
| GET | `/api/train/templates` | 内置训练模板 |
| GET | `/api/train/recipes` | 训练配方列表 |
| POST | `/api/train/recipes` | 创建训练配方 |
| GET | `/api/train/recipes/{id}` | 训练配方详情 |
| PUT | `/api/train/recipes/{id}` | 更新训练配方 |
| DELETE | `/api/train/recipes/{id}` | 删除训练配方 |
| POST | `/api/train/recipes/{id}/inspect` | 检查训练配方 |
| GET | `/api/train/jobs` | 训练作业队列 |
| POST | `/api/train/jobs` | 提交训练作业 |
| GET | `/api/train/jobs/{id}` | 训练作业详情 |
| POST | `/api/train/jobs/{id}/cancel` | 取消训练作业 |
| POST | `/api/train/jobs/{id}/requeue` | 重新排队训练作业 |
| DELETE | `/api/train/jobs/{id}` | 删除训练作业和日志 |
| GET | `/api/train/jobs/{id}/log` | 读取训练日志 |
| PUT | `/api/train/queue` | 调整 queued 作业顺序 |
| GET | `/api/train/pipelines` | 流水线列表 |
| POST | `/api/train/pipelines` | 创建轻量训练流水线 |
| GET | `/api/train/pipelines/{id}` | 流水线详情 |
| GET | `/api/operations/logs` | 读取操作日志 |

### 操作日志

`app/operation_log.py` 提供统一的 `log_operation()` 和 `read_operation_logs()`。日志文件为 `logs/operations.jsonl`，每行一条 JSON 记录：

```json
{"timestamp":"...","action":"edit_apply","status":"success","target":"D:/datasets/src","details":{"output_path":"D:/datasets/out"}}
```

日志只记录操作摘要和错误信息，不参与主业务流程；写日志失败不会阻断数据编辑、合并或回测。

### 视频处理流程

```
video_jobs = [
    {
        "cache_index": 0,           # 源数据集索引（merge 模式）
        "source_episode_index": 3,
        "source_episode": Series,   # 源 episode 行
        "new_episode_index": 0,     # 输出 episode 索引
        "start_frame": 0,           # 从源 episode 的第几帧开始
        "end_frame": 100,           # 到源 episode 的第几帧结束
        "length": 100,              # 输出帧数
    },
    ...
]

for video_key in video_keys:
    ├── 探测源视频编码（ffprobe）
    ├── 检查编码器可用性
    ├── for job in video_jobs:
    │   ├── ffmpeg -ss <start_time> -t <duration> → segment-{i}.mp4
    │   └── 更新 episodes[from_timestamp / to_timestamp / chunk_index / file_index]
    └── ffmpeg concat → 最终输出视频
```

编码器回退优先级（按 codec）：

```
H.264: libx264 → h264_vaapi → h264_nvenc → h264_amf → h264_qsv → h264_v4l2m2m
HEVC:  libx265 → hevc_vaapi → hevc_nvenc → hevc_amf → hevc_qsv → hevc_v4l2m2m
AV1:   libaom-av1 → libsvtav1 → av1_vaapi → av1_nvenc → av1_amf → av1_qsv
VP9:   libvpx-vp9 → vp9_vaapi → vp9_qsv
```

每种编码器有专用的 ffmpeg 参数（`_build_h264_options` / `_build_hevc_options`），硬件编码器不传 `-preset` / `-crf`。

### 类型安全保障

为防止 Pandas `loc` 赋值 + `concat` 导致 int 列泄露为 float64，写盘前统一强制：

- 帧数据：`episode_index`, `frame_index`, `index`, `task_index`
- episode 元数据：`episode_index`, `length`, `task_index`, `dataset_from_index`, `dataset_to_index`, `data/chunk_index`, `data/file_index`, `videos/{key}/chunk_index`, `videos/{key}/file_index`

info.json 中 v3.0 spec 明确要求 int 的字段（`fps`, `chunks_size`）由 `_normalize_info_int_fields()` 在写盘前强制为 int。

## 前端架构

```
web/
├── index.html    # 2 个根工作台 + 子 view + 侧边栏 + episode/edit/model/train 面板
├── app.js        # 全局 state + DOM refs + 所有交互逻辑
└── styles.css    # CSS 变量 + 响应式布局
```

### 页面（Root / View）

| Root | View ID | 说明 |
|------|---------|------|
| `data` | `overviewView` | 数据集总览、features、tasks |
| `data` | `episodeView` | Episode 播放、视频同步、时序图、编辑/导出标记、加入回测样本池 |
| `data` | `datasetEditView` | 编辑操作列表、预览预估、合并区域 |
| `data` | `envView` | 系统环境检测 |
| `model` | `modelManagerView` | 模型注册与管理 |
| `model` | `modelBacktestView` | 回测样本池、模型选择、运行回测与 action 对比 |
| `model` | `trainingRecipeView` | 训练配方 CRUD、检查、提交训练 |
| `model` | `trainingQueueView` | 训练作业状态、日志、取消、重排、删除 |
| `model` | `trainingPipelineView` | 轻量训练→回测流水线记录 |

模型回测不再依赖文本框输入 episode，而是由数据查看页向 `state.backtestEpisodes` 写入结构化样本。每个样本包含 `dataset_path`、`dataset_name`、`episode_index`、`length`、`duration`、`fps`、`tasks` 和 `video_keys`，因此可以同时回测来自不同数据集的 episode。

### 全局状态（`state`）

```javascript
state = {
    datasetId, summary, episodes, history,
    episode, elapsed, series, selectedSeries,
    currentElapsed, duration,
    playing, primaryVideo, videos,
    chartStart, chartEnd, panMode,
    currentRoot, currentView,
    editMode,           // "edit" | "export"
    editOperations,     // 当前标记列表
    trimDraftStart, trimDraftEnd,
    models, modelEnv,
    backtestEpisodes,   // [{dataset_path, episode_index, ...}]
    backtestResult, visibleBacktestModels,
    trainingFrameworks, trainingTemplates, trainingRecipes,
    trainingJobs, trainingPipelines,
}
```

## v4.1 训练队列

训练队列与回测队列不同，所有作业都落盘到 `state/training_jobs/`：

```text
state/training_recipes/{recipe_id}.json
state/training_jobs/{job_id}.json
state/training_jobs/{job_id}.log
state/training_pipelines/{pipeline_id}.json
```

执行流程：

```text
Training Recipe
   │ POST /api/train/jobs
   ▼
Training Job JSON (queued)
   │ scheduler single worker
   ▼
subprocess.Popen(trainer.build_command(recipe))
   │ stdout/stderr
   ▼
job log + progress parse
   │ returncode == 0
   ▼
auto_create_profile() -> state/model_profiles/{profile_id}.json
```

真实训练使用 `lerobot_train` trainer 封装 `lerobot-train` CLI；本地自动化测试使用隐藏的 `mock` trainer，覆盖成功、失败、日志、队列和自动 profile 生成。

### 数据流

```
用户输入 → api() 函数 → FastAPI 路由 → 后端引擎 → JSON 响应
                                                      │
用户操作 ◄── 渲染 (render*) ◄── state 更新 ◄─────────┘
```
