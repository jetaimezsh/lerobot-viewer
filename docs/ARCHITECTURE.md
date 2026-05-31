# 架构设计文档

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | Python 3.10+, FastAPI, Uvicorn |
| 前端 | 单页应用、Vanilla JS、HTML5、CSS3 |
| 数据 | Pandas, PyArrow (Parquet), NumPy |
| 视频 | ffmpeg / ffprobe（可选） |
| 模型推理 | PyTorch, LeRobot official SDK（仅 Linux） |

## 后端架构

```
app/
├── main.py          FastAPI 路由、请求模型、DatasetCache
├── editing.py       编辑引擎、合并引擎、视频处理
├── validation.py    v3.0 严格校验、官方 LeRobotDataset 加载
└── backtesting.py   模型注册、加载、回测
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
| GET | `/api/models` | 模型列表 |
| POST | `/api/models/register` | 注册模型 |
| POST | `/api/models/inspect` | 检查模型 |
| POST | `/api/models/load` | 加载模型 |
| POST | `/api/models/unload` | 卸载模型 |
| POST | `/api/models/delete` | 删除模型 |
| POST | `/api/backtests/run` | 运行回测 |

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
├── index.html    # 7 个 view + 侧边栏 + episode 面板 + edit 面板
├── app.js        # 全局 state + DOM refs + 所有交互逻辑
└── styles.css    # CSS 变量 + 响应式布局
```

### 页面（View）

| View ID | 说明 |
|---------|------|
| `overviewView` | 数据集总览、features、tasks、模型概览 |
| `episodeView` | Episode 播放、视频同步、时序图、编辑/导出标记 |
| `datasetEditView` | 编辑操作列表、预览预估、合并区域 |
| `modelManagerView` | 模型注册与管理 |
| `modelBacktestView` | 回测配置与 action 对比 |
| `envView` | 系统环境检测 |

### 全局状态（`state`）

```javascript
state = {
    datasetId, summary, episodes, history,
    episode, elapsed, series, selectedSeries,
    currentElapsed, duration,
    playing, primaryVideo, videos,
    chartStart, chartEnd, panMode,
    currentView,
    editMode,           // "edit" | "export"
    editOperations,     // 当前标记列表
    trimDraftStart, trimDraftEnd,
    models, modelEnv,
    backtestResult, visibleBacktestModels,
}
```

### 数据流

```
用户输入 → api() 函数 → FastAPI 路由 → 后端引擎 → JSON 响应
                                                      │
用户操作 ◄── 渲染 (render*) ◄── state 更新 ◄─────────┘
```
