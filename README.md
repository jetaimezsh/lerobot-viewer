# LeRobot Dataset v3.0 本地管理与编辑工具

基于 Python/FastAPI 的 LeRobot Dataset v3.0 数据集本地查看、编辑、合并与模型回测工具。前端为静态 HTML/CSS/JS，无需 Node.js。

## 功能概览

| 功能 | 说明 |
|------|------|
| **数据集浏览** | 加载 v3.0 数据集，展示总览、features、tasks、episode 列表 |
| **Episode 播放** | 多视频视角同步播放，数值时序图，帧级 state/action 值显示 |
| **数据集编辑** | 删除 episode、裁剪区间，保留源视频编码格式和关键帧节奏 |
| **选择导出** | 选择特定 episode 或区间，导出为独立新数据集 |
| **数据集合并** | 合并多个 schema 兼容的数据集（含视频），自动合并 task 表 |
| **严格校验** | 按 LeRobot v3.0 规范严格校验，支持官方 `LeRobotDataset` 加载验证 |
| **模型管理** | 注册 checkpoint、检查文件结构、加载/卸载模型 |
| **模型回测** | 离线 action 推理对比（MAE/RMSE/维度级误差），Linux 推理，Windows 管理 |
| **环境检测与日志** | conda/venv 状态、依赖版本、ffmpeg 可用性检查，记录关键操作日志 |

## 项目结构

```text
lerobot-viewer/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 应用、路径补全、历史记录
│   ├── editing.py       # 编辑/选择导出/合并引擎、视频处理
│   ├── validation.py    # v3.0 严格校验、官方 LeRobotDataset 加载验证
│   ├── backtesting.py   # 模型注册/加载/回测
│   └── operation_log.py # JSONL 操作日志
├── web/
│   ├── index.html       # 单页应用
│   ├── app.js           # 前端逻辑
│   └── styles.css       # 样式
├── scripts/
│   ├── setup_venv.sh / .ps1
│   ├── start_backend.sh / .ps1
│   ├── smoke_test.py
│   └── backtest_smoke_test.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET-V3-SPEC.md
│   └── USER-GUIDE.md
├── requirements.txt
├── .gitignore
└── README.md
```

## 支持的数据集格式

严格遵循 [LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3) 规范。数据集根目录结构：

```text
dataset/
├── data/
│   └── chunk-000/
│       └── file-000.parquet         # 多 episode 共享的帧数据
├── meta/
│   ├── info.json                    # 核心元数据
│   ├── stats.json                   # 全局归一化统计
│   ├── tasks.parquet                # 任务定义
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet     # episode 边界元数据
└── videos/
    └── {camera_key}/
        └── chunk-000/
            └── file-000.mp4         # 多 episode 共享的视频文件
```

格式参考：
- <https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3>
- <https://huggingface.co/docs/lerobot/en/porting_datasets_v3>

## 环境准备

### 依赖包

```text
fastapi==0.115.6
uvicorn[standard]==0.34.0
pandas==2.2.3
pyarrow==18.1.0
numpy==2.2.1
pydantic==2.10.4
```

可选安装（模型回测 / 官方校验）：

```bash
pip install lerobot torch safetensors
```

视频编辑需要 ffmpeg：

```bash
# Windows (Chocolatey)
choco install ffmpeg -y

# Linux
sudo apt install ffmpeg

# conda
conda install -c conda-forge ffmpeg -y
```

### Windows

```powershell
.\scripts\setup_venv.ps1
.\scripts\start_backend.ps1
# 打开 http://127.0.0.1:8000
```

### Linux / macOS

```bash
bash scripts/setup_venv.sh
bash scripts/start_backend.sh
# 打开 http://127.0.0.1:8000
```

### conda

```bash
conda create -n lerobot-viewer python=3.12 -y
conda activate lerobot-viewer
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## 使用方式

页面现在分为两个根工作台：

- **LeRobot 数据**：负责数据集加载、episode 播放、数据编辑、选择导出、合并和系统环境检测。
- **模型回测**：负责 checkpoint 管理、回测样本池、模型选择和 action 对比结果。

### 数据集浏览

1. 输入数据集根目录，点击"加载"
2. 左侧 episode 列表点击进入播放
3. 视频、时间轴、时序图、帧数值同步播放
4. 支持播放/暂停/拖动/倍速/慢速，Chart 区域支持缩放、平移
5. 数值型字段可在下拉菜单中勾选绘制

### 数据集编辑与选择导出

1. 在 Episode 播放页，通过顶部的 **删除/导出** 模式切换器选择模式
2. 点击"设为起点"、"设为终点"标记时间点
3. 点击"标记此 episode"标记整 episode，或"标记此区间"标记裁剪范围
4. 再次点击相同标记可取消
5. 切换到"数据集编辑"页面 → "预估修改结果"预览 → 填写输出目录 → "生成新数据集"

### 数据集合并

1. 在"数据集编辑"页面的合并区域输入数据集路径（每行一个）
2. 使用"加入当前数据集"快速添加已加载的数据集
3. "检查合并合法性"验证 schema 兼容性
4. 填写输出目录 → "生成合并数据集"

### 模型回测

1. "模型管理"页面注册 checkpoint（路径 + adapter 类型）
2. 检查 → 加载模型
3. 在 **LeRobot 数据 / Episode 播放** 页面点击"加入回测样本池"
4. 可以切换并加载其他数据集，继续加入不同数据集的 episode
5. 在 **模型回测 / 回测任务** 页面用表格确认样本所属数据集、路径、episode 编号、帧数、时长、任务和视频路数
6. 选择一个或多个模型 → 运行回测
7. 查看 MAE/RMSE / 维度级 action 对比曲线

回测 API 支持多数据集 episode 引用：

```json
{
  "model_ids": ["model-a", "model-b"],
  "episodes": [
    {"dataset_path": "D:/datasets/pusht", "episode_index": 0},
    {"dataset_path": "D:/datasets/pick", "episode_index": 4}
  ],
  "max_frames": 20
}
```

### 操作日志

关键操作会写入 `logs/operations.jsonl`，包括数据集打开、编辑 dry-run/apply、严格校验、合并校验/生成、模型注册/加载/删除和回测运行。

也可以在 **LeRobot 数据 / System 环境** 页面点击"查看操作日志"，或调用：

```text
GET /api/operations/logs?limit=200
```

## 本地验证

```powershell
python scripts/smoke_test.py
python scripts/backtest_smoke_test.py
```

## 设计文档

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — 架构设计与数据流
- [DATASET-V3-SPEC.md](docs/DATASET-V3-SPEC.md) — v3.0 规范合规对照
- [USER-GUIDE.md](docs/USER-GUIDE.md) — 操作指南

## License

MIT
