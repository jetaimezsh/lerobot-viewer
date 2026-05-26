# LeRobot Dataset v3.0 本地查看器

这是一个 Windows 本地运行的 LeRobot Dataset v3.0 查看网页。后端使用 Python/FastAPI，前端为静态 HTML/CSS/JS，不需要 Node.js。

## 功能

- Python 虚拟环境状态查看和依赖安装入口。
- 读取严格遵守 LeRobot Dataset v3.0 的本地数据集。
- 解析 `meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`meta/episodes/**/*.parquet`。
- 按 episode 展示数值型时序数据，数组字段会展开为 `field[0]`、`field[1]`。
- 支持多个视频视角，例如 `observation.images.front`、`observation.images.wrist`。
- 打开某个 episode 后，可播放、暂停、拖动时间轴、调整倍速/慢速。
- 播放视频时，同步显示当前时间点对应的 state/action 等数值。
- 自动记录最近打开过的数据集，可从侧边栏历史记录直接重新打开。
- 输入本地路径时，会自动列出当前路径下的文件夹，并标记包含 `meta/info.json` 的候选数据集目录。

## LeRobot v3.0 格式要求

本工具按当前官方 v3.0 结构读取数据：

```text
dataset/
├── data/
│   └── chunk-000/
│       └── file-000.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
└── videos/
    └── observation.images.front/
        └── chunk-000/
            └── file-000.mp4
```

关键规则：

- 数据和视频都是 file-based storage，一个 Parquet/MP4 文件可以包含多个 episode。
- episode 边界不从文件名猜测，而是从 `meta/episodes/**/*.parquet` 读取。
- 数据文件路径使用 `info.json` 中的 `data_path` 模板。
- 视频文件路径使用 `info.json` 中的 `video_path` 模板。
- task 文件要求为 `meta/tasks.parquet`。

格式依据：

- https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3
- https://huggingface.co/docs/lerobot/main/porting_datasets_v3
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py

## 创建并启动虚拟环境

在 PowerShell 中进入项目目录：

```powershell
cd D:\lerobot
```

一键创建虚拟环境并安装依赖：

```powershell
.\scripts\setup_venv.ps1
```

如果 PowerShell 阻止脚本执行，可先在当前终端临时允许本地脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 启动后端和网页

```powershell
.\scripts\start_backend.ps1
```

然后在浏览器打开：

```text
http://127.0.0.1:8000
```

也可以手动启动：

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## 使用方式

1. 打开 `http://127.0.0.1:8000`。
2. 在页面顶部输入 LeRobot v3.0 dataset 根目录，例如：

   ```text
   D:\datasets\my_lerobot_dataset
   ```

3. 点击“加载数据集”。
4. 在 episode 列表中选择一个 episode。
5. 页面会显示：
   - episode 基本信息；
   - 数值型字段的时序图；
   - 当前时间点的数值表；
   - 所有视频视角。
6. 点击播放后，视频、时间轴、图表游标和当前数值会同步移动。

侧边栏的“最近打开”会保存最近 20 个成功打开的数据集。历史文件存放在项目根目录：

```text
D:\lerobot\.viewer_history.json
```

路径输入框支持目录补全：

- 输入 `D:\` 会列出 D 盘下的文件夹。
- 输入 `D:\data\ro` 会列出 `D:\data` 下以 `ro` 开头的文件夹。
- 如果候选目录包含 `meta/info.json`，会显示 `dataset` 标记。

## 本地样例验证

开发时已用 Hugging Face 官方小数据集 `lerobot/pusht` 验证。它是 LeRobot v3.0 数据集，包含 `meta/tasks.parquet`、共享 Parquet 数据和 `observation.image` 视频视角。

如果项目目录中已有样例数据，可直接在页面输入：

```text
D:\lerobot\sample_datasets\pusht
```

验证结果：

- 数据集总览可读取 `206 episodes`、`25650 frames`、`fps=10`。
- Episode 0 可读取 161 帧。
- `observation.state` 和 `action` 均可展开为数组维度时序。
- `observation.image` 视频可加载并与时间轴同步。

样例数据来源：

```text
https://huggingface.co/datasets/lerobot/pusht
```

## 依赖说明

`requirements.txt` 中只包含本工具读取和服务数据需要的包。项目里也提供了同内容的 `requirement.txt`，用于兼容你指定的文件名。

- `fastapi`
- `uvicorn`
- `pandas`
- `pyarrow`
- `numpy`
- `pydantic`

不强制安装 `lerobot`，因为本工具直接按照官方 v3.0 文件格式读取本地文件，避免因训练库版本变化影响查看功能。

## 注意事项

- 当前版本只绘制数值型字段，`video`、`image`、`string` 字段不会进入时序图。
- 视频同步依赖 `meta/episodes` 中每个视角的 `videos/<video_key>/from_timestamp` 和 `to_timestamp`。
- 如果一个 episode 有多个视角，页面会为每个视角创建一个视频播放器，并使用同一个 episode 时间轴同步。
- 如果某个共享 MP4 包含多个 episode，页面会自动 seek 到当前 episode 的起始时间，并在到达结束时间时停止。
