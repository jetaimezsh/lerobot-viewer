# LeRobot Dataset v3.0 本地查看器

这是一个在 Windows 本地运行的 LeRobot Dataset v3.0 数据集查看网页。后端使用 Python/FastAPI，前端使用静态 HTML/CSS/JS，不需要 Node.js。

## 主要功能

- 读取严格遵守 LeRobot Dataset v3.0 的本地数据集。
- 解析 `meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`meta/episodes/**/*.parquet`。
- 展示 dataset 总览、features、tasks 和 episode 列表。
- 按 episode 展示数值型时序数据。
- 数组字段会自动展开，例如 `observation.state[0]`、`action[1]`。
- 支持多视频视角，例如 `observation.image`、`observation.images.front`、`observation.images.wrist`。
- 支持视频、时间轴、时序图、当前帧数值同步播放。
- 支持播放、暂停、拖动时间轴、倍速和慢速。
- 记录最近打开过的数据集，可从历史记录直接重新打开。
- 输入路径时自动列出候选文件夹，并标记包含 `meta/info.json` 的数据集目录。

## 项目结构

```text
lerobot-viewer/
├── app/
│   ├── __init__.py
│   └── main.py
├── scripts/
│   ├── setup_venv.sh
│   ├── setup_venv.ps1
│   ├── start_backend.sh
│   └── start_backend.ps1
├── web/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── requirements.txt
├── requirement.txt
├── .gitignore
└── README.md
```

## 支持的数据集格式

本工具按当前 LeRobot Dataset v3.0 格式读取数据。数据集根目录应类似：

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
    └── observation.image/
        └── chunk-000/
            └── file-000.mp4
```

关键规则：

- 必须有 `meta/info.json`。
- 必须有 `meta/tasks.parquet`。
- episode 边界从 `meta/episodes/**/*.parquet` 读取。
- 数据文件路径使用 `info.json` 里的 `data_path` 模板。
- 视频文件路径使用 `info.json` 里的 `video_path` 模板。
- 一个 Parquet 或 MP4 文件可以包含多个 episode，本工具不会按文件名猜测 episode。

格式参考：

- https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3
- https://huggingface.co/docs/lerobot/main/porting_datasets_v3
- https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py

## Windows 环境准备

### 1. 安装 Python

建议使用 Python 3.10 或更高版本。当前项目已在 Python 3.12 上验证。

在 PowerShell 中检查 Python：

```powershell
python --version
```

如果系统提示找不到 Python，请先安装 Python，并确保安装时勾选 `Add python.exe to PATH`。

### 2. 克隆项目

```powershell
git clone git@github.com:jetaimezsh/lerobot-viewer.git
cd lerobot-viewer
```

如果你已经把项目放在 `D:\lerobot`，则进入该目录：

```powershell
cd D:\lerobot
```

### 3. 创建虚拟环境并安装依赖

推荐使用项目自带脚本：

```powershell
.\scripts\setup_venv.ps1
```

这个脚本会做三件事：

- 在项目目录下创建 `.venv`。
- 升级虚拟环境里的 `pip`。
- 执行 `pip install -r requirements.txt` 安装后端依赖。

如果 PowerShell 阻止脚本执行，可在当前 PowerShell 窗口临时允许本地脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_venv.ps1
```

也可以手动创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果你习惯使用 conda，也可以用 conda 环境运行本项目：

```powershell
conda create -n lerobot-viewer python=3.12 -y
conda activate lerobot-viewer
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖包包括：

- `fastapi`
- `uvicorn`
- `pandas`
- `pyarrow`
- `numpy`
- `pydantic`

含视频数据集执行删除 episode、裁剪 episode 并生成新数据集时，还需要系统可直接运行 `ffmpeg`。

Windows 推荐用 Chocolatey 安装：

```powershell
choco install ffmpeg -y
ffmpeg -version
```

conda 环境也可以安装 conda-forge 版本：

```powershell
conda install -c conda-forge ffmpeg -y
ffmpeg -version
```

如果无法安装系统级 ffmpeg，也可以在当前 Python 环境中安装 `imageio-ffmpeg`，后端会自动识别它内置的 ffmpeg：

```powershell
python -m pip install imageio-ffmpeg
```

本工具不强制安装 `lerobot`，因为它直接按官方 v3.0 文件格式读取和写入本地数据集；但建议安装 `lerobot`，这样可以在严格校验之外再执行官方 `LeRobotDataset` 打开校验。

## 启动项目

推荐使用启动脚本：

```powershell
.\scripts\start_backend.ps1
```

启动成功后，在浏览器打开：

```text
http://127.0.0.1:8000
```

也可以手动启动：

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

conda 环境下手动启动：

```powershell
conda activate lerobot-viewer
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

也可以在激活 conda 后运行项目脚本。脚本会优先使用当前 conda 环境里的 Python；如果没有激活 conda，才会回退到项目 `.venv`：

```powershell
conda activate lerobot-viewer
.\scripts\start_backend.ps1
```

如果官方 LeRobot 校验提示没有安装 `lerobot`，先确认页面 System 环境里显示的 Python 路径是不是你的 conda 环境。官方校验只会使用当前后端进程的 Python 环境，不会自动跨环境查找包。

如果想在开发时自动重载：

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## 本地验证

如果本地已经有 `sample_datasets/pusht` 和 `tmp_multiview_dataset`，可以运行 smoke test 验证后端能正确读取 v3.0 metadata、episode 边界、数值序列和多视角视频索引：

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_test.py
```

也可以指定数据集路径：

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_test.py --pusht D:\lerobot\sample_datasets\pusht --multiview D:\lerobot\tmp_multiview_dataset
```

## Ubuntu 24.04 环境准备

### 1. 安装系统依赖

Ubuntu 24.04 默认提供 Python 3.12。先安装 Python venv、pip、Git 和 SSH 客户端：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git openssh-client
```

如果后续遇到某些 Python 包需要本地编译，再安装编译工具：

```bash
sudo apt install -y build-essential
```

一般情况下，本项目依赖的 `numpy`、`pandas`、`pyarrow` 在 Ubuntu 24.04 + Python 3.12 下会直接安装 manylinux wheel，不需要本地编译。

### 2. 克隆项目

```bash
git clone git@github.com:jetaimezsh/lerobot-viewer.git
cd lerobot-viewer
```

如果你没有配置 GitHub SSH key，也可以使用 HTTPS 克隆：

```bash
git clone https://github.com/jetaimezsh/lerobot-viewer.git
cd lerobot-viewer
```

### 3. 创建虚拟环境并安装依赖

推荐使用项目自带 Linux/macOS 脚本：

```bash
bash scripts/setup_venv.sh
```

这个脚本会在项目根目录创建 `.venv`，并安装 `requirements.txt` 中的依赖。

也可以手动执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

也可以使用 conda：

```bash
conda create -n lerobot-viewer python=3.12 -y
conda activate lerobot-viewer
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. 启动服务

推荐使用：

```bash
bash scripts/start_backend.sh
```

启动成功后打开：

```text
http://127.0.0.1:8000
```

如果要换端口：

```bash
PORT=8001 bash scripts/start_backend.sh
```

然后打开：

```text
http://127.0.0.1:8001
```

也可以手动启动：

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

conda 环境下手动启动：

```bash
conda activate lerobot-viewer
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

也可以在激活 conda 后运行项目脚本。脚本会优先使用当前 conda 环境里的 Python；如果没有激活 conda，才会回退到项目 `.venv`：

```bash
conda activate lerobot-viewer
bash scripts/start_backend.sh
```

如果官方 LeRobot 校验提示没有安装 `lerobot`，先确认页面 System 环境里显示的 Python 路径是不是你的 conda 环境。官方校验只会使用当前后端进程的 Python 环境，不会自动跨环境查找包。

开发时自动重载：

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Ubuntu 路径输入

Ubuntu 下数据集路径通常类似：

```text
/home/your-user/datasets/my_lerobot_dataset
```

路径补全在 Linux 下会从以下入口开始：

- 当前用户 home 目录，例如 `/home/your-user`
- 项目目录
- 当前工作目录
- 根目录 `/`

候选目录如果包含 `meta/info.json`，页面会显示 `dataset` 标记。

## 使用方式

1. 打开 `http://127.0.0.1:8000`。
2. 在左侧输入 LeRobot v3.0 dataset 根目录，例如：

   ```text
   D:\datasets\my_lerobot_dataset
   ```

3. 点击“加载”。
4. 左侧 episode 列表会显示所有 episode。
5. 点击任意 episode 进入播放页。
6. 页面上方显示该 episode 的视频视角。
7. 时间轴、视频、时序图、当前帧数值会同步。
8. `Numeric Time Series` 区域可通过下拉框选择或取消要绘制的数值字段。
9. `Current Frame` 区域会显示 episode 静态信息和当前时刻的数值。

## System 环境检测

`System环境` 页面会显示当前后端使用的 Python、venv 状态、conda 状态、核心依赖版本，以及数据修改工具检测结果。

conda 检测包括：

- 当前服务是否运行在 conda 环境中。
- conda 环境名和环境前缀。
- `conda` 命令是否可用。
- `conda --version` 返回的版本信息。

数据修改工具检测会说明当前环境是否支持无视频数据集编辑、含视频数据集编辑，以及缺少的工具和修复建议。

## 数据集可用性校验

v2 数据编辑功能会在生成新数据集后执行两层校验：

- 非官方严格校验：不依赖 `lerobot` 包，直接检查 LeRobot v3.0 文件结构、`info.json`、`tasks.parquet`、`episodes` metadata、Parquet 数据行数、`episode_index`、`frame_index`、全局 `index`、`timestamp`、`task_index`、feature shape 和 `stats.json`。
- 官方校验：如果当前环境安装了 Hugging Face `lerobot` 包，会尝试用官方 `LeRobotDataset` 打开输出目录并读取样本。

如果没有安装 `lerobot`，官方校验会显示为 skipped，但非官方严格校验仍会执行。可以在 `数据集编辑` 页面点击 `严格校验当前数据集` 手动检查当前加载的数据集。

安装官方包后再启动服务，可启用官方校验：

```powershell
python -m pip install lerobot
```

## 最近打开和路径补全

成功打开的数据集会记录在项目根目录：

```text
.viewer_history.json
```

这个文件只保存在本地，不会提交到 Git。

路径输入框支持目录补全：

- 输入 `D:\` 会列出 D 盘下的文件夹。
- 输入 `D:\data\ro` 会列出 `D:\data` 下以 `ro` 开头的文件夹。
- 在 Ubuntu 上输入 `/home/`、`~/` 或数据集路径前缀，会列出匹配的文件夹。
- 如果候选目录包含 `meta/info.json`，会显示 `dataset` 标记。

## 本地样例数据

开发时使用 Hugging Face 官方小数据集 `lerobot/pusht` 做过验证。它是 LeRobot v3.0 数据集，包含：

- `meta/tasks.parquet`
- 共享 Parquet 数据
- `observation.image` 视频视角

样例数据来源：

```text
https://huggingface.co/datasets/lerobot/pusht
```

如果你本地已经有样例数据，可以在页面输入：

```text
D:\lerobot\sample_datasets\pusht
```

验证结果：

- 可读取 `206 episodes`。
- 可读取 `25650 frames`。
- `fps=10`。
- Episode 0 可读取 161 帧。
- `observation.state` 和 `action` 可展开为数组维度时序图。
- `observation.image` 视频可加载并与时间轴同步。

## v3 模型离线回测

第三版开始加入模型工作区，页面划分为：

- `模型管理`：注册 checkpoint、检查模型文件、加载/卸载模型、查看 policy 类型、大小和状态。
- `模型回测`：选择一个或多个模型、一个或多个 episode，运行离线回测并对比模型 action 与数据集真实 action。

当前实现先支持 `LeRobot 官方 checkpoint` 的接口框架，并预留 `custom_script` 自定义模型脚本接口。模型推理运行仅支持 Linux；Windows 仍可查看页面、注册 checkpoint、检查文件结构和查看已有结果，但不会执行真实模型推理。

Linux 模型回测环境建议额外安装：

```bash
python -m pip install torch lerobot safetensors
```

回测结果会按 `model x episode` 组合展示：

- 每个组合的执行状态
- MAE / RMSE / max error
- 最差误差 frame
- action 维度级真实值、预测值和误差曲线

当前测试覆盖：

```powershell
.\.venv\Scripts\python.exe .\scripts\backtest_smoke_test.py
```

这个测试不依赖真实模型，使用 mock adapter 验证 episode action 读取、预测 action 对齐、误差指标和 checkpoint 文件检查逻辑。真实 LeRobot checkpoint 推理需要在 Linux 上补充集成测试。

## 常见问题

### PowerShell 不允许执行脚本

执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

然后重新运行：

```powershell
.\scripts\setup_venv.ps1
```

### `pyarrow` 安装失败

先升级 pip：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

再重新安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 端口 8000 被占用

可以换一个端口启动：

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

然后打开：

```text
http://127.0.0.1:8001
```

Ubuntu 下也可以这样换端口：

```bash
PORT=8001 bash scripts/start_backend.sh
```

### Ubuntu 上创建 venv 失败

如果出现 `ensurepip is not available` 或类似错误，通常是没有安装 `python3-venv`：

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

然后重新执行：

```bash
bash scripts/setup_venv.sh
```

## 不会提交到 Git 的内容

`.gitignore` 已排除：

- `.venv/`
- `sample_datasets/`
- `.viewer_history.json`
- `__pycache__/`
- Python 缓存文件
- 临时日志和临时测试文件
