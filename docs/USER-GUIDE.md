# 操作指南

## 启动

```bash
bash scripts/start_backend.sh
# 浏览器打开 http://127.0.0.1:8000
```

页面分为 2 个根工作台，每个工作台内再分子页面：

- **LeRobot 数据** — 总览、Episode 播放、数据集编辑、System 环境
- **模型工作台** — 模型管理、回测任务、训练配方、训练队列、训练流水线

---

## 一、浏览数据集

1. 左侧输入数据集根目录，点击"加载"
2. 总览页显示：FPS、Episodes、Frames、Tasks、Video Keys 等
3. 左侧 episode 列表，点击进入 Episode 播放

### Episode 播放

- **视频**：多视角同步播放，倍速（0.25x~4x），拖动进度条
- **Chart**：选择要绘制的数值字段，滚动缩放、右键/Shift+拖动平移，点击 seek
- **Current Frame**：显示当前帧各字段的值

---

## 二、数据集编辑

### 模式切换

顶部有两个模式按钮：

- **删除（红色）**：标记后删除指定 episode 或裁剪区间
- **导出（绿色）**：标记后仅导出指定 episode 或区间

切换模式会清空已有标记。

### 操作步骤

1. **选择模式**：点击"删除"或"导出"
2. **标记时间点**（如需裁剪）：拖动进度条，点击"设为起点"和"设为终点"
3. **标记**：
   - 点击"标记此 episode" → 标记整个 episode
   - 点击"标记此区间" → 标记时间范围
4. **取消**：再次点击同按钮取消标记
5. 切换到"数据集编辑" Tab
6. **预估**：点击"预估修改结果"查看影响范围
7. **执行**：填写输出目录 → 点击"生成新数据集"

### 操作类型对照

| 模式 | 标记此 episode | 标记此区间 |
|------|----------------|-------------|
| 删除 | delete_episode | trim_episode（保留区间内） |
| 导出 | select_episode（仅导出此 episode） | select_episode_range（仅导出区间） |

---

## 三、数据集合并

1. 在"数据集编辑" Tab 的合并区域输入数据集路径（每行一个）
2. 点击"+ 加入当前数据集"快速添加已加载的数据集
3. 点击"数据集一致性校验"验证 fps / features / video_keys 一致性
4. 填写输出目录 → 勾选是否覆盖 → 点击"生成合并数据集"

要求：
- 至少 2 个数据集
- `fps`、`features` schema、`video_keys` 必须一致
- 支持含视频数据集的合并（视频重新编码拼接）

---

## 四、模型回测

### 模型档案

1. 切换到"模型档案" Tab
2. 选择模板，填写 profile id、名称、checkpoint 路径、设备
3. 在参数表格中编辑运行期参数和自定义参数；点击"添加行"增加 key/type/value
4. 点击"新建档案"保存到 `state/model_profiles/`
5. 点击"检查"读取 checkpoint 文件结构；在 Linux 推理环境中点击"加载"
6. 可在数据查看页选中一个 episode 后，回到模型档案页点击"真实帧测试"

### 运行回测

1. 在 **LeRobot 数据 / Episode 播放** 中打开目标 episode
2. 点击"加入回测样本池"
3. 如需跨数据集回测，加载另一个数据集并继续加入 episode
4. 切换到 **模型回测 / 回测任务**
5. 在样本池表格中确认数据集、路径、episode 编号、帧数、时长、FPS、task 和视频路数
6. 选择 profile（勾选 checkbox）
7. 可选：勾选"限制帧数"以快速验证
8. 点击"运行回测"；多个任务会进入后台队列并依次执行，可在队列表格中刷新查看状态

### 回测结果

- 结果矩阵（profile × dataset × episode）：MAE / RMSE / max error
- Chart：选择具体数据集 episode 和 action 维度，对比 ground truth / predicted / error
- 历史结果保存到 `state/backtests/`，每条结果包含当次 profile 快照，支持 HTML / CSV / JSON 导出

---

## 五、模型训练

### 训练配方

1. 切换到 **模型工作台 / 训练配方**
2. 选择训练模板，填写 recipe id、名称、训练数据集路径、输出 checkpoint 目录
3. 在训练超参数表格中编辑 batch size、epochs、learning rate、policy_type 等参数
4. 按需填写自定义参数；这些参数会随 recipe 一起保存，便于追溯实验
5. 勾选"训练完成后自动创建回测档案"，训练成功后会生成标准 Model Profile
6. 点击"新建配方"或"保存"

### 训练队列

1. 在训练配方页点击"提交训练"，或在配方列表中点击"提交训练"
2. 作业进入 `state/training_jobs/`，单 worker 顺序执行
3. 切换到 **模型工作台 / 训练队列**，点击"刷新队列"查看状态
4. 点击"日志"查看 stdout/stderr 日志
5. 排队或运行中的作业可以取消；已结束作业可以重新排队或删除

说明：
- `lerobot_train` 框架封装 `lerobot-train` CLI，真实训练建议在 Linux 环境执行
- 当前测试使用 `mock` 训练框架覆盖队列、日志、失败、自动 profile 生成等链路
- 服务重启后，已落盘的 queued 作业会在下次访问训练 API 时恢复调度；原 running 作业会标记为 failed

### 训练流水线

**模型工作台 / 训练流水线** 当前提供轻量流水线记录：

- 选择一个训练配方
- 选择要对比的已有 profile
- 使用 Episode 播放页加入的回测样本池
- 创建后会提交训练作业并记录流水线状态

自动训练完成后触发回测属于后续增强，目前训练完成后可直接使用自动生成的 profile 进入回测任务。

---

## 六、环境检测

"System 环境" Tab 显示：

- Python 执行路径、版本
- venv / conda 状态
- 核心依赖包版本（fastapi、pandas、pyarrow 等）
- 编辑工具可用性（ffmpeg 等）
- 缺失项和修复建议
- 操作日志：记录数据集打开、编辑、合并、校验、模型管理和回测任务

---

## 常见问题

### 路径补全无效

重载页面，确认当前数据集目录确实存在。

### 视频无法加载

确认数据集有 video feature，且 `info.json` 的 `video_path` 模板正确。确认视频文件存在于对应路径下。

### 编辑/合并时提示 ffmpeg 不可用

安装 ffmpeg：

```bash
conda install -c conda-forge ffmpeg
# 或
sudo apt install ffmpeg
```

### 官方校验失败："Unknown format code 'd' for object of type 'float'"

`info.json` 中某个整数字段是 float。重新运行编辑/合并，本工具写盘时会自动修正。

### 端口被占用

```bash
PORT=8001 bash scripts/start_backend.sh
```

### 编辑后 episode_index 不连续

这是预期行为：编辑/合并后 episode 从 0 连续重编号，frame_index、index 同样连续。这是 v3.0 规范要求。
