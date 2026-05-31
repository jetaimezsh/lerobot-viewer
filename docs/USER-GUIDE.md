# 操作指南

## 启动

```bash
bash scripts/start_backend.sh
# 浏览器打开 http://127.0.0.1:8000
```

页面分为 7 个 Tab：

- **总览** — 数据集概览
- **Episode 播放** — 视频播放与时序图表
- **数据集编辑** — 编辑/选择导出/合并
- **模型管理** — 注册与管理模型
- **模型回测** — 离线推理对比
- **System 环境** — 后端环境检测

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
3. 点击"检查合并合法性"验证 fps / features / video_keys 一致性
4. 填写输出目录 → 勾选是否覆盖 → 点击"生成合并数据集"

要求：
- 至少 2 个数据集
- `fps`、`features` schema、`video_keys` 必须一致
- 支持含视频数据集的合并（视频重新编码拼接）

---

## 四、模型回测

### 模型管理

1. 切换到"模型管理" Tab
2. 填写 checkpoint 路径（目录或文件）
3. 选择 adapter 类型（LeRobot 官方 / 自定义脚本）
4. 点击"注册"，检查文件 → 加载模型
5. 可以加载多个模型

### 运行回测

1. 切换到"模型回测" Tab
2. 选择模型（勾选 checkbox）
3. 填写 episode 范围（如 `0,2,5-10`）
4. 可选：勾选"限制帧数"以快速验证
5. 点击"运行回测"

### 回测结果

- 结果矩阵（model × episode）：MAE / RMSE / max error
- Chart：选择 episode 和 action 维度，对比 ground truth / predicted / error

---

## 五、环境检测

"System 环境" Tab 显示：

- Python 执行路径、版本
- venv / conda 状态
- 核心依赖包版本（fastapi、pandas、pyarrow 等）
- 编辑工具可用性（ffmpeg 等）
- 缺失项和修复建议

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
