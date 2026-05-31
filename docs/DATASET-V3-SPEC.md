# LeRobot Dataset v3.0 合规对照

本文档逐字段对照 LeRobot v3.0 规范，记录本工具的校验策略和写盘策略。

## info.json 字段

### 必须字段（ERROR）

| 字段 | v3.0 类型 | 本工具策略 |
|------|-----------|-----------|
| `codebase_version` | `"v3.0"` | 写入时继承自源；校验时非 `"v3.0"` → error |
| `fps` | `int` | 写入时 `_normalize_info_int_fields()` 强制 int；校验时 float → error |
| `features` | `dict` | 写入时继承自源 + 更新 video_info |
| `data_path` | `string` | 写入时设为标准模板 `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| `video_path` | `string` | 有 video feature 时写入标准模板；无时移除 |

### 有默认值的字段（WARNING）

| 字段 | v3.0 类型 | v3.0 默认值 | 本工具策略 |
|------|-----------|-------------|-----------|
| `chunks_size` | `int` | `1000` | `setdefault(1000)` |
| `data_files_size_in_mb` | `float` | `100.0` | `setdefault(100.0)` |
| `video_files_size_in_mb` | `float` | `500.0` | `setdefault(500.0)` |

### 从数据推测的字段（WARNING）

| 字段 | v3.0 类型 | 本工具策略 |
|------|-----------|-----------|
| `total_episodes` | `int` | 写入时 `int(len(episodes))` 重新计算 |
| `total_frames` | `int` | 写入时 `int(len(frames))` 重新计算 |
| `total_tasks` | `int` | 写入时 `int(len(tasks_df))` 重新计算 |
| `splits` | `dict` | 写入时 `{"train": "0:{len(episodes)}"}` |

### 可选字段

| 字段 | 说明 | 本工具策略 |
|------|------|-----------|
| `robot_type` | 机器人标识 | 继承自源 |
| `feature.info` | video metadata | 官方 Key 名为 `"info"`；写入时用 `"info"`；缺失 → warning（官方库自动 ffprobe 补全） |

### video_info（`feature.info` 子字段）

| 字段 | 说明 | 写入策略 |
|------|------|----------|
| `video.fps` | 视频帧率 | v3.0 允许 float |
| `video.codec` | 视频编码 | `h264` / `hevc` / `av1` / `vp9` |
| `video.pix_fmt` | 像素格式 | 通常 `yuv420p` |
| `has_audio` | 是否含音频 | 始终 `false`（编辑后无音频） |
| `is_depth_map` | 是否深度图 | `setdefault(False)` |

## data/*.parquet 列

| 列 | dtype | 本工具策略 |
|----|-------|-----------|
| `timestamp` | `float32` | `normalize_frame_columns()` 重算：`np.arange(length) / fps` |
| `frame_index` | `int64` | 每 episode 从 0 重编号 |
| `episode_index` | `int64` | 输出 episode 序号 |
| `index` | `int64` | 全局连续 index |
| `task_index` | `int64` | `rebuild_tasks_for_frames()` 重新映射 |
| `next.done` | `bool` | `normalize_frame_columns()` 最后一帧为 True |

写盘前 `_enforce_frame_int_dtypes()` 确保所有 int 列为 int64。

## meta/episodes/*.parquet 列

| 列 | 类型 | 写入策略 |
|----|------|----------|
| `episode_index` | `int64` | 连续重编号 |
| `length` | `int64` | 帧数 |
| `dataset_from_index` | `int64` | 全局起始 |
| `dataset_to_index` | `int64` | 全局结束（exclusive） |
| `data/chunk_index` | `int64` | 固定 `0` |
| `data/file_index` | `int64` | 固定 `0` |
| `videos/{key}/chunk_index` | `int64` | `write_*_videos()` 填入 |
| `videos/{key}/file_index` | `int64` | `write_*_videos()` 填入 |
| `videos/{key}/from_timestamp` | `float` | 累计时间戳 |
| `videos/{key}/to_timestamp` | `float` | 累计时间戳 |
| `stats/{feature}/{stat}` | `float` | `flatten_stats_for_episode()` 重新计算 |

写盘前 `_enforce_episode_int_dtypes()` 确保所有 int 列为 int64。

## 视频文件

| 要求 | 本工具策略 |
|------|-----------|
| H.264 / HEVC / AV1 / VP9 | 保留源编码格式，`normalize_video_codec()` 做别名映射 |
| yuv420p 像素格式 | 继承源视频的 `pix_fmt` |
| 无音频 | `ffmpeg -an` |
| 多 episode concat | `ffmpeg -f concat -c copy`（segment 提取后） |
| 关键帧间隔保持 | `source_keyframe_interval()` 探测 → `-g -keyint_min` |

## 校验分级总结

```
ERROR   — codebase_version, fps, features, data_path, video_path,
          fps 为 float, meta/stats.json 缺失
WARNING — chunks_size / data_files_size_in_mb / video_files_size_in_mb 缺失,
          total_* 缺失, video feature 缺少 info, 整数字段为 float
INFO    — robot_type, splits, feature.info 完全可选
```
