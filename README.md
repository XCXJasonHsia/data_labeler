# 仿真数据标注

一个用于机器人操作视频标注和只读审核的本地 Web 工具。数据标注与数据审核是两套彼此独立的工作模式和数据源。

| 模式 | 数据源 | 是否写入标注 |
| --- | --- | --- |
| 数据标注 | `/mnt/public2/liushengbang/data/Veified_Data` | 是 |
| 数据审核 | `/mnt/public2/liushengbang/data/RoboDojo_Dataset_to_VMB` | 否，只读播放 |

标注模式只会读取 `Veified_Data` 下在 `tasks.yaml` 中配置的任务；审核模式只会读取 `RoboDojo_Dataset_to_VMB` 下的任务，两者不会互相回退或混用。

## 支持范围

当前支持以下任务：

- `organize_table`
- `store_laptop_and_headphones`
- `arrange_largest_number`
- `fold_clothes`
- `hang_mugs`
- `make_toast`
- `put_bottles_into_dustbin`
- `stack_blocks`
- `sweep_block`

`store_laptop_and_headphones` 对应实际目录名 `store_laptop_and_headphone`。

每个任务均支持三种 metric，各指标读取的数据范围不同：

| Metric | 读取的数据 |
| --- | --- |
| `VOC-MEM` | `ST-1`、`ST-HQ-EMB`、`ST-HQ-ENV` |
| `SIA+CSPC` | `ST-1`、`ST-HQ-EMB`、`ST-HQ-ENV`、`ST-2` |
| `FPL+TRR` | 所有顶层 `FRT-*` 数据，包括常规 `a/b/c`、`EMB` 和 `ENV` |

程序根据各 episode 下的 `observation.images.cam_high/*.mp4` 主视角视频生成列表，并同时加载同一 episode 的以下三个视角：

- `observation.images.cam_high`：主视角。
- `observation.images.cam_left_wrist`：左手腕部相机视角。
- `observation.images.cam_right_wrist`：右手腕部相机视角。

## 环境要求

- Linux 开发机
- Python 3.10+
- PyYAML 6.0+
- ffmpeg（保存标注点截图时使用）
- SSH 客户端（仅在本机找不到配置的视频目录时使用）

本项目不依赖 PyTorch。

## 首次配置环境

进入仓库：

```bash
cd /mnt/public2/liushengbang/data_labeler
```

创建项目专用虚拟环境：

```bash
python3 -m venv .venv
```

激活环境并安装项目：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

可以用下面的命令确认安装成功：

```bash
python -c "import yaml, labeler; print('environment ready')"
```

以上初始化只需要执行一次。当前开发机的 `.venv` 已经创建并安装完成。

## 每次启动

每次重新登录开发机或打开新的终端后，都需要重新激活虚拟环境：

```bash
cd /mnt/public2/liushengbang/data_labeler
source .venv/bin/activate
```

终端提示符前出现 `(.venv)` 表示激活成功。也可以检查当前 Python：

```bash
which python
```

正常情况下应输出：

```text
/mnt/public2/liushengbang/data_labeler/.venv/bin/python
```

启动标注服务：

```bash
python labeler.py --port 8765
```

启动只读审核模式：

```bash
python labeler.py --mode verify --port 8765
```

审核模式不提供任何标注、截图或导出功能。页面会直接列出 `/mnt/public2/liushengbang/data/RoboDojo_Dataset_to_VMB` 下包含 `ST-*` 或 `FRT-*` 数据的任务；选择任务后，程序递归查找该任务下的 `observation.images.cam_high/*.mp4`，并同时显示对应的主视角、左腕和右腕视频。审核时可按 `k` 查看下一个 episode，按 `j` 查看上一个 episode，首尾循环切换；按 `a` 或左方向键后退一帧，按 `d` 或右方向键前进一帧。

启动成功后会输出：

```text
Open http://127.0.0.1:8765
```

通过开发机的端口转发功能打开 `8765` 端口，或在能访问该地址的浏览器中打开 <http://127.0.0.1:8765>。

需要指定启动后的默认任务时，使用 `--task`：

```bash
python labeler.py --task hang_mugs --port 8765
```

查看所有可用参数：

```bash
python labeler.py --help
```

停止服务时，在运行服务的终端按 `Ctrl+C`。退出虚拟环境时执行：

```bash
deactivate
```

## 标注方法

页面左上角可以切换 task 和 metric。主视角显示在左侧，左手腕部相机视角位于右上方，右手腕部相机视角位于右下方。主视角是三个视频的统一控制入口：播放、暂停、倍速调整、拖动跳转、点击标注时间轴和逐帧微调都会同步到两个腕部视角。腕部视频静音播放，避免多个视角的音轨相互叠加。

切换 episode 后会保留当前播放速度。三个视频接口均支持 HTTP Range 分段加载，可以直接拖动主视角的浏览器进度条或点击标注时间轴跳转，不需要等待完整视频下载完成。

页面会始终显示当前 metric 和数据分组。选择 `FPL+TRR` 时，还会从目录名中解析并显示错误类型数字编号和 episode 类别。例如路径中的 `FRT-4/FRT-4-2/FRT-4-2-b` 会显示错误类型 `4-2`、episode 类别 `b`；`FRT-4-EMB` 和 `FRT-4-ENV` 会分别显示 `EMB` 和 `ENV`。

### SIA+CSPC

- `s`：标注当前视频帧。
- `c`：撤销当前视频最近一次标注。

每次按 `s` 或 `c` 后，当前 episode 的结果都会立即写入本地 JSON，无需额外点击保存。

### FPL+TRR

- `s`：标注当前视频帧。
- `c`：撤销当前视频最近一次标注。

FPL+TRR 只显示 `FRT-*` 数据，episode 下拉框和视频上方的信息卡会同时显示错误编号和 `a/b/c` 类别。

### VOC-MEM

- `b`：标记异常区间开始位置。
- `s`：标记异常区间内的特殊帧。
- `e`：标记异常区间结束位置。

VOC-MEM 标注必须从 `b` 开始、以 `e` 结束；`s` 必须位于对应的 `b-e` 区间内。一个视频可以包含多个完整区间。

### 视频切换

- `j`：上一个视频。
- `k`：下一个视频。

切换到列表首尾时会循环跳转。已保存的视频会在 episode 下拉框中显示 `✓`，重新打开时标注点会恢复到时间轴和表格中。

### 逐帧微调

- `a` 或左方向键：后退一帧。
- `d` 或右方向键：前进一帧。

逐帧微调会自动暂停视频，并按照视频标注使用的 `25 FPS` 每次移动 `0.04` 秒，方便精确定位标注帧。

## 标注结果

标注文件保存在仓库根目录，并按照 task 和 metric 分开存储：

```text
<task>_sia_cspc_annotations.json
<task>_voc-mem_annotations.json
<task>_fpl_trr_annotations.json
```

升级前已有的 `<task>_sia_annotations.json` 会作为 `SIA+CSPC` 的历史标注兼容读取；下一次保存后会写入新的 `sia_cspc` 文件。

例如：

```text
hang_mugs_sia_annotations.json
hang_mugs_voc-mem_annotations.json
```

文件是一个以完整视频路径为 key 的 JSON 对象。每条记录主要包含：

- `episode`：视频完整路径。
- `episode_name`：episode 名称。
- `task`：当前任务。
- `metric`：当前标注指标。
- `marks`：标注类型、帧号和时间。
- `duration`：视频时长。
- `gap`：导出采样间隔。

时间以秒记录，帧号按照 25 FPS 计算。本地标注文件和截图目录已加入 `.gitignore`，不会被普通 Git 提交带到代码仓库。

## 导出结果

点击页面上的“同步全部到 dataset_sim”并确认后，程序会扫描所有 task 和 metric 的本地已保存结果，并合并写入：

```text
/mnt/public2/liushengbang/vmbmk/dataset_sim/<task>/episodes/<episode>/annotation.json
```

SIA+CSPC 结果写入 `subtask_segments`，VOC-MEM 结果写入 `voc_mem`。同步只替换对应 metric 的字段，保留 `annotation.json` 中已有的其他标注字段。程序按照原始视频分组和 `dataset_sim` 的 `success/domain` 元数据匹配对应视频；FPL+TRR 或其他找不到匹配视频、标注格式不完整的记录会被跳过，并在页面显示数量。

目标文件名是 VMBMK 数据集现有格式使用的 `annotation.json`（单数）。

点击页面上的 `Transfer all`，输入目标目录的绝对路径。程序会转换当前 task/metric 的全部本地记录，并写入：

```text
<目标目录>/exceptional_intervals.json
```

如果目标目录位于当前开发机，程序会直接写入；否则会尝试通过 SSH 写入配置的远程主机。导出前请确认当前 metric 的标注格式完整，尤其是 VOC-MEM 的每个 `b` 都有对应的 `e`。

## 任务配置

任务的视频目录、预留 annotation 路径以及 metric 配置位于 [tasks.yaml](tasks.yaml)。主要字段如下：

```yaml
hang_mugs:
  video_root: /mnt/public2/liushengbang/data/Veified_Data/hang_mugs
  annotation: /mnt/public2/xiachenxiang/data/VOC-MEM/hang_mugs/exceptional_intervals.json
  metrics:
    SIA+CSPC:
      markers: [s]
      kind: nodes
    VOC-MEM:
      markers: [b, s, e]
      kind: interval
    FPL+TRR:
      markers: [s]
      kind: nodes
```

当前 `annotation` 字段不会被 `Transfer all` 自动采用；实际导出位置仍以页面中手动输入的目标目录为准。

添加任务时，需要确保 `video_root` 下存在对应的 ST 或 FRT 数据目录，并保持 episode 视频路径结构一致。各 metric 的实际目录范围由程序统一规则控制，不需要在每个 task 中重复填写。

## 常见问题

### 端口被占用

如果出现 `OSError: [Errno 98] Address already in use`，说明端口已有服务。可以停止旧进程，或者换一个端口：

```bash
python labeler.py --port 8766
```

### 页面没有显示最新功能

先按 `Ctrl+C` 停止旧服务，重新运行 `labeler.py`，然后在浏览器中使用 `Ctrl+Shift+R` 强制刷新。

### 任务没有视频

检查 `tasks.yaml` 中的 `video_root` 是否存在，并确认当前 metric 对应的数据目录存在：VOC-MEM 使用三类 ST，SIA+CSPC 额外使用 `ST-2`，FPL+TRR 使用 `FRT-*`。

### 出现 SSH 主机指纹提示

这表示配置的视频目录在当前开发机不可用，程序正在回退到 SSH。先停止标注服务，在普通终端中单独完成 SSH 登录和主机指纹确认，再重新启动服务。

### 已有标注没有显示

确认标注文件名中的 task 和 metric 与页面选择一致，例如 `hang_mugs_sia_cspc_annotations.json`。旧的 `hang_mugs_sia_annotations.json` 仍会兼容读取。切换任务后等待 episode 列表加载完成；必要时重启服务并强制刷新浏览器。
