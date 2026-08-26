# VOC-MEM Labeling

一个用于远程机器人操作视频标注的本地 Web 标注工具。程序通过 SSH 读取远程视频，在浏览器中播放，并将标注结果保存到本地 JSON 文件。

## 功能

- 支持 `organize_table` 和 `store_laptop_and_headphones` task。
- 支持 `SIA` 和 `VOC-MEM` metric。
- SIA 使用 `s` 标记；VOC-MEM 使用 `b`、`s`、`e` 标记。
- 使用 `j` 切换上一个视频，`k` 切换下一个视频。
- 已保存的标注会重新加载并显示在时间轴上。
- 标注按 task 和 metric 分别保存，例如 `organize_table_sia_annotations.json`。
- 保存当前视频时只更新该视频记录，不覆盖其他视频的标注。
- `Transfer all` 会将当前 task/metric 的结果转换并发送到指定远程目录。

## 环境要求

- Python 3.10+
- 可用的 SSH 客户端
- 能通过配置中的 SSH 地址访问远程视频服务器
- PyYAML

安装依赖：

```powershell
pip install -e .
```

## 使用

```powershell
python labeler.py --port 8765
```

然后打开 <http://127.0.0.1:8765>。

也可以指定默认 task：

```powershell
python labeler.py --task organize_table --port 8765
```

## 配置

任务、远程视频目录、远程结果目录及 metric 支持情况位于 [tasks.yaml](tasks.yaml)。修改远程主机或目录时，优先修改该文件和 `labeler.py` 中的 SSH 配置。

## 数据格式

本地 annotation 文件是以视频路径为 key 的 JSON 对象。每条记录包含 `episode`、`marks`、`task`、`metric` 等字段。时间以秒保存，帧号按 25 FPS 计算。

## 注意事项

视频不会永久保存到本地；播放时由程序通过 SSH 临时读取。请确保远程视频路径可访问，并在开始标注前确认 SSH 连接正常。
