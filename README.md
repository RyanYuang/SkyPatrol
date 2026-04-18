# UAV Vision MVP0

基于 YOLO 的无人机视觉项目 MVP0 脚手架。

当前目标：先跑通识别功能（目标检测），支持图片、视频和本地摄像头输入，并输出带框结果与 predictions.json。

## 1. 功能范围

MVP0 已包含：
- YOLO 目标检测推理
- 图片 / 视频 / 摄像头输入
- 推理可视化输出
- JSON 结果导出
- 最小测试与脚手架文档

MVP0 暂不包含：
- 语义分割
- 自定义训练
- Jetson / TensorRT 部署
- 无人机实机视频流接入

## 2. 目录结构

```text
uav-vision/
├─ configs/
├─ assets/demo/
├─ checkpoints/
├─ data/
├─ docs/
├─ runs/
├─ scripts/
├─ src/
└─ tests/
```

## 3. 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果是 macOS Apple Silicon，推理时可尝试：
```bash
python scripts/infer.py --source assets/demo/demo.jpg --device mps
```

## 4. 初始化目录

```bash
python scripts/prepare_dirs.py
```

## 5. 运行方式

图片：
```bash
python scripts/infer.py --source assets/demo/demo.jpg
```

视频：
```bash
python scripts/infer.py --source assets/demo/demo.mp4
```

摄像头：
```bash
python scripts/infer.py --source 0
```

自定义参数示例：
```bash
python scripts/infer.py   --source assets/demo/demo.mp4   --device cpu   --output-dir runs/detect/manual-run   --no-show
```

## 6. 输出结果

默认输出到：
```text
runs/detect/<timestamp>/
```

常见输出：
- `annotated_<原文件名>`：带框图片或视频
- `predictions.json`：所有检测结果

## 7. 关键配置

模型配置：`configs/model/yolo_detect.yaml`
运行配置：`configs/runtime/infer.yaml`
类别名映射：`configs/data/class_names.yaml`

## 8. 测试

```bash
pytest -q
```

## 9. 常见问题

1. 首次运行会自动下载 YOLO 权重
- 需要联网
- 默认权重名为 `yolov8n.pt`

2. 没有 GPU 怎么办
- 直接用 CPU
- macOS 可尝试 `mps`

3. 摄像头打不开
- 尝试 `--source 1` 或 `--source 2`

## 10. 下一步路线

- MVP1：接入 VisDrone，自定义检测训练
- MVP2：增加语义分割模块
- MVP3：导出 ONNX / TensorRT 并部署到 Jetson
