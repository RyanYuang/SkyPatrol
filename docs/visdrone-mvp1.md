# VisDrone MVP1

这一阶段开始接入 VisDrone2019-DET 数据集，训练无人机视角目标检测模型。

## 目标
- 自动下载并转换 VisDrone2019-DET 为 YOLO 格式
- 提供统一训练入口
- 保持数据默认存放在项目内：`data/raw/VisDrone`

## 关键文件
- `scripts/train_visdrone.py`
- `src/training/visdrone.py`
- `configs/model/yolo_visdrone_train.yaml`

## 仅下载数据集
```bash
source .venv/bin/activate
python scripts/train_visdrone.py --download-only
```

## 开始训练
```bash
source .venv/bin/activate
python scripts/train_visdrone.py --device mps --epochs 50 --batch 8
```

## 自定义参数
```bash
python scripts/train_visdrone.py \
  --dataset-root data/raw/VisDrone \
  --device cpu \
  --epochs 10 \
  --imgsz 640 \
  --batch 4
```

## 说明
- 首次运行会下载约 2.3 GB 数据。
- 运行脚本时，会自动生成 `data/interim/visdrone_detection_runtime.yaml`。
- 训练结果会默认输出到 `runs/train/visdrone-yolov8n`。
