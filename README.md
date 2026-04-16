# COMP4026 Group Project - Part C (Expression Recognition)

## 中文说明

### 1) 仓库结构

用于小组协作的目录如下：
- `face recognition/`
- `face anonymisation/`
- `facial expression recognition/`

本 README 主要对应第三部分（表情识别）的代码与流程。

### 2) 数据集下载与放置位置

FER-2013 下载地址：
- `https://www.kaggle.com/datasets/msambare/fer2013`

下载并解压后，请将数据放在项目根目录下的：
- `FER-2013/`

最终结构应类似：
- `FER-2013/train/angry/...`
- `FER-2013/train/happy/...`
- `FER-2013/test/angry/...`
- `FER-2013/test/happy/...`

说明：当前清单与配置默认使用 `FER-2013/train` 作为训练集、`FER-2013/test` 作为验证/测试集。

### 3) CSV 清单（manifest）的作用

是的，`data` 文件夹下的 CSV 主要用于“指定图片路径与标签”。训练和评估脚本会读取这些 CSV，而不是直接扫描整个图片目录。

对应文件：
- `facial expression recognition/data/manifests/train_original.csv`
- `facial expression recognition/data/manifests/val_original.csv`
- `facial expression recognition/data/manifests/val_anonymized.csv`
- `facial expression recognition/data/manifests/val_paired.csv`

常用格式：
- 单图分类清单：`image_path,label`
- 配对一致性清单：`orig_path,anon_path,label`

路径规则：
- CSV 中的相对路径会基于 `facial expression recognition/configs/baseline.yaml` 里的 `data.image_root` 解析。
- 例如 `image_root: d:/COMP4026` 时，`FER-2013/train/angry/xxx.jpg` 会被解析为 `d:/COMP4026/FER-2013/train/angry/xxx.jpg`。

### 4) 安装依赖

```bash
pip install -r requirements.txt
```

### 5) 正式训练

```bash
python facial\ expression\ recognition/scripts/train_expression.py --config facial\ expression\ recognition/configs/baseline.yaml
```

输出模型默认保存在：
- `facial expression recognition/outputs/baseline/best_model.pt`

### 6) 评估（当前仅原始图）

```bash
python facial\ expression\ recognition/scripts/evaluate_expression.py --config facial\ expression\ recognition/configs/baseline.yaml --checkpoint facial\ expression\ recognition/outputs/baseline/best_model.pt
```

输出包括：
- 原始图验证集 accuracy / macro F1
- 若提供匿名化 CSV，则额外输出匿名图指标与一致性指标

### 7) 与匿名化模块（Part B）的接口

Part C 可以独立开发，但集成时需要 Part B 提供：
- 匿名化验证集清单：`image_path,label`
- 配对清单：`orig_path,anon_path,label`

---

## English Guide

### 1) Repository Structure

Team collaboration folders:
- `face recognition/`
- `face anonymisation/`
- `facial expression recognition/`

This README focuses on Part C (facial expression recognition).

### 2) Dataset Download and Placement

FER-2013 download URL:
- `https://www.kaggle.com/datasets/msambare/fer2013`

After downloading and extracting, place the dataset at:
- `FER-2013/` (under project root)

Expected structure:
- `FER-2013/train/angry/...`
- `FER-2013/train/happy/...`
- `FER-2013/test/angry/...`
- `FER-2013/test/happy/...`

Note: current manifests/config use `FER-2013/train` for training and `FER-2013/test` for validation/testing.

### 3) What CSV Manifests Do

Yes, the CSV files in `data/manifests` define image paths and labels.
Training/evaluation scripts read these CSV files instead of scanning folders directly.

Manifest files:
- `facial expression recognition/data/manifests/train_original.csv`
- `facial expression recognition/data/manifests/val_original.csv`
- `facial expression recognition/data/manifests/val_anonymized.csv`
- `facial expression recognition/data/manifests/val_paired.csv`

Common formats:
- Single-image classification: `image_path,label`
- Paired consistency evaluation: `orig_path,anon_path,label`

Path rule:
- Relative paths are resolved using `data.image_root` in `facial expression recognition/configs/baseline.yaml`.

### 4) Install Dependencies

```bash
pip install -r requirements.txt
```

### 5) Train (Baseline)

```bash
python facial\ expression\ recognition/scripts/train_expression.py --config facial\ expression\ recognition/configs/baseline.yaml
```

Default best checkpoint:
- `facial expression recognition/outputs/baseline/best_model.pt`

### 6) Evaluate (Original Images for Now)

```bash
python facial\ expression\ recognition/scripts/evaluate_expression.py --config facial\ expression\ recognition/configs/baseline.yaml --checkpoint facial\ expression\ recognition/outputs/baseline/best_model.pt
```

Outputs include:
- Original validation accuracy / macro F1
- Anonymized and paired metrics if anonymized manifests are provided

### 7) Interface Contract with Part B (Anonymization)

Part C can be developed independently, but integration requires Part B to provide:
- Anonymized validation manifest: `image_path,label`
- Paired manifest: `orig_path,anon_path,label`
