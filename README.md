# WaveRec: Wavelet-Based Multi-Modal Fusion for Fine-Grained Urban Flow Inference

## Overview



Fine-grained urban flow inference (FUFI), which involves inferring fine-grained flow maps from their coarse-grained counterparts, is of tremendous interest in the realm of sustainable urban traffic services. WaveRec addresses this challenge by proposing a wavelet-based multi-modal fusion architecture that effectively combines spatial frequency decomposition with external contextual features and road network topology.

The core innovation of WaveRec lies in its **WaveletMultiModalFusionV8Pro** module, which:
- Employs Discrete Wavelet Transform (DWT) to decompose flow maps into low-frequency and high-frequency components
- Captures both local fine-grained details and global coarse-grained semantics through separate processing streams
- Leverages DualStreamCrossAttention to integrate external factors (time, weather, day-of-week) and road network information
- Utilizes FrequencyAwareFusion to adaptively combine different frequency components


## Requirements

* python 3.8+
* pytorch 2.0+
* einops
* scikit-learn
* numpy

## Datasets

TaxiBJ datasets can be obtained from the baseline `https://github.com/yoshall/UrbanFM/tree/master/data`.

## Usage

Before running the code, ensure the package structure is as follows:

```
.
├── datasets
│   └── TaxiBJ
│       ├── P1
│       ├── P2
│       ├── P3
│       └── P4
├── datasets/road_map/
│   ├── TaxiBJ.png
│   ├── ChengDu.png
│   └── XiAn.png
├── experiments
├── model
│   ├── WaveRec.py
│   └── WaveRec_cd.py
├── model_train
├── utils_pack
│   ├── args.py
│   ├── args_cd.py
│   ├── metrics.py
│   └── utils.py
├── train.py
└── train_cd.py
```

### Training

Pre-training with masked reconstruction:

```bash
python train.py --pretrain_epochs 20 --batch_size 16 --dataset TaxiBJ
```

Joint training:

```bash
python train.py --joint_epochs 30 --batch_size 16 --dataset TaxiBJ
```

Fine-tuning:

```bash
python train.py --finetune_epochs 20 --batch_size 16 --dataset TaxiBJ
```

## Model Parameters

Key hyperparameters:
- `--channels`: Number of feature channels (default: 128)
- `--scale_factor`: Upscaling factor for super-resolution (default: 4)
- `--sub_region`: Number of sub-regions for local-global fusion (default: 4)
- `--use_exf`: Enable external factors (default: True)
- `--wave`: Wavelet type: 'haar', 'db1', 'db2', 'db4', 'sym2' (default: 'haar')

## Citing

If you find WaveRec useful in your research, please cite the following paper:

```

```
