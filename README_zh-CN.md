# IDAM-RAW

论文 **“IDAM-RAW: a sensor-aware RAW imaging framework for low-light perception via illumination decoupling and adaptive feature modulation”** 的官方实现，已被 *Journal of Electronic Imaging* 接收（2026）。

## 方法简介

IDAM-RAW 面向极低照度 Bayer RAW 图像目标检测，将任务驱动的可微物理前端、残差照明解耦和自适应多尺度特征调制集成到端到端检测器中。公开代码统一采用论文中的正式名称 DetISP、RIDM、AFMM 和 Self-Boost。

完整安装、训练和评估说明请参阅 [英文 README](README.md)。

## 数据集

GitHub 仓库仅发布代码，数据集单独提供。

> **CR7-RAW 百度网盘下载：`BAIDU_NETDISK_LINK_TO_BE_ADDED`**

请在正式公开数据集前，将上述占位符替换为可访问的百度网盘链接及提取码。CR7-RAW 的采集、标注、划分和完整性要求见 [数据集说明](docs/datasets/CR7_RAW.md)。

公开基准：

- [LOD Dataset 官方仓库](https://github.com/ying-fu/LODDataset)
- [BDD100K 官方仓库](https://github.com/bdd100k/bdd100k)

下载 CR7-RAW 后先运行：

```bash
python scripts/check_dataset.py --root datasets/CR7_RAW --expected-train 1019 --expected-val 140
```

## 训练

```bash
python scripts/train.py --model configs/models/idam_raw_yolov8s.yaml --data configs/datasets/cr7_raw.yaml --device 0 --seed 2024
```

论文的三次独立实验分别使用随机种子 2024、2025 和 2026。

## 许可与联系

代码采用 [GNU AGPL-3.0](LICENSE) 许可证；数据集不适用代码许可证。问题请提交 GitHub Issue，或联系 Minjie Dai：`2023021001000611@ecjtu.edu.cn`。
