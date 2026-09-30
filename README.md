# Super-Resolution on TextZoom: Transformer vs CNN

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A controlled comparison between a **Transformer** and a **CNN** for **Single Image Super-Resolution**, applied to low-resolution text crops photographed from real signs and billboards.

## 📋 Abstract

Two state-of-the-art architectures from opposite families — **HiT-SRF** (Transformer, ECCV 2024) and **SAFMN** (fully convolutional, ICCV 2023) — are evaluated on the same task, the same dataset and with the same metrics. Both are pre-trained by their authors on the same data (DF2K) at the same scale, and both are adapted to the target domain with an identical fine-tuning recipe: **the only variable that changes is the architecture**.

This setup answers a question more interesting than "which model is better": **how much does the choice of architecture weigh, compared to adaptation to the application domain?** The answer is a number — about **one to fifteen**, in favour of adaptation — plus a result that neither model would show on its own: before adaptation, the two architectures are **indistinguishable from each other**.

📄 The full report is available here: [Report.md](Report.md).

## 🎯 Objectives

- Evaluate both models **zero-shot** on a domain they were never trained for
- Fine-tune them on that domain with an **identical** procedure and budget
- Compare them against a bicubic baseline with three complementary metrics
- Establish, through **paired statistics**, whether the observed gaps are real
- Weigh the quality gains against the actual computational cost

## 📊 Dataset

**TextZoom** (ECCV 2020) — text crops from real-world scenes, at an upscaling factor of **2** (16×64 → 32×128 pixels).

What makes it the right target domain is that it is genuinely **out of distribution** with respect to the pre-training data, in two independent ways:

- **the LR images are real, not synthetic**: both versions were photographed by changing the camera focal length, so the LR contains real optical blur, sensor noise and compression artifacts — not the clean bicubic downscaling the models learned to invert;
- **the content is text**, not natural photographs: sharp edges, almost no texture, thin regular strokes.

Data used: **2794 training pairs** and the **1619 images** of the `easy` test subset.

## 🏗️ The two architectures

| | Method 1 — HiT-SRF | Method 2 — SAFMN |
|---|---|---|
| Venue | ECCV 2024 | ICCV 2023 |
| Family | Transformer | CNN |
| Long-range context | attention in hierarchical windows | multi-scale spatial modulation |
| Parameters | 846,944 | 227,820 |
| Input constraint | dimensions must be multiples of 64 (padding required) | any size |
| Pre-training | DF2K, ×2 | DF2K, ×2 |

## 🔧 Experimental setup

**Baseline**: bicubic interpolation — no model, no training. A model that does not beat it is adding nothing.

**Metrics**, computed on all 1619 test images, each with its 95% confidence interval: **PSNR** (pixel fidelity), **SSIM** (local structure), **LPIPS** (perceptual distance). Per-image scores are kept, which enables the paired comparison.

**Fine-tuning recipe**, identical for both models: L1 loss, Adam, learning rate 1e-5, 2000 iterations, batch size 16, the whole network unfrozen, whole images without crops.

## 📈 Results

| Method | PSNR (dB) | SSIM | LPIPS |
|---|---|---|---|
| Bicubic (baseline) | 23.42 ± 0.14 | 0.7359 ± 0.0054 | 0.1691 ± 0.0045 |
| HiT-SRF zero-shot | 23.56 ± 0.14 | 0.7404 ± 0.0056 | 0.1559 ± 0.0045 |
| **HiT-SRF fine-tuned** | **25.56 ± 0.12** | **0.8346 ± 0.0039** | **0.0676 ± 0.0025** |
| SAFMN zero-shot | 23.57 ± 0.14 | 0.7406 ± 0.0056 | 0.1579 ± 0.0045 |
| SAFMN fine-tuned | 24.89 ± 0.11 | 0.8157 ± 0.0041 | 0.0772 ± 0.0025 |

![Trajectories of the two models from zero-shot to fine-tuned](figure/traiettorie.png)
*Figure 1: The two models start out overlapping and just above the baseline; after adaptation they separate, with HiT-SRF ahead.*

![Visual comparison on a text crop](figure/confronto_visivo_2_griglia.png)
*Figure 2: The zero-shot reconstructions are indistinguishable from bicubic upscaling; after fine-tuning both models clearly separate the strokes from the background.*

### Computational cost (T4 GPU)

| Model | Parameters | Training (min) | Inference (ms/img) |
|---|---|---|---|
| HiT-SRF | 846,944 | 44.6 | 173.1 |
| SAFMN | 227,820 | 1.1 | 8.2 |

## 🔍 Key findings

1. **Zero-shot, the architecture is irrelevant.** 23.56 against 23.57 dB, 0.7404 against 0.7406 of SSIM: a Transformer and a CNN that share almost nothing structurally produce the very same result. Since they share the pre-training data but not the architecture, out of domain it is the **pre-training** that decides.

2. **Adaptation is worth about fifteen times the model choice.** Replacing bicubic interpolation with a state-of-the-art model is worth 0.14 dB; adapting that same model to the domain is worth 2.00 dB (HiT-SRF) and 1.32 dB (SAFMN).

3. **After adaptation HiT-SRF wins, systematically but at a price.** The paired comparison gives +0.674 dB [+0.631, +0.716], with HiT-SRF better on **80.7%** of the images and on all three metrics. Yet it costs **21×** the inference time: SAFMN obtains **69%** of the available improvement at one twenty-first of the cost.

4. **Parameter count is an unreliable proxy for cost.** HiT-SRF has 3.7× the parameters but is 21× slower, for two reasons: the padding forced by its input constraint makes it process four times the useful area, and attention is intrinsically more expensive than convolution.

## ⚠️ Limitations

- The timing gap is partly a property of **this dataset**: the padding from 16×64 to 64×64 is caused by TextZoom's unusually small images and would disappear on ordinary-sized ones.
- Evaluation is limited to the `easy` subset, because LR/HR pairs are two distinct shots and their misalignment — which grows with difficulty — penalises PSNR and SSIM.
- Both methods belong to the **regression** family: generative models (GANs, diffusion) were left out, so the comparison is between two ways of building an architecture, not two ways of defining the objective.

## 🏗️ Project structure

```
├── Computer_Vision_Project.ipynb   # Main notebook (both methods, executed)
├── Report.md                       # Full report
├── README.md                       # This file
├── LICENSE                         # MIT License
└── figure/                         # Figures used in the report
```

## 🚀 Usage

The notebook was developed and executed on **Google Colab** with a T4 GPU.

1. Download the `train2` and `test/easy` folders of TextZoom from the [official repository](https://github.com/WenjiaWang0312/TextZoom) and upload them to your Drive under `textzoom/`, renaming `train2` to `train`
2. Open `Computer_Vision_Project.ipynb` in Colab and run the cells in order

Pre-trained weights for both models are downloaded automatically. Fine-tuned weights are cached on Drive, so re-running the notebook skips training that has already been completed.

## 📚 Context

Developed for the **Computer Vision** course at the **University of Catania** — Academic Year 2025/2026, prof. S. Battiato, F. Guarnera.

The project format required both students to tackle the same task on the same dataset with two different architectures, and to compare the results.

## 📖 References

- X. Zhang et al., *HiT-SR: Hierarchical Transformer for Efficient Image Super-Resolution*, ECCV 2024 — [arXiv](https://arxiv.org/abs/2407.05878) · [repo](https://github.com/XiangZ-0/HiT-SR)
- L. Sun et al., *Spatially-Adaptive Feature Modulation for Efficient Image Super-Resolution*, ICCV 2023 — [arXiv](https://arxiv.org/abs/2302.13800) · [repo](https://github.com/sunny2109/SAFMN)
- W. Wang et al., *Scene Text Image Super-Resolution in the Wild*, ECCV 2020 — [arXiv](https://arxiv.org/abs/2005.03341) · [repo](https://github.com/WenjiaWang0312/TextZoom)
- R. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018 — [arXiv](https://arxiv.org/abs/1801.03924)

## 👥 Authors

- **Lorenzo Comis** — [GitHub](https://github.com/LorenzoComis-git) — Method 2 (SAFMN)
- **Alessandro Sciacca** — Method 1 (HiT-SRF)

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*For questions or suggestions, feel free to open an issue on GitHub.*
