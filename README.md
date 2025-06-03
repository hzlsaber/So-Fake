<div align="center">
<h3> So-Fake: Benchmarking and Explaining Social Media Image Forgery Detection </h3>

  <p align="center">
    <a href='https://arxiv.org/abs/2505.18660'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
    <a href='https://hzlsaber.github.io/projects/So-Fake/'  style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>
    <a href='' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow' alt='Huggingface Models'> </a>
    <a href='https://huggingface.co/datasets/saberzl/So-Fake-OOD' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Hugging%20Face-Dataset-orange?logo=huggingface' alt='Dataset'>
    <a href='https://www.youtube.com/embed/ZSShvDB94pc' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Video-Watch%20Now-red' alt='Video'> </a>
  </p>
</div>


[Zhenglin Huang](https://scholar.google.com/citations?user=30SRxRAAAAAJ&hl=en&oi=ao), [Tianxiao Li](https://tianxiao1201.github.io/), [Xiangtai Li](https://lxtgh.github.io/), [Haiquan Wen](https://orcid.org/0009-0009-3804-6753), [Yiwei He](https://orcid.org/0000-0003-0717-8517), [Jiangning Zhang](https://www.researchgate.net/profile/Jiangning-Zhang), [Hao Fei](https://haofei.vip/),[Xi Yang](https://scholar.google.com/citations?user=ddfKpX0AAAAJ&hl=zh-CN),
[Bei Peng](https://beipeng.github.io/), [Xiaowei Huang](https://cgi.csc.liv.ac.uk/~xiaowei/), [Guangliang Cheng](https://sites.google.com/view/guangliangcheng/homepage)


Welcome to our work **So-Fake**, for social media forgery images detection. 

In this work, we propose: 

> ✅ **One Dataset:** **So-Fake-Set:** A large-scale, diverse dataset tailored for social media image forgery detection!
> 
> ✅ **One Benchmark: So-Fake-OOD:** A challenging out-of-distribution benchmark built from real-world Reddit content.
> 
> ✅ **One Method: So-Fake-R1:** A unified, explainable vision-language framework optimized via reinforcement learning.
> 


## Abstract
Recent advances in AI-powered generative models have enabled the creation of increasingly realistic synthetic images, posing significant risks to information integrity and public trust on social media platforms. While robust detection frameworks and diverse, large-scale datasets are essential to mitigate these risks, existing academic efforts remain limited in scope: current datasets lack the diversity, scale, and realism required for social media contexts, while detection methods struggle with generalization to unseen generative technologies.

To bridge this gap, we introduce **So-Fake-Set**, a comprehensive social media-oriented dataset with over **2 million** high-quality images, diverse generative sources, and photorealistic imagery synthesized using 35 state-of-the-art generative models. To rigorously evaluate cross-domain robustness, we establish a novel and large-scale **(100K)** out-of-domain benchmark **(So-Fake-OOD)** featuring synthetic imagery from commercial models explicitly excluded from the training distribution, creating a realistic testbed for evaluating real-world performance.

Leveraging these resources, we present **So-Fake-R1**, an advanced vision-language framework that employs reinforcement learning for highly accurate forgery detection, precise localization, and explainable inference through interpretable visual rationales. Extensive experiments show that So-Fake-R1 outperforms the second-best method, with a 1.3% gain in detection accuracy and a 4.5% increase in localization IoU. By integrating a scalable dataset, a challenging OOD benchmark, and an advanced detection framework, this work establishes a new foundation for social media-centric forgery detection research.

## News
- 🔥 (23-05-2025) We are pleased to announce the release of [So-Fake-OOD](https://huggingface.co/datasets/saberzl/So-Fake-OOD).


## Overview

<div align="left">
  <img src="images/Figure1.png" width="100%">
    <figcaption><strong> Overview of So-Fake Dataset.</strong> (a) Comparison of forgery detection methods from 2020 to 2025 shows our dataset includes the most recent and the largest number of methods. 
              (b) So-Fake-Set covers diverse real social media scenarios, including 12 different categories. 
              (c) Different generative methods and visual examples in <strong>So-Fake-Set</strong> and <strong>So-Fake-OOD</strong>.</figcaption>
</div>

## Dataset Access

We provide two methods to access the So-Fake-OOD:

1. Public Access via [Hugging Face](https://huggingface.co/datasets/saberzl/So-Fake-OOD)


2. Download from Google Drive
[here](https://drive.google.com/drive/folders/1okP2S6LO-VvH69MDqpeRhYZypfJ0ZHoG?usp=sharing)

## Method
<div align="left">
    <img src="images/Method.png" width="100%">
    <figcaption> (a): Overview of the <strong>So-Fake-R1</strong> training process; (b): The detailed <strong>So-Fake-R1</strong> GRPO training process. The example shows a tampered image where a boy has been manipulated.  </figcaption>

</div>

## Generative Methods

<div align="left">
      <figcaption> Details of generative methods used in constructing <strong>So-Fake-Set</strong> and <strong>So-Fake-OOD</strong>. Column abbreviations: Set = So-Fake-Set, OOD = So-Fake-OOD, F = fully synthetic images, T = tampered images. Real data source abbreviations: F30k = Flickr30k, OI = OpenImages, OF = OpenForensics.
  </figcaption>
    <img src="images/generative_method.png" width="100%">

</div>

## Visual Cases

<div align="center">
  <figure style="margin: 0; padding: 0;">
        <figcaption style="margin-top: 8px;"><strong>Visual Cases of full synthetic images</strong></figcaption>
    <img src="images/f_s_case2.png" width="100%">
  </figure>
</div>

<div align="center">
  <figure style="margin: 0; padding: 0;">
        <figcaption style="margin-top: 8px;"><strong>Visual Cases of tampered images</strong></figcaption>
    <img src="images/t_case2.png" width="100%">
  </figure>
</div>

## Visual Output

<div align="center">
  <figure style="margin: 0; padding: 0;">
        <figcaption style="margin-top: 8px;"><strong>Visual output of So-Fake-R1</strong></figcaption>
    <img src="images/SupplFigure7.png" width="100%">
  </figure>
</div>

## Citation 

```
@misc{huang2025sofakebenchmarkingexplainingsocial,
      title={So-Fake: Benchmarking and Explaining Social Media Image Forgery Detection}, 
      author={Zhenglin Huang and Tianxiao Li and Xiangtai Li and Haiquan Wen and Yiwei He and Jiangning Zhang and Hao Fei and Xi Yang and Xiaowei Huang and Bei Peng and Guangliang Cheng},
      journal={arXiv preprint arXiv:2505.13379},
      year={2025}
}
```
