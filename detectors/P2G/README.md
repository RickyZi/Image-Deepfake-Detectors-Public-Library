# ✋ Prompt2Guard: Conditioned Prompt-Optimization for Continual Deepfake Detection
    

[![Official Github Repo](https://img.shields.io/badge/Github%20page-222222.svg?style=for-the-badge&logo=github)](https://github.com/laitifranz/Prompt2Guard)
[![Paper](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/abs/2407.21554)


Original Paper: [Conditioned Prompt-Optimization for Continual Deepfake Detection](https://arxiv.org/abs/2407.21554).

Authors: Francesco Laiti, Benedetta Liberatori, Thomas De Min, Elisa Ricci

## Abstract

The rapid advancement of generative models has significantly enhanced the realism and customization of digital content creation. The increasing power of these tools, coupled with their ease of access, fuels the creation of photorealistic fake content, termed deepfakes, that raises substantial concerns about their potential misuse. In response, there has been notable progress in developing detection mechanisms to identify content produced by these advanced systems. 

However, existing methods often struggle to adapt to the continuously evolving landscape of deepfake generation. This paper introduces Prompt2Guard, a novel solution for exemplar-free continual deepfake detection of images, that leverages Vision-Language Models (VLMs) and domain-specific multimodal prompts. Compared to previous VLM-based approaches that are either bounded by prompt selection accuracy or necessitate multiple forward passes, we leverage a prediction ensembling technique with read-only prompts. Read-only prompts do not interact with VLMs internal representation, mitigating the need for multiple forward passes. Thus, we enhance efficiency and accuracy in detecting generated content. Additionally, our method exploits a text-prompt conditioning tailored to deepfake detection, which we demonstrate is beneficial in our setting. 

We evaluate Prompt2Guard on CDDB-Hard, a continual deepfake detection benchmark composed of five deepfake detection datasets spanning multiple domains and generators, achieving a new state-of-the-art. Additionally, our results underscore the effectiveness of our approach in addressing the challenges posed by continual deepfake detection, paving the way for more robust and adaptable solutions in deepfake detection.

## Please Cite 

```
@misc{laiti2024conditionedpromptoptimizationcontinualdeepfake,
      title={Conditioned Prompt-Optimization for Continual 
      Deepfake Detection}, 
      author={Francesco Laiti and Benedetta Liberatori and 
      Thomas De Min and Elisa Ricci},
      year={2024},
      eprint={2407.21554},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.21554}, 
}
```