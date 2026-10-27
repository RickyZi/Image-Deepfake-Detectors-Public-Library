# TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks

[![Official Github Repo](https://img.shields.io/badge/Github%20page-222222.svg?style=for-the-badge&logo=github)](https://github.com/MMLab-unitn/TrueFake-IJCNN25)
[![Paper](https://img.shields.io/badge/-arXiv-B31B1B.svg?style=for-the-badge)](https://arxiv.org/pdf/2504.20658)

Original Paper:
[TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks](https://arxiv.org/pdf/2504.20658).

Authors: Stefano Dell'Anna, Andrea Montibeller, Giulia Boato 

## Abstract

AI-generated synthetic media are increasingly used in real-world scenarios, often with the purpose of spreading misinformation and propaganda through social media platforms, where compression and other processing can degrade fake detection cues. Currently, many forensic tools fail to account for these in-the-wild challenges. In this work, we introduce TrueFake, a large-scale benchmarking dataset of 600,000 images including top notch generative techniques and sharing via three different social networks. This dataset allows for rigorous evaluation of state-of-the-art fake image detectors under very realistic and challenging conditions. Through extensive experimentation, we analyze how social media sharing impacts detection performance, and identify current most effective detection and training strategies. Our findings highlight the need for evaluating forensic models in conditions that mirror real-world use.

# R50-TF

The R50-TF network uses a ResNet50 architecture pretrained on ImageNet, modified to exclude downsampling at the first layer. During training, the network's backbone remains frozen, and only the classification head is trained. This classification head implements "learned prototypes" to provide robust real vs. fake image detection and to detect out-of-distribution samples by modeling an isotropic Gaussian class-conditional distribution representative of the input data.


# Please Cite

```
@misc{dellanna2025truefake,
      title={TrueFake: A Real World Case Dataset of 
      Last Generation Fake Images also Shared on 
      Social Networks}, 
      author={Stefano Dell'Anna and Andrea Montibeller 
      and Giulia Boato},
      year={2025},
      eprint={2504.20658},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2504.20658}, 
}
```