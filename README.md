# MIRROR
Differentiable Deep Social Projection for Assistive Human-Robot Communication

## Introduction

This work focus on assistive human-robot communication. Communication is a hallmark of intelligence. In this work, we present MIRROR, an approach to (i) quickly learn human models from human demonstrations, and (ii) use the models for subsequent communication planning in assistive shared-control settings. MIRROR is inspired by social projection theory, which hypothesizes that humans use self-models to understand others. Likewise, MIRROR leverages self-models learned using reinforcement learning to bootstrap human modeling. Experiments with simulated humans show that this approach leads to rapid learning and more robust models compared to existing behavioral cloning and state-of-the-art imitation learning methods. We also present a human-subject study using the CARLA simulator which shows that (i) MIRROR is able to scale to complex domains with high-dimensional observations and complicated world physics and (ii) provides effective assistive communication that enabled participants to drive more safely in adverse weather conditions. 

<p align="center">
  <img src="https://github.com/clear-nus/MIRROR/blob/main/mirror.png?raw=true" width="40%">
  <br />
  <span>Fig 1. MIRROR framework is inspired by social projection; the robot reasons using a human model that is constructed from its own internal self-model. Strategically-placed learnable “implants” capture how the human is different from the robot. MIRROR plans communicativeactions by forward simulating possible futures by coupling its own internal  model (to simulate the environment) and the human model (to simulate their actions).</span>
</p>

## Environment Setup 
The code is tested on Ubuntu 16.04, Python 3.7 and CUDA 10.2. Please download the relevant Python packages by running:

Get dependencies:

```
pip3 install torch torchvision torchaudio
pip3 install numpy
```

Install Carla based on the following repo: [carla-simulator/carla](https://github.com/carla-simulator/carla).
 
Install rlpyt based on the following repo: [astooke/rlpyt](https://github.com/astooke/rlpyt).

## Usage

To run MIRROR on carla simulator, run the following:
```
./CarlaUE4.sh (under the folder where you install the Carla)
python ./scripts/main_expt.py # main experiments including no communication and MIRROR
python ./scripts/main_expt_warmup_clear.py # warmup session in clear weather
python ./scripts/main_expt_warmup_fog.py # warmup session in fog weather
```

To train the MIRROR model from scratch, run the following in order:
```
python ./text_compress/main_text.py # train an autoencoder to compress gpt word embedding.
python ./text_compress/main_text_embed_token.py # refine gpt linear model that map word embeddings to tokens.
python ./scripts/main_basic.py # train robot's self model
```
The pretrained MIRROR models can be downloaded from Google Drive: [link](https://drive.google.com/file/d/1zUaEa06tbD0W6tqTq36KaJ6NPjIwXjvG/view?usp=sharing). To use the pretrained models, extract the downloaded `mirror_models.zip` to `./scripts/saved_models/`.

## BibTeX

To cite this work, please use:

```
@inproceedings{chen2022mirror,
    title={MIRROR: Differentiable Deep Social Projection for Assistive Human-Robot Communication}, 
    author={Chen, Kaiqi and Fong, Jeffrey and Soh, Harold},
    year={2022},
    booktitle = {Proceedings of Robotics: Science and Systems}, 
    year      = {2022}, 
    month     = {June}}
```

### Acknowledgement
This repo contains code that's based on the following repos: [carla-simulator/carla](https://github.com/carla-simulator/carla) and [astooke/rlpyt](https://github.com/astooke/rlpyt).

### References
**[Dosovitskiy et al., 2017]** Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, Vladlen Koltun: CARLA: An open urban driving simulator. In CoRL, 2017. 

**[Stooke et al., 2019]** Adam Stooke, Pieter Abbeel: rlpyt: A research code base for deep reinforcement learning in pytorch. arXiv preprint, 2019. 