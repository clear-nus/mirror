# MIRROR
Differentiable Deep Social Projection for Assistive Human-Robot Communication

## Introduction

This work focus on assistive human-robot communication. Communication is a hallmark of intelligence. In this work, we present MIRROR, an approach to (i) quickly learn human models from human demonstrations, and (ii) use the models for subsequent communication planning in assistive shared-control settings. MIRROR is inspired by social projection theory, which hypothesizes that humans use self-models to understand others. Likewise, MIRROR leverages self-models learned using reinforcement learning to bootstrap human modeling. Experiments with simulated humans show that this approach leads to rapid learning and more robust models compared to existing behavioral cloning and state-of-the-art imitation learning methods. We also present a human-subject study using the CARLA simulator which shows that (i) MIRROR is able to scale to complex domains with high-dimensional observations and complicated world physics and (ii) provides effective assistive communication that enabled participants to drive more safely in adverse weather conditions. 

<p align="center">
  <img src="https://github.com/clear-nus/MIRROR/blob/main/mirror.png?raw=true" width="40%">
  <br />
  <span>Fig 1. MIRROR framework is inspired by social projection; the robot reasons using a human model that is constructed from its own internal self-model. Strategically-placed learnable “implants” capture how the human is different from the robot. MIRROR plans communicativeactions by forward simulating possible futures by coupling its own internal  model (to simulate the environment) and the human model (to simulate their actions).</span>
</p>

## Codes
Code coming soon!
