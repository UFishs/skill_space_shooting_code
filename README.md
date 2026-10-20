# Skill-Space Shooting for Autonomous Robot Policy Improvement

[Project Page](https://ufishs.github.io/skill_space_shooting/) | [Paper](https://arxiv.org/abs/2609.38178) 

[![arXiv](https://img.shields.io/badge/arXiv-2609.38178-b31b1b.svg)](https://arxiv.org/abs/2609.38178)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

# Overview

Robots deployed in the physical world must be able to improve beyond their initial training as they encounter new situations and failures. For this improvement to scale across tasks, it must make effective use of experience without requiring human demonstration of each correction. Recent agentic systems offer a way to reduce this reliance on human effort by using foundation models to autonomously compose learned behaviors to complete tasks. Yet completing tasks this way does not itself teach a task policy to overcome its own failures; that requires turning these behaviors into learnable corrections for the policy. Our insight is that many such corrections are familiar short behaviors, or skills: they recur across tasks and describe actions that foundation models can reason about from a scene. We introduce skill-space shooting, which uses foundation model guidance to explore corrections through these reusable skills and turn successful trials into policy improvement. Real-world experiments show repeated improvement in policies acting autonomously, while skills can also be shared to reduce the teaching needed to improve on new tasks. By making reusable skills a source of corrective supervision, skill-space shooting enables scalable and generalizable policy improvement within and across tasks. 


## Repository structure

This repository brings together three components of the skill-space shooting pipeline. Each component has its own environment and detailed setup instructions.

| Directory | Role | Documentation |
| --- | --- | --- |
| `openpi_vlm_improve/` | UMI data conversion, repair-segment extraction, value-supervision data preparation, and ¦Ð?.? policy fine-tuning and serving. | [OpenPI README](openpi_vlm_improve/README.md) |
| `umi_vlm_improve/` | Robot-side execution, policy evaluation, value-model monitoring, VLM-guided skill selection, and rollout recording. | [UMI client README](umi_vlm_improve/README.md) |
| `value_model_vlm_improve/` | Training and serving a conditional value model that predicts future state and value from images and an action chunk. | [Value model README](value_model_vlm_improve/README.md) |
