#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for PPO with A* Guidance (环境定制升级版配置).
"""


class Config:

    # Feature dimensions / 特征维度
    # 升级为 12 维：2(电量比例, 包裹比例) + 8(A*动作One-Hot指引) + 1(目标距离归一化) + 1(是否存在明确目标)
    FEATURES = [15]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间
    # 修正为 8：无人机支持 8 个方向移动 (0-7)
    ACTION_NUM = 8

    # Value head / 价值头
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数 
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5