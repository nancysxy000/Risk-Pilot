"""Stage 1 工具函数"""

import torch
import numpy as np
import yaml


def load_config(config_path='configs/default.yaml'):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    """设置全局随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
