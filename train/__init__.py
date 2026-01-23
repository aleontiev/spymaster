"""
Training modules for SpyMaster models.

This package contains training code split by model type:
- jepa: LeJEPA self-supervised pre-training
- entry: Entry policy training (3-class and 5-class classification)
- regression: Continuous regression entry policy
- directional: Directional move entry policy
"""

from train.common import set_seed, get_device, warmup_cosine_schedule
