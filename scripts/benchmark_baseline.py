"""Benchmark Baseline for Loading Datasets in Parallel."""

import sys
from time import sleep

from boxsup_pytorch.config import GLOBAL_CONFIG
from boxsup_pytorch.data.dataloader import BoxSupDataloader

try:
    num_worker = sys.argv[1]
    GLOBAL_CONFIG.num_worker = int(num_worker)
except IndexError:
    pass

try:
    sleep_time = sys.argv[2]
except IndexError:
    sleep_time = 0

loader = BoxSupDataloader("MASK")

dataloaders = loader.get_data_loader()

print(len(dataloaders["train"]))
max_idx = 0
for idx, data in enumerate(dataloaders["train"]):
    max_idx = max(max_idx, idx)
    sleep(int(sleep_time))
