#!/usr/bin/env bash
set -e

echo "DEVICE_SAMPLE_RATE=16000
TARGET_SAMPLE_RATE=16000
BLOCK_SIZE=128000
CHANNELS=1
LOW_CUTOFF_FREQ=50
HIGH_CUTOFF_FREQ=4000
BANDPASS_Q=0.707
NOISE_REDUCE_STATIONARY=True
NOISE_REDUCE_PROP_DECREASE=0.9
NOISE_TIME_CONSTANT_S=2
NOISE_REDUCE_NFFT=256
NOISE_REDUCE_WIN_LENGTH=100
NOISE_REDUCE_N_STD_THRESH_STATIONARY=0.05
LEVEL_SCALE=35
ENABLE_BANDPASS=True
ENABLE_NOISE_REDUCE=True
ENABLE_SCALE=True
CPU_SPEED_GHZ=1.2
CPU_CORES=4
MAX_MEMORY_USAGE_PROP=0.75
ENABLE_TORCH_PROFILER=False" > .env
