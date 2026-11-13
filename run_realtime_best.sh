#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

exec .venv/bin/python realtime_loop.py \
  --sr-device \
  --block-time 0.4 \
  --extra-time-ce 1.5 \
  --extra-time 0.3 \
  --extra-time-right 0.1 \
  --diffusion-steps 10 \
  --max-prompt-length 5 \
  --checkpoint-path DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth \
  --config-path config_dit_mel_seed_uvit_whisper_small_wavenet.yml \
  "$@"
