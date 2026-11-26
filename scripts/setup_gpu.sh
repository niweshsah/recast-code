#!/usr/bin/env bash
# Third-party models for the GPU stages, pinned (idempotent). Run from the repo root:
#   ./scripts/setup_gpu.sh                  # everything
#   ./scripts/setup_gpu.sh --menagerie-only # just the robot model (simulation, rendering, tests)
# MANO is licensed: register at https://mano.is.tue.mpg.de, then place MANO_RIGHT.pkl in third_party/WiLoR/mano_data/.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
TP=third_party
mkdir -p "$TP/weights"

clone() {  # url dir commit
  [ -d "$TP/$2/.git" ] || git clone -q "$1" "$TP/$2"
  git -C "$TP/$2" fetch -q --depth 1 origin "$3" 2>/dev/null || true
  git -C "$TP/$2" checkout -q "$3"
  echo "$2 @ $(git -C "$TP/$2" rev-parse --short HEAD)"
}
fetch() {  # url dest
  [ -s "$2" ] || { echo "downloading $(basename "$2")"; curl -fL --retry 5 -o "$2" "$1"; }
}

clone https://github.com/google-deepmind/mujoco_menagerie.git mujoco_menagerie eea4c7431b63da8278b0ff31638e0b0e941cec0e
[ "${1:-}" = "--menagerie-only" ] && exit 0

clone https://github.com/rolpotamias/WiLoR.git WiLoR fcb911312a38fa8badd30d9656a167485d61b8f9
clone https://github.com/sczhou/ProPainter.git ProPainter 326782529a63d997cab80ab21439b92ed422b3d3
clone https://github.com/facebookresearch/sam2.git sam2 2b90b9f5ceec907a1c18123530e92e794ad901a4

W=https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models
fetch "$W/wilor_final.ckpt" "$TP/WiLoR/pretrained_models/wilor_final.ckpt"
fetch "$W/detector.pt" "$TP/WiLoR/pretrained_models/detector.pt"
P=https://github.com/sczhou/ProPainter/releases/download/v0.1.0
mkdir -p "$TP/ProPainter/weights"
for f in ProPainter.pth raft-things.pth recurrent_flow_completion.pth; do fetch "$P/$f" "$TP/ProPainter/weights/$f"; done
fetch https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt "$TP/weights/sam2.1_hiera_large.pt"

uv pip install -e ".[perception]" -e "$TP/sam2" -r "$TP/WiLoR/requirements.txt"
uv pip install -r "$TP/ProPainter/requirements.txt"

[ -f "$TP/WiLoR/mano_data/MANO_RIGHT.pkl" ] && echo "MANO ok" \
  || echo "MISSING $TP/WiLoR/mano_data/MANO_RIGHT.pkl (licensed: download it from https://mano.is.tue.mpg.de)"
echo "SAM 3 (default segmentation backend) needs its own Python 3.12 env: see envs/sam3.txt"
