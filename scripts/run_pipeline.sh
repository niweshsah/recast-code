#!/usr/bin/env bash
# Human RGB-D recording -> Recast demonstration. Usage:
#   ./scripts/run_pipeline.sh <recording.db3> <first_frame> <last_frame> <task> <run_dir> [extrinsics.json]
# GPU stages: segmentation, hands, keypoints (CoTracker), inpainting. Pass extrinsics (T_world_cam) from your
# camera-to-robot calibration; without it the robot base is placed virtually on the floor.
set -Eeuo pipefail
BAG=$1 FIRST=$2 LAST=$3 TASK=$4 RUN=$5 EXT=${6:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}
TP=third_party

python -m recast.io.bag "$BAG" --first "$FIRST" --last "$LAST" --out "$RUN/clip"
python -m recast.perception.segmentation "$RUN" --task "$TASK"
python -m recast.perception.hands "$RUN" --wilor "$TP/WiLoR"
python -m recast.perception.keypoints "$RUN" ${EXT:+--extrinsics "$EXT"}
python -m recast.perception.contact "$RUN"
python -m recast.phases decode "$RUN" ${PHASE_MODEL:+--model "$PHASE_MODEL"}
python -m recast.actions "$RUN" --task "$TASK"
python -m recast.editing.inpaint "$RUN" --propainter "$TP/ProPainter"
python -m recast.editing.render "$RUN" --menagerie "$TP/mujoco_menagerie"
echo "done: $RUN  (build training data with: python -m recast.data.episodes build $RUN ... --task $TASK)"
