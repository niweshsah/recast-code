#!/usr/bin/env bash
# CPU-only stages on the committed sample (the GPU outputs are included): keypoints (Lucas-Kanade), contact
# signals, phases (HMM prior), action labels, robot rendering, figure panels. Writes to runs/sample.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
RUN=${1:-runs/sample}
rm -rf "$RUN" && mkdir -p "$(dirname "$RUN")" && cp -r data/sample/cube_demo "$RUN"
python -m recast.perception.keypoints "$RUN" --tracker lk --extrinsics "$RUN/extrinsics.json"
python -m recast.perception.contact "$RUN"
python -m recast.phases decode "$RUN"
python -m recast.actions "$RUN" --task cube
python -m recast.editing.render "$RUN" --menagerie third_party/mujoco_menagerie
python -m recast.viz.panels "$RUN"
echo "edited frames: $RUN/edited   panels: $RUN/viz/steps"
