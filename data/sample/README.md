# Sample: cube demonstration

One recorded demonstration (Intel RealSense RGB-D, third-person view): a person picks up a Rubik's cube and
places it on a box. This is the recording shown in the paper's qualitative figures.

`cube_demo/` is a run directory holding clip frames 41–113, every second frame (37 frames, 7.5 fps), with the
outputs of the GPU stages already filled in so the rest of the pipeline runs on a CPU:

| path | contents | produced by |
|---|---|---|
| `clip/rgb`, `clip/depth` | colour (JPEG) and aligned depth (uint16 mm) | `recast.io.bag` |
| `clip/intrinsics.json`, `clip/timestamps.csv` | RealSense colour intrinsics, frame times | `recast.io.bag` |
| `masks/{hand,arm,object,remove}` | segmentation masks and the dilated removal region | `recast.perception.segmentation` (Grounding DINO + SAM 2 backend) |
| `hands/hands.npz` | WiLoR hand mesh and joints registered to depth by ICP | `recast.perception.hands` |
| `inpainted/` | ProPainter output (hand and arm removed) | `recast.editing.inpaint` |
| `extrinsics.json` | robot-base pose used for rendering | virtual placement, see below |

No camera-to-robot calibration was recorded. `extrinsics.json` places a virtual Panda base on the floor plane
fitted to the sensor depth, out of view, the same placement as the paper's edited frames.

`hands.npz` contains reconstruction outputs only (vertex and joint positions). It does not contain the MANO
model, which is licensed separately.
