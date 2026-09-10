# Setup 1 — Install Guide
Isaac Sim 5.1 / Isaac Lab (training-checkpoints-develop) / cuRobo 0.7.7

## System requirements

| Component     | Version                                      |
|----------------|----------------------------------------------|
| OS             | Ubuntu 22.04.5 LTS (Jammy)                    |
| GPU            | NVIDIA, tested on RTX A2000 8GB Laptop GPU    |
| GPU driver     | 580.178.04 (or compatible)                    |
| CUDA Toolkit   | 12.8 (nvcc required — build-time only)        |

## 1. CUDA Toolkit 12.8

Install system-wide (e.g. to `/usr/local/cuda-12.8`). Needed only for building
cuRobo from source — the runtime CUDA libs come bundled with the PyTorch
wheel (`cu128`) later.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

## 2. Isaac Lab (bundles Isaac Sim)

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout f58361c8f1ea3380ded6ddfde029c40b0e7203ca
```

- Branch this commit sits on: `training-checkpoints-develop` — **not** a tagged
  `v2.3.2` release, so check out by this exact commit hash, not by version number.
- Editable subpackages installed as part of this: `isaaclab`, `isaaclab_assets`,
  `isaaclab_contrib`, `isaaclab_mimic`, `isaaclab_rl`, `isaaclab_tasks`.
- Run Isaac Lab's own installer next (however this checkout was originally set
  up — `./isaaclab.sh --install` or equivalent) so Isaac Sim gets bundled into
  `IsaacLab/_isaac_sim`.

⚠️ **Open risk:** the target Isaac Sim build is a release-candidate —
**5.1.0-rc.19+release.26219.9c81211b.gl** — not a general-release version.
Running the standard installer today may pull a different (later) Isaac Sim
build instead of this exact RC. Worth checking the resulting
`_isaac_sim/VERSION` file against this string after install, and going back to
NVIDIA/Omniverse support if it doesn't match and the exact RC is needed.

## 3. cuRobo

```bash
git clone https://github.com/NVlabs/curobo.git
cd curobo
git checkout d64c4b005459db10c5dd867d8b30a87d5bda9bdb
/path/to/IsaacLab/isaaclab.sh -p -m pip install -e .[isaacsim] --no-build-isolation
```

- Commit tag: `v0.7.7-9-gd64c4b0` (9 commits past the v0.7.7 release tag).
- `--no-build-isolation` is required — cuRobo compiles CUDA extensions against
  the already-installed PyTorch; build isolation would pull in a fresh
  mismatched one.

## 4. Verify

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/Pushing_task/Test_pushing_isaaclab.py
```

## Reference: key package versions

From the captured `pip_freeze.txt` — full file kept for diffing, not for a
literal `pip install -r requirements.txt` (see note below).

- `torch==2.7.0+cu128`
- `numpy==1.26.4`
- `onnxruntime-gpu==1.19.2`
- `warp-lang==1.12.1`
- `scipy==1.15.3`, `pandas==3.0.2`

## Not needed for this stack

The captured environment also contains a full ROS2/MoveIt/UR-driver stack
(`ament-*`, `ros2-*`, `moveit-*`, `ur-*`, `robotiq-tsf`, `tactilesensors4`,
`robotiq-2f-urcap-adapter*`) — that's a separate real-robot ROS2 workspace
sharing this Python environment, unrelated to running the simulation scripts.
Skip these on the new machine unless you're also setting up the real-robot side.
