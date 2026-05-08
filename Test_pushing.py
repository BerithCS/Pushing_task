#!/usr/bin/env python3
"""
================================================================================
 Doosan M0609 — Tactile Sensing Pipeline with cuRobo Motion Planning
================================================================================

OVERVIEW
--------
This script runs a full tactile data collection pipeline in Isaac Sim using
a Doosan M0609 robot arm equipped with a CoRo tactile sensor (deformable
sponge). The robot pushes a cylinder object while simultaneously recording
deformable mesh data and running real-time CNN tactile inference.

PIPELINE PHASES (automatic, no user input required after pressing Play)
------------------------------------------------------------------------
  1. SETTLE      (10 frames)
     The simulation starts and waits for the physics to stabilize.
     The robot holds its initial joint position. No data is collected.
     → Console: "[PHASE] Settle done"

  2. BASELINE    (10 frames)
     The robot remains stationary at the initial pose (no contact).
     The deformable mesh dZ values are recorded and averaged to build
     a no-contact baseline. This baseline is later subtracted from all
     deformation measurements to remove gravity and pre-stress effects.
     No CSV data is saved during this phase.
     → Console: "[PHASE] Baseline done"

  3. PLAN_A      (variable duration)
     cuRobo plans and executes a free-space trajectory from the initial
     pose to Target A (pre-contact position in front of the cylinder).
     Orientation is held constant throughout this motion.
     No CSV data is saved during this phase.
     → Console: "[PHASE] Arrived at Target A"

  4. PUSH_Y      (variable duration)
     The robot executes a constrained straight push in the Y direction
     from Target A to Target B, pressing the sensor into the cylinder.
     Constraints: hold orientation + hold X + hold Z, free Y only.
     CSV recording starts at the beginning of this phase.
     → Console: "[PHASE] Arrived at Target B"

  5. END_WAIT    (10 frames)
     The robot holds its final position at Target B for 10 frames.
     CSV recording continues during this phase.
     After 10 frames the simulation closes automatically.
     → Console: "[PHASE] End wait done. Closing simulation."

OUTPUT FILES (saved to PROJECT_DIR with timestamp)
--------------------------------------------------
  {basename}_deformation_{timestamp}.csv
      One row per mesh node per frame.
      Columns: frame, t, node_id,
               s1_x/y/z (current position),
               s1_vx/vy/vz (velocity),
               s1_Rx/Ry/Rz (rest position),
               s1_Trans_x/y/z (sensor case translation),
               s1_Ori_w/x/y/z (sensor case orientation quaternion)

  {basename}_tactiledata_{timestamp}.csv
      One row per frame (much smaller file).
      Columns: frame, t, T_01 .. T_28
      The 28 values are the CNN tactile prediction for that frame,
      arranged as a 7x4 tactile map.

CNN INFERENCE
-------------
  At each recorded frame:
    1. All mesh node positions and rest positions are read from PhysX
    2. Rest points are rotated + translated to world frame using the
       sensor case pose (orientation quaternion + translation)
    3. Both rest and current points are scaled to a unit cube
    4. dZ = current_z - rest_z  (per filtered node, 216 nodes total)
    5. baseline is subtracted:  dZ = dZ - baseline
    6. dZ is reshaped to (18 x 12) and normalized by X_train_max
    7. ONNX model runs inference → outputs 28 tactile values (7 x 4 map)

KEY SETTINGS (edit at top of file)
-----------------------------------
  SETTLE_FRAMES     : frames to wait for physics stabilization (default 10)
  BASELINE_FRAMES   : frames used to compute no-contact baseline (default 10)
  END_FRAMES        : frames to record after reaching Target B (default 10)
  DISTANCE_TARGETS  : Y distance of the push in metres (default 0.3)
  CSV_BASENAME      : prefix for output CSV filenames (default "sponge_data")
  NODES_FILE        : path to Nodes_id_filtered.csv (216 filtered node IDs)
  ONNX_MODEL_PATH   : path to the trained ONNX tactile CNN model
  X_TRAIN_MAX_NPY   : path to the normalization factor used during training

  TARGET SETUP — HOW TO CONFIGURE PUSH POSITIONS
-----------------------------------------------
The push targets are derived automatically from the cylinder geometry.
To set up a new experiment, adjust the following values:

  OBJECT_PUSHED_BOTTOM_CENTER_WORLD = np.array([X, Y, Z])
      World position of the BOTTOM CENTER of the cylinder being pushed.
      X : lateral position (left/right in world frame)
      Y : depth position (forward/backward in world frame)
      Z : height of the cylinder base above the table surface

  OBJECT_HEIGHT = 0.10
      Height of the cylinder in metres.
      Used to compute the cylinder center Z = bottom_Z + height/2.

  OBJECT_DIAMETER = 0.075
      Diameter of the cylinder in metres (radius = diameter/2).
      Used to offset Target A so the sensor touches the cylinder surface.

  DISTANCE_TARGETS = 0.3
      Y distance in metres between Target A and Target B.
      This is how far the robot pushes through the cylinder.
      Larger value = longer push stroke.

HOW TARGETS ARE COMPUTED AUTOMATICALLY
---------------------------------------
  Cylinder center (world):
      X  = OBJECT_PUSHED_BOTTOM_CENTER_WORLD[0]
      Y  = OBJECT_PUSHED_BOTTOM_CENTER_WORLD[1]
      Z  = OBJECT_PUSHED_BOTTOM_CENTER_WORLD[2] + OBJECT_HEIGHT / 2

  Target A (pre-contact, sensor just touching cylinder):
      X  = cylinder center X
      Y  = cylinder center Y - OBJECT_RADIUS - 0.012 - 0.012 - 0.001
           │                   │               │       │       └─ small safety offset
           │                   │               │       └─ sensor height (12 mm)
           │                   │               └─ link6 to cylinder base distance (12 mm)
           │                   └─ cylinder radius
           └─ same X as cylinder
      Z  = cylinder center Z + 0.087715
           (0.087715 = distance from sensor center to link_6 in Z)

  Target B (end of push stroke):
      X  = same as Target A
      Y  = Target A Y + DISTANCE_TARGETS   ← robot pushes in +Y direction
      Z  = same as Target A

QUICK REFERENCE — WHAT TO CHANGE FOR A NEW EXPERIMENT
------------------------------------------------------
  New cylinder position   → change OBJECT_PUSHED_BOTTOM_CENTER_WORLD
  Different cylinder size → change OBJECT_HEIGHT and OBJECT_DIAMETER
  Longer/shorter push     → change DISTANCE_TARGETS
  Different sensor offset → change the 0.087715 constant (_target_z line)
                            and the 0.012 + 0.012 offsets (_target_y line)
================================================================================

FOLDER STRUCTURE EXPECTED
--------------------------
  Pushing_task/
  ├── Test_pushing.py           ← this script
  ├── scenes/
  │   └── doosan_station_full.usd
  ├── coro_doosan_station/
  │   └── m0609/
  │       └── m0609_adap_curobo.yml
  └── CNN_tactile/
      ├── Nodes_id_filtered.csv
      ├── best.onnx
      └── CNN_max.npy

DEPENDENCIES
------------
  Isaac Sim, cuRobo, onnxruntime, pandas, numpy, scipy
================================================================================
"""

try:
    import isaacsim
except ImportError:
    pass

import torch
_ = torch.zeros(4, device="cuda:0")

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": "1920", "height": "1080"})

import os
import csv
import datetime
import numpy as np
import pandas as pd
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.types import ArticulationAction

try:
    from omni.isaac.core.materials import OmniPBR
except ImportError:
    from isaacsim.core.api.materials import OmniPBR

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util_file import load_yaml
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

# ============================================================
# SETTINGS
# ============================================================
SCRIPT_DIR      = Path(__file__).resolve().parent
PROJECT_DIR     = str(SCRIPT_DIR)
ROBOT_DIR       = str(SCRIPT_DIR / "coro_doosan_station" / "m0609")
SCENE_USD       = str(SCRIPT_DIR / "scenes" / "doosan_station_full.usd")
ROBOT_CFG_FILE  = "m0609_adap_curobo.yml"
ROBOT_PRIM_PATH = "/World/m0609/m0609"
EE_LINK_NAME    = "link_6"

INITIAL_JOINT_DEGREES = np.array(
    [-72.09, 49.03, 57.46, -0.08, 73.51, -73.04],
    dtype=np.float32,
)

# ============================================================
# CNN / INFERENCE SETTINGS
# ============================================================
NODES_FILE      = str(SCRIPT_DIR / "CNN_tactile" / "Nodes_id_filtered.csv")
ONNX_MODEL_PATH = str(SCRIPT_DIR / "CNN_tactile" / "best.onnx")
X_TRAIN_MAX_NPY = str(SCRIPT_DIR / "CNN_tactile" / "CNN_max.npy")

N_ROWS = 18
N_COLS = 12

# ============================================================
# OBJECT PUSHED
# ============================================================
OBJECT_PUSHED_BOTTOM_CENTER_WORLD = np.array([0.22, -0.26, 0.96557], dtype=np.float64)
OBJECT_HEIGHT         = 0.10
OBJECT_DIAMETER       = 0.075
OBJECT_RADIUS         = OBJECT_DIAMETER / 2.0
OBJECT_MASS_KG        = 0.9
OBJECT_COLOR          = np.array([1.0, 0.0, 0.0])
OBJECT_CONTACT_OFFSET = 0.0001
OBJECT_REST_OFFSET    = 0.0
OBJECT_XFORM_PATH     = "/World/Object_pushed"
OBJECT_MESH_PATH      = "/World/Object_pushed/Cylinder"

CYLINDER_SOLVER_POS_ITERS = 240
CYLINDER_SOLVER_VEL_ITERS = 30

# ============================================================
# TARGET COMPUTATION
# ============================================================
DISTANCE_TARGETS = 0.3

_cyl_center_x = float(OBJECT_PUSHED_BOTTOM_CENTER_WORLD[0])
_cyl_center_y = float(OBJECT_PUSHED_BOTTOM_CENTER_WORLD[1])
_cyl_center_z = float(OBJECT_PUSHED_BOTTOM_CENTER_WORLD[2]) + OBJECT_HEIGHT / 2.0

_target_y = _cyl_center_y - OBJECT_RADIUS - 0.012 - 0.012 - 0.001
_target_z = _cyl_center_z + 0.087715

TARGET_A_WORLD = np.array([_cyl_center_x, _target_y,                    _target_z], dtype=np.float32)
TARGET_B_WORLD = np.array([_cyl_center_x, _target_y + DISTANCE_TARGETS, _target_z], dtype=np.float32)

# ============================================================
# DEFORMABLE MESH / CSV
# ============================================================
SPONGE_ROOT_PATH = "/World/CoRo_tactile/CoRo_tactile/Sponge"
SENSOR_POSE_PATH = "/World/CoRo_tactile/CoRo_tactile/Case_m"
CSV_DIR          = PROJECT_DIR
CSV_BASENAME     = "sponge_data"

# ============================================================
# TIMING
# ============================================================
PHYSICS_DT         = 1.0 / 60.0
INTERPOLATION_DT   = 0.009
STEPS_PER_CMD      = 1
TIME_DILATION      = 0.5
MAX_EFFORT         = 1500.0
KP_GAINS           = 200000.0
KD_GAINS           = 10000.0

SETTLE_FRAMES      = 10   # frames to wait for sim to stabilize
BASELINE_FRAMES    = 10   # frames to collect baseline (no contact)
END_FRAMES         = 10   # frames to wait after Target B before closing

# ============================================================
# PLANNING
# ============================================================
MAX_PLAN_FAILS  = 5
PUSH_Y_WEIGHT   = [1, 1, 1, 1, 0, 1]
HOLD_ORI_WEIGHT = [1, 1, 1, 0, 0, 0]

OBSTACLE_IGNORE = [
    "/World/m0609", "/World/looks",
    "/World/target_A", "/World/target_B",
    "/World/Object_pushed", "/World/GroundPlane",
    "/World/defaultGroundPlane", "/World/SphereLight",
    "/World/physicsScene", "/World/robot_mount",
    "/World/Environment/Geometry",
]

# ============================================================
# CNN HELPERS
# ============================================================

def load_ordered_node_ids(path):
    df  = pd.read_csv(path)
    col = "node_id" if "node_id" in df.columns else df.columns[0]
    seen, ids = set(), []
    for nid in df[col].dropna().astype(int):
        if nid not in seen:
            seen.add(nid); ids.append(nid)
    return ids


def rotation_matrix_np(w, x, y, z):
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - z*w),       2*(x*z + y*w)    ],
        [2*(x*y + z*w),        1 - 2*(x**2 + z**2),  2*(y*z - x*w)    ],
        [2*(x*z - y*w),        2*(y*z + x*w),        1 - 2*(x**2+y**2)],
    ])


def scale_to_cube(pts, cx, cy, cz, sx, sy, sz):
    out = np.zeros_like(pts)
    out[:, 0] = (pts[:, 0] - cx) / sx + cx
    out[:, 1] = (pts[:, 1] - cy) / sy + cy
    out[:, 2] = (pts[:, 2] - cz) / sz + cz
    return out


def compute_dz_from_arrays(rest_pts, curr_pts, R_mat, T, ordered_ids, all_node_ids):
    sx = rest_pts[:, 0].max() - rest_pts[:, 0].min()
    sy = rest_pts[:, 1].max() - rest_pts[:, 1].min()
    sz = rest_pts[:, 2].max() - rest_pts[:, 2].min()

    rot_pts = (R_mat @ rest_pts.T).T + T
    cx_w = (rot_pts[:, 0].min() + rot_pts[:, 0].max()) / 2
    cy_w = (rot_pts[:, 1].min() + rot_pts[:, 1].max()) / 2
    cz_w = (rot_pts[:, 2].min() + rot_pts[:, 2].max()) / 2

    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    filter_idx  = [node_to_idx[nid] for nid in ordered_ids if nid in node_to_idx]

    rest_f = rest_pts[filter_idx]
    curr_f = curr_pts[filter_idx]
    rot_f  = (R_mat @ rest_f.T).T + T

    black  = scale_to_cube(rot_f,  cx_w, cy_w, cz_w, sx, sy, sz)
    orange = scale_to_cube(curr_f, cx_w, cy_w, cz_w, sx, sy, sz)

    return orange[:, 2] - black[:, 2]


def get_dz(pos_attr, res_attr, pose_prim, ordered_ids):
    """Helper to compute dz from live sim attributes."""
    P  = pos_attr.Get() if pos_attr else None
    Re = res_attr.Get() if res_attr else None
    if P is None or Re is None:
        return None

    t_attr = pose_prim.GetAttribute("xformOp:translate")
    o_attr = pose_prim.GetAttribute("xformOp:orient")
    trans  = t_attr.Get() if (t_attr and t_attr.IsValid()) else Gf.Vec3d(0, 0, 0)
    ori    = o_attr.Get() if (o_attr and o_attr.IsValid()) else None

    if ori is not None and hasattr(ori, "GetReal"):
        ow = float(ori.GetReal()); img = ori.GetImaginary()
        ox, oy, oz = float(img[0]), float(img[1]), float(img[2])
    else:
        ow, ox, oy, oz = 1.0, 0.0, 0.0, 0.0

    n        = len(P)
    rest_pts = np.array([[float(Re[i][0]), float(Re[i][1]), float(Re[i][2])] for i in range(n)])
    curr_pts = np.array([[float(P[i][0]),  float(P[i][1]),  float(P[i][2])]  for i in range(n)])
    R_mat    = rotation_matrix_np(ow, ox, oy, oz)
    T        = np.array([float(trans[0]), float(trans[1]), float(trans[2])])

    return compute_dz_from_arrays(rest_pts, curr_pts, R_mat, T,
                                  ordered_ids, list(range(n)))


def infer_tactile(dz, baseline, x_train_max, session, input_name):
    dz_corrected = dz - baseline
    dz_grid      = dz_corrected.reshape(N_ROWS, N_COLS).astype(np.float32)
    X            = np.clip(dz_grid / x_train_max, -1.0, 1.0)
    X            = X.reshape(1, N_ROWS, N_COLS, 1).astype(np.float32)
    y_pred       = session.run(None, {input_name: X})[0]
    return y_pred[0].flatten()  # (28,)


# ============================================================
# DEFORMABLE MESH HELPERS
# ============================================================

def _has_any_deformable_api(prim: Usd.Prim) -> bool:
    if prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):    return True
    if prim.HasAPI(PhysxSchema.PhysxDeformableSurfaceAPI): return True
    return any("OmniPhysicsDeformable" in s for s in prim.GetAppliedSchemas())


def find_deformable_simulation_mesh(root_path: str):
    stage = omni.usd.get_context().get_stage()
    root  = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[DEFORMABLE] WARNING: root prim not found: {root_path}")
        return None
    for prim in Usd.PrimRange(root):
        if _has_any_deformable_api(prim):
            for child in prim.GetChildren():
                if "simulation_mesh" in child.GetName().lower():
                    print(f"[DEFORMABLE] Found: {child.GetPath()}")
                    return child
    print(f"[DEFORMABLE] WARNING: simulation_mesh not found under {root_path}")
    return None


# ============================================================
# CSV HELPERS
# ============================================================

def prepare_csv(csv_dir: str, basename: str):
    os.makedirs(csv_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Deformation file — one row per node per frame
    def_path   = os.path.join(csv_dir, f"{basename}_deformation_{ts}.csv")
    def_file   = open(def_path, "w", newline="")
    def_writer = csv.writer(def_file)
    def_writer.writerow([
        "frame", "t", "node_id",
        "s1_x",  "s1_y",  "s1_z",
        "s1_vx", "s1_vy", "s1_vz",
        "s1_Rx", "s1_Ry", "s1_Rz",
        "s1_Trans_x", "s1_Trans_y", "s1_Trans_z",
        "s1_Ori_w",   "s1_Ori_x",   "s1_Ori_y",   "s1_Ori_z",
    ])
    def_file.flush()
    print(f"[CSV] Deformation file : {def_path}")

    # Tactile file — one row per frame
    tactile_cols = [f"T_{i+1:02d}" for i in range(28)]
    tac_path   = os.path.join(csv_dir, f"{basename}_tactiledata_{ts}.csv")
    tac_file   = open(tac_path, "w", newline="")
    tac_writer = csv.writer(tac_file)
    tac_writer.writerow(["frame", "t"] + tactile_cols)
    tac_file.flush()
    print(f"[CSV] Tactile file     : {tac_path}")

    return def_file, def_writer, tac_file, tac_writer


def save_sponge_data(pos_attr, vel_attr, res_attr, pose_prim,
                     def_writer, def_file,
                     tac_writer, tac_file,
                     frame: int, sim_time: float,
                     ordered_ids, baseline, x_train_max, session, input_name):

    P  = pos_attr.Get() if pos_attr else None
    V  = vel_attr.Get() if vel_attr else None
    Re = res_attr.Get() if res_attr else None
    if P is None:
        return

    t_attr = pose_prim.GetAttribute("xformOp:translate")
    o_attr = pose_prim.GetAttribute("xformOp:orient")
    trans  = t_attr.Get() if (t_attr and t_attr.IsValid()) else Gf.Vec3d(0, 0, 0)
    ori    = o_attr.Get() if (o_attr and o_attr.IsValid()) else None

    if ori is not None and hasattr(ori, "GetReal"):
        ori_w = float(ori.GetReal())
        imag  = ori.GetImaginary()
        ori_x, ori_y, ori_z = float(imag[0]), float(imag[1]), float(imag[2])
    else:
        ori_w, ori_x, ori_y, ori_z = 1.0, 0.0, 0.0, 0.0

    n, n_v, n_re = len(P), len(V) if V else 0, len(Re) if Re else 0

    # Compute dZ and infer tactile once per frame
    tactile_vals = np.zeros(28, dtype=np.float32)
    if Re is not None and len(Re) > 0:
        try:
            rest_pts = np.array([[float(Re[i][0]), float(Re[i][1]), float(Re[i][2])]
                                  for i in range(n)], dtype=float)
            curr_pts = np.array([[float(P[i][0]),  float(P[i][1]),  float(P[i][2])]
                                  for i in range(n)], dtype=float)
            R_mat = rotation_matrix_np(ori_w, ori_x, ori_y, ori_z)
            T     = np.array([float(trans[0]), float(trans[1]), float(trans[2])])
            dz    = compute_dz_from_arrays(rest_pts, curr_pts, R_mat, T,
                                           ordered_ids, list(range(n)))
            tactile_vals = infer_tactile(dz, baseline, x_train_max, session, input_name)
        except Exception as e:
            print(f"[WARN] Inference failed at frame {frame}: {e}")

    # Write one row per node to deformation file
    for i in range(n):
        vx = vy = vz = 0.0
        if V  is not None and i < n_v:  vx, vy, vz = float(V[i][0]),  float(V[i][1]),  float(V[i][2])
        rx = ry = rz = 0.0
        if Re is not None and i < n_re: rx, ry, rz = float(Re[i][0]), float(Re[i][1]), float(Re[i][2])

        def_writer.writerow([
            frame, f"{sim_time:.6f}", i,
            float(P[i][0]), float(P[i][1]), float(P[i][2]),
            vx, vy, vz,
            rx, ry, rz,
            float(trans[0]), float(trans[1]), float(trans[2]),
            ori_w, ori_x, ori_y, ori_z,
        ])

    # Write one row per frame to tactile file
    tac_writer.writerow([frame, f"{sim_time:.6f}"] + tactile_vals.tolist())

    if frame % 60 == 0:
        print(f"[CSV] frame={frame}  t={sim_time:.3f}s  nodes={n}  tactile={tactile_vals[:4].round(3)}")
        def_file.flush()
        tac_file.flush()


# ============================================================
# ROBOT / MOTION HELPERS
# ============================================================

def apply_physics_material(stage, prim_path, static_friction, dynamic_friction,
                            restitution=0.0):
    mat_path    = "/World/looks/" + prim_path.replace("/", "_").strip("_") + "_physicsMat"
    mat_prim    = stage.DefinePrim(mat_path, "Material")
    physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    physics_mat.GetStaticFrictionAttr().Set(float(static_friction))
    physics_mat.GetDynamicFrictionAttr().Set(float(dynamic_friction))
    physics_mat.GetRestitutionAttr().Set(float(restitution))
    target_prim = stage.GetPrimAtPath(prim_path)
    if target_prim.IsValid():
        UsdShade.MaterialBindingAPI(target_prim).Bind(
            UsdShade.Material(mat_prim),
            UsdShade.Tokens.weakerThanDescendants, "physics",
        )
        print(f"  Physics mat → {prim_path}  (s={static_friction}, d={dynamic_friction})")
    else:
        print(f"  WARNING: prim not found: {prim_path}")


def create_object_pushed(stage):
    bottom     = OBJECT_PUSHED_BOTTOM_CENTER_WORLD
    xform_prim = UsdGeom.Xform.Define(stage, OBJECT_XFORM_PATH)
    xform_prim.AddTranslateOp().Set(Gf.Vec3d(*bottom.tolist()))

    xform_p  = stage.GetPrimAtPath(OBJECT_XFORM_PATH)
    UsdPhysics.RigidBodyAPI.Apply(xform_p)
    mass_api = UsdPhysics.MassAPI.Apply(xform_p)
    mass_api.GetMassAttr().Set(float(OBJECT_MASS_KG))
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, float(OBJECT_HEIGHT / 2.0)))

    cylinder = UsdGeom.Cylinder.Define(stage, OBJECT_MESH_PATH)
    cylinder.GetHeightAttr().Set(float(OBJECT_HEIGHT))
    cylinder.GetRadiusAttr().Set(float(OBJECT_RADIUS))
    cylinder.GetAxisAttr().Set("Z")
    cyl_xform = UsdGeom.Xformable(cylinder.GetPrim())
    cyl_xform.ClearXformOpOrder()
    cyl_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(OBJECT_HEIGHT / 2.0)))

    cyl_prim = stage.GetPrimAtPath(OBJECT_MESH_PATH)
    UsdPhysics.CollisionAPI.Apply(cyl_prim)
    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(cyl_prim)
    physx_col.GetContactOffsetAttr().Set(float(OBJECT_CONTACT_OFFSET))
    physx_col.GetRestOffsetAttr().Set(float(OBJECT_REST_OFFSET))

    mat_path    = "/World/looks/ObjectPushedMat"
    mat_prim    = stage.DefinePrim(mat_path, "Material")
    shader_prim = stage.DefinePrim(mat_path + "/Shader", "Shader")
    shader_prim.CreateAttribute("info:id", Sdf.ValueTypeNames.Token).Set("UsdPreviewSurface")
    shader_prim.CreateAttribute("inputs:diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*OBJECT_COLOR.tolist()))
    shader_prim.CreateAttribute("inputs:metallic",  Sdf.ValueTypeNames.Float).Set(0.0)
    shader_prim.CreateAttribute("inputs:roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    material = UsdShade.Material(mat_prim)
    shader   = UsdShade.Shader(shader_prim)
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI(cyl_prim).Bind(material)
    print(f"Object_pushed created at Z={bottom[2]:.5f} m")


def apply_cylinder_solver_iters(stage):
    xform_p = stage.GetPrimAtPath(OBJECT_XFORM_PATH)
    if not xform_p.IsValid():
        print("  WARNING: Cylinder Xform not found.")
        return
    for name, val in [
        ("physxRigidBody:solverPositionIterationCount", CYLINDER_SOLVER_POS_ITERS),
        ("physxRigidBody:solverVelocityIterationCount", CYLINDER_SOLVER_VEL_ITERS),
    ]:
        attr = xform_p.GetAttribute(name)
        if attr and attr.IsValid(): attr.Set(val)
        else: xform_p.CreateAttribute(name, Sdf.ValueTypeNames.UInt).Set(val)
        print(f"  Cylinder {name} = {val}")


def world_to_base_frame(point_world, base_pos, base_ori_wxyz):
    w, x, y, z = base_ori_wxyz
    return R.from_quat([x, y, z, w]).inv().apply(point_world - base_pos).astype(np.float32)


def get_curobo_fk(motion_gen, joint_positions, joint_names, tensor_args):
    cu_js = JointState(
        position=tensor_args.to_device(torch.tensor(joint_positions, dtype=torch.float32)),
        velocity=tensor_args.to_device(torch.zeros(len(joint_positions), dtype=torch.float32)),
        acceleration=tensor_args.to_device(torch.zeros(len(joint_positions), dtype=torch.float32)),
        jerk=tensor_args.to_device(torch.zeros(len(joint_positions), dtype=torch.float32)),
        joint_names=joint_names,
    )
    cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
    kin   = motion_gen.kinematics.get_state(cu_js.position.unsqueeze(0))
    return kin.ee_position[0].detach().cpu().numpy(), kin.ee_quaternion[0].detach().cpu().numpy()


def init_targets(robot, ee_prim, motion_gen, init_positions, j_names,
                 tensor_args, target_a, target_b):
    base_pos, base_ori           = robot.get_world_pose()
    curobo_ee_pos, curobo_ee_ori = get_curobo_fk(motion_gen, init_positions, j_names, tensor_args)

    print(f"Robot base pos   : {base_pos.tolist()}")
    print(f"cuRobo FK EE pos : {curobo_ee_pos.tolist()}")
    print(f"Target A (world) : {TARGET_A_WORLD.tolist()}")
    print(f"Target B (world) : {TARGET_B_WORLD.tolist()}")

    goal_ori_base    = curobo_ee_ori.astype(np.float32)
    target_list_base = []
    target_ori_base  = []

    for pt_world, tgt in zip([TARGET_A_WORLD, TARGET_B_WORLD], [target_a, target_b]):
        pt_base = world_to_base_frame(pt_world, base_pos, base_ori)
        target_list_base.append(pt_base)
        target_ori_base.append(goal_ori_base.copy())
        tgt.set_world_pose(position=pt_world, orientation=ee_prim.get_world_pose()[1])

    return target_list_base, target_ori_base


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Scene USD      : {SCENE_USD}")
    print(f"Target A       : {TARGET_A_WORLD.tolist()}")
    print(f"Target B       : {TARGET_B_WORLD.tolist()}")

    # ── Load CNN resources ────────────────────────────────────
    print("Loading CNN resources...")
    ordered_ids = load_ordered_node_ids(NODES_FILE)
    x_train_max = float(np.load(X_TRAIN_MAX_NPY))
    session     = ort.InferenceSession(ONNX_MODEL_PATH,
                                       providers=["CUDAExecutionProvider",
                                                  "CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    print(f"  Nodes       : {len(ordered_ids)}")
    print(f"  X_train_max : {x_train_max:.6f}")
    print(f"  ONNX input  : {input_name}")

    baseline        = None
    baseline_buffer = []

    my_world = World(stage_units_in_meters=1.0)
    stage    = my_world.stage

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    stage.GetRootLayer().subLayerPaths.append(SCENE_USD)
    simulation_app.update()

    # ── GPU dynamics ──────────────────────────────────────────
    physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
    if physics_scene_prim.IsValid():
        physics_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
        physics_scene_api.GetEnableGPUDynamicsAttr().Set(True)
        physics_scene_api.GetBroadphaseTypeAttr().Set("GPU")
        print("GPU dynamics enabled.")

    # ── Disable table_base collision ──────────────────────────
    _tbc = stage.GetPrimAtPath("/World/table_base/table_base/collisions")
    if _tbc.IsValid():
        _tbc.SetActive(False)
        print("  table_base collisions deactivated.")

    # ── Deformable physics tuning ─────────────────────────────
    def _set_attr(prim, name, value, type_name):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid(): attr.Set(value)
        else: prim.CreateAttribute(name, type_name).Set(value)
        print(f"  [Deformable] {name} = {value}")

    _sponge_prim = stage.GetPrimAtPath("/World/CoRo_tactile/CoRo_tactile/Sponge")
    _mat_prim    = stage.GetPrimAtPath("/World/CoRo_tactile/CoRo_tactile/Sponge/Looks/Deformable_Material")

    if _sponge_prim.IsValid():
        _set_attr(_sponge_prim, "omniphysics:mass",                             0.09,    Sdf.ValueTypeNames.Float)
        _set_attr(_sponge_prim, "physxDeformable:solverPositionIterationCount", 80,      Sdf.ValueTypeNames.UInt)
        _set_attr(_sponge_prim, "physxDeformable:sleepThreshold",               0.00001, Sdf.ValueTypeNames.Float)
        _set_attr(_sponge_prim, "physxDeformable:disableGravity",               True,    Sdf.ValueTypeNames.Bool)
    else:
        print("  WARNING: Sponge prim not found.")

    if _mat_prim.IsValid():
        _set_attr(_mat_prim, "omniphysics:youngsModulus",                 1289000.0, Sdf.ValueTypeNames.Float)
        _set_attr(_mat_prim, "omniphysics:poissonsRatio",                 0.1729,    Sdf.ValueTypeNames.Float)
        _set_attr(_mat_prim, "omniphysics:density",                       1240.0,    Sdf.ValueTypeNames.Float)
        _set_attr(_mat_prim, "physxDeformableMaterial:elasticityDamping", 15.0,      Sdf.ValueTypeNames.Float)
    else:
        print("  WARNING: Deformable_Material prim not found.")

    # ── Object pushed + solver ────────────────────────────────
    create_object_pushed(stage)
    apply_cylinder_solver_iters(stage)

    # ── Physics materials ─────────────────────────────────────
    apply_physics_material(stage, OBJECT_MESH_PATH,                                        0.2, 0.15)
    apply_physics_material(stage, "/World/table_cover/Cube",                                0.9,  0.9)
    apply_physics_material(stage, "/World/m0609/m0609/link_6/adapter",                      0.4,  0.35)
    apply_physics_material(stage, "/World/CoRo_tactile/CoRo_tactile/Sponge/collision_mesh", 0.9,  0.8)

    simulation_app.update()

    # ── Scene objects ─────────────────────────────────────────
    robot   = my_world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="robot"))
    ee_prim = XFormPrim(prim_path=f"{ROBOT_PRIM_PATH}/{EE_LINK_NAME}", name="ee_link")

    mat_a    = OmniPBR("/World/looks/tA", color=np.array([0.1, 0.1, 0.1]))
    mat_b    = OmniPBR("/World/looks/tB", color=np.array([0.1, 0.1, 0.1]))
    target_a = cuboid.VisualCuboid("/World/target_A", position=TARGET_A_WORLD,
                                   orientation=np.array([1.,0.,0.,0.]), size=0.04, visual_material=mat_a)
    target_b = cuboid.VisualCuboid("/World/target_B", position=TARGET_B_WORLD,
                                   orientation=np.array([1.,0.,0.,0.]), size=0.04, visual_material=mat_b)
    material_list = [mat_a, mat_b]

    # ── cuRobo setup ──────────────────────────────────────────
    usd_helper = UsdHelper()
    usd_helper.load_stage(my_world.stage)
    world_cfg = usd_helper.get_obstacles_from_stage(
        reference_prim_path="/World", ignore_substring=OBSTACLE_IGNORE,
    )

    tensor_args = TensorDeviceType()
    robot_cfg   = load_yaml(os.path.join(ROBOT_DIR, ROBOT_CFG_FILE))["robot_cfg"]
    robot_cfg["kinematics"]["external_asset_path"]         = ROBOT_DIR
    robot_cfg["kinematics"]["external_robot_configs_path"] = ROBOT_DIR

    j_names        = robot_cfg["kinematics"]["cspace"]["joint_names"]
    init_positions = np.deg2rad(INITIAL_JOINT_DEGREES).tolist()
    robot_cfg["kinematics"]["cspace"]["retract_config"] = init_positions

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg, world_cfg, tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": 10, "mesh": 10},
        interpolation_dt=INTERPOLATION_DT,
        ee_link_name=EE_LINK_NAME,
        num_trajopt_seeds=12, num_graph_seeds=12,
        project_pose_to_goal_frame=False,
    )
    motion_gen = MotionGen(motion_gen_config)
    print("Warming up cuRobo...")
    motion_gen.warmup(warmup_js_trajopt=False)
    print("cuRobo ready.")

    my_world.initialize_physics()
    robot.initialize()
    ctrl = robot.get_articulation_controller()

    plan_config = MotionGenPlanConfig(
        enable_graph=False, enable_graph_attempt=4,
        max_attempts=4, enable_finetune_trajopt=True,
        time_dilation_factor=TIME_DILATION, pose_cost_metric=None,
    )

    # ── Deformable mesh attrs ─────────────────────────────────
    sim_mesh_prim = find_deformable_simulation_mesh(SPONGE_ROOT_PATH)
    if sim_mesh_prim is None:
        pos_attr = vel_attr = res_attr = None
    else:
        pos_attr = sim_mesh_prim.GetAttribute("points")
        vel_attr = sim_mesh_prim.GetAttribute("velocities")
        res_attr = sim_mesh_prim.GetAttribute("omniphysics:restShapePoints")
        print(f"[DEFORMABLE] Nodes: {len(pos_attr.Get()) if pos_attr.Get() else 'N/A'}")

    sensor_prim = omni.usd.get_context().get_stage().GetPrimAtPath(SENSOR_POSE_PATH)
    if not sensor_prim or not sensor_prim.IsValid():
        print(f"[WARNING] Sensor prim not found: {SENSOR_POSE_PATH}")
        sensor_prim = None

    # ── Prepare CSV files ─────────────────────────────────────
    def_file, def_writer, tac_file, tac_writer = prepare_csv(CSV_DIR, CSV_BASENAME)

    # ── State machine ─────────────────────────────────────────
    #
    # Phases (in order):
    #   "settle"      → wait SETTLE_FRAMES for sim to stabilize
    #   "baseline"    → collect BASELINE_FRAMES of dz (no motion, no save)
    #   "plan_A"      → plan + execute trajectory to Target A
    #   "push_y"      → plan + execute push from A to B, SAVING data
    #   "end_wait"    → wait END_FRAMES after Target B then close
    #
    phase              = "settle"
    phase_counter      = 0          # counts frames within current phase
    playing_started    = False      # set True once Play is detected

    cmd_idx = cmd_step_idx = 0
    cmd_plan = idx_list    = None
    targets_initialized    = False
    target_list_base       = []
    target_ori_base        = []
    planned_phase          = None
    last_curobo_goal_pos   = None
    last_curobo_goal_names = None
    plan_fail_count        = 0
    recording_active       = False
    sim_time               = 0.0
    i                      = 0

    # ── Main loop ─────────────────────────────────────────────
    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            if i % 100 == 0: print("**** Click Play to start ****")
            i += 1
            continue

        step      = my_world.current_time_step_index
        sim_time += PHYSICS_DT

        # Joint init on first steps
        if step <= 10:
            robot._articulation_view.initialize()
            idx = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(init_positions, idx)
            robot._articulation_view.set_max_efforts(np.array([MAX_EFFORT]*len(idx)), joint_indices=idx)
            robot._articulation_view.set_gains(
                kps=np.array([KP_GAINS]*len(idx)),
                kds=np.array([KD_GAINS]*len(idx)),
                joint_indices=idx,
            )

        if step < 2:
            continue

        # ── PHASE: settle ─────────────────────────────────────
        if phase == "settle":
            phase_counter += 1
            if phase_counter >= SETTLE_FRAMES:
                print(f"[PHASE] Settle done ({SETTLE_FRAMES} frames). Starting baseline collection.")
                phase         = "baseline"
                phase_counter = 0
            continue

        # ── PHASE: baseline ───────────────────────────────────
        if phase == "baseline":
            if pos_attr is not None and sensor_prim is not None:
                dz = get_dz(pos_attr, res_attr, sensor_prim, ordered_ids)
                if dz is not None:
                    baseline_buffer.append(dz)

            phase_counter += 1
            if phase_counter >= BASELINE_FRAMES:
                if len(baseline_buffer) > 0:
                    baseline = np.mean(baseline_buffer, axis=0)
                    print(f"[PHASE] Baseline done ({len(baseline_buffer)} frames). "
                          f"mean={baseline.mean():.6f}  Starting planning to Target A.")
                else:
                    baseline = np.zeros(N_ROWS * N_COLS, dtype=np.float32)
                    print("[PHASE] Baseline done but no data collected — using zeros.")
                phase         = "plan_A"
                phase_counter = 0

                # Init targets now that baseline is ready
                if not targets_initialized:
                    target_list_base, target_ori_base = init_targets(
                        robot, ee_prim, motion_gen, init_positions, j_names,
                        tensor_args, target_a, target_b,
                    )
                    targets_initialized = True
            continue

        # ── PHASE: plan_A — move to Target A ─────────────────
        if phase == "plan_A":
            # Plan if no active trajectory
            if cmd_plan is None and step % 10 == 0:
                plan_config.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device(HOLD_ORI_WEIGHT),
                )
                print("-"*50 + "\nPHASE: plan_A → Target A\n" + "-"*50)

                for k, m in enumerate(material_list):
                    m.set_color(np.array([0.,1.,0.]) if k==0 else np.array([0.1,0.1,0.1]))

                sim_js = robot.get_joints_state()
                cu_js  = JointState(
                    position=tensor_args.to_device(sim_js.positions),
                    velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                    acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                    jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                    joint_names=robot.dof_names,
                )
                cu_js   = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
                ik_goal = Pose(
                    position=tensor_args.to_device(
                        torch.tensor(target_list_base[0], dtype=torch.float32)),
                    quaternion=tensor_args.to_device(
                        torch.tensor(target_ori_base[0], dtype=torch.float32)),
                )
                result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
                succ   = result.success.item()
                print(f"  Planning to Target A: {'succeeded' if succ else 'failed'}")

                if succ:
                    plan_fail_count        = 0
                    curobo_plan            = result.get_interpolated_plan()
                    last_curobo_goal_pos   = curobo_plan.position[-1].clone()
                    last_curobo_goal_names = curobo_plan.joint_names
                    cmd_plan               = motion_gen.get_full_js(curobo_plan)
                    common                 = [x for x in robot.dof_names if x in cmd_plan.joint_names]
                    idx_list               = [robot.get_dof_index(x) for x in common]
                    cmd_plan               = cmd_plan.get_ordered_joint_state(common)
                    cmd_idx = cmd_step_idx = 0
                    planned_phase          = "plan_A"
                else:
                    plan_fail_count += 1
                    if plan_fail_count >= MAX_PLAN_FAILS:
                        print(f"  {MAX_PLAN_FAILS} failures to Target A. Aborting.")
                        break

            # Execute trajectory
            if cmd_plan is not None:
                cmd = cmd_plan[cmd_idx]
                ctrl.apply_action(ArticulationAction(
                    cmd.position.cpu().numpy(),
                    cmd.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                ))
                cmd_step_idx += 1
                if cmd_step_idx >= STEPS_PER_CMD:
                    cmd_idx += 1; cmd_step_idx = 0

                if cmd_idx >= len(cmd_plan.position):
                    final_pos = cmd_plan.position[-1].cpu().numpy()
                    robot.set_joint_positions(final_pos, joint_indices=idx_list)
                    try: robot.set_joint_velocities(np.zeros_like(final_pos), joint_indices=idx_list)
                    except Exception: pass

                    cmd_idx = cmd_step_idx = 0
                    cmd_plan      = None
                    planned_phase = None
                    plan_fail_count = 0

                    print("[PHASE] Arrived at Target A — starting recording and Y push.")
                    recording_active = True
                    phase            = "push_y"
                    phase_counter    = 0
            continue

        # ── PHASE: push_y — push from A to B, saving data ────
        if phase == "push_y":

            # Save data every frame
            if recording_active and pos_attr is not None and sensor_prim is not None:
                save_sponge_data(
                    pos_attr, vel_attr, res_attr, sensor_prim,
                    def_writer, def_file,
                    tac_writer, tac_file,
                    frame=step, sim_time=sim_time,
                    ordered_ids=ordered_ids, baseline=baseline,
                    x_train_max=x_train_max, session=session,
                    input_name=input_name,
                )

            # Plan if no active trajectory
            if cmd_plan is None and step % 10 == 0:
                plan_config.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device(PUSH_Y_WEIGHT),
                )
                print("-"*50 + "\nPHASE: push_y → Target B\n" + "-"*50)

                for k, m in enumerate(material_list):
                    m.set_color(np.array([0.,1.,0.]) if k==1 else np.array([0.1,0.1,0.1]))

                cu_js = JointState(
                    position=last_curobo_goal_pos,
                    velocity=torch.zeros_like(last_curobo_goal_pos),
                    acceleration=torch.zeros_like(last_curobo_goal_pos),
                    jerk=torch.zeros_like(last_curobo_goal_pos),
                    joint_names=last_curobo_goal_names,
                )
                cu_js   = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
                ik_goal = Pose(
                    position=tensor_args.to_device(
                        torch.tensor(target_list_base[1], dtype=torch.float32)),
                    quaternion=tensor_args.to_device(
                        torch.tensor(target_ori_base[1], dtype=torch.float32)),
                )
                result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
                succ   = result.success.item()
                print(f"  Planning to Target B: {'succeeded' if succ else 'failed'}")

                if succ:
                    plan_fail_count        = 0
                    curobo_plan            = result.get_interpolated_plan()
                    last_curobo_goal_pos   = curobo_plan.position[-1].clone()
                    last_curobo_goal_names = curobo_plan.joint_names
                    cmd_plan               = motion_gen.get_full_js(curobo_plan)
                    common                 = [x for x in robot.dof_names if x in cmd_plan.joint_names]
                    idx_list               = [robot.get_dof_index(x) for x in common]
                    cmd_plan               = cmd_plan.get_ordered_joint_state(common)
                    cmd_idx = cmd_step_idx = 0
                    planned_phase          = "push_y"
                else:
                    plan_fail_count += 1
                    if plan_fail_count >= MAX_PLAN_FAILS:
                        print(f"  {MAX_PLAN_FAILS} failures to Target B. Aborting.")
                        break

            # Execute trajectory
            if cmd_plan is not None:
                cmd = cmd_plan[cmd_idx]
                ctrl.apply_action(ArticulationAction(
                    cmd.position.cpu().numpy(),
                    cmd.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                ))
                cmd_step_idx += 1
                if cmd_step_idx >= STEPS_PER_CMD:
                    cmd_idx += 1; cmd_step_idx = 0

                if cmd_idx >= len(cmd_plan.position):
                    final_pos = cmd_plan.position[-1].cpu().numpy()
                    robot.set_joint_positions(final_pos, joint_indices=idx_list)
                    try: robot.set_joint_velocities(np.zeros_like(final_pos), joint_indices=idx_list)
                    except Exception: pass

                    cmd_idx = cmd_step_idx = 0
                    cmd_plan      = None
                    planned_phase = None

                    print(f"[PHASE] Arrived at Target B — waiting {END_FRAMES} frames then closing.")
                    phase         = "end_wait"
                    phase_counter = 0
            continue

        # ── PHASE: end_wait ───────────────────────────────────
        if phase == "end_wait":

            # Keep saving during end wait
            if recording_active and pos_attr is not None and sensor_prim is not None:
                save_sponge_data(
                    pos_attr, vel_attr, res_attr, sensor_prim,
                    def_writer, def_file,
                    tac_writer, tac_file,
                    frame=step, sim_time=sim_time,
                    ordered_ids=ordered_ids, baseline=baseline,
                    x_train_max=x_train_max, session=session,
                    input_name=input_name,
                )

            phase_counter += 1
            if phase_counter >= END_FRAMES:
                print(f"[PHASE] End wait done ({END_FRAMES} frames). Closing simulation.")
                break
            continue

    # ── Cleanup ───────────────────────────────────────────────
    def_file.flush(); def_file.close()
    tac_file.flush(); tac_file.close()
    print("[CSV] Files closed.")
    simulation_app.close()