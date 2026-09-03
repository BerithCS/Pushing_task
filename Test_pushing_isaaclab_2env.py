#!/usr/bin/env python3
"""
TWO-ENVIRONMENT version of Test_pushing_isaaclab.py  (Isaac Sim 5.1 / Isaac Lab 2.3.2)

Ported from the validated Isaac Lab 3.0 two-env script (Sep 2026). Only three
things differ from that file: PhysxCfg comes from isaaclab.sim, SimulationCfg
takes it as `physx=`, and add_reference_to_stage comes from
isaacsim.core.utils.stage -- exactly as in the single-env 5.1 script.
Everything else (scene cloning, batched articulation, wildcard views, per-env
phase machines, per-env CSVs) is identical.

  env_0 : push 20 cm   (/World/env_0, origin x=0.0)
  env_1 : push 30 cm   (/World/env_1, origin x=+2.0)

Design (validated by smoke_test_two_envs.py, full PASS on this install):
  * The SAME doosan_station_full.usd is referenced twice (/World/env_0,
    /World/env_1). USD composition gives two independent prim subtrees; the
    USD's own physicsScene lives at the absolute root /physicsScene so it is
    NOT duplicated.
  * ONE batched Articulation over regex /World/env_.*/m0609/m0609 (2 instances).
  * ONE wildcard volume-deformable view over /World/env_*/.../Sponge (count=2),
    sliced per env. Body-row -> env mapping is verified from world X positions,
    never assumed.
  * ONE cuRobo MotionGen. Both stations have an identical layout, so the
    collision world is extracted ONCE from env_0 in env-local frame and every
    env's goals are expressed in its own robot-base frame (identical math to
    the single-env script; env offsets cancel).
  * PER-ENV phase machines. env_0 (20 cm) reaches Target B before env_1
    (30 cm), so settle/baseline/plan/push/end state is per env; the loop exits
    when both envs are done.
  * PER-ENV CSVs: sponge_data_env0_20cm_* / sponge_data_env1_30cm_*.

All PORT NOTEs from the single-env script still apply; the validated logic
(Kabsch per-frame fit, settle gating, fail-fast degraded checks) is unchanged
-- just instantiated per env.
"""

# ==============================================================================
# 1) APP LAUNCH  — must come first, before any omni/isaacsim/pxr/cuRobo import.
# ==============================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Doosan M0609 tactile pushing, 2 envs (Isaac Sim 5.1 / Isaac Lab 2.3.2).")
parser.add_argument("--hold", action="store_true",
                    help="Keep the viewer open (physics paused-in-place) after both envs finish.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if getattr(args_cli, "width", None) is None:
    args_cli.width = 1920
if getattr(args_cli, "height", None) is None:
    args_cli.height = 1080

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print(f"[APP] headless={getattr(args_cli, 'headless', None)}  "
      f"livestream={getattr(args_cli, 'livestream', None)}  "
      f"hold={getattr(args_cli, 'hold', None)}")

# ==============================================================================
# 2) EVERYTHING ELSE IMPORTS ONLY AFTER THE APP EXISTS
# ==============================================================================
import os
import csv
import datetime
import numpy as np
import pandas as pd
import onnxruntime as ort
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import torch

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema, Gf, Sdf

import warp as wp

# PORT NOTE (Sim 5.1 / Lab 2.3.2): PhysxCfg lives in isaaclab.sim here, and
# SimulationCfg takes it as `physx=` (Lab 3.0 moved it to isaaclab_physx and
# renamed the field to `physics=`).
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation, ArticulationCfg

# PORT NOTE (Sim 5.1): same helper the single-env 5.1 script uses.
from isaacsim.core.utils.stage import add_reference_to_stage

# cuRobo (unchanged)
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
EE_LINK_NAME    = "link_6"

INITIAL_JOINT_DEGREES = np.array(
    [-72.09, 49.03, 57.46, -0.08, 73.51, -73.04],
    dtype=np.float32,
)

# ── MULTI-ENV ────────────────────────────────────────────────
NUM_ENVS    = 2
ENV_SPACING = 2.0                                    # metres, +X between envs
ENV_ORIGINS = [np.array([i * ENV_SPACING, 0.0, 0.0], dtype=np.float64)
               for i in range(NUM_ENVS)]
ENV_ROOTS   = [f"/World/env_{i}" for i in range(NUM_ENVS)]

# PER-ENV push distance:   env_0 -> 20 cm,  env_1 -> 30 cm
DISTANCE_TARGETS = [0.20, 0.30]

ROBOT_PRIM_REGEX = "/World/env_.*/m0609/m0609"

def env_robot_path(i):  return f"{ENV_ROOTS[i]}/m0609/m0609"
def env_sponge_path(i): return f"{ENV_ROOTS[i]}/CoRo_tactile/CoRo_tactile/Sponge"
def env_case_path(i):   return f"{ENV_ROOTS[i]}/CoRo_tactile/CoRo_tactile/Case_m"
def env_object_xform(i): return f"{ENV_ROOTS[i]}/Object_pushed"
def env_object_mesh(i):  return f"{ENV_ROOTS[i]}/Object_pushed/Cylinder"

SPONGE_WILDCARD = "/World/env_*/CoRo_tactile/CoRo_tactile/Sponge"
CASE_WILDCARD   = "/World/env_*/CoRo_tactile/CoRo_tactile/Case_m"

# ============================================================
# CNN / INFERENCE SETTINGS (unchanged)
# ============================================================
NODES_FILE      = str(SCRIPT_DIR / "CNN_tactile" / "Nodes_id_filtered.csv")
ONNX_MODEL_PATH = str(SCRIPT_DIR / "CNN_tactile" / "best.onnx")
X_TRAIN_MAX_NPY = str(SCRIPT_DIR / "CNN_tactile" / "CNN_max.npy")

N_ROWS = 18
N_COLS = 12

# ============================================================
# OBJECT PUSHED (per-env world position = local + env origin)
# ============================================================
OBJECT_PUSHED_BOTTOM_CENTER_LOCAL = np.array([0.22, -0.26, 0.96557], dtype=np.float64)
OBJECT_HEIGHT         = 0.10
OBJECT_DIAMETER       = 0.075
OBJECT_RADIUS         = OBJECT_DIAMETER / 2.0
OBJECT_MASS_KG        = 0.9
OBJECT_COLOR          = np.array([1.0, 0.0, 0.0])
OBJECT_CONTACT_OFFSET = 0.0001
OBJECT_REST_OFFSET    = 0.0

CYLINDER_SOLVER_POS_ITERS = 240
CYLINDER_SOLVER_VEL_ITERS = 30

# ============================================================
# TARGET COMPUTATION (per env)
#   A is the same LOCAL point in every env; B = A + distance in +Y.
#   World coordinates add the env origin (offset in X only).
# ============================================================
_cyl_center_x = float(OBJECT_PUSHED_BOTTOM_CENTER_LOCAL[0])
_cyl_center_y = float(OBJECT_PUSHED_BOTTOM_CENTER_LOCAL[1])
_cyl_center_z = float(OBJECT_PUSHED_BOTTOM_CENTER_LOCAL[2]) + OBJECT_HEIGHT / 2.0

_target_y = _cyl_center_y - OBJECT_RADIUS - 0.012 - 0.012 - 0.001
_target_z = _cyl_center_z + 0.087715

TARGET_A_LOCAL = np.array([_cyl_center_x, _target_y, _target_z], dtype=np.float32)

TARGET_A_WORLD = [TARGET_A_LOCAL + ENV_ORIGINS[i].astype(np.float32)
                  for i in range(NUM_ENVS)]
TARGET_B_WORLD = [TARGET_A_WORLD[i] + np.array([0.0, DISTANCE_TARGETS[i], 0.0],
                                               dtype=np.float32)
                  for i in range(NUM_ENVS)]

# ============================================================
# CSV
# ============================================================
CSV_DIR      = PROJECT_DIR
CSV_BASENAME = "sponge_data"

# ============================================================
# TIMING (unchanged)
# ============================================================
PHYSICS_DT         = 1.0 / 60.0
INTERPOLATION_DT   = 0.008
STEPS_PER_CMD      = 1
TIME_DILATION      = 0.5
MAX_EFFORT         = 1500.0
KP_GAINS           = 200000.0
KD_GAINS           = 10000.0

SETTLE_FRAMES      = 10
MAX_SETTLE_FRAMES  = 900
BASELINE_FRAMES    = 10
END_FRAMES         = 10

# ============================================================
# PLANNING (unchanged values)
# ============================================================
MAX_PLAN_FAILS  = 5
PUSH_Y_WEIGHT   = [1, 1, 1, 1, 0, 1]
HOLD_ORI_WEIGHT = [1, 1, 1, 0, 0, 0]

# PORT NOTE (2-env): obstacles are extracted from env_0 ONLY, in env-local
# frame (reference_prim_path=/World/env_0). Both stations are identical, so
# one collision world serves both robots -- each plans in its OWN base frame
# and the env offset cancels. Substrings are suffix-style so they match the
# env-prefixed paths.
OBSTACLE_IGNORE = [
    "/m0609", "/looks",
    "/target_A", "/target_B",
    "/Object_pushed", "/GroundPlane",
    "/defaultGroundPlane", "/SphereLight",
    "/physicsScene", "/robot_mount",
    "/Environment/Geometry",
    "/World/env_1",          # belt & suspenders if only_paths is unavailable
]

# ============================================================
# CNN HELPERS (unchanged)
# ============================================================

def load_ordered_node_ids(path):
    df  = pd.read_csv(path)
    col = "node_id" if "node_id" in df.columns else df.columns[0]
    seen, ids = set(), []
    for nid in df[col].dropna().astype(int):
        if nid not in seen:
            seen.add(nid); ids.append(nid)
    return ids


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


def kabsch_fit(rest_pts, cur_pts):
    """Return (R_mat, T) such that R_mat @ rest + T best matches cur.
    (Per-frame direct fit -- see the single-env script for the full Sim 6.0
    diagnosis of why this replaces the Case_m-pose reconstruction.)"""
    A = rest_pts - rest_pts.mean(axis=0)
    B = cur_pts - cur_pts.mean(axis=0)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R_mat = Vt.T @ D @ U.T
    T = cur_pts.mean(axis=0) - R_mat @ rest_pts.mean(axis=0)
    return R_mat, T


def get_dz_from_nodes(cur, rest_ref, ordered_ids):
    """dz from an already-sliced (n,3) nodal array via per-frame Kabsch."""
    try:
        R_mat, T = kabsch_fit(rest_ref, cur)
        n = cur.shape[0]
        return compute_dz_from_arrays(rest_ref, cur, R_mat, T,
                                      ordered_ids, list(range(n)))
    except Exception:
        return None


def infer_tactile(dz, baseline, x_train_max, session, input_name):
    dz_corrected = dz - baseline
    dz_grid      = dz_corrected.reshape(N_ROWS, N_COLS).astype(np.float32)
    X            = np.clip(dz_grid / x_train_max, -1.0, 1.0)
    X            = X.reshape(1, N_ROWS, N_COLS, 1).astype(np.float32)
    y_pred       = session.run(None, {input_name: X})[0]
    return y_pred[0].flatten()  # (28,)


# ============================================================
# DEFORMABLE MESH HELPERS (unchanged)
# ============================================================

def _has_any_deformable_api(prim: Usd.Prim) -> bool:
    for s in prim.GetAppliedSchemas():
        sl = s.lower()
        if "deformable" in sl or "softbody" in sl or "soft_body" in sl:
            return True
    for _api_name in ("PhysxDeformableBodyAPI",
                      "PhysxDeformableSurfaceAPI",
                      "PhysxAutoDeformableAPI",
                      "PhysxDeformableVolumeAPI"):
        _api = getattr(PhysxSchema, _api_name, None)
        if _api is not None:
            try:
                if prim.HasAPI(_api):
                    return True
            except Exception:
                pass
    return False


def find_deformable_simulation_mesh(root_path: str, verbose=True):
    stage = omni.usd.get_context().get_stage()
    root  = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        print(f"[DEFORMABLE] WARNING: root prim not found: {root_path}")
        return None
    if verbose:
        print(f"[DEFORMABLE] scanning under {root_path}:")
    _cands = []
    for prim in Usd.PrimRange(root):
        _schemas = list(prim.GetAppliedSchemas())
        if _schemas and verbose:
            print(f"    {prim.GetPath()}  type={prim.GetTypeName()}  schemas={_schemas}")
        if _has_any_deformable_api(prim):
            for child in prim.GetChildren():
                _cands.append(child)
                if "simulation_mesh" in child.GetName().lower():
                    print(f"[DEFORMABLE] Found: {child.GetPath()}")
                    return child
    print(f"[DEFORMABLE] WARNING: simulation_mesh not found under {root_path}")
    if _cands:
        print(f"[DEFORMABLE]   children of deformable prims seen: "
              f"{[c.GetName() for c in _cands]}")
    return None


# ============================================================
# CSV HELPERS (per env)
# ============================================================

def prepare_csv(csv_dir: str, basename: str, env_i: int, distance_m: float):
    """Per-env CSV pair, distance encoded in the filename."""
    os.makedirs(csv_dir, exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"env{env_i}_{int(round(distance_m * 100))}cm"

    def_path   = os.path.join(csv_dir, f"{basename}_{tag}_deformation_{ts}.csv")
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
    print(f"[CSV] env_{env_i} deformation file : {def_path}")

    tactile_cols = [f"T_{i+1:02d}" for i in range(28)]
    tac_path   = os.path.join(csv_dir, f"{basename}_{tag}_tactiledata_{ts}.csv")
    tac_file   = open(tac_path, "w", newline="")
    tac_writer = csv.writer(tac_file)
    tac_writer.writerow(["frame", "t"] + tactile_cols)
    tac_file.flush()
    print(f"[CSV] env_{env_i} tactile file     : {tac_path}")

    return def_file, def_writer, tac_file, tac_writer


# ============================================================
# SCENE-BUILD HELPERS (env-parameterized)
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
        print(f"  Physics mat -> {prim_path}  (s={static_friction}, d={dynamic_friction})")
    else:
        print(f"  WARNING: prim not found: {prim_path}")


def create_object_pushed(stage, env_i: int):
    """Cylinder to push, per env.

    BUG FIX (run 3): the prim lives UNDER /World/env_i, whose Xform already
    carries the env origin. The translate op is in PARENT space, so it must be
    the LOCAL position only — adding ENV_ORIGINS here applied the offset twice
    and put env_1's cylinder 2 m off its table (it fell to the floor and the
    sensor pushed empty air). env_0 was unaffected because its origin is 0.
    """
    bottom_local = OBJECT_PUSHED_BOTTOM_CENTER_LOCAL
    bottom_world = OBJECT_PUSHED_BOTTOM_CENTER_LOCAL + ENV_ORIGINS[env_i]  # print only
    xform_path = env_object_xform(env_i)
    mesh_path  = env_object_mesh(env_i)

    xform_prim = UsdGeom.Xform.Define(stage, xform_path)
    xf = UsdGeom.Xformable(xform_prim.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*bottom_local.tolist()))

    xform_p  = stage.GetPrimAtPath(xform_path)
    UsdPhysics.RigidBodyAPI.Apply(xform_p)
    mass_api = UsdPhysics.MassAPI.Apply(xform_p)
    mass_api.GetMassAttr().Set(float(OBJECT_MASS_KG))
    mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, float(OBJECT_HEIGHT / 2.0)))

    cylinder = UsdGeom.Cylinder.Define(stage, mesh_path)
    cylinder.GetHeightAttr().Set(float(OBJECT_HEIGHT))
    cylinder.GetRadiusAttr().Set(float(OBJECT_RADIUS))
    cylinder.GetAxisAttr().Set("Z")
    cyl_xform = UsdGeom.Xformable(cylinder.GetPrim())
    cyl_xform.ClearXformOpOrder()
    cyl_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(OBJECT_HEIGHT / 2.0)))

    cyl_prim = stage.GetPrimAtPath(mesh_path)
    UsdPhysics.CollisionAPI.Apply(cyl_prim)
    physx_col = PhysxSchema.PhysxCollisionAPI.Apply(cyl_prim)
    physx_col.GetContactOffsetAttr().Set(float(OBJECT_CONTACT_OFFSET))
    physx_col.GetRestOffsetAttr().Set(float(OBJECT_REST_OFFSET))

    mat_path    = f"/World/looks/ObjectPushedMat_env{env_i}"
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
    print(f"Object_pushed (env_{env_i}) created at world {np.round(bottom_world, 5).tolist()} (local translate, parent carries origin)")


def apply_cylinder_solver_iters(stage, env_i: int):
    xform_p = stage.GetPrimAtPath(env_object_xform(env_i))
    if not xform_p.IsValid():
        print(f"  WARNING: Cylinder Xform not found (env_{env_i}).")
        return
    for name, val in [
        ("physxRigidBody:solverPositionIterationCount", CYLINDER_SOLVER_POS_ITERS),
        ("physxRigidBody:solverVelocityIterationCount", CYLINDER_SOLVER_VEL_ITERS),
    ]:
        attr = xform_p.GetAttribute(name)
        if attr and attr.IsValid(): attr.Set(val)
        else: xform_p.CreateAttribute(name, Sdf.ValueTypeNames.UInt).Set(val)
        print(f"  Cylinder env_{env_i} {name} = {val}")


def create_visual_marker(stage, path, position, size=0.04, color=(0.1, 0.1, 0.1)):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(p) for p in position]))
    xf.AddScaleOp().Set(Gf.Vec3f(size, size, size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cube.GetPrim()


def set_marker_color(prim, color):
    if prim and prim.IsValid():
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in color])])


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


# ==============================================================================
# ISAAC LAB ARTICULATION ADAPTERS — batched over NUM_ENVS
# PORT NOTE (2-env): buffers carry a leading env dimension (now 2, was 1).
# Reads take an env index; the WRITE path is single-shot: each frame every
# env's phase logic updates its row of a (NUM_ENVS, dof) target matrix and ONE
# set_joint_position_target call pushes all rows together.
# ==============================================================================

def as_torch(x):
    if isinstance(x, wp.array):
        return wp.to_torch(x)
    return x


def build_dof_index_map(robot: Articulation):
    names = robot.joint_names
    return {n: i for i, n in enumerate(names)}


def art_get_dof_indices(dof_map, joint_names):
    return [dof_map[n] for n in joint_names]


def art_set_joint_positions_all(robot: Articulation, positions_1d, col_indices, device):
    """Hard-set the SAME joint positions on every env (teleport)."""
    q = as_torch(robot.data.joint_pos).clone()            # (NUM_ENVS, num_dof)
    pos = torch.as_tensor(positions_1d, dtype=torch.float32, device=_torch_dev_of(q))
    q[:, col_indices] = pos
    robot.write_joint_state_to_sim(position=q, velocity=torch.zeros_like(q))


def art_get_joint_state(robot: Articulation, env_i: int, col_indices):
    """(positions, velocities) of ONE env as 1-D numpy arrays in col order."""
    qp = as_torch(robot.data.joint_pos)[env_i, col_indices].detach().cpu().numpy()
    qv = as_torch(robot.data.joint_vel)[env_i, col_indices].detach().cpu().numpy()
    return qp, qv


def _torch_dev_of(t):
    """PORT NOTE (Lab 3.0): `.device` on Lab buffers can be a WARP Device, which
    torch.as_tensor rejects. str() of either device type is 'cuda:0'-style, so
    round-trip through torch.device(str(...))."""
    d = getattr(t, "device", None)
    try:
        return torch.device(str(d))
    except Exception:
        return torch.device("cuda:0")


def art_apply_targets_all(robot: Articulation, targets_2d, col_indices=None, device=None):
    """Send PD position targets for ALL envs at once.
    targets_2d: (NUM_ENVS, n_cols). With col_indices=None the columns must
    already be in robot.joint_names order (full dof) -- ordering-safe path."""
    tgt = torch.as_tensor(np.asarray(targets_2d), dtype=torch.float32,
                          device=_torch_dev_of(as_torch(robot.data.joint_pos)))
    if col_indices is None:
        robot.set_joint_position_target(tgt)
    else:
        robot.set_joint_position_target(tgt, joint_ids=list(col_indices))
    robot.write_data_to_sim()


def get_prim_world_pose(stage, prim_path):
    """World pose (pos_xyz, quat_wxyz) via USD xform cache (see single-env
    PORT NOTE on quaternion conventions -- unchanged)."""
    prim = stage.GetPrimAtPath(prim_path)
    xcache = UsdGeom.XformCache()
    m = xcache.GetLocalToWorldTransform(prim)
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    imag = q.GetImaginary()
    pos  = np.array([t[0], t[1], t[2]], dtype=np.float32)
    quat = np.array([q.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)
    return pos, quat


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Scene USD      : {SCENE_USD}")
    for i in range(NUM_ENVS):
        print(f"env_{i}: origin={ENV_ORIGINS[i].tolist()}  "
              f"push={DISTANCE_TARGETS[i]*100:.0f}cm  "
              f"A={TARGET_A_WORLD[i].round(5).tolist()}  "
              f"B={TARGET_B_WORLD[i].round(5).tolist()}")

    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"

    # ── Load CNN resources (shared: one session serves both envs) ──────────
    print("Loading CNN resources...")
    ordered_ids = load_ordered_node_ids(NODES_FILE)
    x_train_max = float(np.load(X_TRAIN_MAX_NPY).item())
    session     = ort.InferenceSession(ONNX_MODEL_PATH,
                                       providers=["CUDAExecutionProvider",
                                                  "CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    print(f"  Nodes       : {len(ordered_ids)}")
    print(f"  X_train_max : {x_train_max:.6f}")
    print(f"  ONNX input  : {input_name}")

    # ── SimulationContext (PhysX pinned — deformables are PhysX-only) ──────
    _physx_kwargs = dict(
        enable_ccd=False,
        gpu_max_soft_body_contacts=2 ** 20,
    )
    try:
        _physx_cfg = PhysxCfg(**_physx_kwargs)
    except TypeError as e:
        print(f"[BACKEND] WARNING: PhysxCfg rejected a kwarg ({e}).\n"
              f"  Falling back to PhysxCfg() defaults. If the sponge misbehaves or\n"
              f"  PhysX reports a contact-buffer overflow, check the current field\n"
              f"  names in isaaclab.sim.PhysxCfg.")
        _physx_cfg = PhysxCfg()

    sim_cfg = SimulationCfg(dt=PHYSICS_DT, device=device, physx=_physx_cfg)
    print(f"[BACKEND] SimulationCfg.physics = {type(_physx_cfg).__name__} "
          f"(PhysX required for deformables)")

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    stage = omni.usd.get_context().get_stage()

    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # ── Spawn the two envs (validated by the smoke test) ───────────────────
    for i, env_root in enumerate(ENV_ROOTS):
        xf = UsdGeom.Xform.Define(stage, env_root)
        xf.AddTranslateOp().Set(Gf.Vec3d(*ENV_ORIGINS[i].tolist()))
        add_reference_to_stage(usd_path=SCENE_USD, prim_path=env_root)
        print(f"[SCENE] referenced {Path(SCENE_USD).name} -> {env_root} "
              f"(origin {ENV_ORIGINS[i].tolist()})")
    simulation_app.update()

    # ── Physics scene: smoke test showed ONE scene at /physicsScene (the USD
    # authors it at the absolute root, outside /World, so referencing twice
    # does NOT duplicate it). Scan anyway and deactivate any per-env copies —
    # a future USD edit must not silently create competing scenes.
    scene_prims = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    print(f"[PHYS-SCENE] UsdPhysics.Scene prims on stage: "
          f"{[str(p.GetPath()) for p in scene_prims]}")
    kept = None
    for p in scene_prims:
        path = str(p.GetPath())
        if any(path.startswith(r) for r in ENV_ROOTS):
            p.SetActive(False)
            print(f"[PHYS-SCENE] deactivated per-env scene: {path}")
        elif kept is None:
            kept = p
    if kept is not None:
        physics_scene_api = PhysxSchema.PhysxSceneAPI.Apply(kept)
        physics_scene_api.GetEnableGPUDynamicsAttr().Set(True)
        physics_scene_api.GetBroadphaseTypeAttr().Set("GPU")
        print(f"[PHYS-SCENE] GPU dynamics enabled on {kept.GetPath()}")
    else:
        print("[PHYS-SCENE] WARNING: no global physics scene prim found.")

    # ── Per-env scene fixes + deformable tuning (same values as single-env) ─
    def _set_attr(prim, name, value, type_name):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid(): attr.Set(value)
        else: prim.CreateAttribute(name, type_name).Set(value)
        print(f"  [Deformable] {name} = {value}")

    for i, env_root in enumerate(ENV_ROOTS):
        _tbc = stage.GetPrimAtPath(f"{env_root}/table_base/table_base/collisions")
        if _tbc.IsValid():
            _tbc.SetActive(False)
            print(f"  table_base collisions deactivated (env_{i}).")

        _sponge_prim = stage.GetPrimAtPath(env_sponge_path(i))
        _mat_prim    = stage.GetPrimAtPath(env_sponge_path(i) + "/Looks/Deformable_Material")

        if _sponge_prim.IsValid():
            _set_attr(_sponge_prim, "omniphysics:mass",                             0.09,    Sdf.ValueTypeNames.Float)
            _set_attr(_sponge_prim, "physxDeformable:solverPositionIterationCount", 80,      Sdf.ValueTypeNames.UInt)
            _set_attr(_sponge_prim, "physxDeformable:sleepThreshold",               0.00001, Sdf.ValueTypeNames.Float)
            _set_attr(_sponge_prim, "physxDeformable:disableGravity",               True,    Sdf.ValueTypeNames.Bool)
        else:
            print(f"  WARNING: Sponge prim not found (env_{i}).")

        if _mat_prim.IsValid():
            _set_attr(_mat_prim, "omniphysics:youngsModulus",                 1289000.0, Sdf.ValueTypeNames.Float)
            _set_attr(_mat_prim, "omniphysics:poissonsRatio",                 0.1729,    Sdf.ValueTypeNames.Float)
            _set_attr(_mat_prim, "omniphysics:density",                       1240.0,    Sdf.ValueTypeNames.Float)
            _set_attr(_mat_prim, "physxDeformableMaterial:elasticityDamping", 15.0,      Sdf.ValueTypeNames.Float)
        else:
            print(f"  WARNING: Deformable_Material prim not found (env_{i}).")

        # ── Object pushed + solver, per env ─────────────────
        create_object_pushed(stage, i)
        apply_cylinder_solver_iters(stage, i)

        # ── Physics materials, per env ──────────────────────
        apply_physics_material(stage, env_object_mesh(i),                        0.2, 0.15)
        apply_physics_material(stage, f"{env_root}/table_cover/Cube",            0.9,  0.9)
        apply_physics_material(stage, f"{env_root}/m0609/m0609/link_6/adapter",  0.4,  0.35)
        apply_physics_material(stage, env_sponge_path(i) + "/collision_mesh",    0.9,  0.8)

    simulation_app.update()

    # ── ONE batched articulation over both robots ──────────────────────────
    from isaaclab.actuators import ImplicitActuatorCfg
    robot_cfg_lab = ArticulationCfg(
        prim_path=ROBOT_PRIM_REGEX,
        spawn=None,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=MAX_EFFORT,
                stiffness=KP_GAINS,
                damping=KD_GAINS,
            ),
        },
    )
    robot = Articulation(robot_cfg_lab)

    # ── Cosmetic target markers, per env ───────────────────────────────────
    target_a_prims = [create_visual_marker(stage, f"/World/target_A_env{i}", TARGET_A_WORLD[i])
                      for i in range(NUM_ENVS)]
    target_b_prims = [create_visual_marker(stage, f"/World/target_B_env{i}", TARGET_B_WORLD[i])
                      for i in range(NUM_ENVS)]

    # ── cuRobo setup — ONE MotionGen for both (identical stations) ─────────
    usd_helper = UsdHelper()
    usd_helper.load_stage(stage)
    # Extract obstacles from env_0 only, in env_0-LOCAL frame. Both robots plan
    # against this world in their OWN base frame; layouts are identical so the
    # env offset cancels exactly.
    try:
        world_cfg = usd_helper.get_obstacles_from_stage(
            reference_prim_path=ENV_ROOTS[0],
            only_paths=[ENV_ROOTS[0]],
            ignore_substring=OBSTACLE_IGNORE,
        )
    except TypeError:
        # older cuRobo without only_paths: rely on ignore_substring (which
        # includes /World/env_1) to keep the second station out.
        world_cfg = usd_helper.get_obstacles_from_stage(
            reference_prim_path=ENV_ROOTS[0],
            ignore_substring=OBSTACLE_IGNORE,
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

    sim.reset()
    robot.reset()

    n_found = as_torch(robot.data.joint_pos).shape[0]
    print(f"[ROBOT] articulation instances: {n_found} (expected {NUM_ENVS})")
    if n_found != NUM_ENVS:
        print("[ABORT] robot batching failed."); simulation_app.close(); raise SystemExit(1)

    dof_map  = build_dof_index_map(robot)
    idx_cu   = art_get_dof_indices(dof_map, j_names)
    print(f"[JOINTS] Articulation joint order : {robot.joint_names}")
    print(f"[JOINTS] cuRobo joint order        : {j_names}")
    print(f"[JOINTS] cuRobo->column indices    : {idx_cu}")

    plan_config = MotionGenPlanConfig(
        enable_graph=False, enable_graph_attempt=4,
        max_attempts=4, enable_finetune_trajopt=True,
        time_dilation_factor=TIME_DILATION, pose_cost_metric=None,
    )

    # ── JOINT INIT — write, settle, verify (all envs at once) ─────────────
    art_set_joint_positions_all(robot, init_positions, idx_cu, device)
    robot.write_data_to_sim()
    _hold_all = np.tile(np.asarray(init_positions, dtype=np.float32), (NUM_ENVS, 1))
    for _ in range(SETTLE_FRAMES):
        art_apply_targets_all(robot, _hold_all, idx_cu, device)
        sim.step(render=not args_cli.headless)
        robot.update(PHYSICS_DT)
    for i in range(NUM_ENVS):
        qp_check, qv_check = art_get_joint_state(robot, i, idx_cu)
        print(f"[DIAG] env_{i} init target (rad): {np.round(init_positions, 4)}")
        print(f"[DIAG] env_{i} joint pos  (rad) : {np.round(qp_check, 4)}")
        print(f"[DIAG] env_{i} max |pos err|    : "
              f"{np.max(np.abs(qp_check - np.array(init_positions))):.5f}")

    # ── Deformable tensor views ───────────────────────────────────────────
    SimulationManager = None
    _sm_err = []
    for _mod in ("isaacsim.core.simulation_manager",
                 "isaacsim.core.experimental.simulation_manager"):
        try:
            SimulationManager = __import__(_mod, fromlist=["SimulationManager"]).SimulationManager
            print(f"[DEFORMABLE] SimulationManager from {_mod}")
            break
        except (ImportError, AttributeError) as e:
            _sm_err.append(f"{_mod}: {e}")
    if SimulationManager is None:
        raise ImportError(
            "Could not locate SimulationManager -- needed for the deformable "
            "tensor view. Tried:\n  " + "\n  ".join(_sm_err))

    def _to_np(x):
        if isinstance(x, wp.array):
            return x.numpy()
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    deform_view = None
    _pv = None
    try:
        _pv = SimulationManager.get_physics_sim_view()
        _backend = getattr(SimulationManager, "get_physics_backend", None)
        if callable(_backend):
            try:
                print(f"[BACKEND] active physics engine = {_backend()}")
            except Exception:
                pass

        _mk = getattr(_pv, "create_volume_deformable_body_view", None)
        if _mk is None:
            for _alt in ("create_deformable_volume_view", "create_soft_body_view"):
                _mk = getattr(_pv, _alt, None)
                if _mk is not None:
                    print(f"[DEFORMABLE] using fallback factory '{_alt}'")
                    break

        if _mk is not None:
            # ONE wildcard view over both sponges (smoke test: count=2)
            deform_view = _mk(SPONGE_WILDCARD)
            cnt = getattr(deform_view, "count", None) if deform_view is not None else None
            print(f"[DEFORMABLE] wildcard volume view count={cnt} (want {NUM_ENVS})")
            if not cnt or cnt != NUM_ENVS:
                deform_view = None
        else:
            print("[DEFORMABLE] ERROR: no deformable view factory (Newton active?).")
    except Exception as e:
        print(f"[DEFORMABLE] ERROR creating volume view: {e}")

    # ── Rest points per env (per-prim USD reads; smoke test: identical) ────
    rest_refs = [None] * NUM_ENVS
    for i in range(NUM_ENVS):
        _sm = find_deformable_simulation_mesh(env_sponge_path(i), verbose=(i == 0))
        if _sm is not None:
            _ra = _sm.GetAttribute("omniphysics:restShapePoints")
            _rv = _ra.Get() if (_ra and _ra.IsValid()) else None
            if _rv is not None:
                rest_refs[i] = np.array([[float(p[0]), float(p[1]), float(p[2])]
                                         for p in _rv], dtype=float)
        if rest_refs[i] is None:
            print(f"[DEFORMABLE] WARNING: restShapePoints unavailable (env_{i}).")
        else:
            print(f"[DEFORMABLE] env_{i} nodes={rest_refs[i].shape[0]}  "
                  f"first={rest_refs[i][0].round(5).tolist()}")

    N_NODES = rest_refs[0].shape[0] if rest_refs[0] is not None else 0

    # ── Map wildcard-view body rows -> envs (NEVER assume ordering) ────────
    # Read the live nodes once and assign each body block to the env whose
    # origin X its mean X is closest to. Works for both (N_env, nodes, 3) and
    # concatenated (N_env*nodes, 3) layouts.
    deform_row_of_env = list(range(NUM_ENVS))

    def _split_bodies(raw):
        """raw -> list of (nodes,3) blocks, one per body, view-row order."""
        a = _to_np(raw)
        if a.ndim == 3:                          # (N_env, nodes, 3)
            return [a[k] for k in range(a.shape[0])]
        a = a.reshape(-1, 3)                     # concatenated
        n = a.shape[0] // NUM_ENVS
        return [a[k * n:(k + 1) * n] for k in range(NUM_ENVS)]

    if deform_view is not None:
        try:
            _blocks = _split_bodies(deform_view.get_simulation_nodal_positions())
            for k, b in enumerate(_blocks):
                mx = float(b[:, 0].mean())
                env = int(np.argmin([abs(mx - o[0]) for o in ENV_ORIGINS]))
                deform_row_of_env[env] = k
                print(f"[DEFORMABLE] view row {k}: mean_x={mx:.3f} -> env_{env}")
            if len(set(deform_row_of_env)) != NUM_ENVS:
                print("[DEFORMABLE] ERROR: body->env mapping is not a bijection.")
                deform_view = None
        except Exception as e:
            print(f"[DEFORMABLE] ERROR mapping bodies to envs: {e}")
            deform_view = None

    def read_env_nodes(env_i):
        """Live (nodes,3) WORLD-frame positions for env_i."""
        blocks = _split_bodies(deform_view.get_simulation_nodal_positions())
        return blocks[deform_row_of_env[env_i]]

    def read_env_nodevels(env_i):
        blocks = _split_bodies(deform_view.get_simulation_nodal_velocities())
        return blocks[deform_row_of_env[env_i]]

    # ── Case_m rigid-body view (wildcard, mapped like the sponge) ──────────
    case_view = None
    case_row_of_env = list(range(NUM_ENVS))
    try:
        _mkr = getattr(_pv, "create_rigid_body_view", None)
        if _mkr is not None:
            case_view = _mkr(CASE_WILDCARD)
            ccnt = getattr(case_view, "count", None) if case_view is not None else None
            print(f"[CASE] rigid-body view count={ccnt} (want {NUM_ENVS})")
            if not ccnt or ccnt != NUM_ENVS:
                case_view = None
            else:
                _T = _to_np(case_view.get_transforms()).reshape(NUM_ENVS, -1)
                for k in range(NUM_ENVS):
                    mx = float(_T[k, 0])
                    env = int(np.argmin([abs(mx - o[0]) for o in ENV_ORIGINS]))
                    case_row_of_env[env] = k
                    print(f"[CASE] view row {k}: x={mx:.3f} -> env_{env}  "
                          f"pose xyz={_T[k, :3].round(5).tolist()}")
                if len(set(case_row_of_env)) != NUM_ENVS:
                    print("[CASE] ERROR: case->env mapping is not a bijection.")
                    case_view = None
        else:
            print("[CASE] ERROR: create_rigid_body_view missing.")
    except Exception as e:
        print(f"[CASE] ERROR creating rigid-body view: {e}")

    # ── Per-env parent pose (static; differs by the env origin) ────────────
    parent_pos_l     = [np.zeros(3)] * NUM_ENVS
    parent_rot_inv_l = [R.identity()] * NUM_ENVS
    for i in range(NUM_ENVS):
        _parent_path = env_case_path(i).rsplit("/", 1)[0]
        _parent_prim = stage.GetPrimAtPath(_parent_path)
        if _parent_prim and _parent_prim.IsValid():
            _m = UsdGeom.XformCache().GetLocalToWorldTransform(_parent_prim)
            _t = _m.ExtractTranslation()
            _q = _m.ExtractRotationQuat()
            _im = _q.GetImaginary()
            parent_pos_l[i] = np.array([float(_t[0]), float(_t[1]), float(_t[2])])
            _prot = R.from_quat([float(_im[0]), float(_im[1]), float(_im[2]),
                                 float(_q.GetReal())])
            parent_rot_inv_l[i] = _prot.inv()
            print(f"[CASE] env_{i} parent world pos={parent_pos_l[i].round(5).tolist()}")
        else:
            print(f"[CASE] WARNING: parent prim not found (env_{i}); "
                  f"pose will be written in WORLD frame.")

    case_q_corr = R.identity()   # kept for CSV-column compatibility; unused

    # ── DIAGNOSTIC: cylinder rigid-body view (both envs) ───────────────────
    # env_1 showed a full 30cm sweep with ZERO pad deformation, i.e. the
    # sensor never met its cylinder. Without a GUI we track the cylinders from
    # the physics view: world position + tilt, logged at key moments.
    cyl_view = None
    cyl_row_of_env = list(range(NUM_ENVS))
    try:
        _mkc = getattr(_pv, "create_rigid_body_view", None)
        if _mkc is not None:
            cyl_view = _mkc("/World/env_*/Object_pushed")
            ycnt = getattr(cyl_view, "count", None) if cyl_view is not None else None
            print(f"[CYL] rigid-body view count={ycnt} (want {NUM_ENVS})")
            if not ycnt or ycnt != NUM_ENVS:
                cyl_view = None
            else:
                _T = _to_np(cyl_view.get_transforms()).reshape(NUM_ENVS, -1)
                for k in range(NUM_ENVS):
                    mx = float(_T[k, 0])
                    env_k = int(np.argmin([abs(mx - o[0]) for o in ENV_ORIGINS]))
                    cyl_row_of_env[env_k] = k
                if len(set(cyl_row_of_env)) != NUM_ENVS:
                    print("[CYL] WARNING: cylinder->env mapping not a bijection; tracker off.")
                    cyl_view = None
    except Exception as e:
        print(f"[CYL] tracker unavailable: {e}")

    def log_cylinder(env_i, tag):
        """Print one cylinder's world pose + tilt from vertical, in mm/deg."""
        if cyl_view is None:
            return
        try:
            T = _to_np(cyl_view.get_transforms()).reshape(NUM_ENVS, -1)[cyl_row_of_env[env_i]]
            p = T[:3]
            q = R.from_quat([float(T[3]), float(T[4]), float(T[5]), float(T[6])])  # xyzw
            z_body = q.apply([0.0, 0.0, 1.0])
            tilt = float(np.degrees(np.arccos(np.clip(z_body[2], -1.0, 1.0))))
            spawn = OBJECT_PUSHED_BOTTOM_CENTER_LOCAL + ENV_ORIGINS[env_i]
            dxy = np.array([p[0] - spawn[0], p[1] - spawn[1]])
            print(f"[CYL] env_{env_i} {tag}: world xyz="
                  f"[{p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f}]  "
                  f"moved_xy={np.linalg.norm(dxy)*1000:7.1f} mm  "
                  f"dz={(p[2]-spawn[2])*1000:+7.1f} mm  tilt={tilt:5.1f} deg")
        except Exception as e:
            print(f"[CYL] env_{env_i} {tag}: read failed: {e}")

    # ── Kabsch validation at t=0, per env ──────────────────────────────────
    for i in range(NUM_ENVS):
        try:
            if deform_view is not None and rest_refs[i] is not None:
                cur_w0 = read_env_nodes(i)
                cur_l0 = parent_rot_inv_l[i].apply(cur_w0 - parent_pos_l[i])
                R_fit_mat, T_fit = kabsch_fit(rest_refs[i], cur_l0)
                resid = np.linalg.norm((rest_refs[i] @ R_fit_mat.T + T_fit) - cur_l0, axis=1)
                print(f"[CASE] env_{i} rest->cur alignment residual: "
                      f"mean={resid.mean()*1000:.3f}mm max={resid.max()*1000:.3f}mm")
                if resid.mean() > 0.005:
                    print(f"[CASE] WARNING (env_{i}): residual >5mm -- correspondence suspect.")
            else:
                print(f"[CASE] env_{i} validation skipped (missing view or rest points).")
        except Exception as e:
            print(f"[CASE] env_{i} Kabsch validation failed: {e}")

    for _i in range(NUM_ENVS):
        log_cylinder(_i, "initial")

    # ── FAIL FAST (same policy as single-env, checked per env) ─────────────
    _degraded = []
    if deform_view is None:
        _degraded.append("deformable wildcard view is None or count != 2")
    if case_view is None:
        _degraded.append("Case_m wildcard rigid-body view is None or count != 2")
    for i in range(NUM_ENVS):
        if rest_refs[i] is None:
            _degraded.append(f"env_{i}: rest_ref is None (restShapePoints unavailable)")
    if deform_view is not None and all(r is not None for r in rest_refs):
        try:
            for i in range(NUM_ENVS):
                _n_live = read_env_nodes(i).shape[0]
                if _n_live != rest_refs[i].shape[0]:
                    _degraded.append(
                        f"env_{i}: node-count mismatch: live {_n_live} vs "
                        f"rest {rest_refs[i].shape[0]}")
            if len(ordered_ids) != N_ROWS * N_COLS:
                _degraded.append(
                    f"ordered_ids has {len(ordered_ids)} entries but the CNN "
                    f"expects {N_ROWS}x{N_COLS}={N_ROWS*N_COLS}")
            if N_NODES and max(ordered_ids) >= N_NODES:
                _degraded.append(
                    f"ordered_ids max index {max(ordered_ids)} exceeds node "
                    f"count {N_NODES}")
        except Exception as e:
            _degraded.append(f"could not validate node counts: {e}")

    if _degraded:
        _msg = ("\n" + "=" * 70 +
                "\n[ABORT] The deformable/tactile pipeline is not viable:\n" +
                "".join(f"  * {d}\n" for d in _degraded) +
                "\nAny CSV written now would contain zero or meaningless tactile\n"
                "data. Fix the above before trusting output. Set ALLOW_DEGRADED=1\n"
                "to run anyway (robot motion only -- do NOT use the tactile CSV).\n"
                + "=" * 70)
        if os.environ.get("ALLOW_DEGRADED") != "1":
            print(_msg)
            simulation_app.close()
            raise SystemExit(1)
        print(_msg + "\n[ALLOW_DEGRADED=1] Continuing anyway.")

    # ── Per-env CSV recording ─────────────────────────────────────────────
    def save_sponge_data_env(env, frame, sim_time):
        """One deformation row per node + one tactile row, for one env."""
        i = env["i"]
        try:
            cur = read_env_nodes(i)
            vel = read_env_nodevels(i)
            cur = parent_rot_inv_l[i].apply(cur - parent_pos_l[i])
            vel = parent_rot_inv_l[i].apply(vel)
        except Exception as e:
            if frame % 60 == 0:
                print(f"[CSV] env_{i} frame={frame}: deformable read failed: {e}")
            return

        trans = (0.0, 0.0, 0.0)
        ori_w, ori_x, ori_y, ori_z = 1.0, 0.0, 0.0, 0.0
        if case_view is not None:
            try:
                T_all = _to_np(case_view.get_transforms()).reshape(NUM_ENVS, -1)
                T = T_all[case_row_of_env[i]]
                p_world = np.array([float(T[0]), float(T[1]), float(T[2])])
                q_world = R.from_quat([float(T[3]), float(T[4]), float(T[5]), float(T[6])])
                p_local = parent_rot_inv_l[i].apply(p_world - parent_pos_l[i])
                q_local = (parent_rot_inv_l[i] * q_world) * case_q_corr
                trans = (float(p_local[0]), float(p_local[1]), float(p_local[2]))
                qx, qy, qz, qw = q_local.as_quat()
                ori_w, ori_x, ori_y, ori_z = float(qw), float(qx), float(qy), float(qz)
            except Exception as e:
                if frame % 60 == 0:
                    print(f"[CSV] env_{i} frame={frame}: case pose read failed: {e}")

        rest_ref = rest_refs[i]
        n    = cur.shape[0]
        n_re = rest_ref.shape[0] if rest_ref is not None else 0

        tactile_vals = np.zeros(28, dtype=np.float32)
        _kab_resid = float("nan")
        if rest_ref is not None and env["baseline"] is not None:
            try:
                R_mat, T_vec = kabsch_fit(rest_ref, cur)
                _kab_resid = float(np.linalg.norm(
                    (rest_ref @ R_mat.T + T_vec) - cur, axis=1).mean())
                dz = compute_dz_from_arrays(rest_ref, cur, R_mat, T_vec,
                                            ordered_ids, list(range(n)))
                tactile_vals = infer_tactile(dz, env["baseline"], x_train_max,
                                             session, input_name)
            except Exception as e:
                print(f"[WARN] env_{i} inference failed at frame {frame}: {e}")

        w = env["def_writer"]
        for k in range(n):
            cx, cy, cz = float(cur[k, 0]), float(cur[k, 1]), float(cur[k, 2])
            vx, vy, vz = float(vel[k, 0]), float(vel[k, 1]), float(vel[k, 2])
            rx = ry = rz = 0.0
            if k < n_re:
                rx, ry, rz = float(rest_ref[k, 0]), float(rest_ref[k, 1]), float(rest_ref[k, 2])
            w.writerow([
                frame, f"{sim_time:.6f}", k,
                cx, cy, cz,
                vx, vy, vz,
                rx, ry, rz,
                trans[0], trans[1], trans[2],
                ori_w, ori_x, ori_y, ori_z,
            ])
        env["tac_writer"].writerow([frame, f"{sim_time:.6f}"] + tactile_vals.tolist())

        if frame % 60 == 0:
            print(f"[CSV] env_{i} frame={frame}  t={sim_time:.3f}s  nodes={n}  "
                  f"kabsch_resid={_kab_resid*1000:.3f}mm  "
                  f"tactile={tactile_vals[:4].round(3)}")
            if _kab_resid == _kab_resid and _kab_resid > 0.005:
                print(f"[CSV] WARNING env_{i}: Kabsch residual "
                      f"{_kab_resid*1000:.1f}mm is large.")
            env["def_file"].flush()
            env["tac_file"].flush()

    # ── Per-env targets in each robot's own base frame ────────────────────
    def init_targets_env(env_i):
        base_pos, base_ori           = get_prim_world_pose(stage, env_robot_path(env_i))
        curobo_ee_pos, curobo_ee_ori = get_curobo_fk(motion_gen, init_positions,
                                                     j_names, tensor_args)
        print(f"env_{env_i} robot base pos : {base_pos.tolist()}")
        print(f"env_{env_i} Target A world : {TARGET_A_WORLD[env_i].tolist()}")
        print(f"env_{env_i} Target B world : {TARGET_B_WORLD[env_i].tolist()} "
              f"({DISTANCE_TARGETS[env_i]*100:.0f}cm push)")

        goal_ori_base    = curobo_ee_ori.astype(np.float32)
        target_list_base = []
        target_ori_base  = []
        for pt_world, tgt_prim in zip([TARGET_A_WORLD[env_i], TARGET_B_WORLD[env_i]],
                                      [target_a_prims[env_i], target_b_prims[env_i]]):
            pt_base = world_to_base_frame(pt_world, base_pos, base_ori)
            target_list_base.append(pt_base)
            target_ori_base.append(goal_ori_base.copy())
            if tgt_prim and tgt_prim.IsValid():
                xf = UsdGeom.Xformable(tgt_prim)
                for op in xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(Gf.Vec3d(*[float(v) for v in pt_world]))
                        break
        return target_list_base, target_ori_base

    # ── Per-env state machines ────────────────────────────────────────────
    _num_dof = as_torch(robot.data.joint_pos).shape[1]
    _init_hold_full = np.zeros(_num_dof, dtype=np.float32)
    _init_hold_full[idx_cu] = np.asarray(init_positions, dtype=np.float32)

    envs = []
    for i in range(NUM_ENVS):
        d_file, d_writer, t_file, t_writer = prepare_csv(
            CSV_DIR, CSV_BASENAME, i, DISTANCE_TARGETS[i])
        envs.append(dict(
            i=i,
            phase="baseline",           # settle already done in joint init
            phase_counter=0,
            settle_ok=False,
            settle_hist=[],
            baseline=None,
            baseline_buffer=[],
            targets_initialized=False,
            target_list_base=[],
            target_ori_base=[],
            cmd_plan=None, idx_list=None,
            cmd_idx=0, cmd_step_idx=0,
            last_goal_pos=None, last_goal_names=None,
            plan_fail_count=0,
            recording_active=False,
            # full-dof hold vector in robot.joint_names column order
            # (ordering-safe: each writer fills its OWN columns)
            hold_full=_init_hold_full.copy(),
            def_file=d_file, def_writer=d_writer,
            tac_file=t_file, tac_writer=t_writer,
            done=False, aborted=False,
        ))

    def plan_to(env, target_idx, weight, label):
        """cuRobo plan for one env toward its target A (0) or B (1)."""
        i = env["i"]
        plan_config.pose_cost_metric = PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=motion_gen.tensor_args.to_device(weight),
        )
        print("-" * 50 + f"\nPHASE env_{i}: {label}\n" + "-" * 50)

        if target_idx == 0:
            set_marker_color(target_a_prims[i], (0., 1., 0.))
            set_marker_color(target_b_prims[i], (0.1, 0.1, 0.1))
            qp, qv = art_get_joint_state(robot, i, idx_cu)
            cu_js = JointState(
                position=tensor_args.to_device(torch.tensor(qp, dtype=torch.float32)),
                velocity=tensor_args.to_device(torch.tensor(qv, dtype=torch.float32)) * 0.0,
                acceleration=tensor_args.to_device(torch.tensor(qv, dtype=torch.float32)) * 0.0,
                jerk=tensor_args.to_device(torch.tensor(qv, dtype=torch.float32)) * 0.0,
                joint_names=list(j_names),
            )
        else:
            set_marker_color(target_a_prims[i], (0.1, 0.1, 0.1))
            set_marker_color(target_b_prims[i], (0., 1., 0.))
            cu_js = JointState(
                position=env["last_goal_pos"],
                velocity=torch.zeros_like(env["last_goal_pos"]),
                acceleration=torch.zeros_like(env["last_goal_pos"]),
                jerk=torch.zeros_like(env["last_goal_pos"]),
                joint_names=env["last_goal_names"],
            )

        cu_js   = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
        ik_goal = Pose(
            position=tensor_args.to_device(
                torch.tensor(env["target_list_base"][target_idx], dtype=torch.float32)),
            quaternion=tensor_args.to_device(
                torch.tensor(env["target_ori_base"][target_idx], dtype=torch.float32)),
        )
        result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
        succ   = result.success.item()
        print(f"  env_{i} planning {label}: {'succeeded' if succ else 'failed'}")

        if succ:
            env["plan_fail_count"]  = 0
            curobo_plan             = result.get_interpolated_plan()
            env["last_goal_pos"]    = curobo_plan.position[-1].clone()
            env["last_goal_names"]  = curobo_plan.joint_names
            cmd_plan                = motion_gen.get_full_js(curobo_plan)
            common                  = [x for x in robot.joint_names if x in cmd_plan.joint_names]
            env["idx_list"]         = [dof_map[x] for x in common]
            env["cmd_plan"]         = cmd_plan.get_ordered_joint_state(common)
            env["cmd_idx"] = env["cmd_step_idx"] = 0
        else:
            env["plan_fail_count"] += 1
            if env["plan_fail_count"] >= MAX_PLAN_FAILS:
                print(f"  env_{i}: {MAX_PLAN_FAILS} failures ({label}). Aborting this env.")
                env["done"] = env["aborted"] = True

    def execute_plan_step(env):
        """Apply the current command, advance the index; True when finished."""
        cmd = env["cmd_plan"][env["cmd_idx"]]
        cmd_pos = cmd.position.cpu().numpy()
        env["hold_full"][env["idx_list"]] = np.asarray(cmd_pos, dtype=np.float32)

        env["cmd_step_idx"] += 1
        if env["cmd_step_idx"] >= STEPS_PER_CMD:
            env["cmd_idx"] += 1; env["cmd_step_idx"] = 0

        if env["cmd_idx"] >= len(env["cmd_plan"].position):
            final_pos = env["cmd_plan"].position[-1].cpu().numpy()
            env["hold_full"][env["idx_list"]] = np.asarray(final_pos, dtype=np.float32)  # hold via PD, no teleport
            env["cmd_idx"] = env["cmd_step_idx"] = 0
            env["cmd_plan"] = None
            env["plan_fail_count"] = 0
            return True
        return False

    def step_env(env, step, sim_time):
        """One frame of one env's phase machine (state updates only; the joint
        targets are applied ONCE for all envs after every env has run)."""
        i = env["i"]
        if env["done"]:
            return

        # ── baseline (settle-gated, unchanged logic) ─────────
        if env["phase"] == "baseline":
            if deform_view is not None and rest_refs[i] is not None:
                if not env["settle_ok"]:
                    try:
                        _c = read_env_nodes(i)
                        _c = parent_rot_inv_l[i].apply(_c - parent_pos_l[i])
                        _Rm, _T = kabsch_fit(rest_refs[i], _c)
                        _r = float(np.linalg.norm((rest_refs[i] @ _Rm.T + _T) - _c,
                                                  axis=1).mean())
                    except Exception:
                        _r = float("nan")
                    env["settle_hist"].append(_r)
                    _w = env["settle_hist"][-5:]
                    # PORT NOTE (2-env): threshold tightened 1e-3 -> 2.5e-4 m.
                    # Run 1 accepted env_1 at a STABLE 0.96mm plateau, baking a
                    # deformed pad into the baseline. 0.25mm is safely above the
                    # true settled value (~0.055mm) and below any contact.
                    if len(env["settle_hist"]) >= 5 and (max(_w) - min(_w)) < 5e-5 and max(_w) < 2.5e-4:
                        env["settle_ok"] = True
                        print(f"[PHASE] env_{i} pad settled after {len(env['settle_hist'])} "
                              f"frames (residual {_r*1000:.3f}mm). Collecting baseline.")
                    elif len(env["settle_hist"]) > MAX_SETTLE_FRAMES:
                        print(f"[PHASE] WARNING env_{i}: pad did not settle within "
                              f"{MAX_SETTLE_FRAMES} frames (residual {_r*1000:.3f}mm). "
                              f"Proceeding -- baseline may be contaminated.")
                        env["settle_ok"] = True
                    else:
                        if len(env["settle_hist"]) % 60 == 0:
                            print(f"[PHASE] env_{i} waiting for pad to settle... "
                                  f"residual={_r*1000:.3f}mm (frame {len(env['settle_hist'])})")
                        return

                cur = parent_rot_inv_l[i].apply(read_env_nodes(i) - parent_pos_l[i])
                dz = get_dz_from_nodes(cur, rest_refs[i], ordered_ids)
                if dz is not None:
                    env["baseline_buffer"].append(dz)

            env["phase_counter"] += 1
            if env["phase_counter"] >= BASELINE_FRAMES:
                if len(env["baseline_buffer"]) > 0:
                    env["baseline"] = np.mean(env["baseline_buffer"], axis=0)
                    _bs = np.std(env["baseline_buffer"], axis=0).mean()
                    if np.abs(env["baseline"]).mean() > 0.5 or _bs > 0.05:
                        print(f"[PHASE] WARNING env_{i}: baseline looks contaminated "
                              f"(|mean|={np.abs(env['baseline']).mean():.4f}, "
                              f"std={_bs:.4f}).")
                    print(f"[PHASE] env_{i} baseline done ({len(env['baseline_buffer'])} "
                          f"frames). mean={env['baseline'].mean():.6f}  Planning to A.")
                else:
                    env["baseline"] = np.zeros(N_ROWS * N_COLS, dtype=np.float32)
                    print(f"[PHASE] env_{i} baseline done but no data — using zeros.")
                log_cylinder(i, "post-settle")
                env["phase"] = "plan_A"
                env["phase_counter"] = 0
                if not env["targets_initialized"]:
                    env["target_list_base"], env["target_ori_base"] = init_targets_env(i)
                    env["targets_initialized"] = True
            return

        # ── plan_A: move to Target A ─────────────────────────
        if env["phase"] == "plan_A":
            if env["cmd_plan"] is None and step % 10 == 0:
                plan_to(env, 0, HOLD_ORI_WEIGHT, "plan_A -> Target A")
            if env["cmd_plan"] is not None:
                if execute_plan_step(env):
                    print(f"[PHASE] env_{i} arrived at Target A — recording + Y push "
                          f"({DISTANCE_TARGETS[i]*100:.0f}cm).")
                    log_cylinder(i, "at Target A")
                    env["recording_active"] = True
                    env["phase"] = "push_y"
                    env["phase_counter"] = 0
            return

        # ── push_y: A -> B, recording ────────────────────────
        if env["phase"] == "push_y":
            if env["recording_active"] and deform_view is not None:
                save_sponge_data_env(env, frame=step, sim_time=sim_time)
            if step % 60 == 0:
                log_cylinder(i, "during push")

            if env["cmd_plan"] is None and step % 10 == 0:
                plan_to(env, 1, PUSH_Y_WEIGHT, "push_y -> Target B")
            if env["cmd_plan"] is not None:
                if execute_plan_step(env):
                    print(f"[PHASE] env_{i} arrived at Target B — waiting "
                          f"{END_FRAMES} frames then closing this env.")
                    log_cylinder(i, "at Target B")
                    env["phase"] = "end_wait"
                    env["phase_counter"] = 0
            return

        # ── end_wait ─────────────────────────────────────────
        if env["phase"] == "end_wait":
            if env["recording_active"] and deform_view is not None:
                save_sponge_data_env(env, frame=step, sim_time=sim_time)
            env["phase_counter"] += 1
            if env["phase_counter"] >= END_FRAMES:
                print(f"[PHASE] env_{i} end wait done. This env is finished.")
                env["done"] = True
            return

    # ==========================================================
    # MAIN LOOP
    # PORT NOTE (2-env): step physics FIRST (fresh reads), run every env's
    # phase machine (each updates only its OWN hold_target row), then apply
    # ALL rows in one batched PD write. env_0 finishes its 20 cm push before
    # env_1's 30 cm; a finished env simply keeps holding its last target
    # until the other is done.
    # ==========================================================
    sim_time = 0.0
    step     = 0

    while simulation_app.is_running():

        sim.step(render=not args_cli.headless)
        robot.update(PHYSICS_DT)

        step     += 1
        sim_time += PHYSICS_DT

        for env in envs:
            step_env(env, step, sim_time)

        # one batched PD write: every env's current hold/command target,
        # as full-dof vectors in robot column order (ordering-safe).
        targets = np.stack([e["hold_full"] for e in envs], axis=0)
        art_apply_targets_all(robot, targets)

        if all(e["done"] for e in envs):
            print("[PHASE] All envs finished. Closing simulation.")
            break

    # ── Cleanup ───────────────────────────────────────────────
    for env in envs:
        env["def_file"].flush(); env["def_file"].close()
        env["tac_file"].flush(); env["tac_file"].close()
        if env["aborted"]:
            status = "ABORTED (planning failures)"
        elif not env["done"]:
            status = "INTERRUPTED — data truncated (window closed / loop stopped early)"
        else:
            status = "completed"
        print(f"[CSV] env_{env['i']} files closed ({status}, "
              f"{DISTANCE_TARGETS[env['i']]*100:.0f}cm push).")

    if getattr(args_cli, "hold", False) and not args_cli.headless:
        print("[HOLD] Both envs finished — viewer stays open (arms hold position). "
              "Close the window or Ctrl+C to exit.")
        _hold_targets = np.stack([e["hold_full"] for e in envs], axis=0)
        try:
            while simulation_app.is_running():
                art_apply_targets_all(robot, _hold_targets)
                sim.step(render=True)
                robot.update(PHYSICS_DT)
        except KeyboardInterrupt:
            pass

    simulation_app.close()
