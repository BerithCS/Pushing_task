#!/usr/bin/env python3

# ==============================================================================
# 1) APP LAUNCH  — must come first, before any omni/isaacsim/pxr/cuRobo import.
#    PORT NOTE: replaces `from omni.isaac.kit import SimulationApp; SimulationApp(...)`
# ==============================================================================
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Doosan M0609 tactile pushing (Isaac Lab).")
# AppLauncher adds --headless, --device, --width, --height, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Match the standalone window size default; harmless when --headless.
if getattr(args_cli, "width", None) is None:
    args_cli.width = 1920
if getattr(args_cli, "height", None) is None:
    args_cli.height = 1080

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app  # <-- same name as before, so downstream code is unchanged

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

# PORT NOTE: Isaac Lab simulation driver + asset wrapper.
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg
from isaaclab.assets import Articulation, ArticulationCfg
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
# SETTINGS  (unchanged)
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
# CNN / INFERENCE SETTINGS  (restored, same as original standalone)
# ============================================================
NODES_FILE      = str(SCRIPT_DIR / "CNN_tactile" / "Nodes_id_filtered.csv")
ONNX_MODEL_PATH = str(SCRIPT_DIR / "CNN_tactile" / "best.onnx")
X_TRAIN_MAX_NPY = str(SCRIPT_DIR / "CNN_tactile" / "CNN_max.npy")

N_ROWS = 18
N_COLS = 12

# ============================================================
# OBJECT PUSHED  (unchanged)
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
# TARGET COMPUTATION  (unchanged)
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
# DEFORMABLE MESH / CSV  (unchanged)
# ============================================================
SPONGE_ROOT_PATH = "/World/CoRo_tactile/CoRo_tactile/Sponge"
SENSOR_POSE_PATH = "/World/CoRo_tactile/CoRo_tactile/Case_m"
CSV_DIR          = PROJECT_DIR
CSV_BASENAME     = "sponge_data"

# ============================================================
# TIMING  (unchanged)
# ============================================================
PHYSICS_DT         = 1.0 / 60.0
INTERPOLATION_DT   = 0.008
STEPS_PER_CMD      = 1
TIME_DILATION      = 0.5
MAX_EFFORT         = 1500.0
KP_GAINS           = 200000.0
KD_GAINS           = 10000.0

SETTLE_FRAMES      = 10
BASELINE_FRAMES    = 10
END_FRAMES         = 10

# ============================================================
# PLANNING  (unchanged)
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
# CNN HELPERS  (restored VERBATIM from the original standalone,
#  except get_dz_live which replaces the USD-based get_dz)
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


def get_dz_live(deform_view, case_view, rest_ref,
                parent_pos, parent_rot_inv, case_q_corr, ordered_ids):
    """Isaac Lab replacement for the original get_dz (which read USD attrs).
    Assembles the SAME four inputs from the live tensor views + v12 constants:
      rest_pts = mesh-local restShapePoints  (rest_ref)
      curr_pts = live nodes in parent-local frame
      R_mat, T = live Case_m pose (parent-local + Kabsch), same as the CSV.
    Then calls the original compute_dz_from_arrays unchanged."""
    try:
        cur = deform_view.get_simulation_nodal_positions()
        if hasattr(cur, "detach"): cur = cur.detach().cpu().numpy()
        cur = np.asarray(cur).reshape(-1, 3)
        cur = parent_rot_inv.apply(cur - parent_pos)

        T7 = case_view.get_transforms()
        if hasattr(T7, "detach"): T7 = T7.detach().cpu().numpy()
        T7 = np.asarray(T7).reshape(-1)
        p_world = T7[:3].astype(float)
        q_world = R.from_quat([float(T7[3]), float(T7[4]), float(T7[5]), float(T7[6])])
        p_local = parent_rot_inv.apply(p_world - parent_pos)
        q_local = (parent_rot_inv * q_world) * case_q_corr
        qx, qy, qz, qw = q_local.as_quat()

        R_mat = rotation_matrix_np(qw, qx, qy, qz)
        T     = np.asarray(p_local, dtype=float)
        n     = cur.shape[0]
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
# DEFORMABLE MESH HELPERS  (unchanged)
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
# CSV HELPERS  (unchanged)
# ============================================================

def prepare_csv(csv_dir: str, basename: str):
    """DEBUG BUILD: creates ONLY the deformation CSV (no tactile file)."""
    os.makedirs(csv_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

    tactile_cols = [f"T_{i+1:02d}" for i in range(28)]
    tac_path   = os.path.join(csv_dir, f"{basename}_tactiledata_{ts}.csv")
    tac_file   = open(tac_path, "w", newline="")
    tac_writer = csv.writer(tac_file)
    tac_writer.writerow(["frame", "t"] + tactile_cols)
    tac_file.flush()
    print(f"[CSV] Tactile file     : {tac_path}")

    return def_file, def_writer, tac_file, tac_writer


def save_sponge_data(deform_view, rest_ref, case_view,
                     parent_pos, parent_rot_inv, case_q_corr,
                     def_writer, def_file,
                     tac_writer, tac_file,
                     frame: int, sim_time: float,
                     ordered_ids, baseline, x_train_max, session, input_name):
    """Writes one row per mesh node per frame + one tactile prediction row.

    Columns match the ORIGINAL standalone CSV; the dz + inference path is the
    original's, fed by the live tensor-view data in the same local frames.
    """
    def _np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    try:
        cur = _np(deform_view.get_simulation_nodal_positions()).reshape(-1, 3)
        vel = _np(deform_view.get_simulation_nodal_velocities()).reshape(-1, 3)
        # JUNE-26 CONVENTION: nodes and pose all in the PARENT-LOCAL frame.
        cur = parent_rot_inv.apply(cur - parent_pos)
        vel = parent_rot_inv.apply(vel)
    except Exception as e:
        if frame % 60 == 0:
            print(f"[CSV] frame={frame}: deformable view read failed: {e}")
        return

    # LIVE Case_m WORLD pose from the rigid-body view, then convert to LOCAL
    # (relative to the static parent), matching the original xformOp read.
    trans = (0.0, 0.0, 0.0)
    ori_w, ori_x, ori_y, ori_z = 1.0, 0.0, 0.0, 0.0
    if case_view is not None:
        try:
            T = _np(case_view.get_transforms()).reshape(-1)
            p_world = np.array([float(T[0]), float(T[1]), float(T[2])])
            q_world = R.from_quat([float(T[3]), float(T[4]), float(T[5]), float(T[6])])  # xyzw

            p_local = parent_rot_inv.apply(p_world - parent_pos)
            q_local = (parent_rot_inv * q_world) * case_q_corr

            trans = (float(p_local[0]), float(p_local[1]), float(p_local[2]))
            qx, qy, qz, qw = q_local.as_quat()  # scipy returns (x,y,z,w)
            ori_w, ori_x, ori_y, ori_z = float(qw), float(qx), float(qy), float(qz)
        except Exception as e:
            if frame % 60 == 0:
                print(f"[CSV] frame={frame}: case pose read failed: {e}")

    n    = cur.shape[0]
    n_re = rest_ref.shape[0] if rest_ref is not None else 0

    # ── dz + CNN inference (original path, live inputs) ──────────────────
    tactile_vals = np.zeros(28, dtype=np.float32)
    if rest_ref is not None and baseline is not None:
        try:
            R_mat = rotation_matrix_np(ori_w, ori_x, ori_y, ori_z)
            T_vec = np.array([trans[0], trans[1], trans[2]], dtype=float)
            dz    = compute_dz_from_arrays(rest_ref, cur, R_mat, T_vec,
                                           ordered_ids, list(range(n)))
            tactile_vals = infer_tactile(dz, baseline, x_train_max, session, input_name)
        except Exception as e:
            print(f"[WARN] Inference failed at frame {frame}: {e}")

    dmax = 0.0
    for i in range(n):
        cx, cy, cz = float(cur[i, 0]), float(cur[i, 1]), float(cur[i, 2])
        vx, vy, vz = float(vel[i, 0]), float(vel[i, 1]), float(vel[i, 2])
        rx = ry = rz = 0.0
        if i < n_re:
            rx, ry, rz = float(rest_ref[i, 0]), float(rest_ref[i, 1]), float(rest_ref[i, 2])
            d = ((cx - rx)**2 + (cy - ry)**2 + (cz - rz)**2) ** 0.5
            if d > dmax: dmax = d

        def_writer.writerow([
            frame, f"{sim_time:.6f}", i,
            cx, cy, cz,
            vx, vy, vz,
            rx, ry, rz,
            trans[0], trans[1], trans[2],
            ori_w, ori_x, ori_y, ori_z,   # (w,x,y,z) exactly as the original wrote
        ])

    tac_writer.writerow([frame, f"{sim_time:.6f}"] + tactile_vals.tolist())

    if frame % 60 == 0:
        print(f"[CSV] frame={frame}  t={sim_time:.3f}s  nodes={n}  "
              f"case_local_trans=({trans[0]:.3f},{trans[1]:.3f},{trans[2]:.3f})  "
              f"tactile={tactile_vals[:4].round(3)}")
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
        print(f"  Physics mat -> {prim_path}  (s={static_friction}, d={dynamic_friction})")
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


def create_visual_marker(stage, path, position, size=0.04, color=(0.1, 0.1, 0.1)):
    """PORT NOTE: replaces omni.isaac.core.objects.cuboid.VisualCuboid + OmniPBR.
    A purely-cosmetic cube marker built from plain USD (no physics)."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(p) for p in position]))
    xf.AddScaleOp().Set(Gf.Vec3f(size, size, size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cube.GetPrim()


def set_marker_color(prim, color):
    """Recolor a visual marker prim (cosmetic only)."""
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
# ISAAC LAB ARTICULATION ADAPTERS
# PORT NOTE: these wrap isaaclab.assets.Articulation so the rest of the code can
# read/write joints in the same shape the standalone Robot API used (1-D numpy
# arrays indexed by cuRobo joint order). Isaac Lab is batched: all buffers carry
# a leading env dimension (here always 1), and joint order follows
# `robot.joint_names`, so we build a name->column map once.
# ==============================================================================

def build_dof_index_map(robot: Articulation):
    """Return {joint_name: column_index} for the single-env articulation."""
    names = robot.joint_names
    return {n: i for i, n in enumerate(names)}


def art_get_dof_indices(dof_map, joint_names):
    return [dof_map[n] for n in joint_names]


def art_set_joint_positions(robot: Articulation, positions_1d, col_indices, device):
    """Hard-set joint positions (teleport) for the given columns, env 0."""
    q = robot.data.joint_pos.clone()                      # (1, num_dof)
    pos = torch.as_tensor(positions_1d, dtype=torch.float32, device=device)
    q[0, col_indices] = pos
    robot.write_joint_state_to_sim(
        position=q,
        velocity=torch.zeros_like(q),
    )


def art_get_joint_state(robot: Articulation, col_indices):
    """Return (positions, velocities) as 1-D numpy arrays in the given col order."""
    qp = robot.data.joint_pos[0, col_indices].detach().cpu().numpy()
    qv = robot.data.joint_vel[0, col_indices].detach().cpu().numpy()
    return qp, qv


def art_apply_position_target(robot: Articulation, positions_1d, col_indices, device):
    """PORT NOTE: replaces ctrl.apply_action(ArticulationAction(...)).
    Sends a PD position target to the given joint columns for env 0."""
    tgt = torch.as_tensor(positions_1d, dtype=torch.float32, device=device).unsqueeze(0)
    joint_ids = list(col_indices)
    robot.set_joint_position_target(tgt, joint_ids=joint_ids)
    robot.write_data_to_sim()


def get_prim_world_pose(stage, prim_path):
    """World pose (pos_xyz numpy, quat_wxyz numpy) of a prim via USD xform cache."""
    prim = stage.GetPrimAtPath(prim_path)
    xcache = UsdGeom.XformCache()
    m = xcache.GetLocalToWorldTransform(prim)
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    imag = q.GetImaginary()
    pos  = np.array([t[0], t[1], t[2]], dtype=np.float32)
    quat = np.array([q.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)
    return pos, quat


def init_targets(stage, robot, motion_gen, init_positions, j_names,
                 tensor_args, target_a_prim, target_b_prim):
    # PORT NOTE: robot.get_world_pose() -> USD xform cache on the robot root prim.
    base_pos, base_ori           = get_prim_world_pose(stage, ROBOT_PRIM_PATH)
    curobo_ee_pos, curobo_ee_ori = get_curobo_fk(motion_gen, init_positions, j_names, tensor_args)
    ee_pos, ee_ori               = get_prim_world_pose(stage, f"{ROBOT_PRIM_PATH}/{EE_LINK_NAME}")

    print(f"Robot base pos   : {base_pos.tolist()}")
    print(f"cuRobo FK EE pos : {curobo_ee_pos.tolist()}")
    print(f"Target A (world) : {TARGET_A_WORLD.tolist()}")
    print(f"Target B (world) : {TARGET_B_WORLD.tolist()}")

    goal_ori_base    = curobo_ee_ori.astype(np.float32)
    target_list_base = []
    target_ori_base  = []

    for pt_world, tgt_prim in zip([TARGET_A_WORLD, TARGET_B_WORLD],
                                  [target_a_prim, target_b_prim]):
        pt_base = world_to_base_frame(pt_world, base_pos, base_ori)
        target_list_base.append(pt_base)
        target_ori_base.append(goal_ori_base.copy())
        # move the cosmetic marker to the world target (optional)
        if tgt_prim and tgt_prim.IsValid():
            xf = UsdGeom.Xformable(tgt_prim)
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*[float(v) for v in pt_world]))
                    break

    return target_list_base, target_ori_base


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Scene USD      : {SCENE_USD}")
    print(f"Target A       : {TARGET_A_WORLD.tolist()}")
    print(f"Target B       : {TARGET_B_WORLD.tolist()}")

    device = args_cli.device if getattr(args_cli, "device", None) else "cuda:0"

    # ── Load CNN resources (restored, same as original) ────────
    # NOTE: on this laptop (RTX A2000, cuDNN 9.7.1) the CUDA EP can fail on one
    # Conv and ONNX Runtime falls back to CPU — that fallback worked before and
    # is acceptable here. Ensure best.onnx is the cuDNN-patched export.
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

    # ==========================================================
    # PORT NOTE: World(...) -> SimulationContext(SimulationCfg(...))
    #
    # DEFORMABLE READ (final, proven by probe on this stack):
    # Sim 5.1 / Lab 2.3.2 / Physics 107.3. The sponge is a PhysxAutoDeformable
    # VOLUME deformable. USD/Fabric `points` never stream the live mesh (they
    # return a frozen post-reset snapshot — that's why velocities were constant
    # 796mm/s every frame). The ONLY live source is the physics tensor view
    # `create_volume_deformable_body_view`, read via get_simulation_nodal_positions().
    # Fabric setting is irrelevant to that view, so we leave it at default.
    # ==========================================================
    sim_cfg = SimulationCfg(
        dt=PHYSICS_DT,
        device=device,
        physx=PhysxCfg(
            enable_ccd=False,
            # GPU pipeline is required for deformables:
            gpu_max_soft_body_contacts=2 ** 20,
        ),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    stage = omni.usd.get_context().get_stage()

    # Ensure /World exists and is the default prim, then reference the scene USD.
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim.IsValid():
        UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # PORT NOTE: standalone appended SCENE_USD as a sublayer of the root layer.
    # In Isaac Lab we reference it under /World instead (cleaner + matches the
    # prim paths your cuRobo config expects). If your USD's default prim is
    # itself "/World", add_reference_to_stage keeps the same absolute paths.
    add_reference_to_stage(usd_path=SCENE_USD, prim_path="/World")
    simulation_app.update()

    # ── GPU dynamics (kept as a belt-and-suspenders on the scene prim) ──
    physics_scene_prim = stage.GetPrimAtPath("/physicsScene")
    if physics_scene_prim.IsValid():
        physics_scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
        physics_scene_api.GetEnableGPUDynamicsAttr().Set(True)
        physics_scene_api.GetBroadphaseTypeAttr().Set("GPU")
        print("GPU dynamics enabled.")

    # ── Disable table_base collision (unchanged) ──────────────
    _tbc = stage.GetPrimAtPath("/World/table_base/table_base/collisions")
    if _tbc.IsValid():
        _tbc.SetActive(False)
        print("  table_base collisions deactivated.")

    # ── Deformable physics tuning (unchanged) ─────────────────
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

    # ── Object pushed + solver (unchanged) ────────────────────
    create_object_pushed(stage)
    apply_cylinder_solver_iters(stage)

    # ── Physics materials (unchanged) ─────────────────────────
    apply_physics_material(stage, OBJECT_MESH_PATH,                                        0.2, 0.15)
    apply_physics_material(stage, "/World/table_cover/Cube",                                0.9,  0.9)
    apply_physics_material(stage, "/World/m0609/m0609/link_6/adapter",                      0.4,  0.35)
    apply_physics_material(stage, "/World/CoRo_tactile/CoRo_tactile/Sponge/collision_mesh", 0.9,  0.8)

    simulation_app.update()

    # ── Scene objects ─────────────────────────────────────────
    # PORT NOTE: my_world.scene.add(Robot(...)) -> Articulation(ArticulationCfg(...)).
    # We reference the robot that already exists in the stage via its prim path.
    # A single implicit PD actuator group covers all arm joints, matching the
    # KP/KD/effort the standalone script set per joint.
    from isaaclab.actuators import ImplicitActuatorCfg
    robot_cfg_lab = ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
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

    # Cosmetic target markers (optional). PORT NOTE: replaces VisualCuboid+OmniPBR.
    target_a_prim = create_visual_marker(stage, "/World/target_A", TARGET_A_WORLD)
    target_b_prim = create_visual_marker(stage, "/World/target_B", TARGET_B_WORLD)

    # ── cuRobo setup (unchanged except the stage handle) ──────
    usd_helper = UsdHelper()
    usd_helper.load_stage(stage)
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

    # ==========================================================
    # PORT NOTE: my_world.initialize_physics() + robot.initialize()
    #            -> sim.reset() (this initialises the physics view AND the
    #               Articulation buffers in one call).
    # ==========================================================
    sim.reset()
    robot.reset()

    # Build the joint name->column map once, and the cuRobo-order index list.
    dof_map  = build_dof_index_map(robot)
    idx_cu   = art_get_dof_indices(dof_map, j_names)   # columns in cuRobo order
    print(f"[JOINTS] Articulation joint order : {robot.joint_names}")
    print(f"[JOINTS] cuRobo joint order        : {j_names}")
    print(f"[JOINTS] cuRobo->column indices    : {idx_cu}")

    plan_config = MotionGenPlanConfig(
        enable_graph=False, enable_graph_attempt=4,
        max_attempts=4, enable_finetune_trajopt=True,
        time_dilation_factor=TIME_DILATION, pose_cost_metric=None,
    )

    # ==========================================================
    # JOINT INIT — write, settle, verify (fixes "positions don't stick")
    # PORT NOTE: In the standalone loop you were re-setting joints for the first
    # 10 played frames and hoping they'd stick before cuRobo read them. In Isaac
    # Lab we do it deterministically BEFORE the phase loop: write the state to
    # sim, step a few times so PhysX latches it, then read it back and confirm.
    # ==========================================================
    art_set_joint_positions(robot, init_positions, idx_cu, device)
    robot.write_data_to_sim()
    for _ in range(SETTLE_FRAMES):
        # hold the target while physics settles
        art_apply_position_target(robot, init_positions, idx_cu, device)
        sim.step(render=not args_cli.headless)
        robot.update(PHYSICS_DT)
    qp_check, qv_check = art_get_joint_state(robot, idx_cu)
    print(f"[DIAG] init target (rad) : {np.round(init_positions, 4)}")
    print(f"[DIAG] joint pos  (rad)  : {np.round(qp_check, 4)}")
    print(f"[DIAG] joint vel  (rad/s): {np.round(qv_check, 4)}")
    print(f"[DIAG] max |pos err|     : {np.max(np.abs(qp_check - np.array(init_positions))):.5f}")

    # ── Deformable read setup (volume deformable tensor view) ──
    # PROBE-PROVEN on this stack (Sim 5.1 / Lab 2.3.2 / Physics 107.3):
    #   * Sponge is a PhysxAutoDeformable VOLUME deformable.
    #   * USD `points` & Fabric are frozen snapshots (constant velocities) — dead.
    #   * The live source is the physics tensor view created by
    #     create_volume_deformable_body_view(<body_path>), read each frame via
    #     get_simulation_nodal_positions(). count==1 for our single sponge.
    from isaacsim.core.simulation_manager import SimulationManager

    def _to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    deform_view = None
    rest_ref = None
    try:
        _pv = SimulationManager.get_physics_sim_view()
        _mk = getattr(_pv, "create_volume_deformable_body_view", None)
        if _mk is not None:
            deform_view = _mk(SPONGE_ROOT_PATH)
            cnt = getattr(deform_view, "count", None) if deform_view is not None else None
            print(f"[DEFORMABLE] volume view count={cnt}")
            if deform_view is not None and cnt:
                # JUNE-26 CONVENTION: rest = the authored mesh-local
                # omniphysics:restShapePoints (small coords, static), exactly what
                # the standalone recorded in s1_Rx/Ry/Rz. Read once from USD.
                rest_ref = None
                _sm = find_deformable_simulation_mesh(SPONGE_ROOT_PATH)
                if _sm is not None:
                    _ra = _sm.GetAttribute("omniphysics:restShapePoints")
                    _rv = _ra.Get() if (_ra and _ra.IsValid()) else None
                    if _rv is not None:
                        rest_ref = np.array([[float(p[0]), float(p[1]), float(p[2])]
                                             for p in _rv], dtype=float)
                if rest_ref is None:
                    print("[DEFORMABLE] WARNING: restShapePoints unavailable.")
                else:
                    print(f"[DEFORMABLE] nodes={rest_ref.shape[0]}  "
                          f"restShapePoints first={rest_ref[0].round(5).tolist()}")
            else:
                deform_view = None
        else:
            print("[DEFORMABLE] ERROR: create_volume_deformable_body_view missing.")
    except Exception as e:
        print(f"[DEFORMABLE] ERROR creating volume view: {e}")

    sensor_prim = stage.GetPrimAtPath(SENSOR_POSE_PATH)
    if not sensor_prim or not sensor_prim.IsValid():
        print(f"[WARNING] Sensor prim not found: {SENSOR_POSE_PATH}")
        sensor_prim = None

    # ── Live Case_m pose via rigid-body physics view ──────────
    # The sensor Case_m is a rigid body (PhysxRigidBodyAPI), attached to the
    # adapter. Its USD xform is FROZEN during sim (USD isn't the live source),
    # so we read its pose from the rigid-body tensor view instead. get_transforms()
    # returns (count, 7) = [x, y, z, qx, qy, qz, qw] in WORLD frame, live.
    #
    # FRAME MATCH WITH THE STANDALONE CSV: the original Test_pushing.py read
    # xformOp:translate / xformOp:orient on Case_m — i.e. the LOCAL pose,
    # relative to its parent /World/CoRo_tactile/CoRo_tactile. To keep the CSV
    # convention identical (compute_dz expects local), we convert the live
    # WORLD pose to LOCAL each frame using the parent's (static) world pose:
    #     T_local = R_parent^-1 · (T_world - p_parent)
    #     q_local = q_parent^-1 ⊗ q_world
    case_view = None
    parent_pos = np.zeros(3)
    parent_rot_inv = R.identity()
    try:
        _mkr = getattr(_pv, "create_rigid_body_view", None)
        if _mkr is not None:
            case_view = _mkr(SENSOR_POSE_PATH)
            ccnt = getattr(case_view, "count", None) if case_view is not None else None
            print(f"[CASE] rigid-body view count={ccnt}")
            if not ccnt:
                case_view = None
            else:
                _t0 = _to_np(case_view.get_transforms()).reshape(-1)
                print(f"[CASE] initial WORLD pose xyz={_t0[:3].round(5).tolist()} "
                      f"quat(xyzw)={_t0[3:7].round(5).tolist()}")
        else:
            print("[CASE] ERROR: create_rigid_body_view missing.")
    except Exception as e:
        print(f"[CASE] ERROR creating rigid-body view: {e}")

    # Parent world pose (static prim -> USD xform cache is fine, read once).
    _parent_path = SENSOR_POSE_PATH.rsplit("/", 1)[0]
    _parent_prim = stage.GetPrimAtPath(_parent_path)
    if _parent_prim and _parent_prim.IsValid():
        _m = UsdGeom.XformCache().GetLocalToWorldTransform(_parent_prim)
        _t = _m.ExtractTranslation()
        _q = _m.ExtractRotationQuat()
        _im = _q.GetImaginary()
        parent_pos = np.array([float(_t[0]), float(_t[1]), float(_t[2])])
        parent_rot = R.from_quat([float(_im[0]), float(_im[1]), float(_im[2]),
                                  float(_q.GetReal())])  # scipy = (x,y,z,w)
        parent_rot_inv = parent_rot.inv()
        print(f"[CASE] parent {_parent_path} world pos={parent_pos.round(5).tolist()} "
              f"rot(quat xyzw)={parent_rot.as_quat().round(5).tolist()}")
    else:
        print(f"[CASE] WARNING: parent prim not found at {_parent_path}; "
              f"pose will be written in WORLD frame.")

    # One-time ORIENTATION CALIBRATION (Kabsch). June 26's recorded orientation
    # was NOT the authored xformOp:orient (that reads identity) — it was the
    # live physics pose written back by the standalone World. Rather than guess
    # the body-frame convention, we solve for the rotation that compute_dz
    # actually needs: the R that maps the mesh-local rest points onto the
    # current (parent-local) nodes at t=0, i.e. R@rest + T ≈ cur. We then store
    # the constant offset from the view's local quat to that fitted R and apply
    # it every frame. The alignment residual is printed — it should be ~0 mm.
    case_q_corr = R.identity()
    try:
        if case_view is not None and deform_view is not None and rest_ref is not None:
            _T0 = _to_np(case_view.get_transforms()).reshape(-1)
            p_w0 = np.array([float(_T0[0]), float(_T0[1]), float(_T0[2])])
            q_w0 = R.from_quat([float(_T0[3]), float(_T0[4]), float(_T0[5]), float(_T0[6])])
            q_view_local0 = parent_rot_inv * q_w0
            p_local0 = parent_rot_inv.apply(p_w0 - parent_pos)

            cur_w0 = _to_np(deform_view.get_simulation_nodal_positions()).reshape(-1, 3)
            cur_l0 = parent_rot_inv.apply(cur_w0 - parent_pos)

            # Kabsch: find R_fit s.t. R_fit @ (rest - rest_centroid) ≈ cur_l0 - cur_centroid
            A = rest_ref - rest_ref.mean(axis=0)
            B = cur_l0  - cur_l0.mean(axis=0)
            H = A.T @ B
            U, S, Vt = np.linalg.svd(H)
            D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
            R_fit_mat = Vt.T @ D @ U.T
            q_fit = R.from_matrix(R_fit_mat)

            # constant offset applied to the per-frame view-local quat
            case_q_corr = q_view_local0.inv() * q_fit

            # sanity: residual of R_fit@rest + T_fit vs cur (compute_dz's transform)
            T_fit = cur_l0.mean(axis=0) - R_fit_mat @ rest_ref.mean(axis=0)
            resid = np.linalg.norm((rest_ref @ R_fit_mat.T + T_fit) - cur_l0, axis=1)
            print(f"[CASE] Kabsch q_fit (xyzw)      = {q_fit.as_quat().round(5).tolist()}")
            print(f"[CASE] orient offset (xyzw)     = {case_q_corr.as_quat().round(5).tolist()}")
            print(f"[CASE] T_fit (local)            = {T_fit.round(5).tolist()}")
            print(f"[CASE] recorded T_local at t=0  = {p_local0.round(5).tolist()}")
            print(f"[CASE] rest->cur alignment residual: mean={resid.mean()*1000:.3f}mm "
                  f"max={resid.max()*1000:.3f}mm  (should be ~0)")
        else:
            print("[CASE] calibration skipped (missing view or rest points).")
    except Exception as e:
        print(f"[CASE] Kabsch calibration failed: {e}")

    # ── Prepare CSV files (unchanged) ─────────────────────────
    def_file, def_writer, tac_file, tac_writer = prepare_csv(CSV_DIR, CSV_BASENAME)

    # ── State machine (unchanged logic) ───────────────────────
    # PORT NOTE: we start at "baseline" because SETTLE already happened during
    # the deterministic joint-init block above. Set to "settle" if you'd rather
    # keep the original ordering.
    phase              = "baseline"
    phase_counter      = 0

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
    step                   = 0

    material_list_prims = [target_a_prim, target_b_prim]

    # ==========================================================
    # MAIN LOOP
    # PORT NOTE: no is_playing() gate — SimulationContext plays as soon as we
    # step it. `step` is our own frame counter (was current_time_step_index).
    #
    # ORDER MATTERS. We step FIRST so mesh/joint reads are fresh for this frame,
    # then within each phase we apply the current command AND advance the index
    # in the SAME place (matching the standalone apply_action placement). The
    # earlier port split these apart, which caused an off-by-one: the arm was
    # commanded from a stale index and snapped back at the end of each plan.
    # `hold_target` holds the last sent target so the PD arm doesn't sag when
    # there's no active trajectory.
    # ==========================================================
    hold_target = list(init_positions)   # last commanded joint target (cols in hold_idx order)
    hold_idx    = list(idx_cu)

    while simulation_app.is_running():

        # advance physics one frame, then refresh articulation buffers
        sim.step(render=not args_cli.headless)
        robot.update(PHYSICS_DT)

        # re-assert the current hold target every frame (PD position control)
        art_apply_position_target(robot, hold_target, hold_idx, device)

        step     += 1
        sim_time += PHYSICS_DT

        # ── PHASE: settle ─────────────────────────────────────
        if phase == "settle":
            phase_counter += 1
            if phase_counter >= SETTLE_FRAMES:
                print(f"[PHASE] Settle done ({SETTLE_FRAMES} frames). Starting baseline collection.")
                phase         = "baseline"
                phase_counter = 0
            continue

        # ── PHASE: baseline (restored: collect no-contact dz) ─
        if phase == "baseline":
            if deform_view is not None and rest_ref is not None:
                dz = get_dz_live(deform_view, case_view, rest_ref,
                                 parent_pos, parent_rot_inv, case_q_corr,
                                 ordered_ids)
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

                if not targets_initialized:
                    target_list_base, target_ori_base = init_targets(
                        stage, robot, motion_gen, init_positions, j_names,
                        tensor_args, target_a_prim, target_b_prim,
                    )
                    targets_initialized = True
            continue

        # ── PHASE: plan_A — move to Target A ─────────────────
        if phase == "plan_A":
            if cmd_plan is None and step % 10 == 0:
                plan_config.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device(HOLD_ORI_WEIGHT),
                )
                print("-"*50 + "\nPHASE: plan_A -> Target A\n" + "-"*50)

                set_marker_color(target_a_prim, (0., 1., 0.))
                set_marker_color(target_b_prim, (0.1, 0.1, 0.1))

                # PORT NOTE: robot.get_joints_state() -> art_get_joint_state()
                qp, qv = art_get_joint_state(robot, idx_cu)
                cu_js  = JointState(
                    position=tensor_args.to_device(torch.tensor(qp, dtype=torch.float32)),
                    velocity=tensor_args.to_device(torch.tensor(qv, dtype=torch.float32)) * 0.0,
                    acceleration=tensor_args.to_device(torch.tensor(qv, dtype=torch.float32)) * 0.0,
                    jerk=tensor_args.to_device(torch.tensor(qv, dtype=torch.float32)) * 0.0,
                    joint_names=list(j_names),
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
                    common                 = [x for x in robot.joint_names if x in cmd_plan.joint_names]
                    idx_list               = [dof_map[x] for x in common]
                    cmd_plan               = cmd_plan.get_ordered_joint_state(common)
                    cmd_idx = cmd_step_idx = 0
                    planned_phase          = "plan_A"
                else:
                    plan_fail_count += 1
                    if plan_fail_count >= MAX_PLAN_FAILS:
                        print(f"  {MAX_PLAN_FAILS} failures to Target A. Aborting.")
                        break

            # Execute trajectory: apply the current command AND advance the
            # index in the same place (matches standalone apply_action).
            if cmd_plan is not None:
                cmd = cmd_plan[cmd_idx]
                cmd_pos = cmd.position.cpu().numpy()
                art_apply_position_target(robot, cmd_pos, idx_list, device)
                # keep the arm on this target on any frame where we don't re-plan
                hold_target, hold_idx = list(cmd_pos), list(idx_list)

                cmd_step_idx += 1
                if cmd_step_idx >= STEPS_PER_CMD:
                    cmd_idx += 1; cmd_step_idx = 0

                if cmd_idx >= len(cmd_plan.position):
                    final_pos = cmd_plan.position[-1].cpu().numpy()
                    # hold the final target with PD (do NOT teleport — that was
                    # the snap-back). The controller holds the arm at Target A.
                    hold_target, hold_idx = list(final_pos), list(idx_list)
                    art_apply_position_target(robot, final_pos, idx_list, device)

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

            if recording_active and deform_view is not None:
                save_sponge_data(
                    deform_view, rest_ref, case_view,
                    parent_pos, parent_rot_inv, case_q_corr,
                    def_writer, def_file,
                    tac_writer, tac_file,
                    frame=step, sim_time=sim_time,
                    ordered_ids=ordered_ids, baseline=baseline,
                    x_train_max=x_train_max, session=session,
                    input_name=input_name,
                )

            if cmd_plan is None and step % 10 == 0:
                plan_config.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=motion_gen.tensor_args.to_device(PUSH_Y_WEIGHT),
                )
                print("-"*50 + "\nPHASE: push_y -> Target B\n" + "-"*50)

                set_marker_color(target_a_prim, (0.1, 0.1, 0.1))
                set_marker_color(target_b_prim, (0., 1., 0.))

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
                    common                 = [x for x in robot.joint_names if x in cmd_plan.joint_names]
                    idx_list               = [dof_map[x] for x in common]
                    cmd_plan               = cmd_plan.get_ordered_joint_state(common)
                    cmd_idx = cmd_step_idx = 0
                    planned_phase          = "push_y"
                else:
                    plan_fail_count += 1
                    if plan_fail_count >= MAX_PLAN_FAILS:
                        print(f"  {MAX_PLAN_FAILS} failures to Target B. Aborting.")
                        break

            # Execute trajectory: apply current command AND advance index together.
            if cmd_plan is not None:
                cmd = cmd_plan[cmd_idx]
                cmd_pos = cmd.position.cpu().numpy()
                art_apply_position_target(robot, cmd_pos, idx_list, device)
                hold_target, hold_idx = list(cmd_pos), list(idx_list)

                cmd_step_idx += 1
                if cmd_step_idx >= STEPS_PER_CMD:
                    cmd_idx += 1; cmd_step_idx = 0

                if cmd_idx >= len(cmd_plan.position):
                    final_pos = cmd_plan.position[-1].cpu().numpy()
                    # hold Target B with PD instead of teleporting back.
                    hold_target, hold_idx = list(final_pos), list(idx_list)
                    art_apply_position_target(robot, final_pos, idx_list, device)

                    cmd_idx = cmd_step_idx = 0
                    cmd_plan      = None
                    planned_phase = None

                    print(f"[PHASE] Arrived at Target B — waiting {END_FRAMES} frames then closing.")
                    phase         = "end_wait"
                    phase_counter = 0
            continue

        # ── PHASE: end_wait ───────────────────────────────────
        if phase == "end_wait":
            if recording_active and deform_view is not None:
                save_sponge_data(
                    deform_view, rest_ref, case_view,
                    parent_pos, parent_rot_inv, case_q_corr,
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
