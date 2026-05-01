#!/usr/bin/env python3
"""
================================================================================
 Doosan M0609 — Tactile Sensing Pipeline (Static Scene, No Motion Planning)
================================================================================

OVERVIEW
--------
This script loads the Isaac Sim scene with the Doosan M0609 robot and CoRo
tactile sensor, holds the robot at its initial joint position, and records
deformable mesh data + CNN tactile inference for 5 seconds.

PIPELINE PHASES (automatic, starts immediately on launch)
------------------------------------------------------------------------
  1. SETTLE      (10 frames)
     Waits for physics to stabilize. No data collected.
     → Console: "[PHASE] Settle done"

  2. BASELINE    (10 frames)
     Records dZ from the deformable mesh at rest (no contact).
     Computes the average baseline to subtract from future frames.
     → Console: "[PHASE] Baseline done"

  3. RECORD      (5 seconds = 300 frames at 60Hz)
     Robot holds initial position. Both CSV files are written every frame.
     → Console: "[PHASE] Recording complete. Closing simulation."

OUTPUT FILES (saved to script folder with timestamp)
-----------------------------------------------------
  {basename}_deformation_{timestamp}.csv  — one row per node per frame
  {basename}_tactiledata_{timestamp}.csv  — one row per frame (28 tactile values)

KEY SETTINGS
------------
  SETTLE_FRAMES    : frames to wait for physics stabilization (default 10)
  BASELINE_FRAMES  : frames used to compute no-contact baseline (default 10)
  SIM_DURATION     : recording duration in seconds (default 5.0)
  CSV_BASENAME     : prefix for output CSV filenames

FOLDER STRUCTURE EXPECTED
--------------------------
  Pushing_task/
  ├── Static_scene.py           ← this script
  ├── scenes/
  │   └── doosan_station_full.usd
  └── CNN_tactile/
      ├── Nodes_id_filtered.csv
      ├── best.onnx
      └── CNN_max.npy

DEPENDENCIES
------------
  Isaac Sim, onnxruntime, pandas, numpy
================================================================================
"""

try:
    import isaacsim
except ImportError:
    pass

import torch
_ = torch.zeros(4, device="cuda:0")

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": "1920", "height": "1080"})

import os
import csv
import datetime
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

from omni.isaac.core import World
from omni.isaac.core.robots import Robot

# ============================================================
# SETTINGS
# ============================================================
SCRIPT_DIR      = Path(__file__).resolve().parent
PROJECT_DIR     = str(SCRIPT_DIR)
SCENE_USD       = str(SCRIPT_DIR / "scenes" / "doosan_station_full.usd")
ROBOT_PRIM_PATH = "/World/m0609/m0609"

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
# DEFORMABLE MESH / CSV
# ============================================================
SPONGE_ROOT_PATH = "/World/CoRo_tactile/CoRo_tactile/Sponge"
SENSOR_POSE_PATH = "/World/CoRo_tactile/CoRo_tactile/Case_m"
CSV_DIR          = PROJECT_DIR
CSV_BASENAME     = "sponge_data"

# ============================================================
# TIMING
# ============================================================
PHYSICS_DT      = 1.0 / 60.0
MAX_EFFORT      = 1500.0
KP_GAINS        = 200000.0
KD_GAINS        = 10000.0

SETTLE_FRAMES   = 10    # frames to wait for physics stabilization
BASELINE_FRAMES = 10    # frames to collect no-contact baseline
SIM_DURATION    = 5.0   # seconds to record data


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
    return y_pred[0].flatten()


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

    tac_writer.writerow([frame, f"{sim_time:.6f}"] + tactile_vals.tolist())

    if frame % 60 == 0:
        print(f"[CSV] frame={frame}  t={sim_time:.3f}s  nodes={n}  tactile={tactile_vals[:4].round(3)}")
        def_file.flush()
        tac_file.flush()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(f"Scene USD : {SCENE_USD}")

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

    # ── World setup ───────────────────────────────────────────
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

    simulation_app.update()

    # ── Robot ─────────────────────────────────────────────────
    robot = my_world.scene.add(Robot(prim_path=ROBOT_PRIM_PATH, name="robot"))

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

    # ── Start simulation automatically ────────────────────────
    my_world.initialize_physics()
    robot.initialize()
    my_world.play()
    print("[BOOT] Simulation started automatically.")

    # ── State machine ─────────────────────────────────────────
    phase         = "settle"
    phase_counter = 0
    rec_time      = 0.0
    sim_time      = 0.0

    # ── Main loop ─────────────────────────────────────────────
    while simulation_app.is_running():
        my_world.step(render=True)

        step      = my_world.current_time_step_index
        sim_time += PHYSICS_DT

        # Hold robot at initial position
        if step <= 10:
            robot._articulation_view.initialize()
            idx = [robot.get_dof_index(x) for x in robot.dof_names]
            robot.set_joint_positions(np.deg2rad(INITIAL_JOINT_DEGREES).tolist(), idx)
            robot._articulation_view.set_max_efforts(
                np.array([MAX_EFFORT]*len(idx)), joint_indices=idx)
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
                print(f"[PHASE] Settle done ({SETTLE_FRAMES} frames). "
                      f"Starting baseline collection.")
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
                          f"mean={baseline.mean():.6f}  Starting recording.")
                else:
                    baseline = np.zeros(N_ROWS * N_COLS, dtype=np.float32)
                    print("[PHASE] Baseline done but no data — using zeros. "
                          "Starting recording.")
                phase         = "record"
                phase_counter = 0
                rec_time      = 0.0
            continue

        # ── PHASE: record ─────────────────────────────────────
        if phase == "record":
            if pos_attr is not None and sensor_prim is not None:
                save_sponge_data(
                    pos_attr, vel_attr, res_attr, sensor_prim,
                    def_writer, def_file,
                    tac_writer, tac_file,
                    frame=step, sim_time=sim_time,
                    ordered_ids=ordered_ids, baseline=baseline,
                    x_train_max=x_train_max, session=session,
                    input_name=input_name,
                )

            rec_time += PHYSICS_DT
            if rec_time >= SIM_DURATION:
                print(f"[PHASE] Recording complete ({SIM_DURATION}s). "
                      f"Closing simulation.")
                break
            continue

    # ── Cleanup ───────────────────────────────────────────────
    def_file.flush(); def_file.close()
    tac_file.flush(); tac_file.close()
    print("[CSV] Files closed.")
    simulation_app.close()