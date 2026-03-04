#!/usr/bin/env python3
"""
view_exps.py — Web-based experiment viewer for VGGT-Long outputs.

Usage:
    python view_exps.py [--exps_dir ./exps] [--port 8080]
"""

import argparse
import base64
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import viser
import yaml
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Experiment scanning
# ---------------------------------------------------------------------------

def scan_experiments(exps_dir: Path) -> list:
    """Scan exps_dir for experiment directories, return sorted list of dicts."""
    experiments = []
    for config_file in sorted(exps_dir.glob("**/base_config.yaml")):
        exp_dir = config_file.parent
        try:
            parts = exp_dir.relative_to(exps_dir).parts
        except ValueError:
            continue

        if len(parts) < 2:
            continue

        dataset = parts[0]
        timestamp = parts[1]

        has_combined_pcd = (exp_dir / "pcd" / "combined_pcd.ply").exists()
        has_poses = (exp_dir / "camera_poses.txt").exists()

        if has_combined_pcd:
            status = "complete"
            status_prefix = "[✓]"
        elif has_poses:
            status = "partial"
            status_prefix = "[~]"
        else:
            status = "incomplete"
            status_prefix = "[✗]"

        pcd_dir = exp_dir / "pcd"
        chunk_plys = sorted(pcd_dir.glob("[0-9]*_pcd.ply")) if pcd_dir.exists() else []

        experiments.append({
            "path": exp_dir,
            "dataset": dataset,
            "timestamp": timestamp,
            "status": status,
            "status_prefix": status_prefix,
            "label": f"{status_prefix} {dataset} / {timestamp}",
            "has_combined_pcd": has_combined_pcd,
            "has_poses": has_poses,
            "has_chunks": len(chunk_plys) > 0,
            "n_chunks": len(chunk_plys),
        })

    return experiments


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_camera_poses(exp_dir: Path):
    """Return (list[4x4 ndarray], list[(fx,fy,cx,cy)]) from pose/intrinsic files."""
    poses_file = exp_dir / "camera_poses.txt"
    intrinsic_file = exp_dir / "intrinsic.txt"

    c2w_matrices = []
    if poses_file.exists():
        with open(poses_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = np.fromstring(line, sep=" ")
                if len(vals) == 16:
                    c2w_matrices.append(vals.reshape(4, 4).astype(np.float64))

    intrinsics = []
    if intrinsic_file.exists():
        with open(intrinsic_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = list(map(float, line.split()))
                if len(vals) >= 4:
                    intrinsics.append(tuple(vals[:4]))  # (fx, fy, cx, cy)

    return c2w_matrices, intrinsics


def load_config_info(exp_dir: Path) -> dict:
    """Parse base_config.yaml for display info."""
    config_file = exp_dir / "base_config.yaml"
    info = {}
    if config_file.exists():
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        if cfg:
            weights = cfg.get("Weights", {})
            model_cfg = cfg.get("Model", {})
            info["model"] = weights.get("model", "unknown")
            info["chunk_size"] = model_cfg.get("chunk_size", "?")
            info["overlap"] = model_cfg.get("overlap", "?")
            info["loop_enable"] = model_cfg.get("loop_enable", False)
            info["using_sim3"] = model_cfg.get("using_sim3", True)
    return info


def load_ply_pointcloud(ply_path: Path):
    """Load a PLY file, return (points float32 N×3, colors uint8 N×3)."""
    mesh = trimesh.load(str(ply_path), process=False)
    points = np.array(mesh.vertices, dtype=np.float32)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        colors = np.array(mesh.visual.vertex_colors, dtype=np.uint8)[:, :3]
    else:
        colors = np.full((len(points), 3), 180, dtype=np.uint8)
    return points, colors


# ---------------------------------------------------------------------------
# Info markdown builder
# ---------------------------------------------------------------------------

def build_info_markdown(exp: dict, cfg_info: dict, n_frames: int) -> str:
    exp_dir = exp["path"]

    lines = [
        f"**Dataset:** `{exp['dataset']}`",
        f"**Timestamp:** `{exp['timestamp']}`",
        f"**Status:** {exp['status']}",
        f"**Frames:** {n_frames}",
        f"**Model:** {cfg_info.get('model', '?')}",
        (
            f"**Chunk size:** {cfg_info.get('chunk_size', '?')}"
            f"  |  **Overlap:** {cfg_info.get('overlap', '?')}"
        ),
        (
            f"**Loop closure:** {'yes' if cfg_info.get('loop_enable') else 'no'}"
            f"  |  **Alignment:** {'Sim(3)' if cfg_info.get('using_sim3') else 'SE(3)'}"
        ),
    ]

    md = "\n\n".join(lines)

    plot_path = exp_dir / "sim3_opt_result.png"
    if plot_path.exists():
        b64 = base64.b64encode(plot_path.read_bytes()).decode()
        md += f"\n\n![opt](data:image/png;base64,{b64})"

    return md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VGGT-Long Experiment Viewer")
    parser.add_argument("--exps_dir", default="./exps", help="Experiments directory")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    args = parser.parse_args()

    exps_dir = Path(args.exps_dir).resolve()
    if not exps_dir.exists():
        print(f"Experiments directory not found: {exps_dir}")
        return

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")

    # --- Mutable state (nonlocal in callbacks) ---
    state = {
        "experiments": [],
        "current_exp": None,
        "scene_handles": [],
        "pending_delete": None,
        "pending_delete_time": 0.0,
        "pending_clean_time": 0.0,
    }

    # --- GUI layout ---
    with server.gui.add_folder("Experiments"):
        exp_dropdown = server.gui.add_dropdown(
            "Experiment", options=["(none)"], initial_value="(none)"
        )
        load_btn = server.gui.add_button("Load ▶", color="green")
        delete_btn = server.gui.add_button("Delete ✕", color="red")
        clean_btn = server.gui.add_button("Clean Incomplete", color="yellow")
        refresh_btn = server.gui.add_button("↺ Refresh")

    with server.gui.add_folder("Visualization"):
        show_combined = server.gui.add_checkbox("Combined point cloud", initial_value=True)
        show_chunks = server.gui.add_checkbox("Per-chunk PLYs", initial_value=False)
        show_cameras = server.gui.add_checkbox("Camera frustums", initial_value=True)
        point_size_slider = server.gui.add_slider(
            "Point size", min=0.001, max=0.5, step=0.001, initial_value=0.002
        )
        frustum_scale_slider = server.gui.add_slider(
            "Frustum scale", min=0.01, max=2.0, step=0.01, initial_value=0.02
        )

    info_md = server.gui.add_markdown("*No experiment loaded.*")

    # --- Helpers ---

    def clear_scene():
        for h in state["scene_handles"]:
            try:
                h.remove()
            except Exception:
                pass
        state["scene_handles"] = []

    def get_selected_exp() -> Optional[dict]:
        val = exp_dropdown.value
        for exp in state["experiments"]:
            if exp["label"] == val:
                return exp
        return None

    def refresh_experiments():
        state["experiments"] = scan_experiments(exps_dir)
        labels = (
            [exp["label"] for exp in state["experiments"]]
            if state["experiments"]
            else ["(none)"]
        )
        exp_dropdown.options = labels
        if labels:
            exp_dropdown.value = labels[0]

    def load_experiment(exp: dict):
        state["current_exp"] = exp
        state["pending_delete"] = None
        state["pending_delete_time"] = 0.0

        clear_scene()

        exp_dir = exp["path"]
        c2w_matrices, intrinsics = load_camera_poses(exp_dir)
        cfg_info = load_config_info(exp_dir)
        n_frames = len(c2w_matrices)

        info_md.content = build_info_markdown(exp, cfg_info, n_frames)

        # Combined point cloud
        if show_combined.value and exp["has_combined_pcd"]:
            combined_path = exp_dir / "pcd" / "combined_pcd.ply"
            try:
                points, colors = load_ply_pointcloud(combined_path)
                handle = server.scene.add_point_cloud(
                    "/pcd/combined",
                    points=points,
                    colors=colors,
                    point_size=point_size_slider.value,
                )
                state["scene_handles"].append(handle)
            except Exception as e:
                print(f"[warn] Failed to load combined PLY: {e}")

        # Per-chunk PLYs
        if show_chunks.value and exp["has_chunks"]:
            pcd_dir = exp_dir / "pcd"
            for ply_path in sorted(pcd_dir.glob("[0-9]*_pcd.ply")):
                chunk_idx = ply_path.stem.replace("_pcd", "")
                try:
                    points, colors = load_ply_pointcloud(ply_path)
                    handle = server.scene.add_point_cloud(
                        f"/pcd/chunk_{chunk_idx}",
                        points=points,
                        colors=colors,
                        point_size=point_size_slider.value,
                    )
                    state["scene_handles"].append(handle)
                except Exception as e:
                    print(f"[warn] Failed to load chunk PLY {ply_path.name}: {e}")

        # Camera frustums
        if show_cameras.value and c2w_matrices:
            scale = frustum_scale_slider.value
            for i, c2w in enumerate(c2w_matrices):
                R = c2w[:3, :3]
                t = c2w[:3, 3]

                # FOV and aspect from intrinsics (use cy as half-height estimate)
                if i < len(intrinsics):
                    fx, fy, cx, cy = intrinsics[i]
                    fov = float(2.0 * np.arctan(cy / fy))
                    aspect = float(cx / cy)
                else:
                    fov = float(np.radians(60.0))
                    aspect = 16.0 / 9.0

                quat_xyzw = Rotation.from_matrix(R).as_quat()
                wxyz = np.roll(quat_xyzw, 1)  # [x,y,z,w] → [w,x,y,z]

                handle = server.scene.add_camera_frustum(
                    f"/cameras/cam_{i:04d}",
                    fov=fov,
                    aspect=aspect,
                    scale=scale,
                    wxyz=wxyz,
                    position=t,
                )
                state["scene_handles"].append(handle)

    # --- Button callbacks ---

    @load_btn.on_click
    def on_load(_):
        exp = get_selected_exp()
        if exp is None:
            return
        if exp["status"] == "incomplete":
            info_md.content = "⚠️ This experiment is incomplete (no point cloud or poses)."
            return
        load_experiment(exp)

    @delete_btn.on_click
    def on_delete(_):
        exp = get_selected_exp()
        if exp is None:
            return

        selected_path = exp["path"]
        now = time.time()

        if (
            state["pending_delete"] == selected_path
            and now - state["pending_delete_time"] < 10.0
        ):
            # Confirmed — perform delete
            try:
                shutil.rmtree(str(selected_path))
            except Exception as e:
                info_md.content = f"⚠️ Delete failed: {e}"
                return

            state["pending_delete"] = None
            state["pending_delete_time"] = 0.0

            if (
                state["current_exp"] is not None
                and state["current_exp"]["path"] == selected_path
            ):
                state["current_exp"] = None
                clear_scene()
                info_md.content = "*Experiment deleted.*"

            refresh_experiments()
        else:
            state["pending_delete"] = selected_path
            state["pending_delete_time"] = now
            info_md.content = (
                f"⚠️ Click **Delete ✕** again within 10s to confirm deletion of:\n\n"
                f"`{selected_path}`"
            )

    @clean_btn.on_click
    def on_clean(_):
        incomplete = [e for e in state["experiments"] if e["status"] == "incomplete"]
        if not incomplete:
            info_md.content = "*No incomplete experiments to clean.*"
            return

        now = time.time()
        if now - state["pending_clean_time"] < 10.0:
            # Confirmed — delete all incomplete
            failed = []
            deleted = 0
            for exp in incomplete:
                try:
                    shutil.rmtree(str(exp["path"]))
                    deleted += 1
                    if (
                        state["current_exp"] is not None
                        and state["current_exp"]["path"] == exp["path"]
                    ):
                        state["current_exp"] = None
                        clear_scene()
                except Exception as e:
                    failed.append(f"`{exp['path']}`: {e}")

            state["pending_clean_time"] = 0.0
            refresh_experiments()

            if failed:
                info_md.content = (
                    f"Deleted {deleted} incomplete experiment(s). Failures:\n\n"
                    + "\n\n".join(failed)
                )
            else:
                info_md.content = f"*Deleted {deleted} incomplete experiment(s).*"
        else:
            state["pending_clean_time"] = now
            paths = "\n\n".join(f"- `{e['path']}`" for e in incomplete)
            info_md.content = (
                f"⚠️ Click **Clean Incomplete** again within 10s to delete "
                f"{len(incomplete)} incomplete experiment(s):\n\n{paths}"
            )

    @refresh_btn.on_click
    def on_refresh(_):
        refresh_experiments()

    # Visualization option changes → reload current experiment

    @show_combined.on_update
    def _(_):
        if state["current_exp"]:
            load_experiment(state["current_exp"])

    @show_chunks.on_update
    def _(_):
        if state["current_exp"]:
            load_experiment(state["current_exp"])

    @show_cameras.on_update
    def _(_):
        if state["current_exp"]:
            load_experiment(state["current_exp"])

    @point_size_slider.on_update
    def _(_):
        if state["current_exp"]:
            load_experiment(state["current_exp"])

    @frustum_scale_slider.on_update
    def _(_):
        if state["current_exp"]:
            load_experiment(state["current_exp"])

    # --- Initial scan ---
    refresh_experiments()

    print(f"\nVGGT-Long Experiment Viewer")
    print(f"  URL:  http://localhost:{args.port}")
    print(f"  Exps: {exps_dir}")
    print(f"  Found {len(state['experiments'])} experiment(s)\n")

    server.sleep_forever()


if __name__ == "__main__":
    main()
