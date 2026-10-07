import copy
import matplotlib.patches as mpatches
import os
import torch
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple, Callable

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from nuplan.common.actor_state.state_representation import StateSE2

from navsim.common.dataclasses import Trajectory, Scene, TrajectorySampling
from navsim.visualization.bev import TrajVisConfig
from navsim.visualization.bev import add_trajectory_to_bev_ax, add_map_to_bev_ax, add_annotations_to_bev_ax_mask, add_trajectory_to_bev_ax_config
from navsim.visualization.plots import configure_bev_ax
from navsim.visualization.config import TRAJECTORY_CONFIG, BEV_PLOT_CONFIG
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types

def torch_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor


def _get_colors_from_scores(scores: List[float]) -> List[str]:
    return [tuple(cm.viridis(float(v))) for v in scores]

def _get_alphas_from_scores(scores: List[float], min: float = 0.0, max: float = 1.0) -> List[float]:
    return [min + (max - min) * float(v) for v in scores]

def _to_numpy_poses(traj: Union[torch.Tensor, np.ndarray, Trajectory]) -> Optional[np.ndarray]:
    if isinstance(traj, torch.Tensor):
        poses = traj.detach().cpu().numpy()
    elif isinstance(traj, Trajectory):
        poses = traj.poses
    else:
        poses = np.array(traj)
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    if poses.shape[-1] != 3:
        return None
    return poses

BEV_CLASS_NAMES: Dict[int, str] = {
    0: "bg",
    1: "road",
    2: "walkways",
    3: "centerline",
    4: "static_obj",
    5: "veh",
    6: "pedestrian",
    7: "etc",
}

BEV_PALETTE: List[Tuple[int, int, int]] = [
    (0, 0, 0),        # 0 bg
    (90, 90, 90),     # 1 road
    (200, 200, 200),  # 2 walkways
    (255, 200, 0),    # 3 centerline
    (155, 89, 182),   # 4 static_obj
    (231, 76, 60),    # 5 veh
    (46, 204, 113),   # 6 pedestrian
    (26, 188, 156),   # 7 etc
]

def get_traj_vis_config(trajectories: List[np.ndarray],
                        anchor_scores: List[float],
                        cand_scores: List[float],
                        sel_scores: List[float],
                        default_colors: Optional[List[str]] = None,
                        radii: Optional[float] = None,
                        title: Optional[str] = None) -> TrajVisConfig:

    if default_colors is None:
        colors = _get_colors_from_scores(anchor_scores)
    else:
        colors = [default_colors]*len(anchor_scores)
    colors = colors + ["orange"]*len(cand_scores) + ["red"]*len(sel_scores)
    alphas = _get_alphas_from_scores(anchor_scores, min=0.3) + _get_alphas_from_scores(cand_scores, min=0.5) + _get_alphas_from_scores(sel_scores, min=0.5)
    widths = [0.8]*len(anchor_scores) + [1.2]*len(sel_scores) + [1.2]*len(cand_scores)
    zorder = [3]*len(anchor_scores) + [4]*len(cand_scores) + [5]*len(sel_scores)
    # Normalize radius to a single float (take first element if list/array provided)

    if radii is not None:
        radii_list = [None]*len(anchor_scores) + [None]*len(cand_scores) + radii.tolist()
    else:
        radii_list = [None]*len(anchor_scores) + [None]*len(cand_scores) + [None]*len(sel_scores)
    return TrajVisConfig(trajectories, colors, alphas, widths, zorder, radii_list, title)

def get_default_traj_vis_config(trajectories: List[np.ndarray]) -> TrajVisConfig:
    colors = [tuple(cm.viridis(float(i) / max(len(trajectories) - 1, 1))) for i in range(len(trajectories))]
    alphas = [0.3]*len(trajectories)
    widths = [0.8]*len(trajectories)
    zorder = [3]*len(trajectories)
    return TrajVisConfig(trajectories, colors, alphas, widths, zorder)

def get_traj_vis_config_common(trajectories: np.ndarray,
                               scores: np.ndarray) -> TrajVisConfig:
    colors = _get_colors_from_scores(scores)
    alphas = _get_alphas_from_scores(scores, min=0.0, max=0.5)
    widths = [0.8]*len(scores)
    zorder = [5]*len(scores)
    return TrajVisConfig(trajectories, colors, alphas, widths, zorder)

def visualize_anchor_refinement(scene: Scene,
                                stages: List[TrajVisConfig],
                                frame_idx: int = -1,
                                figsize: tuple = (30, 10),
                                score_summary: Optional[str] = None,
                                show: bool = False) -> plt.Figure:
    """
    Simple per-stage visualization with three layers per stage (no rings):
      - anchors: background anchor set (colored by score with cm.viridis, alpha 0.3 + 0.7*score, lw 0.8)
      - orange: overlay trajectories in orange (alpha = score, lw 1.0)
      - red: overlay trajectories in red (alpha = score, lw 1.0)

    Args:
        scene: NAVSIM Scene.
        stages: List of TrajVisConfig objects (one per stage) with attributes:
            - trajectories: List/array of trajectories ([K, T, 3] or list of Trajectory/ndarray/tensor)
            - colors: List/array of colors
            - alphas: List/array of alphas
            - widths: List/array of widths
            - zorder: List/array of zorder values
            - title: Optional title string
        frame_idx: frame index to visualize (default: last).
        figsize: figure size.
        score_summary: optional score summary string to display at the bottom of the figure.
        show: whether to plt.show().

    Returns:
        Matplotlib Figure.
    """
    import matplotlib.cm as cm
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.visualization.bev import add_map_to_bev_ax, add_annotations_to_bev_ax_mask

    num_stages = len(stages)
    fig, axes = plt.subplots(1, num_stages + 1, figsize=figsize)
    if num_stages == 1:
        axes = np.array([axes])

    plot_idx = scene.scene_metadata.num_history_frames - 1 if frame_idx == -1 else frame_idx
    frame = scene.frames[plot_idx]
    ego_pose = StateSE2(*frame.ego_status.ego_pose)

    for s in range(num_stages):
        ax = axes[s]
        add_map_to_bev_ax(ax, scene.map_api, ego_pose)
        add_annotations_to_bev_ax_mask(ax, frame.annotations)

        stage = stages[s]
        trajectories = stage.trajectories
        colors = stage.colors
        alphas = stage.alphas
        widths = stage.widths
        zorder = stage.zorder
        radii = stage.radii

        # Background anchors (color/alpha identical to existing behavior)
        for trajectory, color, alpha, width, zorder_val, radius in zip(trajectories, colors, alphas, widths, zorder, radii):
            poses = _to_numpy_poses(trajectory)
            if poses is None:
                continue
            trajectory_config = TRAJECTORY_CONFIG["agent"].copy()
            trajectory_config["line_color"] = color
            trajectory_config["line_width"] = width
            trajectory_config["line_color_alpha"] = alpha
            trajectory_config["zorder"] = zorder_val
            trajectory_config["marker"] = None
            add_trajectory_to_bev_ax(ax, Trajectory(poses), trajectory_config)

            if radius is not None:
                x, y = poses[-1, 1], poses[-1, 0]
                circle = Circle((x, y), radius=radius,
                                facecolor=(0.196, 0.804, 0.196, 0.6),  # limegreen with high alpha
                                edgecolor="darkgreen",
                                linewidth=1.5,
                                zorder=999)
                ax.add_patch(circle)

        configure_bev_ax(ax)
        ax.set_ylim(-2, 64)
        title = stage.title if stage.title is not None else f"Stage {s+1}"
        ax.set_title(title)

    # GT column
    ax_gt = axes[-1]
    add_map_to_bev_ax(ax_gt, scene.map_api, ego_pose)
    add_annotations_to_bev_ax_mask(ax_gt, frame.annotations)

    draw_human_trajectory(scene, ax_gt)

    configure_bev_ax(ax_gt)
    ax_gt.set_ylim(-2, 64)
    ax_gt.set_title("GT")

    if score_summary:
        plt.figtext(0.5, 0.01, score_summary, ha='center', fontsize=12)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def draw_scene(
    scene: Scene,
    frame_idx: int = -1,
    draw_gt: bool = True,
    figsize: tuple = (30, 10),
    show: bool = False,
    titles: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Draw scene with BEV map and annotations.
    """
    # Use current frame (history last) as BEV origin like in the tutorial
    plot_idx = scene.scene_metadata.num_history_frames - 1 if frame_idx == -1 else frame_idx
    frame = scene.frames[plot_idx]
    ego_pose = StateSE2(*frame.ego_status.ego_pose)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    add_map_to_bev_ax(ax, scene.map_api, ego_pose)
    add_annotations_to_bev_ax_mask(ax, frame.annotations)
    configure_bev_ax(ax)
    ax.set_ylim(-2, 64)

    if draw_gt:
        draw_human_trajectory(scene, ax)

    if titles:
        ax.set_title(titles[plot_idx])
    if show:
        plt.show()
    return fig


def draw_human_trajectory(
    scene: Scene,
    ax: plt.Axes,
    trajectory_config: Optional[Dict[str, Any]] = None,
) -> plt.Figure:
    """
    Draw human trajectory with BEV map and annotations.
    """
    # Same horizon as tutorial default, avoid hard-coded 8 unless desired
    gt_src = scene.get_future_trajectory()
    poses_gt = _to_numpy_poses(gt_src)

    if trajectory_config is None:
        trajectory_config = TRAJECTORY_CONFIG["agent"].copy()
        trajectory_config["line_color"] = "red"
        trajectory_config["line_width"] = 3
        trajectory_config["marker"] = None

    trajectory_sampling = TrajectorySampling(num_poses=len(poses_gt), interval_length=0.5)
    trajectory = Trajectory(poses_gt, trajectory_sampling)
    add_trajectory_to_bev_ax(ax, trajectory, trajectory_config)
    return ax

def _draw_circle(ax,
                 center_xy: np.ndarray,
                 radius: float,
                 color: str = 'red',
                 alpha: float = 0.35,
                 linewidth: float = 2.0,
                 zorder: int = 5,
                 label: str = 'Region') -> None:
    """
    Draw a red translucent circle on BEV axis.
    The BEV axes place y on the horizontal axis and x on the vertical axis.
    """
    circ = Circle(
        (center_xy[1], center_xy[0]),
        radius,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        zorder=zorder,
        label=label,
    )
    ax.add_patch(circ)


def visualize_agent_filtering(
    scene: Scene,
    agents: np.ndarray,
    ego_traj: np.ndarray,
    anchor_trajs: np.ndarray,
    filtering_results: List[np.ndarray],
    ref_filtering_results: List[np.ndarray],
    frame_idx: int = -1,
    figsize: tuple = (20, 20),
    center: np.ndarray = None,
    radii: np.ndarray = None,
    agent_pred_trajs: np.ndarray = None,
    agent_trajs: np.ndarray = None,
    agent_trajs_valid: np.ndarray = None,
    title: str = None,
    show: bool = False
) -> plt.Figure:
    """
    Visualize agent filtering results for 16 anchors in a 4x4 subplot grid.

    Args:
        scene: NavSim Scene object
        agents: [N, 2] array of agent positions (x, y) in global coordinates
        ego_pose: [3] array (x, y, heading) of ego vehicle
        anchor_trajs: [16, T, 3] array of anchor trajectories
        filtering_results: List of 16 boolean masks for selected agents
        ref_filtering_results: List of 16 boolean masks for selected agents
        frame_idx: Frame index for visualization
        figsize: Figure size
        center: [16, 2] array of center points for each anchor
        radii: [16, 2] array of radii for each anchor
        agent_pred_trajs: [16, T, 2] array of agent predicted trajectories
        agent_traj: [16, T, 2] array of agent trajectories
        agent_trajs_valid: [16, T] array of agent trajectories validity masks
        token: Token of the scene
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.flatten()

    # Get frame and ego_pose for BEV
    plot_idx = scene.scene_metadata.num_history_frames - 1 if frame_idx == -1 else frame_idx
    frame = scene.frames[plot_idx]
    ego_pose_se2 = StateSE2(*frame.ego_status.ego_pose)

    for idx, (anchor_traj, selected_mask, ref_selected_mask, ax) in enumerate(zip(anchor_trajs, filtering_results, ref_filtering_results, axes)):
        # Add map and annotations (annotations colored by selection)
        add_map_to_bev_ax(ax, scene.map_api, ego_pose_se2)
        # Map selection masks to index-based styling
        # style_list: 0=default, 1=rejected, 2=ref_selected
        style_list = [
            {},
            {
                "fill_color": "#B0B0B0",
                "fill_color_alpha": 1.0,
                "line_color": "black",
                "line_color_alpha": 1.0,
                "line_width": 1.0,
                "line_style": "-",
            },
            {
                "line_color": "#00FF00",
                "line_width": 1.5,
            },
        ]
        style_indices: List[int] = []
        non_ego_ptr = 0
        for name_value in frame.annotations.names:
            if tracked_object_types[name_value].name == "EGO":
                continue
            idx_val = 0
            if ref_selected_mask is not None and non_ego_ptr < len(ref_selected_mask) and bool(ref_selected_mask[non_ego_ptr]):
                idx_val = 2
            elif selected_mask is not None and non_ego_ptr < len(selected_mask) and not bool(selected_mask[non_ego_ptr]):
                idx_val = 1
            else:
                idx_val = 0
            style_indices.append(idx_val)
            non_ego_ptr += 1

        add_annotations_to_bev_ax_mask(
            ax,
            frame.annotations,
            add_ego=True,
            style_indices=np.array(style_indices, dtype=int),
            style_list=style_list,
        )

        # Configure BEV to match visualize_anchor_refinement_simple
        configure_bev_ax(ax)
        ax.set_ylim(-2, 64)
        ax.set_title(f"Anchor {idx+1}", fontsize=10, fontweight='bold')

        # Optional: draw a red translucent circle for the provided center and radius
        # Accepts either a single center/radius or per-anchor arrays
        if center[idx] is not None and radii[idx] is not None:
            for c_xy, r_val in zip(center[idx], radii[idx]):
                _draw_circle(ax, c_xy, r_val, color="#DE7061", alpha=0.05, linewidth=2.0, zorder=1, label=None)

        # Draw anchor trajectory
        ego_traj = _to_numpy_poses(ego_traj)
        if ego_traj is not None:
            ego_traj_config = TRAJECTORY_CONFIG["agent"].copy()
            ego_traj_config["line_color"] = "blue"
            ego_traj_config["line_width"] = 1.0
            ego_traj_config["line_color_alpha"] = 1.0
            ego_traj_config["marker"] = None
            ego_traj_config["zorder"] = 5
            add_trajectory_to_bev_ax(ax, Trajectory(ego_traj), ego_traj_config)

        anchor_traj = _to_numpy_poses(anchor_traj)
        if anchor_traj is not None:
            # Main trajectory line
            ego_traj_config = TRAJECTORY_CONFIG["agent"].copy()
            ego_traj_config["line_color"] = "red"
            ego_traj_config["line_width"] = 2.5
            ego_traj_config["line_color_alpha"] = 1.0
            ego_traj_config["marker"] = None
            ego_traj_config["zorder"] = 6
            add_trajectory_to_bev_ax(ax, Trajectory(anchor_traj), ego_traj_config)

        # Draw agent trajectories
        if agent_trajs[idx] is not None:
            agent_traj_config = TRAJECTORY_CONFIG["agent"].copy()
            agent_traj_config["line_width"] = 1.0
            agent_traj_config["line_color_alpha"] = 1.0
            agent_traj_config["marker"] = None
            start_poses = frame.annotations.boxes[:, :2]
            agent_traj = agent_trajs[idx]
            agent_traj_valid = agent_trajs_valid[idx]
            for poses, start_pose, is_selected, traj_valid in zip(agent_traj, start_poses, selected_mask, agent_traj_valid):
                if is_selected:
                    agent_traj_config["line_color"] = "blue"
                    agent_traj_config["zorder"] = 5
                else:
                    agent_traj_config["line_color"] = "grey"
                    agent_traj_config["zorder"] = 4
                add_trajectory_to_bev_ax(ax, Trajectory(poses), agent_traj_config, start_pose, traj_valid)

        if agent_pred_trajs[idx] is not None:
            agent_traj_config = TRAJECTORY_CONFIG["agent"].copy()
            agent_traj_config["line_width"] = 1.0
            agent_traj_config["line_color_alpha"] = 1.0
            agent_traj_config["line_color"] = "#C00000"
            agent_traj_config["marker"] = None
            agent_traj_config["zorder"] = 3.5
            start_poses = frame.annotations.boxes[:, :2]
            agent_traj = agent_pred_trajs[idx]
            for poses, start_pose, is_selected in zip(agent_traj, start_poses, selected_mask):
                poses = np.concatenate([poses, np.zeros((poses.shape[0], 1))], axis=1)
                add_trajectory_to_bev_ax(ax, Trajectory(poses), agent_traj_config, start_pose)

        # We now color boxes directly; optional extra scatter disabled

        # Add legend only if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if any(lbl and not lbl.startswith('_') for lbl in labels):
            ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

        # Add count info (only agents within current BEV view)
        try:
            xlim = ax.get_xlim()  # corresponds to y (left-right)
            ylim = ax.get_ylim()  # corresponds to x (front-back)
            if len(agents) > 0 and len(selected_mask) == len(agents):
                in_view = (
                    (agents[:, 1] >= min(xlim)) & (agents[:, 1] <= max(xlim)) &
                    (agents[:, 0] >= min(ylim)) & (agents[:, 0] <= max(ylim))
                )
                sel_in_view = int(np.sum(selected_mask & in_view))
                rej_in_view = int(np.sum((~selected_mask) & in_view))
            else:
                sel_in_view = 0
                rej_in_view = 0
        except Exception:
            sel_in_view = 0
            rej_in_view = 0

        info_text = f"Selected: {sel_in_view}, Rejected: {rej_in_view} (in view)"
        ax.text(
            0.02, 0.98, info_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.suptitle(
        title,
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()

    if show:
        plt.show()

    return fig


def visualize_nc(
    scene: Scene,

    # ground truth
    human_traj: np.ndarray,
    agent_states: np.ndarray,
    agent_trajs: np.ndarray,
    agent_trajs_valid: np.ndarray,

    # prediction
    pred_traj: np.ndarray,
    pred_agent_states: np.ndarray,
    pred_agent_names: List[str],
    pred_agent_trajs: np.ndarray,
    pred_sel_agent_states: np.ndarray | None = None,

    # visualization
    figsize: tuple = (10, 10),
    title: str = None,
    show: bool = False,
    save_path: str = None
) -> plt.Figure:
    """
    Visualize navigation trajectories on a single BEV plot.

    Args:
        scene: NavSim Scene object

        human_traj: [T, 3] array of human trajectory
        agent_states: [N] array of agent states
        agent_trajs: [N, T, 3] array of agent trajectories (x, y, heading)
        agent_trajs_valid: [N, T] boolean array indicating valid agent trajectory points

        pred_traj: [T, 3] array of prediction trajectory
        pred_agent_states: [N] array of prediction agent states
        pred_agent_trajs: [N, T, 3] array of agent trajectories (x, y, heading)

        figsize: Figure size (default: (10, 10))
        title: Optional title for the figure
        show: Whether to display the plot
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()  # get current axes

    # Get frame and ego pose for BEV
    plot_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[plot_idx]
    ego_pose_se2 = StateSE2(*frame.ego_status.ego_pose)

    # Check if ego_traj has multiple trajectories
    pred_trajs_list = []
    if pred_traj.ndim == 3:
        # Multiple ego trajectories [K, T, 3]
        for i in range(pred_traj.shape[0]):
            pred_trajs_list.append(pred_traj[i])
    else:
        # Single ego trajectory [T, 3]
        pred_trajs_list.append(pred_traj)

    # Insert pred_agent_states to frame.annotations
    cur_annos = copy.deepcopy(frame.annotations)
    num_agents, box_dim = cur_annos.boxes.shape

    assert len(pred_agent_names) == len(pred_agent_states)
    cur_annos.names = np.concatenate([cur_annos.names, pred_agent_names], axis=0)
    cur_annos.boxes = np.concatenate([cur_annos.boxes, pred_agent_states[:,:box_dim]], axis=0)

    pred_mask = np.concatenate([np.zeros(num_agents), np.ones(len(pred_agent_states))], axis=0)
    frame.annotations = cur_annos

    # Add map and annotations
    add_map_to_bev_ax(ax, scene.map_api, ego_pose_se2)
    # Index-based styling: 0=default(gt), 1=pred
    style_list_boxes = [
        {},
        {
            "fill_color_alpha": 0.0,
            "line_color": "red",
            "line_width": 1.0,
        },
    ]
    style_indices_boxes: List[int] = []
    non_ego_ptr = 0
    for i, name_value in enumerate(cur_annos.names):
        if tracked_object_types[name_value].name == "EGO":
            continue
        is_pred = bool(pred_mask[i]) if i < len(pred_mask) else False
        style_indices_boxes.append(1 if is_pred else 0)
        non_ego_ptr += 1

    add_annotations_to_bev_ax_mask(
        ax,
        cur_annos,
        add_ego=True,
        style_indices=np.array(style_indices_boxes, dtype=int),
        style_list=style_list_boxes,
    )

    # Configure BEV axis
    configure_bev_ax(ax)
    ax.set_ylim(-2, 64)

    # Draw ground truth trajectory (blue)
    gt_traj_np = _to_numpy_poses(human_traj)
    if gt_traj_np is not None:
        gt_config = TRAJECTORY_CONFIG["agent"].copy()
        gt_config["line_color"] = "blue"
        gt_config["line_width"] = 2.0
        gt_config["line_color_alpha"] = 1.0
        gt_config["marker"] = "o"
        gt_config["marker_size"] = 3
        gt_config["zorder"] = 5
        add_trajectory_to_bev_ax(ax, Trajectory(gt_traj_np), gt_config)

    # Draw ego trajectory/trajectories (green)
    for pred_traj_single in pred_trajs_list:
        pred_traj_np = _to_numpy_poses(pred_traj_single)
        if pred_traj_np is not None:
            pred_traj_config = TRAJECTORY_CONFIG["agent"].copy()
            pred_traj_config["line_color"] = "red"
            pred_traj_config["line_width"] = 2.0
            pred_traj_config["line_color_alpha"] = 1.0
            pred_traj_config["marker"] = "o"
            pred_traj_config["marker_size"] = 3
            pred_traj_config["zorder"] = 6
            add_trajectory_to_bev_ax(ax, Trajectory(pred_traj_np), pred_traj_config)

    # Draw agent trajectories (blue with transparency)
    if agent_trajs is not None and len(agent_trajs) > 0:
        agent_traj_config = TRAJECTORY_CONFIG["agent"].copy()
        agent_traj_config["line_color"] = "blue"
        agent_traj_config["line_width"] = 0.8
        agent_traj_config["line_color_alpha"] = 1.0
        agent_traj_config["marker"] = "o"
        agent_traj_config["marker_size"] = 3
        agent_traj_config["zorder"] = 3

        # Get start poses from current agent positions
        start_poses = agent_states[:, :2]

        for agent_idx, (poses, start_pose) in enumerate(zip(agent_trajs, start_poses)):
            traj_valid = agent_trajs_valid[agent_idx] if agent_trajs_valid is not None else None
            # Only plot if there are valid points
            if traj_valid is None or np.any(traj_valid):
                add_trajectory_to_bev_ax(ax, Trajectory(poses), agent_traj_config, start_pose, traj_valid)

    if pred_agent_trajs is not None and len(pred_agent_trajs) > 0:
        pred_agent_traj_config = TRAJECTORY_CONFIG["agent"].copy()
        pred_agent_traj_config["line_color"] = "red"
        pred_agent_traj_config["line_width"] = 0.8
        pred_agent_traj_config["line_color_alpha"] = 1.0
        pred_agent_traj_config["marker"] = "o"
        pred_agent_traj_config["marker_size"] = 3
        pred_agent_traj_config["zorder"] = 3

        # Get start poses from current agent positions
        start_poses = pred_agent_states[:, :2]

        for poses, start_pose in zip(pred_agent_trajs, start_poses):
            add_trajectory_to_bev_ax(ax, Trajectory(poses), pred_agent_traj_config, start_pose)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=0.8, marker='o', markersize=5, label='prediction'),
        Line2D([0], [0], color='blue', linewidth=0.8, marker='o', markersize=5, label='ground truth'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

    # Add title if provided
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_bev_semantic_map(
    bev_seg_map_pred: np.ndarray=None,
    bev_seg_map_gt: np.ndarray=None,

    class_idx: Optional[int] = None,
    num_classes: Optional[int] = None,

    figsize: tuple = (10, 10),
    title: str = None,
    show: bool = False,
    save_path: str = None,

    fig: plt.Figure = None,
    ax: plt.Axes = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize BEV semantic map.

    Args:
        bev_seg_map_pred: Prediction BEV semantic map in sigmoid format (C, H, W). Default: None.
        bev_seg_map_gt: Ground truth BEV semantic map in one-hot format (C, H, W). Default: None.
        class_idx: If provided, only visualize pixels belonging to this class. Default: None (show all classes).
        num_classes: Number of classes. Default: None (auto-detect from shape).
    """
    # Ensure at least one map is provided
    if bev_seg_map_pred is None and bev_seg_map_gt is None:
        raise ValueError("At least one of bev_seg_map_pred or bev_seg_map_gt must be provided")

    # Determine num_classes
    if num_classes is None:
        if bev_seg_map_pred is not None:
            num_classes = bev_seg_map_pred.shape[0]
        elif bev_seg_map_gt is not None:
            num_classes = bev_seg_map_gt.shape[0]

    # Helper function to convert one-hot to label map (for GT)
    def onehot_to_label(onehot_map: np.ndarray) -> np.ndarray:
        """Convert (C, H, W) one-hot to (H, W) label map."""
        if onehot_map.ndim != 3:
            raise ValueError(f"Expected 3D one-hot map (C, H, W), got shape {onehot_map.shape}")
        return np.argmax(onehot_map, axis=0).astype(np.int32)

    # Helper function to visualize sigmoid map with color intensity based on sigmoid values (heatmap style)

    def sigmoid_to_colored(sigmoid_map: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Convert sigmoid map (C, H, W) to red-blue heatmap-style colored image (H, W, 3).
        High = red, Low = blue.
        """
        if sigmoid_map.ndim != 3:
            raise ValueError(f"Expected 3D sigmoid map (C, H, W), got shape {sigmoid_map.shape}")

        C, H, W = sigmoid_map.shape
        rgb_img = np.zeros((H, W, 3), dtype=np.float32)

        # Non-linear intensity mapping (optional for better visibility)
        def intensity_map(prob: np.ndarray) -> np.ndarray:
            return np.power(np.clip(prob, 0, 1), 0.7)

        cmap = cm.get_cmap('bwr')  # blue–white–red colormap

        if class_idx is not None:
            if class_idx < 0 or class_idx >= C:
                raise ValueError(f"class_idx {class_idx} out of range [0, {C-1}]")
            prob = intensity_map(sigmoid_map[class_idx])  # (H, W)
            rgb_img = cmap(prob)[..., :3]  # RGBA → RGB
        else:
            # Use max probability across channels for heatmap
            max_prob = intensity_map(np.max(sigmoid_map, axis=0))
            rgb_img = cmap(max_prob)[..., :3]

        return (rgb_img * 255).astype(np.uint8)

    # Get palette
    palette = _ensure_palette(num_classes, base_palette=BEV_PALETTE)

    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = ax.figure

    # Visualization: GT only, Pred only, or both
    if bev_seg_map_gt is not None and bev_seg_map_pred is not None:
        # Both provided: GT as base, pred overlaid
        # GT is one-hot, convert to label map first
        if bev_seg_map_gt.ndim == 3:
            label_map_gt = onehot_to_label(bev_seg_map_gt)
        else:
            label_map_gt = bev_seg_map_gt
        if class_idx is not None:
            label_map_gt = np.where(label_map_gt == class_idx, class_idx, 0)
        color_img_gt = _colorize(label_map_gt, palette)

        # Pred is sigmoid, use sigmoid values directly (heatmap style)
        color_img_pred = sigmoid_to_colored(bev_seg_map_pred, class_idx)

        # Draw GT as base layer with reduced opacity to make pred more visible
        ax.imshow(color_img_gt, alpha=0.5)
        # Overlay pred with higher opacity for better visibility
        ax.imshow(color_img_pred, alpha=0.3)
    elif bev_seg_map_gt is not None:
        # GT only (one-hot)
        if bev_seg_map_gt.ndim == 3:
            label_map_gt = onehot_to_label(bev_seg_map_gt)
        else:
            label_map_gt = bev_seg_map_gt
        if class_idx is not None:
            label_map_gt = np.where(label_map_gt == class_idx, class_idx, 0)
        color_img_gt = _colorize(label_map_gt, palette)
        ax.imshow(color_img_gt, alpha=1.0)
    elif bev_seg_map_pred is not None:
        # Pred only (sigmoid)
        color_img_pred = sigmoid_to_colored(bev_seg_map_pred, class_idx)
        ax.imshow(color_img_pred, alpha=1.0)

    if title:
        ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    return fig, ax


def visualize_scene_grouped(
    scene: Scene,

    traj_list: List[np.ndarray],
    traj_category_list: List[str],
    traj_valid_list: List[np.ndarray],

    bbox_list: List[np.ndarray],
    bbox_names_list: List[np.ndarray],
    bbox_style_list: List[str],
    bbox_category_list: List[str],

    points_list: List[np.ndarray]|None = None,

    figsize: tuple = (10, 10),
    title: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize a scene on BEV (bird's-eye view), grouping trajectories and
    bounding boxes by their categories for styling and legend entries.

    Coordinate and unit conventions
    - All positions are in meters in the ego-centric frame at the current frame.
    - x points forward, y points left/right (left is positive).
    - Trajectory poses are expected to contain at least 2 columns (x, y). Extra
      columns (e.g., yaw) are ignored for rendering.

    Parameters
    - scene: NavSim `Scene` object used to fetch the current frame, ego pose,
      and map for background rendering.
    - traj_list: List of trajectories to draw. Each element can be one of:
      - (T, D) array: single trajectory with T points, D >= 2 (uses [:, :2]).
      - (N, T, D) array: N trajectories, each length T, D >= 2.
      Typical usage:
        - ego_gt / ego_pred: (T, 3) with columns [x, y, yaw] or (T, 2)
        - agent_gt / agent_pred: (A, T, 3) for A agents (or (A, T, 2))
    - traj_category_list: List of category strings aligned with `traj_list`.
      Recognized categories (controls colors/linestyles):
        - "ego_gt", "ego_pred", "agent_gt", "agent_pred"
      Unknown categories fall back to a default style.
    - traj_valid_list: List of validity masks aligned with `traj_list`.
      - For (T, D) trajectories: (T,) boolean or None.
      - For (N, T, D) trajectories: (N, T) boolean or None.
      If None, all points are considered valid.
    - bbox_list: List of 2D bounding box arrays, one per group. Each array is
      shaped (M_i, 6) in the format [x, y, sin_yaw, cos_yaw, w, h], in meters.
      Groups are concatenated for rendering and styled by `bbox_category`.
    - bbox_names_list: List of names arrays corresponding to `bbox_list`. Each
      entry is a sequence of strings length M_i. Names are used for filtering
      EGO (if present) and for legend semantics.
    - bbox_category: List of category strings aligned with `bbox_list` that map
      to styles. Recognized values:
        - "gt" (default style), "pred" (red outline), "selected" (green fill)
      Unknown values default to "gt" styling.
    - figsize: Matplotlib figure size, e.g., (10, 10).
    - title: Optional plot title.
    - show: If True, calls plt.show() at the end.
    - save_path: If provided, saves the figure to this path (directories are not
      auto-created here; ensure the parent exists).

    Returns
    - Matplotlib Figure object containing the rendered BEV visualization.

    Notes
    - Trajectory and box groups are rendered on the same axes after the map is
      drawn using the ego pose at the latest history frame.
    - Legends are generated from unique trajectory and bbox categories.
    """

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Current frame and origin
    plot_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[plot_idx]
    ego_pose_se2 = StateSE2(*frame.ego_status.ego_pose)

    # Map
    add_map_to_bev_ax(ax, scene.map_api, ego_pose_se2)

    # Concatenate bbox groups into single Annotations
    all_boxes: List[np.ndarray] = []
    all_names: List[str] = []
    for boxes, names in zip(bbox_list or [], bbox_names_list or []):
        if boxes is None or len(boxes) == 0:
            continue
        all_boxes.append(boxes)
        all_names.extend([str(n) for n in names])

    if len(all_boxes) > 0:
        boxes_cat = np.concatenate(all_boxes, axis=0).astype(np.float32)
        N = boxes_cat.shape[0]
        # placeholders to satisfy Annotations length invariants
        velocity_3d = np.zeros((N, 3), dtype=np.float32)
        instance_tokens = [""] * N
        track_tokens = [""] * N
        annos = type(frame.annotations)(
            boxes=boxes_cat,
            names=all_names,
            velocity_3d=velocity_3d,
            instance_tokens=instance_tokens,
            track_tokens=track_tokens,
        )

        # Build style_list for bboxes with defaults
        bbox_style_defaults: List[Dict[str, Any]] = [
            {},
            {"fill_color_alpha": 0.0, "line_color": "red", "line_width": 1.0},
            {"fill_color_alpha": 0.4, "fill_color": "#32CD32", "line_color": "#228B22", "line_width": 1.2},
            {"fill_color_alpha": 0.0, "line_color": "#32CD32", "line_width": 1.2},
            {"fill_color_alpha": 0.0, "line_color": "#0000FF", "line_width": 1.2},
        ]
        # Map categories to indices
        cat_to_idx = {"default": 0, "red_line": 1, "green_fill": 2, "green_line": 3, "blue_line": 4}

        # Build style_indices aligned with non-ego order
        style_indices: List[int] = []
        # Expand per-group categories to per-box (non-ego)
        group_ptr = 0
        for boxes, names, cat in zip(bbox_list or [], bbox_names_list or [], bbox_style_list or []):
            if boxes is None or len(boxes) == 0:
                continue
            idx_val = cat_to_idx.get(cat, 0)
            for name_value in names:
                if tracked_object_types[name_value].name == "EGO":
                    continue
                style_indices.append(idx_val)
            group_ptr += 1

        add_annotations_to_bev_ax_mask(
            ax,
            annos,
            add_ego=True,
            style_indices=np.array(style_indices, dtype=int) if len(style_indices) > 0 else None,
            style_list=bbox_style_defaults,
        )

    # Trajectory styles by category
    traj_defaults = {
        "ego_gt": {"line_color": "blue", "line_width": 1.5, "zorder": 6, "marker": "o", "marker_size": 3},
        "ego_pred": {"line_color": "red", "line_width": 1.5, "zorder": 7, "marker": "o", "marker_size": 3},
        "agent_gt": {"line_color": "blue", "line_width": 1.0, "zorder": 5, "marker": "o", "marker_size": 3},
        "agent_pred": {"line_color": "red", "line_width": 1.0, "zorder": 4, "marker": "o", "marker_size": 3},
    }

    drawn_traj_cats: List[str] = []
    for poses, cat, valid in zip(traj_list, traj_category_list, traj_valid_list):
        if poses is None or len(poses) == 0:
            continue
        style_base = TRAJECTORY_CONFIG["agent"].copy()
        style_base.update(traj_defaults.get(cat, {}))

        if poses.ndim == 3:
            for i in range(poses.shape[0]):
                v_i = valid[i].astype(bool) if (valid is not None) else None
                add_trajectory_to_bev_ax(ax, poses[i], style_base, traj_valid=v_i)
        else:
            add_trajectory_to_bev_ax(ax, poses, style_base, traj_valid=valid)
        drawn_traj_cats.append(cat)

    # Configure BEV
    configure_bev_ax(ax)
    ax.set_ylim(-2, 64)

    if points_list is not None:
        for points in points_list:
            ax.scatter(points[:, 0], points[:, 1], color="red", marker="o", s=5, zorder=10)

    # Legend: unique categories
    legend_handles: List[Line2D] = []
    added_cats = set()
    for cat in traj_category_list or []:
        if cat in added_cats:
            continue
        if cat in traj_defaults:
            cfg = traj_defaults[cat]
            legend_handles.append(Line2D([0], [0], color=cfg.get("line_color", "black"), linewidth=cfg.get("line_width", 1.0), marker=cfg.get("marker", None), markersize=5, label=f"traj: {cat}"))
            added_cats.add(cat)

    # BBox categories
    if bbox_category_list is not None:
        for cat in bbox_category_list or []:
            if f"box: {cat}" in added_cats:
                continue
            # representative color from style list
            idx = bbox_category_list.index(cat)
            box_style = [{}, {"line_color": "red"}, {"line_color": "#228B22"}, {"line_color": "#32CD32"}][idx]
            legend_handles.append(Line2D([0], [0], color=box_style.get("line_color", "black"), linewidth=1.5, label=f"box: {cat}"))
            added_cats.add(f"box: {cat}")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def visualize_scene(
    scene: Scene,
    trajectories: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Dict[str, Any]]] = None,
    figsize: tuple = (10, 10),
    title: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    grid: bool = False,
    axis_labels: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize a scene on BEV (bird's-eye view) with an intuitive API.

    Coordinate and unit conventions
    - All positions are in meters in the ego-centric frame at the current frame.
    - x points forward, y points left/right (left is positive).
    - Trajectory poses are expected to contain at least 2 columns (x, y). Extra
      columns (e.g., yaw) are ignored for rendering.

    Parameters
    ----------
    scene : Scene
        NavSim `Scene` object used to fetch the current frame, ego pose,
        and map for background rendering.
    trajectories : Optional[List[Dict[str, Any]]], optional
        List of trajectory dictionaries. Each dict contains:
        - `data` (required): trajectory array. Can be:
          - (T, D) array: single trajectory with T points, D >= 2 (uses [:, :2])
          - (N, T, D) array: N trajectories, each length T, D >= 2
        - `style` (optional, default="default"): style string. Options:
          - "blue", "red", "green", etc.: predefined colors
          - "score-adaptive": color based on scores (requires `score` parameter)
        - `label` (optional): label string for legend
        - `valid` (optional): validity mask array
          - For (T, D) trajectories: (T,) boolean or None
          - For (N, T, D) trajectories: (N, T) boolean or None
        - `score` (optional): score array for score-adaptive coloring
          - For (T, D) trajectories: (T,) float array
          - For (N, T, D) trajectories: (N,) float array
    bboxes : Optional[List[Dict[str, Any]]], optional
        List of bbox dictionaries. Each dict contains:
        - `boxes` (required): 2D bounding box array shaped (M, 6) in format
          [x, y, sin_yaw, cos_yaw, w, h], in meters
        - `names` (required): names array of length M (strings)
        - `style` (optional, default="default"): style string. Options:
          - "default": default style
          - "red_line": red outline
          - "green_fill": green fill with outline
          - "green_line": green outline
          - "blue_line": blue outline
        - `label` (optional): label string for legend
    figsize : tuple, optional
        Matplotlib figure size, e.g., (10, 10). Default is (10, 10).
    title : Optional[str], optional
        Optional plot title.
    show : bool, optional
        If True, calls plt.show() at the end. Default is False.
    save_path : Optional[str], optional
        If provided, saves the figure to this path (directories are not
        auto-created here; ensure the parent exists).
    fig : Optional[plt.Figure], optional
        If provided, uses this figure instead of creating a new one.
    ax : Optional[plt.Axes], optional
        If provided, uses this axes instead of creating a new one.
    grid : bool, optional
        If True, displays grid on the plot. Default is False.
    axis_labels : bool, optional
        If True, displays axis labels (x and y labels) on the plot. Default is False.

    Returns
    -------
    plt.Figure
        Matplotlib Figure object containing the rendered BEV visualization.

    Examples
    --------
    >>> visualize_scene(
    ...     scene=scene,
    ...     trajectories=[
    ...         {"data": ego_traj, "style": "blue", "label": "Ego GT"},
    ...         {"data": pred_traj, "style": "score-adaptive", "score": scores, "label": "Predicted"},
    ...         {"data": gt_traj, "style": "red", "valid": valid_mask, "label": "GT"}
    ...     ],
    ...     bboxes=[
    ...         {"boxes": gt_boxes, "names": gt_names, "style": "default", "label": "GT Boxes"},
    ...         {"boxes": pred_boxes, "names": pred_names, "style": "red_line", "label": "Pred Boxes"}
    ...     ]
    ... )
    """
    # Style dictionaries - add new styles here
    traj_style_map = {
        "blue": {"line_color": "blue", "line_width": 1.5, "zorder": 6, "marker": "o", "marker_size": 3},
        "red": {"line_color": "red", "line_width": 1.5, "zorder": 7, "marker": "o", "marker_size": 3},
        "green": {"line_color": "green", "line_width": 1.5, "zorder": 6, "marker": "o", "marker_size": 3},
        "default": {"line_color": "black", "line_width": 1.0, "zorder": 5, "marker": "o", "marker_size": 3},
    }

    bbox_style_defaults: List[Dict[str, Any]] = [
        {},  # default
        {"fill_color_alpha": 0.0, "line_color": "red", "line_width": 1.0},  # red_line
        {"fill_color_alpha": 0.4, "fill_color": "#32CD32", "line_color": "#228B22", "line_width": 1.2},  # green_fill
        {"fill_color_alpha": 0.0, "line_color": "#32CD32", "line_width": 1.2},  # green_line
        {"fill_color_alpha": 0.0, "line_color": "#0000FF", "line_width": 1.2},  # blue_line
    ]
    bbox_style_to_idx = {"default": 0, "red_line": 1, "green_fill": 2, "green_line": 3, "blue_line": 4}

    # Handle fig and ax for subplot compatibility
    if ax is not None:

        if fig is None:
            fig = ax.figure
    elif fig is not None:

        if len(fig.axes) > 0:
            ax = fig.axes[0]
        else:
            ax = fig.add_subplot(1, 1, 1)
    else:

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

    # Current frame and origin
    plot_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[plot_idx]
    ego_pose_se2 = StateSE2(*frame.ego_status.ego_pose)

    # Map
    add_map_to_bev_ax(ax, scene.map_api, ego_pose_se2)

    # Process trajectories
    legend_handles: List[Line2D] = []
    added_labels = set()

    if trajectories is not None:
        for traj_dict in trajectories:
            if traj_dict is None or "data" not in traj_dict:
                continue

            data = traj_dict["data"]
            if data is None or len(data) == 0:
                continue

            style = traj_dict.get("style", "default")
            label = traj_dict.get("label")
            valid = traj_dict.get("valid")
            score = traj_dict.get("score")

            # Handle score-adaptive style
            if style == "score-adaptive":
                if score is None:
                    raise ValueError("score-adaptive style requires 'score' parameter")

                # Convert to numpy if needed
                data_np = np.asarray(data)
                score_np = np.asarray(score)

                if data_np.ndim == 3:
                    # (N, T, D) trajectories
                    N = data_np.shape[0]
                    if score_np.shape[0] != N:
                        raise ValueError(f"score shape {score_np.shape} doesn't match trajectory shape {data_np.shape}")

                    # Get colors from scores
                    colors = _get_colors_from_scores(score_np.tolist())
                    alphas = _get_alphas_from_scores(score_np.tolist(), min=0.0, max=0.5)

                    for i in range(N):
                        v_i = valid[i].astype(bool) if (valid is not None and valid.ndim == 2) else None
                        style_base = TRAJECTORY_CONFIG["agent"].copy()
                        style_base.update({
                            "line_color": colors[i],
                            "line_width": 1.0,
                            "zorder": 5 + score_np[i],
                            "marker": "o",
                            "marker_size": 3,
                        })
                        add_trajectory_to_bev_ax(ax, data_np[i], style_base, traj_valid=v_i)
                else:
                    # (T, D) single trajectory
                    if score_np.ndim != 1 or score_np.shape[0] != data_np.shape[0]:
                        raise ValueError(f"score shape {score_np.shape} doesn't match trajectory shape {data_np.shape}")

                    # Use get_traj_vis_config_common for single trajectory
                    # But we need to handle it differently - score is per point, not per trajectory
                    # For now, use average score for color
                    avg_score = float(np.mean(score_np))
                    color = _get_colors_from_scores([avg_score])[0]
                    style_base = TRAJECTORY_CONFIG["agent"].copy()
                    style_base.update({
                        "line_color": color,
                        "line_width": 1.0,
                        "zorder": 5,
                        "marker": "o",
                        "marker_size": 3,
                    })
                    add_trajectory_to_bev_ax(ax, data_np, style_base, traj_valid=valid)

                # Add to legend if label provided
                if label and label not in added_labels:
                    # Use a representative color (middle of viridis)
                    legend_handles.append(Line2D([0], [0], color=cm.viridis(0.5), linewidth=1.0, marker="o", markersize=5, label=label))
                    added_labels.add(label)
            else:
                # Regular style
                style_base = TRAJECTORY_CONFIG["agent"].copy()
                style_base.update(traj_style_map.get(style, traj_style_map["default"]))

                data_np = np.asarray(data)
                if data_np.ndim == 3:
                    for i in range(data_np.shape[0]):
                        v_i = valid[i].astype(bool) if (valid is not None) else None
                        add_trajectory_to_bev_ax(ax, data_np[i], style_base, traj_valid=v_i)
                else:
                    add_trajectory_to_bev_ax(ax, data_np, style_base, traj_valid=valid)

                # Add to legend if label provided
                if label and label not in added_labels:
                    cfg = traj_style_map.get(style, traj_style_map["default"])
                    legend_handles.append(Line2D([0], [0], color=cfg.get("line_color", "black"), linewidth=cfg.get("line_width", 1.0), marker=cfg.get("marker", None), markersize=5, label=label))
                    added_labels.add(label)

    # Process bboxes
    if bboxes is not None:
        all_boxes: List[np.ndarray] = []
        all_names: List[str] = []
        bbox_style_list: List[str] = []
        bbox_label_list: List[str] = []

        for bbox_dict in bboxes:
            if bbox_dict is None or "boxes" not in bbox_dict or "names" not in bbox_dict:
                continue

            boxes = bbox_dict["boxes"]
            names = bbox_dict["names"]
            if boxes is None or len(boxes) == 0:
                continue

            all_boxes.append(boxes)
            all_names.extend([str(n) for n in names])

            style = bbox_dict.get("style", "default")
            label = bbox_dict.get("label")

            # Store style and label for each box
            for _ in names:
                bbox_style_list.append(style)
                if label:
                    bbox_label_list.append(label)
                else:
                    bbox_label_list.append(None)

        if len(all_boxes) > 0:
            boxes_cat = np.concatenate(all_boxes, axis=0).astype(np.float32)
            N = boxes_cat.shape[0]
            # placeholders to satisfy Annotations length invariants
            velocity_3d = np.zeros((N, 3), dtype=np.float32)
            instance_tokens = [""] * N
            track_tokens = [""] * N
            annos = type(frame.annotations)(
                boxes=boxes_cat,
                names=all_names,
                velocity_3d=velocity_3d,
                instance_tokens=instance_tokens,
                track_tokens=track_tokens,
            )

            # Build style_indices aligned with non-ego order
            style_indices: List[int] = []
            for i, (name_value, style) in enumerate(zip(all_names, bbox_style_list)):
                if tracked_object_types[name_value].name == "EGO":
                    continue
                idx_val = bbox_style_to_idx.get(style, 0)
                style_indices.append(idx_val)

            add_annotations_to_bev_ax_mask(
                ax,
                annos,
                add_ego=True,
                style_indices=np.array(style_indices, dtype=int) if len(style_indices) > 0 else None,
                style_list=bbox_style_defaults,
            )

            # Add bbox labels to legend
            seen_bbox_labels = set()
            for style, label in zip(bbox_style_list, bbox_label_list):
                if label and label not in added_labels and label not in seen_bbox_labels:
                    idx_val = bbox_style_to_idx.get(style, 0)
                    box_style = bbox_style_defaults[idx_val]
                    legend_handles.append(Line2D([0], [0], color=box_style.get("line_color", "black"), linewidth=1.5, label=label))
                    added_labels.add(label)
                    seen_bbox_labels.add(label)
    else:
        annos = scene.frames[plot_idx].annotations
        add_annotations_to_bev_ax_mask(
            ax,
            annos,
            add_ego=True,
            style_indices=None,
            style_list=None,
        )

    # Configure BEV
    configure_bev_ax(ax)
    ax.set_ylim(-2, 64)

    # Grid
    if grid:
        ax.grid(True, alpha=0.3)

    # Axis labels
    if axis_labels:
        ax.set_xlabel("y (m)", fontsize=10)
        ax.set_ylabel("x (m)", fontsize=10)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Legend
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def add_trajectory_with_scores(
    ax: plt.Axes,
    trajectories: np.ndarray,
    scores: np.ndarray,
    meter_to_pixel: Optional[Callable] = None,
) -> plt.Axes:
    """Add trajectory with score to BEV using LineCollection for performance."""
    traj_vis_config = get_traj_vis_config_common(trajectories, scores)

    # Prepare all line segments for LineCollection
    segments = []
    colors = []
    linewidths = []

    for trajectory, color, alpha, width, zorder in zip(
        traj_vis_config.trajectories,
        traj_vis_config.colors,
        traj_vis_config.alphas,
        traj_vis_config.widths,
        traj_vis_config.zorder
    ):
        # Extract xy coordinates (swap x,y for BEV plotting convention)
        if isinstance(trajectory, Trajectory):
            poses = trajectory.poses[:, :2]
        else:
            trajectory_np = np.asarray(trajectory)
            if trajectory_np.ndim == 2 and trajectory_np.shape[1] >= 2:
                poses = trajectory_np[:, :2]
            else:
                poses = trajectory_np

        if meter_to_pixel is not None:
            poses = meter_to_pixel(poses[:, 0], poses[:, 1])

        # Convert to segments: [(y,x) points swapped for BEV]
        points = np.column_stack([poses[:, 1], poses[:, 0]])
        segments.append(points)

        # Apply alpha to color (convert to RGBA)
        if isinstance(color, str):
            import matplotlib.colors as mcolors
            rgba = mcolors.to_rgba(color, alpha=alpha)
        else:
            # color is already a tuple (r,g,b,a) from viridis
            rgba = (*color[:3], alpha)
        colors.append(rgba)
        linewidths.append(width)

    # Create LineCollection for efficient rendering
    lc = LineCollection(segments, colors=colors, linewidths=linewidths, zorder=int(np.mean(traj_vis_config.zorder)))
    ax.add_collection(lc)

    return ax


def _ensure_palette(num_classes: int,
                    base_palette: Optional[List[Tuple[int,int,int]]] = None,
                    seed: int = 2025) -> np.ndarray:
    """Palette of (C, 3) uint8 for num_classes, extended at random if too short."""
    if base_palette is None:
        base_palette = BEV_PALETTE
    palette = list(base_palette)
    if len(palette) < num_classes:
        rng = np.random.default_rng(seed)
        need = num_classes - len(palette)
        extra = rng.integers(0, 256, size=(need, 3), dtype=np.uint8)
        palette.extend([tuple(map(int, c)) for c in extra])
    return np.array(palette[:num_classes], dtype=np.uint8)

def _colorize(label_map: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """(H,W) int -> (H,W,3) uint8"""
    label_map = np.clip(label_map, 0, len(palette)-1)
    return palette[label_map]

def add_bev_segmap(
    ax: plt.Axes,
    bev_semantic_map: torch.Tensor,
    num_classes: int = 8,
    class_names: Optional[Dict[int, str]] = None,
    base_palette: Optional[List[Tuple[int,int,int]]] = None,
) -> plt.Axes:
    """
    Add BEV segmap to BEV.
    :param ax: matplotlib ax object
    :param bev_semantic_map: BEV semantic map (H,W)
    :param num_classes: number of classes
    :param class_names: class names
    :param base_palette: base palette
    """
    assert bev_semantic_map.ndim == 2, f"bev_semantic_map must be (H,W), got {tuple(bev_semantic_map.shape)}"

    # convert to labels
    x = bev_semantic_map
    if not torch.is_floating_point(x):
        labels = x.to(torch.long)
    else:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        labels = x.round().clamp_min(0).to(torch.long)

    # check class range
    max_label = int(labels.max().item()) if labels.numel() > 0 else 0
    if max_label >= num_classes:
        raise ValueError(f"num_classes={num_classes} is less than max_label={max_label}")

    # palette / class names
    palette = _ensure_palette(num_classes, base_palette=base_palette)
    if class_names is None:
        class_names = BEV_CLASS_NAMES

    # create color image
    labels_np = labels.detach().cpu().numpy()
    color_img = _colorize(labels_np, palette)  # (H,W,3) uint8

    ax.imshow(color_img)
    return ax


def visualize_score(
    bev_segmap: torch.Tensor,
    trajectories: torch.Tensor,
    scores: torch.Tensor,
    score_types: List[str],
    figsize: tuple = (18, 6),
    title: Optional[str] = None,
    show: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Render score visualization on BEV."""
    num_score_types = len(score_types)
    assert scores.shape[-1] == num_score_types

    trajectories = torch_to_numpy(trajectories)
    scores = torch_to_numpy(scores)

    H, W = bev_segmap.shape
    def meter_to_pixel(x, y):

        x_scale = H / 32.0
        y_scale = W / 64.0
        y_min = -32.0

        x_px = x * x_scale
        y_px = (y - y_min) * y_scale
        return np.column_stack([x_px, y_px])

    fig, axs = plt.subplots(1, num_score_types+1, figsize=figsize)
    add_bev_segmap(axs[0], bev_segmap)
    for score_idx, score_type in enumerate(score_types):
        ax = axs[score_idx+1]
        add_bev_segmap(ax, bev_segmap)
        add_trajectory_with_scores(ax, trajectories, scores[..., score_idx], meter_to_pixel=meter_to_pixel)
        ax.set_title(score_type, fontsize=14, fontweight='bold')

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    if show:
        plt.show()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig

def visualize_trajectories(trajectories: List[np.ndarray],
                           config: TrajVisConfig=None,
                           title: str=None,
                           show: bool=False) -> plt.Figure:
    if config is None:
        config = get_default_traj_vis_config(trajectories)
    fig, ax = plt.subplots(figsize=(10, 10))
    for trajectory, color, alpha, width, zorder in zip(trajectories, config.colors, config.alphas, config.widths, config.zorder):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=alpha, linewidth=width, zorder=zorder)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def visualize_attn_mask(
    attn_mask: torch.Tensor,
    title: str = "Attention Mask",
    save_path: str = None,
    dpi: int = 300,
    show: bool = False,
) -> plt.Figure:
    """
    Visualize the attention mask for a given batch index.

    Args:
        attn_mask (torch.Tensor): Tensor of shape [B, Q, Q], dtype=bool.
        title (str): Plot title.
    """
    mask = attn_mask.cpu().numpy()  # [Q, Q]

    plt.figure(figsize=(6, 6))
    im = plt.imshow(mask, cmap="gray_r", interpolation="none")
    plt.title(title)
    plt.xlabel("Key Index")
    plt.ylabel("Query Index")

    # Colorbar
    cbar = plt.colorbar(im, label="Masked (1 = mask)")

    # Legend for meaning of colors
    masked_patch = mpatches.Patch(color="black", label="Masked (True)")
    unmasked_patch = mpatches.Patch(color="white", label="Unmasked (False)")
    plt.legend(handles=[masked_patch, unmasked_patch],
               loc="lower right",
               frameon=True)

    plt.grid(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return plt.gcf()


def get_subplots(num_plots: int, max_cols: int = 8, subplot_size: tuple = (5, 5)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Get subplots for a given number of plots.
    :param num_plots: number of plots
    :param max_cols: maximum number of columns
    :param subplot_size: size of each subplot
    :return: figure and axes
    """
    if num_plots <= max_cols:
        fig, axs = plt.subplots(1, num_plots, figsize=(subplot_size[0] * num_plots, subplot_size[1]))
        axs = axs.flatten()
        return fig, axs

    num_rows = (num_plots + max_cols - 1) // max_cols
    fig, axs = plt.subplots(num_rows, max_cols, figsize=(subplot_size[0] * max_cols, subplot_size[1] * num_rows))
    axs = axs.flatten()
    return fig, axs