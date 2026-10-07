import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from matplotlib import cm
from typing import Dict, Any
from pathlib import Path


CLASS_COLOR_MAP = {
    0: [255, 255, 255],
    1: [200, 200, 200],
    2: [180, 255, 180],
    3: [0, 0, 0],
    4: [255, 193, 7],
    5: [52, 152, 219],
    6: [231, 76, 60],
    7: [128, 128, 128],
}

def visualize_attn_mask(attn_mask: torch.Tensor, batch_idx: int = 0, title: str = "Attention Mask"):
    """
    Visualize the attention mask for a given batch index.

    Args:
        attn_mask (torch.Tensor): Tensor of shape [B, Q, Q], dtype=bool.
        batch_idx (int): Index in the batch to visualize.
        title (str): Plot title.
    """
    mask = attn_mask[batch_idx].cpu().numpy()  # [Q, Q]

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray_r", interpolation="none")
    plt.title(title)
    plt.xlabel("Key Index")
    plt.ylabel("Query Index")
    plt.colorbar(label="Masked (1 = mask)")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{title}.png")


def visualize_pwnc_prediction_vs_gt(
    bev_semantic_map,
    trajectory_anchors,
    agent_traj,
    pwnc_pred,
    pwnc_gt,
    batch_id=0,
    save_path="./debug_vis/pwnc_pred_vs_gt.png",
    pixel_size=0.25,
    threshold=0.2
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bev_np = bev_semantic_map[batch_id].cpu().numpy()
    H, W = bev_np.shape
    rgb = np.zeros((H, W, 3), np.uint8)
    for k, c in CLASS_COLOR_MAP.items():
        rgb[bev_np == k] = c

    anchors = trajectory_anchors[batch_id].cpu().numpy()        # [256, 8, 3]
    agents = agent_traj[batch_id, :15].cpu().numpy()            # [15, 8, 3]
    pred_scores = pwnc_pred[batch_id].detach().cpu().numpy()    # [15, 256]
    gt_matrix = pwnc_gt[batch_id, :15].cpu().numpy()            # [15, 256]

    cmap = cm.get_cmap("tab20", 15)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["Predicted PWNC Collision", "GT PWNC Collision"]

    for ax_id, (score_matrix, ax) in enumerate(zip([pred_scores, gt_matrix], axes)):
        ax.imshow(rgb)
        ax.set_title(titles[ax_id])
        ax.axis("off")

        for agent_idx, agent in enumerate(agents):
            traj_xy = agent[:, :2]
            traj_xy = traj_xy[~np.all(traj_xy == 0, axis=1)]
            if len(traj_xy) < 2:
                continue

            color = cmap(agent_idx)[:3]
            agent_px = ego_to_image_coords(traj_xy, H, W, scale=1/pixel_size)
            agent_px[:, 0] = W - agent_px[:, 0]
            agent_px[:, 1] = H - agent_px[:, 1]
            ax.plot(agent_px[:, 0], agent_px[:, 1], lw=3, linestyle='--', color=color)


            if ax_id == 0:
                collision_mask = score_matrix[agent_idx] > threshold
            else:
                collision_mask = score_matrix[agent_idx] > 0.5

            for anchor_idx in np.where(collision_mask)[0]:
                anchor_xy = anchors[anchor_idx][:, :2]
                anchor_px = ego_to_image_coords(anchor_xy, H, W, scale=1/pixel_size)
                anchor_px[:, 0] = W - anchor_px[:, 0]
                anchor_px[:, 1] = H - anchor_px[:, 1]
                ax.plot(anchor_px[:, 0], anchor_px[:, 1], lw=2, color=color, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Saved] PWNC prediction vs GT → {save_path}")


def plot_bev_semantic_maps(
    features: Dict[str, Any],
    targets: Dict[str, Any],
    agent: Any,
    trajectory: np.ndarray,
    save_path: Path,
    scores: Dict[str, Any] = None,
    pixel_size: float = 0.25,
) -> None:
    """
    Plots a single figure showing the current and future BEV semantic maps.
    Overlays the predicted trajectory on the current BEV map.

    Args:
        features: Not used.
        targets: Dictionary containing:
                 - 'bev_semantic_map': (128, 256)
                 - 'fut_bev_semantic_map_all': (8, 128, 256)
        agent: Not used.
        trajectory: (N, 3) predicted trajectory in ego frame.
        save_path: Output path for the PNG image.
        pixel_size: meters per pixel (default: 0.25).
    """

    # Validate required keys
    if 'bev_semantic_map' not in targets or 'fut_bev_semantic_map_all' not in targets:
        raise ValueError("Missing 'bev_semantic_map' or 'fut_bev_semantic_map_all' in targets.")

    cur_bev = targets['bev_semantic_map']            # (H, W)
    fut_bev = targets['fut_bev_semantic_map_all']    # (8, H, W)
    H, W = cur_bev.shape

    # Convert to numpy if torch.Tensor
    if isinstance(cur_bev, torch.Tensor):
        cur_bev = cur_bev.cpu().numpy()
    if isinstance(fut_bev, torch.Tensor):
        fut_bev = fut_bev.cpu().numpy()

    # Create 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    cur_rgb = to_rgb(cur_bev)
    axes[0].imshow(cur_rgb)
    axes[0].set_title("t = 0")
    axes[0].axis("off")

    if trajectory is not None and len(trajectory) > 0:
        traj_xy = trajectory[:, :2]
        traj_px = ego_to_image_coords(traj_xy, H, W, scale=1.0/pixel_size)
        traj_px[:, 0] = W - traj_px[:, 0]
        traj_px[:, 1] = H - traj_px[:, 1]

        axes[0].plot(traj_px[:, 0], traj_px[:, 1], lw=3, color="red", label="Prediction")
        axes[0].legend(loc="upper right", fontsize=8)

    # t+1 to t+8
    for k in range(8):
        ax = axes[k + 1]
        fut_rgb = to_rgb(fut_bev[k])
        ax.imshow(fut_rgb)
        ax.set_title(f"t + {k + 1}")
        ax.axis("off")

    # Hide last unused subplot (bottom-right)
    axes[-1].axis("off")

    # Add title and optional scores
    fig.suptitle("Semantic BEV Maps with Predicted Trajectory", fontsize=16)

    # If scores are provided, add them to the figure
    if scores:
        score_items = [f"{k}: {v:.4f}" for k, v in scores.items()]
        midpoint = len(score_items) // 2
        line1 = " | ".join(score_items[:midpoint])
        line2 = " | ".join(score_items[midpoint:])

        fig.text(0.5, 0.015, line1, ha='center', fontsize=9)
        fig.text(0.5, 0.001, line2, ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save
    save_path = Path(save_path)
    os.makedirs(save_path.parent, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def ego_to_image_coords(xy, h, w, scale=1.0):
    u = w / 2 - (xy[:, 1] * scale)
    v = h - (xy[:, 0] * scale)
    return np.stack([u, v], axis=1)


def to_rgb(bev_map: np.ndarray) -> np.ndarray:
    """Convert a semantic map to an RGB image using CLASS_COLOR_MAP."""
    rgb = np.zeros((bev_map.shape[0], bev_map.shape[1], 3), dtype=np.uint8)
    for k, color in CLASS_COLOR_MAP.items():
        rgb[bev_map == k] = color
    return rgb

