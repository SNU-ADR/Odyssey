from typing import Any, Optional

import numpy as np

import torch
import pytorch_lightning as pl

from navsim.agents.safedrive.safedrive_config import SafeDrive_Config

# ---- helper: Motion Prediction heatmap (local coords, axes swapped) ----


def scale_iou(gt_wh, pred_wh):
    min_wlh = torch.minimum(gt_wh, pred_wh)
    volume_annotation = torch.prod(gt_wh)
    volume_result = torch.prod(pred_wh)
    intersection = torch.prod(min_wlh)
    union = volume_annotation + volume_result - intersection
    iou = intersection / union.clamp(min=1e-6)  # prevent NaN when W or H = 0
    return iou


def yaw_diff(x, y, period: float = 2*np.pi) -> float:
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]
    return abs(diff)


def cummean_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Cumulative mean, NaN aware.
    - all NaN -> a tensor of ones
    - otherwise the NaNs are excluded from the average
    """
    if torch.isnan(x).sum() == len(x):
        return torch.ones_like(x)
    else:
        x_float = x.float()
        x_no_nan = x_float.clone()
        x_no_nan[torch.isnan(x_no_nan)] = 0.0
        sum_vals = torch.cumsum(x_no_nan, dim=0)
        count_vals = torch.cumsum(~torch.isnan(x), dim=0).float()
        cummean = torch.zeros_like(sum_vals)
        cummean[count_vals != 0] = sum_vals[count_vals != 0] / count_vals[count_vals != 0]
        return cummean


def _compute_ap_from_lists(tp_list, fp_list, score_list, num_gts, device):
    """Average precision from per-sample TP/FP lists, on the 101-point recall grid."""
    tp = torch.tensor([x for sub in tp_list for x in sub], device=device, dtype=torch.float32)
    fp = torch.tensor([x for sub in fp_list for x in sub], device=device, dtype=torch.float32)
    scores = torch.cat(score_list) if len(score_list) > 0 else torch.zeros(0, device=device)

    if scores.numel() == 0 or num_gts == 0:
        return torch.tensor(0.0, device=device)

    scores, idx = torch.sort(scores, descending=True)
    tp = torch.cumsum(tp[idx], 0)
    fp = torch.cumsum(fp[idx], 0)

    prec = tp / (tp + fp).clamp_min(1e-12)
    rec = tp / float(num_gts)

    def torch_interp(x_new, x, y):
        if x.numel() == 0:
            return torch.zeros_like(x_new)
        x_new = x_new.clamp(min=x[0], max=x[-1])
        idxs = torch.searchsorted(x, x_new, right=True).clamp(max=len(x) - 1)
        x0 = x[idxs - 1]; x1 = x[idxs]; y0 = y[idxs - 1]; y1 = y[idxs]
        denom = (x1 - x0); denom[denom == 0] = 1e-6
        w = (x_new - x0) / denom
        return y0 + w * (y1 - y0)

    rec_interp = torch.linspace(0, 1, 101, device=device)   # the 101-point recall grid
    prec = torch_interp(rec_interp, rec, prec)
    min_recall, min_precision = 0.1, 0.1
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    if prec.numel() == 0:
        return torch.tensor(0.0, device=device)
    prec = (prec - min_precision).clamp_min(0)
    return (prec.mean() / (1.0 - min_precision))

def compute_map_fast(
    pred_boxes,
    pred_scores,
    gt_boxes,
    distance_treshold=0.5,
    pred_motion=None,
    pred_motion_cls=None,
    target_motion=None,
    target_motion_mask=None,
    device='cpu',
):
    """Greedy centre-distance matching for one frame; returns the AP inputs and the error terms."""
    tp_list = []
    fp_list = []

    trans_err_list = []
    scale_err_list = []
    orien_err_list = []
    vel_err_list = []
    score_list = []

    minade_list = []
    minfde_list = []
    ade_list = []
    fde_list = []

    # last GT pose per matched agent, in that agent's own frame
    tp_last_x_list = []
    tp_last_y_list = []

    scores, indices = torch.sort(pred_scores, descending=True)
    pred_boxes = pred_boxes[indices]

    if pred_motion is not None:  #
        pred_motion = pred_motion[indices]
        pred_motion_cls = pred_motion_cls[indices] if pred_motion_cls is not None else None

    matched_gt_flags = torch.zeros(gt_boxes.size(0), dtype=torch.bool, device=device)

    num_preds = pred_boxes.size(0)
    num_gts = gt_boxes.size(0)

    if gt_boxes.numel() == 0:
        for _ in range(num_preds):
            tp_list.append(0)
            fp_list.append(1)
        return (tp_list, fp_list, scores, num_gts, trans_err_list, scale_err_list, orien_err_list, vel_err_list, score_list,
                minade_list, minfde_list, ade_list, fde_list, tp_last_x_list, tp_last_y_list)

    distances = torch.norm(pred_boxes.float()[:, None, :2] - gt_boxes.float()[None, :, :2], dim=-1, p=2)
    min_dis, min_idxs = distances.min(dim=1)  # Best match GT for each pred

    for i in range(num_preds):
        gt_idx = min_idxs[i]
        if min_dis[i] <= distance_treshold and not matched_gt_flags[gt_idx]:
            # true positive
            tp_list.append(1)
            fp_list.append(0)
            trans_err_list.append(min_dis[i])       # mATE
            scale_err_list.append(1 - scale_iou(gt_boxes[gt_idx, 2:4], pred_boxes[i, 2:4]))
            orien_err_list.append(yaw_diff(gt_boxes[gt_idx, 4], pred_boxes[i, 4]))
            vel_err_list.append(torch.sqrt(((gt_boxes[gt_idx,5:7] - pred_boxes[i,5:7])**2).sum()))
            score_list.append(scores[i])
            matched_gt_flags[gt_idx] = True

            # Prediction
            if pred_motion is not None and target_motion_mask[gt_idx].sum() != 0:
                cur_pred_motion = pred_motion[i][..., :2] + pred_boxes[i][None, None, :2]  # relative motion -> absolute coordinates
                cur_mode = pred_motion_cls[i].argmax() if pred_motion_cls is not None else 0
                cur_tar_motion = target_motion[gt_idx][..., :2]

                valid_mask = target_motion_mask[gt_idx]

                # NaN in model outputs (e.g. early training instability) → skip this sample
                if torch.isnan(cur_pred_motion).any():
                    continue
                pred_err = torch.sqrt(((cur_pred_motion - cur_tar_motion[None])**2).sum(-1))    # (M, T)
                pred_err = pred_err.permute(1, 0)[target_motion_mask[gt_idx]].permute(1, 0)
                minade_list.append(pred_err.mean(-1).min())     # minADE
                minfde_list.append(pred_err[:, -1].min())       # minFDE
                pred_err = torch.sqrt(((cur_pred_motion[cur_mode] - cur_tar_motion)**2)[target_motion_mask[gt_idx]].sum(-1))    # [T]
                ade_list.append(pred_err.mean(-1))
                fde_list.append(pred_err[-1])

                gt_origin = gt_boxes[gt_idx, :2]
                tar_loc = cur_tar_motion - gt_origin[None, :]

                gt_yaw = gt_boxes[gt_idx, 4]
                c, s = torch.cos(-gt_yaw), torch.sin(-gt_yaw)
                R = torch.tensor([[c, -s], [s, c]], device=device, dtype=tar_loc.dtype)
                tar_loc = (tar_loc @ R.T)  # (T, 2)

                last_idx = torch.nonzero(valid_mask, as_tuple=True)[0][-1]
                tp_last_x_list.append(tar_loc[last_idx, 0].detach())
                tp_last_y_list.append(tar_loc[last_idx, 1].detach())

        else:
            tp_list.append(0)
            fp_list.append(1)

    return (tp_list, fp_list, scores, num_gts, trans_err_list, scale_err_list, orien_err_list, vel_err_list, score_list,
            minade_list, minfde_list, ade_list, fde_list, tp_last_x_list, tp_last_y_list)

def _to_log(v, log_device):
    """Ensure value is a finite scalar CUDA tensor for DDP sync_dist logging.
    NaN/Inf → 0.0 to prevent NCCL all_reduce hang in DDP.
    """
    if torch.is_tensor(v):
        t = v.detach().float().to(log_device)
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    val = float(v)
    if val != val or not (val < float('inf')):  # NaN or Inf check
        val = 0.0
    return torch.tensor(val, device=log_device, dtype=torch.float32)


class SafeDrive_Callback(pl.Callback):
    """Visualization Callback for TransFuser during training."""

    def __init__(
        self,
        config: SafeDrive_Config,
        num_plots: int = 3,
        num_rows: int = 2,
        num_columns: int = 2,
    ) -> None:
        """
        Initializes the visualization callback.
        :param config: global config dataclass of TransFuser
        :param num_plots: number of images tiles, defaults to 3
        :param num_rows: number of rows in image tile, defaults to 2
        :param num_columns: number of columns in image tile, defaults to 2
        """

        self._config = config

        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns

        self.no_agent_motion = config.no_planning

    def on_validation_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        all_targets = lightning_module.val_gts
        device = all_targets[0]["agent_states"].device  # CPU (offloaded for memory)
        log_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._log_detection_metrics(lightning_module, device, log_device)
        self._log_trajectory_metrics(lightning_module, device, log_device)
        lightning_module.val_preds.clear()
        lightning_module.val_gts.clear()
        self._log_bev_seg_metrics(lightning_module, device, log_device)
        self._save_test_trajectories(lightning_module, device, log_device)
        self._log_safety_errors(lightning_module, device, log_device)

    def _log_detection_metrics(self, lightning_module, device, log_device):
        """Detection mAP over the distance thresholds, averaged into val_detection/mAP."""
        all_preds = lightning_module.val_preds
        all_targets = lightning_module.val_gts
        mAP_list = []

        # detection, per distance threshold
        for dis_thresh in [0.25, 0.5, 1.0, 2.0]:
            mAP_list.append(self._detection_metrics_at_threshold(
                all_preds, all_targets, dis_thresh, device, log_device))

        self.log("val_detection/mAP", _to_log(sum(mAP_list)/len(mAP_list), log_device), on_epoch=True, prog_bar=True, sync_dist=True)

    def _detection_metrics_at_threshold(self, all_preds, all_targets, dis_thresh,
                                        device, log_device):
        """Detection and prediction metrics at one centre-distance threshold; returns mAP."""
        classes_to_eval = [0, 1]  # vehicle + ped

        per_cls = {
            c: dict(tp=[], fp=[], scores=[], num_gts=[],
                    trans_err=[], scale_err=[], orien_err=[], vel_err=[], tp_scores=[],
                    minade=[], minfde=[], ade=[], fde=[],
                    tp_last_x=[], tp_last_y=[])
            for c in classes_to_eval
        }

        score_thresh = 0.1

        # ---- iterate samples/frames ----
        for i in range(len(all_targets)):
            gt_states_all = all_targets[i]["agent_states"]  # (B, N, 7)
            gt_cls_all    = all_targets[i]["agent_labels"]  # [B, N] {0:veh,1:ped,-1:ignore}

            # Predictions
            layer_num = self._config.ins_decoder_layer - 1
            all_pred = all_preds[i][f'ins_states_{layer_num}']
            sin_yaw = all_pred[..., 2:3].float()
            cos_yaw = all_pred[..., 3:4].float()
            yaw = torch.atan2(sin_yaw, cos_yaw)
            W = all_pred[..., 4:5]
            H = all_pred[..., 5:6]
            pred_states_all = torch.cat([all_pred[..., :2], yaw, W, H, all_pred[..., 6:8]], dim=-1)  # (B,N,5)
            pred_logits_all = all_preds[i][f'ins_labels_{layer_num}']          # [B,N] or [B,N,K+1]

            # Motion
            if not self.no_agent_motion:
                layer_num = self._config.SWNet_num_layers
                pred_motion_all = all_preds[i][f'agent_motion_traj_{layer_num}'][..., :5]
                pred_motion_cls_all = all_preds[i].get(f'agent_motion_cls_{layer_num}',
                                                      pred_motion_all.new_zeros(*pred_motion_all.shape[:3], 1))
            target_motion_all      = all_targets[i]['motion_traj'][..., 1:, :]
            target_motion_mask_all = all_targets[i]['motion_mask'][..., 1:]

            batch_size = gt_states_all.shape[0]

            for b in range(batch_size):
                gt_states = gt_states_all[b]        # (N, 7)
                gt_cls    = gt_cls_all[b].long()    # [N]

                rad_to_ego = torch.arctan2(gt_states[..., 1], gt_states[..., 0])
                in_latent_rad_thresh = torch.logical_and(
                    -self._config.latent_rad_thresh <= rad_to_ego,
                    rad_to_ego <= self._config.latent_rad_thresh,
                )
                gt_keep = (gt_cls != -1) & in_latent_rad_thresh

                gt_boxes_all = torch.stack([
                    gt_states[..., 0],  # x
                    gt_states[..., 1],  # y
                    gt_states[..., 3],  # w
                    gt_states[..., 4],  # h
                    gt_states[..., 2],  # yaw
                    gt_states[..., 5],  # yaw
                    gt_states[..., 6],  # yaw
                ], dim=-1)

                gt_boxes_all = gt_boxes_all[gt_keep]     # (Ng, 5)
                gt_cls_all_f = gt_cls[gt_keep]           # [Ng]

                pred_states = pred_states_all[b]         # (N, 5)
                pred_logits = pred_logits_all[b]         # [N] or [N, K+1]

                pred_boxes_all = torch.stack([
                    pred_states[..., 0],
                    pred_states[..., 1],
                    pred_states[..., 3],
                    pred_states[..., 4],
                    pred_states[..., 2],
                    pred_states[..., 5],
                    pred_states[..., 6],
                ], dim=-1)                                # (N, 5)

                if not self.no_agent_motion:
                    pred_traj_all = pred_motion_all[b]
                    pred_traj_scores_all = pred_motion_cls_all[b]
                else:
                    pred_traj_all = None
                    pred_traj_scores_all = None

                multiclass = (pred_logits.ndim == 2 and pred_logits.shape[-1] > 1)
                if multiclass:
                    # det_pedestrian head: one sigmoid logit per class, no background column
                    probs = pred_logits.sigmoid()       # (N, K)
                    pred_arg = probs.argmax(-1)
                else:
                    probs = pred_logits.sigmoid()       # [N]
                for c in classes_to_eval:
                    gt_mask_c = (gt_cls_all_f == c)
                    gt_boxes_c = gt_boxes_all[gt_mask_c]  # (Ng_c, 5)
                    target_motion = target_motion_all[b][gt_keep][gt_mask_c]
                    target_motion_mask = target_motion_mask_all[b][gt_keep][gt_mask_c]

                    if multiclass:
                        scores_c = probs[:, c]                      # [N]
                        keep_c = (pred_arg == c) & (scores_c >= score_thresh)
                        pred_boxes_c = pred_boxes_all[keep_c]
                        pred_scores_c = scores_c[keep_c]
                        pred_traj_c = pred_traj_all[keep_c] if pred_traj_all is not None else None
                        pred_traj_scores_c = pred_traj_scores_all[keep_c] if pred_traj_scores_all is not None else None
                    else:
                        if c != 0:
                            continue
                        scores = probs                               # [N]
                        keep_c = (scores >= score_thresh)
                        pred_boxes_c = pred_boxes_all[keep_c]
                        pred_scores_c = scores[keep_c]
                        pred_traj_c = pred_traj_all[keep_c] if pred_traj_all is not None else None
                        pred_traj_scores_c = pred_traj_scores_all[keep_c] if pred_traj_scores_all is not None else None

                    (tp, fp, scores, num_gts,
                     trans_err, scale_err, orien_err, vel_err, tp_score,
                     minade, minfde, ade, fde,
                     tp_last_x, tp_last_y) = compute_map_fast(
                        pred_boxes_c, pred_scores_c, gt_boxes_c, dis_thresh,
                        pred_traj_c, pred_traj_scores_c,
                        target_motion, target_motion_mask,
                        device
                    )

                    pc = per_cls[c]
                    pc["tp"].append(tp); pc["fp"].append(fp); pc["scores"].append(scores); pc["num_gts"].append(num_gts)
                    pc["trans_err"].append(trans_err); pc["scale_err"].append(scale_err); pc["orien_err"].append(orien_err)
                    pc["vel_err"].append(vel_err)
                    pc["tp_scores"].append(tp_score)
                    pc["minade"].append(minade); pc["minfde"].append(minfde); pc["ade"].append(ade); pc["fde"].append(fde)
                    pc["tp_last_x"].append(tp_last_x); pc["tp_last_y"].append(tp_last_y)

        ap_per_class = {}
        for c in classes_to_eval:
            pc = per_cls[c]
            num_gts_c = sum(pc["num_gts"])
            ap_c = _compute_ap_from_lists(pc["tp"], pc["fp"], pc["scores"], num_gts_c, device)
            ap_per_class[c] = ap_c
            name = "veh" if c == 0 else "ped"
            self.log(f"val_detection/mAP_{name}_{dis_thresh}", _to_log(ap_c, log_device), on_epoch=True, prog_bar=False, sync_dist=True)

        if len(ap_per_class) > 0:
            mAP_macro = torch.stack([ap_per_class[c] for c in ap_per_class]).mean()
        else:
            mAP_macro = torch.tensor(0.0, device=device)
        self.log(f"val_detection/mAP_{dis_thresh}", _to_log(mAP_macro, log_device), on_epoch=True, prog_bar=True, sync_dist=True)

        trans_err_tensor = torch.tensor(
            [x for c in per_cls for sub in per_cls[c]["trans_err"] for x in sub], device=device)
        scale_err_tensor = torch.tensor(
            [x for c in per_cls for sub in per_cls[c]["scale_err"] for x in sub], device=device)
        orien_err_tensor = torch.tensor(
            [x for c in per_cls for sub in per_cls[c]["orien_err"] for x in sub], device=device)
        vel_err_tensor = torch.tensor(
            [x for c in per_cls for sub in per_cls[c]["vel_err"] for x in sub], device=device)

        tp_score_list_all = [torch.tensor(x, device=device) for c in per_cls for x in per_cls[c]["tp_scores"] if len(x) > 0]
        tp_score_tensor = torch.cat(tp_score_list_all) if tp_score_list_all else torch.zeros(0, device=device)

        def torch_interp(x_new, x, y):
            x = x.clamp_min(0); y = y.clamp_min(0)
            if x.numel() == 0:
                return torch.zeros_like(x_new)
            x_new = x_new.clamp(min=x[0], max=x[-1])
            idxs = torch.searchsorted(x, x_new, right=True).clamp(max=len(x) - 1)
            x0 = x[idxs - 1]; x1 = x[idxs]; y0 = y[idxs - 1]; y1 = y[idxs]
            denom = (x1 - x0); denom[denom == 0] = 1e-6
            w = (x_new - x0) / denom
            return y0 + w * (y1 - y0)

        # scores for interpolation:
        if tp_score_tensor.numel() > 0:
            tp_scores, tp_indices = torch.sort(tp_score_tensor, descending=True)
        else:
            tp_scores = torch.zeros(1, device=device)
            tp_indices = torch.zeros(1, dtype=torch.long, device=device)

        for tp_idx, (tp_err_tensor, tp_err_str) in enumerate([
            (trans_err_tensor, 'mATE'),
            (scale_err_tensor, 'mASE'),
            (orien_err_tensor, 'mAOE'),
            (vel_err_tensor, 'mAVE'),
        ]):
            if tp_err_tensor.numel() == 0 or tp_scores.numel() == 0:
                tp_err_val = 1.0
            else:
                tp_err_sorted = tp_err_tensor[tp_indices]
                tmp = cummean_torch(tp_err_sorted)
                conf = torch.linspace(0, 1, tp_scores.numel(), device=device)  # proxy
                tp_curve = torch_interp(conf.flip(0), conf.flip(0), tmp.flip(0)).flip(0)
                min_recall, _ = 0.1, 0.1
                first_ind = round(100 * min_recall) + 1
                last_ind = tp_scores.nonzero()
                if last_ind.numel() == 0:
                    tp_err_val = 1.0
                else:
                    last_ind = last_ind[-1][0]
                    if last_ind < first_ind:
                        tp_err_val = 1.0
                    else:
                        tp_err_val = float(tp_curve[first_ind: last_ind + 1].mean())
            self.log(f"val_tp_err/{tp_err_str}_{dis_thresh}", _to_log(tp_err_val, log_device), on_epoch=True, prog_bar=True, sync_dist=True)

        minade_tensor = torch.tensor([x for c in per_cls for sub in per_cls[c]["minade"] for x in sub], device=device)
        minfde_tensor = torch.tensor([x for c in per_cls for sub in per_cls[c]["minfde"] for x in sub], device=device)
        ade_tensor    = torch.tensor([x for c in per_cls for sub in per_cls[c]["ade"]    for x in sub], device=device)
        fde_tensor    = torch.tensor([x for c in per_cls for sub in per_cls[c]["fde"]    for x in sub], device=device)

        if ade_tensor.numel() == 0:
            minade = 0; minfde = 0; ade = 0; fde = 0
        else:
            minade = minade_tensor.nanmean()
            minfde = minfde_tensor.nanmean()
            ade = ade_tensor.nanmean()
            fde = fde_tensor.nanmean()

        self.log(f"val_prediction/ade_{dis_thresh}", _to_log(ade, log_device), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"val_prediction/fde_{dis_thresh}", _to_log(fde, log_device), on_epoch=True, prog_bar=True, sync_dist=True)
        return mAP_macro

    def _log_trajectory_metrics(self, lightning_module, device, log_device):
        """Planning ADE / FDE, including the 2D-binned breakdown."""
        if 'trajectory' in lightning_module.val_preds[0].keys():
            plan_l2_err_list = []
            for val_pred, val_gt in zip(lightning_module.val_preds, lightning_module.val_gts):
                # (B, T, 2)
                pred_xy = val_pred['trajectory'][..., :2]
                gt_xy   = val_gt['trajectory'][..., :2]
                T = min(pred_xy.shape[1], gt_xy.shape[1])
                pred_xy = pred_xy[:, :T]
                gt_xy   = gt_xy[:, :T]

                l2_per_t = torch.norm(gt_xy - pred_xy, dim=2)      # (B, T)
                l2_per_b = l2_per_t.mean(dim=1)                    # [B]
                plan_l2_err_list.append(l2_per_b)

            overall_mean = torch.cat(plan_l2_err_list).mean()
            self.log("val_planning/L2_distance", _to_log(overall_mean, log_device), on_epoch=True, prog_bar=True, sync_dist=True)

            x_edges  = torch.tensor([0., 5., 10., 15., 20., 25., 30.], device=device)
            x_labels = ["x_0_5", "x_5_10", "x_10_15", "x_15_20", "x_20_25", "x_25_30", "x_30p"]

            y_edges  = torch.tensor([-14., -10., -6., -2., 2., 6., 10., 14.], device=device)
            y_labels = ["y_n14p", "y_n14_n10", "y_n10_n6", "y_n6_n2", "y_n2_2", "y_2_6", "y_6_10", "y_10_14", "y_14p"]

            bin_x_ade = {k: [] for k in x_labels}
            bin_x_fde = {k: [] for k in x_labels}
            bin_x_l2  = {k: [] for k in x_labels}

            bin_y_ade = {k: [] for k in y_labels}
            bin_y_fde = {k: [] for k in y_labels}
            bin_y_l2  = {k: [] for k in y_labels}

            bin_xy_ade = {(xk, yk): [] for xk in x_labels for yk in y_labels}
            bin_xy_fde = {(xk, yk): [] for xk in x_labels for yk in y_labels}
            bin_xy_l2  = {(xk, yk): [] for xk in x_labels for yk in y_labels}

            def _pairwise_metrics(pred_xy_bt: torch.Tensor, gt_xy_bt: torch.Tensor):
                """ pred_xy_bt, gt_xy_bt: [B, T, 2] -> (ADE[B], FDE[B], L2mean[B]) """
                l2_bt = torch.norm(pred_xy_bt - gt_xy_bt, dim=-1)  # (B, T)
                ade_b = l2_bt.mean(dim=1)                          # [B]
                fde_b = l2_bt[:, -1]                               # [B]
                return ade_b, fde_b, ade_b

            for val_pred, val_gt in zip(lightning_module.val_preds, lightning_module.val_gts):
                pred_xy_bt = val_pred['trajectory'][..., :2]   # (B, T, 2)
                gt_xy_bt   = val_gt['trajectory'][..., :2]     # (B, T, 2)
                T = min(pred_xy_bt.shape[1], gt_xy_bt.shape[1])
                pred_xy_bt = pred_xy_bt[:, :T]
                gt_xy_bt   = gt_xy_bt[:, :T]

                ade_b, fde_b, l2mean_b = _pairwise_metrics(pred_xy_bt, gt_xy_bt)  # [B], [B], [B]

                gt_last = gt_xy_bt[:, -1, :]  # (B, 2)
                last_x  = gt_last[:, 0]
                last_y  = gt_last[:, 1]

                x_idx = torch.bucketize(last_x, x_edges, right=False).clamp(max=len(x_labels)-1)
                y_idx = torch.bucketize(last_y, y_edges, right=False).clamp(min=0, max=len(y_labels)-1)

                for b in range(x_idx.numel()):
                    k = x_labels[int(x_idx[b].item())]
                    bin_x_ade[k].append(ade_b[b].detach())
                    bin_x_fde[k].append(fde_b[b].detach())
                    bin_x_l2[k].append(l2mean_b[b].detach())

                for b in range(y_idx.numel()):
                    ky = y_labels[int(y_idx[b].item())]
                    bin_y_ade[ky].append(ade_b[b].detach())
                    bin_y_fde[ky].append(fde_b[b].detach())
                    bin_y_l2[ky].append(l2mean_b[b].detach())

                for b in range(x_idx.numel()):
                    xk = x_labels[int(x_idx[b].item())]
                    yk = y_labels[int(y_idx[b].item())]
                    bin_xy_ade[(xk, yk)].append(ade_b[b].detach())
                    bin_xy_fde[(xk, yk)].append(fde_b[b].detach())
                    bin_xy_l2[(xk, yk)].append(l2mean_b[b].detach())

            x_ade_bin_list = []
            x_fde_bin_list = []
            for k in x_labels:
                n = len(bin_x_ade[k])
                if n > 0:
                    ade_mean = torch.stack(bin_x_ade[k]).mean()
                    fde_mean = torch.stack(bin_x_fde[k]).mean()
                else:
                    ade_mean = torch.tensor(float(0), device=device)
                    fde_mean = torch.tensor(float(0), device=device)
                self.log(f"val_planning_bins_x/ADE_{k}", _to_log(ade_mean, log_device), on_epoch=True, prog_bar=False, sync_dist=True)
                self.log(f"val_planning_bins_x/FDE_{k}", _to_log(fde_mean, log_device), on_epoch=True, prog_bar=False, sync_dist=True)
                x_ade_bin_list.append(ade_mean)
                x_fde_bin_list.append(fde_mean)
            self.log("val_planning_bins_x/mean_ADE", _to_log(sum(x_ade_bin_list)/len(x_ade_bin_list), log_device), on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val_planning_bins_x/mean_FDE", _to_log(sum(x_fde_bin_list)/len(x_fde_bin_list), log_device), on_epoch=True, prog_bar=False, sync_dist=True)

            y_ade_bin_list = []
            y_fde_bin_list = []
            for ky in y_labels:
                n = len(bin_y_ade[ky])
                if n > 0:
                    ade_mean = torch.stack(bin_y_ade[ky]).mean()
                    fde_mean = torch.stack(bin_y_fde[ky]).mean()
                else:
                    ade_mean = torch.tensor(float(0), device=device)
                    fde_mean = torch.tensor(float(0), device=device)
                self.log(f"val_planning_bins_y/ADE_{ky}", _to_log(ade_mean, log_device), on_epoch=True, prog_bar=False, sync_dist=True)
                self.log(f"val_planning_bins_y/FDE_{ky}", _to_log(fde_mean, log_device), on_epoch=True, prog_bar=False, sync_dist=True)
                y_ade_bin_list.append(ade_mean)
                y_fde_bin_list.append(fde_mean)
            self.log("val_planning_bins_y/mean_ADE", _to_log(sum(y_ade_bin_list)/len(y_ade_bin_list), log_device), on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val_planning_bins_y/mean_FDE", _to_log(sum(y_fde_bin_list)/len(y_fde_bin_list), log_device), on_epoch=True, prog_bar=False, sync_dist=True)

            xy_ade_bin_list = []
            xy_fde_bin_list = []
            for xk in x_labels:
                for yk in y_labels:
                    vals_ade = bin_xy_ade[(xk, yk)]
                    vals_fde = bin_xy_fde[(xk, yk)]
                    n = len(vals_ade)
                    if n > 0:
                        ade_mean = torch.stack(vals_ade).mean()
                        fde_mean = torch.stack(vals_fde).mean()
                    else:
                        ade_mean = torch.tensor(float(0), device=device)
                        fde_mean = torch.tensor(float(0), device=device)

                    self.log(f"val_planning_bins2D/ADE_{xk}_{yk}", _to_log(ade_mean, log_device),
                            on_epoch=True, prog_bar=False, sync_dist=True)
                    self.log(f"val_planning_bins2D/FDE_{xk}_{yk}", _to_log(fde_mean, log_device),
                            on_epoch=True, prog_bar=False, sync_dist=True)
                    xy_ade_bin_list.append(ade_mean)
                    xy_fde_bin_list.append(fde_mean)
            self.log("val_planning_bins2D/mean_ADE", _to_log(sum(xy_ade_bin_list)/len(xy_ade_bin_list), log_device), on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val_planning_bins2D/mean_FDE", _to_log(sum(xy_fde_bin_list)/len(xy_fde_bin_list), log_device), on_epoch=True, prog_bar=False, sync_dist=True)

    def _log_bev_seg_metrics(self, lightning_module, device, log_device):
        """BEV segmentation IoU per class and the mean."""
        if len(lightning_module.val_preds_seg) != 0:
            class_names = [
                "background", "road", "walkways", "centerline",
                "static_objects", "vehicles", "pedestrians"
            ]
            num_classes = len(class_names)

            class_ious = [[] for _ in range(num_classes)]

            for pred_seg, gt_seg in zip(lightning_module.val_preds_seg, lightning_module.val_gts_seg):
                pred_seg = pred_seg.to(device)  # [H, W] or [B, H, W] assumed as single sample per element
                gt_seg = gt_seg.long().to(device)

                for idx in range(num_classes):
                    pred_inds = pred_seg == idx
                    target_inds = gt_seg == idx

                    intersection = (pred_inds & target_inds).sum().float()
                    union = (pred_inds | target_inds).sum().float()

                    if union == 0:
                        continue  # skip (class absent in both pred and GT)

                    iou = intersection / union
                    class_ious[idx].append(iou)

            valid_class_ious = []
            for idx, ious in enumerate(class_ious):
                if len(ious) == 0:
                    continue
                class_iou_avg = torch.nanmean(torch.stack(ious))
                if not torch.isnan(class_iou_avg):
                    valid_class_ious.append(class_iou_avg)
                self.log(f"val_segmentation/IoU_class_{class_names[idx]}", _to_log(class_iou_avg, log_device), on_epoch=True, prog_bar=True, sync_dist=True)

            if len(valid_class_ious) > 0:
                mIoU = torch.stack(valid_class_ious).mean()
            else:
                mIoU = torch.tensor(0.0)
            self.log("val_segmentation/mIoU", _to_log(mIoU, log_device), on_epoch=True, prog_bar=True, sync_dist=True)

            lightning_module.val_preds_seg.clear()
            lightning_module.val_gts_seg.clear()

    def _save_test_trajectories(self, lightning_module, device, log_device):
        """Dump the collected test trajectories to a pickle."""
        if lightning_module.test_traj_save and not lightning_module.debug_mode:
            import pickle
            import torch.distributed as dist

            save_file = lightning_module.scoring_test_save_name
            local_data = lightning_module.test_traj

            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_data)

            if dist.get_rank() == 0:
                merged_test_traj = {}
                for d in gathered:
                    merged_test_traj.update(d)
                out_path = f'exp/training/{lightning_module.agent_name}/{save_file}'
                with open(out_path, 'wb') as f:
                    pickle.dump(merged_test_traj, f)
                print(f"Saved merged dict to {out_path}")

    def _log_safety_errors(self, lightning_module, device, log_device):
        """Safety-head error terms collected during validation."""
        if len(lightning_module.safety_dict['NC_err']) != 0:
            for l in lightning_module.safety_list:
                if f'{l}_err' in lightning_module.safety_dict.keys():
                    lst = lightning_module.safety_dict[f'{l}_err']
                    total = 0.0
                    count = 0
                    if len(lst) != 0:
                        for t in lst:
                            for t_ in t:
                                total += t_.sum()
                                count += t_.numel()
                        safe_score = (total / max(count, 1)).to(torch.float32)
                        self.log(f"val_safety/{l}", _to_log(safe_score, log_device), on_epoch=True, prog_bar=True, sync_dist=True)

        if len(lightning_module.safety_dict['NC_err_only_one']) != 0:
            for l in lightning_module.safety_list:
                if f'{l}_err_only_one' in lightning_module.safety_dict.keys():
                    lst = lightning_module.safety_dict[f'{l}_err_only_one']
                    total = 0.0
                    count = 0
                    if len(lst) != 0:
                        for t in lst:
                            for t_ in t:
                                total += t_.sum()
                                count += t_.numel()
                        safe_score = (total / max(count, 1)).to(torch.float32)
                        self.log(f"val_safety/{l}_only_one", safe_score, on_epoch=True, prog_bar=True, sync_dist=True)

        # ===================== Score Cache Save =====================

    def on_test_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        pass

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        pass

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        pass

    def on_train_epoch_end(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """Inherited, see superclass."""
        pass
