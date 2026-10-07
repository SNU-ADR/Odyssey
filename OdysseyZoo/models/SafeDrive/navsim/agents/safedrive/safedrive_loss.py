"""SafeDrive losses, one function per stage: ProposalNet -> SWNet -> FRNet.

Shape letters: B batch, L decoder layers, Q instance queries, K plan anchors,
A selected agents, T future poses.
"""
from typing import Dict
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F

from navsim.agents.safedrive.safedrive_config import SafeDrive_Config
from navsim.agents.safedrive.safedrive_features import BoundingBox2DIndex

from mmdet.utils import util_mixins

from mmdet.models.utils import multi_apply
from mmdet.models.task_modules import AssignResult

from navsim.agents.safedrive.safedrive_utils import py_sigmoid_focal_loss

def safedrive_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: SafeDrive_Config,
    test_traj_save=False,
    current_epoch=None,
):
    """Total SafeDrive loss, one term per stage: ProposalNet -> SWNet -> FRNet.

    `stage` carries the values one stage's loss hands to the next (the matched GT lists,
    the motion assignment). Returns the dict of every logged term plus 'loss'.
    """
    loss_dict = {}
    stage = {}

    loss = predictions['ins_labels_0'].new_zeros(1)[0]
    loss = _proposal_net_loss(targets, predictions, config, loss_dict, stage, loss)
    loss = _swnet_loss(targets, predictions, config, loss_dict, stage, loss)
    loss = _frnet_loss(targets, predictions, config, loss_dict, stage, loss, test_traj_save)

    loss_dict.update({
        'loss': loss,
        'bev_semantic_loss': config.bev_semantic_weight * stage['bev_semantic_loss'],
    })
    if "trajectory_loss_dict" in predictions:
        trajectory_loss_dict = predictions["trajectory_loss_dict"]
        loss_dict.update(trajectory_loss_dict)

    return loss_dict


def _proposal_net_loss(targets, predictions, config, loss_dict, stage, loss,
                       test_traj_save=False):
    """ProposalNet loss: instance detection (+ denoising queries) and the BEV semantic maps.

    Adds to `loss_dict` and returns the running `loss`.
    """
    # only vehicles (+ pedestrians with det_pedestrian) inside the forward cone are matched; the rest is padding
    rad_to_ego = torch.arctan2(targets['agent_states'][..., BoundingBox2DIndex.Y], targets['agent_states'][..., BoundingBox2DIndex.X],)
    in_latent_rad_thresh = torch.logical_and(-config.latent_rad_thresh <= rad_to_ego, rad_to_ego <= config.latent_rad_thresh,)
    gt_valid = torch.logical_and(in_latent_rad_thresh, targets["agent_labels"]!=-1)
    gt_labels = torch.where(in_latent_rad_thresh,targets["agent_labels"],torch.full_like(targets["agent_labels"], -1))
    if config.det_pedestrian:
        gt_labels = gt_labels.long()        # class index 0 vehicle / 1 pedestrian; -1 rows are dropped by gt_valid
    else:
        gt_labels = gt_labels==0
        gt_valid = gt_labels

    # Per-sample boolean masks cost a sync each. One count sync plus a stable "valid first" gather gives the
    # same rows in the same order.
    _nb = predictions['ins_labels_0'].shape[0]
    _counts = gt_valid.sum(1).tolist()
    _order = torch.argsort((~gt_valid).to(torch.int8), dim=1, stable=True)
    _labels_s = torch.gather(gt_labels, 1, _order)
    _states_s = torch.gather(targets['agent_states'], 1, _order[..., None].expand_as(targets['agent_states']))
    gt_labels_list = [_labels_s[b_id, :_counts[b_id]] for b_id in range(_nb)]
    gt_bboxes_list = [_states_s[b_id, :_counts[b_id]] for b_id in range(_nb)]

    all_gt_bboxes_list = [gt_bboxes_list for _ in range(config.bevformer_decoder['num_layers'])]
    all_gt_labels_list = [gt_labels_list for _ in range(config.bevformer_decoder['num_layers'])]
    config_list = [config for _ in range(config.bevformer_decoder['num_layers'])]

    all_bbox_preds = [predictions[f'ins_states_{l_d}'] for l_d in range(config.bevformer_decoder['num_layers'])]  # x, y, yaw, l, w, vx, vy
    all_cls_scores = [predictions[f'ins_labels_{l_d}'] for l_d in range(config.bevformer_decoder['num_layers'])]
    # -- instance detection --  (Hungarian matched, supervised at every layer)
    losses_cls, losses_bbox, losses_vel = multi_apply(
        _agent_loss_single, all_bbox_preds, all_cls_scores,
        all_gt_bboxes_list, all_gt_labels_list, config_list
        )
    for d_l in range(config.bevformer_decoder['num_layers']):
        loss = loss + (losses_bbox[d_l] * config.agent_box_weight + losses_cls[d_l] * config.agent_class_weight)
        loss_dict[f'ins_states_{d_l}_loss'] = losses_bbox[d_l] * config.agent_box_weight
        loss_dict[f'agent_label_{d_l}_loss'] = losses_cls[d_l] * config.agent_class_weight
        loss_dict[f'agent_veloicy_{d_l}_loss'] = losses_vel[d_l] * config.agent_vel_weight
        loss = loss + losses_vel[d_l] * config.agent_vel_weight

    # -- denoising queries for object detection --
    if predictions['training'] and config.dn_detection:
        dn_md = predictions.get('dn_mask_dict', None)
        dn_lambda = getattr(config, 'dn_loss_weight', 1.0)
        dn_loss_reg = predictions['ins_labels_0'].new_zeros(1)[0]
        dn_loss_cls = predictions['ins_labels_0'].new_zeros(1)[0]
        dn_loss_vel = predictions['ins_labels_0'].new_zeros(1)[0]

        if (isinstance(dn_md, dict) and dn_md.get('pad_size', 0) > 0 and (dn_md.get('output_known_boxes') is not None) and (dn_md.get('output_known_logits') is not None)):
            boxes_all  = dn_md['output_known_boxes']           # (L, B, pad, attr)
            logits_all = dn_md['output_known_logits']          # (L, B, pad, K)
            if logits_all.shape[-1] == 1:
                logits_all = logits_all[..., 0]                # (L, B, pad)

            L, B, pad, K = boxes_all.shape
            known_bid     = dn_md['known_bid']
            map_known_idx = dn_md['map_known_indice']
            gt_abs_m_all  = dn_md['known_boxes_abs_m']        # (S*ΣN,5) [x,y,L,W,yaw]
            num_boxes = known_bid.shape[0]

            cls_targets = None
            bbox_target = gt_abs_m_all
            pos_pred_ind = torch.arange(num_boxes, device=known_bid.device)

            for l in range(L):
                dn_box_pred = boxes_all[l][known_bid, map_known_idx]     # (B,pad,K)
                dn_cls_pred = logits_all[l][known_bid, map_known_idx]    # (B,pad)

                pred_ = dn_box_pred[pos_pred_ind][:, [0,1,4,5,2,3]]
                gt_xywh = bbox_target[:, [0, 1, 3, 4]]

                gt_yaw = bbox_target[:, 2:3]
                gt_sin = gt_yaw.sin()
                gt_cos = gt_yaw.cos()
                gt_ = torch.cat([gt_xywh, gt_sin, gt_cos], dim=-1)
                l1_loss = F.smooth_l1_loss(pred_, gt_, reduction="none")
                l1_loss = l1_loss.sum(-1).mean()

                pred_vel = dn_box_pred[pos_pred_ind][:, 6:8]
                gt_vel = bbox_target[:, 5:7]
                vel_loss = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none")
                dn_loss_vel += torch.nan_to_num(vel_loss.sum(-1).mean())

                pred_logits = dn_cls_pred.flatten()
                prob = torch.sigmoid(pred_logits)

                if cls_targets is None:
                    if dn_cls_pred.dim() == 2:      # (M, K): one-hot on each noised box's class
                        cls_targets = F.one_hot(dn_md['known_labels'].long(), num_classes=dn_cls_pred.shape[-1]).to(pred_logits).flatten()
                    else:
                        cls_targets = dn_cls_pred.new_ones(pred_logits.shape)
                ce_loss = F.binary_cross_entropy_with_logits(pred_logits, cls_targets, reduction="none")
                p_t = prob * cls_targets + (1 - prob) * (1 - cls_targets)
                alpha_t = 0.25 * cls_targets + (1 - 0.25) * (1 - cls_targets)
                focal_weight = alpha_t * (1 - p_t).pow(2)

                ce_loss = focal_weight * ce_loss  # (B, N, ...)
                ce_loss = ce_loss.mean()

                dn_loss_reg += torch.nan_to_num(l1_loss)
                dn_loss_cls += torch.nan_to_num(ce_loss)

        loss_dict['dn_agent_states_loss'] = dn_lambda * dn_loss_reg * config.agent_box_weight
        loss_dict['dn_agent_label_loss']  = dn_lambda * dn_loss_cls * config.agent_class_weight
        loss_dict['dn_agent_veloicy_lss']  = dn_lambda * dn_loss_vel * config.agent_vel_weight
        loss += dn_loss_reg + dn_loss_cls + dn_loss_vel

    bev_semantic_loss = 0.0
    # -- BEV semantic map --
    if "bev_semantic_map" in predictions:
        class_weights = torch.tensor([0.2121, 0.3693, 0.8473, 0.7267, 1.8056, 1.0974, 1.9416]).to(predictions["bev_semantic_map"].device)
        bev_semantic_loss = F.cross_entropy(
            predictions["bev_semantic_map"], targets["bev_semantic_map"].long(), weight = class_weights
        )
    loss += config.bev_semantic_weight * bev_semantic_loss

    # -- future BEV semantic maps --
    if "fut_bev_map" in predictions:
        class_weights = torch.tensor([0.2121, 0.3693, 0.8473, 0.7267, 1.8056, 1.0974, 1.9416]).to(predictions["bev_semantic_map"].device)
        B, C_, H, W = predictions["fut_bev_map"].shape
        fut_bev_map = predictions["fut_bev_map"].reshape(B, config.future_bev_frames, -1, H, W )  # (B, frame, cls, H, W)

        for fut_time in range(config.future_bev_frames):
            cur_fut_bev = fut_bev_map[:, fut_time]
            cur_tar_bev = targets["all_future_frame_bev_semantic_map"][:,fut_time].long()
            fut_bev_semantic_loss = F.cross_entropy(cur_fut_bev, cur_tar_bev, weight = class_weights)
            loss += config.fut_bev_semantic_weight * fut_bev_semantic_loss
            loss_dict[f'bev_fut_loss_{fut_time}']  = config.fut_bev_semantic_weight * fut_bev_semantic_loss

    stage.update({
        'bev_semantic_loss': bev_semantic_loss,
        'gt_bboxes_list': gt_bboxes_list,
        'gt_labels_list': gt_labels_list,
        'gt_valid': gt_valid,
    })
    return loss


def _swnet_loss(targets, predictions, config, loss_dict, stage, loss,
                test_traj_save=False):
    """SWNet loss: plan imitation over the refinement layers, plus agent motion prediction.

    Adds to `loss_dict` and returns the running `loss`.
    """
    gt_bboxes_list = stage['gt_bboxes_list']
    gt_labels_list = stage['gt_labels_list']
    gt_valid = stage['gt_valid']
    assign_result = None          # only set when the motion head ran

    reg_loss = 0.0
    cls_loss = 0.0
    cls_target_list = [predictions['bev_semantic_map'].new_zeros(predictions['bev_semantic_map'].shape[0]).long()]
    num_plan_layers = config.prop_traj_n_layers + config.SWNet_num_layers

    # -- plan imitation --
    if not config.no_planning:
        for l_d in range(num_plan_layers):
            poses_reg = predictions[f'plan_traj_{l_d}']
            poses_cls = predictions[f'plan_traj_cls_{l_d}']
            plan_anchor = predictions['plan_anchor']         # (B, K, T, 3)
            if config.proposalnet_2stage and l_d >= config.prop_traj_n_layers:
                # stage 2 only kept the top anchors, so index the GT match into the same subset
                B, _, T, C = plan_anchor.shape
                topidx = predictions['proposal_2stage_topidx']            # (B, K')
                B, top_A = topidx.shape
                idx = topidx[:, :, None, None].expand(B, top_A, T, C)     # (B, K', T, 3)
                plan_anchor = plan_anchor.gather(dim=1, index=idx)        # (B, K', T, 3)

            bs, num_mode, ts, d = poses_reg.shape
            target_traj = targets["trajectory"]
            # the anchor closest to the GT trajectory is the positive class
            dist = torch.linalg.norm(target_traj.unsqueeze(1)[...,:2] - plan_anchor[..., :2], dim=-1)
            dist = dist.mean(dim=-1)
            mode_idx = torch.argmin(dist, dim=-1)                     # (B,)

            cls_target = mode_idx
            cls_target_list.append(cls_target.long())
            mode_idx = mode_idx[...,None,None,None].repeat(1,1,ts,d)
            best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
            target_classes_onehot = torch.zeros([bs, num_mode],
                                                dtype=poses_cls.dtype,
                                                layout=poses_cls.layout,
                                                device=poses_cls.device)
            target_classes_onehot.scatter_(1, cls_target.unsqueeze(1), 1)

            loss_cls =\
                config.trajectory_weight * \
                config.trajectory_cls_weight * \
                py_sigmoid_focal_loss(
                poses_cls,
                target_classes_onehot,
                weight=None,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                avg_factor=None
            )

            # regression is only on the winning mode, classification on all of them
            reg_loss = config.trajectory_weight * config.trajectory_reg_weight * F.l1_loss(best_reg, target_traj)
            loss_dict[f'trajectory_reg_{l_d}_loss'] = reg_loss
            loss_dict[f'trajectory_cls_{l_d}_loss'] = loss_cls
            loss += loss_cls + reg_loss

        # the motion head reuses these targets, but not the stage-1 layers' copies
        if config.proposalnet_2stage:
            for _ in range(config.prop_traj_n_layers):
                cls_target_list.pop(1)

    # -- agent motion prediction --
    if "agent_motion_traj_0" in predictions:
        if config.AF_topk or config.AF_det_score:
            cls_target_list[0] = cls_target_list[1]

        targets['motion_traj'] = targets['motion_traj'][..., :3]
        reg_loss, cls_loss, assign_result = motion_loss(
            config, predictions, gt_labels_list, gt_bboxes_list, targets['motion_traj'], targets['motion_mask'], gt_valid, cls_target_list
            )
        motion_layers = config.SWNet_num_layers + 1
        for l_d in range(motion_layers):
            loss_dict[f'motion_reg_{l_d}_loss'] = reg_loss[l_d] * config.prediction_loss_weight
            loss_dict[f'motion_cls_{l_d}_loss'] = cls_loss[l_d] * config.prediction_loss_weight
            loss += (reg_loss[l_d] + cls_loss[l_d]) * config.prediction_loss_weight

    stage.update({
        'assign_result': assign_result,
    })
    return loss


def _frnet_loss(targets, predictions, config, loss_dict, stage, loss,
                test_traj_save=False):
    """FRNet loss: scene-level safety, pair-wise NC / Disp and time-wise DAC.

    Adds to `loss_dict` and returns the running `loss`.
    """
    assign_result = stage['assign_result']
    gt_valid = stage['gt_valid']

    if config.scene_level_safety and not test_traj_save:
        # Rollout safety loss, final plan layer only: BCE the last-layer safety heads
        # against targets['safety_target_scores'] produced by the agent.
        # -- scene-level safety --
        if config.use_target_scores:
            if predictions['training'] and 'safety_target_scores' in targets:
                score_loss_dict = score_loss(config, predictions, targets['safety_target_scores'])
                loss_dict.update(score_loss_dict)
                for k, v in score_loss_dict.items():
                    loss += v * config.safety_score_weight
            # validation has no rollout GT, so the safety heads go unsupervised there

        # -- pair-wise no-collision --
        if config.pwnc_check and config.use_target_scores \
                and predictions['training'] and 'pair_collision_gt' in targets:
            # GT is 1 per (agent, anchor, pose); at-fault hits from the live PDM run flip it to 0
            assert 'agent_token_ids' in targets, (
                "use_target_scores + pwnc_check needs targets['agent_token_ids'] "
                "— regenerate the training cache with agent token ids")
            pred = predictions['pwnc_pred'][-1]                          # (B, N, K, T) final layer
            Bp, Np, Kp, Tp = pred.shape
            pair_nc_losses = []
            # matching is per sample, so the loss is too
            for b in range(Bp):
                ar = assign_result[b]
                pos_ind = torch.nonzero(ar.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
                if pos_ind.numel() == 0:
                    continue
                pos_gt_inds = (ar.gt_inds[pos_ind] - 1).long()
                gt_ids = targets['agent_token_ids'][b][gt_valid[b]][pos_gt_inds]   # (P,)
                id2row = {int(i): r for r, i in enumerate(gt_ids.tolist())}   # token -> matched row
                gt = pred.new_ones(pos_ind.numel(), Kp, Tp)
                for k, tokmap in enumerate(targets['pair_collision_gt'][b]):
                    for tid, pose_idcs in tokmap.items():
                        r = id2row.get(int(tid))
                        if r is not None:
                            gt[r, k, pose_idcs] = 0.0
                p_b = torch.clamp(pred[b, pos_ind], -100, 100)                # (P, K, T)
                if config.pwnc_focal_loss:
                    l_b = binary_focal_with_logits(p_b, gt, reduction='none', alpha=config.pwnc_focal_alpha)
                else:
                    l_b = F.binary_cross_entropy_with_logits(p_b, gt, reduction='none')
                # padded agent slots must not dilute the mean
                if 'agent_valid_mask' in predictions:
                    m = predictions['agent_valid_mask'][b][pos_ind][..., None].expand_as(l_b).to(l_b.dtype)
                    pair_nc_losses.append((l_b * m).sum() / m.sum().clamp_min(1.0))
                else:
                    pair_nc_losses.append(l_b.mean())
            pair_nc_loss = (sum(pair_nc_losses) / len(pair_nc_losses)) if pair_nc_losses \
                else predictions['pwnc_pred'].new_tensor(0.0)
            loss_dict['pair_NC_loss'] = pair_nc_loss * config.pair_NC_loss_weight
            loss += pair_nc_loss * config.pair_NC_loss_weight

        # -- pair-wise displacement --
        if config.pwdisp_check:
            pwdisp_pred = predictions['pwdisp_pred']   # (L, B, N, A, T, 2)
            L, B, N, A, T, _ = pwdisp_pred.shape
            pwdisp_pred = pwdisp_pred.permute(1, 0, 2, 3, 4, 5).contiguous()   # (B,L,N,A,T,2)

            plan_traj_per_batch = []
            plan_offset = getattr(config, "prop_traj_n_layers", 0)
            for b in range(B):
                perL = []
                for l in range(L):
                    perL.append(predictions[f'plan_traj_{plan_offset + l}'][b])  # (A, T, 3) (x,y in meters)
                plan_traj_per_batch.append(torch.stack(perL, dim=0))             # (L, A, T, 3)

            gt_motion_traj_list = []
            gt_motion_mask_list = []
            for b in range(B):
                gt_motion_traj_list.append(targets['motion_traj'][b][gt_valid[b]])   # (N_gt, T+1, 3)
                gt_motion_mask_list.append(targets['motion_mask'][b][gt_valid[b]])   # (N_gt, T+1)

            if 'agent_valid_mask' in predictions:
                valid_mask = predictions['agent_valid_mask'].to(dtype=torch.bool, device=pwdisp_pred.device)  # (B, N, A)
            else:
                valid_mask = torch.ones(B, N, A, dtype=torch.bool, device=pwdisp_pred.device)

            config_list = [config for _ in range(B)]

            total_pair_disp_loss = pwdisp_pred.new_tensor(0.0)

            if getattr(config, "pair_Disp_motion_GT_loss", True):
                (pair_disp_gt_loss_list,) = multi_apply(
                    pair_Disp_vec_loss,
                    config_list,
                    plan_traj_per_batch,
                    gt_motion_traj_list,
                    gt_motion_mask_list,
                    pwdisp_pred,
                    assign_result,
                    valid_mask
                )
                pair_disp_gt_loss = sum(pair_disp_gt_loss_list)
                loss_dict['pair_Disp_loss_GT'] = pair_disp_gt_loss * config.pair_Disp_loss_weight
                total_pair_disp_loss = total_pair_disp_loss + loss_dict['pair_Disp_loss_GT']

            if (getattr(config, "motion_GT_disp_loss", True) or getattr(config, "motion_Pred_disp_loss", False)):
                loss += total_pair_disp_loss
                loss_dict['pair_Disp_loss'] = total_pair_disp_loss

        # -- time-wise drivable-area compliance --
        if config.twdac_check and config.use_target_scores \
                and predictions['training'] and 'twdac_gt' in targets:
            # GT is 1 per (anchor, pose); poses outside the drivable area flip it to 0
            twdac_pred = predictions['twdac_pred']            # (L, B, K, T)
            twdac_pred = twdac_pred[-1] if twdac_pred.dim() == 4 else twdac_pred  # final layer (B,K,T)
            Bp, Kp, Tp = twdac_pred.shape
            gt = twdac_pred.new_ones(Bp, Kp, Tp)                      # 1 = compliant / on-road
            for b in range(Bp):
                for k, pose_idcs in enumerate(targets['twdac_gt'][b]):
                    if pose_idcs:
                        gt[b, k, pose_idcs] = 0.0                     # off-road pose -> 0
            p = torch.clamp(twdac_pred, -100, 100)
            if config.twdac_focal_loss:
                twdac_loss = binary_focal_with_logits(p, gt, reduction='mean', alpha=config.time_size_DAC_focal_alpha)
            else:
                twdac_loss = F.binary_cross_entropy_with_logits(p, gt, reduction='mean')
            loss_dict['TwDAC_loss'] = twdac_loss * config.twdac_loss_weight
            loss += twdac_loss * config.twdac_loss_weight
    return loss


def pair_Disp_vec_loss(config,
                       plan_traj_layers,        # (L, A, T, 3)
                       gt_motion_traj_b,        # (N_gt, T+1, 3)
                       gt_motion_mask_b,        # (N_gt, T+1)
                       pair_Disp_pred_b,        # (L, N, A, T, 2)
                       assign_result_b,         # AssignResult for this batch
                       valid_mask_b             # (N, A)
                       ):
    """
    Prediction: per-timestep agent displacement (dx, dy) relative to the plan.
    GT: disp_gt = plan_xy - agent_xy, same frame, in meters.

    - only positive queries, so the query axis lines up with the GT axis
    - averaged under the time mask and the (N, A) valid mask
    """
    device = plan_traj_layers.device
    L, A, T, _ = plan_traj_layers.shape
    _, N, A_pred, T_pred, dim = pair_Disp_pred_b.shape
    assert A_pred == A and T_pred == T, "shape mismatch in pair_Disp_pred_b"

    # positive query
    pos_ind = torch.nonzero(assign_result_b.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    if pos_ind.numel() == 0:
        return (pair_Disp_pred_b.new_tensor(0.0),)

    pos_gt_inds = (assign_result_b.gt_inds[pos_ind] - 1).long()  # (P,)

    # GT agent XY & mask (t=1..T)
    agent_xy   = gt_motion_traj_b[pos_gt_inds, 1:, :2]      # (P, T, 2)
    agent_yaw   = gt_motion_traj_b[pos_gt_inds, 1:, 2:]      # (P, T, 2)
    agent_mask = gt_motion_mask_b[pos_gt_inds, 1:]          # (P, T), bool
    P = agent_xy.shape[0]

    # (N, A) → (P, A)
    vm = valid_mask_b[pos_ind] if valid_mask_b is not None else torch.ones(P, A, dtype=torch.bool, device=device)
    if not vm.any():
        return (pair_Disp_pred_b.new_tensor(0.0),)

    losses = []
    for l in range(L):
        # plan xy: (A, T, 2)
        # displacement
        plan_xy = plan_traj_layers[l, :, :, :2]             # meters
        disp_gt = agent_xy[:, None, :, :] - plan_xy[None, :, :, :]   # (P, A, T, 2)

        if config.pair_Disp_motion_with_yaw:
            disp_yaw = agent_yaw[:, None, :, :] - plan_traj_layers[l, None, :, :, 2:]
            disp_gt = torch.cat([disp_gt, disp_yaw.sin(), disp_yaw.cos()],dim=-1)

        disp_pred = pair_Disp_pred_b[l, pos_ind, :, :, :]            # (P, A, T, 2)

        time_mask = agent_mask[:, None, :, None]                     # (P,1,T,1)
        pair_mask = vm[:, :, None, None]                             # (P,A,1,1)
        m = (time_mask & pair_mask)                                  # (P,A,T,1)

        if not m.any():
            continue

        diff = F.smooth_l1_loss(disp_pred, disp_gt, reduction='none')
        diff = diff * m.to(diff.dtype)                               # (P,A,T,2)
        num = (m.to(diff.dtype)).sum()  # number of valid elements
        loss_l = diff.sum() / (num + 1e-6)
        losses.append(loss_l)

    if len(losses) == 0:
        return (pair_Disp_pred_b.new_tensor(0.0),)

    return (sum(losses) / len(losses), )


def pair_NC_loss_GT_calculate(config,
                       plan_traj_layers,
                       gt_motion_traj_b,
                       gt_motion_mask_b,
                       gt_agent_states_b,
                       pair_NC_pred_b,
                       assign_result_b,
                       valid_mask_b,
                       gt_bev_b,
                       plan_traj_b,
                       current_epoch,
                       ):
    """Pair-wise no-collision GT from box overlap between the plan and each agent."""
    margin_type = config.margin_type
    ins_box_margin = config.ins_box_margin
    ego_box_margin = config.ego_box_margin

    device = plan_traj_layers.device
    L, K, T, _ = plan_traj_layers.shape
    pos_ind = torch.nonzero(assign_result_b.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    if pos_ind.numel() == 0:
        return (pair_NC_pred_b.new_tensor(0.0),)
    pos_gt_inds = (assign_result_b.gt_inds[pos_ind] - 1).long()  # (P,)

    agent_traj   = gt_motion_traj_b[None, pos_gt_inds, None, 1:].repeat(L,1,K,1,1)
    agent_mask = gt_motion_mask_b[None, pos_gt_inds, None, 1:].repeat(L,1,K,1).bool()
    agent_states = gt_agent_states_b[None, pos_gt_inds, None, None].repeat(L,1,K,T,1)

    P = agent_traj.shape[1]

    vm = valid_mask_b[pos_ind].bool() if valid_mask_b is not None else torch.ones(P, K, dtype=torch.bool, device=device)
    if not vm.any():
        return (pair_NC_pred_b.new_tensor(0.0),)

    agent_motion_box = torch.cat([agent_traj, agent_states[..., 3:5]],dim=-1)
    if margin_type == 'fixed':
        agent_motion_box[..., 3] *= ins_box_margin[1]
        agent_motion_box[..., 4] *= ins_box_margin[0]
    elif margin_type == 'distance':
        ins_box_margin_ratio = (torch.norm(agent_motion_box[..., :2], dim=-1) / 10.0).clamp(min=1.0)
        ins_box_margin_width = ins_box_margin_ratio.clamp(max=ins_box_margin[0])
        ins_box_margin_length = ins_box_margin_ratio.clamp(max=ins_box_margin[1])
        agent_motion_box[..., 3] *= ins_box_margin_length
        agent_motion_box[..., 4] *= ins_box_margin_width

    x, y, yaw, w, h = agent_motion_box[..., 0], agent_motion_box[..., 1], -agent_motion_box[..., 2], agent_motion_box[..., 3], agent_motion_box[..., 4]
    dx = w / 2
    dy = h / 2
    corners_local = torch.stack([torch.stack([dx, dy], dim=-1),torch.stack([dx, -dy], dim=-1), torch.stack([-dx, -dy], dim=-1), torch.stack([-dx, dy], dim=-1)], dim=-2)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rot_mat = torch.stack([torch.stack([cos_yaw, -sin_yaw], dim=-1),torch.stack([sin_yaw,  cos_yaw], dim=-1)], dim=-2)  # shape : [4, 11, 8, 2, 2]
    rotated_corners = torch.matmul(corners_local, rot_mat)
    corners_world = rotated_corners + torch.stack([x, y], dim=-1)[..., None, :]

    # planning boxes
    half_width = 1.1485 * ego_box_margin[0]
    front_length=4.049 * ego_box_margin[1]
    rear_length=1.127 * ego_box_margin[1]
    ego_corners = torch.tensor([[half_width, front_length], [-half_width, front_length], [-half_width, -rear_length], [half_width, -rear_length]]).to(agent_states)
    ego_corners = torch.tensor([[front_length, half_width], [front_length, -half_width], [-rear_length, -half_width],[-rear_length, half_width] ]).to(agent_states)
    ego_corners = ego_corners[None, None, None].repeat(L, K, T, 1, 1)
    ego_yaw = -plan_traj_layers[..., 2]
    cos_yaw = torch.cos(ego_yaw)
    sin_yaw = torch.sin(ego_yaw)
    rot_mat = torch.stack([torch.stack([cos_yaw, -sin_yaw], dim=-1),torch.stack([sin_yaw,  cos_yaw], dim=-1)], dim=-2)
    rotated_corners = torch.matmul(ego_corners, rot_mat)
    ego_corners_world = rotated_corners + plan_traj_layers[..., None, :2]

    device = corners_world.device
    B, A, K, T = corners_world.shape[:4]
    ego_rep = ego_corners_world.unsqueeze(1).expand(-1, A, -1, -1, -1, -1)      # (L, A, K, T, 4, 2)
    agent_corners = corners_world                                              # (L, A, K, T, 4, 2)
    edges_ego   = ego_rep[..., [1, 2], :] - ego_rep[..., [0, 1], :]            # (L, A, K, T, 2, 2)
    edges_agent = agent_corners[..., [1, 2], :] - agent_corners[..., [0, 1], :]
    perp = lambda v: torch.stack([-v[..., 1], v[..., 0]], dim=-1)
    axes_ego   = torch.stack([perp(edges_ego[..., 0, :]), perp(edges_ego[..., 1, :])], dim=-2)   # (L, A, K, T, 2, 2)
    axes_agent = torch.stack([perp(edges_agent[..., 0, :]), perp(edges_agent[..., 1, :])], dim=-2)
    all_axes = torch.cat([axes_ego, axes_agent], dim=-2)                     # (L, A, K, T, 4, 2)
    axes_norm = all_axes / (torch.norm(all_axes, dim=-1, keepdim=True) + 1e-8)
    proj_ego = (ego_rep.unsqueeze(-3) * axes_norm.unsqueeze(-2)).sum(dim=-1)     # (L, A, K, T, 4, 4)
    proj_agent = (agent_corners.unsqueeze(-3) * axes_norm.unsqueeze(-2)).sum(dim=-1)
    ego_min, ego_max = proj_ego.min(dim=-1).values, proj_ego.max(dim=-1).values  # (L, A, K, T, 4)
    ag_min, ag_max   = proj_agent.min(dim=-1).values, proj_agent.max(dim=-1).values
    overlap_each_axis = (ego_max >= ag_min) & (ag_max >= ego_min)  # (L, A, K, T, 4)
    collide_mask_time = overlap_each_axis.all(dim=-1)                   # (L, A, K, T)
    collide_mask_time[~agent_mask] = 0

    no_collide = ~collide_mask_time
    pair_mask = vm.unsqueeze(0).unsqueeze(3)            # (1,P,A,1,1)
    m = (agent_mask & pair_mask).to(torch.float32)                    # (1,P,A,T,1)

    if getattr(config, "pair_NC_class_weighting", False):  # up-weight the rare collision class
        pos_w = getattr(config, "pair_NC_pos_weight", 5.0)  # up-weight the rare collision class
        neg_w = getattr(config, "pair_NC_neg_weight", 1.0)
        w = torch.where(no_collide < 0.5,
                        torch.tensor(pos_w, device=no_collide.device, dtype=no_collide.dtype),
                        torch.tensor(neg_w, device=no_collide.device, dtype=no_collide.dtype))
        m = w * m

    denom = m.sum()                                                  # scalar

    disp_NC_all = pair_NC_pred_b[:, pos_ind]     # (L,P,A,T,Dpred)
    disp_NC_all = torch.clamp(disp_NC_all, -100, 100)
    if config.pwnc_focal_loss:
        pair_NC_loss = binary_focal_with_logits(disp_NC_all, no_collide.float(), reduction='none', alpha=config.pwnc_focal_alpha)
    else:
        pair_NC_loss = F.binary_cross_entropy_with_logits(disp_NC_all, no_collide.float(), reduction='none')

    pair_NC_loss = (pair_NC_loss * m).sum()/(denom+1e-6)

    return (pair_NC_loss,)

def pair_NC_loss(config, agent_id, colli_pad, pwnc_pred, assign_result, valid_mask):
    """
    agent_id   : (N_gt,)
    colli_pad    : (L, A, M) padded ids of the colliding agents per layer and anchor
    pwnc_pred : (L, N, A) prediction for this batch (multi_apply splits the B axis)
    assign_result: AssignResult for this batch
    valid_mask   : (N, A) valid mask for this batch
    """
    # positive query index
    pos_ind = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    if pos_ind.numel() == 0:
        return (pwnc_pred.new_tensor(0.0),)

    pos_assigned_gt_inds = (assign_result.gt_inds[pos_ind] - 1).long()
    pos_gt_ids = agent_id[pos_assigned_gt_inds]  # (P,)

    pair_NC_pos = pwnc_pred[:, pos_ind, :]

    L, A, M = colli_pad.shape
    pos_ids_broadcast = pos_gt_ids[None, :, None, None]               # (1, P, 1, 1)
    # (L, A, M) -> (L, 1, A, M)
    colli_broadcast   = colli_pad[:, None, :, :]                      # (L, 1, A, M)
    gt_pair_NC = (pos_ids_broadcast == colli_broadcast).any(dim=-1)   # (L, P, A)

    # focal / bce
    pair_NC_loss = F.binary_cross_entropy_with_logits(pair_NC_pos, gt_pair_NC.float(), reduction='none')  # (L,P,A)
    # valid mask: (N, A) -> (P, A) -> (L,P,A)
    vm = valid_mask[pos_ind]  # (P, A)
    vm = vm[None, ...].expand(L, -1, -1).to(pair_NC_loss.dtype)

    pair_NC_loss = torch.nan_to_num(pair_NC_loss)
    denom = vm.sum().clamp_min(1.0)
    pair_NC_loss = (pair_NC_loss * vm).sum() / denom
    pair_NC_loss = torch.nan_to_num(pair_NC_loss)

    return (pair_NC_loss, )


def binary_focal_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction="mean", eps=1e-6):
    """
    logits: (...,) raw scores
    targets: (...,) in {0,1}
    """
    # CE with logits (per-element)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)        # p_t = p if y=1 else 1-p
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    focal = alpha_t * (1 - p_t).clamp(min=eps).pow(gamma) * ce
    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    else:
        return focal  # "none"


def _three_to_two(x):
    """PDM 3-level {0, 0.5, 1} -> binary (0.5 -> 0). Mirrors EAD_navsim ead_loss_simplebev."""
    x = x.clone()
    x[x == 0.5] = 0.0
    return x


def score_loss(config, predictions, target_score):
    """Safety-score BCE against the simulator-scored GT (final plan layer only).

    Mirrors EAD_navsim's per-proposal BCE block (ead_loss_simplebev.py:734-778):
      - rollout GT column order: pdms  (B,K,7) = [NC, DAC, EP, TTC, comfort, DDC, final]
                                 epdms (B,K,9) = [NC, DAC, EP, TTC, comfort, DDC, TLC, LK, final]
      - NC / DDC: 3-level {0,0.5,1} collapsed to binary (0.5 -> 0)
      - TTC: 2.0 is an invalid/unscored sentinel -> masked out of the mean
    Predictions are the model safety heads sliced to the final plan layer ([-1] -> (B,K)).
    Uses GRAD per-metric loss weights. HC is intentionally NOT supervised (no rollout GT)."""
    # proposalnet_2stage pads 128 proposals to 256 anchors; only the first 128 rows
    # are real, so slice the head to match the rollout GT.
    K_valid = target_score.shape[1]
    def head_last(name):
        return predictions[name][-1][:, :K_valid]  # (L,B,Kpad) -> (B,K_valid), final refined plan layer

    gt = target_score  # (B, K, C)
    s_dtype = predictions['NC'].dtype
    gt_nc, gt_dac, gt_ep = gt[..., 0].to(s_dtype), gt[..., 1].to(s_dtype), gt[..., 2].to(s_dtype)
    gt_ttc, gt_comfort, gt_ddc = gt[..., 3].to(s_dtype), gt[..., 4].to(s_dtype), gt[..., 5].to(s_dtype)

    loss_dict = {}

    nc_loss = F.binary_cross_entropy_with_logits(head_last('NC'), _three_to_two(gt_nc)) * config.NC_loss_weight
    loss_dict['NC_loss'] = nc_loss

    dac_loss = F.binary_cross_entropy_with_logits(head_last('DAC'), gt_dac) * config.DAC_loss_weight
    loss_dict['DAC_loss'] = dac_loss

    ttc_mask = (gt_ttc != 2.0).to(s_dtype)  # 2.0 = invalid/unscored
    ttc_loss = (F.binary_cross_entropy_with_logits(head_last('TTC'), gt_ttc, ttc_mask, reduction='sum')
                / ttc_mask.sum().clamp(min=1.0)) * config.TTC_loss_weight
    loss_dict['TTC_loss'] = ttc_loss

    ep_loss = F.binary_cross_entropy_with_logits(head_last('EP'), gt_ep) * config.EP_loss_weight
    loss_dict['EP_loss'] = ep_loss

    if 'C' in predictions:
        c_loss = F.binary_cross_entropy_with_logits(head_last('C'), gt_comfort) * config.C_loss_weight
        loss_dict['C_loss'] = c_loss

    if 'DDC' in predictions:
        ddc_loss = F.binary_cross_entropy_with_logits(head_last('DDC'), _three_to_two(gt_ddc)) * config.DDC_loss_weight
        loss_dict['DDC_loss'] = ddc_loss

    # pdm_score head: supervised against the rollout's final aggregate (last GT column)
    if 'pdm_score' in predictions:
        pdm_loss = F.binary_cross_entropy_with_logits(head_last('pdm_score'), gt[..., -1].to(s_dtype)) * config.pdm_score_loss_weight
        loss_dict['pdm_score_loss'] = pdm_loss

    # EPDMS extras: TLC (col 6) + LK (col 7) present only when scoring_mode == "epdms" (C >= 9)
    if gt.shape[-1] >= 9:
        if 'TLC' in predictions:
            tlc_loss = F.binary_cross_entropy_with_logits(head_last('TLC'), gt[..., 6].to(s_dtype)) * config.TLC_loss_weight
            loss_dict['TLC_loss'] = tlc_loss
        if 'LK' in predictions:
            lk_loss = F.binary_cross_entropy_with_logits(head_last('LK'), gt[..., 7].to(s_dtype)) * config.LK_loss_weight
            loss_dict['LK_loss'] = lk_loss

    return loss_dict


# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
def motion_loss(config, predictions, gt_labels_list, gt_bboxes_list, gt_motion_traj, gt_motion_mask, gt_valid, cls_target_list):
    """Motion regression and classification against the matched agents."""
    l_d = config.bevformer_decoder['num_layers'] - 1
    num_batches = predictions['ins_labels_0'].shape[0]
    agent_labels = predictions[f'ins_labels_{l_d}']
    agent_states = predictions[f'ins_states_{l_d}']
    gt_motion_traj = [gt_motion_traj[b][gt_valid[b]] for b in range(num_batches)]
    gt_motion_mask = [gt_motion_mask[b][gt_valid[b]] for b in range(num_batches)]

    if predictions['training'] or predictions['visualize'] == True:
        idx_src = cls_target_list[0]
        b_idcs = torch.arange(len(idx_src), device=idx_src.device).long()
        pred_motion_traj = predictions['agent_motion_traj_0'][b_idcs, :, :, idx_src]
    else:
        pred_motion_traj = predictions['agent_motion_traj_0']
    (labels_list, label_weights_list, traj_targets_list, traj_weights_list,
    pos_inds, neg_inds, traj_masks_list, assign_result) = multi_apply(
        _get_motion_target_single,
        agent_labels, agent_states,
        gt_labels_list, gt_bboxes_list,
        gt_motion_traj,
        gt_motion_mask,
        pred_motion_traj
    )
    motion_layers = config.SWNet_num_layers + 1
    if predictions['training'] or predictions['visualize'] == True:
        if config.all_motion_predidction_loss:
            all_pred_trajs = [predictions[f'agent_motion_traj_{l_d}'] for l_d in range(motion_layers)]
        else:
            all_pred_trajs = [predictions[f'agent_motion_traj_{l_d}'][b_idcs, :,:,cls_target_list[l_d]] for l_d in range(motion_layers)]
        if 'agent_motion_cls_0' in predictions:
            all_pred_traj_scores = [predictions[f'agent_motion_cls_{l_d}'][b_idcs, :,:,cls_target_list[l_d]] for l_d in range(motion_layers)]
        else:
            all_pred_traj_scores = [None for l_d in range(motion_layers)]
    else:
        if config.all_motion_predidction_loss:
            all_pred_trajs = [predictions[f'agent_motion_traj_{l_d}'][:,:,:,None] for l_d in range(motion_layers)]
        else:
            all_pred_trajs = [predictions[f'agent_motion_traj_{l_d}'] for l_d in range(motion_layers)]
        if 'agent_motion_cls_0' in predictions:
            all_pred_traj_scores = [predictions[f'agent_motion_cls_{l_d}'] for l_d in range(motion_layers)]
        else:
            all_pred_traj_scores = [None for l_d in range(motion_layers)]

    all_traj_targets_list = [traj_targets_list for l_d in range(motion_layers)]
    all_traj_masks_list = [traj_masks_list for l_d in range(motion_layers)]
    all_config = [config for _ in range(motion_layers)]

    pos_weight_full_layers = [None for _ in range(motion_layers)]

    if 'agent_valid_mask' in predictions:
        agent_filtering_mask = predictions['agent_valid_mask']
        if config.all_motion_predidction_loss:
            agent_filtering_masks = [agent_filtering_mask for l_d in range(motion_layers)]
        else:
            if len(agent_filtering_mask.shape) == 2:
                agent_filtering_masks = [agent_filtering_mask for l_d in range(motion_layers)]
            else:
                idx_src = cls_target_list[0]
                b_idcs = torch.arange(len(idx_src), device=idx_src.device).long()
                agent_filtering_masks = [agent_filtering_mask[b_idcs, :, cls_target_list[l_d]] for l_d in range(motion_layers)]
    else:
        agent_filtering_masks = [None for l_d in range(motion_layers)]

    reg_losses, cls_losses = multi_apply(
        _motion_loss_single,
        all_pred_trajs, all_pred_traj_scores,
        all_traj_targets_list, all_traj_masks_list,
        all_config, agent_filtering_masks, pos_weight_full_layers
    )

    return reg_losses, cls_losses, assign_result


def _motion_loss_single(pred_trajs, pred_traj_scores, traj_targets_list, traj_masks_list,
                        config, agent_filtering_mask, pos_weight_full_list=None):
    """
    pos_weight_full_list: list of length B, each entry [N] or None.
    """
    traj_targets = torch.stack(traj_targets_list, 0)         # (B, N, T+1, D)
    traj_masks   = torch.stack(traj_masks_list, 0)[:, :, 1:] # (B, N, T)
    traj_targets = (traj_targets - traj_targets[:, :, 0:1])[:, :, 1:]  # (B, N, T, D)

    # select best prediction and mode

    if config.all_motion_predidction_loss:
        bs, num_agent, mode, n_anchor, ts, d = pred_trajs.shape
        traj_targets = traj_targets[:,:,None].repeat(1,1,n_anchor,1,1)
        traj_masks = traj_masks[:,:,None].repeat(1,1,n_anchor,1)
    else:
        bs, num_agent, mode, ts, d = pred_trajs.shape

    if mode != 1:
        dist = torch.linalg.norm(traj_targets.unsqueeze(2)[...,:2] - pred_trajs[...,:2], dim=-1)  # (B, N, M, T)
        dist = dist * traj_masks.unsqueeze(2)                    # mask
        dist = dist.mean(dim=-1)                                 # (B, N, M)
        mode_idx = torch.argmin(dist, dim=-1)                    # (B, N)
        mode_idx_for_traj = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
        best_reg = torch.gather(pred_trajs, 2, mode_idx_for_traj).squeeze(2)  # (B, N, T, D)
    else:
        best_reg = pred_trajs[:,:,0]

    reg_loss = F.l1_loss(best_reg, traj_targets, reduction='none')    # (B, N, T, D)
    reg_loss = (reg_loss * traj_masks[..., None]).sum(-1)                  # (B, N, T)

    if pos_weight_full_list is not None:
        posW = torch.stack(pos_weight_full_list, 0)  # (B, N)
        reg_loss = reg_loss * posW[:, :, None]  # (B, N, T)

        denom = traj_masks
        if agent_filtering_mask is not None:
            denom = denom * agent_filtering_mask[:, :, None]
        denom = denom * posW[:, :, None]
        reg_loss = reg_loss.sum() / (denom.sum() + 1e-6)
    else:
        if agent_filtering_mask is not None:
            reg_loss = reg_loss * agent_filtering_mask[..., None]
            reg_loss = reg_loss.sum() / ((traj_masks * agent_filtering_mask[..., None]).sum() + 1e-6)
        else:
            reg_loss = reg_loss.sum() / (traj_masks.sum() + 1e-6)

    # focal loss for classification
    gamma = 2.0
    alpha = 0.25
    if pred_traj_scores is not None:
        ce_loss = F.cross_entropy(pred_traj_scores[traj_masks[:, :, 0]], mode_idx[traj_masks[:, :, 0]], reduction='none')  # [N']
        pt = torch.exp(-ce_loss)  # pt = softmax probability of correct class
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        cls_loss = focal_loss.mean()
    else:
        cls_loss = 0
    return reg_loss, cls_loss


@torch.no_grad()
def _focal_cost_multiclass(gt_labels: torch.Tensor,
                           cls_logits: torch.Tensor,
                           alpha=0.25, gamma=2.0) -> torch.Tensor:
    """
    Sigmoid focal cost, multi-label style.
    cls_logits: [Nq, K+1], last column is background
    gt_labels : [Ng] in {0..K-1}
    return    : [Nq, Ng]
    """
    Nq, C_total = cls_logits.shape
    K_fg = C_total
    logits = cls_logits[:, :K_fg]  # foreground only: [Nq, K]
    Ng = gt_labels.numel()
    tgt_onehot = F.one_hot(gt_labels.clamp_min(0), num_classes=K_fg).to(logits.dtype)  # (Ng, K)

    l = logits[:, None, :].expand(Nq, Ng, K_fg)
    t = tgt_onehot[None, :, :].expand(Nq, Ng, K_fg)

    ce   = F.binary_cross_entropy_with_logits(l, t, reduction='none')
    p    = torch.sigmoid(l)
    p_t  = p * t + (1 - p) * (1 - t)
    a_t  = alpha * t + (1 - alpha) * (1 - t)
    cost = (a_t * (1 - p_t).clamp(min=1e-6).pow(gamma) * ce).sum(dim=-1)   # (Nq, Ng)
    return cost


def _batched_detection_targets(pred_logits, pred_states, gt_labels_list, gt_bboxes_list):
    """Hungarian detection targets for the whole batch with one device-to-host copy.

    Equivalent to _get_target_single per sample (kept as the reference implementation): the same cost functions
    and the same scipy solve, but the cost matrices go to the CPU in one copy and the targets are scattered with
    one index copy, instead of several host syncs per sample per decoder layer.
    """
    Bsz, Q = pred_states.shape[:2]
    device = pred_states.device
    multi = pred_logits.dim() == 3 and pred_logits.shape[-1] > 1
    bg = pred_logits.shape[-1] if multi else 0
    n_attr = pred_states.shape[-1]
    gt_c = gt_bboxes_list[0].shape[-1] if Bsz else n_attr
    n_gts = [int(g.shape[0]) for g in gt_bboxes_list]

    labels = torch.full((Bsz * Q,), bg, dtype=torch.long, device=device)
    bbox_targets = torch.zeros((Bsz * Q, min(gt_c, n_attr)), dtype=gt_bboxes_list[0].dtype if Bsz else pred_states.dtype, device=device)
    bbox_weights = torch.zeros((Bsz * Q, n_attr), dtype=pred_states.dtype, device=device)
    if sum(n_gts) == 0:
        return labels, bbox_targets, bbox_weights

    costs, shapes = [], []
    for b in range(Bsz):
        if n_gts[b] == 0:
            continue
        if multi:
            cls_cost = _focal_cost_multiclass(gt_labels_list[b], pred_logits[b], alpha=0.25, gamma=2.0)
            reg_cost = _get_l1_cost_single(gt_bboxes_list[b], pred_states[b], gt_labels_list[b] >= 0)
        else:
            cls_cost = _get_focal_cost_single(gt_labels_list[b], pred_logits[b])
            reg_cost = _get_l1_cost_single(gt_bboxes_list[b], pred_states[b], gt_labels_list[b])
        cost = torch.nan_to_num(cls_cost + reg_cost, nan=1e6, posinf=1e6, neginf=1e6).detach()
        costs.append(cost.reshape(-1)); shapes.append((b, cost.shape))
    flat = torch.cat(costs).cpu()                                   # the one device-to-host sync

    rows, gts, offset, pos = [], [], [0], 0
    for n in n_gts:
        offset.append(offset[-1] + n)
    for b, shp in shapes:
        numel = shp[0] * shp[1]
        r, c = linear_sum_assignment(flat[pos:pos + numel].view(shp))
        pos += numel
        rows.append(torch.from_numpy(r) + b * Q)
        gts.append(torch.from_numpy(c) + offset[b])
    idx = torch.stack([torch.cat(rows), torch.cat(gts)]).to(device)   # one host-to-device copy
    gt_l = torch.cat(gt_labels_list); gt_b = torch.cat(gt_bboxes_list)
    labels[idx[0]] = gt_l[idx[1]].to(labels.dtype)
    bbox_targets[idx[0]] = gt_b[idx[1]][..., :bbox_targets.shape[-1]].to(bbox_targets.dtype)
    bbox_weights[idx[0]] = 1.0
    return labels, bbox_targets, bbox_weights


def _agent_loss_single(
    pred_states, pred_logits, gt_states, gt_valid, config
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """
    labels, bbox_targets, bbox_weights = _batched_detection_targets(pred_logits, pred_states, gt_valid, gt_states)

    # regression L1 loss
    pred_states = pred_states.reshape(-1, pred_states.size(-1))

    pred_ = pred_states[:, [0,1,4,5,2,3]]
    gt_xywh = bbox_targets[:, [0, 1, 3, 4]]

    gt_yaw = bbox_targets[:, 2:3]
    gt_sin = gt_yaw.sin()
    gt_cos = gt_yaw.cos()
    gt_ = torch.cat([gt_xywh, gt_sin, gt_cos], dim=-1)
    l1_loss = F.smooth_l1_loss(pred_, gt_, reduction="none")
    l1_loss = l1_loss.sum(-1) * bbox_weights[:,0]
    l1_loss = l1_loss.sum() / bbox_weights[:,0].sum()
    vel_loss = None
    pred_vel = pred_states[:, 6:8]
    gt_vel = bbox_targets[:, 5:7]
    vel_loss = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none")
    vel_loss = vel_loss.sum(-1) * bbox_weights[:,0]
    vel_loss = vel_loss.sum() / bbox_weights[:,0].sum()

    if pred_logits.dim() == 3:
        # det_pedestrian: one sigmoid focal term per class on one-hot targets (background = all zeros),
        # summed over classes so a query's loss stays on the scale of the 1-logit head
        K = pred_logits.shape[-1]
        pred_logits = pred_logits.reshape(-1, K)
        gt = F.one_hot(labels.long(), num_classes=K + 1)[:, :K].to(pred_logits)
        prob = torch.sigmoid(pred_logits)
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt, reduction="none")
        p_t = prob * gt + (1 - prob) * (1 - gt)
        alpha_t = 0.25 * gt + (1 - 0.25) * (1 - gt)
        focal_weight = alpha_t * (1 - p_t).pow(2)
        ce_loss = (focal_weight * ce_loss).sum(-1).mean()
    else:
        labels = labels.to(pred_logits)
        pred_logits = pred_logits.flatten()
        prob = torch.sigmoid(pred_logits)
        gt = labels.float()

        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, labels, reduction="none")
        p_t = prob * gt + (1 - prob) * (1 - gt)
        alpha_t = 0.25 * gt + (1 - 0.25) * (1 - gt)
        focal_weight = alpha_t * (1 - p_t).pow(2)

        ce_loss = focal_weight * ce_loss  # (B, N, ...)
        ce_loss = ce_loss.mean()

    l1_loss = torch.nan_to_num(l1_loss)
    ce_loss = torch.nan_to_num(ce_loss)
    return ce_loss, l1_loss, vel_loss

def _get_target_single(cls_score, bbox_pred, gt_labels, gt_bboxes):
    """Hungarian-matched detection targets for one sample."""
    num_bboxes = bbox_pred.size(0)
    # assigner and sampler
    gt_c = gt_bboxes.shape[-1]

    def assign(bbox_pred, cls_pred, gt_bboxes, gt_labels):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        if cls_pred.ndim > 1 and cls_pred.shape[-1] > 1:   # det_pedestrian: gt_labels are class indices
            cls_cost = _focal_cost_multiclass(gt_labels, cls_pred, alpha=0.25, gamma=2.0)
            reg_cost = _get_l1_cost_single(gt_bboxes, bbox_pred, gt_labels >= 0)
        else:
            cls_cost = _get_focal_cost_single(gt_labels, cls_pred)

            # regression L1 cost
            reg_cost = _get_l1_cost_single(gt_bboxes, bbox_pred, gt_labels)

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds].to(assigned_labels)
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

    assign_result = assign(bbox_pred, cls_score, gt_bboxes, gt_labels)
    pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
    gt_flags = bbox_pred.new_zeros(bbox_pred.shape[0], dtype=torch.uint8)
    sampling_result = SamplingResult(pos_inds, neg_inds, bbox_pred, gt_bboxes, assign_result, gt_flags)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds

    # label targets
    # background: 0 for the 1-logit head (targets are 0/1); num_classes for the class-index head
    bg = cls_score.shape[-1] if (cls_score.ndim > 1 and cls_score.shape[-1] > 1) else 0
    labels = gt_bboxes.new_full((num_bboxes,), bg, dtype=torch.long)
    labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].to(labels)
    label_weights = gt_bboxes.new_ones(num_bboxes)

    # bbox targets
    bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c].to(sampling_result.pos_gt_bboxes)
    bbox_weights = torch.zeros_like(bbox_pred)
    bbox_weights[pos_inds] = 1.0

    # DETR
    n_attr = bbox_pred.shape[-1]
    if pos_inds.shape[0] != 0:
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes[..., :n_attr]
    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)


def _get_motion_target_single(cls_score, bbox_pred, gt_labels, gt_bboxes, gt_motion_traj, gt_motion_mask, pred_motion_traj):
    """Motion targets for one sample, reusing the detection matching."""
    num_bboxes = bbox_pred.size(0)
    # assigner and sampler

    def assign(bbox_pred, cls_pred, gt_bboxes, gt_labels):
        multi_class = cls_pred.ndim > 1 and cls_pred.shape[-1] > 1
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ), -1, dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        if multi_class:
            cls_cost = _focal_cost_multiclass(gt_labels, cls_pred, alpha=0.25, gamma=2.0)  # (Nq, Ng)
            reg_cost = _get_l1_cost_single(gt_bboxes, bbox_pred, gt_labels >= 0)
        else:
            cls_cost = _get_focal_cost_single(gt_labels, cls_pred)
            reg_cost = _get_l1_cost_single(gt_bboxes, bbox_pred, gt_labels)

        cost = cls_cost + reg_cost

        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(bbox_pred.device)

        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds].to(assigned_labels)
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

    assign_result = assign(bbox_pred, cls_score, gt_bboxes, gt_labels)

    pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
    neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
    gt_flags = pred_motion_traj.new_zeros(pred_motion_traj.shape[0], dtype=torch.uint8)

    sampling_result = SamplingResult(pos_inds, neg_inds, pred_motion_traj, gt_motion_traj, assign_result, gt_flags)
    sampling_result_mask = SamplingResult(pos_inds, neg_inds, pred_motion_traj, gt_motion_mask, assign_result, gt_flags)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds

    # label targets
    labels = gt_motion_traj.new_full((num_bboxes,), 0, dtype=torch.long)
    labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds].to(labels)
    label_weights = gt_motion_traj.new_ones(num_bboxes)

    # bbox targets
    n_ag, n_mode, n_fut, n_att = pred_motion_traj.shape
    traj_targets = torch.zeros(n_ag, n_fut+1, n_att).to(sampling_result.pos_gt_bboxes)
    traj_masks = torch.zeros(n_ag, n_fut+1).to(sampling_result_mask.pos_gt_bboxes)
    traj_weights = torch.zeros(n_ag, n_fut+1, n_att)
    traj_weights[pos_inds] = 1.0

    if pos_inds.shape[0] != 0:
        traj_targets[pos_inds] = sampling_result.pos_gt_bboxes
        traj_masks[pos_inds] = sampling_result_mask.pos_gt_bboxes
    return (labels, label_weights, traj_targets, traj_weights, pos_inds, neg_inds, traj_masks, assign_result)


@torch.no_grad()
def _get_focal_cost_single(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    gt_valid_exp = gt_valid[:, None].detach().float()     # (B, N, 1)
    pred_logits_exp = pred_logits[None, :].detach()       # (B, 1, N)

    prob = pred_logits_exp.sigmoid()                # (B, 1, N)

    pos_cost = -torch.log(prob + 1e-12) * 0.25 * (1 - prob).pow(2)
    neg_cost = -torch.log(1 - prob + 1e-12) * (1 - 0.25) * prob.pow(2)

    # Combine based on target gt_valid
    focal_cost = gt_valid_exp * pos_cost + (1 - gt_valid_exp) * neg_cost  # (B, N, N)

    # Permute to shape (B, N_pred, N_gt)
    focal_cost = focal_cost.permute(1,0)

    return focal_cost

@torch.no_grad()
def _get_l1_cost_single(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(1,0)
    return l1_cost


class SamplingResult(util_mixins.NiceRepr):
    """Bbox sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_bboxes': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_bboxes': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags=None):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long(), :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data['pos_bboxes'] = data.pop('pos_bboxes').shape
        data['neg_bboxes'] = data.pop('neg_bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            # 'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

    @classmethod
    def random(cls, rng=None, **kwargs):
        """
        Args:
            rng (None | int | numpy.random.RandomState): seed or state.
            kwargs (keyword arguments):
                - num_preds: number of predicted boxes
                - num_gts: number of true boxes
                - p_ignore (float): probability of a predicted box assigned to \
                    an ignored truth.
                - p_assigned (float): probability of a predicted box not being \
                    assigned.
                - p_use_label (float | bool): with labels or not.

        Returns:
            :obj:`SamplingResult`: Randomly generated sampling result.

        Example:
            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random()
            >>> print(self.__dict__)
        """
        from mmdet.core.bbox import demodata
        from mmdet.core.bbox.assigners.assign_result import AssignResult
        from mmdet.core.bbox.samplers.random_sampler import RandomSampler
        rng = demodata.ensure_rng(rng)

        # make probabilistic?
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult.random(rng=rng, **kwargs)

        # Note we could just compute an assignment
        bboxes = demodata.random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = demodata.random_boxes(assign_result.num_gts, rng=rng)

        if rng.rand() > 0.2:
            # sometimes algorithms squeeze their data, be robust to that
            gt_bboxes = gt_bboxes.squeeze()
            bboxes = bboxes.squeeze()

        sampler = RandomSampler(
            num,
            pos_fraction,
            neg_pos_ub=neg_pos_ub,
            rng=rng)
        self = sampler.sample(assign_result, bboxes, gt_bboxes, None)
        return self
