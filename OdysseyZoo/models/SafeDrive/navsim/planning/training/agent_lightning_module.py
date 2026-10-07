import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent
import torch

class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent, check_unused_parameter=False):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

        # mAP
        self.val_preds = []
        self.val_gts = []

        # mIoU
        self.val_preds_seg = []
        self.val_gts_seg = []

        # safety
        self.safety_dict = {}
        self.safety_list = ['NC', "DAC", "TTC", "EP", "C", "pdm_score", "DDC", "TLC", "LK"]
        for l in self.safety_list:
            self.safety_dict[f'{l}_err'] = []
            self.safety_dict[f'{l}_err_only_one'] = []

        self.test_traj = {}
        self.test_traj_save = False
        self.debug_mode = False
        self.agent_name = 'temp'

        self.check_unused_parameter = check_unused_parameter
        self.scoring_test_save_name = 'predictions.pkl'

    def on_after_backward(self):
        if self.check_unused_parameter:
            unused_params = []
            for name, param in self.named_parameters():

                if param.requires_grad and param.grad is None:
                    unused_params.append(name)
            if unused_params:
                print(f"[Unused parameters] {unused_params}")

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features, targets)
        loss_dict = self.agent.compute_loss(features, targets, prediction, self.test_traj_save, self.current_epoch)
        for k, v in loss_dict.items():
            if logging_prefix == 'val' and (k in ['NC_loss' ,'DAC_loss', 'TTC_loss', 'EP_loss', 'C_loss', 'pdm_score_loss', 'DDC_loss', 'TLC_loss', 'LK_loss', 'HC_loss',] or \
            k in ['NC_loss_only_one' ,'DAC_loss_only_one', 'TTC_loss_only_one', 'EP_loss_only_one', 'C_loss_only_one', 'pdm_score_loss_only_one','DDC_loss_only_one','TLC_loss_only_one','LK_loss_only_one','HC_loss_only_one']):
                continue
            if v is not None:
                self.log(f"{logging_prefix}/{k}", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))

        if not self.agent.training:

            def detach_reccurent(x, keep_keys=None):
                """
                Recursively detach & move to CPU.
                - Floating tensors  -> CPU float32
                - Non-floating      -> CPU, keep dtype
                - If a dict key is in keep_keys, that (key, value) pair is SKIPPED (not added).
                i.e., keep_keys acts as an EXCLUDE list.
                """
                def _rec(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.detach().cpu()
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(_rec(t) for t in obj)
                    elif isinstance(obj, dict):
                        out = {}
                        for k, v in obj.items():
                            if k not in keep_keys:

                                continue
                            out[k] = _rec(v)
                        return out
                    return obj

                return _rec(x)




            keep_pred_keys = ['ins_labels_3', 'ins_states_3', 'ins_labels_2', 'ins_states_2', f'agent_motion_traj_{self.agent._safedrive_model._config.SWNet_num_layers}', 'trajectory']
            prediction_cpu = detach_reccurent(prediction, keep_keys=keep_pred_keys)

            keep_gt_keys = {'trajectory', 'agent_states', 'agent_labels', 'agent_tokens', 'scene_frame_token', 'motion_traj', 'motion_mask'}
            targets_cpu = detach_reccurent(targets, keep_keys=keep_gt_keys)
            self.val_preds.append(prediction_cpu)
            self.val_gts.append(targets_cpu)


            if self.test_traj_save:
                for i, token in enumerate(features['token']):
                    self.test_traj[token] = prediction_cpu['trajectory'][i].detach().cpu().numpy()


            # Only the argmax is scored; keep it as uint8 on the CPU, since the float maps of a full
            # validation epoch exhaust host RAM.
            if 'bev_semantic_map' in prediction:
                self.val_preds_seg.append(prediction['bev_semantic_map'].detach().argmax(dim=1).to(torch.uint8).cpu())  # (B,H,W) CPU
                self.val_gts_seg.append(targets['bev_semantic_map'].detach().to(torch.uint8).cpu())                     # (B,H,W) CPU


            for l in self.safety_list:
                if f'{l}_loss' in loss_dict:
                    val = loss_dict[f'{l}_loss']
                    self.safety_dict[f'{l}_err'].append(val.detach() if torch.is_tensor(val) else val)
                if f'{l}_loss_only_one' in loss_dict:
                    val = loss_dict[f'{l}_loss_only_one']
                    self.safety_dict[f'{l}_err_only_one'].append(val.detach() if torch.is_tensor(val) else val)

            del prediction



        #             self.safety_dict[f'{l}_err'].append(loss_dict[f'{l}_loss'])
        #             self.safety_dict[f'{l}_err_only_one'].append(loss_dict[f'{l}_loss_only_one'])



        return loss_dict['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
