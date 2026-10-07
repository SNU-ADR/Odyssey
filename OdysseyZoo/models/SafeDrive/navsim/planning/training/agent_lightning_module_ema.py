import pytorch_lightning as pl

from torch import Tensor
from typing import Dict, Tuple
import os
import torch
import copy

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from pytorch_lightning.callbacks import ModelCheckpoint


class AgentLightningModule_EMA(AgentLightningModule):
    """LightningModule with an auxiliary EMA agent."""

    def __init__(self, agent: AbstractAgent, check_unused_parameter: bool = False):
        super().__init__(agent, check_unused_parameter)

        self.eval_with_ema: bool = False

        # Build EMA agent
        self.ema_agent: AbstractAgent = copy.deepcopy(agent)
        self.ema_agent.eval()
        for p in self.ema_agent.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_ema_agent(self, m: float) -> None:
        for src_params, ema_params in zip(self.agent.parameters(), self.ema_agent.parameters()):
            src = src_params.data
            ema_params.data.mul_(m).add_(src, alpha=1.0 - m)

        for src_buffers, ema_buffers in zip(self.agent.buffers(), self.ema_agent.buffers()):
            ema_buffers.data.copy_(src_buffers.data)

    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer,
            optimizer_closure=None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if epoch < 3:
            m = 0.992 + epoch * 0.002
        else:
            m = 0.998
        self.update_ema_agent(m)

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        if self.eval_with_ema:
            old_agent = self.agent
            self.agent = self.ema_agent

            result =  super().validation_step(batch, batch_idx)
            self.agent = old_agent

        else:
            result = super().validation_step(batch, batch_idx)

        return result

    def on_save_checkpoint(self, checkpoint):
        sd = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {k: v for k, v in sd.items() if not k.startswith("ema_agent.")}


class ModelCheckpointEMA(ModelCheckpoint):
    """Save an extra EMA checkpoint alongside the normal checkpoint.

    - Normal checkpoint behavior is preserved (top-k, last, every_n_epochs, etc.).
    - After the base save finishes, writes an additional file with suffix '_ema.ckpt'
      by temporarily swapping the student's weights with EMA weights.
    """

    def __init__(self, *args, ema_suffix: str = "_ema", **kwargs):
        super().__init__(*args, **kwargs)
        self._ema_suffix = ema_suffix

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        if not trainer.is_global_zero:
            return
        if getattr(trainer, "sanity_checking", False):
            return

        pl_module = trainer.lightning_module
        if not (hasattr(pl_module, "ema_agent")):
            return

        root, ext = os.path.splitext(filepath)
        ema_filepath = f"{root}{self._ema_suffix}{ext}"


        ema_state = {
            "ema_agent": pl_module.ema_agent.state_dict(),
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        torch.save(ema_state, ema_filepath)
