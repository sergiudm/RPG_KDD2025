# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from genrec.dataset import AbstractDataset
from genrec.models.RPG.model import RPG, ResBlock
from genrec.models.diffusion.diffloss import DiffLoss
from genrec.tokenizer import AbstractTokenizer


class RPGDiff(RPG):
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super(RPGDiff, self).__init__(config, dataset, tokenizer)

        continuous_head_list = []
        for _ in range(self.n_pred_head):
            continuous_head_list.append(ResBlock(self.config['n_embd']))
        self.continuous_heads = nn.Sequential(*continuous_head_list)

        self.diffusion_batch_mul = self.config.get('diffusion_batch_mul', 1)
        self.diff_alpha = float(self.config.get('diff_alpha', 0.05))
        warmup_steps = self.config.get(
            'diff_warmup_steps',
            self.config.get('warmup_steps', 0)
        )
        self.diff_warmup_steps = int(warmup_steps or 0)
        self.register_buffer(
            '_diff_loss_step',
            torch.zeros((), dtype=torch.long),
            persistent=False
        )

        self.diffloss = DiffLoss(
            target_channels=self.config['n_embd'],
            z_channels=self.config['n_embd'],
            width=self.config.get('diffloss_w', 1024),
            depth=self.config.get('diffloss_d', 3),
            num_sampling_steps=self.config.get('num_sampling_steps', 100),
            grad_checkpointing=self.config.get('grad_checkpointing', False),
            use_rectified_flow=self.config.get('use_rectified_flow', False),
            rectified_flow_steps=self.config.get('rectified_flow_steps', 1000),
            ode_solver=self.config.get('ode_solver', 'euler'),
        )

    def _current_diff_alpha(self) -> torch.Tensor:
        if self.diff_alpha <= 0:
            return self._diff_loss_step.new_tensor(0.0, dtype=torch.float32)
        if self.diff_warmup_steps <= 0:
            return self._diff_loss_step.new_tensor(self.diff_alpha, dtype=torch.float32)

        warmup_ratio = self._diff_loss_step.float() / float(self.diff_warmup_steps)
        warmup_ratio = torch.clamp(warmup_ratio, max=1.0)
        return warmup_ratio * self.diff_alpha

    def _advance_diff_alpha_step(self) -> None:
        if self.training:
            self._diff_loss_step.add_(1)

    def _diffusion_loss(self, outputs, batch: dict) -> torch.Tensor:
        label_mask = batch['labels'].view(-1) != self.tokenizer.ignored_label
        if label_mask.sum().item() == 0:
            return outputs.final_states.sum() * 0.0

        continuous_states = [
            self.continuous_heads[i](outputs.last_hidden_state).unsqueeze(-2)
            for i in range(self.n_pred_head)
        ]
        continuous_states = torch.cat(continuous_states, dim=-2)
        outputs.continuous_states = continuous_states

        selected_states = continuous_states.view(
            -1,
            self.n_pred_head,
            self.config['n_embd']
        )[label_mask]

        token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
        # Keep diffusion auxiliary from moving the discrete codeword table via targets.
        target_embs = self.gpt2.wte(token_labels).detach()

        z = selected_states.reshape(-1, self.config['n_embd'])
        target = target_embs.reshape(-1, self.config['n_embd'])
        if self.diffusion_batch_mul > 1:
            z = z.repeat(self.diffusion_batch_mul, 1)
            target = target.repeat(self.diffusion_batch_mul, 1)

        return self.diffloss(z=z, target=target)

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        outputs = super(RPGDiff, self).forward(batch, return_loss=return_loss)
        if not return_loss:
            return outputs

        mtp_loss = outputs.loss
        diff_loss = self._diffusion_loss(outputs, batch)
        diff_alpha = self._current_diff_alpha().to(
            device=diff_loss.device,
            dtype=diff_loss.dtype
        )

        outputs.mtp_loss = mtp_loss
        outputs.diff_loss = diff_loss
        outputs.diff_alpha = diff_alpha.detach()
        outputs.loss = mtp_loss + diff_alpha * diff_loss

        self._advance_diff_alpha_step()
        return outputs
