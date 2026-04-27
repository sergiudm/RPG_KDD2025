# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.models.diffusion.diffloss import DiffLoss
from genrec.timing import TimingMonitor


# gpt2是pre-norm架构
class OutputProj(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-8)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear(self.norm(x))


class InputProj(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-8)

    def forward(self, x):
        return self.norm(self.linear(x))


class DiffAR(AbstractModel):
    def __init__(
        self, config: dict, dataset: AbstractDataset, tokenizer: AbstractTokenizer
    ):
        super(DiffAR, self).__init__(config, dataset, tokenizer)

        self.register_buffer("item_id2embs", self._map_item_embs())

        # model
        gpt2config = GPT2Config(
            vocab_size=2,  # 随便设一个小值，不影响训练
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config["n_embd"],
            n_layer=config["n_layer"],
            n_head=config["n_head"],
            n_inner=config["n_inner"],
            activation_function=config["activation_function"],
            resid_pdrop=config["resid_pdrop"],
            embd_pdrop=config["embd_pdrop"],
            attn_pdrop=config["attn_pdrop"],
            layer_norm_epsilon=config["layer_norm_epsilon"],
            initializer_range=config["initializer_range"],
        )
        self.gpt2 = GPT2Model(gpt2config)
        self.hidden_embd = self.config["n_embd"]
        self.feat_embd = self.item_id2embs.shape[1]

        self.use_diffloss = self.config.get("use_diffloss", True)
        self.use_rank_loss = self.config.get("use_rank_loss", self.use_diffloss)
        self.use_diffusion_generation = self.config.get(
            "use_diffusion_generation", self.use_diffloss and not self.use_rank_loss
        )

        # heads
        self.input_proj = InputProj(self.feat_embd, self.hidden_embd)
        if self.use_rank_loss:
            self.rank_proj = OutputProj(self.feat_embd, self.hidden_embd)
        # diffusion/ranking losses consume hidden states; plain MSE needs feat states
        if not self.use_diffloss and not self.use_rank_loss:
            self.output_proj = OutputProj(self.feat_embd, self.hidden_embd)

        # loss
        self.temperature = self.config.get("temperature", 1.0)
        self.rank_temperature = self.config.get("rank_temperature", self.temperature)
        if self.rank_temperature <= 0:
            raise ValueError("rank_temperature must be positive.")
        self.lambda_diff = self.config.get(
            "lambda_diff", 0.05 if self.use_rank_loss else 1.0
        )

        # Diffusion Loss
        self.diffusion_batch_mul = self.config.get("diffusion_batch_mul", 1)
        self.diff_temperature = self.config.get("diff_temperature", 1.0)

        if self.use_diffloss:
            self.diffloss = DiffLoss(
                target_channels=self.feat_embd,
                z_channels=self.hidden_embd,
                width=self.config.get("diffloss_w", 1024),
                depth=self.config.get("diffloss_d", 3),
                num_sampling_steps=self.config.get("num_sampling_steps", 100),
                grad_checkpointing=self.config.get("grad_checkpointing", False),
                use_rectified_flow=self.config.get("use_rectified_flow", False),
                rectified_flow_steps=self.config.get("rectified_flow_steps", 1000),
                ode_solver=self.config.get("ode_solver", "euler"),
            )
        else:
            self.loss_mse = torch.nn.MSELoss()

        # Add timing monitor for performance profiling
        self.timing_monitor = TimingMonitor()

    def _map_item_embs(self) -> torch.Tensor:
        """
        Maps item embeddings to their corresponding item IDs.

        Returns:
            item_id2embs (torch.Tensor): A tensor of shape (n_items, n_embd) where each row represents the embedding of an item.
        """
        # 添加pad以匹配input_ids
        sent_embs = torch.FloatTensor(self.tokenizer.sent_embs)
        pad_embs = torch.zeros((1, sent_embs.shape[1]), dtype=torch.float32)
        item_id2embs = torch.cat([pad_embs, sent_embs], dim=0)
        return item_id2embs

    def _item_similarity_logits(self, pred_embs: torch.Tensor) -> torch.Tensor:
        item_embs = F.normalize(self.item_id2embs[1:], p=2, dim=-1)
        pred_embs = F.normalize(pred_embs, p=2, dim=-1)
        return torch.matmul(pred_embs, item_embs.T) / self.rank_temperature

    def _rank_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pred_embs = self.rank_proj(hidden_states)
        return self._item_similarity_logits(pred_embs)

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(
            p.numel()
            for p in self.gpt2.get_input_embeddings().parameters()
            if p.requires_grad
        )
        return (
            f"#Embedding parameters: {emb_params}\n"
            f"#Non-embedding parameters: {total_params - emb_params}\n"
            f"#Total trainable parameters: {total_params}\n"
        )

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        input_embs = self.item_id2embs[batch["input_ids"]]  # [B N] -> [B N D]
        input_embs = self.input_proj(input_embs)  # 映射到n_embd维度
        outputs = self.gpt2(
            inputs_embeds=input_embs, attention_mask=batch["attention_mask"]
        )
        if self.use_diffloss or self.use_rank_loss:
            final_states = outputs.last_hidden_state  # [B N D] D-hidden
        else:
            final_states = self.output_proj(outputs.last_hidden_state)  # [B N D] D-feat
        outputs.final_states = final_states

        if return_loss:
            assert "labels" in batch, "The batch must contain the labels."
            label_mask = batch["labels"] != -100

            if self.use_rank_loss:
                selected_states = final_states[label_mask]
                if selected_states.numel() == 0:
                    rank_loss = final_states.sum() * 0.0
                else:
                    target_ids = batch["labels"][label_mask].long() - 1
                    rank_logits = self._rank_logits(selected_states)
                    rank_loss = F.cross_entropy(rank_logits, target_ids)
                outputs.rank_loss = rank_loss

            if self.use_diffloss:
                # diffusion loss
                labels = batch["labels"].clone()
                labels[~label_mask] = 0

                label_embs = self.item_id2embs[labels]  # [B D]

                bsz, seq_len, _ = final_states.shape
                target = label_embs.reshape(bsz * seq_len, -1).repeat(
                    self.diffusion_batch_mul, 1
                )
                z = final_states.reshape(bsz * seq_len, -1).repeat(
                    self.diffusion_batch_mul, 1
                )
                mask = (
                    label_mask.float()
                    .reshape(bsz * seq_len)
                    .repeat(self.diffusion_batch_mul)
                )

                if mask.sum().item() == 0:
                    diff_loss = z.sum() * 0.0
                else:
                    diff_loss = self.diffloss(z=z, target=target, mask=mask)
                outputs.diff_loss = diff_loss
                if self.use_rank_loss:
                    outputs.loss = rank_loss + self.lambda_diff * diff_loss
                else:
                    outputs.loss = diff_loss
            elif self.use_rank_loss:
                outputs.loss = rank_loss
            else:
                # mse loss
                # print(label_mask.shape, "label_mask.shape")
                selected_states = final_states[label_mask]  # [B N D][mask] -> [s D]
                label_embs = self.item_id2embs[batch["labels"][label_mask]]  # [s D]
                outputs.loss = self.loss_mse(selected_states, label_embs)

        return outputs

    def generate(self, batch, n_return_sequences=1):
        # Time the forward pass
        self.timing_monitor.start("model_forward")
        outputs = self.forward(batch, return_loss=False)  # [B N D]
        self.timing_monitor.end("model_forward")

        # Time the last token extraction
        self.timing_monitor.start("last_token_extraction")
        # last token of the sequence
        out_embd = (
            self.hidden_embd
            if self.use_diffloss or self.use_rank_loss
            else self.feat_embd
        )
        last_pred = outputs.final_states.gather(
            dim=1,
            index=(batch["seq_lens"] - 1)
            .view(-1, 1, 1)
            .expand(-1, 1, out_embd),  # [B] -> [B 1 1] -> [B 1 D]
        ).squeeze(dim=1)
        self.timing_monitor.end("last_token_extraction")

        # Time the diffusion sampling
        if self.use_diffloss and self.use_diffusion_generation:
            self.timing_monitor.start("diffusion_sampling")
            z = last_pred  # [B D] D-hidden
            last_pred = self.diffloss.sample(z, self.diff_temperature, cfg=1.0)
            self.timing_monitor.end("diffusion_sampling")
        elif self.use_rank_loss:
            last_pred = self.rank_proj(last_pred)  # [B D] D-feat

        # Time the similarity calculation
        self.timing_monitor.start("similarity_calculation")
        item_logits = self._item_similarity_logits(last_pred)  # [B n_items]
        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1

        self.timing_monitor.end("similarity_calculation")

        return preds.unsqueeze(-1)  # [B topk 1]
