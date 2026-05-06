# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # Diffusion reranker configs
        self.use_diffusion_reranker = bool(
            self.config.get("use_diffusion_reranker", False)
        )
        self.diff_rerank_candidate_size = int(
            self.config.get("diff_rerank_candidate_size", 200)
        )
        self.diff_rerank_beta = float(self.config.get("diff_rerank_beta", 0.2))
        self.diff_rerank_score_norm = bool(
            self.config.get("diff_rerank_score_norm", True)
        )
        self.diff_sample_temperature = float(
            self.config.get("diff_sample_temperature", 0.7)
        )
        self.diff_score_temperature = float(
            self.config.get("diff_score_temperature", self.temperature)
        )
        self.diff_condition_source = self.config.get(
            "diff_condition_source", "continuous"
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

        # continuous_states = [
        #     self.continuous_heads[i](outputs.last_hidden_state).unsqueeze(-2)
        #     for i in range(self.n_pred_head)
        # ]
        # continuous_states = torch.cat(continuous_states, dim=-2)
        # outputs.continuous_states = continuous_states

        # selected_states = continuous_states.view(
        #     -1,
        #     self.n_pred_head,
        #     self.config['n_embd']
        # )[label_mask]
        
        selected_states = outputs.final_states.view(
            -1, self.n_pred_head, self.config["n_embd"]
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
    
    def _last_position_index(self, batch: dict, hidden_size: int) -> torch.Tensor:
        return (
            (batch['seq_lens'] - 1)
            .view(-1, 1, 1, 1)
            .expand(-1, 1, self.n_pred_head, hidden_size)
        )

    def _compute_token_logits(self, outputs, batch: dict):
        states = outputs.final_states.gather(
            dim=1,
            index=self._last_position_index(batch, self.config['n_embd'])
        ).squeeze(1)  # [B, n_digit, D]

        states = F.normalize(states, dim=-1)

        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)

        logits = [
            torch.matmul(states[:, i, :], token_embs[i].T) / self.temperature
            for i in range(self.n_pred_head)
        ]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)  # [B, n_digit * codebook_size]

        return token_logits, states

    def _compute_diff_condition_states(self, outputs, batch: dict, pred_states):
        # Option A: use the same continuous heads used during diffusion training.
        if self.diff_condition_source == 'continuous':
            continuous_states = [
                self.continuous_heads[i](outputs.last_hidden_state).unsqueeze(-2)
                for i in range(self.n_pred_head)
            ]
            continuous_states = torch.cat(continuous_states, dim=-2)
            return continuous_states.gather(
                dim=1,
                index=self._last_position_index(batch, self.config['n_embd'])
            ).squeeze(1)  # [B, n_digit, D]

        # Option B: directly use RPG pred head states.
        # Useful if you later change _diffusion_loss() to train on outputs.final_states.
        if self.diff_condition_source == 'pred':
            return pred_states

        raise ValueError(f'Unknown diff_condition_source: {self.diff_condition_source}')

    def _score_candidates_with_rpg(self, token_logits, candidate_ids):
        # candidate_ids: [B, C], item ids
        candidate_tokens = self.item_id2tokens[candidate_ids]  # [B, C, n_digit]

        rpg_scores = torch.gather(
            input=token_logits.unsqueeze(1).expand(-1, candidate_ids.shape[1], -1),
            dim=-1,
            index=candidate_tokens - 1,
        ).mean(dim=-1)  # [B, C]

        return rpg_scores, candidate_tokens

    def _score_candidates_with_diffusion(self, diff_states, candidate_tokens):
        batch_size = diff_states.shape[0]
        hidden_size = self.config['n_embd']

        z = diff_states.reshape(-1, hidden_size)  # [B * n_digit, D]

        # sampled_token_embs = self.diffloss.sample(
        #     z=z,
        #     temperature=self.diff_sample_temperature,
        #     cfg=1.0,
        # )
        sampled_token_embs = diff_states
        sampled_token_embs = sampled_token_embs.view(
            batch_size,
            self.n_pred_head,
            hidden_size,
        )  # [B, n_digit, D]

        sampled_token_embs = F.normalize(sampled_token_embs, dim=-1)

        candidate_token_embs = self.gpt2.wte(candidate_tokens)  # [B, C, n_digit, D]
        candidate_token_embs = F.normalize(candidate_token_embs, dim=-1)

        diff_scores = (
            candidate_token_embs * sampled_token_embs.unsqueeze(1)
        ).sum(dim=-1).mean(dim=-1)  # [B, C]

        return diff_scores / self.diff_score_temperature

    def _row_zscore(self, scores):
        return (
            scores - scores.mean(dim=-1, keepdim=True)
        ) / scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)

    def _rerank_candidates(self, outputs, batch, token_logits, pred_states, candidate_ids):
        rpg_scores, candidate_tokens = self._score_candidates_with_rpg(
            token_logits,
            candidate_ids,
        )

        diff_states = self._compute_diff_condition_states(
            outputs,
            batch,
            pred_states,
        )
        diff_scores = self._score_candidates_with_diffusion(
            diff_states,
            candidate_tokens,
        )

        if self.diff_rerank_score_norm:
            rpg_scores_for_rank = self._row_zscore(rpg_scores)
            diff_scores_for_rank = self._row_zscore(diff_scores)
        else:
            rpg_scores_for_rank = rpg_scores
            diff_scores_for_rank = diff_scores

        final_scores = rpg_scores_for_rank + self.diff_rerank_beta * diff_scores_for_rank
        return final_scores

    def generate(self, batch, n_return_sequences=1):
        # If reranker is disabled, keep exactly the original RPG behavior.
        if not self.use_diffusion_reranker:
            return super(RPGDiff, self).generate(batch, n_return_sequences)

        outputs = super(RPGDiff, self).forward(batch, return_loss=False)
        token_logits, pred_states = self._compute_token_logits(outputs, batch)

        # Need more candidates than final top-k.
        candidate_size = max(
            n_return_sequences,
            self.diff_rerank_candidate_size,
        )

        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True

            # graph_propagation can only return up to num_beams final nodes.
            candidate_size = min(candidate_size, self.num_beams)

            candidate_preds, n_visited_items = self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=candidate_size,
            )
            candidate_ids = candidate_preds.squeeze(-1)  # [B, C]

            final_scores = self._rerank_candidates(
                outputs=outputs,
                batch=batch,
                token_logits=token_logits,
                pred_states=pred_states,
                candidate_ids=candidate_ids,
            )

            rerank_idxs = torch.topk(
                final_scores,
                k=n_return_sequences,
                dim=-1,
            ).indices
            preds = torch.gather(candidate_ids, dim=-1, index=rerank_idxs)

            return preds.unsqueeze(-1), n_visited_items

        # Fallback: full-sort candidates when graph decoding is disabled.
        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
            dim=-1,
            index=(self.item_id2tokens[1:, :] - 1)
                .unsqueeze(0)
                .expand(token_logits.shape[0], -1, -1),
        ).mean(dim=-1)

        candidate_ids = item_logits.topk(candidate_size, dim=-1).indices + 1

        final_scores = self._rerank_candidates(
            outputs=outputs,
            batch=batch,
            token_logits=token_logits,
            pred_states=pred_states,
            candidate_ids=candidate_ids,
        )

        rerank_idxs = torch.topk(
            final_scores,
            k=n_return_sequences,
            dim=-1,
        ).indices
        preds = torch.gather(candidate_ids, dim=-1, index=rerank_idxs)

        return preds.unsqueeze(-1)
