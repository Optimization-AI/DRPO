# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, kl_type='kl'):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    if kl_type == 'low_var_kl':
        ppo_kl = verl_F.masked_mean(-negative_approx_kl + ratio - 1, eos_mask)
    elif kl_type == 'kl':
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss_grpo(old_log_prob, log_prob, advantages, eos_mask, cliprange, kl_type='kl'):
    """Implementaion exactly follows the paper https://arxiv.org/abs/2402.03300

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    if kl_type == 'low_var_kl':
        ppo_kl = verl_F.masked_mean(-negative_approx_kl + ratio - 1, eos_mask)
    elif kl_type == 'kl':
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    #### average method following the paper
    pg_loss = ((torch.max(pg_losses, pg_losses2)*eos_mask).sum(dim=-1)/eos_mask.sum(dim=-1)).mean()
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl





def compute_policy_loss_disco_logL(old_log_prob, log_prob, advantages, eos_mask, uid, seq_level_rewards, delta, beta, tau, kl_type='low_var_kl'):
    """

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """

    if torch.distributed.is_initialized():
        global_old_log_prob = torch.cat(torch.distributed.nn.all_gather(old_log_prob), dim=0)
        global_log_prob = torch.cat(torch.distributed.nn.all_gather(log_prob), dim=0)
        global_eos_mask = torch.cat(torch.distributed.nn.all_gather(eos_mask), dim=0)
        global_uid = torch.cat(torch.distributed.nn.all_gather(uid), dim=0)
        global_rewards = torch.cat(torch.distributed.nn.all_gather(seq_level_rewards), dim=0)
    else:
        global_old_log_prob = old_log_prob
        global_log_prob = log_prob
        global_eos_mask = eos_mask
        global_uid = uid
        global_rewards = seq_level_rewards

    #### do calculation
    negative_approx_kl = global_log_prob - global_old_log_prob
    ratio = torch.exp(negative_approx_kl)

    if kl_type == 'low_var_kl':
        ppo_kl = verl_F.masked_mean(-negative_approx_kl + ratio - 1, global_eos_mask)
    elif kl_type == 'kl':
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, global_eos_mask)


    global_scores= (global_log_prob * global_eos_mask).sum(dim=1) / global_eos_mask.sum(dim=1)
    ### group scores based on global_uid, we assume each question has same number of responses
    sorted_uid, indices = global_uid.sort()
    sorted_scores = global_scores[indices]
    sorted_rewards = global_rewards[indices]

    num_questions = global_uid.unique().numel()
    num_responses_per_question = global_uid.size(0) // num_questions
    grouped_scores = sorted_scores.view(num_questions, num_responses_per_question)
    grouped_rewards = sorted_rewards.view(num_questions, num_responses_per_question)

    pos_mask = grouped_rewards==1
    neg_mask = grouped_rewards==0
    #### remove all zeros and all ones
    valid_mask = (pos_mask.sum(dim=1)!=0) & (neg_mask.sum(dim=1)!=0)

    if valid_mask.sum()>0:
        grouped_scores = grouped_scores[valid_mask]
        grouped_rewards = grouped_rewards[valid_mask]
        pos_mask = pos_mask[valid_mask]
        neg_mask = neg_mask[valid_mask]

        neg_scores_masked = (grouped_scores/tau).masked_fill(~neg_mask, float('-inf'))

        # Compute stable max while keeping dimension
        neg_max, _ = neg_scores_masked.max(dim=-1, keepdim=True)
        # handle all-masked rows safely
        neg_max = torch.where(neg_max == float('-inf'), torch.zeros_like(neg_max), neg_max)

        # Subtract max, exponentiate, and apply mask
        neg_exp = torch.exp(((grouped_scores/tau) - neg_max.detach()) * neg_mask) * neg_mask
        neg_sum_exp = neg_exp.sum(dim=-1, keepdim=True)

        neg_logmeanexp = neg_sum_exp / (neg_sum_exp.detach()+torch.finfo(neg_sum_exp.dtype).eps)

        pg_losses = ((grouped_scores - tau*neg_logmeanexp)*pos_mask
                     ).sum(dim=1, keepdim=True) / pos_mask.sum(dim=1, keepdim=True)
        
        pg_loss = pg_losses.sum() / num_questions
    else:
        pg_loss=torch.tensor(0.)*global_scores.mean()  ### dummy loss 


    constraint = torch.maximum(beta*(ppo_kl-delta),
                                        torch.zeros_like(ppo_kl)).detach() *ppo_kl

    pg_loss = -pg_loss + constraint
    pg_clipfrac = torch.gt(ppo_kl, delta).float()

    return pg_loss, pg_clipfrac, ppo_kl

def compute_policy_loss_drpo(old_log_prob, log_prob, eos_mask, uid, seq_level_rewards, delta, beta, tau, Lambda, kl_type='low_var_kl'):
    """

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """

    if torch.distributed.is_initialized():
        global_old_log_prob = torch.cat(torch.distributed.nn.all_gather(old_log_prob), dim=0)
        global_log_prob = torch.cat(torch.distributed.nn.all_gather(log_prob), dim=0)
        global_eos_mask = torch.cat(torch.distributed.nn.all_gather(eos_mask), dim=0)
        global_uid = torch.cat(torch.distributed.nn.all_gather(uid), dim=0)
        global_rewards = torch.cat(torch.distributed.nn.all_gather(seq_level_rewards), dim=0)
    else:
        global_old_log_prob = old_log_prob
        global_log_prob = log_prob
        global_eos_mask = eos_mask
        global_uid = uid
        global_rewards = seq_level_rewards

    #### do calculation
    negative_approx_kl = global_log_prob - global_old_log_prob
    ratio = torch.exp(negative_approx_kl)
    if kl_type == 'low_var_kl':
        ppo_kl = verl_F.masked_mean(-negative_approx_kl + ratio - 1, global_eos_mask)
    elif kl_type == 'kl':
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, global_eos_mask)

    global_scores= (global_log_prob * global_eos_mask).sum(dim=1) / global_eos_mask.sum(dim=1) # [n,]
    global_length_ratio = global_eos_mask.sum(dim=1)/ global_eos_mask.size(-1)
    ### group scores based on global_uid, we assume each question has same number of responses
    sorted_uid, indices = global_uid.sort()
    sorted_scores = global_scores[indices]
    sorted_rewards = global_rewards[indices]
    sorted_length_ratio = global_length_ratio[indices]
    # print('###########sorted_uid', sorted_uid)

    num_questions = global_uid.unique().numel()
    num_responses_per_question = global_uid.size(0) // num_questions
    grouped_scores = sorted_scores.view(num_questions, num_responses_per_question)
    grouped_rewards = sorted_rewards.view(num_questions, num_responses_per_question)
    grouped_length_ratio = sorted_length_ratio.view(num_questions, num_responses_per_question)

    pos_mask = grouped_rewards==1 # [nq, nr]
    neg_mask = grouped_rewards==0
    #### remove all zeros and all ones
    valid_mask = (pos_mask.sum(dim=1)!=0) & (neg_mask.sum(dim=1)!=0)


    if valid_mask.sum()>0:
        grouped_scores = grouped_scores[valid_mask]
        grouped_rewards = grouped_rewards[valid_mask]
        grouped_length_ratio = grouped_length_ratio[valid_mask]
        pos_mask = pos_mask[valid_mask]
        neg_mask = neg_mask[valid_mask]
    
        neg_scores_masked = (grouped_scores/tau).masked_fill(~neg_mask, float('-inf'))

        # Compute stable max while keeping dimension
        neg_max, _ = neg_scores_masked.max(dim=-1, keepdim=True)
        # handle all-masked rows safely
        neg_max = torch.where(neg_max == float('-inf'), torch.zeros_like(neg_max), neg_max)

        # Subtract max, exponentiate, and apply mask
        neg_exp = torch.exp(((grouped_scores/tau) - neg_max.detach()) * neg_mask) * neg_mask
        neg_sum_exp = neg_exp.sum(dim=-1, keepdim=True)

        neg_logmeanexp = neg_sum_exp / (neg_sum_exp.detach()+torch.finfo(neg_sum_exp.dtype).eps)

        weight = torch.exp((1-grouped_length_ratio)/Lambda)
        pg_losses = (weight*(grouped_scores - tau*neg_logmeanexp)*pos_mask
                     ).sum(dim=1, keepdim=True) / (weight*pos_mask).sum(dim=1, keepdim=True)
        
        pg_loss = pg_losses.sum() / num_questions
    else:
        pg_loss=torch.tensor(0.)*global_scores.mean()  ### dummy loss 

    constraint = torch.maximum(beta*(ppo_kl-delta),
                                        torch.zeros_like(ppo_kl)).detach() *ppo_kl

    pg_loss = -pg_loss + constraint
    pg_clipfrac = torch.gt(ppo_kl, delta).float()

    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
