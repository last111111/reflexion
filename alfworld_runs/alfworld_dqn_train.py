"""
AlfWorld Double DQN (DDQN) Training
完全按照 alfworld_meta_dqn-master 的架构实现，使用 alfworld_qwen_colab_v2.ipynb 的环境调用方式
"""

import os
import sys
import json
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional
import alfworld
import alfworld.agents.environment
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================== Layer Components (from layers.py) ========================

def compute_mask(x):
    """计算 padding mask"""
    mask = torch.ne(x, 0).float()
    if x.is_cuda:
        mask = mask.cuda()
    return mask


def masked_mean(x, m=None, dim=1):
    """带 mask 的平均池化"""
    if m is None:
        return torch.mean(x, dim=dim)
    x = x * m.unsqueeze(-1)
    mask_sum = torch.sum(m, dim=-1)  # batch
    tmp = torch.eq(mask_sum, 0).float()
    if x.is_cuda:
        tmp = tmp.cuda()
    mask_sum = mask_sum + tmp
    res = torch.sum(x, dim=dim)  # batch x h
    res = res / mask_sum.unsqueeze(-1)
    return res


def masked_softmax(x, m=None, axis=-1):
    """带 mask 的 softmax"""
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


class SelfAttention(nn.Module):
    """Multi-head Self-Attention"""
    def __init__(self, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.block_hidden_dim = block_hidden_dim
        self.n_head = n_head
        self.dropout = dropout
        assert block_hidden_dim % n_head == 0
        self.d_k = block_hidden_dim // n_head

        self.linears = nn.ModuleList([nn.Linear(block_hidden_dim, block_hidden_dim) for _ in range(4)])

    def forward(self, query, mask, key, value):
        batch_size = query.size(0)

        # Linear projections in batch
        query, key, value = [l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
                            for l, x in zip(self.linears, (query, key, value))]

        # Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)  # batch x 1 x time x time
            scores = scores * mask + -1e9 * (1 - mask)

        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.block_hidden_dim)
        out = self.linears[-1](out)

        return out, attn


class CQAttention(nn.Module):
    """Context-Query Attention (用于融合 observation 和 task description)"""
    def __init__(self, block_hidden_dim, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        w = torch.empty(block_hidden_dim * 3)
        nn.init.xavier_uniform_(w)
        self.w = nn.Parameter(w)

    def forward(self, C, Q, c_mask, q_mask):
        # C: batch x c_len x hid (observation)
        # Q: batch x q_len x hid (task description)
        batch_size, c_len, q_len = C.size(0), C.size(1), Q.size(1)

        # Compute similarity matrix
        C_expanded = C.unsqueeze(2).expand(-1, -1, q_len, -1)  # batch x c_len x q_len x hid
        Q_expanded = Q.unsqueeze(1).expand(-1, c_len, -1, -1)  # batch x c_len x q_len x hid
        CQ = C_expanded * Q_expanded  # batch x c_len x q_len x hid

        S_inputs = torch.cat([C_expanded, Q_expanded, CQ], dim=-1)  # batch x c_len x q_len x 3*hid
        S = torch.matmul(S_inputs, self.w)  # batch x c_len x q_len

        # Context-to-query attention
        S1 = masked_softmax(S, m=q_mask.unsqueeze(1), axis=2)  # batch x c_len x q_len
        A = torch.bmm(S1, Q)  # batch x c_len x hid

        # Query-to-context attention
        S2 = masked_softmax(S, m=c_mask.unsqueeze(2), axis=1)  # batch x c_len x q_len
        S2_max, _ = torch.max(S2, dim=1, keepdim=True)  # batch x 1 x q_len
        B = torch.bmm(S2_max, Q).expand(-1, c_len, -1)  # batch x c_len x hid

        # Output: [C, A, C*A, C*B]
        out = torch.cat([C, A, C * A, C * B], dim=-1)  # batch x c_len x 4*hid

        return out


def PosEncoder(x):
    """Position Encoding"""
    batch_size, seq_len, d_model = x.size()
    pos = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1).float()
    if x.is_cuda:
        pos = pos.cuda()

    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
    if x.is_cuda:
        div_term = div_term.cuda()

    pe = torch.zeros(batch_size, seq_len, d_model)
    if x.is_cuda:
        pe = pe.cuda()

    pe[:, :, 0::2] = torch.sin(pos.unsqueeze(-1) * div_term)
    pe[:, :, 1::2] = torch.cos(pos.unsqueeze(-1) * div_term)

    return x + pe


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x, mask):
        # x: batch x time x dim
        x = x.transpose(1, 2)  # batch x dim x time
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = x.transpose(1, 2)  # batch x time x dim
        x = x * mask.unsqueeze(-1)
        return x


class EncoderBlock(nn.Module):
    """Encoder Block (from meta-dqn)"""
    def __init__(self, conv_num, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_1 = nn.Linear(ch_num, ch_num)
        self.FFN_2 = nn.Linear(ch_num, ch_num)
        self.norm_C = nn.ModuleList([nn.LayerNorm(block_hidden_dim) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(block_hidden_dim)
        self.norm_2 = nn.LayerNorm(block_hidden_dim)
        self.conv_num = conv_num

    def forward(self, x, mask, squared_mask, l, blks):
        total_layers = (self.conv_num + 2) * blks
        # Position encoding
        out = PosEncoder(x)
        out = out * mask.unsqueeze(-1)

        # Conv layers
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out, mask)
            out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
            out = out * mask.unsqueeze(-1)
            l += 1

        # Self attention
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out, _ = self.self_att(out, squared_mask, out, out)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        out = out * mask.unsqueeze(-1)
        l += 1

        # FFN
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        out = out * mask.unsqueeze(-1)
        l += 1

        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


# ======================== Policy Network (from model.py) ========================

class Policy(nn.Module):
    """
    DQN Policy Network
    完全按照 meta-dqn/alfworld/agents/modules/model.py 的架构
    """
    def __init__(self, bert_model, word_vocab_size, config):
        super().__init__()
        self.bert_model = bert_model
        self.word_vocab_size = word_vocab_size
        self.config = config

        # Read config
        self.encoder_layers = config['encoder_layers']
        self.encoder_conv_num = config['encoder_conv_num']
        self.block_hidden_dim = config['block_hidden_dim']
        self.n_heads = config['n_heads']
        self.block_dropout = config['block_dropout']
        self.dropout = config['dropout']
        self.recurrent = config['recurrent']
        self.noisy_net = config.get('noisy_net', False)

        self._def_layers()

    def _def_layers(self):
        BERT_EMBEDDING_SIZE = 768

        # Word embedding projection: BERT (768) -> hidden_dim (64)
        self.word_embedding_prj = nn.Linear(BERT_EMBEDDING_SIZE, self.block_hidden_dim, bias=False)

        # Encoder stack
        self.encoder = nn.ModuleList([
            EncoderBlock(
                conv_num=self.encoder_conv_num,
                ch_num=self.block_hidden_dim,
                k=5,
                block_hidden_dim=self.block_hidden_dim,
                n_head=self.n_heads,
                dropout=self.block_dropout
            ) for _ in range(self.encoder_layers)
        ])

        # Aggregation attention (combine obs and task description)
        self.aggregation_attention = CQAttention(block_hidden_dim=self.block_hidden_dim, dropout=self.dropout)
        self.aggregation_attention_proj = nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)

        # Recurrent dynamics
        if self.recurrent:
            self.rnncell = nn.GRUCell(self.block_hidden_dim, self.block_hidden_dim)
            self.dynamics_aggregation = nn.Linear(self.block_hidden_dim * 2, self.block_hidden_dim)
        else:
            self.rnncell, self.dynamics_aggregation = None, None

        # Action scorer
        linear_function = nn.Linear  # NoisyLinear if self.noisy_net else nn.Linear
        self.action_scorer_linear_1 = linear_function(self.block_hidden_dim * 2, self.block_hidden_dim)
        self.action_scorer_linear_2 = linear_function(self.block_hidden_dim, 1)
        self.action_scorer_extra_self_attention = SelfAttention(self.block_hidden_dim, self.n_heads, self.dropout)
        self.action_scorer_extra_linear = linear_function(self.block_hidden_dim, self.block_hidden_dim)

    def get_bert_embeddings(self, _input_words, _input_masks):
        """获取 BERT embeddings"""
        # 处理超长序列
        if _input_words.size(1) > 512:
            seg_length = 500
            outputs = []
            num_batch = (_input_words.size(1) + seg_length - 1) // seg_length
            for i in range(num_batch):
                batch_input = _input_words[:, i * seg_length: (i + 1) * seg_length]
                batch_mask = _input_masks[:, i * seg_length: (i + 1) * seg_length]
                out = self.get_bert_embeddings(batch_input, batch_mask)
                outputs.append(out)
            return torch.cat(outputs, 1)

        with torch.no_grad():
            res = self.bert_model.embeddings(_input_words)
            res = res * _input_masks.unsqueeze(-1)
        return res

    def embed(self, input_words, input_word_masks):
        """Embed words"""
        word_embeddings = self.get_bert_embeddings(input_words, input_word_masks)
        word_embeddings = word_embeddings * input_word_masks.unsqueeze(-1)
        word_embeddings = self.word_embedding_prj(word_embeddings)
        word_embeddings = word_embeddings * input_word_masks.unsqueeze(-1)
        return word_embeddings

    def encode_text(self, input_word_ids):
        """编码文本 -> hidden representations"""
        # input_word_ids: batch x seq_len
        input_word_masks = compute_mask(input_word_ids)
        embeddings = self.embed(input_word_ids, input_word_masks)  # batch x seq_len x hid

        squared_mask = torch.bmm(input_word_masks.unsqueeze(-1), input_word_masks.unsqueeze(1))
        encoding_sequence = embeddings

        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder[i](
                encoding_sequence, input_word_masks, squared_mask,
                i * (self.encoder_conv_num + 2) + 1, self.encoder_layers
            )

        return encoding_sequence, input_word_masks

    def aggretate_information(self, h_obs, obs_mask, h_td, td_mask):
        """Aggregate observation and task description"""
        aggregated_obs_representation = self.aggregation_attention(h_obs, h_td, obs_mask, td_mask)
        aggregated_obs_representation = self.aggregation_attention_proj(aggregated_obs_representation)
        aggregated_obs_representation = torch.tanh(aggregated_obs_representation)
        aggregated_obs_representation = aggregated_obs_representation * obs_mask.unsqueeze(-1)
        return aggregated_obs_representation

    def masked_mean(self, h_obs, obs_mask):
        """Mean pooling with mask"""
        _mask = torch.sum(obs_mask, -1)  # batch
        obs_representations = torch.sum(h_obs, -2)  # batch x hid
        tmp = torch.eq(_mask, 0).float()
        if obs_representations.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        obs_representations = obs_representations / _mask.unsqueeze(-1)
        return obs_representations

    def score_actions(self, candidate_representations, cand_mask, h_obs, obs_mask, current_dynamics, fix_shared_components=False):
        """
        为候选动作打分 (Q values)
        Args:
            candidate_representations: batch x num_candidate x hid
            cand_mask: batch x num_candidate
            h_obs: batch x obs_len x hid
            obs_mask: batch x obs_len
            current_dynamics: batch x hid (recurrent hidden state)
            fix_shared_components: bool (for beam search choice)
        Returns:
            action_scores: batch x num_candidate
            action_masks: batch x num_candidate
        """
        batch_size, num_candidate = candidate_representations.size(0), candidate_representations.size(1)

        # Aggregate observation representation
        aggregated_obs_representation = self.masked_mean(h_obs, obs_mask)  # batch x hid

        if self.recurrent:
            aggregated_obs_representation = self.dynamics_aggregation(
                torch.cat([aggregated_obs_representation, current_dynamics], -1)
            )
            aggregated_obs_representation = torch.relu(aggregated_obs_representation)

        if fix_shared_components:
            aggregated_obs_representation = aggregated_obs_representation.detach()
            candidate_representations = candidate_representations.detach()

        # Expand observation representation to match candidate dimensions
        new_h_expanded = torch.stack([aggregated_obs_representation] * num_candidate, 1)

        # Combine candidate and observation representations
        output = self.action_scorer_linear_1(
            torch.cat([candidate_representations, new_h_expanded], -1)
        )
        output = torch.relu(output)
        output = output * cand_mask.unsqueeze(-1)

        if fix_shared_components:
            # Extra attention for beam search choice
            cand_mask_squared = torch.bmm(cand_mask.unsqueeze(-1), cand_mask.unsqueeze(1))
            output, _ = self.action_scorer_extra_self_attention(output, cand_mask_squared, output, output)
            output = self.action_scorer_extra_linear(output)
            output = torch.relu(output)
            output = output * cand_mask.unsqueeze(-1)

        # Score head
        output = self.action_scorer_linear_2(output).squeeze(-1)  # batch x num_candidate
        output = output * cand_mask

        return output, cand_mask


# ======================== Double DQN Agent ========================

Transition = namedtuple('Transition', [
    'obs', 'task', 'action_idx', 'reward', 'next_obs', 'next_task', 'done',
    'admissible', 'next_admissible', 'prev_dynamics'
])


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay"""
    def __init__(self, capacity=500000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, *args):
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            return [], [], []

        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self.size


class DoubleDQNAgent:
    """
    Double DQN Agent (按照 text_dqn_agent.py 实现)

    Double DQN 的核心改进:
    1. 用 online network 选择动作 (argmax Q_online)
    2. 用 target network 评估该动作的 Q 值 (Q_target[best_action])
    3. 这样可以减少 Q 值的过估计问题
    """
    def __init__(self, online_net, target_net, tokenizer, word2id, device, config):
        self.online_net = online_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(online_net.state_dict())
        self.target_net.eval()

        self.tokenizer = tokenizer
        self.word2id = word2id
        self.device = device
        self.config = config

        # Training hyperparameters
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.target_update_frequency = config['target_update_frequency']
        self.update_per_k_game_steps = config['update_per_k_game_steps']
        self.multi_step = config.get('multi_step', 3)

        # Epsilon greedy
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_anneal_episodes = config['epsilon_anneal_episodes']
        self.epsilon = self.epsilon_start

        # Optimizer
        self.optimizer = optim.Adam(online_net.parameters(), lr=self.learning_rate)

        # Replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=config['replay_memory_capacity'])

        # Counters
        self.step_in_total = 0
        self.episode_no = 0
        self.mode = "train"

    def update_epsilon(self):
        """Update epsilon for epsilon-greedy"""
        if self.episode_no < self.epsilon_anneal_episodes:
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.episode_no / self.epsilon_anneal_episodes)
        else:
            self.epsilon = self.epsilon_end

    def words_to_ids(self, words):
        """Convert words to token IDs"""
        ids = self.tokenizer.encode(words, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long).to(self.device)

    def encode(self, texts, use_model="online"):
        """Encode texts"""
        model = self.online_net if use_model == "online" else self.target_net

        input_ids_list = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
            input_ids_list.append(ids)

        # Pad to same length
        max_len = max(len(ids) for ids in input_ids_list)
        input_ids_padded = [ids + [0] * (max_len - len(ids)) for ids in input_ids_list]
        input_ids = torch.tensor(input_ids_padded, dtype=torch.long).to(self.device)

        h, mask = model.encode_text(input_ids)
        return h, mask

    def choose_random_action(self, action_scores, action_candidate_list):
        """Choose random action"""
        batch_size = action_scores.size(0)
        indices = []
        for j in range(batch_size):
            indices.append(np.random.choice(len(action_candidate_list[j])))
        return np.array(indices)

    def choose_maxQ_action(self, action_scores, action_mask=None):
        """Choose action with maximum Q value"""
        action_scores = action_scores - torch.min(action_scores, -1, keepdim=True)[0] + 1e-2
        if action_mask is not None:
            action_scores = action_scores * action_mask
        action_indices = torch.argmax(action_scores, -1)
        return action_indices.cpu().numpy()

    def action_scoring(self, action_candidate_list, h_obs, obs_mask, h_td, td_mask, previous_dynamics, use_model="online"):
        """
        Score admissible actions
        Returns:
            action_scores: batch x num_actions
            action_masks: batch x num_actions
            current_dynamics: batch x hid
        """
        model = self.online_net if use_model == "online" else self.target_net
        batch_size = len(action_candidate_list)

        # Aggregate obs and task
        aggregated_obs_representation = model.aggretate_information(h_obs, obs_mask, h_td, td_mask)

        # Update dynamics
        if model.recurrent:
            averaged_representation = model.masked_mean(aggregated_obs_representation, obs_mask)
            current_dynamics = model.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else model.rnncell(averaged_representation)
        else:
            current_dynamics = None

        # Encode candidate actions
        max_num_candidate = max(len(candidates) for candidates in action_candidate_list)
        candidate_representations_list = []
        candidate_masks_list = []

        for candidates in action_candidate_list:
            if len(candidates) == 0:
                candidate_reps = torch.zeros(max_num_candidate, model.block_hidden_dim).to(self.device)
                candidate_mask = torch.zeros(max_num_candidate).to(self.device)
            else:
                h_cand, cand_mask = self.encode(candidates, use_model=use_model)
                candidate_reps = model.masked_mean(h_cand, cand_mask)  # num_cand x hid

                # Pad to max_num_candidate
                if len(candidates) < max_num_candidate:
                    padding = torch.zeros(max_num_candidate - len(candidates), model.block_hidden_dim).to(self.device)
                    candidate_reps = torch.cat([candidate_reps, padding], dim=0)

                    mask_padding = torch.zeros(max_num_candidate - len(candidates)).to(self.device)
                    candidate_mask = torch.cat([cand_mask.sum(dim=1) > 0, mask_padding.bool()], dim=0).float()
                else:
                    candidate_mask = (cand_mask.sum(dim=1) > 0).float()

            candidate_representations_list.append(candidate_reps)
            candidate_masks_list.append(candidate_mask)

        candidate_representations = torch.stack(candidate_representations_list)  # batch x num_cand x hid
        candidate_masks = torch.stack(candidate_masks_list)  # batch x num_cand

        # Score actions
        action_scores, action_masks = model.score_actions(
            candidate_representations, candidate_masks,
            aggregated_obs_representation, obs_mask,
            current_dynamics, fix_shared_components=False
        )

        return action_scores, action_masks, current_dynamics

    def admissible_commands_act(self, obs_strings, task_strings, action_candidate_list, previous_dynamics, random=False):
        """
        Select action from admissible commands (epsilon-greedy)
        """
        with torch.no_grad():
            if self.mode == "eval" or not random:
                # Greedy
                h_obs, obs_mask = self.encode(obs_strings, use_model="online")
                h_td, td_mask = self.encode(task_strings, use_model="online")
                action_scores, action_masks, current_dynamics = self.action_scoring(
                    action_candidate_list, h_obs, obs_mask, h_td, td_mask, previous_dynamics, use_model="online"
                )
                action_indices = self.choose_maxQ_action(action_scores, action_masks)
                chosen_actions = [candidates[idx] for candidates, idx in zip(action_candidate_list, action_indices)]
                return chosen_actions, action_indices, current_dynamics

            # Epsilon-greedy
            batch_size = len(obs_strings)
            h_obs, obs_mask = self.encode(obs_strings, use_model="online")
            h_td, td_mask = self.encode(task_strings, use_model="online")
            action_scores, action_masks, current_dynamics = self.action_scoring(
                action_candidate_list, h_obs, obs_mask, h_td, td_mask, previous_dynamics, use_model="online"
            )

            action_indices_maxq = self.choose_maxQ_action(action_scores, action_masks)
            action_indices_random = self.choose_random_action(action_scores, action_candidate_list)

            # Epsilon-greedy selection
            rand_num = np.random.uniform(low=0.0, high=1.0, size=(batch_size,))
            less_than_epsilon = (rand_num < self.epsilon).astype("float32")
            greater_than_epsilon = 1.0 - less_than_epsilon

            chosen_indices = less_than_epsilon * action_indices_random + greater_than_epsilon * action_indices_maxq
            chosen_indices = chosen_indices.astype(int)
            chosen_actions = [candidates[idx] for candidates, idx in zip(action_candidate_list, chosen_indices)]

            return chosen_actions, chosen_indices, current_dynamics

    def update(self, beta=0.4):
        """
        Update online network using Double DQN

        Double DQN 更新规则:
        1. 用 online network 选择下一状态的最佳动作: a* = argmax_a Q_online(s', a)
        2. 用 target network 评估该动作: Q_target(s', a*)
        3. 目标值: y = r + gamma * Q_target(s', a*)
        4. 最小化: (Q_online(s, a) - y)^2
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        transitions, indices, weights = self.memory.sample(self.batch_size, beta=beta)
        batch = Transition(*zip(*transitions))
        weights = torch.FloatTensor(weights).to(self.device)

        # Multi-step returns (简化版，实际应该累积多步)
        batch_size = len(transitions)

        # Compute current Q values
        h_obs, obs_mask = self.encode(batch.obs, use_model="online")
        h_td, td_mask = self.encode(batch.task, use_model="online")
        action_scores, _, current_dynamics = self.action_scoring(
            batch.admissible, h_obs, obs_mask, h_td, td_mask, batch.prev_dynamics, use_model="online"
        )

        # Gather Q values for chosen actions
        q_values = []
        for i in range(batch_size):
            q_values.append(action_scores[i, batch.action_idx[i]])
        q_values = torch.stack(q_values)

        # Compute target Q values using Double DQN
        next_q_values = []
        for i in range(batch_size):
            if batch.done[i]:
                next_q_values.append(torch.tensor(0.0).to(self.device))
            else:
                # DOUBLE DQN: Use online net to SELECT action
                h_obs_next, obs_mask_next = self.encode([batch.next_obs[i]], use_model="online")
                h_td_next, td_mask_next = self.encode([batch.next_task[i]], use_model="online")

                # Get dynamics for next state
                aggregated_obs_next = self.online_net.aggretate_information(h_obs_next, obs_mask_next, h_td_next, td_mask_next)
                if self.online_net.recurrent:
                    averaged_next = self.online_net.masked_mean(aggregated_obs_next, obs_mask_next)
                    next_dynamics = self.online_net.rnncell(averaged_next, current_dynamics[i:i+1])
                else:
                    next_dynamics = None

                # Step 1: Use ONLINE network to SELECT the best action
                action_scores_online, _, _ = self.action_scoring(
                    [batch.next_admissible[i]], h_obs_next, obs_mask_next, h_td_next, td_mask_next,
                    next_dynamics, use_model="online"
                )
                best_action_idx = action_scores_online[0].argmax().item()  # argmax over online Q values

                # Step 2: Use TARGET network to EVALUATE that action
                action_scores_target, _, _ = self.action_scoring(
                    [batch.next_admissible[i]], h_obs_next, obs_mask_next, h_td_next, td_mask_next,
                    next_dynamics, use_model="target"
                )
                next_q_values.append(action_scores_target[0, best_action_idx])  # Q_target[best_action from online]

        next_q_values = torch.stack(next_q_values)

        # Target Q values
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        target_q_values = rewards + (self.gamma ** self.multi_step) * next_q_values

        # TD errors for prioritized replay
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors + 1e-6)

        # Weighted loss
        loss = (weights * F.smooth_l1_loss(q_values, target_q_values, reduction='none')).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=0.1)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network"""
        self.target_net.load_state_dict(self.online_net.state_dict())


# ======================== Training Loop ========================

def process_ob(ob):
    """Process observation"""
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


def run_episode(env, agent, task_desc, max_steps=50, train=True):
    """Run one episode"""
    obs, info = env.reset()
    obs_text = '\n'.join(obs[0].split('\n\n')[1:])
    obs_text = process_ob(obs_text)

    total_reward = 0
    step = 0
    losses = []
    prev_dynamics = None

    while step < max_steps:
        # Get admissible actions
        if 'admissible_commands' not in info or len(info['admissible_commands']) == 0:
            break

        admissible = info['admissible_commands'][0]

        # Select action
        actions, action_indices, current_dynamics = agent.admissible_commands_act(
            [obs_text], [task_desc], [admissible], prev_dynamics, random=train
        )
        action = actions[0]
        action_idx = action_indices[0]

        # Execute action
        next_obs, _, done, next_info = env.step([action])
        next_obs_text = process_ob(next_obs[0])
        reward = 1.0 if next_info['won'][0] else 0.0
        done = done[0]

        # Store transition
        if train:
            next_admissible = next_info.get('admissible_commands', [[]])[0] if not done else []
            agent.memory.push(
                obs_text, task_desc, action_idx, reward, next_obs_text, task_desc,
                done, admissible, next_admissible, prev_dynamics
            )

            # Update network
            if agent.step_in_total % agent.update_per_k_game_steps == 0 and len(agent.memory) >= agent.batch_size:
                loss = agent.update(beta=0.4)
                if loss is not None:
                    losses.append(loss)

        total_reward += reward
        obs_text = next_obs_text
        info = next_info
        prev_dynamics = current_dynamics
        step += 1
        agent.step_in_total += 1

        if done:
            break

    agent.episode_no += 1
    agent.update_epsilon()

    # Update target network
    if agent.episode_no % agent.target_update_frequency == 0:
        agent.update_target_network()

    avg_loss = np.mean(losses) if losses else 0.0
    return total_reward, step, avg_loss, done


def train_dqn(bert_model, tokenizer, word2id, config_path='base_config.yaml',
              num_episodes=1000, eval_freq=50, save_path='dqn_policy.pt'):
    """Train Double DQN agent"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # Load environment config
    with open(config_path) as f:
        env_config = yaml.safe_load(f)

    # Create environments
    train_env = alfworld.agents.environment.get_environment(env_config["env"]["type"])(env_config, train_eval='train')
    train_env = train_env.init_env(batch_size=1)

    eval_env = alfworld.agents.environment.get_environment(env_config["env"]["type"])(env_config, train_eval='eval_out_of_distribution')
    eval_env = eval_env.init_env(batch_size=1)

    # Model config (从 meta-dqn 的 dqn_config.yaml)
    model_config = {
        'encoder_layers': 1,
        'encoder_conv_num': 5,
        'block_hidden_dim': 64,
        'n_heads': 1,
        'block_dropout': 0.1,
        'dropout': 0.1,
        'recurrent': True,
        'noisy_net': False
    }

    # Training config
    training_config = {
        'learning_rate': 0.001,
        'gamma': 0.9,
        'batch_size': 64,
        'target_update_frequency': 500,
        'update_per_k_game_steps': 5,
        'multi_step': 3,
        'epsilon_start': 0.3,
        'epsilon_end': 0.1,
        'epsilon_anneal_episodes': 1000,
        'replay_memory_capacity': 500000
    }

    # Create networks
    word_vocab_size = len(word2id)
    online_net = Policy(bert_model, word_vocab_size, model_config)
    target_net = Policy(bert_model, word_vocab_size, model_config)

    # Create agent
    agent = DoubleDQNAgent(online_net, target_net, tokenizer, word2id, device, training_config)

    # Training
    train_rewards = []
    eval_rewards = []

    logger.info("开始训练 Double DQN...")
    logger.info("="*80)

    task_desc = "Your task is to: put some object on some receptacle."  # 简化版任务描述

    for episode in tqdm(range(num_episodes), desc="训练进度"):
        # Train
        reward, steps, loss, success = run_episode(train_env, agent, task_desc, train=True)
        train_rewards.append(reward)

        # Evaluate
        if (episode + 1) % eval_freq == 0:
            eval_reward_list = []
            eval_success_list = []

            agent.mode = "eval"
            for _ in range(5):
                eval_reward, _, _, eval_success = run_episode(eval_env, agent, task_desc, train=False)
                eval_reward_list.append(eval_reward)
                eval_success_list.append(eval_success)
            agent.mode = "train"

            avg_eval_reward = np.mean(eval_reward_list)
            avg_eval_success = np.mean(eval_success_list)
            eval_rewards.append(avg_eval_reward)

            logger.info(f"\nEpisode {episode+1}/{num_episodes}")
            logger.info(f"  训练奖励: {np.mean(train_rewards[-eval_freq:]):.3f}")
            logger.info(f"  评估奖励: {avg_eval_reward:.3f}")
            logger.info(f"  评估成功率: {avg_eval_success:.1%}")
            logger.info(f"  Epsilon: {agent.epsilon:.3f}")
            logger.info(f"  Loss: {loss:.4f}")
            logger.info(f"  缓冲区: {len(agent.memory)}")
            logger.info("="*80)

    # Save model
    torch.save({
        'online_net': online_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'episodes': num_episodes,
        'train_rewards': train_rewards,
        'eval_rewards': eval_rewards,
        'config': {**model_config, **training_config}
    }, save_path)

    logger.info(f"\n✓ 模型已保存: {save_path}")

    train_env.close()
    eval_env.close()

    return agent, train_rewards, eval_rewards


if __name__ == "__main__":
    logger.info("请在 notebook 或主脚本中调用 train_dqn() 函数")
    logger.info("需要提供: bert_model, tokenizer, word2id")
