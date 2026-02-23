import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trace_enabled(config) -> bool:
    return getattr(config, "enable_trace", False)


def _trace_level(config) -> str:
    return getattr(config, "trace_level", "input_flow").lower()


def _trace_verbose(config) -> bool:
    return _trace_level(config) == "verbose"


def _trace_compact_mode(config) -> bool:
    return _trace_level(config) == "compact"


def _trace_input_mode(config) -> bool:
    return _trace_level(config) in {"input_flow", "flow", "input"}


def _trace_text(config, message: str) -> None:
    if _trace_enabled(config):
        print(f"[TRACE] {message}")


def _trace_compact(config, message: str) -> None:
    if _trace_enabled(config) and _trace_compact_mode(config):
        print(f"[TRACE] {message}")


def _trace_input(config, message: str) -> None:
    if _trace_enabled(config) and _trace_input_mode(config):
        print(f"[TRACE] {message}")


def _trace_tensor(config, name: str, tensor: torch.Tensor) -> None:
    if _trace_enabled(config) and _trace_verbose(config):
        print(
            f"[TRACE] {name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}"
        )


@dataclass
class Qwen3MoeConfig:
    vocab_size: int = 151669
    hidden_size: int = 4096
    intermediate_size: int = 11008
    moe_intermediate_size: int = 1408
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = False
    enable_trace: bool = False
    trace_level: str = "input_flow"

    # MoE config
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    norm_topk_prob: bool = True
    moe_layer_freq: int = 1
    router_aux_loss_coef: float = 0.0


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = Qwen3RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        prefix = f"layer{layer_idx:02d}.attn" if layer_idx is not None else "attn"
        batch_size, seq_len, _ = hidden_states.size()
        attn_input = hidden_states
        _trace_tensor(self.config, f"{prefix}.input_hidden_states", hidden_states)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        _trace_compact(
            self.config,
            f"{prefix}.qkv_proj: q{tuple(query_states.shape)} k{tuple(key_states.shape)} v{tuple(value_states.shape)}",
        )
        _trace_tensor(self.config, f"{prefix}.q_proj", query_states)
        _trace_tensor(self.config, f"{prefix}.k_proj", key_states)
        _trace_tensor(self.config, f"{prefix}.v_proj", value_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        _trace_compact(
            self.config,
            f"{prefix}.qkv_head: q{tuple(query_states.shape)} k{tuple(key_states.shape)} v{tuple(value_states.shape)}",
        )
        _trace_tensor(self.config, f"{prefix}.q_reshape", query_states)
        _trace_tensor(self.config, f"{prefix}.k_reshape", key_states)
        _trace_tensor(self.config, f"{prefix}.v_reshape", value_states)

        # Qwen3 官方实现顺序：QK-Norm -> RoPE（而不是 RoPE -> QK-Norm）。
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        _trace_tensor(self.config, f"{prefix}.q_after_qknorm", query_states)
        _trace_tensor(self.config, f"{prefix}.k_after_qknorm", key_states)

        if position_embeddings is None:
            if position_ids is None:
                raise ValueError("position_ids or position_embeddings must be provided for RoPE.")
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        _trace_tensor(self.config, f"{prefix}.q_after_rope", query_states)
        _trace_tensor(self.config, f"{prefix}.k_after_rope", key_states)

        if self.num_kv_heads != self.num_heads:
            key_states = self.repeat_kv(key_states, self.num_heads // self.num_kv_heads)
            value_states = self.repeat_kv(value_states, self.num_heads // self.num_kv_heads)
        _trace_compact(
            self.config,
            f"{prefix}.after_qknorm_rope_gqa: q{tuple(query_states.shape)} k{tuple(key_states.shape)} v{tuple(value_states.shape)}",
        )
        _trace_tensor(self.config, f"{prefix}.k_after_gqa", key_states)
        _trace_tensor(self.config, f"{prefix}.v_after_gqa", value_states)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        _trace_tensor(self.config, f"{prefix}.attn_scores", attn_weights)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            _trace_tensor(self.config, f"{prefix}.attn_scores_masked", attn_weights)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        _trace_tensor(self.config, f"{prefix}.attn_probs", attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        _trace_compact(
            self.config,
            f"{prefix}.score_to_context: scores{tuple(attn_weights.shape)} -> context{tuple(attn_output.shape)}",
        )
        _trace_tensor(self.config, f"{prefix}.attn_context", attn_output)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        _trace_input(
            self.config,
            f"{prefix}.flow: in{tuple(attn_input.shape)} -> out{tuple(attn_output.shape)}",
        )
        _trace_compact(
            self.config,
            f"{prefix}.flow: in{tuple(attn_input.shape)} -> out{tuple(attn_output.shape)}",
        )
        _trace_tensor(self.config, f"{prefix}.output", attn_output)
        return attn_output


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        inner_dim = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inner_dim, bias=False)
        self.down_proj = nn.Linear(inner_dim, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        prefix = f"layer{layer_idx:02d}.mlp" if layer_idx is not None else "mlp"
        mlp_input = x
        _trace_tensor(self.config, f"{prefix}.input", x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        _trace_tensor(self.config, f"{prefix}.gate_proj", gate)
        _trace_tensor(self.config, f"{prefix}.up_proj", up)
        hidden = self.act_fn(gate) * up
        _trace_tensor(self.config, f"{prefix}.swiglu_hidden", hidden)
        output = self.down_proj(hidden)
        _trace_input(
            self.config,
            f"{prefix}.flow: in{tuple(mlp_input.shape)} -> out{tuple(output.shape)}",
        )
        _trace_compact(
            self.config,
            f"{prefix}.flow: in{tuple(mlp_input.shape)} -> gate/up{tuple(gate.shape)} -> swiglu{tuple(hidden.shape)} -> out{tuple(output.shape)}",
        )
        _trace_tensor(self.config, f"{prefix}.output", output)
        return output


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        if config.num_experts_per_tok > config.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({config.num_experts_per_tok}) must be <= num_experts ({config.num_experts})"
            )
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size

        # Gate -> Linear：为每个 token 生成到所有 expert 的路由打分（router logits）。
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # Experts：由多个独立 MLP 组成的专家池；每个 token 只会激活 Top-K 个 expert（稀疏 MoE）。
        self.experts = nn.ModuleList(
            [
                Qwen3MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        稀疏 MoE 前向流程（与示意图一一对应）：
        1) 输入 [B, S, H] 拉平到 [B*S, H]
        2) gate 计算 router_logits: [B*S, E]
        3) softmax 后取 Top-K，得到 routing_weights / selected_experts
        4) 初始化全零输出 final_hidden_states: [B*S, H]
        5) 按 expert 聚合：Expert(x) * p，再用 index_add_ 写回
        6) reshape 回 [B, S, H]，并返回 (final_hidden_states, router_logits)

        核心公式（对每个 token）：
            y_t = sum_{i in TopK(t)} p_{t,i} * Expert_i(x_t)
        """
        prefix = f"layer{layer_idx:02d}.moe" if layer_idx is not None else "moe"
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # [B, S, H] -> [B*S, H]
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        _trace_tensor(self.config, f"{prefix}.input", hidden_states)

        # router_logits: [B*S, E]
        router_logits = self.gate(flat_hidden_states)
        _trace_tensor(self.config, f"{prefix}.router_logits", router_logits)

        # 先 softmax 成概率，再选 Top-K expert（每个 token 仅激活 K 个 expert）。
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # 可选：对 Top-K 概率再归一化，使其和为 1（Qwen/Mixtral 的常见差异点之一）。
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(flat_hidden_states.dtype)
        # 初始化最终聚合输出。
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=flat_hidden_states.dtype,
            device=flat_hidden_states.device,
        )

        # one-hot 后得到 expert_mask，便于反查“某个 expert 被哪些 token 选中”。
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            # idx: 该 token 命中的是 Top-K 中第几个位置；token_indices: 对应 token 的平铺索引。
            idx, token_indices = torch.where(expert_mask[expert_idx])
            if token_indices.numel() == 0:
                continue

            # 只取被当前 expert 选中的 token，走该 expert 的 MLP。
            current_state = flat_hidden_states.index_select(0, token_indices)
            # expert 输出乘上对应路由权重（图中的“×”），再准备聚合（图中的“+”）。
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[token_indices, idx].unsqueeze(-1)
            # 使用 index_add_ 将当前 expert 的贡献累加回总输出。
            final_hidden_states.index_add_(0, token_indices, current_hidden_states.to(flat_hidden_states.dtype))

        # [B*S, H] -> [B, S, H]
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # 额外返回 router_logits，便于训练时加入路由相关损失（如负载均衡 loss）。
        router_logits = router_logits.reshape(batch_size, sequence_length, self.num_experts)

        if _trace_enabled(self.config) and _trace_compact_mode(self.config):
            expert_hist = torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts)
            _trace_compact(
                self.config,
                f"{prefix}.routing: top_k={self.top_k} expert_hist={expert_hist.tolist()}",
            )
        _trace_input(
            self.config,
            f"{prefix}.flow: in{tuple(hidden_states.shape)} -> out{tuple(final_hidden_states.shape)}",
        )
        _trace_tensor(self.config, f"{prefix}.output", final_hidden_states)
        return final_hidden_states, router_logits


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.self_attn = Qwen3Attention(config)
        if config.moe_layer_freq <= 0:
            raise ValueError(f"moe_layer_freq must be positive, got {config.moe_layer_freq}.")
        # 与 HF Qwen3-MoE 一致：按 1-based 层号判断稀疏层位置。
        self.use_moe = config.use_moe and ((layer_idx + 1) % config.moe_layer_freq == 0)
        if self.use_moe:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        idx = self.layer_idx if layer_idx is None else layer_idx
        prefix = f"layer{idx:02d}"
        layer_input = hidden_states
        _trace_tensor(self.config, f"{prefix}.input", hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        _trace_tensor(self.config, f"{prefix}.after_input_layernorm", hidden_states)
        attn_output = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            position_embeddings=position_embeddings,
            layer_idx=idx,
        )
        hidden_states = residual + attn_output
        after_attn_residual = hidden_states
        _trace_tensor(self.config, f"{prefix}.after_attn_residual", hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        _trace_tensor(self.config, f"{prefix}.after_post_attn_layernorm", hidden_states)

        router_logits = None
        if self.use_moe:
            mlp_output, router_logits = self.mlp(hidden_states, layer_idx=idx)
        else:
            mlp_output = self.mlp(hidden_states, layer_idx=idx)
        hidden_states = residual + mlp_output
        _trace_input(
            self.config,
            f"{prefix}.flow: in{tuple(layer_input.shape)} -> attn{tuple(attn_output.shape)} -> mlp{tuple(mlp_output.shape)} -> out{tuple(hidden_states.shape)}",
        )
        _trace_compact(
            self.config,
            f"{prefix}.flow: in{tuple(layer_input.shape)} -> attn{tuple(attn_output.shape)} -> res1{tuple(after_attn_residual.shape)} -> mlp{tuple(mlp_output.shape)} -> out{tuple(hidden_states.shape)}",
        )
        _trace_tensor(self.config, f"{prefix}.output", hidden_states)
        return hidden_states, router_logits


def load_balancing_loss_func(
    router_logits: List[torch.Tensor],
    num_experts: int,
    top_k: int,
) -> Optional[torch.Tensor]:
    if len(router_logits) == 0:
        return None

    losses = []
    for layer_router_logits in router_logits:
        layer_router_logits = layer_router_logits.reshape(-1, num_experts)
        routing_weights = F.softmax(layer_router_logits, dim=-1, dtype=torch.float32)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts).float()
        # [tokens, top_k, experts] -> [experts]
        tokens_per_expert = expert_mask.mean(dim=(0, 1))
        router_prob_per_expert = routing_weights.mean(dim=0)
        layer_loss = torch.sum(tokens_per_expert * router_prob_per_expert) * num_experts
        losses.append(layer_loss)
    return torch.stack(losses).mean()


class Qwen3MoeModel(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 与官方实现一致：在 model 级计算 RoPE，再传给各层 Attention 复用。
        self.rotary_emb = Qwen3RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        _trace_text(
            self.config,
            f"Trace level={_trace_level(self.config)}. Available: input_flow / compact / verbose",
        )
        _trace_tensor(self.config, "model.input_ids", input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            _trace_text(self.config, "position_ids was not provided, generated from input_ids")
        _trace_tensor(self.config, "model.position_ids", position_ids)
        if attention_mask is not None:
            _trace_tensor(self.config, "model.attention_mask", attention_mask)

        hidden_states = self.embed_tokens(input_ids)
        _trace_input(self.config, f"model.embedding: input_ids -> hidden_states{tuple(hidden_states.shape)}")
        _trace_compact(self.config, f"model.embedding: {tuple(hidden_states.shape)}")
        _trace_tensor(self.config, "model.embedding_output", hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        _trace_input(
            self.config,
            f"model.rotary_emb: position_ids{tuple(position_ids.shape)} -> cos/sin{tuple(position_embeddings[0].shape)}",
        )

        all_router_logits: List[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            if _trace_verbose(self.config):
                _trace_text(self.config, f"enter decoder layer {layer_idx}")
            elif _trace_compact_mode(self.config):
                _trace_text(self.config, f"===== Decoder Layer {layer_idx:02d} =====")

            hidden_states, layer_router_logits = layer(
                hidden_states,
                attention_mask,
                position_ids,
                position_embeddings=position_embeddings,
                layer_idx=layer_idx,
            )
            if output_router_logits and layer_router_logits is not None:
                all_router_logits.append(layer_router_logits)
            if _trace_verbose(self.config):
                _trace_text(self.config, f"leave decoder layer {layer_idx}")

        hidden_states = self.norm(hidden_states)
        _trace_input(self.config, f"model.final_norm: {tuple(hidden_states.shape)}")
        _trace_compact(self.config, f"model.final_norm: {tuple(hidden_states.shape)}")
        _trace_tensor(self.config, "model.final_norm_output", hidden_states)

        if output_router_logits:
            return hidden_states, all_router_logits
        return hidden_states, None


class Qwen3MoeForCausalLM(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_router_logits: bool = False,
    ):
        hidden_states, router_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_router_logits=output_router_logits,
        )
        logits = self.lm_head(hidden_states)
        _trace_input(
            self.config,
            f"lm_head.flow: hidden_states{tuple(hidden_states.shape)} -> logits{tuple(logits.shape)}",
        )
        _trace_compact(
            self.config,
            f"lm_head.flow: hidden{tuple(hidden_states.shape)} -> logits{tuple(logits.shape)}",
        )
        _trace_tensor(self.config, "lm_head.logits", logits)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            _trace_text(self.config, f"lm_head.ce_loss={loss.item():.6f}")

        aux_loss = None
        if output_router_logits and self.config.router_aux_loss_coef > 0:
            aux_loss = load_balancing_loss_func(
                router_logits if router_logits is not None else [],
                num_experts=self.config.num_experts,
                top_k=self.config.num_experts_per_tok,
            )
            if aux_loss is not None:
                scaled_aux = self.config.router_aux_loss_coef * aux_loss
                loss = scaled_aux if loss is None else (loss + scaled_aux)
                _trace_text(self.config, f"lm_head.router_aux_loss={aux_loss.item():.6f}")

        if output_router_logits:
            return logits, loss, router_logits, aux_loss
        return logits, loss


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask.unsqueeze(0).unsqueeze(0)


def _extract_first_tensor(obj) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            tensor = _extract_first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _preview_single_data(tensor: Optional[torch.Tensor], max_items: int = 4) -> str:
    if tensor is None:
        return "n/a"
    if tensor.numel() == 0:
        return "[]"

    view_tensor = tensor.detach()
    if view_tensor.dim() >= 3:
        vec = view_tensor[0, 0, : min(max_items, view_tensor.shape[-1])]
    elif view_tensor.dim() == 2:
        vec = view_tensor[0, : min(max_items, view_tensor.shape[-1])]
    elif view_tensor.dim() == 1:
        vec = view_tensor[: min(max_items, view_tensor.shape[0])]
    else:
        vec = view_tensor.reshape(-1)[:max_items]

    vec = vec.to(torch.float32).cpu().tolist()
    return "[" + ", ".join(f"{x:.4f}" for x in vec) + "]"


def _tensor_stats(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return "n/a"
    if tensor.numel() == 0:
        return "empty"
    t = tensor.detach().to(torch.float32)
    mean = t.mean().item()
    std = t.std(unbiased=False).item()
    t_min = t.min().item()
    t_max = t.max().item()
    return f"mean={mean:.4f}, std={std:.4f}, min={t_min:.4f}, max={t_max:.4f}"


def _module_desc_zh(module_name: str) -> str:
    if module_name == "model.embed_tokens":
        return "词嵌入：把 token id 映射为向量表示"
    if module_name == "model.rotary_emb":
        return "旋转位置编码：生成 RoPE 的 cos/sin，并供各层注意力复用"
    if module_name == "model.norm":
        return "最终归一化：稳定输出分布"
    if module_name == "lm_head":
        return "语言模型头：把隐藏状态映射到词表 logits"

    if ".input_layernorm" in module_name:
        return "输入层归一化（Pre-Norm）"
    if ".self_attn" in module_name:
        return "自注意力：让每个 token 融合上下文信息"
    if ".post_attention_layernorm" in module_name:
        return "注意力后归一化"
    if ".mlp" in module_name:
        return "前馈网络/MoE：进行非线性变换（MoE 时含路由）"

    return "模块计算"


def _module_extra_detail_zh(
    module_name: str,
    module: nn.Module,
    inputs,
    outputs,
) -> List[str]:
    lines: List[str] = []
    in_tensor = _extract_first_tensor(inputs)

    if isinstance(module, nn.Embedding):
        lines.append(
            f"参数: vocab_size={module.num_embeddings}, hidden_size={module.embedding_dim}"
        )
    elif isinstance(module, Qwen3Attention):
        lines.append(
            f"参数: num_heads={module.num_heads}, num_kv_heads={module.num_kv_heads}, head_dim={module.head_dim}"
        )
        lines.append(
            "内部步骤: QKV投影 -> QK-Norm -> RoPE -> GQA -> Softmax -> 加权求和"
        )
        if in_tensor is not None and in_tensor.dim() == 3:
            b, s, _ = in_tensor.shape
            lines.append(
                "内部维度: "
                f"Q=({b},{s},{module.num_heads},{module.head_dim}), "
                f"K/V=({b},{s},{module.num_kv_heads},{module.head_dim}), "
                f"AttnScore=({b},{module.num_heads},{s},{s})"
            )
    elif isinstance(module, Qwen3RotaryEmbedding):
        lines.append(
            f"参数: head_dim={module.dim}, max_position_embeddings={module.max_position_embeddings}, rope_theta={module.base}"
        )
        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[0], torch.Tensor):
            lines.append(
                f"输出: cos{tuple(outputs[0].shape)}, sin{tuple(outputs[1].shape)}"
            )
    elif isinstance(module, Qwen3MoeSparseMoeBlock):
        lines.append(
            f"参数: num_experts={module.num_experts}, top_k={module.top_k}, norm_topk_prob={module.norm_topk_prob}"
        )
        lines.append("内部步骤: gate打分 -> softmax -> top-k路由 -> expert并行 -> 加权聚合")
        if isinstance(outputs, tuple) and len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
            router_logits = outputs[1].detach().to(torch.float32)
            routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
            topk_probs, topk_indices = torch.topk(routing_probs, k=module.top_k, dim=-1)
            if module.norm_topk_prob:
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
            expert_hist = torch.bincount(topk_indices.reshape(-1), minlength=module.num_experts)
            lines.append(f"路由命中计数: expert_hist={expert_hist.cpu().tolist()}")
            first_token_experts = topk_indices[0, 0].cpu().tolist()
            first_token_probs = [round(float(x), 4) for x in topk_probs[0, 0].cpu().tolist()]
            lines.append(
                f"首token路由: experts={first_token_experts}, probs={first_token_probs}"
            )
    elif isinstance(module, Qwen3MLP):
        lines.append(
            f"参数: hidden_size={module.gate_proj.in_features}, intermediate_size={module.gate_proj.out_features}"
        )
        lines.append("内部步骤: SiLU(gate_proj(x)) * up_proj(x) -> down_proj")
    elif isinstance(module, Qwen3RMSNorm):
        lines.append(f"参数: eps={module.variance_epsilon}")
    elif isinstance(module, nn.Linear) and module_name == "lm_head":
        lines.append(f"参数: hidden_size={module.in_features}, vocab_size={module.out_features}")

    return lines


def register_dimension_flow_hooks(
    model: nn.Module,
    max_items: int = 4,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    注册前向 hook，打印模块级维度流：
    - 当前模块名
    - 输入维度 -> 输出维度
    - 一个样本数据切片（默认取第 0 个 batch 第 0 个 token 的前几个元素）
    """

    def _should_trace(name: str) -> bool:
        if name in {"model.embed_tokens", "model.rotary_emb", "model.norm", "lm_head"}:
            return True
        if ".layers." in name:
            tracked_suffixes = (
                ".input_layernorm",
                ".self_attn",
                ".post_attention_layernorm",
                ".mlp",
            )
            return name.endswith(tracked_suffixes)
        return False

    step_state = {"step": 0}

    def _build_hook(module_name: str):
        def _hook(_module, inputs, outputs):
            step_state["step"] += 1
            in_tensor = _extract_first_tensor(inputs)
            out_tensor = _extract_first_tensor(outputs)
            in_shape = tuple(in_tensor.shape) if in_tensor is not None else None
            out_shape = tuple(out_tensor.shape) if out_tensor is not None else None
            sample = _preview_single_data(out_tensor, max_items=max_items)
            desc = _module_desc_zh(module_name)
            input_stats = _tensor_stats(in_tensor)
            output_stats = _tensor_stats(out_tensor)
            extra_lines = _module_extra_detail_zh(module_name, _module, inputs, outputs)
            print(f"[流程 {step_state['step']:02d}] 模块：{module_name}")
            print(f"  作用：{desc}")
            print(f"  输入形状：{in_shape}")
            print(f"  输入统计：{input_stats}")
            print(f"  输出形状：{out_shape}")
            print(f"  输出统计：{output_stats}")
            print(f"  输出样本（第0个样本/第0个token前{max_items}维）：{sample}")
            for detail in extra_lines:
                print(f"  细节：{detail}")

            if (
                isinstance(_module, Qwen3MoeSparseMoeBlock)
                and isinstance(outputs, tuple)
                and len(outputs) > 1
                and isinstance(outputs[1], torch.Tensor)
            ):
                router_tensor = outputs[1]
                print(f"  附加输出 router_logits 形状：{tuple(router_tensor.shape)}")
                print(f"  附加输出 router_logits 统计：{_tensor_stats(router_tensor)}")
                print("  含义：每个 token 对每个 expert 的路由打分（训练可用于负载均衡损失）")

        return _hook

    handles: List[torch.utils.hooks.RemovableHandle] = []
    for name, module in model.named_modules():
        if _should_trace(name):
            handles.append(module.register_forward_hook(_build_hook(name)))
    return handles


def run_dimension_flow_demo(device: torch.device) -> None:
    """
    用一组小尺寸自造输入，展示“一个数据在整个模型流程中的维度变化”。
    """
    print("\n===== 维度流动演示（中文详细版）=====")
    print("符号说明：")
    print("  B = batch size（批大小）")
    print("  S = sequence length（序列长度）")
    print("  H = hidden size（隐藏维度）")
    print("  E = num_experts（专家数量）")
    print("  K = top-k（每个 token 选择的专家数）")
    print("-" * 60)
    demo_config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        enable_trace=False,
    )

    demo_model = Qwen3MoeForCausalLM(demo_config).to(device)
    demo_model.eval()

    # 自造输入：B=1, S=6
    input_ids = torch.tensor([[1, 7, 3, 9, 5, 11]], device=device)
    attention_mask = generate_causal_mask(seq_len=input_ids.shape[1], device=device)

    print(f"[输入] input_ids 形状：{tuple(input_ids.shape)}，数值：{input_ids.tolist()}")
    print(f"[输入] attention_mask 形状：{tuple(attention_mask.shape)}")
    print("-" * 60)

    handles = register_dimension_flow_hooks(demo_model, max_items=4)
    with torch.no_grad():
        logits, _, router_logits, _ = demo_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_router_logits=True,
        )
    for handle in handles:
        handle.remove()

    print("-" * 60)
    print(f"[输出] 最终 logits 形状：{tuple(logits.shape)}")
    print(f"[输出] logits 样本（第0个样本第0个token前4维）：{_preview_single_data(logits, max_items=4)}")
    print(f"[输出] 捕获到的 router 层数：{len(router_logits)}")
    for idx, layer_router in enumerate(router_logits):
        print(f"  - 第 {idx} 层 router_logits 形状：{tuple(layer_router.shape)}")
    print("说明：如果该层是 MoE 层，router_logits 形状通常是 [B, S, E]。")


# Backward-compatible aliases with the dense-file naming style.
Qwen3Config = Qwen3MoeConfig
Qwen3Model = Qwen3MoeModel
Qwen3ForCausalLM = Qwen3MoeForCausalLM


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    run_dimension_flow_demo(device)

    # 如需保留原始较大配置示例，可把该开关改为 True。
    run_full_example = False
    if run_full_example:
        config = Qwen3MoeConfig(
            vocab_size=32000,
            hidden_size=1024,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=128,
            moe_intermediate_size=512,
            use_moe=True,
            num_experts=4,
            num_experts_per_tok=2,
            enable_trace=True,
            trace_level="input_flow",
        )

        model = Qwen3MoeForCausalLM(config).to(device)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
        attention_mask = generate_causal_mask(10, device)

        with torch.no_grad():
            logits, loss, router_logits, aux_loss = model(
                input_ids,
                attention_mask=attention_mask,
                output_router_logits=True,
            )

        print(f"输入形状: {input_ids.shape}")
        print(f"输出 logits 形状: {logits.shape}")
        print(f"捕获到的 router 层数: {len(router_logits)}")
        print(f"总损失: {loss}")
        print(f"路由辅助损失: {aux_loss}")
