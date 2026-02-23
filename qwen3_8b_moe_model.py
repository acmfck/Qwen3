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
        position_ids: torch.Tensor,
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

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        _trace_tensor(self.config, f"{prefix}.q_after_rope", query_states)
        _trace_tensor(self.config, f"{prefix}.k_after_rope", key_states)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        _trace_tensor(self.config, f"{prefix}.q_after_qknorm", query_states)
        _trace_tensor(self.config, f"{prefix}.k_after_qknorm", key_states)

        if self.num_kv_heads != self.num_heads:
            key_states = self.repeat_kv(key_states, self.num_heads // self.num_kv_heads)
            value_states = self.repeat_kv(value_states, self.num_heads // self.num_kv_heads)
        _trace_compact(
            self.config,
            f"{prefix}.after_rope_qknorm_gqa: q{tuple(query_states.shape)} k{tuple(key_states.shape)} v{tuple(value_states.shape)}",
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

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
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
        prefix = f"layer{layer_idx:02d}.moe" if layer_idx is not None else "moe"
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        _trace_tensor(self.config, f"{prefix}.input", hidden_states)

        router_logits = self.gate(flat_hidden_states)
        _trace_tensor(self.config, f"{prefix}.router_logits", router_logits)

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(flat_hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=flat_hidden_states.dtype,
            device=flat_hidden_states.device,
        )

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            idx, token_indices = torch.where(expert_mask[expert_idx])
            if token_indices.numel() == 0:
                continue

            current_state = flat_hidden_states.index_select(0, token_indices)
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[token_indices, idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, current_hidden_states.to(flat_hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
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
        self.use_moe = config.use_moe and (layer_idx % config.moe_layer_freq == 0)
        if self.use_moe:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        idx = self.layer_idx if layer_idx is None else layer_idx
        prefix = f"layer{idx:02d}"
        layer_input = hidden_states
        _trace_tensor(self.config, f"{prefix}.input", hidden_states)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        _trace_tensor(self.config, f"{prefix}.after_input_layernorm", hidden_states)
        attn_output = self.self_attn(hidden_states, attention_mask, position_ids, layer_idx=idx)
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
        tokens_per_expert = expert_mask.mean(dim=0)
        router_prob_per_expert = routing_weights.mean(dim=0)
        layer_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0)) * num_experts
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


# Backward-compatible aliases with the dense-file naming style.
Qwen3Config = Qwen3MoeConfig
Qwen3Model = Qwen3MoeModel
Qwen3ForCausalLM = Qwen3MoeForCausalLM


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Router layers captured: {len(router_logits)}")
    print(f"Loss: {loss}")
    print(f"Aux loss: {aux_loss}")
