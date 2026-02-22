import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List


def _trace_enabled(config) -> bool:
    return getattr(config, "enable_trace", False)


def _trace_text(config, message: str) -> None:
    if _trace_enabled(config):
        print(f"[TRACE] {message}")


def _trace_tensor(config, name: str, tensor: torch.Tensor) -> None:
    if _trace_enabled(config):
        print(
            f"[TRACE] {name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, device={tensor.device}"
        )

# ==========================================
# 1. 配置类 (Config)
# ==========================================
@dataclass
class Qwen3Config:
    vocab_size: int = 151669          # 报告 Section 2: vocab size 151,669
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8      # 报告 Section 2: 使用 GQA
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768 # 报告 Section 3.2: 长上下文阶段支持 32K+
    rope_theta: float = 1000000.0     # 报告 Section 3.2: 基频从 10k 提升到 1M
    tie_word_embeddings: bool = False # 报告 Table 1: 部分模型 Tie Embedding 为 No
    enable_trace: bool = False        # 是否打印数据流向（调试用）

# ==========================================
# 2. RMSNorm (报告中全程使用 RMSNorm)
# ==========================================
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # 计算方差，沿最后一个维度
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

# ==========================================
# 3. RoPE (旋转位置编码)
# ==========================================
class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        # x: (batch, heads, seq_len, head_dim)
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# ==========================================
# 4. Attention 模块 (包含 QK-Norm, GQA, 无 Bias)
# ==========================================
class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        # 报告 Section 2: 移除 QKV-bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # 报告 Section 2: 引入 QK-Norm 确保训练稳定性
        # 注意：QK-Norm 是在 head_dim 上进行归一化
        self.q_norm = Qwen3RMSNorm(self.head_dim)
        self.k_norm = Qwen3RMSNorm(self.head_dim)
        
        self.rotary_emb = Qwen3RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(self, hidden_states, attention_mask, position_ids, layer_idx=None):
        prefix = f"layer{layer_idx:02d}.attn" if layer_idx is not None else "attn"
        batch_size, seq_len, _ = hidden_states.size()
        _trace_tensor(self.config, f"{prefix}.input_hidden_states", hidden_states)
        
        # 1. 线性投影
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        _trace_tensor(self.config, f"{prefix}.q_proj", query_states)
        _trace_tensor(self.config, f"{prefix}.k_proj", key_states)
        _trace_tensor(self.config, f"{prefix}.v_proj", value_states)
        
        # 2. 重塑形状 (batch, seq_len, heads, head_dim) -> (batch, heads, seq_len, head_dim)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        _trace_tensor(self.config, f"{prefix}.q_reshape", query_states)
        _trace_tensor(self.config, f"{prefix}.k_reshape", key_states)
        _trace_tensor(self.config, f"{prefix}.v_reshape", value_states)
        
        # 3. 应用 RoPE
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        _trace_tensor(self.config, f"{prefix}.q_after_rope", query_states)
        _trace_tensor(self.config, f"{prefix}.k_after_rope", key_states)
        
        # 4. 应用 QK-Norm (报告关键特性)
        # 输入形状：(batch, heads, seq_len, head_dim)，RMSNorm 沿最后一个维度 head_dim 归一化
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        _trace_tensor(self.config, f"{prefix}.q_after_qknorm", query_states)
        _trace_tensor(self.config, f"{prefix}.k_after_qknorm", key_states)
        
        # 5. GQA 处理：重复 KV 头以匹配 Q 头数量
        if self.num_kv_heads != self.num_heads:
            key_states = self.repeat_kv(key_states, self.num_heads // self.num_kv_heads)
            value_states = self.repeat_kv(value_states, self.num_heads // self.num_kv_heads)
        _trace_tensor(self.config, f"{prefix}.k_after_gqa", key_states)
        _trace_tensor(self.config, f"{prefix}.v_after_gqa", value_states)
        
        # 6. 计算 Attention 分数
        # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        _trace_tensor(self.config, f"{prefix}.attn_scores", attn_weights)
        
        # 7. 应用因果掩码 (Causal Mask)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            _trace_tensor(self.config, f"{prefix}.attn_scores_masked", attn_weights)
        
        # 8. Softmax 和 Dropout (此处省略 dropout)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        _trace_tensor(self.config, f"{prefix}.attn_probs", attn_weights)
        
        # 9. 加权求和
        attn_output = torch.matmul(attn_weights, value_states)
        _trace_tensor(self.config, f"{prefix}.attn_context", attn_output)
        
        # 10. 重塑并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        _trace_tensor(self.config, f"{prefix}.output", attn_output)
        
        return attn_output

    def repeat_kv(self, hidden_states, n_rep):
        """重复 KV 头以匹配 GQA 需求"""
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

# ==========================================
# 5. MLP 模块 (SwiGLU 结构)
# ==========================================
class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        # 报告 Section 2: 使用 SwiGLU
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x, layer_idx=None):
        prefix = f"layer{layer_idx:02d}.mlp" if layer_idx is not None else "mlp"
        _trace_tensor(self.config, f"{prefix}.input", x)
        # SwiGLU 公式：SiLU(Gate) * Up
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        _trace_tensor(self.config, f"{prefix}.gate_proj", gate)
        _trace_tensor(self.config, f"{prefix}.up_proj", up)
        hidden = self.act_fn(gate) * up
        _trace_tensor(self.config, f"{prefix}.swiglu_hidden", hidden)
        output = self.down_proj(hidden)
        _trace_tensor(self.config, f"{prefix}.output", output)
        return output

# ==========================================
# 6. Decoder Layer (整体结构)
# ==========================================
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        # 报告 Section 2: Pre-Norm 架构
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size)
        
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)

    def forward(self, hidden_states, attention_mask, position_ids, layer_idx=None):
        prefix = f"layer{layer_idx:02d}" if layer_idx is not None else "layer"
        _trace_tensor(self.config, f"{prefix}.input", hidden_states)
        # Attention 子层 (残差连接)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        _trace_tensor(self.config, f"{prefix}.after_input_layernorm", hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_ids, layer_idx=layer_idx)
        hidden_states = residual + hidden_states
        _trace_tensor(self.config, f"{prefix}.after_attn_residual", hidden_states)
        
        # MLP 子层 (残差连接)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        _trace_tensor(self.config, f"{prefix}.after_post_attn_layernorm", hidden_states)
        hidden_states = self.mlp(hidden_states, layer_idx=layer_idx)
        hidden_states = residual + hidden_states
        _trace_tensor(self.config, f"{prefix}.output", hidden_states)
        
        return hidden_states

# ==========================================
# 7. Qwen3 主模型
# ==========================================
class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size) # 最终归一化

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        _trace_tensor(self.config, "model.input_ids", input_ids)
        if position_ids is None:
            # 生成默认 position_ids
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
            _trace_text(self.config, "position_ids 未传入，已根据 input_ids 自动生成")
        _trace_tensor(self.config, "model.position_ids", position_ids)
        if attention_mask is not None:
            _trace_tensor(self.config, "model.attention_mask", attention_mask)
        
        hidden_states = self.embed_tokens(input_ids)
        _trace_tensor(self.config, "model.embedding_output", hidden_states)
        
        for layer_idx, layer in enumerate(self.layers):
            _trace_text(self.config, f"进入 Decoder Layer {layer_idx}")
            hidden_states = layer(hidden_states, attention_mask, position_ids, layer_idx=layer_idx)
            _trace_text(self.config, f"离开 Decoder Layer {layer_idx}")
            
        hidden_states = self.norm(hidden_states)
        _trace_tensor(self.config, "model.final_norm_output", hidden_states)
        return hidden_states

# ==========================================
# 8. 用于生成任务的模型 (带 LM Head)
# ==========================================
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 报告 Table 1: 部分模型 Tie Embedding
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None, labels=None):
        outputs = self.model(input_ids, attention_mask, position_ids)
        logits = self.lm_head(outputs)
        _trace_tensor(self.config, "lm_head.logits", logits)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            _trace_text(self.config, f"lm_head.loss={loss.item():.6f}")
            
        return logits, loss

# ==========================================
# 9. 辅助函数：生成因果掩码
# ==========================================
def generate_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len)

# ==========================================
# 10. 使用示例
# ==========================================
if __name__ == "__main__":
    # 初始化配置 (模拟一个小模型)
    config = Qwen3Config(
        hidden_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=2048,
        enable_trace=True,
    )
    
    model = Qwen3ForCausalLM(config)
    model.eval()
    
    # 模拟输入 (batch_size=1, seq_len=10)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    attention_mask = generate_causal_mask(10, input_ids.device)
    
    # 前向传播
    with torch.no_grad():
        logits, loss = model(input_ids, attention_mask=attention_mask)
        
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print("Qwen3 架构代码补充完成！")
    
    # 关于 Thinking Mode 的说明：
    # 报告中提到的 Thinking Mode (思考模式) 主要通过 Chat Template 和 特殊 Token 控制
    # 例如在输入中加入 <think> 标签，或在推理时控制生成长度 (Thinking Budget)
    # 架构本身支持动态生成长度，无需修改模型结构
