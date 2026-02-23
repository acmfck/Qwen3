import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# --- 1. Mock 依赖类，保证代码可运行 ---
class Qwen3MoeConfig:
    def __init__(self):
        self.hidden_size = 2
        self.num_experts = 4
        self.num_experts_per_tok = 2
        self.norm_topk_prob = True
        self.moe_intermediate_size = 4

class Qwen3MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        # 简单弄一个线性层模拟专家，为了展示效果，权重初始化为1
        self.fc = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        nn.init.constant_(self.fc.weight, 1.0) 

    def forward(self, x):
        return self.fc(x)

# 忽略 trace 函数
def _trace_tensor(*args, **kwargs): pass
def _trace_enabled(*args, **kwargs): return False
def _trace_compact_mode(*args, **kwargs): return False
def _trace_compact(*args, **kwargs): pass
def _trace_input(*args, **kwargs): pass

# --- 2. 你提供的核心代码 ---
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

        # 维度变化 1: [B, S, H] -> [B*S, H]
        print(f"[shape] hidden_states      : {tuple(hidden_states.shape)}")
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        print(f"[shape] flat_hidden_states : {tuple(flat_hidden_states.shape)}")

        # 维度变化 2: gate 输出 router logits [B*S, E]
        router_logits = self.gate(flat_hidden_states)
        print(f"[shape] router_logits      : {tuple(router_logits.shape)}")

        # 维度变化 3: softmax + topk，得到 [B*S, K]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        print(f"[shape] routing_weights    : {tuple(routing_weights.shape)}")
        print(f"[shape] selected_experts   : {tuple(selected_experts.shape)}")
        
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        routing_weights = routing_weights.to(flat_hidden_states.dtype)
        # 维度变化 4: 初始化聚合缓冲区 [B*S, H]
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=flat_hidden_states.dtype,
            device=flat_hidden_states.device,
        )
        print(f"[shape] final_hidden_init  : {tuple(final_hidden_states.shape)}")

        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        # [模拟打印]: 看一下每个 token 选了谁，权重是多少
        print(f"--- 路由分配结果 ---")
        for i in range(batch_size * sequence_length):
            print(f"Token {i} 选择了专家: {selected_experts[i].tolist()}, 权重为: {routing_weights[i].tolist()}")
        print("-" * 20)

        for expert_idx in range(self.num_experts):
            idx, token_indices = torch.where(expert_mask[expert_idx])
            if token_indices.numel() == 0:
                continue

            current_state = flat_hidden_states.index_select(0, token_indices)
            current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[token_indices, idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_indices, current_hidden_states.to(flat_hidden_states.dtype))

            # [模拟打印]: 看一下每个专家处理了哪些 token
            print(
                f"专家 E{expert_idx} 被唤醒，处理 Token={token_indices.tolist()} | "
                f"current_state={tuple(current_state.shape)} -> expert_out={tuple(current_hidden_states.shape)}"
            )

        # 维度变化 5: [B*S, H] -> [B, S, H]
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # 维度变化 6: router logits 同步 reshape 回 [B, S, E]
        router_logits = router_logits.reshape(batch_size, sequence_length, self.num_experts)
        print(f"[shape] final_hidden_out   : {tuple(final_hidden_states.shape)}")
        print(f"[shape] router_logits_out  : {tuple(router_logits.shape)}")

        return final_hidden_states, router_logits

# --- 3. 运行测试用例 ---
def run_case(moe_block: Qwen3MoeSparseMoeBlock, x: torch.Tensor, name: str) -> None:
    print("\n" + "=" * 60)
    print(f"Case: {name}")
    print(f"输入 x 形状: {tuple(x.shape)}")
    print(x)
    out, logits = moe_block(x)
    print("-" * 20)
    print(f"输出 out 形状   : {tuple(out.shape)}")
    print(f"输出 logits 形状: {tuple(logits.shape)}")
    print("输出 out 数值:")
    print(out)


if __name__ == "__main__":
    torch.manual_seed(42) # 固定随机种子以复现
    config = Qwen3MoeConfig()
    moe_block = Qwen3MoeSparseMoeBlock(config)

    # Case 1: B=1, S=3, H=2
    x_case1 = torch.tensor([
        [[1.0, 1.0],  # Token 0
         [2.0, 2.0],  # Token 1
         [3.0, 3.0]]  # Token 2
    ], dtype=torch.float32)

    # Case 2: B=2, S=4, H=2
    x_case2 = torch.arange(1, 2 * 4 * 2 + 1, dtype=torch.float32).reshape(2, 4, 2)

    # Case 3: B=3, S=1, H=2
    x_case3 = torch.tensor(
        [
            [[1.0, -1.0]],
            [[2.0, -2.0]],
            [[3.0, -3.0]],
        ],
        dtype=torch.float32,
    )

    # 依次运行，观察不同输入维度下的形状变化
    run_case(moe_block, x_case1, "B=1, S=3, H=2")
    run_case(moe_block, x_case2, "B=2, S=4, H=2")
    run_case(moe_block, x_case3, "B=3, S=1, H=2")
