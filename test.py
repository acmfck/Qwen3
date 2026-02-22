import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 定义参数
hidden_size = 2
intermediate_size = 4
batch_size = 1

# 2. 创建输入 (模拟数据)
x = torch.tensor([[1.0, 2.0]])

# 3. 创建线性层 (bias=False 符合 Qwen3 风格)
gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

# 4. 初始化权重为 1 (为了方便手动验证)
nn.init.ones_(gate_proj.weight)
nn.init.ones_(up_proj.weight)
nn.init.ones_(down_proj.weight)

# 5. 前向传播 (SwiGLU 流程)
# Gate 分支
gate_output = gate_proj(x)        # [3.0, 3.0, 3.0, 3.0]
gate_activated = F.silu(gate_output) # SiLU 激活

# Up 分支
up_output = up_proj(x)            # [3.0, 3.0, 3.0, 3.0]

# 乘法操作 (Gating)
hidden_state = gate_activated * up_output # [8.55..., 8.55..., ...]

# Down 分支
output = down_proj(hidden_state)  # [34.2..., 34.2...]

print(f"输入：{x}")
print(f"Gate 激活后：{gate_activated}")
print(f"Up 输出：{up_output}")
print(f"相乘后：{hidden_state}")
print(f"最终输出：{output}")