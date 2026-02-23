# qwen3_8b_moe_model.py 模型流向说明（详细版）

本文档对应实现文件 `qwen3_8b_moe_model.py`，目标是把这份 Qwen3 MoE 教学实现的“数据如何在模型中流动”讲清楚，方便你对照代码、日志和结构图定位问题。

## 1. 快速结论

- 这是一个 `Decoder-only` 架构，整体流向是：
  `input_ids -> Embedding -> (RMSNorm -> Attention -> Residual -> RMSNorm -> MoE/MLP -> Residual) x N -> Final RMSNorm -> LM Head`
- Attention 内部关键顺序是：
  `QKV 投影 -> QK-Norm -> RoPE -> GQA -> Scaled Dot-Product Attention -> O 投影`
- MoE 内部关键顺序是：
  `Gate 打分 -> softmax -> Top-K -> Expert 计算 -> 按路由权重加权 -> 聚合回写`

## 2. 代码结构总览

主要类与职责：

- `Qwen3MoeConfig`
  模型超参、MoE 超参、trace 开关。
- `Qwen3Attention`
  GQA + QK-Norm + RoPE + 因果自注意力。
- `Qwen3MLP`
  Dense 前馈（SwiGLU）。
- `Qwen3MoeSparseMoeBlock`
  稀疏 MoE 路由与专家聚合。
- `Qwen3DecoderLayer`
  一个解码层：Attention 子层 + MLP/MoE 子层（带残差）。
- `Qwen3MoeModel`
  token embedding、rotary embedding、多层堆叠、最终 norm。
- `Qwen3MoeForCausalLM`
  加 `lm_head`，可选 CE loss 和 router aux loss。

## 3. 记号与维度

- `B`: batch size
- `S`: sequence length
- `H`: hidden size
- `A`: attention heads (`num_attention_heads`)
- `Kv`: key/value heads (`num_key_value_heads`)
- `D`: head_dim
- `E`: num_experts
- `K`: top-k experts per token (`num_experts_per_tok`)
- `T`: token 总数，`T = B * S`

常见张量形状：

- `input_ids`: `[B, S]`
- `hidden_states`: `[B, S, H]`
- `router_logits`: `[B, S, E]`
- `logits`: `[B, S, vocab_size]`

## 4. 从输入到输出：端到端流向

### 4.1 进入模型

1. `input_ids` 进入 `embed_tokens`，得到 `hidden_states: [B, S, H]`
2. 若未传 `position_ids`，自动生成 `0..S-1` 的位置索引
3. 在 model 级计算 RoPE：
   - `cos, sin = rotary_emb(hidden_states, position_ids)`
   - `cos/sin` 形状一般是 `[B, S, D]`
4. 进入 `N` 个 Decoder Layer 循环

### 4.2 每个 Decoder Layer 内部

设当前层输入为 `x`：

1. Attention 子层
   - `residual1 = x`
   - `x = RMSNorm(x)`
   - `attn_out = Attention(x, mask, position_ids, position_embeddings=(cos,sin))`
   - `x = residual1 + attn_out`
2. FFN 子层（Dense 或 MoE）
   - `residual2 = x`
   - `x = RMSNorm(x)`
   - 若该层启用 MoE：`ffn_out, router_logits = MoE(x)`
   - 否则：`ffn_out = MLP(x)`
   - `x = residual2 + ffn_out`
3. 输出层结果给下一层

### 4.3 层堆叠结束后

1. `x = final RMSNorm(x)`
2. `logits = lm_head(x)`
3. 如果请求 `output_router_logits=True`，返回每个 MoE 层的 `router_logits`

## 5. Attention 详细流向

输入：`hidden_states: [B, S, H]`

1. 线性投影
   - `q = q_proj(x): [B, S, A*D]`
   - `k = k_proj(x): [B, S, Kv*D]`
   - `v = v_proj(x): [B, S, Kv*D]`
2. 变形为多头
   - `q -> [B, A, S, D]`
   - `k, v -> [B, Kv, S, D]`
3. **QK-Norm（先于 RoPE）**
   - `q = q_norm(q)`
   - `k = k_norm(k)`
4. RoPE
   - `q, k = apply_rotary_pos_emb(q, k, cos, sin)`
5. GQA 扩展 KV 头
   - 若 `Kv < A`，重复 `k/v` 到 `A` 个头
6. 注意力计算
   - `scores = q @ k^T / sqrt(D)`，形状 `[B, A, S, S]`
   - `scores += attention_mask`（因果 mask）
   - `probs = softmax(scores)`
   - `context = probs @ v`，形状 `[B, A, S, D]`
7. 输出投影
   - `context -> [B, S, A*D]`
   - `attn_out = o_proj(context)`，形状 `[B, S, H]`

## 6. MoE 详细流向

输入：`hidden_states: [B, S, H]`

1. 展平 token 维
   - `x = reshape -> [T, H]`
2. 路由打分
   - `router_logits = gate(x): [T, E]`
3. 概率与 Top-K
   - `routing_probs = softmax(router_logits, dim=-1): [T, E]`
   - `topk_probs, topk_idx = topk(routing_probs, K): [T, K]`
   - 可选再归一化：`topk_probs /= sum(topk_probs, dim=-1, keepdim=True)`
4. 稀疏派发 + 聚合
   - 初始化 `final_hidden = zeros([T, H])`
   - 对每个专家 `e`：
     - 找到命中该专家的 token 索引
     - 取出 token 子集跑 `expert_e(·)`
     - 乘上该 token 对专家 `e` 的路由权重
     - `index_add_` 累加回 `final_hidden`
5. 回到序列形状
   - `final_hidden -> [B, S, H]`
   - `router_logits -> [B, S, E]`
6. 返回
   - `return final_hidden, router_logits`

每个 token 的核心公式：

```text
y_t = Σ_{i ∈ TopK(t)} p_{t,i} * Expert_i(x_t)
```

## 7. 哪些层用 MoE

层选择逻辑：

```text
use_moe_layer(l) = use_moe and ((l + 1) % moe_layer_freq == 0)
```

- `l` 是 0-based 层号
- 这是 1-based 频率语义
- 例子：
  - `moe_layer_freq = 1` -> 每层都是 MoE
  - `moe_layer_freq = 2` -> 第 2、4、6...层是 MoE（0-based 下为 1,3,5...）

## 8. 损失计算流向

### 8.1 语言模型主损失（可选）

如果传 `labels`：

- `shift_logits = logits[..., :-1, :]`
- `shift_labels = labels[..., 1:]`
- `ce_loss = CrossEntropy(shift_logits, shift_labels)`

### 8.2 路由辅助损失（可选）

触发条件：

- `output_router_logits=True`
- 且 `router_aux_loss_coef > 0`

按每个 MoE 层：

1. `routing_weights = softmax(router_logits)`
2. `selected_experts = topk(routing_weights, K)`
3. `expert_mask = one_hot(selected_experts)`，形状 `[T, K, E]`
4. `tokens_per_expert = mean(expert_mask, dim=(0,1))`，形状 `[E]`
5. `router_prob_per_expert = mean(routing_weights, dim=0)`，形状 `[E]`
6. `layer_aux = sum(tokens_per_expert * router_prob_per_expert) * E`

多层平均后得到 `aux_loss`，最终：

```text
total_loss = ce_loss + router_aux_loss_coef * aux_loss
```

## 9. 与结构图节点的一一对应

图中节点到代码模块映射：

- `Embedding` -> `model.embed_tokens`
- `RotaryEmbedding` -> `model.rotary_emb`
- `Decoder Layer x n` -> `model.layers`
- `RMSNorm(Attention前)` -> `input_layernorm`
- `Attention` -> `self_attn`
- `RMSNorm(FFN前)` -> `post_attention_layernorm`
- `MoE/MLP` -> `mlp`
- `Final RMSNorm` -> `model.norm`
- `Output Linear` -> `lm_head`

## 10. Trace 与流程调试

配置项：

- `enable_trace`: 是否输出 trace
- `trace_level`:
  - `input_flow`: 仅主路径输入输出
  - `compact`: 中等细节（含路由分布摘要）
  - `verbose`: 张量级完整细节

辅助能力：

- `register_dimension_flow_hooks(model)`：
  注册模块级 forward hook，打印每步输入/输出形状、统计值、样本值、关键参数。
- `run_dimension_flow_demo(device)`：
  用小尺寸示例自动跑一遍完整流程并输出详细解释。

## 11. 最小运行示例

```bash
python qwen3_8b_moe_model.py
```

脚本入口默认会：

1. 选择设备（优先 `cuda:0`，否则 `cpu`）
2. 跑 `run_dimension_flow_demo`
3. 打印每个关键模块的维度流和 MoE 路由信息

## 12. 常见“看起来像流向错误”的点

- 图是抽象结构，代码是可执行细节：
  图里一个方框（如 Attention）在代码里会拆成很多子步骤。
- `RoPE` 放在 `model` 级预先计算并不改变数学逻辑：
  只是把 cos/sin 生成提到更外层复用。
- MoE 返回的是 `(hidden_states, router_logits)`：
  训练时可用 `router_logits` 计算辅助损失，推理时一般可忽略。
- `moe_layer_freq` 用 1-based 语义：
  不同实现若按 0-based，会造成“哪几层是 MoE”看起来不一致。

---

## 13. 固定配置逐步维度表（`B=1, S=10, H=1024`）

下面给出一套固定配置，所有 shape 都按这组超参精确展开，方便和 trace 日志逐行对照。

### 13.1 固定配置

```text
vocab_size = 32000
B = 1
S = 10
H = 1024
N = 4                  # num_hidden_layers
A = 8                  # num_attention_heads
Kv = 4                 # num_key_value_heads
D = 128                # head_dim, 且 A*D = 1024
E = 4                  # num_experts
K = 2                  # num_experts_per_tok
moe_intermediate_size = 512
moe_layer_freq = 1     # 每层都用 MoE
use_moe = True
```

派生量：

- `T = B * S = 10`
- 注意力打分矩阵维度是 `[B, A, S, S] = [1, 8, 10, 10]`

### 13.2 端到端主路径（整模）

| 步骤 | 模块/操作 | 输入 shape | 输出 shape | 作用 |
|---|---|---|---|---|
| 1 | `input_ids` | - | `[1, 10]` | 提供离散 token 索引，作为整条计算链起点。 |
| 2 | `position_ids`（自动生成） | - | `[1, 10]` | 提供每个 token 的位置编号，用于构造 RoPE。 |
| 3 | `attention_mask`（因果） | - | `[1, 1, 10, 10]` | 屏蔽未来位置信息，保证自回归因果性。 |
| 4 | `embed_tokens` | `[1, 10]` | `[1, 10, 1024]` | 把 token id 映射到连续语义空间。 |
| 5 | `rotary_emb` 生成 `cos/sin` | `hidden:[1,10,1024], position_ids:[1,10]` | `cos:[1,10,128], sin:[1,10,128]` | 生成旋转位置编码参数，供每层注意力共享。 |
| 6 | `Decoder Layer 0` | `[1, 10, 1024]` | `[1, 10, 1024]` | 第 1 层上下文建模与 MoE 非线性变换。 |
| 7 | `Decoder Layer 1` | `[1, 10, 1024]` | `[1, 10, 1024]` | 第 2 层进一步融合上下文并重编码。 |
| 8 | `Decoder Layer 2` | `[1, 10, 1024]` | `[1, 10, 1024]` | 第 3 层继续提炼语义表征。 |
| 9 | `Decoder Layer 3` | `[1, 10, 1024]` | `[1, 10, 1024]` | 第 4 层输出最终隐藏特征。 |
| 10 | `final RMSNorm` | `[1, 10, 1024]` | `[1, 10, 1024]` | 稳定输出分布，便于线性读出。 |
| 11 | `lm_head` | `[1, 10, 1024]` | `[1, 10, 32000]` | 把隐藏状态投影到词表 logits。 |

### 13.3 单个 Decoder Layer 内部（Layer i）

| 子步骤 | 操作 | 输入 shape | 输出 shape | 作用 |
|---|---|---|---|---|
| 1 | `residual1 = x` | `[1,10,1024]` | `[1,10,1024]` | 缓存 Attention 子层残差分支。 |
| 2 | `input_layernorm` | `[1,10,1024]` | `[1,10,1024]` | Pre-Norm，稳定注意力输入尺度。 |
| 3 | `self_attn` | `[1,10,1024]` | `[1,10,1024]` | 通过 token 间交互注入上下文信息。 |
| 4 | `x = residual1 + attn_out` | `[1,10,1024] + [1,10,1024]` | `[1,10,1024]` | 残差连接，保留原特征并叠加注意力增量。 |
| 5 | `residual2 = x` | `[1,10,1024]` | `[1,10,1024]` | 缓存 FFN/MoE 子层残差分支。 |
| 6 | `post_attention_layernorm` | `[1,10,1024]` | `[1,10,1024]` | 对进入 MoE/MLP 的特征再做归一化。 |
| 7 | `moe(x)`（本配置每层都启用 MoE） | `[1,10,1024]` | `hidden:[1,10,1024], router_logits:[1,10,4]` | 路由到 Top-K 专家并做稀疏非线性变换。 |
| 8 | `x = residual2 + moe_out` | `[1,10,1024] + [1,10,1024]` | `[1,10,1024]` | 残差融合，形成该层最终输出。 |

### 13.4 Attention 内部逐步 shape（精确）

输入：`hidden_states = [1,10,1024]`

| 步骤 | 操作 | shape | 作用 |
|---|---|---|---|
| 1 | `q_proj(hidden)` | `[1,10,1024]` | 生成查询向量，表示“我需要什么信息”。 |
| 2 | `k_proj(hidden)` | `[1,10,512]` | 生成键向量，表示“我能提供什么信息”。 |
| 3 | `v_proj(hidden)` | `[1,10,512]` | 生成值向量，表示“真实要聚合的内容”。 |
| 4 | `q -> view/transpose` | `[1,8,10,128]` | 拆成多头并调整到注意力计算布局。 |
| 5 | `k -> view/transpose` | `[1,4,10,128]` | KV 按较少头数布局，准备 GQA。 |
| 6 | `v -> view/transpose` | `[1,4,10,128]` | 同上，供后续加权求和使用。 |
| 7 | `q_norm(q)` | `[1,8,10,128]` | 归一化查询向量，稳定训练与数值。 |
| 8 | `k_norm(k)` | `[1,4,10,128]` | 归一化键向量，匹配 QK-Norm 设计。 |
| 9 | `apply_rotary_pos_emb(q,k,cos,sin)` | `q:[1,8,10,128], k:[1,4,10,128]` | 注入相对位置信息。 |
| 10 | GQA 扩展 `k` | `[1,8,10,128]` | 将 KV 头复制到与 Q 头数一致。 |
| 11 | GQA 扩展 `v` | `[1,8,10,128]` | 与扩展后的 K 对齐，便于逐头计算。 |
| 12 | `scores = q @ k^T / sqrt(D)` | `[1,8,10,10]` | 计算 token 间相关性打分。 |
| 13 | `scores + attention_mask` | `[1,8,10,10]` | 屏蔽未来位置，满足因果约束。 |
| 14 | `softmax(scores)` | `[1,8,10,10]` | 把打分转成归一化注意力权重。 |
| 15 | `context = probs @ v` | `[1,8,10,128]` | 按权重聚合 value，得到上下文表示。 |
| 16 | `context -> transpose/view` | `[1,10,1024]` | 合并多头，恢复隐藏维布局。 |
| 17 | `o_proj(context)` | `[1,10,1024]` | 线性融合多头结果，输出 attention 子层增量。 |

### 13.5 MoE 内部逐步 shape（精确）

输入：`hidden_states = [1,10,1024]`

| 步骤 | 操作 | shape | 作用 |
|---|---|---|---|
| 1 | flatten token 维 | `[10,1024]` | 把 batch+seq 合并，便于统一路由。 |
| 2 | `router_logits = gate(x)` | `[10,4]` | 计算每个 token 对所有专家的打分。 |
| 3 | `routing_probs = softmax(router_logits)` | `[10,4]` | 把打分转成可解释概率分布。 |
| 4 | `topk(routing_probs, k=2)` | `topk_probs:[10,2], topk_idx:[10,2]` | 为每个 token 选最重要的 2 个专家。 |
| 5 | （可选）Top-K 概率归一化 | `[10,2]` | 让被选专家权重和为 1，便于加权求和。 |
| 6 | `final_hidden` 初始化 | `[10,1024]` | 创建聚合缓冲区，累加专家输出。 |
| 7 | 对每个 expert `e` 收集 token | `token_indices:[n_e]` | 找出当前专家需要处理的 token 子集。 |
| 8 | `current_state = x[token_indices]` | `[n_e,1024]` | 抽取对应 token 的输入特征。 |
| 9 | expert MLP: `gate/up` | `[n_e,512]` | 专家内部线性扩展到中间维。 |
| 10 | expert MLP: `SiLU(gate)*up` | `[n_e,512]` | 完成 SwiGLU 非线性混合。 |
| 11 | expert MLP: `down` | `[n_e,1024]` | 映射回 hidden 维度。 |
| 12 | 乘对应路由权重 | `[n_e,1024]` | 体现“专家贡献度”加权。 |
| 13 | `index_add_` 聚合到 `final_hidden` | `[10,1024]` | 把所有专家贡献按 token 索引累加。 |
| 14 | reshape 回序列 | `[1,10,1024]` | 恢复到 `[B,S,H]` 的主干格式。 |
| 15 | `router_logits` reshape | `[1,10,4]` | 输出训练可用的路由打分。 |

说明：

- `n_e` 是当前 expert 实际命中的 token 数（动态值，`0 <= n_e <= 10`）。
- 全部 expert 的 `n_e` 之和等于 `T*K = 20`（每个 token 选择 2 个 expert）。

### 13.6 `output_router_logits=True` 时的返回维度

在本固定配置 `N=4`、且 `moe_layer_freq=1` 时：

- `router_logits` 列表长度 = 4
- 每层 router 形状 = `[1,10,4]`

即：

```text
all_router_logits = [
  [1,10,4],  # layer0
  [1,10,4],  # layer1
  [1,10,4],  # layer2
  [1,10,4],  # layer3
]
```

### 13.7 训练损失的固定维度对应

#### CE loss（语言模型主损失）

| 步骤 | 张量/操作 | shape | 作用 |
|---|---|---|---|
| 1 | `logits` | `[1,10,32000]` | 每个位置对词表的预测分数。 |
| 2 | `shift_logits = logits[..., :-1, :]` | `[1,9,32000]` | 去掉最后一个时间步，与右移标签对齐。 |
| 3 | `shift_labels = labels[..., 1:]` | `[1,9]` | 去掉第一个标签，构造 next-token 监督。 |
| 4 | `shift_logits.view(-1,32000)` | `[9,32000]` | 展平 batch/seq，输入交叉熵。 |
| 5 | `shift_labels.view(-1)` | `[9]` | 展平监督标签。 |

#### Router aux loss（路由辅助损失）

| 步骤 | 张量/操作 | shape | 作用 |
|---|---|---|---|
| 1 | 单层 `router_logits` reshape 后 | `[10,4]` | 把该层 token 路由打分拉平成二维。 |
| 2 | `selected_experts = topk(...)` | `[10,2]` | 确定每个 token 选中的专家索引。 |
| 3 | `expert_mask = one_hot(selected_experts)` | `[10,2,4]` | 构造 token-专家分配掩码。 |
| 4 | `tokens_per_expert` | `[4]` | 统计专家负载（命中频率）。 |
| 5 | `router_prob_per_expert` | `[4]` | 统计路由概率质量在专家上的分布。 |

这套固定维度可以直接对照 `trace_level=compact/verbose` 的日志逐项核查。
