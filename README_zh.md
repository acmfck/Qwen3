# Qwen3

<p align="center">
    <img src="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/logo_qwen3.png" width="400"/>
<p>

<p align="center">
          💜 <a href="https://chat.qwen.ai/"><b>Qwen Chat</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2505.09388">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://qwenlm.github.io/blog/qwen3/">Blog</a> &nbsp&nbsp ｜ &nbsp&nbsp📖 <a href="https://qwen.readthedocs.io/">Documentation</a>
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-Demo">Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp
</p>


访问我们的 Hugging Face 或 ModelScope 组织（点击上方链接），搜索以 `Qwen3-` 开头的权重，或访问 [Qwen3 collection](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)，即可找到所需内容！祝使用愉快！

想了解更多 Qwen3，欢迎阅读我们的文档 \[[EN](https://qwen.readthedocs.io/en/latest/)|[ZH](https://qwen.readthedocs.io/zh-cn/latest/)\]。文档包含以下章节：

- Quickstart：基础用法与演示；
- Inference：基于 Transformers 的推理指南，包括批量推理、流式输出等；
- Run Locally：在 CPU/GPU 本地运行 LLM 的说明，涵盖 llama.cpp、Ollama、LM Studio 等框架；
- Deployment：使用 SGLang、vLLM、TGI 等框架进行大规模推理部署的示例；
- Quantization：使用 GPTQ、AWQ 进行量化，以及制作高质量 GGUF 量化文件的指南；
- Training：后训练指南，包括使用 Axolotl、LLaMA-Factory 等框架进行 SFT 和 RLHF（TODO）；
- Framework：在 RAG、Agent 等应用框架中使用 Qwen 的方法。

## 简介

### Qwen3-2507

在过去三个月里，我们持续探索 Qwen3 系列的潜力，并很高兴推出更新版本 **Qwen3-2507**。该版本包含 Qwen3-Instruct-2507 和 Qwen3-Thinking-2507 两个变体，并提供 235B-A22B、30B-A3B、4B 三种规模。

**Qwen3-Instruct-2507** 是此前 Qwen3 非思考模式的更新版本，带来如下关键提升：  

- **通用能力显著提升**，覆盖 **指令跟随、逻辑推理、文本理解、数学、科学、代码与工具使用**。  
- **多语言长尾知识覆盖显著增强**。  
- **在主观与开放式任务上更好对齐用户偏好**，生成更有帮助且更高质量的回复。  
- **256K 长上下文理解能力增强**，可扩展至 **100 万 tokens**。

**Qwen3-Thinking-2507** 是 Qwen3 思考模型的延续，在推理质量与深度上进一步提升，关键改进包括：
- **推理任务表现显著提升**，覆盖逻辑推理、数学、科学、代码与学术基准（通常需要人类专家参与），达到 **开源权重思考模型的 SOTA 水平**。
- **通用能力显著增强**，包括指令跟随、工具使用、文本生成与人类偏好对齐。
- **256K 长上下文理解能力增强**，可扩展至 **100 万 tokens**。


<details>
    <summary><b>此前的 Qwen3 版本</b></summary>
    <h3>Qwen3（亦称 Qwen3-2504）</h3>
    <p>
    我们很高兴发布 Qwen3，这是 Qwen 大语言模型家族的最新成员。
    这些模型基于我们在 QwQ 与 Qwen2.5 方面的积累，是迄今最先进、最智能的系统。
    我们已将 Qwen3 权重开放，包括稠密模型与混合专家（MoE）模型。
    <br><br>
    Qwen3 的亮点包括：
        <ul>
            <li><b>多种规模的稠密与 MoE 模型</b>，提供 0.6B、1.7B、4B、8B、14B、32B、30B-A3B 与 235B-A22B。</li>
            <li><b>思考模式</b>（用于复杂逻辑推理、数学与编码）与<b>非思考模式</b>（高效通用对话）之间可无缝切换，确保不同场景下的最佳表现。</li>
            <li><b>推理能力显著增强</b>，在数学、代码生成与常识逻辑推理等方面超过此前 QwQ（思考模式）与 Qwen2.5 Instruct（非思考模式）。</li>
            <li><b>更好的人类偏好对齐</b>，在创意写作、角色扮演、多轮对话与指令遵循方面表现出色，带来更自然、更有沉浸感的对话体验。</li>
            <li><b>具备强大 Agent 能力</b>，可在思考与非思考模式下精准调用外部工具，在复杂 Agent 任务上达到开源模型领先水平。</li>
            <li><b>支持 100+ 语言与方言</b>，具备出色的<b>多语言指令跟随</b>与<b>翻译</b>能力。</li>
        </ul>
    </p>
</details>


## 新闻
- 2025.08.08：Qwen3-2507 已支持 **100 万 tokens** 的超长输入！请查看更新的模型卡（[235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)、[235B-A22B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507)、[A30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)、[A30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)）了解如何启用该特性。
- 2025.08.06：Qwen3-2507 最终开放版本 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 与 [Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) 发布！
- 2025.07.31：Qwen3-30B-A3B-Thinking-2507 发布。详情见 [modelcard](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)。
- 2025.07.30：Qwen3-30B-A3B-Instruct-2507 发布。详情见 [modelcard](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)。
- 2025.07.25：Qwen3-235B-A22B 思考模式更新版本 Qwen3-235B-A22B-Thinking-2507 发布。详情见 [modelcard](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507)。
- 2025.07.21：Qwen3-235B-A22B 非思考模式更新版本 Qwen3-235B-A22B-Instruct-2507 发布，带来显著增强并支持 256K 长上下文理解。详情见 [modelcard](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)。
- 2025.04.29：Qwen3 系列发布。详情见 [blog](https://qwenlm.github.io/blog/qwen3)。
- 2024.09.19：Qwen2.5 系列发布，并新增 3B、14B、32B 三种规模。详情见 [blog](https://qwenlm.github.io/blog/qwen2.5)。
- 2024.06.06：Qwen2 系列发布。详情见 [blog](https://qwenlm.github.io/blog/qwen2/)！
- 2024.03.28：发布首个 Qwen MoE 模型：Qwen1.5-MoE-A2.7B！目前仅 HF transformers 与 vLLM 支持该模型，后续将支持 llama.cpp、mlx-lm 等。详情见 [blog](https://qwenlm.github.io/blog/qwen-moe/)。
- 2024.02.05：Qwen1.5 系列发布。

## 性能

详细评测结果见 [📑 blog (Qwen3-2504)](https://qwenlm.github.io/blog/qwen3/) 与 [📑 blog (Qwen3-2507) \[即将发布\]]()。

GPU 显存需求与吞吐量结果可参考[此处](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)。

## 运行 Qwen3

### 🤗 Transformers

Transformers 是用于推理与训练的预训练 NLP 库。
推荐使用最新版，并要求 `transformers>=4.51.0`。

#### Qwen3-Instruct-2507

下面的代码片段演示如何使用 Qwen3-30B-A3B-Instruct-2507 根据输入生成内容。
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

> [!Note]
> Qwen3-Instruct-2507 仅支持非思考模式，输出中不会包含 ``<think></think>`` 块。同时不再需要显式设置 `enable_thinking=False`。


#### Qwen3-Thinking-2507

下面的代码片段演示如何使用 Qwen3-30B-A3B-Thinking-2507 根据输入生成内容。
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-30B-A3B-Thinking-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)  # no opening <think> tag
print("content:", content)

```

> [!Note]
> Qwen3-Thinking-2507 仅支持思考模式。
> 另外，为强制模型思考，默认聊天模板会自动加入 `<think>`。因此模型输出中可能只出现 `</think>` 而没有显式的 `<think>` 起始标签，这是正常现象。
> 
> Qwen3-Thinking-2507 还支持更长的思考长度。我们强烈建议在复杂推理任务中为其设置足够的最大生成长度。



<details>
    <summary><b>此前 Qwen3 模型的思考/非思考模式切换</b></summary>
    <p>
    默认情况下，Qwen3 模型会在回复前进行思考。
    可通过以下方式控制：
        <ul>
            <li><code>enable_thinking=False</code>：在 `tokenizer.apply_chat_template` 中传入 <code>enable_thinking=False</code> 可严格禁止模型生成思考内容。</li>
            <li><code>/think</code> 与 <code>/no_think</code> 指令：在 system 或 user 消息中使用这些词来指示 Qwen3 是否思考。在多轮对话中，遵循最新指令。</li>
        </ul>
    </p>
</details>


### ModelScope

我们强烈建议（尤其是中国大陆用户）使用 ModelScope。
ModelScope 提供与 Transformers 类似的 Python API。
`modelscope download` CLI 工具可帮助解决权重下载问题。
对于 vLLM 与 SGLang，可分别设置环境变量 `VLLM_USE_MODELSCOPE=true` 与 `SGLANG_USE_MODELSCOPE=true`。


### llama.cpp

[`llama.cpp`](https://github.com/ggml-org/llama.cpp) 可在广泛硬件上以极少配置实现高性能 LLM 推理。
建议使用 `llama.cpp>=b5401` 以完整支持 Qwen3。

在终端中使用 CLI：
```shell
./llama-cli -hf Qwen/Qwen3-8B-GGUF:Q8_0 --jinja --color -ngl 99 -fa -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift
# CTRL+C to exit
```

在终端中启动 API server：
```shell
./llama-server -hf Qwen/Qwen3-8B-GGUF:Q8_0 --jinja --reasoning-format deepseek -ngl 99 -fa -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift --port 8080
```
简单的 Web 前端位于 `http://localhost:8080`，OpenAI 兼容 API 位于 `http://localhost:8080/v1`。

更多指南请参考[文档](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)。

> [!Note]
> llama.cpp 采用“旋转上下文管理”，通过逐出早期 token 来实现无限生成。
> 可通过参数配置，上述命令实际上禁用了这一机制。
> 详情请参考[文档](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html#llama-cli)。

### Ollama

在[安装 Ollama](https://ollama.com/) 后，可使用以下命令启动服务（推荐 Ollama v0.9.0 或更高版本）：
```shell
ollama serve
# You need to keep this service running whenever you are using ollama
```

使用 `ollama run` 拉取并运行模型。可通过 `qwen3` 的后缀指定模型规模，例如 `:8b` 或 `:30b-a3b`：
```shell
ollama run qwen3:8b
# Setting parameters, type "/set parameter num_ctx 40960" and "/set parameter num_predict 32768"
# To exit, type "/bye" and press ENTER
# For Qwen3-2504 models,
# - To enable thinking, which is the default, type "/set think"
# - To disable thinking, type "/set nothink"
```

也可通过 OpenAI 兼容 API 使用 Ollama。
请注意：(1) 使用 API 时需保持 `ollama serve` 运行；(2) 在调用 API 前需执行 `ollama run qwen3:8b`，以确保模型权重就绪。
默认 API 地址为 `http://localhost:11434/v1/`。

更多信息请访问 [ollama.ai](https://ollama.com/)。

> [!Note]
> Ollama 的命名可能与 Qwen 的原始命名不完全一致。
> 例如，截至 2025 年 8 月，Ollama 中的 `qwen3:30b-a3b` 指向 `qwen3:30b-a3b-thinking-2507-q4_K_M`。
> 使用前请查看 <https://ollama.com/library/qwen3/tags>。


> [!Note]
> Ollama 与 llama.cpp 一样采用“旋转上下文管理”。
> 但其默认设置（`num_ctx` 2048、`num_predict` -1）意味着在 2048 token 上下文内的无限生成，
> 可能导致 Qwen3 模型出现问题。
> 建议合理设置 `num_ctx` 与 `num_predict`。

### LMStudio

Qwen3 已被 [lmstudio.ai](https://lmstudio.ai/) 支持，可直接使用我们的 GGUF 文件。

### ExecuTorch

如需导出并在 ExecuTorch（iOS、Android、Mac、Linux 等）上运行，请参考此[示例](https://github.com/pytorch/executorch/blob/main/examples/models/qwen3/README.md)。

### MNN

如需导出并在移动端支持 Qwen3 的 MNN 上运行，请访问 [Alibaba MNN](https://github.com/alibaba/MNN)。

### MLX LM

在 Apple Silicon 上运行时，[`mlx-lm`](https://github.com/ml-explore/mlx-lm) 也支持 Qwen3（`mlx-lm>=0.24.0`）。
可在 Hugging Face Hub 中寻找以 MLX 结尾的模型。


### OpenVINO

在 Intel CPU 或 GPU 上运行时，[OpenVINO toolkit](https://github.com/openvinotoolkit) 支持 Qwen3。
可参考此[聊天机器人示例](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/llm-chatbot/llm-chatbot.ipynb)。


## 部署 Qwen3

Qwen3 支持多种推理框架。
这里演示 `SGLang`、`vLLM` 与 `TensorRT-LLM` 的用法。
你也可以在不同推理服务商处获取 Qwen3 模型，例如 [Alibaba Cloud Model Studio](https://www.alibabacloud.com/en/product/modelstudio)。


### SGLang

[SGLang](https://github.com/sgl-project/sglang) 是用于大语言模型与视觉语言模型的高速推理框架。
SGLang 可启动 OpenAI 兼容 API 服务。
需要 `sglang>=0.4.6.post1`。

对于 Qwen3-Instruct-2507：
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --port 30000 --context-length 262144
```

对于 Qwen3-Thinking-2507：
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B-Thinking-2507 --port 30000 --context-length 262144 --reasoning-parser deepseek-r1
```

对于 Qwen3：
```shell
python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --context-length 131072 --reasoning-parser qwen3
```
OpenAI 兼容 API 地址为 `http://localhost:30000/v1`。

> [!Note]
> 由于 SGLang 预处理 API 请求时会移除所有 `reasoning_content` 字段，导致 **Qwen3 思考模型在多步工具调用** 中的效果可能不理想（这类任务需要相关思考内容）。修复正在进行中。
> 作为临时方案，建议直接传入原始内容，不要抽取思考内容，聊天模板将正确处理。


### vLLM

[vLLM](https://github.com/vllm-project/vllm) 是高吞吐、内存高效的 LLM 推理与服务引擎。
推荐使用 `vllm>=0.9.0`。

对于 Qwen3-Instruct-2507：
```shell
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000 --max-model-len 262144
```

对于 Qwen3-Thinking-2507：
```shell
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --port 8000 --max-model-len 262144 --enable-reasoning --reasoning-parser deepseek_r1
```

对于 Qwen3：
```shell
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 131072 --enable-reasoning --reasoning-parser qwen3
```
OpenAI 兼容 API 地址为 `http://localhost:8000/v1`。

> [!Note]
> 由于 vLLM 预处理 API 请求时会移除所有 `reasoning_content` 字段，导致 **Qwen3 思考模型在多步工具调用** 中的效果可能不理想（这类任务需要相关思考内容）。修复正在进行中。
> 作为临时方案，建议直接传入原始内容，不要抽取思考内容，聊天模板将正确处理。

### TensorRT-LLM

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 是 NVIDIA 开源的 LLM 推理引擎，提供自定义 attention kernel、量化等优化。
Qwen3 已在其重构后的 [PyTorch backend](https://nvidia.github.io/TensorRT-LLM/torch.html) 中得到支持。
推荐 `tensorrt_llm>=0.20.0rc3`。
更多细节请参考 [README](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/models/core/qwen/README.md#qwen3)。

```shell
trtllm-serve Qwen/Qwen3-8B --host localhost --port 8000 --backend pytorch
```
OpenAI 兼容 API 地址为 `http://localhost:8000/v1`。

### MindIE

在 Ascend NPU 上部署请访问 [Modelers](https://modelers.cn/) 并搜索 Qwen3。

<!-- 
### OpenLLM

[OpenLLM](https://github.com/bentoml/OpenLLM) allows you to easily run Qwen2.5 as OpenAI-compatible APIs. You can start a model server using `openllm serve`. For example:

```bash
openllm serve qwen2.5:7b
```

The server is active at `http://localhost:3000/`, providing OpenAI-compatible APIs. You can create an OpenAI client to call its chat API. For more information, refer to [our documentation](https://qwen.readthedocs.io/en/latest/deployment/openllm.html). -->


## 基于 Qwen3 构建

### Tool Use

针对工具调用能力，建议查看 [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)，它提供了这些 API 的封装，并支持 MCP 的工具调用或函数调用。
Qwen3 的工具调用也可通过 SGLang、vLLM、Transformers、llama.cpp、Ollama 等实现。
具体启用方式请参考文档。


### Finetuning

建议使用训练框架（如 [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)、[UnSloth](https://github.com/unslothai/unsloth)、[Swift](https://github.com/modelscope/swift)、[Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) 等）进行 SFT、DPO、GRPO 等微调。


## 许可协议

我们的开源权重模型均采用 Apache 2.0 许可证。
许可证文件可在对应的 Hugging Face 仓库中找到。

## 引用

如果我们的工作对你有帮助，欢迎引用。

```bibtex
@article{qwen3,
    title={Qwen3 Technical Report}, 
    author={An Yang and Anfeng Li and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Gao and Chengen Huang and Chenxu Lv and Chujie Zheng and Dayiheng Liu and Fan Zhou and Fei Huang and Feng Hu and Hao Ge and Haoran Wei and Huan Lin and Jialong Tang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jing Zhou and Jingren Zhou and Junyang Lin and Kai Dang and Keqin Bao and Kexin Yang and Le Yu and Lianghao Deng and Mei Li and Mingfeng Xue and Mingze Li and Pei Zhang and Peng Wang and Qin Zhu and Rui Men and Ruize Gao and Shixuan Liu and Shuang Luo and Tianhao Li and Tianyi Tang and Wenbiao Yin and Xingzhang Ren and Xinyu Wang and Xinyu Zhang and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yinger Zhang and Yu Wan and Yuqiong Liu and Zekun Wang and Zeyu Cui and Zhenru Zhang and Zhipeng Zhou and Zihan Qiu},
    journal = {arXiv preprint arXiv:2505.09388},
    year={2025}
}

@article{qwen2.5,
    title   = {Qwen2.5 Technical Report}, 
    author  = {An Yang and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chengyuan Li and Dayiheng Liu and Fei Huang and Haoran Wei and Huan Lin and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jingren Zhou and Junyang Lin and Kai Dang and Keming Lu and Keqin Bao and Kexin Yang and Le Yu and Mei Li and Mingfeng Xue and Pei Zhang and Qin Zhu and Rui Men and Runji Lin and Tianhao Li and Tingyu Xia and Xingzhang Ren and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yu Wan and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zihan Qiu},
    journal = {arXiv preprint arXiv:2412.15115},
    year    = {2024}
}

@article{qwen2,
    title   = {Qwen2 Technical Report}, 
    author  = {An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan},
    journal = {arXiv preprint arXiv:2407.10671},
    year    = {2024}
}
```

## 联系我们
如需联系研究团队或产品团队，请加入我们的 [Discord](https://discord.gg/z3GAxXZ9Ce) 或 [微信群](assets/wechat.png)！
