This folder provides scripts to reproduce evaluation results across various benchmarks for the **Qwen** series of large language models.

## Supported Benchmarks

Currently, we support the following benchmark:

| Model | Dataset | Config | Reproduced Score |
|-------|--------|--------|------------------|
| Qwen3-235B-A22B-Instruct-2507 | ARC-AGI 1 (pass@1) | [./configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml](./configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml) | 40.75 |

In the meantime, you can find the model outputs and final evaluation results in the [`./output`](./output) and [`./eval_res`](./eval_res) directories, respectively.

Additional benchmarks will be added in future updates. 


## Evaluation Guide

Follow the steps below to reproduce the reported scores.

### Step 0: Prerequisites

Ensure you have:
- Python ≥ 3.9
- Either [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sgl) installed

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Step 1: Start vLLM Server

Launch the vLLM inference server using the command below:

```bash
export MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"  # Replace with desired model
export MODEL_PATH="$MODEL_NAME"  # Or path to local checkpoint
export NUM_GPUS=8

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $NUM_GPUS \
    --enforce-eager \
    --port 8030
```

> 💡 Adjust `tensor_parallel_size` according to your GPU setup.

### Optional: Start SGLang Router (Recommended for Faster Evaluation)

Since evaluations can take several days, we recommend using **SGLang** with data parallelism to accelerate inference. See the [SGLang Router documentation](https://docs.sglang.ai/router/router.html) for details.

Start the SGLang router server:

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

> ⚠️ Adjust `dp_size` based on available resources, and ensure consistency in port configuration for subsequent steps.


### Step 2: Run Inference

Once the inference server is running, generate model responses using the multithreaded inference script.

```bash
mkdir -p output

# Example: Evaluate on ARC-AGI
python generate_api_answers/infer_multithread.py \
    --config configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml
```

#### Resume Interrupted Inference

If the process is interrupted, simply re-run the same command. The script will automatically detect existing outputs and resume generation for incomplete prompts.

### Step 3: Compute Scores

After inference completes, evaluate the results using the scoring script:

```bash
mkdir -p eval_res

python eval/eval.py \
    --config configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml \
    > eval_res/ARCAGI-Qwen3-235B-A22B-Instruct-2507_eval_result.txt
```

The final score will be saved to the specified output file.

---

## 中文翻译

本文件夹提供脚本，用于复现 **Qwen** 系列大语言模型在各类评测基准上的结果。

## 支持的评测

目前支持以下评测：

| 模型 | 数据集 | 配置 | 复现分数 |
|------|--------|------|----------|
| Qwen3-235B-A22B-Instruct-2507 | ARC-AGI 1 (pass@1) | [./configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml](./configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml) | 40.75 |

同时，你可以在 [`./output`](./output) 与 [`./eval_res`](./eval_res) 目录中分别找到模型输出和最终评测结果。

后续更新将会加入更多评测。


## 评测指南

按照以下步骤复现报告中的分数。

### 步骤 0：前置条件

请确保具备：
- Python ≥ 3.9
- 已安装 [vLLM](https://github.com/vllm-project/vllm) 或 [SGLang](https://github.com/sgl-project/sgl) 之一

安装依赖：

```bash
pip install -r requirements.txt
```

### 步骤 1：启动 vLLM 服务

使用以下命令启动 vLLM 推理服务：

```bash
export MODEL_NAME="Qwen/Qwen3-235B-A22B-Instruct-2507"  # 可替换为目标模型
export MODEL_PATH="$MODEL_NAME"  # 或本地权重路径
export NUM_GPUS=8

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size $NUM_GPUS \
    --enforce-eager \
    --port 8030
```

> 💡 根据 GPU 资源调整 `tensor_parallel_size`。

### 可选：启动 SGLang Router（推荐加速评测）

由于评测可能持续数天，建议使用 **SGLang** 的数据并行来加速推理。详情见 [SGLang Router 文档](https://docs.sglang.ai/router/router.html)。

启动 SGLang 路由服务：

```bash
python -m sglang_router.launch_server \
    --model-path Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --dp-size 4 \
    --host 0.0.0.0 \
    --port 30000
```

> ⚠️ 请根据资源调整 `dp_size`，并保证后续步骤的端口配置一致。


### 步骤 2：运行推理

推理服务启动后，使用多线程推理脚本生成模型输出。

```bash
mkdir -p output

# 示例：在 ARC-AGI 上评测
python generate_api_answers/infer_multithread.py \
    --config configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml
```

#### 断点续跑

如果中途中断，直接重复执行同一命令即可。脚本会自动检测已有输出，并继续生成未完成的部分。

### 步骤 3：计算分数

推理完成后，使用评分脚本计算结果：

```bash
mkdir -p eval_res

python eval/eval.py \
    --config configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml \
    > eval_res/ARCAGI-Qwen3-235B-A22B-Instruct-2507_eval_result.txt
```

最终分数将保存到指定的输出文件中。
