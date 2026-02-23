# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official Qwen3 repository — a family of large language models (dense and MoE) by Alibaba/Qwen. The repo contains documentation, evaluation pipelines, custom model implementations for educational/tracing purposes, and deployment guides. It does NOT contain the actual model weights or the upstream HuggingFace Transformers integration code.

## Repository Structure

- `qwen3_8b_model.py` — Educational dense Qwen3 implementation (Transformer decoder with GQA, QK-Norm, SwiGLU, RoPE). Includes a configurable tracing system for debugging data flow.
- `qwen3_8b_moe_model.py` — Educational MoE variant adding sparse expert routing (`Qwen3MoeSparseMoeBlock`) with Top-K gating, load-balancing loss, and per-layer MoE frequency control.
- `test.py` — Standalone MoE block test with mock config and simplified experts.
- `docs/` — Sphinx documentation site. Sources in `docs/source/`, Chinese translations in `docs/locales/zh_CN/`.
- `eval/` — Evaluation pipeline: configs in `eval/configs/`, data in `eval/data/`, inference in `eval/generate_api_answers/`, scoring in `eval/eval/`.
- `examples/` — Legacy demo scripts (deprecated for Qwen3, carried over from Qwen2.5).
- `docker/` — Dockerfiles and helper scripts for demos.

## Build & Run Commands

### Build documentation
```bash
pip install -r docs/requirements-docs.txt
make -C docs html
# Output: docs/build/html
```

### Run evaluation (requires a running vLLM server)
```bash
pip install -r eval/requirements.txt
# Step 1: Generate answers
python eval/generate_api_answers/infer_multithread.py --config eval/configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml
# Step 2: Score
python eval/eval/eval.py --config eval/configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml
```

### Run model implementations directly
```bash
python qwen3_8b_model.py       # Dense model forward pass demo
python qwen3_8b_moe_model.py   # MoE model forward pass demo
python test.py                  # MoE block isolated test
```

## Architecture Notes

### Dense Model (`qwen3_8b_model.py`)
Data flow per decoder layer: `Input → RMSNorm → Attention(QKV proj → RoPE → QK-Norm → GQA expand → Scaled Dot-Product → O proj) + Residual → RMSNorm → MLP(SwiGLU: gate/up → SiLU*up → down) + Residual → Output`

Key design choices from the Qwen3 technical report:
- GQA (Grouped Query Attention) with `num_kv_heads < num_attention_heads`
- QK-Norm (RMSNorm on Q/K after RoPE) for training stability
- No bias in QKV/O projections
- RoPE base frequency = 1M (extended context support)
- Pre-Norm (RMSNorm before attention and MLP)

### MoE Model (`qwen3_8b_moe_model.py`)
Extends the dense model by replacing MLP with `Qwen3MoeSparseMoeBlock` at configurable layer intervals (`moe_layer_freq`). Each MoE block:
- Routes tokens via a learned gate (`nn.Linear → softmax → Top-K`)
- Dispatches to selected experts using `index_select` / `index_add_`
- Optionally normalizes Top-K probabilities (`norm_topk_prob`)
- Supports load-balancing auxiliary loss (`router_aux_loss_coef`)

### Tracing System
Both model files share a tracing utility (`_trace_*` functions) with three levels:
- `input_flow` — High-level input/output shapes per module
- `compact` — Intermediate shapes and routing histograms
- `verbose` — Full tensor metadata (shape, dtype, device)

Enable via config: `enable_trace=True, trace_level="compact"`

## Coding Conventions

- Python 4-space indentation, standard library imports first
- `snake_case` for modules and functions
- No repo-wide linter/formatter configured — match surrounding code style
- Eval configs named as `{DATASET}-{MODEL}.yaml` in `eval/configs/`
- Comments in model files are in Chinese, referencing sections of the Qwen3 technical report

## Key Dependencies

- PyTorch (`torch`) — core framework for model implementations
- `transformers>=4.51.0` — for loading official Qwen3 weights from HuggingFace
- `vllm>=0.9.0` or `sglang>=0.4.6.post1` — for serving/deployment
- Eval: `openai`, `numpy`, `tqdm`, `datasets==2.14.6`, `pyyaml`
- Docs: see `docs/requirements-docs.txt`

## License

Apache 2.0
