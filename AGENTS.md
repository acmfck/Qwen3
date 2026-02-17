# Repository Guidelines(acmfck)

## Project Structure & Module Organization
- `README.md` and `Qwen3_Technical_Report.pdf` provide high-level model and release context.
- `docs/` is the Sphinx documentation site (sources in `docs/source/`, translations in `docs/locales/`).
- `eval/` contains evaluation configs, data, and scripts (`eval/configs/`, `eval/data/`, `eval/generate_api_answers/`, `eval/eval/`).
- `examples/` hosts demos and benchmarks; note `examples/README.md` marks these as deprecated for Qwen3.
- `docker/` includes demo Dockerfiles and helper scripts.

## Build, Test, and Development Commands
- Build docs: `pip install -r docs/requirements-docs.txt`, then `make -C docs html` (outputs to `docs/build/html`).
- Evaluate a model: `pip install -r eval/requirements.txt`, start a vLLM server, then run:
  - `python eval/generate_api_answers/infer_multithread.py --config eval/configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml`
  - `python eval/eval/eval.py --config eval/configs/ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml > eval/eval_res/ARCAGI-Qwen3-235B-A22B-Instruct-2507_eval_result.txt`
- Demo scripts (legacy Qwen2.5): `python examples/demo/cli_demo.py` or `python examples/demo/web_demo.py --server-port 8000`.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard library-first import grouping; follow existing file patterns.
- Use `snake_case` for Python modules (e.g., `infer_multithread.py`).
- Keep evaluation configs in `eval/configs/` named with the dataset/model (e.g., `ARCAGI-Qwen3-235B-A22B-Instruct-2507.yaml`).
- No repo-wide formatter or linter is configured; keep changes minimal and consistent with surrounding code.

## Testing Guidelines
- There is no unit test suite. Validation is done via evaluation scripts in `eval/` and by building docs.
- When changing eval logic or configs, rerun the relevant `eval/` command and update `eval/output/` or `eval_res/` only if explicitly intended.

## Commit & Pull Request Guidelines
- Commit messages in history are short and imperative (e.g., `Update README.md`, `add eval`) and often include a PR number like `(#1600)`.
- PRs should include a clear summary, affected areas (`docs/`, `eval/`, `examples/`), and any commands run (e.g., doc build or evaluation).
- If a change updates published documentation, note whether translations in `docs/locales/` were updated or left for follow-up.

## Configuration & Resource Notes
- Evaluation and demos assume external model weights and may require large GPU resources.
- For eval, keep `MODEL_NAME`, `MODEL_PATH`, and GPU parallelism settings aligned with the config file you run.
