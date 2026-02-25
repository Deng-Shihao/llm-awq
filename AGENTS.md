# Repository Guidelines

## Project Structure & Module Organization
- `awq/`: core quantization package (search, quantizer, kernels, utilities) and CLI entrypoint in `awq/entry.py`.
- `awq/kernels/`: CUDA/C++ extensions and kernel build scripts.
- `tinychat/`: inference/demo stack for LLM/VLM chat, including serving code in `tinychat/serve/`.
- `scripts/`: model-family example workflows (`llama3_example.sh`, `qwen_example.sh`, etc.).
- `examples/`: notebooks and conversion utilities (e.g., `convert_to_hf.py`).
- `figures/`: docs/demo assets only; no runtime logic.

## Build, Test, and Development Commands
- Create env (preferred): `uv venv --python 3.10 .venv && source .venv/bin/activate`
- Install package: `uv pip install -e .`
- Build kernels: `cd awq/kernels && python setup.py install`
- Optional FlashAttention: `uv pip install flash-attn --no-build-isolation`
- Run AWQ flow (example): `python -m awq.entry --model_path <hf-model> --w_bit 4 --q_group_size 128 --run_awq --dump_awq awq_cache/model-w4-g128.pt`
- Run TinyChat demo: `python tinychat/demo.py --model_type llama --model_path <model> --load_quant <quant-ckpt>`
- Launch VLM server components: `python -m tinychat.serve.controller`, `python -m tinychat.serve.gradio_web_server`, and `python -m tinychat.serve.model_worker_new ...`

## Coding Style & Naming Conventions
- Python style follows existing code: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep CLI args descriptive and long-form (see `awq/entry.py` pattern).
- Prefer small, focused modules under `awq/utils/`, `awq/quantize/`, and `tinychat/utils/`.
- No repo-enforced formatter/linter config is currently committed; keep changes PEP 8-aligned and consistent with neighboring files.

## Testing Guidelines
- There is no dedicated `tests/` suite in this repository.
- Validate changes with targeted smoke runs:
  - AWQ CLI path (`python -m awq.entry ...` with a small model/checkpoint).
  - TinyChat inference path (`python tinychat/demo.py ...`).
  - If serving changes, boot controller/web/model-worker locally.

## Commit & Pull Request Guidelines
- Follow existing history style: concise imperative subject, optionally prefixed with `[Minor]` or `[Major]`, often with PR number (e.g., `(#294)`).
- Keep commits scoped (one logical change per commit when possible).
- PRs should include:
  - What changed and why.
  - Repro steps/commands used.
  - Model/config used for validation.
  - Screenshots/log snippets for demo or serving UI changes.
