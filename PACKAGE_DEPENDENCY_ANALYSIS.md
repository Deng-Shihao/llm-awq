# llm-awq Package Version Dependency Analysis

## Scope
- Repository: `llm-awq`
- Analysis targets:
  - Declared Python dependencies (`pyproject.toml`)
  - CUDA extension build dependencies (`awq/kernels/setup.py`, `awq/kernels/csrc/attention/setup.py`)
  - Runtime optional dependencies from README and code paths (TinyChat, VILA/NVILA, FlashAttention)
  - Current local environment versions (from `.venv`)

## Sources
- `pyproject.toml`
- `awq/kernels/setup.py`
- `awq/kernels/csrc/attention/setup.py`
- `README.md`
- `tinychat/README.md`
- Runtime import/use points in `awq/` and `tinychat/`

## 1. Declared Python dependencies (authoritative)
From `pyproject.toml`:

| Package | Constraint in repo |
|---|---|
| accelerate | `==0.34.2` |
| sentencepiece | unpinned |
| tokenizers | `>=0.12.1` |
| torch | `==2.3.0` |
| torchvision | `==0.18.0` |
| transformers | `==4.46.0` |
| lm_eval | `==0.3.0` |
| texttable | unpinned |
| toml | unpinned |
| attributedict | unpinned |
| protobuf | unpinned |
| gradio | `==3.35.2` |
| gradio_client | `==0.2.9` |
| fastapi | unpinned |
| uvicorn | unpinned |
| pydantic | `==1.10.19` |

Build-system:
- `setuptools>=61.0`

## 2. Build-time dependencies (CUDA extensions)

### awq kernel extension (`awq_inference_engine`)
- Defined in `awq/kernels/setup.py`
- Requires:
  - `torch` with C++ extension toolchain
  - CUDA toolkit + `nvcc`
  - C++17 compiler (`-std=c++17`)

### ft attention extension (`ft_attention`)
- Defined in `awq/kernels/csrc/attention/setup.py`
- Hard checks include:
  - CUDA toolkit available (`CUDA_HOME`, `nvcc`)
  - CUDA version compatibility with `torch.version.cuda`
  - CUDA >= 11.0 for `ft_attention`

## 3. Runtime optional dependencies and feature mapping

| Feature path | Dependency notes |
|---|---|
| AWQ quantization (`python -m awq.entry`) | Depends on core stack above and CUDA kernels for real INT4 runtime performance |
| TinyChat LLM/VLM inference | Uses `torch`, `transformers`, `tokenizers`, `sentencepiece`, and optional FlashAttention |
| FlashAttention acceleration (`--flash_attn`) | Optional; wheel must match Python, Torch, CUDA, and `cxx11abi` |
| VILA/NVILA related workflows | Optional external project install (`VILA`) per README |
| Gradio serving (`tinychat/serve`) | `gradio`, `fastapi`, `uvicorn`, plus model/runtime stack |

## 4. Version coupling and compatibility constraints

### 4.1 Torch stack coupling
- Repo pins:
  - `torch==2.3.0`
  - `torchvision==0.18.0`
- This pair should be kept aligned.

### 4.2 Transformers and Hugging Face Hub coupling
- Repo pins `transformers==4.46.0`
- This transformers line requires `huggingface-hub<1.0` at runtime.
- If `huggingface-hub>=1.0` is installed, import can fail.

Recommended safe constraint:
- `huggingface-hub>=0.23.2,<1.0`

### 4.3 FlashAttention wheel coupling
FlashAttention wheel must match all of:
- Python ABI (`cp310` for Python 3.10)
- Torch major/minor (`torch2.3`)
- CUDA line (`cu12` vs local Torch CUDA build)
- ABI flavor (`cxx11abiTRUE` vs `cxx11abiFALSE`)

For this environment (see section 5), `cxx11abiFALSE` is required.

## 5. Current local environment (validated)
Observed in `.venv`:

| Package | Installed version |
|---|---|
| torch | `2.3.0` |
| torchvision | `0.18.0` |
| transformers | `4.46.0` |
| accelerate | `0.34.2` |
| huggingface-hub | `0.36.2` |
| tokenizers | `0.20.3` |
| sentencepiece | `0.2.1` |
| gradio | `3.35.2` |
| gradio-client | `0.2.9` |
| fastapi | `0.125.0` |
| uvicorn | `0.41.0` |
| pydantic | `1.10.19` |
| lm-eval | `0.3.0` |
| flash-attn | `2.7.4.post1` |
| einops | `0.8.2` |
| triton | `2.3.0` |

Additional environment fact:
- `torch.compiled_with_cxx11_abi() == False`

## 6. Known issues and risk points

1. Hub version drift:
- Installing `huggingface-hub` 1.x can break `transformers==4.46.0`.

2. FlashAttention wheel mismatch:
- Wrong `cxx11abi` wheel can import-fail with undefined symbols.
- Wrong Torch/CUDA/Python wheel tags can also fail at import or runtime.

3. Unpinned packages in `pyproject.toml`:
- `sentencepiece`, `fastapi`, `uvicorn`, `protobuf`, `toml`, etc. are unpinned.
- This increases long-term reproducibility risk.

4. Documentation drift:
- Some docs mention `--flash`; current TinyChat code path uses `--flash_attn`.

## 7. Recommended pinning policy for reproducibility

If strict reproducibility is needed, add/keep:
- `torch==2.3.0`
- `torchvision==0.18.0`
- `transformers==4.46.0`
- `accelerate==0.34.2`
- `huggingface-hub>=0.23.2,<1.0`
- `pydantic==1.10.19`
- Exact FlashAttention wheel matching local ABI/toolchain

Optionally pin currently unbounded dependencies (`fastapi`, `uvicorn`, `sentencepiece`, `protobuf`, `toml`, `texttable`, `attributedict`) to reduce future breakage.

