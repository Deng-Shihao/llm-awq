# llm-awq 模型量化依赖分析

## 一、量化流程概览

llm-awq 实现了 **AWQ (Activation-aware Weight Quantization)**，支持 INT3/4 低比特量化。完整量化流程分四步：

```
AWQ搜索 → 应用结果 → 伪/实量化 → 推理评估
```

---

## 二、核心文件依赖图

```
awq/entry.py                          ← 主入口
├── awq/quantize/pre_quant.py         ← AWQ 搜索（run_awq / apply_awq）
│   ├── awq/quantize/auto_scale.py   ← 自动缩放搜索
│   ├── awq/quantize/auto_clip.py    ← 自动裁剪搜索
│   └── awq/utils/calib_data.py      ← 校准数据集加载
├── awq/quantize/quantizer.py         ← 伪/实量化实现
│   └── awq/quantize/qmodule.py      ← WQLinear 量化层（调用 CUDA 核心）
├── awq/utils/module.py               ← 层操作工具
├── awq/utils/parallel.py             ← 模型并行设置
└── awq/utils/lm_eval_adaptor.py      ← 评估框架适配器
```

---

## 三、必须的源码文件

| 文件路径 | 作用 | 关键类/函数 |
|---------|------|-----------|
| `awq/entry.py` | 主程序入口，协调全流程 | `main()`, `build_model_and_enc()` |
| `awq/quantize/pre_quant.py` | AWQ 搜索核心 | `run_awq()`, `apply_awq()`, `get_blocks()`, `get_named_linears()` |
| `awq/quantize/quantizer.py` | 权重量化实现 | `pseudo_quantize_tensor()`, `pseudo_quantize_model_weight()`, `real_quantize_model_weight()` |
| `awq/quantize/qmodule.py` | 量化线性层定义 | `WQLinear`, `ScaledActivation` |
| `awq/quantize/auto_scale.py` | 激活-权重缩放搜索 | `auto_scale_block()`, `apply_scale()`, `scale_ln_fcs()` |
| `awq/quantize/auto_clip.py` | 权重裁剪范围搜索 | `auto_clip_block()`, `auto_clip_layer()`, `apply_clip()` |
| `awq/quantize/smooth.py` | SmoothQuant 激活缩放（VILA 模型用） | `get_smooth_scale()`, `smooth_lm()` |
| `awq/utils/calib_data.py` | 加载校准数据集 | `get_calib_dataset()` |
| `awq/utils/module.py` | 模型层操作工具 | `get_op_by_name()`, `set_op_by_name()`, `get_op_name()` |
| `awq/utils/parallel.py` | 多卡并行配置 | `auto_parallel()` |
| `awq/utils/lm_eval_adaptor.py` | lm-eval 接口适配 | `LMEvalAdaptor` |
| `awq/utils/utils.py` | 模型分发工具 | `simple_dispatch_model()` |

---

## 四、必须的第三方 Python 包

### 最小运行依赖

```bash
pip install \
  torch==2.3.0 \
  transformers==4.46.0 \
  accelerate==0.34.2 \
  datasets \
  sentencepiece \
  tokenizers>=0.12.1 \
  lm_eval==0.3.0 \
  toml \
  protobuf \
  pydantic==1.10.19
```

| 包名 | 版本 | 用途 |
|-----|------|------|
| `torch` | 2.3.0 | 深度学习框架，张量计算 |
| `transformers` | 4.46.0 | 模型加载（`AutoModelForCausalLM`, `AutoTokenizer`） |
| `accelerate` | 0.34.2 | 多卡分发（`init_empty_weights`, `dispatch_model`） |
| `datasets` | 最新 | PILEval 校准数据集加载 |
| `sentencepiece` | 最新 | LLaMA 等模型的分词器 |
| `tokenizers` | ≥0.12.1 | 快速分词库 |
| `lm_eval` | 0.3.0 | 语言模型评估（WikiText PPL 等） |
| `toml` | 最新 | 配置文件解析 |
| `protobuf` | 最新 | tokenizer 序列化 |
| `pydantic` | 1.10.19 | 数据验证 |

### 完整安装（含 CUDA 推理核心）

```bash
pip install -e .   # 自动编译 awq_inference_engine CUDA 扩展
```

---

## 五、必须的 CUDA 编译产物

`awq_inference_engine` 是量化推理的核心，**实量化（real backend）必须编译**，伪量化（fake backend）可跳过。

### CUDA 源码文件（编译后生成 `awq_inference_engine` 模块）

| 源文件 | 作用 |
|-------|------|
| `awq/kernels/csrc/pybind.cpp` | Python 绑定入口 |
| `awq/kernels/csrc/quantization/gemm_cuda_gen.cu` | W4A16 GEMM 内核 |
| `awq/kernels/csrc/quantization/gemv_cuda.cu` | W4A16 GEMV 内核（生成阶段） |
| `awq/kernels/csrc/quantization_new/gemm/gemm_cuda.cu` | 新版 GEMM |
| `awq/kernels/csrc/quantization_new/gemv/gemv_cuda.cu` | 新版 GEMV |
| `awq/kernels/csrc/w8a8/w8a8_gemm_cuda.cu` | W8A8 GEMM |
| `awq/kernels/csrc/layernorm/layernorm.cu` | 融合层标准化 |
| `awq/kernels/csrc/rope_new/fused_rope_with_pos.cu` | 融合 RoPE 位置编码 |

### 调用时机

```python
# qmodule.py 中
import awq_inference_engine

# 小批量（token < 8）：生成阶段
awq_inference_engine.gemv_forward_cuda_new(...)

# 大批量：prefill 阶段
awq_inference_engine.gemm_forward_cuda_new(...)
```

### 编译环境要求

- CUDA Toolkit 11.8+
- NVCC 编译器
- g++ / clang++ (支持 C++17)

---

## 六、量化配置参数

```python
# 必须指定的参数
w_bit        = 4          # 量化位数（支持 3, 4）
q_group_size = 128        # 分组大小（128 / 64 / 32 / -1 全层）
zero_point   = True       # 是否使用零点量化

# q_config 字典传入量化函数
q_config = {
    "zero_point": True,
    "q_group_size": 128,
}
```

---

## 七、完整调用链（实量化示例）

```
entry.main()
│
├─ build_model_and_enc()
│  ├─ AutoModelForCausalLM.from_pretrained()   # 加载原始模型
│  └─ AutoTokenizer.from_pretrained()           # 加载分词器
│
├─ apply_awq(model, awq_results)               # 加载预计算的 AWQ 结果
│  ├─ apply_scale(model, awq_results["scale"]) # 吸收缩放到权重/LayerNorm
│  └─ apply_clip(model, awq_results["clip"])   # 裁剪权重范围
│
├─ real_quantize_model_weight(model, w_bit, q_config)
│  └─ for each Linear layer:
│     ├─ pseudo_quantize_tensor(w, get_scale_zp=True)  # 计算 scale/zp
│     └─ WQLinear.from_linear(linear, scales, zeros)   # 打包为 INT4
│        └─ pack_intweight()                            # 位打包
│
└─ (推理) WQLinear.forward(x)
   └─ awq_inference_engine.gemm_forward_cuda_new()     # CUDA 加速
```

---

## 八、AWQ 搜索调用链（可选，通常用预计算缓存）

```
run_awq(model, enc, w_bit=4, q_config, n_samples=128, seqlen=512)
│
├─ get_calib_dataset("pileval", enc, n_samples, seqlen)
│
└─ for each transformer block:
   ├─ auto_scale_block(module, linears, x)
   │  ├─ pseudo_quantize_tensor()         # 模拟量化误差
   │  └─ scale_ln_fcs() / scale_fc_fc()  # 更新缩放因子
   │
   └─ auto_clip_block(module, w_bit, q_config, input_feat)
      └─ auto_clip_layer()               # 格点搜索最优裁剪值
```

---

## 九、支持的模型架构

| 架构类型 | 代表模型 |
|---------|---------|
| LlamaForCausalLM | LLaMA-1/2/3, CodeLlama |
| Qwen2ForCausalLM | Qwen2 系列 |
| OPTForCausalLM | Meta OPT |
| BloomForCausalLM | BLOOM |
| MPT | MosaicML MPT |
| Falcon | Falcon |
| GPT-NeoX | EleutherAI GPT-NeoX |
| InternVL3 | InternVL3 |
| LlavaLlamaForCausalLM | LLaVA（多模态） |

---

## 十、快速参考：最小文件集

仅需量化（不含推理评估），必须的文件：

```
awq/
├── entry.py
├── quantize/
│   ├── __init__.py
│   ├── pre_quant.py
│   ├── quantizer.py
│   ├── qmodule.py        # 实量化还需编译 awq_inference_engine
│   ├── auto_scale.py
│   └── auto_clip.py
└── utils/
    ├── calib_data.py
    └── module.py
```
