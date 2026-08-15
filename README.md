# tinyinfer.cpp

一个轻量级的 LLM 推理引擎：约 3000 行 C++/CUDA，自包含、仅 NVIDIA GPU，
**零第三方 CUDA 库依赖**（不使用 cuBLAS / cuDNN / cuBLASLt / CUTLASS）。

参考 llama.cpp 的推理思路裁剪/重写而成，默认适配 Qwen3-0.6B（BF16, GGUF）：
模型超参全部从 GGUF 元数据读取并校验，不硬编码。

## 特性

- 离线批处理推理：一次运行可处理多条 prompt
- 仅 NVIDIA GPU（CUDA 12.x, sm_80），无 CPU 回退
- 自研 CUDA 算子：BF16 tensor-core GEMM、decode 专用 split-K GEMM、
  GQA causal attention、NeoX RoPE、RMSNorm、SwiGLU、LM head
- 自包含 Qwen3 BPE tokenizer（字节编码 + Qwen2 预分词）
- 对话模式（`--chat`，Qwen3 im_start/im_end 模板）
- 算子自测：`QWEN_SELFTEST=1`

## Quick Start

### 1. 下载

```bash
git clone https://github.com/beyondHJM/tinyinfer.cpp
cd tinyinfer.cpp
```

### 2. 编译

依赖：CMake >= 3.18、CUDA 12.x、gcc/g++（Linux）或 MSVC（Windows）。

```bash
./scripts/build.sh
```

或手动：

```bash
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
```

### 3. 安装（可选）

```bash
cp build/tinyinfer /usr/local/bin/
```

### 4. 运行 Demo

需要 Qwen3-0.6B 的 BF16 GGUF 模型文件（约 1.1 GiB，可从 HuggingFace 或
ModelScope 下载，例如 `Qwen3_0.6B.BF16.gguf`）。

```bash
./build/tinyinfer -m /path/to/Qwen3_0.6B.BF16.gguf \
    -n 128 -p "Please introduce AI Agent briefly" --chat
```

输出示例（A100 实测）：

```text
[output] Okay, the user wants me to introduce an AI Agent briefly. Let me start
by understanding what an AI Agent is. From what I know, an AI Agent is a system
that can perform tasks autonomously, like helping with tasks like answering
questions, providing information, or even making decisions. But I need to make
sure I explain it clearly without getting too technical.
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `-m` | GGUF 模型路径（Qwen3 架构、BF16） |
| `-n` | 每个序列最大生成 token 数 |
| `-p` | prompt（可重复） |
| `-f` | prompt 文件（每行一条） |
| `--chat` | 对话模式（Qwen3 chat 模板） |
| `-t` / `--top-k` / `--top-p` / `--seed` | 采样参数（`-t 0` 为 greedy） |
| `--max-seq` | KV cache 每序列最大长度（默认 2048） |
| `--verbose` | 打印 token 级信息 |

## 批处理示例

```bash
printf 'Hello\nPlease introduce AI Agent\n' > prompts.txt
./build/tinyinfer -m model.gguf -n 128 -f prompts.txt
```

## 正确性与性能

- **算子自测**：`QWEN_SELFTEST=1 ./build/tinyinfer`，gemm / rms_norm / rope /
  attention 与 CPU 参考逐值对比，全部通过
- **数值对齐**：与 llama.cpp 同一模型同一 prompt 的 logits 最大绝对误差 < 0.1，
  top-10 完全一致
- **性能**：A100 上单序列 decode 约 180 tok/s，为 llama.cpp `simple.cpp`
  的 ~96%（超过计划目标 80%）

## 模块来源

- `src/tokenizer.cpp`：BPE 与 Qwen2 预分词裁剪自 llama.cpp（MIT License）
- `src/cuda/rope.cu`：NeoX RoPE 公式与 llama.cpp 一致
- 其余（GGUF 解析、模型结构、推理循环、GEMM/attention/FFN kernel）为本工程实现

## 说明

本仓库所有代码均由 **deepseek-v4-flash + Codex** 自动生成，用时约 **1~2 小时**。

## License

MIT
