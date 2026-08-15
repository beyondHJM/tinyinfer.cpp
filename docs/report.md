# qwen3-gpu-infer 验证报告

按 `/root/hujianmin/qwen3-gpu-demo-plan.md` 的里程碑 M1-M6 执行与验收。

## 环境

- 服务器：Ubuntu，104 核，251 GB 内存，4 × NVIDIA A100-PCIE-40GB（sm_80）
- 工具链：CUDA 12.4、gcc 11.4、cmake 3.28
- 模型：`Qwen3_0.6B.BF16.gguf`（1.12 GiB，28 层、head_dim=128、词表 151936，
  BF16；位于 `/root/llm-resource/Models/`）
- 参考：llama.cpp b10430（`/root/hujianmin/llama.cpp`），llama-server 运行于 8080

## M1 工程骨架 / GGUF 读取 / 权重加载

- 工程位于 `/root/hujianmin/qwen3-gpu-infer`，CMake 构建通过
- 极简 GGUF V3 解析器（自写）读取超参与全部 310 个张量；BF16 权重上 GPU
- 超参从 GGUF 读取（n_layer=28、n_embd=1024、n_head=16、n_head_kv=8、
  head_dim=128、n_ff=3072、n_vocab=151936、rope_theta=1e6、norm_eps=1e-6），
  与 llama.cpp 打印一致

## M2 Prefill 数值对齐

用 llama.cpp（`ref/ref_logits.cpp`）导出 prompt "Hello world" 的 logits，
与本引擎 prefill logits 对比：

| 指标 | 数值 |
|------|------|
| logits 最大绝对误差 | 0.074（目标 ~1e-2 量级，BF16 容差内） |
| logits 平均绝对误差 | 0.013 |
| top-10 一致率 | 10/10 |

另用独立的 NumPy 全模型参考交叉验证：NumPy 与 llama.cpp 的最大误差 0.049，
本引擎各层激活与 NumPy 逐层一致（误差随层数缓慢放大，属 BF16 数值累积）。

## M3 Decode 完整生成

`-p "Hello world" -n 96` 的生成（greedy）：

> ! This is a simple example of a Python script that prints "Hello World" to the
> console. The script is written in Python and uses the print function to output
> the message...

语义合理、无乱码，与 llama.cpp 原始续写一致。

## M4 批处理（5 条 prompt）

`demo_prompts.txt`（技术/历史/饮食/旅行/文学）批处理输出与 llama-server 原始续写
逐条对比，5/5 主题一致、几乎逐字相同，例如：

- "写一段描述川菜麻婆豆腐的文字" → 双方均输出
  "，要求包含以下要素：1.麻婆豆腐的起源，2.麻婆豆腐的特色，..."
- "如果去深圳旅游，推荐三天行程" → 双方均输出
  "，包括景点、美食、交通和住宿，还要有特色活动和注意事项。三天行程..."

批处理 decode 526 tok/s（5 序列并行）。

## M5 性能对比

同机 A100，相同模型 / prompt / `max_tokens`：

| 实现 | decode tok/s（单序列，n=96） |
|------|------|
| llama.cpp `simple.cpp`（`ref_tools/simple_ref`） | 187.1 |
| **qwen3-gpu-infer** | **179.6（96.0%）** |

目标 ≥ 80%（150 tok/s）达成。

主要优化：BF16 tensor-core GEMM（`mma.m16n8k16`）、decode 专用 split-K
小 M GEMM、QKV 融合为单 GEMM、attention 按 head 并行、KV cache BF16。

> 注：服务器为多人共享，存在外部任务持续占满 4 卡 GPU 的情况。上述 179.6 tok/s
> 为 GPU 空闲窗口测得；受外部负载干扰时（4 卡利用率 100%）测得 49~71 tok/s，
> 属环境竞争而非引擎性能波动。`simple_ref` 的 187.1 tok/s 同样在空闲窗口测得。

## M6 收尾

- README（本仓库根目录）
- 构建/运行/对比脚本：`scripts/build.sh`、`scripts/run_demo.sh`、
  `scripts/compare_llama.cpp.sh`
- 本报告：`docs/report.md`
- 算子自测：`QWEN_SELFTEST=1` 全部通过（gemm / rms_norm / rope / attention，
  与 CPU 参考最大误差 < 1e-5）

## 调试中修复的关键缺陷

1. GGUF tensor offset 为相对 tensor 数据段的偏移（最初按文件头偏移读取，全部权重错位）
2. BF16 tensor-core GEMM 的 mma k16 只累加了一半 K
3. 多行 SwiGLU 原地计算存在跨行写读竞争（gate 输出覆盖上一行 up 源）
4. split-K GEMM 在 ksplit=1 时仍按 K/4 切分，丢失 3/4 的 K（LM head 受影响）
5. Q/K head 维度 RMSNorm 与 RoPE 之间缺少 `__syncthreads`
6. Qwen3 使用 NeoX RoPE（半拆分对），最初误用相邻对
