# Qwen2.5-VL-32B 模型并行计划

本文档记录 HERMES 中 Qwen2.5-VL-32B 从单卡运行迁移到多卡模型并行的需求、实现步骤和测试流程。这里讨论的是模型并行，不是数据并行。

## 1. 背景与目标

Qwen2.5-VL-32B 的 fp16 权重约 64GB。80G 单卡即使勉强装下权重，也很难同时容纳：

- 视频 KV cache；
- HERMES 压缩阶段的注意力打分临时张量；
- vision/text forward 的中间激活；
- `lm_head` 等回答阶段临时张量；
- CUDA allocator 预留和碎片。

因此，32B 在 `fps=0.5, kv_size=4000` 等配置下容易出现 CPU offload 或 OOM。目标是让一个 32B 推理进程同时使用多张 GPU，通过模型并行分摊权重和每层 KV cache，尽量避免 CPU offload，并为 HERMES、Sliding Window 和 ViSpec draft latency 后端保留统一入口。

## 2. 模型并行边界

### 2.1 不采用数据并行

数据并行会让每张 GPU 都保存一份完整 32B 权重：

```text
GPU0: full Qwen2.5-VL-32B
GPU1: full Qwen2.5-VL-32B
...
```

这无法解决单卡显存不足问题，也不适合当前单视频流式推理场景。

### 2.2 采用层级模型并行

HuggingFace/Accelerate 的 `device_map="auto"` 默认更接近层级模型并行。它通常按模块或 decoder layer 切分权重，例如：

```text
GPU0: vision encoder + embedding + layer 0-30
GPU1: layer 31-63 + final norm + lm_head
```

实际切分取决于模型结构、可见 GPU 数量、每张卡可用显存和 `max_memory` 设置。

## 3. 通信与 KV Cache 分布

### 3.1 跨设备通信内容

层级模型并行的跨卡通信主要发生在相邻模块的设备边界。跨卡传输的是 hidden states，而不是 KV cache：

```text
layer 30 on GPU0:
  更新 layer 30 自己的 KV cache
  输出 hidden_states

hidden_states 从 GPU0 拷贝到 GPU1

layer 31 on GPU1:
  读取 layer 31 自己的历史 KV cache
  更新 layer 31 自己的新 KV cache
```

因此通信成本主要取决于：

```text
seq_len * hidden_size * dtype
```

prefill / video chunk encoding 阶段 `seq_len` 较长，跨卡传输更重；decode 阶段每步通常只有 1 个新 token，单步通信较小，但会重复很多次。HERMES 的 `pseudo_forward()` 和压缩阶段会额外触发 forward 或手工 attention，因此也会产生额外通信和显存压力。

如果机器有 NVLink，模型并行 latency 会更可接受；如果只有 PCIe，也可以先解决能否跑通，但速度未必优于单卡小模型。

### 3.2 KV cache 跟随层所在设备

Transformer 的 KV cache 是按层私有的。下一层不会使用上一层的 KV cache。若模型被切成：

```text
GPU0: layer 0-30
GPU1: layer 31-63
```

则 KV cache 应自然分布为：

```text
GPU0:
  layer 0 cache
  layer 1 cache
  ...
  layer 30 cache

GPU1:
  layer 31 cache
  layer 32 cache
  ...
  layer 63 cache
```

Python 侧的 `past_key_values` / `DynamicCache` 只是一个容器，容器中每层 `k_cache, v_cache` tensor 保存在对应层所在 GPU 上。

### 3.3 压缩应按层本地执行

HERMES 压缩不能把所有 cache 统一搬到 `self.device`。正确策略是按层使用 `k_layer.device`：

```python
for layer_idx, (k_layer, v_layer) in enumerate(kv_cache):
    layer_device = k_layer.device
    keep_indices = keep_indices_per_layer[layer_idx].to(layer_device)

    k_new = torch.index_select(k_layer, dim=2, index=keep_indices)
    v_new = torch.index_select(v_layer, dim=2, index=keep_indices)
```

同理，position ids、attention scores、summary key/value、`torch.arange()` 产生的索引张量，都应放在当前层的 `layer_device` 上。

## 4. 实现需求

1. 一个 32B 推理进程能够看到多张 GPU，例如 `CUDA_VISIBLE_DEVICES=0,1`。
2. `device_map="auto"` 或显式 `max_memory` 能将权重切到多张 GPU。
3. 运行日志能打印实际 `hf_device_map`，用于确认权重分布。
4. 运行日志能明确提示是否存在 CPU/disk offload。
5. Qwen2.5-VL 的 HERMES、Sliding Window、ViSpec draft latency 后端共享同一套模型并行参数。
6. HERMES 自定义 KV 操作必须支持每层 cache 位于不同 GPU。
7. 压缩、裁剪、summary token 和位置重索引不能把跨卡 cache 隐式搬回主卡。
8. 保留已有 `--use_flash_attention` 行为：默认 eager attention，显式传参才启用 Flash Attention 2。

## 5. 编写步骤

### 5.1 第一阶段：不改调度器，手工 smoke test

先绕开 `video_qa/run_infer.py` 的单卡子进程分配，直接让一个 `hermes_vqa.py` 进程看到两张卡：

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1 python video_qa/hermes_vqa.py \
  --model qwen2.5_vl_32b \
  --sample_fps 0.5 \
  --save_dir results/qwen2.5_vl_32b/rvs_movie/fps0.5-kv1000-mp2-smoke \
  --anno_path data/rvs/movie/movienet_oe.json \
  --debug true \
  --num_chunks 1 \
  --chunk_idx 0 \
  --kv_size 1000 \
  --streaming True
```

第一轮建议先用 `kv_size=1000`，目标是验证模型并行和自定义 KV 逻辑是否兼容，而不是直接追求 kv4000 完整结果。

### 5.2 第二阶段：给加载函数增加可控显存配置

在 Qwen 后端 `load_model()` 中增加可选参数，例如：

```text
max_memory_per_gpu
disallow_cpu_offload
print_device_map
```

加载时优先使用受控 `max_memory`：

```python
max_memory = {
    gpu_idx: max_memory_per_gpu
    for gpu_idx in range(torch.cuda.device_count())
}

base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    max_memory=max_memory,
    torch_dtype=torch.float16,
    attn_implementation=attn_implementation,
)
```

80G 卡建议先给每张卡留出较多余量，例如：

```text
max_memory_per_gpu = "72GiB"
```

如果 `hf_device_map` 中出现 `"cpu"` 或 `"disk"`，应在日志中显式警告；严格模式下可以直接报错，避免误把 CPU offload 结果当作 GPU 模型并行结果。

### 5.3 第三阶段：适配 HERMES 跨设备 KV 操作

重点排查以下文件：

- `inference/qwenvl_hermes.py`
- `inference/qwenvl_slidingwindow.py`
- `inference/qwenvl_vispec_draft_latency.py`
- `inference/abstract_hermes.py`
- `inference/reindex_3d.py`

重点函数：

- `encode_init_prompt()`
- `encode_video_chunk()`
- `predict_and_compress()`
- `pseudo_forward()`
- `_compute_attention_scores_manually()`
- `prune_kv_cache_by_attention()`
- `apply_kv_cache_pruning_strict()`
- `_shrink_positions_and_rerotate_keys()`
- `question_answering()`

需要将这类写法：

```python
tensor.to(self.device)
torch.arange(..., device=self.device)
torch.tensor(..., device=self.device)
```

逐步改为 layer-local device：

```python
layer_device = k_layer.device
tensor = tensor.to(layer_device)
idx = torch.arange(..., device=layer_device)
```

对于非逐层的输入张量，例如 `input_ids`、`pixel_values`、`position_ids`，应遵循 HuggingFace 模型入口期望的主设备或 embedding 设备；对于逐层 cache、逐层位置缓存、逐层裁剪索引，必须跟随该层 cache device。

### 5.4 第四阶段：给调度器增加每个 chunk 多 GPU

当前 `video_qa/run_infer.py` 会给普通模型的每个 chunk 分配一张 GPU。模型并行需要一个 chunk 看到多张 GPU。建议新增参数：

```bash
--gpus_per_chunk 2
```

调度逻辑改为：

```text
chunk 0 -> CUDA_VISIBLE_DEVICES=0,1
chunk 1 -> CUDA_VISIBLE_DEVICES=2,3
...
```

示例命令：

```bash
python video_qa/run_infer.py \
  --num_chunks 1 \
  --gpus_per_chunk 2 \
  --gpu_ids 0,1 \
  --model qwen2.5_vl_32b \
  --dataset rvs_movie \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --debug true \
  --skip_eval
```

如果未来要并行跑两个 32B chunk，可使用：

```bash
python video_qa/run_infer.py \
  --num_chunks 2 \
  --gpus_per_chunk 2 \
  --gpu_ids 0,1,2,3 \
  --model qwen2.5_vl_32b \
  --dataset rvs_movie \
  --sample_fps 0.5 \
  --kv_size 1000 \
  --debug true \
  --skip_eval
```

## 6. 测试流程

### 6.1 Load-only 检查

目标：确认权重能分布到多张 GPU，且没有 CPU/disk offload。

检查项：

- `torch.cuda.device_count()` 是否等于当前进程可见 GPU 数；
- 日志是否打印 `hf_device_map`；
- `hf_device_map` 中是否只包含 GPU；
- 日志中是否没有 `offloaded to the cpu`；
- `nvidia-smi` 中多张卡都出现该 Python 进程；
- 每张卡显存是否低于安全阈值，例如 72GB。

### 6.2 Qwen Sliding Window kv1000

目标：先验证最简单的逐层 KV 裁剪能跨设备工作。

建议命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python video_qa/hermes_vqa.py \
  --model qwen2.5_vl_32b_slidingwindow \
  --sample_fps 0.5 \
  --save_dir results/qwen2.5_vl_32b_slidingwindow/rvs_movie/fps0.5-kv1000-mp2-smoke \
  --anno_path data/rvs/movie/movienet_oe.json \
  --debug true \
  --num_chunks 1 \
  --chunk_idx 0 \
  --kv_size 1000 \
  --streaming True
```

通过标准：

- 不出现跨设备 tensor error；
- 不出现 CPU offload；
- 能生成至少一个视频的完整 `results.csv`；
- token timing 与单卡 kv1000 相比没有异常数量级劣化。

### 6.3 Qwen HERMES kv1000

目标：验证 HERMES 的伪查询、attention scoring、strict shrink 和回答阶段都能跨设备运行。

建议命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python video_qa/hermes_vqa.py \
  --model qwen2.5_vl_32b \
  --sample_fps 0.5 \
  --save_dir results/qwen2.5_vl_32b/rvs_movie/fps0.5-kv1000-mp2-smoke \
  --anno_path data/rvs/movie/movienet_oe.json \
  --debug true \
  --num_chunks 1 \
  --chunk_idx 0 \
  --kv_size 1000 \
  --streaming True
```

重点检查：

- `_compute_attention_scores_manually()` 是否出现跨设备错误；
- `apply_kv_cache_pruning_strict()` 后每层 cache 长度是否符合预期；
- `_position_ids_cache[layer_idx]` 是否与对应层 cache 在同一设备；
- 回答结束后的 KV rollback 是否按层正确截断；
- 是否再次出现 fp32 attention 临时张量导致的 OOM。

### 6.4 Qwen HERMES kv4000

目标：验证目标实验配置能否在多卡 GPU-only 下完成。

建议命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 python video_qa/hermes_vqa.py \
  --model qwen2.5_vl_32b \
  --sample_fps 0.5 \
  --save_dir results/qwen2.5_vl_32b/rvs_movie/fps0.5-kv4000-mp2 \
  --anno_path data/rvs/movie/movienet_oe.json \
  --debug true \
  --num_chunks 1 \
  --chunk_idx 0 \
  --kv_size 4000 \
  --streaming True
```

通过标准：

- 能完整处理 115 个问题；
- 不出现 CPU/disk offload；
- 不出现 OOM；
- token timing、prefill timing、压缩耗时均能稳定记录；
- GPU 显存曲线没有持续泄漏。

## 7. 日志与数据记录

每次模型并行实验至少记录：

- `CUDA_VISIBLE_DEVICES`；
- `num_chunks`；
- `gpus_per_chunk`；
- `gpu_ids`；
- `model`；
- `sample_fps`；
- `kv_size`；
- `attn_implementation`；
- `max_memory_per_gpu`；
- `hf_device_map`；
- 是否出现 CPU/disk offload；
- 每张 GPU 的峰值显存；
- token timing summary；
- prefill timing summary；
- strict shrink 次数与耗时；
- 最终完成问题数。

建议日志文件名显式包含模型并行配置，例如：

```text
qwen32b_hermes_fps0.5_kv4000_mp2_gpu0-1_YYYYMMDD_HHMMSS.log
```

## 8. 风险点与判断标准

### 8.1 主要风险

1. `self.device` 假设单卡，导致逐层 cache 被搬到错误 GPU。
2. 手工 attention scoring 创建大 fp32 临时张量，仍可能在某层所在 GPU OOM。
3. `hf_device_map` 自动切分不理想，导致某张卡过满或 CPU offload。
4. PCIe 跨卡通信导致 latency 明显升高。
5. vision encoder、embedding、lm_head 集中在同一张卡，造成局部显存热点。
6. Flash Attention 2 与 3D position ids、手写 patch 或跨设备 cache 存在兼容风险。

### 8.2 通过标准

模型并行版本只有同时满足以下条件，才可作为正式实验口径：

- 权重和 KV cache 均在 GPU 上；
- 无 CPU/disk offload；
- 无跨设备 tensor error；
- 无 OOM；
- 完成完整数据集或明确完成指定 chunk；
- 日志中有实际 `hf_device_map` 和 token timing；
- HERMES 与 Sliding Window 的参数、attention 后端和 GPU 数量记录清楚。

## 9. 推荐执行顺序

1. 手工 `CUDA_VISIBLE_DEVICES=0,1` 跑 load-only 或极小样本。
2. 跑 `qwen2.5_vl_32b_slidingwindow, kv_size=1000`。
3. 修复 Sliding Window 暴露出的跨设备 KV 裁剪问题。
4. 跑 `qwen2.5_vl_32b, kv_size=1000`。
5. 修复 HERMES attention scoring、summary token、position cache 的跨设备问题。
6. 增加 `max_memory_per_gpu`、`print_device_map`、`disallow_cpu_offload` 等加载参数。
7. 增加 `run_infer.py --gpus_per_chunk`。
8. 跑 `qwen2.5_vl_32b, kv_size=4000`。
9. 汇总 GPU 显存、完成问题数、token latency、压缩耗时和 OOM/offload 状态。

短期优先级是先证明两卡 GPU-only 能跑通 `kv_size=1000`。只有这个链路稳定后，`kv_size=4000` 的结果才有解释价值。
