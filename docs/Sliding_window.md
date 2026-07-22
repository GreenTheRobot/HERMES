# Sliding-window 基线实现说明

## 1. 修改目的

本次新增一个只依赖滑动窗口的 KV cache 基线，用于和 HERMES 的分层记忆策略做可控对比。新实现不使用 HERMES 的 local/global/mixed 伪查询，不计算查询到历史 KV 的注意力分数，不做分层 Top-K，也不生成长期记忆摘要 token。

原始实现没有被修改：

- `inference/llavaov_hermes.py` 保持为 LLaVA-OneVision 的 HERMES 实现；
- `inference/qwenvl_hermes.py` 保持为 Qwen2.5-VL 的 HERMES 实现。

两个新文件均先从对应 HERMES 文件完整复制，再只在副本中调整运行路径：

- `inference/llavaov_slidingwindow.py`；
- `inference/qwenvl_slidingwindow.py`。

这样保留了原项目已有的模型加载、视频特征提取、KV 生命周期、问答模板、位置编码修正和结果生成约定，也避免滑动窗口实验污染原 HERMES 后端。

## 2. 文件改动总览

| 文件 | 修改内容 |
|---|---|
| `inference/llavaov_slidingwindow.py` | 新增 LLaVA-OneVision 滑动窗口后端，使用 1D RoPE 位置维护，并加入逐 token 计时 |
| `inference/qwenvl_slidingwindow.py` | 新增 Qwen2.5-VL 滑动窗口后端，保留 3D M-RoPE 位置维护，并加入逐 token 计时 |
| `video_qa/base.py` | 注册六个带 `_slidingwindow` 后缀的模型入口，Qwen 仍采用延迟导入以兼容不同 Transformers 环境 |
| `video_qa/run_infer.py` | 将六个新入口加入 `--model` choices，并让 LLaVA 72B 滑动窗口版本沿用四卡映射约定 |
| `video_qa/hermes_vqa.py` | 将当前问题的逐 token 计时序列化到结果 CSV 的 `token_inference_times` 字段 |
| `docs/Sliding_window.md` | 本说明文档 |

## 3. 模型入口与类名

新类名用于明确区分实验方法：

- `LlavaOneVision_SlidingWindow`；
- `QwenVL_SlidingWindow`。

可用的命令行模型名如下：

- `llava_ov_0.5b_slidingwindow`；
- `llava_ov_7b_slidingwindow`；
- `llava_ov_72b_slidingwindow`；
- `qwen2.5_vl_3b_slidingwindow`；
- `qwen2.5_vl_7b_slidingwindow`；
- `qwen2.5_vl_32b_slidingwindow`。

模型权重路径完全复用相同尺寸的原版配置，因此无需创建另一份权重目录。

## 4. 滑动窗口的精确定义

### 4.1 `kv_size` 的含义

本实现将 `kv_size` 解释为“最多保留的视觉 token 数”，与原 HERMES 压缩代码中对视觉预算的用法保持一致。初始化系统提示对应的 KV 前缀始终保留，不计入 `kv_size`。

压缩后的每层 KV 长度满足：

```text
初始化文本前缀长度 + min(kv_size, 当前视觉 token 数)
```

因此总 KV 长度通常会略大于 `kv_size`，多出的部分是固定文本前缀。`load_model()` 会检查 `kv_size`，缺失、为零或为负数都会直接抛出 `ValueError`，避免实验在预算无效时静默运行。

### 4.2 保留索引

视频块仍由 `HermesVQA.analyze_a_video()` 按原约定分块编码。每次调用 `encode_video_chunk()` 后，驱动层继续调用兼容接口 `predict_and_compress()`。新实现中的该接口不再预测伪问题，而是直接构造每层保留索引：

```text
[0, visual_start_idx)                         固定初始化文本前缀
[max(visual_start_idx, seq_len-kv_size), ...) 最近的视觉 token
```

索引在所有层遵循同一时间窗口，不依赖注意力、层深或历史活跃度。当当前长度不超过 `visual_start_idx + kv_size` 时直接返回，不发生 KV 拷贝或位置重排。

如果单个视频块产生的视觉 token 已经超过窗口，窗口会保留该块末尾最新的 `kv_size` 个 token；更早的 token（包括同一块开头）会被移除。

### 4.3 明确禁用的 HERMES 行为

由于新文件是从原实现复制的，一些 HERMES 辅助函数仍保留，以最大程度保持文件结构并方便对照，但正常滑动窗口入口不会调用下列路径：

- `predict_next_question()`；
- `pseudo_forward()`；
- `_compute_attention_scores_manually()`；
- `prune_kv_cache_by_attention()`；
- `allocate_budget_by_depth()`；
- `apply_kv_cache_pruning_strict()`。

`predict_and_compress()` 只执行确定性的窗口截断。截断时会暂时把 `long_term_threshold` 设为层数，借此复用两个模型后端已经验证过的 RoPE/M-RoPE KV 重排代码，同时关闭其“深层被删除 token 聚合为摘要 token”的分支。调用结束后原阈值会在 `finally` 中恢复。最终窗口中没有摘要 token、注意力选中的旧 token或其他分层记忆内容。

后续若重构公共位置编码代码，可把这段窗口重排抽成独立 helper；在此之前，不应删除临时阈值保护，否则深层会重新产生 HERMES 摘要 token，滑动窗口基线将不再纯净。

## 5. 两个模型的位置编码处理

直接切片 KV 后必须同步维护逻辑位置，否则后续视频或文本 token 的 RoPE 会和缓存 Key 不一致。两个新后端分别沿用各自原实现的处理：

### 5.1 LLaVA-OneVision

LLaVA 使用 1D RoPE 和每层 `_position_ids_cache`。

- `streaming=True`：通常保留被选 token 的原逻辑位置，只在接近 `max_position_embeddings` 上限时压紧位置并对 Key 做旋转差修正；
- `streaming=False`：每次窗口截断都把保留 token 压到连续位置，并对 Key 应用对应的旋转差。

### 5.2 Qwen2.5-VL

Qwen 使用时间、高度、宽度三个维度的 M-RoPE。窗口截断同时切片三维 `_position_ids_cache`。需要压紧时，三维位置分别按仍然存在的唯一坐标重新映射，并使用 `mrope_section` 对缓存 Key 做旋转差修正。

不能把 LLaVA 的 1D 位置重排直接复制到 Qwen 文件中；二者虽共享窗口索引定义，位置修正必须继续分开维护。

## 6. 逐 token 推理计时

两个滑动窗口后端的 `question_answering()` 都会为每个实际输出 token 记录一次计时。记录保存在模型实例的：

```python
model.last_token_inference_times
```

列表中每个元素的结构为：

```json
{
  "token_index": 0,
  "token_id": 123,
  "phase": "prefill",
  "latency_seconds": 0.123456789
}
```

字段含义：

- `token_index`：当前回答内从 0 开始的 token 序号；
- `token_id`：Tokenizer 的输出 token id；
- `phase`：首 token 为 `prefill`，其余 token 为 `decode`；
- `latency_seconds`：以 `time.perf_counter()` 测得的秒数。

首 token 的延迟是 TTFT，包含问题文本 prefill 到首 token 可选取为止的时间。后续 token 的延迟只包围单 token 的 `language_model(...)` 前向，不包含日志、Tokenizer 解码和下一轮采样。CUDA 输入下在计时边界调用 `torch.cuda.synchronize(device)`，避免异步 kernel 导致读数偏小；非 CUDA 输入不执行同步。

每条记录同时以 `[TokenTiming]` 日志输出，并由 `video_qa/hermes_vqa.py` 写入结果 CSV 的 `token_inference_times` 列。CSV 单元格内容是 JSON 数组，可以在 Pandas 中这样展开：

```python
import json
import pandas as pd

df = pd.read_csv("results.csv")
timings = df["token_inference_times"].map(json.loads).explode().apply(pd.Series)
decode_latency = timings.loc[timings["phase"] == "decode", "latency_seconds"]
print(decode_latency.describe())
```

每次问答开始时 `last_token_inference_times` 都会清空，因此 CSV 中一行只对应该问题自己的 token 时延，不会混入上一个问题。

## 7. 运行示例

运行方式和原项目一致，只需替换模型名：

```bash
python video_qa/run_infer.py \
  --model llava_ov_7b_slidingwindow \
  --dataset streamingbench \
  --num_chunks 1 \
  --sample_fps 0.5 \
  --kv_size 6000
```

Qwen 示例：

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_7b_slidingwindow \
  --dataset videomme \
  --num_chunks 1 \
  --sample_fps 1 \
  --kv_size 6000
```

结果目录仍使用 `results/{model}/{dataset}/fps{sample_fps}-kv{kv_size}`，所以滑动窗口结果会自然落在带 `_slidingwindow` 的独立模型目录中，不会覆盖原 HERMES 结果。

## 8. 后续修改建议

### 修改窗口单位

当前窗口单位是视觉 token，不是帧。若要改成“最近 N 帧”，需要在 `encode_video_chunk()` 中保存每帧对应的 token 边界，再由 `predict_and_compress()` 按帧边界生成索引；仅把 `kv_size` 政名为帧数会产生错误结果。

### 修改触发频率

当前每编码一个视频块检查一次窗口。若要在编码前先腾出空间，应同时考虑新块产生的 token 数，并调整 `HermesVQA.analyze_a_video()` 的调用顺序。无论采用哪种方式，都要保证单块大于窗口时仍能在编码完成后截到严格预算。

### 修改固定前缀

固定前缀边界由 `encode_init_prompt()` 设置的 `visual_start_idx` 决定。不要硬编码为 14，因为不同 Tokenizer、系统提示或模型版本可能得到不同长度。

### 增加统计指标

如果需要吞吐量，可从 `phase == "decode"` 的记录计算 `1 / mean(latency_seconds)`；如果需要端到端回答耗时，应另加一个覆盖采样、解码和日志的外层计时，不要把它与当前单 token 模型前向时延混用。

## 9. 无权重环境下的验证范围

本机没有模型权重，因此本次不执行真实加载和推理。应至少运行 Python 语法编译检查，覆盖两个新后端及三个被修改的入口文件。获得权重后建议补做以下 smoke test：

1. 令 `kv_size` 很小，编码多个视频块，确认每层长度始终不超过 `visual_start_idx + kv_size`；
2. 确认所有层的 KV 长度与各自 `_position_ids_cache` 长度一致；
3. 确认日志中没有 local/global/mixed 伪查询；
4. 确认深层长度与浅层一致，没有额外摘要 token；
5. 确认输出 token 数等于 `token_inference_times` 元素数；
6. 分别测试 `streaming=True` 和 `streaming=False` 的位置压紧路径；
7. 对 Qwen 额外检查三维位置缓存形状始终为 `[3, seq_len]`。
