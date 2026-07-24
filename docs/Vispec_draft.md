# ViSpec Draft Head Latency Backend 设计草案

## 1. 目标

本文档描述在 HERMES 中接入 ViSpec-Qwen2.5-VL-3B-Instruct 与 ViSpec-Qwen2.5-VL-7B-Instruct 草稿头的第一阶段方案。

第一阶段只实现一个 **ViSpec 草稿头 latency backend**：

- 保留 HERMES 原有视频 KV 编码、驱逐和压缩流程。
- 每次回答前，目标 Qwen2.5-VL 对当前问题执行一次 prefill forward。
- 使用这次目标 forward 产生的真实 hidden states 驱动 ViSpec 草稿头。
- 草稿头沿用 ViSpec 的 tree draft 设计，仅生成一次 draft tree。
- 不执行目标模型 tree verify。
- 不执行 speculative accept / reject。
- 不把 draft token 写回目标模型 KV。
- 记录 ViSpec 草稿头每个 tree depth 层的 latency 与这一层实际并行 forward 的输入 token 数。

这个阶段的目的不是提升端到端速度，而是回答一个更基础的问题：在 HERMES 当前视频 KV 状态下，ViSpec 草稿头本身的起草开销是多少，以及这个开销如何随 depth、top-k、total-token 和保留视觉上下文变化。

## 2. 设计理念

### 2.1 先测真实输入分布下的草稿头开销

ViSpec 草稿头不是独立小 VLM。它的核心输入包括：

```text
target model last_hidden_states
target token embeddings
optional visual compressed context
```

因此不能简单把草稿头当作 `input_ids -> logits` 的 standalone draft model。若使用随机 hidden states 或 token embedding 代替目标 hidden states，虽然可以得到算子级 microbenchmark，但无法反映真实接入时的输入分布。

本方案选择每次问题 prefill 后让目标模型跑一次 forward，取得真实 hidden states，再让草稿头起草一次。这保留了 ViSpec 的关键语义，同时避免过早处理完整 speculative decoding 中最复杂的 verify / accept / KV gather。

### 2.2 目标模型 KV 与草稿头 KV 分离

HERMES 的主 KV cache 是长期视频记忆：

```text
init prompt KV + retained video KV
```

问题 token 和回答 token 只在 `question_answering()` 中临时 append，回答结束后会 truncate 回回答前长度。HERMES 的压缩主要管理视觉 KV，并维护 layer-wise position cache 与 RoPE/M-RoPE 重索引。

ViSpec 草稿头的 KV 是短生命周期状态，只服务当前一次 draft tree 生成。它不应进入 HERMES 的长期视频 KV，也不应参与 HERMES 的驱逐、summary token、RoPE rerotate。

推荐生命周期：

```text
profile_vispec_draft():
  reset draft text/tree KV
  lazy build draft visual context
  target question prefill once
  draft tree once
  record latency
  clear draft text/tree KV
```

### 2.3 草稿头视觉上下文采用 lazy rebuild

ViSpec 的视觉压缩模块较轻。第一阶段不在每次 `encode_video_chunk()` 或 HERMES `predict_and_compress()` 时持续维护草稿头视觉 KV，而是在每次起草前根据当前 HERMES 保留的视觉源信息做一次 lazy rebuild。

这避免了两个问题：

- 草稿头视觉压缩 token 与 HERMES target KV token 不一定一一对应。
- HERMES 压缩后可能存在 long-term summary KV，直接裁剪草稿头 KV 很难保证语义一致。

需要注意的是，ViSpec 视觉压缩模块原本需要的是视觉 token 的 embeddings 和目标模型 hidden states，而不是目标模型 KV tensor 本身。因此不能只拿 HERMES 的 K/V 直接喂给 ImgAdaptor。应维护一份轻量 source buffer，用来在起草前重建草稿视觉上下文。

## 3. 相关现有代码

ViSpec 侧核心文件：

- `ViSpec/vispec/model/spec_model_ours.py`
  - `SpecModel` 加载 base model 与草稿层。
  - `specgenerate()` 实现完整 speculative decoding。
- `ViSpec/vispec/model/cnets_ours.py`
  - `Model` 是草稿头主体。
  - `ImgAdaptor` 与 `img_fc` 实现 vision-aware draft context。
  - `topK_genrate()` 生成 draft tree。
- `ViSpec/vispec/model/utils.py`
  - `initialize_tree()` 执行 target prefill 后初始化 draft tree。
  - `tree_decoding()` 执行 target tree verify。
  - `evaluate_posterior()` 与 `update_inference_inputs()` 处理 accept / cache update。
- `ViSpec/vispec/train/qwen2.5_vl_3B_config.json`
- `ViSpec/vispec/train/qwen2.5_vl_7B_config.json`

HERMES 侧核心文件：

- `HERMES/inference/qwenvl_hermes.py`
  - `encode_init_prompt()` 初始化固定文本前缀 KV。
  - `encode_video_chunk()` 将视频 chunk 编入目标模型 KV。
  - `predict_and_compress()` 触发 HERMES 伪查询与压缩。
  - `question_answering()` 执行当前问题 prefill 与回答 decode。
- `HERMES/video_qa/base.py`
  - 注册模型后端。
- `HERMES/video_qa/hermes_vqa.py`
  - 视频级推理循环，负责逐 chunk 编码和逐问题回答。

## 4. 第一阶段后端边界

### 4.1 新增后端形态

建议新增独立后端，而不是直接替换现有 Qwen HERMES 后端：

```text
qwen2.5_vl_3b_vispec_draft_latency
qwen2.5_vl_7b_vispec_draft_latency
```

输出结果和 baseline 结果隔离。

### 4.2 输入

一次 draft profile 的输入包括：

```text
input_text:
  question
  prompt
  formatted_question, optional

HERMES state:
  self.kv_cache
  self._position_ids_cache
  self.visual_start_idx
  self.processor

ViSpec config:
  spec_model_path
  depth
  top_k
  total_token
  num_q
  temperature
```

### 4.3 输出

第一阶段不输出 draft answer。推荐输出结构化 profile 结果：

```json
{
  "target_prefill_latency_seconds": 0.123,
  "draft_head_total_latency_seconds": 0.009,
  "draft_tree_pack_latency_seconds": 0.001,
  "draft_depth": 3,
  "draft_top_k": 8,
  "draft_total_token": 30,
  "draft_tokens_shape": [1, 30],
  "retrieve_indices_shape": [16, 4],
  "layer_timings": [
    {
      "tree_depth": 0,
      "phase": "draft_initial",
      "frontier_tokens": 1,
      "candidate_tokens": 8,
      "selected_frontier_tokens": 8,
      "latency_seconds": 0.002
    },
    {
      "tree_depth": 1,
      "phase": "draft_expand",
      "frontier_tokens": 8,
      "candidate_tokens": 64,
      "selected_frontier_tokens": 8,
      "latency_seconds": 0.003
    }
  ]
}
```

其中 `tree_depth` 指 draft tree 层级，不是 transformer layer。ViSpec Qwen 草稿头配置里通常只有 1 层 decoder。

## 5. 执行流程

### 5.1 视频编码阶段

HERMES 原流程保持不变：

```text
clear_cache()
encode_init_prompt()
for each video chunk:
  encode_video_chunk(video_chunk)
  predict_and_compress()
```

第一阶段可新增一个 source buffer，但不立即更新草稿头视觉 KV：

```text
encode_video_chunk():
  video_chunk -> processor -> video_features
  video_features -> target language_model -> target hidden_states + target KV
  save draft visual source:
    chunk_id
    global token indices
    video_features
    target hidden_states
    position_ids_3d
```

`predict_and_compress()` 后，需要根据 HERMES 当前 layer-wise keep 结果同步 compact source buffer：

```text
predict_and_compress():
  HERMES computes keep_indices
  HERMES prunes/rerotates target KV
  compute source keep set from layer-wise keep_indices
  compact source buffer using the keep set
```

第一版的正式默认策略是：

```text
source_keep_policy = union_all
```

也就是只要某个原始视觉 token 仍被 HERMES 任意一层 KV 显式保留，就保留它对应的 source embedding、target hidden states 和 position ids。这样语义最保守，不会提前删除目标模型某一层仍在使用的视觉证据。

HERMES long-term summary KV 第一版不向草稿头暴露，因为 summary KV 没有直接对应的原始视觉 embedding。compact 时只管理原始视觉 token source，并在 profile 中记录 summary token 被排除。

### 5.2 起草前 lazy rebuild 视觉上下文

起草前执行：

```text
reset spec_layer.stable_kv
reset spec_layer.last_img_hidden
collect compacted visual source from source buffer
run ViSpec visual compression once
store temporary draft visual context for this draft call
```

只构造 `last_img_hidden` / visual summary，不持久化草稿视觉 KV。


### 5.3 目标问题 prefill

沿用 HERMES Qwen 的 position 逻辑：

```text
past_lens_prefill = self._get_cache_seq_len_per_layer()
global_offset_prefill = self._get_next_global_offset_per_layer()
input_ids = tokenizer(prompt)
inputs_embeds = self.get_input_embeddings()(input_ids)
position_ids_3d = build text 3D positions
out = self.language_model(
  inputs_embeds=inputs_embeds,
  past_key_values=self.kv_cache,
  use_cache=True,
  position_ids=position_ids_3d,
  output_hidden_states=True or return last_hidden_state
)
target_hidden_states = out.last_hidden_state
target_logits = self.lm_head(target_hidden_states)
```

这次 prefill 的 KV 是临时的。profile 结束后，仍应 truncate 回 `past_lens_prefill`，与 HERMES 原 `question_answering()` 语义一致。

### 5.4 草稿头 tree draft

用 ViSpec 的 `topK_genrate()` 语义生成一次 tree：

```text
sample_token = argmax(target_logits[:, -1])
draft_input_ids = concat(input_ids, sample_token)
draft_tokens, retrieve_indices, tree_mask, tree_position_ids =
  spec_layer.topK_genrate(
    hidden_states=target_hidden_states,
    input_ids=draft_input_ids,
    head=self.lm_head,
    logits_processor=None,
    inputs_embeds=question_inputs_embeds or reconstructed embeds,
    image_mask=optional
  )
```

第一阶段不调用：

```text
tree_decoding()
evaluate_posterior()
update_inference_inputs()
```

也不把 `draft_tokens` 作为答案输出。

### 5.5 结束清理

profile 结束后：

```text
spec_layer.reset_kv()
truncate target KV back to past_lens_prefill
truncate position_ids_cache back to past_lens_prefill
torch.cuda.empty_cache(), optional
```

## 6. KV Cache 管理

### 6.1 HERMES target KV

HERMES target KV 长期保存：

```text
init prompt KV + retained video KV
```

问题 KV 和回答 KV 是临时状态。draft latency backend 也应遵守这个语义。

### 6.2 ViSpec draft text/tree KV

草稿头 text/tree KV 是短生命周期状态：

```text
scope: one profile_vispec_draft_head call
owner: spec_layer
reset before draft
reuse inside topK_genrate
reset after draft
```

它不跨问题、不跨视频、不进入 HERMES 压缩。

### 6.3 ViSpec draft visual context

第一阶段采用 lazy rebuild：

```text
encode chunk:
  save visual source buffer only

compress:
  compact source buffer using source_keep_policy

draft:
  rebuild temporary draft visual context from retained source
```

原因是 ViSpec ImgAdaptor 需要 visual embeddings 和 target hidden states，而不是 target KV tensor。若后续想直接根据 K/V 构建草稿视觉上下文，需要额外训练 `KV -> visual summary` adapter，这已经偏离原版 ViSpec 语义。

### 6.4 Source Buffer 建议结构

可维护：

```python
self.vispec_visual_sources = [
    {
        "chunk_id": int,
        "global_indices": Tensor,       # [n], original visual token indices in HERMES target KV
        "video_features": Tensor,       # [1, n, hidden]
        "target_hidden_states": Tensor, # [1, n, hidden]
        "position_ids_3d": Tensor,      # [3, 1, n]
    }
]
```

每次 HERMES 压缩后，source buffer 根据当前策略做物理 compact，删除不再被策略保留的 token source。为了控制内存，source buffer 可以只保存 CPU bf16/fp16 tensor，并在起草前搬回 GPU。若 latency profile 需要包含视觉 lazy rebuild 开销，应把 H2D 拷贝计入单独字段；若只测草稿头计算，应预先搬到 GPU 后再开始计时。

### 6.5 Source Compact Policy

第一版默认策略：

```text
union_all:
  keep_global = union(keep_indices_all_layers)
  keep_global_visual = keep_global restricted to original visual token indices
```

即保留所有仍被任意一层使用的原始视觉 token source。该策略与 HERMES 的 layer-wise KV 管理兼容，并最大限度避免草稿头视觉上下文弱于目标模型可见的视觉证据。

后续将 compact policy 作为消融变量：

| Policy | 含义 |
|---|---|
| `union_all` | 保留所有层 keep indices 的并集，第一版默认 |
| `shallow` | 只保留浅层区间 keep indices 的并集 |
| `mid` | 只保留中层区间 keep indices 的并集 |
| `deep` | 只保留深层区间 keep indices 的并集 |
| `layer:N` | 只使用第 N 层 keep indices |

层区间复用 HERMES 当前划分：

```text
shallow: [0, short_term_threshold)
mid:     [short_term_threshold, long_term_threshold)
deep:    [long_term_threshold, num_layers)
```

compact 时应排除没有原始视觉 source 的 HERMES summary token，并记录：

```json
{
  "source_keep_policy": "union_all",
  "source_token_count_before_compact": 8120,
  "source_token_count_after_compact": 5340,
  "summary_visible_to_draft": false,
  "summary_token_count_excluded": 12
}
```

## 7. Latency 记录设计

### 7.1 计时边界

推荐记录以下阶段：

| 字段                                   | 含义                                                       |
| -------------------------------------- | ---------------------------------------------------------- |
| `target_prefill_latency_seconds`       | 当前问题目标模型 prefill forward 时延                      |
| `draft_visual_rebuild_latency_seconds` | 起草前视觉 lazy rebuild 时延                               |
| `draft_initial_latency_seconds`        | 第一层 draft token 生成时延                                |
| `draft_expand_latency_seconds`         | 每个 tree depth expansion 时延                             |
| `draft_tree_pack_latency_seconds`      | scores/top_scores/retrieve_indices/tree_mask 打包时延      |
| `draft_head_total_latency_seconds`     | 视觉 rebuild 之后或包含 rebuild 的草稿头总时延，需明确口径 |

建议同时输出两种 total：

```text
draft_head_compute_latency_seconds
draft_head_with_visual_rebuild_latency_seconds
```

这样后续分析时可以区分草稿头核心计算和 lazy visual context 构建成本。

### 7.2 每个 tree depth 层的记录

`topK_genrate()` 内部需要打点：

```text
tree_depth=0:
  hidden_states + input_ids -> first top-k draft tokens

tree_depth=1..depth:
  frontier top-k tokens -> expand top-k * top-k candidates -> select next frontier
```

每条记录建议包含：

```json
{
  "tree_depth": 1,
  "phase": "draft_expand",
  "frontier_tokens": 8,
  "candidate_tokens": 64,
  "selected_frontier_tokens": 8,
  "latency_seconds": 0.003
}
```

注意 `candidate_tokens` 是用于归一化的候选规模，不等于最终被放入 draft tree 的 token 数。最终 tree 中保留多少 token 由 `total_token` 决定。

### 7.3 CUDA 同步

GPU 计时必须处理 CUDA 异步。可选方式：

```python
torch.cuda.synchronize(device)
start = time.perf_counter()
...
torch.cuda.synchronize(device)
latency = time.perf_counter() - start
```

或使用 CUDA event：

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
...
end_event.record()
torch.cuda.synchronize(device)
latency = start_event.elapsed_time(end_event) / 1000
```

第一版建议沿用 HERMES 当前 token timing 的 `torch.cuda.synchronize()` 风格，便于和已有 CSV 口径一致。

### 7.4 CSV 字段

在 `results.csv` 每个问题行中可新增：

```text
vispec_draft_profile
```

值为 JSON 字符串。后续再新增分析脚本将它展开为长表：

```text
video_index
video_id
question_index
tree_depth
phase
frontier_tokens
candidate_tokens
selected_frontier_tokens
latency_seconds
latency_per_candidate_token_seconds
```

## 8. 配置参数

当前实现已经注册两个独立后端：

```text
qwen2.5_vl_3b_vispec_draft_latency
qwen2.5_vl_7b_vispec_draft_latency
```

它们复用 HERMES 原 Qwen2.5-VL 视频 KV 编码、压缩和回答流程；在每次正式回答前额外执行一次 ViSpec 草稿头 profile，并把结果写入 `results.csv` 的 `vispec_draft_profile` 字段。

实现侧已经把 ViSpec 草稿层的最小源码依赖 vendor 到 HERMES 内部：

```text
HERMES/inference/vispec_draft/
  cnets_ours.py
  configs.py
  choices.py
  utils_c.py
  train/qwen2.5_vl_3B_config.json
  train/qwen2.5_vl_7B_config.json
```

因此部署时不再要求 HERMES 同级目录存在完整 `ViSpec/` 源码树。运行时仍需要 ViSpec 草稿头权重，可通过 `--vispec_spec_model_path` 指向本地权重目录，或使用默认 Hugging Face repo 自动下载。

### 8.1 后端参数

以下参数可通过 `video_qa/run_infer.py` 或 `video_qa/hermes_vqa.py` 传入：

| 参数                                       |              默认值 | 含义                                          |
| ------------------------------------------ | ------------------: | --------------------------------------------- |
| `--vispec_spec_model_path`                 | 根据 model 自动推断 | ViSpec 草稿头权重路径      |
| `--vispec_depth`                           |                 `3` | draft tree depth                              |
| `--vispec_top_k`                           |                 `8` | 每层 top-k frontier 宽度                      |
| `--vispec_total_token`                     |                `30` | 最终 draft tree 选择 token 数                 |
| `--vispec_num_q`                           |                 `2` | ImgAdaptor query 数                           |
| `--vispec_temperature`                     |               `0.0` | 第一版只按 greedy profile，非零值会记录 warning |
| `--vispec_profile_visual_rebuild`          |              `true` | 是否记录视觉 lazy rebuild                     |
| `--vispec_include_visual_rebuild_in_total` |             `false` | total latency 是否包含视觉 rebuild            |
| `--vispec_ignore_hermes_summary`           |              `true` | 第一版是否忽略 HERMES long-term summary token |
| `--vispec_source_keep_policy`              |       `union_all` | source buffer compact 策略，可选 `union_all`、`shallow`、`mid`、`deep`、`layer:N` |

`video_qa/run_infer.py` 额外提供 ViSpec profile 展开脚本的控制参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--skip_vispec_draft_timing_analysis` | `false` | 跳过 `eval/analyze_vispec_draft_timings.py` |
| `--vispec_timing_detail_prefix` | `vispec_draft_timings` | draft tree depth 明细 CSV 文件名前缀 |
| `--vispec_timing_summary_prefix` | `vispec_draft_timing_summary` | draft latency summary CSV 文件名前缀 |

默认模型路径：

```text
qwen2.5_vl_3b_vispec_draft_latency:
  base model: models/Qwen2.5-VL-3B-Instruct
  spec model: JLKang/ViSpec-Qwen2.5-VL-3B-Instruct

qwen2.5_vl_7b_vispec_draft_latency:
  base model: models/Qwen2.5-VL-7B-Instruct
  spec model: JLKang/ViSpec-Qwen2.5-VL-7B-Instruct
```

如果 ViSpec 草稿头权重已经下载到本地，建议显式传入：

```bash
--vispec_spec_model_path /path/to/ViSpec-Qwen2.5-VL-3B-Instruct
```

该路径中应包含 `pytorch_model.bin` 或 `model.safetensors`。如果路径不存在，当前实现会尝试从 Hugging Face repo 下载对应权重。

### 8.2 启动方法

推荐从 `HERMES` 仓库根目录启动：

```bash
cd HERMES
```

完整 benchmark 入口示例：

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_3b_vispec_draft_latency \
  --dataset videomme \
  --num_chunks 1 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --debug true \
  --skip_eval
```

7B 后端只需替换 model：

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_7b_vispec_draft_latency \
  --dataset videomme \
  --num_chunks 1 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --debug true \
  --skip_eval
```

常用 profile 参数示例：

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_3b_vispec_draft_latency \
  --dataset videomme \
  --num_chunks 1 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --vispec_depth 3 \
  --vispec_top_k 8 \
  --vispec_total_token 30 \
  --vispec_num_q 2 \
  --vispec_source_keep_policy union_all \
  --vispec_include_visual_rebuild_in_total false \
  --debug true \
  --skip_eval
```

多 GPU chunk 入口示例：

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_3b_vispec_draft_latency \
  --dataset videomme \
  --num_chunks 4 \
  --gpu_ids 0,1,2,3 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --skip_eval
```

`run_infer.py` 会在推理结束后自动合并 chunk CSV，并在 ViSpec latency backend 下自动运行：

```bash
python eval/analyze_vispec_draft_timings.py \
  --results_path results/<model>/<dataset>/fps<sample_fps>-kv<kv_size>/results.csv \
  --output_dir results/<model>/<dataset>/fps<sample_fps>-kv<kv_size>
```

输出包括：

```text
results.csv
vispec_draft_timings_N.csv
vispec_draft_timing_summary_N.csv
```

其中 `results.csv` 每个问题行包含 `vispec_draft_profile` JSON；`vispec_draft_timings_N.csv` 将每个 `tree_depth` 的 latency 展开成长表；`vispec_draft_timing_summary_N.csv` 统计 per-video 和 dataset-level 的平均 latency。

单 worker/debug 入口也可以直接调用 `hermes_vqa.py`：

```bash
python video_qa/hermes_vqa.py \
  --model qwen2.5_vl_3b_vispec_draft_latency \
  --anno_path data/videomme/videomme.json \
  --save_dir results/debug_vispec \
  --num_chunks 1 \
  --chunk_idx 0 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --streaming false \
  --debug true
```

直接调用 `hermes_vqa.py` 时只会生成当前 worker 的 CSV，例如：

```text
results/debug_vispec/1_0.csv
```

如需展开 profile，可手动运行：

```bash
python eval/analyze_vispec_draft_timings.py \
  --results_path results/debug_vispec/1_0.csv \
  --output_dir results/debug_vispec
```

## 9. 实施步骤

### Step 1: 抽取 ViSpec 草稿头加载逻辑

目标：

- 不直接替换 HERMES 当前 Qwen modeling 文件。
- 只复用 ViSpec 的草稿层结构与权重。
- target model 仍使用 HERMES 当前 transformers 版本加载的 Qwen2.5-VL。

执行：

1. 新增草稿头 loader。
2. 根据 base model hidden size / vocab size 检查 3B/7B 配置匹配。
3. 加载 `model.safetensors` 或 `pytorch_model.bin`。
4. 将 spec layer 放到 target `lm_head` 所在 device 或最后 decoder layer device。
5. 初始化 `spec_layer.init_tree()`。

验收：

- 3B/7B spec layer 均可 load。
- missing/unexpected keys 明确打印。
- embedding、lm_head、hidden size 对齐。

### Step 2: 增加 source buffer

目标：

- 记录被 HERMES 视觉记忆管理影响的视觉源信息。
- 支持起草前 lazy rebuild。

执行：

1. 在 Qwen ViSpec latency backend 中维护 `vispec_visual_sources`。
2. `encode_video_chunk()` 中保存 video features、target hidden states、position ids 和 token index range。
3. `predict_and_compress()` 后根据 `--vispec_source_keep_policy` 计算全局 source keep set。
4. 默认 `union_all`：保留所有仍被任意层显式使用的原始视觉 token source。
5. 对 source buffer 做物理 compact，删除不再保留的 embedding / hidden / position。
6. 第一版忽略 HERMES summary token，并记录被排除数量。

验收：

- source buffer token 数随 HERMES 压缩后 compact，不随完整历史视频无限增长。
- `union_all` 下，source buffer 保留 token 等于 layer-wise explicit visual keep indices 的并集。
- `shallow/mid/deep/layer:N` policy 下，source buffer compact 结果与配置策略一致。
- clear cache 后 source buffer 清空。

### Step 3: 实现 lazy visual rebuild

目标：

- 每次起草前构建临时草稿视觉上下文。
- 不跨问题维护草稿视觉 KV。

执行：

1. 收集 retained visual source。
2. 调用 ViSpec ImgAdaptor 或等价封装。
3. 更新 `spec_layer.last_img_hidden` 或临时 visual context。
4. 记录 rebuild latency。

验收：

- 无 retained visual source 时可退化为零视觉上下文。
- retained visual source 存在时 `last_img_hidden` 非空且 shape 正确。
- 视觉 rebuild 不改变 HERMES target KV。

### Step 4: 实现 target question prefill profile

目标：

- 用目标模型真实 forward 提供 hidden states。
- 回答结束后回滚临时问题 KV。

执行：

1. 复用 `question_answering()` 中 Qwen 3D position 构造逻辑。
2. forward 时获取 `out.last_hidden_state`。
3. `lm_head` 得到 target logits。
4. 记录 prefill latency。
5. 保存 `past_lens_prefill` 用于 profile 后 truncate。

验收：

- prefill 后 KV 长度增加问题 token 数。
- profile 后 KV 长度恢复。
- `_position_ids_cache` 同步恢复。

### Step 5: 实现 timed topK draft tree

目标：

- 沿用 ViSpec `topK_genrate()` 语义。
- 增加每个 tree depth 层的 timing。

执行：

1. 在草稿层增加可选 `return_layer_timings` 参数，或新建 wrapper 函数。
2. 对 initial layer、每个 expansion layer、tree packing 分别计时。
3. 返回 draft tree 和 profile JSON。
4. 起草前后 reset `spec_layer.stable_kv`。

验收：

- `draft_tokens`、`retrieve_indices`、`tree_mask`、`tree_position_ids` shape 正常。
- layer timing 数量为 `depth + 1` 或与实现定义一致。
- CUDA 同步口径一致。

### Step 6: 接入结果记录

目标：

- 每个问题记录一份 `vispec_draft_profile`。
- 不影响现有 token timing 字段。

执行：

1. 在 `HermesVQA.analyze_a_video()` 记录 profile JSON。
2. 新增独立分析脚本或扩展 token timing 分析脚本。
3. 生成 draft layer timing 明细 CSV 和 summary CSV。

验收：

- `results.csv` 每个问题行包含合法 JSON。
- 多进程 chunk 合并后仍可按 `video_index/question_index` 展开。
- 空 profile 或异常 profile 有明确错误信息。

## 10. 风险与处理

### 10.1 Transformers 版本差异

ViSpec README 使用 `transformers==4.51.3`，HERMES Qwen 环境当前更高。不要直接替换 HERMES 的 Qwen modeling 文件，否则可能破坏 DynamicCache、Flash Attention、M-RoPE 与 HERMES hooks。

处理策略：

- target model 使用 HERMES 当前加载方式。
- 只抽取 ViSpec 草稿层、权重加载和 tree generation 逻辑。
- 对依赖 ViSpec 自定义 KVCache 的部分做隔离。

### 10.2 HERMES summary token 与 source buffer 不一致

HERMES 深层可能把被驱逐视觉 KV 聚合为 summary token。这个 summary KV 没有直接对应的原始 visual embedding。

第一版策略：

- 草稿头只看 compact policy 保留的原始视觉 source。
- 默认 `union_all`，即所有仍被任意层显式使用的原始视觉 token source。
- 忽略 HERMES summary token。
- 在 profile 中记录 `summary_visible_to_draft=false` 和 `summary_token_count_excluded`。

后续可补：

- source-level mean hidden pseudo token。
- 从 summary KV 训练 `KV -> draft visual summary` adapter。
- 在 HERMES summary 生成时同步生成 draft-visible summary source。

### 10.3 视觉 source buffer 内存

保存每个 chunk 的 video features 与 hidden states 会增加显存或内存。

处理策略：

- dtype 使用 bf16/fp16。
- 每次 compact 后，可选择立即丢弃已驱逐 token 的 source。
- 如果只测当前 compact 后 source，可在压缩后物理 compact source buffer。

### 10.4 计时口径混淆

需要明确区分：

```text
target prefill latency
visual lazy rebuild latency
draft head compute latency
tree packing latency
```

不要把 HERMES 视频编码和压缩开销混入 draft token latency。

## 11. 后续延伸点

### 11.1 完整 speculative decoding

在 latency backend 跑通后，可补：

```text
draft tree
target tree verify
posterior evaluation
accept/reject
target KV update
draft KV update
```

主要难点是 target tree verify 后如何在 HERMES 的 target KV 中接受部分 candidate，并同步 `_position_ids_cache`。这一步需要单独设计 cache gather/copy，不能直接照搬 ViSpec 的预分配 KVCache。

### 11.2 支持 temperature > 0

第一版建议只做 greedy `temperature=0`。后续可接入 ViSpec 的 `prepare_logits_processor()`，支持 top-p、top-k sampling 和 posterior sampling。

### 11.3 让 HERMES summary 对草稿头可见

可尝试三条路线：

1. 忽略 summary，作为 baseline。
2. 使用被驱逐 source hidden 的均值构造 pseudo visual token。
3. 训练一个 adapter，将 HERMES summary KV 映射到 ViSpec draft visual summary。

### 11.4 多轮回答内的 draft KV 复用

第一阶段每次 profile 后清空 draft text/tree KV。完整 speculative decoding 中，可以在同一次回答生成内跨 accept round 复用 draft KV，但回答结束后仍应清空，不能进入长期视频记忆。

### 11.5 分析脚本

新增 `eval/analyze_vispec_draft_timings.py`，输出：

- 每问题 draft total latency。
- 每 tree depth 平均 latency。
- 每 candidate token latency。
- 按视频长度、数据集、kv_size、depth、top-k 的分组统计。
- 与 HERMES 原 token timing 的联合分析。

### 11.6 消融实验

建议消融：

- 无视觉 lazy rebuild，仅用 target question hidden。
- lazy rebuild 但忽略 summary。
- lazy rebuild 加 pseudo summary。
- 不同 `num_q`。
- 不同 `depth/top_k/total_token`。
- 不同 `kv_size`。
- streaming 与 offline reindex 模式。

## 12. 推荐第一版验收标准

第一版不要求提升生成速度，只要求 profile 可靠：

1. Qwen2.5-VL 3B/7B 均可加载对应 ViSpec 草稿头。
2. HERMES 原视频 KV 编码、压缩和问题 KV 回滚语义不变。
3. 每个问题可产生合法 `vispec_draft_profile` JSON。
4. 每个 profile 包含 target prefill、visual rebuild、每个 tree depth、tree packing 和 total latency。
5. 起草前后 HERMES target KV 长度一致。
6. 起草前后草稿头 `stable_kv` 不跨问题残留。
7. 无模型权重环境下，新增分析脚本可用合成 JSON 做展开测试。
8. 有模型权重环境下，至少跑通 debug 模式单视频单问题。
