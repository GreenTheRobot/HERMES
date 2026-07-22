# 完整数据集逐 Token 推理计时设计

## 1. 目标与适用范围

本功能用于在一个完整数据集的推理结束后，将每个实际生成 token 的时延整理为可直接分析的三维长表，并输出每个视频及整个数据集的平均解码时长。

三维坐标定义为：

```text
(video_index, question_index, token_index)
```

三个索引都从 `0` 开始：

- `video_index`：视频在完整标注 JSON 顶层列表中的位置；
- `question_index`：问题在该视频 `conversations` 列表中的位置；
- `token_index`：模型回答中生成 token 的位置。

该坐标不依赖多进程 chunk 的划分。因此，同一个标注文件使用不同 `num_chunks` 运行时，只要样本内容和顺序不变，生成的三维坐标也不变。

本功能同时支持：

- 原版 LLaVA-OneVision HERMES；
- 原版 Qwen2.5-VL HERMES；
- LLaVA-OneVision Sliding Window；
- Qwen2.5-VL Sliding Window。

## 2. 设计理念

### 2.1 HERMES 与 Sliding Window 使用相同计时口径

两个 Sliding Window 后端已经在 `question_answering()` 中保存结构化逐 token 计时。本次对以下原版 HERMES 文件做了最小化修改：

- `inference/llavaov_hermes.py`；
- `inference/qwenvl_hermes.py`。

修改只涉及计时和元数据初始化，不改变：

- local/global/mixed 伪查询；
- 注意力评分；
- 分层 KV 预算；
- Top-K 选择；
- 长期摘要 token；
- RoPE/M-RoPE 修正；
- 回答采样策略和最终回答内容。

两个 HERMES 后端和两个 Sliding Window 后端现在都通过：

```python
model.last_token_inference_times
```

暴露当前问题的计时列表。每次回答开始前该列表都会清空，避免不同问题的数据混合。

### 2.2 区分首 Token 与增量解码

每个计时元素包含：

```json
{
  "token_index": 2,
  "token_id": 12345,
  "phase": "decode",
  "latency_seconds": 0.038214527
}
```

`phase` 有两种取值：

- `prefill`：`token_index == 0`，表示从问题 prefill 开始到首 token 可用为止的 TTFT；
- `decode`：`token_index >= 1`，表示产生该 token 所依赖的单 token 增量前向时长。

每视频平均值和数据集平均值只使用 `phase=decode`，首 token 的 `prefill` 时长仍保留在明细 CSV 中，方便单独分析 TTFT。

### 2.3 CUDA 同步保证计时有效

CUDA kernel 默认异步执行。如果只在 Python 中包围 `time.perf_counter()`，结束时间可能只反映 kernel 提交时间，而不是 GPU 实际完成时间。

因此，当输入 tensor 位于 CUDA 设备时，四个后端都在计时边界执行：

```python
torch.cuda.synchronize(device)
```

CPU 推理不调用 CUDA 同步。后续 decode 时延只包围模型单 token 前向；日志输出、最终 Tokenizer 解码和下一 token 的采样不计入该时延。Qwen 的 `lm_head` 也包含在计时范围中，从而使计时结束点对应“下一个 token 的 logits 已经可用”。

### 2.4 不把 HERMES 压缩开销混入 Token 解码时延

本功能测量的是回答生成阶段的 token 时延。HERMES 在视频块编码后执行的伪查询、注意力打分和 KV 压缩不包含在 `decode` 时延中。

因此该结果适合回答：

- 当前 KV 状态下，每个回答 token 的增量生成有多快；
- HERMES 与 Sliding Window 的 KV 结构对 decode latency 有何影响。

它不能单独表示整个方法的端到端成本。如果要比较完整系统开销，还需要另行记录视频编码、压缩和整体运行时间。

## 3. 数据如何跨多进程保持全局编号

`video_qa/base.py` 在切分数据集之前，先为每个视频的浅拷贝写入内部字段：

```text
_dataset_video_index
```

之后才调用 `get_chunk()`。这保证每个子进程拿到的是完整数据集中的全局视频编号，而不是 chunk 内从 0 开始的局部编号。

`video_qa/hermes_vqa.py` 使用 `enumerate(video_sample['conversations'])` 得到问题编号，并把以下字段写入每条问答结果：

- `video_index`；
- `question_index`；
- `token_inference_times`。

原有 chunk 结果仍按照项目既有方式合并成一个 `results.csv`。计时分析只在完整 `results.csv` 形成后运行，因此不会产生多个进程争抢最终计时文件的问题。

## 4. 输出文件自动编号

每次分析会生成一对具有相同序号的文件：

```text
token_timings_1.csv
token_timing_summary_1.csv
```

下次运行时从 `1` 开始查找两个文件名均未占用的最小序号。例如：

```text
token_timings_1.csv              已存在
token_timing_summary_1.csv       已存在
token_timings_2.csv              不存在
token_timing_summary_2.csv       不存在
token_timings_3.csv              已存在
```

此时选择 `_2`，而不是简单使用最大序号加一。

若某个序号只有明细文件或只有汇总文件，也视为已占用，防止新的一对输出和旧文件错误配对。脚本还会为选中的序号创建原子锁文件；并发启动两个分析进程时，只有一个进程能取得该序号。正常结束或 Python 异常时锁文件都会移除。

默认前缀可以通过命令行修改，但两个文件仍共享同一个 `_N`。

## 5. Token 明细 CSV

默认文件名：

```text
token_timings_N.csv
```

每个实际输出 token 占一行，按三维坐标排序：

| 字段 | 类型 | 含义 |
|---|---|---|
| `video_index` | int | 视频在完整数据集中的 0-based 序号 |
| `video_id` | string | 标注中的原始视频 ID，便于人工核对 |
| `question_index` | int | 问题在当前视频中的 0-based 序号 |
| `token_index` | int | Token 在当前回答中的 0-based 序号 |
| `token_id` | int | Tokenizer 输出的 token ID |
| `phase` | string | `prefill` 或 `decode` |
| `latency_seconds` | float | 该 token 对应阶段的秒数 |

示例：

```csv
video_index,video_id,question_index,token_index,token_id,phase,latency_seconds
0,video_001,0,0,198,prefill,0.311828
0,video_001,0,1,892,decode,0.039117
0,video_001,0,2,13,decode,0.038924
0,video_001,1,0,7,prefill,0.287613
1,video_002,0,0,301,prefill,0.421107
```

脚本会检查 `(video_index, question_index, token_index)` 是否重复。发现重复坐标、非法 JSON、负数索引、未知 phase、负数或非有限时延时会立即报错，不会静默生成不可靠统计。

## 6. 汇总 CSV

默认文件名：

```text
token_timing_summary_N.csv
```

字段如下：

| 字段 | 含义 |
|---|---|
| `scope` | `video`、`dataset_token_weighted` 或 `dataset_video_macro` |
| `video_index` | 视频级行的全局序号；数据集级行为空 |
| `video_id` | 视频级行的 ID；数据集级行是 `ALL` |
| `video_count` | 当前统计覆盖的视频数 |
| `decode_token_count` | 参与统计的 decode token 数 |
| `mean_decode_latency_seconds` | 平均 decode 时延，单位秒 |
| `mean_decode_latency_ms` | 同一个平均值换算为毫秒 |

汇总包含三类行。

### 6.1 每视频平均值

`scope=video`，计算该视频所有问题中全部 `phase=decode` token 的算术平均值。某个视频若只生成了首 token，没有 decode token，则：

- `decode_token_count=0`；
- 平均值为空；
- 控制台显示 `N/A`。

### 6.2 Token 加权数据集总平均

`scope=dataset_token_weighted`，公式为：

```text
数据集中所有 decode token 的时延之和
──────────────────────────────────
数据集中全部 decode token 数
```

这是默认意义上的“总平均时长”。回答较长的视频会按照实际 token 数贡献更多样本。

### 6.3 视频宏平均

`scope=dataset_video_macro`，先计算每视频平均值，再对具有至少一个 decode token 的视频平均：

```text
各个有效视频的平均 decode 时延之和
────────────────────────────────
具有 decode token 的视频数
```

该指标让每个视频具有相同权重，可用于观察数据集是否被少数长回答主导。没有 decode token 的视频不参与宏平均，但仍保留自己的 `scope=video` 行。

## 7. 自动运行方式

正常运行完整数据集时，`video_qa/run_infer.py` 会在所有 chunk 合并为 `results.csv` 后自动调用计时分析脚本：

```bash
python video_qa/run_infer.py \
  --model llava_ov_7b \
  --dataset streamingbench \
  --num_chunks 8 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --debug false
```

Sliding Window 示例：

```bash
python video_qa/run_infer.py \
  --model llava_ov_7b_slidingwindow \
  --dataset streamingbench \
  --num_chunks 8 \
  --sample_fps 0.5 \
  --kv_size 6000 \
  --debug false
```

输出位于原结果目录，例如：

```text
results/llava_ov_7b/streamingbench/fps0.5-kv6000/
├── results.csv
├── token_timings_1.csv
└── token_timing_summary_1.csv
```

生成 CSV 后，控制台会依次打印：

- 每个视频的平均 decode 时延；
- 数据集 token 加权总平均；
- 数据集视频宏平均；
- 两个实际输出路径。

## 8. `run_infer.py` 新增命令行参数

### `--skip_token_timing_analysis`

不生成明细和汇总 CSV。适用于：

- 对旧版、不包含计时字段的 `results.csv` 执行 `--only_eval`；
- 临时只需要原有准确率或开放问答评测；
- 已经生成计时文件，不希望本次再产生新序号。

示例：

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_7b \
  --dataset videomme \
  --only_eval \
  --kv_size 6000 \
  --skip_token_timing_analysis
```

### `--timing_detail_prefix`

设置 token 明细文件前缀，默认值为 `token_timings`。

### `--timing_summary_prefix`

设置汇总文件前缀，默认值为 `token_timing_summary`。

例如：

```bash
python video_qa/run_infer.py \
  --model qwen2.5_vl_7b_slidingwindow \
  --dataset mvbench \
  --num_chunks 4 \
  --kv_size 6000 \
  --timing_detail_prefix qwen_sw_tokens \
  --timing_summary_prefix qwen_sw_summary
```

第一次运行会生成：

```text
qwen_sw_tokens_1.csv
qwen_sw_summary_1.csv
```

文件前缀只能是文件名，不能包含目录部分。输出目录由当前实验的 `save_dir` 决定。

## 9. 独立分析脚本

核心脚本为：

```text
eval/analyze_token_timings.py
```

可以对已经完成合并的结果单独运行：

```bash
python eval/analyze_token_timings.py \
  --results_path results/llava_ov_7b/streamingbench/fps0.5-kv6000/results.csv
```

### 命令行参数

| 参数 | 必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--results_path` | 是 | 无 | 一个完整数据集已合并的 `results.csv` |
| `--output_dir` | 否 | `results_path` 所在目录 | 明细和汇总文件输出目录 |
| `--detail_prefix` | 否 | `token_timings` | 明细文件前缀 |
| `--summary_prefix` | 否 | `token_timing_summary` | 汇总文件前缀 |

指定输出目录和自定义前缀：

```bash
python eval/analyze_token_timings.py \
  --results_path results/qwen2.5_vl_7b/videomme/fps1-kv6000/results.csv \
  --output_dir analysis/qwen_videomme \
  --detail_prefix hermes_tokens \
  --summary_prefix hermes_summary
```

重复运行同一命令不会覆盖旧文件，而是依次寻找 `_1`、`_2` 等最小空闲序号。

## 10. 使用 Pandas 做进一步分析

读取三维明细并计算问题级平均值：

```python
import pandas as pd

timings = pd.read_csv("token_timings_1.csv")
decode = timings[timings["phase"] == "decode"]

question_means = (
    decode.groupby(["video_index", "question_index"], as_index=False)
    ["latency_seconds"]
    .mean()
)
print(question_means)
```

比较两个方法时，建议先用三维坐标对齐：

```python
import pandas as pd

hermes = pd.read_csv("hermes/token_timings_1.csv")
sliding = pd.read_csv("sliding/token_timings_1.csv")

keys = ["video_index", "question_index", "token_index"]
paired = hermes.merge(
    sliding,
    on=keys,
    suffixes=("_hermes", "_sliding"),
)
paired = paired[
    (paired["phase_hermes"] == "decode")
    & (paired["phase_sliding"] == "decode")
]
paired["latency_delta_seconds"] = (
    paired["latency_seconds_hermes"]
    - paired["latency_seconds_sliding"]
)
```

不同方法可能生成不同文本或不同 token 数，因此 merge 后的行数可能少于任一原始文件。这种情况下应同时报告各自的非配对总体平均和成功对齐部分的配对平均。

## 11. 边界情况与失败行为

- `results.csv` 缺少 `video_index`、`question_index` 或 `token_inference_times` 时，脚本明确报错；
- `token_inference_times` 为空时，该问题不产生明细行，但视频仍出现在汇总中；
- 只有首 token 时没有 decode 样本，每视频平均为 `N/A`；
- 输出文件使用 exclusive create，不覆盖已有文件；
- 如果明细已成功创建、汇总写入失败，该序号会因为明细存在而保持占用，避免下次运行错误配对；
- `--only_eval` 默认也会重新分析现有结果并产生下一个 `_N`；若不需要，应添加 `--skip_token_timing_analysis`；
- `--debug true` 只处理每个 chunk 的首个样本，不代表完整数据集计时，正式统计应使用 `--debug false`。

## 12. 无模型权重环境下的验证

没有模型权重时无法验证真实 CUDA 数值，但可以完成：

1. 所有修改文件的 Python 编译检查；
2. HERMES 与 Sliding Window 计时字段、phase 和同步边界的静态一致性检查；
3. 使用合成 `results.csv` 验证三维展开；
4. 验证乱序输入会按三维坐标重新排序；
5. 验证每视频平均、token 加权总平均和视频宏平均；
6. 验证已有 `_1`、空闲 `_2`、已有 `_3` 时会选择 `_2`；
7. 验证重复三维坐标会被拒绝；
8. 验证分析脚本和 `run_infer.py` 的 `--help` 参数可正常解析。

获得模型权重后，建议额外确认：

- 每个回答生成的 token 数与 `last_token_inference_times` 长度一致；
- CUDA 计时均为有限非负数；
- HERMES 与 Sliding Window 在相同模型、问题和生成参数下使用相同计时口径；
- 计时日志、`results.csv` 内 JSON、最终 `token_timings_N.csv` 三者的 token 序号一致。
