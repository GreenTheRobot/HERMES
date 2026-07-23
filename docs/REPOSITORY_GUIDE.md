# HERMES 仓库结构与核心入口

本文档面向准备阅读、复现或修改 HERMES 的开发者，说明仓库中各目录的职责、从启动脚本到 KV Cache 压缩的完整调用链，以及常见改动应从哪里进入。

HERMES 是一个无需训练的流式视频理解框架。它以 LLaVA-OneVision 或 Qwen2.5-VL 为基础模型，将历史视频的 KV Cache 作为常驻 GPU 的分层记忆，并在视频流到达期间持续压缩。用户真正提问时，模型直接复用已经维护好的 KV Cache，不再检索或重新预填充历史视频。

## 1. 仓库结构

```text
HERMES/
├── asset/                         # README 使用的图片资源
├── data/                          # 各评测集的 JSON 标注；视频文件需另外下载
│   ├── egoschema/
│   ├── mvbench/
│   ├── ovobench/
│   ├── rvs/
│   ├── streamingbench/
│   └── videomme/
├── docs/                          # 项目开发文档
├── eval/
│   ├── eval_multiple_choice.py    # 选择题评测
│   └── eval_open_ended.py         # 使用外部语言模型评测开放式回答
├── inference/
│   ├── abstract_hermes.py         # 两个模型后端共享的最小状态与接口
│   ├── llavaov_hermes.py          # LLaVA-OneVision 的 HERMES 实现
│   ├── qwenvl_hermes.py           # Qwen2.5-VL 的 HERMES 实现
│   ├── reindex_1d.py              # LLaVA 1D RoPE 位置重索引
│   └── reindex_3d.py              # Qwen 3D M-RoPE 位置重索引
├── scripts/
│   └── run_infer.sh               # 推荐的命令行启动脚本
├── video_qa/
│   ├── base.py                    # 模型注册、视频采样、QA 格式和通用运行逻辑
│   ├── hermes_vqa.py              # 视频级流式/离线推理循环
│   └── run_infer.py               # 数据集配置、多进程调度、结果合并与评测调度
├── requirements_llava.txt         # LLaVA 后端依赖
├── requirements_qwen.txt          # Qwen 后端依赖
└── README.md
```

仓库不包含训练代码。主要可修改内容都位于推理阶段，包括视频分块、伪查询、token 重要性打分、层间平滑、Top-K 淘汰、摘要 token 和位置重索引。

## 2. 顶层运行入口

### 2.1 推荐入口：`scripts/run_infer.sh`

通常从 [`scripts/run_infer.sh`](../scripts/run_infer.sh) 启动实验：

```bash
bash scripts/run_infer.sh
```

脚本最终执行：

```bash
python video_qa/run_infer.py \
    --num_chunks 8 \
    --model llava_ov_7b \
    --dataset streamingbench \
    --sample_fps 0.5 \
    --kv_size 6000
```

主要参数：

| 参数 | 含义 |
|---|---|
| `model` | 模型后端与规模，定义在 `video_qa/base.py` 的 `MODELS` 中 |
| `dataset` | 数据集名称，定义在 `video_qa/run_infer.py` 的 `BENCHMARK_CONFIGS` 中 |
| `num_chunks` | 数据分片数量，通常与可用 GPU 数量一致 |
| `sample_fps` | 从原视频采样的帧率 |
| `kv_size` | 每层最多保留的视频 KV token 数量，不含固定系统提示前缀 |
| `gpu_ids` | 可选。逗号分隔的物理 GPU 编号，用于覆盖默认分配，例如 `1,3` |
| `only_eval` | 跳过推理，直接评测已有结果 |

### 2.2 实验调度入口：`video_qa/run_infer.py`

[`video_qa/run_infer.py`](../video_qa/run_infer.py) 负责：

1. 根据 `dataset` 查找标注路径、流式/离线模式和评测命令。
2. 将标注数据拆成 `num_chunks` 份。
3. 为每个分片启动一个 `video_qa/hermes_vqa.py` 子进程。
4. 为子进程设置 `CUDA_VISIBLE_DEVICES`。
5. 合并各分片生成的 CSV。
6. 调用 `eval/` 下对应的评测脚本。

新增数据集时，需要首先修改该文件中的 `BENCHMARK_CONFIGS`。新增模型名称时，还要同步修改这里的命令行 `choices` 和 `video_qa/base.py` 中的 `MODELS`。

当前结果目录格式为：

```text
results/{model}/{dataset}/fps{sample_fps}-kv{kv_size}/results.csv
```

### 2.3 GPU 设置规则

`video_qa/run_infer.py` 会为每个数据分片启动一个子进程，并在子进程环境里设置
`CUDA_VISIBLE_DEVICES`。当前实现的规则如下：

- 普通模型（如 `llava_ov_0.5b`、`llava_ov_7b`、`qwen2.5_vl_7b`）：第 `idx`
  个分片使用 `CUDA_VISIBLE_DEVICES=str(idx)`。
- 72B 模型（`llava_ov_72b` 和 `llava_ov_72b_slidingwindow`）：第 `idx`
  个分片使用 4 张卡，即 `4*idx,4*idx+1,4*idx+2,4*idx+3`。
- 如果传入 `--gpu_ids`，则按该列表覆盖默认分配。例如普通模型下
  `--num_chunks 2 --gpu_ids 1,3` 会让 chunk 0 使用物理 GPU 1，chunk 1
  使用物理 GPU 3。
- 因为子进程会重新赋值 `CUDA_VISIBLE_DEVICES`，所以应优先使用 `--gpu_ids`
  指定非连续 GPU，而不是只在外层命令里写 `export CUDA_VISIBLE_DEVICES=1,3`。

因此，默认启动方式适合使用从 0 开始连续编号的 GPU。例如使用物理 GPU 0 和 1：

```bash
python video_qa/run_infer.py \
    --num_chunks 2 \
    --model llava_ov_0.5b \
    --dataset rvs_ego \
    --sample_fps 0.5 \
    --kv_size 6000 \
    --debug false \
    --skip_eval
```

如果要指定非连续 GPU，例如只使用物理 GPU 1 和 3：

```bash
python video_qa/run_infer.py \
    --num_chunks 2 \
    --gpu_ids 1,3 \
    --model llava_ov_0.5b \
    --dataset rvs_ego \
    --sample_fps 0.5 \
    --kv_size 6000 \
    --debug false \
    --skip_eval
```

72B 模型每个 chunk 需要 4 张 GPU，因此 `--num_chunks 2 --gpu_ids 0,1,2,3,4,5,6,7`
会让 chunk 0 使用 `0,1,2,3`，chunk 1 使用 `4,5,6,7`。

## 3. 主调用链

一次完整实验的调用关系如下：

```text
scripts/run_infer.sh
└── video_qa/run_infer.py
    └── video_qa/hermes_vqa.py
        └── video_qa/base.py::work(HermesVQA)
            ├── inference/*_hermes.py::load_model()
            └── HermesVQA.analyze()
                └── HermesVQA.analyze_a_video()
                    ├── clear_cache()
                    ├── encode_init_prompt()
                    ├── encode_video_chunk()
                    ├── predict_and_compress()
                    │   ├── predict_next_question()
                    │   └── pseudo_forward()
                    │       ├── 计算 local/global/mixed 注意力
                    │       ├── prune_kv_cache_by_attention()
                    │       └── apply_kv_cache_pruning_strict()
                    └── question_answering()
```

理解或修改 HERMES 时，最重要的是 `HermesVQA.analyze_a_video()`、模型后端的 `predict_and_compress()`，以及 `prune_kv_cache_by_attention()` 这三个层级。

## 4. 视频级入口：`HermesVQA.analyze_a_video`

[`video_qa/hermes_vqa.py`](../video_qa/hermes_vqa.py) 中的 `HermesVQA.analyze_a_video()` 是连接数据和模型的核心入口。

其流程如下：

1. 根据标注读取视频、`.npy` 文件或图片帧目录。
2. 按 `sample_fps` 采样并转换为视频 tensor。
3. 对每个新视频调用 `clear_cache()` 和 `encode_init_prompt()`。
4. 默认每 16 帧组成一个视频块。
5. 每个视频块依次调用：
   - `encode_video_chunk(video_chunk)`：视觉编码并写入 KV Cache；
   - `predict_and_compress()`：使用伪查询计算重要性，必要时压缩 KV Cache。
6. 到达问题对应时间后，调用选择题或开放题回答接口。
7. 将预测结果追加到当前分片的 CSV 记录。

流式与离线数据使用同一个循环：

- 对话项包含 `end_time` 时，只编码到该时间点，模拟用户在视频播放期间提问。
- 不包含 `end_time` 时，先编码完整段视频再回答，作为离线模式。

若要修改视频块大小、压缩触发频率或场景变化检测，首先从这个函数进入。

## 5. 通用数据和模型入口：`video_qa/base.py`

[`video_qa/base.py`](../video_qa/base.py) 包含四类公共逻辑。

### 5.1 模型注册

`MODELS` 将命令行模型名映射到：

- 后端 `load_model()` 函数；
- 本地模型权重路径。

当前支持：

- `llava_ov_0.5b`
- `llava_ov_7b`
- `llava_ov_72b`
- `qwen2.5_vl_3b`
- `qwen2.5_vl_7b`
- `qwen2.5_vl_32b`

论文中的 Qwen3-VL 尚未在这个仓库入口中实现。

### 5.2 视频读取

- `load_video()`：读取普通视频或 `.npy`。
- `load_video_frames()`：读取按图片帧保存的视频。

这两个函数负责按 `sample_fps` 采样。若要接入摄像头、RTSP 或在线帧队列，可以保留 `HermesVQA` 的后续逻辑，仅替换这里的数据源。

### 5.3 问答接口

- `video_open_qa()`：开放式回答。
- `video_close_qa()`：格式化多选题并抽取选项字母。
- `pseudo_qa()`：旧的通用伪问答接口；当前主流程直接调用模型的 `predict_and_compress()`。

### 5.4 命令行工作入口

`work(QA_CLASS)` 负责解析子进程参数、加载模型和标注，并创建 `HermesVQA` 实例。`video_qa/hermes_vqa.py` 在模块末尾调用：

```python
work(HermesVQA)
```

## 6. 模型后端入口

### 6.1 公共状态：`Abstract_Hermes`

[`inference/abstract_hermes.py`](../inference/abstract_hermes.py) 保存：

- `kv_cache`：当前视频记忆；
- `kv_size`：视频 token 预算；
- `visual_start_idx`：固定系统提示之后，视频 token 开始的位置；
- `conv_history`：历史问题和回答，用于生成下一轮压缩提示。

主要公共方法：

- `clear_cache()`：开始新视频前清理状态；
- `encode_init_prompt()`：将固定系统提示写入 KV，并确定 `visual_start_idx`；
- `get_prompt()`：构造真实用户问题的文本模板。

### 6.2 LLaVA-OneVision

[`inference/llavaov_hermes.py`](../inference/llavaov_hermes.py) 是最适合首先阅读的实现，因为它使用标准 1D RoPE，逻辑相对直接。

关键函数：

| 函数 | 职责 |
|---|---|
| `load_model()` | 加载基础模型，并挂接 HERMES 状态、位置缓存和 forward hook |
| `encode_video_chunk()` | 调用视觉编码器和 projector，将视频 token 预填充到语言模型 KV |
| `predict_next_question()` | 根据最近一轮对话生成 local/global 伪查询 |
| `pseudo_forward()` | 分别计算 local、global、mixed 三组查询注意力 |
| `_compute_attention_scores_manually()` | Flash Attention/SDPA 下手工计算查询对历史 KV 的注意力 |
| `prune_kv_cache_by_attention()` | 分层打分、跨层平滑、预算分配和 Top-K 选择 |
| `apply_kv_cache_pruning_strict()` | 执行真正的 KV 裁剪 |
| `_shrink_positions_and_rerotate_keys()` | 位置重索引、Key 相位修正和深层摘要 token |
| `question_answering()` | 复用视频 KV 生成回答，结束后回滚临时文本 KV |

### 6.3 Qwen2.5-VL

[`inference/qwenvl_hermes.py`](../inference/qwenvl_hermes.py) 与 LLaVA 后端保持类似接口，但需要额外维护 Qwen 的 3D M-RoPE：

- 时间坐标；
- 高度坐标；
- 宽度坐标。

修改通用打分策略时，通常需要同步修改两个后端。修改位置编码时则应分别处理，不能直接复制 LLaVA 的 1D 实现。

## 7. KV 压缩入口

核心算法集中在两个后端同名的 `prune_kv_cache_by_attention()` 中。

### 7.1 层划分

官方 `load_model()` 最终设置：

- 前 10% 层：short-term / sensory memory；
- 中间 60% 层：mid-term / working memory；
- 后 30% 层：long-term memory。

注意：类构造函数中还保留了 30% 浅层的默认值，但官方加载入口会覆盖为 10%。增加配置系统时应消除这组重复默认值。

### 7.2 分层分数

- 浅层：只依赖指数近期性，优先保留最新 token。
- 中层：融合近期性和 mixed 伪查询注意力。
- 深层：依赖 global 伪查询注意力。

打分后进行从深层到浅层的相邻层平滑，再对每层独立 Top-K。

### 7.3 预算

`allocate_budget_by_depth()` 当前实际上是均匀分配：每层视频预算都约为 `kv_size`。深层会预留一个位置给摘要 token。

如果要实现动态层预算，应修改：

1. `allocate_budget_by_depth()`；
2. `prune_kv_cache_by_attention()` 中深层预留摘要位置的逻辑；
3. 结果目录或日志，使实验能够记录实际预算配置。

### 7.4 摘要 token

深层被淘汰的 token 会聚合为一个摘要：

- Value 直接求均值；
- Key 先旋转到统一目标位置，再求均值。

若要实现多摘要、聚类摘要或事件摘要，应从 `_shrink_positions_and_rerotate_keys()` 进入，并保证最终 token 数量不超过预算。

## 8. 位置重索引入口

压缩会造成 KV 物理索引与逻辑位置不连续。两个后端通过每层 `_position_ids_cache` 保存逻辑位置，并在必要时修正缓存 Key。

- [`inference/reindex_1d.py`](../inference/reindex_1d.py)：标准 1D RoPE。
- [`inference/reindex_3d.py`](../inference/reindex_3d.py)：Qwen 3D M-RoPE。

运行模式由数据集配置传入：

- `streaming=True`：lazy re-index，接近最大位置范围时才压紧；
- `streaming=False`：eager re-index，每次压缩都压紧。

位置修正属于高风险改动。至少需要验证：

- 固定系统前缀不被删除；
- token 时间顺序保持不变；
- 每层 KV 长度与位置缓存长度一致；
- 重索引前后，同一查询的注意力结果在数值误差范围内一致；
- Qwen 的三个坐标维度和 `mrope_section` 一致。

## 9. 回答阶段的 KV 生命周期

`question_answering()` 不会把真实问题永久写入视频记忆：

1. 保存回答前各层 KV 长度。
2. 将问题 token 临时预填充到当前视频 KV。
3. 自回归生成答案。
4. 将问答文本写入 Python 侧的 `conv_history`。
5. 把 KV 截断回回答前长度。

因此，后续视频只保留视觉 KV；历史对话通过伪查询文本影响下一次压缩，而不是作为永久文本 KV 保留。

修改多轮对话行为时，应同时检查：

- `Abstract_Hermes.conv_history`；
- `predict_next_question()`；
- `question_answering()` 末尾的 KV 回滚。

## 10. 数据和评测入口

### 10.1 标注

`data/` 当前包含 JSON 标注，但视频需要从各数据集官方来源下载。标注中的 `video_path` 必须与本地实际位置一致。

单个视频样本至少需要：

```json
{
  "video_id": "example",
  "video_path": "/path/to/video.mp4",
  "conversations": [
    {
      "question": "...",
      "answer": "...",
      "choices": ["...", "..."],
      "end_time": 12.5
    }
  ]
}
```

其中 `choices` 和 `end_time` 均为可选字段。

### 10.2 评测

[`eval/eval_multiple_choice.py`](../eval/eval_multiple_choice.py) 提供：

- `general`：总体准确率和任务拆分；
- `videomme`：按视频长度统计；
- `egoschema`：生成提交文件。

[`eval/eval_open_ended.py`](../eval/eval_open_ended.py) 使用外部语言模型评分。当前文件中的 `base_url` 和 `api_key` 是空占位符，运行前需要自行配置；不应将真实密钥提交到仓库。

## 11. 常见改动对应入口

| 目标 | 首要修改位置 |
|---|---|
| 修改视频分块大小 | `HermesVQA.analyze_a_video()` |
| 改成按场景变化触发压缩 | `HermesVQA.analyze_a_video()`、`predict_and_compress()` |
| 修改 local/global 提示 | 两个后端的 `predict_next_question()` |
| 修改近期性衰减 | 两个后端的 `prune_kv_cache_by_attention()` |
| 修改层划分 | 两个后端的 `load_model()`，建议抽为统一配置 |
| 动态分配每层预算 | `allocate_budget_by_depth()` |
| 修改跨层平滑 | `prune_kv_cache_by_attention()` 的 `refined_scores` 循环 |
| 帧级或事件级选择 | `prune_kv_cache_by_attention()` 的 Top-K 前后 |
| 多摘要 token | `_shrink_positions_and_rerotate_keys()` |
| 修改 lazy/eager 重索引 | `_shrink_positions_and_rerotate_keys()` 和 `reindex_*.py` |
| 接入新模型后端 | 新增 `inference/*_hermes.py`，并更新 `MODELS` 和 CLI choices |
| 接入新数据集 | `BENCHMARK_CONFIGS` 和相应评测函数 |

## 12. 开发注意事项

1. 两个模型后端存在较多重复实现。通用算法改动必须同步，否则容易产生实验不可比的参数漂移。
2. 当前每个视频块都会运行 local、global、mixed 三组伪查询，即使 KV 尚未超过预算。这部分是降低流式摄取开销的重要优化点。
3. LLaVA 中层近期性权重的实现与论文公式并不完全一致；进行算法改动前，应先将其作为显式配置并记录实际值。
4. `token_activity_cache` 会累计注意力，但当前 Top-K 分数仍使用本轮注意力；若要使用历史活跃度，需要明确融合方式。
5. 模型加载通过复制基础模型实例状态并注册 forward hook 实现，依赖特定 Transformers 内部结构。升级依赖后应首先运行小模型 smoke test。
6. `video_qa/run_infer.py` 使用 Bash/POSIX 命令合并和删除结果文件，默认运行环境是 Linux。
7. 当前仓库没有测试目录。位置重索引、预算边界和摘要 token 是最应优先补测试的模块。

## 13. 推荐阅读顺序

第一次阅读代码时，建议按以下顺序：

1. `scripts/run_infer.sh`
2. `video_qa/run_infer.py`
3. `video_qa/base.py` 中的 `work()` 和 `MODELS`
4. `video_qa/hermes_vqa.py` 中的 `analyze_a_video()`
5. `inference/abstract_hermes.py`
6. `inference/llavaov_hermes.py` 中的：
   - `encode_video_chunk()`
   - `predict_and_compress()`
   - `pseudo_forward()`
   - `prune_kv_cache_by_attention()`
   - `_shrink_positions_and_rerotate_keys()`
   - `question_answering()`
7. `inference/reindex_1d.py`
8. 对照阅读 `qwenvl_hermes.py` 和 `reindex_3d.py`

完成上述路径后，就能覆盖从视频输入、KV 增长、压缩、位置修正到用户回答的完整生命周期。
