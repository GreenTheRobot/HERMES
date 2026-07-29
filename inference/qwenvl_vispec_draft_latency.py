import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from logzero import logger

from inference.qwenvl_hermes import (
    QwenVL_Hermes,
    get_qwen2_5_vl_position_ids,
    load_model as load_qwenvl_hermes_model,
)
from inference.reindex_3d import contiguous_kv, get_cache_seq_len
from inference.vispec_draft.cnets_ours import Model as ViSpecDraftModel
from inference.vispec_draft.configs import EConfig


_VENDORED_VISPEC_ROOT = Path(__file__).resolve().parent / "vispec_draft"
_HERMES_ROOT = Path(__file__).resolve().parents[1]


def _cuda_sync_if_needed(device):
    device = torch.device(device)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def _default_spec_model_path(model_path):
    model_path_lower = str(model_path).lower()
    if "3b" in model_path_lower:
        local_path = _HERMES_ROOT / "models" / "ViSpec-Qwen2.5-VL-3B-Instruct"
        if local_path.exists():
            return str(local_path)
        return "JLKang/ViSpec-Qwen2.5-VL-3B-Instruct"
    if "7b" in model_path_lower:
        local_path = _HERMES_ROOT / "models" / "ViSpec-Qwen2.5-VL-7B-Instruct"
        if local_path.exists():
            return str(local_path)
        return "JLKang/ViSpec-Qwen2.5-VL-7B-Instruct"
    raise ValueError(
        "Cannot infer ViSpec draft head path for this base model. "
        "Pass --vispec_spec_model_path explicitly."
    )


def _local_train_config_for_hidden_size(hidden_size):
    if hidden_size == 2048:
        return _VENDORED_VISPEC_ROOT / "train" / "qwen2.5_vl_3B_config.json"
    if hidden_size == 3584:
        return _VENDORED_VISPEC_ROOT / "train" / "qwen2.5_vl_7B_config.json"
    raise ValueError(f"Unsupported Qwen2.5-VL hidden size for ViSpec: {hidden_size}")


def _hf_download(repo_id, filename):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required to download ViSpec draft head files. "
            "Install it or pass a local --vispec_spec_model_path."
        ) from exc
    return Path(hf_hub_download(repo_id, filename))


def _resolve_config_path(spec_model_path, hidden_size):
    path = Path(spec_model_path)
    if path.is_file():
        return path
    if path.exists():
        candidate = path / "config.json"
        if candidate.exists():
            return candidate
        return _local_train_config_for_hidden_size(hidden_size)
    fallback = _local_train_config_for_hidden_size(hidden_size)
    logger.info(
        "Using vendored ViSpec architecture config %s for draft weights %s",
        fallback,
        spec_model_path,
    )
    return fallback


def _load_spec_state_dict(spec_model_path):
    path = Path(spec_model_path)
    if path.is_file():
        candidates = [path]
    elif path.exists():
        candidates = [path / "pytorch_model.bin", path / "model.safetensors"]
    else:
        candidates = []
        try:
            candidates.append(_hf_download(spec_model_path, "pytorch_model.bin"))
        except Exception as bin_exc:
            try:
                candidates.append(_hf_download(spec_model_path, "model.safetensors"))
            except Exception as safe_exc:
                raise FileNotFoundError(
                    f"Could not locate ViSpec draft head weights for "
                    f"{spec_model_path!r}. Tried pytorch_model.bin and "
                    f"model.safetensors."
                ) from safe_exc

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(candidate), device="cpu")
        return torch.load(candidate, map_location="cpu")

    raise FileNotFoundError(
        f"Could not locate pytorch_model.bin or model.safetensors in {spec_model_path!r}"
    )


class QwenVL_ViSpecDraftLatency(QwenVL_Hermes):
    """Qwen2.5-VL HERMES backend with one-shot ViSpec draft-head profiling."""

    def _init_vispec_draft_latency_backend(
        self,
        model_path,
        vispec_spec_model_path=None,
        vispec_depth=3,
        vispec_top_k=8,
        vispec_total_token=30,
        vispec_num_q=2,
        vispec_temperature=0.0,
        vispec_profile_visual_rebuild=True,
        vispec_include_visual_rebuild_in_total=False,
        vispec_ignore_hermes_summary=True,
        vispec_source_keep_policy="union_all",
    ):
        self.vispec_spec_model_path = (
            vispec_spec_model_path or _default_spec_model_path(model_path)
        )
        self.vispec_depth = int(vispec_depth)
        self.vispec_top_k = int(vispec_top_k)
        self.vispec_total_token = int(vispec_total_token)
        self.vispec_num_q = int(vispec_num_q)
        self.vispec_temperature = float(vispec_temperature)
        self.vispec_profile_visual_rebuild = bool(vispec_profile_visual_rebuild)
        self.vispec_include_visual_rebuild_in_total = bool(
            vispec_include_visual_rebuild_in_total
        )
        self.vispec_ignore_hermes_summary = bool(vispec_ignore_hermes_summary)
        self.vispec_source_keep_policy = vispec_source_keep_policy
        self._vispec_policy_layers(self.vispec_source_keep_policy)

        self._reset_vispec_visual_source_state()
        self.last_vispec_draft_profile = None
        self.vispec_spec_layer = self._load_vispec_spec_layer()

    def _load_vispec_spec_layer(self):
        hidden_size = self.language_model.config.hidden_size
        vocab_size = self.lm_head.weight.shape[0]
        config_path = _resolve_config_path(self.vispec_spec_model_path, hidden_size)
        config = EConfig.from_pretrained(str(config_path))

        if config.hidden_size != hidden_size:
            raise ValueError(
                "ViSpec draft head hidden_size does not match base model: "
                f"{config.hidden_size} != {hidden_size}"
            )
        if config.vocab_size != vocab_size:
            raise ValueError(
                "ViSpec draft head vocab_size does not match base lm_head: "
                f"{config.vocab_size} != {vocab_size}"
            )

        with open(config_path, "r", encoding="utf-8") as handle:
            raw_config = json.load(handle)
        bias = raw_config.get("bias", True)

        spec_layer = ViSpecDraftModel(
            config,
            bias=bias,
            total_tokens=self.vispec_total_token,
            depth=self.vispec_depth,
            top_k=self.vispec_top_k,
            threshold=1.0,
            num_q=self.vispec_num_q,
        )
        state_dict = _load_spec_state_dict(self.vispec_spec_model_path)
        missing_keys, unexpected_keys = spec_layer.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys:
            logger.warning("ViSpec draft head missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.warning("ViSpec draft head unexpected keys: %s", unexpected_keys)
        del state_dict

        device = self.lm_head.weight.device
        dtype = self.lm_head.weight.dtype
        spec_layer.to(device=device, dtype=dtype)
        spec_layer.init_tree()
        spec_layer.eval()
        logger.info(
            "Loaded ViSpec draft head from %s with depth=%d top_k=%d total_token=%d",
            self.vispec_spec_model_path,
            self.vispec_depth,
            self.vispec_top_k,
            self.vispec_total_token,
        )
        return spec_layer

    def _reset_vispec_visual_source_state(self):
        self.vispec_visual_sources = []
        self.vispec_next_chunk_id = 0
        self.vispec_summary_token_count_excluded = 0
        self.last_vispec_source_compact = {
            "source_keep_policy": getattr(
                self, "vispec_source_keep_policy", "union_all"
            ),
            "source_token_count_before_compact": 0,
            "source_token_count_after_compact": 0,
            "summary_visible_to_draft": False,
            "summary_token_count_excluded": 0,
            "summary_token_count_excluded_cumulative": 0,
        }

    def clear_cache(self):
        super().clear_cache()
        self._reset_vispec_visual_source_state()
        self.last_vispec_draft_profile = None
        if hasattr(self, "vispec_spec_layer"):
            self.vispec_spec_layer.reset_kv()
            self.vispec_spec_layer.last_img_hidden = None

    def encode_init_prompt(self):
        super().encode_init_prompt()
        self._reset_vispec_visual_source_state()

    @torch.inference_mode()
    def encode_video_chunk(self, video_chunk):
        if video_chunk is None or (
            hasattr(video_chunk, "shape") and video_chunk.shape[0] == 0
        ):
            return

        if len(video_chunk.shape) == 4 and video_chunk.shape[-1] == 3:
            video_chunk = video_chunk.permute(0, 3, 1, 2)

        video_input = self.processor(
            text=[""], videos=video_chunk, return_tensors="pt"
        ).to(self.device, self.dtype)
        pixel_values_videos = video_input["pixel_values_videos"]
        video_grid_thw = video_input["video_grid_thw"]
        video_features = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )[0].unsqueeze(0)

        self._ensure_dynamic_cache()

        cache_lens_before = self._get_cache_seq_len_per_layer()
        global_offset_per_layer = self._get_next_global_offset_per_layer()
        q_len = video_features.shape[1]
        batch = video_features.shape[0]

        base_offset = global_offset_per_layer[0]
        grid_pos_ids = get_qwen2_5_vl_position_ids(
            video_grid_thw[0].tolist(),
            q_len,
            offset=base_offset,
            vision_config=self.config.vision_config,
            sample_fps=self.sample_fps,
        ).to(self.device)

        self._layer_position_ids.clear()
        for layer_idx in range(self.num_layers):
            layer_offset = global_offset_per_layer[layer_idx]
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos = current_layer_pos + (layer_offset - base_offset)
            position_ids_3d = self._build_position_ids_3d_for_vision(
                current_layer_pos, batch
            )
            self._layer_position_ids[layer_idx] = position_ids_3d

        default_position_ids_3d = self._build_position_ids_3d_for_vision(
            grid_pos_ids, batch
        )

        out = self.language_model(
            inputs_embeds=video_features,
            past_key_values=self.kv_cache,
            use_cache=True,
            return_dict=True,
            position_ids=default_position_ids_3d,
        )
        self.kv_cache = out.past_key_values
        contiguous_kv(self.kv_cache)

        for layer_idx in range(self.num_layers):
            layer_offset = global_offset_per_layer[layer_idx]
            current_layer_pos = grid_pos_ids.clone()
            if layer_offset != base_offset:
                current_layer_pos = current_layer_pos + (layer_offset - base_offset)
            self._append_position_ids_layer_explicit(layer_idx, current_layer_pos)

        self._record_vispec_visual_source(
            cache_lens_before=cache_lens_before,
            video_features=video_features,
            target_hidden_states=out.last_hidden_state,
            position_ids_3d=default_position_ids_3d,
        )

        self.last_encoded_frames = video_chunk.shape[0]
        self.total_processed_frames += video_chunk.shape[0]

        self._layer_position_ids.clear()
        torch.cuda.empty_cache()

    def _record_vispec_visual_source(
        self,
        cache_lens_before,
        video_features,
        target_hidden_states,
        position_ids_3d,
    ):
        q_len = video_features.shape[1]
        layer_indices = []
        for layer_idx in range(self.num_layers):
            start = cache_lens_before[layer_idx]
            layer_indices.append(torch.arange(start, start + q_len, dtype=torch.long))
        layer_indices = torch.stack(layer_indices, dim=0)

        source = {
            "chunk_id": self.vispec_next_chunk_id,
            "global_indices": layer_indices[0].clone(),
            "layer_indices": layer_indices,
            "video_features": video_features.detach().to("cpu"),
            "target_hidden_states": target_hidden_states.detach().to("cpu"),
            "position_ids_3d": position_ids_3d.detach().to("cpu"),
        }
        self.vispec_visual_sources.append(source)
        self.vispec_next_chunk_id += 1

    @torch.inference_mode()
    def apply_kv_cache_pruning_strict(self, keep_indices_all_layers):
        if self.kv_cache is None:
            logger.warning("No KV-Cache to prune")
            return
        if not keep_indices_all_layers or len(keep_indices_all_layers[0]) == 0:
            logger.warning("Empty keep_indices; skip pruning")
            return

        old_lens = self._get_cache_seq_len_per_layer()
        safe_keep_indices = []
        for layer_idx, keep_indices in enumerate(keep_indices_all_layers):
            if not isinstance(keep_indices, torch.Tensor):
                keep_indices = torch.as_tensor(keep_indices, device=self.device)
            safe_keep_indices.append(
                self._sanitize_keep_indices(
                    keep_indices, old_lens[layer_idx]
                ).detach().cpu()
            )

        self._shrink_positions_and_rerotate_keys(keep_indices_all_layers)
        self._compact_vispec_visual_sources(safe_keep_indices, old_lens)
        logger.info(f"Strict-shrunk KV cache. New length: {get_cache_seq_len(self.kv_cache)}")

    def _vispec_policy_layers(self, policy):
        if policy == "union_all":
            return list(range(self.num_layers))
        if policy == "shallow":
            return list(range(0, self.short_term_threshold))
        if policy == "mid":
            return list(range(self.short_term_threshold, self.long_term_threshold))
        if policy == "deep":
            return list(range(self.long_term_threshold, self.num_layers))
        if policy.startswith("layer:"):
            layer_idx = int(policy.split(":", 1)[1])
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(
                    f"ViSpec source keep policy layer index out of range: {policy}"
                )
            return [layer_idx]
        raise ValueError(f"Unsupported ViSpec source keep policy: {policy}")

    def _current_vispec_source_token_count(self):
        return sum(
            source["video_features"].shape[1] for source in self.vispec_visual_sources
        )

    def _compact_vispec_visual_sources(self, safe_keep_indices, old_lens):
        before = self._current_vispec_source_token_count()
        policy_layers = self._vispec_policy_layers(self.vispec_source_keep_policy)

        mappings = []
        for layer_idx, keep_indices in enumerate(safe_keep_indices):
            mapping = torch.full((old_lens[layer_idx],), -1, dtype=torch.long)
            if keep_indices.numel() > 0:
                mapping[keep_indices.long()] = torch.arange(
                    keep_indices.numel(), dtype=torch.long
                )
            mappings.append(mapping)

        summary_count_excluded = 0
        if self.vispec_ignore_hermes_summary:
            for layer_idx in range(self.long_term_threshold, self.num_layers):
                if old_lens[layer_idx] > safe_keep_indices[layer_idx].numel():
                    summary_count_excluded += 1

        compacted_sources = []
        for source in self.vispec_visual_sources:
            layer_indices = source["layer_indices"].long()
            new_layer_indices = torch.full_like(layer_indices, -1)
            for layer_idx in range(self.num_layers):
                old_idx = layer_indices[layer_idx]
                valid = (old_idx >= 0) & (old_idx < old_lens[layer_idx])
                if valid.any():
                    new_layer_indices[layer_idx, valid] = mappings[layer_idx][
                        old_idx[valid]
                    ]

            keep_mask = (new_layer_indices[policy_layers] >= 0).any(dim=0)
            if not keep_mask.any():
                continue

            kept_layer_indices = new_layer_indices[:, keep_mask].contiguous()
            compacted_sources.append(
                {
                    "chunk_id": source["chunk_id"],
                    "global_indices": self._first_available_indices(
                        kept_layer_indices
                    ),
                    "layer_indices": kept_layer_indices,
                    "video_features": source["video_features"][
                        :, keep_mask, :
                    ].contiguous(),
                    "target_hidden_states": source["target_hidden_states"][
                        :, keep_mask, :
                    ].contiguous(),
                    "position_ids_3d": source["position_ids_3d"][
                        :, :, keep_mask
                    ].contiguous(),
                }
            )

        self.vispec_visual_sources = compacted_sources
        after = self._current_vispec_source_token_count()
        self.vispec_summary_token_count_excluded += summary_count_excluded
        self.last_vispec_source_compact = {
            "source_keep_policy": self.vispec_source_keep_policy,
            "source_token_count_before_compact": int(before),
            "source_token_count_after_compact": int(after),
            "summary_visible_to_draft": False,
            "summary_token_count_excluded": int(summary_count_excluded),
            "summary_token_count_excluded_cumulative": int(
                self.vispec_summary_token_count_excluded
            ),
        }

    def _first_available_indices(self, layer_indices):
        n_tokens = layer_indices.shape[1]
        result = torch.full((n_tokens,), -1, dtype=torch.long)
        for layer_idx in range(layer_indices.shape[0]):
            layer_values = layer_indices[layer_idx]
            fill_mask = (result < 0) & (layer_values >= 0)
            result[fill_mask] = layer_values[fill_mask]
        return result

    def _collect_vispec_visual_sources(self, device, dtype):
        if not self.vispec_visual_sources:
            return None, None
        video_features = torch.cat(
            [source["video_features"] for source in self.vispec_visual_sources],
            dim=1,
        ).to(device=device, dtype=dtype, non_blocking=True)
        target_hidden_states = torch.cat(
            [
                source["target_hidden_states"]
                for source in self.vispec_visual_sources
            ],
            dim=1,
        ).to(device=device, dtype=dtype, non_blocking=True)
        return video_features, target_hidden_states

    def _rebuild_vispec_visual_context(self):
        spec_layer = self.vispec_spec_layer
        device = spec_layer.embed_tokens.weight.device
        dtype = spec_layer.embed_tokens.weight.dtype
        token_count = self._current_vispec_source_token_count()

        _cuda_sync_if_needed(device)
        start_time = time.perf_counter()
        used_visual_context = False
        target_hidden_shape = None

        if self.vispec_profile_visual_rebuild and token_count > 0:
            video_features, target_hidden_states = self._collect_vispec_visual_sources(
                device, dtype
            )
            target_hidden_shape = list(target_hidden_states.shape)
            adapted = spec_layer.imadpt(video_features)
            spec_layer.last_img_hidden = adapted[0, -1:].detach()
            used_visual_context = True
        else:
            spec_layer.last_img_hidden = torch.zeros(
                (1, spec_layer.embed_tokens.embedding_dim),
                device=device,
                dtype=dtype,
            )

        spec_layer.preserve_last_img_hidden = True
        _cuda_sync_if_needed(device)
        latency = time.perf_counter() - start_time
        return {
            "draft_visual_rebuild_latency_seconds": latency,
            "draft_visual_source_token_count": int(token_count),
            "draft_visual_context_used": used_visual_context,
            "draft_visual_target_hidden_shape": target_hidden_shape,
        }

    @torch.inference_mode()
    def profile_vispec_draft_head(self, input_text):
        if self.vispec_temperature != 0.0:
            logger.warning(
                "ViSpec draft latency backend currently profiles greedy "
                "temperature=0 only; got temperature=%s",
                self.vispec_temperature,
            )

        device = self.device
        spec_layer = self.vispec_spec_layer
        spec_device = spec_layer.embed_tokens.weight.device
        past_lens_prefill = self._get_cache_seq_len_per_layer()
        profile = None

        try:
            spec_layer.reset_kv()
            spec_layer.last_img_hidden = None
            spec_layer.preserve_last_img_hidden = False
            visual_profile = self._rebuild_vispec_visual_context()

            prompt = input_text["prompt"]
            input_ids = self.processor.tokenizer(prompt).input_ids
            input_ids = torch.as_tensor([input_ids], device=device)

            self._ensure_dynamic_cache()
            global_offset_prefill = self._get_next_global_offset_per_layer()
            inputs_embeds = self.get_input_embeddings()(input_ids)
            q_len_prefill = inputs_embeds.shape[1]
            batch = inputs_embeds.shape[0]

            self._layer_position_ids.clear()
            for layer_idx in range(self.num_layers):
                position_ids_3d = self._build_position_ids_3d_for_text(
                    global_offset_prefill[layer_idx], q_len_prefill, batch
                )
                self._layer_position_ids[layer_idx] = position_ids_3d

            position_ids_3d = self._build_position_ids_3d_for_text(
                global_offset_prefill[0], q_len_prefill, batch
            )

            _cuda_sync_if_needed(input_ids.device)
            prefill_start_time = time.perf_counter()
            out = self.language_model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=self.kv_cache,
                position_ids=position_ids_3d,
                return_dict=True,
            )
            _cuda_sync_if_needed(input_ids.device)
            target_prefill_latency = time.perf_counter() - prefill_start_time

            target_hidden_states = out.last_hidden_state
            target_logits = self.lm_head(target_hidden_states)

            for layer_idx in range(self.num_layers):
                offset = global_offset_prefill[layer_idx]
                self._append_position_ids_layer(
                    layer_idx, [offset, offset, offset], q_len_prefill
                )
            self._layer_position_ids.clear()

            sample_token = target_logits[:, -1].argmax(dim=-1, keepdim=True)
            draft_input_ids = torch.cat(
                (input_ids, sample_token.to(input_ids.device)), dim=-1
            ).to(spec_device)
            draft_hidden_states = target_hidden_states.detach().to(spec_device)
            draft_inputs_embeds = inputs_embeds.detach().to(spec_device)

            _cuda_sync_if_needed(spec_device)
            draft_start_time = time.perf_counter()
            (
                draft_tokens,
                retrieve_indices,
                tree_mask,
                tree_position_ids,
                layer_timings,
                pack_latency,
            ) = spec_layer.topK_genrate(
                hidden_states=draft_hidden_states,
                input_ids=draft_input_ids,
                head=self._vispec_head,
                logits_processor=None,
                inputs_embeds=draft_inputs_embeds,
                image_mask=None,
                return_layer_timings=True,
                timing_sync_device=spec_device,
            )
            _cuda_sync_if_needed(spec_device)
            draft_compute_latency = time.perf_counter() - draft_start_time

            draft_with_visual_latency = (
                draft_compute_latency
                + visual_profile["draft_visual_rebuild_latency_seconds"]
            )
            draft_total_latency = (
                draft_with_visual_latency
                if self.vispec_include_visual_rebuild_in_total
                else draft_compute_latency
            )

            profile = {
                "target_prefill_latency_seconds": target_prefill_latency,
                "draft_head_total_latency_seconds": draft_total_latency,
                "draft_head_compute_latency_seconds": draft_compute_latency,
                "draft_head_with_visual_rebuild_latency_seconds": (
                    draft_with_visual_latency
                ),
                "draft_tree_pack_latency_seconds": pack_latency,
                "draft_depth": self.vispec_depth,
                "draft_top_k": self.vispec_top_k,
                "draft_total_token": self.vispec_total_token,
                "draft_num_q": self.vispec_num_q,
                "draft_temperature": self.vispec_temperature,
                "draft_tokens_shape": list(draft_tokens.shape),
                "retrieve_indices_shape": list(retrieve_indices.shape),
                "tree_mask_shape": list(tree_mask.shape),
                "tree_position_ids_shape": list(tree_position_ids.shape),
                "sample_token_id": int(sample_token.item()),
                "layer_timings": layer_timings,
                "target_kv_lengths_before_prefill": [
                    int(length) for length in past_lens_prefill
                ],
                "target_kv_lengths_after_prefill": [
                    int(length) for length in self._get_cache_seq_len_per_layer()
                ],
                **visual_profile,
                **self.last_vispec_source_compact,
            }
            return profile
        finally:
            spec_layer.reset_kv()
            spec_layer.last_img_hidden = None
            spec_layer.preserve_last_img_hidden = False
            self._truncate_kv_cache(past_lens_prefill)
            for layer_idx in range(self.num_layers):
                cache = self._position_ids_cache[layer_idx]
                if cache is not None and cache.shape[1] > past_lens_prefill[layer_idx]:
                    self._position_ids_cache[layer_idx] = cache[
                        :, : past_lens_prefill[layer_idx]
                    ].contiguous()
            self._layer_position_ids.clear()
            if profile is not None:
                profile["target_kv_lengths_after_cleanup"] = [
                    int(length) for length in self._get_cache_seq_len_per_layer()
                ]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _vispec_head(self, hidden_states):
        weight = self.lm_head.weight
        bias = getattr(self.lm_head, "bias", None)
        if weight.device == hidden_states.device:
            return self.lm_head(hidden_states)
        return F.linear(
            hidden_states,
            weight.detach().to(hidden_states.device),
            None if bias is None else bias.detach().to(hidden_states.device),
        )

    @torch.inference_mode()
    def question_answering(
        self,
        input_text,
        max_new_tokens=128,
        temperature=0,
        repetition_penalty=1.1,
        pseudo_forward=False,
    ):
        if not pseudo_forward:
            try:
                self.last_vispec_draft_profile = self.profile_vispec_draft_head(
                    input_text
                )
            except Exception as exc:
                logger.exception("ViSpec draft-head profiling failed")
                self.last_vispec_draft_profile = {
                    "error": str(exc),
                    "draft_depth": self.vispec_depth,
                    "draft_top_k": self.vispec_top_k,
                    "draft_total_token": self.vispec_total_token,
                    "source_keep_policy": self.vispec_source_keep_policy,
                    "summary_visible_to_draft": False,
                }

        return super().question_answering(
            input_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pseudo_forward=pseudo_forward,
        )


def load_model(
    model_path="models/Qwen2.5-VL-7B-Instruct",
    n_init=None,
    kv_size=None,
    streaming=True,
    device="cuda",
    sample_fps=1,
    vispec_spec_model_path=None,
    vispec_depth=3,
    vispec_top_k=8,
    vispec_total_token=30,
    vispec_num_q=2,
    vispec_temperature=0.0,
    vispec_profile_visual_rebuild=True,
    vispec_include_visual_rebuild_in_total=False,
    vispec_ignore_hermes_summary=True,
    vispec_source_keep_policy="union_all",
    use_flash_attention=False,
    max_memory_per_gpu=None,
    disallow_cpu_offload=False,
    print_device_map=False,
):
    model, processor = load_qwenvl_hermes_model(
        model_path=model_path,
        n_init=n_init,
        kv_size=kv_size,
        streaming=streaming,
        device=device,
        sample_fps=sample_fps,
        use_flash_attention=use_flash_attention,
        max_memory_per_gpu=max_memory_per_gpu,
        disallow_cpu_offload=disallow_cpu_offload,
        print_device_map=print_device_map,
    )
    model.__class__ = QwenVL_ViSpecDraftLatency
    model._init_vispec_draft_latency_backend(
        model_path=model_path,
        vispec_spec_model_path=vispec_spec_model_path,
        vispec_depth=vispec_depth,
        vispec_top_k=vispec_top_k,
        vispec_total_token=vispec_total_token,
        vispec_num_q=vispec_num_q,
        vispec_temperature=vispec_temperature,
        vispec_profile_visual_rebuild=vispec_profile_visual_rebuild,
        vispec_include_visual_rebuild_in_total=vispec_include_visual_rebuild_in_total,
        vispec_ignore_hermes_summary=vispec_ignore_hermes_summary,
        vispec_source_keep_policy=vispec_source_keep_policy,
    )
    return model, processor
