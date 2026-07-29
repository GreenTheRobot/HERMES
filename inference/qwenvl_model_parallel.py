from collections import Counter

import torch
from logzero import logger


_OFFLOAD_DEVICES = {"cpu", "disk"}


def build_max_memory(max_memory_per_gpu):
    if not max_memory_per_gpu:
        return None
    return {idx: max_memory_per_gpu for idx in range(torch.cuda.device_count())}


def _module_device(module, fallback_device):
    if module is None:
        return torch.device(fallback_device)
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device(fallback_device)


def get_input_device(model, fallback_device):
    return _module_device(model.get_input_embeddings(), fallback_device)


def get_layer_device(model, layer_idx, fallback_device):
    kv_cache = getattr(model, "kv_cache", None)
    if kv_cache is not None and layer_idx < len(kv_cache):
        return kv_cache[layer_idx][0].device

    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is not None and layer_idx < len(layers):
        return _module_device(layers[layer_idx], fallback_device)

    return torch.device(fallback_device)


def log_model_parallel_state(model, max_memory=None, print_device_map=False, disallow_cpu_offload=False):
    logger.info(f"cuda_visible_device_count: {torch.cuda.device_count()}")
    if max_memory is not None:
        logger.info(f"max_memory: {max_memory}")

    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map is None:
        logger.info("hf_device_map: <not available>")
        return

    device_counts = Counter(str(device) for device in hf_device_map.values())
    logger.info(f"hf_device_map_summary: {dict(device_counts)}")

    if print_device_map:
        for module_name, device in sorted(hf_device_map.items()):
            logger.info(f"hf_device_map[{module_name}]: {device}")

    offloaded = {
        module_name: str(device)
        for module_name, device in hf_device_map.items()
        if str(device).lower() in _OFFLOAD_DEVICES
    }
    if not offloaded:
        logger.info("hf_device_map_offload: none")
        return

    logger.warning(f"hf_device_map_offload: {offloaded}")
    if disallow_cpu_offload:
        raise RuntimeError(
            "CPU/disk offload detected in hf_device_map while "
            "disallow_cpu_offload=True. Reduce max_memory pressure or use more GPUs."
        )


def sync_cuda_devices():
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_idx)
