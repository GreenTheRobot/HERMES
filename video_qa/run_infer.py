import os
import argparse
import subprocess
import multiprocessing


def exec(cmd, sub=False, device=None):
    print(f'exec: {cmd}')
    if not sub:
        if isinstance(cmd, list):
            cmd = ' '.join(cmd)
        os.system(cmd)
    else:
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = device
        print(f'CUDA_VISIBLE_DEVICES={device}')
        subprocess.run(cmd, env=my_env)


BENCHMARK_CONFIGS = {
    "videomme": {
        "streaming": False,
        "anno_path": "data/videomme/videomme.json",
        "eval_cmds": [
            "python eval/eval_multiple_choice.py general --results_path {results_path}",
            "python eval/eval_multiple_choice.py videomme --results_path {results_path}",
        ],
    },
    "mvbench": {
        "streaming": False,
        "anno_path": "data/mvbench/mvbench.json",
        "eval_cmds": [
            "python eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
    "egoschema": {
        "streaming": False,
        "anno_path": "data/egoschema/egoschema.json",
        "eval_cmds": [
            "python eval/eval_multiple_choice.py egoschema --results_path {results_path}",
        ],
    },
    "rvs_ego": {
        "streaming": True,
        "anno_path": "data/rvs/ego/ego4d_oe.json",
        "eval_cmds": [
            "python eval/eval_open_ended.py --pred_path {results_path} --output_dir {save_dir}/tmp --output_json {save_dir}/results.json",
        ],
    },
    "rvs_movie": {
        "streaming": True,
        "anno_path": "data/rvs/movie/movienet_oe.json",
        "eval_cmds": [
            "python eval/eval_open_ended.py --pred_path {results_path} --output_dir {save_dir}/tmp --output_json {save_dir}/results.json",
        ],
    },
    "ovobench": {
        "streaming": True,
        "anno_path": "data/ovobench/ovobench_realtime_backeward.json",
        "eval_cmds": [
            "python eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
    "streamingbench": {
        "streaming": True,
        "anno_path": "data/streamingbench/streamingbench_realtime.json",
        "eval_cmds": [
            "python eval/eval_multiple_choice.py general --results_path {results_path}",
        ],
    },
}


def run_eval(args, config):
    num_chunks = args.num_chunks
    save_dir = f"results/{args.model}/{args.dataset}/fps{args.sample_fps}-kv{args.kv_size}"
    streaming = config["streaming"]
    results_path = f"{save_dir}/results.csv"
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [gpu_id.strip() for gpu_id in args.gpu_ids.split(",") if gpu_id.strip()]
        gpus_per_chunk = 4 if args.model in {'llava_ov_72b', 'llava_ov_72b_slidingwindow'} else 1
        required_gpus = num_chunks * gpus_per_chunk
        if len(gpu_ids) < required_gpus:
            raise ValueError(
                f"--gpu_ids provides {len(gpu_ids)} GPU(s), but "
                f"{num_chunks} chunk(s) require {required_gpus} GPU(s) for {args.model}."
            )

    if not args.only_eval:
        processes = []
        for idx in range(num_chunks):
            cmd = [
                "python", "video_qa/hermes_vqa.py",
                "--model", args.model,
                "--sample_fps", str(args.sample_fps),
                "--save_dir", save_dir,
                "--anno_path", config["anno_path"],
                "--debug", args.debug,
                "--num_chunks", str(num_chunks),
                "--chunk_idx", str(idx),
                "--kv_size", str(args.kv_size),
                "--streaming", str(streaming),
            ]
            if args.model in {'llava_ov_72b', 'llava_ov_72b_slidingwindow'}:
                if gpu_ids:
                    device = ",".join(gpu_ids[4 * idx : 4 * idx + 4])
                else:
                    device = f'{4*idx},{4*idx+1},{4*idx+2},{4*idx+3}'
            else:
                device = gpu_ids[idx] if gpu_ids else str(idx)
            p = multiprocessing.Process(target=exec, args=(cmd, True, device))
            processes.append(p)
            p.start()

        failed_chunks = []
        for idx, p in enumerate(processes):
            p.join()
            chunk_file = f"{save_dir}/{num_chunks}_{idx}.csv"
            if not os.path.exists(chunk_file):
                failed_chunks.append(idx)
                print(f"WARNING: Chunk {idx} failed - file {chunk_file} not found!")

        if failed_chunks:
            raise RuntimeError(
                f"The following chunks failed: {failed_chunks}. Please rerun them manually."
            )

        exec(f"> {results_path}")
        for idx in range(num_chunks):
            chunk_file = f"{save_dir}/{num_chunks}_{idx}.csv"
            if idx == 0:
                exec(f"head -n 1 {chunk_file} > {results_path}")
            exec(f"tail -n +2 {chunk_file} >> {results_path}")
            exec(f"rm {chunk_file}")

    if not args.skip_token_timing_analysis:
        subprocess.run(
            [
                "python",
                "eval/analyze_token_timings.py",
                "--results_path",
                results_path,
                "--output_dir",
                save_dir,
                "--detail_prefix",
                args.timing_detail_prefix,
                "--summary_prefix",
                args.timing_summary_prefix,
            ],
            check=True,
        )

    if args.skip_eval:
        return

    fmt = {"results_path": results_path, "save_dir": save_dir, "anno_path": config["anno_path"]}
    for cmd_template in config["eval_cmds"]:
        exec(cmd_template.format(**fmt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="llava_ov_7b",
        choices=[
            'llava_ov_0.5b',
            'llava_ov_7b',
            'llava_ov_72b',
            'qwen2.5_vl_3b',
            'qwen2.5_vl_7b',
            'qwen2.5_vl_32b',
            'llava_ov_0.5b_slidingwindow',
            'llava_ov_7b_slidingwindow',
            'llava_ov_72b_slidingwindow',
            'qwen2.5_vl_3b_slidingwindow',
            'qwen2.5_vl_7b_slidingwindow',
            'qwen2.5_vl_32b_slidingwindow',
        ],
    )
    parser.add_argument("--dataset", type=str, default=None, choices=list(BENCHMARK_CONFIGS.keys()))
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--sample_fps", type=float, default=1)
    parser.add_argument("--debug", type=str, default='false')
    parser.add_argument("--kv_size", type=int)
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids assigned to chunks, e.g. 1,3.",
    )
    parser.add_argument(
        "--skip_token_timing_analysis",
        action="store_true",
        help="Skip generation of token timing detail and summary CSV files.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip benchmark evaluation after inference and token timing analysis.",
    )
    parser.add_argument(
        "--timing_detail_prefix",
        type=str,
        default="token_timings",
        help="Filename prefix for token-level timing CSV files.",
    )
    parser.add_argument(
        "--timing_summary_prefix",
        type=str,
        default="token_timing_summary",
        help="Filename prefix for timing summary CSV files.",
    )
    args = parser.parse_args()

    if args.dataset in BENCHMARK_CONFIGS:
        print(f'Execute {args.dataset} evaluation')
        run_eval(args, BENCHMARK_CONFIGS[args.dataset])
