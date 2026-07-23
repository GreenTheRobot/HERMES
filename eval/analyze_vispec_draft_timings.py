import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import fmean


DETAIL_FIELDS = [
    "video_index",
    "video_id",
    "question_index",
    "tree_depth",
    "phase",
    "frontier_tokens",
    "candidate_tokens",
    "selected_frontier_tokens",
    "latency_seconds",
    "latency_per_candidate_token_seconds",
    "draft_depth",
    "draft_top_k",
    "draft_total_token",
    "draft_visual_source_token_count",
]

SUMMARY_FIELDS = [
    "scope",
    "video_index",
    "video_id",
    "video_count",
    "profile_count",
    "profile_error_count",
    "layer_timing_count",
    "mean_target_prefill_latency_seconds",
    "mean_draft_head_compute_latency_seconds",
    "mean_draft_head_with_visual_rebuild_latency_seconds",
    "mean_draft_tree_pack_latency_seconds",
    "mean_layer_latency_seconds",
    "mean_latency_per_candidate_token_seconds",
]


def _parse_non_negative_int(value, field_name, row_number):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number}: {field_name} must be an integer, got {value!r}"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"Row {row_number}: {field_name} must be non-negative, got {parsed}"
        )
    return parsed


def _parse_non_negative_float(value, field_name, row_number):
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number}: {field_name} must be a float, got {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"Row {row_number}: {field_name} must be finite and non-negative"
        )
    return parsed


def _optional_float(profile, field_name, row_number):
    value = profile.get(field_name)
    if value is None:
        return None
    return _parse_non_negative_float(value, field_name, row_number)


def load_vispec_profiles(results_path):
    results_path = Path(results_path)
    detail_rows = []
    profile_rows = []
    videos = {}

    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "video_index",
            "video_id",
            "question_index",
            "vispec_draft_profile",
        }
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"{results_path} is missing required columns: {missing}")

        for row_number, result_row in enumerate(reader, start=2):
            video_index = _parse_non_negative_int(
                result_row["video_index"], "video_index", row_number
            )
            question_index = _parse_non_negative_int(
                result_row["question_index"], "question_index", row_number
            )
            video_id = result_row["video_id"]
            previous_video_id = videos.setdefault(video_index, video_id)
            if previous_video_id != video_id:
                raise ValueError(
                    f"video_index {video_index} maps to both "
                    f"{previous_video_id!r} and {video_id!r}"
                )

            raw_profile = result_row["vispec_draft_profile"]
            if raw_profile is None or not raw_profile.strip():
                continue
            try:
                profile = json.loads(raw_profile)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Row {row_number}: vispec_draft_profile is not valid JSON"
                ) from exc
            if not isinstance(profile, dict):
                raise ValueError(
                    f"Row {row_number}: vispec_draft_profile must be a JSON object"
                )

            profile_error = profile.get("error")
            profile_row = {
                "video_index": video_index,
                "video_id": video_id,
                "question_index": question_index,
                "error": profile_error,
                "target_prefill_latency_seconds": _optional_float(
                    profile, "target_prefill_latency_seconds", row_number
                ),
                "draft_head_compute_latency_seconds": _optional_float(
                    profile, "draft_head_compute_latency_seconds", row_number
                ),
                "draft_head_with_visual_rebuild_latency_seconds": _optional_float(
                    profile,
                    "draft_head_with_visual_rebuild_latency_seconds",
                    row_number,
                ),
                "draft_tree_pack_latency_seconds": _optional_float(
                    profile, "draft_tree_pack_latency_seconds", row_number
                ),
            }
            profile_rows.append(profile_row)

            if profile_error:
                continue

            layer_timings = profile.get("layer_timings", [])
            if not isinstance(layer_timings, list):
                raise ValueError(
                    f"Row {row_number}: layer_timings must be a JSON list"
                )

            for timing in layer_timings:
                if not isinstance(timing, dict):
                    raise ValueError(
                        f"Row {row_number}: every layer timing must be an object"
                    )
                tree_depth = _parse_non_negative_int(
                    timing.get("tree_depth"), "tree_depth", row_number
                )
                frontier_tokens = _parse_non_negative_int(
                    timing.get("frontier_tokens"), "frontier_tokens", row_number
                )
                candidate_tokens = _parse_non_negative_int(
                    timing.get("candidate_tokens"), "candidate_tokens", row_number
                )
                selected_frontier_tokens = _parse_non_negative_int(
                    timing.get("selected_frontier_tokens"),
                    "selected_frontier_tokens",
                    row_number,
                )
                latency = _parse_non_negative_float(
                    timing.get("latency_seconds"), "latency_seconds", row_number
                )
                detail_rows.append(
                    {
                        "video_index": video_index,
                        "video_id": video_id,
                        "question_index": question_index,
                        "tree_depth": tree_depth,
                        "phase": timing.get("phase", ""),
                        "frontier_tokens": frontier_tokens,
                        "candidate_tokens": candidate_tokens,
                        "selected_frontier_tokens": selected_frontier_tokens,
                        "latency_seconds": latency,
                        "latency_per_candidate_token_seconds": (
                            latency / candidate_tokens
                            if candidate_tokens > 0
                            else None
                        ),
                        "draft_depth": profile.get("draft_depth"),
                        "draft_top_k": profile.get("draft_top_k"),
                        "draft_total_token": profile.get("draft_total_token"),
                        "draft_visual_source_token_count": profile.get(
                            "draft_visual_source_token_count"
                        ),
                    }
                )

    detail_rows.sort(
        key=lambda row: (
            row["video_index"],
            row["question_index"],
            row["tree_depth"],
        )
    )
    return detail_rows, profile_rows, videos


def _mean(values):
    clean = [value for value in values if value is not None]
    return fmean(clean) if clean else None


def _build_summary_row(scope, video_index, video_id, profiles, detail_rows, video_count):
    profile_error_count = sum(1 for row in profiles if row["error"])
    return {
        "scope": scope,
        "video_index": video_index,
        "video_id": video_id,
        "video_count": video_count,
        "profile_count": len(profiles),
        "profile_error_count": profile_error_count,
        "layer_timing_count": len(detail_rows),
        "mean_target_prefill_latency_seconds": _mean(
            [row["target_prefill_latency_seconds"] for row in profiles]
        ),
        "mean_draft_head_compute_latency_seconds": _mean(
            [row["draft_head_compute_latency_seconds"] for row in profiles]
        ),
        "mean_draft_head_with_visual_rebuild_latency_seconds": _mean(
            [
                row["draft_head_with_visual_rebuild_latency_seconds"]
                for row in profiles
            ]
        ),
        "mean_draft_tree_pack_latency_seconds": _mean(
            [row["draft_tree_pack_latency_seconds"] for row in profiles]
        ),
        "mean_layer_latency_seconds": _mean(
            [row["latency_seconds"] for row in detail_rows]
        ),
        "mean_latency_per_candidate_token_seconds": _mean(
            [
                row["latency_per_candidate_token_seconds"]
                for row in detail_rows
            ]
        ),
    }


def build_summary(detail_rows, profile_rows, videos):
    detail_by_video = {video_index: [] for video_index in videos}
    profile_by_video = {video_index: [] for video_index in videos}

    for row in detail_rows:
        detail_by_video[row["video_index"]].append(row)
    for row in profile_rows:
        profile_by_video[row["video_index"]].append(row)

    summary_rows = []
    for video_index in sorted(videos):
        summary_rows.append(
            _build_summary_row(
                "video",
                video_index,
                videos[video_index],
                profile_by_video[video_index],
                detail_by_video[video_index],
                1,
            )
        )

    summary_rows.append(
        _build_summary_row(
            "dataset",
            "",
            "ALL",
            profile_rows,
            detail_rows,
            len(videos),
        )
    )
    return summary_rows


def _validate_prefix(prefix, argument_name):
    if not prefix or Path(prefix).name != prefix:
        raise ValueError(
            f"{argument_name} must be a filename prefix without directory components"
        )


def reserve_output_paths(output_dir, detail_prefix, summary_prefix):
    _validate_prefix(detail_prefix, "detail_prefix")
    _validate_prefix(summary_prefix, "summary_prefix")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        detail_path = output_dir / f"{detail_prefix}_{index}.csv"
        summary_path = output_dir / f"{summary_prefix}_{index}.csv"
        lock_path = output_dir / f".{detail_prefix}_{index}.lock"

        if detail_path.exists() or summary_path.exists() or lock_path.exists():
            index += 1
            continue

        try:
            file_descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            index += 1
            continue
        os.close(file_descriptor)

        if detail_path.exists() or summary_path.exists():
            lock_path.unlink(missing_ok=True)
            index += 1
            continue
        return detail_path, summary_path, lock_path


def write_csv_exclusive(path, fieldnames, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_mean(value):
    if value is None:
        return "N/A"
    return f"{value:.9f} s ({value * 1000:.6f} ms)"


def print_summary(summary_rows, detail_path, summary_path):
    dataset_row = next(row for row in summary_rows if row["scope"] == "dataset")
    print(
        "Dataset mean draft-head compute latency: "
        f"{_format_mean(dataset_row['mean_draft_head_compute_latency_seconds'])}"
    )
    print(
        "Dataset mean draft-head with visual rebuild latency: "
        f"{_format_mean(dataset_row['mean_draft_head_with_visual_rebuild_latency_seconds'])}"
    )
    print(f"ViSpec draft timing detail: {detail_path}")
    print(f"ViSpec draft timing summary: {summary_path}")


def analyze_results(
    results_path,
    output_dir=None,
    detail_prefix="vispec_draft_timings",
    summary_prefix="vispec_draft_timing_summary",
):
    results_path = Path(results_path)
    if output_dir is None:
        output_dir = results_path.parent

    detail_rows, profile_rows, videos = load_vispec_profiles(results_path)
    summary_rows = build_summary(detail_rows, profile_rows, videos)
    detail_path, summary_path, lock_path = reserve_output_paths(
        output_dir,
        detail_prefix,
        summary_prefix,
    )
    try:
        write_csv_exclusive(detail_path, DETAIL_FIELDS, detail_rows)
        write_csv_exclusive(summary_path, SUMMARY_FIELDS, summary_rows)
    finally:
        lock_path.unlink(missing_ok=True)

    print_summary(summary_rows, detail_path, summary_path)
    return detail_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Expand ViSpec draft-head profile JSON from a complete HERMES "
            "results.csv and calculate per-video and dataset latency summaries."
        )
    )
    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--detail_prefix",
        type=str,
        default="vispec_draft_timings",
    )
    parser.add_argument(
        "--summary_prefix",
        type=str,
        default="vispec_draft_timing_summary",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_results(
        results_path=args.results_path,
        output_dir=args.output_dir,
        detail_prefix=args.detail_prefix,
        summary_prefix=args.summary_prefix,
    )
