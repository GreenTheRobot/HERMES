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
    "token_index",
    "token_id",
    "phase",
    "latency_seconds",
]

SUMMARY_FIELDS = [
    "scope",
    "video_index",
    "video_id",
    "video_count",
    "decode_token_count",
    "mean_decode_latency_seconds",
    "mean_decode_latency_ms",
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


def _parse_latency(value, row_number, token_index):
    try:
        latency = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number}, token {token_index}: invalid latency {value!r}"
        ) from exc
    if not math.isfinite(latency) or latency < 0:
        raise ValueError(
            f"Row {row_number}, token {token_index}: latency must be finite and non-negative"
        )
    return latency


def load_token_timings(results_path):
    results_path = Path(results_path)
    detail_rows = []
    videos = {}
    seen_coordinates = set()

    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "video_index",
            "video_id",
            "question_index",
            "token_inference_times",
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

            raw_timings = result_row["token_inference_times"]
            if raw_timings is None or not raw_timings.strip():
                continue
            try:
                token_timings = json.loads(raw_timings)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Row {row_number}: token_inference_times is not valid JSON"
                ) from exc
            if not isinstance(token_timings, list):
                raise ValueError(
                    f"Row {row_number}: token_inference_times must be a JSON list"
                )

            for timing in token_timings:
                if not isinstance(timing, dict):
                    raise ValueError(
                        f"Row {row_number}: every timing entry must be a JSON object"
                    )
                missing_timing_fields = {
                    "token_index",
                    "token_id",
                    "phase",
                    "latency_seconds",
                }.difference(timing)
                if missing_timing_fields:
                    missing = ", ".join(sorted(missing_timing_fields))
                    raise ValueError(
                        f"Row {row_number}: timing entry is missing fields: {missing}"
                    )

                token_index = _parse_non_negative_int(
                    timing["token_index"], "token_index", row_number
                )
                token_id = _parse_non_negative_int(
                    timing["token_id"], "token_id", row_number
                )
                phase = timing["phase"]
                if phase not in {"prefill", "decode"}:
                    raise ValueError(
                        f"Row {row_number}, token {token_index}: "
                        f"phase must be 'prefill' or 'decode', got {phase!r}"
                    )
                latency = _parse_latency(
                    timing["latency_seconds"], row_number, token_index
                )

                coordinate = (video_index, question_index, token_index)
                if coordinate in seen_coordinates:
                    raise ValueError(
                        "Duplicate timing coordinate detected: "
                        f"(video={video_index}, question={question_index}, "
                        f"token={token_index})"
                    )
                seen_coordinates.add(coordinate)

                detail_rows.append(
                    {
                        "video_index": video_index,
                        "video_id": video_id,
                        "question_index": question_index,
                        "token_index": token_index,
                        "token_id": token_id,
                        "phase": phase,
                        "latency_seconds": latency,
                    }
                )

    detail_rows.sort(
        key=lambda row: (
            row["video_index"],
            row["question_index"],
            row["token_index"],
        )
    )
    return detail_rows, videos


def build_summary(detail_rows, videos):
    decode_latencies_by_video = {video_index: [] for video_index in videos}
    for row in detail_rows:
        if row["phase"] == "decode":
            decode_latencies_by_video[row["video_index"]].append(
                row["latency_seconds"]
            )

    summary_rows = []
    video_means = []
    all_decode_latencies = []
    for video_index in sorted(videos):
        latencies = decode_latencies_by_video[video_index]
        all_decode_latencies.extend(latencies)
        mean_seconds = fmean(latencies) if latencies else None
        if mean_seconds is not None:
            video_means.append(mean_seconds)
        summary_rows.append(
            {
                "scope": "video",
                "video_index": video_index,
                "video_id": videos[video_index],
                "video_count": 1,
                "decode_token_count": len(latencies),
                "mean_decode_latency_seconds": mean_seconds,
                "mean_decode_latency_ms": (
                    mean_seconds * 1000 if mean_seconds is not None else None
                ),
            }
        )

    token_weighted_mean = (
        fmean(all_decode_latencies) if all_decode_latencies else None
    )
    video_macro_mean = fmean(video_means) if video_means else None
    summary_rows.extend(
        [
            {
                "scope": "dataset_token_weighted",
                "video_index": "",
                "video_id": "ALL",
                "video_count": len(videos),
                "decode_token_count": len(all_decode_latencies),
                "mean_decode_latency_seconds": token_weighted_mean,
                "mean_decode_latency_ms": (
                    token_weighted_mean * 1000
                    if token_weighted_mean is not None
                    else None
                ),
            },
            {
                "scope": "dataset_video_macro",
                "video_index": "",
                "video_id": "ALL",
                "video_count": len(video_means),
                "decode_token_count": len(all_decode_latencies),
                "mean_decode_latency_seconds": video_macro_mean,
                "mean_decode_latency_ms": (
                    video_macro_mean * 1000
                    if video_macro_mean is not None
                    else None
                ),
            },
        ]
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
        return index, detail_path, summary_path, lock_path


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
    print("Per-video mean decode latency:")
    for row in summary_rows:
        if row["scope"] != "video":
            continue
        print(
            f"  video_index={row['video_index']} video_id={row['video_id']} "
            f"decode_tokens={row['decode_token_count']} "
            f"mean={_format_mean(row['mean_decode_latency_seconds'])}"
        )

    token_weighted = next(
        row for row in summary_rows if row["scope"] == "dataset_token_weighted"
    )
    video_macro = next(
        row for row in summary_rows if row["scope"] == "dataset_video_macro"
    )
    print(
        "Dataset token-weighted mean: "
        f"{_format_mean(token_weighted['mean_decode_latency_seconds'])}"
    )
    print(
        "Dataset video-macro mean: "
        f"{_format_mean(video_macro['mean_decode_latency_seconds'])}"
    )
    print(f"Token timing detail: {detail_path}")
    print(f"Token timing summary: {summary_path}")


def analyze_results(
    results_path,
    output_dir=None,
    detail_prefix="token_timings",
    summary_prefix="token_timing_summary",
):
    results_path = Path(results_path)
    if output_dir is None:
        output_dir = results_path.parent

    detail_rows, videos = load_token_timings(results_path)
    summary_rows = build_summary(detail_rows, videos)
    _, detail_path, summary_path, lock_path = reserve_output_paths(
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
            "Expand token timing metadata from a complete HERMES results.csv "
            "and calculate per-video and dataset-level decode latency."
        )
    )
    parser.add_argument(
        "--results_path",
        type=Path,
        required=True,
        help="Path to the merged results.csv for one complete dataset run.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the directory containing results_path.",
    )
    parser.add_argument(
        "--detail_prefix",
        type=str,
        default="token_timings",
        help="Detail CSV prefix; the script appends the smallest free _N.csv suffix.",
    )
    parser.add_argument(
        "--summary_prefix",
        type=str,
        default="token_timing_summary",
        help="Summary CSV prefix; it shares the detail file's numeric suffix.",
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
