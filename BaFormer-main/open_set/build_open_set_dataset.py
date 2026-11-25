#!/usr/bin/env python3
"""Materialize the paper-style open-set TAS protocol for BaFormer.

This script only constructs the protocol.  It does not alter BaFormer or train
a model.  The source dataset remains read-only and the generated directory
contains labels, masks, targets, split manifests, and a symlink to features.

For the GTEA protocol from *Learning Action Distribution Flow for Open-Set
Temporal Action Segmentation* (LADF), the nested unknown action sets are:

    10%: take
    20%: take, open
    30%: take, open, put
    40%: take, open, put, pour

``background`` is deliberately kept as a known class.  Training targets use
``-100`` for frames whose action class was held out.  Test targets use one
additional class id for ``unknown``.  This keeps the original temporal input
while making the train/test supervision policy explicit for a future BaFormer
wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


LADF_GTEA_UNKNOWN_ACTIONS: Mapping[int, tuple[str, ...]] = {
    10: ("take",),
    20: ("take", "open"),
    30: ("take", "open", "put"),
    40: ("take", "open", "put", "pour"),
}

IGNORE_TARGET = -100
UNKNOWN_LABEL = "unknown"


def read_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    action_to_id: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        index_text, action = line.split(maxsplit=1)
        if action in action_to_id:
            raise ValueError(f"Duplicate action {action!r} in {path}")
        action_to_id[action] = int(index_text)
    if not action_to_id:
        raise ValueError(f"No actions found in {path}")
    ids = sorted(action_to_id.values())
    if ids != list(range(len(ids))):
        raise ValueError(f"Class ids in {path} must be contiguous from zero: {ids}")
    return action_to_id, {index: action for action, index in action_to_id.items()}


def read_bundle(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Empty split file: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate video names in {path}")
    return names


def write_mapping(path: Path, actions: Sequence[str]) -> None:
    path.write_text("".join(f"{index} {action}\n" for index, action in enumerate(actions)), encoding="utf-8")


def choose_unknown_actions(
    dataset: str,
    action_to_id: Mapping[str, int],
    percentage: int | None,
    explicit: str | None,
) -> list[str]:
    if percentage is not None and explicit:
        raise ValueError("Use either --unknown-percentage or --unknown-actions")
    if dataset == "gtea" and percentage is not None:
        try:
            selected = list(LADF_GTEA_UNKNOWN_ACTIONS[percentage])
        except KeyError as exc:
            raise ValueError("GTEA percentage must be one of 10, 20, 30, 40") from exc
    elif explicit:
        selected = list(dict.fromkeys(item.strip() for item in explicit.split(",") if item.strip()))
    else:
        raise ValueError("Provide --unknown-percentage or --unknown-actions")
    if not selected:
        raise ValueError("At least one unknown action must be selected")
    missing = [action for action in selected if action not in action_to_id]
    if missing:
        raise ValueError(f"Unknown actions absent from mapping.txt: {missing}")
    if "background" in selected:
        raise ValueError("background is a known class in the LADF TAS protocol")
    if len(selected) >= len(action_to_id):
        raise ValueError("At least one known action is required")
    return selected


def ensure_output_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}. Choose a new --output-root")
    path.mkdir(parents=True)


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)


def copy_split_files(source_splits: Path, destination_splits: Path) -> list[str]:
    destination_splits.mkdir()
    copied: list[str] = []
    for source in sorted(source_splits.glob("*.bundle")):
        if source.is_file():
            shutil.copy2(source, destination_splits / source.name)
            copied.append(source.name)
    if not copied:
        raise ValueError(f"No .bundle files found in {source_splits}")
    return copied


def all_split_entries(source_splits: Path) -> list[tuple[str, int, str]]:
    entries: list[tuple[str, int, str]] = []
    for kind in ("train", "test"):
        for bundle in sorted(source_splits.glob(f"{kind}.split*.bundle")):
            suffix = bundle.name.removeprefix(f"{kind}.split").removesuffix(".bundle")
            try:
                split = int(suffix)
            except ValueError as exc:
                raise ValueError(f"Unexpected split filename: {bundle.name}") from exc
            entries.extend((name, split, kind) for name in read_bundle(bundle))
    if not entries:
        raise ValueError(f"No train/test split entries found in {source_splits}")
    return entries


def read_labels(path: Path) -> list[str]:
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"Empty ground-truth file: {path}")
    return labels


def build_one(
    source_root: Path,
    output_root: Path,
    dataset: str,
    unknown_actions: Sequence[str],
    percentage: int | None,
) -> dict:
    mapping_path = source_root / "mapping.txt"
    source_features = source_root / "features"
    source_ground_truth = source_root / "groundTruth"
    source_splits = source_root / "splits"
    for path in (mapping_path, source_features, source_ground_truth, source_splits):
        if not path.exists():
            raise FileNotFoundError(path)

    full_mapping, _ = read_mapping(mapping_path)
    if UNKNOWN_LABEL in full_mapping or "<ignore>" in full_mapping:
        raise ValueError("Source mapping cannot contain the reserved labels 'unknown' or '<ignore>'")
    unknown_set = set(unknown_actions)
    known_actions = [
        action for action, _ in sorted(full_mapping.items(), key=lambda item: item[1])
        if action not in unknown_set
    ]
    known_mapping = {action: index for index, action in enumerate(known_actions)}
    eval_actions = known_actions + [UNKNOWN_LABEL]
    eval_mapping = {action: index for index, action in enumerate(eval_actions)}

    entries = all_split_entries(source_splits)
    partitions: dict[int, dict[str, set[str]]] = {}
    for video_name, split, partition in entries:
        partitions.setdefault(split, {}).setdefault(partition, set()).add(video_name)
    for split, sets in partitions.items():
        if set(sets) != {"train", "test"}:
            raise ValueError(f"Split {split} requires both train and test bundles")
        overlap = sets["train"] & sets["test"]
        if overlap:
            raise ValueError(f"Train/test overlap in split {split}: {sorted(overlap)}")

    ensure_output_dir(output_root)
    (output_root / "labels" / "train").mkdir(parents=True)
    (output_root / "labels" / "test").mkdir(parents=True)
    (output_root / "targets" / "train").mkdir(parents=True)
    (output_root / "targets" / "test").mkdir(parents=True)
    (output_root / "supervision_masks" / "train").mkdir(parents=True)
    copied_splits = copy_split_files(source_splits, output_root / "splits")
    relative_symlink(source_features, output_root / "features")

    manifest: list[dict] = []
    seen_partition: set[tuple[str, str]] = set()
    split_counts: dict[int, dict[str, dict[str, int]]] = {}
    for video_name, split, partition in entries:
        key = (partition, video_name)
        label_path = source_ground_truth / video_name
        labels = read_labels(label_path)
        feature_path = source_features / f"{Path(video_name).stem}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(feature_path)
        features = np.load(feature_path, mmap_mode="r")
        if features.ndim != 2 or features.shape != (2048, len(labels)):
            raise ValueError(
                f"Feature/label mismatch for {video_name}: features={features.shape}, labels={len(labels)}"
            )
        unknown_mask = np.asarray([label in unknown_set for label in labels], dtype=np.bool_)
        if any(label not in full_mapping for label in labels):
            bad = sorted({label for label in labels if label not in full_mapping})
            raise ValueError(f"Labels absent from mapping.txt in {video_name}: {bad}")
        train_targets = np.asarray(
            [known_mapping[label] if not is_unknown else IGNORE_TARGET for label, is_unknown in zip(labels, unknown_mask)],
            dtype=np.int64,
        )
        eval_labels = [UNKNOWN_LABEL if is_unknown else label for label, is_unknown in zip(labels, unknown_mask)]
        eval_targets = np.asarray([eval_mapping[label] for label in eval_labels], dtype=np.int64)
        label_dir = output_root / "labels" / partition
        target_dir = output_root / "targets" / partition
        output_label_path = label_dir / video_name
        output_target_path = target_dir / f"{Path(video_name).stem}.npy"
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        if key not in seen_partition:
            train_labels = ["<ignore>" if is_unknown else label for label, is_unknown in zip(labels, unknown_mask)]
            output_label_path.write_text("".join(f"{label}\n" for label in (train_labels if partition == "train" else eval_labels)), encoding="utf-8")
            np.save(output_target_path, train_targets if partition == "train" else eval_targets)
            if partition == "train":
                np.save(output_root / "supervision_masks" / "train" / f"{Path(video_name).stem}.npy", (~unknown_mask).astype(np.uint8))
            seen_partition.add(key)
        unknown_count = int(unknown_mask.sum())
        counts = split_counts.setdefault(split, {}).setdefault(partition, {"videos": 0, "frames": 0, "unknown_frames": 0})
        counts["videos"] += 1
        counts["frames"] += len(labels)
        counts["unknown_frames"] += unknown_count
        manifest.append({
            "video": video_name,
            "split": split,
            "partition": partition,
            "source_labels": str(label_path),
            "generated_labels": str(output_label_path),
            "feature_shape": [int(features.shape[0]), int(features.shape[1])],
            "frames": len(labels),
            "unknown_frames": unknown_count,
            "known_frames": len(labels) - unknown_count,
        })

    write_mapping(output_root / "mapping_known.txt", known_actions)
    write_mapping(output_root / "mapping_eval.txt", eval_actions)
    (output_root / "unknown_actions.txt").write_text("".join(f"{action}\n" for action in unknown_actions), encoding="utf-8")
    metadata = {
        "protocol": "ladf_tas_class_holdout" if dataset == "gtea" else "class_holdout",
        "dataset": dataset,
        "source_root": str(source_root.resolve()),
        "unknown_percentage": percentage,
        "unknown_actions": list(unknown_actions),
        "known_actions": known_actions,
        "background_is_known": "background" in full_mapping and "background" not in unknown_set,
        "unknown_label": UNKNOWN_LABEL,
        "ignore_target": IGNORE_TARGET,
        "train_policy": "unknown action frames stay in the video but have target -100 and supervision mask 0",
        "test_policy": "unknown action frames are relabeled as one evaluation class named unknown",
        "transition_policy": "original frame labels are preserved; no extra transition class is introduced",
        "feature_policy": "features are exposed through a relative symlink and are never copied or modified",
        "copied_split_files": copied_splits,
        "total_videos": len({item[0] for item in entries}),
        "split_counts": split_counts,
        "manifest": "manifest.json",
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, required=True, help="Original dataset directory, e.g. .../TAS_dataset/gtea")
    parser.add_argument("--output-root", type=Path, required=True, help="Base directory for generated open-set datasets")
    parser.add_argument("--dataset", default="gtea", help="Dataset name used for protocol metadata")
    parser.add_argument("--unknown-percentage", type=int, choices=(10, 20, 30, 40), default=None)
    parser.add_argument("--unknown-actions", default=None, help="Comma-separated explicit held-out action classes")
    parser.add_argument("--all-percentages", action="store_true", help="Build all four nested GTEA LADF settings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all_percentages and (args.unknown_percentage is not None or args.unknown_actions):
        raise ValueError("--all-percentages cannot be combined with a single unknown setting")
    percentages = [10, 20, 30, 40] if args.all_percentages else [args.unknown_percentage]
    for percentage in percentages:
        unknown = choose_unknown_actions(args.dataset, read_mapping(args.source_root / "mapping.txt")[0], percentage, args.unknown_actions)
        output = args.output_root / (f"pct{percentage}" if percentage is not None else "explicit")
        metadata = build_one(args.source_root, output, args.dataset, unknown, percentage)
        print(json.dumps({"output": str(output), "unknown_actions": unknown, "split_counts": metadata["split_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
