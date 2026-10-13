from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds


BG = Color.BLACK
LINE_COLOR = Color.GREY
ORIENTATIONS = ("main", "anti")
OBJECT_COLORS = [color for color in Color.NOT_BLACK if color != LINE_COLOR]
MAX_GENERATION_ATTEMPTS = 400
MAX_ERASE_ATTEMPTS = 24


def _reflect_point(point: tuple[int, int], orientation: str, const: int) -> tuple[int, int]:
    r, c = map(int, point)
    if orientation == "main":
        return int(c + const), int(r - const)
    return int(const - c), int(const - r)


def _signed_offset(point: tuple[int, int], orientation: str, const: int) -> int:
    r, c = map(int, point)
    if orientation == "main":
        return int(r - c - const)
    return int(r + c - const)


def _axis_constants(shape: tuple[int, int], orientation: str) -> range:
    height, width = map(int, shape)
    if orientation == "main":
        return range(-(width - 1), height)
    return range(height + width - 1)


def _axis_cells(shape: tuple[int, int], orientation: str, const: int) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    cells: list[tuple[int, int]] = []
    if orientation == "main":
        for r in range(height):
            c = r - int(const)
            if 0 <= c < width:
                cells.append((int(r), int(c)))
    else:
        for r in range(height):
            c = int(const) - r
            if 0 <= c < width:
                cells.append((int(r), int(c)))
    return cells


def _segment_signature(coords: np.ndarray) -> tuple[str, int] | None:
    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) < 2:
        return None

    rows = sorted(r for r, _c in pts)
    if len(rows) != len(set(rows)):
        return None
    if rows != list(range(rows[0], rows[-1] + 1)):
        return None

    main_consts = {r - c for r, c in pts}
    if len(main_consts) == 1:
        const = int(next(iter(main_consts)))
        if all((r, r - const) in pts for r in rows):
            return "main", const

    anti_consts = {r + c for r, c in pts}
    if len(anti_consts) == 1:
        const = int(next(iter(anti_consts)))
        if all((r, const - r) in pts for r in rows):
            return "anti", const

    return None


def _complete_object(object_grid: np.ndarray, axis: tuple[str, int]) -> np.ndarray | None:
    orientation, const = axis
    out = np.asarray(object_grid, dtype=int).copy()
    for r0, c0 in np.argwhere(out != BG):
        r, c = int(r0), int(c0)
        rr, cc = _reflect_point((r, c), orientation, const)
        if not in_bounds(out.shape, rr, cc):
            return None
        color = int(out[r, c])
        if out[rr, cc] not in (BG, color):
            return None
        if out[rr, cc] == BG:
            out[rr, cc] = color
    return out


def _candidate_completions(object_grid: np.ndarray) -> list[tuple[tuple[str, int], np.ndarray]]:
    candidates: list[tuple[tuple[str, int], np.ndarray]] = []
    for orientation in ORIENTATIONS:
        for const in _axis_constants(object_grid.shape, orientation):
            out = _complete_object(object_grid, (orientation, int(const)))
            if out is not None:
                candidates.append(((orientation, int(const)), out))
    return candidates


def _unique_completed_outputs(candidates: list[tuple[tuple[str, int], np.ndarray]]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for _axis, out in candidates:
        if not any(np.array_equal(out, existing) for existing in unique):
            unique.append(out)
    return unique


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    expanded = np.zeros_like(mask, dtype=bool)
    for r0, c0 in np.argwhere(mask):
        r, c = int(r0), int(c0)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if in_bounds(mask.shape, rr, cc):
                    expanded[rr, cc] = True
    return expanded


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if in_bounds(blocked.shape, rr, cc):
                    blocked[rr, cc] = True


def _segment_candidates(
    shape: tuple[int, int],
    blocked: np.ndarray,
    orientation: str,
    length: int,
) -> list[list[tuple[int, int]]]:
    height, width = map(int, shape)
    candidates: list[list[tuple[int, int]]] = []
    if orientation == "main":
        for r0 in range(height - length + 1):
            for c0 in range(width - length + 1):
                coords = [(int(r0 + i), int(c0 + i)) for i in range(length)]
                if all(not blocked[r, c] for r, c in coords):
                    candidates.append(coords)
    else:
        for r0 in range(height - length + 1):
            for c0 in range(length - 1, width):
                coords = [(int(r0 + i), int(c0 - i)) for i in range(length)]
                if all(not blocked[r, c] for r, c in coords):
                    candidates.append(coords)
    return candidates


def _true_axis_candidates(
    line: list[tuple[int, int]],
    obj_start: int,
    obj_len: int,
    blocked: np.ndarray,
    length: int,
) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    obj_end = obj_start + obj_len - 1
    for start in range(len(line) - length + 1):
        end = start + length - 1
        separated = end <= obj_start - 3 or start >= obj_end + 3
        if not separated:
            continue
        coords = line[start : start + length]
        if all(not blocked[r, c] for r, c in coords):
            candidates.append(coords)
    return candidates


def _extract_object_and_segments(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int | np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    object_comp = max(comps, key=lambda comp: int(comp["area"]))
    object_grid = np.full_like(g, bg)
    obj_coords = object_comp["coords"]
    object_grid[obj_coords[:, 0], obj_coords[:, 1]] = g[obj_coords[:, 0], obj_coords[:, 1]]

    segments: list[dict[str, int | np.ndarray]] = []
    for comp in comps:
        if comp is object_comp:
            continue
        coords = comp["coords"]
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1, "each guide segment must be monochromatic"
        signature = _segment_signature(coords)
        assert signature is not None, "every extra component must be a straight diagonal segment"
        orientation, const = signature
        segments.append(
            {
                "coords": coords,
                "orientation": orientation,
                "const": int(const),
                "length": int(coords.shape[0]),
                "color": int(colors[0]),
            }
        )

    assert segments, "expected at least one guide segment"
    return object_grid, segments


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Use the voted diagonal axis to mirror-complete the damaged object, then erase all guide bars.

    Input structure:
    - The background is black.
    - One large multicolor foreground object is missing some mirrored pixels.
    - All other foreground components are isolated one-cell-thick diagonal bars in the two diagonal orientations.

    Rule:
    - Separate the large foreground object from the scattered diagonal bars.
    - Sum bar lengths by diagonal orientation and pick the majority orientation.
    - Within that orientation, take the unique longest bar; extend its supporting diagonal line across the grid and use it as the mirror axis.
    - Mirror every surviving object pixel across that axis, filling only background cells, and remove every bar from the output.

    Key insight:
    - The bars act as a vote plus a key: orientation majority chooses the diagonal family, and the longest bar inside that family chooses the exact axis position.
    """
    object_grid, segments = _extract_object_and_segments(np.asarray(input_grid, dtype=int))

    totals = {orientation: 0 for orientation in ORIENTATIONS}
    for seg in segments:
        totals[str(seg["orientation"])] += int(seg["length"])
    assert totals["main"] != totals["anti"], "bar orientations must have a unique majority"
    majority = "main" if totals["main"] > totals["anti"] else "anti"

    majority_segments = [seg for seg in segments if seg["orientation"] == majority]
    longest = max(int(seg["length"]) for seg in majority_segments)
    longest_segments = [seg for seg in majority_segments if int(seg["length"]) == longest]
    assert len(longest_segments) == 1, "majority orientation must have a unique longest bar"
    axis = (majority, int(longest_segments[0]["const"]))

    completed = _complete_object(object_grid, axis)
    assert completed is not None, "the selected axis must produce a valid completion"
    return completed


def _connected_object(grid: np.ndarray) -> bool:
    comps = find_components(grid, bg=BG, connectivity=8, monochromatic=False, return_format="dict")
    return len(comps) == 1


def _build_object(
    shape: tuple[int, int],
    axis: tuple[str, int],
    rng: random.Random,
) -> dict[str, object] | None:
    orientation, const = axis
    line = _axis_cells(shape, orientation, const)
    if len(line) < 15:
        return None

    max_obj_len = min(10, len(line) - 6)
    if max_obj_len < 6:
        return None
    obj_len = rng.randint(6, max_obj_len)
    start_min = 2
    start_max = len(line) - obj_len - 2
    if start_max < start_min:
        return None
    obj_start = rng.randint(start_min, start_max)
    axis_run = line[obj_start : obj_start + obj_len]
    palette = rng.sample(OBJECT_COLORS, rng.randint(3, 5))

    full = np.zeros(shape, dtype=int)
    pair_records: list[dict[str, object]] = []
    for idx, (r, c) in enumerate(axis_run):
        full[r, c] = int(palette[idx % len(palette)])
        for family, step_sign in (("south", 1), ("north", -1)):
            feasible_depths: list[int] = []
            for depth in range(1, 4):
                src = (int(r + step_sign * depth), int(c))
                dst = _reflect_point(src, orientation, const)
                if in_bounds(shape, *src) and in_bounds(shape, *dst):
                    feasible_depths.append(depth)
            if not feasible_depths:
                continue

            min_depth = 1 if (family == "south" and idx % 2 == 0) else 0
            min_depth = min(min_depth, feasible_depths[-1])
            depth_count = rng.randint(min_depth, feasible_depths[-1])
            for depth in range(1, depth_count + 1):
                src = (int(r + step_sign * depth), int(c))
                dst = _reflect_point(src, orientation, const)
                color = int(palette[(idx + depth + (0 if family == "south" else 1)) % len(palette)])
                full[src] = color
                full[dst] = color
                pair_records.append(
                    {
                        "index": int(idx),
                        "family": family,
                        "depth": int(depth),
                        "cells": (src, dst),
                    }
                )

    if len(pair_records) < 10:
        return None
    if np.count_nonzero(full) < 28:
        return None
    if len(set(map(int, full[full != BG]))) < 3:
        return None
    if not _connected_object(full):
        return None

    for _ in range(MAX_ERASE_ATTEMPTS):
        partial = full.copy()
        remove_sign = rng.choice((-1, 1))
        interval_len = rng.randint(max(2, obj_len // 2), obj_len - 1)
        interval_start = rng.randint(0, obj_len - interval_len)
        chosen_families = rng.sample(["south", "north"], k=rng.randint(1, 2))
        min_depth = rng.randint(1, 2)
        removed = 0

        for rec in pair_records:
            idx = int(rec["index"])
            if not (interval_start <= idx < interval_start + interval_len):
                continue
            if str(rec["family"]) not in chosen_families:
                continue
            if int(rec["depth"]) < min_depth:
                continue
            for cell in rec["cells"]:
                if np.sign(_signed_offset(cell, orientation, const)) == remove_sign and partial[cell] != BG:
                    partial[cell] = BG
                    removed += 1

        if removed < 6:
            continue
        if np.count_nonzero(partial) >= np.count_nonzero(full):
            continue
        if not _connected_object(partial):
            continue

        true_completion = _complete_object(partial, axis)
        if true_completion is None or not np.array_equal(true_completion, full):
            continue
        unique_outputs = _unique_completed_outputs(_candidate_completions(partial))
        if len(unique_outputs) < 2:
            continue

        return {
            "full": full,
            "partial": partial,
            "line": line,
            "obj_start": int(obj_start),
            "obj_len": int(obj_len),
        }

    return None


def _segment_plan(
    rng: random.Random,
    true_len: int,
) -> tuple[list[int], list[int], bool]:
    other_major = [rng.randint(2, true_len - 1) for _ in range(rng.randint(2, 3))]
    majority_total = true_len + sum(other_major)
    decoy_mode = rng.random() < 0.55

    if decoy_mode:
        decoy_len = min(7, true_len + rng.randint(1, 2))
        minority = [decoy_len]
        remaining = majority_total - decoy_len - 1
        if remaining >= 2 and rng.random() < 0.5:
            minority.append(rng.randint(2, min(4, remaining)))
    else:
        minority = [rng.randint(2, true_len - 1) for _ in range(rng.randint(1, 2))]

    while sum(minority) >= majority_total:
        if len(minority) > 1:
            minority.pop()
        else:
            minority[0] = max(2, minority[0] - 1)
    return other_major, minority, decoy_mode


def _paint_segments(grid: np.ndarray, segments: list[list[tuple[int, int]]]) -> None:
    for coords in segments:
        for r, c in coords:
            grid[r, c] = LINE_COLOR


def _place_segments(
    partial: np.ndarray,
    full: np.ndarray,
    axis: tuple[str, int],
    meta: dict[str, object],
    rng: random.Random,
) -> tuple[list[list[tuple[int, int]]], bool] | None:
    orientation, _const = axis
    blocked = _expand_mask(full != BG)
    true_candidates_by_len: list[tuple[int, list[list[tuple[int, int]]]]] = []
    for true_len in range(4, 7):
        candidates = _true_axis_candidates(
            meta["line"],
            int(meta["obj_start"]),
            int(meta["obj_len"]),
            blocked,
            true_len,
        )
        if candidates:
            true_candidates_by_len.append((true_len, candidates))
    if not true_candidates_by_len:
        return None

    true_len, true_candidates = rng.choice(true_candidates_by_len)
    placed: list[list[tuple[int, int]]] = []
    true_segment = rng.choice(true_candidates)
    placed.append(true_segment)
    _mark_blocked(blocked, true_segment)

    other_major, minority, decoy_mode = _segment_plan(rng, true_len)

    for length in other_major:
        candidates = _segment_candidates(partial.shape, blocked, orientation, length)
        if not candidates:
            return None
        coords = rng.choice(candidates)
        placed.append(coords)
        _mark_blocked(blocked, coords)

    minority_orientation = "anti" if orientation == "main" else "main"
    for length in minority:
        candidates = _segment_candidates(partial.shape, blocked, minority_orientation, length)
        if not candidates:
            return None
        coords = rng.choice(candidates)
        placed.append(coords)
        _mark_blocked(blocked, coords)

    return placed, decoy_mode


def _analyze_instance(grid: np.ndarray) -> dict[str, object]:
    object_grid, segments = _extract_object_and_segments(grid)
    totals = {orientation: 0 for orientation in ORIENTATIONS}
    max_len = {orientation: 0 for orientation in ORIENTATIONS}
    for seg in segments:
        orientation = str(seg["orientation"])
        length = int(seg["length"])
        totals[orientation] += length
        max_len[orientation] = max(max_len[orientation], length)
    majority = "main" if totals["main"] > totals["anti"] else "anti"
    majority_segments = [seg for seg in segments if seg["orientation"] == majority]
    longest = max(int(seg["length"]) for seg in majority_segments)
    axis = (majority, int(next(seg["const"] for seg in majority_segments if int(seg["length"]) == longest)))
    return {
        "object_grid": object_grid,
        "segments": segments,
        "totals": totals,
        "max_len": max_len,
        "axis": axis,
    }


def _generate_case(rng: random.Random | None = None) -> dict[str, object]:
    rng = rng or random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = rng.randint(18, 26)
        width = rng.randint(18, 26)
        shape = (height, width)

        valid_axes = [
            (orientation, const)
            for orientation in ORIENTATIONS
            for const in _axis_constants(shape, orientation)
            if len(_axis_cells(shape, orientation, const)) >= 15
        ]
        if not valid_axes:
            continue
        axis = rng.choice(valid_axes)
        meta = _build_object(shape, axis, rng)
        if meta is None:
            continue

        segment_info = _place_segments(meta["partial"], meta["full"], axis, meta, rng)
        if segment_info is None:
            continue
        segments, decoy_mode = segment_info

        inp = np.asarray(meta["partial"], dtype=int).copy()
        _paint_segments(inp, segments)

        analysis = _analyze_instance(inp)
        if tuple(analysis["axis"]) != axis:
            continue
        if analysis["totals"]["main"] == analysis["totals"]["anti"]:
            continue
        majority = axis[0]
        minority = "anti" if majority == "main" else "main"
        if analysis["max_len"][majority] <= 0:
            continue
        if decoy_mode and analysis["max_len"][minority] <= analysis["max_len"][majority]:
            continue
        if np.count_nonzero(inp == LINE_COLOR) < 8:
            continue
        if not np.array_equal(main(inp), meta["full"]):
            continue

        unique_outputs = _unique_completed_outputs(_candidate_completions(meta["partial"]))
        if len(unique_outputs) < 2:
            continue

        return {
            "input": inp,
            "output": np.asarray(meta["full"], dtype=int),
            "axis": axis,
            "decoy_mode": bool(decoy_mode),
        }
    raise RuntimeError("failed to generate a valid ARC input")


def generate_input() -> np.ndarray:
    return np.asarray(_generate_case()["input"], dtype=int)
