from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

REGION_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE, TEAL)
FOLD_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MAX_GEN_ATTEMPTS = 400


def _as_int_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return g


def _separator_rows_cols(g: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [int(r) for r in range(g.shape[0]) if bool(np.all(g[r, :] == GREY))]
    cols = [int(c) for c in range(g.shape[1]) if bool(np.all(g[:, c] == GREY))]
    assert rows and cols, "expected full grey separator rows and columns"
    return rows, cols


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start <= sep - 1:
            out.append((int(start), int(sep - 1)))
        start = int(sep + 1)
    if start <= size - 1:
        out.append((int(start), int(size - 1)))
    return out


def _region_index_grid(
    shape: tuple[int, int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> tuple[list[dict], np.ndarray]:
    H, W = shape
    region_idx = np.full((H, W), -1, dtype=int)
    regions: list[dict] = []
    rid = 0
    for ri, (r0, r1) in enumerate(row_bands):
        for cj, (c0, c1) in enumerate(col_bands):
            region_idx[r0 : r1 + 1, c0 : c1 + 1] = rid
            regions.append(
                {
                    "id": rid,
                    "ri": int(ri),
                    "cj": int(cj),
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
            rid += 1
    return regions, region_idx


def _region_base_color(g: np.ndarray, region: dict) -> int:
    patch = g[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1]
    vals = [int(v) for v in patch.ravel() if int(v) not in {GREY, PINK, MAROON}]
    counts = Counter(vals)
    assert counts, "region must contain a visible base color"
    color, _ = counts.most_common(1)[0]
    return int(color)


def _step_sign(a: int, b: int) -> int:
    if a < b:
        return 1
    if a > b:
        return -1
    return 0


def _line_cells_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = _step_sign(a[0], b[0])
    dc = _step_sign(a[1], b[1])
    cur = a
    out = [a]
    while cur != b:
        cur = (int(cur[0] + dr), int(cur[1] + dc))
        out.append(cur)
    return out


def _fold_direction(shape: tuple[int, int], hinge: tuple[int, int], orientation: str) -> tuple[int, int]:
    H, W = shape
    hr, hc = hinge
    if orientation == "h":
        up_space = int(hr)
        down_space = int(H - 1 - hr)
        assert up_space != down_space, "horizontal hinge must prefer one vertical side"
        return (-1, 0) if up_space > down_space else (1, 0)
    left_space = int(hc)
    right_space = int(W - 1 - hc)
    assert left_space != right_space, "vertical hinge must prefer one horizontal side"
    return (0, -1) if left_space > right_space else (0, 1)


def _first_target_region(
    g: np.ndarray,
    region_idx: np.ndarray,
    source_region_id: int,
    hinge: tuple[int, int],
    fold_step: tuple[int, int],
) -> int:
    H, W = g.shape
    rr, cc = hinge
    crossed_separator = False
    for _ in range(H + W):
        rr += int(fold_step[0])
        cc += int(fold_step[1])
        assert 0 <= rr < H and 0 <= cc < W, "fold must stay in bounds"
        if int(region_idx[rr, cc]) == int(source_region_id):
            continue
        if int(g[rr, cc]) == GREY:
            crossed_separator = True
            continue
        rid = int(region_idx[rr, cc])
        assert crossed_separator and rid >= 0 and rid != int(source_region_id), "fold must enter a neighboring region"
        return rid
    raise AssertionError("failed to locate fold target region")


def _parse_rod(g: np.ndarray, region: dict, region_idx: np.ndarray) -> dict | None:
    r0, r1, c0, c1 = region["r0"], region["r1"], region["c0"], region["c1"]
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.isin(patch, [GREY, PINK, MAROON])
    if not bool(mask.any()):
        return None

    coords_local = np.argwhere(mask)
    coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in coords_local]
    vals = [int(g[r, c]) for r, c in coords]
    assert vals.count(MAROON) == 1, "each rod needs exactly one endpoint marker"
    assert vals.count(GREY) == 1, "each rod needs exactly one hinge marker"
    assert all(v in {GREY, PINK, MAROON} for v in vals)
    endpoint = next((r, c) for r, c in coords if int(g[r, c]) == MAROON)
    hinge = next((r, c) for r, c in coords if int(g[r, c]) == GREY)

    row_set = {r for r, _ in coords}
    col_set = {c for _, c in coords}
    if len(row_set) == 1:
        orientation = "h"
        line_r = next(iter(row_set))
        expected = {(line_r, c) for c in range(c0, c1 + 1)}
    else:
        assert len(col_set) == 1, "rod must be straight"
        orientation = "v"
        line_c = next(iter(col_set))
        expected = {(r, line_c) for r in range(r0, r1 + 1)}
    assert set(coords) == expected, "rod must span its region fully"

    all_line = _line_cells_between(endpoint, hinge)
    fixed_reds = [cell for cell in all_line[1:-1]]
    dr = _step_sign(endpoint[0], hinge[0])
    dc = _step_sign(endpoint[1], hinge[1])
    tail_step = (dr, dc)
    tail: list[tuple[int, int]] = []
    rr, cc = hinge
    while True:
        rr += int(tail_step[0])
        cc += int(tail_step[1])
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            break
        assert int(g[rr, cc]) == PINK, "tail must contain only magenta pixels"
        tail.append((int(rr), int(cc)))
    assert tail, "rod needs a movable tail"

    fold_step = _fold_direction(g.shape, hinge, orientation)
    target_region_id = _first_target_region(g, region_idx, int(region["id"]), hinge, fold_step)
    folded = [
        (int(hinge[0] + fold_step[0] * k), int(hinge[1] + fold_step[1] * k))
        for k in range(1, len(tail) + 1)
    ]
    return {
        "source_region_id": int(region["id"]),
        "source_color": int(region["color"]),
        "source_area": int(region["area"]),
        "endpoint": endpoint,
        "hinge": hinge,
        "fixed_reds": fixed_reds,
        "tail": tail,
        "tail_len": int(len(tail)),
        "orientation": orientation,
        "fold_step": fold_step,
        "folded": folded,
        "target_region_id": int(target_region_id),
    }


def _analyze_grid(g: np.ndarray) -> tuple[list[dict], np.ndarray, list[dict]]:
    sep_rows, sep_cols = _separator_rows_cols(g)
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    regions, region_idx = _region_index_grid(g.shape, row_bands, col_bands)
    for region in regions:
        region["color"] = _region_base_color(g, region)
    rods: list[dict] = []
    for region in regions:
        rod = _parse_rod(g, region, region_idx)
        if rod is not None:
            rods.append(rod)
    assert 2 <= len(rods) <= 4, "expected 2-4 rods"
    return regions, region_idx, rods


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked full-span rod and recolor the first neighboring region it points into.

    Input structure:
    - Grey full rows and columns partition the grid into 4-6 rectangular monochrome regions.
    - Each region has one base color chosen from the non-marker palette, and 2-4 regions contain a full-width or full-height magenta rod.
    - Every rod has one maroon endpoint marker at one end and one grey hinge marker at an interior position.

    Rule:
    - Keep the segment from the maroon endpoint through the grey hinge fixed.
    - Move the tail on the far side of the hinge to a perpendicular ray whose direction is the grid side with more space from the hinge.
    - The first region reached after crossing a grey separator is the target region; repaint that whole region with the source region's original color.
    - If two rods point to the same target region, the larger source region wins; all other regions keep their color.
    - Preserve the folded rods in the output, including their magenta pixels plus the maroon endpoint and grey hinge markers.

    Key insight:
    - The fold direction is not only a shape change: it acts as a pointer that selects which neighboring region gets recolored, and source-region area resolves the only allowed target conflict.
    """
    g = _as_int_grid(grid)
    regions, _region_idx, rods = _analyze_grid(g)
    region_by_id = {int(region["id"]): region for region in regions}

    target_to_best: dict[int, dict] = {}
    for rod in rods:
        target = int(rod["target_region_id"])
        prev = target_to_best.get(target)
        if prev is None or int(rod["source_area"]) > int(prev["source_area"]):
            target_to_best[target] = rod
        elif int(rod["source_area"]) == int(prev["source_area"]):
            raise AssertionError("conflicting rods must have different source areas")

    final_region_colors = {int(region["id"]): int(region["color"]) for region in regions}
    for target, rod in target_to_best.items():
        final_region_colors[int(target)] = int(rod["source_color"])

    out = np.full(g.shape, GREY, dtype=int)
    for region in regions:
        rid = int(region["id"])
        out[region["r0"] : region["r1"] + 1, region["c0"] : region["c1"] + 1] = final_region_colors[rid]

    occupied: set[tuple[int, int]] = set()
    for rod in rods:
        structure = set(rod["fixed_reds"]) | set(rod["folded"]) | {rod["endpoint"], rod["hinge"]}
        assert occupied.isdisjoint(structure), "folded rods must not overlap"
        occupied |= structure
        for r, c in rod["fixed_reds"]:
            out[r, c] = PINK
        for r, c in rod["folded"]:
            out[r, c] = PINK
        out[rod["endpoint"]] = MAROON
        out[rod["hinge"]] = GREY

    return out


def _sample_partition_sizes(parts: int, low: int, high: int) -> list[int]:
    for _ in range(200):
        vals = [random.randint(low, high) for _ in range(parts)]
        if len(set(vals)) == 1 and parts > 1:
            continue
        return vals
    raise RuntimeError("failed to sample partition sizes")


def _make_layout() -> tuple[np.ndarray, list[dict], dict[tuple[int, int], int]]:
    row_parts, col_parts = random.choice([(2, 2), (2, 3), (3, 2)])
    heights = _sample_partition_sizes(row_parts, 4, 8)
    widths = _sample_partition_sizes(col_parts, 4, 8)
    H = int(sum(heights) + (row_parts - 1))
    W = int(sum(widths) + (col_parts - 1))
    assert H <= 30 and W <= 30

    grid = np.full((H, W), GREY, dtype=int)
    colors = random.sample(list(REGION_COLORS), row_parts * col_parts)
    regions: list[dict] = []
    pos_to_id: dict[tuple[int, int], int] = {}
    rid = 0
    r = 0
    color_idx = 0
    for ri, height in enumerate(heights):
        c = 0
        for cj, width in enumerate(widths):
            r0, r1 = int(r), int(r + height - 1)
            c0, c1 = int(c), int(c + width - 1)
            color = int(colors[color_idx])
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
            regions.append(
                {
                    "id": rid,
                    "ri": int(ri),
                    "cj": int(cj),
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "area": int(height * width),
                    "color": color,
                }
            )
            pos_to_id[(int(ri), int(cj))] = rid
            rid += 1
            color_idx += 1
            c += width
            if cj != col_parts - 1:
                grid[:, c] = GREY
                c += 1
        r += height
        if ri != row_parts - 1:
            grid[r, :] = GREY
            r += 1
    return grid, regions, pos_to_id


def _target_region_id(region: dict, fold_dir: str, pos_to_id: dict[tuple[int, int], int]) -> int | None:
    ri, cj = int(region["ri"]), int(region["cj"])
    if fold_dir == "up":
        return pos_to_id.get((ri - 1, cj))
    if fold_dir == "down":
        return pos_to_id.get((ri + 1, cj))
    if fold_dir == "left":
        return pos_to_id.get((ri, cj - 1))
    return pos_to_id.get((ri, cj + 1))


def _enumerate_candidates(
    shape: tuple[int, int],
    regions: list[dict],
    pos_to_id: dict[tuple[int, int], int],
) -> dict[int, list[dict]]:
    H, W = shape
    region_by_id = {int(region["id"]): region for region in regions}
    out: dict[int, list[dict]] = defaultdict(list)

    for region in regions:
        r0, r1, c0, c1 = region["r0"], region["r1"], region["c0"], region["c1"]
        # Horizontal rods.
        if c1 - c0 + 1 >= 3:
            for line_r in range(r0, r1 + 1):
                up_space = int(line_r)
                down_space = int(H - 1 - line_r)
                if up_space == down_space:
                    continue
                fold_dir = "up" if up_space > down_space else "down"
                target_id = _target_region_id(region, fold_dir, pos_to_id)
                if target_id is None:
                    continue
                target = region_by_id[target_id]
                enter_min = int((line_r - r0) + 2) if fold_dir == "up" else int((r1 - line_r) + 2)
                stay_max = int(line_r - target["r0"]) if fold_dir == "up" else int(target["r1"] - line_r)
                if enter_min > stay_max:
                    continue
                for endpoint_side in ("left", "right"):
                    endpoint_c = c0 if endpoint_side == "left" else c1
                    for hinge_c in range(c0 + 1, c1):
                        tail_len = int(c1 - hinge_c) if endpoint_side == "left" else int(hinge_c - c0)
                        if not (enter_min <= tail_len <= stay_max):
                            continue
                        endpoint = (int(line_r), int(endpoint_c))
                        hinge = (int(line_r), int(hinge_c))
                        fixed_reds = (
                            [(int(line_r), int(c)) for c in range(c0 + 1, hinge_c)]
                            if endpoint_side == "left"
                            else [(int(line_r), int(c)) for c in range(hinge_c + 1, c1)]
                        )
                        tail = (
                            [(int(line_r), int(c)) for c in range(hinge_c + 1, c1 + 1)]
                            if endpoint_side == "left"
                            else [(int(line_r), int(c)) for c in range(c0, hinge_c)]
                        )
                        fold_step = FOLD_DIRS[fold_dir]
                        folded = [
                            (int(line_r + fold_step[0] * k), int(hinge_c + fold_step[1] * k))
                            for k in range(1, tail_len + 1)
                        ]
                        candidate = {
                            "source_region_id": int(region["id"]),
                            "source_color": int(region["color"]),
                            "source_area": int(region["area"]),
                            "target_region_id": int(target_id),
                            "orientation": "h",
                            "endpoint": endpoint,
                            "hinge": hinge,
                            "fixed_reds": fixed_reds,
                            "tail": tail,
                            "tail_len": tail_len,
                            "fold_dir": fold_dir,
                            "fold_step": fold_step,
                            "folded": folded,
                            "output_cells": frozenset(set(fixed_reds) | set(folded) | {endpoint, hinge}),
                        }
                        out[int(region["id"])].append(candidate)

        # Vertical rods.
        if r1 - r0 + 1 >= 3:
            for line_c in range(c0, c1 + 1):
                left_space = int(line_c)
                right_space = int(W - 1 - line_c)
                if left_space == right_space:
                    continue
                fold_dir = "left" if left_space > right_space else "right"
                target_id = _target_region_id(region, fold_dir, pos_to_id)
                if target_id is None:
                    continue
                target = region_by_id[target_id]
                enter_min = int((line_c - c0) + 2) if fold_dir == "left" else int((c1 - line_c) + 2)
                stay_max = int(line_c - target["c0"]) if fold_dir == "left" else int(target["c1"] - line_c)
                if enter_min > stay_max:
                    continue
                for endpoint_side in ("top", "bottom"):
                    endpoint_r = r0 if endpoint_side == "top" else r1
                    for hinge_r in range(r0 + 1, r1):
                        tail_len = int(r1 - hinge_r) if endpoint_side == "top" else int(hinge_r - r0)
                        if not (enter_min <= tail_len <= stay_max):
                            continue
                        endpoint = (int(endpoint_r), int(line_c))
                        hinge = (int(hinge_r), int(line_c))
                        fixed_reds = (
                            [(int(r), int(line_c)) for r in range(r0 + 1, hinge_r)]
                            if endpoint_side == "top"
                            else [(int(r), int(line_c)) for r in range(hinge_r + 1, r1)]
                        )
                        tail = (
                            [(int(r), int(line_c)) for r in range(hinge_r + 1, r1 + 1)]
                            if endpoint_side == "top"
                            else [(int(r), int(line_c)) for r in range(r0, hinge_r)]
                        )
                        fold_step = FOLD_DIRS[fold_dir]
                        folded = [
                            (int(hinge_r + fold_step[0] * k), int(line_c + fold_step[1] * k))
                            for k in range(1, tail_len + 1)
                        ]
                        candidate = {
                            "source_region_id": int(region["id"]),
                            "source_color": int(region["color"]),
                            "source_area": int(region["area"]),
                            "target_region_id": int(target_id),
                            "orientation": "v",
                            "endpoint": endpoint,
                            "hinge": hinge,
                            "fixed_reds": fixed_reds,
                            "tail": tail,
                            "tail_len": tail_len,
                            "fold_dir": fold_dir,
                            "fold_step": fold_step,
                            "folded": folded,
                            "output_cells": frozenset(set(fixed_reds) | set(folded) | {endpoint, hinge}),
                        }
                        out[int(region["id"])].append(candidate)

    return out


def _conflict_profile(chosen: list[dict]) -> tuple[dict[int, int], int]:
    counts: dict[int, int] = Counter(int(c["target_region_id"]) for c in chosen)
    twice = sum(1 for v in counts.values() if int(v) == 2)
    return counts, int(twice)


def _select_rods(candidate_map: dict[int, list[dict]], region_count: int) -> list[dict] | None:
    source_regions = [rid for rid, options in candidate_map.items() if options]
    if len(source_regions) < 2:
        return None

    count_choices = [2, 3]
    if region_count >= 6:
        count_choices.append(4)
    random.shuffle(count_choices)

    for desired_count in count_choices:
        if desired_count > len(source_regions):
            continue
        for want_conflict in random.sample([False, True], k=2):
            for _ in range(250):
                chosen: list[dict] = []
                used_sources: set[int] = set()
                occupied: set[tuple[int, int]] = set()

                if want_conflict:
                    conflict_pairs: list[tuple[dict, dict]] = []
                    for i, src_a in enumerate(source_regions):
                        for src_b in source_regions[i + 1 :]:
                            for cand_a in candidate_map[src_a]:
                                for cand_b in candidate_map[src_b]:
                                    if int(cand_a["target_region_id"]) != int(cand_b["target_region_id"]):
                                        continue
                                    if int(cand_a["source_area"]) == int(cand_b["source_area"]):
                                        continue
                                    if not set(cand_a["output_cells"]).isdisjoint(cand_b["output_cells"]):
                                        continue
                                    conflict_pairs.append((cand_a, cand_b))
                    if not conflict_pairs:
                        continue
                    pair = random.choice(conflict_pairs)
                    chosen.extend(pair)
                    used_sources.update(int(c["source_region_id"]) for c in pair)
                    occupied |= set(pair[0]["output_cells"]) | set(pair[1]["output_cells"])

                random_sources = source_regions[:]
                random.shuffle(random_sources)
                for source_id in random_sources:
                    if len(chosen) >= desired_count:
                        break
                    if source_id in used_sources:
                        continue
                    options = candidate_map[source_id][:]
                    random.shuffle(options)
                    for cand in options:
                        if not occupied.isdisjoint(cand["output_cells"]):
                            continue
                        trial = chosen + [cand]
                        target_counts, twice = _conflict_profile(trial)
                        if max(target_counts.values(), default=0) > 2:
                            continue
                        if twice > 1:
                            continue
                        if not want_conflict and any(v > 1 for v in target_counts.values()):
                            continue
                        if want_conflict and len(trial) == desired_count and twice != 1:
                            continue
                        chosen.append(cand)
                        used_sources.add(source_id)
                        occupied |= set(cand["output_cells"])
                        break

                if len(chosen) != desired_count:
                    continue
                target_counts, twice = _conflict_profile(chosen)
                if want_conflict and twice != 1:
                    continue
                if not want_conflict and any(v > 1 for v in target_counts.values()):
                    continue
                orientations = {str(c["orientation"]) for c in chosen}
                if desired_count >= 3 and len(orientations) == 1 and random.random() < 0.7:
                    continue
                return chosen
    return None


def _paint_input_rods(grid: np.ndarray, rods: Iterable[dict]) -> None:
    for rod in rods:
        for r, c in rod["fixed_reds"]:
            grid[r, c] = PINK
        for r, c in rod["tail"]:
            grid[r, c] = PINK
        grid[rod["endpoint"]] = MAROON
        grid[rod["hinge"]] = GREY


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        regions, _region_idx, rods = _analyze_grid(grid)
        if not (2 <= len(rods) <= 4):
            return False
        target_counts = Counter(int(rod["target_region_id"]) for rod in rods)
        if max(target_counts.values(), default=0) > 2:
            return False
        if sum(1 for v in target_counts.values() if int(v) == 2) > 1:
            return False
        for target, count in target_counts.items():
            if int(count) == 2:
                areas = sorted(int(rod["source_area"]) for rod in rods if int(rod["target_region_id"]) == int(target))
                if areas[0] == areas[1]:
                    return False
        colors = [int(region["color"]) for region in regions]
        if len(colors) != len(set(colors)):
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid, regions, pos_to_id = _make_layout()
        candidate_map = _enumerate_candidates(grid.shape, regions, pos_to_id)
        rods = _select_rods(candidate_map, len(regions))
        if not rods:
            continue
        _paint_input_rods(grid, rods)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid folded-pointer recolor task")
