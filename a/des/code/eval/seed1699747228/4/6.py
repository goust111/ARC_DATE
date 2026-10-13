from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
MISSING_BY_DIR = {
    "right": (0, 1),
    "left": (0, 0),
    "down": (1, 0),
    "up": (1, 1),
}
EMITTER_BY_DIR = {
    "right": (1, 1),
    "left": (1, 0),
    "down": (1, 1),
    "up": (0, 1),
}
GEN_RETRIES = 500


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (
                        _in_bounds(h, w, nr, nc)
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _find_legend(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[tuple[int, int, int, int], list[tuple[int, int, int]]]] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width != 5:
            continue
        if not (r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                is_grey = int(g[r, c]) == GREY
                if on_border != is_grey:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.shape[1] != 3 or inner.shape[0] < 1:
            continue
        rows: list[tuple[int, int, int]] = []
        for rr in range(inner.shape[0]):
            triple = tuple(int(v) for v in inner[rr])
            if any(v in (BG, GREY) for v in triple):
                ok = False
                break
            rows.append(triple)
        if ok:
            candidates.append(((r0, c0, r1, c1), rows))
    assert len(candidates) == 1, "expected exactly one grey legend"
    return candidates[0]


def _find_towers(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    legend = _legend_mask(g.shape, legend_bbox)
    usable = (g != BG) & (g != GREY) & (~legend)
    towers: list[dict[str, object]] = []
    for comp in _components(usable):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1 or len(comp) != 3:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        present = {(r - r0, c - c0) for r, c in comp}
        missing = None
        for cand in ((0, 0), (0, 1), (1, 0), (1, 1)):
            if cand not in present:
                missing = cand
                break
        if missing is None:
            continue
        direction = None
        for name, miss in MISSING_BY_DIR.items():
            if miss == missing:
                direction = name
                break
        if direction is None:
            continue
        towers.append(
            {
                "anchor": (r0, c0),
                "color": int(next(iter(colors))),
                "direction": direction,
            }
        )
    assert towers, "expected at least one tower"
    return towers


def _ray_cells(
    tower: dict[str, object], obstacle_mask: np.ndarray
) -> list[tuple[int, int]]:
    h, w = obstacle_mask.shape
    r0, c0 = tower["anchor"]
    direction = tower["direction"]
    dr, dc = DIRS[direction]
    er, ec = EMITTER_BY_DIR[direction]
    r = int(r0) + int(er) + int(dr)
    c = int(c0) + int(ec) + int(dc)
    cells: list[tuple[int, int]] = []
    while _in_bounds(h, w, r, c) and not obstacle_mask[r, c]:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _mapping_from_rows(rows: list[tuple[int, int, int]]) -> dict[tuple[int, int], int]:
    mapping: dict[tuple[int, int], int] = {}
    for a, b, out in rows:
        mapping[(a, b)] = out
        mapping[(b, a)] = out
    return mapping


def _sorted_towers(towers: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        towers,
        key=lambda tower: (
            int(tower["anchor"][1]),
            int(tower["anchor"][0]),
        ),
    )


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    legend_bbox, legend_rows = _find_legend(g)
    legend = _legend_mask(g.shape, legend_bbox)
    mapping = _mapping_from_rows(legend_rows)
    towers = _sorted_towers(_find_towers(g, legend_bbox))
    obstacle_mask = g != BG

    out = g.copy()
    paths: list[dict[str, object]] = []
    for tower in towers:
        ray = _ray_cells(tower, obstacle_mask)
        paths.append({"tower": tower, "cells": ray})
        for r, c in ray:
            current = int(out[r, c])
            color = int(tower["color"])
            if current == BG:
                out[r, c] = color
            elif current == color:
                continue
            else:
                out[r, c] = mapping.get((current, color), current)

    out[legend] = g[legend]
    return out, {
        "legend_bbox": legend_bbox,
        "legend_rows": legend_rows,
        "legend_mask": legend,
        "mapping": mapping,
        "towers": towers,
        "paths": paths,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast rays from 2x2 missing-corner towers and mix crossing colors using a grey-framed legend.
    """
    return _solve(grid)[0]


def _spaced_values(
    candidates: list[int], count: int, rng: random.Random, min_gap: int = 4
) -> list[int] | None:
    if count == 0:
        return []
    if count == 1:
        if not candidates:
            return None
        return [rng.choice(candidates)]
    pairs = [
        (a, b)
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
        if b - a >= min_gap
    ]
    if not pairs:
        return None
    pair = rng.choice(pairs)
    return [pair[0], pair[1]]


def _tower_cells(anchor: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0 = anchor
    missing = MISSING_BY_DIR[direction]
    cells: list[tuple[int, int]] = []
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                cells.append((r0 + dr, c0 + dc))
    return cells


def _place_legend(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    rows: list[tuple[int, int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    for i, triple in enumerate(rows, start=1):
        grid[r0 + i, c0 + 1 : c0 + 4] = np.array(triple, dtype=int)


def _legend_bbox_for_edge(
    h: int, w: int, edge: str, n_rows: int, rng: random.Random
) -> tuple[int, int, int, int]:
    legend_h = n_rows + 2
    legend_w = 5
    if edge == "top":
        top = 0
        left = rng.randint(0, w - legend_w)
    elif edge == "bottom":
        top = h - legend_h
        left = rng.randint(0, w - legend_w)
    elif edge == "left":
        top = rng.randint(0, h - legend_h)
        left = 0
    else:
        top = rng.randint(0, h - legend_h)
        left = w - legend_w
    return top, left, top + legend_h - 1, left + legend_w - 1


def _candidate_rows(
    h: int,
    edge: str,
    legend_bbox: tuple[int, int, int, int],
    top_band: int,
    bottom_band: int,
    need_down: bool,
    need_up: bool,
) -> list[int]:
    r0, _, r1, _ = legend_bbox
    row_min = 2
    row_max = h - 3
    if need_down:
        row_min = max(row_min, top_band + 4)
    if need_up:
        row_max = min(row_max, bottom_band - 2)
    rows: list[int] = []
    for r in range(row_min, row_max + 1):
        if edge in ("left", "right") and r0 - 1 <= r <= r1 + 1:
            continue
        rows.append(r)
    return rows


def _candidate_cols(
    w: int,
    edge: str,
    legend_bbox: tuple[int, int, int, int],
    left_band: int,
    right_band: int,
    need_right: bool,
    need_left: bool,
) -> list[int]:
    _, c0, _, c1 = legend_bbox
    col_min = 2
    col_max = w - 3
    if need_right:
        col_min = max(col_min, left_band + 4)
    if need_left:
        col_max = min(col_max, right_band - 2)
    cols: list[int] = []
    for c in range(col_min, col_max + 1):
        if edge in ("top", "bottom") and c0 - 1 <= c <= c1 + 1:
            continue
        cols.append(c)
    return cols


def _sample_decoy_row(
    rng: random.Random, used_colors: set[int], used_pairs: set[tuple[int, int]]
) -> tuple[int, int, int] | None:
    available = [c for c in COLOR_POOL if c not in used_colors]
    if len(available) < 3:
        return None
    for _ in range(30):
        a, b, out = rng.sample(available, 3)
        key = tuple(sorted((a, b)))
        if key not in used_pairs:
            return a, b, out
    return None


def _build_mode_specs(
    mode: str,
    rows: list[int],
    cols: list[int],
    variant: int = 0,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str]]]:
    if mode == "mapped_single":
        combos = [
            ("H", "right", rows[0], "V", "down", cols[0]),
            ("H", "right", rows[0], "V", "up", cols[0]),
            ("H", "left", rows[0], "V", "down", cols[0]),
            ("H", "left", rows[0], "V", "up", cols[0]),
        ]
        combo = combos[variant]
        specs = [(combo[0], combo[1], combo[2]), (combo[3], combo[4], combo[5])]
        return specs, [("H", "V")]
    if mode == "unmapped_single":
        combo = [
            [("H", "right", rows[0]), ("V", "up", cols[0])],
            [("H", "left", rows[0]), ("V", "down", cols[0])],
        ][variant]
        return combo, []
    if mode == "mixed_three_a":
        specs = [("H1", "right", rows[0]), ("H2", "left", rows[1]), ("V", "up", cols[0])]
        return specs, [("H2", "V")]
    if mode == "mixed_three_b":
        specs = [("H1", "right", rows[0]), ("H2", "left", rows[1]), ("V", "down", cols[0])]
        return specs, [("H1", "V")]
    if mode == "mixed_three_c":
        specs = [("H", "right", rows[0]), ("V1", "down", cols[0]), ("V2", "up", cols[1])]
        return specs, [("H", "V1")]
    if mode == "mixed_three_d":
        specs = [("H", "left", rows[0]), ("V1", "down", cols[0]), ("V2", "up", cols[1])]
        return specs, [("H", "V2")]
    specs = [
        ("H1", "right", rows[0]),
        ("H2", "left", rows[1]),
        ("V1", "down", cols[0]),
        ("V2", "up", cols[1]),
    ]
    return specs, [("H1", "V1"), ("H1", "V2"), ("H2", "V1"), ("H2", "V2")]


def _place_towers(
    grid: np.ndarray,
    specs: list[tuple[str, str, int]],
    colors: dict[str, int],
    legend_bbox: tuple[int, int, int, int],
    left_band: int,
    right_band: int,
    top_band: int,
    bottom_band: int,
) -> dict[str, dict[str, object]] | None:
    h, w = grid.shape
    legend = _legend_mask(grid.shape, legend_bbox)
    placed: dict[str, dict[str, object]] = {}
    for name, direction, line in specs:
        if direction == "right":
            anchor = (line - 1, left_band)
        elif direction == "left":
            anchor = (line - 1, right_band)
        elif direction == "down":
            anchor = (top_band, line - 1)
        else:
            anchor = (bottom_band, line - 1)
        cells = _tower_cells(anchor, direction)
        for r, c in cells:
            if not _in_bounds(h, w, r, c) or legend[r, c] or int(grid[r, c]) != BG:
                return None
        for r, c in cells:
            grid[r, c] = colors[name]
        placed[name] = {
            "anchor": anchor,
            "direction": direction,
            "color": colors[name],
        }
    return placed


def _grid_for_mode(rng: random.Random, mode: str) -> np.ndarray | None:
    tower_count = {
        "mapped_single": 2,
        "unmapped_single": 2,
        "mixed_three_a": 3,
        "mixed_three_b": 3,
        "mixed_three_c": 3,
        "mixed_three_d": 3,
        "dense_four": 4,
    }[mode]

    mapped_target = {
        "mapped_single": 1,
        "unmapped_single": 0,
        "mixed_three_a": 1,
        "mixed_three_b": 1,
        "mixed_three_c": 1,
        "mixed_three_d": 1,
        "dense_four": 4,
    }[mode]

    add_decoy = mode == "unmapped_single" or (
        mode.startswith("mixed_three") and rng.random() < 0.5
    ) or (mode == "mapped_single" and rng.random() < 0.5)
    legend_rows_needed = max(1, mapped_target + (1 if add_decoy else 0))

    h = rng.randint(20, 26)
    w = rng.randint(20, 26)
    if mode == "dense_four":
        h = max(h, 22)
        w = max(w, 22)

    if mode == "mapped_single":
        variant = rng.randrange(4)
    elif mode == "unmapped_single":
        variant = rng.randrange(2)
    else:
        variant = 0

    edge = rng.choice(["top", "bottom", "left", "right"])
    legend_bbox = _legend_bbox_for_edge(h, w, edge, legend_rows_needed, rng)
    lr0, lc0, lr1, lc1 = legend_bbox
    left_band = 0 if edge != "left" else lc1 + 2
    right_band = w - 2 if edge != "right" else lc0 - 3
    top_band = 0 if edge != "top" else lr1 + 2
    bottom_band = h - 2 if edge != "bottom" else lr0 - 3
    if min(left_band, top_band) < 0 or right_band + 1 >= w or bottom_band + 1 >= h:
        return None
    if left_band >= right_band or top_band >= bottom_band:
        return None

    if mode in ("mapped_single", "unmapped_single"):
        temp_specs, _ = _build_mode_specs(mode, [0], [0], variant)
        need_rows = 1
        need_cols = 1
    elif mode in ("mixed_three_a", "mixed_three_b"):
        temp_specs = [("H1", "right", 0), ("H2", "left", 0), ("V", "up", 0)]
        if mode == "mixed_three_b":
            temp_specs[2] = ("V", "down", 0)
        need_rows = 2
        need_cols = 1
    elif mode in ("mixed_three_c", "mixed_three_d"):
        temp_specs = [("H", "right", 0), ("V1", "down", 0), ("V2", "up", 0)]
        if mode == "mixed_three_d":
            temp_specs[0] = ("H", "left", 0)
        need_rows = 1
        need_cols = 2
    else:
        temp_specs = [
            ("H1", "right", 0),
            ("H2", "left", 0),
            ("V1", "down", 0),
            ("V2", "up", 0),
        ]
        need_rows = 2
        need_cols = 2

    need_right = any(direction == "right" for _, direction, _ in temp_specs)
    need_left = any(direction == "left" for _, direction, _ in temp_specs)
    need_down = any(direction == "down" for _, direction, _ in temp_specs)
    need_up = any(direction == "up" for _, direction, _ in temp_specs)
    row_candidates = _candidate_rows(
        h, edge, legend_bbox, top_band, bottom_band, need_down, need_up
    )
    col_candidates = _candidate_cols(
        w, edge, legend_bbox, left_band, right_band, need_right, need_left
    )
    rows = _spaced_values(row_candidates, need_rows, rng)
    cols = _spaced_values(col_candidates, need_cols, rng)
    if rows is None or cols is None:
        return None

    specs, mapped_pairs_by_name = _build_mode_specs(mode, rows, cols, variant)
    names = [name for name, _, _ in specs]
    tower_colors = rng.sample(list(COLOR_POOL), tower_count)
    color_by_name = {name: tower_colors[i] for i, name in enumerate(names)}

    mapped_pairs = [
        tuple(sorted((color_by_name[a], color_by_name[b])))
        for a, b in mapped_pairs_by_name
    ]
    mapped_pairs = list(dict.fromkeys(mapped_pairs))

    used_colors = set(tower_colors)
    remaining = [c for c in COLOR_POOL if c not in used_colors]
    if len(remaining) < len(mapped_pairs):
        return None
    result_colors = rng.sample(remaining, len(mapped_pairs))
    used_colors.update(result_colors)

    used_pair_keys = set(mapped_pairs)
    legend_rows: list[tuple[int, int, int]] = []
    for pair, result in zip(mapped_pairs, result_colors):
        a, b = pair
        if rng.random() < 0.5:
            legend_rows.append((a, b, result))
        else:
            legend_rows.append((b, a, result))

    if add_decoy and len(legend_rows) < legend_rows_needed:
        decoy = _sample_decoy_row(rng, used_colors, used_pair_keys)
        if decoy is None:
            return None
        legend_rows.append(decoy)
        used_colors.update(decoy)
        used_pair_keys.add(tuple(sorted(decoy[:2])))

    while len(legend_rows) < legend_rows_needed:
        decoy = _sample_decoy_row(rng, used_colors, used_pair_keys)
        if decoy is None:
            return None
        legend_rows.append(decoy)
        used_colors.update(decoy)
        used_pair_keys.add(tuple(sorted(decoy[:2])))
    rng.shuffle(legend_rows)

    grid = np.zeros((h, w), dtype=int)
    _place_legend(grid, legend_bbox, legend_rows)
    placed = _place_towers(
        grid,
        specs,
        color_by_name,
        legend_bbox,
        left_band,
        right_band,
        top_band,
        bottom_band,
    )
    if placed is None:
        return None

    try:
        out, info = _solve(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not np.array_equal(grid[info["legend_mask"]], out[info["legend_mask"]]):
        return None
    if len(info["towers"]) != tower_count:
        return None

    path_sets = {
        int(item["tower"]["color"]): set(item["cells"])
        for item in info["paths"]
    }
    intersections = []
    colors = sorted(path_sets)
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            overlap = path_sets[a] & path_sets[b]
            if overlap:
                intersections.append(tuple(sorted((a, b))))
    intersections = sorted(set(intersections))
    expected_intersections = {
        "mapped_single": 1,
        "unmapped_single": 1,
        "mixed_three_a": 2,
        "mixed_three_b": 2,
        "mixed_three_c": 2,
        "mixed_three_d": 2,
        "dense_four": 4,
    }[mode]
    if len(intersections) != expected_intersections:
        return None

    mapped_in_scene = sum(pair in set(mapped_pairs) for pair in intersections)
    if mapped_in_scene != mapped_target:
        return None

    for pair in intersections:
        a, b = pair
        cell = next(iter(path_sets[a] & path_sets[b]))
        if pair in set(mapped_pairs):
            if int(out[cell]) == a or int(out[cell]) == b:
                return None
        else:
            if int(out[cell]) not in pair:
                return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    modes = [
        "mapped_single",
        "unmapped_single",
        "mixed_three_a",
        "mixed_three_b",
        "mixed_three_c",
        "mixed_three_d",
        "dense_four",
    ]
    for _ in range(GEN_RETRIES):
        grid = _grid_for_mode(rng, rng.choice(modes))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-tower input")
