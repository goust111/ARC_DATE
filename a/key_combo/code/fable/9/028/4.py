from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
ORIENTATIONS = ("h", "v", "h", "v")
COLOR_IDS = tuple(range(10))

GRID_MIN = 12
GRID_MAX = 20
MIN_COMPLETE_SEGMENTS = 5
MAX_COMPLETE_SEGMENTS = 10
MIN_REMAINT_SEGMENTS = 4
MAX_REMAINT_SEGMENTS = 8
GENERATION_ATTEMPTS = 500
REMNANT_LAYOUT_ATTEMPTS = 120


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1


def _stage_buckets(n: int) -> list[list[int]]:
    if n <= 5:
        mid = max(1, n // 2)
        return [list(range(0, mid)), list(range(mid, n))]
    cuts = [0, n // 3, (2 * n) // 3, n]
    buckets = []
    for a, b in zip(cuts, cuts[1:]):
        if a < b:
            buckets.append(list(range(a, b)))
    return buckets


def _segment_options(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if len(cells) <= 2:
        return ()
    return cells[1:-1]


def _build_model(shape: tuple[int, int], center: tuple[int, int]) -> dict:
    r, c = center
    h_cells = {center}
    v_cells: set[tuple[int, int]] = set()
    all_cells = {center}
    cell_segment = {center: 0}
    segments = []
    length = 2
    dir_index = 0

    while True:
        dr, dc = DIRS[dir_index]
        new_cells = tuple((r + dr * step, c + dc * step) for step in range(1, length + 1))
        if any(not _in_bounds(shape, nr, nc) for nr, nc in new_cells):
            break

        if dir_index == 0:
            cells = (center,) + new_cells
        else:
            cells = new_cells

        orientation = ORIENTATIONS[dir_index]
        for pos in cells:
            all_cells.add(pos)
            cell_segment[pos] = len(segments)
        if orientation == "h":
            h_cells.update(cells)
        else:
            v_cells.update(cells)
        segments.append(
            {
                "index": len(segments),
                "orientation": orientation,
                "cells": cells,
                "options": _segment_options(cells),
            }
        )
        r, c = new_cells[-1]
        length += 1
        dir_index = (dir_index + 1) % 4

    return {
        "center": center,
        "segments": tuple(segments),
        "h_cells": frozenset(h_cells),
        "v_cells": frozenset(v_cells),
        "all_cells": frozenset(all_cells),
        "cell_segment": cell_segment,
        "failed_dir": dir_index,
    }


@lru_cache(maxsize=None)
def _models_for_shape(shape: tuple[int, int]) -> tuple[dict, ...]:
    h, w = shape
    return tuple(_build_model(shape, (r, c)) for r in range(h) for c in range(w))


def _render_model(model: dict, shape: tuple[int, int], bg: int, h_color: int, v_color: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for r, c in model["h_cells"]:
        out[r, c] = h_color
    for r, c in model["v_cells"]:
        out[r, c] = v_color
    return out


def _solve_candidates(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 2, "expected exactly two remnant colors"

    coords_by_color = {
        color: {tuple(map(int, p)) for p in np.argwhere(g == color)}
        for color in colors
    }
    assert all(coords_by_color[color] for color in colors), "both remnant colors must appear"

    candidates = []
    for h_color, v_color in (tuple(colors), tuple(reversed(colors))):
        h_need = coords_by_color[h_color]
        v_need = coords_by_color[v_color]
        for model in _models_for_shape(g.shape):
            if h_need <= model["h_cells"] and v_need <= model["v_cells"]:
                candidates.append(
                    {
                        "model": model,
                        "center": model["center"],
                        "h_color": h_color,
                        "v_color": v_color,
                        "output": _render_model(model, g.shape, bg, h_color, v_color),
                    }
                )
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the unique clipped square spiral from its surviving directional remnants.

    Input structure:
    - The grid is dominated by one background color.
    - All foreground pixels are isolated remnants in exactly two colors.
    - One remnant color appears only on horizontal spiral segments, and the other
      only on vertical segments; the true spiral center is not marked.

    Rule:
    - Infer the unique square spiral whose center and all remnants are consistent
      with a left, down, right, up walk and segment lengths 2, 3, 4, ... .
    - A segment is drawn only if its entire requested length stays inside the grid;
      otherwise the spiral stops before that segment.
    - Repaint the whole spiral, including the center cell as part of the first
      horizontal segment, using the horizontal remnant color on horizontal cells
      and the vertical remnant color on vertical cells.

    Key insight:
    - The remnants are not independent dots: their colors encode segment
      orientation, and only one center can satisfy those colored collinear traces
      together with the fixed turning order and increasing segment lengths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    candidates = _solve_candidates(g)
    assert len(candidates) == 1, "expected a unique compatible spiral"
    return candidates[0]["output"]


def _valid_source_model(model: dict) -> bool:
    segments = model["segments"]
    if not (MIN_COMPLETE_SEGMENTS <= len(segments) <= MAX_COMPLETE_SEGMENTS):
        return False
    if len([seg for seg in segments if seg["orientation"] == "h"]) < 2:
        return False
    if len([seg for seg in segments if seg["orientation"] == "v"]) < 2:
        return False
    if any(not seg["options"] for seg in segments):
        return False
    return True


def _pick_positions_for_segments(model: dict, segment_ids: list[int]) -> list[tuple[int, int]] | None:
    seg_options = {idx: list(model["segments"][idx]["options"]) for idx in segment_ids}
    order = sorted(segment_ids, key=lambda idx: len(seg_options[idx]))
    chosen: list[tuple[int, int]] = []
    out: dict[int, tuple[int, int]] = {}

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        seg_id = order[k]
        options = seg_options[seg_id][:]
        random.shuffle(options)
        for pos in options:
            if all(_far_enough(pos, prev) for prev in chosen):
                chosen.append(pos)
                out[seg_id] = pos
                if dfs(k + 1):
                    return True
                chosen.pop()
                del out[seg_id]
        return False

    if not dfs(0):
        return None
    return [out[idx] for idx in segment_ids]


def _is_isolated(grid: np.ndarray, bg: int) -> bool:
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = rr + dr, cc + dc
                if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != bg:
                    return False
    return True


def _semantic_ok(inp: np.ndarray, model: dict, bg: int, h_color: int, v_color: int) -> bool:
    if not _is_isolated(inp, bg):
        return False

    candidates = _solve_candidates(inp)
    if len(candidates) != 1:
        return False
    winner = candidates[0]
    if winner["center"] != model["center"]:
        return False
    if winner["h_color"] != h_color or winner["v_color"] != v_color:
        return False

    remnant_segment_ids = set()
    by_orientation = {"h": set(), "v": set()}
    by_stage = 0
    buckets = _stage_buckets(len(model["segments"]))
    remnant_positions = [tuple(map(int, p)) for p in np.argwhere(inp != bg)]
    for bucket in buckets:
        if any(pos in model["cell_segment"] and model["cell_segment"][pos] in bucket for pos in remnant_positions):
            by_stage += 1

    for color, orientation in ((h_color, "h"), (v_color, "v")):
        for pos_arr in np.argwhere(inp == color):
            pos = tuple(map(int, pos_arr))
            if pos not in model["cell_segment"]:
                return False
            seg_id = model["cell_segment"][pos]
            remnant_segment_ids.add(seg_id)
            by_orientation[orientation].add(seg_id)
            if model["segments"][seg_id]["orientation"] != orientation:
                return False

    if len(remnant_segment_ids) < MIN_REMAINT_SEGMENTS:
        return False
    if len(by_orientation["h"]) < 2 or len(by_orientation["v"]) < 2:
        return False
    if by_stage < len(buckets):
        return False
    if not any(seg_id >= len(model["segments"]) - 2 for seg_id in remnant_segment_ids):
        return False
    return True


def _gen_input_from_selection(
    shape: tuple[int, int],
    selected: list[int],
    positions: list[tuple[int, int]],
    segments: tuple[dict, ...],
    bg: int,
    h_color: int,
    v_color: int,
) -> np.ndarray:
    inp = np.full(shape, bg, dtype=int)
    for seg_id, pos in zip(selected, positions):
        color = h_color if segments[seg_id]["orientation"] == "h" else v_color
        inp[pos] = color
    return inp


def _sample_remnants(model: dict, shape: tuple[int, int], bg: int, h_color: int, v_color: int) -> np.ndarray | None:
    segments = model["segments"]
    n = len(segments)
    all_segment_ids = list(range(n))
    buckets = _stage_buckets(n)

    for _ in range(REMNANT_LAYOUT_ATTEMPTS):
        required: set[int] = set()
        for bucket in buckets:
            required.add(random.choice(bucket))
        required.add(random.choice(list(range(max(0, n - 2), n))))

        for orientation in ("h", "v"):
            orient_ids = [idx for idx in all_segment_ids if segments[idx]["orientation"] == orientation]
            while sum(1 for idx in required if segments[idx]["orientation"] == orientation) < 2 and orient_ids:
                options = [idx for idx in orient_ids if idx not in required]
                if not options:
                    break
                required.add(random.choice(options))

        low = max(len(required), 5)
        high = min(MAX_REMAINT_SEGMENTS, n)
        if low > high:
            continue

        remaining = [idx for idx in all_segment_ids if idx not in required]
        random.shuffle(remaining)
        target_count = random.randint(low, high)
        selected = list(required) + remaining[: target_count - len(required)]
        random.shuffle(selected)

        positions = _pick_positions_for_segments(model, selected)
        if positions is None:
            continue

        inp = _gen_input_from_selection(shape, selected, positions, segments, bg, h_color, v_color)
        if _semantic_ok(inp, model, bg, h_color, v_color):
            return inp
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.choice(COLOR_IDS)
    h_color, v_color = random.sample([c for c in COLOR_IDS if c != bg], 2)
    shape = (h, w)

    centers = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(centers)
    for center in centers:
        model = _build_model(shape, center)
        if not _valid_source_model(model):
            continue
        inp = _sample_remnants(model, shape, bg, h_color, v_color)
        if inp is None:
            continue
        out = _render_model(model, shape, bg, h_color, v_color)
        try:
            pred = main(inp)
        except AssertionError:
            continue
        if np.array_equal(pred, out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-remnant input")
