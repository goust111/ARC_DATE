from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 6)
DIMENSIONS = tuple((h, w) for h in range(1, 5) for w in range(1, 5))
GRID_SIDE_RANGE = (15, 20)
RECT_COUNT_CHOICES = (3, 4, 4, 5, 5)
COUNT_RANGE = (4, 12)
MAX_GENERATION_ATTEMPTS = 500
_CANDIDATE_CACHE: dict[tuple[int, int], dict[tuple[int, int], list[dict]]] = {}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _stamp_stats(r0: int, c0: int, height: int, width: int, grid_shape: tuple[int, int]) -> dict:
    rows, cols = grid_shape
    cells = {r * cols + c for r in range(r0, r0 + height) for c in range(c0, c0 + width)}
    left_steps = 0
    right_steps = 0
    clipped = 0
    full_repeats = 0
    for direction in (-1, 1):
        step = 1
        while True:
            rr = r0 + step * height
            cc = c0 + direction * step * width
            if rr >= rows or cc >= cols or cc + width <= 0:
                break
            ir0 = max(0, rr)
            ir1 = min(rows, rr + height)
            ic0 = max(0, cc)
            ic1 = min(cols, cc + width)
            if ir0 >= ir1 or ic0 >= ic1:
                break
            if direction < 0:
                left_steps += 1
            else:
                right_steps += 1
            if rr >= 0 and rr + height <= rows and cc >= 0 and cc + width <= cols:
                full_repeats += 1
            else:
                clipped += 1
            for pr in range(ir0, ir1):
                for pc in range(ic0, ic1):
                    cells.add(pr * cols + pc)
            step += 1
    return {
        "count": 1 + left_steps + right_steps,
        "left": left_steps,
        "right": right_steps,
        "clipped": clipped,
        "full_repeats": full_repeats,
        "cells": cells,
    }


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    components = _find_components(g)
    assert 3 <= len(components) <= 5, "expected 3-5 rectangles"

    colors = [comp["color"] for comp in components]
    assert len(colors) == len(set(colors)), "rectangle colors must be unique"

    rows, cols = g.shape
    specs: list[dict] = []
    size_pairs: set[tuple[int, int]] = set()
    all_traj_cells: set[int] = set()
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert comp["area"] == height * width, "foreground components must be solid rectangles"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == comp["color"]), "bbox must be monochrome"
        size_pair = (height, width)
        assert size_pair not in size_pairs, "rectangle sizes must be distinct"
        size_pairs.add(size_pair)

        stats = _stamp_stats(r0, c0, height, width, (rows, cols))
        assert stats["left"] >= 1 and stats["right"] >= 1, "both diagonal directions must contribute"
        overlap = all_traj_cells & stats["cells"]
        assert not overlap, "different rectangle trajectories must not overlap"
        all_traj_cells |= stats["cells"]
        specs.append(
            {
                "color": comp["color"],
                "r0": r0,
                "c0": c0,
                "h": height,
                "w": width,
                "area": comp["area"],
                **stats,
            }
        )

    counts = [spec["count"] for spec in specs]
    assert len(counts) == len(set(counts)), "stamp counts must be unique"
    assert any(spec["h"] != spec["w"] for spec in specs), "not all rectangles may be square"
    return specs


def _histogram_from_specs(specs: list[dict], heights_key: str, order_key) -> np.ndarray:
    ordered = sorted(specs, key=order_key)
    max_height = max(int(spec[heights_key]) for spec in ordered)
    out = np.zeros((max_height, len(ordered)), dtype=int)
    for col, spec in enumerate(ordered):
        out[: int(spec[heights_key]), col] = int(spec["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each rectangle's diagonal self-stamps and summarize the counts as a sorted top-aligned histogram.

    Input structure:
    - The background is black.
    - There are 3-5 non-overlapping single-color filled rectangles, each with a distinct color and distinct size.
    - For a rectangle with top-left corner (r, c), height h, and width w, the rectangle itself defines the sprite and the diagonal stride.

    Rule:
    - For each rectangle, keep the original and stamp copies at offsets (k*h, -k*w) and (k*h, +k*w) for k = 1, 2, ... .
    - Count every copy whose stamped rectangle still has at least one in-bounds cell; partially clipped copies still count.
    - Sort rectangles by this total stamp count from largest to smallest, breaking ties by smaller original left column.
    - Output a new grid whose height is the largest count and whose width is the number of rectangles; column k is filled from the top with the kth rectangle's color for exactly its count cells.

    Key insight:
    - The rectangle size controls the diagonal step, while the starting position controls how long each trajectory stays visible, so the histogram height is neither the rectangle area nor its original height.
    """
    specs = _extract_rectangles(np.asarray(grid, dtype=int))
    ordered = sorted(specs, key=lambda spec: (-int(spec["count"]), int(spec["c0"])))
    out = np.zeros((max(int(spec["count"]) for spec in ordered), len(ordered)), dtype=int)
    for col, spec in enumerate(ordered):
        out[: int(spec["count"]), col] = int(spec["color"])
    return out


def _candidate_pool(rows: int, cols: int) -> dict[tuple[int, int], list[dict]]:
    key = (rows, cols)
    if key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[key]

    by_dim = {dim: [] for dim in DIMENSIONS}
    for height, width in DIMENSIONS:
        for r0 in range(rows - height + 1):
            for c0 in range(cols - width + 1):
                stats = _stamp_stats(r0, c0, height, width, (rows, cols))
                if stats["left"] < 1 or stats["right"] < 1:
                    continue
                if stats["full_repeats"] < 1:
                    continue
                if not (COUNT_RANGE[0] <= stats["count"] <= COUNT_RANGE[1]):
                    continue
                by_dim[(height, width)].append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "h": height,
                        "w": width,
                        "area": height * width,
                        **stats,
                    }
                )
    _CANDIDATE_CACHE[key] = by_dim
    return by_dim


def _choose_layout(rows: int, cols: int, dims: list[tuple[int, int]]) -> list[dict] | None:
    pools = _candidate_pool(rows, cols)
    dim_order = sorted(dims, key=lambda dim: len(pools[dim]))
    chosen: list[dict] = []
    used_counts: set[int] = set()
    used_cells: set[int] = set()

    def dfs(index: int) -> bool:
        if index == len(dim_order):
            return True
        dim = dim_order[index]
        candidates = pools[dim][:]
        random.shuffle(candidates)
        for cand in candidates:
            if cand["count"] in used_counts:
                continue
            if used_cells & cand["cells"]:
                continue
            chosen.append(cand)
            used_counts.add(int(cand["count"]))
            used_cells.update(cand["cells"])
            if dfs(index + 1):
                return True
            chosen.pop()
            used_counts.remove(int(cand["count"]))
            used_cells.difference_update(cand["cells"])
        return False

    if not dfs(0):
        return None

    dim_to_choice = {(cand["h"], cand["w"]): cand for cand in chosen}
    return [dim_to_choice[dim] for dim in dims]


def _render_input(rows: int, cols: int, rects: list[dict]) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    colors = random.sample(list(PALETTE), len(rects))
    for color, rect in zip(colors, rects):
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        h = int(rect["h"])
        w = int(rect["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(color)
    return grid


def _alt_order(specs: list[dict], key_name: str) -> list[int]:
    return [int(spec["color"]) for spec in sorted(specs, key=lambda spec: (-int(spec[key_name]), int(spec["c0"])))]


def _sample_is_informative(grid: np.ndarray, specs: list[dict], out: np.ndarray) -> bool:
    true_order = [int(spec["color"]) for spec in sorted(specs, key=lambda spec: (-int(spec["count"]), int(spec["c0"])))]
    if true_order == [int(spec["color"]) for spec in sorted(specs, key=lambda spec: int(spec["c0"]))]:
        return False

    if not any(spec["clipped"] > 0 for spec in specs):
        return False
    if not any(spec["clipped"] == 0 for spec in specs):
        return False
    if not any(spec["h"] != spec["w"] for spec in specs):
        return False

    distinct_alt_orders = 0
    for metric in ("area", "h", "w", "left", "right"):
        if _alt_order(specs, metric) != true_order:
            distinct_alt_orders += 1
    if distinct_alt_orders < 3:
        return False

    alternate_outputs = [
        _histogram_from_specs(specs, "count", lambda spec: int(spec["c0"])),
        _histogram_from_specs(specs, "area", lambda spec: (-int(spec["area"]), int(spec["c0"]))),
        _histogram_from_specs(specs, "h", lambda spec: (-int(spec["h"]), int(spec["c0"]))),
        _histogram_from_specs(specs, "w", lambda spec: (-int(spec["w"]), int(spec["c0"]))),
        _histogram_from_specs(specs, "left", lambda spec: (-int(spec["left"]), int(spec["c0"]))),
        _histogram_from_specs(specs, "right", lambda spec: (-int(spec["right"]), int(spec["c0"]))),
    ]
    if any(np.array_equal(out, alt) for alt in alternate_outputs):
        return False

    recomputed = _extract_rectangles(grid)
    if sorted((spec["color"], spec["count"]) for spec in recomputed) != sorted((spec["color"], spec["count"]) for spec in specs):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_SIDE_RANGE)
    cols = random.randint(*GRID_SIDE_RANGE)
    rect_count = random.choice(RECT_COUNT_CHOICES)

    dims = random.sample(list(DIMENSIONS), rect_count)
    if all(h == w for h, w in dims):
        non_square = [dim for dim in DIMENSIONS if dim[0] != dim[1] and dim not in dims]
        if not non_square:
            return None
        dims[random.randrange(rect_count)] = random.choice(non_square)

    rects = _choose_layout(rows, cols, dims)
    if rects is None:
        return None

    grid = _render_input(rows, cols, rects)
    try:
        specs = _extract_rectangles(grid)
        out = main(grid)
    except AssertionError:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(specs) != rect_count:
        return None
    if not _sample_is_informative(grid, specs, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-stamp histogram example")
