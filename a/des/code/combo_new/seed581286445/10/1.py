from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORS = tuple(range(10))
BRANCHES = ("under", "exact", "over")

GRID_MIN = 14
GRID_MAX = 30
FRAME_MARGIN = 1
FRAME_GAP = 1
PLACEMENT_TRIES = 200
GENERATION_TRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    expected: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0 + 1, r1):
        expected.add((r, c0))
        expected.add((r, c1))

    if set(points) != expected:
        return None
    return (int(r0), int(c0), int(r1), int(c1))


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[tuple[int, int, int, int], int]]:
    frames: list[tuple[tuple[int, int, int, int], int]] = []
    for color in sorted(set(map(int, grid.reshape(-1))) - {int(bg)}):
        for comp in _components(grid == int(color)):
            bbox = _frame_bbox(comp)
            if bbox is not None:
                frames.append((bbox, int(color)))
    frames.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[1]))
    return frames


def _ring_count(inner_h: int, inner_w: int) -> int:
    return int(min((inner_h + 1) // 2, (inner_w + 1) // 2))


def _paint_ring(grid: np.ndarray, top: int, left: int, bottom: int, right: int, color: int) -> None:
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region with inward color rings determined by its interior seed order.

    Input structure:
    - The background is the most common color.
    - One or more one-cell-thick rectangular frames sit on the background.
    - Each frame encloses background cells plus several single-pixel seed colors inside.

    Rule:
    - For each frame, read all non-background interior seed pixels in row-major order.
    - Starting from the ring directly inside the frame, fill successive inward rings with
      the corresponding seed colors.
    - If the frame has more rings than seeds, any deeper unassigned interior stays as background.
    - If there are more seeds than rings, ignore the extra seeds; the frame itself stays unchanged.

    Key insight:
    - Seed positions matter only through their top-to-bottom, left-to-right scan order, which
      defines the outward-to-inward ring color sequence for each closed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for (r0, c0, r1, c1), _frame_color in _find_frames(g, bg):
        seeds: list[int] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                color = int(g[r, c])
                if color != int(bg):
                    seeds.append(color)

        for layer, color in enumerate(seeds):
            top = int(r0 + 1 + layer)
            left = int(c0 + 1 + layer)
            bottom = int(r1 - 1 - layer)
            right = int(c1 - 1 - layer)
            if top > bottom or left > right:
                break
            _paint_ring(out, top, left, bottom, right, int(color))

    return out


def _choose_frame_count(rng: random.Random, height: int, width: int) -> int:
    span = min(height, width)
    if span < 20:
        return int(rng.choices([1, 2], weights=[1, 3])[0])
    if span < 24:
        return int(rng.choices([1, 2, 3], weights=[1, 3, 3])[0])
    return int(rng.choices([2, 3, 4], weights=[2, 3, 2])[0])


def _branch_mix(rng: random.Random, frame_count: int) -> list[str]:
    if frame_count == 1:
        return [str(rng.choice(BRANCHES))]

    chosen = list(rng.sample(BRANCHES, k=min(frame_count, len(BRANCHES))))
    while len(chosen) < frame_count:
        chosen.append(str(rng.choice(BRANCHES)))
    rng.shuffle(chosen)
    return chosen


def _sample_inner_dims(
    rng: random.Random, branch: str
) -> tuple[int, int, int, int] | None:
    for _ in range(200):
        if rng.random() < 0.3:
            if rng.random() < 0.5:
                inner_h = int(rng.randint(1, 3))
                inner_w = int(rng.randint(4, 8))
            else:
                inner_h = int(rng.randint(4, 8))
                inner_w = int(rng.randint(1, 3))
        else:
            inner_h = int(rng.randint(2, 8))
            inner_w = int(rng.randint(2, 8))

        if inner_h * inner_w <= 1:
            continue

        layers = _ring_count(inner_h, inner_w)
        max_seeds = min(8, inner_h * inner_w)
        if branch == "under" and layers < 2:
            continue
        if branch == "over" and layers + 1 > max_seeds:
            continue

        if branch == "under":
            seed_count = int(rng.randint(1, layers - 1))
        elif branch == "exact":
            seed_count = int(layers)
        else:
            seed_upper = min(max_seeds, layers + 3)
            seed_count = int(rng.randint(layers + 1, seed_upper))

        return inner_h, inner_w, layers, seed_count
    return None


def _placeable(
    reserved: np.ndarray, top: int, left: int, outer_h: int, outer_w: int
) -> bool:
    r0 = top - FRAME_GAP
    r1 = top + outer_h + FRAME_GAP
    c0 = left - FRAME_GAP
    c1 = left + outer_w + FRAME_GAP
    return not bool(reserved[r0:r1, c0:c1].any())


def _reserve(
    reserved: np.ndarray, top: int, left: int, outer_h: int, outer_w: int
) -> None:
    r0 = top - FRAME_GAP
    r1 = top + outer_h + FRAME_GAP
    c0 = left - FRAME_GAP
    c1 = left + outer_w + FRAME_GAP
    reserved[r0:r1, c0:c1] = True


def _draw_frame(grid: np.ndarray, top: int, left: int, outer_h: int, outer_w: int, color: int) -> None:
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    grid[top, left : right + 1] = int(color)
    grid[bottom, left : right + 1] = int(color)
    grid[top : bottom + 1, left] = int(color)
    grid[top : bottom + 1, right] = int(color)


def _unique_bg_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    return all(int(counts[bg]) > int(count) for color, count in counts.items() if int(color) != int(bg))


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GENERATION_TRIES):
        height = int(rng.randint(GRID_MIN, GRID_MAX))
        width = int(rng.randint(GRID_MIN, GRID_MAX))
        frame_count = _choose_frame_count(rng, height, width)
        branches = _branch_mix(rng, frame_count)

        bg = int(rng.choice(COLORS))
        border = int(rng.choice([color for color in COLORS if color != bg]))
        seed_palette = [color for color in COLORS if color not in (bg, border)]

        specs: list[dict[str, int | str]] = []
        ok = True
        for branch in branches:
            dims = _sample_inner_dims(rng, branch)
            if dims is None:
                ok = False
                break
            inner_h, inner_w, _layers, seed_count = dims
            specs.append(
                {
                    "branch": branch,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "outer_h": inner_h + 2,
                    "outer_w": inner_w + 2,
                    "seed_count": seed_count,
                }
            )
        if not ok:
            continue

        grid = np.full((height, width), bg, dtype=int)
        reserved = np.zeros((height, width), dtype=bool)
        placed: list[dict[str, int | str]] = []

        for spec in sorted(specs, key=lambda item: int(item["outer_h"]) * int(item["outer_w"]), reverse=True):
            outer_h = int(spec["outer_h"])
            outer_w = int(spec["outer_w"])
            top_min = FRAME_MARGIN
            left_min = FRAME_MARGIN
            top_max = height - outer_h - FRAME_MARGIN
            left_max = width - outer_w - FRAME_MARGIN
            if top_min > top_max or left_min > left_max:
                ok = False
                break

            placed_here = False
            for _ in range(PLACEMENT_TRIES):
                top = int(rng.randint(top_min, top_max))
                left = int(rng.randint(left_min, left_max))
                if not _placeable(reserved, top, left, outer_h, outer_w):
                    continue
                _reserve(reserved, top, left, outer_h, outer_w)
                spec_copy = dict(spec)
                spec_copy["top"] = top
                spec_copy["left"] = left
                placed.append(spec_copy)
                placed_here = True
                break

            if not placed_here:
                ok = False
                break

        if not ok:
            continue

        for spec in placed:
            top = int(spec["top"])
            left = int(spec["left"])
            inner_h = int(spec["inner_h"])
            inner_w = int(spec["inner_w"])
            outer_h = int(spec["outer_h"])
            outer_w = int(spec["outer_w"])
            seed_count = int(spec["seed_count"])

            _draw_frame(grid, top, left, outer_h, outer_w, border)

            interior_cells = [
                (r, c)
                for r in range(top + 1, top + 1 + inner_h)
                for c in range(left + 1, left + 1 + inner_w)
            ]
            seed_cells = rng.sample(interior_cells, seed_count)
            seed_colors = rng.sample(seed_palette, seed_count)
            for (r, c), color in zip(seed_cells, seed_colors):
                grid[r, c] = int(color)

        if len(_find_frames(grid, bg)) != frame_count:
            continue
        if not _unique_bg_dominant(grid, bg):
            continue

        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape or np.array_equal(out, grid):
            continue

        return grid

    raise RuntimeError("generate_input: exhausted retries")
