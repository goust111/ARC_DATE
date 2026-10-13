from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class ObjectInfo:
    color: int
    coords: np.ndarray
    top: int
    bottom: int
    left: int
    right: int


def _find_objects(grid: np.ndarray) -> list[ObjectInfo]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[ObjectInfo] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (BLACK, GREY):
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.asarray(pts, dtype=int)
            rows = coords[:, 0]
            cols = coords[:, 1]
            objects.append(
                ObjectInfo(
                    color=color,
                    coords=coords,
                    top=int(rows.min()),
                    bottom=int(rows.max()),
                    left=int(cols.min()),
                    right=int(cols.max()),
                )
            )

    objects.sort(key=lambda obj: (-obj.bottom, obj.top, obj.left))
    return objects


def _interval_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    left = max(a[0], b[0])
    right = min(a[1], b[1])
    return max(0, right - left + 1)


def _resolve_colors(objects: list[ObjectInfo]) -> list[int]:
    resolved = [obj.color for obj in objects]

    for i, obj in enumerate(objects):
        candidates: list[tuple[int, int, int]] = []
        for j in range(i):
            lower = objects[j]
            if lower.bottom <= obj.bottom:
                continue
            overlap = _interval_overlap((obj.left, obj.right), (lower.left, lower.right))
            if overlap <= 0:
                continue
            candidates.append((overlap, lower.bottom, j))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        resolved[i] = resolved[candidates[0][2]]

    return resolved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor higher floating objects by the color of the closest lower object whose
    shadow interval overlaps theirs, propagating colors upward through the stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 5 and g.shape[1] >= 6, "grid is too small"
    assert np.all(g[-1, :] == GREY), "expected a grey horizon on the bottom row"
    assert not np.any(g[:-1, :] == GREY), "grey is reserved for the horizon"

    objects = _find_objects(g)
    assert len(objects) >= 2, "expected multiple floating foreground objects"

    out = g.copy()
    resolved = _resolve_colors(objects)
    for obj, color in zip(objects, resolved):
        if color == obj.color:
            continue
        out[obj.coords[:, 0], obj.coords[:, 1]] = color

    return out


GEN_MIN_OBJECTS = 3
GEN_MAX_OBJECTS = 6
GEN_MIN_GRID_W = 12
GEN_MAX_GRID_W = 20
GEN_MAX_GRID_H = 18
GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 100
POLYOMINO_ATTEMPTS = 80
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SCENARIOS = ("simple", "chain", "multi", "mixed")
SCENARIO_WEIGHTS = (4, 3, 2, 1)


def _random_polyomino(rng: random.Random) -> np.ndarray:
    for _ in range(POLYOMINO_ATTEMPTS):
        max_h = rng.randint(2, 4)
        max_w = rng.randint(2, 5)
        target = rng.randint(3, min(max_h * max_w, 8))

        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    pt = (r + dr, c + dc)
                    if pt in cells:
                        continue
                    trial = cells | {pt}
                    rows = [rr for rr, _ in trial]
                    cols = [cc for _, cc in trial]
                    if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                        frontier.add(pt)
            if not frontier:
                break

            candidates = sorted(frontier)
            weights = [
                1 + sum((pt[0] + dr, pt[1] + dc) in cells for dr, dc in DIR4)
                for pt in candidates
            ]
            cells.add(rng.choices(candidates, weights=weights, k=1)[0])

        if len(cells) != target:
            continue

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < 2 or w < 2:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        return mask

    h = rng.randint(2, 4)
    w = rng.randint(2, 4)
    return np.ones((h, w), dtype=bool)


def _sample_vertical_layout(
    heights: list[int], rng: random.Random
) -> tuple[list[int], int] | None:
    n = len(heights)
    top_margin = rng.randint(0, 2)
    between = [rng.randint(1, 2) for _ in range(n - 1)]
    gap_to_horizon = rng.randint(1, 3)

    tops = [0] * n
    current_top = top_margin
    for idx in range(n - 1, -1, -1):
        tops[idx] = current_top
        if idx > 0:
            current_top += heights[idx] + between[idx - 1]

    grid_h = tops[0] + heights[0] + gap_to_horizon + 1
    if grid_h > GEN_MAX_GRID_H:
        return None
    return tops, grid_h


def _guided_left(
    width: int,
    grid_w: int,
    lowers: list[dict[str, int]],
    scenario: str,
    rng: random.Random,
) -> int:
    max_left = grid_w - width
    if not lowers:
        return rng.randint(0, max_left)

    if scenario in ("multi", "mixed") and len(lowers) >= 2 and rng.random() < 0.7:
        subset = lowers[-2:] if rng.random() < 0.7 else rng.sample(lowers, 2)
        lo = max(0, max(obj["left"] - width + 1 for obj in subset))
        hi = min(max_left, min(obj["right"] for obj in subset))
        if lo <= hi:
            return rng.randint(lo, hi)

    if scenario in ("chain", "mixed") and rng.random() < 0.7:
        anchor = lowers[-1]
        lo = max(0, anchor["left"] - width + 1)
        hi = min(max_left, anchor["right"])
        if lo <= hi:
            return rng.randint(lo, hi)

    if rng.random() < 0.65:
        anchor = rng.choice(lowers)
        lo = max(0, anchor["left"] - width + 1)
        hi = min(max_left, anchor["right"])
        if lo <= hi:
            return rng.randint(lo, hi)

    return rng.randint(0, max_left)


def _relation_summary(objects: list[ObjectInfo]) -> dict[str, int | bool]:
    parents = [None] * len(objects)
    depths = [0] * len(objects)
    changed = 0
    multi = 0
    ambiguous = False

    for i, obj in enumerate(objects):
        candidates: list[tuple[int, int]] = []
        for j in range(i):
            lower = objects[j]
            if lower.bottom <= obj.bottom:
                continue
            overlap = _interval_overlap((obj.left, obj.right), (lower.left, lower.right))
            if overlap > 0:
                candidates.append((overlap, j))

        if len(candidates) >= 2:
            multi += 1
        if not candidates:
            continue

        best_overlap = max(overlap for overlap, _ in candidates)
        best = [j for overlap, j in candidates if overlap == best_overlap]
        if len(best) != 1:
            ambiguous = True
            continue

        parent = best[0]
        parents[i] = parent
        depths[i] = depths[parent] + 1
        changed += 1

    return {
        "changed": changed,
        "multi": multi,
        "max_depth": max(depths, default=0),
        "ambiguous": ambiguous,
    }


def _scenario_ok(summary: dict[str, int | bool], scenario: str) -> bool:
    if bool(summary["ambiguous"]) or int(summary["changed"]) == 0:
        return False
    if scenario == "chain":
        return int(summary["max_depth"]) >= 2
    if scenario == "multi":
        return int(summary["multi"]) >= 1
    if scenario == "mixed":
        return int(summary["max_depth"]) >= 2 and int(summary["multi"]) >= 1
    return True


def _build_grid(
    masks: list[np.ndarray],
    tops: list[int],
    lefts: list[int],
    colors: list[int],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    grid[-1, :] = GREY

    for mask, top, left, color in zip(masks, tops, lefts, colors):
        rows, cols = np.nonzero(mask)
        rr = rows + top
        cc = cols + left
        assert not np.any(grid[rr, cc] != BLACK), "objects must not overlap"
        grid[rr, cc] = color

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    scenario = rng.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]
    n_objects = rng.randint(GEN_MIN_OBJECTS, GEN_MAX_OBJECTS)

    masks = [_random_polyomino(rng) for _ in range(n_objects)]
    heights = [int(mask.shape[0]) for mask in masks]
    layout = _sample_vertical_layout(heights, rng)
    if layout is None:
        return None
    tops, grid_h = layout

    max_width = max(int(mask.shape[1]) for mask in masks)
    min_grid_w = max(GEN_MIN_GRID_W, max_width + 2)
    if min_grid_w > GEN_MAX_GRID_W:
        return None
    grid_w = rng.randint(min_grid_w, GEN_MAX_GRID_W)
    colors = rng.sample(PALETTE, n_objects)

    for _ in range(PLACEMENT_ATTEMPTS):
        lefts: list[int] = []
        lowers: list[dict[str, int]] = []
        for mask, top in zip(masks, tops):
            width = int(mask.shape[1])
            left = _guided_left(width, grid_w, lowers, scenario, rng)
            lefts.append(left)
            lowers.append(
                {
                    "top": top,
                    "bottom": top + int(mask.shape[0]) - 1,
                    "left": left,
                    "right": left + width - 1,
                }
            )

        grid = _build_grid(masks, tops, lefts, colors, grid_h, grid_w)
        objects = _find_objects(grid)
        if len(objects) != n_objects:
            continue

        summary = _relation_summary(objects)
        if not _scenario_ok(summary, scenario):
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
