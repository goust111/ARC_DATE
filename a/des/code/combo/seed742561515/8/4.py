from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
OUTER_H_RANGE = (5, 8)
OUTER_W_RANGE = (5, 8)
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (2, 4, 3, 1)
MAX_MARKERS = 8


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _ray_cells_to_edge(
    h: int, w: int, start: tuple[int, int], direction: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _frame_direction(
    gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int
) -> tuple[int, int] | None:
    gr, gc = gap
    choices: list[tuple[int, int]] = []
    if gr == r0 and c0 < gc < c1:
        choices.append((-1, 0))
    if gr == r1 and c0 < gc < c1:
        choices.append((1, 0))
    if gc == c0 and r0 < gr < r1:
        choices.append((0, -1))
    if gc == c1 and r0 < gr < r1:
        choices.append((0, 1))
    if len(choices) != 1:
        return None
    return choices[0]


def _extract_frames(grid: np.ndarray, strict: bool = True) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    infos: list[dict] = []
    for comp in _components(g == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            if strict:
                raise AssertionError("grey component too small to be a framed region")
            continue

        comp_set = set(comp)
        border = set(_border_cells(r0, c0, r1, c1))
        if not comp_set <= border:
            if strict:
                raise AssertionError("grey component must lie on a rectangle border")
            continue

        missing = list(border - comp_set)
        if len(missing) != 1:
            if strict:
                raise AssertionError("each frame must have exactly one border gap")
            continue

        gap = missing[0]
        direction = _frame_direction(gap, r0, c0, r1, c1)
        if direction is None:
            if strict:
                raise AssertionError("gap must lie on exactly one non-corner edge")
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            if strict:
                raise AssertionError("frame interior must be non-empty")
            continue
        if np.any(inner == GREY):
            if strict:
                raise AssertionError("frame interior may not contain grey border pixels")
            continue

        marker_mask = (inner != BG) & (inner != GREY)
        marker_count = int(marker_mask.sum())
        colors = np.unique(inner[marker_mask])
        if marker_count == 0 or colors.size != 1:
            if strict:
                raise AssertionError("frame interior must contain one marker color")
            continue

        marker_color = int(colors[0])
        marker_coords = [
            (int(r), int(c))
            for r, c in np.argwhere(marker_mask)
            for r, c in [(r + r0 + 1, c + c0 + 1)]
        ]
        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "direction": direction,
                "marker_color": marker_color,
                "marker_count": marker_count,
                "marker_coords": marker_coords,
            }
        )
    if strict and not infos:
        raise AssertionError("expected at least one grey frame")
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = g.copy()
    frames = _extract_frames(g, strict=True)

    for frame in frames:
        for r, c in frame["marker_coords"]:
            out[r, c] = BG

    for frame in frames:
        color = int(frame["marker_color"])
        length = int(frame["marker_count"])
        gr, gc = frame["gap"]
        dr, dc = frame["direction"]
        for step in range(length):
            rr = gr + dr * step
            cc = gc + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if g[rr, cc] != BG or out[rr, cc] != BG:
                break
            out[rr, cc] = color

    return out


def _sample_object_count(rng: random.Random) -> int:
    return rng.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]


def _gap_for_side(
    rng: random.Random, r0: int, c0: int, r1: int, c1: int, direction: tuple[int, int]
) -> tuple[int, int]:
    if direction == (-1, 0):
        return (r0, rng.randint(c0 + 1, c1 - 1))
    if direction == (1, 0):
        return (r1, rng.randint(c0 + 1, c1 - 1))
    if direction == (0, -1):
        return (rng.randint(r0 + 1, r1 - 1), c0)
    return (rng.randint(r0 + 1, r1 - 1), c1)


def _try_place_frame(
    rng: random.Random, grid: np.ndarray, blocked: np.ndarray
) -> dict | None:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        outer_h = rng.randint(OUTER_H_RANGE[0], min(OUTER_H_RANGE[1], h - 2))
        outer_w = rng.randint(OUTER_W_RANGE[0], min(OUTER_W_RANGE[1], w - 2))
        if outer_h >= h or outer_w >= w:
            continue

        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        direction = rng.choice(DIRS4)
        gap = _gap_for_side(rng, r0, c0, r1, c1, direction)

        ray_cells = _ray_cells_to_edge(h, w, gap, direction)
        ray_capacity = len(ray_cells)
        inner_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        inner_area = len(inner_cells)
        if inner_area == 0:
            continue

        truncate = rng.random() < 0.5
        if truncate:
            min_markers = ray_capacity + 1
            max_markers = min(inner_area, MAX_MARKERS)
            if min_markers > max_markers:
                continue
            marker_count = rng.randint(min_markers, max_markers)
            reserved_ray = ray_cells
        else:
            max_markers = min(inner_area, ray_capacity, MAX_MARKERS)
            if max_markers < 1:
                continue
            marker_count = rng.randint(1, max_markers)
            reserved_ray = ray_cells[:marker_count]

        border_cells = [cell for cell in _border_cells(r0, c0, r1, c1) if cell != gap]
        frame_box = {
            (rr, cc)
            for rr in range(max(0, r0 - 1), min(h, r1 + 2))
            for cc in range(max(0, c0 - 1), min(w, c1 + 2))
        }
        needed = frame_box | set(reserved_ray)
        if any(blocked[r, c] for r, c in needed):
            continue

        marker_cells = rng.sample(inner_cells, marker_count)
        color = rng.choice(MARKER_COLORS)

        for r, c in border_cells:
            grid[r, c] = GREY
        for r, c in marker_cells:
            grid[r, c] = color
        for r, c in needed:
            blocked[r, c] = True

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "direction": direction,
            "marker_color": color,
            "marker_count": marker_count,
            "marker_cells": marker_cells,
            "truncate": truncate,
        }
    return None


def _validate_generated_grid(grid: np.ndarray, expected_count: int) -> None:
    frames = _extract_frames(grid, strict=True)
    if len(frames) != expected_count:
        raise AssertionError("generator placed an unexpected number of frames")

    out = main(grid)
    if np.array_equal(out, grid):
        raise AssertionError("generator produced a no-op sample")


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(*GRID_H_RANGE)
        w = rng.randint(*GRID_W_RANGE)
        grid = np.full((h, w), BG, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        target_count = _sample_object_count(rng)
        placed = 0

        for _ in range(target_count):
            info = _try_place_frame(rng, grid, blocked)
            if info is None:
                break
            placed += 1

        if placed != target_count:
            continue

        _validate_generated_grid(grid, placed)
        return grid

    raise RuntimeError("generate_input: exhausted attempts")
