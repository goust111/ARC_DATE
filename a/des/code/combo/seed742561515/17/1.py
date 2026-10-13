from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_RETRIES = 400
GRID_SIZE_RANGE = (22, 30)
REGION_COUNT_RANGE = (2, 5)
COLOR_POOL = tuple(range(1, 10))
PATCH_MARGIN = 2
PATCH_PADDING = 1


def _neighbors(r: int, c: int, h: int, w: int, dirs: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors(cr, cc, h, w, dirs):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _edge_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(values.tolist()):
        if int(value) != BG:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(values)))
    return runs


def _detect_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray, str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, str, int, int, list[int]]] = []

    def consider(side: str, edge: np.ndarray, inward: np.ndarray | None) -> None:
        for start, end in _edge_runs(edge):
            colors = [int(x) for x in edge[start:end].tolist()]
            if len(colors) < 2 or len(set(colors)) != len(colors):
                continue
            if inward is not None and np.any(inward[start:end] != BG):
                continue
            candidates.append((len(colors), side, start, end, colors))

    consider("top", g[0, :], g[1, :] if h > 1 else None)
    consider("bottom", g[h - 1, :], g[h - 2, :] if h > 1 else None)
    consider("left", g[:, 0], g[:, 1] if w > 1 else None)
    consider("right", g[:, w - 1], g[:, w - 2] if w > 1 else None)

    assert candidates, "expected a legend bar on one edge"
    best = max(candidates, key=lambda item: item[0])
    _, side, start, end, colors = best

    mask = np.zeros_like(g, dtype=bool)
    if side == "top":
        mask[0, start:end] = True
    elif side == "bottom":
        mask[h - 1, start:end] = True
    elif side == "left":
        mask[start:end, 0] = True
    else:
        mask[start:end, w - 1] = True
    return colors, mask, side


def _classify_colors(grid: np.ndarray, legend_mask: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(x) for x in np.unique(g[~legend_mask]) if int(x) != BG)
    assert len(colors) == 2, "expected outline and marker colors outside the legend"

    marker_candidates: list[int] = []
    for color in colors:
        comps = _components((g == color) & (~legend_mask), connectivity=4)
        if comps and all(len(comp) == 1 for comp in comps):
            marker_candidates.append(color)

    assert len(marker_candidates) == 1, "expected a unique singleton marker color"
    marker = marker_candidates[0]
    outline = colors[0] if colors[1] == marker else colors[1]
    return outline, marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region according to the count of interior markers.

    The legend bar on the outer edge maps counts 1..N to fill colors in order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_colors, legend_mask, _legend_side = _detect_legend(g)
    outline_color, marker_color = _classify_colors(g, legend_mask)

    region_mask = (g != outline_color) & (~legend_mask)
    out = g.copy()
    h, w = g.shape

    for comp in _components(region_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        marker_count = sum(1 for r, c in comp if int(g[r, c]) == marker_color)
        if marker_count == 0:
            continue
        assert 1 <= marker_count <= len(legend_colors), "marker count must match legend range"
        fill_color = legend_colors[marker_count - 1]
        for r, c in comp:
            if int(g[r, c]) == BG:
                out[r, c] = fill_color
    return out


def _dilate(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    h, w = mask.shape
    dirs = DIR8 if connectivity == 8 else DIR4
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors(r, c, h, w, dirs):
            out[nr, nc] = True
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "mask must be nonempty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _random_blob_mask(inner_h: int, inner_w: int, min_area: int, max_area: int) -> np.ndarray | None:
    area = random.randint(min_area, max_area)
    start = (random.randrange(inner_h), random.randrange(inner_w))
    cells = {start}
    frontier = [start]

    while len(cells) < area and frontier:
        fr, fc = random.choice(frontier)
        candidates = []
        weights = []
        for nr, nc in _neighbors(fr, fc, inner_h, inner_w, DIR4):
            if (nr, nc) in cells:
                continue
            candidates.append((nr, nc))
            touch_count = sum((ar, ac) in cells for ar, ac in _neighbors(nr, nc, inner_h, inner_w, DIR4))
            weights.append(1 + touch_count)
        if not candidates:
            frontier.remove((fr, fc))
            continue
        picked = random.choices(candidates, weights=weights, k=1)[0]
        cells.add(picked)
        frontier.append(picked)

    if len(cells) < area:
        return None

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    r0, c0, r1, c1 = _bbox(mask)
    mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    if min(mask.shape) < 2 or int(mask.sum()) < min_area:
        return None
    return mask


def _notched_rect_mask(inner_h: int, inner_w: int, min_area: int) -> np.ndarray | None:
    mask = np.ones((inner_h, inner_w), dtype=bool)
    notch_count = random.randint(1, 3)
    for _ in range(notch_count):
        candidate = mask.copy()
        side = random.choice(["top", "bottom", "left", "right"])
        if side in {"top", "bottom"}:
            if candidate.shape[0] < 3 or candidate.shape[1] < 4:
                continue
            width = random.randint(1, candidate.shape[1] - 2)
            start = random.randint(0, candidate.shape[1] - width)
            depth = random.randint(1, min(2, candidate.shape[0] - 2))
            rows = slice(0, depth) if side == "top" else slice(candidate.shape[0] - depth, candidate.shape[0])
            candidate[rows, start : start + width] = False
        else:
            if candidate.shape[1] < 3 or candidate.shape[0] < 4:
                continue
            height = random.randint(1, candidate.shape[0] - 2)
            start = random.randint(0, candidate.shape[0] - height)
            depth = random.randint(1, min(2, candidate.shape[1] - 2))
            cols = slice(0, depth) if side == "left" else slice(candidate.shape[1] - depth, candidate.shape[1])
            candidate[start : start + height, cols] = False
        if int(candidate.sum()) < min_area:
            continue
        if len(_components(candidate, connectivity=4)) == 1:
            mask = candidate

    if int(mask.sum()) < min_area or min(mask.shape) < 2:
        return None
    if mask.all():
        return None
    return mask


def _sample_markers(interior_mask: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(interior_mask)]
    random.shuffle(coords)
    chosen: list[tuple[int, int]] = []
    for r, c in coords:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    return None


def _make_region_patch(marker_count: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(80):
        style = random.choice(["blob", "room"])
        inner_h = random.randint(3, 7)
        inner_w = random.randint(3, 7)
        min_area = max(marker_count + 3, 6)
        max_area = min(inner_h * inner_w, marker_count + random.randint(5, 12))
        if min_area > max_area:
            continue

        if style == "blob":
            interior = _random_blob_mask(inner_h, inner_w, min_area, max_area)
        else:
            if random.random() < 0.5:
                inner_h = random.randint(4, 7)
                inner_w = random.randint(4, 7)
            interior = _notched_rect_mask(inner_h, inner_w, min_area)
        if interior is None:
            continue

        patch_h, patch_w = interior.shape[0] + 2, interior.shape[1] + 2
        shifted = np.zeros((patch_h, patch_w), dtype=bool)
        shifted[1:-1, 1:-1] = interior
        outline = np.zeros_like(shifted)
        for r, c in np.argwhere(shifted):
            r = int(r)
            c = int(c)
            for nr, nc in _neighbors(r, c, patch_h, patch_w, DIR4):
                if not shifted[nr, nc]:
                    outline[nr, nc] = True
        if not outline.any():
            continue

        markers = _sample_markers(shifted, marker_count)
        if markers is None:
            continue
        region_cells = [(int(r), int(c)) for r, c in np.argwhere(shifted)]
        return outline, region_cells, markers
    return None


def _legend_start(length: int, total: int) -> int:
    assert total >= length + 2
    return random.randint(1, total - length - 1)


def _place_legend(grid: np.ndarray, colors: list[int], side: str) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    h, w = grid.shape
    n = len(colors)

    if side == "top":
        start = _legend_start(n, w)
        grid[0, start : start + n] = colors
        mask[0, start : start + n] = True
    elif side == "bottom":
        start = _legend_start(n, w)
        grid[h - 1, start : start + n] = colors
        mask[h - 1, start : start + n] = True
    elif side == "left":
        start = _legend_start(n, h)
        for idx, color in enumerate(colors):
            grid[start + idx, 0] = int(color)
            mask[start + idx, 0] = True
    else:
        start = _legend_start(n, h)
        for idx, color in enumerate(colors):
            grid[start + idx, w - 1] = int(color)
            mask[start + idx, w - 1] = True
    return mask


def _can_place(blocked: np.ndarray, patch_mask: np.ndarray, top: int, left: int) -> bool:
    ph, pw = patch_mask.shape
    return not np.any(blocked[top : top + ph, left : left + pw] & patch_mask)


def _apply_patch(
    grid: np.ndarray,
    outline_color: int,
    marker_color: int,
    outline: np.ndarray,
    region_cells: list[tuple[int, int]],
    markers: list[tuple[int, int]],
    top: int,
    left: int,
) -> dict[str, object]:
    for r, c in np.argwhere(outline):
        grid[top + int(r), left + int(c)] = outline_color
    for r, c in markers:
        grid[top + r, left + c] = marker_color
    return {
        "cells": [(top + r, left + c) for r, c in region_cells],
        "markers": {(top + r, left + c) for r, c in markers},
        "count": len(markers),
    }


def _expected_output(grid: np.ndarray, regions: list[dict[str, object]], legend_colors: list[int]) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        fill = legend_colors[int(region["count"]) - 1]
        marker_cells = region["markers"]
        for r, c in region["cells"]:
            if (r, c) not in marker_cells:
                out[r, c] = fill
    return out


def _generate_one() -> np.ndarray | None:
    h = random.randint(*GRID_SIZE_RANGE)
    w = random.randint(*GRID_SIZE_RANGE)
    region_count = random.randint(*REGION_COUNT_RANGE)
    if min(h, w) < region_count + 6:
        return None

    grid = np.full((h, w), BG, dtype=int)
    colors = random.sample(COLOR_POOL, region_count + 2)
    outline_color = int(colors[0])
    marker_color = int(colors[1])
    legend_colors = [int(x) for x in colors[2:]]
    side = random.choice(["top", "bottom", "left", "right"])
    legend_mask = _place_legend(grid, legend_colors, side)

    blocked = _dilate(legend_mask, connectivity=8)
    regions: list[dict[str, object]] = []
    counts = list(range(1, region_count + 1))
    random.shuffle(counts)

    for marker_count in counts:
        placed = False
        for _ in range(120):
            patch = _make_region_patch(marker_count)
            if patch is None:
                continue
            outline, region_cells, markers = patch
            patch_mask = outline.copy()
            for r, c in region_cells:
                patch_mask[r, c] = True
            ph, pw = patch_mask.shape
            if h - ph - PATCH_MARGIN < PATCH_MARGIN or w - pw - PATCH_MARGIN < PATCH_MARGIN:
                continue
            top = random.randint(PATCH_MARGIN, h - ph - PATCH_MARGIN)
            left = random.randint(PATCH_MARGIN, w - pw - PATCH_MARGIN)
            if not _can_place(blocked, patch_mask, top, left):
                continue

            region = _apply_patch(
                grid=grid,
                outline_color=outline_color,
                marker_color=marker_color,
                outline=outline,
                region_cells=region_cells,
                markers=markers,
                top=top,
                left=left,
            )
            occupied = np.zeros_like(grid, dtype=bool)
            occupied[top : top + ph, left : left + pw] = patch_mask
            blocked |= _dilate(occupied, connectivity=8)
            regions.append(region)
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, regions, legend_colors)
    if not np.array_equal(main(grid), expected):
        return None

    legend_detected, legend_mask2, _ = _detect_legend(grid)
    if legend_detected != legend_colors or not np.array_equal(legend_mask, legend_mask2):
        return None

    outline_detected, marker_detected = _classify_colors(grid, legend_mask)
    if outline_detected != outline_color or marker_detected != marker_color:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
