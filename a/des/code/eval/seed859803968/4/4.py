from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BG = 8
BORDER = 3
PEEL_COLORS = (4, 2, 9)  # yellow, red, maroon
ISLAND_COLORS = (1, 5, 6, 7)

GEN_RETRIES = 400
LOCAL_RETRIES = 200

# Canonical polyominoes up to area 4. Orientations are sampled by rotation/flip.
BASE_ISLAND_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    labels = np.zeros((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    next_id = 0

    for r in range(h):
        for c in range(w):
            if not m[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            labels[r, c] = next_id
            q = deque([(int(r), int(c))])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not m[nr, nc] or labels[nr, nc] != 0:
                        continue
                    labels[nr, nc] = next_id
                    q.append((nr, nc))
            comps.append(cells)
    return labels, comps


def _touches_edge(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells)


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _region_info(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    _, border_comps = _label_components(g == BORDER)
    info: list[dict] = []
    for border_cells in border_comps:
        br0, bc0, br1, bc1 = _bbox(border_cells)
        if br1 - br0 < 2 or bc1 - bc0 < 2:
            continue

        local = g[br0 : br1 + 1, bc0 : bc1 + 1]
        local_h, local_w = local.shape
        border_mask = local == BORDER
        open_mask = ~border_mask

        outside = np.zeros_like(border_mask, dtype=bool)
        q = deque()
        for r in range(local_h):
            for c in range(local_w):
                if (r not in (0, local_h - 1)) and (c not in (0, local_w - 1)):
                    continue
                if not open_mask[r, c] or outside[r, c]:
                    continue
                outside[r, c] = True
                q.append((r, c))

        while q:
            r, c = q.popleft()
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < local_h and 0 <= nc < local_w):
                    continue
                if not open_mask[nr, nc] or outside[nr, nc]:
                    continue
                outside[nr, nc] = True
                q.append((nr, nc))

        inside_mask = open_mask & (~outside)
        if not inside_mask.any():
            continue

        island_mask = inside_mask & (local != BG)
        _, island_comps = _label_components(island_mask)
        island_count = len(island_comps)

        peel_mask = inside_mask & (local == BG)
        layer0_local = [
            (r, c)
            for r in range(local_h)
            for c in range(local_w)
            if peel_mask[r, c]
            and any(
                0 <= r + dr < local_h
                and 0 <= c + dc < local_w
                and border_mask[r + dr, c + dc]
                for dr, dc in DIR4
            )
        ]

        dist: dict[tuple[int, int], int] = {}
        q = deque()
        for cell in layer0_local:
            dist[cell] = 0
            q.append(cell)

        while q:
            r, c = q.popleft()
            next_dist = dist[(r, c)] + 1
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                nxt = (nr, nc)
                if not (0 <= nr < local_h and 0 <= nc < local_w):
                    continue
                if not peel_mask[nr, nc] or nxt in dist:
                    continue
                dist[nxt] = next_dist
                q.append(nxt)

        layers = [[] for _ in range(len(PEEL_COLORS))]
        peel_cells: list[tuple[int, int]] = []
        for (r, c), depth in dist.items():
            gr = br0 + r
            gc = bc0 + c
            peel_cells.append((gr, gc))
            if depth < len(PEEL_COLORS):
                layers[depth].append((gr, gc))

        bg_cells = [
            (br0 + r, bc0 + c)
            for r, c in zip(*np.where(inside_mask & (local == BG)), strict=False)
        ]

        info.append(
            {
                "boundary_bbox": (br0, bc0, br1, bc1),
                "bbox": _bbox(bg_cells) if bg_cells else (br0, bc0, br1, bc1),
                "cells": peel_cells,
                "islands": island_count,
                "layers": layers,
            }
        )

    info.sort(key=lambda item: item["boundary_bbox"])
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count islands inside each green-bounded region, then peel that many background
    layers inward with the fixed yellow/red/maroon sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for region in _region_info(g):
        n_layers = min(int(region["islands"]), len(PEEL_COLORS))
        for depth in range(n_layers):
            if not region["layers"][depth]:
                break
            for r, c in region["layers"][depth]:
                out[r, c] = PEEL_COLORS[depth]
    return out


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(norm)


def _transform_shape(
    cells: tuple[tuple[int, int], ...], rot: int, flip: bool
) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in cells]
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    return _normalize_shape(tuple(pts))


def _sample_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    base = rng.choice(BASE_ISLAND_SHAPES)
    return _transform_shape(base, rng.randrange(4), rng.random() < 0.5)


def _sample_region_counts(rng: random.Random, n_regions: int) -> list[int]:
    counts = [0, 1, 2, 3]
    rng.shuffle(counts)
    chosen = counts[:n_regions]
    if all(count == 0 for count in chosen):
        chosen[0] = rng.choice((1, 2, 3))
    return chosen


def _sample_region_spec(rng: random.Random, island_count: int) -> dict:
    min_side = max(4, 2 * island_count + 2)
    max_side = min_side + 2
    inner_h = rng.randint(min_side, max_side)
    inner_w = rng.randint(min_side, max_side + 1)
    if rng.random() < 0.5:
        inner_h, inner_w = inner_w, inner_h
    return {
        "islands": island_count,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
    }


def _split_rows(specs: list[dict], rng: random.Random) -> list[list[dict]]:
    items = specs[:]
    rng.shuffle(items)
    n = len(items)
    if n == 2:
        return [items] if rng.random() < 0.5 else [[items[0]], [items[1]]]
    if n == 3:
        return [items[:2], items[2:]] if rng.random() < 0.5 else [items[:1], items[1:]]
    return [items[:2], items[2:]]


def _arrange_regions(
    specs: list[dict], rng: random.Random
) -> tuple[list[dict], int, int] | None:
    rows = _split_rows(specs, rng)
    row_used_widths = []
    row_band_heights = []
    row_gaps = []
    for row in rows:
        gaps = [rng.randint(2, 4) for _ in range(max(0, len(row) - 1))]
        row_used_width = sum(spec["outer_w"] for spec in row) + sum(gaps)
        band_height = max(spec["outer_h"] for spec in row) + rng.randint(0, 2)
        row_used_widths.append(row_used_width)
        row_band_heights.append(band_height)
        row_gaps.append(gaps)

    top_margin = rng.randint(1, 3)
    bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)
    between_rows = [rng.randint(2, 4) for _ in range(max(0, len(rows) - 1))]
    extra_w = rng.randint(0, 3)

    grid_h = top_margin + bottom_margin + sum(row_band_heights) + sum(between_rows)
    grid_w = left_margin + right_margin + max(row_used_widths) + extra_w
    if not (14 <= grid_h <= 30 and 14 <= grid_w <= 30):
        return None

    placements: list[dict] = []
    y = top_margin
    for row_index, row in enumerate(rows):
        row_used_width = row_used_widths[row_index]
        band_height = row_band_heights[row_index]
        x = rng.randint(left_margin, grid_w - right_margin - row_used_width)
        for item_index, spec in enumerate(row):
            r0 = y + rng.randint(0, band_height - spec["outer_h"])
            c0 = x
            placed = dict(spec)
            placed["top"] = r0
            placed["left"] = c0
            placements.append(placed)
            x += spec["outer_w"]
            if item_index < len(row_gaps[row_index]):
                x += row_gaps[row_index][item_index]
        if row_index < len(between_rows):
            y += band_height + between_rows[row_index]
    return placements, grid_h, grid_w


def _place_islands(grid: np.ndarray, spec: dict, rng: random.Random) -> bool:
    top = int(spec["top"])
    left = int(spec["left"])
    inner_h = int(spec["inner_h"])
    inner_w = int(spec["inner_w"])
    n_islands = int(spec["islands"])

    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    for _ in range(n_islands):
        placed = False
        for _ in range(LOCAL_RETRIES):
            shape = _sample_shape(rng)
            shape_h = max(r for r, _ in shape) + 1
            shape_w = max(c for _, c in shape) + 1
            max_r = inner_h - shape_h - 1
            max_c = inner_w - shape_w - 1
            if max_r < 1 or max_c < 1:
                continue
            base_r = rng.randint(1, max_r)
            base_c = rng.randint(1, max_c)
            cells = [(base_r + r, base_c + c) for r, c in shape]
            if any(occupied[r, c] for r, c in cells):
                continue
            if any(
                0 <= r + dr < inner_h
                and 0 <= c + dc < inner_w
                and occupied[r + dr, c + dc]
                for r, c in cells
                for dr, dc in DIR4
            ):
                continue
            color = rng.choice(ISLAND_COLORS)
            for r, c in cells:
                occupied[r, c] = True
                grid[top + 1 + r, left + 1 + c] = color
            placed = True
            break
        if not placed:
            return False
    return True


def _build_grid(specs: list[dict], rng: random.Random) -> tuple[np.ndarray, list[dict]] | None:
    arranged = _arrange_regions(specs, rng)
    if arranged is None:
        return None

    placements, grid_h, grid_w = arranged
    if not (14 <= grid_h <= 30 and 14 <= grid_w <= 30):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in placements:
        top = int(spec["top"])
        left = int(spec["left"])
        bottom = top + int(spec["outer_h"]) - 1
        right = left + int(spec["outer_w"]) - 1
        grid[top : bottom + 1, left : right + 1] = BORDER
        grid[top + 1 : bottom, left + 1 : right] = BG

    for spec in placements:
        if not _place_islands(grid, spec, rng):
            return None
    return grid, placements


def _validate_case(grid: np.ndarray, expected_specs: list[dict]) -> bool:
    region_info = _region_info(grid)
    if len(region_info) != len(expected_specs):
        return False

    expected = sorted(
        (
            spec["top"] + 1,
            spec["left"] + 1,
            int(spec["islands"]),
        )
        for spec in expected_specs
    )
    actual = sorted(
        (
            info["bbox"][0],
            info["bbox"][1],
            int(info["islands"]),
        )
        for info in region_info
    )
    if expected != actual:
        return False

    if not any(info["islands"] > 0 for info in region_info):
        return False

    for info in region_info:
        for depth in range(min(int(info["islands"]), len(PEEL_COLORS))):
            if not info["layers"][depth]:
                return False

    out = main(grid)
    color_counts = {color: int(np.count_nonzero(out == color)) for color in PEEL_COLORS}
    needed_depths = max(int(info["islands"]) for info in region_info)
    for depth in range(needed_depths):
        if depth >= len(PEEL_COLORS):
            break
        if any(int(info["islands"]) > depth for info in region_info) and color_counts[PEEL_COLORS[depth]] == 0:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.choices((2, 3, 4), weights=(2, 2, 1), k=1)[0]
    island_counts = _sample_region_counts(rng, n_regions)
    specs = [_sample_region_spec(rng, count) for count in island_counts]
    built = _build_grid(specs, rng)
    if built is None:
        return None
    grid, placements = built
    if not _validate_case(grid, placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
