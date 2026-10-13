from __future__ import annotations

from collections import deque, defaultdict
import random

import numpy as np


Coord = tuple[int, int]
MarkerInfo = tuple[int, int, int, int]  # (row, col, color, black_count)

BLACK = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4, 5)
OUTPUT_WIDTH = 3

GRID_H_RANGE = range(12, 17)
GRID_W_RANGE = range(8, 13)
GEN_ATTEMPTS = 300
PROFILE_CYCLE = ((5, 9), (9, 5), (5, 5), (9, 9))

_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], tuple[int, ...]]] = set()


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trace_bounce_path(h: int, w: int, start_c: int, dc: int) -> list[Coord]:
    r = h - 1
    c = start_c
    out: list[Coord] = []
    step_c = dc
    while True:
        out.append((r, c))
        nr = r - 1
        if nr < 0:
            break
        nc = c + step_c
        if nc < 0 or nc >= w:
            step_c = -step_c
            nc = c + step_c
        r, c = nr, nc
    return out


def _label_passable(passable: np.ndarray) -> tuple[np.ndarray, list[list[Coord]]]:
    h, w = passable.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[list[Coord]] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if not passable[r, c] or labels[r, c] >= 0:
                continue
            q: deque[Coord] = deque([(r, c)])
            labels[r, c] = cid
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(comp)
            cid += 1
    return labels, comps


def _pack_summary(markers: list[MarkerInfo]) -> np.ndarray:
    rows = sum((black_count + OUTPUT_WIDTH - 1) // OUTPUT_WIDTH for _r, _c, _color, black_count in markers)
    out = np.full((rows, OUTPUT_WIDTH), BLACK, dtype=int)
    out_row = 0
    for _r, _c, color, black_count in markers:
        for idx in range(black_count):
            out[out_row + idx // OUTPUT_WIDTH, idx % OUTPUT_WIDTH] = color
        out_row += (black_count + OUTPUT_WIDTH - 1) // OUTPUT_WIDTH
    return out


def _collect_markers(grid: np.ndarray) -> list[MarkerInfo]:
    g = np.asarray(grid, dtype=int)
    passable = g != BLUE
    labels, comps = _label_passable(passable)

    marker_points = [(int(r), int(c)) for r, c in np.argwhere((g != BLACK) & (g != BLUE))]
    assert 2 <= len(marker_points) <= 4

    used_components: set[int] = set()
    markers: list[MarkerInfo] = []
    for r, c in marker_points:
        cid = int(labels[r, c])
        assert cid >= 0
        assert cid not in used_components
        used_components.add(cid)
        black_count = sum(1 for rr, cc in comps[cid] if int(g[rr, cc]) == BLACK)
        assert 1 <= black_count <= 9
        markers.append((r, c, int(g[r, c]), black_count))
    markers.sort(key=lambda item: (item[1], item[0]))
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """Use the blue bouncing diagonal as a wall, count each marked region's black area, and compress those counts into 3-wide color rows.

    Input structure:
    - A black background contains one blue diagonal bouncing path starting from a single bottom-edge blue pixel and continuing upward with left/right wall reflections until the top edge.
    - The remaining non-blue foreground cells are 2 to 4 isolated single-pixel markers of distinct colors chosen from red, green, yellow, and grey.
    - The blue path together with the grid boundary partitions the non-blue-free space into 4-connected regions, with exactly one marker in each marked region.

    Rule:
    - Treat every blue cell as a wall and treat the area outside the grid as blocked by the outer border.
    - For each marker, flood-fill its 4-connected non-blue region and count only the black cells in that region, excluding the marker itself and excluding blue wall cells.
    - Sort markers by input column, breaking ties by row, then for each marker start on a fresh output row and fill its color left-to-right in a 3-column summary for as many cells as its counted black area requires, padding unfinished rows with black.

    Key insight:
    - The blue diagonal is not recolored decoration: its exact reflected staircase geometry creates the region boundaries whose areas are later encoded by the marker colors in packed row blocks.
    """
    markers = _collect_markers(np.asarray(grid, dtype=int))
    return _pack_summary(markers)


def _analyze_base_config(h: int, w: int, start_c: int, start_dc: int) -> dict | None:
    path = _trace_bounce_path(h, w, start_c, start_dc)
    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in path:
        grid[r, c] = BLUE

    labels, comps = _label_passable(grid != BLUE)
    regions = []
    for cid, comp in enumerate(comps):
        black_count = len(comp) - 1
        if not (1 <= black_count <= 9):
            continue
        marker_cells = [(r, c) for r, c in comp if 0 < r < h - 1 and 0 < c < w - 1]
        if not marker_cells:
            continue
        regions.append(
            {
                "cid": cid,
                "cells": tuple(comp),
                "marker_cells": tuple(marker_cells),
                "black_count": black_count,
                "min_col": min(c for _r, c in comp),
                "region_size": len(comp),
            }
        )

    if len(regions) < 2:
        return None

    regions.sort(key=lambda info: (info["min_col"], info["cid"]))
    ordered_counts = tuple(int(info["black_count"]) for info in regions[:2])
    return {
        "shape": (h, w),
        "start_c": start_c,
        "start_dc": start_dc,
        "path": tuple(path),
        "regions": tuple(regions[:2]),
        "ordered_counts": ordered_counts,
        "total_regions": len(comps),
        "grid": grid,
        "labels": labels,
    }


def _build_valid_configs() -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for h in GRID_H_RANGE:
        for w in GRID_W_RANGE:
            for start_c in range(w):
                for start_dc in (-1, 1):
                    cfg = _analyze_base_config(h, w, start_c, start_dc)
                    if cfg is None:
                        continue
                    grouped[cfg["ordered_counts"]].append(cfg)
    assert grouped
    return dict(grouped)


VALID_CONFIGS = _build_valid_configs()


def _choose_marker_cell(region: dict) -> Coord:
    cells = list(region["marker_cells"])
    cells.sort(key=lambda item: (item[0], item[1]))
    if len(cells) == 1:
        return cells[0]
    pick_mode = random.randrange(3)
    if pick_mode == 0:
        return cells[0]
    if pick_mode == 1:
        return cells[-1]
    return random.choice(cells)


def _output_sig(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return (tuple(map(int, grid.shape)), tuple(int(x) for x in grid.ravel()))


def _build_instance(cfg: dict, colors_lr: tuple[int, int]) -> np.ndarray:
    h, w = cfg["shape"]
    grid = np.full((h, w), BLACK, dtype=int)
    for r, c in cfg["path"]:
        grid[r, c] = BLUE

    placements = []
    for region, color in zip(cfg["regions"], colors_lr):
        r, c = _choose_marker_cell(region)
        placements.append((r, c, color))

    assert len({(r, c) for r, c, _color in placements}) == len(placements)
    for r, c, color in placements:
        assert grid[r, c] == BLACK
        grid[r, c] = color
    return grid


def _validate_generated_instance(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    blue_points = np.argwhere(g == BLUE)
    assert blue_points.shape[0] >= g.shape[0]
    assert int(np.count_nonzero(g[-1] == BLUE)) == 1
    assert int(np.count_nonzero(g[0] == BLUE)) == 1

    marker_points = np.argwhere((g != BLACK) & (g != BLUE))
    marker_colors = [int(g[r, c]) for r, c in marker_points]
    assert len(marker_points) == len(set(marker_colors)) == 2

    for r, c in marker_points:
        rr = int(r)
        cc = int(c)
        assert 0 < rr < g.shape[0] - 1 and 0 < cc < g.shape[1] - 1

    passable = g != BLUE
    labels, comps = _label_passable(passable)
    marker_cids = [int(labels[int(r), int(c)]) for r, c in marker_points]
    assert len(marker_cids) == len(set(marker_cids))

    counts_lr = []
    for r, c in sorted([(int(r), int(c)) for r, c in marker_points], key=lambda item: (item[1], item[0])):
        cid = int(labels[r, c])
        black_count = sum(1 for rr, cc in comps[cid] if int(g[rr, cc]) == BLACK)
        assert black_count in (5, 9)
        counts_lr.append(black_count)

    assert len(comps) >= 3
    out = main(g)
    assert np.array_equal(out, _pack_summary(_collect_markers(g)))
    return out, tuple(counts_lr)


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for attempt in range(GEN_ATTEMPTS):
        profile = PROFILE_CYCLE[_GEN_CALL_COUNT % len(PROFILE_CYCLE)]
        pool = VALID_CONFIGS.get(profile)
        if not pool:
            pool = random.choice(list(VALID_CONFIGS.values()))
        cfg = random.choice(pool)

        colors_lr = tuple(random.sample(MARKER_COLORS, 2))
        if attempt % 5 == 4:
            colors_lr = tuple(reversed(colors_lr))

        grid = _build_instance(cfg, colors_lr)
        out, counts_lr = _validate_generated_instance(grid)
        assert counts_lr == cfg["ordered_counts"]

        sig = _output_sig(out)
        if sig in _SEEN_OUTPUT_SIGS and attempt < GEN_ATTEMPTS - 1:
            continue

        _SEEN_OUTPUT_SIGS.add(sig)
        _GEN_CALL_COUNT += 1
        return grid
    raise RuntimeError("generate_input: failed to construct a fresh valid instance")
