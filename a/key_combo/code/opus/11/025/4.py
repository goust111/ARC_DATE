from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

_DIR_BY_SIDE = {
    "L": (0, 1),
    "R": (0, -1),
    "T": (1, 0),
    "B": (-1, 0),
}
_CORNERS = ("TL", "TR", "BL", "BR")
_GEN_COUNTER = 0
_GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[tuple[np.ndarray, int]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[tuple[np.ndarray, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            border_touch_count = 0

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    border_touch_count += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            out.append((np.asarray(coords, dtype=int), border_touch_count))

    return out


def _marker_sets(grid: np.ndarray) -> tuple[list[int], list[int], list[int], list[int]]:
    h, w = grid.shape
    left_rows = [r for r in range(h) if int(grid[r, 0]) == RED]
    right_rows = [r for r in range(h) if int(grid[r, w - 1]) == RED]
    top_cols = [c for c in range(w) if int(grid[0, c]) == RED]
    bottom_cols = [c for c in range(w) if int(grid[h - 1, c]) == RED]
    return left_rows, right_rows, top_cols, bottom_cols


def _line_mask_and_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    h, w = grid.shape
    left_rows, right_rows, top_cols, bottom_cols = _marker_sets(grid)

    full_rows = sorted(set(left_rows) & set(right_rows))
    full_cols = sorted(set(top_cols) & set(bottom_cols))
    left_only = sorted(set(left_rows) - set(right_rows))
    right_only = sorted(set(right_rows) - set(left_rows))
    top_only = sorted(set(top_cols) - set(bottom_cols))
    bottom_only = sorted(set(bottom_cols) - set(top_cols))

    line = np.zeros((h, w), dtype=bool)
    for r in full_rows:
        line[r, :] = True
    for c in full_cols:
        line[:, c] = True

    rays: list[list[int | bool | str]] = []
    for r in left_only:
        dr, dc = _DIR_BY_SIDE["L"]
        rays.append([r, 0, dr, dc, True, "L"])
    for r in right_only:
        dr, dc = _DIR_BY_SIDE["R"]
        rays.append([r, w - 1, dr, dc, True, "R"])
    for c in top_only:
        dr, dc = _DIR_BY_SIDE["T"]
        rays.append([0, c, dr, dc, True, "T"])
    for c in bottom_only:
        dr, dc = _DIR_BY_SIDE["B"]
        rays.append([h - 1, c, dr, dc, True, "B"])

    for r, c, _dr, _dc, _active, _tag in rays:
        line[int(r), int(c)] = True

    stats = {
        "full_rows": len(full_rows),
        "full_cols": len(full_cols),
        "partial_rays": len(rays),
        "collision_steps": 0,
        "painted_partial_cells": 0,
    }

    while True:
        claims: dict[tuple[int, int], list[int]] = {}
        any_active = False

        for i, ray in enumerate(rays):
            r, c, dr, dc, active, _tag = ray
            if not active:
                continue
            nr = int(r) + int(dr)
            nc = int(c) + int(dc)
            if not (0 <= nr < h and 0 <= nc < w) or line[nr, nc]:
                ray[4] = False
                continue
            claims.setdefault((nr, nc), []).append(i)
            any_active = True

        if not any_active:
            break

        progressed = False
        for (nr, nc), idxs in claims.items():
            if len(idxs) == 1:
                idx = idxs[0]
                line[nr, nc] = True
                rays[idx][0] = nr
                rays[idx][1] = nc
                stats["painted_partial_cells"] += 1
                progressed = True
            else:
                stats["collision_steps"] += 1
                for idx in idxs:
                    rays[idx][4] = False

        if not progressed:
            break

    return line, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Extend border-marker rays to build a red lattice, then fill each enclosed cell by whether it is rectangular.

    Input structure:
    - The grid is black except for red singleton markers placed on the outer border.
    - Matching left/right markers indicate complete horizontal guides, and matching top/bottom markers indicate complete vertical guides.
    - Unmatched markers indicate one-sided rays that travel inward from the border.

    Rule:
    - Draw every matched guide as a full red row or column.
    - Advance all unmatched rays simultaneously one cell per step; a ray stops before an already red cell, and competing rays that would enter the same next cell both stop there without painting it.
    - After the red framework is finished, keep the unique largest remaining black component as the outside background; recolor every other black component blue when it exactly fills its bounding box, otherwise yellow.

    Key insight:
    - The marker pairs define a coarse lattice, while the missing pair endpoints create truncated rays whose collision pattern merges some nominal cells into non-rectangular chambers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    line, _stats = _line_mask_and_stats(g)
    regions = _components(~line)
    assert regions, "expected at least one black region"

    region_info = []
    for idx, (coords, border_touch_count) in enumerate(regions):
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, c0 = int(rs.min()), int(cs.min())
        r1, c1 = int(rs.max()), int(cs.max())
        area = int(coords.shape[0])
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        region_info.append((idx, area, bbox_area, border_touch_count, r0, c0, coords))

    outside_idx = max(region_info, key=lambda item: (item[1], item[3], -item[4], -item[5]))[0]

    out = np.zeros_like(g)
    out[line] = RED
    for idx, area, bbox_area, _border_touch_count, _r0, _c0, coords in region_info:
        if idx == outside_idx:
            continue
        out[coords[:, 0], coords[:, 1]] = BLUE if area == bbox_area else YELLOW
    return out


def _sample_pair(rng: np.random.Generator, lo: int, hi: int) -> tuple[int, int] | None:
    values = list(range(int(lo), int(hi) + 1))
    if len(values) < 2:
        return None
    a, b = sorted(map(int, rng.choice(values, size=2, replace=False).tolist()))
    return a, b


def _make_candidate(rng: np.random.Generator, corner: str) -> np.ndarray | None:
    h = int(rng.integers(12, 17))
    w = int(rng.integers(12, 17))
    max_diag = min(h, w) - 5
    if max_diag < 2:
        return None
    diag = int(rng.integers(2, max_diag + 1))

    if corner == "BR":
        unmatched_row = h - 1 - diag
        unmatched_col = w - 1 - diag
        row_pair = _sample_pair(rng, 2, unmatched_row - 2)
        col_pair = _sample_pair(rng, 2, unmatched_col - 2)
        if row_pair is None or col_pair is None:
            return None
        left_rows = list(row_pair)
        right_rows = list(row_pair) + [unmatched_row]
        top_cols = list(col_pair)
        bottom_cols = list(col_pair) + [unmatched_col]
    elif corner == "BL":
        unmatched_row = h - 1 - diag
        unmatched_col = diag
        row_pair = _sample_pair(rng, 2, unmatched_row - 2)
        col_pair = _sample_pair(rng, unmatched_col + 2, w - 3)
        if row_pair is None or col_pair is None:
            return None
        left_rows = list(row_pair) + [unmatched_row]
        right_rows = list(row_pair)
        top_cols = list(col_pair)
        bottom_cols = list(col_pair) + [unmatched_col]
    elif corner == "TR":
        unmatched_row = diag
        unmatched_col = w - 1 - diag
        row_pair = _sample_pair(rng, unmatched_row + 2, h - 3)
        col_pair = _sample_pair(rng, 2, unmatched_col - 2)
        if row_pair is None or col_pair is None:
            return None
        left_rows = list(row_pair)
        right_rows = list(row_pair) + [unmatched_row]
        top_cols = list(col_pair) + [unmatched_col]
        bottom_cols = list(col_pair)
    else:
        unmatched_row = diag
        unmatched_col = diag
        row_pair = _sample_pair(rng, unmatched_row + 2, h - 3)
        col_pair = _sample_pair(rng, unmatched_col + 2, w - 3)
        if row_pair is None or col_pair is None:
            return None
        left_rows = list(row_pair) + [unmatched_row]
        right_rows = list(row_pair)
        top_cols = list(col_pair) + [unmatched_col]
        bottom_cols = list(col_pair)

    grid = np.zeros((h, w), dtype=int)
    for r in left_rows:
        grid[int(r), 0] = RED
    for r in right_rows:
        grid[int(r), w - 1] = RED
    for c in top_cols:
        grid[0, int(c)] = RED
    for c in bottom_cols:
        grid[h - 1, int(c)] = RED
    return grid


def _generated_instance_ok(grid: np.ndarray) -> bool:
    left_rows, right_rows, top_cols, bottom_cols = _marker_sets(grid)
    if not all(2 <= len(side) <= 3 for side in (left_rows, right_rows, top_cols, bottom_cols)):
        return False

    total_line_count = len(set(left_rows) | set(right_rows)) + len(set(top_cols) | set(bottom_cols))
    unmatched_count = len(set(left_rows) ^ set(right_rows)) + len(set(top_cols) ^ set(bottom_cols))
    if total_line_count != 6 or unmatched_count != 2:
        return False

    line, stats = _line_mask_and_stats(grid)
    if stats["collision_steps"] < 1 or stats["painted_partial_cells"] < 2:
        return False

    regions = _components(~line)
    if len(regions) < 4:
        return False

    region_info = []
    for coords, border_touch_count in regions:
        rs = coords[:, 0]
        cs = coords[:, 1]
        area = int(coords.shape[0])
        bbox_area = (int(rs.max()) - int(rs.min()) + 1) * (int(cs.max()) - int(cs.min()) + 1)
        region_info.append((area, bbox_area, border_touch_count, int(rs.min()), int(cs.min()), coords))
    region_info.sort(key=lambda item: (-item[0], -item[2], item[3], item[4]))

    if region_info[0][0] == region_info[1][0]:
        return False

    out = main(grid)
    if set(map(int, np.unique(out))) != {BLACK, BLUE, RED, YELLOW}:
        return False

    yellow_regions = 0
    blue_regions = 0
    for area, bbox_area, _border_touch_count, _r0, _c0, coords in region_info[1:]:
        color = int(out[coords[0, 0], coords[0, 1]])
        if area == bbox_area:
            if color != BLUE:
                return False
            blue_regions += 1
        else:
            if color != YELLOW:
                return False
            yellow_regions += 1
    if blue_regions < 2 or yellow_regions < 1:
        return False

    right_set = set(right_rows)
    bottom_set = set(bottom_cols)
    unmatched_markers: list[tuple[int, int]] = []
    for r in sorted(set(left_rows) ^ set(right_rows)):
        unmatched_markers.append((r, 0) if int(grid[r, 0]) == RED and r not in right_set else (r, grid.shape[1] - 1))
    for c in sorted(set(top_cols) ^ set(bottom_cols)):
        unmatched_markers.append((0, c) if int(grid[0, c]) == RED and c not in bottom_set else (grid.shape[0] - 1, c))

    for r, c in unmatched_markers:
        altered = grid.copy()
        altered[int(r), int(c)] = BLACK
        if np.array_equal(main(altered), out):
            return False

    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    branch_index = _GEN_COUNTER
    _GEN_COUNTER += 1
    preferred_corner = _CORNERS[branch_index % len(_CORNERS)]

    for attempt in range(_GEN_ATTEMPTS):
        rng = np.random.default_rng()
        corner = preferred_corner if attempt < _GEN_ATTEMPTS // 2 else _CORNERS[int(rng.integers(len(_CORNERS)))]
        grid = _make_candidate(rng, corner)
        if grid is not None and _generated_instance_ok(grid):
            return grid

    raise RuntimeError("failed to generate a valid ray-lattice input")
