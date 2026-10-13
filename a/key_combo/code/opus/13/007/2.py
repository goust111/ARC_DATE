import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers = [
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere((grid != BLACK) & (grid != GREY))
    ]
    markers.sort(key=lambda item: (item[0], item[1]))
    return markers


def _ray_owner_map(shape: tuple[int, int], markers: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    owner = np.full((h, w), -1, dtype=int)
    best_dist = np.full((h, w), np.iinfo(np.int32).max, dtype=int)

    for idx, (mr, mc, _color) in enumerate(markers):
        seen: set[tuple[int, int]] = set()
        for dr, dc in DIAGONALS:
            rr, cc = mr, mc
            while _in_bounds(shape, rr, cc):
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    dist = (rr - mr) * (rr - mr) + (cc - mc) * (cc - mc)
                    prev = int(owner[rr, cc])
                    if dist < int(best_dist[rr, cc]) or (
                        dist == int(best_dist[rr, cc])
                        and (prev == -1 or (mr, mc) < markers[prev][:2])
                    ):
                        best_dist[rr, cc] = dist
                        owner[rr, cc] = idx
                rr += dr
                cc += dc
    return owner


def _passable_components(passable: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[list[tuple[int, int]], bool]] = []

    for r in range(h):
        for c in range(w):
            if not passable[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in ORTHOGONAL:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(passable.shape, nr, nc) and passable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append((cells, touches_border))
    return components


def _nearest_marker_index(
    cells: Iterable[tuple[int, int]], markers: list[tuple[int, int, int]]
) -> int:
    best_score: tuple[int, int, int, int] | None = None
    best_idx = -1
    for idx, (mr, mc, _color) in enumerate(markers):
        for rr, cc in cells:
            dist = (rr - mr) * (rr - mr) + (cc - mc) * (cc - mc)
            score = (dist, mr, mc, idx)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
    return best_idx


def _analyze_regions(
    shape: tuple[int, int], markers: list[tuple[int, int, int]]
) -> tuple[np.ndarray, list[dict], list[dict]]:
    owner = _ray_owner_map(shape, markers)
    passable = owner == -1
    closed: list[dict] = []
    open_regions: list[dict] = []
    for cells, touches_border in _passable_components(passable):
        region = {
            "cells": cells,
            "area": len(cells),
            "owner": _nearest_marker_index(cells, markers),
        }
        if touches_border:
            open_regions.append(region)
        else:
            closed.append(region)
    return owner, closed, open_regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend colored marker diagonals to partition the canvas, then fill only closed labeled chambers with the nearest marker color.

    Input structure:
    - The grid is mostly black background with 3-4 distinct colored marker pixels and several grey label pixels.
    - Marker colors come from the non-grey, non-black palette and each marker is a single pixel.
    - Grey labels sit in the background spaces that may later become enclosed by the diagonal rays.

    Rule:
    - Through every marker, activate the full NW-SE and NE-SW diagonals to the grid edges.
    - If multiple markers claim the same ray cell, keep the color of the closer marker by Euclidean distance; exact ties go to the earlier marker in row-major order.
    - Treat the activated diagonal cells as walls, then find 4-connected background regions on the remaining black/grey cells.
    - Open regions stay unchanged. A closed region with no grey labels stays black.
    - A closed region with one or more grey labels fills all of its black cells with the color of the marker that is globally nearest to any cell in that region, while the grey labels themselves stay grey.

    Key insight:
    - The grey labels only decide whether a chamber activates; the chamber's fill color comes independently from the nearest external marker that generated the diagonal partition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers = _find_markers(g)
    assert 1 <= len(markers), "expected at least one marker"

    ray_owner, closed_regions, _open_regions = _analyze_regions(g.shape, markers)

    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            idx = int(ray_owner[r, c])
            if idx != -1 and int(g[r, c]) == BLACK:
                out[r, c] = markers[idx][2]

    for region in closed_regions:
        label_count = sum(int(g[r, c]) == GREY for r, c in region["cells"])
        if label_count == 0:
            continue
        fill_color = markers[int(region["owner"])][2]
        for r, c in region["cells"]:
            if int(g[r, c]) == BLACK:
                out[r, c] = fill_color
    return out


GRID_MIN = 12
GRID_MAX = 15
MARKER_COUNT_OPTIONS = (3, 4)
MIN_ROW_GAP = 4
MIN_COL_GAP = 4
MAX_ATTEMPTS = 400
MIN_ELIGIBLE_REGION_AREA = 4
LABEL_TOTAL_MIN = 4
LABEL_TOTAL_MAX = 8


def _pick_rows_and_cols(h: int, w: int) -> tuple[list[int], list[int]] | None:
    row_choices = [r for r in range(2, h - 2)]
    col_choices = [c for c in range(2, w - 2)]
    row_pairs = [pair for pair in _pair_choices(row_choices) if pair[1] - pair[0] >= MIN_ROW_GAP]
    col_pairs = [pair for pair in _pair_choices(col_choices) if pair[1] - pair[0] >= MIN_COL_GAP]
    if not row_pairs or not col_pairs:
        return None
    rows = list(random.choice(row_pairs))
    cols = list(random.choice(col_pairs))
    return rows, cols


def _pair_choices(values: list[int]) -> list[tuple[int, int]]:
    return [(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]


def _sample_marker_layout() -> tuple[tuple[int, int], list[tuple[int, int, int]]] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    rows_cols = _pick_rows_and_cols(h, w)
    if rows_cols is None:
        return None
    rows, cols = rows_cols
    corners = [
        (rows[0], cols[0]),
        (rows[0], cols[1]),
        (rows[1], cols[0]),
        (rows[1], cols[1]),
    ]
    marker_count = random.choice(MARKER_COUNT_OPTIONS)
    positions = corners if marker_count == 4 else random.sample(corners, 3)
    colors = random.sample(MARKER_COLORS, marker_count)
    markers = [(r, c, color) for (r, c), color in zip(positions, colors)]
    markers.sort(key=lambda item: (item[0], item[1]))
    return (h, w), markers


def _choose_distinct_owner_regions(regions: list[dict], count: int) -> list[dict]:
    shuffled = regions[:]
    random.shuffle(shuffled)
    chosen: list[dict] = []
    seen_owners: set[int] = set()
    for region in shuffled:
        owner = int(region["owner"])
        if owner not in seen_owners:
            chosen.append(region)
            seen_owners.add(owner)
            if len(chosen) == count:
                return chosen
    while len(chosen) < count and shuffled:
        region = shuffled.pop()
        if region not in chosen:
            chosen.append(region)
    return chosen


def _label_cap(region: dict, *, closed_region: bool) -> int:
    area = int(region["area"])
    if closed_region:
        return max(0, min(4, area - 1))
    return max(0, min(2, area))


def _pick_label_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    pool = cells[:]
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) >= 2 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    if len(chosen) == count:
        return chosen
    return None


def _place_labels(
    shape: tuple[int, int], markers: list[tuple[int, int, int]], closed_regions: list[dict], open_regions: list[dict]
) -> np.ndarray | None:
    eligible_closed = [region for region in closed_regions if int(region["area"]) >= MIN_ELIGIBLE_REGION_AREA]
    if len(eligible_closed) < 2:
        return None

    unlabeled_region = random.choice(eligible_closed)
    remaining_closed = [region for region in eligible_closed if region is not unlabeled_region]
    if not remaining_closed:
        return None

    desired_closed_count = 2 if len(remaining_closed) >= 2 and random.random() < 0.75 else 1
    desired_closed_count = min(desired_closed_count, len(remaining_closed))
    labeled_closed = _choose_distinct_owner_regions(remaining_closed, desired_closed_count)

    large_open = [region for region in open_regions if int(region["area"]) >= 3]
    include_open = bool(large_open) and random.random() < 0.5
    labeled_open = random.choice(large_open) if include_open else None

    region_specs: list[tuple[dict, bool]] = [(region, True) for region in labeled_closed]
    if labeled_open is not None:
        region_specs.append((labeled_open, False))

    caps = [_label_cap(region, closed_region=is_closed) for region, is_closed in region_specs]
    if any(cap <= 0 for cap in caps):
        return None

    assignments = [1 for _ in region_specs]
    max_total = min(LABEL_TOTAL_MAX, sum(caps))
    min_total = max(LABEL_TOTAL_MIN, sum(assignments))
    if min_total > max_total:
        return None

    target_total = random.randint(min_total, max_total)
    while sum(assignments) < target_total:
        choices = [i for i, cap in enumerate(caps) if assignments[i] < cap]
        if not choices:
            return None
        weights = [3 if region_specs[i][1] else 1 for i in choices]
        pick = random.choices(choices, weights=weights, k=1)[0]
        assignments[pick] += 1

    grid = np.zeros(shape, dtype=int)
    for r, c, color in markers:
        grid[r, c] = color

    for (region, _is_closed), label_count in zip(region_specs, assignments):
        label_cells = _pick_label_cells(region["cells"], label_count)
        if label_cells is None:
            return None
        for r, c in label_cells:
            grid[r, c] = GREY
    return grid


def _candidate_is_valid(grid: np.ndarray) -> bool:
    markers = _find_markers(grid)
    if len(markers) not in MARKER_COUNT_OPTIONS:
        return False
    colors = {color for _r, _c, color in markers}
    if len(colors) != len(markers):
        return False

    for i, (r1, c1, _color1) in enumerate(markers):
        for r2, c2, _color2 in markers[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) < 3:
                return False

    ray_owner, closed_regions, _open_regions = _analyze_regions(grid.shape, markers)
    eligible_closed = [region for region in closed_regions if int(region["area"]) >= MIN_ELIGIBLE_REGION_AREA]
    if len(eligible_closed) < 2:
        return False

    labeled_closed = []
    unlabeled_closed = []
    for region in closed_regions:
        label_count = sum(int(grid[r, c]) == GREY for r, c in region["cells"])
        if label_count >= 1:
            labeled_closed.append(region)
        else:
            unlabeled_closed.append(region)
    if not labeled_closed or not unlabeled_closed:
        return False

    label_total = int(np.sum(grid == GREY))
    if not (LABEL_TOTAL_MIN <= label_total <= LABEL_TOTAL_MAX):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) == GREY and int(ray_owner[r, c]) != -1:
                return False

    if not any(
        any(int(grid[r, c]) == BLACK and int(out[r, c]) != BLACK for r, c in region["cells"])
        for region in labeled_closed
    ):
        return False
    if not any(
        any(int(grid[r, c]) == BLACK and int(out[r, c]) == BLACK for r, c in region["cells"])
        for region in unlabeled_closed
    ):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_marker_layout()
    if sampled is None:
        return None
    shape, markers = sampled
    _ray_owner, closed_regions, open_regions = _analyze_regions(shape, markers)
    grid = _place_labels(shape, markers, closed_regions, open_regions)
    if grid is None:
        return None
    if not _candidate_is_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
