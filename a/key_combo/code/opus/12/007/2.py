from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_FROM_DELTA = {delta: name for name, delta in DIRS.items()}
PERPENDICULAR = {
    "up": (0, 1),
    "down": (0, 1),
    "left": (1, 0),
    "right": (1, 0),
}

GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 24
GEN_RETRIES = 500
MIN_BBOX_GAP = 5
MIN_RAY_SPACE = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _ray_space(shape: tuple[int, int], r: int, c: int, direction: str) -> int:
    h, w = shape
    if direction == "up":
        return r
    if direction == "down":
        return h - 1 - r
    if direction == "left":
        return c
    return w - 1 - c


def _direction_bbox(r: int, c: int, direction: str) -> tuple[int, int, int, int]:
    dr, dc = DIRS[direction]
    rr, cc = r + dr, c + dc
    return min(r, rr), min(c, cc), max(r, rr), max(c, cc)


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    row_gap = max(0, a[0] - b[2] - 1, b[0] - a[2] - 1)
    col_gap = max(0, a[1] - b[3] - 1, b[1] - a[3] - 1)
    return max(row_gap, col_gap)


def _parse_markers(grid: np.ndarray) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    markers = np.argwhere((g != BACKGROUND) & (g != GREY))
    assert 2 <= len(markers) <= 4, "expected 2 to 4 colored markers"

    records: list[dict[str, int | str]] = []
    seen_colors: set[int] = set()
    for r_raw, c_raw in markers:
        r, c = int(r_raw), int(c_raw)
        color = int(g[r, c])
        assert color in MARKER_COLORS, "marker colors must use the visible palette"
        assert color not in seen_colors, "marker colors must be distinct"
        seen_colors.add(color)

        grey_neighbors: list[tuple[int, int]] = []
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == GREY:
                grey_neighbors.append((nr, nc))
        assert len(grey_neighbors) == 1, "each marker needs exactly one grey direction cell"

        gr, gc = grey_neighbors[0]
        delta = (gr - r, gc - c)
        assert delta in DIR_FROM_DELTA, "grey direction cell must be 4-neighbor adjacent"
        records.append(
            {
                "row": r,
                "col": c,
                "color": color,
                "grey_row": gr,
                "grey_col": gc,
                "direction": DIR_FROM_DELTA[delta],
            }
        )

    grey_cells = np.argwhere(g == GREY)
    assert len(grey_cells) == len(records), "grey cells must correspond one-to-one with markers"
    for gr_raw, gc_raw in grey_cells:
        gr, gc = int(gr_raw), int(gc_raw)
        adjacent = 0
        for marker in records:
            if abs(gr - int(marker["row"])) + abs(gc - int(marker["col"])) == 1:
                adjacent += 1
        assert adjacent == 1, "each grey cell must belong to exactly one marker"

    assert len({str(marker["direction"]) for marker in records}) >= 2, "need at least two distinct directions"
    return sorted(records, key=lambda item: (int(item["row"]), int(item["col"])))


def _solve_records(
    records: list[dict[str, int | str]],
    shape: tuple[int, int],
    grey_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray | bool | int]]:
    h, w = shape
    marker_mask = np.zeros(shape, dtype=bool)
    for marker in records:
        marker_mask[int(marker["row"]), int(marker["col"])] = True

    inf = h + w + 10
    best_dist = np.full(shape, inf, dtype=int)
    best_src_r = np.full(shape, h + 1, dtype=int)
    best_src_c = np.full(shape, w + 1, dtype=int)
    best_color = np.zeros(shape, dtype=int)
    candidate_counts = np.zeros(shape, dtype=int)
    saw_clip = False

    for marker in records:
        r0 = int(marker["row"])
        c0 = int(marker["col"])
        color = int(marker["color"])
        direction = str(marker["direction"])
        dr, dc = DIRS[direction]
        pr, pc = PERPENDICULAR[direction]

        distance = 1
        while True:
            axis_r = r0 + distance * dr
            axis_c = c0 + distance * dc
            if not _in_bounds(shape, axis_r, axis_c):
                break

            for offset in range(-(distance - 1), distance):
                rr = axis_r + offset * pr
                cc = axis_c + offset * pc
                if not _in_bounds(shape, rr, cc):
                    saw_clip = True
                    continue
                if grey_mask[rr, cc] or marker_mask[rr, cc]:
                    continue

                candidate_counts[rr, cc] += 1
                dist = max(abs(rr - r0), abs(cc - c0))
                better = dist < best_dist[rr, cc]
                tied = dist == best_dist[rr, cc]
                earlier = r0 < best_src_r[rr, cc] or (
                    r0 == best_src_r[rr, cc] and c0 < best_src_c[rr, cc]
                )
                if better or (tied and earlier):
                    best_dist[rr, cc] = dist
                    best_src_r[rr, cc] = r0
                    best_src_c[rr, cc] = c0
                    best_color[rr, cc] = color
            distance += 1

    out = np.zeros(shape, dtype=int)
    paint_mask = best_dist < inf
    out[paint_mask] = best_color[paint_mask]
    for marker in records:
        out[int(marker["row"]), int(marker["col"])] = int(marker["color"])

    meta: dict[str, np.ndarray | bool | int] = {
        "candidate_counts": candidate_counts,
        "saw_clip": saw_clip,
        "painted_cells": int(np.count_nonzero(paint_mask)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored marker into a direction-coded triangular ray wedge and resolve overlaps by source distance.

    Input structure:
    - The grid is a black background with 2 to 4 isolated colored marker pixels.
    - Each marker uses one distinct color from blue, red, green, yellow.
    - Every marker has exactly one grey 4-neighbor whose relative position encodes the ray direction.

    Rule:
    - Keep each colored marker pixel in place and erase every grey direction pixel to black.
    - For each marker, cast a ray in the direction indicated by its grey neighbor.
    - At distance d along that ray, fill the in-bounds row or column segment of width 2d-1 centered on the ray axis and perpendicular to the ray direction.
    - If multiple wedges cover one cell, keep the color whose source marker has smaller Chebyshev distance to that cell; if still tied, prefer the source with smaller row, then smaller column.

    Key insight:
    - The grey neighbor is not decoration: it selects the wedge axis, and the wedge width grows by one cell per step away from the marker.
    """
    g = np.asarray(grid, dtype=int)
    records = _parse_markers(g)
    out, _meta = _solve_records(records, g.shape, g == GREY)
    return out


def _row_choices(h: int) -> list[int]:
    return sorted({value for value in (2, 3, h // 2, h - 4, h - 3) if 1 <= value < h - 1})


def _col_choices(w: int) -> list[int]:
    return sorted({value for value in (2, 3, w // 2, w - 4, w - 3) if 1 <= value < w - 1})


def _can_place(
    placed: list[dict[str, int | str]],
    shape: tuple[int, int],
    r: int,
    c: int,
    direction: str,
) -> bool:
    h, w = shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if _ray_space(shape, r, c, direction) < MIN_RAY_SPACE:
        return False

    dr, dc = DIRS[direction]
    gr, gc = r + dr, c + dc
    if not _in_bounds(shape, gr, gc):
        return False

    bbox = _direction_bbox(r, c, direction)
    for item in placed:
        other_cells = {
            (int(item["row"]), int(item["col"])),
            (int(item["grey_row"]), int(item["grey_col"])),
        }
        if (r, c) in other_cells or (gr, gc) in other_cells:
            return False
        other_bbox = _direction_bbox(int(item["row"]), int(item["col"]), str(item["direction"]))
        if _bbox_gap(bbox, other_bbox) < MIN_BBOX_GAP:
            return False
    return True


def _place_marker(
    placed: list[dict[str, int | str]],
    shape: tuple[int, int],
    r: int,
    c: int,
    direction: str,
    color: int,
) -> bool:
    if not _can_place(placed, shape, r, c, direction):
        return False
    dr, dc = DIRS[direction]
    placed.append(
        {
            "row": r,
            "col": c,
            "color": color,
            "grey_row": r + dr,
            "grey_col": c + dc,
            "direction": direction,
        }
    )
    return True


def _try_random_marker(
    placed: list[dict[str, int | str]],
    shape: tuple[int, int],
    color: int,
    preferred_direction: str | None = None,
) -> bool:
    h, w = shape
    directions = list(DIRS)
    if preferred_direction is not None:
        directions.remove(preferred_direction)
        directions.insert(0, preferred_direction)

    for _ in range(200):
        direction = random.choice(directions)
        boundary_bias = random.random() < 0.7

        if direction in ("left", "right"):
            row_pool = _row_choices(h) if boundary_bias else list(range(2, h - 2))
            col_min = 3 if direction == "left" else 2
            col_max = w - 3 if direction == "right" else w - 4
            if col_min > col_max:
                continue
            r = random.choice(row_pool)
            c = random.randint(col_min, col_max)
        else:
            col_pool = _col_choices(w) if boundary_bias else list(range(2, w - 2))
            row_min = 3 if direction == "up" else 2
            row_max = h - 3 if direction == "down" else h - 4
            if row_min > row_max:
                continue
            r = random.randint(row_min, row_max)
            c = random.choice(col_pool)

        if _place_marker(placed, shape, r, c, direction, color):
            return True
    return False


def _build_horizontal_overlap_case(shape: tuple[int, int], colors: list[int]) -> list[dict[str, int | str]] | None:
    h, w = shape
    if w < 13:
        return None

    placed: list[dict[str, int | str]] = []
    row = random.choice(_row_choices(h))
    left_col_max = w - 12
    if left_col_max < 2:
        return None
    c1 = random.randint(2, left_col_max)
    c2 = random.randint(c1 + 8, w - 3)

    if not _place_marker(placed, shape, row, c1, "right", colors[0]):
        return None
    if not _place_marker(placed, shape, row, c2, "left", colors[1]):
        return None

    for color in colors[2:]:
        if not _try_random_marker(placed, shape, color):
            return None
    return placed


def _build_vertical_overlap_case(shape: tuple[int, int], colors: list[int]) -> list[dict[str, int | str]] | None:
    h, w = shape
    if h < 13:
        return None

    placed: list[dict[str, int | str]] = []
    col = random.choice(_col_choices(w))
    top_row_max = h - 12
    if top_row_max < 2:
        return None
    r1 = random.randint(2, top_row_max)
    r2 = random.randint(r1 + 8, h - 3)

    if not _place_marker(placed, shape, r1, col, "down", colors[0]):
        return None
    if not _place_marker(placed, shape, r2, col, "up", colors[1]):
        return None

    for color in colors[2:]:
        if not _try_random_marker(placed, shape, color):
            return None
    return placed


def _build_spread_case(shape: tuple[int, int], colors: list[int]) -> list[dict[str, int | str]] | None:
    placed: list[dict[str, int | str]] = []
    directions = [random.choice(tuple(DIRS)) for _ in colors]
    while len(set(directions)) < 2:
        directions = [random.choice(tuple(DIRS)) for _ in colors]

    for color, direction in zip(colors, directions):
        if not _try_random_marker(placed, shape, color, preferred_direction=direction):
            return None
    return placed


def _records_to_grid(shape: tuple[int, int], records: list[dict[str, int | str]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for marker in records:
        grid[int(marker["row"]), int(marker["col"])] = int(marker["color"])
        grid[int(marker["grey_row"]), int(marker["grey_col"])] = GREY
    return grid


def _validate_case(grid: np.ndarray, mode: str) -> bool:
    try:
        records = _parse_markers(grid)
    except AssertionError:
        return False

    for marker in records:
        r = int(marker["row"])
        c = int(marker["col"])
        if r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1):
            return False

    for i, a in enumerate(records):
        bbox_a = _direction_bbox(int(a["row"]), int(a["col"]), str(a["direction"]))
        for b in records[i + 1 :]:
            bbox_b = _direction_bbox(int(b["row"]), int(b["col"]), str(b["direction"]))
            if _bbox_gap(bbox_a, bbox_b) < MIN_BBOX_GAP:
                return False

    out, meta = _solve_records(records, grid.shape, grid == GREY)
    if np.array_equal(out, grid):
        return False
    if np.any(out == GREY):
        return False
    if any(int(out[int(m["grey_row"]), int(m["grey_col"])]) != BACKGROUND for m in records):
        return False
    if any(int(out[int(m["row"]), int(m["col"])]) != int(m["color"]) for m in records):
        return False

    candidate_counts = np.asarray(meta["candidate_counts"])
    overlap_area = int(np.count_nonzero(candidate_counts > 1))
    has_overlap = overlap_area > 0
    has_clip = bool(meta["saw_clip"])
    painted_cells = int(meta["painted_cells"])

    if painted_cells < len(records):
        return False
    if painted_cells > int(grid.size * 0.72):
        return False
    if overlap_area > int(grid.size * 0.2):
        return False

    if mode in ("horizontal_overlap", "vertical_overlap"):
        return has_overlap and has_clip
    return (not has_overlap) and has_clip


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        shape = (
            random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX),
            random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX),
        )
        count = random.randint(2, 4)
        colors = random.sample(list(MARKER_COLORS), count)
        mode = random.choice(("horizontal_overlap", "vertical_overlap", "spread"))

        if mode == "horizontal_overlap":
            records = _build_horizontal_overlap_case(shape, colors)
        elif mode == "vertical_overlap":
            records = _build_vertical_overlap_case(shape, colors)
        else:
            records = _build_spread_case(shape, colors)

        if records is None:
            continue

        grid = _records_to_grid(shape, records)
        if _validate_case(grid, mode):
            return grid

    raise RuntimeError("failed to generate a valid wedge-ray input")
