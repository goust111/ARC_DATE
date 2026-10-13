from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds


GREEN = 3
RED = 2
BLACK = 0
EDGE_DISTANCES = (2, 3, 4, 5, 6)
GRID_SIDE_RANGE = (18, 24)
GEN_RETRIES = 400

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _find_anchor(mask: np.ndarray) -> BBox:
    h, w = mask.shape
    anchors: list[BBox] = []
    for r in range(h - 2):
        for c in range(w - 2):
            if bool(mask[r : r + 3, c : c + 3].all()):
                anchors.append((r, c, r + 2, c + 2))
    assert len(anchors) == 1, "expected exactly one solid 3x3 green anchor"
    return anchors[0]


def _anchor_cells(anchor: BBox) -> set[Coord]:
    r0, c0, r1, c1 = anchor
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _ray_groups(mask: np.ndarray, anchor: BBox) -> dict[tuple[Coord, Coord], list[tuple[int, int, int]]]:
    r0, c0, r1, c1 = anchor
    groups: dict[tuple[Coord, Coord], list[tuple[int, int, int]]] = defaultdict(list)
    for raw_r, raw_c in np.argwhere(mask):
        r, c = map(int, (raw_r, raw_c))
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        if r0 <= r <= r1 and c < c0:
            terminal, step = (r, c0 - 1), (0, 1)
            dist = c0 - 1 - c
        elif r0 <= r <= r1 and c > c1:
            terminal, step = (r, c1 + 1), (0, -1)
            dist = c - (c1 + 1)
        elif c0 <= c <= c1 and r < r0:
            terminal, step = (r0 - 1, c), (1, 0)
            dist = r0 - 1 - r
        elif c0 <= c <= c1 and r > r1:
            terminal, step = (r1 + 1, c), (-1, 0)
            dist = r - (r1 + 1)
        else:
            raise AssertionError("loose green pixels must lie on anchor-aligned orthogonal rays")
        assert dist >= 1
        groups[(terminal, step)].append((dist, r, c))
    return groups


def _compressed_wall_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mask = (g == GREEN)
    anchor = _find_anchor(mask)
    occupied = np.zeros_like(mask, dtype=bool)
    for r, c in _anchor_cells(anchor):
        occupied[r, c] = True

    for (terminal, step), cells in _ray_groups(mask, anchor).items():
        distances = [dist for dist, _r, _c in cells]
        assert len(distances) == len(set(distances)), "a single lane cannot reuse a distance"
        tr, tc = terminal
        dr, dc = step
        for i, _cell in enumerate(sorted(cells)):
            rr = tr - i * dr
            cc = tc - i * dc
            assert in_bounds(g.shape, rr, cc)
            assert not occupied[rr, cc], "compressed green pixels must not overlap"
            occupied[rr, cc] = True

    assert int(occupied.sum()) == int(mask.sum()), "compression must preserve green pixel count"
    return occupied


def _scanline_fill(grid: np.ndarray, wall_mask: np.ndarray, marker_col: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    wall_rows = np.flatnonzero(np.any(wall_mask, axis=1))
    assert wall_rows.size > 0, "expected at least one wall row"
    top_row = int(wall_rows.min())
    bottom_row = int(wall_rows.max())

    for r in range(g.shape[0]):
        wall_cols = np.flatnonzero(wall_mask[r])
        if wall_cols.size:
            left = wall_cols[wall_cols < marker_col]
            right = wall_cols[wall_cols >= marker_col]
            if left.size and right.size:
                start = int(left.max()) + 1
                stop = int(right.min())
            elif left.size:
                start = int(left.max()) + 1
                stop = g.shape[1]
            elif right.size:
                start = 0
                stop = int(right.min())
            else:
                continue
        elif top_row < r < bottom_row:
            start = 0
            stop = g.shape[1]
        else:
            continue

        if start >= stop:
            continue
        span = out[r, start:stop]
        span[span == bg] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress green ray pixels into a wall around the anchor, then fill scanline spans using the red marker column.

    Input structure:
    - A black-background grid contains one solid green 3x3 anchor block and additional isolated green pixels.
    - Every loose green pixel lies on a row or column ray extending orthogonally from the anchor, never on a diagonal.
    - One red singleton marker sits in a quadrant outside the anchor and shares neither a row nor a column with any green pixel.

    Rule:
    - Keep the 3x3 green anchor fixed and compress each loose green ray inward along its own row or column until each lane becomes a contiguous run touching the anchor.
    - Treat every green pixel after compression, including the anchor, as a wall.
    - Let the red marker column be the scan reference. On each row with walls, fill background cells between the nearest wall strictly left of the marker column and the nearest wall at or right of it; if only one side exists, fill from that wall toward the opposite grid edge.
    - On rows without walls that lie strictly between the topmost and bottommost wall rows, fill every background cell red. Preserve all original green pixels only in their compressed positions, and preserve the original red marker pixel.

    Key insight:
    - The same loose green pixels must first be interpreted as movable ray pixels and only afterward as fixed scanline boundaries; filling from the input walls directly gives a different result.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    marker_coords = np.argwhere(g == RED)
    assert marker_coords.shape == (1, 2), "expected exactly one red marker"
    marker_col = int(marker_coords[0, 1])

    wall_mask = _compressed_wall_mask(g)
    out = np.full_like(g, int(get_bg(g)))
    out[wall_mask] = GREEN
    out[g == RED] = RED
    out = _scanline_fill(out, wall_mask, marker_col)
    return out


def _direction_counts_from_positions(anchor: BBox, loose: set[Coord]) -> dict[str, int]:
    r0, c0, r1, c1 = anchor
    counts = {"left": 0, "right": 0, "up": 0, "down": 0}
    for r, c in loose:
        if r0 <= r <= r1 and c < c0:
            counts["left"] += 1
        elif r0 <= r <= r1 and c > c1:
            counts["right"] += 1
        elif c0 <= c <= c1 and r < r0:
            counts["up"] += 1
        elif c0 <= c <= c1 and r > r1:
            counts["down"] += 1
        else:
            raise AssertionError("loose green pixels must stay on orthogonal rays")
    return counts


def _sample_lane_counts(total: int, *, require_zero_lane: bool) -> tuple[int, int, int]:
    choices = []
    for a in range(total + 1):
        for b in range(total - a + 1):
            c = total - a - b
            counts = (a, b, c)
            if sum(v > 0 for v in counts) < 2:
                continue
            if require_zero_lane and 0 not in counts:
                continue
            choices.append(counts)
    assert choices
    return random.choice(choices)


def _place_direction(side: str, anchor: BBox, counts: tuple[int, int, int]) -> set[Coord]:
    r0, c0, r1, c1 = anchor
    cells: set[Coord] = set()
    for lane_index, amount in enumerate(counts):
        if amount == 0:
            continue
        chosen = random.sample(EDGE_DISTANCES, amount)
        if side == "left":
            row = r0 + lane_index
            for dist in chosen:
                cells.add((row, c0 - dist))
        elif side == "right":
            row = r0 + lane_index
            for dist in chosen:
                cells.add((row, c1 + dist))
        elif side == "up":
            col = c0 + lane_index
            for dist in chosen:
                cells.add((r0 - dist, col))
        elif side == "down":
            col = c0 + lane_index
            for dist in chosen:
                cells.add((r1 + dist, col))
        else:
            raise ValueError("invalid side")
    return cells


def _marker_column_candidates(
    width: int,
    anchor: BBox,
    green_cols: set[int],
    horizontal_side: str,
    wall_mask: np.ndarray,
) -> list[int]:
    _r0, c0, _r1, c1 = anchor
    wall_cols = np.flatnonzero(np.any(wall_mask, axis=0))
    min_wall = int(wall_cols.min())
    max_wall = int(wall_cols.max())
    if horizontal_side == "left":
        return [
            c
            for c in range(0, c0)
            if c not in green_cols and c < min_wall
        ]
    return [
        c
        for c in range(c1 + 1, width)
        if c not in green_cols and c > max_wall
    ]


def _marker_row_candidates(height: int, anchor: BBox, green_rows: set[int], vertical_side: str) -> list[int]:
    r0, _c0, r1, _c1 = anchor
    if vertical_side == "top":
        return [r for r in range(0, r0) if r not in green_rows]
    return [r for r in range(r1 + 1, height) if r not in green_rows]


def _build_input_from_spec(
    shape: tuple[int, int],
    anchor: BBox,
    loose: set[Coord],
    marker: Coord,
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for r, c in _anchor_cells(anchor):
        grid[r, c] = GREEN
    for r, c in loose:
        grid[r, c] = GREEN
    mr, mc = marker
    assert grid[mr, mc] == BLACK
    grid[mr, mc] = RED
    return grid


def _validate_scene(inp: np.ndarray) -> None:
    g = np.asarray(inp, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert int(np.count_nonzero(g == RED)) == 1

    green_mask = (g == GREEN)
    anchor = _find_anchor(green_mask)
    loose = {tuple(map(int, rc)) for rc in np.argwhere(green_mask)} - _anchor_cells(anchor)
    counts = _direction_counts_from_positions(anchor, loose)
    assert set(counts.values()) <= {2, 3, 4}
    assert all(2 <= v <= 4 for v in counts.values())

    marker = tuple(map(int, np.argwhere(g == RED)[0]))
    green_rows = {r for r, _c in np.argwhere(green_mask)}
    green_cols = {c for _r, c in np.argwhere(green_mask)}
    assert marker[0] not in green_rows and marker[1] not in green_cols

    r0, c0, r1, c1 = anchor
    assert marker[0] < r0 or marker[0] > r1
    assert marker[1] < c0 or marker[1] > c1

    wall_mask = _compressed_wall_mask(g)
    direct_wall_out = _scanline_fill(g, green_mask, marker[1])
    actual = main(g)
    assert np.array_equal(actual == GREEN, wall_mask)
    assert int(np.count_nonzero(actual == RED)) > 1, "fill step must add red pixels"
    assert not np.array_equal(actual, direct_wall_out), "compression must materially affect the fill result"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    r0 = random.randint(6, height - 9)
    c0 = random.randint(6, width - 9)
    anchor = (r0, c0, r0 + 2, c0 + 2)

    vertical_side = random.choice(("top", "bottom"))
    horizontal_side = random.choice(("left", "right"))

    totals = {side: random.randint(2, 4) for side in ("left", "right", "up", "down")}
    lane_counts = {
        side: _sample_lane_counts(
            totals[side],
            require_zero_lane=(side == horizontal_side),
        )
        for side in ("left", "right", "up", "down")
    }

    loose: set[Coord] = set()
    for side in ("left", "right", "up", "down"):
        loose |= _place_direction(side, anchor, lane_counts[side])

    green_rows = {r for r, _c in loose | _anchor_cells(anchor)}
    green_cols = {c for _r, c in loose | _anchor_cells(anchor)}
    probe = np.full((height, width), BLACK, dtype=int)
    for r, c in _anchor_cells(anchor):
        probe[r, c] = GREEN
    for r, c in loose:
        probe[r, c] = GREEN
    wall_mask = _compressed_wall_mask(probe)

    marker_rows = _marker_row_candidates(height, anchor, green_rows, vertical_side)
    marker_cols = _marker_column_candidates(width, anchor, green_cols, horizontal_side, wall_mask)
    if not marker_rows or not marker_cols:
        return None

    marker = (random.choice(marker_rows), random.choice(marker_cols))
    inp = _build_input_from_spec((height, width), anchor, loose, marker)
    try:
        _validate_scene(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 4")
