from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color


BLACK = int(Color.BLACK)
OBSTACLE = int(Color.GREY)
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
CORNER_NAMES = ("tl", "tr", "bl", "br")

Square = tuple[int, int, int]
Coord = tuple[int, int]


def _corner_positions(shape: tuple[int, int]) -> dict[str, Coord]:
    h, w = map(int, shape)
    return {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }


def _corner_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    seen_colors: set[int] = set()
    for name in CORNER_NAMES:
        r, c = _corner_positions(grid.shape)[name]
        color = int(grid[r, c])
        if color in (BLACK, OBSTACLE):
            continue
        assert color not in seen_colors, "corner marker colors must be distinct"
        seen_colors.add(color)
        markers.append((r, c, color))
    assert markers, "expected at least one colored corner marker"
    return markers


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), ((1, 0), (1, 0)))
    return padded.cumsum(axis=0).cumsum(axis=1)


def _square_area(prefix: np.ndarray, r0: int, c0: int, size: int) -> int:
    r1 = r0 + size
    c1 = c0 + size
    return int(prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0])


def _enumerate_black_squares(black_mask: np.ndarray) -> list[Square]:
    h, w = map(int, black_mask.shape)
    prefix = _prefix_sum(black_mask)
    candidates: list[Square] = []
    for size in range(min(h, w), 1, -1):
        target = size * size
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                if _square_area(prefix, r0, c0, size) == target:
                    candidates.append((r0, c0, size))
    return candidates


def _select_squares(candidates: list[Square], shape: tuple[int, int]) -> list[Square]:
    occupied = np.zeros(shape, dtype=bool)
    selected: list[Square] = []
    for r0, c0, size in candidates:
        block = occupied[r0 : r0 + size, c0 : c0 + size]
        if block.any():
            continue
        occupied[r0 : r0 + size, c0 : c0 + size] = True
        selected.append((r0, c0, size))
    return selected


def _square_mask(shape: tuple[int, int], squares: list[Square]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, size in squares:
        mask[r0 : r0 + size, c0 : c0 + size] = True
    return mask


def _square_cells(square: Square) -> set[Coord]:
    r0, c0, size = square
    return {(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)}


def _nearest_marker_color(square: Square, markers: list[tuple[int, int, int]]) -> int:
    r0, c0, size = square
    center_r2 = 2 * r0 + size - 1
    center_c2 = 2 * c0 + size - 1
    distances = [
        (
            abs(center_r2 - 2 * marker_r) + abs(center_c2 - 2 * marker_c),
            int(color),
        )
        for marker_r, marker_c, color in markers
    ]
    distances.sort()
    assert len(distances) == 1 or distances[0][0] != distances[1][0], "nearest corner marker must be unique"
    return int(distances[0][1])


def _paint_x(grid: np.ndarray, square: Square, color: int) -> None:
    r0, c0, size = square
    for delta in range(size):
        grid[r0 + delta, c0 + delta] = int(color)
        grid[r0 + delta, c0 + size - 1 - delta] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Find all maximal non-overlapping black squares by largest-first priority and stamp each with the nearest corner marker's X color.

    Input structure:
    - The canvas contains black free space, grey obstacle pixels, and colored single-pixel markers on some grid corners.
    - The black region may include square rooms, one-cell corridors, and offset protrusions inside the grey obstacle field.
    - Candidate square rooms are all-black axis-aligned squares of side at least 2.

    Rule:
    - Enumerate every all-black square candidate of side at least 2 and process them from larger to smaller.
    - Keep a candidate iff it does not overlap any previously kept square; then, for each kept square, find the unique nearest marked corner by Manhattan distance from the square center.
    - Draw an X across that square in the chosen marker color, leaving grey obstacles, corner markers, and all unselected black cells unchanged.

    Key insight:
    - Object discovery happens entirely inside the black geometry, but the color comes from a separate addressing layer at the corners, so both the largest-first square selection and the corner-distance lookup are necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    markers = _corner_markers(g)
    black_mask = g == BLACK
    candidates = _enumerate_black_squares(black_mask)
    selected = _select_squares(candidates, g.shape)

    out = g.copy()
    for square in selected:
        _paint_x(out, square, _nearest_marker_color(square, markers))
    return out


def _room_shape(
    size: int,
    *,
    decoy_orientation: str | None = None,
    decoy_shift: int | None = None,
) -> set[Coord]:
    cells = {(r, c) for r in range(size) for c in range(size)}
    if decoy_orientation is None:
        return cells

    assert decoy_shift is not None and 2 <= decoy_shift <= size - 2
    extras = {(size, c) for c in range(decoy_shift, size + 1)}
    extras |= {(r, size) for r in range(decoy_shift, size + 1)}

    for r, c in extras:
        rr, cc = r, c
        if "n" in decoy_orientation:
            rr = size - 1 - rr
        if "w" in decoy_orientation:
            cc = size - 1 - cc
        cells.add((rr, cc))
    return cells


def _bounds(cells: set[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _zones(shape: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    h, w = map(int, shape)
    mid_r = h // 2
    mid_c = w // 2
    return {
        "tl": (2, mid_r - 2, 2, mid_c - 2),
        "tr": (2, mid_r - 2, mid_c + 1, w - 3),
        "bl": (mid_r + 1, h - 3, 2, mid_c - 2),
        "br": (mid_r + 1, h - 3, mid_c + 1, w - 3),
    }


def _place_room(
    zone: tuple[int, int, int, int],
    size: int,
    shape_cells: set[Coord],
) -> tuple[int, int] | None:
    zr0, zr1, zc0, zc1 = zone
    min_r, min_c, max_r, max_c = _bounds(shape_cells)
    r_lo = zr0 - min_r
    r_hi = zr1 - max_r
    c_lo = zc0 - min_c
    c_hi = zc1 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return None

    options = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
    random.shuffle(options)
    for r0, c0 in options:
        return r0, c0
    return None


def _room_center(square: Square) -> tuple[float, float]:
    r0, c0, size = square
    return r0 + (size - 1) / 2.0, c0 + (size - 1) / 2.0


def _boundary_contact(square: Square, target_square: Square) -> Coord:
    r0, c0, size = square
    src_r, src_c = _room_center(square)
    dst_r, dst_c = _room_center(target_square)
    if abs(dst_c - src_c) >= abs(dst_r - src_r):
        if dst_c > src_c:
            return r0 + size // 2, c0 + size - 1
        return r0 + size // 2, c0
    if dst_r > src_r:
        return r0 + size - 1, c0 + size // 2
    return r0, c0 + size // 2


def _segment(a: Coord, b: Coord) -> list[Coord]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    assert a[1] == b[1]
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _corridor_cells(square_a: Square, square_b: Square, occupied: set[Coord]) -> set[Coord] | None:
    a = _boundary_contact(square_a, square_b)
    b = _boundary_contact(square_b, square_a)
    pivots = [(a[0], b[1]), (b[0], a[1])]
    random.shuffle(pivots)
    allowed = _square_cells(square_a) | _square_cells(square_b)
    for pivot in pivots:
        path = set(_segment(a, pivot)) | set(_segment(pivot, b))
        if any(cell in occupied and cell not in allowed for cell in path):
            continue
        return path
    return None


def _component_labels(mask: np.ndarray) -> np.ndarray:
    h, w = map(int, mask.shape)
    labels = np.zeros((h, w), dtype=int)
    next_label = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_label += 1
            stack = [(r, c)]
            labels[r, c] = next_label
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
    return labels


def _expected_output(grid: np.ndarray, squares: list[Square]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    markers = _corner_markers(out)
    for square in squares:
        _paint_x(out, square, _nearest_marker_color(square, markers))
    return out


def _has_offset_overlap_candidate(candidates: list[Square], selected: list[Square]) -> bool:
    selected_sets = {square: _square_cells(square) for square in selected}
    selected_keys = set(selected)
    for candidate in candidates:
        if candidate in selected_keys:
            continue
        cand_cells = _square_cells(candidate)
        for square, sq_cells in selected_sets.items():
            if cand_cells & sq_cells and not cand_cells.issubset(sq_cells) and not sq_cells.issubset(cand_cells):
                return True
    return False


def _valid_layout(grid: np.ndarray, intended: list[Square]) -> bool:
    black_mask = grid == BLACK
    candidates = _enumerate_black_squares(black_mask)
    selected = _select_squares(candidates, grid.shape)
    if set(selected) != set(intended):
        return False
    if not _has_offset_overlap_candidate(candidates, intended):
        return False

    selected_mask = _square_mask(grid.shape, intended)
    if not np.any(black_mask & ~selected_mask):
        return False

    labels = _component_labels(black_mask)
    counts: dict[int, int] = {}
    for r0, c0, _size in intended:
        label = int(labels[r0, c0])
        counts[label] = counts.get(label, 0) + 1
    if max(counts.values()) < 2:
        return False

    markers = _corner_markers(grid)
    try:
        marker_colors = {_nearest_marker_color(square, markers) for square in intended}
    except AssertionError:
        return False
    if len(marker_colors) < 2:
        return False

    try:
        expected = _expected_output(grid, intended)
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    return not np.array_equal(grid, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 29)
    w = random.randint(20, 29)
    shape = (h, w)
    grid = np.full(shape, OBSTACLE, dtype=int)

    zones = _zones(shape)
    room_count = random.randint(2, 4)
    chosen_zones = random.sample(CORNER_NAMES, room_count)
    random.shuffle(chosen_zones)

    active_corner_count = random.choice((2, 3, 3, 4))
    active_corners = random.sample(CORNER_NAMES, active_corner_count)
    marker_colors = random.sample(MARKER_PALETTE, active_corner_count)
    for name in CORNER_NAMES:
        r, c = _corner_positions(shape)[name]
        grid[r, c] = OBSTACLE
    for name, color in zip(active_corners, marker_colors):
        r, c = _corner_positions(shape)[name]
        grid[r, c] = int(color)

    decoy_zone = chosen_zones[0]
    rooms: list[dict[str, object]] = []
    occupied: set[Coord] = set()
    for zone_name in chosen_zones:
        if zone_name == decoy_zone:
            size = random.randint(4, 6)
            shift = random.randint(2, size - 2)
            orientation = random.choice(("se", "sw", "ne", "nw"))
        else:
            size = random.randint(2, 6)
            shift = None
            orientation = None

        local_cells = _room_shape(size, decoy_orientation=orientation, decoy_shift=shift)
        placed = _place_room(zones[zone_name], size, local_cells)
        if placed is None:
            return None
        r0, c0 = placed

        shape_cells = {(r0 + dr, c0 + dc) for dr, dc in local_cells}
        if shape_cells & occupied:
            return None

        square = (r0, c0, size)
        rooms.append(
            {
                "zone": zone_name,
                "square": square,
                "shape_cells": shape_cells,
            }
        )
        occupied |= shape_cells

    corridor_edges = [(0, 1)]
    if room_count >= 3 and random.random() < 0.7:
        corridor_edges.append((1, 2))
    if room_count == 4 and random.random() < 0.4:
        corridor_edges.append((2, 3))

    for a_idx, b_idx in corridor_edges:
        room_a = rooms[a_idx]
        room_b = rooms[b_idx]
        corridor = _corridor_cells(
            room_a["square"],  # type: ignore[arg-type]
            room_b["square"],  # type: ignore[arg-type]
            occupied,
        )
        if corridor is None:
            return None
        occupied |= corridor
        room_a["shape_cells"] = set(room_a["shape_cells"]) | corridor
        room_b["shape_cells"] = set(room_b["shape_cells"]) | corridor

    for room in rooms:
        for r, c in room["shape_cells"]:  # type: ignore[assignment]
            grid[r, c] = BLACK

    intended = [room["square"] for room in rooms]  # type: ignore[misc]
    if not _valid_layout(grid, intended):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
