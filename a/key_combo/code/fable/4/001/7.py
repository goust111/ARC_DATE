from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, bbox_h, bbox_w, components_mask


BLACK = int(Color.BLACK)
WALL = int(Color.GREY)
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _rooms(grid: np.ndarray) -> list[np.ndarray]:
    return components_mask(grid != WALL, connectivity=4)


def _component_labels(shape: tuple[int, int], comps: list[np.ndarray]) -> np.ndarray:
    labels = np.zeros(shape, dtype=int)
    for idx, coords in enumerate(comps, start=1):
        labels[coords[:, 0], coords[:, 1]] = idx
    return labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill black cells that share both a room and a coordinate parity lattice with each marker.

    Input structure:
    - The grid uses black background cells and grey wall cells.
    - Grey walls partition the non-wall cells into closed 4-connected rooms.
    - Some rooms contain exactly one uniquely colored marker pixel; other rooms may be empty.

    Rule:
    - For each marker, use its global row parity and column parity.
    - Recolor only black cells that are 4-connected to that marker without crossing grey walls
      and whose row and column parities both match the marker.
    - Leave grey walls, marker pixels, empty rooms, other rooms, and parity-mismatched black cells unchanged.

    Key insight:
    - The selected cells are the intersection of a local room component and a global parity lattice;
      either test alone colors cells that should stay black.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    marker_mask = (g != BLACK) & (g != WALL)
    marker_coords = np.argwhere(marker_mask)
    assert marker_coords.shape[0] >= 1, "expected at least one colored marker"
    marker_values = [int(g[r, c]) for r, c in marker_coords]
    assert len(marker_values) == len(set(marker_values)), "marker colors must be unique single pixels"

    out = g.copy()
    for coords in _rooms(g):
        room_marker_coords = [tuple(map(int, rc)) for rc in coords if marker_mask[int(rc[0]), int(rc[1])]]
        assert len(room_marker_coords) <= 1, "each marked room must contain exactly one marker"
        if not room_marker_coords:
            continue

        mr, mc = room_marker_coords[0]
        color = int(g[mr, mc])
        same_parity = (coords[:, 0] % 2 == mr % 2) & (coords[:, 1] % 2 == mc % 2)
        targets = coords[same_parity]
        for r, c in targets:
            rr, cc = int(r), int(c)
            if int(g[rr, cc]) == BLACK:
                out[rr, cc] = color

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 500
ROOM_ROWS_CHOICES = (2, 2, 3)
ROOM_COLS_CHOICES = (2, 2, 3)
ROOM_SIZE_RANGE = (3, 7)
MAX_MARKERS = 6


def _build_rectangular_rooms() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    row_count = random.choice(ROOM_ROWS_CHOICES)
    col_count = random.choice(ROOM_COLS_CHOICES)
    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(row_count)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(col_count)]

    height = sum(room_heights) + row_count + 1
    width = sum(room_widths) + col_count + 1
    grid = np.full((height, width), WALL, dtype=int)

    boxes: list[tuple[int, int, int, int]] = []
    r0 = 1
    for room_h in room_heights:
        c0 = 1
        for room_w in room_widths:
            grid[r0 : r0 + room_h, c0 : c0 + room_w] = BLACK
            boxes.append((r0, c0, r0 + room_h - 1, c0 + room_w - 1))
            c0 += room_w + 1
        r0 += room_h + 1

    return grid, boxes


def _try_add_protrusion(grid: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    options: list[list[tuple[int, int]]] = []
    v_max = min(2, height - 2)
    h_max = min(2, width - 2)
    for col in range(c0 + 1, c1):
        for length in range(1, v_max + 1):
            options.append([(r0 + dr, col) for dr in range(length)])
            options.append([(r1 - dr, col) for dr in range(length)])
    for row in range(r0 + 1, r1):
        for length in range(1, h_max + 1):
            options.append([(row, c0 + dc) for dc in range(length)])
            options.append([(row, c1 - dc) for dc in range(length)])

    random.shuffle(options)
    for cells in options:
        if any(int(grid[r, c]) != BLACK for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = WALL

        patch = grid[r0 : r1 + 1, c0 : c1 + 1] != WALL
        comps = components_mask(patch, connectivity=4)
        if len(comps) == 1 and comps[0].shape[0] >= 6:
            return True

        for r, c in cells:
            grid[r, c] = BLACK

    return False


def _add_room_protrusions(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> int:
    shuffled = boxes[:]
    random.shuffle(shuffled)
    target_count = random.randint(1, min(4, len(shuffled)))
    placed = 0
    for box in shuffled:
        if _try_add_protrusion(grid, box):
            placed += 1
        if placed >= target_count:
            break
    return placed


def _room_is_nonrectangular(coords: np.ndarray) -> bool:
    box = bbox_coords(coords)
    return int(coords.shape[0]) < bbox_h(box) * bbox_w(box)


def _marker_candidates(grid: np.ndarray, coords: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    rr = coords[:, 0]
    cc = coords[:, 1]
    for r, c in coords:
        r = int(r)
        c = int(c)
        if int(grid[r, c]) != BLACK:
            continue
        same_count = int(np.count_nonzero((rr % 2 == r % 2) & (cc % 2 == c % 2)))
        mismatch_count = int(coords.shape[0]) - same_count
        if same_count >= 2 and mismatch_count >= 1:
            out.append((r, c))
    return out


def _place_markers(grid: np.ndarray) -> bool:
    comps = _rooms(grid)
    if len(comps) < 3:
        return False

    candidate_indices = [i for i, coords in enumerate(comps) if _marker_candidates(grid, coords)]
    nonrect_candidates = [i for i in candidate_indices if _room_is_nonrectangular(comps[i])]
    if len(candidate_indices) < 2 or not nonrect_candidates:
        return False

    max_markers = min(MAX_MARKERS, len(candidate_indices), len(comps) - 1, len(MARKER_COLORS))
    if max_markers < 2:
        return False

    marker_count = random.randint(2, max_markers)
    chosen = [random.choice(nonrect_candidates)]
    remaining = [i for i in candidate_indices if i not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[: marker_count - 1])
    random.shuffle(chosen)

    colors = random.sample(MARKER_COLORS, len(chosen))
    for idx, color in zip(chosen, colors):
        r, c = random.choice(_marker_candidates(grid, comps[idx]))
        grid[r, c] = int(color)

    return True


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    marker_mask = (grid != BLACK) & (grid != WALL)
    for coords in _rooms(grid):
        markers = [tuple(map(int, rc)) for rc in coords if marker_mask[int(rc[0]), int(rc[1])]]
        if not markers:
            continue
        assert len(markers) == 1
        mr, mc = markers[0]
        color = int(grid[mr, mc])
        same_parity = (coords[:, 0] % 2 == mr % 2) & (coords[:, 1] % 2 == mc % 2)
        targets = coords[same_parity]
        for r, c in targets:
            rr, cc = int(r), int(c)
            if int(grid[rr, cc]) == BLACK:
                out[rr, cc] = color
    return out


def _semantically_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    comps = _rooms(grid)
    labels = _component_labels(grid.shape, comps)
    marker_coords = np.argwhere((grid != BLACK) & (grid != WALL))
    if marker_coords.shape[0] < 2:
        return False

    colors = [int(grid[r, c]) for r, c in marker_coords]
    if len(colors) != len(set(colors)):
        return False

    marker_room_ids = [int(labels[r, c]) for r, c in marker_coords]
    if any(room_id == 0 for room_id in marker_room_ids):
        return False
    if len(marker_room_ids) != len(set(marker_room_ids)):
        return False
    if len(comps) - len(marker_room_ids) < 1:
        return False
    if len({(int(r) % 2, int(c) % 2) for r, c in marker_coords}) < 2:
        return False

    has_nonrect_marked_room = False
    total_fill = 0
    rr_all, cc_all = np.indices(grid.shape)
    for r, c in marker_coords:
        r = int(r)
        c = int(c)
        room_id = int(labels[r, c])
        coords = comps[room_id - 1]
        same = (rr_all % 2 == r % 2) & (cc_all % 2 == c % 2)
        in_room = labels == room_id

        fill_targets = (grid == BLACK) & in_room & same
        parity_misses = (grid == BLACK) & in_room & (~same)
        blocked_same_parity = (grid == BLACK) & (labels != room_id) & same
        if int(np.count_nonzero(fill_targets)) < 1:
            return False
        if int(np.count_nonzero(parity_misses)) < 1:
            return False
        if int(np.count_nonzero(blocked_same_parity)) < 1:
            return False

        total_fill += int(np.count_nonzero(fill_targets))
        has_nonrect_marked_room = has_nonrect_marked_room or _room_is_nonrectangular(coords)

    if not has_nonrect_marked_room or total_fill < int(marker_coords.shape[0]):
        return False

    expected = _expected_output(grid)
    if np.array_equal(expected, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid, boxes = _build_rectangular_rooms()
    if max(grid.shape) > 30:
        return None

    if _add_room_protrusions(grid, boxes) == 0:
        return None

    if len(_rooms(grid)) != len(boxes):
        return None

    if not _place_markers(grid):
        return None

    if not _semantically_valid(grid):
        return None

    expected = _expected_output(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-room input")
