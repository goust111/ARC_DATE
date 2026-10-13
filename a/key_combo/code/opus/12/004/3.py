from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MAIN = 1
DISTRACTOR = 2
MARKER = 7

GRID_MIN = 13
GRID_MAX = 18
BLUE_AREA_MIN = 15
BLUE_AREA_MAX = 25
MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = set()
    queue = deque([next(iter(cells))])
    seen.add(queue[0])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _axis_from_markers(grid: np.ndarray) -> tuple[str, int]:
    markers = np.argwhere(grid == MARKER)
    assert 3 <= len(markers) <= 4, "expected 3-4 marker pixels"
    rows = {int(r) for r in markers[:, 0]}
    cols = {int(c) for c in markers[:, 1]}
    if len(cols) == 1 and len(rows) >= 3:
        return "vertical", next(iter(cols))
    if len(rows) == 1 and len(cols) >= 3:
        return "horizontal", next(iter(rows))
    raise AssertionError("marker pixels must be strictly collinear on one row or column")


def _mirror_cell(r: int, c: int, axis_kind: str, axis_index: int) -> tuple[int, int]:
    if axis_kind == "vertical":
        return r, 2 * axis_index - c
    return 2 * axis_index - r, c


def _mirror_targets(
    blue_cells: set[tuple[int, int]], axis_kind: str, axis_index: int
) -> set[tuple[int, int]]:
    return {_mirror_cell(r, c, axis_kind, axis_index) for r, c in blue_cells}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden symmetry axis from marker pixels, mirror the blue object across it, repair blocked targets, and erase the markers.

    Input structure:
    - The background is black.
    - A single irregular blue 4-connected object lies entirely on one side of a hidden symmetry axis.
    - Three or four orange marker pixels are strictly collinear on one row or one column, encoding that axis.
    - Red single-pixel distractors include some cells that sit exactly on mirrored target positions and some that do not.

    Rule:
    - Use the orange markers' common row or common column as the symmetry axis.
    - Reflect every blue pixel across that axis.
    - Paint every mirrored target blue whether it was black or occupied by a red distractor.
    - Erase every orange marker pixel and every red distractor not used as a mirrored target.
    - Keep the original blue object unchanged.

    Key insight:
    - The axis is not a visible centerline; it is implicitly specified by the orange markers' alignment, and only after finding it can you tell which red pixels are repair sites versus removable clutter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    axis_kind, axis_index = _axis_from_markers(g)
    blue_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == MAIN)}
    assert blue_cells, "expected a blue object"

    out = np.zeros_like(g)
    for r, c in blue_cells:
        out[r, c] = MAIN

    for r, c in blue_cells:
        mr, mc = _mirror_cell(r, c, axis_kind, axis_index)
        assert _in_bounds(g.shape, mr, mc), "mirrored targets must stay in bounds"
        out[mr, mc] = MAIN

    return out


def _grow_blob_in_bbox(
    top: int, left: int, height: int, width: int, target_area: int
) -> set[tuple[int, int]] | None:
    cells_in_box = [(top + r, left + c) for r in range(height) for c in range(width)]
    start = random.choice(cells_in_box)
    blob = {start}
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(*start)
        if top <= nr < top + height and left <= nc < left + width
    }

    while len(blob) < target_area:
        if not frontier:
            return None
        frontier_list = list(frontier)
        weights = []
        for r, c in frontier_list:
            degree = sum((nr, nc) in blob for nr, nc in _neighbors4(r, c))
            weights.append((degree + 1) ** 3)
        choice = random.choices(frontier_list, weights=weights, k=1)[0]
        frontier.remove(choice)
        blob.add(choice)
        for nr, nc in _neighbors4(*choice):
            if (
                top <= nr < top + height
                and left <= nc < left + width
                and (nr, nc) not in blob
            ):
                frontier.add((nr, nc))
    return blob


def _choose_bbox_vertical(
    h: int, w: int, axis: int, side: str
) -> tuple[int, int, int, int] | None:
    max_dist = min(axis, w - 1 - axis)
    max_width = min(7, max_dist)
    if max_width < 4:
        return None

    height = random.randint(5, min(8, h))
    width = random.randint(4, max_width)
    top = random.randint(0, h - height)

    if side == "left":
        col_lo = max(0, 2 * axis - (w - 1))
        col_hi = axis - width
    else:
        col_lo = axis + 1
        col_hi = min(w - width, 2 * axis - width + 1)

    if col_lo > col_hi:
        return None
    left = random.randint(col_lo, col_hi)
    return top, left, height, width


def _choose_bbox_horizontal(
    h: int, w: int, axis: int, side: str
) -> tuple[int, int, int, int] | None:
    max_dist = min(axis, h - 1 - axis)
    max_height = min(7, max_dist)
    if max_height < 4:
        return None

    height = random.randint(4, max_height)
    width = random.randint(5, min(8, w))
    left = random.randint(0, w - width)

    if side == "top":
        row_lo = max(0, 2 * axis - (h - 1))
        row_hi = axis - height
    else:
        row_lo = axis + 1
        row_hi = min(h - height, 2 * axis - height + 1)

    if row_lo > row_hi:
        return None
    top = random.randint(row_lo, row_hi)
    return top, left, height, width


def _marker_candidates(
    grid: np.ndarray, axis_kind: str, axis_index: int
) -> list[tuple[int, int]]:
    h, w = grid.shape
    if axis_kind == "vertical":
        cells = [(r, axis_index) for r in range(h)]
    else:
        cells = [(axis_index, c) for c in range(w)]

    candidates = []
    for r, c in cells:
        if grid[r, c] != BG:
            continue
        if all(
            (not _in_bounds(grid.shape, nr, nc)) or grid[nr, nc] not in {MAIN, DISTRACTOR}
            for nr, nc in _neighbors4(r, c)
        ):
            candidates.append((r, c))
    return candidates


def _sample_spaced(candidates: list[tuple[int, int]], count: int) -> list[tuple[int, int]] | None:
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _semantic_check(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not set(np.unique(grid)).issubset({BG, MAIN, DISTRACTOR, MARKER}):
        return False

    try:
        axis_kind, axis_index = _axis_from_markers(grid)
    except AssertionError:
        return False

    h, w = grid.shape
    if axis_kind == "vertical" and w % 2 == 1 and axis_index == w // 2:
        return False
    if axis_kind == "horizontal" and h % 2 == 1 and axis_index == h // 2:
        return False

    blue_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == MAIN)}
    if not (BLUE_AREA_MIN <= len(blue_cells) <= BLUE_AREA_MAX):
        return False
    if not _is_connected(blue_cells):
        return False

    if axis_kind == "vertical":
        if not (all(c < axis_index for _, c in blue_cells) or all(c > axis_index for _, c in blue_cells)):
            return False
    else:
        if not (all(r < axis_index for r, _ in blue_cells) or all(r > axis_index for r, _ in blue_cells)):
            return False

    targets = _mirror_targets(blue_cells, axis_kind, axis_index)
    if len(targets) != len(blue_cells):
        return False
    if any(not _in_bounds(grid.shape, r, c) for r, c in targets):
        return False
    if blue_cells & targets:
        return False

    red_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == DISTRACTOR)}
    if not (2 <= len(red_cells) <= 4):
        return False
    embedded = red_cells & targets
    isolated = red_cells - targets
    if not embedded or not isolated:
        return False
    if len(embedded) == len(targets):
        return False

    marker_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)}
    if not (3 <= len(marker_cells) <= 4):
        return False
    for r, c in marker_cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] in {MAIN, DISTRACTOR}:
                return False

    expected = np.zeros_like(grid)
    for r, c in blue_cells | targets:
        expected[r, c] = MAIN
    if not np.array_equal(main(grid), expected):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    axis_kind = random.choice(["vertical", "horizontal"])

    if axis_kind == "vertical":
        axis_candidates = [
            c
            for c in range(4, w - 4)
            if not (w % 2 == 1 and c == w // 2)
        ]
        if not axis_candidates:
            return None
        axis_index = random.choice(axis_candidates)
        side = random.choice(["left", "right"])
        bbox = _choose_bbox_vertical(h, w, axis_index, side)
    else:
        axis_candidates = [
            r
            for r in range(4, h - 4)
            if not (h % 2 == 1 and r == h // 2)
        ]
        if not axis_candidates:
            return None
        axis_index = random.choice(axis_candidates)
        side = random.choice(["top", "bottom"])
        bbox = _choose_bbox_horizontal(h, w, axis_index, side)

    if bbox is None:
        return None

    top, left, height, width = bbox
    max_area = min(BLUE_AREA_MAX, height * width - 1)
    min_area = max(BLUE_AREA_MIN, (height * width + 1) // 2)
    if min_area > max_area:
        return None
    blue_area = random.randint(min_area, max_area)
    blue_cells = _grow_blob_in_bbox(top, left, height, width, blue_area)
    if blue_cells is None:
        return None

    r0, c0, r1, c1 = _bbox(blue_cells)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    targets = _mirror_targets(blue_cells, axis_kind, axis_index)
    if any(not _in_bounds((h, w), r, c) for r, c in targets):
        return None
    if blue_cells & targets:
        return None

    repair_max = min(2, len(targets) - 1)
    if repair_max < 1:
        return None
    repair_count = random.randint(1, repair_max)
    repair_cells = set(random.sample(list(targets), repair_count))

    grid = np.zeros((h, w), dtype=int)
    for r, c in blue_cells:
        grid[r, c] = MAIN
    for r, c in repair_cells:
        grid[r, c] = DISTRACTOR

    isolated_candidates = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or (r, c) in targets:
                continue
            if any((nr, nc) in blue_cells for nr, nc in _neighbors4(r, c)):
                continue
            isolated_candidates.append((r, c))
    random.shuffle(isolated_candidates)

    isolated_count = random.randint(1, min(2, 4 - repair_count))
    if len(isolated_candidates) < isolated_count:
        return None
    for r, c in isolated_candidates[:isolated_count]:
        grid[r, c] = DISTRACTOR

    marker_candidates = _marker_candidates(grid, axis_kind, axis_index)
    marker_count_options = [count for count in (3, 4) if len(marker_candidates) >= count]
    if not marker_count_options:
        return None
    marker_count = random.choice(marker_count_options)
    chosen_markers = _sample_spaced(marker_candidates, marker_count)
    if chosen_markers is None:
        return None
    for r, c in chosen_markers:
        grid[r, c] = MARKER

    return grid if _semantic_check(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-axis symmetry instance")
