import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (BLUE, GREEN, YELLOW)

MAX_GENERATION_ATTEMPTS = 400
MIN_REGION_BLACK_CELLS = 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow bottom-up red divider paths, then flood-fill each resulting open region by its internal marker color.

    Input structure:
    - A black background contains 2-3 red start pixels on the bottom row.
    - Grey obstacle pixels lie strictly inside the grid and can redirect growing paths.
    - Blue, green, or yellow single-pixel markers sit inside the open areas cut out by the future paths.

    Rule:
    - Starting from each bottom-row red pixel, repeatedly try to move one cell upward; if the cell above is grey, move one cell right instead, and stop if that right step is blocked or leaves the grid.
    - After all red paths have grown, treat the red path pixels and grey obstacles as barriers and find the 4-connected open regions they leave behind, with markers counting as part of those regions.
    - Each open region contains exactly one marker pixel; recolor every black pixel in that region to the marker's color while leaving the marker, red paths, and grey obstacles unchanged.

    Key insight:
    - The red paths are only intermediate dividers: you must simulate their obstacle-driven geometry first, because that geometry determines which marker controls each final filled region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grown = _simulate_paths(g)
    out = grown.copy()

    for coords in _open_components(grown):
        colors = [int(grown[r, c]) for r, c in coords if int(grown[r, c]) in MARKER_COLORS]
        assert len(colors) == 1, "each open region must contain exactly one marker"
        fill_color = colors[0]
        for r, c in coords:
            if int(out[r, c]) == BLACK:
                out[r, c] = fill_color

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for opus/12/009/3")


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _neighbors8(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(mask.shape, cr, cc):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _open_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    open_mask = (grid != RED) & (grid != GREY)
    return _components(open_mask)


def _simulate_paths(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = out.shape
    starts = [int(c) for c in np.flatnonzero(out[h - 1] == RED)]

    for c0 in starts:
        r = h - 1
        c = c0
        while r > 0:
            if int(out[r - 1, c]) == GREY:
                if c + 1 >= w or int(out[r, c + 1]) == GREY:
                    break
                out[r, c + 1] = RED
                c += 1
            else:
                out[r - 1, c] = RED
                r -= 1
    return out


def _fill_open_regions(grown: np.ndarray) -> np.ndarray:
    out = grown.copy()
    for coords in _open_components(grown):
        marker_cells = [(r, c) for r, c in coords if int(grown[r, c]) in MARKER_COLORS]
        if len(marker_cells) != 1:
            raise AssertionError("expected exactly one marker per open region")
        color = int(grown[marker_cells[0]])
        for r, c in coords:
            if int(out[r, c]) == BLACK:
                out[r, c] = color
    return out


def _sample_start_columns(width: int, count: int) -> list[int] | None:
    min_gap = 4
    cols: list[int] = []
    lo = 2
    for i in range(count):
        hi = width - 4 - (count - 1 - i) * min_gap
        if lo > hi:
            return None
        cols.append(random.randint(lo, hi))
        lo = cols[-1] + min_gap
    return cols


def _interior(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 0 < r < h - 1 and 0 < c < w - 1


def _trace_full_path(
    shape: tuple[int, int],
    start_col: int,
    detour_rows: list[int],
    reserved_red: set[tuple[int, int]],
    reserved_grey: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    path = [(h - 1, start_col)]
    current = (h - 1, start_col)
    local_red = {current}
    local_grey: list[tuple[int, int]] = []
    pending = set(detour_rows)

    while current[0] > 0:
        r, c = current
        if r in pending:
            blocker = (r - 1, c)
            step = (r, c + 1)
            if not _interior(blocker, shape):
                return None
            if step[1] >= w:
                return None
            if blocker in reserved_red or blocker in reserved_grey or blocker in local_red:
                return None
            if step in reserved_grey or step in reserved_red or step in local_red:
                return None
            local_grey.append(blocker)
            path.append(step)
            local_red.add(step)
            current = step
            pending.remove(r)
        else:
            step = (r - 1, c)
            if step in reserved_grey or step in reserved_red or step in local_red:
                return None
            path.append(step)
            local_red.add(step)
            current = step

    if pending:
        return None
    return path, local_grey


def _trace_terminal_path(
    shape: tuple[int, int],
    start_col: int,
    stop_row: int,
    detour_rows: list[int],
    reserved_red: set[tuple[int, int]],
    reserved_grey: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    path = [(h - 1, start_col)]
    current = (h - 1, start_col)
    local_red = {current}
    local_grey: list[tuple[int, int]] = []
    pending = set(detour_rows)

    while current[0] > stop_row:
        r, c = current
        if r in pending:
            blocker = (r - 1, c)
            step = (r, c + 1)
            if not _interior(blocker, shape):
                return None
            if step[1] >= w:
                return None
            if blocker in reserved_red or blocker in reserved_grey or blocker in local_red:
                return None
            if step in reserved_grey or step in reserved_red or step in local_red:
                return None
            local_grey.append(blocker)
            path.append(step)
            local_red.add(step)
            current = step
            pending.remove(r)
        else:
            step = (r - 1, c)
            if step in reserved_grey or step in reserved_red or step in local_red:
                return None
            path.append(step)
            local_red.add(step)
            current = step

    if pending:
        return None

    r, c = current
    top_blocker = (r - 1, c)
    right_blocker = (r, c + 1)
    if not _interior(top_blocker, shape) or not _interior(right_blocker, shape):
        return None
    if right_blocker in reserved_red or right_blocker in reserved_grey or right_blocker in local_red:
        return None
    if top_blocker in reserved_red or top_blocker in reserved_grey or top_blocker in local_red:
        return None
    local_grey.extend([top_blocker, right_blocker])
    return path, local_grey


def _sample_full_path_plans(
    shape: tuple[int, int],
    starts: list[int],
    full_count: int,
) -> list[tuple[int, list[int]]] | None:
    h, w = shape
    plans: list[tuple[int, list[int]]] = []

    for i in range(full_count):
        start = starts[i]
        right_limit = (starts[i + 1] - 3) if i + 1 < len(starts) else (w - 3)
        max_shift = right_limit - start
        if max_shift < 0:
            return None
        shift = random.randint(0, min(3, max_shift))
        if shift == 0:
            rows: list[int] = []
        else:
            rows = sorted(random.sample(list(range(2, h)), shift), reverse=True)
        plans.append((start, rows))

    return plans


def _sample_terminal_plan(
    shape: tuple[int, int],
    start_col: int,
) -> tuple[int, list[int]] | None:
    h, w = shape
    if start_col > w - 4:
        return None
    stop_row = random.randint(3, h - 4)
    max_shift = min(2, (w - 3) - start_col)
    if max_shift < 0:
        return None
    shift = random.randint(0, max_shift)
    candidate_rows = list(range(stop_row + 1, h))
    if shift > len(candidate_rows):
        return None
    rows = sorted(random.sample(candidate_rows, shift), reverse=True)
    return stop_row, rows


def _place_markers(base: np.ndarray, grown: np.ndarray) -> np.ndarray | None:
    regions = _open_components(grown)
    if len(regions) < 3 or len(regions) > 6:
        return None

    black_regions = []
    for coords in regions:
        blacks = [(r, c) for r, c in coords if int(grown[r, c]) == BLACK]
        if len(blacks) < MIN_REGION_BLACK_CELLS:
            return None
        black_regions.append(blacks)

    marker_grid = base.copy()
    preferred_per_region: list[list[tuple[int, int]]] = []
    for blacks in black_regions:
        candidates = []
        preferred = []
        for r, c in blacks:
            if any(int(grown[nr, nc]) == GREY for nr, nc in _neighbors8(grown.shape, r, c)):
                continue
            candidates.append((r, c))
            if all(int(grown[nr, nc]) != RED for nr, nc in _neighbors4(grown.shape, r, c)):
                preferred.append((r, c))
        if not candidates:
            return None
        preferred_per_region.append(preferred or candidates)

    region_count = len(regions)
    if region_count <= 3:
        palette = random.sample(MARKER_COLORS, region_count)
    else:
        palette = [random.choice(MARKER_COLORS) for _ in range(region_count)]
        if len(set(palette)) < 2:
            palette[0] = random.choice(MARKER_COLORS)
            palette[1] = next(color for color in MARKER_COLORS if color != palette[0])
    random.shuffle(palette)

    for color, candidates in zip(palette, preferred_per_region):
        r, c = random.choice(candidates)
        marker_grid[r, c] = color

    grown_with_markers = _simulate_paths(marker_grid)
    for coords in _open_components(grown_with_markers):
        markers = [(r, c) for r, c in coords if int(grown_with_markers[r, c]) in MARKER_COLORS]
        if len(markers) != 1:
            return None
    return marker_grid


def _semantic_checks(inp: np.ndarray, planned_red: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    starts = [int(c) for c in np.flatnonzero(inp[h - 1] == RED)]
    if len(starts) not in (2, 3):
        return False
    if np.any(inp[:-1] == RED):
        return False

    grey_coords = np.argwhere(inp == GREY)
    if grey_coords.size == 0:
        return False
    if any(r in (0, h - 1) or c in (0, w - 1) for r, c in grey_coords):
        return False

    marker_coords = np.argwhere(np.isin(inp, MARKER_COLORS))
    if marker_coords.size == 0:
        return False
    for r, c in marker_coords:
        if any(int(inp[nr, nc]) == GREY for nr, nc in _neighbors8(inp.shape, int(r), int(c))):
            return False

    grown = _simulate_paths(inp)
    actual_red = {(int(r), int(c)) for r, c in np.argwhere(grown == RED)}
    if actual_red != planned_red:
        return False

    detour_count = 0
    for r, c in sorted(actual_red):
        if r > 0 and int(grown[r, c]) == RED and int(inp[r - 1, c]) == GREY:
            detour_count += 1
    if detour_count == 0:
        return False

    components = _open_components(grown)
    if len(components) < 3:
        return False
    for coords in components:
        markers = [(r, c) for r, c in coords if int(grown[r, c]) in MARKER_COLORS]
        black_count = sum(int(grown[r, c]) == BLACK for r, c in coords)
        if len(markers) != 1 or black_count < 1:
            return False

    out = _fill_open_regions(grown)
    if np.array_equal(inp, out):
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(15, 19)
    width = random.randint(15, 20)
    start_count = random.choice((2, 3))
    starts = _sample_start_columns(width, start_count)
    if starts is None:
        return None

    include_terminal = start_count == 3 and random.random() < 0.4
    full_count = start_count - 1 if include_terminal else start_count
    plans = _sample_full_path_plans((height, width), starts, full_count)
    if plans is None:
        return None

    if sum(len(rows) for _start, rows in plans) == 0 and not include_terminal:
        shiftable = [i for i, (start, rows) in enumerate(plans) if len(rows) == 0]
        if not shiftable:
            return None
        idx = random.choice(shiftable)
        start = starts[idx]
        right_limit = (starts[idx + 1] - 3) if idx + 1 < len(starts) else (width - 3)
        if right_limit - start < 1:
            return None
        detour_row = random.randint(2, height - 1)
        plans[idx] = (start, [detour_row])

    grid = np.zeros((height, width), dtype=int)
    grid[height - 1, starts] = RED

    reserved_red = {(height - 1, c) for c in starts}
    reserved_grey: set[tuple[int, int]] = set()
    all_path_cells: set[tuple[int, int]] = set(reserved_red)

    for start_col, detour_rows in plans:
        traced = _trace_full_path((height, width), start_col, detour_rows, reserved_red, reserved_grey)
        if traced is None:
            return None
        path_cells, grey_cells = traced
        for r, c in grey_cells:
            grid[r, c] = GREY
            reserved_grey.add((r, c))
        for cell in path_cells:
            reserved_red.add(cell)
            all_path_cells.add(cell)

    if include_terminal:
        terminal = _sample_terminal_plan((height, width), starts[-1])
        if terminal is None:
            return None
        stop_row, detour_rows = terminal
        traced = _trace_terminal_path(
            (height, width), starts[-1], stop_row, detour_rows, reserved_red, reserved_grey
        )
        if traced is None:
            return None
        path_cells, grey_cells = traced
        for r, c in grey_cells:
            grid[r, c] = GREY
            reserved_grey.add((r, c))
        for cell in path_cells:
            reserved_red.add(cell)
            all_path_cells.add(cell)

    grown = _simulate_paths(grid)
    if {(int(r), int(c)) for r, c in np.argwhere(grown == RED)} != all_path_cells:
        return None

    marked = _place_markers(grid, grown)
    if marked is None:
        return None
    if not _semantic_checks(marked, all_path_cells):
        return None
    return marked
