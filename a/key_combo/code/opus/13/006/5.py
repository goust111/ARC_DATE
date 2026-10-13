from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_PALETTE = (BLUE, RED, GREEN, YELLOW, ORANGE, TEAL, MAROON)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _nearest_squared_distance(source: np.ndarray, target: np.ndarray) -> int:
    diffs = source[:, None, :] - target[None, :, :]
    return int((diffs * diffs).sum(axis=2).min())


def _distance_map_to_cells(shape: tuple[int, int], target_cells: list[tuple[int, int]]) -> np.ndarray:
    rr, cc = np.indices(shape)
    target = np.array(target_cells, dtype=int)
    dr = rr[:, :, None] - target[:, 0]
    dc = cc[:, :, None] - target[:, 1]
    return (dr * dr + dc * dc).min(axis=2)


def _top_pink_run_length(grid: np.ndarray) -> int:
    row = np.asarray(grid[0], dtype=int)
    run = 0
    while run < row.size and int(row[run]) == PINK:
        run += 1
    assert run > 0, "expected a left-anchored pink run on the top row"
    assert not np.any(row[run:] == PINK), "top-row pink cells must form one contiguous left segment"
    return run


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the nearest marker color to recolor the grey object, erase losing markers, and stack the top pink run by the winner count.

    Input structure:
    - The grid has a black background, one contiguous pink run starting at the left edge of the top row, and one 4-connected grey object away from the top row.
    - The remaining foreground cells are isolated marker pixels in 2-3 non-grey, non-pink colors.
    - Marker colors may appear multiple times, but each colored marker cell is a single isolated pixel.

    Rule:
    - For each marker color, compute the minimum squared Euclidean distance from any marker pixel of that color to any grey-object pixel.
    - The unique color with the smallest such distance is the winning color, and its total marker count is N.
    - Repaint the entire grey object with the winning color, erase all losing-color markers to black, and keep the winning-color markers.
    - Starting from the top row, preserve the input pink run on row 1 and create N total stacked rows, extending the run one cell to the right per row, capped by the grid width.

    Key insight:
    - The same marker set contributes two different signals: distance decides which color wins, while the winner's global pixel count decides how many stacked pink rows are drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    run_len = _top_pink_run_length(g)

    grey_mask = g == GREY
    grey_comps = _components(grey_mask)
    assert len(grey_comps) == 1, "expected exactly one grey object"
    assert not grey_mask[0].any(), "grey object must not occupy the top row"
    grey_coords = np.argwhere(grey_mask)

    marker_colors = sorted(int(v) for v in np.unique(g) if int(v) not in {BLACK, GREY, PINK})
    assert 2 <= len(marker_colors) <= 3, "expected 2-3 marker colors"

    scores: list[tuple[int, int]] = []
    counts: dict[int, int] = {}
    for color in marker_colors:
        mask = g == color
        comps = _components(mask)
        assert comps and all(len(comp) == 1 for comp in comps), "markers must be isolated single pixels"
        coords = np.argwhere(mask)
        counts[color] = int(coords.shape[0])
        scores.append((_nearest_squared_distance(coords, grey_coords), color))

    best = min(dist for dist, _ in scores)
    winners = [color for dist, color in scores if dist == best]
    assert len(winners) == 1, "closest marker color must be unique"
    winner = winners[0]
    stack_rows = counts[winner]

    out = g.copy()
    for color in marker_colors:
        if color != winner:
            out[out == color] = BLACK
    out[grey_mask] = winner

    for r in range(stack_rows):
        end = min(run_len + r, w)
        out[r, :end] = PINK

    return out


GEN_ATTEMPTS = 500


def _random_connected_shape(h: int, w: int, min_row: int, area: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(min_row, h) for c in range(w)]
    if len(cells) < area:
        return None

    for _ in range(80):
        seed = random.choice(cells)
        shape = {seed}
        frontier = set()
        for dr, dc in DIRS4:
            nr, nc = seed[0] + dr, seed[1] + dc
            if _in_bounds((h, w), nr, nc) and nr >= min_row:
                frontier.add((nr, nc))

        while len(shape) < area and frontier:
            cand = random.choice(tuple(frontier))
            frontier.remove(cand)
            shape.add(cand)
            nbrs = list(DIRS4)
            random.shuffle(nbrs)
            for dr, dc in nbrs:
                nr, nc = cand[0] + dr, cand[1] + dc
                if _in_bounds((h, w), nr, nc) and nr >= min_row and (nr, nc) not in shape:
                    frontier.add((nr, nc))

        if len(shape) != area:
            continue
        rows = {r for r, _ in shape}
        cols = {c for _, c in shape}
        if len(rows) < 2 or len(cols) < 2:
            continue
        # Reject overly regular rectangles to keep the object visibly irregular.
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if bbox_area == len(shape):
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in shape:
            mask[r, c] = True
        if len(_components(mask)) == 1:
            return sorted(shape)
    return None


def _far_from_foreground(cell: tuple[int, int], occupied: set[tuple[int, int]], grey_cells: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in occupied:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    for rr, cc in grey_cells:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _choose_marker_positions(
    h: int,
    w: int,
    grey_cells: list[tuple[int, int]],
    reserved_rows: int,
    colors: list[int],
    counts: dict[int, int],
    winner: int,
) -> dict[int, list[tuple[int, int]]] | None:
    grey_set = set(grey_cells)
    dist_map = _distance_map_to_cells((h, w), grey_cells)

    candidates = [
        (r, c)
        for r in range(reserved_rows, h)
        for c in range(w)
        if (r, c) not in grey_set and _far_from_foreground((r, c), set(), grey_set)
    ]
    if not candidates:
        return None

    distances = sorted({int(dist_map[r, c]) for r, c in candidates})
    if len(distances) < len(colors):
        return None

    usable = distances[: min(len(distances), 10)]
    if len(usable) < len(colors):
        return None
    chosen_distances = sorted(random.sample(usable, len(colors)))

    color_to_min: dict[int, int] = {}
    other_colors = [color for color in colors if color != winner]
    color_to_min[winner] = chosen_distances[0]
    random.shuffle(other_colors)
    for color, dist in zip(other_colors, chosen_distances[1:]):
        color_to_min[color] = dist

    occupied: set[tuple[int, int]] = set()
    placed: dict[int, list[tuple[int, int]]] = {color: [] for color in colors}

    order = sorted(colors, key=lambda color: (color_to_min[color], -counts[color]))
    for color in order:
        dmin = color_to_min[color]
        anchor_pool = [
            (r, c)
            for r, c in candidates
            if int(dist_map[r, c]) == dmin and _far_from_foreground((r, c), occupied, grey_set)
        ]
        random.shuffle(anchor_pool)
        if not anchor_pool:
            return None
        anchor = anchor_pool[0]
        placed[color].append(anchor)
        occupied.add(anchor)

        filler_pool = [
            (r, c)
            for r, c in candidates
            if int(dist_map[r, c]) >= dmin and (r, c) not in occupied
        ]
        random.shuffle(filler_pool)
        for cell in filler_pool:
            if len(placed[color]) == counts[color]:
                break
            if _far_from_foreground(cell, occupied, grey_set):
                placed[color].append(cell)
                occupied.add(cell)
        if len(placed[color]) != counts[color]:
            return None

    mins = {}
    grey_arr = np.array(grey_cells, dtype=int)
    for color, cells in placed.items():
        mins[color] = _nearest_squared_distance(np.array(cells, dtype=int), grey_arr)
        if mins[color] != color_to_min[color]:
            return None
    best = min(mins.values())
    if sum(1 for v in mins.values() if v == best) != 1 or mins[winner] != best:
        return None

    return placed


def _render_input(
    h: int,
    w: int,
    run_len: int,
    grey_cells: list[tuple[int, int]],
    marker_cells: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[0, :run_len] = PINK
    for r, c in grey_cells:
        grid[r, c] = GREY
    for color, cells in marker_cells.items():
        for r, c in cells:
            grid[r, c] = color
    return grid


def _valid_generated_input(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if grid.ndim != 2 or h > 30 or w > 30:
        return False
    try:
        out = main(grid)
    except Exception:
        return False

    grey_comps = _components(grid == GREY)
    if len(grey_comps) != 1:
        return False

    run_len = _top_pink_run_length(grid)
    marker_colors = [int(v) for v in np.unique(grid) if int(v) not in {BLACK, GREY, PINK}]
    if len(marker_colors) not in {2, 3}:
        return False
    grey_coords = np.argwhere(grid == GREY)
    scores = []
    for color in marker_colors:
        coords = np.argwhere(grid == color)
        if not (1 <= len(coords) <= 4):
            return False
        if not all(len(comp) == 1 for comp in _components(grid == color)):
            return False
        scores.append((_nearest_squared_distance(coords, grey_coords), color))
    best = min(dist for dist, _ in scores)
    winners = [color for dist, color in scores if dist == best]
    if len(winners) != 1:
        return False
    winner = winners[0]
    n = int(np.count_nonzero(grid == winner))
    if run_len + n - 1 > w:
        return False

    if not np.array_equal(out, main(grid)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(10, 14)
    w = random.randint(10, 14)

    color_count = random.choice([2, 3])
    colors = random.sample(MARKER_PALETTE, color_count)
    winner = random.choice(colors)
    counts = {color: random.randint(1, 4) for color in colors}
    winner_count = counts[winner]

    max_run = min(5, w - winner_count + 1)
    if max_run < 2:
        return None
    run_len = random.randint(2, max_run)

    reserved_rows = winner_count + 1
    min_grey_row = reserved_rows + 1
    if min_grey_row > h - 3:
        return None

    grey_area = random.randint(5, min(15, (h - min_grey_row) * w))
    grey_cells = _random_connected_shape(h, w, min_grey_row, grey_area)
    if grey_cells is None:
        return None

    marker_cells = _choose_marker_positions(h, w, grey_cells, reserved_rows, colors, counts, winner)
    if marker_cells is None:
        return None

    grid = _render_input(h, w, run_len, grey_cells, marker_cells)
    if not _valid_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
