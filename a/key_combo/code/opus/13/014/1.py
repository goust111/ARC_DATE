import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GRID_H_RANGE = (14, 18)
GRID_W_RANGE = (19, 23)
ANCHORS_PER_PANEL = (2, 3)
GREEN_COUNT_RANGE = (4, 6)
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_divider_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one full-height grey divider"
    return cols[0]


def _find_anchors(grid: np.ndarray, divider_col: int) -> list[tuple[int, int]]:
    grey = grid == GREY
    grey[:, divider_col] = False
    anchors: list[tuple[int, int]] = []
    for comp in _components(grey):
        assert comp.shape[0] == 4, "grey non-divider component must be a 2x2 anchor"
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0, c1 - c0) == (1, 1), "anchor must be 2x2"
        anchors.append((r0, c0))
    anchors.sort()
    return anchors


def _obstacle_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    mask = grid == GREY
    mask[h - 2 :, :] |= grid[h - 2 :, :] == BLUE
    mask[h - 2 :, :] |= grid[h - 2 :, :] == RED
    return mask


def _blocked_mask(grid: np.ndarray) -> np.ndarray:
    obstacle = _obstacle_mask(grid)
    blocked = obstacle.copy()
    rows, cols = np.nonzero(obstacle)
    for r, c in zip(rows, cols):
        if r - 1 >= 0:
            blocked[r - 1, c] = True
    return blocked


def _settled_green_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    blocked = _blocked_mask(grid)
    greens = [tuple(map(int, p)) for p in np.argwhere(grid == GREEN)]
    occupied = np.zeros(grid.shape, dtype=bool)
    settled: list[tuple[int, int]] = []
    by_col: dict[int, list[int]] = {}
    for r, c in greens:
        by_col.setdefault(c, []).append(r)
    for c in sorted(by_col):
        for start_r in sorted(by_col[c], reverse=True):
            r = start_r
            while r + 1 < grid.shape[0] and not blocked[r + 1, c] and not occupied[r + 1, c]:
                r += 1
            occupied[r, c] = True
            settled.append((r, c))
    settled.sort()
    return settled


def _cheb_to_anchor(point: tuple[int, int], anchor: tuple[int, int]) -> int:
    r, c = point
    ar, ac = anchor
    dr = 0 if ar <= r <= ar + 1 else min(abs(r - ar), abs(r - (ar + 1)))
    dc = 0 if ac <= c <= ac + 1 else min(abs(c - ac), abs(c - (ac + 1)))
    return max(dr, dc)


def _frame_cells(anchor: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    ar, ac = anchor
    cells: list[tuple[int, int]] = []
    for r in range(ar - 1, ar + 3):
        for c in range(ac - 1, ac + 3):
            if 0 <= r < h and 0 <= c < w and not (ar <= r <= ar + 1 and ac <= c <= ac + 1):
                cells.append((r, c))
    return cells


def _nearest_anchor(point: tuple[int, int], anchors: list[tuple[int, int]]) -> tuple[int, int]:
    dists = [(_cheb_to_anchor(point, anchor), i) for i, anchor in enumerate(anchors)]
    dists.sort()
    assert len(dists) >= 2 or dists[0][0] >= 0
    if len(dists) >= 2:
        assert dists[0][0] < dists[1][0], "nearest anchor must be unique"
    return anchors[dists[0][1]]


def _snap_assignments(
    settled: list[tuple[int, int]],
    anchors: list[tuple[int, int]],
    divider_col: int,
    shape: tuple[int, int],
) -> list[tuple[tuple[int, int], int]]:
    used: set[tuple[int, int]] = set()
    assignments: list[tuple[tuple[int, int], int]] = []
    for point in settled:
        anchor = _nearest_anchor(point, anchors)
        frame = sorted(
            _frame_cells(anchor, shape),
            key=lambda cell: (
                max(abs(cell[0] - point[0]), abs(cell[1] - point[1])),
                abs(cell[0] - point[0]) + abs(cell[1] - point[1]),
                cell[0],
                cell[1],
            ),
        )
        dest = next((cell for cell in frame if cell not in used), None)
        assert dest is not None, "anchor frame ran out of free cells"
        used.add(dest)
        color = RED if anchor[1] < divider_col else BLUE
        assignments.append((dest, color))
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """Drop green singletons, snap them to the nearest anchor frame, then recolor by opposite panel.

    Input structure:
    - The grid has a black background, a full-height grey divider column, blue/red two-row floors, grey 2x2 anchors, and green singleton pixels above the anchors.
    - The divider splits the board into left and right panels; left-floor cells are blue and right-floor cells are red.

    Rule:
    - Move each green pixel downward within its column, stopping above the one-row cushion over the floor or above the cushion over a grey anchor; lower pixels in the same column settle first and higher ones stack above them.
    - For each settled pixel, find the unique nearest grey 2x2 anchor by Chebyshev distance, then place the pixel onto the nearest free cell of that anchor's 1-cell outer frame.
    - Recolor pixels snapped to a left-panel anchor red and pixels snapped to a right-panel anchor blue; keep the divider, floors, and anchors unchanged, and clear the original green source cells.

    Key insight:
    - Gravity changes which anchor is nearest, and the chosen anchor's panel determines the final opposite panel color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    divider_col = _find_divider_col(g)
    anchors = _find_anchors(g, divider_col)
    assert anchors, "expected at least one anchor"
    settled = _settled_green_positions(g)

    out = g.copy()
    out[g == GREEN] = BG
    for dest, color in _snap_assignments(settled, anchors, divider_col, g.shape):
        out[dest] = color
    return out


def _anchor_cell_manhattan_gap(a: tuple[int, int], b: tuple[int, int]) -> int:
    best = 10**9
    for ra in range(a[0], a[0] + 2):
        for ca in range(a[1], a[1] + 2):
            for rb in range(b[0], b[0] + 2):
                for cb in range(b[1], b[1] + 2):
                    best = min(best, abs(ra - rb) + abs(ca - cb))
    return best


def _sample_panel_anchors(
    h: int,
    w: int,
    divider_col: int,
    left: bool,
    count: int,
) -> list[tuple[int, int]] | None:
    rows = list(range(3, h - 6))
    if left:
        cols = list(range(1, divider_col - 2))
    else:
        cols = list(range(divider_col + 2, w - 2))
    candidates = [(r, c) for r in rows for c in cols]
    random.shuffle(candidates)

    def search(start: int, chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return chosen
        if len(candidates) - start < count - len(chosen):
            return None
        for i in range(start, len(candidates)):
            cand = candidates[i]
            if all(_anchor_cell_manhattan_gap(cand, old) >= 5 for old in chosen):
                out = search(i + 1, chosen + [cand])
                if out is not None:
                    return out
        return None

    return search(0, [])


def _build_base_grid(
    h: int,
    w: int,
    divider_col: int,
    anchors: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[:, divider_col] = GREY
    grid[h - 2 :, :divider_col] = BLUE
    grid[h - 2 :, divider_col + 1 :] = RED
    for r, c in anchors:
        grid[r : r + 2, c : c + 2] = GREY
    return grid


def _first_blocked_row(grid: np.ndarray, col: int) -> int:
    blocked = _blocked_mask(grid)
    rows = np.flatnonzero(blocked[:, col])
    assert rows.size > 0
    return int(rows[0])


def _pick_green_columns(
    grid: np.ndarray,
    anchors: list[tuple[int, int]],
    divider_col: int,
    total: int,
) -> dict[int, int] | None:
    anchor_cols_left = sorted({c + dc for r, c in anchors if c < divider_col for dc in (0, 1)})
    anchor_cols_right = sorted({c + dc for r, c in anchors if c > divider_col for dc in (0, 1)})
    free_left = [c for c in range(divider_col) if c not in anchor_cols_left]
    free_right = [c for c in range(divider_col + 1, grid.shape[1]) if c not in anchor_cols_right]
    if not anchor_cols_left or not anchor_cols_right or not free_left or not free_right:
        return None

    base = [
        random.choice(anchor_cols_left),
        random.choice(free_left),
        random.choice(anchor_cols_right),
        random.choice(free_right),
    ]
    counts: dict[int, int] = {}
    for c in base:
        counts[c] = counts.get(c, 0) + 1

    choices = sorted(set(anchor_cols_left + anchor_cols_right + free_left + free_right))
    while sum(counts.values()) < total:
        if random.random() < 0.6:
            c = random.choice(list(counts))
        else:
            c = random.choice(choices)
        counts[c] = counts.get(c, 0) + 1

    min_anchor_row = min(r for r, _ in anchors)
    for c, k in counts.items():
        top_blocked = _first_blocked_row(grid, c)
        capacity = top_blocked
        if k > min(min_anchor_row, capacity):
            return None
    return counts


def _place_sources(
    grid: np.ndarray,
    counts: dict[int, int],
    min_anchor_row: int,
) -> bool:
    for c, k in counts.items():
        rows = list(range(min_anchor_row))
        if len(rows) < k:
            return False
        chosen = sorted(random.sample(rows, k))
        for r in chosen:
            grid[r, c] = GREEN
    return True


def _validate_input(grid: np.ndarray) -> bool:
    divider_col = _find_divider_col(grid)
    anchors = _find_anchors(grid, divider_col)
    if len([a for a in anchors if a[1] < divider_col]) < 2:
        return False
    if len([a for a in anchors if a[1] > divider_col]) < 2:
        return False

    greens = [tuple(map(int, p)) for p in np.argwhere(grid == GREEN)]
    if not (GREEN_COUNT_RANGE[0] <= len(greens) <= GREEN_COUNT_RANGE[1]):
        return False
    min_anchor_row = min(r for r, _ in anchors)
    if any(r >= min_anchor_row for r, _ in greens):
        return False

    settled = _settled_green_positions(grid)
    if len(settled) != len(greens):
        return False
    try:
        assignments = _snap_assignments(settled, anchors, divider_col, grid.shape)
    except AssertionError:
        return False

    colors = {color for _, color in assignments}
    if colors != {BLUE, RED}:
        return False

    anchor_loads: dict[tuple[int, int], int] = {}
    for point in settled:
        anchor = _nearest_anchor(point, anchors)
        anchor_loads[anchor] = anchor_loads.get(anchor, 0) + 1
    if max(anchor_loads.values()) > 3:
        return False

    settled_by_support = {"anchor": 0, "floor": 0}
    blocked = _blocked_mask(grid)
    for r, c in settled:
        if r + 1 < grid.shape[0] and blocked[r + 1, c] and grid[min(r + 2, grid.shape[0] - 1), c] == GREY:
            settled_by_support["anchor"] += 1
        else:
            settled_by_support["floor"] += 1
    if min(settled_by_support.values()) == 0:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    divider_col = w // 2

    left_count = random.randint(*ANCHORS_PER_PANEL)
    right_count = random.randint(*ANCHORS_PER_PANEL)
    left_anchors = _sample_panel_anchors(h, w, divider_col, True, left_count)
    right_anchors = _sample_panel_anchors(h, w, divider_col, False, right_count)
    if left_anchors is None or right_anchors is None:
        return None
    anchors = sorted(left_anchors + right_anchors)
    if any(_anchor_cell_manhattan_gap(a, b) < 5 for i, a in enumerate(anchors) for b in anchors[i + 1 :]):
        return None

    grid = _build_base_grid(h, w, divider_col, anchors)
    green_total = random.randint(*GREEN_COUNT_RANGE)
    counts = _pick_green_columns(grid, anchors, divider_col, green_total)
    if counts is None:
        return None
    if not _place_sources(grid, counts, min(r for r, _ in anchors)):
        return None
    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-snap-opposite-panel input")
