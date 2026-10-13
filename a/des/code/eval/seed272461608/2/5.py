import random

import numpy as np


BG = 0
OBSTACLE = 5
RING_COLORS = (2, 7, 4, 3, 8, 1, 6, 9)
RING_OFFSETS = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
RING_COORDS = [(1 + dr, 1 + dc) for dr, dc in RING_OFFSETS]

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
SEED_COUNT_RANGE = (4, 6)
MIN_BRANCH_LEN = 2
MAX_BRANCH_LEN = 8
GEN_RETRIES = 300


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _legend_mapping(grid: np.ndarray) -> dict[int, tuple[int, int, int]]:
    ring = [int(grid[r, c]) for r, c in RING_COORDS]
    assert int(grid[1, 1]) == BG
    assert len(set(ring)) == 8

    mapping: dict[int, tuple[int, int, int]] = {}
    for idx, source in enumerate(ring):
        next_idx = (idx + 1) % 8
        target = ring[next_idx]
        dr, dc = RING_OFFSETS[next_idx]
        mapping[source] = (target, dr, dc)
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the 3x3 color ring in the top-left corner and grow each seed in the
    direction encoded by its clockwise successor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping = _legend_mapping(g)

    seeds: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if r < 3 and c < 3:
                continue
            color = int(g[r, c])
            if color in mapping:
                seeds.append((r, c, color))

    out = g.copy()
    occupied = g != BG
    tips = [(r, c) for r, c, _color in seeds]
    targets = [mapping[color][0] for _r, _c, color in seeds]
    directions = [mapping[color][1:] for _r, _c, color in seeds]
    active = [True] * len(seeds)

    while any(active):
        proposals: dict[tuple[int, int], list[int]] = {}
        for idx, is_active in enumerate(active):
            if not is_active:
                continue
            r, c = tips[idx]
            dr, dc = directions[idx]
            nr, nc = r + dr, c + dc
            if not _in_bounds(h, w, nr, nc) or occupied[nr, nc]:
                active[idx] = False
                continue
            proposals.setdefault((nr, nc), []).append(idx)

        if not proposals:
            break

        grew = [False] * len(seeds)
        for (nr, nc), owners in proposals.items():
            if len(owners) != 1:
                for idx in owners:
                    active[idx] = False
                continue
            idx = owners[0]
            out[nr, nc] = targets[idx]
            occupied[nr, nc] = True
            tips[idx] = (nr, nc)
            grew[idx] = True

        if not any(grew):
            break

        for idx, is_active in enumerate(active):
            if is_active and not grew[idx]:
                active[idx] = False

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                out[nr, nc] = True
    return out


def _is_input_isolated(input_occ: np.ndarray, r: int, c: int) -> bool:
    h, w = input_occ.shape
    for nr in range(max(0, r - 1), min(h, r + 2)):
        for nc in range(max(0, c - 1), min(w, c + 2)):
            if (nr, nc) != (r, c) and input_occ[nr, nc]:
                return False
    return True


def _trace_clear_cells(
    h: int,
    w: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    future_occ: np.ndarray,
    buffer_occ: np.ndarray,
) -> tuple[list[tuple[int, int]], str]:
    cells: list[tuple[int, int]] = []
    cr, cc = r, c
    while True:
        nr, nc = cr + dr, cc + dc
        if not _in_bounds(h, w, nr, nc):
            return cells, "edge"
        if future_occ[nr, nc] or buffer_occ[nr, nc]:
            return cells, "occupied"
        cells.append((nr, nc))
        cr, cc = nr, nc


def _candidate_plans(
    h: int,
    w: int,
    dr: int,
    dc: int,
    stop_type: str,
    future_occ: np.ndarray,
    buffer_occ: np.ndarray,
    input_occ: np.ndarray,
) -> list[tuple[tuple[int, int], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]]:
    plans = []
    for r in range(h):
        for c in range(w):
            if future_occ[r, c] or buffer_occ[r, c]:
                continue
            if not _is_input_isolated(input_occ, r, c):
                continue

            clear_cells, reason = _trace_clear_cells(h, w, r, c, dr, dc, future_occ, buffer_occ)
            if stop_type == "edge":
                if reason != "edge":
                    continue
                if not (MIN_BRANCH_LEN <= len(clear_cells) <= MAX_BRANCH_LEN):
                    continue
                plans.append(((r, c), tuple(clear_cells), ()))
                continue

            max_len = min(len(clear_cells) - 1, MAX_BRANCH_LEN)
            if max_len < MIN_BRANCH_LEN:
                continue
            for length in range(MIN_BRANCH_LEN, max_len + 1):
                branch = tuple(clear_cells[:length])
                blocker = (clear_cells[length],)
                plans.append(((r, c), branch, blocker))
    return plans


def _add_extra_obstacle(
    grid: np.ndarray,
    future_occ: np.ndarray,
    input_occ: np.ndarray,
    buffer_occ: np.ndarray,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for _ in range(80):
        rect_h = rng.randint(1, 3)
        rect_w = rng.randint(2, 4)
        if rect_h > h or rect_w > w:
            continue
        r0 = rng.randint(0, h - rect_h)
        c0 = rng.randint(0, w - rect_w)
        if np.any(buffer_occ[r0 : r0 + rect_h, c0 : c0 + rect_w]):
            continue
        grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = OBSTACLE
        future_occ[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        input_occ[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        return True
    return False


def _place_legend(grid: np.ndarray, ring: list[int]) -> None:
    for (r, c), color in zip(RING_COORDS, ring):
        grid[r, c] = color


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    ring = rng.sample(list(RING_COLORS), len(RING_COLORS))
    _place_legend(grid, ring)
    mapping = _legend_mapping(grid)

    future_occ = grid != BG
    input_occ = future_occ.copy()
    buffer_occ = _dilate(future_occ)
    if h > 3 and w > 3:
        buffer_occ[:4, :4] = True

    seed_count = rng.randint(*SEED_COUNT_RANGE)
    seed_colors = rng.sample(ring, seed_count)
    stop_types = ["edge"] * (seed_count // 2) + ["obstacle"] * (seed_count - seed_count // 2)
    rng.shuffle(stop_types)

    remaining = list(zip(seed_colors, stop_types))
    planned_paths: list[tuple[tuple[tuple[int, int], ...], int]] = []
    obstacle_cells = 0

    while remaining:
        candidate_sets = []
        for idx, (seed_color, stop_type) in enumerate(remaining):
            _target_color, dr, dc = mapping[seed_color]
            plans = _candidate_plans(h, w, dr, dc, stop_type, future_occ, buffer_occ, input_occ)
            if not plans:
                return None
            candidate_sets.append((len(plans), idx, plans))

        _count, chosen_idx, plans = min(candidate_sets, key=lambda item: item[0])
        seed_color, stop_type = remaining.pop(chosen_idx)
        branch_color, _dr, _dc = mapping[seed_color]
        seed_pos, branch_cells, blockers = rng.choice(plans)

        sr, sc = seed_pos
        grid[sr, sc] = seed_color
        future_occ[sr, sc] = True
        input_occ[sr, sc] = True

        for br, bc in branch_cells:
            future_occ[br, bc] = True
        planned_paths.append((branch_cells, branch_color))

        for orow, ocol in blockers:
            grid[orow, ocol] = OBSTACLE
            future_occ[orow, ocol] = True
            input_occ[orow, ocol] = True
            obstacle_cells += 1

        buffer_occ = _dilate(future_occ)

    extra_rects = rng.randint(0, 2)
    for _ in range(extra_rects):
        if _add_extra_obstacle(grid, future_occ, input_occ, buffer_occ, rng):
            obstacle_cells += 1
            buffer_occ = _dilate(future_occ)

    if obstacle_cells == 0:
        if not _add_extra_obstacle(grid, future_occ, input_occ, buffer_occ, rng):
            return None

    out = main(grid)
    changed = out != grid
    expected = np.zeros_like(changed, dtype=bool)
    for cells, color in planned_paths:
        for r, c in cells:
            if int(out[r, c]) != color:
                return None
            expected[r, c] = True
    if not np.array_equal(changed, expected):
        return None

    seed_cells = []
    for r in range(h):
        for c in range(w):
            if r < 3 and c < 3:
                continue
            if int(grid[r, c]) in mapping:
                seed_cells.append((r, c))
                if not _is_input_isolated(input_occ & (grid != OBSTACLE), r, c):
                    return None
    if len(seed_cells) != seed_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
