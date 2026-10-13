from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (18, 28)
MAX_GENERATION_ATTEMPTS = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _find_single_seed(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == BLUE)
    assert len(coords) == 1, "expected exactly one blue seed"
    return int(coords[0, 0]), int(coords[0, 1])


def _marker_set(grid: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(grid == GREY)}


def _first_pass(grid: np.ndarray, seed: tuple[int, int]) -> tuple[np.ndarray, set[tuple[int, int]]]:
    seen = np.zeros(grid.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque([seed])
    seen[seed] = True
    reachable_markers: set[tuple[int, int]] = set()

    while q:
        r, c = q.popleft()
        if grid[r, c] == GREY:
            reachable_markers.add((r, c))
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc) or seen[nr, nc] or grid[nr, nc] == RED:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))

    return seen, reachable_markers


def _drop_gates(grid: np.ndarray, active_markers: set[tuple[int, int]]) -> np.ndarray:
    """
    Draw all gate rays independently against the original red walls.

    This keeps the operation order-independent even when multiple reachable
    markers are present.
    """
    out = np.array(grid, copy=True)
    original_walls = grid == RED
    h, w = grid.shape

    for mr, mc in sorted(active_markers):
        for dr, dc in DIRS4:
            r, c = mr + dr, mc + dc
            while 0 <= r < h and 0 <= c < w and not original_walls[r, c]:
                if out[r, c] == BLACK:
                    out[r, c] = RED
                r += dr
                c += dc

    return out


def _second_pass(grid: np.ndarray, seed: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    seen = np.zeros(grid.shape, dtype=bool)
    seen[seed] = True
    out = np.array(grid, copy=True)
    q: deque[tuple[int, int]] = deque([seed])

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc) or seen[nr, nc] or grid[nr, nc] != BLACK:
                continue
            seen[nr, nc] = True
            out[nr, nc] = BLUE
            q.append((nr, nc))

    return seen, out


def _solve_details(
    grid: np.ndarray,
    *,
    forced_active_markers: set[tuple[int, int]] | None = None,
) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BLACK, BLUE, RED, GREY})

    seed = _find_single_seed(g)
    initial_seen, initial_active = _first_pass(g, seed)
    active_markers = initial_active if forced_active_markers is None else set(forced_active_markers)
    gated = _drop_gates(g, active_markers)
    final_seen, out = _second_pass(gated, seed)
    return {
        "seed": seed,
        "initial_seen": initial_seen,
        "initial_active": initial_active,
        "used_active": active_markers,
        "gated": gated,
        "final_seen": final_seen,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let only the initially reachable gray markers drop red cross-gates, then refill the remaining black water region from the blue seed.

    Input structure:
    - Black background cells are split into rooms by red wall pixels, with one-cell or two-cell door gaps left in some walls.
    - Exactly one blue seed pixel lies in an open region.
    - Several gray marker pixels are sparsely scattered in open cells and are not orthogonally adjacent to each other.

    Rule:
    - First, ignore any falling gates and 4-neighbor flood from the blue seed through non-wall cells; gray markers count as passable in this first pass.
    - Every gray marker reached by that initial flood extends a red line up, down, left, and right until a red wall or the grid edge, while the marker cell itself stays gray.
    - Then, treating the original walls plus all new red gate lines as walls, flood again from the original blue seed through black cells only; recolor the black cells reached in this second pass to blue and leave every other cell unchanged.

    Key insight:
    - The same gray marker can matter twice in opposite ways: it drops only if the seed could reach it before any gates exist, but the gate it drops can later cut that route and leave its own former neighborhood dry.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _sample_nonadjacent(
    candidates: list[tuple[int, int]],
    count: int,
    *,
    forbid: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    forbid = set() if forbid is None else set(forbid)
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if cell in forbid:
            continue
        if any(abs(cell[0] - r) + abs(cell[1] - c) == 1 for r, c in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _rect_cells(r0: int, r1: int, c0: int, c1: int, *, exclude_cols: set[int] | None = None) -> list[tuple[int, int]]:
    exclude_cols = set() if exclude_cols is None else set(exclude_cols)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if c not in exclude_cols]


def _one_pass_no_gates(grid: np.ndarray) -> np.ndarray:
    seed = _find_single_seed(grid)
    seen = np.zeros(grid.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque([seed])
    seen[seed] = True
    out = np.array(grid, copy=True)
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc) or seen[nr, nc] or grid[nr, nc] != BLACK:
                continue
            seen[nr, nc] = True
            out[nr, nc] = BLUE
            q.append((nr, nc))
    return out


def _validate_instance(grid: np.ndarray, roles: dict) -> bool:
    a_marker = roles["a_marker"]
    b_marker = roles["b_marker"]
    c_marker = roles["c_marker"]
    all_markers = {a_marker, b_marker, c_marker}

    details = _solve_details(grid)
    out = details["output"]

    if details["initial_active"] != {a_marker, b_marker}:
        return False
    if c_marker in details["initial_active"]:
        return False

    # The dry reachable marker in the right wing forces a true first pass.
    if any(
        in_bounds(out.shape, nr, nc) and out[nr, nc] == BLUE
        for nr, nc in neighbors4(*b_marker)
    ):
        return False

    # Water must genuinely lose territory after the gate drop.
    if np.count_nonzero(out == BLUE) >= np.count_nonzero(details["initial_seen"]):
        return False

    # Wrong shortcuts must disagree with the intended output.
    if np.array_equal(out, _one_pass_no_gates(grid)):
        return False
    if np.array_equal(out, _solve_details(grid, forced_active_markers={a_marker})["output"]):
        return False
    if np.array_equal(out, _solve_details(grid, forced_active_markers=all_markers)["output"]):
        return False

    # The gate line from A must dry out one side of the seed room.
    a_row = a_marker[0]
    hr = roles["left_split_row"]
    vc = roles["vertical_wall_col"]
    seed_r = roles["seed"][0]
    if seed_r < a_row:
        dry_slice = out[a_row + 1 : out.shape[0] - 1, 1:vc]
    else:
        dry_slice = out[hr + 1 : a_row, 1:vc]
    if dry_slice.size == 0 or not np.any(dry_slice == BLACK):
        return False

    return True


def _generate_attempt() -> tuple[np.ndarray, dict] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    vc = random.randint(5, w - 6)
    hr = random.randint(3, h - 10)
    rr = random.randint(hr + 3, h - 6)
    a_row = random.randint(rr + 2, h - 4)
    gap2_w = random.randint(1, 2)
    gap2_c0 = random.randint(vc + 2, w - 2 - gap2_w)
    gap2_cols = set(range(gap2_c0, gap2_c0 + gap2_w))

    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = RED
    grid[-1, :] = RED
    grid[:, 0] = RED
    grid[:, -1] = RED
    grid[1 : h - 1, vc] = RED
    grid[a_row, vc] = BLACK
    grid[hr, 1:vc] = RED
    grid[rr, vc + 1 : w - 1] = RED
    for c in gap2_cols:
        grid[rr, c] = BLACK

    a_col_choices = list(range(2, vc - 1))
    if not a_col_choices:
        return None
    a_col = random.choice(a_col_choices)
    a_marker = (a_row, a_col)
    grid[a_marker] = GREY

    top_right_cells = _rect_cells(1, rr - 1, vc + 1, w - 2, exclude_cols=gap2_cols)
    b_choice = _sample_nonadjacent(top_right_cells, 1, forbid={a_marker})
    if b_choice is None:
        return None
    b_marker = b_choice[0]
    grid[b_marker] = GREY

    top_left_cells = _rect_cells(1, hr - 1, 1, vc - 1)
    c_choice = _sample_nonadjacent(top_left_cells, 1, forbid={a_marker, b_marker})
    if c_choice is None:
        return None
    c_marker = c_choice[0]
    grid[c_marker] = GREY

    seed_side = random.choice(("upper", "lower"))
    if seed_side == "upper":
        seed_rows = list(range(hr + 1, a_row))
    else:
        seed_rows = list(range(a_row + 1, h - 1))
    seed_cells = [(r, c) for r in seed_rows for c in range(1, vc) if grid[r, c] == BLACK]
    random.shuffle(seed_cells)
    seed = None
    for cand in seed_cells:
        if any(in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK for nr, nc in neighbors4(*cand)):
            seed = cand
            break
    if seed is None:
        return None
    grid[seed] = BLUE

    roles = {
        "a_marker": a_marker,
        "b_marker": b_marker,
        "c_marker": c_marker,
        "seed": seed,
        "left_split_row": hr,
        "right_split_row": rr,
        "vertical_wall_col": vc,
    }
    return grid, roles


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _generate_attempt()
        if built is None:
            continue
        grid, roles = built
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        if _validate_instance(grid, roles):
            return grid
    raise RuntimeError("failed to generate a valid gate-drop puzzle input")
