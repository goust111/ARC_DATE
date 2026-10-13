import itertools

import numpy as np


BLACK = 0
RED = 2
GREY = 5
GRID_SIZE = 5
OUTPUT_SIZE = GRID_SIZE * GRID_SIZE
MOTIF_COLORS = np.array([1, 3, 4, 6, 7, 8, 9], dtype=int)
RNG = np.random.default_rng()
GEN_RETRY_LIMIT = 600


def _trace_path(grid: np.ndarray, start: tuple[int, int]) -> dict:
    """Trace one control path using only the puzzle's upward-then-right rule."""
    h, w = grid.shape
    r, c = start
    path = [(r, c)]
    up_moves = 0
    grey_turns = []
    top_rights = 0

    while True:
        nr = r - 1
        if nr >= 0 and grid[nr, c] != GREY:
            r = nr
            path.append((r, c))
            up_moves += 1
            continue

        nc = c + 1
        if nc < w and grid[r, nc] != GREY:
            if nr >= 0 and grid[nr, c] == GREY:
                grey_turns.append((nr, c))
            else:
                top_rights += 1
            c = nc
            path.append((r, c))
            continue

        stop_reason = "grey" if nr >= 0 and grid[nr, c] == GREY else "top"
        return {
            "start": start,
            "path": path,
            "up_moves": up_moves,
            "grey_turns": grey_turns,
            "top_rights": top_rights,
            "stop_reason": stop_reason,
        }


def _collect_mask(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Return the union stamp mask and per-control path metadata."""
    h, w = grid.shape
    starts = [(h - 1, int(c)) for c in np.flatnonzero(grid[h - 1] == RED)]
    info = [_trace_path(grid, start) for start in starts]
    mask = np.zeros((h, w), dtype=bool)
    for item in info:
        rows, cols = zip(*item["path"])
        mask[list(rows), list(cols)] = True
    return mask, info


def _stamp_output(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for r, c in zip(*np.where(mask)):
        out[r * GRID_SIZE : (r + 1) * GRID_SIZE, c * GRID_SIZE : (c + 1) * GRID_SIZE] = grid
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow paths upward from bottom red controls and use the visited cells as recursive stamp slots.

    Input structure:
    - The input is a 5x5 grid.
    - The bottom row contains 1 to 3 red control pixels, and every other bottom-row cell is black.
    - Grey pixels are obstacles above the bottom row; other non-black, non-red, non-grey pixels form a small motif.

    Rule:
    - From each red control pixel, repeatedly try to move one cell upward; if that cell is in bounds and not grey, visit it.
    - If the cell above is grey or out of bounds, try moving one cell right instead; if that right cell is in bounds and not grey, visit it and continue from there, otherwise stop that path.
    - Collect every visited cell from every path, then create a 25x25 output where each visited input cell selects one 5x5 block that receives an exact copy of the full input grid; all other blocks stay black.

    Key insight:
    - The path geometry is not drawn directly in the output: it becomes a macro-scale stamp mask, while the same grey obstacles both steer the paths and reappear inside every stamped copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 5x5 grid"
    assert np.all(np.isin(g, np.arange(10))), "expected ARC color ids"
    bottom = g[-1]
    starts = np.flatnonzero(bottom == RED)
    assert 1 <= len(starts) <= 3, "expected 1 to 3 bottom controls"
    assert np.all(np.isin(bottom, [BLACK, RED])), "bottom row must contain only black and red"
    assert not np.any(g[-1] == GREY), "grey obstacles cannot be on the bottom row"

    mask, _info = _collect_mask(g)
    return _stamp_output(g, mask)


def _random_starts() -> tuple[int, ...]:
    choices = []
    for count in (1, 2, 3):
        for combo in itertools.combinations(range(GRID_SIZE), count):
            if any(c < GRID_SIZE - 1 for c in combo):
                choices.append(combo)
    weights = np.array([2 if len(combo) == 1 else 4 if len(combo) == 2 else 3 for combo in choices], dtype=float)
    weights /= weights.sum()
    idx = int(RNG.choice(len(choices), p=weights))
    return choices[idx]


def _build_base_grid(starts: tuple[int, ...], obstacles: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for r, c in obstacles:
        grid[r, c] = GREY
    grid[GRID_SIZE - 1, list(starts)] = RED
    return grid


def _sample_base_candidate() -> np.ndarray | None:
    starts = _random_starts()
    focus_choices = [c for c in starts if c < GRID_SIZE - 1]
    if not focus_choices:
        return None
    focus_col = int(RNG.choice(focus_choices))
    blocker_row = int(RNG.choice([0, 1]))
    forced_blocker = (blocker_row, focus_col)
    forced_target = (blocker_row + 1, focus_col + 1)

    protected = {(3, c) for c in starts} | {(2, c) for c in starts}
    if forced_blocker in protected or forced_target in protected:
        return None

    eligible = [
        (r, c)
        for r in range(GRID_SIZE - 1)
        for c in range(GRID_SIZE)
        if (r, c) not in protected
        and (r, c) != forced_blocker
        and (r, c) != forced_target
    ]
    extra_count = int(RNG.integers(1, 5))
    if extra_count > len(eligible):
        return None

    sampled = set(map(tuple, RNG.choice(np.array(eligible), size=extra_count, replace=False))) if extra_count else set()
    obstacles = sampled | {forced_blocker}
    return _build_base_grid(starts, obstacles)


def _paths_are_disjoint(info: list[dict]) -> bool:
    seen = set()
    for item in info:
        for cell in item["path"]:
            if cell in seen:
                return False
            seen.add(cell)
    return True


def _decorate_motif(base: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    grid = base.copy()
    free = {(r, c) for r in range(GRID_SIZE - 1) for c in range(GRID_SIZE) if grid[r, c] == BLACK}
    visited = sorted(cell for cell in free if mask[cell])
    unvisited = sorted(cell for cell in free if not mask[cell])
    if len(visited) < 2 or len(unvisited) < 2:
        return None

    color_count = int(RNG.integers(2, 4))
    colors = RNG.choice(MOTIF_COLORS, size=color_count, replace=False)

    max_v = min(len(visited) - 1, 3)
    max_u = min(len(unvisited) - 1, 4)
    if max_v < 1 or max_u < 2:
        return None

    visit_n = int(RNG.integers(1, max_v + 1))
    unvisit_n = int(RNG.integers(2, max_u + 1))
    motif_cells = list(visited[:])
    RNG.shuffle(motif_cells)
    chosen = motif_cells[:visit_n]
    unvisited_cells = list(unvisited[:])
    RNG.shuffle(unvisited_cells)
    chosen.extend(unvisited_cells[:unvisit_n])
    RNG.shuffle(chosen)

    for idx, (r, c) in enumerate(chosen):
        grid[r, c] = int(colors[idx % len(colors)])
    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    if grid.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if not np.all(np.isin(grid[-1], [BLACK, RED])):
        return False
    if np.any(grid[-1] == GREY):
        return False

    mask, info = _collect_mask(grid)
    starts = [tuple(start) for start in np.argwhere(grid[-1] == RED)]
    if not (1 <= len(starts) <= 3):
        return False
    if not all(item["up_moves"] >= 2 for item in info):
        return False
    if sum(len(item["grey_turns"]) for item in info) < 1:
        return False
    if not _paths_are_disjoint(info):
        return False
    if sum(len(item["path"]) for item in info) != int(mask.sum()):
        return False
    if int(mask.sum()) < 5 or int(mask.sum()) > 14:
        return False

    blockers_used = [cell for item in info for cell in item["grey_turns"]]
    all_obstacles = list(map(tuple, np.argwhere(grid == GREY)))
    if not blockers_used or not all_obstacles:
        return False

    motif_mask = (grid != BLACK) & (grid != RED) & (grid != GREY)
    if len(np.unique(grid[motif_mask])) < 2:
        return False
    if not np.any(mask & motif_mask):
        return False
    if not np.any((~mask) & motif_mask):
        return False
    if not np.any(mask & (grid == BLACK)):
        return False
    if not np.any((~mask) & (grid == BLACK)):
        return False

    if np.array_equal(mask, grid == RED):
        return False
    if np.array_equal(mask, grid == GREY):
        return False
    if np.array_equal(mask, grid != BLACK):
        return False

    out = main(grid)
    expected = _stamp_output(grid, mask)
    if not np.array_equal(out, expected):
        return False

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            block = out[r * GRID_SIZE : (r + 1) * GRID_SIZE, c * GRID_SIZE : (c + 1) * GRID_SIZE]
            if mask[r, c]:
                if not np.array_equal(block, grid):
                    return False
            elif np.any(block != BLACK):
                return False

    used_blocker = blockers_used[0]
    mutated = grid.copy()
    mutated[used_blocker] = BLACK
    mutated_mask, _ = _collect_mask(mutated)
    if np.array_equal(mask, mutated_mask):
        return False
    if np.array_equal(main(mutated), out):
        return False

    unvisited_motif = tuple(np.argwhere((~mask) & motif_mask)[0])
    recolored = grid.copy()
    recolored[unvisited_motif] = BLACK
    recolored_mask, _ = _collect_mask(recolored)
    if not np.array_equal(mask, recolored_mask):
        return False
    if np.array_equal(main(recolored), out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    base = _sample_base_candidate()
    if base is None:
        return None

    mask, info = _collect_mask(base)
    if not all(item["up_moves"] >= 2 for item in info):
        return None
    if sum(len(item["grey_turns"]) for item in info) < 1:
        return None
    if not _paths_are_disjoint(info):
        return None

    grid = _decorate_motif(base, mask)
    if grid is None:
        return None
    return grid if _validate_candidate(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recursive-stamp path puzzle")
