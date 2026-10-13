import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    idx = DIRS.index(direction)
    return DIRS[(idx - 1) % 4]


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    idx = DIRS.index(direction)
    return DIRS[(idx + 1) % 4]


def _perimeter_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = [(0, c) for c in range(w)]
    cells += [(r, w - 1) for r in range(1, h - 1)]
    cells += [(h - 1, c) for c in range(w - 1, -1, -1)]
    cells += [(r, 0) for r in range(h - 2, 0, -1)]
    return cells


def _find_gap_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    gaps = [(r, c) for r, c in _perimeter_cells(grid.shape) if int(grid[r, c]) != GREY]
    assert len(gaps) == 1, "expected exactly one wall gap"
    gap = gaps[0]
    r, c = gap
    assert gap not in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}, "gap cannot be a corner"
    assert int(grid[r, c]) == BLACK, "the unique entry gap must be black"
    if r == 0:
        direction = (1, 0)
        start = (1, c)
    elif r == h - 1:
        direction = (-1, 0)
        start = (h - 2, c)
    elif c == 0:
        direction = (0, 1)
        start = (r, 1)
    else:
        direction = (0, -1)
        start = (r, w - 2)
    return gap, start, direction


def _simulate(
    grid: np.ndarray,
    *,
    consume_turners: bool = True,
    swap_turn_meaning: bool = False,
    turn_after_enter: bool = False,
) -> dict:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    assert g.ndim == 2, "input must be a 2D grid"
    assert 5 <= h <= 30 and 5 <= w <= 30, "grid size must stay within ARC bounds"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, YELLOW, GREY}, "unexpected colors"
    assert not np.any(g[1:-1, 1:-1] == GREY), "grey is reserved for the outer wall"

    gap, start, direction = _find_gap_and_direction(g)
    sr, sc = start
    assert int(g[sr, sc]) not in {RED, BLUE}, "the first interior cell must be walkable"

    active = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere((g == RED) | (g == BLUE))
    }
    out = g.copy()
    r, c = start
    path = [start]
    consumed = []
    seen = set()

    while True:
        state = (r, c, direction, tuple(sorted(active.items())))
        assert state not in seen, "ball should eventually breach the wall instead of looping forever"
        seen.add(state)

        out[r, c] = YELLOW
        dr, dc = direction
        nr, nc = r + dr, c + dc
        assert 0 <= nr < h and 0 <= nc < w, "valid puzzles keep the ball inside the framed grid until breach"

        if (nr, nc) in active:
            color = active[(nr, nc)]
            if turn_after_enter:
                if consume_turners:
                    del active[(nr, nc)]
                out[nr, nc] = YELLOW
                r, c = nr, nc
                path.append((r, c))
            else:
                if consume_turners:
                    del active[(nr, nc)]
                    out[nr, nc] = BLACK
                else:
                    out[nr, nc] = color

            right_on_red = not swap_turn_meaning
            if (color == RED and right_on_red) or (color == BLUE and not right_on_red):
                direction = _turn_right(direction)
            else:
                direction = _turn_left(direction)
            consumed.append((nr, nc))
            continue

        if int(out[nr, nc]) == GREY:
            out[nr, nc] = YELLOW
            breach = (nr, nc)
            break

        r, c = nr, nc
        path.append((r, c))

    return {
        "output": out,
        "gap": gap,
        "start": start,
        "path": path,
        "consumed": consumed,
        "remaining": sorted(active),
        "breach": breach,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a marble through a walled chamber, consuming one-shot turn pixels and punching a new wall breach.

    Input structure:
    - A grey perimeter wall frames the whole grid, with exactly one non-corner wall gap as the entry.
    - The interior contains black empty cells plus isolated red and blue one-cell turners.
    - Red and blue cells are the only non-wall interior objects, and the marble starts by entering from the gap along the inward normal.

    Rule:
    - Color every visited interior cell yellow as the marble travels.
    - If the next cell straight ahead is red, the marble turns right in place; if it is blue, it turns left in place.
    - A triggered turner is consumed immediately, disappears to an ordinary walkable cell, and can later be crossed straight through like any yellow or black cell.
    - When the next cell ahead is a grey wall cell, that wall cell is broken and recolored yellow, then the simulation stops; untriggered turners remain unchanged.

    Key insight:
    - Turn pixels are single-use triggers, not permanent arrows: the first approach rotates the marble without entering the trigger cell, while later approaches can pass through that same location after it has vanished.
    """
    return _simulate(grid)["output"]


def _build_base_case(rng: np.random.Generator) -> np.ndarray:
    h = int(rng.integers(8, 19))
    w = int(rng.integers(8, 19))
    g = np.full((h, w), BLACK, dtype=int)
    g[0, :] = GREY
    g[-1, :] = GREY
    g[:, 0] = GREY
    g[:, -1] = GREY

    go_down_first = bool(rng.integers(2))
    c1 = int(rng.integers(2, w - 3))

    if go_down_first:
        r0 = int(rng.integers(2, h - 4))
        r2 = int(rng.integers(r0 + 2, h - 2))
        g[r0, 0] = BLACK
        g[r0, c1 + 1] = RED
        g[r2 + 1, c1] = BLUE
        g[r2, c1 + 2] = BLUE
    else:
        r0 = int(rng.integers(4, h - 2))
        r2 = int(rng.integers(2, r0 - 1))
        g[r0, 0] = BLACK
        g[r0, c1 + 1] = BLUE
        g[r2 - 1, c1] = RED
        g[r2, c1 + 2] = RED

    turns = int(rng.integers(4))
    return np.rot90(g, turns).copy()


def _candidate_decoys(grid: np.ndarray, trace: dict, min_dist: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    important = set(trace["path"]) | set(trace["consumed"]) | {trace["gap"], trace["breach"]}
    cells = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in important:
                continue
            if int(grid[r, c]) != BLACK:
                continue
            if all(abs(r - rr) + abs(c - cc) >= min_dist for rr, cc in important):
                cells.append((r, c))
    return cells


def _place_decoys(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    trace = _simulate(grid)
    g = grid.copy()
    candidates = _candidate_decoys(g, trace, min_dist=2)
    if len(candidates) < 1:
        candidates = _candidate_decoys(g, trace, min_dist=1)
    assert candidates, "need room for at least one untouched turner"

    rng.shuffle(candidates)
    target = int(rng.integers(1, min(6, len(candidates) + 1)))
    chosen = []
    for cell in candidates:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= 2 for rr, cc in chosen):
            chosen.append(cell)
        if len(chosen) == target:
            break
    if not chosen:
        chosen = [candidates[0]]

    for r, c in chosen:
        g[r, c] = int(rng.choice([RED, BLUE]))
    return g


def _wrong_output(grid: np.ndarray, **kwargs) -> np.ndarray | None:
    try:
        return _simulate(grid, **kwargs)["output"]
    except AssertionError:
        return None


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        correct = _simulate(grid)
    except AssertionError:
        return False

    out = correct["output"]
    consumed = correct["consumed"]
    revisited = [cell for cell in consumed if int(out[cell]) == YELLOW]
    if len(consumed) < 3 or not revisited:
        return False
    if not correct["remaining"]:
        return False
    if correct["gap"] == correct["breach"]:
        return False
    if int(out[correct["gap"]]) != BLACK or int(out[correct["breach"]]) != YELLOW:
        return False

    wrong_permanent = _wrong_output(grid, consume_turners=False)
    if wrong_permanent is not None and np.array_equal(wrong_permanent, out):
        return False

    wrong_swapped = _wrong_output(grid, swap_turn_meaning=True)
    if wrong_swapped is not None and np.array_equal(wrong_swapped, out):
        return False

    wrong_enter = _wrong_output(grid, turn_after_enter=True)
    if wrong_enter is not None and np.array_equal(wrong_enter, out):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _build_base_case(rng)
        grid = _place_decoys(grid, rng)
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate puzzle instance")
