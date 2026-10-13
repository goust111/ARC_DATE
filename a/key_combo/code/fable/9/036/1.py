import random

import numpy as np


BACKGROUND = 0
MOVE_SWITCH = 5
KEEP_SWITCH = 6
DATA_COLORS = (1, 2, 3, 4, 7, 8, 9)

MIN_SIZE = 7
MAX_SIZE = 12
GENERATION_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _diag_run_count(diag: np.ndarray) -> int:
    changes = int(np.count_nonzero(diag[1:] != diag[:-1]))
    return changes + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the main-diagonal switch colors to decide which rows transpose into their matching columns.

    Input structure:
    - The input is a square grid with a dominant background.
    - Every main-diagonal cell is filled, using only the two dedicated switch colors: gray and magenta.
    - Those two switch colors appear only on the main diagonal; all other foreground cells lie off the diagonal.

    Rule:
    - Keep the main diagonal unchanged.
    - For each row whose diagonal cell is gray, move every off-diagonal foreground cell from (r, c) to (c, r), then clear its original position.
    - For each row whose diagonal cell is magenta, leave its off-diagonal foreground cells in place.

    Key insight:
    - The diagonal is not a mirror axis for the whole picture: it is a row-by-row control strip, so the output is a mixture of untouched rows and transposed rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "expected a square grid"
    assert 1 <= h <= 30, "grid size must satisfy ARC bounds"

    bg = _most_common_color(g)
    diag = np.diag(g)
    diag_colors = set(map(int, np.unique(diag)))
    assert diag_colors == {MOVE_SWITCH, KEEP_SWITCH}, "diagonal must use both switch colors"

    off_diag = g.copy()
    idx = np.arange(h)
    off_diag[idx, idx] = bg
    assert not np.any(off_diag == MOVE_SWITCH), "move-switch color must stay on the diagonal"
    assert not np.any(off_diag == KEEP_SWITCH), "keep-switch color must stay on the diagonal"

    out = g.copy()
    moved_cells: list[tuple[int, int, int]] = []
    for r in range(h):
        if int(diag[r]) == MOVE_SWITCH:
            cols = np.flatnonzero((g[r] != bg) & (np.arange(h) != r))
            for c in cols:
                moved_cells.append((r, int(c), int(g[r, c])))
            if cols.size:
                out[r, cols] = bg
        else:
            assert int(diag[r]) == KEEP_SWITCH

    for r, c, color in moved_cells:
        assert out[c, r] == bg, "moved pixels must land on empty cells"
        out[c, r] = color

    return out


def _sample_partition(n: int) -> tuple[list[int], list[int]] | None:
    move_count = random.randint(2, n - 3)
    move_rows = sorted(random.sample(range(n), move_count))
    keep_rows = [i for i in range(n) if i not in move_rows]
    if len(keep_rows) < 3:
        return None
    diag = np.array([MOVE_SWITCH if i in move_rows else KEEP_SWITCH for i in range(n)], dtype=int)
    if _diag_run_count(diag) < 3:
        return None
    return move_rows, keep_rows


def _grow_object(allowed: np.ndarray, blocked: np.ndarray, max_size: int) -> list[tuple[int, int]] | None:
    free = np.argwhere(allowed & (~blocked))
    if free.size == 0:
        return None

    seed_r, seed_c = map(int, free[random.randrange(len(free))])
    target_size = random.randint(1, max_size)
    cells = {(seed_r, seed_c)}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for rr, cc in _neighbors4(allowed.shape, r, c):
            if allowed[rr, cc] and not blocked[rr, cc] and (rr, cc) not in cells:
                frontier.add((rr, cc))

    add_frontier(seed_r, seed_c)
    while len(cells) < target_size and frontier:
        ranked = sorted(
            frontier,
            key=lambda rc: sum((nr, nc) in cells for nr, nc in _neighbors4(allowed.shape, rc[0], rc[1])),
            reverse=True,
        )
        top_k = min(len(ranked), 3)
        chosen = ranked[random.randrange(top_k)]
        frontier.remove(chosen)
        cells.add(chosen)
        add_frontier(*chosen)

    return sorted(cells)


def _reserve_neighborhood(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True
        for rr, cc in _neighbors4(mask.shape, r, c):
            mask[rr, cc] = True


def _paint_zone(
    grid: np.ndarray,
    allowed: np.ndarray,
    min_cells: int,
    max_cells: int,
    palette: list[int],
) -> int:
    if int(np.count_nonzero(allowed)) < min_cells:
        return -1

    target_total = random.randint(min_cells, min(max_cells, int(np.count_nonzero(allowed))))
    placed = 0
    blocked = np.zeros_like(allowed, dtype=bool)
    attempts = 0
    while placed < target_total and attempts < 80:
        attempts += 1
        remaining = target_total - placed
        max_size = min(4, remaining)
        cells = _grow_object(allowed, blocked, max_size)
        if cells is None:
            break
        color = random.choice(palette)
        for r, c in cells:
            grid[r, c] = color
        placed += len(cells)
        _reserve_neighborhood(blocked, cells)
    return placed


def _build_output(n: int, move_rows: list[int], keep_rows: list[int]) -> np.ndarray | None:
    out = np.full((n, n), BACKGROUND, dtype=int)
    for i in move_rows:
        out[i, i] = MOVE_SWITCH
    for i in keep_rows:
        out[i, i] = KEEP_SWITCH

    keep_zone = np.zeros((n, n), dtype=bool)
    move_zone = np.zeros((n, n), dtype=bool)
    keep_set = set(keep_rows)
    move_set = set(move_rows)

    for r in keep_rows:
        for c in keep_rows:
            if r != c:
                keep_zone[r, c] = True
        for c in move_rows:
            move_zone[r, c] = True

    palette = random.sample(list(DATA_COLORS), random.randint(3, min(5, len(DATA_COLORS))))
    keep_cells = _paint_zone(out, keep_zone, min_cells=2, max_cells=max(4, len(keep_rows) + 1), palette=palette)
    if keep_cells < 2:
        return None

    moved_cells = _paint_zone(out, move_zone, min_cells=2, max_cells=max(4, len(move_rows) + 2), palette=palette)
    if moved_cells < 2:
        return None

    if int(np.count_nonzero(out == BACKGROUND)) <= n * n // 2:
        return None

    # Reject overly simple keep-zone symmetry that could mimic a full transpose.
    asym_keep = any(out[r, c] != BACKGROUND and out[c, r] == BACKGROUND for r in keep_rows for c in keep_rows if r != c)
    if not asym_keep:
        return None

    # Prevent accidental use of switch colors outside the diagonal.
    off_diag = out.copy()
    idx = np.arange(n)
    off_diag[idx, idx] = BACKGROUND
    if np.any((off_diag == MOVE_SWITCH) | (off_diag == KEEP_SWITCH)):
        return None

    # Output family is intentionally restricted: all off-diagonal output cells live in keep rows.
    for r in move_rows:
        row = off_diag[r]
        if np.any(row != BACKGROUND):
            return None
    for c in keep_rows:
        for r in move_rows:
            if off_diag[r, c] != BACKGROUND:
                return None
    return out


def _derive_input_from_output(out: np.ndarray) -> np.ndarray:
    n = out.shape[0]
    inp = out.copy()
    diag = np.diag(out)
    move_rows = [i for i in range(n) if int(diag[i]) == MOVE_SWITCH]
    keep_rows = [i for i in range(n) if int(diag[i]) == KEEP_SWITCH]

    for r in keep_rows:
        for c in move_rows:
            color = int(out[r, c])
            if color == BACKGROUND:
                continue
            inp[r, c] = BACKGROUND
            inp[c, r] = color
    return inp


def _invert_generated_family(out: np.ndarray) -> np.ndarray:
    return _derive_input_from_output(out)


def _validate_generated(inp: np.ndarray) -> bool:
    n = inp.shape[0]
    diag = np.diag(inp)
    move_rows = [i for i in range(n) if int(diag[i]) == MOVE_SWITCH]
    keep_rows = [i for i in range(n) if int(diag[i]) == KEEP_SWITCH]
    keep_set = set(keep_rows)
    move_set = set(move_rows)

    if not move_rows or not keep_rows:
        return False
    if set(map(int, np.unique(diag))) != {MOVE_SWITCH, KEEP_SWITCH}:
        return False
    if _diag_run_count(diag) < 3:
        return False

    off_diag = inp.copy()
    idx = np.arange(n)
    off_diag[idx, idx] = BACKGROUND
    if np.any((off_diag == MOVE_SWITCH) | (off_diag == KEEP_SWITCH)):
        return False

    # Injective generator family: every input off-diagonal cell sits in a keep-column.
    coords = np.argwhere(off_diag != BACKGROUND)
    if coords.size == 0:
        return False
    if any(int(c) not in keep_set for _, c in coords):
        return False
    if not any(int(r) in move_set for r, _ in coords):
        return False
    if not any(int(r) in keep_set for r, _ in coords):
        return False

    try:
        out = main(inp)
    except AssertionError:
        return False

    if np.array_equal(out, inp):
        return False
    if np.array_equal(out, inp.T):
        return False

    out_off = out.copy()
    out_off[idx, idx] = BACKGROUND
    if any(np.any(out_off[r] != BACKGROUND) for r in move_rows):
        return False

    # The restricted family must be perfectly invertible from the output.
    reconstructed = _invert_generated_family(out)
    if not np.array_equal(reconstructed, inp):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    n = random.randint(MIN_SIZE, MAX_SIZE)
    sampled = _sample_partition(n)
    if sampled is None:
        return None
    move_rows, keep_rows = sampled

    out = _build_output(n, move_rows, keep_rows)
    if out is None:
        return None
    inp = _derive_input_from_output(out)
    if not _validate_generated(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-switch transpose input")
