from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
MAROON = 9
FILL_PALETTE = np.array([1, 2, 3, 4, 6, 7], dtype=int)
MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _lattice_candidate(grid: np.ndarray, color: int) -> dict[str, object] | None:
    mask = grid == color
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    patch = mask[r0 : r1 + 1, c0 : c1 + 1]
    full_rows = np.where(patch.all(axis=1))[0]
    full_cols = np.where(patch.all(axis=0))[0]
    if full_rows.size < 4 or full_cols.size < 4:
        return None
    row_diffs = np.diff(full_rows)
    col_diffs = np.diff(full_cols)
    if not (np.all(row_diffs == row_diffs[0]) and np.all(col_diffs == col_diffs[0])):
        return None
    step = int(row_diffs[0])
    if step <= 1 or step != int(col_diffs[0]):
        return None
    if full_rows[0] != 0 or full_rows[-1] != patch.shape[0] - 1:
        return None
    if full_cols[0] != 0 or full_cols[-1] != patch.shape[1] - 1:
        return None
    n = int(full_rows.size - 1)
    if n != int(full_cols.size - 1) or n < 2:
        return None
    expected = np.zeros_like(patch, dtype=bool)
    expected[full_rows, :] = True
    expected[:, full_cols] = True
    if not np.array_equal(patch, expected):
        return None
    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "line_rows": full_rows.astype(int),
        "line_cols": full_cols.astype(int),
        "cell_size": step - 1,
        "n": n,
        "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
    }


def _find_lattice(grid: np.ndarray) -> dict[str, object]:
    candidates: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == BLACK:
            continue
        candidate = _lattice_candidate(grid, color)
        if candidate is not None:
            candidates.append(candidate)
    assert candidates, "could not locate the lattice grid"
    candidates.sort(key=lambda item: int(item["area"]), reverse=True)
    return candidates[0]


def _extract_slot_colors(
    board: np.ndarray, line_rows: np.ndarray, line_cols: np.ndarray, lattice_color: int
) -> np.ndarray:
    n = int(line_rows.size - 1)
    slots = np.full((n, n), BLACK, dtype=int)
    for r in range(n):
        for c in range(n):
            patch = board[line_rows[r] + 1 : line_rows[r + 1], line_cols[c] + 1 : line_cols[c + 1]]
            vals = [int(v) for v in np.unique(patch) if int(v) != BLACK]
            assert lattice_color not in vals, "slot interiors must not contain lattice color"
            if not vals:
                assert np.all(patch == BLACK)
                continue
            assert len(vals) == 1 and np.all(patch == vals[0]), "each slot must be monochrome"
            slots[r, c] = vals[0]
    return slots


def _solve_latin_unique(partial: np.ndarray) -> np.ndarray:
    grid = np.array(partial, dtype=int, copy=True)
    n = int(grid.shape[0])
    colors = [int(v) for v in np.unique(grid) if int(v) != BLACK]
    assert len(colors) == n, "valid inputs must expose all Latin-square colors"

    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    blanks: list[tuple[int, int]] = []

    for r in range(n):
        for c in range(n):
            value = int(grid[r, c])
            if value == BLACK:
                blanks.append((r, c))
                continue
            assert value not in row_used[r], "row duplicates violate the Latin constraint"
            assert value not in col_used[c], "column duplicates violate the Latin constraint"
            row_used[r].add(value)
            col_used[c].add(value)

    solutions: list[np.ndarray] = []

    def search() -> None:
        if len(solutions) >= 2:
            return
        best_cell: tuple[int, int] | None = None
        best_domain: list[int] | None = None
        for r, c in blanks:
            if int(grid[r, c]) != BLACK:
                continue
            domain = [color for color in colors if color not in row_used[r] and color not in col_used[c]]
            if not domain:
                return
            if best_domain is None or len(domain) < len(best_domain):
                best_cell = (r, c)
                best_domain = domain
                if len(domain) == 1:
                    break
        if best_cell is None:
            solutions.append(grid.copy())
            return

        r, c = best_cell
        assert best_domain is not None
        for value in best_domain:
            grid[r, c] = value
            row_used[r].add(value)
            col_used[c].add(value)
            search()
            col_used[c].remove(value)
            row_used[r].remove(value)
            grid[r, c] = BLACK

    search()
    assert len(solutions) == 1, "empty slots must have a unique Latin-square completion"
    return solutions[0]


def _find_seed_keep_mask(grid: np.ndarray, board_bbox: tuple[int, int, int, int], cell_size: int) -> np.ndarray:
    r0, c0, r1, c1 = board_bbox
    seed_pixels = np.isin(grid, (AZURE, MAROON))
    seed_pixels[r0 : r1 + 1, c0 : c1 + 1] = False
    box = _bbox(seed_pixels)
    assert box is not None, "could not locate the maroon/azure seed"
    sr0, sc0, sr1, sc1 = box
    assert (sr1 - sr0 + 1, sc1 - sc0 + 1) == (cell_size, cell_size), "seed must match slot size"
    assert sc0 > c1 or sr0 > r1, "seed must sit to the right of or below the lattice"
    seed_patch_mask = seed_pixels[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert np.all(seed_patch_mask), "seed bbox must be fully occupied"
    seed_pixels[sr0 : sr1 + 1, sc0 : sc1 + 1] = False
    assert not seed_pixels.any(), "expected exactly one seed"
    seed = grid[sr0 : sr1 + 1, sc0 : sc1 + 1]
    assert set(map(int, np.unique(seed))) == {AZURE, MAROON}
    return seed == MAROON


def _render_board(slot_colors: np.ndarray, keep_mask: np.ndarray, lattice_color: int) -> np.ndarray:
    n = int(slot_colors.shape[0])
    cell_size = int(keep_mask.shape[0])
    size = n * cell_size + n + 1
    out = np.full((size, size), int(lattice_color), dtype=int)
    for r in range(n):
        for c in range(n):
            rr = 1 + r * (cell_size + 1)
            cc = 1 + c * (cell_size + 1)
            patch = out[rr : rr + cell_size, cc : cc + cell_size]
            patch[:] = BLACK
            patch[keep_mask] = int(slot_colors[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing slot colors from a Latin-square grid, then carve every slot with the maroon/azure seed mask.

    Input structure:
    - The background is black.
    - A single-pixel lattice color forms a square N×N grid of equal slots, with some slot interiors filled by solid colors and some left black.
    - A separate seed square of the same size as one slot sits to the right of or below the lattice and is filled only with maroon and azure.

    Rule:
    - Detect the lattice and read each slot as either one solid non-black color or an empty black slot.
    - Complete the empty slots so that every row and every column uses each observed slot color exactly once.
    - Remove the external seed by cropping to the lattice, then repaint every slot interior: keep the slot color where the seed is maroon and turn the seed's azure positions black.

    Key insight:
    The black slots are not empty output areas; they are missing Latin-square entries whose colors must be deduced before the seed can be used as a cutout mask.
    """
    g = np.asarray(grid, dtype=int)
    lattice = _find_lattice(g)
    lattice_color = int(lattice["color"])
    board_bbox = lattice["bbox"]
    line_rows = np.asarray(lattice["line_rows"], dtype=int)
    line_cols = np.asarray(lattice["line_cols"], dtype=int)
    cell_size = int(lattice["cell_size"])

    r0, c0, r1, c1 = board_bbox
    board = g[r0 : r1 + 1, c0 : c1 + 1]
    assert board.shape == (line_rows[-1] + 1, line_cols[-1] + 1)

    partial_slots = _extract_slot_colors(board, line_rows, line_cols, lattice_color)
    filled_slots = _solve_latin_unique(partial_slots)
    keep_mask = _find_seed_keep_mask(g, board_bbox, cell_size)
    assert keep_mask.shape == (cell_size, cell_size)

    return _render_board(filled_slots, keep_mask, lattice_color)


def _random_latin_square(n: int, colors: np.ndarray) -> np.ndarray:
    base = (np.arange(n)[:, None] + np.arange(n)[None, :]) % n
    row_perm = RNG.permutation(n)
    col_perm = RNG.permutation(n)
    symbol_perm = RNG.permutation(n)
    return np.asarray(colors[symbol_perm[base[row_perm][:, col_perm]]], dtype=int)


def _choose_partial_board(full_slots: np.ndarray) -> np.ndarray | None:
    n = int(full_slots.shape[0])
    max_blanks = min(n + 1, n * n - 1)
    positions = np.arange(n * n)
    for _ in range(200):
        blank_count = int(RNG.integers(1, max_blanks + 1))
        chosen = RNG.choice(positions, size=blank_count, replace=False)
        partial = full_slots.copy()
        partial.flat[chosen] = BLACK
        if len(np.unique(partial[partial != BLACK])) != n:
            continue
        try:
            solved = _solve_latin_unique(partial)
        except AssertionError:
            continue
        if np.array_equal(solved, full_slots):
            return partial
    return None


def _seed_mask(size: int) -> np.ndarray:
    rows, cols = np.indices((size, size))
    builders = [
        lambda: rows == cols,
        lambda: rows + cols == size - 1,
        lambda: (rows == size // 2) | (cols == size // 2),
        lambda: (rows == cols) | (rows + cols == size - 1),
        lambda: (rows == 0) | (cols == 0),
        lambda: (rows + cols) % 2 == 0,
        lambda: (rows == 0) | (rows == size - 1) | (cols == 0) | (cols == size - 1),
    ]
    for _ in range(200):
        mask = np.array(builders[int(RNG.integers(0, len(builders)))](), dtype=bool)
        mask = np.rot90(mask, int(RNG.integers(0, 4)))
        if bool(RNG.integers(0, 2)):
            mask = np.fliplr(mask)
        keep_count = int(mask.sum())
        if 2 <= keep_count <= size * size - max(2, size - 1):
            return mask
    raise RuntimeError("failed to build a usable seed mask")


def _render_plain_board(slot_colors: np.ndarray, cell_size: int, lattice_color: int) -> np.ndarray:
    n = int(slot_colors.shape[0])
    size = n * cell_size + n + 1
    board = np.full((size, size), int(lattice_color), dtype=int)
    for r in range(n):
        for c in range(n):
            rr = 1 + r * (cell_size + 1)
            cc = 1 + c * (cell_size + 1)
            board[rr : rr + cell_size, cc : cc + cell_size] = int(slot_colors[r, c])
    return board


def _place_seed(board: np.ndarray, seed: np.ndarray, orientation: str, gap: int) -> np.ndarray:
    bh, bw = map(int, board.shape)
    sh, sw = map(int, seed.shape)
    assert sh == sw
    if orientation == "right":
        canvas = np.zeros((bh, bw + gap + sw), dtype=int)
        canvas[:bh, :bw] = board
        sr = int(RNG.integers(0, bh - sh + 1))
        sc = bw + gap
    else:
        canvas = np.zeros((bh + gap + sh, bw), dtype=int)
        canvas[:bh, :bw] = board
        sr = bh + gap
        sc = int(RNG.integers(0, bw - sw + 1))
    canvas[sr : sr + sh, sc : sc + sw] = seed
    return canvas


def _pad_canvas(core: np.ndarray) -> np.ndarray:
    max_extra_h = 30 - core.shape[0]
    max_extra_w = 30 - core.shape[1]
    top = int(RNG.integers(0, max_extra_h + 1))
    bottom = int(RNG.integers(0, max_extra_h - top + 1))
    left = int(RNG.integers(0, max_extra_w + 1))
    right = int(RNG.integers(0, max_extra_w - left + 1))
    out = np.zeros((core.shape[0] + top + bottom, core.shape[1] + left + right), dtype=int)
    out[top : top + core.shape[0], left : left + core.shape[1]] = core
    return out


def _gen_attempt() -> np.ndarray | None:
    n = 3 if bool(RNG.integers(0, 2)) else 4
    max_cell_size = 6 if n == 3 else 4
    cell_size = int(RNG.integers(3, max_cell_size + 1))
    colors = RNG.choice(FILL_PALETTE, size=n, replace=False)
    full_slots = _random_latin_square(n, colors)
    partial_slots = _choose_partial_board(full_slots)
    if partial_slots is None:
        return None

    keep_mask = _seed_mask(cell_size)
    seed = np.where(keep_mask, MAROON, AZURE)
    board = _render_plain_board(partial_slots, cell_size, GREY)
    expected = _render_board(full_slots, keep_mask, GREY)

    orientation = "right" if bool(RNG.integers(0, 2)) else "below"
    gap = int(RNG.integers(1, 3))
    core = _place_seed(board, seed, orientation, gap)
    if max(core.shape) > 30:
        return None

    grid = _pad_canvas(core)
    if max(grid.shape) > 30:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
