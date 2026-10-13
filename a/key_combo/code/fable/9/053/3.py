from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
HALF_H_RANGE = (3, 5)
HALF_W_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 600


def _split_halves(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0
    h = g.shape[0] // 2
    return g[:h, :], g[h:, :]


def _foreground_color(mask_grid: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(mask_grid))) - {BACKGROUND})
    assert len(colors) == 1, "each half must contain exactly one foreground color"
    return colors[0]


def _stamp_output(select_mask: np.ndarray, stamp_mask: np.ndarray, stamp_color: int, *, carve: bool) -> np.ndarray:
    stamp_pattern = np.where(stamp_mask & ~(select_mask if carve else np.zeros_like(select_mask, dtype=bool)), stamp_color, BACKGROUND)
    assert np.any(select_mask), "selection mask must be non-empty"
    assert np.any(stamp_pattern != BACKGROUND), "stamp pattern must be non-empty"
    return np.kron(select_mask.astype(int), stamp_pattern)


def _solve_from_halves(select_half: np.ndarray, stamp_half: np.ndarray, *, carve: bool) -> np.ndarray:
    select_mask = np.asarray(select_half, dtype=int) != BACKGROUND
    stamp_mask = np.asarray(stamp_half, dtype=int) != BACKGROUND
    stamp_color = _foreground_color(stamp_half)
    return _stamp_output(select_mask, stamp_mask, stamp_color, carve=carve)


def main(grid: np.ndarray) -> np.ndarray:
    """Use the sparser half as both placement mask and carving mask for the denser half's stamp.

    Input structure
    The grid is a vertical stack of two equal-size black-background masks. Each
    half contains exactly one foreground color, the two foreground colors are
    different, and some aligned cells are foreground in both halves.

    Rule
    Count foreground pixels in the two halves. The half with fewer foreground
    pixels is the selection mask; the other half is the stamp mask. Build the
    stamp pattern from the stamp mask's foreground cells after removing every
    cell where the selection mask is also foreground. Then partition the output
    into blocks with the same size as one half, and place one full copy of that
    carved stamp pattern into every block whose corresponding selection-mask
    cell is foreground; leave all other blocks black.

    Key insight
    The sparse mask has a double role: it chooses which output blocks receive a
    stamp, and it also punches matching holes into the denser partner before the
    stamp is copied.
    """
    top, bottom = _split_halves(grid)
    top_color = _foreground_color(top)
    bottom_color = _foreground_color(bottom)
    assert top_color != bottom_color, "foreground colors must differ"

    top_count = int(np.count_nonzero(top))
    bottom_count = int(np.count_nonzero(bottom))
    assert top_count != bottom_count, "foreground counts must differ"

    if top_count < bottom_count:
        select_half, stamp_half = top, bottom
    else:
        select_half, stamp_half = bottom, top

    overlap = (select_half != BACKGROUND) & (stamp_half != BACKGROUND)
    assert np.any(overlap), "expected shared foreground cells"
    return _solve_from_halves(select_half, stamp_half, carve=True)


def _build_input(
    h: int,
    w: int,
    sparse_cells: set[int],
    dense_cells: set[int],
    sparse_color: int,
    dense_color: int,
    *,
    sparse_on_top: bool,
) -> np.ndarray:
    sparse = np.zeros((h, w), dtype=int)
    dense = np.zeros((h, w), dtype=int)
    for idx in sparse_cells:
        r, c = divmod(idx, w)
        sparse[r, c] = sparse_color
    for idx in dense_cells:
        r, c = divmod(idx, w)
        dense[r, c] = dense_color
    return np.vstack([sparse, dense]) if sparse_on_top else np.vstack([dense, sparse])


def _wrong_fixed_role(grid: np.ndarray, *, top_is_select: bool, carve: bool) -> np.ndarray | None:
    top, bottom = _split_halves(grid)
    select_half, stamp_half = (top, bottom) if top_is_select else (bottom, top)
    try:
        return _solve_from_halves(select_half, stamp_half, carve=carve)
    except AssertionError:
        return None


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    top, bottom = _split_halves(grid)
    top_count = int(np.count_nonzero(top))
    bottom_count = int(np.count_nonzero(bottom))
    sparse_half = top if top_count < bottom_count else bottom
    dense_half = bottom if top_count < bottom_count else top
    sparse_mask = sparse_half != BACKGROUND
    dense_mask = dense_half != BACKGROUND

    overlap = sparse_mask & dense_mask
    sparse_only = sparse_mask & ~dense_mask
    dense_only = dense_mask & ~sparse_mask
    if not (np.any(overlap) and np.any(sparse_only) and np.any(dense_only)):
        return False

    stamp_cells = int(np.count_nonzero(dense_only))
    block_count = int(np.count_nonzero(sparse_mask))
    if stamp_cells < 2 or block_count < 2:
        return False

    no_carve = _solve_from_halves(sparse_half, dense_half, carve=False)
    if np.array_equal(out, no_carve):
        return False

    wrong_role = _solve_from_halves(dense_half, sparse_half, carve=True)
    if np.array_equal(out, wrong_role):
        return False

    wrong_top = _wrong_fixed_role(grid, top_is_select=True, carve=True)
    wrong_bottom = _wrong_fixed_role(grid, top_is_select=False, carve=True)
    if wrong_top is None or wrong_bottom is None:
        return False
    if np.array_equal(wrong_top, wrong_bottom):
        return False
    if top_count < bottom_count and not np.array_equal(out, wrong_top):
        return False
    if bottom_count < top_count and not np.array_equal(out, wrong_bottom):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HALF_H_RANGE)
    w = random.randint(*HALF_W_RANGE)
    total = h * w
    sparse_on_top = bool(random.getrandbits(1))
    sparse_color, dense_color = random.sample(PALETTE, 2)

    sparse_max = min(6, total - 3)
    if sparse_max < 2:
        return None
    sparse_count = random.randint(2, sparse_max)

    dense_min = sparse_count + 1
    dense_max = min(total - 1, sparse_count + 5)
    if dense_min > dense_max:
        return None
    dense_count = random.randint(dense_min, dense_max)

    overlap_min = max(1, sparse_count + dense_count - total)
    overlap_max = min(sparse_count - 1, dense_count - 2)
    if overlap_min > overlap_max:
        return None
    overlap_count = random.randint(overlap_min, overlap_max)

    all_cells = list(range(total))
    overlap_cells = set(random.sample(all_cells, overlap_count))
    remaining = [idx for idx in all_cells if idx not in overlap_cells]

    sparse_only_count = sparse_count - overlap_count
    dense_only_count = dense_count - overlap_count
    sparse_only = set(random.sample(remaining, sparse_only_count))
    remaining2 = [idx for idx in remaining if idx not in sparse_only]
    dense_only = set(random.sample(remaining2, dense_only_count))

    sparse_cells = overlap_cells | sparse_only
    dense_cells = overlap_cells | dense_only
    grid = _build_input(h, w, sparse_cells, dense_cells, sparse_color, dense_color, sparse_on_top=sparse_on_top)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
