import random

import numpy as np


BG = 0
RED = 2
GREY = 5

GEN_RETRIES = 300
CELL_H_RANGE = (8, 14)
CELL_W_RANGE = (8, 9)
KEY_COLORS = (1, 3, 4, 6, 7, 8, 9)

KEY_BASES = (
    np.array(
        [
            [1, 0],
            [1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0],
            [1, 0],
            [1, 1],
        ],
        dtype=bool,
    ),
)

LOCK_BASES = (
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    ),
)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1], (int(r0), int(c0), int(r1), int(c1))


def _normalize_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    rel = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in rel))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _check_catalogs() -> None:
    for base in KEY_BASES:
        sigs = {_normalize_mask(np.rot90(base, k)) for k in range(4)}
        assert len(sigs) == 4
    for base in LOCK_BASES:
        cropped, _ = _crop_mask(base)
        assert max(cropped.shape) <= 5
        assert 6 <= int(cropped.sum()) <= 8
        assert _is_connected(cropped)
        sigs = {_normalize_mask(np.rot90(cropped, k)) for k in range(4)}
        assert len(sigs) == 4


_check_catalogs()


def _split_cells(grid: np.ndarray) -> tuple[tuple[slice, slice], tuple[slice, slice, slice]]:
    grey_rows = np.where(np.all(grid == GREY, axis=1))[0]
    grey_cols = np.where(np.all(grid == GREY, axis=0))[0]
    assert len(grey_rows) == 1
    assert len(grey_cols) == 2

    split_r = int(grey_rows[0])
    split_c1, split_c2 = (int(grey_cols[0]), int(grey_cols[1]))
    assert 0 < split_r < grid.shape[0] - 1
    assert 0 < split_c1 < split_c2 < grid.shape[1] - 1

    row_slices = (slice(0, split_r), slice(split_r + 1, grid.shape[0]))
    col_slices = (
        slice(0, split_c1),
        slice(split_c1 + 1, split_c2),
        slice(split_c2 + 1, grid.shape[1]),
    )

    heights = {sl.stop - sl.start for sl in row_slices}
    widths = {sl.stop - sl.start for sl in col_slices}
    assert len(heights) == 1
    assert len(widths) == 1
    return row_slices, col_slices


def _match_key_rotation(mask: np.ndarray) -> int:
    matches: list[int] = []
    for base in KEY_BASES:
        if int(mask.sum()) != int(base.sum()):
            continue
        for turns in range(4):
            rotated = np.rot90(base, turns)
            if mask.shape == rotated.shape and np.array_equal(mask, rotated):
                matches.append(turns)
    assert len(matches) == 1
    return matches[0]


def _extract_key(cell: np.ndarray) -> tuple[int, int]:
    fg = cell != BG
    assert fg.any()
    colors = np.unique(cell[fg])
    assert len(colors) == 1
    color = int(colors[0])
    assert color not in (BG, RED, GREY)
    mask, _ = _crop_mask(fg)
    turns = _match_key_rotation(mask)
    return color, turns


def _extract_lock(cell: np.ndarray) -> np.ndarray:
    red_mask = cell == RED
    assert red_mask.any()
    others = np.unique(cell[(cell != BG) & (cell != RED)])
    assert len(others) == 0
    mask, _ = _crop_mask(red_mask)
    return mask


def _paint_mask(cell: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = bbox
    assert mask.shape == (r1 - r0 + 1, c1 - c0 + 1)
    patch = cell[r0 : r1 + 1, c0 : c1 + 1]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Copy each row's lock sprite into the empty output cell after rotating it by the key sprite's orientation.

    Input structure:
    - The background is black, with one full grey separator row and two full grey separator columns splitting the grid into a 2×3 array of equal cells.
    - In each row, the left cell contains one small non-red L-shaped key, the middle cell contains one larger red lock sprite, and the right cell is empty.
    - The key is a rotated version of a canonical L shape, and the lock sprite is a rotated copy of the same larger shape across rows.

    Rule:
    - Use the grey separator lines to isolate the six cells.
    - For each row, determine the key rotation by matching its tight binary mask against the canonical L shape under 0°, 90°, 180°, or 270° rotation.
    - Rotate that row's red lock mask by the same amount, then draw the rotated lock in the right cell using the key's color, centered in the destination cell.

    Key insight:
    - The key and lock do not share shape; the key supplies only a rotation instruction, and that instruction must be transferred to a different sprite before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    row_slices, col_slices = _split_cells(g)
    out = g.copy()

    for row_slice in row_slices:
        key_cell = g[row_slice, col_slices[0]]
        lock_cell = g[row_slice, col_slices[1]]
        out_cell = out[row_slice, col_slices[2]]

        key_color, turns = _extract_key(key_cell)
        lock_mask = _extract_lock(lock_cell)

        out_cell[:] = BG
        rotated_lock = np.rot90(lock_mask, turns)
        top, left = _center_anchor(out_cell.shape, rotated_lock.shape)
        _stamp_mask(out_cell, rotated_lock, key_color, top, left)

    return out


def _random_anchor(cell_shape: tuple[int, int], mask_shape: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    max_r = cell_shape[0] - mask_shape[0]
    max_c = cell_shape[1] - mask_shape[1]
    assert max_r >= 0 and max_c >= 0

    lo_r = 1 if max_r >= 2 else 0
    hi_r = max_r - 1 if max_r >= 2 else max_r
    lo_c = 1 if max_c >= 2 else 0
    hi_c = max_c - 1 if max_c >= 2 else max_c

    top = rng.randint(lo_r, hi_r)
    left = rng.randint(lo_c, hi_c)
    return top, left


def _center_anchor(cell_shape: tuple[int, int], mask_shape: tuple[int, int]) -> tuple[int, int]:
    assert cell_shape[0] >= mask_shape[0] and cell_shape[1] >= mask_shape[1]
    return (cell_shape[0] - mask_shape[0]) // 2, (cell_shape[1] - mask_shape[1]) // 2


def _stamp_mask(cell: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    patch = cell[top : top + h, left : left + w]
    assert not np.any(patch[mask] != BG)
    patch[mask] = int(color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.randint(*CELL_H_RANGE)
    cell_w = rng.randint(*CELL_W_RANGE)
    grid = np.full((2 * cell_h + 1, 3 * cell_w + 2), BG, dtype=int)
    grid[cell_h, :] = GREY
    grid[:, cell_w] = GREY
    grid[:, 2 * cell_w + 1] = GREY

    row_slices, col_slices = _split_cells(grid)
    key_base = KEY_BASES[rng.randrange(len(KEY_BASES))]
    lock_base = LOCK_BASES[rng.randrange(len(LOCK_BASES))]
    key_turns = rng.sample(range(4), 2)
    key_colors = rng.sample(KEY_COLORS, 2)
    lock_turns = [rng.randrange(4) for _ in range(2)]

    for idx, row_slice in enumerate(row_slices):
        key_cell = grid[row_slice, col_slices[0]]
        key_mask = np.rot90(key_base, key_turns[idx])
        key_top, key_left = _random_anchor(key_cell.shape, key_mask.shape, rng)
        _stamp_mask(key_cell, key_mask, key_colors[idx], key_top, key_left)

        lock_cell = grid[row_slice, col_slices[1]]
        lock_mask, _ = _crop_mask(np.rot90(lock_base, lock_turns[idx]))
        lock_top, lock_left = _center_anchor(lock_cell.shape, lock_mask.shape)
        _stamp_mask(lock_cell, lock_mask, RED, lock_top, lock_left)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
