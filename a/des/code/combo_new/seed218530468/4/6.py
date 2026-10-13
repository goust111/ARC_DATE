import random
from itertools import permutations

import numpy as np


BG = 0
DIVIDER = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GAP = 2
LOCK_INTERIOR_MAX = 6
BOTTOM_PACK_MAX_W = 20
BOTTOM_PACK_MAX_H = 18
GEN_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_divider_row(grid: np.ndarray) -> tuple[int, int]:
    for r in range(grid.shape[0]):
        if int(grid[r, 0]) == BG:
            continue
        if np.all(grid[r] == grid[r, 0]):
            return r, int(grid[r, 0])
    raise AssertionError("expected a full-width non-black divider row")


def _resize_nearest(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    src_h, src_w = mask.shape
    dst_h, dst_w = shape
    row_idx = (np.arange(dst_h) * src_h) // dst_h
    col_idx = (np.arange(dst_w) * src_w) // dst_w
    return mask[row_idx[:, None], col_idx[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each colored lock interior with a nearest-neighbor scaled copy of the matching key.

    Input structure:
    - The background is black and one full-width grey row splits the grid into an upper key area and a lower lock area.
    - The upper area contains several isolated monochrome key patterns, each occupying a 3x3 or 4x4 bounding box.
    - The lower area contains several hollow rectangular frames with one-pixel colored borders and black interiors; each frame color matches exactly one key color.

    Rule:
    - Find the divider row, then extract every upper key as a binary template inside its own bounding box.
    - Match each lower frame to the key of the same color and measure that frame's interior height and width.
    - Resize the key template to the interior by nearest-neighbor scaling, paint the resized 1-cells in the key color, keep resized 0-cells black, erase the original upper keys, and preserve the divider plus the lower colored borders.

    Key insight:
    - Color equality identifies which key belongs to which lock, but the transferred shape is a scaled binary bitmap rather than a direct copy.
    """
    g = np.asarray(grid, dtype=int)
    divider_row, divider_color = _find_divider_row(g)

    top = g[:divider_row]
    lower = g[divider_row + 1 :]

    key_templates: dict[int, np.ndarray] = {}
    for color in np.unique(top):
        color = int(color)
        if color in (BG, divider_color):
            continue
        mask = top == color
        if not mask.any():
            continue
        r0, c0, r1, c1 = _bbox(mask)
        key_templates[color] = mask[r0 : r1 + 1, c0 : c1 + 1]

    assert key_templates, "expected at least one key above the divider"

    out = g.copy()
    out[:divider_row, :] = BG

    for color in np.unique(lower):
        color = int(color)
        if color in (BG, divider_color):
            continue
        assert color in key_templates, "every lock color must match an upper key"
        mask = lower == color
        r0, c0, r1, c1 = _bbox(mask)
        r0 += divider_row + 1
        r1 += divider_row + 1
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h > 0 and inner_w > 0, "frame must have a non-empty interior"
        scaled = _resize_nearest(key_templates[color], (inner_h, inner_w))
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:] = BG
        inner[scaled] = color

    return out


def _neighbors4(r: int, c: int, size: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _random_key_mask(size: int, rng: random.Random) -> np.ndarray:
    for _ in range(300):
        target = rng.randint(size + 1, size * size - 2)
        cells = {(rng.randrange(size), rng.randrange(size))}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                frontier.update(_neighbors4(r, c, size))
            frontier.difference_update(cells)
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if 0 not in rows or size - 1 not in rows or 0 not in cols or size - 1 not in cols:
            continue
        if len(rows) < 2 or len(cols) < 2:
            continue
        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to sample a valid key mask")


def _sample_lock_size(key_size: int, rng: random.Random) -> tuple[int, int]:
    inner_h = rng.randint(key_size, LOCK_INTERIOR_MAX)
    inner_w = rng.randint(key_size, LOCK_INTERIOR_MAX)
    if inner_h == key_size and inner_w == key_size and rng.random() < 0.75:
        if rng.random() < 0.5:
            inner_h = min(LOCK_INTERIOR_MAX, key_size + 1)
        else:
            inner_w = min(LOCK_INTERIOR_MAX, key_size + 1)
    return inner_h, inner_w


def _top_pack(key_sizes: list[int], rng: random.Random) -> tuple[dict[int, tuple[int, int]], int, int]:
    order = list(range(len(key_sizes)))
    rng.shuffle(order)
    pack_h = max(key_sizes)
    pack_w = sum(key_sizes[i] for i in order) + GAP * (len(order) - 1)
    positions: dict[int, tuple[int, int]] = {}
    x = 0
    for idx in order:
        dy = rng.randint(0, pack_h - key_sizes[idx])
        positions[idx] = (dy, x)
        x += key_sizes[idx] + GAP
    return positions, pack_h, pack_w


def _build_row_pack(
    rows: list[tuple[int, ...]],
    outer_sizes: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[dict[int, tuple[int, int]], int, int]:
    row_heights = [max(outer_sizes[i][0] for i in row) for row in rows]
    row_widths = [sum(outer_sizes[i][1] for i in row) + GAP * (len(row) - 1) for row in rows]
    pack_h = sum(row_heights) + GAP * (len(rows) - 1)
    pack_w = max(row_widths)
    positions: dict[int, tuple[int, int]] = {}
    y = 0
    for row, row_h, row_w in zip(rows, row_heights, row_widths):
        x = rng.randint(0, pack_w - row_w)
        for idx in row:
            h, w = outer_sizes[idx]
            dy = rng.randint(0, row_h - h)
            positions[idx] = (y + dy, x)
            x += w + GAP
        y += row_h + GAP
    return positions, pack_h, pack_w


def _bottom_pack(
    outer_sizes: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[dict[int, tuple[int, int]], int, int] | None:
    ids = tuple(range(len(outer_sizes)))
    one_row: list[list[tuple[int, ...]]] = []
    two_rows: list[list[tuple[int, ...]]] = []
    for perm in permutations(ids):
        rows1 = [perm]
        _, pack_h, pack_w = _build_row_pack(rows1, outer_sizes, rng)
        if pack_w <= BOTTOM_PACK_MAX_W and pack_h <= BOTTOM_PACK_MAX_H:
            one_row.append(rows1)
        for cut in range(1, len(ids)):
            rows2 = [perm[:cut], perm[cut:]]
            _, pack_h, pack_w = _build_row_pack(rows2, outer_sizes, rng)
            if pack_w <= BOTTOM_PACK_MAX_W and pack_h <= BOTTOM_PACK_MAX_H:
                two_rows.append(rows2)

    if one_row and two_rows:
        row_choices = one_row if rng.random() < 0.5 else two_rows
    elif one_row:
        row_choices = one_row
    elif two_rows:
        row_choices = two_rows
    else:
        return None

    return _build_row_pack(rng.choice(row_choices), outer_sizes, rng)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, color: int) -> None:
    h = inner_h + 2
    w = inner_w + 2
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 4)
    colors = rng.sample(KEY_COLORS, count)

    key_masks: list[np.ndarray] = []
    key_sizes: list[int] = []
    lock_sizes: list[tuple[int, int]] = []
    for _ in range(count):
        key_size = 3 if rng.random() < 0.5 else 4
        key_sizes.append(key_size)
        key_masks.append(_random_key_mask(key_size, rng))
        lock_sizes.append(_sample_lock_size(key_size, rng))

    top_pos, top_pack_h, top_pack_w = _top_pack(key_sizes, rng)
    outer_sizes = [(h + 2, w + 2) for h, w in lock_sizes]
    bottom_pack = _bottom_pack(outer_sizes, rng)
    if bottom_pack is None:
        return None
    bottom_pos, bottom_pack_h, bottom_pack_w = bottom_pack

    min_grid_h = top_pack_h + 1 + 1 + bottom_pack_h + 1
    min_grid_w = max(top_pack_w, bottom_pack_w) + 2
    if min_grid_h > 30 or min_grid_w > 30:
        return None

    grid_h = rng.randint(min_grid_h, min(30, min_grid_h + 4))
    grid_w = rng.randint(min_grid_w, min(30, min_grid_w + 6))

    extra_h = grid_h - min_grid_h
    extra_top = rng.randint(0, extra_h)
    extra_bottom = extra_h - extra_top
    top_area_h = top_pack_h + 1 + extra_top
    bottom_area_h = bottom_pack_h + 1 + extra_bottom
    divider_row = top_area_h

    top_y0 = rng.randint(0, top_area_h - top_pack_h - 1)
    top_x0 = rng.randint(0, grid_w - top_pack_w)
    bottom_y0 = divider_row + 1 + rng.randint(1, bottom_area_h - bottom_pack_h)
    bottom_x0 = rng.randint(0, grid_w - bottom_pack_w)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[divider_row, :] = DIVIDER

    for idx, color in enumerate(colors):
        y, x = top_pos[idx]
        mask = key_masks[idx]
        size = key_sizes[idx]
        window = grid[top_y0 + y : top_y0 + y + size, top_x0 + x : top_x0 + x + size]
        window[mask] = color

    for idx, color in enumerate(colors):
        y, x = bottom_pos[idx]
        inner_h, inner_w = lock_sizes[idx]
        _draw_frame(grid, bottom_y0 + y, bottom_x0 + x, inner_h, inner_w, color)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid key-lock scaling task")
