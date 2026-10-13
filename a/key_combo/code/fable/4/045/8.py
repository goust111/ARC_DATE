from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _infer_colors(grid: np.ndarray) -> tuple[int, int]:
    colors = sorted(int(v) for v in np.unique(grid))
    assert len(colors) == 2, "input must use exactly one foreground color plus background"
    if 0 in colors:
        bg = 0
        fg = colors[1] if colors[0] == 0 else colors[0]
    else:
        counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
        bg = max(colors, key=lambda color: counts[color])
        fg = min(colors, key=lambda color: counts[color])
    assert bg != fg
    return bg, fg


def _try_factor(mask: np.ndarray, block_h: int, block_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = mask.shape
    if h % block_h != 0 or w % block_w != 0:
        return None

    layout_h = h // block_h
    layout_w = w // block_w
    stamp = None
    layout = np.zeros((layout_h, layout_w), dtype=bool)

    for br in range(layout_h):
        for bc in range(layout_w):
            block = mask[
                br * block_h : (br + 1) * block_h,
                bc * block_w : (bc + 1) * block_w,
            ]
            if block.any():
                if stamp is None:
                    stamp = block.copy()
                elif not np.array_equal(block, stamp):
                    return None
                layout[br, bc] = True

    if stamp is None:
        return None
    if stamp.all() or not stamp.any():
        return None
    if layout.all() or not layout.any():
        return None
    if stamp.shape == layout.shape and np.array_equal(stamp, layout):
        return None
    return stamp, layout


def _find_factorization(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for block_h in _divisors(h):
        for block_w in _divisors(w):
            result = _try_factor(mask, block_h, block_w)
            if result is not None:
                candidates.append(result)
    assert len(candidates) == 1, f"expected exactly one non-trivial factorization, found {len(candidates)}"
    return candidates[0]


def _swap_product(stamp: np.ndarray, layout: np.ndarray) -> np.ndarray:
    out = np.zeros((stamp.shape[0] * layout.shape[0], stamp.shape[1] * layout.shape[1]), dtype=bool)
    for r, c in np.argwhere(stamp):
        r = int(r)
        c = int(c)
        out[
            r * layout.shape[0] : (r + 1) * layout.shape[0],
            c * layout.shape[1] : (c + 1) * layout.shape[1],
        ] = layout
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Swap the micro stamp and the macro layout in a single-color block-product picture.

    Input structure:
    - The grid uses one foreground color on a background color.
    - It can be uniquely partitioned into equal blocks where every non-empty block is the same stamp A.
    - The positions of those non-empty blocks form a binary macro layout B.

    Rule:
    - Recover the unique block size that decomposes the input into stamp A and layout B.
    - Build a same-size output whose block size is the shape of B.
    - For every foreground cell of A, place one copy of B in the corresponding output block; background cells of A map to all-background blocks.

    Key insight:
    - The input is a Kronecker-style product B ⊗ A, so solving the task means factoring that product and reconstructing the swapped product A ⊗ B.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, fg = _infer_colors(g)
    mask = g == fg
    stamp, layout = _find_factorization(mask)
    swapped = _swap_product(stamp, layout)
    return np.where(swapped, fg, bg).astype(int)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(v) for v in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _is_primitive(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for block_h in _divisors(h):
        for block_w in _divisors(w):
            if block_h == h and block_w == w:
                continue
            if _try_factor(mask, block_h, block_w) is not None:
                return False
    return True


def _random_connected_mask(height: int, width: int) -> np.ndarray | None:
    area = height * width
    min_fg = max(2, int(np.ceil(area * 0.35)))
    max_fg = min(area - 2, int(np.floor(area * 0.7)))
    if min_fg > max_fg:
        min_fg = 2
        max_fg = area - 2
    if min_fg > max_fg:
        return None

    target = random.randint(min_fg, max_fg)
    start = (random.randrange(height), random.randrange(width))
    cells = {start}

    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if not _touches_all_edges(mask):
        return None
    if not _is_connected(mask):
        return None
    if not _is_primitive(mask):
        return None
    return mask


def _generate_mask(height: int, width: int) -> np.ndarray | None:
    for _ in range(120):
        mask = _random_connected_mask(height, width)
        if mask is not None:
            return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    stamp_h = random.randint(2, 6)
    stamp_w = random.randint(2, 6)
    layout_h = random.randint(2, min(6, 30 // stamp_h))
    layout_w = random.randint(2, min(6, 30 // stamp_w))

    stamp = _generate_mask(stamp_h, stamp_w)
    layout = _generate_mask(layout_h, layout_w)
    if stamp is None or layout is None:
        return None
    if stamp.shape == layout.shape and np.array_equal(stamp, layout):
        return None

    fg = random.randint(1, 9)
    inp_mask = np.kron(layout.astype(int), stamp.astype(int)).astype(bool)
    expected_mask = _swap_product(stamp, layout)
    inp = np.where(inp_mask, fg, 0).astype(int)

    try:
        recovered_stamp, recovered_layout = _find_factorization(inp_mask)
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(recovered_stamp, stamp):
        return None
    if not np.array_equal(recovered_layout, layout):
        return None
    expected = np.where(expected_mask, fg, 0).astype(int)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
