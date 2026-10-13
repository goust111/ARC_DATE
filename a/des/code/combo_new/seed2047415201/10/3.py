import random
from collections import deque

import numpy as np


BG = 0
SWATCH = 2
COLORS = list(range(1, 10))
MAX_ATTEMPTS = 300


def _bottom_legend(g: np.ndarray) -> tuple[list[int], np.ndarray]:
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 4

    band = g[h - SWATCH : h, :]
    used_cols = np.flatnonzero(np.any(band != BG, axis=0))
    assert len(used_cols) >= 2 * SWATCH
    c0, c1 = int(used_cols[0]), int(used_cols[-1])
    assert np.array_equal(used_cols, np.arange(c0, c1 + 1))
    assert (c1 - c0 + 1) % SWATCH == 0
    assert np.all(band[:, :c0] == BG)
    assert np.all(band[:, c1 + 1 :] == BG)

    colors = []
    for c in range(c0, c1 + 1, SWATCH):
        block = band[:, c : c + SWATCH]
        vals = np.unique(block)
        assert len(vals) == 1
        color = int(vals[0])
        assert color != BG
        colors.append(color)
    assert len(colors) >= 2
    assert len(set(colors)) == len(colors)

    legend_mask = np.zeros(g.shape, dtype=bool)
    legend_mask[h - SWATCH : h, c0 : c1 + 1] = True
    return colors, legend_mask


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object to the color immediately left of its color in the bottom legend.

    Input structure:
    - The background is black. The bottom two rows contain one horizontal legend strip made of
      adjacent 2x2 swatches, with distinct non-black colors read from left to right.
    - Above the legend are separated monochrome 4-connected foreground objects, each using one
      of the legend colors.

    Rule:
    - Read the legend colors from left to right as a cyclic order.
    - For every foreground object outside the legend, replace its color by the legend color one
      swatch to the left; objects using the leftmost swatch color become the rightmost color.
    - Leave black background pixels and the legend strip unchanged.

    Key insight:
    - The visible order of the bottom swatches defines a cyclic color substitution, so the
      object colors shift left through that ordered legend rather than by color id.
    """
    g = np.asarray(grid, dtype=int)
    colors, legend_mask = _bottom_legend(g)
    mapping = {color: colors[(i - 1) % len(colors)] for i, color in enumerate(colors)}

    target_mask = (g != BG) & (~legend_mask)
    out = g.copy()
    for pts in _components(target_mask):
        comp_colors = {int(g[r, c]) for r, c in pts}
        assert len(comp_colors) == 1
        color = next(iter(comp_colors))
        assert color in mapping
        new_color = mapping[color]
        for r, c in pts:
            out[r, c] = new_color
    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_blob(area: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < area:
        r, c = random.choice(frontier)
        nr, nc = random.choice(((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
        if (nr, nc) not in cells:
            cells.add((nr, nc))
            frontier.append((nr, nc))
    rmin = min(r for r, _c in cells)
    cmin = min(c for _r, c in cells)
    shifted = [(r - rmin, c - cmin) for r, c in cells]
    h = max(r for r, _c in shifted) + 1
    w = max(c for _r, c in shifted) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _random_shape() -> np.ndarray:
    kind = random.choice(("dot", "line", "rect", "corner", "blob"))
    if kind == "dot":
        mask = np.ones((1, 1), dtype=bool)
    elif kind == "line":
        length = random.randint(2, 6)
        if random.choice((True, False)):
            mask = np.ones((1, length), dtype=bool)
        else:
            mask = np.ones((length, 1), dtype=bool)
    elif kind == "rect":
        mask = np.ones((random.randint(2, 4), random.randint(2, 4)), dtype=bool)
    elif kind == "corner":
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        mask = np.zeros((height, width), dtype=bool)
        mask[:, 0] = True
        mask[-1, :] = True
        if random.choice((True, False)):
            mask = np.fliplr(mask)
        if random.choice((True, False)):
            mask = np.flipud(mask)
    else:
        mask = _random_blob(random.randint(4, 10))

    rotations = random.randint(0, 3)
    mask = np.rot90(mask, rotations)
    return _crop(mask)


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if r0 < 0 or c0 < 0 or r0 + mh > h or c0 + mw > w:
        return False
    region = occupied[r0 : r0 + mh, c0 : c0 + mw]
    return not np.any(region & mask)


def _reserve(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    mh, mw = mask.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + mh + 1)
    cc1 = min(occupied.shape[1], c0 + mw + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _paint(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    mh, mw = mask.shape
    grid[r0 : r0 + mh, c0 : c0 + mw][mask] = color


def _expected(grid: np.ndarray, colors: list[int], legend_col: int) -> np.ndarray:
    out = grid.copy()
    mapping = {color: colors[(i - 1) % len(colors)] for i, color in enumerate(colors)}
    out[:-SWATCH, :] = grid[:-SWATCH, :]
    legend_mask = np.zeros(grid.shape, dtype=bool)
    legend_mask[-SWATCH:, legend_col : legend_col + SWATCH * len(colors)] = True
    for color, new_color in mapping.items():
        mask = (grid == color) & (~legend_mask)
        out[mask] = new_color
    return out


def _gen_attempt() -> np.ndarray | None:
    color_count = random.randint(3, 5)
    width = random.randint(max(12, 2 * color_count + 2), 30)
    height = random.randint(10, 30)
    colors = random.sample(COLORS, color_count)
    legend_width = SWATCH * color_count
    legend_col = random.randint(0, width - legend_width)

    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for i, color in enumerate(colors):
        c0 = legend_col + SWATCH * i
        grid[-SWATCH:, c0 : c0 + SWATCH] = color
    occupied[-3:, :] = True

    object_colors = colors[:]
    object_colors.extend(random.choice(colors) for _ in range(random.randint(2, 7)))
    random.shuffle(object_colors)

    for color in object_colors:
        placed = False
        for _ in range(80):
            shape = _random_shape()
            sh, sw = shape.shape
            if sh > height - 3 or sw > width:
                continue
            r0 = random.randint(0, height - 3 - sh)
            c0 = random.randint(0, width - sw)
            if _can_place(occupied, shape, r0, c0):
                _paint(grid, shape, r0, c0, color)
                _reserve(occupied, shape, r0, c0)
                placed = True
                break
        if not placed:
            return None

    expected = _expected(grid, colors, legend_col)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
