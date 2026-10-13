from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
QUADRANTS = ("tl", "tr", "bl", "br")

GEN_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _separator_candidates(length: int, *, min_side: int = 3, ratio_lo: float = 0.30, ratio_hi: float = 0.70) -> list[int]:
    return [
        idx
        for idx in range(min_side, length - min_side)
        if ratio_lo <= idx / (length - 1) <= ratio_hi
    ]


def _quadrant_bounds(shape: tuple[int, int], sep_r: int, sep_c: int, quadrant: str) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    if quadrant == "tl":
        return 0, 0, sep_r - 1, sep_c - 1
    if quadrant == "tr":
        return 0, sep_c + 1, sep_r - 1, w - 1
    if quadrant == "bl":
        return sep_r + 1, 0, h - 1, sep_c - 1
    if quadrant == "br":
        return sep_r + 1, sep_c + 1, h - 1, w - 1
    raise AssertionError(f"unknown quadrant: {quadrant}")


def _opposite_across_vertical(quadrant: str) -> str:
    return {"tl": "tr", "tr": "tl", "bl": "br", "br": "bl"}[quadrant]


def _opposite_across_horizontal(quadrant: str) -> str:
    return {"tl": "bl", "bl": "tl", "tr": "br", "br": "tr"}[quadrant]


def _middle(start: int, end: int) -> int:
    return (int(start) + int(end)) // 2


def _single_component(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected foreground pixels"
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    assert len(seen) == int(mask.sum()), "foreground mask must be a single 4-connected component"
    return np.array(sorted(seen), dtype=int)


def _find_structure(grid: np.ndarray) -> tuple[int, int, np.ndarray, int, int, str]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(sep_rows) == 1 and len(sep_cols) == 1, "expected one full grey row and one full grey column"
    sep_r = int(sep_rows[0])
    sep_c = int(sep_cols[0])

    fg_mask = (g != BLACK) & (g != GREY)
    fg_colors = sorted(set(map(int, g[fg_mask])))
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    color = int(fg_colors[0])

    coords = _single_component(fg_mask)
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows < sep_r) and np.all(cols < sep_c):
        quadrant = "tl"
    elif np.all(rows < sep_r) and np.all(cols > sep_c):
        quadrant = "tr"
    elif np.all(rows > sep_r) and np.all(cols < sep_c):
        quadrant = "bl"
    elif np.all(rows > sep_r) and np.all(cols > sep_c):
        quadrant = "br"
    else:
        raise AssertionError("foreground mask must lie entirely within one quadrant")

    return sep_r, sep_c, fg_mask, color, len(coords), quadrant


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten the lone irregular foreground mask into a line whose area decides both direction and mirror axis.

    Input structure:
    - A black background contains one full grey horizontal separator row and one
      full grey vertical separator column, splitting the grid into four quadrants.
    - Exactly one quadrant contains a single 4-connected monochromatic mask in a
      non-black, non-grey color; the other quadrants are empty.
    - The mask area is small enough that a straight segment of that length fits in
      the quadrant according to the branch condition.

    Rule:
    - Count the foreground pixels N, keep their color, and erase the original mask.
    - Let W be the width of the occupied quadrant.
    - If N <= W, place a horizontal segment of length N on the occupied side,
      centered in that quadrant's row span and flush against the vertical separator,
      then mirror it across the vertical separator into the opposite quadrant.
    - If N > W, place a vertical segment of length N on the occupied side,
      centered in that quadrant's column span and flush against the horizontal
      separator, then mirror it across the horizontal separator into the opposite quadrant.

    Key insight:
    - The foreground area plays two roles at once: it is the segment length, and
      comparing it to the quadrant width chooses both the straightening direction
      and which separator line becomes the reflection axis.
    """
    g = np.asarray(grid, dtype=int)
    sep_r, sep_c, fg_mask, color, count, quadrant = _find_structure(g)
    out = g.copy()
    out[fg_mask] = BLACK

    r0, c0, r1, c1 = _quadrant_bounds(g.shape, sep_r, sep_c, quadrant)
    q_height = int(r1 - r0 + 1)
    q_width = int(c1 - c0 + 1)

    if count <= q_width:
        mirror_quadrant = _opposite_across_vertical(quadrant)
        mr0, mc0, mr1, mc1 = _quadrant_bounds(g.shape, sep_r, sep_c, mirror_quadrant)
        assert (mr1 - mr0 + 1) == q_height and (mc1 - mc0 + 1) >= count, "mirrored horizontal segment must fit"

        row = _middle(r0, r1)
        if quadrant in ("tl", "bl"):
            cols = np.arange(sep_c - count, sep_c)
        else:
            cols = np.arange(sep_c + 1, sep_c + count + 1)
        mirror_cols = 2 * sep_c - cols

        out[row, cols] = color
        out[row, mirror_cols] = color
    else:
        assert count <= q_height, "vertical segment must fit in the source quadrant"
        mirror_quadrant = _opposite_across_horizontal(quadrant)
        mr0, mc0, mr1, mc1 = _quadrant_bounds(g.shape, sep_r, sep_c, mirror_quadrant)
        assert (mc1 - mc0 + 1) == q_width and (mr1 - mr0 + 1) >= count, "mirrored vertical segment must fit"

        col = _middle(c0, c1)
        if quadrant in ("tl", "tr"):
            rows = np.arange(sep_r - count, sep_r)
        else:
            rows = np.arange(sep_r + 1, sep_r + count + 1)
        mirror_rows = 2 * sep_r - rows

        out[rows, col] = color
        out[mirror_rows, col] = color

    return out


def _random_connected_shape(cell_count: int, max_h: int, max_w: int) -> tuple[np.ndarray, int, int] | None:
    candidates = [
        (h, w)
        for h in range(2, min(max_h, cell_count) + 1)
        for w in range(2, min(max_w, cell_count) + 1)
        if h * w > cell_count and 0.35 <= cell_count / float(h * w) <= 0.85
    ]
    if not candidates:
        candidates = [
            (h, w)
            for h in range(2, min(max_h, cell_count) + 1)
            for w in range(2, min(max_w, cell_count) + 1)
            if h * w > cell_count
        ]
    if not candidates:
        return None

    for _ in range(GEN_SHAPE_ATTEMPTS):
        box_h, box_w = random.choice(candidates)
        cells = {(
            random.randrange(box_h),
            random.randrange(box_w),
        )}
        while len(cells) < cell_count:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) < 2 or len(cols) < 2 or len(cells) == box_h * box_w:
            continue

        patch = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            patch[r, c] = True
        return patch, box_h, box_w
    return None


def _position_candidates(
    shape: tuple[int, int],
    quadrant: str,
    sep_r: int,
    sep_c: int,
    box_h: int,
    box_w: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _quadrant_bounds(shape, sep_r, sep_c, quadrant)
    all_positions = [
        (r, c)
        for r in range(r0, r1 - box_h + 2)
        for c in range(c0, c1 - box_w + 2)
    ]
    if not all_positions:
        return []

    preferred = []
    for r, c in all_positions:
        bbox_r1 = r + box_h - 1
        bbox_c1 = c + box_w - 1
        row_gap_ok = bbox_r1 <= sep_r - 2 if quadrant in ("tl", "tr") else r >= sep_r + 2
        col_gap_ok = bbox_c1 <= sep_c - 2 if quadrant in ("tl", "bl") else c >= sep_c + 2
        if row_gap_ok and col_gap_ok:
            preferred.append((r, c))
    return preferred or all_positions


def _build_grid(
    height: int,
    width: int,
    sep_r: int,
    sep_c: int,
    quadrant: str,
    count: int,
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    grid[sep_r, :] = GREY
    grid[:, sep_c] = GREY

    color = random.choice(PALETTE)
    q_r0, q_c0, q_r1, q_c1 = _quadrant_bounds(grid.shape, sep_r, sep_c, quadrant)
    q_height = q_r1 - q_r0 + 1
    q_width = q_c1 - q_c0 + 1

    shape_data = _random_connected_shape(count, q_height, q_width)
    if shape_data is None:
        return None
    patch, box_h, box_w = shape_data

    positions = _position_candidates(grid.shape, quadrant, sep_r, sep_c, box_h, box_w)
    if not positions:
        return None
    top, left = random.choice(positions)
    grid[top : top + box_h, left : left + box_w][patch] = color
    return grid


def _gen_horizontal_case() -> np.ndarray | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        height = random.randint(11, 17)
        width = random.randint(11, 17)
        row_candidates = _separator_candidates(height, min_side=3)
        col_candidates = _separator_candidates(width, min_side=3)
        if not row_candidates or not col_candidates:
            continue

        sep_r = random.choice(row_candidates)
        sep_c = random.choice(col_candidates)
        top_h, bottom_h = sep_r, height - sep_r - 1
        left_w, right_w = sep_c, width - sep_c - 1

        row_side = random.choice(["top", "bottom"])
        q_height = top_h if row_side == "top" else bottom_h
        max_count = min(8, q_height, left_w, right_w)
        if max_count < 3:
            continue

        count = random.randint(3, max_count)
        side = random.choice(["left", "right"])
        quadrant = ("t" if row_side == "top" else "b") + ("l" if side == "left" else "r")
        grid = _build_grid(height, width, sep_r, sep_c, quadrant, count)
        if grid is not None:
            return grid
    return None


def _gen_vertical_case() -> np.ndarray | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        height = random.randint(13, 18)
        width = random.randint(10, 15)
        row_candidates = _separator_candidates(height, min_side=4)
        col_candidates = _separator_candidates(width, min_side=3)
        if not row_candidates or not col_candidates:
            continue

        sep_r = random.choice(row_candidates)
        sep_c = random.choice(col_candidates)
        top_h, bottom_h = sep_r, height - sep_r - 1
        left_w, right_w = sep_c, width - sep_c - 1

        side = random.choice(["left", "right"])
        q_width = left_w if side == "left" else right_w
        min_count = max(3, q_width + 1)
        max_count = min(8, top_h, bottom_h)
        if min_count > max_count:
            continue

        count = random.randint(min_count, max_count)
        row_side = random.choice(["top", "bottom"])
        quadrant = ("t" if row_side == "top" else "b") + ("l" if side == "left" else "r")
        grid = _build_grid(height, width, sep_r, sep_c, quadrant, count)
        if grid is not None:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        branch = random.choice(["horizontal", "vertical"])
        grid = _gen_horizontal_case() if branch == "horizontal" else _gen_vertical_case()
        if grid is None:
            continue

        sep_r, sep_c, fg_mask, _color, count, quadrant = _find_structure(grid)
        r0, c0, r1, c1 = _quadrant_bounds(grid.shape, sep_r, sep_c, quadrant)
        q_width = c1 - c0 + 1
        q_height = r1 - r0 + 1
        if count <= q_width:
            if branch != "horizontal":
                continue
            if count > q_height:
                continue
        else:
            if branch != "vertical":
                continue
            if count > q_height:
                continue

        out = main(grid)
        if np.array_equal(grid, out):
            continue
        if np.count_nonzero(out == GREY) != np.count_nonzero(grid == GREY):
            continue
        if int(fg_mask.sum()) != count:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle 9 input")
