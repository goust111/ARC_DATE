import random

import numpy as np


BG = 0
SEP = 2
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
BRANCHES = ("horizontal", "vertical", "diagonal", "none")
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *separators, length]
    spans = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in spans]
    assert spans and min(sizes) > 0
    assert len(set(sizes)) == 1
    return spans


def _partition_slices(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_separators = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    col_separators = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert row_separators and col_separators

    row_spans = _intervals(row_separators, grid.shape[0])
    col_spans = _intervals(col_separators, grid.shape[1])
    side = row_spans[0][1] - row_spans[0][0]
    assert all(r1 - r0 == side for r0, r1 in row_spans)
    assert all(c1 - c0 == side for c0, c1 in col_spans)
    assert len(row_spans) == len(col_spans)
    assert side % 2 == 1
    return row_spans, col_spans, side


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expand_horizontal(src: np.ndarray) -> np.ndarray:
    return np.concatenate([src, np.fliplr(src[:, :-1])], axis=1)


def _expand_vertical(src: np.ndarray) -> np.ndarray:
    return np.vstack([src, np.flipud(src[:-1, :])])


def _expand_none(src: np.ndarray) -> np.ndarray:
    top = _expand_horizontal(src)
    return np.vstack([top, np.flipud(top[:-1, :])])


def _expand_diagonal(src: np.ndarray) -> np.ndarray:
    side = src.shape[0] * 2 - 1
    offset = src.shape[0] - 1
    out = np.zeros((side, side), dtype=bool)
    out[: src.shape[0], : src.shape[1]] = src

    rr, cc = np.where(np.triu(src))
    out[offset + cc, offset + rr] = True
    return out


def _expand_cell(cell: np.ndarray) -> np.ndarray:
    mask = cell != BG
    assert mask.any()

    colors = {int(color) for color in np.unique(cell[mask])}
    assert len(colors) == 1 and SEP not in colors
    color = colors.pop()

    r0, c0, r1, c1 = _bbox(mask)
    assert r0 == 0 and c0 == 0

    src = mask[: r1 + 1, : c1 + 1]
    side = cell.shape[0]
    h, w = src.shape

    h_sym = bool(np.array_equal(src, np.flipud(src)))
    v_sym = bool(np.array_equal(src, np.fliplr(src)))
    d_sym = h == w and bool(np.array_equal(src, src.T))

    branch: str | None = None
    if h == side and 2 * w - 1 == side and h_sym and not v_sym:
        branch = "horizontal"
        out_mask = _expand_horizontal(src)
    elif w == side and 2 * h - 1 == side and v_sym and not h_sym:
        branch = "vertical"
        out_mask = _expand_vertical(src)
    elif h == w and 2 * h - 1 == side and d_sym and not h_sym and not v_sym:
        branch = "diagonal"
        out_mask = _expand_diagonal(src)
    elif h == w and 2 * h - 1 == side and not h_sym and not v_sym and not d_sym:
        branch = "none"
        out_mask = _expand_none(src)
    else:
        raise AssertionError("cell seed does not match any supported symmetry branch")

    assert out_mask.shape == cell.shape, branch
    out = np.full_like(cell, BG)
    out[out_mask] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each top-left monochrome seed inside a red block grid according to the seed's own symmetry.

    Input structure:
    - The background is black.
    - One-cell red divider rows and columns partition the canvas into equal odd-sized square cells.
    - Each cell contains a single-color non-red seed touching the cell's upper-left corner and occupying only a left-half, top-half, or upper-left-square subregion.
    - Different seeds are unambiguously horizontal-symmetric, vertical-symmetric, main-diagonal-symmetric, or asymmetric.

    Rule:
    - Detect the equal square cells from the full red divider rows and columns.
    - In each cell, crop the occupied upper-left seed and test it against top-bottom flip, left-right flip, and main-diagonal transpose.
    - A horizontally symmetric seed is mirrored to the right; a vertically symmetric seed is mirrored downward.
    - A diagonal-symmetric square seed stays in the upper-left corner and its upper-triangle pixels are transposed into the complementary lower-right triangle.
    - An asymmetric square seed is mirrored right, down, and both, forming a four-quadrant reflected tiling.
    - Red divider lines and black background remain unchanged.

    Key insight:
    - The visible seed in each cell declares its own completion rule through symmetry, so the solver must classify every cell before expanding it.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, side = _partition_slices(g)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = g[r0:r1, c0:c1]
            assert cell.shape == (side, side)
            out[r0:r1, c0:c1] = _expand_cell(cell)
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not coords:
        return False

    seen = {coords[0]}
    stack = [coords[0]]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, *mask.shape):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _full_bbox(mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(mask)
    return r0 == 0 and c0 == 0 and r1 == mask.shape[0] - 1 and c1 == mask.shape[1] - 1


def _random_connected_mask(h: int, w: int, min_fill: int, max_fill: int) -> np.ndarray:
    target = random.randint(min_fill, max_fill)
    filled = {(0, 0)}
    frontier = set(_neighbors4(0, 0, h, w))

    while len(filled) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        filled.add(cell)
        for nb in _neighbors4(cell[0], cell[1], h, w):
            if nb not in filled:
                frontier.add(nb)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in filled:
        mask[r, c] = True
    return mask


def _make_horizontal_source(side: int) -> np.ndarray | None:
    half = side // 2 + 1
    for _ in range(200):
        top = _random_connected_mask(half, half, half + 1, half * half)
        if not top[:, -1].any():
            continue
        src = np.vstack([top, np.flipud(top[:-1, :])])
        if not _full_bbox(src):
            continue
        if np.array_equal(src, np.fliplr(src)):
            continue
        return src
    return None


def _make_vertical_source(side: int) -> np.ndarray | None:
    half = side // 2 + 1
    for _ in range(200):
        left = _random_connected_mask(half, half, half + 1, half * half)
        if not left[-1, :].any():
            continue
        src = np.hstack([left, np.fliplr(left[:, :-1])])
        if not _full_bbox(src):
            continue
        if np.array_equal(src, np.flipud(src)):
            continue
        return src
    return None


def _make_diagonal_source(side: int) -> np.ndarray | None:
    half = side // 2 + 1
    for _ in range(300):
        mask = np.zeros((half, half), dtype=bool)
        mask[0, 0] = True
        mask[half - 1, half - 1] = True
        mask[0, half - 1] = True
        mask[half - 1, 0] = True

        for r in range(half):
            for c in range(r, half):
                if (r, c) in {(0, 0), (0, half - 1), (half - 1, half - 1)}:
                    continue
                if random.random() < 0.45:
                    mask[r, c] = True
                    mask[c, r] = True

        if not _full_bbox(mask):
            continue
        if int(mask.sum()) < half + 2:
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _make_none_source(side: int) -> np.ndarray | None:
    half = side // 2 + 1
    for _ in range(250):
        mask = _random_connected_mask(half, half, half + 1, half * half - 1)
        if not _full_bbox(mask):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, mask.T):
            continue
        return mask
    return None


def _make_source(branch: str, side: int) -> np.ndarray:
    makers = {
        "horizontal": _make_horizontal_source,
        "vertical": _make_vertical_source,
        "diagonal": _make_diagonal_source,
        "none": _make_none_source,
    }
    src = makers[branch](side)
    if src is None:
        raise RuntimeError(f"failed to build a {branch} seed")
    return src


def _paint(mask: np.ndarray, color: int) -> np.ndarray:
    arr = np.full(mask.shape, BG, dtype=int)
    arr[mask] = int(color)
    return arr


def _apply_branch(mask: np.ndarray, branch: str) -> np.ndarray:
    if branch == "horizontal":
        return _expand_horizontal(mask)
    if branch == "vertical":
        return _expand_vertical(mask)
    if branch == "diagonal":
        return _expand_diagonal(mask)
    if branch == "none":
        return _expand_none(mask)
    raise ValueError(branch)


def _build_empty_grid(n: int, side: int) -> np.ndarray:
    size = n * side + (n - 1)
    grid = np.full((size, size), BG, dtype=int)
    for k in range(1, n):
        idx = k * side + (k - 1)
        grid[idx, :] = SEP
        grid[:, idx] = SEP
    return grid


def _branch_schedule(cell_count: int) -> list[str]:
    branches = list(BRANCHES) * (cell_count // len(BRANCHES))
    branches.extend(random.sample(BRANCHES, cell_count % len(BRANCHES)))
    random.shuffle(branches)
    return branches


def _attempt_grid() -> tuple[np.ndarray, np.ndarray] | None:
    side = random.choice((5, 7, 9))
    max_n = 31 // (side + 1)
    n = random.randint(2, max_n)

    grid = _build_empty_grid(n, side)
    expected = grid.copy()
    branches = _branch_schedule(n * n)

    for index, branch in enumerate(branches):
        r = index // n
        c = index % n
        r0 = r * side + r
        c0 = c * side + c

        color = random.choice(PALETTE)
        src_mask = _make_source(branch, side)
        grid[r0 : r0 + src_mask.shape[0], c0 : c0 + src_mask.shape[1]] = _paint(src_mask, color)

        full_mask = _apply_branch(src_mask, branch)
        expected[r0 : r0 + side, c0 : c0 + side] = _paint(full_mask, color)

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _attempt_grid()
        if pair is not None:
            grid, _expected = pair
            return grid
    raise RuntimeError("failed to generate a valid symmetry-grid ARC input")
