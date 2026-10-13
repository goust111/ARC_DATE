from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
LINE_COLORS = (BLUE, RED, GREEN)
SEED_COLORS = (4, 5, 6, 7, 8, 9)

LAYOUTS = ((2, 2), (2, 3), (3, 2))
REGION_SIZE_RANGE = (4, 8)
MAX_GENERATION_ATTEMPTS = 300
CORNERS = ("TL", "TR", "BL", "BR")


def _constant_color(values: np.ndarray) -> int:
    uniq = set(map(int, np.asarray(values).ravel()))
    assert len(uniq) == 1
    return uniq.pop()


def _separator_indices(grid: np.ndarray) -> tuple[list[int], list[int]]:
    line_mask = np.isin(grid, LINE_COLORS)
    row_seps = [int(r) for r in range(grid.shape[0]) if bool(line_mask[r, :].all())]
    col_seps = [int(c) for c in range(grid.shape[1]) if bool(line_mask[:, c].all())]
    assert len(row_seps) >= 2 and len(col_seps) >= 2
    assert row_seps[0] == 0 and row_seps[-1] == grid.shape[0] - 1
    assert col_seps[0] == 0 and col_seps[-1] == grid.shape[1] - 1
    assert all(b - a >= 2 for a, b in zip(row_seps, row_seps[1:]))
    assert all(b - a >= 2 for a, b in zip(col_seps, col_seps[1:]))
    return row_seps, col_seps


def _grow_region(seed_mask: np.ndarray, top: int, bottom: int, left: int, right: int) -> np.ndarray:
    filled = seed_mask.copy()
    if top == bottom == GREEN and left == right == GREEN:
        filled |= np.rot90(seed_mask, 2)
        return filled

    if top == bottom == RED:
        filled |= np.flipud(filled)
    if left == right == BLUE:
        filled |= np.fliplr(filled)
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each corner seed inside a line-delimited subregion into the symmetric full motif encoded by that region's border colors.

    Input structure:
    - Black background fills a rectangular grid.
    - Single-pixel-wide red, blue, and green separator lines form a table of rectangular subregions with an outer border.
    - Each subregion contains one small single-color seed pattern in a corner; seed colors are different from the separator colors.

    Rule:
    - Read the four border segments around each subregion: top and bottom come from the horizontal separators, left and right from the vertical separators.
    - Red on both top and bottom means reflect the seed across the region's horizontal midline; blue on both left and right means reflect it across the vertical midline.
    - If both conditions hold, apply both reflections so the seed appears in all four symmetric corner positions; if all four borders are green, instead add the 180-degree rotated copy.
    - Keep the original seed, write the transformed copies inside the same subregion, and leave all separator lines and background unchanged.

    Key insight:
    - The colored divider segments are not decoration: each region's pair of horizontal border colors and pair of vertical border colors together encode which symmetry operation should complete the corner seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_seps, col_seps = _separator_indices(g)
    out = g.copy()

    for i in range(len(row_seps) - 1):
        r0, r1 = row_seps[i], row_seps[i + 1]
        for j in range(len(col_seps) - 1):
            c0, c1 = col_seps[j], col_seps[j + 1]
            patch = g[r0 + 1 : r1, c0 + 1 : c1]
            if patch.size == 0:
                continue

            colors = set(map(int, patch.ravel())) - {BLACK}
            colors -= set(LINE_COLORS)
            if not colors:
                continue
            assert len(colors) == 1
            seed_color = colors.pop()
            seed_mask = patch == seed_color

            top = _constant_color(g[r0, c0 + 1 : c1])
            bottom = _constant_color(g[r1, c0 + 1 : c1])
            left = _constant_color(g[r0 + 1 : r1, c0])
            right = _constant_color(g[r0 + 1 : r1, c1])

            grown = _grow_region(seed_mask, top, bottom, left, right)
            new_patch = np.full_like(patch, BLACK)
            new_patch[grown] = seed_color
            out[r0 + 1 : r1, c0 + 1 : c1] = new_patch

    return out


def _sample_classes(length: int, first: int, second: int) -> list[int]:
    values = [first, second]
    while len(values) < length:
        values.append(values[int(np.random.randint(0, 2))])
    np.random.shuffle(values)
    return [int(v) for v in values]


def _separator_positions(sizes: list[int]) -> list[int]:
    seps = [0]
    pos = 0
    for size in sizes:
        pos += int(size) + 1
        seps.append(pos)
    return seps


def _random_mask(h: int, w: int) -> np.ndarray:
    max_cells = h * w
    assert max_cells >= 1
    if max_cells == 1:
        return np.ones((1, 1), dtype=bool)

    low = 2
    upper = max_cells if max_cells <= 2 or bool(np.random.rand() < 0.25) else max_cells - 1
    target = int(np.random.randint(low, upper + 1))

    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    frontier = []

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in frontier:
                frontier.append((nr, nc))

    add_frontier(0, 0)
    filled = 1
    while filled < target and frontier:
        idx = int(np.random.randint(len(frontier)))
        r, c = frontier.pop(idx)
        if mask[r, c]:
            continue
        mask[r, c] = True
        filled += 1
        add_frontier(r, c)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[: rows[-1] + 1, : cols[-1] + 1]


def _choose_seed_dims(region_h: int, region_w: int) -> tuple[int, int]:
    options = [
        (h, w)
        for h in range(1, region_h // 2 + 1)
        for w in range(1, region_w // 2 + 1)
        if h * w >= 2 or (region_h // 2 == 1 and region_w // 2 == 1)
    ]
    assert options
    idx = int(np.random.randint(len(options)))
    return tuple(map(int, options[idx]))


def _place_seed(region: np.ndarray, color: int) -> None:
    region_h, region_w = map(int, region.shape)
    seed_h, seed_w = _choose_seed_dims(region_h, region_w)
    mask = _random_mask(seed_h, seed_w)
    corner = CORNERS[int(np.random.randint(len(CORNERS)))]

    oriented = mask
    if corner in ("TR", "BR"):
        oriented = np.fliplr(oriented)
    if corner in ("BL", "BR"):
        oriented = np.flipud(oriented)

    h, w = oriented.shape
    r0 = 0 if corner in ("TL", "TR") else region_h - h
    c0 = 0 if corner in ("TL", "BL") else region_w - w
    patch = region[r0 : r0 + h, c0 : c0 + w]
    assert not bool(patch.any())
    patch[oriented] = int(color)


def _region_kind(top_bottom: int, left_right: int) -> str:
    if top_bottom == GREEN and left_right == GREEN:
        return "rot180"
    if top_bottom == RED and left_right == BLUE:
        return "both"
    if top_bottom == RED:
        return "vertical"
    if left_right == BLUE:
        return "horizontal"
    raise AssertionError("unsupported border combination")


def _build_grid() -> np.ndarray:
    n_rows, n_cols = LAYOUTS[int(np.random.randint(len(LAYOUTS)))]
    row_modes = _sample_classes(n_rows, BLUE, GREEN)
    col_modes = _sample_classes(n_cols, RED, GREEN)

    row_sizes = [int(np.random.randint(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1] + 1)) for _ in range(n_rows)]
    col_sizes = [int(np.random.randint(REGION_SIZE_RANGE[0], REGION_SIZE_RANGE[1] + 1)) for _ in range(n_cols)]

    row_seps = _separator_positions(row_sizes)
    col_seps = _separator_positions(col_sizes)
    height = row_seps[-1] + 1
    width = col_seps[-1] + 1
    assert height <= 30 and width <= 30

    kinds = {
        _region_kind(col_modes[j], row_modes[i])
        for i in range(n_rows)
        for j in range(n_cols)
    }
    assert len(kinds) == 4

    grid = np.full((height, width), BLACK, dtype=int)

    for bi, r in enumerate(row_seps):
        for j in range(n_cols):
            c0, c1 = col_seps[j], col_seps[j + 1]
            grid[r, c0 + 1 : c1] = int(col_modes[j])

    for i in range(n_rows):
        r0, r1 = row_seps[i], row_seps[i + 1]
        for bj, c in enumerate(col_seps):
            grid[r0 + 1 : r1, c] = int(row_modes[i])

    for bi, r in enumerate(row_seps):
        for bj, c in enumerate(col_seps):
            candidates = []
            if bi > 0:
                candidates.append(row_modes[bi - 1])
            if bi < n_rows:
                candidates.append(row_modes[bi])
            if bj > 0:
                candidates.append(col_modes[bj - 1])
            if bj < n_cols:
                candidates.append(col_modes[bj])
            grid[r, c] = GREEN if GREEN in candidates else int(candidates[0])

    seed_palette = np.random.permutation(SEED_COLORS)[: n_rows * n_cols]
    idx = 0
    for i in range(n_rows):
        r0, r1 = row_seps[i], row_seps[i + 1]
        for j in range(n_cols):
            c0, c1 = col_seps[j], col_seps[j + 1]
            region = grid[r0 + 1 : r1, c0 + 1 : c1]
            _place_seed(region, int(seed_palette[idx]))
            idx += 1

    return grid


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if grid.ndim != 2 or out.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.array_equal(grid, out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_grid()
        if _valid_generated(grid):
            return grid
    raise RuntimeError("failed to generate valid input")
