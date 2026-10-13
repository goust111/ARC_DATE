import random

import numpy as np


BG = 0
DIVIDER = 5
QUADRANTS = ("tl", "tr", "bl", "br")
NON_BG_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ROTATION_BY_CORNER = {
    "tl": 0,
    "tr": 3,  # clockwise 90
    "bl": 1,  # clockwise 270
    "br": 2,
}
MIN_QUADRANT_SPAN = 5
PATCH_RETRIES = 200
GEN_RETRIES = 400


def _quadrant_specs(shape: tuple[int, int], divider_row: int, divider_col: int) -> dict[str, dict[str, object]]:
    height, width = map(int, shape)
    specs: dict[str, dict[str, object]] = {}

    slices = {
        "tl": (slice(0, divider_row), slice(0, divider_col)),
        "tr": (slice(0, divider_row), slice(divider_col + 1, width)),
        "bl": (slice(divider_row + 1, height), slice(0, divider_col)),
        "br": (slice(divider_row + 1, height), slice(divider_col + 1, width)),
    }
    for key, (row_slice, col_slice) in slices.items():
        sub_h = int(row_slice.stop - row_slice.start)
        sub_w = int(col_slice.stop - col_slice.start)
        if key == "tl":
            local_outer = (0, 0)
            local_inner = (sub_h - 1, sub_w - 1)
            corner = (0, 0)
        elif key == "tr":
            local_outer = (0, sub_w - 1)
            local_inner = (sub_h - 1, 0)
            corner = (0, width - 1)
        elif key == "bl":
            local_outer = (sub_h - 1, 0)
            local_inner = (0, sub_w - 1)
            corner = (height - 1, 0)
        else:
            local_outer = (sub_h - 1, sub_w - 1)
            local_inner = (0, 0)
            corner = (height - 1, width - 1)
        specs[key] = {
            "rows": row_slice,
            "cols": col_slice,
            "local_outer": local_outer,
            "local_inner": local_inner,
            "corner": corner,
        }
    return specs


def _find_divider(grid: np.ndarray) -> tuple[int, int, int]:
    height, width = map(int, grid.shape)
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        rows = [r for r in range(height) if np.all(grid[r, :] == color)]
        cols = [c for c in range(width) if np.all(grid[:, c] == color)]
        if len(rows) == 1 and len(cols) == 1:
            candidates.append((color, rows[0], cols[0]))
    assert len(candidates) == 1, "expected one complete divider row and one complete divider column"
    return candidates[0]


def _indicator_in_quadrant(subgrid: np.ndarray, divider_color: int, inner_corner: tuple[int, int]) -> tuple[tuple[int, int], int]:
    mask = (subgrid != BG) & (subgrid != divider_color)
    coords = np.argwhere(mask)
    assert coords.size > 0, "each quadrant must contain at least an indicator"
    distances = np.abs(coords[:, 0] - inner_corner[0]) + np.abs(coords[:, 1] - inner_corner[1])
    best = int(np.argmin(distances))
    best_dist = int(distances[best])
    assert int(np.count_nonzero(distances == best_dist)) == 1, "indicator must be the unique closest colored cell to the cross"
    r, c = map(int, coords[best])
    return (r, c), int(subgrid[r, c])


def _square_window(mask: np.ndarray) -> tuple[int, int, int]:
    rows, cols = np.where(mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    size = max(height, width)
    top = r0 - (size - height) // 2
    left = c0 - (size - width) // 2
    top = max(0, min(top, mask.shape[0] - size))
    left = max(0, min(left, mask.shape[1] - size))
    return int(top), int(left), int(size)


def _rotate_pattern(subgrid: np.ndarray, divider_color: int, outer_corner: tuple[int, int], inner_corner: tuple[int, int], k: int) -> np.ndarray:
    indicator_pos, _ = _indicator_in_quadrant(subgrid, divider_color, inner_corner)
    pattern_mask = (subgrid != BG) & (subgrid != divider_color)
    pattern_mask[outer_corner] = False
    pattern_mask[indicator_pos] = False
    if not np.any(pattern_mask):
        return subgrid.copy()

    top, left, size = _square_window(pattern_mask)
    window_mask = pattern_mask[top : top + size, left : left + size]
    patch = np.full((size, size), BG, dtype=int)
    source = subgrid[top : top + size, left : left + size]
    patch[window_mask] = source[window_mask]
    rotated = np.rot90(patch, k=k)

    out = subgrid.copy()
    out[pattern_mask] = BG
    rr, cc = np.where(rotated != BG)
    for r, c in zip(rr.tolist(), cc.tolist(), strict=False):
        tr = top + r
        tc = left + c
        assert out[tr, tc] == BG, "rotated pattern must stay separate from markers and divider"
        out[tr, tc] = int(rotated[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each quadrant's small motif by the angle encoded through its nearby indicator color and the matching corner marker.

    Input structure:
    - Background is black.
    - One full grey row and one full grey column divide the grid into four quadrants.
    - Each quadrant contains one small non-symmetric foreground motif plus one colored indicator nearest the central cross.
    - The four outer corners each contain a single colored pixel whose color matches exactly one indicator color.

    Rule:
    - Read the corner colors as a rotation code: top-left means 0 degrees, top-right means clockwise 90 degrees, bottom-left means clockwise 270 degrees, and bottom-right means 180 degrees.
    - For each quadrant, find the unique colored cell closest to the cross inside that quadrant; this is that quadrant's indicator.
    - Match the indicator color to the corner carrying the same color, then rotate only the quadrant's motif by the corresponding angle around its own local center.
    - Keep the grey divider, the indicators, and the corner markers unchanged.

    Key insight:
    - The indicator does not directly encode an angle; its color must first be matched to the same color in a corner, and the corner position is what reveals the rotation amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    divider_color, divider_row, divider_col = _find_divider(g)
    specs = _quadrant_specs(g.shape, divider_row, divider_col)

    corner_colors = {
        "tl": int(g[0, 0]),
        "tr": int(g[0, -1]),
        "bl": int(g[-1, 0]),
        "br": int(g[-1, -1]),
    }
    assert len(set(corner_colors.values())) == 4, "corner markers must be distinct"
    rotation_by_color = {
        color: ROTATION_BY_CORNER[corner_key]
        for corner_key, color in corner_colors.items()
    }

    out = g.copy()
    for quadrant in QUADRANTS:
        spec = specs[quadrant]
        rows = spec["rows"]
        cols = spec["cols"]
        sub = out[rows, cols].copy()
        inner_corner = spec["local_inner"]
        _, indicator_color = _indicator_in_quadrant(sub, divider_color, inner_corner)
        k = rotation_by_color[indicator_color]
        rotated = _rotate_pattern(sub, divider_color, spec["local_outer"], inner_corner, k)
        out[rows, cols] = rotated
    return out


def _neighbors4(r: int, c: int, size: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc


def _random_mask(size: int) -> np.ndarray | None:
    min_cells = size + 1
    max_cells = size * size - 2
    if min_cells > max_cells:
        return None

    for _ in range(PATCH_RETRIES):
        target = random.randint(min_cells, max_cells)
        mask = np.zeros((size, size), dtype=bool)
        seed = (random.randrange(size), random.randrange(size))
        mask[seed] = True
        frontier = set(_neighbors4(seed[0], seed[1], size))

        while int(mask.sum()) < target:
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            for nb in _neighbors4(cell[0], cell[1], size):
                if not mask[nb]:
                    frontier.add(nb)
        if int(mask.sum()) != target:
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        rotations = {np.rot90(mask, k).tobytes() for k in range(4)}
        if len(rotations) < 4:
            continue

        mask = np.rot90(mask, k=random.randrange(4))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask
    return None


def _random_patch(size: int, color: int) -> np.ndarray | None:
    mask = _random_mask(size)
    if mask is None:
        return None
    patch = np.full((size, size), BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_patch(grid: np.ndarray, row0: int, col0: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    grid[row0 : row0 + h, col0 : col0 + w] = patch


def _build_expected(inp: np.ndarray, placements: dict[str, tuple[int, int, np.ndarray]]) -> np.ndarray:
    expected = inp.copy()
    divider_color, divider_row, divider_col = _find_divider(inp)
    specs = _quadrant_specs(inp.shape, divider_row, divider_col)
    corner_colors = {
        "tl": int(inp[0, 0]),
        "tr": int(inp[0, -1]),
        "bl": int(inp[-1, 0]),
        "br": int(inp[-1, -1]),
    }
    rotation_by_color = {
        color: ROTATION_BY_CORNER[corner_key]
        for corner_key, color in corner_colors.items()
    }

    for quadrant in QUADRANTS:
        spec = specs[quadrant]
        rows = spec["rows"]
        cols = spec["cols"]
        sub = inp[rows, cols]
        _, indicator_color = _indicator_in_quadrant(sub, divider_color, spec["local_inner"])
        k = rotation_by_color[indicator_color]
        row0, col0, patch = placements[quadrant]
        expected[row0 : row0 + patch.shape[0], col0 : col0 + patch.shape[1]] = np.rot90(patch, k=k)
    return expected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 29)
    width = random.randint(13, 29)
    divider_row = random.randint(MIN_QUADRANT_SPAN, height - MIN_QUADRANT_SPAN - 1)
    divider_col = random.randint(MIN_QUADRANT_SPAN, width - MIN_QUADRANT_SPAN - 1)

    grid = np.full((height, width), BG, dtype=int)
    grid[divider_row, :] = DIVIDER
    grid[:, divider_col] = DIVIDER
    specs = _quadrant_specs(grid.shape, divider_row, divider_col)

    palette = random.sample(NON_BG_COLORS, len(NON_BG_COLORS))
    indicator_colors = {quadrant: palette[i] for i, quadrant in enumerate(QUADRANTS)}
    motif_colors = {quadrant: palette[i + 4] for i, quadrant in enumerate(QUADRANTS)}

    indicator_values = [indicator_colors[q] for q in QUADRANTS]
    random.shuffle(indicator_values)
    for corner_key, color in zip(QUADRANTS, indicator_values, strict=False):
        cr, cc = specs[corner_key]["corner"]
        grid[cr, cc] = color

    placements: dict[str, tuple[int, int, np.ndarray]] = {}
    for quadrant in QUADRANTS:
        spec = specs[quadrant]
        rows = spec["rows"]
        cols = spec["cols"]
        local_inner = spec["local_inner"]
        global_indicator = (rows.start + local_inner[0], cols.start + local_inner[1])
        grid[global_indicator] = indicator_colors[quadrant]

        sub_h = int(rows.stop - rows.start)
        sub_w = int(cols.stop - cols.start)
        max_size = min(6, sub_h - 2, sub_w - 2)
        if max_size < 3:
            return None
        size = random.randint(3, max_size)
        patch = _random_patch(size, motif_colors[quadrant])
        if patch is None:
            return None

        top = random.randint(1, sub_h - size - 1)
        left = random.randint(1, sub_w - size - 1)
        row0 = rows.start + top
        col0 = cols.start + left
        _place_patch(grid, row0, col0, patch)
        placements[quadrant] = (row0, col0, patch)

    expected = _build_expected(grid, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-indicator puzzle")
