from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

RETRY_LIMIT = 400
BASE_SHAPES = (
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)},
    {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)},
    {(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)},
)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bbox")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _partition_bands(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    grey_rows = np.all(grid == GREY, axis=1)
    grey_cols = np.all(grid == GREY, axis=0)
    row_bands = _true_runs(~grey_rows)
    col_bands = _true_runs(~grey_cols)
    if not row_bands or not col_bands:
        raise ValueError("expected divider-defined row and column regions")
    return row_bands, col_bands


def _extract_puzzle(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray, np.ndarray]:
    row_bands, col_bands = _partition_bands(grid)
    row_heights = {r1 - r0 for r0, r1 in row_bands}
    col_widths = {c1 - c0 for c0, c1 in col_bands}
    if len(row_heights) != 1 or len(col_widths) != 1:
        raise ValueError("all regions must be equal-sized rectangles")

    control = np.zeros((len(row_bands), len(col_bands)), dtype=bool)
    stamp: np.ndarray | None = None
    empty_regions = 0

    for i, (r0, r1) in enumerate(row_bands):
        for j, (c0, c1) in enumerate(col_bands):
            region = np.array(grid[r0:r1, c0:c1], dtype=int)
            sprite_mask = (region == BLUE) | (region == RED)
            yellow_mask = region == YELLOW
            colors = set(map(int, np.unique(region)))

            if sprite_mask.any():
                if yellow_mask.any():
                    raise ValueError("sprite region cannot also contain yellow control pixels")
                if stamp is not None:
                    raise ValueError("expected exactly one sprite region")
                if not colors <= {BLACK, BLUE, RED}:
                    raise ValueError("sprite region contains unexpected colors")
                if not _connected4(sprite_mask):
                    raise ValueError("sprite must be 4-connected")
                if int(sprite_mask.sum()) < 3 or int(sprite_mask.sum()) > 6:
                    raise ValueError("sprite size must be 3..6 pixels")
                if not (np.any(region == BLUE) and np.any(region == RED)):
                    raise ValueError("sprite must use both blue and red")
                rmin, cmin, rmax, cmax = _bbox(sprite_mask)
                if rmin == 0 or cmin == 0 or rmax == region.shape[0] - 1 or cmax == region.shape[1] - 1:
                    raise ValueError("sprite must not touch the region border")
                stamp = region[rmin : rmax + 1, cmin : cmax + 1].copy()
                inner_mask = sprite_mask[rmin : rmax + 1, cmin : cmax + 1]
                stamp[~inner_mask] = BLACK
                control[i, j] = True
                continue

            if not colors <= {BLACK, YELLOW}:
                raise ValueError("non-sprite regions may only contain black or one yellow control pixel")
            if yellow_mask.any():
                if int(yellow_mask.sum()) != 1:
                    raise ValueError("control region must contain exactly one yellow pixel")
                center = (region.shape[0] // 2, region.shape[1] // 2)
                if not yellow_mask[center]:
                    raise ValueError("yellow control pixel must be centered in its region")
                control[i, j] = True
            else:
                empty_regions += 1

    if stamp is None:
        raise ValueError("missing sprite region")
    if int(control.sum()) < 2:
        raise ValueError("expected at least two active regions including the sprite region")
    if empty_regions < 1:
        raise ValueError("expected at least one empty region")
    return row_bands, col_bands, control, stamp


def _render_output(control: np.ndarray, stamp: np.ndarray) -> np.ndarray:
    rows, cols = control.shape
    sh, sw = stamp.shape
    out = np.zeros((rows * sh, cols * sw), dtype=int)
    for r in range(rows):
        for c in range(cols):
            if control[r, c]:
                out[r * sh : (r + 1) * sh, c * sw : (c + 1) * sw] = stamp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert a grey-partitioned region grid into a stamp tiling whose size is set by the partition layout.

    Input structure:
    - Black background contains full grey divider rows and columns that split the canvas into an R x C grid of equal rectangular regions.
    - Exactly one region contains a small 4-connected blue/red sprite that stays away from that region's border.
    - Some other regions contain a single centered yellow control pixel, while the remaining regions are empty black.

    Rule:
    - Detect the grey divider topology to recover the region grid dimensions R and C.
    - Find the unique sprite region, crop its blue/red content to the minimal bounding box, and use that crop as the stamp template.
    - Build an R x C control matrix where the sprite region and every yellow-control region are active, and every empty region is inactive.
    - Output an R-by-C tiling of stamp-sized blocks, copying the template into each active block and leaving inactive blocks black.

    Key insight:
    - The dividers encode both the macro output lattice size and the per-region switch positions, while the sprite region separately supplies the micro-scale stamp to tile across that lattice.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D ARC grid")
    _row_bands, _col_bands, control, stamp = _extract_puzzle(g)
    return _render_output(control, stamp)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _transform_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    if random.random() < 0.5:
        out = {(c, r) for r, c in out}
    out = _normalize_cells(out)
    max_r = max(r for r, _ in out)
    max_c = max(c for _, c in out)
    if random.random() < 0.5:
        out = {(r, max_c - c) for r, c in out}
    if random.random() < 0.5:
        out = {(max_r - r, c) for r, c in out}
    return _normalize_cells(out)


def _shape_grid(cells: set[tuple[int, int]]) -> np.ndarray:
    cells = _normalize_cells(cells)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _sample_stamp_patch(cell_h: int, cell_w: int) -> np.ndarray:
    for _ in range(200):
        mask = _shape_grid(_transform_shape(set(random.choice(BASE_SHAPES))))
        h, w = mask.shape
        if h > cell_h - 2 or w > cell_w - 2:
            continue
        top = random.randint(1, cell_h - h - 1)
        left = random.randint(1, cell_w - w - 1)
        patch = np.zeros((cell_h, cell_w), dtype=int)
        coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
        red_cells = set(random.sample(coords, random.randint(1, len(coords) - 1)))
        for r, c in coords:
            patch[top + r, left + c] = RED if (r, c) in red_cells else BLUE
        if _connected4(patch != BLACK):
            return patch
    raise RuntimeError("failed to construct a valid sprite patch")


def _eligible_sizes(n_regions: int) -> list[int]:
    return [size for size in (5, 7) if n_regions * size + (n_regions - 1) <= 30]


def _sample_control_matrix(rows: int, cols: int, sprite_rc: tuple[int, int]) -> np.ndarray:
    sr, sc = sprite_rc
    total = rows * cols
    for _ in range(200):
        mode = random.choice(("scatter", "row", "col", "checker"))
        control = np.zeros((rows, cols), dtype=bool)
        control[sr, sc] = True

        if mode == "scatter":
            p = random.uniform(0.3, 0.7)
            for r in range(rows):
                for c in range(cols):
                    if (r, c) != (sr, sc) and random.random() < p:
                        control[r, c] = True
        elif mode == "row":
            rr = random.randrange(rows)
            chosen = random.sample(range(cols), random.randint(1, cols))
            control[rr, chosen] = True
            if random.random() < 0.6:
                rr2 = random.randrange(rows)
                cc2 = random.randrange(cols)
                control[rr2, cc2] = True
        elif mode == "col":
            cc = random.randrange(cols)
            chosen = random.sample(range(rows), random.randint(1, rows))
            control[chosen, cc] = True
            if random.random() < 0.6:
                rr2 = random.randrange(rows)
                cc2 = random.randrange(cols)
                control[rr2, cc2] = True
        else:
            parity = random.randrange(2)
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 2 == parity:
                        control[r, c] = True
            flips = random.randint(0, max(1, total // 3))
            for _ in range(flips):
                rr = random.randrange(rows)
                cc = random.randrange(cols)
                control[rr, cc] = ~control[rr, cc]

        control[sr, sc] = True
        on_count = int(control.sum())
        if on_count < 2 or on_count >= total:
            continue
        return control
    raise RuntimeError("failed to sample a non-degenerate control matrix")


def _assemble_input(
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    control: np.ndarray,
    sprite_rc: tuple[int, int],
    sprite_patch: np.ndarray,
) -> np.ndarray:
    height = rows * cell_h + (rows - 1)
    width = cols * cell_w + (cols - 1)
    grid = np.zeros((height, width), dtype=int)

    for r in range(1, rows):
        grid[r * cell_h + (r - 1), :] = GREY
    for c in range(1, cols):
        grid[:, c * cell_w + (c - 1)] = GREY

    center = (cell_h // 2, cell_w // 2)
    sprite_r, sprite_c = sprite_rc
    for r in range(rows):
        for c in range(cols):
            r0 = r * (cell_h + 1)
            c0 = c * (cell_w + 1)
            if (r, c) == (sprite_r, sprite_c):
                grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = sprite_patch
            elif control[r, c]:
                grid[r0 + center[0], c0 + center[1]] = YELLOW
    return grid


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        row_bands, col_bands, control, stamp = _extract_puzzle(grid)
    except ValueError:
        return False

    if not np.array_equal(main(grid), expected):
        return False
    if not np.array_equal(expected, _render_output(control, stamp)):
        return False

    cell_h = row_bands[0][1] - row_bands[0][0]
    cell_w = col_bands[0][1] - col_bands[0][0]
    if stamp.shape[0] >= cell_h or stamp.shape[1] >= cell_w:
        return False
    if stamp.shape[0] * len(row_bands) > 30 or stamp.shape[1] * len(col_bands) > 30:
        return False
    if int(control.sum()) <= 1 or int(control.sum()) >= control.size:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    cell_h = random.choice(_eligible_sizes(rows))
    cell_w = random.choice(_eligible_sizes(cols))
    sprite_rc = (random.randrange(rows), random.randrange(cols))
    control = _sample_control_matrix(rows, cols, sprite_rc)
    sprite_patch = _sample_stamp_patch(cell_h, cell_w)

    stamp_mask = (sprite_patch == BLUE) | (sprite_patch == RED)
    r0, c0, r1, c1 = _bbox(stamp_mask)
    stamp = sprite_patch[r0 : r1 + 1, c0 : c1 + 1].copy()
    stamp[~stamp_mask[r0 : r1 + 1, c0 : c1 + 1]] = BLACK
    expected = _render_output(control, stamp)
    grid = _assemble_input(rows, cols, cell_h, cell_w, control, sprite_rc, sprite_patch)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if not _validate_generated(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
