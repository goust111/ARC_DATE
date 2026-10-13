from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
SEP = 5
PATTERN_COLORS = np.array([3, 4, 6, 7, 8, 9], dtype=int)
LAYOUTS = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2)]
MIN_CELL_SIDE = 5
MAX_ATTEMPTS = 400
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(separators):
        if start <= idx - 1:
            bands.append((int(start), int(idx - 1)))
        start = idx + 1
    if start <= length - 1:
        bands.append((int(start), int(length - 1)))
    return bands


def _find_separator_color(grid: np.ndarray) -> int:
    h, w = grid.shape
    row_colors = {int(grid[r, 0]) for r in range(h) if np.all(grid[r, :] == grid[r, 0])}
    col_colors = {int(grid[0, c]) for c in range(w) if np.all(grid[:, c] == grid[0, c])}
    candidates = (row_colors & col_colors) - {BG, BLUE, RED}
    assert len(candidates) == 1, "expected exactly one separator color"
    return int(next(iter(candidates)))


def _parse_panels(grid: np.ndarray) -> tuple[list[tuple[np.ndarray, int, int]], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    sep = _find_separator_color(g)
    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert sep_rows and sep_cols, "separator lattice must have both row and column lines"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert row_bands and col_bands

    heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "all panels must share one size"
    panel_shape = (int(next(iter(heights))), int(next(iter(widths))))
    assert len(row_bands) * len(col_bands) >= 2, "input must contain multiple panels"

    panels: list[tuple[np.ndarray, int, int]] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            control = int(cell[-1, -1])
            assert control in {RED, BLUE}, "panel control pixel must be red or blue"
            colors = set(map(int, np.unique(cell))) - {BG, control}
            assert len(colors) == 1, "each panel must have exactly one pattern color"
            color = int(next(iter(colors)))
            mask = cell == color
            mask[-1, -1] = False
            assert mask.any(), "each panel must contain at least one pattern pixel"
            panels.append((mask, color, control))

    return panels, panel_shape


def _compose_panels(panels: list[tuple[np.ndarray, int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for mask, color, control in panels:
        if control == RED:
            out[mask] = int(color)
        else:
            write_mask = mask & (out == BG)
            out[write_mask] = int(color)
    return out


def _compose_with_stats(
    masks: list[np.ndarray], colors: list[int], controls: list[int], shape: tuple[int, int]
) -> tuple[np.ndarray, bool, bool]:
    out = np.full(shape, BG, dtype=int)
    saw_blue_block = False
    saw_red_overwrite = False
    for mask, color, control in zip(masks, colors, controls):
        visible_overlap = mask & (out != BG) & (out != int(color))
        if control == BLUE:
            saw_blue_block |= bool(visible_overlap.any())
            write_mask = mask & (out == BG)
            out[write_mask] = int(color)
        else:
            saw_red_overwrite |= bool(visible_overlap.any())
            out[mask] = int(color)
    return out, saw_blue_block, saw_red_overwrite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite separator-lattice panels into one canvas using bottom-right red/blue control pixels.

    Input structure:
    - The background is black.
    - Single-pixel separator lines of one gray color divide the grid into equal rectangular panels.
    - Each panel contains a binary pattern drawn in one foreground color over the black background.
    - The bottom-right pixel of every panel is a control pixel colored red or blue, distinct from that panel's pattern color.

    Rule:
    - Split the grid at the separator rows and columns; the output canvas has the same size as one panel interior.
    - Read panels in row-major order, from left to right and then top to bottom.
    - If a panel's control pixel is red, copy all of that panel's pattern pixels onto the output, overwriting whatever color is already there.
    - If a panel's control pixel is blue, copy that panel's pattern pixels only onto output cells that are still black.
    - Ignore every control pixel itself during composition.

    Key insight:
    Every panel is a layer for the same canvas, and the red-versus-blue control pixel chooses overwrite or fill-only composition while panel order decides the final conflicts.
    """
    panels, panel_shape = _parse_panels(np.asarray(grid, dtype=int))
    return _compose_panels(panels, panel_shape)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _random_cell(rng: np.random.Generator, h: int, w: int) -> tuple[int, int]:
    while True:
        r = _randint(rng, 0, h - 1)
        c = _randint(rng, 0, w - 1)
        if (r, c) != (h - 1, w - 1):
            return int(r), int(c)


def _grow_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    sr, sc = _random_cell(rng, h, w)
    mask[sr, sc] = True
    frontier = {(sr, sc)}

    while int(mask.sum()) < target and frontier:
        fr, fc = list(frontier)[_randint(rng, 0, len(frontier) - 1)]
        candidates = []
        for dr, dc in DIRS4:
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) != (h - 1, w - 1):
                candidates.append((int(nr), int(nc)))
        if not candidates:
            frontier.discard((fr, fc))
            continue
        nr, nc = candidates[_randint(rng, 0, len(candidates) - 1)]
        mask[nr, nc] = True
        frontier.add((nr, nc))
        if rng.random() < 0.35:
            frontier.add((nr, nc))

    return mask


def _rect_union_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    n_rects = _randint(rng, 1, 3)
    for _ in range(n_rects):
        rect_h = _randint(rng, 1, max(1, min(h, target)))
        rect_w = _randint(rng, 1, max(1, min(w, target)))
        r0 = _randint(rng, 0, h - rect_h)
        c0 = _randint(rng, 0, w - rect_w)
        mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
    mask[-1, -1] = False
    return mask


def _stroke_mask(rng: np.random.Generator, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r, c = _random_cell(rng, h, w)
    mask[r, c] = True
    for _ in range(max(12, target * 6)):
        if int(mask.sum()) >= target:
            break
        dr, dc = DIRS4[_randint(rng, 0, len(DIRS4) - 1)]
        steps = _randint(rng, 1, max(h, w))
        for _ in range(steps):
            nr = min(max(r + dr, 0), h - 1)
            nc = min(max(c + dc, 0), w - 1)
            if (nr, nc) == (h - 1, w - 1):
                break
            r, c = int(nr), int(nc)
            mask[r, c] = True
            if int(mask.sum()) >= target:
                break
        if rng.random() < 0.45:
            coords = np.argwhere(mask)
            pick = coords[_randint(rng, 0, len(coords) - 1)]
            r, c = int(pick[0]), int(pick[1])
    mask[-1, -1] = False
    return mask


def _random_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    usable = h * w - 1
    min_area = max(3, usable // 6)
    max_area = max(min_area, min(usable - 1, (usable * 2) // 5))

    for _ in range(50):
        target = _randint(rng, min_area, max_area)
        style = _randint(rng, 0, 2)
        if style == 0:
            mask = _grow_mask(rng, h, w, target)
        elif style == 1:
            mask = _rect_union_mask(rng, h, w, target)
        else:
            mask = _stroke_mask(rng, h, w, target)
        mask[-1, -1] = False
        area = int(mask.sum())
        if min_area <= area <= max_area:
            return mask

    fallback = _grow_mask(rng, h, w, max(min_area, usable // 4))
    fallback[-1, -1] = False
    return fallback


def _critical_controls(rng: np.random.Generator) -> list[int]:
    if rng.random() < 0.5:
        return [RED, BLUE, RED]
    return [BLUE, RED, BLUE]


def _balanced_controls(rng: np.random.Generator, count: int) -> list[int]:
    controls = _critical_controls(rng)
    base_reds = sum(control == RED for control in controls)
    if count % 2 == 0:
        target_reds = count // 2
    else:
        target_reds = _randint(rng, count // 2, count // 2 + 1)
    rem_reds = target_reds - base_reds
    rem_blues = (count - 3) - rem_reds
    assert rem_reds >= 0 and rem_blues >= 0
    tail = [RED] * rem_reds + [BLUE] * rem_blues
    rng.shuffle(tail)
    return controls + tail


def _generate_masks(rng: np.random.Generator, count: int, h: int, w: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = [_random_mask(rng, h, w)]

    for _ in range(80):
        candidate = _random_mask(rng, h, w)
        if np.any(candidate & masks[0]) and np.any(candidate & ~masks[0]):
            masks.append(candidate)
            break
    else:
        return None

    occupied = masks[0] | masks[1]
    for _ in range(80):
        candidate = _random_mask(rng, h, w)
        if np.any(candidate & occupied) and np.any(candidate & ~occupied):
            masks.append(candidate)
            break
    else:
        return None

    for _ in range(count - 3):
        masks.append(_random_mask(rng, h, w))
    return masks


def _assign_colors(rng: np.random.Generator, count: int) -> list[int]:
    first = list(rng.choice(PATTERN_COLORS, size=3, replace=False).astype(int))
    rest = list(rng.choice(PATTERN_COLORS, size=max(0, count - 3), replace=True).astype(int))
    return first + rest


def _grid_shape(rows: int, cols: int, cell_h: int, cell_w: int, bordered: bool) -> tuple[int, int]:
    if bordered:
        return rows * cell_h + rows + 1, cols * cell_w + cols + 1
    return rows * cell_h + rows - 1, cols * cell_w + cols - 1


def _panel_starts(count: int, cell_side: int, bordered: bool) -> list[int]:
    if bordered:
        return [1 + idx * (cell_side + 1) for idx in range(count)]
    return [idx * (cell_side + 1) for idx in range(count)]


def _assemble_input(
    masks: list[np.ndarray],
    colors: list[int],
    controls: list[int],
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    bordered: bool,
) -> np.ndarray:
    grid_h, grid_w = _grid_shape(rows, cols, cell_h, cell_w, bordered)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    row_starts = _panel_starts(rows, cell_h, bordered)
    col_starts = _panel_starts(cols, cell_w, bordered)
    sep_rows = sorted(set(range(grid_h)) - {r + off for r in row_starts for off in range(cell_h)})
    sep_cols = sorted(set(range(grid_w)) - {c + off for c in col_starts for off in range(cell_w)})
    grid[sep_rows, :] = SEP
    grid[:, sep_cols] = SEP

    for idx, (mask, color, control) in enumerate(zip(masks, colors, controls)):
        pr = idx // cols
        pc = idx % cols
        r0 = row_starts[pr]
        c0 = col_starts[pc]
        cell = np.full((cell_h, cell_w), BG, dtype=int)
        cell[mask] = int(color)
        cell[-1, -1] = int(control)
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell

    return grid


def _informative_sample(
    masks: list[np.ndarray], colors: list[int], controls: list[int], shape: tuple[int, int]
) -> bool:
    out, saw_blue_block, saw_red_overwrite = _compose_with_stats(masks, colors, controls, shape)
    if not (saw_blue_block and saw_red_overwrite):
        return False

    filled = int(np.count_nonzero(out != BG))
    usable = shape[0] * shape[1] - 1
    if filled < max(4, usable // 5) or filled > (usable * 4) // 5:
        return False

    visible_colors = set(map(int, np.unique(out))) - {BG}
    if len(visible_colors) < 2:
        return False

    all_red = _compose_panels([(m, c, RED) for m, c in zip(masks, colors)], shape)
    all_blue = _compose_panels([(m, c, BLUE) for m, c in zip(masks, colors)], shape)
    reversed_out = _compose_panels(list(zip(masks[::-1], colors[::-1], controls[::-1])), shape)
    return (
        not np.array_equal(out, all_red)
        and not np.array_equal(out, all_blue)
        and not np.array_equal(out, reversed_out)
    )


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows, cols = LAYOUTS[_randint(rng, 0, len(LAYOUTS) - 1)]
    bordered = bool(_randint(rng, 0, 1))

    sep_rows = rows + 1 if bordered else rows - 1
    sep_cols = cols + 1 if bordered else cols - 1
    max_cell_h = (30 - sep_rows) // rows
    max_cell_w = (30 - sep_cols) // cols
    if max_cell_h < MIN_CELL_SIDE or max_cell_w < MIN_CELL_SIDE:
        return None

    cell_h = _randint(rng, MIN_CELL_SIDE, max_cell_h)
    cell_w = _randint(rng, MIN_CELL_SIDE, max_cell_w)
    count = rows * cols

    controls = _balanced_controls(rng, count)
    masks = _generate_masks(rng, count, cell_h, cell_w)
    if masks is None:
        return None
    colors = _assign_colors(rng, count)

    if not _informative_sample(masks, colors, controls, (cell_h, cell_w)):
        return None

    grid = _assemble_input(masks, colors, controls, rows, cols, cell_h, cell_w, bordered)
    expected = _compose_panels(list(zip(masks, colors, controls)), (cell_h, cell_w))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
