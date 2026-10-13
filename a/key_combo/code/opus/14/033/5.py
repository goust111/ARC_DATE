from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FRAME_COLORS = (1, 2, 3)
YELLOW = 4
FRAME_SIZE = 3
FRAME_STRIDE = 4
MAX_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask_1d)
    assert idx.size > 0
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for value in map(int, idx[1:]):
        if value == prev + 1:
            prev = value
        else:
            runs.append((start, prev))
            start = prev = value
    runs.append((start, prev))
    return runs


def _frame_border_color(block: np.ndarray) -> int:
    border = np.concatenate(
        [block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1]]
    )
    colors = {int(v) for v in border if int(v) != BACKGROUND}
    assert len(colors) == 1
    return colors.pop()


def _downsample_mask(yellow_crop: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = map(int, yellow_crop.shape)
    assert h % rows == 0 and w % cols == 0
    scale_r = h // rows
    scale_c = w // cols
    assert scale_r == scale_c
    k = scale_r
    mask = np.zeros((rows, cols), dtype=bool)
    for i in range(rows):
        for j in range(cols):
            block = yellow_crop[i * k : (i + 1) * k, j * k : (j + 1) * k]
            mask[i, j] = bool(block.any())
    return mask


def _group_complete(frame_crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(frame_crop, dtype=int, copy=True)
    row_runs = _runs(np.any(frame_crop != BACKGROUND, axis=1))
    col_runs = _runs(np.any(frame_crop != BACKGROUND, axis=0))
    rows, cols = len(row_runs), len(col_runs)
    assert mask.shape == (rows, cols)
    assert all(r1 - r0 + 1 == FRAME_SIZE for r0, r1 in row_runs)
    assert all(c1 - c0 + 1 == FRAME_SIZE for c0, c1 in col_runs)

    interior_h = FRAME_SIZE - 2
    interior_w = FRAME_SIZE - 2
    union_by_group = {
        False: np.zeros((interior_h, interior_w), dtype=bool),
        True: np.zeros((interior_h, interior_w), dtype=bool),
    }
    frames: list[tuple[int, int, int, bool]] = []

    for i, (r0, r1) in enumerate(row_runs):
        for j, (c0, c1) in enumerate(col_runs):
            block = frame_crop[r0 : r1 + 1, c0 : c1 + 1]
            color = _frame_border_color(block)
            interior = block[1:-1, 1:-1] == color
            group = bool(mask[i, j])
            union_by_group[group] |= interior
            frames.append((r0, c0, color, group))

    for r0, c0, color, group in frames:
        interior_view = out[r0 + 1 : r0 + FRAME_SIZE - 1, c0 + 1 : c0 + FRAME_SIZE - 1]
        interior_view[union_by_group[group]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Downsample the large yellow mask, group frames by its coverage bits, and complete each group's motif union.

    Input structure:
    - A black background contains one large yellow object in the upper region and a lower lattice of separated 3x3 colored frames.
    - Every frame has a solid 1-pixel border in blue, red, or green, plus a 1x1 center cell that is either black or matches the frame color.
    - The yellow object's bbox is an integer-scaled version of a binary mask whose row/column counts match the frame lattice.

    Rule:
    - Crop the yellow object to its bbox and downsample it to the frame-lattice size by marking a mask cell as covered iff its corresponding k x k block contains any yellow pixel.
    - Map each frame to the mask cell at the same lattice row and column; covered frames form one group and uncovered frames form the other.
    - Within each group, take the union of same-color interior positions across that group's frames and paint that union into every frame using each frame's own border color.
    - Erase the yellow region and return only the completed lower frame lattice.

    Key insight:
    - The grouping signal is hidden in an external scaled mask, so identical border colors can split into different completion groups depending only on the downsampled yellow coverage.
    """
    g = np.asarray(grid, dtype=int)

    yellow_box = _bbox(g == YELLOW)
    assert yellow_box is not None
    frame_box = _bbox((g != BACKGROUND) & (g != YELLOW))
    assert frame_box is not None

    frame_crop = _crop(g, frame_box)
    row_runs = _runs(np.any(frame_crop != BACKGROUND, axis=1))
    col_runs = _runs(np.any(frame_crop != BACKGROUND, axis=0))
    assert all(r1 - r0 + 1 == FRAME_SIZE for r0, r1 in row_runs)
    assert all(c1 - c0 + 1 == FRAME_SIZE for c0, c1 in col_runs)

    mask = _downsample_mask(_crop(g == YELLOW, yellow_box), len(row_runs), len(col_runs))
    assert mask.any() and (~mask).any()
    return _group_complete(frame_crop, mask)


def _random_mask(rows: int, cols: int) -> np.ndarray | None:
    for _ in range(100):
        mask = np.random.rand(rows, cols) < random.uniform(0.35, 0.65)
        ones = int(mask.sum())
        zeros = rows * cols - ones
        if ones < 2 or zeros < 2:
            continue
        if not bool(np.all(mask.any(axis=1))) or not bool(np.all(mask.any(axis=0))):
            continue
        mixed_rows = sum(
            1 for i in range(rows) if mask[i].any() and (~mask[i]).any()
        )
        mixed_cols = sum(
            1 for j in range(cols) if mask[:, j].any() and (~mask[:, j]).any()
        )
        if mixed_rows < min(2, rows) or mixed_cols < min(2, cols):
            continue
        return mask
    return None


def _random_colors(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = mask.shape
    for _ in range(100):
        colors = np.random.choice(FRAME_COLORS, size=(rows, cols))
        if len(set(map(int, colors.flatten()))) < 2:
            continue
        crosses = False
        for color in FRAME_COLORS:
            cells = colors == color
            if cells.any() and bool((mask & cells).any()) and bool(((~mask) & cells).any()):
                crosses = True
                break
        if crosses:
            return colors.astype(int)
    return None


def _paint_frame(grid: np.ndarray, r0: int, c0: int, color: int, filled: bool) -> None:
    grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = BACKGROUND
    grid[r0, c0 : c0 + FRAME_SIZE] = color
    grid[r0 + FRAME_SIZE - 1, c0 : c0 + FRAME_SIZE] = color
    grid[r0 : r0 + FRAME_SIZE, c0] = color
    grid[r0 : r0 + FRAME_SIZE, c0 + FRAME_SIZE - 1] = color
    if filled:
        grid[r0 + 1, c0 + 1] = color


def _build_frame_region(
    colors: np.ndarray, mask: np.ndarray, active_group: bool, carrier_cells: set[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = mask.shape
    height = rows * FRAME_SIZE + (rows - 1)
    width = cols * FRAME_SIZE + (cols - 1)
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    for i in range(rows):
        for j in range(cols):
            r0 = i * FRAME_STRIDE
            c0 = j * FRAME_STRIDE
            color = int(colors[i, j])
            filled_in = (i, j) in carrier_cells
            filled_out = bool(mask[i, j]) == bool(active_group)
            _paint_frame(inp, r0, c0, color, filled_in)
            _paint_frame(out, r0, c0, color, filled_out)
    return inp, out


def _color_group_completion(frame_region: np.ndarray) -> np.ndarray:
    out = np.array(frame_region, dtype=int, copy=True)
    row_runs = _runs(np.any(frame_region != BACKGROUND, axis=1))
    col_runs = _runs(np.any(frame_region != BACKGROUND, axis=0))
    carriers: dict[int, bool] = {color: False for color in FRAME_COLORS}
    frames: list[tuple[int, int, int]] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = frame_region[r0 : r1 + 1, c0 : c1 + 1]
            color = _frame_border_color(block)
            carriers[color] = carriers[color] or bool(block[1, 1] == color)
            frames.append((r0, c0, color))
    for r0, c0, color in frames:
        if carriers[color]:
            out[r0 + 1, c0 + 1] = color
    return out


def _global_completion(frame_region: np.ndarray) -> np.ndarray:
    out = np.array(frame_region, dtype=int, copy=True)
    has_carrier = False
    row_runs = _runs(np.any(frame_region != BACKGROUND, axis=1))
    col_runs = _runs(np.any(frame_region != BACKGROUND, axis=0))
    frames: list[tuple[int, int, int]] = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = frame_region[r0 : r1 + 1, c0 : c1 + 1]
            color = _frame_border_color(block)
            has_carrier = has_carrier or bool(block[1, 1] == color)
            frames.append((r0, c0, color))
    if has_carrier:
        for r0, c0, color in frames:
            out[r0 + 1, c0 + 1] = color
    return out


def _build_full_grid(mask: np.ndarray, frame_in: np.ndarray, k: int) -> np.ndarray:
    rows, cols = mask.shape
    pattern = np.kron(mask.astype(int) * YELLOW, np.ones((k, k), dtype=int))
    pattern_h, pattern_w = map(int, pattern.shape)
    frame_h, frame_w = map(int, frame_in.shape)

    height_choices = [
        (top_pad, bottom_pad, gap)
        for top_pad in (0, 1)
        for bottom_pad in (0, 1)
        for gap in (1, 2)
        if top_pad + pattern_h + gap + frame_h + bottom_pad <= 30
    ]
    assert height_choices
    top_pad, bottom_pad, gap = random.choice(height_choices)
    height = top_pad + pattern_h + gap + frame_h + bottom_pad

    min_width = max(pattern_w, frame_w)
    width = min_width + random.randint(0, min(3, 30 - min_width))
    assert width <= 30

    pattern_c0 = random.randint(0, width - pattern_w)
    frame_c0 = random.randint(0, width - frame_w)
    pattern_r0 = top_pad
    frame_r0 = top_pad + pattern_h + gap

    grid = np.zeros((height, width), dtype=int)
    grid[pattern_r0 : pattern_r0 + pattern_h, pattern_c0 : pattern_c0 + pattern_w] = pattern
    frame_mask = frame_in != BACKGROUND
    target = grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w]
    target[frame_mask] = frame_in[frame_mask]
    return grid


def _attempt() -> np.ndarray | None:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    frame_h = rows * FRAME_SIZE + (rows - 1)
    frame_w = cols * FRAME_SIZE + (cols - 1)

    valid_k = [
        k
        for k in (2, 3)
        if any(
            top_pad + rows * k + gap + frame_h + bottom_pad <= 30
            for top_pad in (0, 1)
            for bottom_pad in (0, 1)
            for gap in (1, 2)
        )
        and max(cols * k, frame_w) <= 30
    ]
    if not valid_k:
        return None
    k = random.choice(valid_k)

    mask = _random_mask(rows, cols)
    if mask is None:
        return None
    colors = _random_colors(mask)
    if colors is None:
        return None

    active_group = random.choice([False, True])
    active_cells = [
        (i, j) for i in range(rows) for j in range(cols) if bool(mask[i, j]) == active_group
    ]
    inactive_cells = [
        (i, j) for i in range(rows) for j in range(cols) if bool(mask[i, j]) != active_group
    ]
    if len(active_cells) < 2 or len(inactive_cells) < 2:
        return None

    n_carriers = random.randint(1, len(active_cells) - 1)
    carrier_cells = set(random.sample(active_cells, n_carriers))
    frame_in, frame_out = _build_frame_region(colors, mask, active_group, carrier_cells)
    grid = _build_full_grid(mask, frame_in, k)

    solved = main(grid)
    if not np.array_equal(solved, frame_out):
        return None
    if np.array_equal(frame_in, frame_out):
        return None
    if np.array_equal(_color_group_completion(frame_in), frame_out):
        return None
    if np.array_equal(_global_completion(frame_in), frame_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
