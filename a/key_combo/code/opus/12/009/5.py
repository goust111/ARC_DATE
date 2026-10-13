from __future__ import annotations

import random

import numpy as np


BG = 0
SCAFFOLD = 3
FILL = 4
PATTERN_COLORS = [1, 2, 5, 6, 7, 8, 9]

ROW_STRIDE = 4
COL_STRIDE = 4
FRAME_SIZE = 3
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _scaffold_meta(grid: np.ndarray) -> tuple[int, int, int, int, int, int, list[int]]:
    box = _bbox(grid == SCAFFOLD)
    assert box is not None, "missing green scaffold"
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert (height + 1) % ROW_STRIDE == 0, "invalid scaffold height"
    assert (width + 1) % COL_STRIDE == 0, "invalid scaffold width"
    h = (height + 1) // ROW_STRIDE
    k = (width + 1) // COL_STRIDE
    assert h >= 2 and k >= 1, "scaffold must have at least two rows and one column"

    offsets: list[int] = []
    for rr in range(h):
        center_r = r0 + 1 + rr * ROW_STRIDE
        row_count = 0
        for cc in range(k):
            frame_r0 = r0 + rr * ROW_STRIDE
            frame_c0 = c0 + cc * COL_STRIDE
            frame = grid[frame_r0 : frame_r0 + FRAME_SIZE, frame_c0 : frame_c0 + FRAME_SIZE]
            assert frame.shape == (FRAME_SIZE, FRAME_SIZE), "truncated frame"
            border = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=bool)
            border[1, 1] = False
            assert np.all(frame[border] == SCAFFOLD), "invalid frame border"
            assert frame[1, 1] in (BG, FILL), "invalid frame center"
            row_count += int(frame[1, 1] == FILL)

        gap_r = frame_r0 + FRAME_SIZE
        if rr < h - 1:
            assert np.all(grid[gap_r, c0 : c1 + 1] == BG), "expected empty row gap"
        offsets.append(row_count)
        assert center_r == frame_r0 + 1

    for cc in range(k - 1):
        gap_c = c0 + FRAME_SIZE + cc * COL_STRIDE
        assert np.all(grid[r0 : r1 + 1, gap_c] == BG), "expected empty column gap"

    return r0, c0, h, k, r1, c1, offsets


def _pattern_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = (grid != BG) & (grid != SCAFFOLD) & (grid != FILL)
    box = _bbox(mask)
    assert box is not None, "missing colored pattern"
    r0, c0, r1, c1 = box
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((crop != BG) & (crop != SCAFFOLD) & (crop != FILL)), "pattern must be solid"
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode row-wise offsets from the scaffold's filled frame centers and shear the neighboring pattern rightward.

    Input structure:
    - A black background contains a left-side lattice of green hollow 3x3 frames, separated by 1-cell black gaps.
    - Some frame centers are yellow; the number of yellow centers varies by scaffold row.
    - A separate right-side rectangular pattern uses non-black, non-green, non-yellow colors.

    Rule:
    - Find the scaffold lattice and treat each scaffold row as one code row.
    - Count yellow frame centers in each scaffold row to obtain that row's horizontal offset.
    - Extract the colored rectangle, preserve each row's pixels, and place row r on a black output canvas starting at the decoded offset for scaffold row r.

    Key insight:
    - The scaffold is not transformed itself; it is a row-by-row instruction table whose fill counts encode the pattern's non-uniform shear offsets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sr0, _sc0, h, _k, _sr1, scaffold_c1, offsets = _scaffold_meta(g)
    pr0, pc0, pr1, pc1 = _pattern_box(g)
    assert pc0 >= scaffold_c1 + 3, "pattern must be to the right of the scaffold with a 2-column gap"

    pattern = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
    assert pattern.shape[0] == h, "pattern height must match scaffold row count"

    max_offset = max(offsets)
    out = np.zeros((h, pattern.shape[1] + max_offset), dtype=int)
    for r, offset in enumerate(offsets):
        out[r, offset : offset + pattern.shape[1]] = pattern[r]
    return out


def _ensure_palette_coverage(arr: np.ndarray, palette: list[int]) -> None:
    cells = random.sample(range(arr.size), len(palette))
    rows, cols = np.divmod(cells, arr.shape[1])
    arr[rows, cols] = np.asarray(palette, dtype=int)


def _fill_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    mode = random.choice(("noise", "row_runs", "column_bands", "diagonal"))
    arr = np.zeros((h, w), dtype=int)

    if mode == "noise":
        arr[:] = np.random.choice(palette, size=(h, w))
    elif mode == "row_runs":
        for r in range(h):
            if w == 1:
                arr[r, 0] = random.choice(palette)
                continue
            cut_count = random.randint(0, min(3, w - 1))
            cuts = sorted(random.sample(range(1, w), cut_count))
            starts = [0] + cuts
            ends = cuts + [w]
            last = None
            for c0, c1 in zip(starts, ends):
                choices = [color for color in palette if color != last] or list(palette)
                color = random.choice(choices)
                arr[r, c0:c1] = color
                last = color
    elif mode == "column_bands":
        for c in range(w):
            arr[:, c] = random.choice(palette)
        if random.random() < 0.6:
            for r in range(h):
                arr[r, random.randrange(w)] = random.choice(palette)
    else:
        band = random.randint(1, 3)
        phase = random.randrange(len(palette))
        rows, cols = np.indices((h, w))
        arr[:] = np.asarray(palette, dtype=int)[((rows + cols) // band + phase) % len(palette)]
        if random.random() < 0.5:
            arr = np.fliplr(arr)

    _ensure_palette_coverage(arr, palette)
    return arr


def _sample_offsets(h: int, k: int) -> list[int]:
    mode = random.choice(("extremes", "duplicates", "mixed"))
    while True:
        if mode == "extremes":
            values = [random.randint(0, k) for _ in range(h)]
            values[random.randrange(h)] = 0
            values[random.randrange(h)] = k
        elif mode == "duplicates":
            base = [random.randint(0, k) for _ in range(max(2, min(3, h)))]
            values = [random.choice(base) for _ in range(h)]
        else:
            values = random.sample(range(k + 1), h) if h <= k + 1 and random.random() < 0.5 else [random.randint(0, k) for _ in range(h)]

        if len(set(values)) < 2:
            continue
        if random.random() < 0.65 and (values == sorted(values) or values == sorted(values, reverse=True)):
            continue
        return [int(v) for v in values]


def _offset_positions(offsets: list[int], k: int) -> list[set[int]]:
    positions: list[set[int]] = []
    for count in offsets:
        if count == 0:
            positions.append(set())
        else:
            positions.append(set(random.sample(range(k), count)))
    return positions


def _render_input(
    h: int,
    k: int,
    pattern: np.ndarray,
    fill_positions: list[set[int]],
    top: int,
    left: int,
    pattern_top: int,
    right_pad: int,
) -> np.ndarray:
    scaffold_h = ROW_STRIDE * h - 1
    scaffold_w = COL_STRIDE * k - 1
    total_h = top + scaffold_h
    total_w = left + scaffold_w + 2 + pattern.shape[1] + right_pad
    grid = np.zeros((total_h, total_w), dtype=int)

    for rr in range(h):
        for cc in range(k):
            r0 = top + rr * ROW_STRIDE
            c0 = left + cc * COL_STRIDE
            grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = SCAFFOLD
            grid[r0 + 1, c0 + 1] = BG
            if cc in fill_positions[rr]:
                grid[r0 + 1, c0 + 1] = FILL

    pattern_left = left + scaffold_w + 2
    grid[pattern_top : pattern_top + h, pattern_left : pattern_left + pattern.shape[1]] = pattern
    return grid


def _expected_output(pattern: np.ndarray, offsets: list[int]) -> np.ndarray:
    out = np.zeros((pattern.shape[0], pattern.shape[1] + max(offsets)), dtype=int)
    for r, offset in enumerate(offsets):
        out[r, offset : offset + pattern.shape[1]] = pattern[r]
    return out


def _semantic_checks(grid: np.ndarray, pattern: np.ndarray, offsets: list[int], k: int) -> bool:
    expected = _expected_output(pattern, offsets)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    # The instruction scaffold must matter: perturbing one row count should change the output.
    candidate_rows = [r for r, value in enumerate(offsets) if 0 < value < k]
    if not candidate_rows:
        candidate_rows = [r for r, value in enumerate(offsets) if value < k]
    row = candidate_rows[0]
    mutated = grid.copy()
    scaffold_box = _bbox(mutated == SCAFFOLD)
    if scaffold_box is None:
        return False
    sr0, sc0, _sr1, _sc1 = scaffold_box
    current = offsets[row]
    centers = [sc0 + 1 + cc * COL_STRIDE for cc in range(k)]
    if current < k:
        for c in centers:
            if mutated[sr0 + 1 + row * ROW_STRIDE, c] == BG:
                mutated[sr0 + 1 + row * ROW_STRIDE, c] = FILL
                break
    else:
        for c in centers:
            if mutated[sr0 + 1 + row * ROW_STRIDE, c] == FILL:
                mutated[sr0 + 1 + row * ROW_STRIDE, c] = BG
                break
    try:
        changed = main(mutated)
    except AssertionError:
        return False
    return not np.array_equal(changed, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(3, 6)
    k = random.randint(2, 5)
    top = random.randint(0, 1)
    left = random.randint(0, 1)
    right_pad = random.randint(0, 1)

    scaffold_h = ROW_STRIDE * h - 1
    max_pattern_w = 30 - (left + (COL_STRIDE * k - 1) + 2 + right_pad)
    if max_pattern_w < 3:
        return None
    w = random.randint(3, min(8, max_pattern_w))

    pattern_top_max = scaffold_h - h
    if pattern_top_max < top:
        return None
    pattern_top = random.randint(top, pattern_top_max)

    color_count = random.randint(2, min(4, w, len(PATTERN_COLORS)))
    palette = random.sample(PATTERN_COLORS, color_count)
    pattern = _fill_pattern(h, w, palette)
    if len(set(map(int, pattern.flatten()))) < 2:
        return None

    offsets = _sample_offsets(h, k)
    if max(offsets) + w > 30:
        return None
    fill_positions = _offset_positions(offsets, k)

    grid = _render_input(h, k, pattern, fill_positions, top, left, pattern_top, right_pad)
    if max(grid.shape) > 30:
        return None
    if not _semantic_checks(grid, pattern, offsets, k):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
