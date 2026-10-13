from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


BG = 0
SLOT_SIZE = 3
EXPANDED_SIZE = SLOT_SIZE * 2
OUTPUT_STRIDE = EXPANDED_SIZE + 1
GEN_ATTEMPTS = 400
PALETTE = np.arange(1, 10, dtype=int)


def _dense_runs(counts: np.ndarray) -> list[tuple[int, int]]:
    values = sorted(set(map(int, counts)))
    assert len(values) >= 2, "need at least two distinct density levels"

    gaps = [(b - a, b) for a, b in zip(values, values[1:])]
    max_gap = max(gap for gap, _ in gaps)
    thresholds = [upper for gap, upper in gaps if gap == max_gap]
    assert len(thresholds) == 1, "dense-band threshold must be unique"

    high = np.asarray(counts) >= thresholds[0]
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(high):
        if not high[i]:
            i += 1
            continue
        j = i
        while j < len(high) and high[j]:
            j += 1
        runs.append((i, j))
        i = j

    assert len(runs) >= 2, "need multiple dense bands"
    assert len({r1 - r0 for r0, r1 in runs}) == 1, "band heights must be uniform"
    assert len({runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)}) == 1, "band gaps must be uniform"
    return runs


def _expand_slot(pattern: np.ndarray) -> np.ndarray:
    top = np.concatenate((pattern, np.fliplr(pattern)), axis=1)
    bottom = np.concatenate((np.flipud(pattern), np.flipud(np.fliplr(pattern))), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _detect_slots(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg = g != bg
    row_runs = _dense_runs(fg.sum(axis=1))
    col_runs = _dense_runs(fg.sum(axis=0))
    assert {r1 - r0 for r0, r1 in row_runs} == {SLOT_SIZE}
    assert {c1 - c0 for c0, c1 in col_runs} == {SLOT_SIZE}
    return int(bg), row_runs, col_runs


def main(grid: np.ndarray) -> np.ndarray:
    """Detect a dense slot array and mirror-expand each 3x3 slot into a separated 2x2 block layout.

    Input structure:
    - The background is black and most cells are background.
    - Non-background cells form a regular array of 3x3 slots located by dense row bands and dense column bands.
    - Each slot contains a small single-color pattern, while sparse noise pixels may appear outside the slot rows and columns.

    Rule:
    - Find the dense row bands and dense column bands; their Cartesian product gives the slot positions.
    - Ignore every pixel outside those slot positions and extract each slot's 3x3 content.
    - For each slot pattern P, build a 6x6 tile with quadrants [P, horizontal mirror, vertical mirror, 180-degree mirror].
    - Reassemble the expanded tiles in the original row-major slot layout with one black separator row and column between neighboring tiles.

    Key insight:
    - Dense-band detection is only a locator here: once the slot lattice is found, each slot must be transformed independently, so a whole-array mirror or simple 2x scaling does not match the examples.
    """
    g = np.asarray(grid, dtype=int)
    bg, row_runs, col_runs = _detect_slots(g)

    out_h = len(row_runs) * EXPANDED_SIZE + max(0, len(row_runs) - 1)
    out_w = len(col_runs) * EXPANDED_SIZE + max(0, len(col_runs) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    for br, (r0, r1) in enumerate(row_runs):
        for bc, (c0, c1) in enumerate(col_runs):
            slot = g[r0:r1, c0:c1]
            mask = slot != bg
            assert mask.any(), "each detected slot must contain a pattern"
            assert np.all(mask.sum(axis=0) >= 1), "each slot column must participate in the dense band"
            assert np.all(mask.sum(axis=1) >= 1), "each slot row must participate in the dense band"
            colors = set(map(int, slot[mask].ravel()))
            assert len(colors) == 1, "each slot must use a single non-background color"
            rr = br * OUTPUT_STRIDE
            cc = bc * OUTPUT_STRIDE
            out[rr : rr + EXPANDED_SIZE, cc : cc + EXPANDED_SIZE] = _expand_slot(slot)

    return out


def _is_connected8(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    stack = [tuple(map(int, coords[0]))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                    nxt = (rr, cc)
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
    return len(seen) == int(mask.sum())


def _build_pattern_library() -> list[np.ndarray]:
    patterns: list[np.ndarray] = []
    for bits in range(1 << (SLOT_SIZE * SLOT_SIZE)):
        mask = np.zeros((SLOT_SIZE, SLOT_SIZE), dtype=bool)
        for idx in range(SLOT_SIZE * SLOT_SIZE):
            if bits & (1 << idx):
                mask[idx // SLOT_SIZE, idx % SLOT_SIZE] = True
        cells = int(mask.sum())
        if not 3 <= cells <= 5:
            continue
        if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
            continue
        if not _is_connected8(mask):
            continue
        patterns.append(mask)
    patterns.sort(key=lambda x: (int(x.sum()), x.astype(np.uint8).tobytes()))
    return patterns


PATTERN_LIBRARY = _build_pattern_library()


def _sample_layout(rng: np.random.Generator) -> tuple[int, int, list[int], list[int]] | None:
    slot_rows = int(rng.integers(3, 5))
    slot_cols = int(rng.integers(3, 5))
    row_gap = int(rng.integers(1, 4))
    col_gap = int(rng.integers(1, 4))

    dense_h = slot_rows * SLOT_SIZE + (slot_rows - 1) * row_gap
    dense_w = slot_cols * SLOT_SIZE + (slot_cols - 1) * col_gap

    total_h_min = max(16, dense_h)
    total_h_max = min(24, 30)
    total_w_min = max(16, dense_w)
    total_w_max = min(24, 30)
    if total_h_min > total_h_max or total_w_min > total_w_max:
        return None

    total_h = int(rng.integers(total_h_min, total_h_max + 1))
    total_w = int(rng.integers(total_w_min, total_w_max + 1))
    extra_h = total_h - dense_h
    extra_w = total_w - dense_w
    top = int(rng.integers(0, extra_h + 1))
    left = int(rng.integers(0, extra_w + 1))
    bottom = extra_h - top
    right = extra_w - left
    assert top + bottom == extra_h and left + right == extra_w

    row_starts = [top + i * (SLOT_SIZE + row_gap) for i in range(slot_rows)]
    col_starts = [left + i * (SLOT_SIZE + col_gap) for i in range(slot_cols)]
    if row_starts[-1] + SLOT_SIZE + bottom != total_h:
        return None
    if col_starts[-1] + SLOT_SIZE + right != total_w:
        return None
    return total_h, total_w, row_starts, col_starts


def _choose_slot_masks(slot_count: int, rng: np.random.Generator) -> list[np.ndarray]:
    choice = rng.choice(len(PATTERN_LIBRARY), size=slot_count, replace=False)
    return [PATTERN_LIBRARY[int(idx)] for idx in choice]


def _asymmetric_counts(patterns: list[np.ndarray]) -> tuple[int, int, int]:
    lr = sum(not np.array_equal(p, np.fliplr(p)) for p in patterns)
    ud = sum(not np.array_equal(p, np.flipud(p)) for p in patterns)
    scale = sum(not np.array_equal(_expand_slot(p.astype(int)), np.kron(p.astype(int), np.ones((2, 2), dtype=int))) for p in patterns)
    return lr, ud, scale


def _assemble_expected(slot_grids: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    out_h = rows * EXPANDED_SIZE + (rows - 1)
    out_w = cols * EXPANDED_SIZE + (cols - 1)
    out = np.full((out_h, out_w), BG, dtype=int)
    for idx, slot in enumerate(slot_grids):
        br, bc = divmod(idx, cols)
        rr = br * OUTPUT_STRIDE
        cc = bc * OUTPUT_STRIDE
        out[rr : rr + EXPANDED_SIZE, cc : cc + EXPANDED_SIZE] = _expand_slot(slot)
    return out


def _wrong_whole_array_transform(grid: np.ndarray, row_runs: list[tuple[int, int]], col_runs: list[tuple[int, int]]) -> np.ndarray:
    dense_rows = np.concatenate([np.arange(r0, r1) for r0, r1 in row_runs])
    dense_cols = np.concatenate([np.arange(c0, c1) for c0, c1 in col_runs])
    compact = grid[np.ix_(dense_rows, dense_cols)]
    return _expand_slot(compact)


def _validate_example(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, row_runs, col_runs = _detect_slots(grid)
    except AssertionError:
        return False

    if bg != BG:
        return False

    slot_patterns: list[np.ndarray] = []
    dense_rows = {r for r0, r1 in row_runs for r in range(r0, r1)}
    dense_cols = {c for c0, c1 in col_runs for c in range(c0, c1)}
    slot_mask = np.zeros_like(grid, dtype=bool)
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            slot_mask[r0:r1, c0:c1] = True
            slot = grid[r0:r1, c0:c1]
            mask = slot != BG
            if not mask.any():
                return False
            if np.any(mask.sum(axis=0) == 0) or np.any(mask.sum(axis=1) == 0):
                return False
            colors = set(map(int, slot[mask].ravel()))
            if len(colors) != 1:
                return False
            slot_patterns.append(slot.copy())

    signatures = {slot.astype(np.uint8).tobytes() for slot in slot_patterns}
    if len(signatures) != len(slot_patterns):
        return False

    noise_coords = np.argwhere((grid != BG) & ~slot_mask)
    if len(noise_coords) < 2:
        return False
    if any(int(r) in dense_rows or int(c) in dense_cols for r, c in noise_coords):
        return False
    if len({int(r) for r, _ in noise_coords}) != len(noise_coords):
        return False
    if len({int(c) for _, c in noise_coords}) != len(noise_coords):
        return False

    lr_asym, ud_asym, scale_asym = _asymmetric_counts([slot != BG for slot in slot_patterns])
    if lr_asym == 0 or ud_asym == 0 or scale_asym == 0:
        return False

    if np.array_equal(expected, _wrong_whole_array_transform(grid, row_runs, col_runs)):
        return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    layout = _sample_layout(rng)
    if layout is None:
        return None
    height, width, row_starts, col_starts = layout
    slot_rows = len(row_starts)
    slot_cols = len(col_starts)
    slot_count = slot_rows * slot_cols

    slot_masks = _choose_slot_masks(slot_count, rng)
    lr_asym, ud_asym, scale_asym = _asymmetric_counts(slot_masks)
    if lr_asym == 0 or ud_asym == 0 or scale_asym == 0:
        return None

    color_count = int(rng.integers(2, min(6, slot_count, len(PALETTE)) + 1))
    colors = rng.choice(PALETTE, size=color_count, replace=False)
    slot_colors = rng.choice(colors, size=slot_count, replace=True)
    slot_colors[:color_count] = colors
    rng.shuffle(slot_colors)

    grid = np.full((height, width), BG, dtype=int)
    slot_grids: list[np.ndarray] = []
    for idx, (r0, c0) in enumerate((rc for rc in ((r, c) for r in row_starts for c in col_starts))):
        slot = np.where(slot_masks[idx], int(slot_colors[idx]), BG).astype(int)
        grid[r0 : r0 + SLOT_SIZE, c0 : c0 + SLOT_SIZE] = slot
        slot_grids.append(slot)

    dense_rows = {r for r0 in row_starts for r in range(r0, r0 + SLOT_SIZE)}
    dense_cols = {c for c0 in col_starts for c in range(c0, c0 + SLOT_SIZE)}
    free_rows = [r for r in range(height) if r not in dense_rows]
    free_cols = [c for c in range(width) if c not in dense_cols]
    max_noise = min(len(free_rows), len(free_cols), 8)
    if max_noise < 2:
        return None
    noise_count = int(rng.integers(2, max_noise + 1))
    noise_rows = rng.choice(free_rows, size=noise_count, replace=False)
    noise_cols = rng.choice(free_cols, size=noise_count, replace=False)
    rng.shuffle(noise_cols)
    for r, c in zip(noise_rows, noise_cols):
        grid[int(r), int(c)] = int(rng.choice(colors))

    expected = _assemble_expected(slot_grids, slot_rows, slot_cols)
    if not _validate_example(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dense-slot mirror-expansion input")
