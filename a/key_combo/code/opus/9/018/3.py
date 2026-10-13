from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
FOREGROUND_POOL = (2, 3, 7, 8)


def _mode_color(values: Iterable[int]) -> int:
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _split_intervals(separators: list[int], size: int) -> list[tuple[int, int]]:
    edges = [-1, *sorted(int(x) for x in separators), int(size)]
    intervals: list[tuple[int, int]] = []
    for left, right in zip(edges, edges[1:]):
        if right - left > 1:
            intervals.append((left + 1, right))
    return intervals


def _extract_blocks(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_rows = [r for r in range(h) if bool(np.all(g[r, :] == GREY))]
    sep_cols = [c for c in range(w) if bool(np.all(g[:, c] == GREY))]
    row_intervals = _split_intervals(sep_rows, h)
    col_intervals = _split_intervals(sep_cols, w)
    assert len(row_intervals) == 3 and len(col_intervals) == 3, "expected a 3x3 separator grid"

    block_heights = {r1 - r0 for r0, r1 in row_intervals}
    block_widths = {c1 - c0 for c0, c1 in col_intervals}
    assert len(block_heights) == 1 and len(block_widths) == 1, "all blocks must share one size"
    assert next(iter(block_heights)) == next(iter(block_widths)), "motif blocks must be square"

    blocks: list[np.ndarray] = []
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            blocks.append(g[r0:r1, c0:c1].copy())
    return blocks


def _consensus_block(blocks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(blocks, axis=0)
    _, bh, bw = stack.shape
    out = np.zeros((bh, bw), dtype=int)
    for r in range(bh):
        for c in range(bw):
            out[r, c] = _mode_color(stack[:, r, c])
    return out


def _segments_from_scaled_pattern(block: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(block, dtype=int)
    bh, bw = g.shape

    row_changes = [r for r in range(bh - 1) if bool(np.any(g[r, :] != g[r + 1, :]))]
    col_changes = [c for c in range(bw - 1) if bool(np.any(g[:, c] != g[:, c + 1]))]

    row_starts = [0, *[r + 1 for r in row_changes]]
    row_ends = [*[r + 1 for r in row_changes], bh]
    col_starts = [0, *[c + 1 for c in col_changes]]
    col_ends = [*[c + 1 for c in col_changes], bw]
    row_segments = list(zip(row_starts, row_ends))
    col_segments = list(zip(col_starts, col_ends))

    assert row_segments and col_segments, "compressed grid must be non-empty"
    assert min(r1 - r0 for r0, r1 in row_segments) >= 2, "each scaled row band must be at least 2 high"
    assert min(c1 - c0 for c0, c1 in col_segments) >= 2, "each scaled col band must be at least 2 wide"

    row_sizes = {r1 - r0 for r0, r1 in row_segments}
    col_sizes = {c1 - c0 for c0, c1 in col_segments}
    assert len(row_sizes) == 1 and len(col_sizes) == 1, "expected one global rectangular scale factor"

    for r0, r1 in row_segments:
        for rr in range(r0 + 1, r1):
            assert bool(np.array_equal(g[r0, :], g[rr, :])), "rows inside one scaled band must match"
    for c0, c1 in col_segments:
        for cc in range(c0 + 1, c1):
            assert bool(np.array_equal(g[:, c0], g[:, cc])), "cols inside one scaled band must match"

    return row_segments, col_segments


def _segment_index(segments: list[tuple[int, int]], pos: int) -> int:
    for idx, (start, end) in enumerate(segments):
        if int(start) <= int(pos) < int(end):
            return int(idx)
    raise AssertionError(f"position {pos} outside of segments")


def _compress_and_mark(consensus: np.ndarray, damaged_pixels: np.ndarray) -> np.ndarray:
    row_segments, col_segments = _segments_from_scaled_pattern(consensus)
    out = np.zeros((len(row_segments), len(col_segments)), dtype=int)

    for i, (r0, r1) in enumerate(row_segments):
        for j, (c0, c1) in enumerate(col_segments):
            patch = consensus[r0:r1, c0:c1]
            color = int(patch[0, 0])
            assert bool(np.all(patch == color)), "each logical cell must scale to one solid rectangle"
            out[i, j] = color

    for r, c in np.argwhere(damaged_pixels):
        i = _segment_index(row_segments, int(r))
        j = _segment_index(col_segments, int(c))
        out[i, j] = BLUE

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a damaged repeated motif by voting, compress the repaired motif, and encode damaged logical cells in blue.

    Input structure:
    - A grey separator grid divides the canvas into a 3x3 array of equal square blocks.
    - Each block is one copy of the same block-scaled motif, using black plus 2-3 foreground colors.
    - Exactly one copy is damaged by turning a few non-black pixels into black.

    Rule:
    - Split the input into its 9 blocks and take the majority color at every local block coordinate to recover the consensus motif.
    - Find the unique block whose pixels disagree with that consensus and record every mismatching local coordinate.
    - Detect the motif's logical row and column bands from the repaired block-scaled rectangles, then compress each uniform rectangle to one logical cell.
    - Recolor each compressed logical cell blue if any recorded damage coordinate falls inside its rectangle; otherwise keep the consensus color.

    Key insight:
    - The damaged copy identifies which logical rectangles should become blue, but only the vote-repaired consensus reveals the true block partition needed to decode those positions.
    """
    blocks = _extract_blocks(np.asarray(grid, dtype=int))
    consensus = _consensus_block(blocks)
    mismatch_masks = [block != consensus for block in blocks]
    damaged_ids = [idx for idx, mask in enumerate(mismatch_masks) if bool(mask.any())]
    assert len(damaged_ids) == 1, "expected exactly one damaged replica"
    damaged_mask = mismatch_masks[damaged_ids[0]]
    return _compress_and_mark(consensus, damaged_mask)


SPEC_OPTIONS = (
    (2, 2, 2, 2),
    (2, 2, 3, 3),
    (2, 2, 4, 4),
    (3, 3, 2, 2),
    (4, 4, 2, 2),
    (2, 3, 3, 2),
    (3, 2, 2, 3),
    (2, 4, 4, 2),
    (4, 2, 2, 4),
)
MAX_GENERATION_ATTEMPTS = 400


def _build_logical_grid(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray | None:
    fg_count = int(rng.integers(2, 4))
    palette = [BLACK, *rng.choice(FOREGROUND_POOL, size=fg_count, replace=False).tolist()]

    for _ in range(200):
        logical = np.full((rows, cols), -1, dtype=int)
        for r in range(rows):
            for c in range(cols):
                banned = set()
                if r > 0:
                    banned.add(int(logical[r - 1, c]))
                if c > 0:
                    banned.add(int(logical[r, c - 1]))
                choices = [color for color in palette if color not in banned]
                weights = np.array([1 if color == BLACK else 3 for color in choices], dtype=float)
                weights /= weights.sum()
                logical[r, c] = int(rng.choice(choices, p=weights))

        non_black = int(np.count_nonzero(logical != BLACK))
        used_fg = sorted(set(int(v) for v in logical.flatten()) - {BLACK})
        if np.any(logical == BLACK) and non_black >= 3 and len(used_fg) >= 2:
            return logical
    return None


def _pick_damage_pixels(
    rng: np.random.Generator,
    logical: np.ndarray,
    scale_r: int,
    scale_c: int,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    candidates = [tuple(map(int, rc)) for rc in np.argwhere(logical != BLACK)]
    if len(candidates) < 2:
        return None

    cell_area = int(scale_r * scale_c)
    max_per_cell = max(1, (cell_area - 1) // 2)
    total_damage = int(rng.integers(2, 5))
    distinct_count = int(rng.integers(2, min(len(candidates), total_damage, 4) + 1))
    chosen_cells = [candidates[idx] for idx in rng.choice(len(candidates), size=distinct_count, replace=False)]

    counts = {cell: 1 for cell in chosen_cells}
    remaining = total_damage - distinct_count
    extras = [cell for cell in chosen_cells if max_per_cell > 1]
    while remaining > 0 and extras:
        cell = extras[int(rng.integers(len(extras)))]
        if counts[cell] < max_per_cell:
            counts[cell] += 1
            remaining -= 1
        extras = [candidate for candidate in extras if counts[candidate] < max_per_cell]
    if remaining > 0:
        return None

    pixel_coords: list[tuple[int, int]] = []
    selected_logical: set[tuple[int, int]] = set()
    any_non_topleft = False

    for cell_r, cell_c in chosen_cells:
        local_pixels = [
            (rr, cc)
            for rr in range(scale_r)
            for cc in range(scale_c)
        ]
        rng.shuffle(local_pixels)
        needed = counts[(cell_r, cell_c)]
        chosen_local = local_pixels[:needed]
        for rr, cc in chosen_local:
            if (rr, cc) != (0, 0):
                any_non_topleft = True
            pixel_coords.append((cell_r * scale_r + rr, cell_c * scale_c + cc))
        selected_logical.add((cell_r, cell_c))

    if not any_non_topleft:
        return None
    return pixel_coords, selected_logical


def _assemble_input(block: np.ndarray, damaged_block: np.ndarray, damaged_idx: int) -> np.ndarray:
    block_size = int(block.shape[0])
    grid = np.full((block_size * 3 + 2, block_size * 3 + 2), GREY, dtype=int)
    for idx in range(9):
        br = idx // 3
        bc = idx % 3
        r0 = br * (block_size + 1)
        c0 = bc * (block_size + 1)
        grid[r0:r0 + block_size, c0:c0 + block_size] = damaged_block if idx == damaged_idx else block
    return grid


def _expected_output(logical: np.ndarray, selected_logical: set[tuple[int, int]]) -> np.ndarray:
    out = logical.copy()
    for r, c in selected_logical:
        out[r, c] = BLUE
    return out


def _detect_shortcut_failure(consensus: np.ndarray, damaged_block: np.ndarray) -> bool:
    true_rows, true_cols = _segments_from_scaled_pattern(consensus)
    try:
        bad_rows, bad_cols = _segments_from_scaled_pattern(damaged_block)
    except AssertionError:
        return True
    return bad_rows != true_rows or bad_cols != true_cols


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    logical_rows, logical_cols, scale_r, scale_c = SPEC_OPTIONS[int(rng.integers(len(SPEC_OPTIONS)))]
    logical = _build_logical_grid(rng, logical_rows, logical_cols)
    if logical is None:
        return None

    consensus = np.kron(logical, np.ones((scale_r, scale_c), dtype=int))
    damage_info = _pick_damage_pixels(rng, logical, scale_r, scale_c)
    if damage_info is None:
        return None
    damaged_pixels, selected_logical = damage_info

    damaged_block = consensus.copy()
    for r, c in damaged_pixels:
        damaged_block[r, c] = BLACK

    if not _detect_shortcut_failure(consensus, damaged_block):
        return None

    expected = _expected_output(logical, selected_logical)
    if np.count_nonzero(expected == BLUE) >= expected.size:
        return None
    if not np.any((logical == BLACK) & (expected != BLUE)):
        return None

    damaged_idx = int(rng.integers(9))
    inp = _assemble_input(consensus, damaged_block, damaged_idx)
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = _generate_case(rng)
        if generated is None:
            continue
        inp, expected = generated
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue
        if inp.ndim != 2 or actual.ndim != 2:
            continue
        if max(inp.shape) > 30 or max(actual.shape) > 30:
            continue
        return inp
    raise RuntimeError("failed to generate a valid voting-repair compression puzzle")
