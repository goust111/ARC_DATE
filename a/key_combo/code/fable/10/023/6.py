from __future__ import annotations

from collections import Counter
import random

import numpy as np


MASK_SIZE = 3
SLOT_STRIDE = 4
BG = 0
FILL = 5
HEADER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
NON_CORNERS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
HEADER_COUNT_MIN = 3
HEADER_COUNT_MAX = 6
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask_1d.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d)))
    return runs


def _corner_bits(slot: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    return tuple(int(slot[r, c] != bg) for r, c in CORNERS)


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in mask.ravel())


def _bit_and(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(int(x and y) for x, y in zip(a, b))


def _parse_structure(grid: np.ndarray) -> tuple[int, int, int, list[int], list[int], list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _get_bg(g)
    row_runs = _runs(np.any(g != bg, axis=1))
    col_runs = _runs(np.any(g != bg, axis=0))
    assert len(row_runs) >= 2 and len(col_runs) >= 2, "expected a header row and a header column"
    assert all(end - start == MASK_SIZE for start, end in row_runs), "every occupied row band must be 3 cells tall"
    assert all(end - start == MASK_SIZE for start, end in col_runs), "every occupied column band must be 3 cells wide"
    assert all(row_runs[i + 1][0] - row_runs[i][0] == SLOT_STRIDE for i in range(len(row_runs) - 1)), "row bands must be evenly spaced"
    assert all(col_runs[i + 1][0] - col_runs[i][0] == SLOT_STRIDE for i in range(len(col_runs) - 1)), "column bands must be evenly spaced"

    row_slot_counts = [
        sum(int(np.any(g[r0:r1, c0:c1] != bg)) for c0, c1 in col_runs)
        for r0, r1 in row_runs
    ]
    col_slot_counts = [
        sum(int(np.any(g[r0:r1, c0:c1] != bg)) for r0, r1 in row_runs)
        for c0, c1 in col_runs
    ]

    header_row_idx = int(np.argmax(row_slot_counts))
    header_col_idx = int(np.argmax(col_slot_counts))
    assert row_slot_counts[header_row_idx] >= 2, "top header band must contain multiple masks"
    assert col_slot_counts[header_col_idx] >= 2, "left header band must contain multiple masks"
    assert all(
        count == 1 for i, count in enumerate(row_slot_counts) if i != header_row_idx
    ), "non-header row bands must hold exactly one left mask"
    assert all(
        count == 1 for i, count in enumerate(col_slot_counts) if i != header_col_idx
    ), "non-header column bands must hold exactly one top mask"

    header_r0, header_r1 = row_runs[header_row_idx]
    header_c0, header_c1 = col_runs[header_col_idx]
    assert not np.any(g[header_r0:header_r1, header_c0:header_c1] != bg), "top-left corner slot must be empty"

    body_rows = [start for i, (start, _end) in enumerate(row_runs) if i != header_row_idx]
    body_cols = [start for i, (start, _end) in enumerate(col_runs) if i != header_col_idx]
    assert HEADER_COUNT_MIN <= len(body_rows) <= HEADER_COUNT_MAX
    assert HEADER_COUNT_MIN <= len(body_cols) <= HEADER_COUNT_MAX

    row_masks: list[tuple[int, int, int, int]] = []
    for r0 in body_rows:
        slot = g[r0 : r0 + MASK_SIZE, header_c0 : header_c0 + MASK_SIZE]
        assert np.any(slot != bg), "each left header slot must contain one mask"
        row_masks.append(_corner_bits(slot, bg))

    col_masks: list[tuple[int, int, int, int]] = []
    for c0 in body_cols:
        slot = g[header_r0 : header_r0 + MASK_SIZE, c0 : c0 + MASK_SIZE]
        assert np.any(slot != bg), "each top header slot must contain one mask"
        col_masks.append(_corner_bits(slot, bg))

    for r0 in body_rows:
        for c0 in body_cols:
            assert np.all(g[r0 : r0 + MASK_SIZE, c0 : c0 + MASK_SIZE] == bg), "interior slots must start as background"

    return bg, header_r0, header_c0, body_rows, body_cols, row_masks, col_masks


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each interior 3x3 cell region with the cornerwise intersection of its row and column header masks.

    Input structure:
    - The top edge contains a row of colored 3x3 masks, and the left edge contains a column of colored 3x3 masks.
    - Each header mask sits in its own 3x3 slot, with one background cell separating neighboring slots.
    - The top-left 3x3 corner slot is empty, and the interior aligned slots are blank 3x3 cell regions.

    Rule:
    - Keep the headers and separators unchanged.
    - For each interior 3x3 cell region, compare the left header mask for that row with the top header mask for that column.
    - Paint a grey pixel at a cell-region corner exactly when both corresponding header masks occupy that same corner; leave every other pixel as background.

    Key insight:
    - The meaningful header value is neither color nor full silhouette; it is the 4-bit pattern of occupied corners, and every interior cell is the bitwise AND table of those two corner patterns.
    """
    g = np.asarray(grid, dtype=int)
    bg, _header_r0, _header_c0, body_rows, body_cols, row_masks, col_masks = _parse_structure(g)

    out = g.copy()
    for i, r0 in enumerate(body_rows):
        for j, c0 in enumerate(body_cols):
            bits = _bit_and(row_masks[i], col_masks[j])
            for bit, (dr, dc) in zip(bits, CORNERS):
                if bit:
                    out[r0 + dr, c0 + dc] = FILL
    assert FILL != bg, "fill color must differ from the background"
    return out


def _all_bit_patterns() -> list[tuple[int, int, int, int]]:
    return [tuple((value >> shift) & 1 for shift in (3, 2, 1, 0)) for value in range(16)]


def _candidate_masks_by_bits() -> dict[tuple[int, int, int, int], list[np.ndarray]]:
    candidates: dict[tuple[int, int, int, int], list[np.ndarray]] = {}
    for bits in _all_bit_patterns():
        masks: list[np.ndarray] = []
        for extra_mask in range(1, 1 << len(NON_CORNERS)):
            occ = np.zeros((MASK_SIZE, MASK_SIZE), dtype=int)
            for bit, (r, c) in zip(bits, CORNERS):
                if bit:
                    occ[r, c] = 1
            for idx, (r, c) in enumerate(NON_CORNERS):
                if (extra_mask >> idx) & 1:
                    occ[r, c] = 1
            if np.any(occ.sum(axis=0) == 0) or np.any(occ.sum(axis=1) == 0):
                continue
            if occ.sum() > 7:
                continue
            masks.append(occ)
        assert masks, f"expected mask candidates for bit pattern {bits}"
        candidates[bits] = masks
    return candidates


MASK_CANDIDATES = _candidate_masks_by_bits()


def _row_and_col_signatures(
    row_bits: list[tuple[int, int, int, int]],
    col_bits: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[tuple[int, int, int, int], ...]], list[tuple[tuple[int, int, int, int], ...]]]:
    row_sigs = [tuple(_bit_and(rbits, cbits) for cbits in col_bits) for rbits in row_bits]
    col_sigs = [tuple(_bit_and(rbits, cbits) for rbits in row_bits) for cbits in col_bits]
    return row_sigs, col_sigs


def _same_count_pair_exists(
    bits_a: list[tuple[int, int, int, int]],
    bits_b: list[tuple[int, int, int, int]],
) -> bool:
    for i in range(len(bits_a)):
        for j in range(i + 1, len(bits_a)):
            if sum(bits_a[i]) != sum(bits_a[j]) or bits_a[i] == bits_a[j]:
                continue
            sig_i = tuple(_bit_and(bits_a[i], other) for other in bits_b)
            sig_j = tuple(_bit_and(bits_a[j], other) for other in bits_b)
            if sig_i != sig_j:
                return True
    return False


def _bits_valid(
    row_bits: list[tuple[int, int, int, int]],
    col_bits: list[tuple[int, int, int, int]],
) -> bool:
    all_bits = row_bits + col_bits
    counts = [sum(bits) for bits in all_bits]
    if len(set(counts)) < 3:
        return False

    for corner_idx in range(4):
        if len({bits[corner_idx] for bits in row_bits}) < 2:
            return False
        if len({bits[corner_idx] for bits in col_bits}) < 2:
            return False

    cell_patterns = {_bit_and(rbits, cbits) for rbits in row_bits for cbits in col_bits}
    if len(cell_patterns) < 5:
        return False
    if len({pattern for pattern in cell_patterns if any(pattern)}) < 2:
        return False
    for corner_idx in range(4):
        if {pattern[corner_idx] for pattern in cell_patterns} != {0, 1}:
            return False

    row_sigs, col_sigs = _row_and_col_signatures(row_bits, col_bits)
    if len(set(row_sigs)) != len(row_sigs):
        return False
    if len(set(col_sigs)) != len(col_sigs):
        return False
    if not all(len(set(sig)) >= 2 for sig in row_sigs):
        return False
    if not all(len(set(sig)) >= 2 for sig in col_sigs):
        return False

    if not (_same_count_pair_exists(row_bits, col_bits) or _same_count_pair_exists(col_bits, row_bits)):
        return False
    return True


def _sample_bits(n_rows: int, n_cols: int) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]] | None:
    total = n_rows + n_cols
    all_patterns = _all_bit_patterns()
    for _ in range(200):
        chosen = random.sample(all_patterns, total)
        random.shuffle(chosen)
        row_bits = chosen[:n_rows]
        col_bits = chosen[n_rows:]
        if _bits_valid(row_bits, col_bits):
            return row_bits, col_bits
    return None


def _colors_valid(bits: list[tuple[int, int, int, int]], colors: list[int]) -> bool:
    counts = [sum(bit) for bit in bits]
    same_color_diff_count = any(
        colors[i] == colors[j] and counts[i] != counts[j] and bits[i] != bits[j]
        for i in range(len(bits))
        for j in range(i + 1, len(bits))
    )
    same_count_diff_color = any(
        counts[i] == counts[j] and colors[i] != colors[j] and bits[i] != bits[j]
        for i in range(len(bits))
        for j in range(i + 1, len(bits))
    )
    return same_color_diff_count and same_count_diff_color


def _sample_colors(bits: list[tuple[int, int, int, int]]) -> list[int] | None:
    max_palette = min(6, len(HEADER_COLORS), len(bits))
    min_palette = min(3, max_palette)
    for _ in range(200):
        palette_size = random.randint(min_palette, max_palette)
        palette = random.sample(list(HEADER_COLORS), palette_size)
        colors = [random.choice(palette) for _ in bits]
        if len(set(colors)) == len(colors):
            continue
        if _colors_valid(bits, colors):
            return colors
    return None


def _sample_mask_shapes(bits: list[tuple[int, int, int, int]]) -> list[np.ndarray] | None:
    used: set[tuple[int, ...]] = set()
    masks: list[np.ndarray] = []
    for bit_pattern in bits:
        choices = MASK_CANDIDATES[bit_pattern][:]
        random.shuffle(choices)
        picked: np.ndarray | None = None
        for mask in choices:
            sig = _mask_signature(mask)
            if sig not in used:
                picked = mask
                used.add(sig)
                break
        if picked is None:
            return None
        masks.append(picked.copy())
    return masks


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    view = grid[r0 : r0 + MASK_SIZE, c0 : c0 + MASK_SIZE]
    view[mask.astype(bool)] = int(color)


def _build_grid(
    row_bits: list[tuple[int, int, int, int]],
    col_bits: list[tuple[int, int, int, int]],
    row_masks: list[np.ndarray],
    col_masks: list[np.ndarray],
    row_colors: list[int],
    col_colors: list[int],
) -> np.ndarray:
    n_rows = len(row_bits)
    n_cols = len(col_bits)
    height = SLOT_STRIDE * n_rows + MASK_SIZE
    width = SLOT_STRIDE * n_cols + MASK_SIZE
    grid = np.full((height, width), BG, dtype=int)

    for j, (mask, color) in enumerate(zip(col_masks, col_colors)):
        _paint_mask(grid, 0, SLOT_STRIDE * (j + 1), mask, color)
    for i, (mask, color) in enumerate(zip(row_masks, row_colors)):
        _paint_mask(grid, SLOT_STRIDE * (i + 1), 0, mask, color)

    return grid


def _expected_output_from_bits(
    grid: np.ndarray,
    row_bits: list[tuple[int, int, int, int]],
    col_bits: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for i, rbits in enumerate(row_bits):
        for j, cbits in enumerate(col_bits):
            bits = _bit_and(rbits, cbits)
            r0 = SLOT_STRIDE * (i + 1)
            c0 = SLOT_STRIDE * (j + 1)
            for bit, (dr, dc) in zip(bits, CORNERS):
                if bit:
                    out[r0 + dr, c0 + dc] = FILL
    return out


def _semantic_checks(
    grid: np.ndarray,
    out: np.ndarray,
    row_bits: list[tuple[int, int, int, int]],
    col_bits: list[tuple[int, int, int, int]],
    row_colors: list[int],
    col_colors: list[int],
    row_masks: list[np.ndarray],
    col_masks: list[np.ndarray],
) -> bool:
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if np.count_nonzero(out == FILL) == 0:
        return False

    all_bits = row_bits + col_bits
    all_colors = row_colors + col_colors
    all_masks = row_masks + col_masks

    if len({_mask_signature(mask) for mask in all_masks}) != len(all_masks):
        return False
    if not _colors_valid(all_bits, all_colors):
        return False
    if not _bits_valid(row_bits, col_bits):
        return False

    # Exact corner positions, not just corner counts, must matter.
    same_count_position_decoy = False
    for i in range(len(all_bits)):
        for j in range(i + 1, len(all_bits)):
            if sum(all_bits[i]) == sum(all_bits[j]) and all_bits[i] != all_bits[j]:
                same_count_position_decoy = True
    if not same_count_position_decoy:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows = random.randint(HEADER_COUNT_MIN, HEADER_COUNT_MAX)
    n_cols = random.randint(HEADER_COUNT_MIN, HEADER_COUNT_MAX)

    sampled = _sample_bits(n_rows, n_cols)
    if sampled is None:
        return None
    row_bits, col_bits = sampled

    row_colors = _sample_colors(row_bits)
    col_colors = _sample_colors(col_bits)
    if row_colors is None or col_colors is None:
        return None

    row_masks = _sample_mask_shapes(row_bits)
    col_masks = _sample_mask_shapes(col_bits)
    if row_masks is None or col_masks is None:
        return None

    grid = _build_grid(row_bits, col_bits, row_masks, col_masks, row_colors, col_colors)
    expected = _expected_output_from_bits(grid, row_bits, col_bits)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(grid, actual, row_bits, col_bits, row_colors, col_colors, row_masks, col_masks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-header-value input")
