from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 2
PALETTE = (1, 3, 4, 7)
MAX_GENERATION_ATTEMPTS = 500


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive (r0, c0, r1, c1) bbox of an (N,2) coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_lattice(grid: np.ndarray) -> tuple[list[int], list[int]]:
    red = np.argwhere(grid == FRAME)
    assert red.size > 0, "expected a red lattice frame"
    r0, c0, r1, c1 = bbox_coords(red)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    row_lines = [r0 + r for r in range(sub.shape[0]) if bool(np.all(sub[r, :] == FRAME))]
    col_lines = [c0 + c for c in range(sub.shape[1]) if bool(np.all(sub[:, c] == FRAME))]
    assert len(row_lines) >= 3 and len(col_lines) >= 3
    assert row_lines[0] == r0 and row_lines[-1] == r1
    assert col_lines[0] == c0 and col_lines[-1] == c1
    assert all(a + 1 < b for a, b in zip(row_lines, row_lines[1:]))
    assert all(a + 1 < b for a, b in zip(col_lines, col_lines[1:]))

    expected = np.zeros_like(sub, dtype=bool)
    expected[[r - r0 for r in row_lines], :] = True
    expected[:, [c - c0 for c in col_lines]] = True
    assert np.array_equal(sub == FRAME, expected), "frame must be a complete rectangular lattice"

    slot_rows = len(row_lines) - 1
    slot_cols = len(col_lines) - 1
    assert slot_rows == 2 and slot_cols in (2, 3)
    return row_lines, col_lines


def _settle_slot(slot: np.ndarray) -> np.ndarray:
    h, w = slot.shape
    settled = np.full((h, w), BG, dtype=int)
    for c in range(w):
        stack = [int(slot[r, c]) for r in range(h - 1, -1, -1) if int(slot[r, c]) != BG]
        for depth, color in enumerate(stack):
            settled[h - 1 - depth, c] = int(color)
    return settled


def _row_sequence(slot: np.ndarray) -> list[int]:
    seq: list[int] = []
    for r in range(slot.shape[0] - 1, -1, -1):
        vals = sorted({int(v) for v in slot[r] if int(v) != BG})
        assert len(vals) <= 1, "each non-empty row must have exactly one foreground color"
        if vals:
            seq.append(int(vals[0]))
    assert seq, "each slot must contain foreground pixels"
    return seq


def _compress_adjacent(seq: list[int]) -> list[int]:
    out: list[int] = []
    for color in seq:
        if not out or out[-1] != int(color):
            out.append(int(color))
    assert out
    return out


def _fill_rows(height: int, width: int, seq: list[int]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for depth in range(height):
        out[height - 1 - depth, :] = int(seq[depth % len(seq)])
    return out


def _transform_slot(slot: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    settled = _settle_slot(slot)
    raw_seq = _row_sequence(settled)
    compact = _compress_adjacent(raw_seq)
    filled = _fill_rows(slot.shape[0], slot.shape[1], compact)
    return filled, settled, raw_seq, compact


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply slot-local gravity, compress the resulting color layers, and retile each slot with the compressed sequence.

    Input structure:
    - The background is black.
    - A one-pixel-thick red rectangular lattice partitions the grid into a 2x2 or 2x3 array of rectangular slots.
    - Each slot contains a few scattered blue/green/yellow/orange pixels on black.

    Rule:
    - Process each slot independently inside its own frame boundaries.
    - Let the slot pixels fall downward within each column until they stack on the slot floor or on each other.
    - Read the settled non-empty rows from bottom to top, merge adjacent equal colors, and use that compressed sequence as a repeating bottom-up stripe pattern for the whole slot.
    - Keep the red lattice and the outside black background unchanged.

    Key insight:
    - The final stripe order is not visible in the scatter itself; it is created by local gravity, then thickness is discarded by adjacent deduplication before the sequence is tiled back over the full slot.
    """
    g = np.asarray(grid, dtype=int)
    row_lines, col_lines = _parse_lattice(g)

    out = g.copy()
    for sr in range(len(row_lines) - 1):
        for sc in range(len(col_lines) - 1):
            r0 = row_lines[sr] + 1
            r1 = row_lines[sr + 1]
            c0 = col_lines[sc] + 1
            c1 = col_lines[sc + 1]
            slot = g[r0:r1, c0:c1]
            filled, _settled, _raw_seq, _compact = _transform_slot(slot)
            out[r0:r1, c0:c1] = filled
    return out


def _frame_lines(lengths: list[int]) -> list[int]:
    lines = [0]
    pos = 0
    for size in lengths:
        pos += int(size) + 1
        lines.append(int(pos))
    return lines


def _build_frame_patch(slot_heights: list[int], slot_widths: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    row_lines = _frame_lines(slot_heights)
    col_lines = _frame_lines(slot_widths)
    patch = np.full((row_lines[-1] + 1, col_lines[-1] + 1), BG, dtype=int)
    patch[row_lines, :] = FRAME
    patch[:, col_lines] = FRAME
    return patch, row_lines, col_lines


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    vals: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        vals.append(int(cut - prev))
        prev = cut
    return vals


def _choose_compact_sequence(k: int, force_duplicate: bool, force_three_colors: bool) -> tuple[list[int], list[int]]:
    assert k >= 2
    for _ in range(200):
        if force_three_colors:
            if k < 3:
                continue
            a, b, c = random.sample(PALETTE, 3)
            compact = [a, b, c]
        else:
            if k >= 3 and random.random() < 0.35:
                a, b, c = random.sample(PALETTE, 3)
                compact = [a, b, c]
            else:
                a, b = random.sample(PALETTE, 2)
                if k >= 3 and random.random() < 0.45:
                    compact = [a, b, a]
                else:
                    compact = [a, b]

        if len(compact) >= k:
            if force_duplicate:
                continue
            if len(compact) == k:
                return compact, compact[:]
            continue

        thickness = _random_composition(k, len(compact))
        if force_duplicate and max(thickness) == 1:
            continue
        raw: list[int] = []
        for color, width in zip(compact, thickness):
            raw.extend([int(color)] * int(width))
        if len(raw) == k:
            return compact, raw

    raise RuntimeError("failed to choose a valid layer sequence")


def _generate_slot(
    height: int,
    width: int,
    *,
    force_duplicate: bool,
    force_three_colors: bool,
) -> np.ndarray:
    for _ in range(300):
        if force_duplicate and force_three_colors:
            k_min = 4
        elif force_duplicate or force_three_colors:
            k_min = 3
        else:
            k_min = 2
        k_max = min(height, 4)
        if k_min > k_max:
            continue
        k = random.randint(k_min, k_max)

        max_cols = min(width, 4, height - k + 1)
        if max_cols < 2:
            continue
        used_cols = random.randint(2, max_cols)

        heights = [k] + [1] * (used_cols - 1)
        extra_budget = height - sum(heights)
        extra_total = random.randint(0, extra_budget) if extra_budget > 0 else 0
        for _ in range(extra_total):
            choices = [i for i, cur in enumerate(heights) if cur < k]
            if not choices:
                break
            heights[random.choice(choices)] += 1
        random.shuffle(heights)

        compact, raw_seq = _choose_compact_sequence(k, force_duplicate, force_three_colors)
        if height <= len(compact):
            continue
        if len(set(compact)) < 2 or len(set(compact)) > 3:
            continue

        cols = sorted(random.sample(range(width), used_cols))
        settled = np.full((height, width), BG, dtype=int)
        chains: list[list[int]] = []
        for col, col_height in zip(cols, heights):
            stack = raw_seq[:col_height]
            for depth, color in enumerate(stack):
                settled[height - 1 - depth, col] = int(color)
            chains.append(list(reversed(stack)))

        pixel_count = int(sum(heights))
        assert 3 <= pixel_count <= 8
        rows = sorted(random.sample(range(height), pixel_count))
        order: list[tuple[int, int]] = []
        temp = [chain[:] for chain in chains]
        while any(temp):
            options = [idx for idx, chain in enumerate(temp) if chain]
            idx = random.choice(options)
            order.append((int(cols[idx]), int(temp[idx].pop(0))))

        slot = np.full((height, width), BG, dtype=int)
        for row, (col, color) in zip(rows, order):
            slot[int(row), int(col)] = int(color)

        if np.array_equal(slot, settled):
            continue
        if not np.array_equal(_settle_slot(slot), settled):
            continue

        input_seq = _row_sequence(slot)
        _filled, _settled, actual_raw, actual_compact = _transform_slot(slot)
        if actual_raw != raw_seq or actual_compact != compact:
            continue
        if _compress_adjacent(input_seq) == compact and input_seq == actual_raw:
            continue
        return slot

    raise RuntimeError("failed to generate a valid slot")


def _validate_instance(inp: np.ndarray) -> None:
    out = main(inp)
    row_lines, col_lines = _parse_lattice(inp)
    assert np.array_equal(out[row_lines, :], inp[row_lines, :])
    assert np.array_equal(out[:, col_lines], inp[:, col_lines])
    assert np.all(inp[(inp != BG) & (inp != FRAME)] != FRAME)

    gravity_matters = False
    compression_matters = False
    fill_cycles = False

    for sr in range(len(row_lines) - 1):
        for sc in range(len(col_lines) - 1):
            r0 = row_lines[sr] + 1
            r1 = row_lines[sr + 1]
            c0 = col_lines[sc] + 1
            c1 = col_lines[sc + 1]

            slot = inp[r0:r1, c0:c1]
            actual_slot = out[r0:r1, c0:c1]
            filled, settled, raw_seq, compact = _transform_slot(slot)
            assert np.array_equal(actual_slot, filled)

            pixel_count = int(np.count_nonzero(slot))
            assert 3 <= pixel_count <= 8
            colors = {int(v) for v in slot.flatten() if int(v) != BG}
            assert 2 <= len(colors) <= 3
            assert slot.shape[0] > len(compact)

            input_seq = _row_sequence(slot)
            no_gravity = _fill_rows(slot.shape[0], slot.shape[1], _compress_adjacent(input_seq))
            no_compress = _fill_rows(slot.shape[0], slot.shape[1], raw_seq)

            if not np.array_equal(actual_slot, no_gravity):
                gravity_matters = True
            if not np.array_equal(actual_slot, no_compress):
                compression_matters = True
            if slot.shape[0] > len(compact):
                fill_cycles = True

            for r in range(slot.shape[0]):
                vals = {int(v) for v in slot[r] if int(v) != BG}
                assert len(vals) <= 1, "input rows should not mix colors"
            for r in range(settled.shape[0]):
                vals = {int(v) for v in settled[r] if int(v) != BG}
                assert len(vals) <= 1, "settled rows should form monochrome layers"

    assert gravity_matters, "generator must include slots where gravity changes the inferred sequence"
    assert compression_matters, "generator must include slots where dedup changes the final pattern"
    assert fill_cycles, "generator must include slots where cyclic tiling uses more than one cycle"
    assert np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    slot_cols = random.choice([2, 3])
    slot_heights = [random.randint(4, 6) for _ in range(2)]
    slot_widths = [random.randint(4, 6) for _ in range(slot_cols)]
    frame_patch, row_offsets, col_offsets = _build_frame_patch(slot_heights, slot_widths)

    top = random.randint(1, 3)
    left = random.randint(1, 3)
    bottom = random.randint(1, 3)
    right = random.randint(1, 3)
    grid_h = frame_patch.shape[0] + top + bottom
    grid_w = frame_patch.shape[1] + left + right
    if grid_h > 30 or grid_w > 30:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[top : top + frame_patch.shape[0], left : left + frame_patch.shape[1]] = frame_patch

    slots = [(sr, sc) for sr in range(2) for sc in range(slot_cols)]
    duplicate_slot = random.choice(slots)
    three_color_slot = None
    if random.random() < 0.6:
        candidates = [slot for slot in slots if slot != duplicate_slot]
        if candidates:
            three_color_slot = random.choice(candidates)

    for sr, sc in slots:
        h = slot_heights[sr]
        w = slot_widths[sc]
        try:
            slot = _generate_slot(
                h,
                w,
                force_duplicate=(sr, sc) == duplicate_slot or random.random() < 0.35,
                force_three_colors=(three_color_slot is not None and (sr, sc) == three_color_slot),
            )
        except RuntimeError:
            return None
        r0 = top + row_offsets[sr] + 1
        c0 = left + col_offsets[sc] + 1
        inp[r0 : r0 + h, c0 : c0 + w] = slot

    try:
        _validate_instance(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 5")
