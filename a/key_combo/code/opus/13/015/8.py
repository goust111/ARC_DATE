from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
BARRIER = 6

MARKER_COLORS = (BLUE, YELLOW)
GEN_RETRIES = 400


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == RED)
    assert coords.size > 0, "expected a red frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 4 and c1 - c0 >= 6, "frame must have a visible interior"
    assert np.all(grid[r0, c0 : c1 + 1] == RED)
    assert np.all(grid[r1, c0 : c1 + 1] == RED)
    assert np.all(grid[r0 : r1 + 1, c0] == RED)
    assert np.all(grid[r0 : r1 + 1, c1] == RED)
    assert not np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == RED), "frame must be hollow"
    return r0, c0, r1, c1


def _infer_periodic_columns(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = frame
    inner_cols = list(range(c0 + 1, c1))
    observed = []
    for c in inner_cols:
        col = grid[r0 + 1 : r1, c]
        observed.append(bool(np.all(col == GREEN)))
    observed_mask = np.array(observed, dtype=bool)
    assert observed_mask.any() and (~observed_mask).any(), "frame stripes need both bars and gaps"

    candidates: list[tuple[int, set[int]]] = []
    offsets = [c - c0 for c in inner_cols]
    for period in (2, 3):
        residues = {offset % period for offset, active in zip(offsets, observed_mask) if active}
        if not residues:
            continue
        rebuilt = np.array([(offset % period) in residues for offset in offsets], dtype=bool)
        if np.array_equal(rebuilt, observed_mask):
            candidates.append((period, residues))

    assert len(candidates) == 1, "period mask must be uniquely inferable from the frame"
    period, residues = candidates[0]
    return [c for c in range(grid.shape[1]) if ((c - c0) % period) in residues]


def _segments_above_frame(
    grid: np.ndarray, c: int, top_row: int
) -> list[tuple[int, int]]:
    barrier_rows = [r for r in range(top_row, grid.shape[0]) if int(grid[r, c]) == BARRIER]
    segments: list[tuple[int, int]] = []
    start = top_row
    for barrier_row in barrier_rows + [grid.shape[0]]:
        if start <= barrier_row - 1:
            segments.append((start, barrier_row - 1))
        start = barrier_row + 1
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select periodic columns from the framed stripe mask, then propagate colors through each chosen column with barrier-triggered direction flips.

    Input structure:
    - A black background contains a hollow red rectangle near the upper-left.
    - Green pixels fill full interior columns of that frame on a strict period-2 or period-3 pattern.
    - Below the frame, every selected propagation column contains a lowest blue or yellow marker, additional blue/yellow divider pixels, and optional magenta barrier pixels.

    Rule:
    - Infer the periodic column mask from the green frame stripes and extend that mask across the full grid width.
    - Only those matching columns are propagated, and propagation is confined to rows below the frame.
    - In each selected column, start from the lowest blue/yellow marker with its color, fill background cells along the current segment direction, update the current color when another blue/yellow pixel is encountered, and keep barrier pixels unchanged.
    - Barrier pixels split the column into segments; the bottom segment propagates upward, and each higher segment alternates direction.

    Key insight:
    - The frame contributes only a periodic selector for which columns matter, while the barrier rows independently decide how each chosen column alternates its fill direction segment by segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _find_frame(g)
    _r0, _c0, frame_bottom, _c1 = frame
    selected_cols = _infer_periodic_columns(g, frame)
    out = g.copy()

    top_row = frame_bottom + 1
    for c in selected_cols:
        marker_rows = [r for r in range(top_row, g.shape[0]) if int(g[r, c]) in MARKER_COLORS]
        assert marker_rows, "each selected column must contain a starting marker"
        start_row = max(marker_rows)
        current = int(g[start_row, c])

        segments = _segments_above_frame(g, c, top_row)
        start_idx = None
        for idx, (lo, hi) in enumerate(segments):
            if lo <= start_row <= hi:
                start_idx = idx
                break
        assert start_idx is not None, "start marker must lie in a non-barrier segment"

        for step, idx in enumerate(range(start_idx, -1, -1)):
            lo, hi = segments[idx]
            if step == 0:
                rows = range(start_row, lo - 1, -1)
            elif step % 2 == 1:
                rows = range(lo, hi + 1)
            else:
                rows = range(hi, lo - 1, -1)

            for r in rows:
                color = int(g[r, c])
                if color == BG:
                    out[r, c] = current
                elif color in MARKER_COLORS:
                    current = color
                else:
                    assert color != BARRIER, "barriers must not appear inside a segment"

    return out


def _opposite_color(color: int) -> int:
    return YELLOW if color == BLUE else BLUE


def _random_composition(total: int, parts: int, minimum: int) -> list[int]:
    values = [minimum] * parts
    extra = total - minimum * parts
    for _ in range(extra):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _scan_order_positions(
    segments: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for step, seg_idx in enumerate(range(len(segments) - 1, -1, -1)):
        lo, hi = segments[seg_idx]
        if step % 2 == 0:
            rows = range(hi, lo - 1, -1)
        else:
            rows = range(lo, hi + 1)
        for r in rows:
            positions.append((seg_idx, r))
    return positions


def _segment_marker_candidates(
    segments: list[tuple[int, int]], seg_idx: int, bottom_row: int
) -> list[int]:
    lo, hi = segments[seg_idx]
    candidates = list(range(lo, hi + 1))
    if seg_idx > 0 and lo in candidates:
        candidates.remove(lo)
    if seg_idx < len(segments) - 1 and hi in candidates:
        candidates.remove(hi)
    if bottom_row in candidates:
        candidates.remove(bottom_row)
    if candidates:
        return candidates
    return [r for r in range(lo, hi + 1) if r != bottom_row]


def _place_column_features(
    grid: np.ndarray,
    c: int,
    top_row: int,
    obstacle_count: int,
) -> tuple[int, list[int]]:
    usable_len = grid.shape[0] - top_row
    segment_count = obstacle_count + 1
    lengths = _random_composition(usable_len - obstacle_count, segment_count, 3)

    segments: list[tuple[int, int]] = []
    row = top_row
    for seg_idx, seg_len in enumerate(lengths):
        lo = row
        hi = row + seg_len - 1
        segments.append((lo, hi))
        row = hi + 1
        if seg_idx < obstacle_count:
            grid[row, c] = BARRIER
            row += 1
    assert row == grid.shape[0]

    bottom_row = grid.shape[0] - 1
    bottom_color = random.choice(MARKER_COLORS)
    grid[bottom_row, c] = bottom_color

    marker_count = random.randint(2, min(4, usable_len - obstacle_count - 1))
    required_segments = {len(segments) - 1}
    if obstacle_count >= 1:
        required_segments.add(0)
    if obstacle_count >= 2 and random.random() < 0.6:
        required_segments.add(random.randrange(1, len(segments) - 1))

    chosen_rows: list[tuple[int, int]] = []
    used_rows = {bottom_row}
    for seg_idx in sorted(required_segments):
        candidates = [r for r in _segment_marker_candidates(segments, seg_idx, bottom_row) if r not in used_rows]
        if not candidates:
            continue
        row = random.choice(candidates)
        chosen_rows.append((seg_idx, row))
        used_rows.add(row)

    all_candidates: list[tuple[int, int]] = []
    for seg_idx in range(len(segments)):
        for row in _segment_marker_candidates(segments, seg_idx, bottom_row):
            if row not in used_rows:
                all_candidates.append((seg_idx, row))
    random.shuffle(all_candidates)
    for seg_idx, row in all_candidates:
        if len(chosen_rows) >= marker_count:
            break
        chosen_rows.append((seg_idx, row))
        used_rows.add(row)

    assert len(chosen_rows) >= 2, "each column needs visible divider markers"
    order_index = {
        pair: i for i, pair in enumerate(_scan_order_positions(segments))
    }
    chosen_rows.sort(key=lambda pair: order_index[pair])

    current = bottom_color
    changed = False
    for i, pair in enumerate(chosen_rows):
        if i == 0 or random.random() < 0.75:
            color = _opposite_color(current)
        else:
            color = current
        if color != current:
            changed = True
        grid[pair[1], c] = color
        current = color

    if not changed:
        pair = random.choice(chosen_rows)
        grid[pair[1], c] = _opposite_color(bottom_color)

    return bottom_color, [int(grid[r, c]) for _seg_idx, r in chosen_rows]


def _gen_attempt() -> np.ndarray | None:
    period = random.choice((2, 3))
    width = random.randint(14, 18) if period == 2 else random.randint(16, 22)
    height = random.randint(18, 24)

    frame_height = random.randint(5, 6)
    frame_width = random.randint(8, 10)
    frame_top = random.randint(1, 3)
    frame_left = random.randint(1, 3)
    frame_bottom = frame_top + frame_height - 1
    frame_right = frame_left + frame_width - 1
    if frame_bottom + 9 >= height or frame_right >= width:
        return None

    inner_offsets = [offset for offset in range(1, frame_width - 1)]
    valid_residues = [
        residue
        for residue in range(period)
        if sum((offset % period) == residue for offset in inner_offsets) >= 2
        and sum((offset % period) != residue for offset in inner_offsets) >= 2
    ]
    if not valid_residues:
        return None
    residue = random.choice(valid_residues)

    grid = np.zeros((height, width), dtype=int)
    grid[frame_top, frame_left : frame_right + 1] = RED
    grid[frame_bottom, frame_left : frame_right + 1] = RED
    grid[frame_top : frame_bottom + 1, frame_left] = RED
    grid[frame_top : frame_bottom + 1, frame_right] = RED
    for c in range(frame_left + 1, frame_right):
        if (c - frame_left) % period == residue:
            grid[frame_top + 1 : frame_bottom, c] = GREEN

    selected_cols = [c for c in range(width) if ((c - frame_left) % period) == residue]
    if len(selected_cols) < 4:
        return None

    top_row = frame_bottom + 1
    usable_len = height - top_row
    allowed_obstacles = [k for k in range(4) if usable_len >= 4 * k + 3]
    if not allowed_obstacles:
        return None

    planned_counts = []
    for base in (0, 1, 2, 3):
        if base in allowed_obstacles:
            planned_counts.append(base)
    while len(planned_counts) < len(selected_cols):
        planned_counts.append(random.choice(allowed_obstacles))
    random.shuffle(planned_counts)
    planned_counts = planned_counts[: len(selected_cols)]

    bottom_colors: list[int] = []
    extra_colors: list[int] = []
    for c, obstacle_count in zip(selected_cols, planned_counts):
        bottom_color, marker_colors = _place_column_features(grid, c, top_row, obstacle_count)
        bottom_colors.append(bottom_color)
        extra_colors.extend(marker_colors)

    present_marker_colors = set(bottom_colors + extra_colors)
    if present_marker_colors != set(MARKER_COLORS):
        return None
    if 0 not in planned_counts or max(planned_counts) == 0:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    changed = int(np.count_nonzero(out != grid))
    if changed < 3 * len(selected_cols):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-mask propagation instance")
