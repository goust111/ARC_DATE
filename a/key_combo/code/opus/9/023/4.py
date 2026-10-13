from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
CHEVRON = 5
LINE_COLORS = (1, 2, 3, 4, 9)
GEN_RETRIES = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_segments(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    segments: list[dict] = []
    mask = (grid != BG) & (grid != CHEVRON)
    for r in range(h):
        c = 0
        while c < w:
            if not mask[r, c]:
                c += 1
                continue
            color = int(grid[r, c])
            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == color:
                c += 1
            c1 = c
            assert c1 > c0, "horizontal line segments must be length >= 2"
            segments.append(
                {
                    "row": int(r),
                    "start": int(c0),
                    "end": int(c1),
                    "length": int(c1 - c0 + 1),
                    "color": color,
                }
            )
            c += 1
    segments.sort(key=lambda seg: seg["row"])
    return segments


def _bottom_gray_by_col(grid: np.ndarray) -> dict[int, int]:
    cols: dict[int, int] = {}
    h, w = grid.shape
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] == CHEVRON)
        if len(rows) > 0:
            cols[int(c)] = int(rows.max())
    return cols


def _render_from_reference(
    grid: np.ndarray,
    segments: list[dict],
    ref_index: int,
    *,
    keep_original_lines: bool = False,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    if not keep_original_lines:
        for seg in segments:
            out[seg["row"], seg["start"] : seg["end"] + 1] = BG

    ref = segments[ref_index]
    c0 = int(ref["start"])
    target_len = int(ref["length"])
    c1 = c0 + target_len - 1
    seq = [int(seg["color"]) for seg in segments]

    if not keep_original_lines:
        for seg in segments:
            out[seg["row"], c0 : c1 + 1] = int(seg["color"])

    bottom_gray = _bottom_gray_by_col(grid)
    h = grid.shape[0]
    for c in range(c0, c1 + 1):
        if c not in bottom_gray:
            continue
        r_bottom = bottom_gray[c]
        for r in range(r_bottom + 1, h):
            out[r, c] = seq[(r - r_bottom - 1) % len(seq)]
    return out


def _gray_component(grid: np.ndarray) -> set[tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid == CHEVRON)]
    assert pts, "expected one gray chevron"
    start = pts[0]
    seen = {start}
    q = deque([start])
    point_set = set(pts)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in point_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    assert seen == point_set, "gray chevron must be one 8-connected component"
    return seen


def _validate_input_structure(grid: np.ndarray) -> tuple[list[dict], int, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    segments = _find_segments(g)
    assert len(segments) >= 3 and len(segments) % 2 == 1
    assert len({seg["row"] for seg in segments}) == len(segments)
    assert len({seg["color"] for seg in segments}) == len(segments)

    rows = [seg["row"] for seg in segments]
    ref_index = len(segments) // 2
    ref = segments[ref_index]
    ref_len = int(ref["length"])

    gray = _gray_component(g)
    gray_rows = sorted({r for r, _c in gray})
    gray_cols = sorted({c for _r, c in gray})
    assert rows[-1] + 2 < gray_rows[0], "lines and chevron need a blank gap"
    assert len(gray_rows) >= 4
    assert len(gray_cols) >= 5

    bottom_gray = _bottom_gray_by_col(g)
    active_cols = [c for c in range(ref["start"], ref["end"] + 1) if c in bottom_gray]
    assert active_cols, "reference interval must overlap the chevron"

    return segments, ref_index, active_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the horizontal color bars to the median bar, then cycle-fill only chevron-covered columns below it.

    Input structure:
    - A black background contains several 1-cell-thick horizontal line segments of different colors in the upper half.
    - A single gray chevron object sits lower in the grid, separated from the lines by blank rows.
    - The chevron and the line stack partially overlap in columns but do not touch.

    Rule:
    - Sort the horizontal line segments by row and use the median row segment as the reference.
    - Shift every segment horizontally so its start column matches the reference start; trim longer segments on the right and extend shorter ones on the right with their own color so every aligned segment matches the reference length.
    - For each column inside that aligned interval, check whether the column contains any gray chevron pixel; if so, find the lowest gray pixel in that column and fill downward from the next row to the bottom using the aligned top-to-bottom color stack as a repeating cycle.
    - Keep the aligned line segments and the gray chevron visible, and leave uncovered columns unchanged.

    Key insight:
    - The median segment defines the common aligned interval, and that alignment creates the color stack later reused as the per-column fill cycle under the chevron.
    """
    segments, ref_index, _active_cols = _validate_input_structure(grid)
    return _render_from_reference(np.asarray(grid, dtype=int), segments, ref_index)


def _make_chevron(
    top: int,
    left: int,
    arm_len: int,
    orientation: str,
) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if orientation == "caret":
        center = left + arm_len - 1
        for k in range(arm_len):
            cells.add((top + k, center - k))
            cells.add((top + k, center + k))
    else:
        center = left + arm_len - 1
        for k in range(arm_len):
            offset = arm_len - 1 - k
            cells.add((top + k, center - offset))
            cells.add((top + k, center + offset))
    return sorted(cells)


def _random_rows() -> list[int]:
    top = random.randint(1, 3)
    rows = [top]
    for _ in range(4):
        rows.append(rows[-1] + random.randint(1, 2))
    return rows


def _sample_lengths(ref_len: int, width: int) -> list[int]:
    short_pool = list(range(max(2, ref_len - 4), ref_len))
    long_pool = list(range(ref_len + 1, min(width - 2, ref_len + 5) + 1))
    assert short_pool and long_pool
    lengths = [ref_len] * 5
    non_ref = [0, 1, 3, 4]
    short_slots = random.sample(non_ref, 2)
    long_slots = [idx for idx in non_ref if idx not in short_slots]
    for idx in short_slots:
        lengths[idx] = random.choice(short_pool)
    for idx in long_slots:
        lengths[idx] = random.choice(long_pool)
    return lengths


def _sample_starts(ref_start: int, ref_len: int, lengths: list[int], width: int) -> list[int] | None:
    starts = [0] * len(lengths)
    starts[2] = ref_start
    non_ref = [0, 1, 3, 4]
    neg_slots = random.sample(non_ref, 2)
    pos_slots = [idx for idx in non_ref if idx not in neg_slots]

    for idx in neg_slots:
        max_start = width - lengths[idx]
        choices = [
            s
            for s in range(max(0, ref_start - 4), ref_start)
            if 0 <= s <= max_start
        ]
        if not choices:
            return None
        starts[idx] = random.choice(choices)

    for idx in pos_slots:
        max_start = width - lengths[idx]
        choices = [
            s
            for s in range(ref_start + 1, min(max_start, ref_start + 4) + 1)
            if 0 <= s <= max_start
        ]
        if not choices:
            return None
        starts[idx] = random.choice(choices)

    return starts


def _place_chevron_interval(
    width: int,
    ref_start: int,
    ref_len: int,
    chevron_width: int,
) -> int | None:
    ref_end = ref_start + ref_len - 1
    overlap_max = min(ref_len - 2, chevron_width - 2)
    if overlap_max < 2:
        return None

    choices: list[int] = []
    overlap = random.randint(2, overlap_max)

    left_end = ref_start + overlap - 1
    left_start = left_end - chevron_width + 1
    if left_start >= 0:
        choices.append(left_start)

    right_start = ref_end - overlap + 1
    right_end = right_start + chevron_width - 1
    if right_end < width:
        choices.append(right_start)

    if not choices:
        return None
    return random.choice(choices)


def _attempt_generate() -> np.ndarray | None:
    width = random.randint(20, 28)
    rows = _random_rows()
    ref_len = random.randint(6, 9)
    if width - ref_len - 5 < 4:
        return None
    ref_start = random.randint(4, width - ref_len - 5)

    lengths = _sample_lengths(ref_len, width)
    starts = _sample_starts(ref_start, ref_len, lengths, width)
    if starts is None:
        return None

    arm_len = random.choice((5, 6, 7))
    chevron_width = 2 * arm_len - 1
    chevron_left = _place_chevron_interval(width, ref_start, ref_len, chevron_width)
    if chevron_left is None:
        return None

    min_top = rows[-1] + 3
    min_height = min_top + arm_len + 1
    if min_height > 26:
        return None
    height = random.randint(max(18, min_height), 26)
    chevron_top = random.randint(min_top, height - arm_len - 1)
    orientation = random.choice(("caret", "vee"))

    grid = np.zeros((height, width), dtype=int)
    colors = list(LINE_COLORS)
    random.shuffle(colors)
    for row, start, length, color in zip(rows, starts, lengths, colors):
        grid[row, start : start + length] = int(color)

    for r, c in _make_chevron(chevron_top, chevron_left, arm_len, orientation):
        grid[r, c] = CHEVRON

    try:
        _validate_generated_example(grid)
    except AssertionError:
        return None
    return grid


def _validate_generated_example(grid: np.ndarray) -> None:
    segments, ref_index, active_cols = _validate_input_structure(grid)
    ref = segments[ref_index]
    lengths = [seg["length"] for seg in segments]
    starts = [seg["start"] for seg in segments]

    assert len(segments) == 5
    assert min(lengths) < ref["length"] < max(lengths)
    assert sum(length < ref["length"] for length in lengths) >= 2
    assert sum(length > ref["length"] for length in lengths) >= 2
    assert any(start < ref["start"] for start in starts)
    assert any(start > ref["start"] for start in starts)
    assert len(active_cols) <= ref["length"] - 2
    assert len(active_cols) >= 2

    original_cover_counts = [
        sum(seg["start"] <= c <= seg["end"] for seg in segments) for c in active_cols
    ]
    assert min(original_cover_counts) < len(segments), "alignment must matter for at least one active column"

    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "the puzzle must produce a non-trivial output"

    no_align = _render_from_reference(grid, segments, ref_index, keep_original_lines=True)
    assert not np.array_equal(out, no_align), "aligned output must differ from keeping original bars"

    alt_refs = sorted(
        {0, len(segments) - 1, max(range(len(segments)), key=lambda i: segments[i]["length"])}
        - {ref_index}
    )
    assert alt_refs
    assert any(
        not np.array_equal(out, _render_from_reference(grid, segments, alt))
        for alt in alt_refs
    ), "wrong reference choices should not reproduce the true output"

    bottom_gray = _bottom_gray_by_col(grid)
    inactive_cols = [c for c in range(ref["start"], ref["end"] + 1) if c not in bottom_gray]
    assert inactive_cols, "need uncovered reference columns as counterexamples"

    new_fill = [
        (r, c)
        for c in active_cols
        for r in range(bottom_gray[c] + 1, grid.shape[0])
        if out[r, c] != grid[r, c]
    ]
    assert new_fill, "active columns must create visible fill"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
