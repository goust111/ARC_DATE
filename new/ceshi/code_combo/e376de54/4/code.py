# Combo variant of e376de54 (index 4)
# combo_spec: D1/3+D7/0
# Name: 频率基准+共享颜色

from collections import Counter
from itertools import combinations
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# concretization_and_trace:
#   visual_form:
#   - The canvas still contains several separated parallel monochrome segments on a
#     16x16 grid, all following one shared direction from {(0,1), (1,0), (1,1), (1,-1)}.
#   - Unlike the color-unique reading of the seed task, 2-3 segments may now share the
#     same color while sitting on different parallel tracks. They remain distinct objects
#     because their perpendicular offsets differ.
#   - Every segment still shares one anchored endpoint convention with the others:
#     either all starts coincide in t-space or all ends coincide in t-space.
#   behavior_trace:
#   - Example with 5 horizontal segments ordered by row:
#       colors  = [2, 8, 2, 5, 2]
#       lengths = [4, 6, 8, 6, 7]
#     Original seed logic would read the middle track and copy length 8.
#     The combo logic must first separate the three red-looking tracks by position,
#     then count all lengths: 6 occurs twice, all others once, so target length = 6.
#     Output becomes lengths [6, 6, 6, 6, 6] while colors stay [2, 8, 2, 5, 2].
#   - The key fusion effect is that repeated color is a distraction: two same-colored
#     tracks can behave differently in the input because the decision comes from global
#     length frequency, not from color identity.
# complexity_calibration:
#   - The generator uses 5 segments only. This is the smallest count that can show both
#     a unique modal length and a nontrivial repeated-color subgroup without collapsing
#     into ambiguous "maybe the middle line matters" examples.
#   - The modal length appears exactly twice, and the middle-offset segment is never one
#     of those two tracks. This keeps the D1 change visible by forcing disagreement with
#     the original median-track rule on every generated instance.
#   - A shared-color group of size 2 or 3 is enforced, and its member lengths are not all
#     equal. That makes the D7 change visible instead of allowing color sharing to occur
#     only in trivial same-length cases.
# implicit_assumption_review:
#   - The seed solver grouped cells by color before tracing segments. That is no longer a
#     safe conceptual model once multiple segments may share one color. The new solver
#     scans the whole grid and traces runs by spatial starts along the detected direction.
#   - Direction detection still relies on one direction having the most same-color
#     adjacencies; the spacing constraints are preserved so this remains stable.
#   - The new solver assumes:
#       1. all multi-cell objects are straight monochrome segments on the main direction
#       2. segment tracks have distinct perpendicular offsets
#       3. there is a unique modal segment length
#       4. all segments share either one common start or one common end in t-space
#     The generator constructs exactly these properties.
# placement_legality:
#   - Output is built first with all tracks already at the modal length and therefore in
#     bounds. Input is derived by preserving the shared anchor and varying only lengths.
#   - Distinct perpendicular offsets guarantee that parallel segments never overlap, even
#     when colors repeat. Diagonal gaps stay at least 4 to avoid visual touching.
# visual_observability_and_cleanliness:
#   - Human solvers can see two cues at once: one length repeats across different tracks,
#     and at least one color repeats across tracks with different lengths.
#   - Because repeated colors are spatially separated and gaps are irregular, the grid
#     stays readable: each segment is visually distinct despite some colors recurring.
# change_inferability:
#   - No hidden use of color ids is introduced. Colors are preserved, not remapped.
#   - The only inferred relation is geometric: count lengths across all tracks, even when
#     some tracks share a color. Repeated color provides a visible counterexample to the
#     shortcut "same color means same segment role".

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _pick_main_dir(g: np.ndarray, bg: int) -> tuple[int, int]:
    counts = []
    h, w = g.shape
    for dr, dc in _DIR_CANDIDATES:
        cnt = 0
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v == bg:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == v:
                    cnt += 1
        counts.append(((dr, dc), cnt))

    best_cnt = max(cnt for _, cnt in counts)
    assert best_cnt > 0, "main direction must be detectable"
    best_dirs = [d for d, cnt in counts if cnt == best_cnt]
    assert len(best_dirs) == 1, "main direction must be unique"
    return best_dirs[0]


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _extract_segments(g: np.ndarray, d: tuple[int, int], bg: int) -> list[dict]:
    h, w = g.shape
    dr, dc = d
    visited = np.zeros((h, w), dtype=bool)
    segments = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue

            col = int(g[r, c])
            pr, pc = r - dr, c - dc
            if 0 <= pr < h and 0 <= pc < w and int(g[pr, pc]) == col:
                continue

            cells = []
            rr, cc = r, c
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == col:
                cells.append((rr, cc))
                visited[rr, cc] = True
                rr += dr
                cc += dc

            if len(cells) < 2:
                continue

            start = cells[0]
            end = cells[-1]
            segments.append(
                {
                    "color": col,
                    "cells": cells,
                    "start": start,
                    "end": end,
                    "t_start": _dot(start, d),
                    "t_end": _dot(end, d),
                    "s": _perp_dot(start, d),
                    "len": len(cells),
                }
            )

    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize parallel segments to the length that appears most often, even when colors repeat across tracks.

    Input structure:
    - bg is the most common color.
    - Several multi-cell monochrome segments lie on distinct parallel tracks and all follow one shared direction.
    - Two or three of those segments may share the same color, so color alone does not identify a segment.
    - Their lengths vary, but all segments already share one anchored endpoint orientation: either a common start or a common end in t-space.

    Rule:
    - Detect the common direction, then trace each segment by spatial position along that axis rather than grouping by color.
    - Count all segment lengths and choose the unique modal length, meaning the length with the highest frequency.
    - Preserve the shared anchored endpoint for each segment, then trim or extend every segment so its length becomes that modal length.
    - Keep the grid size unchanged and leave isolated single cells alone.

    Key insight: Repeated colors are a decoy; the real reference is the length value that repeats most often across distinct tracks.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    length_counts = Counter(s["len"] for s in segments)
    best_freq = max(length_counts.values())
    assert best_freq > 1, "at least one repeated length is required"
    mode_lens = [length for length, freq in length_counts.items() if freq == best_freq]
    assert len(mode_lens) == 1, "modal length must be unique"
    target_len = mode_lens[0]

    common_starts = {s["t_start"] for s in segments}
    common_ends = {s["t_end"] for s in segments}
    use_start_anchor = len(common_starts) == 1
    use_end_anchor = len(common_ends) == 1
    assert use_start_anchor or use_end_anchor, "segments must share one anchored endpoint"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        if use_start_anchor:
            r0, c0 = s["start"]
        else:
            r0 = s["end"][0] - (target_len - 1) * dr
            c0 = s["end"][1] - (target_len - 1) * dc

        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "normalized segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNT = 5
_MODE_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d: tuple[int, int], t: int, s: int):
    if d == (0, 1):
        r, c = s, t
    elif d == (1, 0):
        r, c = t, -s
    elif d == (1, 1):
        if (t + s) % 2 or (t - s) % 2:
            return None
        r, c = (t + s) // 2, (t - s) // 2
    else:
        if (t - s) % 2 or (-s - t) % 2:
            return None
        r, c = (t - s) // 2, (-s - t) // 2
    if 0 <= r < _H and 0 <= c < _W:
        return r, c
    return None


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int):
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d: tuple[int, int], n: int) -> list[list[int]]:
    allowed = _PERP_GAP_DIAG if d in [(1, 1), (1, -1)] else _PERP_GAP_ORTHO
    lo, hi = _s_limits(d)
    max_span = hi - lo
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [g] for p in patterns for g in allowed]
    return [p for p in patterns if sum(p) <= max_span and len(set(p)) > 1]


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], mode_len: int):
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, mode_len) is None for s in s_vals):
            continue

        t1 = t0 + (mode_len - 1) * _step(d)
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if sum(cap > mode_len for cap in start_caps) >= 2:
            placements.append((t0, "start", start_caps))

        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if sum(cap > mode_len for cap in end_caps) >= 2:
            placements.append((t0, "end", end_caps))
    return placements


def _choose_lengths(mode_len: int, caps: list[int]) -> list[int] | None:
    mid = _SEGMENT_COUNT // 2
    mode_pairs = list(combinations([i for i in range(_SEGMENT_COUNT) if i != mid], 2))
    random.shuffle(mode_pairs)

    for mode_pair in mode_pairs:
        remaining = [i for i in range(_SEGMENT_COUNT) if i not in mode_pair]
        longable = [i for i in remaining if caps[i] > mode_len]
        long_pairs = list(combinations(longable, 2))
        random.shuffle(long_pairs)

        for long_pair in long_pairs:
            short_i = [i for i in remaining if i not in long_pair][0]
            long_i, long_j = long_pair

            long_opts_i = list(range(mode_len + 1, min(_MAX_SEG_LEN, caps[long_i]) + 1))
            long_opts_j = list(range(mode_len + 1, min(_MAX_SEG_LEN, caps[long_j]) + 1))
            long_value_pairs = [(a, b) for a in long_opts_i for b in long_opts_j if a != b]
            if not long_value_pairs:
                continue

            short_len = random.randint(2, mode_len - 1)
            long_len_i, long_len_j = random.choice(long_value_pairs)

            lengths = [None] * _SEGMENT_COUNT
            for idx in mode_pair:
                lengths[idx] = mode_len
            lengths[short_i] = short_len
            lengths[long_i] = long_len_i
            lengths[long_j] = long_len_j

            counts = Counter(lengths)
            if counts[mode_len] != 2:
                continue
            if max(freq for length, freq in counts.items() if length != mode_len) > 1:
                continue
            if lengths[mid] == mode_len:
                continue
            return lengths

    return None


def _choose_colors(lengths: list[int], mode_len: int) -> tuple[int, ...] | None:
    idxs = list(range(_SEGMENT_COUNT))
    valid_shared_groups = []
    for size in (2, 3):
        for group in combinations(idxs, size):
            group_lengths = {lengths[i] for i in group}
            if len(group_lengths) < 2:
                continue
            if any(lengths[i] == mode_len for i in group) and any(lengths[i] != mode_len for i in group):
                valid_shared_groups.append(group)

    if not valid_shared_groups:
        return None

    shared_group = random.choice(valid_shared_groups)
    shared_color = random.choice(_COLORS)
    remaining_colors = [c for c in _COLORS if c != shared_color]
    other_palette_size = random.choice([1, 2])
    other_palette = random.sample(remaining_colors, other_palette_size)

    colors = [None] * _SEGMENT_COUNT
    for i in shared_group:
        colors[i] = shared_color

    for i in idxs:
        if colors[i] is None:
            colors[i] = random.choice(other_palette)

    if len(set(colors)) < 2:
        lone = random.choice([i for i in idxs if i not in shared_group])
        colors[lone] = other_palette[0]

    color_counts = Counter(colors)
    repeated = [col for col, freq in color_counts.items() if freq >= 2]
    assert repeated
    for col in repeated:
        same_lengths = {lengths[i] for i, c in enumerate(colors) if c == col}
        if len(same_lengths) >= 2:
            return tuple(colors)
    return None


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals: tuple[int, ...], starts: list[int], lengths: list[int], colors: tuple[int, ...]) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt():
    d = random.choice(_DIRS)
    mode_len = random.randint(*_MODE_LEN_RANGE)
    s_vals = _choose_offsets(d, _SEGMENT_COUNT)
    placements = _feasible_placements(d, s_vals, mode_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    lengths = _choose_lengths(mode_len, caps)
    if lengths is None:
        return None

    colors = _choose_colors(lengths, mode_len)
    if colors is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * _SEGMENT_COUNT, [mode_len] * _SEGMENT_COUNT, colors):
        return None

    delta = _step(d)
    t1 = t0 + (mode_len - 1) * delta
    if anchor == "start":
        starts = [t0] * _SEGMENT_COUNT
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
