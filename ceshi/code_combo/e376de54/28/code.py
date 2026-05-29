# Combo variant of e376de54 (index 28)
# combo_spec: D1/3+D7/0+D12/1
# Name: 频率基准+共享颜色+大量线段

from collections import Counter
import random

import numpy as np

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# concretization_and_trace:
#   visual_form:
#   - The canvas contains 7-10 separated monochrome line segments on distinct parallel
#     tracks. They all share one direction from {(0,1), (1,0), (1,1), (1,-1)}.
#   - Colors are no longer unique identifiers. Several segments may reuse the same
#     color, so a human must distinguish tracks by position and length rather than by
#     color labels alone.
#   - Every segment still shares one anchored endpoint orientation with the others:
#     either all starts coincide in t-space or all ends coincide in t-space.
#   behavior_trace:
#   - Example with 8 horizontal segments sharing the same left endpoint and colors
#     [2, 5, 2, 8, 5, 2, 8, 5]:
#       lengths by top-to-bottom track = [9, 6, 4, 6, 7, 6, 10, 5]
#       original code would read the middle-offset track, whose length here is 7
#       combo rule must count the full multiset and pick 6 because it appears 3 times
#       output becomes [6, 6, 6, 6, 6, 6, 6, 6] with the common left endpoint preserved
#   - The key difference is twofold: the target is a repeated length class rather than
#     a central track, and repeated colors cannot be used to pair or rank segments.
# complexity_calibration:
#   - Using 7-10 segments makes the frequency signal meaningful at larger scale; the
#     generator therefore enforces a unique modal length with frequency 3 or 4.
#   - To avoid collapsing back to the older median-track heuristic, the generator makes
#     the middle-offset segment non-modal whenever an odd number of tracks is used.
#   - To keep the large-track layout physically possible while preserving clean gaps,
#     the canvas is enlarged from 16x16 to 24x24. Keeping the old canvas together with
#     10 tracks and the original minimum inter-track spacing would be geometrically
#     inconsistent.
# implicit_assumption_review:
#   - _pick_main_dir assumes same-color adjacency is strongest only along the true
#     segment direction. The generator preserves this by spacing tracks apart and never
#     allowing segments to touch.
#   - The original solver effectively treated "color" as the outer grouping key when
#     tracing segments. That is unsafe once colors are shared. The new extractor scans
#     all non-background cells and detects segment starts by local predecessor tests,
#     so multiple disconnected segments may share a color.
#   - The old "odd count + unique median perpendicular offset" assumption is removed.
#     The new solver instead assumes:
#       1. all traced segments lie on distinct parallel tracks
#       2. there is a unique modal length
#       3. all segments share either one common start or one common end
#     The generator constructs all three properties explicitly.
# placement_legality:
#   - Output is built first with every track already at the modal length, guaranteeing
#     the target configuration is in bounds.
#   - Input is derived by keeping one endpoint anchored and changing only the lengths.
#     Distinct perpendicular offsets prevent overlaps even when colors repeat.
# visual_observability_and_cleanliness:
#   - The modal length appears on several tracks at once, so a solver can visually spot
#     the repeated extent. Shared colors add clutter, but the tracks remain separated by
#     fixed gaps and aligned endpoints, keeping the scene legible.
#   - The palette is capped at 2-4 colors. That is enough to make color reuse obvious
#     without turning the grid into noise.
# inferability_of_change:
#   - No hidden ordering uses numeric color ids. All decisions are visible from geometry:
#     count how many tracks have each length, find the most frequent length, then snap
#     every track to that shared extent.

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


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _extract_segments(g: np.ndarray, d: tuple[int, int], bg: int) -> list[dict]:
    h, w = g.shape
    dr, dc = d
    segments = []
    for r in range(h):
        for c in range(w):
            col = int(g[r, c])
            if col == bg:
                continue
            pr, pc = r - dr, c - dc
            if 0 <= pr < h and 0 <= pc < w and int(g[pr, pc]) == col:
                continue

            cells = []
            rr, cc = r, c
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == col:
                cells.append((rr, cc))
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
    Normalize many parallel line segments to the length that occurs most often.

    Input structure:
    - The background is the most common color.
    - There are 7-10 multi-cell monochrome segments on distinct parallel tracks, all
      running in one common horizontal, vertical, or diagonal direction.
    - Multiple segments may share a color, so color is not a unique identity cue.
    - Their lengths vary, but they already share one anchored endpoint orientation:
      either all starts match in direction-space or all ends match.

    Rule:
    - Detect the shared direction and trace every multi-cell segment as a separate
      object, even when several objects have the same color.
    - Count the segment lengths and choose the unique modal length, meaning the length
      used by the largest number of tracks.
    - Keep the common anchored endpoint fixed for each segment, then trim or extend the
      segment so its new length equals that modal length.
    - Leave the grid size unchanged and keep any isolated single pixels untouched.

    Key insight: The reference is not a special track at all; it is the repeated length
    pattern across the whole crowded set of segments.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    d = _pick_main_dir(g, bg)
    dr, dc = d

    segments = _extract_segments(g, d, bg)
    assert 7 <= len(segments) <= 10, "combo variant expects 7-10 segments"

    s_vals = [s["s"] for s in segments]
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    if _step(d) == 2:
        ss = sorted(s_vals)
        for i in range(len(ss) - 1):
            assert ss[i + 1] - ss[i] >= 4, "diagonal tracks must be well separated"

    length_counts = Counter(s["len"] for s in segments)
    best_freq = max(length_counts.values())
    assert best_freq >= 2, "at least one repeated length is required"
    mode_lens = [length for length, freq in length_counts.items() if freq == best_freq]
    assert len(mode_lens) == 1, "modal length must be unique"
    target_len = mode_lens[0]

    common_starts = {s["t_start"] for s in segments}
    common_ends = {s["t_end"] for s in segments}
    use_start_anchor = len(common_starts) == 1
    use_end_anchor = len(common_ends) == 1
    assert use_start_anchor ^ use_end_anchor, "segments must share exactly one endpoint orientation"

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
            assert 0 <= rr < h and 0 <= cc < w, "resized segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

_H, _W = 24, 24
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 4, 5, 6, 8, 9]

_SEGMENT_COUNTS = [7, 8, 9, 10]
_MODE_LEN_RANGE = (4, 9)
_MAX_SEG_LEN = 15

_PERP_GAP_ORTHO = [2, 3]
_PERP_GAP_DIAG = [4, 5]

_MIN_COLORS = 2
_MAX_DISTINCT_COLORS = 4

_T_SCAN_RANGE = (-60, 61)
_GEN_RETRIES = 800


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
        return 0, _H - 1
    if d == (1, 0):
        return -(_W - 1), 0
    if d == (1, 1):
        return -(_W - 1), _H - 1
    return -(_H + _W - 2), 0


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


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(_MIN_COLORS, min(_MAX_DISTINCT_COLORS, n - 1))
    palette = random.sample(_COLORS, n_colors)
    colors = [None] * n

    repeated_color = random.choice(palette)
    repeated_idxs = random.sample(range(n), random.randint(2, min(4, n)))
    for idx in repeated_idxs:
        colors[idx] = repeated_color

    unused_colors = [c for c in palette if c != repeated_color]
    remaining_slots = [i for i, col in enumerate(colors) if col is None]
    random.shuffle(remaining_slots)
    for color in unused_colors:
        idx = remaining_slots.pop()
        colors[idx] = color

    for idx in remaining_slots:
        colors[idx] = random.choice(palette)

    assert len(set(colors)) == n_colors
    assert max(Counter(colors).values()) >= 2
    return tuple(colors)


def _is_originally_degenerate(lengths: list[int], mode_len: int) -> bool:
    n = len(lengths)
    sorted_lens = sorted(lengths)
    if n % 2 == 1 and lengths[n // 2] == mode_len:
        return True
    if n >= 2 and sorted_lens[-2] == mode_len:
        return True
    return False


def _choose_lengths(mode_len: int, caps: list[int]) -> list[int] | None:
    n = len(caps)
    mode_freq = random.randint(3, min(4, n - 2))
    mid = n // 2

    mode_pool = list(range(n))
    if n % 2 == 1 and mid in mode_pool:
        mode_pool.remove(mid)

    for _ in range(400):
        mode_indices = set(random.sample(mode_pool, mode_freq))
        non_mode = [i for i in range(n) if i not in mode_indices]
        longable = [i for i in non_mode if caps[i] > mode_len]
        if len(longable) < 2:
            continue

        lengths = [mode_len if i in mode_indices else None for i in range(n)]

        long_i, long_j = random.sample(longable, 2)
        long_opts_i = list(range(mode_len + 1, min(_MAX_SEG_LEN, caps[long_i]) + 1))
        long_opts_j = list(range(mode_len + 1, min(_MAX_SEG_LEN, caps[long_j]) + 1))
        long_pairs = [(a, b) for a in long_opts_i for b in long_opts_j if a != b]
        if not long_pairs:
            continue
        long_len_i, long_len_j = random.choice(long_pairs)
        lengths[long_i] = long_len_i
        lengths[long_j] = long_len_j

        short_choices = [i for i in non_mode if i not in (long_i, long_j)]
        if not short_choices:
            continue
        short_i = random.choice(short_choices)
        lengths[short_i] = random.randint(2, mode_len - 1)

        remaining = [i for i in non_mode if lengths[i] is None]
        used_counts = Counter(length for length in lengths if length is not None)
        for idx in remaining:
            options = [length for length in range(2, min(_MAX_SEG_LEN, caps[idx]) + 1) if length != mode_len]
            if not options:
                break
            random.shuffle(options)
            options.sort(key=lambda length: (used_counts[length], abs(length - mode_len), random.random()))
            chosen = options[0]
            lengths[idx] = chosen
            used_counts[chosen] += 1
        else:
            counts = Counter(lengths)
            if counts[mode_len] != mode_freq:
                continue
            if max(freq for length, freq in counts.items() if length != mode_len) >= mode_freq:
                continue
            if min(lengths) >= mode_len or max(lengths) <= mode_len:
                continue
            if _is_originally_degenerate(lengths, mode_len):
                continue
            return lengths

    return None


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals, starts, lengths, colors) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _gen_attempt():
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    mode_len = random.randint(*_MODE_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, mode_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    lengths = _choose_lengths(mode_len, caps)
    if lengths is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [mode_len] * n, colors):
        return None

    delta = _step(d)
    t1 = t0 + (mode_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
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
