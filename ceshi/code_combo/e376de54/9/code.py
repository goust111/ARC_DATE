# Combo variant of e376de54 (index 9)
# combo_spec: D6/1+D7/2
# Name: 按长度排序位置+颜色映射排名

import random

import numpy as np

from arc_utils import get_bg

"""
Design analysis for combo D6/1 + D7/2
=====================================

a_concretization_and_trace:
  picture:
    input:
      - 3 or 5 parallel multi-cell segments on distinct lines, all sharing one direction
      - one common anchored end exactly as in the base task
      - each segment has a unique input color and a unique visible length
    output:
      - every segment is still normalized to the median-offset reference extent
      - the parallel lines are treated as ordered slots from low s to high s
      - the longest input segment is redrawn on the first slot, the second-longest on the next slot,
        and so on
      - colors no longer come from the input identities; slot rank i uses canonical rank color i
  traced_example:
    example:
      - slots from low s to high s contain input colors [8, 9, 6]
      - lengths on those slots are [4, 7, 5]
      - base task output would keep slots/colors and only make all lengths 7
      - D6-only output would reorder positions to length order, giving slot colors [9, 6, 8]
      - D7-only output would keep positions but recolor by rank, giving slot colors [3, 1, 2]
      - combo output sorts positions and colors together by rank, giving slot colors [1, 2, 3]
    difference:
      - geometry depends on the median reference extent as before
      - slot occupancy depends on length rank
      - output color also depends on the same rank, so the solver must combine both cues

b_complexity_calibration:
  controls:
    - segment count stays at 3 or 5, keeping the rank ladder readable
    - lengths are pairwise distinct, removing tie ambiguity
    - output colors come from a short fixed palette [1, 2, 3, 5, 6]
    - offsets are evenly spaced so the reordered slot pattern is easy to see
  anti_simplification:
    - generator rejects identity rank orders, so spatial reordering always happens
    - generator also rejects weak recolorings, so the fusion does not collapse into D6-only

c_implicit_assumption_audit_main:
  preserved_from_base:
    - dominant direction is still uniquely detectable by same-color adjacency
    - the reference segment is still the unique median by perpendicular offset
    - all output segments still share the reference start and reference length
  broken_from_base:
    - original painting used each segment's own color; invalid after D7/2
    - original painting kept each segment on its own line; invalid after D6/1
    - original task tolerated equal lengths; invalid now because ranking would be ambiguous
  new_requirements:
    - slot order must be well-defined from sorted perpendicular offsets
    - the rank palette must have at least as many colors as the segment count
    - destination cells must be reconstructed from `(ref_t0, destination_s)`, not from source starts

d_placement_legality:
  generator_guards:
    - choose evenly spaced offsets that fit the 16x16 board for the chosen direction
    - find a common aligned reference extent that is in bounds on every slot
    - derive the input by shared-start or shared-end anchoring while keeping all segments in bounds
    - paint with overlap checks so distinct slots never collide

e_visual_observability:
  - humans can read the ranking directly from input line lengths
  - the output places ranks in monotone slot order, so the spatial sorting is visible
  - the fixed rank palette repeats across examples, so the recoloring rule is learnable

f_visual_cleanliness:
  - only one family of thin parallel lines appears
  - uniform slot spacing prevents clutter and makes the permutation legible
  - no extra markers or auxiliary objects are introduced

g_change_inferability:
  - no hidden metadata such as numeric color ids or scan order is used for ranking
  - the same visible variable, input length rank, drives both the slot permutation and the color map
  - generator measures both non-identity slot movement and nontrivial recoloring to avoid degeneration

new_main_assumptions_checked_against_generator:
  - odd segment count -> enforced by _SEGMENT_COUNTS
  - unique offsets -> enforced by _choose_offsets
  - unique lengths -> enforced by _choose_lengths
  - enough canonical colors -> enforced by max count 5 and palette length 5
  - destination slots support the aligned reference extent -> enforced by _feasible_placements
  - color-based segment tracing remains sound -> enforced by unique input colors
"""

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]
_RANK_OUTPUT_COLORS = [1, 2, 3, 5, 6]

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_INPUT_COLOR_POOL = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = {3: (3, 8), 5: (4, 7)}
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 600


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


def _cell_from_ts(
    shape: tuple[int, int], d: tuple[int, int], t: int, s: int
) -> tuple[int, int] | None:
    h, w = shape
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
    if 0 <= r < h and 0 <= c < w:
        return int(r), int(c)
    return None


def _segment_cells(
    shape: tuple[int, int], d: tuple[int, int], t0: int, s: int, length: int
) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(shape, d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _trace_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
    h, w = g.shape
    dr, dc = d
    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue
                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
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


def _rank_indices_by_length(lengths: list[int], s_vals: tuple[int, ...]) -> list[int]:
    return sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), int(s_vals[i]), i))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize parallel segments to the reference extent, then sort their destination slots and colors by input-length rank.

    Input structure:
    - The background is the most common color.
    - Several multi-cell segments of distinct colors lie on parallel lines and share one dominant direction.
    - Their visible lengths are all different, and the ordered slot set is given by their perpendicular offsets.

    Rule:
    - Detect the common direction and trace every multi-cell segment on it.
    - Use the segment at the median perpendicular offset as the geometric reference; its start and length define the common output extent.
    - Rank the input segments from longest to shortest, rank the slot positions from low perpendicular offset to high, and place rank i onto slot i.
    - Recolor rank i with the fixed rank palette `[1, 2, 3, 5, 6]` while drawing every output segment at the reference extent; isolated pixels remain untouched.

    Key insight: Length rank now controls two things at once: which parallel line a segment moves to and which output color it receives, while the aligned geometry still comes from the median-offset reference segment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    delta_t = _step(d)

    segments = _trace_segments(g, bg, d)
    assert segments, "at least one multi-cell segment must exist"
    assert len(segments) <= len(_RANK_OUTPUT_COLORS), "rank palette too short"

    s_vals = sorted(int(s["s"]) for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    lengths = [int(s["len"]) for s in segments]
    assert len(lengths) == len(set(lengths)), "segment lengths must be pairwise distinct"

    ref_s = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if int(s["s"]) == ref_s]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_t0 = int(ref["t_start"])
    ref_t1 = int(ref["t_end"])
    target_len = int(ref["len"])
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    slots = sorted(segments, key=lambda s: (int(s["s"]), int(s["t_start"]), int(s["color"])))
    ranked = sorted(segments, key=lambda s: (-int(s["len"]), int(s["s"]), int(s["color"])))

    for rank, (slot, _) in enumerate(zip(slots, ranked)):
        start_cell = _cell_from_ts((h, w), d, ref_t0, int(slot["s"]))
        assert start_cell is not None, "destination slot must admit the reference start"
        rr, cc = start_cell
        color = _RANK_OUTPUT_COLORS[rank]
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "reordered segment must stay in bounds"
            out[rr, cc] = color
            rr += d[0]
            cc += d[1]

    return out


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts((_H, _W), d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts((_H, _W), d, t1 - length * delta, s) is not None:
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


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    allowed = _PERP_GAP_DIAG if d in [(1, 1), (1, -1)] else _PERP_GAP_ORTHO
    feasible = [gap for gap in allowed if lo + (n - 1) * gap <= hi]
    gap = random.choice(feasible)
    s0 = random.randint(lo, hi - (n - 1) * gap)
    return tuple(s0 + i * gap for i in range(n))


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int):
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells((_H, _W), d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_lengths(ref_len: int, caps: list[int], s_vals: tuple[int, ...]) -> list[int] | None:
    n = len(s_vals)
    ref_idx = n // 2
    max_lens = [min(_MAX_SEG_LEN, int(cap)) for cap in caps]
    if max_lens[ref_idx] < ref_len:
        return None

    domains = {}
    for i in range(n):
        if i == ref_idx:
            domains[i] = [ref_len]
        else:
            domains[i] = [length for length in range(2, max_lens[i] + 1) if length != ref_len]
            if not domains[i]:
                return None

    order = [ref_idx] + [i for i in range(n) if i != ref_idx]
    for _ in range(200):
        lengths = [None] * n
        used = set()

        def backtrack(pos: int) -> bool:
            if pos == len(order):
                return True
            idx = order[pos]
            candidates = domains[idx][:]
            random.shuffle(candidates)
            for length in candidates:
                if length in used:
                    continue
                lengths[idx] = length
                used.add(length)
                if backtrack(pos + 1):
                    return True
                used.remove(length)
                lengths[idx] = None
            return False

        if not backtrack(0):
            continue
        lengths = [int(x) for x in lengths]
        if not any(length < ref_len for length in lengths):
            continue
        if not any(length > ref_len for length in lengths):
            continue
        rank_order = _rank_indices_by_length(lengths, s_vals)
        move_count = sum(rank_order[i] != i for i in range(n))
        if move_count < 2:
            continue
        return lengths
    return None


def _choose_input_colors(n: int, rank_order: list[int]) -> tuple[int, ...]:
    out_colors = tuple(_RANK_OUTPUT_COLORS[:n])
    min_mismatches = max(2, n // 2)
    out_set = set(out_colors)
    for _ in range(200):
        colors = random.sample(_INPUT_COLOR_POOL, n)
        random.shuffle(colors)
        slot_mismatches = sum(colors[i] != out_colors[i] for i in range(n))
        rank_mismatches = sum(colors[rank_order[i]] != out_colors[i] for i in range(n))
        has_outside_color = any(color not in out_set for color in colors)
        if slot_mismatches >= min_mismatches and rank_mismatches >= min_mismatches and has_outside_color:
            return tuple(colors)
    return None


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells((_H, _W), d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE[n])
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    lengths = _choose_lengths(ref_len, caps, s_vals)
    if lengths is None:
        return None

    rank_order = _rank_indices_by_length(lengths, s_vals)
    input_colors = _choose_input_colors(n, rank_order)
    if input_colors is None:
        return None

    out_colors = tuple(_RANK_OUTPUT_COLORS[:n])
    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, out_colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, input_colors):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
