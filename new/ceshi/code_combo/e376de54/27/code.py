# Combo variant of e376de54 (index 27)
# combo_spec: D6/1+D7/2+D10/0
# Name: 按长度排序+颜色映射+参考消失

"""
Design analysis for combo D6/1 + D7/2 + D10/0
==============================================

a_concretization_and_trace:
  picture:
    input:
      - A 16x16 grid with 3 or 5 parallel multi-cell segments on distinct offsets.
      - Every input segment has a unique color so it can be traced reliably.
      - The segment at the median perpendicular offset is still the geometric reference.
    output:
      - The reference lane is empty.
      - The remaining segments are redrawn on the non-reference lanes at the reference extent.
      - Lane order now follows descending INPUT length rank.
      - Output colors are canonical rank colors computed from the full length ranking, so the
        hidden reference removes one interior rank color from the visible set.
  traced_example:
    - Suppose the input has horizontal segments on rows [2, 5, 8] with lengths [4, 7, 5].
    - Original task:
      row 5 is the reference, and all three rows become length 7 while keeping colors on rows
      [2, 5, 8].
    - This combo:
      row 5 is still the reference extent, but row 5 disappears in the output.
      Among the visible segments, length 5 outranks length 4, so the row-8 source moves to the
      upper visible lane and the row-2 source moves to the lower visible lane.
      Global length ranks are [7, 5, 4], so visible outputs use rank colors [2, 3] rather than
      simply [1, 2]; the missing rank color 1 belongs to the hidden reference.
    - Difference:
      the solver must combine hidden reference recovery, slot permutation, and rank-color mapping.

b_complexity_calibration:
  - Unique input lengths remove tie ambiguity in both the positional permutation and the color map.
  - Offsets are evenly spaced so "slot order" is easy to read; otherwise D6-style reordering would
    be visually noisy once the middle lane is removed.
  - The reference is forced to have at least one shorter and one longer neighbor, so its global
    rank is internal rather than extreme; this makes the hidden color gap observable.
  - The generator rejects cases where visible rank order already matches slot order, preventing the
    D6 component from collapsing back to "stay on the same lane".

c_implicit_assumption_audit_main:
  - Segment extraction by input color is no longer safe if colors repeat, so the generator now
    enforces unique input colors.
  - Painting with the source color is invalid because output colors are rank-derived instead.
  - Repainting on each segment's original offset is invalid because visible lanes are reassigned by
    length order.
  - Repainting every segment is invalid because the median-offset reference must disappear.
  - A fixed visible palette [1, 2, 3, 5, 6] is only sound because segment count stays in {3, 5}.

d_placement_legality:
  - Generation is output-first in (t, s) coordinates.
  - A common reference extent must fit on every slot before any input is built.
  - Distinct offsets guarantee that parallel segments never overlap.
  - Capacity checks guarantee that all chosen input lengths fit when anchored by common start or end.

e_visual_observability:
  - Humans can read the sortable attribute directly from visible segment lengths.
  - The empty median lane is a stable clue for the hidden reference.
  - The missing interior rank color is a second clue that the hidden segment still participates in
    the ranking even though it is not drawn.

f_visual_cleanliness:
  - Only one family of thin parallel lines is present.
  - Uniform lane spacing keeps the permutation readable after the center lane disappears.
  - No extra markers or overlapping decorations are introduced.

g_change_inferability:
  - Ranking never depends on numeric color ids or any invisible metadata.
  - Output colors come from a short canonical palette, so rank-color transfer is learnable from a
    few examples.
  - Because the hidden reference has an interior global rank, visible output colors always skip a
    meaningful canonical color, which prevents the color rule from degenerating to a plain prefix.

new_main_assumptions_checked_against_generator:
  - segment count is odd and at least 3, so the median reference is unique;
  - input colors are pairwise distinct, so color-based segment extraction is reliable;
  - input lengths are pairwise distinct, so both visible and global rankings are unique;
  - visible rank order differs from original slot order, so positional reordering is active;
  - the reference has both shorter and longer neighbors, so the hidden-rank color gap is active.
"""

import random

import numpy as np

from arc_utils import get_bg

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_GLOBAL_RANK_COLORS = [1, 2, 3, 5, 6]
_INPUT_COLOR_POOL = [1, 2, 3, 5, 6, 8, 9]


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


def _cell_from_ts(shape: tuple[int, int], d: tuple[int, int], t: int, s: int) -> tuple[int, int] | None:
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


def _extract_segments(g: np.ndarray, d: tuple[int, int], bg: int) -> list[dict]:
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


def _global_rank_colors(segments: list[dict]) -> dict[int, int]:
    lengths = [int(seg["len"]) for seg in segments]
    assert len(lengths) <= len(_GLOBAL_RANK_COLORS), "rank palette too short"
    assert len(lengths) == len(set(lengths)), "segment lengths must be pairwise distinct"
    order = sorted(range(len(segments)), key=lambda i: (-int(segments[i]["len"]), int(segments[i]["s"]), int(segments[i]["color"])))
    return {idx: _GLOBAL_RANK_COLORS[rank] for rank, idx in enumerate(order)}


def _draw_segment(out: np.ndarray, d: tuple[int, int], t0: int, s: int, length: int, color: int) -> None:
    dr, dc = d
    cell = _cell_from_ts(out.shape, d, t0, s)
    assert cell is not None, "destination slot must admit the reference start"
    rr, cc = cell
    for _ in range(length):
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "drawn segment must stay in bounds"
        out[rr, cc] = color
        rr += dr
        cc += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel segments to a hidden reference extent, then encode length order through both lane permutation and rank colors.

    Input structure:
    - The most frequent color is background.
    - Several solid multi-cell segments lie on distinct parallel lanes with one shared direction.
    - Input colors are unique, input lengths are all different, and the median lane by perpendicular offset is the reference that stays visible only in the input.

    Rule:
    - Detect the shared direction and extract every multi-cell segment on that direction.
    - Use the median-offset segment as the reference extent, but erase that lane from the output entirely.
    - Rank all segments by input length to determine canonical rank colors, then rank only the non-reference segments by the same input lengths and place them onto the visible lanes from low offset to high offset.
    - Every visible output segment is drawn at the reference start and reference length; isolated pixels remain unchanged.

    Key insight: The missing middle lane still participates in the hidden length ranking, so the output combines three clues at once: an invisible geometric standard, a lane permutation driven by input lengths, and a color sequence with a missing interior rank.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    delta_t = _step(d)

    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(int(seg["s"]) for seg in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) >= 3, "combo variant needs a visible neighborhood around the hidden reference"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    lengths = [int(seg["len"]) for seg in segments]
    assert len(lengths) == len(set(lengths)), "combo variant requires pairwise distinct segment lengths"

    ref_s = s_vals[len(s_vals) // 2]
    ref_candidates = [seg for seg in segments if int(seg["s"]) == ref_s]
    assert len(ref_candidates) == 1, "median reference segment must be unique"
    ref = ref_candidates[0]
    ref_t0 = int(ref["t_start"])
    ref_t1 = int(ref["t_end"])
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = int(ref["len"])

    rank_colors = _global_rank_colors(segments)
    visible_sources = [(idx, seg) for idx, seg in enumerate(segments) if int(seg["s"]) != ref_s]
    visible_slots = sorted(int(seg["s"]) for seg in segments if int(seg["s"]) != ref_s)
    ranked_visible = sorted(visible_sources, key=lambda item: (-int(item[1]["len"]), int(item[1]["s"]), int(item[1]["color"])))

    out = g.copy()
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = bg

    for slot_s, (seg_idx, _seg) in zip(visible_slots, ranked_visible):
        _draw_segment(out, d, ref_t0, int(slot_s), target_len, int(rank_colors[seg_idx]))

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = {3: (3, 8), 5: (4, 7)}
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 800


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts((_H, _W), d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


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
    feasible_gaps = [gap for gap in allowed if lo + (n - 1) * gap <= hi]
    gap = random.choice(feasible_gaps)
    s0 = random.randint(lo, hi - (n - 1) * gap)
    return tuple(s0 + i * gap for i in range(n))


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if min(start_caps) >= ref_len:
            placements.append((t0, "start", start_caps))
        if min(end_caps) >= ref_len:
            placements.append((t0, "end", end_caps))
    return placements


def _rank_indices(lengths: list[int], indices: list[int], s_vals: tuple[int, ...]) -> list[int]:
    return sorted(indices, key=lambda i: (-int(lengths[i]), int(s_vals[i]), i))


def _choose_lengths(ref_len: int, caps: list[int], s_vals: tuple[int, ...]) -> list[int] | None:
    n = len(s_vals)
    ref_idx = n // 2
    max_lens = [min(_MAX_SEG_LEN, int(cap)) for cap in caps]
    if max_lens[ref_idx] < ref_len:
        return None

    domains: dict[int, list[int]] = {}
    visible_indices = [i for i in range(n) if i != ref_idx]
    for i in visible_indices:
        domains[i] = [length for length in range(2, max_lens[i] + 1) if length != ref_len]
        if not domains[i]:
            return None

    order = sorted(visible_indices, key=lambda i: (len(domains[i]), i))
    for _ in range(160):
        lengths = [None] * n
        lengths[ref_idx] = ref_len
        used = {ref_len}

        def backtrack(pos: int) -> bool:
            if pos == len(order):
                return True
            idx = order[pos]
            candidates = domains[idx][:]
            random.shuffle(candidates)
            for length in candidates:
                if length in used:
                    continue
                used.add(length)
                lengths[idx] = length
                if backtrack(pos + 1):
                    return True
                lengths[idx] = None
                used.remove(length)
            return False

        if not backtrack(0):
            continue
        if not any(int(lengths[i]) < ref_len for i in visible_indices):
            continue
        if not any(int(lengths[i]) > ref_len for i in visible_indices):
            continue
        if _rank_indices(lengths, visible_indices, s_vals) == visible_indices:
            continue
        return [int(v) for v in lengths]
    return None


def _global_rank_colors_for_lengths(lengths: list[int], s_vals: tuple[int, ...]) -> tuple[int, ...]:
    n = len(lengths)
    assert n <= len(_GLOBAL_RANK_COLORS)
    assert len(lengths) == len(set(lengths)), "length ranks must be unique"
    order = _rank_indices(lengths, list(range(n)), s_vals)
    colors = [None] * n
    for rank, idx in enumerate(order):
        colors[idx] = _GLOBAL_RANK_COLORS[rank]
    return tuple(int(color) for color in colors)


def _choose_input_colors(global_rank_colors: tuple[int, ...], visible_indices: list[int]) -> tuple[int, ...]:
    output_color_set = set(global_rank_colors)
    for _ in range(200):
        colors = random.sample(_INPUT_COLOR_POOL, len(global_rank_colors))
        mismatch_count = sum(colors[i] != global_rank_colors[i] for i in visible_indices)
        outside_count = sum(colors[i] not in output_color_set for i in visible_indices)
        if mismatch_count >= max(2, len(visible_indices) // 2) and outside_count >= 1:
            return tuple(int(c) for c in colors)
    raise RuntimeError("failed to choose sufficiently distinct input colors")


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals: tuple[int, ...], starts: list[int], lengths: list[int], colors: tuple[int, ...], keep_mask: list[bool] | None = None) -> bool:
    if keep_mask is None:
        keep_mask = [True] * len(s_vals)
    for keep, s, t0, length, color in zip(keep_mask, s_vals, starts, lengths, colors):
        if not keep:
            continue
        cells = _segment_cells(d, int(t0), int(s), int(length))
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = int(color)
    return True


def _build_output(d: tuple[int, int], s_vals: tuple[int, ...], t0: int, ref_len: int, lengths: list[int]) -> tuple[np.ndarray, tuple[int, ...]]:
    n = len(s_vals)
    ref_idx = n // 2
    visible_indices = [i for i in range(n) if i != ref_idx]
    visible_slots = [int(s_vals[i]) for i in visible_indices]
    ranked_visible = _rank_indices(lengths, visible_indices, s_vals)
    global_colors = _global_rank_colors_for_lengths(lengths, s_vals)

    out = np.full((_H, _W), _BG, dtype=int)
    draw_lengths = [ref_len] * len(visible_slots)
    draw_starts = [t0] * len(visible_slots)
    draw_colors = tuple(global_colors[idx] for idx in ranked_visible)
    if not _paint(out, d, tuple(visible_slots), draw_starts, draw_lengths, draw_colors):
        raise RuntimeError("failed to build output")
    return out, global_colors


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_idx = n // 2
    ref_len = random.randint(*_REF_LEN_RANGE[n])
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    random.shuffle(placements)
    for t0, anchor, caps in placements[:40]:
        lengths = _choose_lengths(ref_len, caps, s_vals)
        if lengths is None:
            continue

        visible_indices = [i for i in range(n) if i != ref_idx]
        global_colors = _global_rank_colors_for_lengths(lengths, s_vals)
        ref_rank_color = global_colors[ref_idx]
        if ref_rank_color in (_GLOBAL_RANK_COLORS[0], _GLOBAL_RANK_COLORS[n - 1]):
            continue

        out, global_colors = _build_output(d, s_vals, t0, ref_len, lengths)
        input_colors = _choose_input_colors(global_colors, visible_indices + [ref_idx])

        delta = _step(d)
        t1 = t0 + (ref_len - 1) * delta
        if anchor == "start":
            starts = [t0] * n
        else:
            starts = [t1 - (length - 1) * delta for length in lengths]

        inp = np.full((_H, _W), _BG, dtype=int)
        if not _paint(inp, d, s_vals, starts, lengths, input_colors):
            continue

        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
