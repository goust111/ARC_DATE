# Combo variant of e376de54 (index 32)
# combo_spec: D1/1+D5/1
# Name: 最短基准+倒数第二短

"""
Design analysis for the combo variant

a) Concrete behavior and trace
   - The visible objects stay in the same family as the source task: a 16x16 background with several
     parallel multi-cell segments, all in one shared direction and on distinct perpendicular offsets.
   - The fused change is a short-end mirror of the original "second longest" rule. We still normalize
     every segment to one reference span, but the reference is now chosen by sorting lengths from short
     to long and taking the unique second-shortest segment.
   - Small trace, horizontal / common-end case:
       input lengths by offset order = [2, 4, 7, 9, 11], all ending at the same column.
       original code would use length 9 because that is the second longest.
       D1 alone would use length 2 because that is the shortest.
       this combo uses length 4 because it is the second shortest.
       The output therefore redraws every row onto the 4-cell span, aligned to the 4-cell segment's
       start and end positions.
   - The key difference from the original is not merely "shorter output"; it is that ranking and
     reference selection both move to the short side, while still skipping the absolute minimum.

b) Complexity calibration
   - I fix the generator to 5 segments. With only 3 segments, "second shortest" collapses to the same
     rank as "second longest", which would visually erase the D5 half of the fusion.
   - Lengths are forced into one strict pattern: exactly one shorter segment, exactly one reference
     segment, and exactly three distinct longer segments. This gives a unique second-shortest witness
     without adding extra object types.
   - The reference offset is forced away from the median perpendicular line, so the samples cannot be
     explained by the original median-by-position solver.

c) Hidden assumptions reviewed against main()
   - `_pick_main_dir()` assumes same-color adjacency is strongest along one direction. The generator
     preserves this by drawing only clean segments and keeping perpendicular gaps large.
   - `_extract_segments()` assumes each segment has a unique predecessor-free start along the detected
     direction. That remains true because segments are straight, non-overlapping, and lie on distinct
     offsets.
   - Reference selection no longer depends on perpendicular-order median. Instead, `main()` assumes:
       1. there is a unique shortest segment below the reference length;
       2. there is a unique segment at the second-smallest length.
     The generator enforces both by making all five lengths distinct and ordered as
     shortest < reference < long1 < long2 < long3.
   - The redraw loop still aligns every segment start to the reference start. That is only safe if the
     chosen reference span is valid on every offset. The generator builds the common output span first,
     then derives the input by lengthening or shortening away from a shared anchor, which guarantees the
     shifted redraw remains in bounds.

d) Placement legality
   - Output-first placement chooses a reference span that fits on every offset before any input-specific
     variation is introduced.
   - Start-anchored inputs extend only forward from that span; end-anchored inputs extend only backward.
     This keeps all input segments legal while preserving the shared-endpoint structure needed by the task.
   - Distinct offsets plus orthogonal gaps >= 2 and diagonal gaps >= 4 prevent overlap or accidental
     touching between neighboring segments.

e) Visual observability
   - A human can rank the five segment lengths by sight and see that the output matches neither the
     shortest nor the second longest, but the second-shortest segment.
   - Because the reference is not allowed to sit on the median offset, the examples also visibly reject
     the original position-based explanation.
   - No invisible property such as color id or scan order participates in ranking; only segment length
     matters.

f) Visual cleanliness
   - The grid contains only the background and the line segments, so the transformation remains easy to
     inspect.
   - Irregular offset gaps keep lines separate even when colors repeat.
   - The output is visually compact but not collapsed: one line stays shorter than most inputs, yet the
     chosen span is not the absolute minimum, which makes the fusion effect legible.

g) Change inferability
   - Colors are preserved exactly and never remapped, so the learner does not need to infer a palette
     rule in addition to the geometric one.
   - Repeated colors are allowed, but they do not carry semantics; repeated structural evidence comes from
     multiple lengths on multiple offsets instead.

Post-change solver assumptions guaranteed by generate_input()
   - There are exactly 5 multi-cell segments.
   - All segment offsets are distinct, and diagonal offsets are separated enough for clean direction
     detection.
   - Segment lengths are all distinct, with exactly one shorter than the reference and exactly one equal
     to the reference.
   - The second-shortest segment is not on the median perpendicular offset, so the samples do not
     degenerate to the original reference rule.
   - The common reference span is valid on every offset, so repainting to that span cannot leave bounds.
"""

import random

import numpy as np

from arc_utils import get_bg

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


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
    # perp to (dr,dc) can be (dc,-dr); dot(p, perp) = r*dc - c*dr
    return p[0] * d[1] - p[1] * d[0]


def _extract_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel colored segments so every one matches the span of the second-shortest segment.

    Input structure:
    - The background is the most frequent color.
    - Five horizontal, vertical, or diagonal multi-cell segments lie on distinct parallel offsets and
      share one detectable main direction.
    - Their visible lengths are all different, so the shortest and the second-shortest segments are both
      uniquely identifiable by sight.

    Rule:
    - Detect the common direction and trace each multi-cell segment along it.
    - Sort the segment lengths from short to long and choose the unique second-shortest segment as the
      reference.
    - Read that reference segment's start and end coordinates along the main direction.
    - Shift every traced segment so its start aligns with the reference start, then redraw exactly the
      reference length.
    - Keep the grid size unchanged and leave isolated single pixels untouched.

    Key insight: The task is a short-end mirror of the original long-end ranking. The answer is governed
    by the second-shortest segment, not by the median offset line, the shortest segment, or the
    second-longest segment.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _extract_segments(g, bg, d)
    assert len(segments) >= 2, "at least two multi-cell segments must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    sorted_lens = sorted(s["len"] for s in segments)
    ref_len = sorted_lens[1]
    assert sum(s["len"] < ref_len for s in segments) == 1, "reference must be the unique second-shortest"
    ref_cands = [s for s in segments if s["len"] == ref_len]
    assert len(ref_cands) == 1, "second-shortest reference segment must be unique"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1
    assert target_len == ref_len

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = s["start"][0] + k * dr
        c0 = s["start"][1] + k * dc
        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
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
_REF_LEN_RANGE = (3, 5)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 800


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d: tuple[int, int], t: int, s: int) -> tuple[int, int] | None:
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


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int) -> list[tuple[int, int]] | None:
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


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        placements.append((t0, "start", start_caps))
        placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _ref_index_choices(n: int) -> list[int]:
    center = n // 2
    return [i for i in range(n) if i != center]


def _choose_distinct_longs(ref_len: int, long_idxs: list[int], caps: list[int]) -> dict[int, int] | None:
    order = long_idxs[:]
    random.shuffle(order)
    assign: dict[int, int] = {}
    used = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        i = order[pos]
        hi = min(_MAX_SEG_LEN, caps[i])
        options = list(range(ref_len + 1, hi + 1))
        random.shuffle(options)
        for value in options:
            if value in used:
                continue
            used.add(value)
            assign[i] = value
            if dfs(pos + 1):
                return True
            used.remove(value)
            del assign[i]
        return False

    return assign if dfs(0) else None


def _choose_lengths(ref_len: int, caps: list[int], ref_idx: int) -> tuple[int, ...] | None:
    nonrefs = [i for i in range(len(caps)) if i != ref_idx]
    random.shuffle(nonrefs)
    for short_idx in nonrefs:
        long_idxs = [i for i in nonrefs if i != short_idx]
        if any(min(_MAX_SEG_LEN, caps[i]) <= ref_len for i in long_idxs):
            continue
        long_assign = _choose_distinct_longs(ref_len, long_idxs, caps)
        if long_assign is None:
            continue
        lengths = [None] * len(caps)
        lengths[ref_idx] = ref_len
        lengths[short_idx] = random.randint(2, ref_len - 1)
        for i, value in long_assign.items():
            lengths[i] = value
        assert len(set(lengths)) == len(lengths)
        return tuple(lengths)
    return None


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: tuple[int, ...] | list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = _SEGMENT_COUNT
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    random.shuffle(placements)
    ref_indices = _ref_index_choices(n)
    colors = _choose_colors(n)
    delta = _step(d)

    for t0, anchor, caps in placements:
        ref_order = ref_indices[:]
        random.shuffle(ref_order)
        for ref_idx in ref_order:
            lengths = _choose_lengths(ref_len, caps, ref_idx)
            if lengths is None:
                continue

            out = np.full((_H, _W), _BG, dtype=int)
            if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
                continue

            t1 = t0 + (ref_len - 1) * delta
            if anchor == "start":
                starts = [t0] * n
            else:
                starts = [t1 - (length - 1) * delta for length in lengths]

            inp = np.full((_H, _W), _BG, dtype=int)
            if not _paint(inp, d, s_vals, starts, lengths, colors):
                continue

            try:
                if not np.array_equal(main(inp), out):
                    continue
            except Exception:
                continue

            segments = sorted(_extract_segments(inp, _BG, d), key=lambda s: s["s"])
            if len(segments) != n:
                continue
            observed_lengths = [s["len"] for s in segments]
            if sorted(observed_lengths) != sorted(lengths):
                continue
            if observed_lengths[n // 2] == ref_len:
                continue
            if sum(length < ref_len for length in observed_lengths) != 1:
                continue
            if sum(length == ref_len for length in observed_lengths) != 1:
                continue
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
