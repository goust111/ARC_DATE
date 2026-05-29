# Variant of e376de54
# Dimension: D12 — 线段数量 (meta)
# Change: 3-5条平行线段... → 包含干扰线段（非平行方向的线段）

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import get_bg


# -----------------------------------------------------------------------------
# Design analysis for the D12 variant
#
# a) Concrete behavior and trace
# - The original puzzle contains one family of 3 or 5 parallel segments. Every
#   multi-cell line belongs to that family, so the solver can treat the dominant
#   direction as "the only relevant direction".
# - This variant still has that same target family, but now also adds 1 or 2
#   extra line segments in another direction. Those distractors stay unchanged in
#   the output. Only the larger parallel family is aligned to the median-offset
#   reference segment.
# - Small trace:
#   Original-style input: three horizontal lines -> detect horizontal -> take the
#   middle row as reference -> align all three.
#   Variant input: the same three horizontal lines plus one or two long vertical
#   or diagonal lines -> solver must first decide that the horizontal family is
#   the operative group because it has more parallel segments, then align only
#   those three and leave the off-direction lines untouched.
#   Difference: the "which direction matters?" step moves from total same-color
#   adjacency toward counting maximal segments.
#
# b) Complexity calibration
# - The distractor count is capped at 1-2 so the scene remains readable.
# - The operative family remains 3 or 5 segments, preserving the original median
#   reference logic.
# - Distractors are often longer than the main-group segments so the variant is
#   not visually trivial and can defeat the old adjacency-based direction choice.
#
# c) Original implicit assumptions that break here
# - Old _pick_main_dir assumed the relevant direction is the one with the most
#   same-color adjacent pairs. That is no longer safe once a single long
#   distractor can outscore a larger-but-shorter parallel family.
# - The rest of main() only clears and redraws segments traced in the chosen
#   direction. That part is still valid, but only if the chosen direction is the
#   intended parallel family.
# - Therefore the critical fix is to extract maximal segments in every candidate
#   direction, then choose the unique direction that has the largest count of
#   multi-cell segments and still satisfies the original odd-count / unique-median
#   structure.
#
# d) Placement legality
# - Generator builds the transformed parallel family first, derives the input
#   family by varying lengths while keeping one endpoint anchored, then places
#   distractors only on cells disjoint from both input-family cells and
#   output-family cells.
# - Distractors also use distinct colors from the main family so separate objects
#   cannot accidentally merge into a single same-color run.
#
# e) Visual observability
# - A human can see one direction contains more parallel segments while another
#   direction contains only 1-2 off-angle lines. Across examples, the unchanged
#   off-angle lines make it clear that not every line participates in alignment.
# - No invisible metadata such as color id ordering is used; all decisions depend
#   on visible geometry: direction, multiplicity, offset, and length.
#
# f) Visual cleanliness
# - The generator forbids overlap between distractors and the target family in
#   both input and output states, so the scene stays separable.
# - Distractors are lines, not pixel markers, so the added structure matches the
#   visual language of the original task.
#
# g) Color inferability
# - Colors are preserved. There is no color remapping, only geometric change for
#   the main family and no change for distractors. This keeps the rule easy to
#   infer from 2-4 examples.
#
# h) Non-degenerate variant enforcement
# - The generator rejects samples unless the strongest distractor direction would
#   beat the main family under the old adjacency-score heuristic. This guarantees
#   the added lines are not merely decorative.
#
# i) New solver assumptions and generator guarantees
# - main() now assumes exactly one direction has the maximal eligible segment
#   count (3 or 5 segments with distinct perpendicular offsets). Generator
#   guarantees this by giving the true family 3 or 5 segments and each distractor
#   direction at most 2 segments.
# - main() assumes transformed target cells stay in-bounds. Generator still uses
#   output-first placement exactly to guarantee that.
# -----------------------------------------------------------------------------


_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _collect_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
    dr, dc = d
    h, w = g.shape
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


def _pick_parallel_group(g: np.ndarray, bg: int) -> tuple[tuple[int, int], list[dict]]:
    candidates = []
    for d in _DIR_CANDIDATES:
        segs = _collect_segments(g, bg, d)
        s_vals = [s["s"] for s in segs]
        if len(segs) not in (3, 5):
            continue
        if len(s_vals) != len(set(s_vals)):
            continue
        candidates.append((d, segs))

    assert candidates, "a unique target parallel family must exist"
    max_count = max(len(segs) for _, segs in candidates)
    best = [(d, segs) for d, segs in candidates if len(segs) == max_count]
    assert len(best) == 1, "target parallel family must be unique"
    return best[0]


def _adjacency_score(lengths: list[int] | tuple[int, ...]) -> int:
    return sum(length - 1 for length in lengths)


def _pair_count_for_dir(g: np.ndarray, bg: int, d: tuple[int, int]) -> int:
    dr, dc = d
    h, w = g.shape
    cnt = 0
    for r in range(h):
        for c in range(w):
            v = int(g[r, c])
            if v == bg:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == v:
                cnt += 1
    return cnt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align only the true parallel segment family while preserving off-direction distractor lines.

    Input structure:
    - bg is the most frequent color.
    - One direction contains the real task group: 3 or 5 multi-cell line segments on distinct parallel tracks.
    - The grid may also contain 1 or 2 extra multi-cell segments in other directions; these are distractors and are not part of the alignment group.

    Rule:
    - Detect which direction forms the unique larger parallel family.
    - Within that family, choose the reference segment by median perpendicular offset, exactly as in the original task.
    - Shift every segment in that family so its start matches the reference start, then trim or extend it to the reference length.
    - Leave all non-family distractor segments unchanged.

    Key insight:
    The important distinction is not "all visible lines" but "the direction with the parallel family";
    only that family is normalized, while off-angle lines are inert clutter.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d, segments = _pick_parallel_group(g, bg)
    dr, dc = d
    delta_t = _step(d)

    assert segments, "at least one target segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "target family must have a single median"
    assert len(s_vals) == len(set(s_vals)), "target family offsets must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "reference segment must be unique"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "segment tracing must move forward"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference extent must align with grid step"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()

    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must match grid step"
        k = dt // delta_t
        r0 = s["start"][0] + k * dr
        c0 = s["start"][1] + k * dc
        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "aligned target segment must stay in bounds"
            out[rr, cc] = s["color"]
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75
_MAIN_LONG_EXTRA_MAX = 4
_OTHER_LEN_EXTRA_MAX = 2

_DISTRACTOR_COUNT_OPTIONS = [1, 2]
_DISTRACTOR_MIN_LEN = 5

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 600


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


def _bundle_cells(
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
) -> list[list[tuple[int, int]]] | None:
    bundles = []
    for s, t0, length in zip(s_vals, starts, lengths):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return None
        bundles.append(cells)
    return bundles


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
    return [
        p
        for p in patterns
        if sum(p) <= max_span and (n == 3 and p[0] != p[1] or n == 5 and len(set(p)) > 1)
    ]


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
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(s_vals: tuple[int, ...]) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len: int, caps: list[int], n: int) -> list[int] | None:
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])

    long_hi = min(_MAX_SEG_LEN, caps[long_i], ref_len + _MAIN_LONG_EXTRA_MAX)
    if long_hi <= ref_len:
        return None
    lengths[long_i] = random.randint(ref_len + 1, long_hi)
    lengths[short_i] = random.randint(2, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        capped_hi = min(_MAX_SEG_LEN, caps[i], ref_len + _OTHER_LEN_EXTRA_MAX)
        options = list(range(2, capped_hi + 1))
        if not options:
            return None
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals: tuple[int, ...], starts: list[int], lengths: list[int], colors: tuple[int, ...]) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _choose_distractor_colors(main_colors: tuple[int, ...], count: int) -> tuple[int, ...]:
    pool = [c for c in _COLORS if c not in set(main_colors)]
    return tuple(random.sample(pool, count))


def _try_place_distractor_dir(
    d: tuple[int, int],
    count: int,
    blocked: set[tuple[int, int]],
    main_adj: int,
) -> list[dict] | None:
    placements = []
    used_s = set()
    current_blocked = set(blocked)
    total_adj = 0

    for _ in range(count):
        candidates = []
        for length in range(_MAX_SEG_LEN, _DISTRACTOR_MIN_LEN - 1, -1):
            for s in range(_s_limits(d)[0], _s_limits(d)[1] + 1):
                if s in used_s:
                    continue
                for t0 in range(*_T_SCAN_RANGE):
                    cells = _segment_cells(d, t0, s, length)
                    if cells is None:
                        continue
                    cell_set = set(cells)
                    if cell_set & current_blocked:
                        continue
                    candidates.append({"d": d, "s": s, "t0": t0, "length": length, "cells": cells})
            if candidates:
                break
        if not candidates:
            return None
        choice = random.choice(candidates)
        placements.append(choice)
        used_s.add(choice["s"])
        current_blocked |= set(choice["cells"])
        total_adj += choice["length"] - 1

    if total_adj <= main_adj:
        return None
    return placements


def _place_distractors(
    main_d: tuple[int, int],
    blocked: set[tuple[int, int]],
    main_colors: tuple[int, ...],
    main_adj: int,
) -> list[dict] | None:
    other_dirs = [d for d in _DIRS if d != main_d]
    count_order = random.sample(_DISTRACTOR_COUNT_OPTIONS, k=len(_DISTRACTOR_COUNT_OPTIONS))
    for count in count_order:
        for d in random.sample(other_dirs, k=len(other_dirs)):
            placements = _try_place_distractor_dir(d, count, blocked, main_adj)
            if placements is None:
                continue
            colors = _choose_distractor_colors(main_colors, count)
            for item, color in zip(placements, colors):
                item["color"] = color
            return placements
    return None


def _paint_distractors(grid: np.ndarray, distractors: list[dict]) -> bool:
    for item in distractors:
        for r, c in item["cells"]:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = item["color"]
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    out_starts = [t0] * n
    out_lengths = [ref_len] * n

    out_bundles = _bundle_cells(d, s_vals, out_starts, out_lengths)
    in_bundles = _bundle_cells(d, s_vals, starts, lengths)
    if out_bundles is None or in_bundles is None:
        return None

    blocked = set()
    for cells in out_bundles + in_bundles:
        blocked.update(cells)

    main_adj = _adjacency_score(lengths)
    distractors = _place_distractors(d, blocked, colors, main_adj)
    if distractors is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, out_starts, out_lengths, colors):
        return None
    if not _paint_distractors(out, distractors):
        return None

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None
    if not _paint_distractors(inp, distractors):
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
        # Reject decorative distractors: the old adjacency heuristic must be misled.
        main_pair_count = _pair_count_for_dir(inp, _BG, d)
        other_best = max(_pair_count_for_dir(inp, _BG, dd) for dd in _DIRS if dd != d)
        if other_best <= main_pair_count:
            return None
        return inp
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
