# Combo variant of e376de54 (index 11)
# combo_spec: D10/0+D12/2
# Name: 参考消失+干扰线段

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import get_bg


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
# - Base task: one direction contains 3 or 5 parallel segments. The median by
#   perpendicular offset is the reference. Every segment is shifted so its start
#   matches the reference start, and every segment is resized to the reference
#   length.
# - D10/0 changes the redraw stage: the reference still exists in the input and
#   still defines the target interval, but it is erased in the output instead of
#   being repainted.
# - D12/2 changes the scene composition: besides the real parallel family, the
#   input also contains 1 or 2 segments in a different direction. Those
#   distractors are copied through unchanged.
# - Small trace:
#   Input has five horizontal segments at rows [1, 4, 7, 10, 13], sharing the
#   same start anchor, plus one long diagonal distractor. Row 7 is the median
#   reference and spans columns 5..9. The original task would align all five
#   horizontal rows to columns 5..9. The combo output aligns rows 1, 4, 10, 13
#   to columns 5..9, leaves row 7 blank, and keeps the diagonal distractor
#   exactly as it was.
# - Difference point: the solver must first separate "operative parallel family"
#   from off-angle clutter, then infer a target interval from a line that is not
#   visible in the output.
#
# b) Complexity calibration
# - The operative family stays at 3 or 5 segments so the hidden median remains
#   uniquely inferable and still has visible neighbors on both sides.
# - Distractors are capped at 1-2 and use exactly one non-main direction. That
#   is enough to break the original adjacency heuristic without making the scene
#   unreadable.
# - The generator blocks distractors from both the input family and the full
#   aligned target bundle, including the hidden reference lane, so the missing
#   lane remains visually legible.
# - Colors still come from only 1-3 main-family colors plus up to 2 distractor
#   colors, with distractor colors disjoint from the main-family palette.
#
# c) Implicit assumption audit
# - Original _pick_main_dir assumed the relevant direction maximizes same-color
#   adjacency. That fails here because a single long distractor can beat the true
#   family under that heuristic. The combo solver instead extracts maximal
#   segments in every candidate direction and picks the unique eligible family
#   with the largest segment count.
# - Original redraw repainted every traced segment. That no longer holds because
#   the median reference must disappear. The combo redraw skips only the median
#   lane and repaints the other family members.
# - Clearing target-family cells remains safe because distractors are guaranteed
#   to be disjoint from every target-family cell in both the input and the full
#   aligned layout.
# - The solver assumes the chosen family has odd cardinality, distinct
#   perpendicular offsets, and a unique median. The generator enforces exactly
#   that structure.
#
# d) Placement legality
# - Generation is output-first. It first chooses the hidden reference extent,
#   paints the aligned family without the reference, then derives the input by
#   restoring the reference and varying only the non-reference lengths while
#   keeping a common start or end anchor.
# - Distractors are placed afterward on cells disjoint from all cells used by
#   the input family and by the full aligned family. This prevents overlap,
#   clipping, and accidental covering of the hidden target span.
#
# e) Visual observability
# - Across examples, humans can see one direction has 3 or 5 parallel tracks and
#   another direction has only 1-2 off-angle lines that remain unchanged.
# - The output always shows a conspicuous empty middle lane exactly where the
#   aligned family would otherwise be complete, making the hidden-reference rule
#   visible from examples rather than from color ids or metadata.
#
# f) Visual cleanliness
# - No objects overlap. The main family occupies separated parallel offsets with
#   irregular gaps, and distractors sit on different cells entirely.
# - The missing middle lane is not occluded by distractors because the generator
#   reserves the full target bundle, not just the visible output cells.
#
# g) Change inferability
# - No color remapping is introduced. Colors act only as object identity.
# - The two changes reinforce each other structurally: first detect the parallel
#   family, then apply the hidden-reference transform only to that family. A
#   solver that ignores either half will fail on accepted samples.
#
# h) Non-degenerate enforcement
# - Generator rejects samples unless the strongest off-angle direction beats the
#   main family under the old adjacency heuristic, ensuring the distractors are
#   functionally relevant.
# - Generator also builds outputs that exclude the reference lane, so the combo
#   output is different from the visible-reference D12-only variant on every
#   accepted sample.
#
# i) New solver assumptions and generator guarantees
# - main() assumes exactly one candidate direction yields an eligible family with
#   maximal segment count. Generator guarantees this by giving the real family 3
#   or 5 segments and each distractor direction at most 2.
# - main() assumes the reference lane is present in the input, because it must be
#   inferred from the median offset there. Generator always restores that segment
#   in the input even though it omits it from the output.
# - main() assumes aligned non-reference redraws stay in bounds. Output-first
#   construction guarantees the reference span is valid for every family member.
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
    Align the true parallel family to a hidden reference span, remove that reference, and keep distractors unchanged.

    Input structure:
    - The background is the most common color.
    - One direction contains the real task family: 3 or 5 multi-cell segments on distinct parallel tracks.
    - The grid also contains 1 or 2 off-direction multi-cell distractor segments that do not belong to that family.
    - Within the real family, the median track by perpendicular offset is still present in the input and acts as the hidden standard.

    Rule:
    - Detect which direction forms the unique larger eligible parallel family.
    - Use the family member at the median perpendicular offset as the reference extent.
    - Clear only that family from the grid, then redraw every non-reference member so its start matches the reference start and its length matches the reference length.
    - Do not redraw the reference itself, and leave all off-direction distractors untouched.

    Key insight:
    The operative structure is the larger parallel stack, not every visible line; its middle member defines an invisible target interval that the other members snap to before that middle line disappears.
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
    assert len(s_vals) >= 3, "hidden reference needs visible neighbors"
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
        if s["s"] == s_ref:
            continue
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must match grid step"
        k = dt // delta_t
        rr = s["start"][0] + k * dr
        cc = s["start"][1] + k * dc
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
_GEN_RETRIES = 800


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
    if not longable or ref_len <= 2:
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


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
    keep_mask: list[bool] | None = None,
) -> bool:
    if keep_mask is None:
        keep_mask = [True] * len(s_vals)
    for keep, s, t0, length, color in zip(keep_mask, s_vals, starts, lengths, colors):
        if not keep:
            continue
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
    ref_idx = n // 2
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

    main_adj = sum(length - 1 for length in lengths)
    distractors = _place_distractors(d, blocked, colors, main_adj)
    if distractors is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    keep_mask = [i != ref_idx for i in range(n)]
    if not _paint(out, d, s_vals, out_starts, out_lengths, colors, keep_mask=keep_mask):
        return None
    if not _paint_distractors(out, distractors):
        return None

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None
    if not _paint_distractors(inp, distractors):
        return None

    visible_ref_out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(visible_ref_out, d, s_vals, out_starts, out_lengths, colors):
        return None
    if not _paint_distractors(visible_ref_out, distractors):
        return None

    try:
        solved = main(inp)
        if not np.array_equal(solved, out):
            return None
        if np.array_equal(out, visible_ref_out):
            return None
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
