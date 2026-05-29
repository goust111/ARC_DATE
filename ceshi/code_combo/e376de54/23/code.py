# Combo variant of e376de54 (index 23)
# combo_spec: D11/1+D13/1
# Name: 断裂线段+随机起始位置

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        _ROOT = _parent
        break
assert _ROOT is not None
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg

# Design analysis for combo D11/1 + D13/1
# ----------------------------------------
# a) Concrete behavior:
#    The source task already normalizes several parallel segments to the reference span chosen by
#    median perpendicular offset. This combo keeps that target, but the input now hides each logical
#    line in two ways at once:
#    - D11/1: every logical line is split into exactly two same-color runs with a 1-2 cell gap.
#    - D13/1: non-reference lines no longer share one anchored endpoint; their full logical spans
#      start and end at different places along the common direction.
#    Small horizontal example with reference length 7:
#      reference track:      . R R . R R R . .    -> logical span is columns 1..7
#      inside short track:   . . A A . A A . .    -> merged span is 2..6, shorter and shifted
#      outside long track:   B B . B B B B . B B  -> merged span is 0..9, longer and shifted
#      output: all three tracks become continuous segments on columns 1..7.
#    Relative to the original code, extraction must merge fragments before measuring lengths, and
#    alignment can no longer rely on a shared input start or end.
#
# b) Complexity calibration:
#    The break is mandatory on every logical line, so the D11 signal is always visible. To make D13
#    equally visible, generator samples always contain one non-reference line strictly inside the
#    reference span and one strictly outside it. That guarantees varied starts and varied ends in the
#    same board, instead of a weak near-original case where only one endpoint family changes.
#
# c) Implicit-assumption audit against the original main():
#    - Original tracing assumed one continuous run per logical segment. Broken input violates this,
#      so solver now extracts runs first and merges by (color, perpendicular offset).
#    - Original generator guaranteed all starts or all ends were shared. Random span placement
#      removes that, but solver logic itself only needs each logical segment's recovered t_start and
#      t_end, so once merged segments are reconstructed the redraw rule still applies unchanged.
#    - Original "one recovered segment per offset" assumption becomes "one merged logical segment per
#      offset". Generator preserves unique offsets and never creates two logical segments on one
#      track, even when colors repeat elsewhere.
#    - Clearing remains safe because solver erases only traced fragment cells, then redraws the
#      continuous target span.
#
# d) Placement legality:
#    The generator remains output-first on a 16x16 board. It first chooses a reference span that is
#    simultaneously drawable on every parallel track, then derives per-track input spans that stay on
#    the same track, stay in bounds, and are later split into two valid pieces with piece lengths >=2.
#    Parallel offsets keep the original spacing constraints, so tracks never overlap.
#
# e) Visual observability:
#    Across 2-4 examples, a human sees three repeated cues: same-color fragment pairs belong
#    together, one track sits inside the reference, another extends outside it, and output snaps all
#    tracks to the reference extent. No invisible ordering such as numeric color rank is used.
#
# f) Visual cleanliness:
#    The grid still contains only one object family: broken parallel line segments. The added
#    difficulty comes from fragment grouping and varied span placement, not from extra markers or
#    clutter. Gaps are short and offsets are well separated, so each track remains readable.
#
# g) Change inferability:
#    Colors are preserved exactly and are used only as visible grouping keys for broken fragments on a
#    single track. Because the transformation depends on geometry relative to the reference span, not
#    on color identity, repeated colors across different offsets do not create hidden mappings.
#
# h) Post-change solver assumptions and generator guarantees:
#    - main() assumes each logical segment contributes one or two runs with identical color on one
#      perpendicular offset; generate_input() constructs exactly two runs per track.
#    - main() assumes merged segments have odd count and unique perpendicular offsets; generator keeps
#      the original 3-or-5 count and irregular gap constraints.
#    - main() assumes the reference track is the median by offset, not the longest by length; this
#      follows the supplied original solver/spec and generator preserves that geometry.
#    - main() assumes the random-start variant is not degenerate back to a common-anchor case;
#      generator enforces at least one line strictly inside and one strictly outside the reference
#      span, yielding distinct start and end sets on every accepted sample.

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


def _extract_runs(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
    h, w = g.shape
    dr, dc = d
    runs = []
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
                runs.append(
                    {
                        "color": col,
                        "cells": cells,
                        "start": start,
                        "end": end,
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "s": _perp_dot(start, d),
                    }
                )
    return runs


def _merge_runs_to_segments(runs: list[dict], d: tuple[int, int]) -> list[dict]:
    delta_t = _step(d)
    groups: dict[tuple[int, int], list[dict]] = {}
    for run in runs:
        groups.setdefault((run["color"], run["s"]), []).append(run)

    segments = []
    for (col, s), pieces in groups.items():
        pieces = sorted(pieces, key=lambda part: part["t_start"])
        assert 1 <= len(pieces) <= 2, "each logical segment must have one or two pieces"

        cells = []
        gap_cells = 0
        for i, part in enumerate(pieces):
            if i > 0:
                jump = part["t_start"] - pieces[i - 1]["t_end"]
                assert jump % delta_t == 0, "piece spacing must align with direction"
                gap = jump // delta_t - 1
                assert gap >= 1, "pieces must be separated by a real break"
                gap_cells += gap
            cells.extend(part["cells"])

        t_start = pieces[0]["t_start"]
        t_end = pieces[-1]["t_end"]
        total_len = (t_end - t_start) // delta_t + 1
        segments.append(
            {
                "color": col,
                "pieces": pieces,
                "piece_count": len(pieces),
                "cells": cells,
                "start": pieces[0]["start"],
                "end": pieces[-1]["end"],
                "t_start": t_start,
                "t_end": t_end,
                "s": s,
                "len": total_len,
                "gap_cells": gap_cells,
            }
        )
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct broken parallel lines and then snap every logical line to the hidden reference span.

    Input structure:
    - The background is the most frequent color.
    - Several parallel horizontal, vertical, or diagonal tracks appear at distinct perpendicular offsets.
    - Each logical line is split into two same-color runs on one track, and different tracks may begin and end at unrelated positions along the shared direction.

    Rule:
    - Detect the common direction, trace each continuous run, and merge same-color runs that lie on the same parallel track into one logical segment.
    - Select the logical segment at the median perpendicular offset as the reference span.
    - Erase the original fragments, then redraw every logical segment as one continuous line whose start and end both match the reference segment.
    - Grid size stays fixed, and isolated single pixels remain untouched.

    Key insight:
    The visible objects are fragments, but the transformation is controlled by hidden full spans: first recover which two fragments form one line, then ignore their random placement and normalize all lines to the reference track's start and end.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = _step(d)

    runs = _extract_runs(g, bg, d)
    assert runs, "at least one multi-cell run must exist"
    segments = _merge_runs_to_segments(runs, d)
    assert segments, "at least one logical segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]

    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

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


# --------------- generator (o5x) ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (7, 8)
_MIN_LOGICAL_LEN = 5
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_BREAK_GAPS = [1, 2]
_NONREF_EXACT_REF_PROB = 0.1

_T_SCAN_RANGE = (-30, 31)
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


def _choose_colors(s_vals: tuple[int, ...]) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _plan_candidates(d: tuple[int, int], ref_t0: int, ref_len: int, s: int) -> list[dict]:
    delta = _step(d)
    ref_t1 = ref_t0 + (ref_len - 1) * delta
    left_room = _max_len_to(d, ref_t0, s) - 1
    right_room = _max_len_from(d, ref_t1, s) - 1
    max_len = min(_MAX_SEG_LEN, left_room + ref_len + right_room)
    plans = []
    for length in range(_MIN_LOGICAL_LEN, max_len + 1):
        k_min = -left_room
        k_max = right_room + ref_len - length
        for start_shift in range(k_min, k_max + 1):
            end_shift = start_shift + length - ref_len
            if start_shift == 0 and end_shift == 0:
                kind = "exact_ref"
            elif start_shift > 0 and end_shift < 0:
                kind = "inside_both"
            elif start_shift < 0 and end_shift > 0:
                kind = "outside_both"
            elif start_shift == 0:
                kind = "same_start"
            elif end_shift == 0:
                kind = "same_end"
            elif length == ref_len:
                kind = "shifted_same_len"
            else:
                kind = "mixed"
            plans.append(
                {
                    "kind": kind,
                    "start_shift": start_shift,
                    "end_shift": end_shift,
                    "start_t": ref_t0 + start_shift * delta,
                    "length": length,
                }
            )
    return plans


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int):
    ref_idx = len(s_vals) // 2
    nonrefs = [i for i in range(len(s_vals)) if i != ref_idx]
    placements = []
    for ref_t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, ref_t0, s, ref_len) is None for s in s_vals):
            continue
        per_track = [_plan_candidates(d, ref_t0, ref_len, s) for s in s_vals]
        pairable = []
        for inside_i in nonrefs:
            if not any(p["kind"] == "inside_both" for p in per_track[inside_i]):
                continue
            for outside_i in nonrefs:
                if inside_i == outside_i:
                    continue
                if any(p["kind"] == "outside_both" for p in per_track[outside_i]):
                    pairable.append((inside_i, outside_i))
        if pairable:
            placements.append((ref_t0, per_track, pairable))
    return placements


def _pick_plan(plans: list[dict], allowed_kinds: set[str]) -> dict | None:
    options = [p for p in plans if p["kind"] in allowed_kinds]
    if not options:
        return None
    return dict(random.choice(options))


def _pick_any_nonref_plan(plans: list[dict]) -> dict | None:
    preferred = [p for p in plans if p["kind"] != "exact_ref"]
    if preferred and random.random() > _NONREF_EXACT_REF_PROB:
        return dict(random.choice(preferred))
    if plans:
        return dict(random.choice(plans))
    return None


def _choose_input_plans(
    d: tuple[int, int], ref_t0: int, ref_len: int, s_vals: tuple[int, ...], per_track, pairable
) -> list[dict] | None:
    ref_idx = len(s_vals) // 2
    inside_i, outside_i = random.choice(pairable)
    plans = [None] * len(s_vals)
    plans[ref_idx] = {
        "kind": "exact_ref",
        "start_shift": 0,
        "end_shift": 0,
        "start_t": ref_t0,
        "length": ref_len,
    }
    plans[inside_i] = _pick_plan(per_track[inside_i], {"inside_both"})
    plans[outside_i] = _pick_plan(per_track[outside_i], {"outside_both"})
    if plans[inside_i] is None or plans[outside_i] is None:
        return None

    for i in range(len(s_vals)):
        if plans[i] is not None:
            continue
        plans[i] = _pick_any_nonref_plan(per_track[i])
        if plans[i] is None:
            return None
    return plans


def _split_options(total_len: int) -> list[tuple[int, int, int]]:
    options = []
    for gap in _BREAK_GAPS:
        remaining = total_len - gap
        for first_len in range(2, remaining - 1):
            second_len = remaining - first_len
            if second_len >= 2:
                options.append((first_len, gap, second_len))
    return options


def _choose_breaks(lengths: list[int]) -> list[tuple[int, int, int]] | None:
    breaks = []
    for length in lengths:
        options = _split_options(length)
        if not options:
            return None
        breaks.append(random.choice(options))
    return breaks


def _paint(
    grid: np.ndarray, d: tuple[int, int], s_vals: tuple[int, ...], starts, lengths, colors
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _paint_broken(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    plans: list[dict],
    breaks: list[tuple[int, int, int]],
    colors: tuple[int, ...],
) -> bool:
    delta = _step(d)
    for s, plan, split, color in zip(s_vals, plans, breaks, colors):
        first_len, gap_len, second_len = split
        total_len = plan["length"]
        t0 = plan["start_t"]
        assert first_len + gap_len + second_len == total_len
        cells1 = _segment_cells(d, t0, s, first_len)
        cells2 = _segment_cells(d, t0 + (first_len + gap_len) * delta, s, second_len)
        if cells1 is None or cells2 is None:
            return False
        for r, c in cells1 + cells2:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _analyze_generated_input(inp: np.ndarray) -> dict:
    bg = get_bg(inp)
    d = _pick_main_dir(inp, bg)
    segments = _merge_runs_to_segments(_extract_runs(inp, bg, d), d)
    segments = sorted(segments, key=lambda s: s["s"])
    ref = segments[len(segments) // 2]
    starts = [s["t_start"] for s in segments]
    ends = [s["t_end"] for s in segments]
    nonrefs = [s for i, s in enumerate(segments) if i != len(segments) // 2]
    return {
        "direction": d,
        "segments": segments,
        "ref": ref,
        "unique_starts": len(set(starts)),
        "unique_ends": len(set(ends)),
        "inside_count": sum(
            s["t_start"] > ref["t_start"] and s["t_end"] < ref["t_end"] for s in nonrefs
        ),
        "outside_count": sum(
            s["t_start"] < ref["t_start"] and s["t_end"] > ref["t_end"] for s in nonrefs
        ),
        "all_broken": all(s["piece_count"] == 2 and s["gap_cells"] >= 1 for s in segments),
    }


def fusion_metrics(inp: np.ndarray) -> dict:
    info = _analyze_generated_input(inp)
    return {
        "segment_count": len(info["segments"]),
        "unique_starts": info["unique_starts"],
        "unique_ends": info["unique_ends"],
        "inside_count": info["inside_count"],
        "outside_count": info["outside_count"],
        "all_broken": info["all_broken"],
    }


def _is_strong_fusion_instance(inp: np.ndarray) -> bool:
    info = _analyze_generated_input(inp)
    return (
        info["all_broken"]
        and info["unique_starts"] >= 3
        and info["unique_ends"] >= 3
        and info["inside_count"] >= 1
        and info["outside_count"] >= 1
    )


def _gen_attempt():
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    ref_t0, per_track, pairable = random.choice(placements)
    plans = _choose_input_plans(d, ref_t0, ref_len, s_vals, per_track, pairable)
    if plans is None:
        return None
    colors = _choose_colors(s_vals)
    lengths = [plan["length"] for plan in plans]
    breaks = _choose_breaks(lengths)
    if breaks is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [ref_t0] * n, [ref_len] * n, colors):
        return None

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_broken(inp, d, s_vals, plans, breaks, colors):
        return None
    if not _is_strong_fusion_instance(inp):
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
