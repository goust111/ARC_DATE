# Variant of e376de54
# Dimension: D11 — 线段的连续性与断裂 (meta)
# Change: 每条线段是连续的像素链（对角线每像素步进1行1列，水平/垂直每像素步进1格）... → 线段中间有断裂（两段子线段组成一条逻辑线段）

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg

# Design analysis for the D11 variant
# -----------------------------------
# a) Concrete behavior:
#    Input still contains 3 or 5 parallel same-direction lines on distinct perpendicular offsets,
#    but each logical line is split into exactly two same-color runs with a 1-2 cell gap in between.
#    Output keeps the original task's endpoint-alignment effect: each logical line becomes one
#    continuous segment aligned to the reference line's full start/end extent.
#    Example along a horizontal row:
#      original input line:  A A A A A A
#      variant input line:   A A . A A A
#      solver must treat this as length 6, not two separate lengths 2 and 3.
#      After alignment/cropping, the output draws one continuous aligned line.
#    The affected pipeline step is segment extraction: original code assumes one continuous run per
#    logical segment, while the variant requires run detection first and then same-(color, offset)
#    merging across one internal break.
#
# b) Complexity calibration:
#    The gap is mandatory on every generated logical segment, so the variant is visually obvious.
#    Gap size is restricted to 1-2 cells and each piece has length >= 2, which keeps the rule
#    learnable while still forcing nontrivial grouping. Colors can still repeat across different
#    offsets as in the source task, but never twice on the same offset, so grouping remains visible.
#
# c) Original-code assumptions that break under D11:
#    - "trace while same-color cells continue" used to recover an entire segment; now it recovers
#      only one fragment.
#    - "one recovered segment -> one perpendicular offset" used to hold; now a single logical line
#      can yield two fragments at the same offset.
#    - reference length was computed from one continuous run; now it must use the outer endpoints of
#      the merged fragments, including the gap between them.
#    Clearing and redrawing remain safe once fragments are merged because all original colored cells
#    are explicitly collected before drawing the continuous output.
#
# d) Placement legality:
#    The generator still uses output-first placement on a 16x16 canvas. Parallel offsets obey the
#    original spacing constraints, so different lines do not overlap. Each logical line's two pieces
#    sit on the same offset and stay within the span of a feasible continuous segment, guaranteeing
#    in-bounds output after the solver fills the gap and aligns lengths.
#
# e) Visual observability:
#    Across a few examples, the recurring cue is "two separated same-color pieces on one line become
#    one aligned continuous line". No invisible property such as color id ordering is used. Median
#    reference selection still depends only on visible spatial order of offsets.
#
# f) Visual cleanliness:
#    Only one object family exists: colored line fragments on well-separated parallel tracks.
#    There are no extra markers or overlays, so the grids remain legible despite the added break.
#
# g) Color inference:
#    Colors are preserved exactly. The transform does not depend on which color is used, only on the
#    geometry of same-color fragments sharing an offset, so repeated colors do not introduce a hidden
#    mapping problem.
#
# h) Post-change solver assumptions and generator guarantees:
#    - main() assumes each logical segment contributes one or two runs for a given (color, offset).
#      generate_input() guarantees exactly two runs per logical segment.
#    - main() assumes those runs are non-overlapping and separated by at least one missing cell.
#      generate_input() enforces gap length 1 or 2.
#    - main() assumes the merged logical segments still have unique perpendicular offsets and odd
#      count, which the generator preserves from the source task.

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
                        "len": len(cells),
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
    Align parallel colored line fragments by first reconstructing the broken logical segments they belong to.

    Input structure:
    - bg is the most common color.
    - Several parallel horizontal, vertical, or diagonal lines appear on distinct offsets.
    - Each logical line is broken into two same-color runs on the same offset, with a small gap between them; isolated single pixels may also exist.

    Rule:
    - Detect the dominant shared direction, trace each continuous run, and merge runs that share both color and perpendicular offset into one logical segment spanning across the gap.
    - Choose the logical segment at the median perpendicular offset as the reference.
    - Shift every logical segment so its start matches the reference start, then redraw the full reference span continuously, filling any internal break.
    - Grid size stays the same; isolated single pixels are left untouched.

    Key insight:
    The gap does not split the object logically: two same-color fragments on one parallel track define one hidden segment extent, and that hidden extent is what controls the alignment and resizing.
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
_REF_LEN_RANGE = (6, 8)
_MIN_LOGICAL_LEN = 5
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75
_BREAK_GAPS = [1, 2]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400


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


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int):
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * _step(d)
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


def _choose_lengths(ref_len: int, caps: list[int], n: int):
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if min(_MAX_SEG_LEN, caps[i]) > ref_len]
    shortable = [i for i in nonrefs if _MIN_LOGICAL_LEN < ref_len]
    if not longable or len(shortable) < 2:
        return None

    long_i = random.choice(longable)
    short_choices = [i for i in nonrefs if i != long_i]
    if not short_choices:
        return None
    short_i = random.choice(short_choices)

    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))
    lengths[short_i] = random.randint(_MIN_LOGICAL_LEN, ref_len - 1)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(_MIN_LOGICAL_LEN, min(_MAX_SEG_LEN, caps[i]) + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


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


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals: tuple[int, ...], starts, lengths, colors) -> bool:
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
    starts,
    lengths,
    breaks,
    colors,
) -> bool:
    delta = _step(d)
    for s, t0, total_len, split, color in zip(s_vals, starts, lengths, breaks, colors):
        first_len, gap_len, second_len = split
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


def _all_groups_broken(inp: np.ndarray) -> bool:
    bg = get_bg(inp)
    d = _pick_main_dir(inp, bg)
    segments = _merge_runs_to_segments(_extract_runs(inp, bg, d), d)
    return bool(segments) and all(s["piece_count"] == 2 and s["gap_cells"] >= 1 for s in segments)


def _gen_attempt():
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
    breaks = _choose_breaks(lengths)
    if breaks is None:
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_broken(inp, d, s_vals, starts, lengths, breaks, colors):
        return None
    if not _all_groups_broken(inp):
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
