# Combo variant of e376de54 (index 17)
# combo_spec: D1/0+D8/2
# Name: 最长基准+动态网格扩展

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[5] / "solvers"))
    from arc_utils import get_bg

# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - Keep the same object family: 3 or 5 parallel straight segments, all on distinct tracks, all
#      sharing one direction and one anchored end convention in the input.
#    - D1/0 changes the reference choice from the median-offset segment to the unique longest input
#      segment. D8/2 then matters because that longest span may be valid on its own track but not on
#      every other parallel diagonal track inside the original 16x16 canvas.
#    - Example: three "\"-diagonal segments share the same start along t. Lengths are [7, 4, 5], and
#      the 7-cell segment sits on the topmost diagonal. Old logic would copy the middle track's 4-cell
#      extent onto all tracks inside the fixed 16x16 grid. The fused logic copies the 7-cell extent
#      instead; on a lower diagonal that extra tail falls below the old border, so the output adds
#      rows/cols and then paints the fully extended segment there.
#    - The visible difference is therefore twofold: the common target extent comes from the longest
#      line, and the canvas itself may grow when another parallel track cannot host that extent inside
#      the old boundary.
# b) Complexity calibration:
#    - Solver still supports all four original directions, but the generator concentrates on the two
#      diagonal directions because orthogonal lines cannot make D8/2 observable: copying the same row
#      or column range to another row/column never needs expansion.
#    - The segment count remains odd and the offset gaps stay irregular so the scene still reads like
#      the original family instead of collapsing into a trivial repeated-pattern task.
#    - The unique longest reference is forced onto an outer track rather than the median track, making
#      the D1 change visually strong across few-shot examples.
# c) Implicit assumptions review:
#    - Direction detection still relies on one adjacency direction being uniquely dominant; generator
#      keeps all meaningful runs parallel and well separated.
#    - Segment extraction still assumes each useful object is one contiguous run of length >= 2.
#      Generator paints only such runs and never overlaps them.
#    - Reference selection now assumes a unique maximum length; generator enforces exactly one longest
#      segment and makes every other segment strictly shorter.
#    - The original solver assumed the redrawn extent stayed in-bounds. That assumption is removed:
#      this solver first computes the full target coordinates in world space, then pads the canvas as
#      needed before repainting.
# d) Placement legality:
#    - Generation is output-first in world coordinates. The full longest-reference extent is chosen
#      first, then shorter input segments are carved as anchored prefixes/suffixes that fit in 16x16.
#    - Every input segment is checked for in-bounds placement, and every output segment lies on its
#      own parallel track, so there is no overlap or accidental touching.
# e) Visual observability:
#    - Every generated sample has two measurable properties: the longest segment is not the median
#      offset segment, and at least one non-reference segment would leave the original 16x16 frame if
#      extended to the longest span. A human can therefore see both the changed reference rule and the
#      new expanding canvas behavior.
# f) Visual cleanliness:
#    - Large even gaps between diagonal tracks avoid corner-touching and keep each segment visually
#      isolated. The output only adds blank padding plus the extended tails, so the scene stays tidy.
# g) Color inference:
#    - Colors are preserved; they only label segments. No hidden color remapping is introduced, and
#      the rule is entirely geometric.
# h) New solver assumptions and generator guarantees:
#    - main() assumes all meaningful segments share one direction, have unique perpendicular offsets,
#      and include one unique longest run; generate_input() guarantees these.
#    - main() now allows negative / over-edge target coordinates by expanding the canvas, and the
#      generator guarantees at least one sample-side extension actually occurs instead of degenerating
#      to the fixed-size behavior.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


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


def _cell_from_ts(
    d: tuple[int, int], t: int, s: int, bounds: tuple[int, int] | None = None
) -> tuple[int, int] | None:
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

    if bounds is not None:
        h, w = bounds
        if not (0 <= r < h and 0 <= c < w):
            return None
    return int(r), int(c)


def _segment_cells_from_t(
    d: tuple[int, int],
    t0: int,
    s: int,
    length: int,
    bounds: tuple[int, int] | None = None,
) -> list[tuple[int, int]] | None:
    delta_t = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta_t, s, bounds=bounds)
        if cell is None:
            return None
        cells.append(cell)
    return cells


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


def _select_reference(segments: list[dict], d: tuple[int, int], mode: str) -> tuple[dict, int, int]:
    delta_t = _step(d)
    if mode == "longest":
        max_len = max(s["len"] for s in segments)
        ref_cands = [s for s in segments if s["len"] == max_len]
        assert len(ref_cands) == 1, "reference segment must be the unique longest"
        ref = ref_cands[0]
    elif mode == "median":
        s_vals = sorted(s["s"] for s in segments)
        assert len(s_vals) % 2 == 1, "segment count must have a single median"
        assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
        s_ref = s_vals[len(s_vals) // 2]
        ref_cands = [s for s in segments if s["s"] == s_ref]
        assert len(ref_cands) == 1, "median reference segment must be unique"
        ref = ref_cands[0]
    else:
        raise ValueError(mode)

    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1
    return ref, ref_t0, target_len


def _render_dynamic(
    g: np.ndarray,
    bg: int,
    d: tuple[int, int],
    segments: list[dict],
    ref_t0: int,
    target_len: int,
) -> np.ndarray:
    h, w = g.shape
    dr, dc = d
    delta_t = _step(d)

    shifted = []
    min_r, max_r = 0, h - 1
    min_c, max_c = 0, w - 1
    for seg in segments:
        dt = ref_t0 - seg["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        rr = seg["start"][0] + k * dr
        cc = seg["start"][1] + k * dc
        cells = []
        for _ in range(target_len):
            cells.append((rr, cc))
            min_r = min(min_r, rr)
            max_r = max(max_r, rr)
            min_c = min(min_c, cc)
            max_c = max(max_c, cc)
            rr += dr
            cc += dc
        shifted.append((seg["color"], cells))

    row_off, col_off = -min_r, -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)

    out[row_off : row_off + h, col_off : col_off + w] = g
    for seg in segments:
        for r, c in seg["cells"]:
            out[r + row_off, c + col_off] = bg
    for color, cells in shifted:
        for r, c in cells:
            out[r + row_off, c + col_off] = color
    return out


def _solve_with_mode(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    segments = _extract_segments(g, bg, d)
    assert segments, "at least one multi-cell segment must exist"
    assert len({s["s"] for s in segments}) == len(segments), "segment line coordinates must be unique"
    _ref, ref_t0, target_len = _select_reference(segments, d, mode)
    return _render_dynamic(g, bg, d, segments, ref_t0, target_len)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize several parallel colored segments to the span of the longest one, expanding the canvas if needed.

    Input structure:
    - The most frequent color is the background.
    - Foreground objects are multiple parallel straight segments on distinct horizontal, vertical, or
      diagonal tracks, with varied lengths and one shared anchoring convention along their direction.

    Rule:
    - Detect the common direction and extract each multi-cell segment.
    - Choose the unique longest segment as the reference and copy its start/end range to every other
      segment by shifting them along the shared direction.
    - If any redrawn segment would extend beyond the current grid border, enlarge the output canvas
      just enough to contain the original 16x16 area plus every extended tail. Isolated non-segment
      pixels, if any, are preserved in place relative to the expanded canvas.

    Key insight:
    - The target extent comes from the visibly longest line, and the board boundary is not sacred:
      all segments must reach that same span even when doing so forces the output grid to grow.
    """
    return _solve_with_mode(grid, "longest")


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (4, 8)
_MAX_DISTINCT_COLORS = 3
_PERP_GAP_DIAG = [4, 6]
_T_SCAN_RANGE = (-30, 31)
_MIN_TOTAL_PADDING = 2
_GEN_RETRIES = 400


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(n: int) -> list[list[int]]:
    lo, hi = -30, 30
    max_span = hi - lo
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [g] for p in patterns for g in _PERP_GAP_DIAG]
    return [
        p
        for p in patterns
        if sum(p) <= max_span and ((n == 3 and p[0] != p[1]) or (n == 5 and len(set(p)) > 1))
    ]


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(n))
    parity = random.randrange(2)
    s0_choices = [s0 for s0 in range(lo, hi - sum(gaps) + 1) if s0 % 2 == parity]
    s0 = random.choice(s0_choices)
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _max_prefix_in_input(d: tuple[int, int], t0: int, s: int) -> int:
    delta_t = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta_t, s, bounds=(_H, _W)) is not None:
        length += 1
    return length


def _max_suffix_in_input(d: tuple[int, int], t1: int, s: int) -> int:
    delta_t = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta_t, s, bounds=(_H, _W)) is not None:
        length += 1
    return length


def _padding_amount(cells: list[tuple[int, int]]) -> int:
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)
    return (
        max(0, -min_r)
        + max(0, max_r - (_H - 1))
        + max(0, -min_c)
        + max(0, max_c - (_W - 1))
    )


def _candidate_plans(
    d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int, ref_idx: int
) -> list[tuple[int, str, list[int], int]]:
    plans = []
    delta_t = _step(d)
    for t0 in range(*_T_SCAN_RANGE):
        ref_cells = _segment_cells_from_t(d, t0, s_vals[ref_idx], ref_len, bounds=(_H, _W))
        if ref_cells is None:
            continue
        t1 = t0 + (ref_len - 1) * delta_t
        start_caps = [_max_prefix_in_input(d, t0, s) for s in s_vals]
        end_caps = [_max_suffix_in_input(d, t1, s) for s in s_vals]
        full_cells = []
        valid = True
        for s in s_vals:
            seg_cells = _segment_cells_from_t(d, t0, s, ref_len, bounds=None)
            if seg_cells is None:
                valid = False
                break
            full_cells.extend(seg_cells)
        if not valid:
            continue
        padding = _padding_amount(full_cells)
        if padding < _MIN_TOTAL_PADDING:
            continue

        for anchor, caps in (("start", start_caps), ("end", end_caps)):
            if caps[ref_idx] < ref_len:
                continue
            if any(caps[i] < 2 for i in range(len(s_vals)) if i != ref_idx):
                continue
            if not any(caps[i] < ref_len for i in range(len(s_vals)) if i != ref_idx):
                continue
            plans.append((t0, anchor, caps, padding))
    return plans


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len: int, caps: list[int], ref_idx: int) -> list[int] | None:
    n = len(caps)
    lengths = [None] * n
    lengths[ref_idx] = ref_len
    constrained = [i for i in range(n) if i != ref_idx and caps[i] < ref_len]
    if not constrained:
        return None

    border_i = random.choice(constrained)
    lengths[border_i] = caps[border_i]

    for i in range(n):
        if i == ref_idx or i == border_i:
            continue
        hi = min(ref_len - 1, caps[i])
        if hi < 2:
            return None
        lengths[i] = random.randint(2, hi)

    return [int(x) for x in lengths]


def _paint_input(
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> np.ndarray | None:
    grid = np.full((_H, _W), _BG, dtype=int)
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells_from_t(d, t0, s, length, bounds=(_H, _W))
        if cells is None:
            return None
        for r, c in cells:
            if grid[r, c] != _BG:
                return None
            grid[r, c] = color
    return grid


def _expected_output(
    d: tuple[int, int], s_vals: tuple[int, ...], t0: int, ref_len: int, colors: tuple[int, ...]
) -> np.ndarray | None:
    all_cells = []
    for s in s_vals:
        cells = _segment_cells_from_t(d, t0, s, ref_len, bounds=None)
        if cells is None:
            return None
        all_cells.append(cells)

    flat = [cell for cells in all_cells for cell in cells]
    min_r = min([0] + [r for r, _ in flat])
    max_r = max([_H - 1] + [r for r, _ in flat])
    min_c = min([0] + [c for _, c in flat])
    max_c = max([_W - 1] + [c for _, c in flat])
    row_off, col_off = -min_r, -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), _BG, dtype=int)
    for color, cells in zip(colors, all_cells):
        for r, c in cells:
            out[r + row_off, c + col_off] = color
    return out


def _choose_ref_idx(n: int) -> int:
    return random.choice([0, n - 1])


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    ref_idx = _choose_ref_idx(n)
    plans = _candidate_plans(d, s_vals, ref_len, ref_idx)
    if not plans:
        return None

    t0, anchor, caps, _padding = random.choice(plans)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, caps, ref_idx)
    if lengths is None:
        return None

    delta_t = _step(d)
    t1 = t0 + (ref_len - 1) * delta_t
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta_t for length in lengths]

    inp = _paint_input(d, s_vals, starts, lengths, colors)
    if inp is None:
        return None

    expected = _expected_output(d, s_vals, t0, ref_len, colors)
    if expected is None or expected.shape == inp.shape:
        return None

    try:
        solved = main(inp)
        if not np.array_equal(solved, expected):
            return None
        if np.array_equal(_solve_with_mode(inp, "median"), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
