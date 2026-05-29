# Combo variant of e376de54 (index 24)
# combo_spec: D1/0+D4/0+D8/2
# Name: 最长基准+仅延伸+网格扩展

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "solvers" / "arc_utils.py"
        if candidate.exists():
            sys.path.append(str(parent / "solvers"))
            break
    from arc_utils import get_bg


# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - Keep the original object family: 3 or 5 straight parallel segments on distinct offsets, all
#      sharing one dominant direction. The output still realigns every segment to one reference start.
#    - D1 changes which reference is chosen: use the unique longest segment instead of the median
#      offset segment. D4 then becomes "extend only" in a fused way: because the reference is already
#      the longest, every non-reference segment is strictly shorter and therefore only grows.
#    - D8 matters on diagonal scenes. Example with direction (1,1): if the longest visible segment on
#      one diagonal occupies t=10..20, another offset line may fit only t=12..20 inside 16x16. That
#      shorter line is legal in the input, but after alignment to the longest segment's start it must
#      occupy t=10..20 too, so two new cells appear outside the old canvas. The output therefore keeps
#      the original 16x16 area and adds only the missing border rows/cols needed to show those cells.
#    - Compared with the original seed, the visible differences are: the target span is set by the
#      visibly longest line rather than the median line, no segment is ever shortened, and the frame
#      can grow beyond 16x16.
# b) Complexity calibration:
#    - If axis-aligned directions were kept, grid expansion would never trigger under this task's
#      "same reference t-range on every line" geometry. To make D8/2 observable rather than dead
#      metadata, the generator restricts to the two diagonal directions where the same t-range lands at
#      different absolute coordinates on different offsets.
#    - The generator also forces the longest segment away from the median offset and requires at least
#      one non-reference line whose full aligned span would leave the source grid. This prevents the
#      combo from collapsing back to either the original task or to plain D1 alone.
# c) Implicit assumptions review:
#    - `_pick_main_dir()` assumes one direction wins the same-color adjacency count. The generator keeps
#      all useful objects as separated diagonal runs, so the true direction is unique.
#    - Segment extraction assumes each logical object is a contiguous monochrome run of length >= 2.
#      The generator paints only such runs and forbids overlap.
#    - Reference selection now assumes a unique longest segment. The generator sets exactly one segment
#      to `ref_len` and makes every other segment strictly shorter.
#    - Original in-bounds redraw assertions no longer hold after fusion: on diagonal offsets, copying
#      the reference t-range to another line can produce negative rows/cols or overflow the far edge.
#      The solver therefore renders in absolute coordinates first, then expands the canvas just enough
#      to include both the old grid rectangle and every transformed segment cell.
#    - Because D4 is fused with D1, `draw_len` must still be computed as `max(old_len, target_len)`;
#      generator guarantees this equals `target_len` for non-reference lines, but the solver keeps the
#      one-way logic explicit.
# d) Placement legality:
#    - Input construction stays anchor-based: all segments share either the same start t or the same end
#      t, matching the base family. For each non-reference diagonal, the generator checks the maximum
#      in-bounds length available from that shared anchor and chooses an input length inside that limit.
#    - At least one non-reference line has capacity `< ref_len`, which is exactly what forces expansion
#      after alignment. Distinct offsets with gaps 4 or 6 keep diagonal lines from touching.
# e) Visual observability:
#    - A human sees one line is longest, all others end up matching that longer span, and the new pixels
#      appear only on the outer border created by the larger output frame. The rule does not depend on
#      color ids, hidden ordering, or any non-visual attribute.
# f) Visual cleanliness:
#    - The scene remains a sparse bundle of diagonals. Expansion only adds blank margin plus the newly
#      extended tips, so the output is easy to parse; there is no dense overlap or mixed-object clutter.
# g) Color inference:
#    - Colors are preserved exactly. They identify segments only; the combo adds no recoloring step and
#      therefore no extra color-mapping burden.
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes the longest segment is unique, perpendicular offsets are unique, and transformed
#      segments do not overlap after alignment. `generate_input()` guarantees all three.
#    - `main()` assumes any out-of-bounds transformed cells can be handled by expanding the canvas
#      instead of clipping. The generator verifies every produced input by comparing `main(input)` with
#      an independently rendered expected output.

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


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _coord_from_ts(d: tuple[int, int], t: int, s: int) -> tuple[int, int] | None:
    if d == (0, 1):
        return s, t
    if d == (1, 0):
        return t, -s
    if d == (1, 1):
        if (t + s) % 2 or (t - s) % 2:
            return None
        return (t + s) // 2, (t - s) // 2
    assert d == (1, -1)
    if (t - s) % 2 or (-s - t) % 2:
        return None
    return (t - s) // 2, (-s - t) // 2


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int] | None) -> bool:
    if cell is None:
        return False
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _segment_cells(
    d: tuple[int, int],
    t0: int,
    s: int,
    length: int,
    *,
    shape: tuple[int, int] | None = None,
) -> list[tuple[int, int]] | None:
    delta_t = _step(d)
    cells = []
    for k in range(length):
        cell = _coord_from_ts(d, t0 + k * delta_t, s)
        if cell is None:
            return None
        if shape is not None and not _in_bounds(shape, cell):
            return None
        cells.append(cell)
    return cells


def _max_len_from(shape: tuple[int, int], d: tuple[int, int], t0: int, s: int) -> int:
    delta_t = _step(d)
    length = 0
    while _in_bounds(shape, _coord_from_ts(d, t0 + length * delta_t, s)):
        length += 1
    return length


def _max_len_to(shape: tuple[int, int], d: tuple[int, int], t1: int, s: int) -> int:
    delta_t = _step(d)
    length = 0
    while _in_bounds(shape, _coord_from_ts(d, t1 - length * delta_t, s)):
        length += 1
    return length


def _extract_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
    h, w = g.shape
    dr, dc = d
    segments = []
    for col in [int(c) for c in np.unique(g) if int(c) != bg]:
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


def _render_expanded(
    base_grid: np.ndarray,
    bg: int,
    segments: list[dict],
    d: tuple[int, int],
    ref_t0: int,
    target_len: int,
) -> np.ndarray:
    h, w = base_grid.shape
    dr, dc = d
    delta_t = _step(d)

    transformed = []
    for seg in segments:
        dt = ref_t0 - seg["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = seg["start"][0] + k * dr
        c0 = seg["start"][1] + k * dc
        draw_len = max(seg["len"], target_len)
        rr, cc = r0, c0
        for _ in range(draw_len):
            transformed.append((rr, cc, seg["color"]))
            rr += dr
            cc += dc

    assert transformed, "transformed output must contain segment pixels"
    min_r = min([0] + [r for r, _, _ in transformed])
    max_r = max([h - 1] + [r for r, _, _ in transformed])
    min_c = min([0] + [c for _, c, _ in transformed])
    max_c = max([w - 1] + [c for _, c, _ in transformed])

    row_off = -min_r
    col_off = -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    out[row_off : row_off + h, col_off : col_off + w] = base_grid

    for seg in segments:
        for r, c in seg["cells"]:
            out[r + row_off, c + col_off] = bg

    for r, c, color in transformed:
        rr = r + row_off
        cc = c + col_off
        cur = int(out[rr, cc])
        assert cur in (bg, color), "transformed segments must not overlap unrelated colors"
        out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align several parallel segments to the span of the unique longest segment, extending the canvas only when that longer span does not fit on every parallel line.

    Input structure:
    - The background is the most frequent color.
    - Foreground objects are multiple multi-cell segments in one shared direction, each on a distinct parallel track, with one common start-anchor or end-anchor pattern from the base task.
    - Segment lengths differ, and the visibly longest segment is unique.

    Rule:
    - Detect the dominant direction and trace every multi-cell segment in that direction.
    - Use the unique longest segment as the reference, taking its start position and length as the target span.
    - Shift every segment along the shared direction so its start matches the reference start, then redraw it with `max(original_length, reference_length)`.
    - If any redrawn cell would fall outside the original canvas, enlarge the output grid just enough to contain both the old canvas and all transformed cells.

    Key insight:
    - The longest visible run fixes the common span; every other run only grows toward that span, and diagonal offsets can force the output frame to expand because the same reference t-range does not land inside the old grid on every parallel line.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    delta_t = _step(d)
    segments = _extract_segments(g, bg, d)

    assert segments, "at least one multi-cell segment must exist"
    s_vals = sorted(seg["s"] for seg in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    max_len = max(seg["len"] for seg in segments)
    ref_cands = [seg for seg in segments if seg["len"] == max_len]
    assert len(ref_cands) == 1, "reference segment must be the unique longest"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    return _render_expanded(g, bg, segments, d, ref_t0, target_len)


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

# Diagonals only: same t-range can go out of bounds on another parallel line, making D8/2 observable.
_DIRS = [(1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (5, 8)
_PERP_GAP_DIAG = [4, 6]
_MAX_DISTINCT_COLORS = 3

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 1200


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (1, 1):
        return -15, 15
    assert d == (1, -1)
    return -30, 0


def _gap_patterns(n: int) -> list[list[int]]:
    lo_gap = _PERP_GAP_DIAG
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [gap] for p in patterns for gap in lo_gap]
    return [p for p in patterns if ((n == 3 and p[0] != p[1]) or (n == 5 and len(set(p)) > 1))]


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    patterns = []
    for gaps in _gap_patterns(n):
        if sum(gaps) <= hi - lo:
            patterns.append(gaps)
    gaps = random.choice(patterns)
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _build_segments_meta(
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> list[dict] | None:
    segments = []
    shape = (_H, _W)
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length, shape=shape)
        if cells is None:
            return None
        start = cells[0]
        end = cells[-1]
        segments.append(
            {
                "color": color,
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


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    shape = grid.shape
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length, shape=shape)
        if cells is None:
            return False
        for r, c in cells:
            if int(grid[r, c]) != _BG:
                return False
            grid[r, c] = color
    return True


def _feasible_layouts(
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    ref_idx: int,
    ref_len: int,
) -> list[tuple[int, str, list[int], tuple[int, ...]]]:
    shape = (_H, _W)
    delta_t = _step(d)
    layouts = []
    for t0 in range(*_T_SCAN_RANGE):
        ref_cells = _segment_cells(d, t0, s_vals[ref_idx], ref_len, shape=shape)
        if ref_cells is None:
            continue
        t1 = t0 + (ref_len - 1) * delta_t
        for anchor in ("start", "end"):
            caps = []
            expanding = []
            ok = True
            for i, s in enumerate(s_vals):
                if anchor == "start":
                    cap = _max_len_from(shape, d, t0, s)
                else:
                    cap = _max_len_to(shape, d, t1, s)
                caps.append(cap)
                if i == ref_idx:
                    if cap < ref_len:
                        ok = False
                        break
                else:
                    if cap < 2:
                        ok = False
                        break
                    if cap < ref_len:
                        expanding.append(i)
            if ok and expanding:
                layouts.append((t0, anchor, caps, tuple(expanding)))
    return layouts


def _choose_lengths(ref_len: int, caps: list[int], ref_idx: int) -> list[int] | None:
    lengths = [None] * len(caps)
    lengths[ref_idx] = ref_len
    for i, cap in enumerate(caps):
        if i == ref_idx:
            continue
        hi = min(cap, ref_len - 1)
        if hi < 2:
            return None
        lengths[i] = random.randint(2, hi)
    return lengths


def _old_median_same_canvas(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    segments = _extract_segments(g, bg, d)
    delta_t = _step(d)
    s_vals = sorted(seg["s"] for seg in segments)
    assert len(s_vals) % 2 == 1
    s_ref = s_vals[len(s_vals) // 2]
    ref = [seg for seg in segments if seg["s"] == s_ref][0]
    target_len = (ref["t_end"] - ref["t_start"]) // delta_t + 1
    return _render_expanded(g, bg, segments, d, ref["t_start"], target_len)


def _build_case():
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    median_idx = n // 2
    ref_idx = random.choice([i for i in range(n) if i != median_idx])
    layouts = _feasible_layouts(d, s_vals, ref_idx, ref_len)
    if not layouts:
        return None

    t0, anchor, caps, expanding = random.choice(layouts)
    lengths = _choose_lengths(ref_len, caps, ref_idx)
    if lengths is None:
        return None
    colors = _choose_colors(n)

    delta_t = _step(d)
    t1 = t0 + (ref_len - 1) * delta_t
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta_t for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    segments = _build_segments_meta(d, s_vals, starts, lengths, colors)
    if segments is None:
        return None
    expected = _render_expanded(inp, _BG, segments, d, t0, ref_len)

    meta = {
        "direction": d,
        "anchor": anchor,
        "ref_idx": ref_idx,
        "median_idx": median_idx,
        "ref_len": ref_len,
        "input_lengths": tuple(lengths),
        "expanding_indices": tuple(expanding),
        "expanded_shape": tuple(expected.shape),
        "expanded_area_delta": int(expected.size - inp.size),
    }
    return inp, expected, meta


def _gen_attempt():
    case = _build_case()
    if case is None:
        return None
    inp, expected, meta = case
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if meta["expanded_shape"] == inp.shape:
        return None
    if meta["ref_idx"] == meta["median_idx"]:
        return None
    if np.array_equal(_old_median_same_canvas(inp), solved):
        return None
    return inp


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
