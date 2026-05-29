# Combo variant of e376de54 (index 25)
# combo_spec: D1/1+D4/1+D8/0
# Name: 最短基准+仅裁剪+紧凑裁剪

"""
Design analysis for the combo variant

a) Concrete behavior and trace
   - The object family stays the same: a 16x16 background with 3 or 5 parallel monochrome segments,
     all in one of four directions and on distinct perpendicular offsets.
   - The fused rule changes three linked stages at once:
       1. reference choice: use the unique shortest segment;
       2. length action: only cut longer segments, never extend anything;
       3. framing: crop the final aligned segments to their minimal bounding box.
   - Small trace, horizontal/end-anchored case:
       input lengths by offset = [8, 3, 6], all share the same right endpoint.
       base code would pick the median-offset row, align every row to that row's span, keep 16x16,
       and could extend or trim depending on the median length.
       combo code picks the 3-cell row as reference, shifts every segment so its start matches the
       short row's start, trims the 8-cell and 6-cell rows to length 3, and finally crops away the
       surrounding background. The visible result is a compact 3-column stripe block.
   - Because the reference is already the minimum length, "clip only" becomes a strict one-way
     contraction: every non-reference segment is longer and therefore gets shortened.

b) Complexity calibration
   - The generator keeps the original direction set, odd segment counts, irregular offset gaps, and
     1-3 colors so the task still looks like the same puzzle family.
   - To keep the D1 change visible, the shortest segment is forced to be off the median offset line.
     Otherwise many samples would accidentally agree with the original positional reference rule.
   - To keep the D8 change visible, the aligned output layout must sit at least one cell away from
     every canvas border, guaranteeing that tight crop reduces the 16x16 frame.
   - Freedom is limited to direction, segment count, anchor side, shortest-line offset, shortest
     length, and the longer lengths. That is enough variation without making the relation noisy.

c) Hidden assumptions reviewed against main()
   - Direction detection still assumes same-color adjacency is uniquely strongest along one of the
     four candidate directions. The generator preserves that by drawing only proper separated runs.
   - Segment extraction assumes each segment has a unique predecessor-free start on its offset line.
     Distinct offsets and straight runs preserve this.
   - The original median-offset reference assumption is invalid under the combo and is replaced with
     a new assumption: exactly one segment has minimum length. The generator enforces uniqueness.
   - The original redraw loop always drew target_len cells. Under the combo this is valid only after
     replacing that with min(original_len, target_len); with a shortest reference this simplifies to
     trimming longer lines while leaving the reference unchanged.
   - Tight crop also breaks the old `out = g.copy()` habit because stale foreground inside the crop
     would survive. The combo solver therefore redraws onto a clean background canvas and crops from
     the transformed cells only.

d) Placement legality
   - Generation is output-first. It first chooses the compact aligned target span, then only accepts
     placements where that span is in bounds for every offset and has an outer margin for cropping.
   - Input segments are derived by lengthening away from a shared start or shared end anchor. Every
     non-reference segment must be strictly longer than the shortest reference, so no extension is
     ever needed in the target transformation.
   - Perpendicular offsets remain distinct, with diagonal gaps >= 4, so segments never overlap or
     visually fuse after alignment.

e) Visual observability
   - A human sees one uniquely shortest line in the input, then sees all output lines contract to
     that short span and the canvas shrink around them. Those are visible geometric cues.
   - The shortest line changes position across samples and is explicitly decoupled from the median
     offset, so the puzzle does not collapse back to the source rule.

f) Visual cleanliness
   - No extra markers are introduced. The scene remains only parallel segments on plain background.
   - Irregular spacing keeps tracks visually separate, and tight crop produces a compact stripe block
     rather than a cluttered arrangement.

g) Color inference
   - Colors are preserved as identities of segments. No color ordering, recoloring, or hidden
     color-dependent rule is introduced, so the inference burden stays geometric.

Post-change solver assumptions guaranteed by generate_input()
   - There is at least one multi-cell segment and all extracted segments share one dominant direction.
   - All segment line coordinates are distinct, and diagonal spacing is large enough for clean tracing.
   - Exactly one segment is shortest, and it is not on the median perpendicular offset.
   - Every other segment is strictly longer than the reference, so the one-sided trim rule is active.
   - The transformed layout has a nontrivial outer margin, so tight crop changes the output size.
"""

from pathlib import Path
import random
import sys

import numpy as np

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        _solvers_dir = str(_parent / "solvers")
        if _solvers_dir not in sys.path:
            sys.path.insert(0, _solvers_dir)
        break

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
    return p[0] * d[1] - p[1] * d[0]


def _bbox_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Re-align parallel colored segments to the shortest segment's span, trim only the excess, and return the tight cropped result.

    Input structure:
    - The background is the most frequent color.
    - The foreground consists of several multi-cell monochrome segments on distinct parallel tracks.
    - All segments share one dominant direction, and exactly one segment is the shortest.

    Rule:
    - Detect the common direction and trace every multi-cell segment along it.
    - Use the unique shortest segment as the reference, taking its start as the shared aligned start and its length as the maximum allowed output length.
    - Shift every segment so its start matches the reference start, then draw only `min(segment_length, reference_length)` cells.
    - After redrawing, crop the canvas to the minimal rectangle containing the transformed segments.

    Key insight: The deciding signal is the minimum length, and because the rule never extends shorter lines, the whole pattern contracts to a compact block framed only by the surviving clipped spans.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc
    segments = _extract_segments(g, bg, d)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    min_len = min(s["len"] for s in segments)
    ref_cands = [s for s in segments if s["len"] == min_len]
    assert len(ref_cands) == 1, "shortest reference segment must be unique"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = np.full((h, w), bg, dtype=int)
    transformed_cells = []
    for s in segments:
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = s["start"][0] + k * dr
        c0 = s["start"][1] + k * dc
        draw_len = min(s["len"], target_len)
        rr, cc = r0, c0
        for _ in range(draw_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = s["color"]
            transformed_cells.append((rr, cc))
            rr += dr
            cc += dc

    assert transformed_cells, "transformed output must contain segment pixels"
    r0, c0, r1, c1 = _bbox_from_cells(transformed_cells)
    return out[r0 : r1 + 1, c0 : c1 + 1]


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (2, 6)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_OUTER_MARGIN = 1

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500


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


def _segment_cells(
    d: tuple[int, int], t0: int, s: int, length: int
) -> list[tuple[int, int]] | None:
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
        patterns = [p + [gap] for p in patterns for gap in allowed]
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


def _all_cells_for_layout(
    d: tuple[int, int], s_vals: tuple[int, ...], starts: list[int], lengths: list[int]
) -> list[tuple[int, int]] | None:
    cells = []
    for s, t0, length in zip(s_vals, starts, lengths):
        seg_cells = _segment_cells(d, t0, s, length)
        if seg_cells is None:
            return None
        cells.extend(seg_cells)
    return cells


def _layout_bbox(
    d: tuple[int, int], s_vals: tuple[int, ...], starts: list[int], lengths: list[int]
) -> tuple[int, int, int, int] | None:
    cells = _all_cells_for_layout(d, s_vals, starts, lengths)
    if cells is None:
        return None
    return _bbox_from_cells(cells)


def _has_outer_margin(d: tuple[int, int], s_vals: tuple[int, ...], t0: int, ref_len: int) -> bool:
    bbox = _layout_bbox(d, s_vals, [t0] * len(s_vals), [ref_len] * len(s_vals))
    if bbox is None:
        return False
    r0, c0, r1, c1 = bbox
    return (
        r0 >= _OUTER_MARGIN
        and c0 >= _OUTER_MARGIN
        and r1 <= _H - 1 - _OUTER_MARGIN
        and c1 <= _W - 1 - _OUTER_MARGIN
    )


def _feasible_placements(
    d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int, ref_idx: int
) -> list[tuple[int, str, list[int]]]:
    delta = _step(d)
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        if not _has_outer_margin(d, s_vals, t0, ref_len):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if all(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if all(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len: int, caps: list[int], ref_idx: int) -> list[int] | None:
    lengths = []
    for i, cap in enumerate(caps):
        if i == ref_idx:
            lengths.append(ref_len)
            continue
        hi = min(_MAX_SEG_LEN, cap)
        if hi <= ref_len:
            return None
        lengths.append(random.randint(ref_len + 1, hi))
    assert sum(length == ref_len for length in lengths) == 1
    return lengths


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _crop_to_cells(grid: np.ndarray, cells: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_cells(cells)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    median_idx = n // 2
    ref_idx = random.choice([i for i in range(n) if i != median_idx])
    placements = _feasible_placements(d, s_vals, ref_len, ref_idx)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, caps, ref_idx)
    if lengths is None:
        return None

    out_full = np.full((_H, _W), _BG, dtype=int)
    out_starts = [t0] * n
    out_lengths = [min(length, ref_len) for length in lengths]
    aligned_cells = _all_cells_for_layout(d, s_vals, out_starts, out_lengths)
    if aligned_cells is None:
        return None
    if not _paint(out_full, d, s_vals, out_starts, out_lengths, colors):
        return None
    out = _crop_to_cells(out_full, aligned_cells)
    if out.shape == (_H, _W):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    starts = []
    for length in lengths:
        if anchor == "start":
            starts.append(t0)
        else:
            starts.append(t1 - (length - 1) * delta)

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    segments = _extract_segments(inp, _BG, d)
    if len(segments) != n:
        return None
    seg_lengths = [s["len"] for s in segments]
    if seg_lengths.count(min(seg_lengths)) != 1:
        return None
    ordered = sorted(segments, key=lambda s: s["s"])
    ordered_shortest_idx = min(range(n), key=lambda i: ordered[i]["len"])
    if ordered_shortest_idx == median_idx:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
