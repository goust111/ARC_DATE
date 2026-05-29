# Combo variant of e376de54 (index 33)
# combo_spec: D1/0+D9/0
# Name: 最长基准+迭代缩减

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import get_bg

# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - The input still contains one family of parallel monochrome segments on distinct offsets, all
#      sharing one anchored endpoint direction. The original task would keep every segment and snap all
#      of them to one reference extent. This combo changes both "which segment defines the extent" and
#      "which segments survive".
#    - Small example: four horizontal segments with shared left endpoint and lengths [9, 6, 4, 3].
#      Original rule (median-offset reference) might output four copies of length 4. D1 alone would
#      output four copies of length 9. This combo first removes the unique longest segment (length 9),
#      then uses the new unique longest segment (length 6) as the reference and redraws only the
#      remaining segments to length 6. Final output therefore has three segments, all length 6.
#    - Because snapping every survivor to the new longest instantly equalizes their lengths, the
#      "iterative" wording collapses to one effective convergence round. The solver still implements the
#      loop literally so the code matches the stated process, but the generator makes that one round
#      geometrically visible and decisive.
# b) Complexity calibration:
#    - The generator fixes the input segment count at 5 instead of mixing 3 and 5. That keeps the
#      scene rich enough that deleting one line and then reusing a non-median survivor as reference is
#      visually obvious, while still leaving enough empty space on a 16x16 board.
#    - The removed longest segment and the post-removal reference are both forced onto non-median
#      offsets. This prevents the combo from looking like the original median-based task with one line
#      merely erased.
# c) Implicit assumptions review:
#    - `_pick_main_dir(...)` assumes one direction wins the same-color adjacency count. The generator
#      places only one separated family of runs, so that remains true.
#    - `_extract_segments(...)` assumes each meaningful object is one contiguous run of length >= 2 on a
#      unique parallel track. The generator never overlaps runs and preserves the original gap rules.
#    - The combo loop assumes there is a unique current longest segment to remove and then a unique new
#      longest segment to use as reference. The generator enforces a strict top-two length ordering:
#      one longest input segment, one second-longest input segment, all others strictly shorter.
#    - Clearing and redrawing only detected segment cells still leaves unrelated isolated pixels intact.
#      The generator does not add such pixels, but the solver stays compatible with them.
# d) Placement legality:
#    - Construction is output-first over the surviving four segments. Their final common extent is placed
#      fully in bounds before any input variation is added.
#    - The deleted longest segment is then added on its own offset with the same anchored endpoint and a
#      strictly larger length, so there are no overlaps and all cells stay inside the canvas.
# e) Visual observability:
#    - Every sample visibly loses exactly one whole line from input to output.
#    - Among the remaining lines, the output span matches the second-largest visible input length rather
#      than the largest one and rather than the median-offset line. That makes both fused components
#      observable from geometry alone.
# f) Visual cleanliness:
#    - Offsets remain irregular and well separated; diagonal gaps stay at least 4 to avoid accidental
#      touching. The board contains only one family of lines, so the deletion and re-alignment are easy
#      to parse.
# g) Color inference:
#    - Colors are purely identity markers. No ranking, deletion, or alignment decision depends on color
#      ids, so the rule is inferable without hidden color semantics.
# h) New solver assumptions and generator guarantees:
#    - `main()` now assumes the combo's convergence happens after removing a unique longest segment and
#      snapping the survivors to a unique second-longest extent; `generate_input()` guarantees that.
#    - `main()` assumes the deleted line is not needed later and that the target extent stays in bounds
#      for every survivor; output-first placement guarantees both conditions.

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


def _rebuild_segment(seg: dict, ref_t0: int, target_len: int, d: tuple[int, int], shape: tuple[int, int]) -> dict:
    h, w = shape
    dr, dc = d
    delta_t = _step(d)
    dt = ref_t0 - seg["t_start"]
    assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
    k = dt // delta_t
    rr = seg["start"][0] + k * dr
    cc = seg["start"][1] + k * dc
    cells = []
    for _ in range(target_len):
        assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
        cells.append((rr, cc))
        rr += dr
        cc += dc
    start = cells[0]
    end = cells[-1]
    rebuilt = dict(seg)
    rebuilt.update(
        {
            "cells": cells,
            "start": start,
            "end": end,
            "t_start": _dot(start, d),
            "t_end": _dot(end, d),
            "len": target_len,
            "s": _perp_dot(start, d),
        }
    )
    assert rebuilt["s"] == seg["s"], "parallel offset must stay unchanged after shifting"
    return rebuilt


def _snap_group_to_reference(active: list[dict], ref: dict, d: tuple[int, int], shape: tuple[int, int]) -> list[dict]:
    return [_rebuild_segment(seg, ref["t_start"], ref["len"], d, shape) for seg in active]


def _combo_transform(segments: list[dict], d: tuple[int, int], shape: tuple[int, int]) -> list[dict]:
    active = [dict(seg) for seg in segments]
    while len({seg["len"] for seg in active}) > 1:
        max_len = max(seg["len"] for seg in active)
        longest = [seg for seg in active if seg["len"] == max_len]
        assert len(longest) == 1, "current longest segment must be unique"
        removed = longest[0]
        active = [seg for seg in active if seg is not removed]
        assert active, "at least one segment must remain after deleting the longest"

        new_max = max(seg["len"] for seg in active)
        refs = [seg for seg in active if seg["len"] == new_max]
        assert len(refs) == 1, "post-removal reference segment must be the unique new longest"
        active = _snap_group_to_reference(active, refs[0], d, shape)
    return active


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the longest segment from one family of parallel colored lines, then normalize the survivors to the next reference span.

    Input structure:
    - The background is the most frequent color.
    - Foreground objects are five separated horizontal, vertical, or diagonal line segments on distinct
      parallel tracks, all sharing one anchored endpoint direction but having different lengths.

    Rule:
    - Detect the common direction and extract every multi-cell segment on that direction.
    - Repeatedly delete the unique current longest segment. After each deletion, pick the new unique
      longest survivor as the reference and shift every remaining segment so its start matches that
      survivor's start, then redraw all survivors with the reference length.
    - Stop when the remaining segments all have the same length. Deleted segments stay absent; grid
      size is unchanged and isolated non-segment pixels are untouched.

    Key insight:
    - The decisive step is not "align everything to the biggest line". The biggest line disappears
      first, so the final common span comes from the second tier: the segment that becomes longest only
      after the original maximum has been removed.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    delta_t = _step(d)

    segments = _extract_segments(g, bg, d)
    assert len(segments) >= 3, "at least three multi-cell segments must exist"
    s_vals = sorted(seg["s"] for seg in segments)
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4, "diagonal offsets must stay well separated"

    active = _combo_transform(segments, d, g.shape)

    out = g.copy()
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = bg
    for seg in active:
        for r, c in seg["cells"]:
            out[r, c] = seg["color"]
    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [5]
_REF_LEN_RANGE = (4, 8)
_MAX_SEG_LEN = 12
_MAX_DISTINCT_COLORS = 3

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500


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
        if any(cap > ref_len for cap in start_caps):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(cap > ref_len for cap in end_caps):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_removed_and_ref(caps: list[int], ref_len: int) -> tuple[int, int] | None:
    n = len(caps)
    median_idx = n // 2
    nonmedian = [i for i in range(n) if i != median_idx]
    removed_options = [i for i in nonmedian if caps[i] > ref_len]
    if not removed_options:
        return None
    removed_idx = random.choice(removed_options)
    ref_options = [i for i in nonmedian if i != removed_idx]
    if not ref_options:
        return None
    ref_idx = random.choice(ref_options)
    return removed_idx, ref_idx


def _choose_input_lengths(
    ref_len: int, caps: list[int], removed_idx: int, ref_idx: int
) -> list[int] | None:
    n = len(caps)
    lengths = [None] * n
    removed_hi = min(_MAX_SEG_LEN, caps[removed_idx])
    if removed_hi <= ref_len:
        return None
    lengths[removed_idx] = random.randint(ref_len + 1, removed_hi)
    lengths[ref_idx] = ref_len

    for i in range(n):
        if i in (removed_idx, ref_idx):
            continue
        hi = min(_MAX_SEG_LEN, ref_len - 1, caps[i])
        if hi < 2:
            return None
        lengths[i] = random.randint(2, hi)
    return lengths


def _paint(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
    indices: list[int] | None = None,
) -> bool:
    idxs = range(len(s_vals)) if indices is None else indices
    for i in idxs:
        cells = _segment_cells(d, starts[i], s_vals[i], lengths[i])
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = colors[i]
    return True


def _median_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = _step(d)
    segments = _extract_segments(g, bg, d)
    s_vals = sorted(seg["s"] for seg in segments)
    ref_s = s_vals[len(s_vals) // 2]
    ref = [seg for seg in segments if seg["s"] == ref_s][0]
    target_len = (ref["t_end"] - ref["t_start"]) // delta_t + 1

    out = g.copy()
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = bg
    for seg in segments:
        redrawn = _rebuild_segment(seg, ref["t_start"], target_len, d, g.shape)
        for r, c in redrawn["cells"]:
            out[r, c] = seg["color"]
    return out


def _longest_keep_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    segments = _extract_segments(g, bg, d)
    max_len = max(seg["len"] for seg in segments)
    ref = [seg for seg in segments if seg["len"] == max_len][0]

    out = g.copy()
    for seg in segments:
        for r, c in seg["cells"]:
            out[r, c] = bg
    for seg in segments:
        redrawn = _rebuild_segment(seg, ref["t_start"], ref["len"], d, g.shape)
        for r, c in redrawn["cells"]:
            out[r, c] = seg["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    choice = _choose_removed_and_ref(caps, ref_len)
    if choice is None:
        return None
    removed_idx, ref_idx = choice
    lengths = _choose_input_lengths(ref_len, caps, removed_idx, ref_idx)
    if lengths is None:
        return None
    colors = _choose_colors(n)

    survivors = [i for i in range(n) if i != removed_idx]
    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, colors, survivors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    try:
        solved = main(inp)
        if not np.array_equal(solved, out):
            return None
        if np.array_equal(_median_output(inp), out):
            return None
        if np.array_equal(_longest_keep_output(inp), out):
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
