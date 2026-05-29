# Variant of e376de54
# Dimension: D9 — 多步处理：参考线段选取与线段变换的顺序 (multi_step)
# Change: 步骤1：识别所有平行线段及其颜色；步骤2：计算每条线段长度；步骤3：找到第二长线段为参考；步骤4：将所有线段端点对齐到参... → 分组处理：按方向或位置分组独立对齐

# Design analysis for the grouped variant
# a) Concrete behavior:
#    The original puzzle has one family of parallel segments. This variant places two independent
#    subproblems in one 16x16 grid. Each subproblem is still a small set of parallel segments with
#    one endpoint already aligned in the input and full-range alignment in the output. The new visible
#    behavior is that alignment is local to each cluster: one cluster does not borrow the reference
#    extent of the other cluster, even when both clusters use the same direction.
#    Example: two horizontal triplets can sit in different rows and with different shared starts/ends.
#    The old logic would merge all six lines into one global family; the new logic splits them by the
#    endpoint-alignment pattern seen inside each local cluster, then aligns each cluster separately.
# b) Complexity calibration:
#    Exactly two groups are generated. This is enough to make the grouped behavior obvious, but avoids
#    the clutter of three or more local problems in a 16x16 grid. Each group keeps three segments, so
#    the median-by-offset reference rule from the original code remains unambiguous inside each group.
#    Some samples use two different directions; others use the same direction but spatially separated.
# c) Old implicit assumptions that break:
#    - `_pick_main_dir(...)` assumed one dominant direction globally. That fails when the grid contains
#      two different direction families or two separated families that should not share one reference.
#    - The original `segments` list was treated as one global set. Its median `s` was meaningful only
#      because every segment belonged to the same alignment problem.
#    - Clearing and redrawing every segment against one `ref_t0/target_len` is only harmless when all
#      segments should end up on the same span. In this variant that would destroy already-correct data
#      from the other group.
# d) Placement legality:
#    The generator constructs each group output-first and rejects any candidate whose input/output cells
#    overlap or even touch another group's cells. For same-direction groups it also enforces disjoint
#    `t_start` and `t_end` sets, so the solver's grouping-by-shared-endpoint rule cannot accidentally
#    merge the two groups.
# e) Visual observability:
#    Group membership is visible from geometry alone. In every sample there are two separate sets of
#    parallel lines, each with its own locally aligned endpoint. The output shows two independent
#    alignment ranges, so a human can see that "one global reference for all lines" is no longer valid.
# f) Visual cleanliness:
#    Groups are separated by blank space in both input and output. No overlaps, crossings, or touching
#    endpoints are allowed between groups, which keeps the local subproblems easy to parse.
# g) Color inferability:
#    Colors stay as identity markers only. No sorting or transformation depends on color ids; the rule
#    is driven entirely by visible direction, offset order, and shared endpoints inside each group.

import random

import numpy as np

from arc_utils import get_bg

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _extract_segments_in_dir(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
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
                        "dir": d,
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


def _find_groups(segments: list[dict]) -> list[list[dict]]:
    groups = []
    for d in _DIR_CANDIDATES:
        dir_segments = [s for s in segments if s["dir"] == d]
        if not dir_segments:
            continue

        parent = list(range(len(dir_segments)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(len(dir_segments)):
            for j in range(i + 1, len(dir_segments)):
                if (
                    dir_segments[i]["t_start"] == dir_segments[j]["t_start"]
                    or dir_segments[i]["t_end"] == dir_segments[j]["t_end"]
                ):
                    union(i, j)

        comps: dict[int, list[dict]] = {}
        for i, seg in enumerate(dir_segments):
            comps.setdefault(find(i), []).append(seg)

        for comp in comps.values():
            if len(comp) >= 3:
                groups.append(comp)

    groups.sort(key=lambda grp: min(min(seg["cells"]) for seg in grp))
    return groups


def _group_metrics(g: np.ndarray) -> list[dict]:
    bg = get_bg(g)
    segments = []
    for d in _DIR_CANDIDATES:
        segments.extend(_extract_segments_in_dir(g, bg, d))
    groups = _find_groups(segments)
    info = []
    for grp in groups:
        info.append(
            {
                "dir": grp[0]["dir"],
                "size": len(grp),
                "t_starts": tuple(sorted({seg["t_start"] for seg in grp})),
                "t_ends": tuple(sorted({seg["t_end"] for seg in grp})),
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve several independent line-alignment subproblems that coexist in one grid.

    Input structure:
    - bg is the most common color.
    - The grid contains two or more local groups of multi-cell parallel line segments.
    - Segments inside one group share a direction and already share one endpoint coordinate, but
      different groups may use a different direction or a different aligned endpoint range.

    Rule:
    - Detect every horizontal, vertical, and diagonal segment of length at least 2.
    - Split them into independent groups by direction and by the endpoint-alignment relation seen in
      the input: segments belong together only if they share a start coordinate or share an end coordinate.
    - Inside each group, keep the original reference rule: order the parallel lines by perpendicular
      offset, use the median line as the reference, then shift every segment so its start matches the
      reference start and trim/extend it to the reference length.
    - Leave any pixels outside those grouped segments unchanged.

    Key insight:
    The alignment reference is no longer global. Each visible cluster is its own miniature copy of the
    original task, so reference selection and endpoint snapping must be done group-by-group.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    segments = []
    for d in _DIR_CANDIDATES:
        segments.extend(_extract_segments_in_dir(g, bg, d))

    groups = _find_groups(segments)
    assert groups, "at least one valid segment group must exist"

    out = g.copy()
    for group in groups:
        d = group[0]["dir"]
        dr, dc = d
        delta_t = _step(d)

        s_vals = sorted(seg["s"] for seg in group)
        assert len(group) % 2 == 1, "each group must have a single median reference"
        assert len(s_vals) == len(set(s_vals)), "perpendicular offsets inside a group must be unique"
        if delta_t == 2:
            for i in range(len(s_vals) - 1):
                assert s_vals[i + 1] - s_vals[i] >= 4

        s_ref = s_vals[len(s_vals) // 2]
        ref = [seg for seg in group if seg["s"] == s_ref][0]
        ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
        assert ref_t1 >= ref_t0
        assert (ref_t1 - ref_t0) % delta_t == 0
        target_len = (ref_t1 - ref_t0) // delta_t + 1

        for seg in group:
            for r, c in seg["cells"]:
                out[r, c] = bg

        for seg in group:
            dt = ref_t0 - seg["t_start"]
            assert dt % delta_t == 0
            k = dt // delta_t
            rr = seg["start"][0] + k * dr
            cc = seg["start"][1] + k * dc
            for _ in range(target_len):
                assert 0 <= rr < h and 0 <= cc < w, "shifted group segment must stay in bounds"
                out[rr, cc] = seg["color"]
                rr += dr
                cc += dc

    return out


# --------------- generator (o5x) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

# --- Group shape ---
_GROUP_COUNT = 2
_SEGMENT_COUNT = 3
_REF_LEN_RANGE = (3, 6)
_MAX_SEG_LEN = 10

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 500
_GROUP_RETRIES = 160

# --- Separation ---
_MIN_CHEBYSHEV_GAP = 2


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
    random.shuffle(placements)
    return placements


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
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
    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))
    lengths[short_i] = random.randint(2, ref_len - 1)
    return lengths


def _chebyshev_dist(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _cells_well_separated(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> bool:
    for p in cells_a:
        for q in cells_b:
            if _chebyshev_dist(p, q) < _MIN_CHEBYSHEV_GAP:
                return False
    return True


def _group_ok(group: dict, groups: list[dict]) -> bool:
    for other in groups:
        if not _cells_well_separated(group["input_cells"], other["input_cells"]):
            return False
        if not _cells_well_separated(group["output_cells"], other["output_cells"]):
            return False
        if group["dir"] == other["dir"]:
            if set(group["t_starts"]) & set(other["t_starts"]):
                return False
            if set(group["t_ends"]) & set(other["t_ends"]):
                return False
    return True


def _build_group(d: tuple[int, int], groups: list[dict]) -> dict | None:
    for _ in range(_GROUP_RETRIES):
        ref_len = random.randint(*_REF_LEN_RANGE)
        s_vals = _choose_offsets(d, _SEGMENT_COUNT)
        placements = _feasible_placements(d, s_vals, ref_len)
        if not placements:
            continue
        colors = _choose_colors(_SEGMENT_COUNT)
        for t0, anchor, caps in placements:
            lengths = _choose_lengths(ref_len, caps, _SEGMENT_COUNT)
            if lengths is None:
                continue

            delta = _step(d)
            t1 = t0 + (ref_len - 1) * delta
            if anchor == "start":
                starts = [t0] * _SEGMENT_COUNT
            else:
                starts = [t1 - (length - 1) * delta for length in lengths]

            input_segments = []
            output_segments = []
            ok = True
            for s, start, length, color in zip(s_vals, starts, lengths, colors):
                inp_cells = _segment_cells(d, start, s, length)
                out_cells = _segment_cells(d, t0, s, ref_len)
                if inp_cells is None or out_cells is None:
                    ok = False
                    break
                input_segments.append(
                    {"cells": inp_cells, "color": color, "t_start": start, "t_end": start + (length - 1) * delta}
                )
                output_segments.append({"cells": out_cells, "color": color})
            if not ok:
                continue

            group = {
                "dir": d,
                "anchor": anchor,
                "input_segments": input_segments,
                "output_segments": output_segments,
                "input_cells": {cell for seg in input_segments for cell in seg["cells"]},
                "output_cells": {cell for seg in output_segments for cell in seg["cells"]},
                "t_starts": tuple(seg["t_start"] for seg in input_segments),
                "t_ends": tuple(seg["t_end"] for seg in input_segments),
            }
            if _group_ok(group, groups):
                return group
    return None


def _paint_group(grid: np.ndarray, group: dict, key: str) -> None:
    for seg in group[key]:
        for r, c in seg["cells"]:
            grid[r, c] = seg["color"]


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["same_dir", "diff_dir"])
    if mode == "same_dir":
        d = random.choice(_DIRS)
        dirs = [d, d]
    else:
        d0, d1 = random.sample(_DIRS, 2)
        dirs = [d0, d1]

    groups = []
    for d in dirs:
        group = _build_group(d, groups)
        if group is None:
            return None
        groups.append(group)

    inp = np.full((_H, _W), _BG, dtype=int)
    out = np.full((_H, _W), _BG, dtype=int)
    for group in groups:
        _paint_group(inp, group, "input_segments")
        _paint_group(out, group, "output_segments")

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None

    metrics = _group_metrics(inp)
    if len(metrics) != _GROUP_COUNT:
        return None
    if mode == "same_dir" and len({m["dir"] for m in metrics}) != 1:
        return None
    if mode == "diff_dir" and len({m["dir"] for m in metrics}) != 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid grouped input")
