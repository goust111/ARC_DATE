# Combo variant of e376de54 (index 15)
# combo_spec: D2/2+D9/1
# Name: 混合方向+分组处理

# Design analysis for the combo variant
# a) Concrete behavior:
#    The original solver assumes one global direction and one global reference span. Here a 16x16 grid
#    contains 2-3 independent families of parallel segments, and each family uses a different direction.
#    Example: one horizontal triplet may share a left endpoint, one vertical triplet may share a bottom
#    endpoint, and one diagonal triplet may share a start on a / diagonal. In the output, each family is
#    aligned only inside its own direction-specific cluster. The old logic would try to collapse all
#    colored pixels into one dominant direction; the combo logic must first separate visible families by
#    direction, then solve each family independently.
# b) Complexity calibration:
#    Each group has exactly three segments. That keeps the reference-by-median-offset rule unambiguous
#    while still making multi-group reasoning necessary. The puzzle uses 2 or 3 groups with distinct
#    directions, and their reference lengths are forced to differ, so the grouped mixed-direction nature
#    is obvious instead of looking like one accidental cluttered scene.
# c) Old implicit assumptions that break:
#    - `_pick_main_dir(...)` is invalid because there is no single global direction anymore.
#    - A single `segments` list cannot be assigned one shared `ref_t0/target_len`; every group has its
#      own direction, own aligned endpoint, and own reference span.
#    - Clearing and redrawing all segments at once would wrongly overwrite already-correct content from
#      other groups. The solver therefore clears and redraws one group at a time.
#    - Direction-specific coordinate assumptions still matter inside a group, so the generator must keep
#      each group's perpendicular offsets unique and diagonal gaps large enough to avoid ambiguity.
# d) Placement legality:
#    The generator builds each group output-first, derives its input by varying only that group's lengths,
#    and rejects any candidate whose input cells or output cells overlap or touch another group. Groups
#    also use distinct directions, stay fully in bounds, and preserve the "all share one start or one
#    end" condition needed by the grouping rule.
# e) Visual observability:
#    A human can see several separate alignment mini-problems because each direction family occupies its
#    own blank-space-separated region. The output keeps those regions separate while regularizing each
#    family's span. Different target lengths across groups make it clear that reference selection is
#    local rather than global.
# f) Visual cleanliness:
#    Only three segments appear per group, and groups are separated by at least one blank moat in both
#    input and output. This avoids crossings, touching endpoints, and accidental merged chains.
# g) Change inferability:
#    Colors remain identity markers only. The rule depends on visible geometry: direction class,
#    endpoint-sharing within a class, and the median-offset reference inside that class. No color-id
#    ordering or hidden numeric property is used.

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
        dir_segments = [seg for seg in segments if seg["dir"] == d]
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
            starts = {seg["t_start"] for seg in comp}
            ends = {seg["t_end"] for seg in comp}
            if len(comp) >= 3 and (len(starts) == 1 or len(ends) == 1):
                groups.append(comp)

    groups.sort(key=lambda grp: min(min(seg["cells"]) for seg in grp))
    return groups


def _segment_group_info(g: np.ndarray) -> list[dict]:
    bg = get_bg(g)
    segments = []
    for d in _DIR_CANDIDATES:
        segments.extend(_extract_segments_in_dir(g, bg, d))
    groups = _find_groups(segments)
    info = []
    for grp in groups:
        s_vals = sorted(seg["s"] for seg in grp)
        ref = [seg for seg in grp if seg["s"] == s_vals[len(s_vals) // 2]][0]
        info.append(
            {
                "dir": grp[0]["dir"],
                "size": len(grp),
                "t_starts": tuple(sorted({seg["t_start"] for seg in grp})),
                "t_ends": tuple(sorted({seg["t_end"] for seg in grp})),
                "ref_len": ref["len"],
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve several independent line-alignment puzzles that use different line directions in one grid.

    Input structure:
    - The background is the most common color.
    - The foreground contains 2-3 groups of multi-cell line segments, and each group is internally
      parallel but uses a different direction from the other groups.
    - Inside one group, the segments already share one endpoint coordinate along that direction, while
      their lengths vary and their perpendicular offsets are distinct.

    Rule:
    - Detect all horizontal, vertical, and diagonal segments of length at least 2.
    - Split them into independent groups by direction and by the visible endpoint-sharing relation:
      segments belong to one group only when they share the same start coordinate or the same end
      coordinate along their own direction.
    - For each group separately, order segments by perpendicular offset, use the median-offset segment
      as the reference, then shift every group member so its start matches the reference start and trim
      or extend it to the reference length.
    - Leave any pixels outside these grouped segments unchanged.

    Key insight:
    The puzzle is no longer "find one dominant direction". Each direction family is its own copy of the
    original task, so the solver must classify first and only then perform reference selection locally.
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
        assert len(group) % 2 == 1, "each group must have a unique median reference"
        assert len(s_vals) == len(set(s_vals)), "group perpendicular offsets must be unique"
        if delta_t == 2:
            for i in range(len(s_vals) - 1):
                assert s_vals[i + 1] - s_vals[i] >= 4, "diagonal offsets must be well separated"

        s_ref = s_vals[len(s_vals) // 2]
        ref_cands = [seg for seg in group if seg["s"] == s_ref]
        assert len(ref_cands) == 1, "median reference must be unique inside a group"
        ref = ref_cands[0]
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
                assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
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
_GROUP_COUNTS = [2, 3]
_SEGMENT_COUNT = 3
_REF_LEN_RANGES = {2: (3, 6), 3: (3, 5)}
_MAX_SEG_LEN = 9

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 900
_GROUP_RETRIES = 220

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
        patterns = [p + [gap] for p in patterns for gap in allowed]
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
        if all(start_caps[i] >= 2 for i in range(len(s_vals))) and any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if all(end_caps[i] >= 2 for i in range(len(s_vals))) and any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
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


def _choose_lengths(ref_len: int, caps: list[int]) -> list[int] | None:
    ref_idx = _SEGMENT_COUNT // 2
    nonrefs = [i for i in range(_SEGMENT_COUNT) if i != ref_idx]
    short_max = min(ref_len - 1, _MAX_SEG_LEN)
    if short_max < 2:
        return None
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None
    long_i = random.choice(longable)
    short_i = nonrefs[0] if nonrefs[1] == long_i else nonrefs[1]
    lengths = [ref_len] * _SEGMENT_COUNT
    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))
    lengths[short_i] = random.randint(2, short_max)
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
            return False
    return True


def _build_group(d: tuple[int, int], ref_len: int, groups: list[dict]) -> dict | None:
    for _ in range(_GROUP_RETRIES):
        s_vals = _choose_offsets(d, _SEGMENT_COUNT)
        placements = _feasible_placements(d, s_vals, ref_len)
        if not placements:
            continue
        colors = _choose_colors(_SEGMENT_COUNT)
        for t0, anchor, caps in placements:
            lengths = _choose_lengths(ref_len, caps)
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
                    {
                        "cells": inp_cells,
                        "color": color,
                        "t_start": start,
                        "t_end": start + (length - 1) * delta,
                        "len": length,
                    }
                )
                output_segments.append(
                    {
                        "cells": out_cells,
                        "color": color,
                        "t_start": t0,
                        "t_end": t1,
                        "len": ref_len,
                    }
                )
            if not ok:
                continue

            group = {
                "dir": d,
                "anchor": anchor,
                "ref_len": ref_len,
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


def _build_ref_lengths(group_count: int) -> list[int]:
    lo, hi = _REF_LEN_RANGES[group_count]
    choices = list(range(lo, hi + 1))
    return random.sample(choices, group_count)


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice(_GROUP_COUNTS)
    dirs = random.sample(_DIRS, group_count)
    ref_lengths = _build_ref_lengths(group_count)
    random.shuffle(ref_lengths)

    groups = []
    for d, ref_len in zip(dirs, ref_lengths):
        group = _build_group(d, ref_len, groups)
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

    metrics = _segment_group_info(inp)
    if len(metrics) != group_count:
        return None
    if len({m["dir"] for m in metrics}) != group_count:
        return None
    if len({m["ref_len"] for m in metrics}) != group_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid mixed-direction grouped input")
