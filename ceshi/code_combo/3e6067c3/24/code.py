# Combo variant of 3e6067c3 (index 24)
# combo_spec: D1/2+D4/0+D5/2
# Name: 尺寸匹配+不等大方块+单像素桥梁
#
# Design analysis
# a) Concrete behavior and trace
#    Original task:
#    - framed squares have equal geometry and are identified by inner color;
#    - control row contains isolated pixels whose colors name the next square;
#    - bridges fill the whole shared inner band between adjacent aligned squares.
#    Combined variant:
#    - every frame shares the same inner fill color, so color no longer identifies a target;
#    - the measurable key is the inner opening side length (1/3/5), encoded by the control
#      row as monochrome horizontal bars of length 1/3/5;
#    - frames no longer have equal geometry: both outer size and border thickness may vary,
#      but lattice rows/columns still share centerlines;
#    - each bridge is a single-pixel line along the common center row/column of the pair.
#    Example: if the current control bar has length 5, the solver must find the frame whose
#    visible opening is 5 cells wide. If the next bar has length 3 and that frame lies on the
#    same lattice row, the output draws only the center row of the corridor, not the whole
#    3-row or 5-row band that the original task or a pure D4 variant would draw.
# b) Complexity calibration
#    The scene uses only three size classes (1/3/5) so the coding is learnable, but samples
#    enforce repeated sizes so matching is not trivial one-to-one lookup. Variable thicknesses
#    and outer sizes make the geometry visibly different from the source task. The generator
#    also requires at least one edge whose smaller endpoint has inner size >=3, so the thin-line
#    bridge differs visibly from the thick bridge that older variants would have drawn.
# c) Hidden-assumption audit of the original main()
#    - Grouping by `inner` color is invalid because all inner fills are identical.
#      The solver groups by measured `inner_size` instead.
#    - Reading the control row as isolated pixels is invalid because control items are now runs.
#      The solver parses monochrome segments and uses their lengths as the match key.
#    - Equality of inner bboxes no longer characterizes alignment once frames have variable size.
#      The solver switches to shared doubled center coordinates (`center2`) for row/column
#      alignment, which the generator preserves across each lattice row/column.
#    - Drawing the full source inner band is no longer correct. Under D5/2 the bridge is a
#      one-pixel centerline, so only the central row or column of the corridor is painted.
#    - A new implicit assumption appears: the intended bridge centerline must be visually
#      well-defined. The generator therefore only uses odd inner sizes (1/3/5), so every
#      opening has a unique center row/column.
# d) Placement legality
#    Frames are placed on a 2x3 lattice using per-row/per-column maximum radii, which guarantees
#    non-overlap even when outer sizes differ. Consecutive path frames are adjacent in lattice
#    coordinates, centerline-aligned, and separated by at least two background cells, so bridges
#    are non-empty. All bridge cells are checked for overlap and bounds before a sample is kept.
# e) Visual observability
#    A human can see three coupled facts from a few I/O pairs:
#    - all square interiors have the same color, so the control bars cannot be naming colors;
#    - bar lengths match visible opening sizes of target frames;
#    - completed bridges are thin centerlines even when the openings themselves are wide.
#    The combination is therefore readable from examples without relying on hidden numeric
#    properties of colors.
# f) Visual cleanliness
#    Frames stay on a coarse lattice, control bars sit on a dedicated bottom row, and the input
#    contains no bridges. In the output, bridge lines occupy only empty corridors, avoiding the
#    heavy overpainting that a thick-bridge version would create.
# g) Color inference
#    Colors carry only visible roles: background, border, common inner fill, and per-step bridge
#    paint. No numeric color ordering or invisible tie-break is used. Matching is purely by
#    geometric size, while bridge color comes from the current control bar color.
# 9) New solver assumptions and generator guarantees
#    - Assumption: control items are monochrome runs separated by background.
#      Guarantee: generator writes one bar per step with a one-cell gap.
#    - Assumption: all relevant frames are square border components enclosing one solid inner area.
#      Guarantee: generator constructs exactly that pattern.
#    - Assumption: frames in the same lattice row/column share a centerline.
#      Guarantee: generator places them from shared row/column center arrays.
#    - Assumption: repeated size classes are still disambiguated by align-first/nearest scoring.
#      Guarantee: generator forbids repeated sizes within the same lattice row/column and validates
#      every sample with `main(inp) == out`.
#    - Assumption: at least one sample feature exercises each fused dimension.
#      Guarantee: generator enforces repeated sizes, multiple thicknesses, variable-size path
#      edges, and at least one edge with min(inner_size)>=3.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask


SIZE_VALUES = (1, 3, 5)
THICKNESS_VALUES = (1, 2)
_DIGITS = [int(i) for i in range(10)]


def _parse_control_runs(row: np.ndarray, bg: int) -> list[dict]:
    runs: list[dict] = []
    c = 0
    W = int(row.shape[0])
    while c < W:
        if int(row[int(c)]) == int(bg):
            c += 1
            continue
        color = int(row[int(c)])
        c0 = int(c)
        while c < W and int(row[int(c)]) == int(color):
            c += 1
        runs.append({"color": int(color), "size": int(c - c0)})
    return runs


def _most_common_nonbg(g: np.ndarray, bg: int) -> int | None:
    cnt = Counter(map(int, g.flatten()))
    for col, _k in cnt.most_common():
        if int(col) != int(bg):
            return int(col)
    return None


def _detect_frames(g: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = int(get_bg(g))
    border = _most_common_nonbg(g, bg)
    assert border is not None

    lab, n = label_mask(g == int(border), connectivity=4)
    frames: list[dict] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if int(pts.shape[0]) < 8:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3 or h != w:
            continue

        t = None
        inner_col = None
        inner_bbox = None
        for tt in range(1, int(min(h, w) // 2) + 1):
            inner = g[int(r0 + tt) : int(r1 - tt) + 1, int(c0 + tt) : int(c1 - tt) + 1]
            if inner.size == 0:
                continue
            if bool((inner == int(border)).any()):
                continue
            u = set(map(int, inner.flatten()))
            if len(u) != 1:
                continue

            ok = True
            if not bool((g[int(r0) : int(r0 + tt), int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r1 - tt) + 1 : int(r1) + 1, int(c0) : int(c1) + 1] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c0) : int(c0 + tt)] == int(border)).all()):
                ok = False
            if not bool((g[int(r0) : int(r1) + 1, int(c1 - tt) + 1 : int(c1) + 1] == int(border)).all()):
                ok = False
            if not ok:
                continue

            t = int(tt)
            inner_col = int(next(iter(u)))
            inner_bbox = (int(r0 + tt), int(c0 + tt), int(r1 - tt), int(c1 - tt))
            break

        if t is None or inner_col is None or inner_bbox is None:
            continue

        inner_h = int(inner_bbox[2] - inner_bbox[0] + 1)
        inner_w = int(inner_bbox[3] - inner_bbox[1] + 1)
        if inner_h != inner_w:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "t": int(t),
                "outer": int(h),
                "inner": int(inner_col),
                "inner_bbox": inner_bbox,
                "inner_size": int(inner_h),
                "center2": (int(r0 + r1), int(c0 + c1)),
            }
        )

    assert frames
    return int(bg), int(border), frames


def _frames_by_size(frames: list[dict]) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner_size"])].append(f)
    return by


def _alignment_kind(f1: dict, f2: dict) -> str | None:
    r1, c1 = map(int, f1["center2"])
    r2, c2 = map(int, f2["center2"])
    if int(r1) == int(r2) and int(c1) != int(c2):
        return "row"
    if int(c1) == int(c2) and int(r1) != int(r2):
        return "col"
    return None


def _frame_center(f: dict) -> tuple[int, int]:
    r2, c2 = map(int, f["center2"])
    return int(r2 // 2), int(c2 // 2)


def _center_bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    kind = _alignment_kind(fa, fb)
    if kind is None:
        return set()

    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    ia = fa["inner_bbox"]
    ib = fb["inner_bbox"]

    if kind == "row":
        row_a = int((int(ia[0]) + int(ia[2])) // 2)
        row_b = int((int(ib[0]) + int(ib[2])) // 2)
        if int(row_a) != int(row_b):
            return set()
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return set()
        return {(int(row_a), int(c)) for c in range(int(cc0), int(cc1) + 1)}

    col_a = int((int(ia[1]) + int(ia[3])) // 2)
    col_b = int((int(ib[1]) + int(ib[3])) // 2)
    if int(col_a) != int(col_b):
        return set()
    if int(r1a) < int(r0b):
        rr0, rr1 = int(r1a + 1), int(r0b - 1)
    else:
        rr0, rr1 = int(r1b + 1), int(r0a - 1)
    if int(rr0) > int(rr1):
        return set()
    return {(int(r), int(col_a)) for r in range(int(rr0), int(rr1) + 1)}


def _candidate_score(f1: dict, f2: dict) -> tuple[int, int]:
    al = 1 if _center_bridge_cells(f1, f2) else 0
    r1, c1 = _frame_center(f1)
    r2, c2 = _frame_center(f2)
    d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return int(al), -int(d)


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _trace_path(frames: list[dict], ctrl: list[dict]) -> list[dict]:
    by_size = _frames_by_size(frames)
    start_size = int(ctrl[0]["size"])
    assert start_size in by_size
    start_cands = by_size[start_size]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_size = int(ctrl[1]["size"])
        nxt_cands = by_size.get(nxt_size, [])
        best = None
        bestf = None
        for f in start_cands:
            for g2 in (nxt_cands if nxt_cands else [f]):
                cand = _candidate_score(f, g2)
                if best is None or cand > best:
                    best = cand
                    bestf = f
        assert bestf is not None
        cur = bestf

    traced = [cur]
    for item in ctrl[1:]:
        cands = by_size.get(int(item["size"]), [])
        assert cands
        best = None
        bestf = None
        for f in cands:
            cand = _candidate_score(cur, f)
            if best is None or cand > best:
                best = cand
                bestf = f
        assert bestf is not None
        traced.append(bestf)
        cur = bestf
    return traced


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, _border, frames = _detect_frames(g)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = _parse_control_runs(g[int(ctrl_r)], int(bg))
    assert len(ctrl) >= 2

    path_frames = _trace_path(frames, ctrl)
    out = g.copy()

    for i in range(int(len(ctrl)) - 1):
        cur = path_frames[int(i)]
        nxt = path_frames[int(i + 1)]
        assert int(cur["inner_size"]) == int(ctrl[int(i)]["size"])
        for r, c in _center_bridge_cells(cur, nxt):
            if 0 <= int(r) < H and 0 <= int(c) < W:
                out[int(r), int(c)] = int(ctrl[int(i)]["color"])

    return out


def _scene_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg, _border, frames = _detect_frames(g)
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(g.shape[0] - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = _parse_control_runs(g[int(ctrl_r)], int(bg))
    assert len(ctrl) >= 2

    path_frames = _trace_path(frames, ctrl)
    sizes = [int(f["inner_size"]) for f in frames]
    counts = Counter(sizes)
    variable_edges = 0
    thick_line_edges = 0
    for i in range(int(len(path_frames)) - 1):
        sa = int(path_frames[int(i)]["inner_size"])
        sb = int(path_frames[int(i + 1)]["inner_size"])
        if int(sa) != int(sb):
            variable_edges += 1
        if min(int(sa), int(sb)) >= 3:
            thick_line_edges += 1

    return {
        "frames": int(len(frames)),
        "distinct_inner_sizes": int(len(set(sizes))),
        "repeated_size_max": int(max(counts.values())),
        "distinct_thicknesses": int(len({int(f["t"]) for f in frames})),
        "distinct_outer_sizes": int(len({int(f["outer"]) for f in frames})),
        "variable_edges": int(variable_edges),
        "thick_line_edges": int(thick_line_edges),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_choices = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_choices) < 2:
        return None
    inner_col = int(np.random.choice(np.array(inner_choices, dtype=int)))
    ctrl_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border), int(inner_col))]
    if len(ctrl_palette) < 5:
        return None

    R, C = 2, 3
    n_frames = int(np.random.randint(4, 6))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(400):
        p = [(0, 0)]
        vis = {(0, 0)}
        ok = True
        for _ in range(int(n_frames) - 1):
            r, c = map(int, p[-1])
            opts = [
                (int(rr), int(cc))
                for rr, cc in neighbors4(int(r), int(c))
                if in_bounds((int(R), int(C)), int(rr), int(cc)) and (int(rr), int(cc)) not in vis
            ]
            if not opts:
                ok = False
                break
            nxt = opts[int(np.random.randint(0, len(opts)))]
            p.append((int(nxt[0]), int(nxt[1])))
            vis.add((int(nxt[0]), int(nxt[1])))
        if not ok:
            continue
        if len({r for r, _c in p}) < 2 or len({c for _r, c in p}) < 2:
            continue
        path = p
        break
    if path is None:
        return None

    size_by_slot: dict[tuple[int, int], int] = {}
    t_by_slot: dict[tuple[int, int], int] = {}
    for _geom_try in range(500):
        size_by_slot = {}
        t_by_slot = {}
        used_in_row: dict[int, set[int]] = defaultdict(set)
        used_in_col: dict[int, set[int]] = defaultdict(set)
        prev_size = None
        ok = True
        for rr, cc in path:
            rr = int(rr)
            cc = int(cc)
            choices = [
                int(sz)
                for sz in SIZE_VALUES
                if int(sz) not in used_in_row[int(rr)]
                and int(sz) not in used_in_col[int(cc)]
                and int(sz) != int(prev_size if prev_size is not None else -1)
            ]
            if not choices:
                ok = False
                break
            size = int(np.random.choice(np.array(choices, dtype=int)))
            t = int(np.random.choice(np.array(THICKNESS_VALUES, dtype=int)))
            size_by_slot[(int(rr), int(cc))] = int(size)
            t_by_slot[(int(rr), int(cc))] = int(t)
            used_in_row[int(rr)].add(int(size))
            used_in_col[int(cc)].add(int(size))
            prev_size = int(size)
        if not ok:
            continue

        counts = Counter(size_by_slot.values())
        variable_edges = sum(
            int(size_by_slot[(int(a[0]), int(a[1]))]) != int(size_by_slot[(int(b[0]), int(b[1]))])
            for a, b in zip(path, path[1:])
        )
        thick_line_edges = sum(
            min(
                int(size_by_slot[(int(a[0]), int(a[1]))]),
                int(size_by_slot[(int(b[0]), int(b[1]))]),
            )
            >= 3
            for a, b in zip(path, path[1:])
        )
        if len(counts) < 2:
            continue
        if max(int(v) for v in counts.values()) < 2:
            continue
        if len(set(t_by_slot.values())) < 2:
            continue
        if len({int(size_by_slot[k] + 2 * t_by_slot[k]) for k in size_by_slot}) < 2:
            continue
        if int(variable_edges) < 1:
            continue
        if int(thick_line_edges) < 1:
            continue
        break
    else:
        return None

    row_radius: dict[int, int] = defaultdict(int)
    col_radius: dict[int, int] = defaultdict(int)
    for slot in path:
        rr, cc = map(int, slot)
        outer = int(size_by_slot[(int(rr), int(cc))] + 2 * t_by_slot[(int(rr), int(cc))])
        rad = int(outer // 2)
        row_radius[int(rr)] = max(int(row_radius[int(rr)]), int(rad))
        col_radius[int(cc)] = max(int(col_radius[int(cc)]), int(rad))

    top_margin = int(np.random.randint(1, 3))
    left_margin = int(np.random.randint(1, 3))
    row_gap = [int(np.random.randint(2, 5)) for _ in range(int(R) - 1)]
    col_gap = [int(np.random.randint(2, 5)) for _ in range(int(C) - 1)]

    row_centers = [int(top_margin + row_radius[0])]
    for rr in range(1, int(R)):
        row_centers.append(int(row_centers[-1] + row_radius[int(rr - 1)] + row_radius[int(rr)] + row_gap[int(rr - 1)]))
    col_centers = [int(left_margin + col_radius[0])]
    for cc in range(1, int(C)):
        col_centers.append(int(col_centers[-1] + col_radius[int(cc - 1)] + col_radius[int(cc)] + col_gap[int(cc - 1)]))

    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    for rr, cc in path:
        rr = int(rr)
        cc = int(cc)
        inner_size = int(size_by_slot[(int(rr), int(cc))])
        t = int(t_by_slot[(int(rr), int(cc))])
        outer = int(inner_size + 2 * t)
        rad = int(outer // 2)
        r_ctr = int(row_centers[int(rr)])
        c_ctr = int(col_centers[int(cc)])
        r0 = int(r_ctr - rad)
        c0 = int(c_ctr - rad)
        r1 = int(r0 + outer - 1)
        c1 = int(c0 + outer - 1)
        f = {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "inner_bbox": (int(r0 + t), int(c0 + t), int(r1 - t), int(c1 - t)),
            "slot": (int(rr), int(cc)),
            "t": int(t),
            "outer": int(outer),
            "inner_size": int(inner_size),
            "center2": (int(r0 + r1), int(c0 + c1)),
        }
        frames.append(f)
        slot_to_frame[(int(rr), int(cc))] = f

    max_r1 = max(int(f["bbox"][2]) for f in frames)
    max_c1 = max(int(f["bbox"][3]) for f in frames)

    ctrl_colors = [int(x) for x in np.random.choice(np.array(ctrl_palette, dtype=int), size=int(n_frames), replace=False)]
    ctrl_sizes = [int(size_by_slot[(int(r), int(c))]) for r, c in path]
    ctrl_spans: list[tuple[int, int, int]] = []
    cur_c = 1
    for color, size in zip(ctrl_colors, ctrl_sizes):
        ctrl_spans.append((int(cur_c), int(cur_c + size - 1), int(color)))
        cur_c += int(size + 1)
    ctrl_end = int(ctrl_spans[-1][1])

    ctrl_gap = int(np.random.randint(2, 5))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 3))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max(int(max_c1 + 1 + right_margin), int(ctrl_end + 1 + right_margin)))
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(f["outer"]),
            t=int(f["t"]),
            border=int(border),
            inner_col=int(inner_col),
        )
        f["inner"] = int(inner_col)

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for i, (a_slot, b_slot) in enumerate(zip(path, path[1:])):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _center_bridge_cells(fa, fb)
        if not cells:
            ok = False
            break
        if cells & bridge_cells_all:
            ok = False
            break
        for r, c in cells:
            if not (0 <= int(r) < int(H) and 0 <= int(c) < int(W)):
                ok = False
                break
            if int(out[int(r), int(c)]) != int(bg):
                ok = False
                break
        if not ok:
            break
        paint = int(ctrl_colors[int(i)])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    for c0, c1, color in ctrl_spans:
        out[int(ctrl_r), int(c0) : int(c1) + 1] = int(color)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None

    try:
        metrics = _scene_metrics(inp)
    except Exception:
        return None
    if int(metrics["distinct_inner_sizes"]) < 2:
        return None
    if int(metrics["repeated_size_max"]) < 2:
        return None
    if int(metrics["distinct_thicknesses"]) < 2:
        return None
    if int(metrics["distinct_outer_sizes"]) < 2:
        return None
    if int(metrics["variable_edges"]) < 1:
        return None
    if int(metrics["thick_line_edges"]) < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
