# Variant of 3e6067c3
# Dimension: D4 — 目标对象：带框方块的结构 (dst)
# Change: 等尺寸的带框方块（外层边框色 + 内层纯色填充），排列在网格状的格点上... → 不等尺寸方块

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import get_bg, in_bounds, neighbors4, label_mask

"""
Design analysis for the D4 variant
==================================

a) Concrete behavior and trace
The original task uses equal-sized framed squares, so two horizontally aligned
targets have identical inner-row spans and the bridge simply fills that full
band across the gap. In this variant, each framed square is still a square with
a uniform border and a solid-color inner square, but different squares may have
different outer sizes and different border thicknesses. The lattice structure
is preserved through shared row/column centerlines rather than shared identical
inner bboxes.

Concrete example:
- Old: square A has inner rows 10..12, square B also has inner rows 10..12, so
  the bridge uses rows 10..12.
- New: A may have inner rows 10..10 while B has inner rows 9..11. They are
  still on the same lattice row because their centers match, but the bridge
  should only use the smaller inner span, here row 10..10.
- Therefore the equal-bbox test from the original logic is no longer correct;
  centerline alignment and smaller-inner-span drawing are the visible D4 change.

b) Complexity calibration
The variant must be visibly different from the original task, so the generator
forces geometry diversity:
- at least two inner sizes across the scene,
- at least two border thickness values across the scene,
- at least one consecutive path edge whose endpoints have different inner sizes.
This keeps the new bridge-width rule active in most samples while preserving a
small discrete set of sizes so the scene stays readable.

c) Hidden assumptions in the original solver that no longer hold
Relevant old assumptions:
- `aligned()` assumed equal-sized squares by checking identical `inner_bbox`
  row/col spans. This breaks immediately once inner sizes differ.
- `draw_bridge()` copied the current frame's full inner span. That was harmless
  only because every aligned square had the same span. With variable sizes that
  would over-thicken bridges and no longer match the intended smaller endpoint.
- The frame detector itself is mostly reusable: it already searches each border
  component for a uniform-thickness border enclosing one solid-color inner area.
New solver assumptions introduced here:
- Target objects are still squares, not arbitrary rectangles.
- Frames that belong to the same lattice row/column share the same doubled
  center coordinate (`r0+r1` or `c0+c1`), even if their sizes differ.
- The bridge cross-section is exactly the inner span of the smaller endpoint in
  the orthogonal direction. The generator enforces these conditions.

d) Placement legality
The generator assigns each path square an inner size and border thickness, then
places all squares on lattice row/column centers. Row/column center spacing is
computed from the maximum half-size used in each row/column plus an extra gap of
at least 2 cells, guaranteeing:
- no frame overlap,
- at least one background cell between neighboring frames,
- non-empty bridge rectangles,
- all bridge cells remain background in the input.

e) Visual observability
Humanly visible cues:
- framed squares are plainly different sizes within the same grid,
- some bridges are thin because they match the smaller endpoint,
- other bridges are thicker when both endpoints are large.
The rule is still recoverable from 2-4 examples because the control row and the
path order are unchanged; only bridge thickness depends on visible geometry.

f) Visual cleanliness
The generator keeps a coarse lattice layout, uses only one border color, and
prevents bridge overlap. Control pixels remain isolated on a separate bottom row.
That preserves the original scene grammar while making the new geometry readable.

g) Color inferability
Colors are not repurposed for hidden metadata. Inner colors still identify the
visit sequence items, and bridge colors still come from the source square color.
The only new variable is object geometry, which is directly visible.

Post-change solver/generator contract
The solver now assumes square frames with matched lattice centers; the generator
guarantees this by construction and validates every sample via `main(input)`.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect variable-sized framed squares via colored bridges following the
    control-pixel sequence.

    D4 change:
    - aligned targets are recognized by shared lattice centers, not equal inner
      bbox spans;
    - bridge thickness adapts to the smaller endpoint's inner size.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, _border, frames = _detect_frames(g)
    by = _frames_by_color(frames)

    max_frame_r = max(int(f["bbox"][2]) for f in frames)

    ctrl_r = None
    for r in range(int(H - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None

    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(W)) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    start_color = int(ctrl[0])
    assert start_color in by
    start_cands = by[start_color]
    if len(start_cands) == 1:
        cur = start_cands[0]
    else:
        nxt_color = int(ctrl[1])
        nxt_cands = by.get(nxt_color, [])
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

    out = g.copy()
    for i in range(int(len(ctrl)) - 1):
        a = int(ctrl[i])
        b = int(ctrl[i + 1])
        assert int(cur["inner"]) == int(a)
        if b not in by:
            continue
        best = None
        bestf = None
        for f in by[b]:
            cand = _candidate_score(cur, f)
            if best is None or cand > best:
                best = cand
                bestf = f
        assert bestf is not None
        _draw_bridge(out, cur, bestf, int(a))
        cur = bestf

    return out


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


def _frames_by_color(frames: list[dict]) -> dict[int, list[dict]]:
    by: dict[int, list[dict]] = defaultdict(list)
    for f in frames:
        by[int(f["inner"])].append(f)
    return by


def _alignment_kind(f1: dict, f2: dict) -> str | None:
    r1, c1 = map(int, f1["center2"])
    r2, c2 = map(int, f2["center2"])
    if int(r1) == int(r2) and int(c1) != int(c2):
        return "row"
    if int(c1) == int(c2) and int(r1) != int(r2):
        return "col"
    return None


def _candidate_score(f1: dict, f2: dict) -> tuple[int, int]:
    kind = _alignment_kind(f1, f2)
    al = 1 if kind is not None and _bridge_rect(f1, f2) is not None else 0
    r1, c1 = map(int, f1["center2"])
    r2, c2 = map(int, f2["center2"])
    d = abs(int(r1) - int(r2)) + abs(int(c1) - int(c2))
    return int(al), -int(d)


def _bridge_rect(fa: dict, fb: dict) -> tuple[int, int, int, int] | None:
    kind = _alignment_kind(fa, fb)
    if kind is None:
        return None

    (r0a, c0a, r1a, c1a) = map(int, fa["bbox"])
    (r0b, c0b, r1b, c1b) = map(int, fb["bbox"])
    narrow = fa if int(fa["inner_size"]) <= int(fb["inner_size"]) else fb
    ia = narrow["inner_bbox"]

    if kind == "row":
        rr0, rr1 = int(ia[0]), int(ia[2])
        if int(c1a) < int(c0b):
            cc0, cc1 = int(c1a + 1), int(c0b - 1)
        else:
            cc0, cc1 = int(c1b + 1), int(c0a - 1)
        if int(cc0) > int(cc1):
            return None
        return int(rr0), int(rr1), int(cc0), int(cc1)

    cc0, cc1 = int(ia[1]), int(ia[3])
    if int(r1a) < int(r0b):
        rr0, rr1 = int(r1a + 1), int(r0b - 1)
    else:
        rr0, rr1 = int(r1b + 1), int(r0a - 1)
    if int(rr0) > int(rr1):
        return None
    return int(rr0), int(rr1), int(cc0), int(cc1)


def _draw_bridge(out: np.ndarray, fa: dict, fb: dict, color: int) -> None:
    rect = _bridge_rect(fa, fb)
    if rect is None:
        return
    rr0, rr1, cc0, cc1 = map(int, rect)
    H, W = out.shape
    for r in range(int(rr0), int(rr1) + 1):
        for c in range(int(cc0), int(cc1) + 1):
            if 0 <= int(r) < int(H) and 0 <= int(c) < int(W):
                out[int(r), int(c)] = int(color)


def _draw_frame(g: np.ndarray, *, r0: int, c0: int, outer: int, t: int, border: int, inner_col: int) -> None:
    g[int(r0) : int(r0 + outer), int(c0) : int(c0 + outer)] = int(border)
    g[int(r0 + t) : int(r0 + outer - t), int(c0 + t) : int(c0 + outer - t)] = int(inner_col)


def _bridge_cells(fa: dict, fb: dict) -> set[tuple[int, int]]:
    rect = _bridge_rect(fa, fb)
    if rect is None:
        return set()
    rr0, rr1, cc0, cc1 = map(int, rect)
    return {(int(r), int(c)) for r in range(int(rr0), int(rr1) + 1) for c in range(int(cc0), int(cc1) + 1)}


def _edge_geometry_count(frames_by_slot: dict[tuple[int, int], dict], path: list[tuple[int, int]]) -> int:
    count = 0
    for a_slot, b_slot in zip(path, path[1:]):
        fa = frames_by_slot[(int(a_slot[0]), int(a_slot[1]))]
        fb = frames_by_slot[(int(b_slot[0]), int(b_slot[1]))]
        if int(fa["inner_size"]) != int(fb["inner_size"]):
            count += 1
    return int(count)


def _extract_path_and_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg, _border, frames = _detect_frames(g)
    by = _frames_by_color(frames)
    max_frame_r = max(int(f["bbox"][2]) for f in frames)
    ctrl_r = None
    for r in range(int(g.shape[0] - 1), int(max_frame_r), -1):
        if int((g[int(r)] != int(bg)).sum()) >= 2:
            ctrl_r = int(r)
            break
    assert ctrl_r is not None
    ctrl = [int(g[int(ctrl_r), int(c)]) for c in range(int(g.shape[1])) if int(g[int(ctrl_r), int(c)]) != int(bg)]
    assert len(ctrl) >= 2

    cur = by[int(ctrl[0])][0]
    path_frames = [cur]
    if len(by[int(ctrl[0])]) > 1:
        best = None
        for cand in by[int(ctrl[0])]:
            nxt_cands = by.get(int(ctrl[1]), [])
            for g2 in (nxt_cands if nxt_cands else [cand]):
                score = _candidate_score(cand, g2)
                if best is None or score > best[0]:
                    best = (score, cand)
        assert best is not None
        cur = best[1]
        path_frames = [cur]

    for col in ctrl[1:]:
        bestf = None
        best = None
        for cand in by[int(col)]:
            score = _candidate_score(cur, cand)
            if best is None or score > best:
                best = score
                bestf = cand
        assert bestf is not None
        path_frames.append(bestf)
        cur = bestf

    inner_sizes = {int(f["inner_size"]) for f in frames}
    thicknesses = {int(f["t"]) for f in frames}
    variable_edges = sum(
        int(path_frames[i]["inner_size"]) != int(path_frames[i + 1]["inner_size"]) for i in range(int(len(path_frames)) - 1)
    )
    return {
        "frames": int(len(frames)),
        "distinct_inner_sizes": int(len(inner_sizes)),
        "distinct_thicknesses": int(len(thicknesses)),
        "variable_edges": int(variable_edges),
    }


_DIGITS = [int(i) for i in range(10)]
_GEOMS = [(1, 1), (3, 1), (1, 2), (3, 2)]


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.choice(np.array(_DIGITS, dtype=int)))
    border = int(np.random.choice(np.array([d for d in _DIGITS if int(d) != int(bg)], dtype=int)))
    inner_palette = [int(d) for d in _DIGITS if int(d) not in (int(bg), int(border))]
    if len(inner_palette) < 4:
        return None

    R = int(np.random.randint(2, 6))
    C = int(np.random.randint(2, 6))
    max_frames = int(R * C)
    if max_frames < 4:
        return None
    n_frames = int(np.random.randint(4, int(min(12, max_frames)) + 1))

    path: list[tuple[int, int]] | None = None
    for _path_try in range(500):
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

    slot_geom: dict[tuple[int, int], tuple[int, int]] = {}
    for _geom_try in range(300):
        slot_geom = {}
        for slot in path:
            geom = _GEOMS[int(np.random.randint(0, len(_GEOMS)))]
            slot_geom[(int(slot[0]), int(slot[1]))] = (int(geom[0]), int(geom[1]))
        inner_sizes = {int(v[0]) for v in slot_geom.values()}
        thicknesses = {int(v[1]) for v in slot_geom.values()}
        if len(inner_sizes) < 2 or len(thicknesses) < 2:
            continue
        tmp_frames = {
            slot: {"inner_size": int(geom[0]), "t": int(geom[1]), "outer": int(geom[0] + 2 * geom[1])}
            for slot, geom in slot_geom.items()
        }
        if _edge_geometry_count(tmp_frames, path) < 1:
            continue
        break
    else:
        return None

    row_radius: dict[int, int] = defaultdict(int)
    col_radius: dict[int, int] = defaultdict(int)
    for (rr, cc), (inner_size, t) in slot_geom.items():
        outer = int(inner_size + 2 * t)
        rad = int(outer // 2)
        row_radius[int(rr)] = max(int(row_radius[int(rr)]), int(rad))
        col_radius[int(cc)] = max(int(col_radius[int(cc)]), int(rad))

    top_margin = int(np.random.randint(1, 4))
    left_margin = int(np.random.randint(1, 4))

    row_centers: list[int] = [int(top_margin + row_radius[0])]
    for rr in range(1, int(R)):
        gap = int(np.random.randint(2, 6))
        row_centers.append(int(row_centers[-1] + row_radius[int(rr - 1)] + row_radius[int(rr)] + gap))
    col_centers: list[int] = [int(left_margin + col_radius[0])]
    for cc in range(1, int(C)):
        gap = int(np.random.randint(2, 6))
        col_centers.append(int(col_centers[-1] + col_radius[int(cc - 1)] + col_radius[int(cc)] + gap))

    frames: list[dict] = []
    slot_to_frame: dict[tuple[int, int], dict] = {}
    for rr, cc in path:
        inner_size, t = slot_geom[(int(rr), int(cc))]
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

    ctrl_gap = int(np.random.randint(2, 6))
    ctrl_r = int(max_r1 + ctrl_gap)
    bottom_margin = 1
    right_margin = int(np.random.randint(1, 5))
    H = int(ctrl_r + 1 + bottom_margin)
    W = int(max_c1 + 1 + right_margin)
    if not (7 <= int(H) <= 30 and 7 <= int(W) <= 30):
        return None

    used_in_row: dict[int, set[int]] = defaultdict(set)
    used_in_col: dict[int, set[int]] = defaultdict(set)

    start_color = int(np.random.choice(np.array(inner_palette, dtype=int)))
    colors_by_slot: dict[tuple[int, int], int] = {(0, 0): int(start_color)}
    used_in_row[0].add(int(start_color))
    used_in_col[0].add(int(start_color))

    for rr, cc in path[1:]:
        rr = int(rr)
        cc = int(cc)
        forbid = set(used_in_row[int(rr)]) | set(used_in_col[int(cc)]) | {int(start_color)}
        choices = [int(x) for x in inner_palette if int(x) not in forbid]
        if not choices:
            return None
        col = int(np.random.choice(np.array(choices, dtype=int)))
        colors_by_slot[(int(rr), int(cc))] = int(col)
        used_in_row[int(rr)].add(int(col))
        used_in_col[int(cc)].add(int(col))

    if len(colors_by_slot) != len(path):
        return None

    out = np.full((int(H), int(W)), int(bg), dtype=int)
    for f in frames:
        rr, cc = map(int, f["slot"])
        r0, c0, _r1, _c1 = map(int, f["bbox"])
        _draw_frame(
            out,
            r0=int(r0),
            c0=int(c0),
            outer=int(f["outer"]),
            t=int(f["t"]),
            border=int(border),
            inner_col=int(colors_by_slot[(int(rr), int(cc))]),
        )

    bridge_cells_all: set[tuple[int, int]] = set()
    ok = True
    for a_slot, b_slot in zip(path, path[1:]):
        fa = slot_to_frame[(int(a_slot[0]), int(a_slot[1]))]
        fb = slot_to_frame[(int(b_slot[0]), int(b_slot[1]))]
        cells = _bridge_cells(fa, fb)
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
        paint = int(colors_by_slot[(int(a_slot[0]), int(a_slot[1]))])
        for r, c in cells:
            out[int(r), int(c)] = int(paint)
        bridge_cells_all |= set(cells)
    if not ok:
        return None

    ctrl_colors = [int(colors_by_slot[(int(r), int(c))]) for r, c in path]
    if len(ctrl_colors) < 2:
        return None
    ctrl_cols = [int(1 + 2 * i) for i in range(int(len(ctrl_colors)))]
    if int(ctrl_cols[-1]) >= int(W):
        return None
    for col, cc in zip(ctrl_colors, ctrl_cols):
        out[int(ctrl_r), int(cc)] = int(col)

    inp = out.copy()
    for r, c in bridge_cells_all:
        inp[int(r), int(c)] = int(bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if int(_most_common_nonbg(inp, int(bg)) or -1) != int(border):
        return None

    metrics = _extract_path_and_metrics(inp)
    if int(metrics["distinct_inner_sizes"]) < 2 or int(metrics["distinct_thicknesses"]) < 2:
        return None
    if int(metrics["variable_edges"]) < 1:
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if pred.shape != out.shape:
        return None
    if not bool(np.array_equal(pred, out)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(700):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 700 attempts")
