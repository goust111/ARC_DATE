# Combo variant of 4a21e3da (index 1)
# combo_spec: D1/0+D5/1
# Name: 对角线切割+镜像翻转
#
# Design analysis for the fused variant
# a) Concretization / behavior trace
#    - Base task: border control pixels define one vertical and/or one horizontal cut.
#      Pieces in rectangular regions that touch a control edge slide to their corner;
#      pieces in non-touching regions vanish; fg on the split line stays in place.
#    - Fused task: the top/bottom control now defines a "/" diagonal cut
#      (constant r+c), while the left/right control defines a "\" diagonal cut
#      (constant r-c). Fg exactly on either diagonal is preserved in the center.
#      Non-line fg is classified by the sign of (r+c-k_sum) and/or (r-c-k_diff),
#      so the old quadrant labels still exist but are produced by diagonal half-planes.
#    - Example with controls on top and left:
#      top at (0, 10) gives r+c=10; left at (8, 0) gives r-c=8.
#      A pixel with (r+c<10, r-c<8) belongs to TL; (r+c>10, r-c<8) is TR;
#      (r+c<10, r-c>8) is BL; (r+c>10, r-c>8) is BR.
#      TL/TR/BL survive because they touch top or left; BR vanishes.
#      Each surviving piece is normalized inside its own bbox, mirrored across the
#      row axis implied by the vertical corner choice and across the column axis
#      implied by the horizontal corner choice, then anchored in the corner.
#    - The visible difference from the base task is twofold:
#      the central partition is diagonal/triangular instead of axis-aligned, and
#      the corner silhouette is mirrored instead of merely translated.
# b) Complexity calibration
#    - Freedom stays low: there are still only 1-2 controls, only two line families
#      (r+c and r-c), and only the four old corner labels.
#    - The generator rejects samples where mirror placement collapses to slide-only
#      or where diagonal and straight-cut solvers would coincide, so the fusion is
#      visually observable rather than a cosmetic rewrite.
# c) Hidden assumptions audit
#    - Original code assumes split pixels form 4-connected straight lines. A true
#      diagonal line does not: consecutive diagonal cells are only 8-neighbors.
#      The generator therefore inserts off-line bridge cells from adjacent regions
#      so the input sprite remains 4-connected while the diagonal cut is still clear.
#    - Original Stage 2 preserves local offsets under translation. That assumption
#      no longer holds because surviving pieces are mirrored inside their local bbox
#      before corner anchoring.
#    - Ray drawing now relies on exact diagonal traces from the control pixel to the
#      last fg on that trace. The generator guarantees that only designated split
#      pixels lie on those traces, so the ray stop is unambiguous.
# d) Placement legality
#    - The sprite bbox is still centered with margin >= 4 from every border.
#    - Every generated region piece stays strictly in its diagonal half-plane(s),
#      never overlaps preserved line pixels, and stays inside the bbox.
#    - Two-control samples enforce an integer line intersection inside the bbox so
#      both diagonals participate in one coherent central structure.
# e) Visual observability
#    - The preserved center contains a short diagonal slash and/or backslash, and
#      region pieces hug different sides of those cuts. Across samples, the solver
#      can infer that corner survival is controlled by which global edge-side a
#      triangular piece belongs to, not by color ids or hidden ordering.
# f) Visual cleanliness
#    - Only three colors are used exactly as in the original family: bg, fg, ctrl.
#      Rays remain intermittent and distinct from fg, and corner pieces stay far
#      from the centered split structure because of the original 4-cell margin.
# g) Color inference
#    - No new color mapping exists. The entire fusion is geometric: diagonal cut,
#      edge-touch survival, and mirror-then-corner placement.

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


Coord = tuple[int, int]
Region = tuple[str, str]


def _normalize_points(
    pts: set[Coord] | frozenset[Coord],
) -> tuple[frozenset[Coord], int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, cmin = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - rmin), int(c - cmin)) for r, c in pts)
    h = int(max(r for r, _ in shape) + 1)
    w = int(max(c for _, c in shape) + 1)
    return shape, h, w


def _flip_shape(
    shape: frozenset[Coord],
    h: int,
    w: int,
    *,
    flip_v: bool,
    flip_h: bool,
) -> frozenset[Coord]:
    out = set()
    for r, c in shape:
        rr = int(h - 1 - r) if flip_v else int(r)
        cc = int(w - 1 - c) if flip_h else int(c)
        out.add((int(rr), int(cc)))
    return frozenset(out)


def _place_shape_in_corner(
    shape: frozenset[Coord],
    h: int,
    w: int,
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[Coord]:
    r0 = 0 if dh == "T" else int(H - h)
    c0 = 0 if dv == "L" else int(W - w)
    return frozenset((int(r0 + r), int(c0 + c)) for r, c in shape)


def _corner_transform(
    pts: set[Coord] | frozenset[Coord],
    dh: str,
    dv: str,
    H: int,
    W: int,
    *,
    mirror: bool,
) -> frozenset[Coord]:
    shape, h, w = _normalize_points(pts)
    if mirror:
        shape = _flip_shape(shape, h, w, flip_v=True, flip_h=True)
    return _place_shape_in_corner(shape, h, w, dh, dv, H, W)


def _touches_edges(hs: str, vs: str) -> set[str]:
    te: set[str] = set()
    if hs in ("T", "A"):
        te.add("top")
    if hs in ("B", "A"):
        te.add("bottom")
    if vs in ("L", "A"):
        te.add("left")
    if vs in ("R", "A"):
        te.add("right")
    return te


def _pick_dest(hs: str, vs: str, *, has_h_split: bool, has_v_split: bool, edges: set[str]) -> tuple[str, str]:
    if has_h_split:
        dh = hs
    elif "top" in edges and "bottom" not in edges:
        dh = "T"
    elif "bottom" in edges and "top" not in edges:
        dh = "B"
    else:
        dh = "T"

    if has_v_split:
        dv = vs
    elif "left" in edges and "right" not in edges:
        dv = "L"
    elif "right" in edges and "left" not in edges:
        dv = "R"
    else:
        dv = "L"
    return str(dh), str(dv)


def _ctrl_meta(r: int, c: int, H: int, W: int, *, path_mode: str) -> dict[str, int | str]:
    if path_mode == "straight":
        if r == 0:
            return {"edge": "top", "family": "col", "const": int(c), "dr": 1, "dc": 0, "r": int(r), "c": int(c)}
        if r == H - 1:
            return {"edge": "bottom", "family": "col", "const": int(c), "dr": -1, "dc": 0, "r": int(r), "c": int(c)}
        if c == 0:
            return {"edge": "left", "family": "row", "const": int(r), "dr": 0, "dc": 1, "r": int(r), "c": int(c)}
        return {"edge": "right", "family": "row", "const": int(r), "dr": 0, "dc": -1, "r": int(r), "c": int(c)}

    if r == 0:
        return {"edge": "top", "family": "sum", "const": int(c), "dr": 1, "dc": -1, "r": int(r), "c": int(c)}
    if r == H - 1:
        return {
            "edge": "bottom",
            "family": "sum",
            "const": int(r + c),
            "dr": -1,
            "dc": 1,
            "r": int(r),
            "c": int(c),
        }
    if c == 0:
        return {"edge": "left", "family": "diff", "const": int(r), "dr": 1, "dc": 1, "r": int(r), "c": int(c)}
    return {
        "edge": "right",
        "family": "diff",
        "const": int(r - c),
        "dr": -1,
        "dc": -1,
        "r": int(r),
        "c": int(c),
    }


def _trace_cells(meta: dict[str, int | str], H: int, W: int) -> list[Coord]:
    r = int(meta["r"])
    c = int(meta["c"])
    dr = int(meta["dr"])
    dc = int(meta["dc"])
    out: list[Coord] = []
    while 0 <= r < H and 0 <= c < W:
        out.append((int(r), int(c)))
        r += int(dr)
        c += int(dc)
    return out


def _on_line(r: int, c: int, meta: dict[str, int | str]) -> bool:
    fam = str(meta["family"])
    const = int(meta["const"])
    if fam == "col":
        return int(c) == const
    if fam == "row":
        return int(r) == const
    if fam == "sum":
        return int(r + c) == const
    return int(r - c) == const


def _classify_region(
    r: int,
    c: int,
    *,
    row_meta: dict[str, int | str] | None,
    col_meta: dict[str, int | str] | None,
    sum_meta: dict[str, int | str] | None,
    diff_meta: dict[str, int | str] | None,
    path_mode: str,
) -> Region | None:
    if path_mode == "straight":
        if col_meta is not None and int(c) == int(col_meta["const"]):
            return None
        if row_meta is not None and int(r) == int(row_meta["const"]):
            return None
        hs = "A"
        vs = "A"
        if row_meta is not None:
            hs = "T" if int(r) < int(row_meta["const"]) else "B"
        if col_meta is not None:
            vs = "L" if int(c) < int(col_meta["const"]) else "R"
        return str(hs), str(vs)

    if sum_meta is not None and int(r + c) == int(sum_meta["const"]):
        return None
    if diff_meta is not None and int(r - c) == int(diff_meta["const"]):
        return None
    hs = "A"
    vs = "A"
    if diff_meta is not None:
        hs = "T" if int(r - c) < int(diff_meta["const"]) else "B"
    if sum_meta is not None:
        vs = "L" if int(r + c) < int(sum_meta["const"]) else "R"
    return str(hs), str(vs)


def _solve(grid: np.ndarray, *, path_mode: str, mirror: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    metas = [_ctrl_meta(r, c, H, W, path_mode=path_mode) for r, c in ctrl_pts]
    edges = {str(meta["edge"]) for meta in metas}

    row_meta = next((m for m in metas if str(m["family"]) == "row"), None)
    col_meta = next((m for m in metas if str(m["family"]) == "col"), None)
    sum_meta = next((m for m in metas if str(m["family"]) == "sum"), None)
    diff_meta = next((m for m in metas if str(m["family"]) == "diff"), None)

    out = np.full((H, W), int(bg), dtype=int)

    for meta in metas:
        trace = _trace_cells(meta, H, W)
        fg_steps = [i for i, (r, c) in enumerate(trace) if int(g[r, c]) == int(fg)]
        if not fg_steps:
            continue
        stop = int(max(fg_steps))
        for r, c in trace[: stop + 1]:
            if int(g[r, c]) == int(bg):
                out[int(r), int(c)] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if any(_on_line(rr, cc, meta) for meta in metas):
            out[rr, cc] = int(fg)

    regions: dict[Region, list[Coord]] = {}
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        reg = _classify_region(
            rr,
            cc,
            row_meta=row_meta,
            col_meta=col_meta,
            sum_meta=sum_meta,
            diff_meta=diff_meta,
            path_mode=path_mode,
        )
        if reg is None:
            continue
        regions.setdefault(reg, []).append((int(rr), int(cc)))

    has_h_split = (row_meta is not None) if path_mode == "straight" else (diff_meta is not None)
    has_v_split = (col_meta is not None) if path_mode == "straight" else (sum_meta is not None)

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & edges):
            continue
        dh, dv = _pick_dest(hs, vs, has_h_split=has_h_split, has_v_split=has_v_split, edges=edges)
        for rr, cc in _corner_transform(frozenset(pts), dh, dv, H, W, mirror=mirror):
            if 0 <= int(rr) < H and 0 <= int(cc) < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Diagonally cut a centered sprite with 45-degree border rays, keep fg on the
    diagonal cut line(s), and send only control-edge-touching triangular regions to
    corners after mirroring them inside their own bbox.
    """
    return _solve(grid, path_mode="diagonal", mirror=True)


def _bbox_cells(r0: int, c0: int, h: int, w: int) -> list[Coord]:
    return [(int(r), int(c)) for r in range(int(r0), int(r0 + h)) for c in range(int(c0), int(c0 + w))]


def _line_cells_in_bbox(meta: dict[str, int | str], H: int, W: int, r0: int, c0: int, h: int, w: int) -> list[Coord]:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    return [(r, c) for r, c in _trace_cells(meta, H, W) if int(r0) <= int(r) <= int(r1) and int(c0) <= int(c) <= int(c1)]


def _pick_segment(line_cells: list[Coord], rng: random.Random, *, include: Coord | None, min_len: int, max_len: int) -> list[Coord] | None:
    if not line_cells:
        return None
    hi = min(int(max_len), len(line_cells))
    lo = min(int(min_len), hi)
    if lo <= 0:
        return None
    lengths = list(range(int(lo), int(hi) + 1))
    rng.shuffle(lengths)
    if include is None:
        for ln in lengths:
            starts = list(range(0, len(line_cells) - ln + 1))
            if not starts:
                continue
            s = rng.choice(starts)
            return list(line_cells[s : s + ln])
        return None
    idx = line_cells.index(include)
    for ln in lengths:
        starts = [s for s in range(0, len(line_cells) - ln + 1) if s <= idx < s + ln]
        if not starts:
            continue
        s = rng.choice(starts)
        return list(line_cells[s : s + ln])
    return None


def _ctrl_from_anchor(edge: str, anchor: Coord, H: int, W: int) -> Coord | None:
    ar, ac = map(int, anchor)
    if edge == "top":
        cc = int(ar + ac)
        if 1 <= cc <= int(W - 2):
            return 0, int(cc)
        return None
    if edge == "bottom":
        cc = int(ar + ac - (H - 1))
        if 1 <= cc <= int(W - 2):
            return int(H - 1), int(cc)
        return None
    if edge == "left":
        rr = int(ar - ac)
        if 1 <= rr <= int(H - 2):
            return int(rr), 0
        return None
    rr = int(ar - ac + (W - 1))
    if 1 <= rr <= int(H - 2):
        return int(rr), int(W - 1)
    return None


def _bridge_candidates(meta: dict[str, int | str], a: Coord, b: Coord) -> list[Coord]:
    (r1, c1), (r2, c2) = a, b
    if int(r1) > int(r2):
        r1, c1, r2, c2 = int(r2), int(c2), int(r1), int(c1)
    fam = str(meta["family"])
    if fam == "sum":
        return [(int(r1), int(c1 - 1)), (int(r2), int(c1))]
    return [(int(r1), int(c1 + 1)), (int(r2), int(c1))]


def _neighbors4(p: Coord) -> list[Coord]:
    r, c = p
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _shortest_path(allowed: set[Coord], sources: set[Coord], target: Coord) -> list[Coord] | None:
    if target in sources:
        return [target]
    frontier = list(sources)
    prev: dict[Coord, Coord | None] = {s: None for s in frontier}
    q_idx = 0
    while q_idx < len(frontier):
        cur = frontier[q_idx]
        q_idx += 1
        for nxt in _neighbors4(cur):
            if nxt not in allowed or nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == target:
                path = [target]
                while path[-1] not in sources:
                    parent = prev[path[-1]]
                    assert parent is not None
                    path.append(parent)
                path.reverse()
                return path
            frontier.append(nxt)
    return None


def _connected_piece_from_required(allowed: set[Coord], required: set[Coord]) -> set[Coord] | None:
    if not required:
        return None
    required = set(required)
    piece = {next(iter(required))}
    for target in list(required - piece):
        path = _shortest_path(allowed, piece, target)
        if path is None:
            return None
        piece.update(path)
    return piece


def _grow_piece(
    allowed: set[Coord],
    required: set[Coord],
    target_px: int,
    union_seed: set[Coord],
    rng: random.Random,
) -> frozenset[Coord] | None:
    if not allowed:
        return None
    req = set(required)
    if not req:
        anchors = [p for p in allowed if any(n in union_seed for n in _neighbors4(p))]
        if not anchors:
            return None
        req = {rng.choice(anchors)}
    piece = _connected_piece_from_required(allowed, req)
    if piece is None or len(piece) > int(target_px):
        return None

    while len(piece) < int(target_px):
        frontier = []
        for cell in piece:
            for nxt in _neighbors4(cell):
                if nxt in allowed and nxt not in piece:
                    score = sum(1 for nn in _neighbors4(nxt) if nn in piece)
                    score += 2 if any(nn in union_seed for nn in _neighbors4(nxt)) else 0
                    frontier.append((int(score), nxt))
        if not frontier:
            return None
        best = max(score for score, _ in frontier)
        choices = [cell for score, cell in frontier if score >= best - 1]
        piece.add(rng.choice(choices))
    return frozenset(piece)


def _distribute_budget(
    regions: list[Region],
    counts: dict[Region, int],
    total_budget: int,
    mins: dict[Region, int],
    rng: random.Random,
) -> dict[Region, int] | None:
    if not regions:
        return {}
    if total_budget < sum(mins[r] for r in regions):
        return None
    if total_budget > sum(counts[r] for r in regions):
        return None
    out = {r: int(mins[r]) for r in regions}
    remaining = int(total_budget - sum(out.values()))
    order = list(regions)
    rng.shuffle(order)
    for reg in order:
        if remaining <= 0:
            break
        cap = int(counts[reg] - out[reg])
        if cap <= 0:
            continue
        add = rng.randint(0, min(cap, remaining))
        out[reg] += int(add)
        remaining -= int(add)
    while remaining > 0:
        movable = [reg for reg in regions if out[reg] < counts[reg]]
        if not movable:
            return None
        reg = rng.choice(movable)
        out[reg] += 1
        remaining -= 1
    return out


def _build_input_and_metrics(rng: random.Random) -> tuple[np.ndarray, dict[str, int]] | None:
    S = rng.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = rng.choices([1, 2], weights=[1, 12])[0]
    if int(num_ctrl) == 1:
        ctrl_edges = [rng.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [rng.choice(["top", "bottom"]), rng.choice(["left", "right"])]

    sprite_h = rng.randint(6, 8)
    sprite_w = rng.randint(8, 10)

    candidates: list[
        tuple[
            int,
            int,
            list[Coord],
            list[dict[str, int | str]],
            dict[str, list[Coord]],
            Coord | None,
            dict[Region, int],
            list[Region],
            list[Region],
            set[str],
            dict[str, int | str] | None,
            dict[str, int | str] | None,
        ]
    ] = []
    for r0 in range(4, int(H - 5 - sprite_h + 2)):
        for c0 in range(4, int(W - 5 - sprite_w + 2)):
            bbox = set(_bbox_cells(r0, c0, sprite_h, sprite_w))
            inner_rows = range(int(r0 + 1), int(r0 + sprite_h - 1))
            inner_cols = range(int(c0 + 1), int(c0 + sprite_w - 1))
            anchors = [(int(ar), int(ac)) for ar in inner_rows for ac in inner_cols]
            rng.shuffle(anchors)
            for anchor in anchors:
                ctrl_pts = []
                ok = True
                for edge in ctrl_edges:
                    pt = _ctrl_from_anchor(edge, anchor, H, W)
                    if pt is None:
                        ok = False
                        break
                    ctrl_pts.append(pt)
                if not ok or len(set(ctrl_pts)) != len(ctrl_pts):
                    continue

                metas = [_ctrl_meta(r, c, H, W, path_mode="diagonal") for r, c in ctrl_pts]
                edges = {str(meta["edge"]) for meta in metas}
                sum_meta = next((m for m in metas if str(m["family"]) == "sum"), None)
                diff_meta = next((m for m in metas if str(m["family"]) == "diff"), None)
                has_v = sum_meta is not None
                has_h = diff_meta is not None

                if has_h and has_v:
                    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
                elif has_v:
                    all_regions = [("A", "L"), ("A", "R")]
                else:
                    all_regions = [("T", "A"), ("B", "A")]

                kept = [r for r in all_regions if _touches_edges(*r) & edges]
                vanish = [r for r in all_regions if r not in kept]
                used_regions = list(kept + vanish)

                line_cells: dict[str, list[Coord]] = {}
                for meta in metas:
                    cells = _line_cells_in_bbox(meta, H, W, r0, c0, sprite_h, sprite_w)
                    if len(cells) < 3:
                        ok = False
                        break
                    line_cells[str(meta["family"])] = cells
                if not ok:
                    continue

                intersection: Coord | None = None
                if sum_meta is not None and diff_meta is not None:
                    common = set(line_cells["sum"]) & set(line_cells["diff"])
                    if set(common) != {anchor}:
                        continue
                    intersection = anchor

                counts: dict[Region, int] = defaultdict(int)
                for r, c in bbox:
                    reg = _classify_region(
                        r,
                        c,
                        row_meta=None,
                        col_meta=None,
                        sum_meta=sum_meta,
                        diff_meta=diff_meta,
                        path_mode="diagonal",
                    )
                    if reg is not None:
                        counts[reg] += 1
                if any(counts.get(reg, 0) < 5 for reg in used_regions):
                    continue

                candidates.append(
                    (
                        int(r0),
                        int(c0),
                        list(ctrl_pts),
                        list(metas),
                        line_cells,
                        intersection,
                        counts,
                        kept,
                        vanish,
                        edges,
                        sum_meta,
                        diff_meta,
                    )
                )

    if not candidates:
        return None

    r0, c0, ctrl_pts, metas, line_cells, intersection, region_counts, kept, vanish, edges, sum_meta, diff_meta = rng.choice(candidates)
    used_regions = list(kept + vanish)
    area = int(sprite_h * sprite_w)

    split_pixels: set[Coord] = set()
    ordered_segments: dict[str, list[Coord]] = {}
    for fam, cells in line_cells.items():
        segment = _pick_segment(cells, rng, include=intersection, min_len=3, max_len=4)
        if segment is None:
            return None
        ordered_segments[fam] = list(segment)
        split_pixels |= set(segment)

    region_required: dict[Region, set[Coord]] = {reg: set() for reg in used_regions}
    for fam, segment in ordered_segments.items():
        meta = sum_meta if fam == "sum" else diff_meta
        assert meta is not None
        for a, b in zip(segment, segment[1:]):
            cands = []
            for cell in _bridge_candidates(meta, a, b):
                rr, cc = cell
                if not (r0 <= rr < r0 + sprite_h and c0 <= cc < c0 + sprite_w):
                    continue
                reg = _classify_region(
                    rr,
                    cc,
                    row_meta=None,
                    col_meta=None,
                    sum_meta=sum_meta,
                    diff_meta=diff_meta,
                    path_mode="diagonal",
                )
                if reg is None or reg not in region_required:
                    continue
                cands.append((reg, cell))
            if not cands:
                return None
            reg, cell = rng.choice(cands)
            region_required[reg].add(cell)

    bbox_set = set(_bbox_cells(r0, c0, sprite_h, sprite_w))
    region_allowed: dict[Region, set[Coord]] = {reg: set() for reg in used_regions}
    for r, c in bbox_set:
        if (r, c) in split_pixels:
            continue
        reg = _classify_region(
            r,
            c,
            row_meta=None,
            col_meta=None,
            sum_meta=sum_meta,
            diff_meta=diff_meta,
            path_mode="diagonal",
        )
        if reg is not None and reg in region_allowed:
            region_allowed[reg].add((r, c))

    for reg in used_regions:
        if not region_required[reg]:
            anchors = [p for p in region_allowed[reg] if any(n in split_pixels for n in _neighbors4(p))]
            if not anchors:
                return None
            region_required[reg].add(rng.choice(anchors))

    split_count = len(split_pixels)
    lower_total = max(int(np.ceil(0.40 * area)), split_count + 3 * len(used_regions))
    upper_total = min(int(np.floor(0.58 * area)), split_count + sum(len(region_allowed[r]) for r in used_regions))
    if lower_total > upper_total:
        return None
    target_total = rng.randint(int(lower_total), int(upper_total))
    piece_budget = int(target_total - split_count)

    mins = {reg: max(3, len(region_required[reg])) for reg in used_regions}
    budgets = _distribute_budget(used_regions, {r: len(region_allowed[r]) for r in used_regions}, piece_budget, mins, rng)
    if budgets is None:
        return None

    pieces: dict[Region, frozenset[Coord]] = {}
    current_union = set(split_pixels)
    order = list(used_regions)
    order.sort(key=lambda reg: (0 if reg in kept else 1, reg))
    for reg in order:
        piece = _grow_piece(region_allowed[reg], region_required[reg], budgets[reg], current_union, rng)
        if piece is None:
            return None
        pieces[reg] = piece
        current_union |= set(piece)

    combined = set(split_pixels)
    for pts in pieces.values():
        combined |= set(pts)

    mask = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    arr = np.array(list(combined))
    bb_h = int(arr[:, 0].max() - arr[:, 0].min() + 1)
    bb_w = int(arr[:, 1].max() - arr[:, 1].min() + 1)
    if bb_h < 5 or bb_h > 8 or bb_w < 8 or bb_w > 10:
        return None
    density = len(combined) / float(bb_h * bb_w)
    if density < 0.39 or density > 0.60:
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for r, c in ctrl_pts:
        inp[int(r), int(c)] = int(ctrl_color)

    fused = _solve(inp, path_mode="diagonal", mirror=True)
    diagonal_slide = _solve(inp, path_mode="diagonal", mirror=False)
    straight_mirror = _solve(inp, path_mode="straight", mirror=True)
    mirror_delta = int(np.count_nonzero(fused != diagonal_slide))
    path_delta = int(np.count_nonzero(fused != straight_mirror))
    if mirror_delta < 4 or path_delta < 6:
        return None

    return inp, {"mirror_delta": int(mirror_delta), "path_delta": int(path_delta), "split_pixels": int(split_count)}


def measure_variant_metrics(grid: np.ndarray) -> dict[str, int]:
    fused = _solve(grid, path_mode="diagonal", mirror=True)
    diagonal_slide = _solve(grid, path_mode="diagonal", mirror=False)
    straight_mirror = _solve(grid, path_mode="straight", mirror=True)
    return {
        "mirror_delta": int(np.count_nonzero(fused != diagonal_slide)),
        "path_delta": int(np.count_nonzero(fused != straight_mirror)),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1000):
        built = _build_input_and_metrics(rng)
        if built is None:
            continue
        inp, _metrics = built
        return inp
    raise RuntimeError("Failed to generate valid input after 1000 attempts")
