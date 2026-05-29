# Combo variant of 4a21e3da (index 37)
# combo_spec: D3/3+D4/3
# Name: 箭头指示方向+距离编码射线长度

from __future__ import annotations

import itertools
import random
import sys

import numpy as np

try:
    from arc_utils import colors_present, get_bg, is_contiguous
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import colors_present, get_bg, is_contiguous


"""
Design analysis for combo D3/3 + D4/3
=====================================

a) concretization_and_behavior_trace:
  original:
    - each red border marker is a single pixel; its border side determines ray
      direction and its own row/column is the split line.
    - the ray runs on that split line until the farthest fg pixel on the same
      line, skipping over fg and painting only background.
  combo:
    - each control is now a 3-cell L-shaped arrow touching exactly one border:
        top:    {(0,s), (0,s+1), (1,s)}    launch column = s, ray goes downward
        bottom: {(H-1,s-1), (H-1,s), (H-2,s)} launch column = s, ray goes upward
        left:   {(s-1,0), (s,0), (s,1)}    launch row = s, ray goes rightward
        right:  {(s,W-2), (s,W-1), (s+1,W-1)} launch row = s, ray goes leftward
    - after decoding direction from shape, the same launch row/column still
      defines the split line exactly as in the source task.
    - the ray stop is no longer always the farthest aligned fg. Instead, the
      arrow's border position encodes one of three visible depth bins by distance
      from the edge midpoint:
        distance 0  -> long   -> farthest aligned fg
        distance 1  -> medium -> middle aligned fg
        distance >=2 -> short -> nearest aligned fg
    - small trace:
        top arrow at column 8 on a 20-wide grid is near the top-edge midpoint,
        so after decoding a downward arrow the solver draws a long vertical ray on
        column 8. A top arrow at column 7 is farther from the midpoint, so the
        same downward split at column 7 becomes short or medium depending on the
        exact offset. Piece partitioning and corner sliding are unchanged.

b) complexity_calibration:
  - I keep only one extra shape family: four cardinal arrow orientations. That
    adds a single direction-decoding step without introducing many classes.
  - I quantize length coding into exactly three bins. This keeps the mapping
    learnable from a few examples and avoids brittle fine-grained arithmetic.
  - Every generated active split line contains exactly three spaced fg anchors, so
    short / medium / long always land on three visibly different stops.
  - Generator rejects samples where all controls are "long", because those would
    collapse back to the source farthest-stop behavior.

c) implicit_assumption_audit_against_original_main:
  - `np.argwhere(g == ctrl)` used to enumerate independent emitters. That breaks:
    one arrow would look like three controls. New solver first extracts 4-connected
    ctrl components, requires size-3 2x2 L-shapes, and decodes one launch site per
    component.
  - Original side-detection logic assumed any ctrl-colored border cell determines
    the split. Now only the decoded launch cell does.
  - Original ray painter assumed stop = farthest aligned fg. That assumption is
    replaced by a bin lookup from edge-midpoint distance plus the ordered aligned
    fg anchors on the split line.
  - Original output restoration only had to redraw one ctrl pixel per source. Now
    the full 3-cell arrow must be preserved in output.
  - The region-splitting and corner-slide logic still relies only on `r_split`,
    `c_split`, and edge-touching; those assumptions remain valid once split lines
    come from decoded arrow launches.

d) placement_legality:
  - Arrows stay away from corners by keeping split coordinates in `[1, W-2]` or
    `[1, H-2]`, so every arrow touches exactly one border.
  - The sprite still keeps a margin of at least four cells from every edge, so
    arrows never overlap fg.
  - Generator chooses split coordinates from a widened near-center band, then
    places the sprite so each active split passes through the sprite interior with
    at least two fg columns/rows on both sides.
  - Each active split line gets three non-adjacent fg anchor pixels, guaranteeing
    distinct short / medium / long stops while staying inside the sprite bbox.

e) visual_observability:
  - A human can see arrow orientation directly from the 3-cell corner shape and
    correlate it with the output ray direction.
  - Because otherwise similar samples can have short, medium, or long rays on the
    same kind of split, the distance-to-midpoint cue is visible across examples.
  - No hidden color ordering is used: only geometry and relative border position
    determine the output.

f) visual_cleanliness:
  - Only the original three colors remain: bg, fg, ctrl.
  - Arrows occupy a tiny border-local 2x2 zone, while rays stay on one row/col and
    moved fg pieces go to corners, so the three roles are spatially separated.
  - Requiring split-line fg anchors to be sparse avoids the ray being visually
    indistinguishable across bins.

g) new_solver_assumptions_backed_by_generator:
  - All ctrl cells belong to 1 or 2 size-3 L-shaped arrow components.
  - There is at most one vertical launch line and at most one horizontal launch
    line, exactly as in the source task.
  - Every active split line contains three aligned fg anchors with gaps of at
    least one background cell between consecutive anchors, so all three depth bins
    produce different painted ray cells.
  - At least one control per generated sample is medium or short; otherwise the
    sample is rejected as degenerate.
"""


TOP_ARROW = frozenset({(0, 0), (0, 1), (1, 0)})
RIGHT_ARROW = frozenset({(0, 0), (0, 1), (1, 1)})
LEFT_ARROW = frozenset({(0, 0), (1, 0), (1, 1)})
BOTTOM_ARROW = frozenset({(0, 1), (1, 0), (1, 1)})


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((int(rr), int(cc)))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
        comps.append(comp)
    return comps


def _decode_arrow(comp: list[tuple[int, int]], H: int, W: int) -> dict[str, object]:
    rs = [int(r) for r, _ in comp]
    cs = [int(c) for _, c in comp]
    rmin = int(min(rs))
    rmax = int(max(rs))
    cmin = int(min(cs))
    cmax = int(max(cs))
    norm = frozenset((int(r - rmin), int(c - cmin)) for r, c in comp)
    touches: list[str] = []
    if rmin == 0:
        touches.append("top")
    if rmax == H - 1:
        touches.append("bottom")
    if cmin == 0:
        touches.append("left")
    if cmax == W - 1:
        touches.append("right")

    assert len(comp) == 3
    assert rmax - rmin == 1 and cmax - cmin == 1
    assert len(touches) == 1

    edge = str(touches[0])
    if norm == TOP_ARROW:
        assert edge == "top"
        launch = (0, int(cmin))
    elif norm == BOTTOM_ARROW:
        assert edge == "bottom"
        launch = (int(H - 1), int(cmax))
    elif norm == LEFT_ARROW:
        assert edge == "left"
        launch = (int(rmax), 0)
    elif norm == RIGHT_ARROW:
        assert edge == "right"
        launch = (int(rmin), int(W - 1))
    else:
        raise AssertionError("Unexpected control arrow shape")

    return {"edge": edge, "launch": launch, "cells": tuple(sorted((int(r), int(c)) for r, c in comp))}


def _find_arrows(g: np.ndarray, ctrl: int) -> list[dict[str, object]]:
    arrows = [_decode_arrow(comp, *g.shape) for comp in _components4(g == int(ctrl))]
    arrows.sort(key=lambda a: (str(a["edge"]), tuple(a["launch"])))
    return arrows


def _edge_mid_distance(pos: int, size: int) -> int:
    if int(size) % 2 == 1:
        mids = [int(size // 2)]
    else:
        mids = [int(size // 2 - 1), int(size // 2)]
    return int(min(abs(int(pos) - m) for m in mids))


def _distance_bin(pos: int, size: int) -> int:
    d = int(_edge_mid_distance(int(pos), int(size)))
    if d == 0:
        return 0
    if d == 1:
        return 1
    return 2


def _ordered_hits(scan_grid: np.ndarray, fg: int, edge: str, launch: tuple[int, int]) -> list[int]:
    r0, c0 = launch
    if edge == "top":
        return [int(rr) for rr in np.flatnonzero(scan_grid[:, int(c0)] == int(fg))]
    if edge == "bottom":
        return [int(rr) for rr in np.flatnonzero(scan_grid[:, int(c0)] == int(fg))[::-1]]
    if edge == "left":
        return [int(cc) for cc in np.flatnonzero(scan_grid[int(r0), :] == int(fg))]
    if edge == "right":
        return [int(cc) for cc in np.flatnonzero(scan_grid[int(r0), :] == int(fg))[::-1]]
    raise ValueError(edge)


def _select_stop(hits: list[int], edge: str, launch: tuple[int, int], H: int, W: int, *, use_distance_stop: bool) -> int:
    assert hits
    if not use_distance_stop:
        return int(hits[-1])
    coord = int(launch[1]) if edge in ("top", "bottom") else int(launch[0])
    size = int(W) if edge in ("top", "bottom") else int(H)
    b = int(_distance_bin(coord, size))
    if b == 0:
        return int(hits[-1])
    if b == 1:
        return int(hits[len(hits) // 2])
    return int(hits[0])


def _paint_ray(
    scan_grid: np.ndarray,
    out: np.ndarray,
    edge: str,
    launch: tuple[int, int],
    fg: int,
    ctrl: int,
    bg: int,
    *,
    use_distance_stop: bool,
) -> None:
    H, W = scan_grid.shape
    r0, c0 = tuple(int(x) for x in launch)
    hits = _ordered_hits(scan_grid, fg, edge, (int(r0), int(c0)))
    if not hits:
        return
    stop = int(_select_stop(hits, edge, (int(r0), int(c0)), H, W, use_distance_stop=use_distance_stop))
    if edge == "top":
        for rr in range(0, int(stop) + 1):
            if int(scan_grid[int(rr), int(c0)]) == int(bg):
                out[int(rr), int(c0)] = int(ctrl)
    elif edge == "bottom":
        for rr in range(int(H - 1), int(stop) - 1, -1):
            if int(scan_grid[int(rr), int(c0)]) == int(bg):
                out[int(rr), int(c0)] = int(ctrl)
    elif edge == "left":
        for cc in range(0, int(stop) + 1):
            if int(scan_grid[int(r0), int(cc)]) == int(bg):
                out[int(r0), int(cc)] = int(ctrl)
    elif edge == "right":
        for cc in range(int(W - 1), int(stop) - 1, -1):
            if int(scan_grid[int(r0), int(cc)]) == int(bg):
                out[int(r0), int(cc)] = int(ctrl)
    else:
        raise ValueError(edge)


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


def _pick_dest(E: set[str], r_split: int | None, c_split: int | None, hs: str, vs: str) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif "top" in E and "bottom" not in E:
        dh = "T"
    elif "bottom" in E and "top" not in E:
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif "left" in E and "right" not in E:
        dv = "L"
    elif "right" in E and "left" not in E:
        dv = "R"
    else:
        dv = "L"
    return str(dh), str(dv)


def _solve(grid: np.ndarray, *, use_distance_stop: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    arrows = _find_arrows(g, ctrl)
    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for arrow in arrows:
        edge = str(arrow["edge"])
        launch_r, launch_c = tuple(int(x) for x in arrow["launch"])
        E.add(edge)
        if edge in ("top", "bottom"):
            assert c_split is None or int(c_split) == int(launch_c)
            c_split = int(launch_c)
        else:
            assert r_split is None or int(r_split) == int(launch_r)
            r_split = int(launch_r)

    out = np.full((H, W), int(bg), dtype=int)
    for arrow in arrows:
        _paint_ray(
            g,
            out,
            str(arrow["edge"]),
            tuple(int(x) for x in arrow["launch"]),
            fg,
            ctrl,
            bg,
            use_distance_stop=use_distance_stop,
        )

    for arrow in arrows:
        for r, c in tuple(arrow["cells"]):
            out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            out[int(r), int(c)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if c < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue
        dh, dv = _pick_dest(E, r_split, c_split, hs, vs)
        rs = [int(r) for r, _ in pts]
        cs = [int(c) for _, c in pts]
        rmin = int(min(rs))
        rmax = int(max(rs))
        cmin = int(min(cs))
        cmax = int(max(cs))
        dr = int(-rmin) if dh == "T" else int(H - 1 - rmax)
        dc = int(-cmin) if dv == "L" else int(W - 1 - cmax)
        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each border arrow, use its orientation as ray direction and its border
    offset from the edge midpoint as a short/medium/long stop code, then apply the
    original split-and-slide rule.
    """
    return _solve(grid, use_distance_stop=True)


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) not in cells:
                    neighbors.add((int(nr), int(nc)))
        valid = [
            (int(nr), int(nc))
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(int(nr), int(nc))}) - min(r for r, _ in cells | {(int(nr), int(nc))}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(int(nr), int(nc))}) - min(c for _, c in cells | {(int(nr), int(nc))}) + 1 <= int(max_w)
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _make_arrow(edge: str, split: int, H: int, W: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    if edge == "top":
        return {(0, int(split)), (0, int(split + 1)), (1, int(split))}, (0, int(split))
    if edge == "bottom":
        return {(int(H - 1), int(split - 1)), (int(H - 1), int(split)), (int(H - 2), int(split))}, (int(H - 1), int(split))
    if edge == "left":
        return {(int(split - 1), 0), (int(split), 0), (int(split), 1)}, (int(split), 0)
    if edge == "right":
        return {(int(split), int(W - 2)), (int(split), int(W - 1)), (int(split + 1), int(W - 1))}, (int(split), int(W - 1))
    raise ValueError(edge)


def _split_band(size: int) -> tuple[int, int]:
    lo = max(1, int(round(size * 0.35)))
    hi = min(int(size - 2), int(round(size * 0.65)))
    return int(lo), int(hi)


def _pick_split_for_bin(size: int, wanted_bin: int) -> int | None:
    lo, hi = _split_band(int(size))
    cands = [int(i) for i in range(int(lo), int(hi) + 1) if int(_distance_bin(int(i), int(size))) == int(wanted_bin)]
    if not cands:
        return None
    return int(random.choice(cands))


def _choose_three_spaced(values: list[int]) -> list[int] | None:
    vals = sorted(int(v) for v in values)
    valid = [list(map(int, comb)) for comb in itertools.combinations(vals, 3) if int(comb[1] - comb[0]) >= 2 and int(comb[2] - comb[1]) >= 2]
    if not valid:
        return None
    return random.choice(valid)


def _mask_from_pts(H: int, W: int, pts: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=int)
    for r, c in pts:
        mask[int(r), int(c)] = 1
    return mask


def _count_nonlong_arrows(arrows: list[dict[str, object]], H: int, W: int) -> int:
    total = 0
    for arrow in arrows:
        launch_r, launch_c = tuple(int(x) for x in arrow["launch"])
        edge = str(arrow["edge"])
        coord = int(launch_c) if edge in ("top", "bottom") else int(launch_r)
        size = int(W) if edge in ("top", "bottom") else int(H)
        total += int(_distance_bin(coord, size) > 0)
    return int(total)


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(18, 20)
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[3, 2])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    wanted_bins = [random.choice([0, 1, 2]) for _ in ctrl_edges]
    if not any(int(b) > 0 for b in wanted_bins):
        wanted_bins[random.randrange(len(wanted_bins))] = random.choice([1, 2])

    c_split: int | None = None
    r_split: int | None = None
    arrows: list[dict[str, object]] = []
    E = set(ctrl_edges)
    has_v = False
    has_h = False

    for edge, wanted_bin in zip(ctrl_edges, wanted_bins):
        if edge in ("top", "bottom"):
            c_split = _pick_split_for_bin(W, int(wanted_bin))
            if c_split is None:
                return None
            cells, launch = _make_arrow(edge, int(c_split), H, W)
            arrows.append({"edge": edge, "cells": cells, "launch": launch})
            has_v = True
        else:
            r_split = _pick_split_for_bin(H, int(wanted_bin))
            if r_split is None:
                return None
            cells, launch = _make_arrow(edge, int(r_split), H, W)
            arrows.append({"edge": edge, "cells": cells, "launch": launch})
            has_h = True

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
        return _pick_dest(E, r_split, c_split, hs, vs)

    def touches_E(hs: str, vs: str) -> bool:
        return bool(_touches_edges(hs, vs) & E)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if touches_E(*reg)]
    vanish = [reg for reg in all_regions if reg not in kept]

    sprite_h = random.randint(6, 8)
    sprite_w = random.randint(8, 10)

    if has_v:
        left_span_cands = [
            int(x)
            for x in range(2, int(sprite_w - 2))
            if int(c_split - x) >= 4 and int(c_split + (sprite_w - x - 1)) <= int(W - 5)
        ]
        if not left_span_cands:
            return None
        left_span = int(random.choice(left_span_cands))
        sprite_c0 = int(c_split - left_span)
    else:
        hi = int(W - 5 - sprite_w + 1)
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, int(hi))

    if has_h:
        top_span_cands = [
            int(x)
            for x in range(2, int(sprite_h - 2))
            if int(r_split - x) >= 4 and int(r_split + (sprite_h - x - 1)) <= int(H - 5)
        ]
        if not top_span_cands:
            return None
        top_span = int(random.choice(top_span_cands))
        sprite_r0 = int(r_split - top_span)
    else:
        hi = int(H - 5 - sprite_h + 1)
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, int(hi))

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0) if hs in ("T", "A") else int(r_split + 1)
        r1 = int(r_split - 1 if hs == "T" else sprite_r1) if has_h else int(sprite_r1)
        c0 = int(sprite_c0) if vs in ("L", "A") else int(c_split + 1)
        c1 = int(c_split - 1 if vs == "L" else sprite_c1) if has_v else int(sprite_c1)
        return int(r0), int(r1), int(c0), int(c1)

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        rows = [int(r) for r in range(int(sprite_r0), int(sprite_r1) + 1) if r != r_split]
        picked_rows = _choose_three_spaced(rows)
        if picked_rows is None:
            return None
        for r in picked_rows:
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        cols = [int(c) for c in range(int(sprite_c0), int(sprite_c1) + 1) if c != c_split]
        picked_cols = _choose_three_spaced(cols)
        if picked_cols is None:
            return None
        for c in picked_cols:
            split_pixels.add((int(r_split), int(c)))

    n_split = int(len(split_pixels))
    area = int(sprite_h * sprite_w)
    target_density = random.uniform(0.39, 0.58)
    target_total = int(target_density * area)

    n_kept = int(len(kept))
    n_vanish = int(len(vanish))
    budget_pieces = int(target_total - n_split)
    if n_vanish > 0:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= int(vanish_budget)
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for i in range(n_kept):
        r0, r1, c0, c1 = sub_bbox(*kept[i])
        max_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if i == n_kept - 1:
            px = int(remaining)
        else:
            avg = int(remaining // (n_kept - i))
            lo_px = max(3, int(avg - 2))
            hi_px = min(int(max_area), int(avg + 2))
            if lo_px > hi_px:
                return None
            px = random.randint(int(lo_px), int(hi_px))
        px = max(3, min(int(px), int(max_area), int(remaining - (n_kept - i - 1) * 3)))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(int(b) < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = pick_dest(hs, vs)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = int(max(r for r, _ in piece) + 1)
        pw = int(max(c for _, c in piece) + 1)

        if has_h:
            inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
        else:
            split_rows = sorted(int(r) for r, c in split_pixels if int(c) == int(c_split))
            if split_rows:
                mid = int(split_rows[len(split_rows) // 2])
                inp_r0 = max(int(r0), min(int(mid - ph // 2), int(r1 - ph + 1)))
            else:
                inp_r0 = int(r0)

        if has_v:
            inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
        else:
            split_cols = sorted(int(c) for r, c in split_pixels if int(r) == int(r_split))
            if split_cols:
                mid = int(split_cols[len(split_cols) // 2])
                inp_c0 = max(int(c0), min(int(mid - pw // 2), int(c1 - pw + 1)))
            else:
                inp_c0 = int(c0)

        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split))):
                return None
            if has_h and ((hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split))):
                return None

        inp_rs = [int(r) for r, _ in inp_pts]
        inp_cs = [int(c) for _, c in inp_pts]
        dr = int(-min(inp_rs)) if dh == "T" else int(H - 1 - max(inp_rs))
        dc = int(-min(inp_cs)) if dv == "L" else int(W - 1 - max(inp_cs))
        out_pts = frozenset((int(r + dr), int(c + dc)) for r, c in inp_pts)

        for r, c in out_pts:
            if has_v and ((vs == "L" and c >= int(c_split)) or (vs == "R" and c <= int(c_split))):
                return None
            if has_h and ((hs == "T" and r >= int(r_split)) or (hs == "B" and r <= int(r_split))):
                return None
        input_pieces[reg] = inp_pts

    all_inp_fg: set[tuple[int, int]] = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    all_inp_fg |= set(split_pixels)
    if not all_inp_fg:
        return None

    fg_arr = np.array(list(all_inp_fg), dtype=int)
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    if not bool(is_contiguous(_mask_from_pts(H, W, all_inp_fg))):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh = int(vr1 - vr0 + 1)
            vw = int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = min(int(vanish_budget), int(vh * vw))
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph = int(max(r for r, _ in piece) + 1)
            ppw = int(max(c for _, c in piece) + 1)
            vr_off = int(vr1 - pph + 1) if vhs == "T" else int(vr0)
            vc_off = int(vc1 - ppw + 1) if vvs == "L" else int(vc0)
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    return None
                if (int(r), int(c)) in all_inp_fg:
                    return None
            vanish_fg |= vpts

    combined = set(all_inp_fg) | set(vanish_fg)
    if not bool(is_contiguous(_mask_from_pts(H, W, combined))):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for arrow in arrows:
        for r, c in set(arrow["cells"]):
            inp[int(r), int(c)] = int(ctrl_color)

    if _count_nonlong_arrows(arrows, H, W) < 1:
        return None

    combo = main(inp)
    farthest = _solve(inp, use_distance_stop=False)
    if np.array_equal(combo, farthest):
        return None
    if int(np.count_nonzero(combo != farthest)) <= 0:
        return None

    fg_all = np.array(list(combined), dtype=int)
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / float(final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    return inp


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))
    arrows = _find_arrows(g, ctrl)

    nonlong_controls = int(_count_nonlong_arrows(arrows, H, W))
    active_lines = 0
    ray_shortening = 0
    for arrow in arrows:
        edge = str(arrow["edge"])
        launch = tuple(int(x) for x in arrow["launch"])
        hits = _ordered_hits(g, fg, edge, launch)
        if hits:
            active_lines += 1
            short_stop = int(_select_stop(hits, edge, launch, H, W, use_distance_stop=True))
            long_stop = int(_select_stop(hits, edge, launch, H, W, use_distance_stop=False))
            ray_shortening += abs(int(long_stop) - int(short_stop))

    combo = main(g)
    farthest = _solve(g, use_distance_stop=False)
    return {
        "nonlong_controls": int(nonlong_controls),
        "active_lines": int(active_lines),
        "ray_shortening": int(ray_shortening),
        "ray_diff": int(np.count_nonzero(combo != farthest)),
    }


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
