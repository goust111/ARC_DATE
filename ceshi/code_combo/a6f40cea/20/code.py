# Combo variant of a6f40cea (index 20)
# combo_spec: D12/2+D14/1
# Name: 假源过滤距离排序
#
# Design analysis
# a) Concrete behavior:
#    - Base task: every ring-touching outer component is treated as a real source and each one is
#      painted independently into the framed interior.
#    - Combo task: first discard fake sources, then sort the remaining real sources by visible
#      distance to the frame center, then paint them in that order so later/farther paths can cover
#      earlier/nearer ones.
#    - Small example: two top components can both project onto the same interior row. In the base
#      task either scan order or independence would make order irrelevant. Here the thin top-left and
#      top-right sources are both valid, but the one whose outer elbow protrudes farther from the
#      frame center is painted later and wins on overlap. A stamped source with the same ring-touch
#      pattern is ignored entirely.
# b) Complexity calibration:
#    - Only one fake marker is introduced: a filled 2x2 elbow stamp. This keeps the filter stage
#      binary and geometric.
#    - Only one ordering cue is introduced: distance of the visible outer source from the frame
#      center. Generator spaces these distances apart to avoid ties.
#    - The number of real sources stays fixed at two, so overlap ordering is visible rather than noisy.
# c) Implicit-assumption review:
#    - Original assumption "every ring-touching component is actionable" is false now; components must
#      be classified before endpoint pairing.
#    - Original assumption "components can be solved independently" is false now; draw order matters
#      because later paths may overwrite earlier ones.
#    - Original generator assumption "inner paths should avoid overlap" is also false now; the combo
#      must deliberately create overlaps or the distance stage becomes inert.
#    - The alternating-color endpoint logic still holds after filtering/sorting because real sources
#      keep the original perimeter coloring semantics.
# d) Placement legality:
#    - Outer components remain 4-disconnected from one another and from the frame.
#    - Fake sources are ordinary L-sources plus one extra stamp cell that completes a 2x2 block.
#    - Real interior paths are allowed to overlap; only outside placement must remain collision-free.
# e) Visual observability:
#    - Fake vs real is visible from local geometry alone: thin corner trace versus stamped corner.
#    - Near/far order is visible because components protrude by different offsets from the frame and
#      the generator rejects distance ties.
#    - Samples are accepted only if sorted painting differs from unsorted painting and if fake-source
#      filtering changes the result, so both fused effects are present.
# f) Visual cleanliness:
#    - Real overlaps are concentrated on one shared row or column instead of many chaotic crossings.
#    - The fake stamp is local and does not add a second annotation layer elsewhere in the board.
# g) Inferability of changes:
#    - No color-id lookup is introduced. Colors still only indicate path paint / alternation.
#    - The two new decisions are both tied to visible structure: "has a 2x2 stamp?" and "which
#      source sits farther from the frame center?".

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill == int(bc):
                continue
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _has_full_2x2(mask: np.ndarray) -> bool:
    """Return True iff the occupied cells contain a complete 2x2 block."""
    coords = np.argwhere(mask)
    if len(coords) < 4:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    local = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return bool(np.any(local[:-1, :-1] & local[1:, :-1] & local[:-1, 1:] & local[1:, 1:]))


def _ring_cells(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> list[tuple[int, int]]:
    ring: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                p = (int(rr), int(cc))
                if p not in seen:
                    seen.add(p)
                    ring.append(p)
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                p = (int(rr), int(cc))
                if p not in seen:
                    seen.add(p)
                    ring.append(p)
    return ring


def _distance_key_from_coords(
    coords: list[tuple[int, int]] | np.ndarray,
    frame: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    """Sort by the farthest visible source pixel from the frame center."""
    r0, c0, r1, c1 = map(int, frame)
    center_r2 = int(r0 + r1)
    center_c2 = int(c0 + c1)
    pts = [(int(r), int(c)) for r, c in coords]
    max_d2 = max((2 * r - center_r2) ** 2 + (2 * c - center_c2) ** 2 for r, c in pts)
    min_r = min(r for r, _c in pts)
    min_c = min(c for _r, c in pts)
    return int(max_d2), int(min_r), int(min_c)


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
        return
    a, b = colors
    vals = [a if i % 2 == 0 else b for i in range(len(path))]
    grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != bg:
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _build_manhattan(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        step = 1 if cb >= ca else -1
        return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    if ca == cb:
        step = 1 if rb >= ra else -1
        return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
    if corner_a:
        step = 1 if rb >= ra else -1
        path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
        step = 1 if cb >= ca else -1
        path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb) + step, step)]
        return path
    step = 1 if cb >= ca else -1
    path = [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    step = 1 if rb >= ra else -1
    path += [(int(r), int(cb)) for r in range(int(ra) + step, int(rb) + step, step)]
    return path


def _inside_corner(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
    outH: int,
    outW: int,
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if corner_a else (ra, cb)
    rr, cc = corner
    return (0 < rr < outH - 1) and (0 < cc < outW - 1)


def _choose_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    outH: int,
    outW: int,
) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    if _inside_corner(p1, p2, False, outH, outW) and not _inside_corner(p1, p2, True, outH, outW):
        return path_b
    return path_a


def _farthest_pair(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    best = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            if best is None or d > best[0]:
                best = (d, coords[i], coords[j])
    if best is None:
        return None
    _d, a, b = best
    return a, b


def _solve(
    grid: np.ndarray,
    *,
    filter_fake: bool,
    sort_by_distance: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)
    ring = _ring_cells(r0, c0, r1, c1, H, W)

    records: list[dict[str, object]] = []
    for cid in range(1, int(cn) + 1):
        cmask = clab == int(cid)
        coords_arr = np.argwhere(cmask)
        if len(coords_arr) == 0:
            continue
        ring_pts: list[tuple[int, int, int]] = []
        for rr, cc in ring:
            if int(clab[rr, cc]) != int(cid):
                continue
            mr, mc = map_to_out(rr, cc)
            ring_pts.append((int(mr), int(mc), int(g[rr, cc])))
        if not ring_pts:
            continue
        fake = _has_full_2x2(cmask)
        if filter_fake and fake:
            continue
        cols = sorted(set(map(int, g[cmask].flatten())))
        dist_key = _distance_key_from_coords(coords_arr.tolist(), (r0, c0, r1, c1))
        records.append(
            {
                "cid": int(cid),
                "fake": bool(fake),
                "cols": cols,
                "pts": ring_pts,
                "dist_key": dist_key,
            }
        )

    if sort_by_distance:
        records.sort(key=lambda rec: tuple(rec["dist_key"]))  # near first, far last
    else:
        records.sort(key=lambda rec: int(rec["cid"]))

    for rec in records:
        pts = list(rec["pts"])
        cols = list(rec["cols"])
        if len(pts) < 2 or len(cols) not in (1, 2):
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1
        coords = list(hist.keys())
        pair = _farthest_pair(coords)
        if pair is None:
            continue
        a, b = pair
        path = _choose_path(a, b, outH, outW)

        if len(cols) == 1:
            for rr, cc in path:
                out[int(rr), int(cc)] = int(cols[0])
            continue

        ca, cb = map(int, cols)
        ha = int(hist[a].most_common(1)[0][0])
        hb = int(hist[b].most_common(1)[0][0])
        chosen = None
        for path_pts in (path, list(reversed(path))):
            for start_color in (ca, cb):
                end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb if start_color == ca else ca)
                if int(start_color) != int(ha):
                    continue
                if int(end_color) != int(hb):
                    continue
                chosen = (path_pts, int(start_color))
                break
            if chosen is not None:
                break
        assert chosen is not None
        path_pts, start_color = chosen
        cur = int(start_color)
        for rr, cc in path_pts:
            out[int(rr), int(cc)] = int(cur)
            cur = int(cb) if cur == int(ca) else int(ca)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the frame interior from outer traces after filtering decoys and resolving draw order.

    Input structure:
    - One closed 1px rectangular frame with a uniform fill.
    - Colored outer components touch the frame ring at two points.
    - Thin open-rectangle traces are genuine sources; traces with a filled 2x2 elbow stamp are fake.

    Rule:
    - Crop out the frame interior and start from the fill color.
    - Discard stamped fake sources.
    - Sort the remaining genuine sources from nearer to farther relative to the frame center.
    - Project each genuine source inward and draw its monochrome or alternating Manhattan path; later,
      farther sources may overwrite cells painted by earlier, nearer ones.

    Key insight:
    - The puzzle is no longer solved by treating every touching component independently. The correct
      reasoning chain is geometric filtering first, then center-distance ordering, then path drawing.
    """
    return _solve(grid, filter_fake=True, sort_by_distance=True)


_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.55
_INTERIOR_H_RANGE = (5, 12)
_INTERIOR_W_RANGE = (7, 13)
_GRID_SIZE_RANGE = (22, 30)
_GRID_INTERIOR_PAD = 14
_FRAME_MARGIN = 6
_FRAME_MARGIN_FAR = 6
_OUTER_DEPTH_VALUES = (2, 3, 4, 5, 6)
_FAKE_SOURCE_COUNT_RANGE = (1, 1)
_DUAL_COLOR_PROB = 0.35
_PLACEMENT_TRIES = 80
_GEN_RETRIES = 500


def _l_specs_from_elbow(
    kind: str,
    er: int,
    ec: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    depth: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + ec)
    ring_b = (r0 + 1 + er, c0 - 1 if is_left else c1 + 1)
    outer_r = (r0 - depth) if is_top else (r1 + depth)
    outer_c = (c0 - depth) if is_left else (c1 + depth)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_vpath(edge_r, er, ec), _hpath(er, ec, edge_c))
    outer = _join(
        _vpath(ring_a[0], outer_r, ring_a[1]),
        _hpath(outer_r, ring_a[1], outer_c),
        _vpath(outer_r, ring_b[0], outer_c),
        _hpath(ring_b[0], outer_c, ring_b[1]),
    )
    return inner, outer


def _fake_l_specs_from_elbow(
    kind: str,
    er: int,
    ec: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    depth: int,
) -> list[tuple[int, int]]:
    _inner, outer = _l_specs_from_elbow(kind, er, ec, H, W, r0, c0, r1, c1, depth)
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    outer_r = (r0 - depth) if is_top else (r1 + depth)
    outer_c = (c0 - depth) if is_left else (c1 + depth)
    stamp = (outer_r + (1 if is_top else -1), outer_c + (1 if is_left else -1))
    return outer + [stamp]


def _sample_corner_elbow(kind: str, H: int, W: int) -> tuple[int, int]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    er = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
    ec = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)
    return int(er), int(ec)


def _choose_color_tuple(palette: list[int], idx: int) -> tuple[tuple[int, ...], int]:
    if idx + 1 < len(palette) and random.random() < _DUAL_COLOR_PROB:
        return (palette[idx], palette[idx + 1]), idx + 2
    return (palette[idx],), idx + 1


def _paint_expected(
    H: int,
    W: int,
    fill: int,
    real_specs: list[dict[str, object]],
) -> np.ndarray:
    out = np.full((H, W), int(fill), dtype=int)
    for spec in sorted(real_specs, key=lambda rec: tuple(rec["dist_key"])):
        _paint_path(out, list(spec["inner"]), tuple(spec["colors"]))
    return out


def _build_overlap_specs(
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[tuple[str, int, int]]:
    family = random.choice(["main_diag", "anti_diag"])
    col = random.randint(max(1, W // 3), min(W - 2, max(1, (2 * W) // 3)))

    if family == "main_diag":
        r_tl = random.randint(max(1, H // 2), H - 2)
        r_br = random.randint(1, r_tl)
        return [("tl", r_tl, col), ("br", r_br, col)]

    r_tr = random.randint(max(1, H // 2), H - 2)
    r_bl = random.randint(1, r_tr)
    return [("tr", r_tr, col), ("bl", r_bl, col)]


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    sorted_out = _solve(grid, filter_fake=True, sort_by_distance=True)
    unsorted_out = _solve(grid, filter_fake=True, sort_by_distance=False)
    no_filter_out = _solve(grid, filter_fake=False, sort_by_distance=True)

    bg = int(get_bg(grid))
    found = _find_framed_rect_uniform_fill(np.array(grid, dtype=int), bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    comp_mask = (grid != int(bg)) & (grid != int(fill)) & (grid != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)
    real_distances: list[int] = []
    fake_count = 0
    ring = _ring_cells(r0, c0, r1, c1, grid.shape[0], grid.shape[1])
    for cid in range(1, int(cn) + 1):
        cmask = clab == int(cid)
        if not any(int(clab[rr, cc]) == int(cid) for rr, cc in ring):
            continue
        if _has_full_2x2(cmask):
            fake_count += 1
            continue
        real_distances.append(int(_distance_key_from_coords(np.argwhere(cmask).tolist(), (r0, c0, r1, c1))[0]))

    return {
        "changed_vs_unsorted": int(np.count_nonzero(sorted_out != unsorted_out)),
        "changed_vs_unfiltered": int(np.count_nonzero(sorted_out != no_filter_out)),
        "fake_count": int(fake_count),
        "real_count": int(len(real_distances)),
        "unique_real_distances": int(len(real_distances) == len(set(real_distances))),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    max_depth = max(_OUTER_DEPTH_VALUES)
    r0_lo = max(_FRAME_MARGIN, max_depth)
    c0_lo = max(_FRAME_MARGIN, max_depth)
    r0_hi = rows - H - 2 - max(_FRAME_MARGIN_FAR, max_depth)
    c0_hi = cols - W - 2 - max(_FRAME_MARGIN_FAR, max_depth)
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)
    r1, c1 = r0 + H + 1, c0 + W + 1

    base_grid = np.full((rows, cols), bg, dtype=int)
    base_grid[r0, c0 : c1 + 1] = bc
    base_grid[r1, c0 : c1 + 1] = bc
    base_grid[r0 : r1 + 1, c0] = bc
    base_grid[r0 : r1 + 1, c1] = bc
    base_grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)

    grid: np.ndarray | None = None
    comp_mask: np.ndarray | None = None
    real_specs: list[dict[str, object]] | None = None
    for _layout_try in range(_PLACEMENT_TRIES):
        grid_try = np.array(base_grid, copy=True)
        comp_mask_try = np.zeros_like(grid_try, dtype=bool)
        overlap_specs = _build_overlap_specs(H, W, r0, c0, r1, c1)
        if len(overlap_specs) < 2:
            continue
        real_specs_plan = overlap_specs[:2]
        far_depth = random.choice((5, 6))
        near_depth = random.choice((2, 3))
        depths = [far_depth, near_depth]
        real_specs_try: list[dict[str, object]] = []
        color_idx = 0
        ok = True

        for (kind, er, ec), depth in zip(real_specs_plan, depths):
            if color_idx >= len(palette):
                ok = False
                break
            colors, color_idx = _choose_color_tuple(palette, color_idx)
            inner, outer = _l_specs_from_elbow(kind, er, ec, H, W, r0, c0, r1, c1, int(depth))
            if not _place_component(grid_try, comp_mask_try, outer, colors, bg):
                ok = False
                break
            real_specs_try.append(
                {
                    "kind": kind,
                    "inner": inner,
                    "outer": outer,
                    "colors": colors,
                    "dist_key": _distance_key_from_coords(outer, (r0, c0, r1, c1)),
                }
            )

        if not ok:
            continue

        dist_values = [int(spec["dist_key"][0]) for spec in real_specs_try]
        if len(dist_values) != len(set(dist_values)):
            continue
        grid = grid_try
        comp_mask = comp_mask_try
        real_specs = real_specs_try
        break

    if grid is None or comp_mask is None or real_specs is None:
        return None

    target_fake = random.randint(*_FAKE_SOURCE_COUNT_RANGE)
    fake_palette = palette[:] if palette else [x for x in range(10) if x not in (bg, bc, fill)]
    placed_fake = 0
    real_kinds = {str(spec["kind"]) for spec in real_specs}
    if real_kinds == {"tl", "br"}:
        fake_kinds = ["tr", "bl"]
    elif real_kinds == {"tr", "bl"}:
        fake_kinds = ["tl", "br"]
    else:
        fake_kinds = ["tl", "tr", "bl", "br"]
    for _ in range(target_fake):
        placed = False
        for _try in range(_PLACEMENT_TRIES):
            kind = random.choice(fake_kinds)
            er, ec = _sample_corner_elbow(kind, H, W)
            depth = random.choice((3, 4, 5))
            outer = _fake_l_specs_from_elbow(kind, er, ec, H, W, r0, c0, r1, c1, int(depth))
            color = random.choice(fake_palette)
            if _place_component(grid, comp_mask, outer, (int(color),), bg):
                placed_fake += 1
                placed = True
                break
        if not placed:
            return None
    if placed_fake != target_fake:
        return None

    expected = _paint_expected(H, W, fill, real_specs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    metrics = _behavior_metrics(grid)
    if metrics["fake_count"] < 1:
        return None
    if metrics["real_count"] < 2 or not metrics["unique_real_distances"]:
        return None
    if metrics["changed_vs_unsorted"] <= 0:
        return None
    if metrics["changed_vs_unfiltered"] <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
