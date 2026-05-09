# Combo variant of a6f40cea (index 38)
# combo_spec: D4/0+D6/1+D13/0
# Name: U形矩形填充跨组件连接

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for combo 38
# --------------------------------
# a) concretization and behavior trace
#    - Keep one closed 1px rectangular frame with uniform fill, exactly as in the source task.
#    - Always place four meaningful outside components:
#      1. one top-left L component,
#      2. one bottom-right L component,
#      3. one U component on one side of the frame,
#      4. its same-color opposite-side U partner.
#    - The L pair shares one visible palette and is solved across components:
#      top touch of the top-left component connects to the right touch of the bottom-right component,
#      and left touch connects to bottom touch. This is the D4 change.
#    - The D6 change is applied after that geometric pairing: each cross-component L no longer stays
#      a 1px line. Instead it fills the corner rectangle fenced by that L path and the crop boundary.
#      So the paired Ls create two filled corner blocks: top-right and bottom-left.
#    - The D13 U pair no longer means two stripes. Its four touchpoints define one rectangle spanning
#      between the opposite sides. Because D6 is also active, that rectangle is flood-filled, not just outlined.
#    - Concrete example with interior H=10, W=12:
#      top-left L touches top@(0,8), left@(7,0); bottom-right L touches bottom@(9,2), right@(3,11).
#      Original source task would locally repair top-left and bottom-right corners.
#      This combo instead fills rows 0..3, cols 8..11 and rows 7..9, cols 0..2.
#      If a top/bottom U pair touches columns 4 and 6, the combo also fills rows 0..9, cols 4..6.
#      The visible output therefore contains three disjoint filled regions: two cross-paired corner blocks
#      and one U-derived central rectangle.
# b) complexity calibration
#    - The generator always includes both semantic mechanisms at once, so the combo cannot collapse to
#      "just D4" or "just D13". This makes the fusion visible from 2-4 samples.
#    - The L pair is restricted to one opposite-corner palette group, and the U pair uses a different mono color.
#      That keeps grouping unambiguous and avoids hidden sorting by color id.
#    - The two L-filled regions are forced to be disjoint by choosing right_row < left_row and bottom_col < top_col.
#      The U rectangle is forced into the gap between them, so overlap ambiguity is removed.
# c) implicit assumption audit against the original solver
#    - Original assumption: each component can be solved from its own touchpoints. Broken by D4.
#      New solver groups by shared palette and pairs two different L components.
#    - Original assumption: monochrome opposite-side touches can be deferred into stripes. Broken by D13.
#      New solver detects same-color U pairs and converts them into one filled rectangle.
#    - Original assumption: path painting is enough. Broken by D6.
#      New solver reconstructs filled rectangles, and the generator reserves whole output areas instead of only lines.
#    - New assumption introduced here: the only repeated mono palette on the ring is the U pair, and the only repeated
#      1- or 2-color L palette is the cross-component pair. The generator enforces palette uniqueness.
# d) placement legality
#    - Outside paths are still 4-connected, non-overlapping, in-bounds, and non-4-adjacent.
#    - Output regions are selected first and checked with a bbox occupancy mask before any outside component is placed.
#    - Candidate searches over outside offsets are used so the four components fit simultaneously around one frame.
# e) visual observability
#    - Cross-component pairing is directly visible because the outside corner shapes do not complete their own corners:
#      the filled blocks appear in the opposite corners instead.
#    - U behavior is directly visible because the U pair produces a solid rectangle/band in the middle gap,
#      not two separate stripes and not merely an outline.
# f) visual cleanliness
#    - The three filled regions are disjoint by construction.
#    - Distractors never touch the frame ring, so they cannot be mistaken for semantic components.
# g) color inference
#    - Monochrome U rectangles copy one visible color inward.
#    - The L pair may be monochrome or 2-color alternating. In the 2-color case, each filled block is a checkerboard
#      whose phase is fixed by the endpoint colors seen on the frame ring; no hidden numeric color ordering is used.


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Return the largest 1px complete rectangle with uniform fill."""
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


def _other_color(colors: tuple[int, int], color: int) -> int:
    return int(colors[1]) if int(color) == int(colors[0]) else int(colors[0])


def _paint_path_with_start(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    *,
    start: int | None = None,
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    palette = (int(colors[0]), int(colors[1]))
    cur = int(palette[0] if start is None else start)
    vals = []
    for _ in path:
        vals.append(int(cur))
        cur = _other_color(palette, cur)
    grid[rs, cs] = vals


def _paint_checker_rect(
    out: np.ndarray,
    bounds: tuple[int, int, int, int],
    colors: tuple[int, int],
    corner: tuple[int, int],
    corner_color: int,
) -> None:
    r_lo, r_hi, c_lo, c_hi = bounds
    other = _other_color(colors, int(corner_color))
    rs = np.arange(r_lo, r_hi + 1)[:, None]
    cs = np.arange(c_lo, c_hi + 1)[None, :]
    parity = (np.abs(rs - int(corner[0])) + np.abs(cs - int(corner[1]))) % 2
    block = np.where(parity == 0, int(corner_color), int(other))
    out[r_lo : r_hi + 1, c_lo : c_hi + 1] = block


def _checker_color_at(
    pt: tuple[int, int],
    colors: tuple[int, int],
    corner: tuple[int, int],
    corner_color: int,
) -> int:
    d = abs(int(pt[0]) - int(corner[0])) + abs(int(pt[1]) - int(corner[1]))
    if d % 2 == 0:
        return int(corner_color)
    return _other_color(colors, int(corner_color))


def _paint_solid_rect(out: np.ndarray, bounds: tuple[int, int, int, int], color: int) -> None:
    r_lo, r_hi, c_lo, c_hi = bounds
    out[r_lo : r_hi + 1, c_lo : c_hi + 1] = int(color)


def _bbox_is_free(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> bool:
    r_lo, r_hi, c_lo, c_hi = bounds
    return not mask[r_lo : r_hi + 1, c_lo : c_hi + 1].any()


def _mark_bbox(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    r_lo, r_hi, c_lo, c_hi = bounds
    mask[r_lo : r_hi + 1, c_lo : c_hi + 1] = True


def _side_of_point(r: int, c: int, outH: int, outW: int) -> str:
    sides = []
    if int(r) == 0:
        sides.append("top")
    if int(r) == int(outH - 1):
        sides.append("bottom")
    if int(c) == 0:
        sides.append("left")
    if int(c) == int(outW - 1):
        sides.append("right")
    assert len(sides) == 1
    return sides[0]


def _ring_coords(r0: int, c0: int, r1: int, c1: int, H: int, W: int) -> set[tuple[int, int]]:
    ring: set[tuple[int, int]] = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))
    return ring


def _extract_component_infos(
    g: np.ndarray,
    *,
    bg: int,
    bc: int,
    fill: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    outH: int,
    outW: int,
) -> list[dict]:
    H, W = g.shape
    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)
    ring = _ring_coords(r0, c0, r1, c1, H, W)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((mr, mc, int(g[rr, cc])))

    infos: list[dict] = []
    for cid, pts in comp_pts.items():
        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1

        side_pts: dict[str, list[dict]] = defaultdict(list)
        for coord, ctr in hist.items():
            side = _side_of_point(int(coord[0]), int(coord[1]), outH, outW)
            side_pts[side].append({"coord": (int(coord[0]), int(coord[1])), "color": int(ctr.most_common(1)[0][0])})

        for side in side_pts:
            side_pts[side].sort(key=lambda item: item["coord"])

        palette = tuple(sorted(set(map(int, g[clab == int(cid)].flatten()))))
        if sum(len(v) for v in side_pts.values()) != 2:
            continue
        if len(side_pts) == 1 and sorted(len(v) for v in side_pts.values()) == [2]:
            kind = "U"
        elif len(side_pts) == 2 and sorted(len(v) for v in side_pts.values()) == [1, 1]:
            kind = "L"
        else:
            continue

        touches = {}
        if kind == "L":
            touches = {side: items[0] for side, items in side_pts.items()}

        infos.append(
            {
                "cid": int(cid),
                "kind": kind,
                "palette": palette,
                "side_pts": dict(side_pts),
                "touches": touches,
            }
        )
    return infos


def _find_u_pair(groups: dict[tuple[int, ...], list[dict]]) -> tuple[int, dict, dict] | None:
    for palette, infos in groups.items():
        if len(palette) != 1 or len(infos) != 2:
            continue
        if not all(info["kind"] == "U" for info in infos):
            continue
        side_a = next(iter(infos[0]["side_pts"]))
        side_b = next(iter(infos[1]["side_pts"]))
        if {side_a, side_b} not in ({"top", "bottom"}, {"left", "right"}):
            continue
        return int(palette[0]), infos[0], infos[1]
    return None


def _find_l_pair(groups: dict[tuple[int, ...], list[dict]]) -> tuple[tuple[int, ...], dict, dict] | None:
    for palette, infos in groups.items():
        if len(infos) != 2:
            continue
        if not all(info["kind"] == "L" for info in infos):
            continue
        tl = None
        br = None
        for info in infos:
            sides = set(info["touches"])
            if sides == {"top", "left"}:
                tl = info
            elif sides == {"bottom", "right"}:
                br = info
        if tl is not None and br is not None:
            return palette, tl, br
    return None


def _paint_u_region(
    out: np.ndarray,
    color: int,
    info_a: dict,
    info_b: dict,
    outH: int,
    outW: int,
) -> tuple[int, int, int, int]:
    side_a = next(iter(info_a["side_pts"]))
    side_b = next(iter(info_b["side_pts"]))
    if {side_a, side_b} == {"top", "bottom"}:
        cols = sorted(item["coord"][1] for info in (info_a, info_b) for item in next(iter(info["side_pts"].values())))
        bounds = (0, outH - 1, int(cols[0]), int(cols[-1]))
    else:
        rows = sorted(item["coord"][0] for info in (info_a, info_b) for item in next(iter(info["side_pts"].values())))
        bounds = (int(rows[0]), outH - 1 if False else int(rows[-1]), 0, outW - 1)
        bounds = (int(rows[0]), int(rows[-1]), 0, outW - 1)
    _paint_solid_rect(out, bounds, int(color))
    return bounds


def _paint_cross_l_regions(
    out: np.ndarray,
    palette: tuple[int, ...],
    tl: dict,
    br: dict,
    outH: int,
    outW: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    top = tl["touches"]["top"]
    left = tl["touches"]["left"]
    bottom = br["touches"]["bottom"]
    right = br["touches"]["right"]

    bounds_tr = (0, int(right["coord"][0]), int(top["coord"][1]), outW - 1)
    bounds_bl = (int(left["coord"][0]), outH - 1, 0, int(bottom["coord"][1]))
    assert bounds_tr[1] < bounds_bl[0]
    assert bounds_bl[3] < bounds_tr[2]

    if len(palette) == 1:
        color = int(palette[0])
        _paint_solid_rect(out, bounds_tr, color)
        _paint_solid_rect(out, bounds_bl, color)
        return bounds_tr, bounds_bl

    colors = (int(palette[0]), int(palette[1]))
    corner_tr = (0, outW - 1)
    corner_bl = (outH - 1, 0)

    seed_tr = None
    for candidate in colors:
        if _checker_color_at(tuple(top["coord"]), colors, corner_tr, int(candidate)) != int(top["color"]):
            continue
        if _checker_color_at(tuple(right["coord"]), colors, corner_tr, int(candidate)) != int(right["color"]):
            continue
        seed_tr = int(candidate)
        break
    assert seed_tr is not None

    seed_bl = None
    for candidate in colors:
        if _checker_color_at(tuple(left["coord"]), colors, corner_bl, int(candidate)) != int(left["color"]):
            continue
        if _checker_color_at(tuple(bottom["coord"]), colors, corner_bl, int(candidate)) != int(bottom["color"]):
            continue
        seed_bl = int(candidate)
        break
    assert seed_bl is not None

    _paint_checker_rect(out, bounds_tr, colors, corner_tr, seed_tr)
    _paint_checker_rect(out, bounds_bl, colors, corner_bl, seed_bl)
    return bounds_tr, bounds_bl


def _baseline_local_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)

    infos = _extract_component_infos(
        g,
        bg=int(bg),
        bc=int(bc),
        fill=int(fill),
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
        outH=outH,
        outW=outW,
    )
    groups: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for info in infos:
        groups[info["palette"]].append(info)

    u_found = _find_u_pair(groups)
    assert u_found is not None
    u_color, u_a, u_b = u_found
    _paint_u_region(out, int(u_color), u_a, u_b, outH, outW)

    l_found = _find_l_pair(groups)
    assert l_found is not None
    palette, tl, br = l_found
    top = tl["touches"]["top"]
    left = tl["touches"]["left"]
    bottom = br["touches"]["bottom"]
    right = br["touches"]["right"]

    bounds_tl = (0, int(left["coord"][0]), 0, int(top["coord"][1]))
    bounds_br = (int(right["coord"][0]), outH - 1, int(bottom["coord"][1]), outW - 1)

    if len(palette) == 1:
        color = int(palette[0])
        _paint_solid_rect(out, bounds_tl, color)
        _paint_solid_rect(out, bounds_br, color)
        return out

    colors = (int(palette[0]), int(palette[1]))
    corner_tl = (0, 0)
    corner_br = (outH - 1, outW - 1)

    seed_tl = None
    for candidate in colors:
        if _checker_color_at(tuple(top["coord"]), colors, corner_tl, int(candidate)) != int(top["color"]):
            continue
        if _checker_color_at(tuple(left["coord"]), colors, corner_tl, int(candidate)) != int(left["color"]):
            continue
        seed_tl = int(candidate)
        break
    assert seed_tl is not None

    seed_br = None
    for candidate in colors:
        if _checker_color_at(tuple(bottom["coord"]), colors, corner_br, int(candidate)) != int(bottom["color"]):
            continue
        if _checker_color_at(tuple(right["coord"]), colors, corner_br, int(candidate)) != int(right["color"]):
            continue
        seed_br = int(candidate)
        break
    assert seed_br is not None

    _paint_checker_rect(out, bounds_tl, colors, corner_tl, seed_tl)
    _paint_checker_rect(out, bounds_br, colors, corner_br, seed_br)
    return out


def measure_fusion_behavior(grid: np.ndarray) -> dict[str, int | str]:
    """
    Return visible combo metrics for one generated sample.

    cross_diff:
      number of output pixels that differ from a local-completion baseline.
      Positive means the D4 cross-component pairing is materially active.
    u_interior:
      number of strictly interior pixels of the inferred U-rectangle that carry the U color.
      Positive means the U result is flood-filled, not merely an outline/stripe surrogate.
    """
    g = np.array(grid, dtype=int)
    out = main(g)
    base = _baseline_local_output(g)

    bg = int(get_bg(g))
    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    infos = _extract_component_infos(
        g,
        bg=int(bg),
        bc=int(bc),
        fill=int(fill),
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
        outH=outH,
        outW=outW,
    )
    groups: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for info in infos:
        groups[info["palette"]].append(info)
    u_found = _find_u_pair(groups)
    assert u_found is not None
    u_color, u_a, u_b = u_found
    side_a = next(iter(u_a["side_pts"]))
    side_b = next(iter(u_b["side_pts"]))
    if {side_a, side_b} == {"top", "bottom"}:
        cols = sorted(item["coord"][1] for info in (u_a, u_b) for item in next(iter(info["side_pts"].values())))
        interior = out[1 : outH - 1, int(cols[0]) + 1 : int(cols[-1])]
        orientation = "vertical"
    else:
        rows = sorted(item["coord"][0] for info in (u_a, u_b) for item in next(iter(info["side_pts"].values())))
        interior = out[int(rows[0]) + 1 : int(rows[-1]), 1 : outW - 1]
        orientation = "horizontal"
    return {
        "cross_diff": int(np.count_nonzero(out != base)),
        "u_interior": int(np.count_nonzero(interior == int(u_color))),
        "u_orientation": orientation,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct filled interior regions by jointly interpreting opposite-corner L pairs and opposite-side U pairs.

    Input structure:
    - One closed 1px rectangular frame with a uniform interior fill sits on a uniform background.
    - Outside the frame there are four meaningful ring-touching components: two corner L traces sharing a palette,
      plus two same-color U traces on opposite sides. Small distractor blobs may also exist away from the ring.

    Rule:
    - Crop the output to the frame interior and start from the fill color.
    - Group ring-touching components by their visible palette.
    - The two L traces are paired across components, not completed locally: top->right and left->bottom.
      Each such Manhattan connection is then interpreted with the flood-fill rule, producing two filled corner rectangles.
      Monochrome pairs fill solid blocks; 2-color pairs fill checkerboards whose phase is fixed by the ring endpoint colors.
    - The same-color opposite U pair defines one closed rectangle spanning between the two touched sides.
      That rectangle is flood-filled with the U color.

    Key insight:
    - The repeated palette tells which separate components belong together, while the touched sides tell whether the pair
      should become two cross-component corner fills (L case) or one filled spanning rectangle (U case).
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)

    infos = _extract_component_infos(
        g,
        bg=int(bg),
        bc=int(bc),
        fill=int(fill),
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
        outH=outH,
        outW=outW,
    )
    groups: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    for info in infos:
        groups[info["palette"]].append(info)

    u_found = _find_u_pair(groups)
    assert u_found is not None
    u_color, u_a, u_b = u_found
    _paint_u_region(out, int(u_color), u_a, u_b, outH, outW)

    l_found = _find_l_pair(groups)
    assert l_found is not None
    palette, tl, br = l_found
    _paint_cross_l_regions(out, palette, tl, br, outH, outW)
    return out


# --------------- generator (combo 38) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (9, 13)
_INTERIOR_W_RANGE = (10, 14)
_GRID_SIZE_RANGE = (24, 30)
_GRID_INTERIOR_PAD = 12
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 7
_OUTER_OFFSET_RANGE = (2, 4)
_L_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 800


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
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
    *,
    start: int | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path_with_start(grid, path, colors, start=start)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _component_outer(
    kind: str,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    inner_r: int,
    inner_c: int,
    off_r: int,
    off_c: int,
) -> tuple[list[tuple[int, int]], dict[str, tuple[int, int]]]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + inner_c)
    ring_b = (r0 + 1 + inner_r, c0 - 1 if is_left else c1 + 1)
    side_a = "top" if is_top else "bottom"
    side_b = "left" if is_left else "right"
    R = (r0 - int(off_r)) if is_top else (r1 + int(off_r))
    C = (c0 - int(off_c)) if is_left else (c1 + int(off_c))

    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return outer, {side_a: ring_a, side_b: ring_b}


def _candidate_l_components(
    kind: str,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    inner_r: int,
    inner_c: int,
) -> list[dict]:
    out = []
    lo, hi = _OUTER_OFFSET_RANGE
    for off_r in range(lo, hi + 1):
        for off_c in range(lo, hi + 1):
            path, touches = _component_outer(kind, r0, c0, r1, c1, inner_r, inner_c, off_r, off_c)
            if len(set(path)) != len(path):
                continue
            out.append(
                {
                    "path": path,
                    "touches": touches,
                    "same_endpoints": (len(path) % 2 == 1),
                }
            )
    return out


def _candidate_u_pairs(
    orientation: str,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    a: int,
    b: int,
) -> list[dict]:
    out = []
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        a_top = (r0 - 1, c0 + 1 + a)
        b_top = (r0 - 1, c0 + 1 + b)
        a_bot = (r1 + 1, c0 + 1 + a)
        b_bot = (r1 + 1, c0 + 1 + b)
        for top_off in range(lo, hi + 1):
            for bot_off in range(lo, hi + 1):
                top_r = r0 - int(top_off)
                bot_r = r1 + int(bot_off)
                outer_a = _join(
                    _vpath(a_top[0], top_r, a_top[1]),
                    _hpath(top_r, a_top[1], b_top[1]),
                    _vpath(top_r, b_top[0], b_top[1]),
                )
                outer_b = _join(
                    _vpath(a_bot[0], bot_r, a_bot[1]),
                    _hpath(bot_r, a_bot[1], b_bot[1]),
                    _vpath(bot_r, b_bot[0], b_bot[1]),
                )
                if len(set(outer_a)) != len(outer_a) or len(set(outer_b)) != len(outer_b):
                    continue
                out.append({"outer_a": outer_a, "outer_b": outer_b})
    else:
        a_left = (r0 + 1 + a, c0 - 1)
        b_left = (r0 + 1 + b, c0 - 1)
        a_right = (r0 + 1 + a, c1 + 1)
        b_right = (r0 + 1 + b, c1 + 1)
        for left_off in range(lo, hi + 1):
            for right_off in range(lo, hi + 1):
                left_c = c0 - int(left_off)
                right_c = c1 + int(right_off)
                outer_a = _join(
                    _hpath(a_left[0], a_left[1], left_c),
                    _vpath(a_left[0], b_left[0], left_c),
                    _hpath(b_left[0], left_c, b_left[1]),
                )
                outer_b = _join(
                    _hpath(a_right[0], a_right[1], right_c),
                    _vpath(a_right[0], b_right[0], right_c),
                    _hpath(b_right[0], right_c, b_right[1]),
                )
                if len(set(outer_a)) != len(outer_a) or len(set(outer_b)) != len(outer_b):
                    continue
                out.append({"outer_a": outer_a, "outer_b": outer_b})
    return out


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r = random.randint(1, H - 2)
        c = random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _paths_fit(
    grid: np.ndarray,
    bg: int,
    base_mask: np.ndarray,
    paths: list[list[tuple[int, int]]],
) -> bool:
    mask = np.array(base_mask, copy=True)
    for path in paths:
        if not _can_place(grid, path, bg, mask):
            return False
        rs, cs = zip(*path)
        mask[rs, cs] = True
    return True


def _choose_geometry(H: int, W: int) -> tuple[int, int, int, int]:
    c_bottom = random.randint(1, max(1, W // 4))
    c_top = random.randint(c_bottom + 5, W - 2)
    r_right = random.randint(1, max(1, H // 4))
    r_left = random.randint(r_right + 5, H - 2)
    return c_bottom, c_top, r_right, r_left


def _paint_expected_combo(
    expected: np.ndarray,
    *,
    u_bounds: tuple[int, int, int, int],
    u_color: int,
    l_palette: tuple[int, ...],
    c_bottom: int,
    c_top: int,
    r_right: int,
    r_left: int,
    tr_seed: int | None = None,
    bl_seed: int | None = None,
) -> None:
    H, W = expected.shape
    _paint_solid_rect(expected, u_bounds, int(u_color))
    bounds_tr = (0, int(r_right), int(c_top), W - 1)
    bounds_bl = (int(r_left), H - 1, 0, int(c_bottom))
    if len(l_palette) == 1:
        color = int(l_palette[0])
        _paint_solid_rect(expected, bounds_tr, color)
        _paint_solid_rect(expected, bounds_bl, color)
        return
    colors = (int(l_palette[0]), int(l_palette[1]))
    assert tr_seed is not None and bl_seed is not None
    _paint_checker_rect(expected, bounds_tr, colors, (0, W - 1), int(tr_seed))
    _paint_checker_rect(expected, bounds_bl, colors, (H - 1, 0), int(bl_seed))


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])

    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    used_inner = np.zeros((H, W), dtype=bool)
    comp_mask = np.zeros_like(grid, dtype=bool)

    palette_pool = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette_pool)
    if len(palette_pool) < 3:
        return None

    c_bottom, c_top, r_right, r_left = _choose_geometry(H, W)
    bounds_tr = (0, r_right, c_top, W - 1)
    bounds_bl = (r_left, H - 1, 0, c_bottom)
    if not _bbox_is_free(used_inner, bounds_tr) or not _bbox_is_free(used_inner, bounds_bl):
        return None
    _mark_bbox(used_inner, bounds_tr)
    _mark_bbox(used_inner, bounds_bl)

    orientation_options = []
    if c_top - c_bottom >= 5:
        orientation_options.append("vertical")
    if r_left - r_right >= 5:
        orientation_options.append("horizontal")
    if not orientation_options:
        return None
    orientation = random.choice(orientation_options)
    if orientation == "vertical":
        u1 = random.randint(c_bottom + 1, c_top - 3)
        u2 = random.randint(u1 + 2, c_top - 1)
        u_bounds = (0, H - 1, u1, u2)
    else:
        u1 = random.randint(r_right + 1, r_left - 3)
        u2 = random.randint(u1 + 2, r_left - 1)
        u_bounds = (u1, u2, 0, W - 1)
    if not _bbox_is_free(used_inner, u_bounds):
        return None
    _mark_bbox(used_inner, u_bounds)

    u_color = int(palette_pool[0])
    next_idx = 1
    if random.random() < _L_DUAL_COLOR_PROB and next_idx + 1 < len(palette_pool):
        l_palette: tuple[int, ...] = (int(palette_pool[next_idx]), int(palette_pool[next_idx + 1]))
        next_idx += 2
    else:
        l_palette = (int(palette_pool[next_idx]),)
        next_idx += 1

    tl_opts = _candidate_l_components("tl", r0, c0, r1, c1, r_left, c_top)
    br_opts = _candidate_l_components("br", r0, c0, r1, c1, r_right, c_bottom)
    u_opts = _candidate_u_pairs(orientation, r0, c0, r1, c1, u1, u2)
    if not tl_opts or not br_opts or not u_opts:
        return None
    random.shuffle(tl_opts)
    random.shuffle(br_opts)
    random.shuffle(u_opts)

    chosen = None
    if len(l_palette) == 1:
        mono = int(l_palette[0])
        for u_opt in u_opts:
            for tl_opt in tl_opts:
                for br_opt in br_opts:
                    if not _paths_fit(grid, bg, comp_mask, [u_opt["outer_a"], u_opt["outer_b"], tl_opt["path"], br_opt["path"]]):
                        continue
                    chosen = {
                        "u": u_opt,
                        "tl": tl_opt,
                        "br": br_opt,
                        "tl_start": mono,
                        "br_start": mono,
                        "tr_seed": None,
                        "bl_seed": None,
                    }
                    break
                if chosen is not None:
                    break
            if chosen is not None:
                break
    else:
        colors = (int(l_palette[0]), int(l_palette[1]))
        top_pt = (0, c_top)
        left_pt = (r_left, 0)
        bottom_pt = (H - 1, c_bottom)
        right_pt = (r_right, W - 1)
        for tr_seed in colors:
            top_color = _checker_color_at(top_pt, colors, (0, W - 1), int(tr_seed))
            right_color = _checker_color_at(right_pt, colors, (0, W - 1), int(tr_seed))
            for bl_seed in colors:
                left_color = _checker_color_at(left_pt, colors, (H - 1, 0), int(bl_seed))
                bottom_color = _checker_color_at(bottom_pt, colors, (H - 1, 0), int(bl_seed))
                tl_need_same = int(top_color) == int(left_color)
                br_need_same = int(bottom_color) == int(right_color)
                tl_filtered = [opt for opt in tl_opts if bool(opt["same_endpoints"]) == bool(tl_need_same)]
                br_filtered = [opt for opt in br_opts if bool(opt["same_endpoints"]) == bool(br_need_same)]
                for u_opt in u_opts:
                    for tl_opt in tl_filtered:
                        for br_opt in br_filtered:
                            if not _paths_fit(grid, bg, comp_mask, [u_opt["outer_a"], u_opt["outer_b"], tl_opt["path"], br_opt["path"]]):
                                continue
                            chosen = {
                                "u": u_opt,
                                "tl": tl_opt,
                                "br": br_opt,
                                "tl_start": int(top_color),
                                "br_start": int(bottom_color),
                                "tr_seed": int(tr_seed),
                                "bl_seed": int(bl_seed),
                            }
                            break
                        if chosen is not None:
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                break
    if chosen is None:
        return None

    if not _place_component(grid, comp_mask, chosen["u"]["outer_a"], (u_color,), bg):
        return None
    if not _place_component(grid, comp_mask, chosen["u"]["outer_b"], (u_color,), bg):
        return None
    if not _place_component(grid, comp_mask, chosen["tl"]["path"], l_palette, bg, start=chosen["tl_start"]):
        return None
    if not _place_component(grid, comp_mask, chosen["br"]["path"], l_palette, bg, start=chosen["br_start"]):
        return None

    _paint_expected_combo(
        expected,
        u_bounds=u_bounds,
        u_color=u_color,
        l_palette=l_palette,
        c_bottom=c_bottom,
        c_top=c_top,
        r_right=r_right,
        r_left=r_left,
        tr_seed=chosen["tr_seed"],
        bl_seed=chosen["bl_seed"],
    )

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if next_idx >= len(palette_pool):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(palette_pool[next_idx]))
        next_idx += 1

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    metrics = measure_fusion_behavior(grid)
    if int(metrics["cross_diff"]) <= 0 or int(metrics["u_interior"]) <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
