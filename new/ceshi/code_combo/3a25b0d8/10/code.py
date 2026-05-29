# Combo variant of 3a25b0d8 (index 10)
# combo_spec: D2/3+D4/0
# Name: 多色渐变子精灵+颜色映射
#
# Design analysis
# =============================================================================
# a) Concrete fused behavior and execution trace
#    Base task:
#    - There are two large 8-connected outline objects.
#    - src contains one colored subsprite per cavity.
#    - dst contains matching empty holes.
#    - Solver matches cavities to holes by centroid row/col rank and fills each
#      dst hole with one scalar src color.
#
#    Fused variant:
#    - Keep the same src/dst geometry and centroid-rank matching.
#    - Change each src cavity from "single fill color" to a two-color local
#      pattern anchored to that cavity bbox. To keep the rule human-inferable,
#      one sample uses exactly one family globally:
#        checkerboard, horizontal stripes, or vertical stripes.
#    - Add a third object: a small 5-column legend card. Each legend row says
#      "source color -> mapped color".
#    - When copying a matched src cavity into dst, solver must preserve the full
#      local pattern structure, but recolor each of the two source colors through
#      the legend before painting dst.
#
#    Tiny example:
#    - Suppose the global pattern family is checkerboard.
#    - A matched src cavity shows:
#        2 4 2
#        4 2 4
#    - Legend rows contain 2->7 and 4->3.
#    - Original task would have output a flat 2 or 4 region.
#    - D2-only would copy the checkerboard as 2/4.
#    - This fused task outputs:
#        7 3 7
#        3 7 3
#      in the matched dst hole.
#
# b) Complexity calibration
#    - Pattern family count is intentionally small: one of three global families.
#      That is enough visual variety to make the D2 change obvious, but avoids
#      per-cavity free-form patterns that would be hard to infer from 2-4 demos.
#    - Each cavity always uses exactly two colors, and those colors are reused
#      across cavities. This gives repeated evidence for both the legend mapping
#      and the pattern-copy rule.
#    - The legend covers 4-6 colors and is a derangement, so every mapped output
#      visibly differs from direct-copy behavior.
#
# c) Implicit-assumption audit of the original main()
#    - Original "_extract_two_objects()" assumed only src and dst exist.
#      Broken by D4: the fused solver must scan all 8-connected components and
#      explicitly detect the legend instead of taking the two largest blindly.
#    - Original "_outline_col()" assumed the dominant non-bg color is the outline.
#      Broken by D2: multi-color patterned cavities can outnumber outline cells.
#      The fused solver reads outline from the component border instead.
#    - Original "_colored_components()" assumed one color per cavity and stored
#      just that scalar. Broken by D2: fused solver must infer bbox-anchored
#      pattern family plus the two phase colors.
#    - Original fill loop wrote "dest[rr, cc] = color". Broken by both changes:
#      fused logic computes a local pattern phase for each cell, then maps the
#      chosen source color through the legend before writing it.
#
# d) Placement legality
#    - Generator places src, dst, and legend independently with the existing
#      non-overlap helper, shared border, and padding constraints.
#    - The legend is much smaller than src/dst and cannot merge under
#      8-connectivity because placement uses the same transparent padding rules.
#
# e) Visual observability
#    - Humans can see three separate signals directly on the grid:
#      1. src cavities are not flat; they contain visible two-color patterns.
#      2. dst holes are empty but arranged like the src cavities.
#      3. the legend visibly pairs colors row-by-row.
#    - Because the output keeps the pattern geometry but changes both colors via
#      the legend, the fused rule is inferable from a few examples without using
#      any hidden numeric-color ordering.
#
# f) Visual cleanliness
#    - Pattern family is global per sample, not mixed freely per cavity.
#    - Cavity seed sizes avoid 1-row / 1-col shapes so both colors are visible.
#    - The legend is a compact framed card, visually distinct from the two main
#      outline objects.
#
# g) Color-change inferability
#    - Every legend source color is guaranteed to appear in src at least once.
#    - Each cavity uses only two colors, repeatedly, so the recoloring rule has
#      local structural support rather than looking like arbitrary per-pixel noise.
#
# h) Non-degeneration target
#    - A generated sample is considered healthy only if mapped output differs from
#      the "copy original cavity colors directly" counterfactual on some filled
#      cells. The post-edit validation script measures this explicitly.
#
# i) New solver assumptions introduced here, and generator guarantees
#    - Assumption: legend follows the fixed 5-column schema.
#      Guarantee: generator builds exactly that card.
#    - Assumption: every src cavity pattern is disambiguated by its visible cells.
#      Guarantee: generator only accepts cavity shapes whose occupied cells
#      distinguish the chosen family from the other candidate families.
#    - Assumption: every source color used in patterned cavities has a legend row.
#      Guarantee: generator enforces full legend coverage before oracle-checking.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


PATTERN_FAMILIES = ("checker", "hstripe", "vstripe")


def _components(mask: np.ndarray) -> list[dict]:
    lab, n = label_mask(mask, connectivity=8)
    comps = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        comps.append(
            {
                "cid": int(cid),
                "pts": pts,
                "size": int(len(pts)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    comps.sort(key=lambda d: int(d["size"]), reverse=True)
    return comps


def _extract_subgrid(grid: np.ndarray, pts: np.ndarray, bg: int) -> np.ndarray:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for rr, cc in pts:
        sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
    return sub


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    m = grid != int(bg)
    if not np.any(m):
        return grid.copy()
    pts = np.argwhere(m)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = sub.shape
    holes = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 == 0 or c0 == 0 or r1 == int(h - 1) or c1 == int(w - 1)):
            continue
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        holes.append(
            {
                "pts": pts,
                "cent": cent,
                "size": int(len(pts)),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return holes


def _outline_col(sub: np.ndarray, bg: int) -> int:
    if sub.size == 0:
        return int(bg)
    border_vals = []
    border_vals.extend(map(int, sub[0, :].tolist()))
    border_vals.extend(map(int, sub[-1, :].tolist()))
    if sub.shape[0] > 2:
        border_vals.extend(map(int, sub[1:-1, 0].tolist()))
        border_vals.extend(map(int, sub[1:-1, -1].tolist()))
    border_non_bg = [v for v in border_vals if int(v) != int(bg)]
    if border_non_bg:
        return int(Counter(border_non_bg).most_common(1)[0][0])
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _pattern_phase(local_r: int, local_c: int, family: str) -> int:
    if family == "checker":
        return int((int(local_r) + int(local_c)) % 2)
    if family == "hstripe":
        return int(local_r % 2)
    return int(local_c % 2)


def _pattern_signature(mask: np.ndarray, family: str) -> tuple[int, ...]:
    pts = np.argwhere(mask)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    return tuple(_pattern_phase(int(r), int(c), family) for r, c in pts)


def _infer_pattern_spec(sub: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    vals = [int(sub[int(rr), int(cc)]) for rr, cc in pts]
    uniq = sorted(set(vals))
    if not uniq:
        return {
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "family": "checker",
            "phase0_color": int(bg),
            "phase1_color": int(bg),
        }
    color_pairs = []
    if len(uniq) == 1:
        color_pairs = [(int(uniq[0]), int(uniq[0]))]
    else:
        for a in uniq:
            for b in uniq:
                if int(a) != int(b):
                    color_pairs.append((int(a), int(b)))
    best = None
    best_mismatch = None
    for family in PATTERN_FAMILIES:
        for phase0, phase1 in color_pairs:
            mismatch = 0
            for rr, cc in pts:
                local_r = int(rr) - int(r0)
                local_c = int(cc) - int(c0)
                expect = int(phase0) if _pattern_phase(local_r, local_c, family) == 0 else int(phase1)
                mismatch += int(int(sub[int(rr), int(cc)]) != expect)
            if best_mismatch is None or mismatch < best_mismatch:
                best_mismatch = mismatch
                best = (family, int(phase0), int(phase1))
    family, phase0, phase1 = best if best is not None else ("checker", int(bg), int(bg))
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "family": str(family),
        "phase0_color": int(phase0),
        "phase1_color": int(phase1),
    }


def _source_cavity_specs(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    specs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        spec = _infer_pattern_spec(sub, pts, bg)
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        spec["cent"] = cent
        specs.append(spec)
    return specs


def _rank_keys(pts: list[tuple[int, int]]):
    rows = sorted(set(int(r) for r, _c in pts))
    cols = sorted(set(int(c) for _r, c in pts))
    rrank = {r: i for i, r in enumerate(rows)}
    crank = {c: i for i, c in enumerate(cols)}
    keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
    return keys, rrank, crank


def _transform_point(p: tuple[int, int], h0: int, w0: int, k: int, flip: bool) -> tuple[int, int]:
    r, c = int(p[0]), int(p[1])
    if int(k) == 0:
        rr, cc = int(r), int(c)
        _h2, w2 = int(h0), int(w0)
    elif int(k) == 1:
        rr, cc = int(w0 - 1 - c), int(r)
        _h2, w2 = int(w0), int(h0)
    elif int(k) == 2:
        rr, cc = int(h0 - 1 - r), int(w0 - 1 - c)
        _h2, w2 = int(h0), int(w0)
    else:
        rr, cc = int(c), int(h0 - 1 - r)
        _h2, w2 = int(w0), int(h0)
    if bool(flip):
        cc = int(w2 - 1 - cc)
    return int(rr), int(cc)


def _parse_legend(sub: np.ndarray, bg: int) -> dict[int, int] | None:
    outc = _outline_col(sub, bg)
    h, w = sub.shape
    if int(w) != 5 or int(h) < 5:
        return None
    if np.any(sub == int(bg)):
        return None
    if not np.all(sub[0, :] == int(outc)) or not np.all(sub[-1, :] == int(outc)):
        return None
    if not np.all(sub[:, 0] == int(outc)) or not np.all(sub[:, -1] == int(outc)):
        return None
    if not np.all(sub[:, 2] == int(outc)):
        return None
    mapping: dict[int, int] = {}
    seen_dst: set[int] = set()
    for r in range(1, int(h - 1)):
        src_col = int(sub[int(r), 1])
        dst_col = int(sub[int(r), 3])
        if src_col in (int(bg), int(outc)) or dst_col in (int(bg), int(outc)):
            return None
        if src_col in mapping or dst_col in seen_dst:
            return None
        mapping[int(src_col)] = int(dst_col)
        seen_dst.add(int(dst_col))
    return mapping if mapping else None


def _analyze_scene(grid: np.ndarray, bg: int) -> dict:
    comps = _components(grid != int(bg))
    infos = []
    for comp in comps:
        sub = _extract_subgrid(grid, comp["pts"], bg)
        infos.append(
            {
                "sub": sub,
                "size": int(comp["size"]),
                "holes": _enclosed_bg_components(sub, bg),
                "legend_map": _parse_legend(sub, bg),
            }
        )
    legend_candidates = [i for i, info in enumerate(infos) if info["legend_map"]]
    legend_idx = None
    if legend_candidates:
        legend_idx = min(
            legend_candidates,
            key=lambda i: (int(infos[i]["size"]), -len(infos[i]["legend_map"])),
        )
    color_map = infos[legend_idx]["legend_map"] if legend_idx is not None else {}
    object_infos = [info for i, info in enumerate(infos) if i != legend_idx]
    object_infos.sort(key=lambda d: int(d["size"]), reverse=True)
    return {
        "color_map": color_map,
        "object_infos": object_infos,
        "legend_idx": legend_idx,
    }


def _match_cavities(dest: np.ndarray, src: np.ndarray, bg: int) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    dest_holes = _enclosed_bg_components(dest, bg)
    src_specs = _source_cavity_specs(src, bg)
    if not dest_holes or not src_specs:
        return [], dest_holes, src_specs

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in (False, True):
            tpts = [
                _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
                for s in src_specs
            ]
            tkeys, _s_rr, _s_cc = _rank_keys(tpts)
            if sorted(tkeys) == sorted(dest_keys):
                best = (int(k), bool(flip))
                break
        if best is not None:
            break
    if best is None:
        best = (0, False)
    k, flip = best

    dest_list = []
    for hole in dest_holes:
        key = (int(d_rr[int(hole["cent"][0])]), int(d_cc[int(hole["cent"][1])]))
        dest_list.append((key, hole["cent"], hole))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [
        _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        for s in src_specs
    ]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for spec in src_specs:
        tp = _transform_point(spec["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, spec))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    pairs = [(spec, hole) for (_k1, _p1, spec), (_k2, _p2, hole) in zip(src_list, dest_list)]
    return pairs, dest_holes, src_specs


def _fill_dest(dest: np.ndarray, pairs: list[tuple[dict, dict]], color_map: dict[int, int] | None) -> np.ndarray:
    out = dest.copy()
    cmap = color_map or {}
    for spec, hole in pairs:
        r0, c0, _r1, _c1 = hole["bbox"]
        phase0 = int(cmap.get(int(spec["phase0_color"]), int(spec["phase0_color"])))
        phase1 = int(cmap.get(int(spec["phase1_color"]), int(spec["phase1_color"])))
        for rr, cc in hole["pts"]:
            local_r = int(rr) - int(r0)
            local_c = int(cc) - int(c0)
            col = phase0 if _pattern_phase(local_r, local_c, spec["family"]) == 0 else phase1
            out[int(rr), int(cc)] = int(col)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the target holes by copying matched src cavity patterns, but recolor
    each pattern phase through the visible legend card first.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    scene = _analyze_scene(g, bg)
    object_infos = scene["object_infos"]
    if len(object_infos) < 2:
        return _tight_crop_non_bg(g, bg)

    objs = [object_infos[0]["sub"], object_infos[1]["sub"]]
    holes_list = [object_infos[0]["holes"], object_infos[1]["holes"]]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    pairs, dest_holes, src_specs = _match_cavities(dest, src, bg)
    if not dest_holes or not src_specs:
        return _tight_crop_non_bg(dest, bg)

    filled = _fill_dest(dest, pairs, scene["color_map"])
    return _tight_crop_non_bg(filled, bg)


# --------------- generator ---------------

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (4, 6)

ROW_HEIGHT_CHOICES = [2, 2, 3, 3, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]

CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40
SHAPE_RETRY_LIMIT = 40
PAIR_BUILD_RETRY_LIMIT = 50
COLOR_MISMATCH_RATE = 0.10

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 400
MAX_LAYOUT_R_BUDGET = 24


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool = False) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape
    n_remove = random.randint(1, max(1, n_remove_max))
    tried = 0
    removed = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        cand = []
        for r in range(h):
            for c in range(w):
                if not m[r, c]:
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and m[nr, nc]:
                        nb += 1
                if nb <= 2:
                    cand.append((r, c))
        if not cand:
            break
        r, c = random.choice(cand)
        to_remove = [(r, c)]
        if symmetric:
            mc = w - 1 - c
            if (r, mc) != (r, c):
                to_remove.append((r, mc))
        m2 = m.copy()
        for rr, cc in to_remove:
            m2[rr, cc] = False
        if not m2.any():
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += len(to_remove)
    if not m.any():
        return mask.copy()
    return m


def _shape_uses_full_bbox(mask: np.ndarray) -> bool:
    b = bbox(mask)
    if b is None:
        return False
    h, w = mask.shape
    return tuple(map(int, b)) == (0, 0, int(h - 1), int(w - 1))


def _shape_supports_family(mask: np.ndarray, family: str) -> bool:
    pts = np.argwhere(mask)
    if len(pts) < 2:
        return False
    if not _shape_uses_full_bbox(mask):
        return False
    sig = _pattern_signature(mask, family)
    if len(set(sig)) < 2:
        return False
    for other in PATTERN_FAMILIES:
        if other != family and sig == _pattern_signature(mask, other):
            return False
    return True


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool, family: str) -> np.ndarray:
    rect = _random_rect_shape(h, w)
    if _shape_supports_family(rect, family) and (not nonrect or h * w < 4):
        return rect
    if not nonrect or h * w < 4:
        return rect
    for _ in range(SHAPE_RETRY_LIMIT):
        m = _mutate_shape(rect, n_remove_max=max(1, (h * w) // 2), symmetric=symmetric)
        if _shape_supports_family(m, family):
            return m
    return rect


def _mirror_shape(mask: np.ndarray) -> np.ndarray:
    return mask[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool, family: str) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PAIR_BUILD_RETRY_LIMIT):
        dst = _gen_shape(h, w, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=symmetric, family=family)
        src = _gen_shape(h, w, nonrect=(random.random() < NONRECT_PROBABILITY), symmetric=symmetric, family=family)
        if _shape_supports_family(dst, family) and _shape_supports_family(src, family) and not np.array_equal(dst, src):
            return dst, src
    rect = _random_rect_shape(h, w)
    if symmetric and w >= 3:
        src = rect.copy()
        src[0, 0] = False
        src[0, -1] = False
        if _shape_supports_family(src, family):
            return rect, src
    if h >= 2 and w >= 2:
        src = rect.copy()
        src[0, 0] = False
        if _shape_supports_family(src, family):
            return rect, src
    return rect, rect


def _add_centered_slot(slots: list[dict], cur_r: int, pair_id: int, row_h: int, family: str) -> None:
    w = random.choice(CENTERED_SLOT_WIDTHS)
    dst, src = _gen_dst_src_pair(row_h, w, symmetric=True, family=family)
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(w // 2),
            "shape_dst": dst,
            "shape_src": src,
            "pair_id": pair_id,
        }
    )


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int, family: str) -> None:
    w = random.choice(PAIR_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w, symmetric=False, family=family)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap % 2 == 0:
        axis_gap += 1
    half = axis_gap // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(half + w),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": half + 1,
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": pair_id,
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int, family: str) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w_side, symmetric=False, family=family)
    dst_r = _mirror_shape(dst_l)
    src_r = _mirror_shape(src_l)
    dst_m, src_m = _gen_dst_src_pair(row_h, w_mid, symmetric=True, family=family)
    mid_half = w_mid // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(mid_half + w_side + TRIO_MID_SIDE_GAP),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -mid_half,
            "shape_dst": dst_m,
            "shape_src": src_m,
            "pair_id": pair_id + 1,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": mid_half + TRIO_MID_SIDE_GAP + 1,
            "shape_dst": dst_r,
            "shape_src": src_r,
            "pair_id": pair_id,
        }
    )


def _pick_layout_k(budget: int) -> int:
    if budget == 1:
        return 1
    if budget == 2:
        return random.choice(LAYOUT_K_CHOICES_SMALL)
    return random.choice(LAYOUT_K_CHOICES_FULL)


def _design_layout(n_target: int, family: str) -> list[dict]:
    slots: list[dict] = []
    cur_r = 0
    pair_id = 0
    budget = n_target
    while len(slots) < n_target and cur_r < MAX_LAYOUT_R_BUDGET and budget > 0:
        row_h = random.choice(ROW_HEIGHT_CHOICES)
        k = _pick_layout_k(budget)
        if k == 1:
            _add_centered_slot(slots, cur_r, pair_id, row_h, family)
            pair_id += 1
            budget -= 1
        elif k == 2:
            _add_mirror_pair_slots(slots, cur_r, pair_id, row_h, family)
            pair_id += 1
            budget -= 2
        else:
            _add_trio_slots(slots, cur_r, pair_id, row_h, family)
            pair_id += 2
            budget -= 3
        cur_r += row_h + random.choice(ROW_GAP_CHOICES)
    return slots


def _slot_pattern_color(spec: dict, local_r: int, local_c: int) -> int:
    phase = _pattern_phase(local_r, local_c, spec["family"])
    return int(spec["phase0_color"]) if phase == 0 else int(spec["phase1_color"])


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    pattern_specs: list[dict],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc, rr, cc))
    if not placed_slots:
        return None, None

    all_rows = np.concatenate([p[1] for p in placed_slots])
    all_cols = np.concatenate([p[2] for p in placed_slots])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    h = (r_max - r_min + 1) + 2 * pad
    w = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((h, w), bg, dtype=int)
    fill_mask = np.zeros((h, w), dtype=bool)
    for slot_idx, rows, cols, local_rr, local_cc in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        spec = pattern_specs[slot_idx]
        for gr, gc, lr, lc in zip(rr, cc, local_rr, local_cc):
            canvas[int(gr), int(gc)] = _slot_pattern_color(spec, int(lr), int(lc))
        fill_mask[rr, cc] = True

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = outline_col

    filled_grid = canvas.copy()
    if not filled:
        canvas[fill_mask] = bg

    b = bbox(canvas != bg)
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[r0b : r1b + 1, c0b : c1b + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != int(bg)) & (filled_grid != int(outline_col))
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1:
            count += 1
    return count


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _assign_pair_pattern_specs(layout: list[dict], fill_pool: list[int], family: str) -> list[dict]:
    unique_pair_ids: list[int] = []
    for slot in layout:
        if slot["pair_id"] not in unique_pair_ids:
            unique_pair_ids.append(slot["pair_id"])
    random.shuffle(unique_pair_ids)
    n_fill = len(fill_pool)
    offset = random.randint(1, max(1, n_fill - 1)) if n_fill > 1 else 0
    base_specs: dict[int, dict] = {}
    for i, pid in enumerate(unique_pair_ids):
        a = int(fill_pool[i % n_fill])
        b = int(fill_pool[(i + offset) % n_fill])
        if a == b:
            b = int(fill_pool[(i + offset + 1) % n_fill])
        if random.random() < 0.5:
            a, b = b, a
        base_specs[int(pid)] = {"family": family, "phase0_color": int(a), "phase1_color": int(b)}
    specs = [dict(base_specs[int(slot["pair_id"])]) for slot in layout]
    for spec in specs:
        if random.random() < COLOR_MISMATCH_RATE:
            a, b = random.sample(fill_pool, 2)
            if random.random() < 0.5:
                a, b = b, a
            spec["phase0_color"] = int(a)
            spec["phase1_color"] = int(b)
    return specs


def _ensure_color_coverage(pattern_specs: list[dict], fill_pool: list[int]) -> list[dict]:
    specs = [dict(spec) for spec in pattern_specs]
    for col in fill_pool:
        used = any(int(spec["phase0_color"]) == int(col) or int(spec["phase1_color"]) == int(col) for spec in specs)
        if used:
            continue
        idx = random.randrange(len(specs))
        replace_phase0 = random.random() < 0.5
        if replace_phase0:
            specs[idx]["phase0_color"] = int(col)
            if int(specs[idx]["phase0_color"]) == int(specs[idx]["phase1_color"]):
                other_choices = [c for c in fill_pool if int(c) != int(col)]
                specs[idx]["phase1_color"] = int(random.choice(other_choices))
        else:
            specs[idx]["phase1_color"] = int(col)
            if int(specs[idx]["phase0_color"]) == int(specs[idx]["phase1_color"]):
                other_choices = [c for c in fill_pool if int(c) != int(col)]
                specs[idx]["phase0_color"] = int(random.choice(other_choices))
    return specs


def _used_source_colors(pattern_specs: list[dict]) -> list[int]:
    vals = []
    for spec in pattern_specs:
        vals.append(int(spec["phase0_color"]))
        vals.append(int(spec["phase1_color"]))
    return sorted(set(vals))


def _make_derangement(colors: list[int]) -> dict[int, int]:
    src = list(colors)
    dst = list(colors)
    while True:
        random.shuffle(dst)
        if all(int(a) != int(b) for a, b in zip(src, dst)):
            return {int(a): int(b) for a, b in zip(src, dst)}


def _map_pattern_specs(pattern_specs: list[dict], color_map: dict[int, int]) -> list[dict]:
    out = []
    for spec in pattern_specs:
        out.append(
            {
                "family": spec["family"],
                "phase0_color": int(color_map[int(spec["phase0_color"])]),
                "phase1_color": int(color_map[int(spec["phase1_color"])]),
            }
        )
    return out


def _build_legend(color_map: dict[int, int], outline_col: int) -> np.ndarray:
    rows = list(color_map.items())
    random.shuffle(rows)
    legend = np.full((len(rows) + 2, 5), outline_col, dtype=int)
    for i, (src_col, dst_col) in enumerate(rows, start=1):
        legend[int(i), 1] = int(src_col)
        legend[int(i), 3] = int(dst_col)
    return legend


def _build_object_pair(
    layout: list[dict],
    src_specs: list[dict],
    mapped_specs: list[dict],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, mapped_specs, bg, outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(layout, src_shapes, src_specs, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_cropped):
        return None

    return src_cropped, dst_cropped, dst_filled


def _place_on_canvas(patches: list[np.ndarray], bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    for patch in patches:
        try:
            r, c = random_free_location(
                canvas,
                patch,
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
            )
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = palette[0]
    family = random.choice(PATTERN_FAMILIES)

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE), family)
    if len(layout) < CAVITY_COUNT_RANGE[0]:
        return None

    max_cols = min(FILL_COLOR_COUNT_RANGE[1], len(palette) - 1)
    if max_cols < FILL_COLOR_COUNT_RANGE[0]:
        return None
    n_fill_cols = random.randint(FILL_COLOR_COUNT_RANGE[0], max_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]

    src_specs = _assign_pair_pattern_specs(layout, fill_pool, family)
    src_specs = _ensure_color_coverage(src_specs, fill_pool)
    used_colors = _used_source_colors(src_specs)
    if len(used_colors) < FILL_COLOR_COUNT_RANGE[0]:
        return None
    color_map = _make_derangement(used_colors)
    mapped_specs = _map_pattern_specs(src_specs, color_map)

    built = _build_object_pair(layout, src_specs, mapped_specs, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built
    legend = _build_legend(color_map, outline_col)

    canvas = _place_on_canvas([src, dst, legend], bg)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None

    metrics = sample_metrics(canvas)
    if int(metrics["mapped_changed_cells"]) <= 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")


def sample_metrics(grid: np.ndarray) -> dict[str, int | str]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    scene = _analyze_scene(g, bg)
    object_infos = scene["object_infos"]
    if len(object_infos) < 2:
        return {
            "hole_count": 0,
            "legend_rows": len(scene["color_map"]),
            "mapped_changed_cells": 0,
            "filled_cells": 0,
            "family_count": 0,
        }
    objs = [object_infos[0]["sub"], object_infos[1]["sub"]]
    holes_list = [object_infos[0]["holes"], object_infos[1]["holes"]]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    pairs, dest_holes, src_specs = _match_cavities(dest, src, bg)
    direct = _fill_dest(dest, pairs, {})
    mapped = _fill_dest(dest, pairs, scene["color_map"])
    hole_mask = np.zeros_like(dest, dtype=bool)
    for hole in dest_holes:
        for rr, cc in hole["pts"]:
            hole_mask[int(rr), int(cc)] = True
    changed = int(np.sum((direct != mapped) & hole_mask))
    families = sorted({spec["family"] for spec in src_specs})
    return {
        "hole_count": len(dest_holes),
        "legend_rows": len(scene["color_map"]),
        "mapped_changed_cells": changed,
        "filled_cells": int(np.sum(hole_mask)),
        "family_count": len(families),
    }
