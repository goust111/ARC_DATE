# Combo variant of f560132c (index 3)
# combo_spec: D1/3+D2/1
# Name: 连接器颜色匹配+锚点像素

import itertools
import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# fusion_summary:
#   preserved_backbone:
#     - 仍然是 4 个 4-connected piece，在唯一的 S*S 方形里 exact-cover 拼合。
#     - 仍然有一个 legend piece，输入继续由 output-first 生成，再反向散落成 ARC 输入。
#     - 输出仍是把整个拼好的方形紧凑裁剪后，按 legend 给出的 2x2 颜色图例统一重着色。
#   fused_change:
#     - D1/3 把“按形状/空间象限直接配色”改成“读 legend 上的连接器颜色”。
#     - D2/1 再加入 piece 自带锚点像素，piece 放到哪个大象限不再由质心决定，而由锚点所在象限决定。
#     - 两者融合后不采用 4 个独立 connector 色或 4 个独立 anchor 色，否则任一层单独就能唯一确定象限。
#     - 这里改成 2 个 anchor 色(top/bottom) + 2 个 connector 色(left/right)，两者组合成 4 个象限码：
#         TL = (anchor_top, connector_left)
#         TR = (anchor_top, connector_right)
#         BL = (anchor_bottom, connector_left)
#         BR = (anchor_bottom, connector_right)
#       这样必须同时读取两层信息，任意一层单独都不够。
#   concrete_pixels:
#     - 所有 piece 共用同一种 body color。
#     - legend 内嵌固定 4x4 微结构：
#         aT  bd  cR  aT
#         cL  pTL pTR bd
#         bd  pBL pBR cR
#         aB  cL  bd  aB
#       其中：
#         p** = 4 个不同的输出 palette 色
#         aT/aB = 上半/下半锚点色（2 色）
#         cL/cR = 左半/右半连接器色（2 色）
#         bd = body 色
#     - 非 legend piece 只有 1 个锚点像素 + 1 个连接器像素；legend 额外再带 1 个 piece-anchor 和 1 个 piece-connector，
#       用来说明 legend 自己这块 piece 最终该涂哪一个 palette 色。
#   behavior_trace:
#     - 原题：exact-cover 后，取每块 piece 的质心，量化到 2x2 空间格，再从 legend_map 拿颜色。
#     - 单独 D1/3：exact-cover 后，直接看 piece 的 connector 色，查 legend connector->palette。
#     - 本融合版：先从 piece 读出 (anchor_color, connector_color) 对；这个组合给出目标象限。
#       exact-cover 时再要求“piece 的锚点单元必须落在该目标大象限内”，最后用对应象限的 palette 色重着色。
#     - 差异点：空间位置不再由质心决定，而是由 piece 自己携带的锚点定位；connector 只给左右位信息，
#       anchor 只给上下位信息，两者都缺一不可。
#   small_example:
#     - 假设某块 piece 的锚点色是 aT、连接器色是 cR，那么它只能对应 TR 象限。
#     - 原题会问：“这块形状最终质心落在哪个象限？”；本题改成：“这块的锚点单元有没有被放进上半+右半区域？”
#     - 即便这块 piece 的大部分面积跨到了下半区，只要锚点单元位于右上大象限，它仍应涂成 legend 的 TR 颜色。
#   complexity_calibration:
#     - 若 4 个 anchor 色都唯一，connector 会退化成装饰；若 4 个 connector 色都唯一，anchor 也会退化。
#     - 2x2 组合编码把自由度压到 2+2 个离散取值，既不会颜色爆炸，也能让样例中重复出现“同锚点异连接器 /
#       同连接器异锚点”的对照，肉眼可归纳。
#   implicit_assumptions_audit:
#     - 原 main() 假设 piece 可以视作单色几何体，只在 legend 内读颜色；这里失效，因为每块 piece 都有两类标记色。
#     - 原 main() 假设输出颜色由装配后的空间质心直接决定；这里失效，必须改成由 (anchor, connector) 组合决定。
#     - 原 exact-cover 对非 legend piece 的所有放置都等价；这里失效，放置还要满足“锚点落在指定大象限”。
#     - 新 solver 的输入假设：
#         1. legend 的 4x4 微结构唯一存在，且未被旋转。
#         2. 非 legend piece 恰有 1 个 anchor 像素和 1 个 connector 像素。
#         3. legend 除 4x4 微结构外，恰有 1 个额外 anchor 像素和 1 个额外 connector 像素。
#         4. 2 个 anchor 色与 2 个 connector 色彼此不同，且都与 4 个 palette 色、body 色不同。
#       generator 会显式构造并验证这些条件。
#   placement_legality:
#     - 锚点和连接器都只是把 piece 既有占据单元改色，不增加凸起，因此不会破坏 exact-cover 面积。
#     - legend 微结构固定嵌在其左上 4x4 满块里，额外标记放在该块之外，避免与签名冲突。
#     - 散落仍使用 random_free_location，bbox 之间保留 padding，保证不重叠、不越界。
#   visual_observability_and_tidiness:
#     - 输入里所有 piece 的 body 色一致，解题者会自然关注“少量重复出现的 2 个 anchor 色和 2 个 connector 色”。
#     - legend 上 4 个 palette 色、4 个角锚点、4 个边连接器排布整齐，不会与 piece 的散落位置混淆。
#     - 同一 anchor 色会在两个 piece 上复用、同一 connector 色也会在两个 piece 上复用，因此人眼能从 2-3 个样例看出
#       “上/下由 anchor 决定，左/右由 connector 决定”，而不是把任一单色误判为完整标签。


K = 2
NUM_PIECES = K * K
SIDE_CHOICES = [8, 10, 12]
AREA_FRAC_LO = 0.08
AREA_FRAC_HI = 0.43
AREA_ABS_LO = 11
AREA_ABS_HI = 62
LEGEND_THRESHOLD = {8: 4, 10: 5, 12: 9}
LEGEND_CORE_SIZE = 4
LEGEND_ARM_RANGE = range(1, 4)
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
GEN_RETRIES = 400

PATTERN_SIZE = 4
PALETTE_POS = ((1, 1), (1, 2), (2, 1), (2, 2))  # TL, TR, BL, BR
ANCHOR_POS = ((0, 0), (0, 3), (3, 0), (3, 3))  # TL, TR, BL, BR
CONNECTOR_POS = ((1, 0), (0, 2), (3, 1), (2, 3))  # TL, TR, BL, BR
BODY_POS = tuple(
    (r, c)
    for r in range(PATTERN_SIZE)
    for c in range(PATTERN_SIZE)
    if (r, c) not in set(PALETTE_POS) | set(ANCHOR_POS) | set(CONNECTOR_POS)
)
QUADRANT_TO_BITS = {
    0: (0, 0),  # TL
    1: (0, 1),  # TR
    2: (1, 0),  # BL
    3: (1, 1),  # BR
}


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(n**0.5) + 1):
        if n % h == 0:
            w = n // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    return out


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, _w = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _h, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


def _rot90_point(pt: tuple[int, int], h: int) -> tuple[int, int]:
    r, c = pt
    return int(c), int(h - 1 - r)


def _flip_h_point(pt: tuple[int, int], w: int) -> tuple[int, int]:
    r, c = pt
    return int(r), int(w - 1 - c)


def _dihedral_orientations(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]] = []
    cur = coords
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), False))
        cur = _rot90(cur)[0]
    cur = _flip_h(coords)[0]
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), True))
        cur = _rot90(cur)[0]
    return out


def _dihedral_orientations_with_anchor(
    coords: tuple[tuple[int, int], ...],
    anchor: tuple[int, int],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool, tuple[int, int]]]:
    norm, _h, _w = _normalize_coords(coords)
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[tuple[tuple[tuple[int, int], ...], int, int, int, bool, tuple[int, int]]] = []

    cur_coords = norm
    cur_anchor = tuple(map(int, anchor))
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur_coords)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), False, cur_anchor))
        _c2, old_h, _old_w = _normalize_coords(cur_coords)
        cur_coords, _nh, _nw = _rot90(cur_coords)
        cur_anchor = _rot90_point(cur_anchor, old_h)

    cur_coords, _fh, old_w = _flip_h(norm)
    cur_anchor = _flip_h_point(anchor, old_w)
    for rot_k in range(4):
        c, h, w = _normalize_coords(cur_coords)
        if c not in seen:
            seen.add(c)
            out.append((c, h, w, int(rot_k), True, cur_anchor))
        _c2, old_h, _old_w = _normalize_coords(cur_coords)
        cur_coords, _nh, _nw = _rot90(cur_coords)
        cur_anchor = _rot90_point(cur_anchor, old_h)
    return out


def _fill_holes_in_component(g: np.ndarray, pts: np.ndarray, bg: int) -> tuple[tuple[tuple[int, int], ...], int]:
    r0, c0, r1, c1 = bbox_coords(pts)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    occ = np.zeros((hh, ww), dtype=bool)
    for r, c in pts:
        occ[r - r0, c - c0] = True

    bgmask = ~occ
    vis = np.zeros_like(bgmask, dtype=bool)
    st: list[tuple[int, int]] = []

    for rr in range(hh):
        for cc in range(ww):
            if rr in (0, hh - 1) or cc in (0, ww - 1):
                if bgmask[rr, cc] and not vis[rr, cc]:
                    vis[rr, cc] = True
                    st.append((rr, cc))
                    while st:
                        r, c = st.pop()
                        for r2, c2 in neighbors4(r, c):
                            if 0 <= r2 < hh and 0 <= c2 < ww and bgmask[r2, c2] and not vis[r2, c2]:
                                vis[r2, c2] = True
                                st.append((r2, c2))

    holes = bgmask & ~vis
    hole_coords = [(r0 + rr, c0 + cc) for rr, cc in zip(*np.where(holes))]
    filled = tuple(sorted(set(map(tuple, pts)).union(hole_coords)))
    return filled, int(holes.sum())


def _cell_id_from_point(r: int, c: int, H: int, W: int) -> int:
    rr = min(K - 1, int(r * K / H))
    cc = min(K - 1, int(c * K / W))
    return int(rr * K + cc)


def _extract_legend_signature(
    g: np.ndarray,
    pts: np.ndarray,
    bg: int,
) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    candidates = []

    for rr in range(hh - PATTERN_SIZE + 1):
        for cc in range(ww - PATTERN_SIZE + 1):
            window = sub[rr : rr + PATTERN_SIZE, cc : cc + PATTERN_SIZE]
            body_vals = [int(window[r, c]) for r, c in BODY_POS]
            if len(set(body_vals)) != 1:
                continue
            body = int(body_vals[0])
            if body == bg:
                continue

            pal_vals = [int(window[r, c]) for r, c in PALETTE_POS]
            if len(set(pal_vals)) != 4:
                continue

            anc_vals = [int(window[r, c]) for r, c in ANCHOR_POS]
            if not (anc_vals[0] == anc_vals[1] and anc_vals[2] == anc_vals[3] and anc_vals[0] != anc_vals[2]):
                continue

            conn_vals = [int(window[r, c]) for r, c in CONNECTOR_POS]
            if not (conn_vals[0] == conn_vals[2] and conn_vals[1] == conn_vals[3] and conn_vals[0] != conn_vals[1]):
                continue

            palette_set = set(map(int, pal_vals))
            anchor_set = {int(anc_vals[0]), int(anc_vals[2])}
            connector_set = {int(conn_vals[0]), int(conn_vals[1])}
            if body in palette_set or body in anchor_set or body in connector_set:
                continue
            if palette_set & anchor_set or palette_set & connector_set or anchor_set & connector_set:
                continue

            pair_to_palette = {}
            for idx in range(4):
                pair = (int(anc_vals[idx]), int(conn_vals[idx]))
                pair_to_palette[pair] = int(pal_vals[idx])
            if len(pair_to_palette) != 4:
                continue

            pattern_block = {
                (r0 + rr + dr, c0 + cc + dc)
                for dr in range(PATTERN_SIZE)
                for dc in range(PATTERN_SIZE)
            }
            candidates.append(
                {
                    "pair_to_palette": pair_to_palette,
                    "palette": np.array([[pal_vals[0], pal_vals[1]], [pal_vals[2], pal_vals[3]]], dtype=int),
                    "anchor_colors": (int(anc_vals[0]), int(anc_vals[2])),
                    "connector_colors": (int(conn_vals[0]), int(conn_vals[1])),
                    "body_color": body,
                    "pattern_block": pattern_block,
                }
            )

    assert len(candidates) == 1, 'legend must contain exactly one valid anchor+connector+palette micro-pattern'
    info = candidates[0]
    assert len({int(g[r, c]) for r, c in pts if int(g[r, c]) != bg}) == 9, 'legend must use 9 distinct non-bg colors'
    return info


def _solve_exact_cover(
    pieces: list[dict],
    legend_idx: int,
) -> tuple[int, int, dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]]:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _, _ in p["oris"]) for p in pieces if p["oris"])
    max_w = max(max(w for _, _, w, _, _, _ in p["oris"]) for p in pieces if p["oris"])

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    assert len(dims) == 1, 'piece areas and bounds must imply a unique output rectangle'

    H, W = dims[0]
    nbits = H * W
    all_mask = (1 << nbits) - 1

    legend_coords = pieces[legend_idx]["fixed_coords"]
    legend_mask = 0
    for dr, dc in legend_coords:
        legend_mask |= 1 << (int(dr) * int(W) + int(dc))

    other_pis = [i for i in range(len(pieces)) if i != legend_idx]
    placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
    for pi in other_pis:
        opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
        for coords, ph, pw, _rot_k, _flipped, anchor_rel in pieces[pi]["oris"]:
            for r0 in range(0, H - ph + 1):
                for c0 in range(0, W - pw + 1):
                    aid = _cell_id_from_point(r0 + anchor_rel[0], c0 + anchor_rel[1], H, W)
                    if aid != pieces[pi]["target_id"]:
                        continue
                    m = 0
                    for dr, dc in coords:
                        idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                        m |= 1 << idx
                    if (m & legend_mask) != 0:
                        continue
                    opts.append((m, int(r0), int(c0), coords))
        assert opts, 'piece must have at least one anchor-compatible placement'
        placements_other.append(opts)

    remaining = set(range(len(other_pis)))
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] = {
        legend_idx: (legend_mask, 0, 0, legend_coords)
    }

    def search(used: int) -> bool:
        if not remaining:
            return used == all_mask
        empty = (~used) & all_mask
        if empty == 0:
            return False
        cell = (empty & -empty).bit_length() - 1

        candidates = []
        for k in sorted(remaining):
            opts_k = []
            for pl in placements_other[k]:
                m = pl[0]
                if (m >> cell) & 1 and (m & used) == 0:
                    opts_k.append(pl)
            if opts_k:
                candidates.append((len(opts_k), k, opts_k))
        if not candidates:
            return False

        candidates.sort(key=lambda x: (x[0], x[1]))
        for _, k, opts_k in candidates:
            remaining.remove(k)
            pi = other_pis[k]
            for pl in opts_k:
                chosen[pi] = pl
                if search(used | pl[0]):
                    return True
            chosen.pop(pi, None)
            remaining.add(k)
        return False

    assert search(legend_mask), 'exact-cover assembly failed under anchor constraints'
    return int(H), int(W), chosen


def _render_solution(
    H: int,
    W: int,
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]],
    pieces: list[dict],
) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for pi, pl in chosen.items():
        _mask, r0, c0, coords = pl
        col = int(pieces[pi]["out_color"])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a square, then recolor them by fused anchor+connector matching.

    Input structure:
    - Background plus 4 disconnected foreground pieces that still exact-cover one unique square under D4 moves.
    - One piece is the legend: it contains a fixed 4x4 micro-pattern with 4 palette colors, 2 repeated anchor colors, and 2 repeated connector colors.
    - Every piece also carries its own anchor pixel and connector pixel; for the legend these are extra markers outside the 4x4 signature.

    Rule:
    - Reassemble the 4 shapes into the tight square with the legend fixed at the square's top-left in its observed orientation.
    - Read each piece's `(anchor_color, connector_color)` pair and match it against the legend's 2x2 quadrant code.
    - During assembly, only placements whose anchor pixel lands inside that target large quadrant are valid; then recolor the full piece with that quadrant's palette color.

    Key insight: Shape still determines how the square tiles, but color no longer comes from a piece's centroid; the anchor gives the vertical half, the connector gives the horizontal half, and only the pair resolves the correct quadrant.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = components_mask(g != bg, connectivity=4)
    assert comps, 'grid must contain at least one non-background component'
    assert len(comps) == NUM_PIECES, 'this combo variant expects exactly 4 pieces'

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max(len(x) for x in comp_colors)]
    assert len(legend_candidates) == 1, 'legend component must be uniquely identified by color richness'
    legend_idx = legend_candidates[0]

    legend_info = _extract_legend_signature(g, comps[legend_idx], bg)
    anchor_colors = set(legend_info["anchor_colors"])
    connector_colors = set(legend_info["connector_colors"])
    pair_to_palette = legend_info["pair_to_palette"]

    pieces: list[dict] = []
    for i, pts in enumerate(comps):
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        bbox_pts = np.asarray(list(filled_abs), dtype=int)
        r0, c0, _r1, _c1 = bbox_coords(bbox_pts)
        rel = tuple((int(r - r0), int(c - c0)) for r, c in filled_abs)
        rel_norm, _hh, _ww = _normalize_coords(rel)

        if i == legend_idx:
            outside = [
                (int(r), int(c))
                for r, c in pts
                if (int(r), int(c)) not in legend_info["pattern_block"]
            ]
            anchor_pts = [(r, c) for r, c in outside if int(g[r, c]) in anchor_colors]
            connector_pts = [(r, c) for r, c in outside if int(g[r, c]) in connector_colors]
            assert len(anchor_pts) == 1, 'legend must have exactly one extra anchor marker'
            assert len(connector_pts) == 1, 'legend must have exactly one extra connector marker'
            fixed_oris = [(rel_norm, _hh, _ww, 0, False, (anchor_pts[0][0] - r0, anchor_pts[0][1] - c0))]
        else:
            anchor_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) in anchor_colors]
            connector_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) in connector_colors]
            assert len(anchor_pts) == 1, 'non-legend piece must have exactly one anchor marker'
            assert len(connector_pts) == 1, 'non-legend piece must have exactly one connector marker'
            fixed_oris = _dihedral_orientations_with_anchor(rel_norm, (anchor_pts[0][0] - r0, anchor_pts[0][1] - c0))

        anchor_color = int(g[anchor_pts[0][0], anchor_pts[0][1]])
        connector_color = int(g[connector_pts[0][0], connector_pts[0][1]])
        pair = (anchor_color, connector_color)
        assert pair in pair_to_palette, 'piece markers must match one legend quadrant'

        target_id = None
        for qid, bits in QUADRANT_TO_BITS.items():
            ac = legend_info["anchor_colors"][bits[0]]
            cc = legend_info["connector_colors"][bits[1]]
            if (anchor_color, connector_color) == (ac, cc):
                target_id = qid
                break
        assert target_id is not None

        pieces.append(
            {
                "area": len(filled_abs),
                "oris": fixed_oris,
                "fixed_coords": rel_norm,
                "target_id": int(target_id),
                "out_color": int(pair_to_palette[pair]),
            }
        )

    H, W, chosen = _solve_exact_cover(pieces, legend_idx)
    return _render_solution(H, W, chosen, pieces)


def _area_bounds(side: int) -> tuple[int, int]:
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _has_hole(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return True
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    empty = ~sub
    if not empty.any():
        return False
    for comp in components_mask(empty, connectivity=4):
        if not (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == sub.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == sub.shape[1] - 1)
        ):
            return True
    return False


def _forces_unique_square(masks: list[np.ndarray], side: int) -> bool:
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _make_legend_mask(side: int, min_area: int, max_area: int) -> np.ndarray:
    threshold = LEGEND_THRESHOLD[side]
    opts = []
    for h0 in range(threshold + 1, side + 1):
        for w0 in range(threshold + 1, side + 1):
            for arm_w in LEGEND_ARM_RANGE:
                for arm_h in LEGEND_ARM_RANGE:
                    mask = np.zeros((side, side), dtype=bool)
                    mask[:LEGEND_CORE_SIZE, :LEGEND_CORE_SIZE] = True
                    mask[:arm_h, :w0] = True
                    mask[:h0, :arm_w] = True
                    area = int(mask.sum())
                    if min_area <= area <= max_area:
                        opts.append(mask)
    return random.choice(opts).copy()


def _partition_rest(rest: np.ndarray) -> np.ndarray | None:
    side = rest.shape[0]
    labels = np.full((side, side), -1, dtype=int)
    available = set(map(tuple, np.argwhere(rest)))
    anchors = [
        (random.randint(0, side // 3), side - 1),
        (side - 1, random.randint(0, side // 3)),
        (side - 1, side - 1),
    ]

    seeds = []
    for ar, ac in anchors:
        seed = min(available, key=lambda p: abs(p[0] - ar) + abs(p[1] - ac) + random.random() * PARTITION_SEED_JITTER)
        seeds.append(seed)
        available.remove(seed)

    areas = [0, 0, 0]
    frontiers = [set() for _ in range(3)]

    def add_frontier(i: int, r: int, c: int) -> None:
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < side and 0 <= cc < side and rest[rr, cc] and labels[rr, cc] < 0:
                frontiers[i].add((rr, cc))

    for i, seed in enumerate(seeds):
        labels[seed] = i
        areas[i] = 1
        add_frontier(i, seed[0], seed[1])

    unassigned = int(rest.sum()) - len(seeds)
    while unassigned:
        candidates = []
        for i in range(3):
            frontiers[i] = {p for p in frontiers[i] if labels[p] < 0}
            if frontiers[i]:
                candidates.append(i)
        if not candidates:
            return None

        weights = [
            (1.0 / (areas[i] ** PARTITION_GROWTH_EXP)) * (len(frontiers[i]) ** PARTITION_FRONTIER_EXP)
            for i in candidates
        ]
        i = random.choices(candidates, weights=weights, k=1)[0]
        cell = random.choice(tuple(frontiers[i]))
        frontiers[i].discard(cell)
        labels[cell] = i
        areas[i] += 1
        unassigned -= 1
        add_frontier(i, cell[0], cell[1])

    return labels


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _boundary_cells(mask: np.ndarray, forbidden: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    forbidden = forbidden or set()
    h, w = mask.shape
    out = []
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if (rr, cc) in forbidden:
            continue
        for r2, c2 in neighbors4(rr, cc):
            if not (0 <= r2 < h and 0 <= c2 < w and mask[r2, c2]):
                out.append((rr, cc))
                break
    return out


def _occupied_cells(mask: np.ndarray, forbidden: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    forbidden = forbidden or set()
    return [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in forbidden]


def _random_orientation_with_anchor(mask: np.ndarray, anchor: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    coords = tuple(map(tuple, np.argwhere(mask)))
    norm, h, w = _normalize_coords(coords)
    assert (h, w) == mask.shape
    opts = _dihedral_orientations_with_anchor(norm, anchor)
    coords2, hh, ww, _rot_k, _flipped, anchor2 = random.choice(opts)
    out = np.zeros((hh, ww), dtype=bool)
    for r, c in coords2:
        out[r, c] = True
    return out, anchor2


def _piece_candidate_cells(
    mask: np.ndarray,
    side: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> dict[int, list[tuple[int, int]]]:
    cands = {qid: [] for qid in range(NUM_PIECES)}
    for r, c in _occupied_cells(mask, forbidden):
        qid = _cell_id_from_point(r, c, side, side)
        cands[qid].append((r, c))
    return cands


def _centroid_cell_id(mask: np.ndarray, side: int) -> int:
    pts = np.argwhere(mask)
    rr = min(K - 1, int(float(pts[:, 0].mean()) * K / side))
    cc = min(K - 1, int(float(pts[:, 1].mean()) * K / side))
    return int(rr * K + cc)


def _choose_piece_targets(
    masks: list[np.ndarray],
    side: int,
) -> tuple[list[int], list[tuple[int, int]], list[int]] | None:
    pattern_forbidden = {(r, c) for r in range(PATTERN_SIZE) for c in range(PATTERN_SIZE)}
    candidate_maps = []
    for pi, mask in enumerate(masks):
        forbidden = pattern_forbidden if pi == 0 else set()
        candidate_maps.append(_piece_candidate_cells(mask, side, forbidden))

    centroid_ids = [_centroid_cell_id(mask, side) for mask in masks]
    perms = list(itertools.permutations(range(NUM_PIECES)))
    random.shuffle(perms)
    for perm in perms:
        if any(not candidate_maps[pi][perm[pi]] for pi in range(NUM_PIECES)):
            continue
        mismatch = sum(int(perm[pi] != centroid_ids[pi]) for pi in range(NUM_PIECES))
        if mismatch < 2:
            continue
        anchors = [random.choice(candidate_maps[pi][perm[pi]]) for pi in range(NUM_PIECES)]
        return list(map(int, perm)), anchors, centroid_ids
    return None


def _paint_simple_piece(
    mask: np.ndarray,
    body_color: int,
    anchor_rel: tuple[int, int],
    anchor_color: int,
    connector_color: int,
) -> np.ndarray | None:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(body_color)
    patch[anchor_rel] = int(anchor_color)
    boundary = _boundary_cells(mask, forbidden={anchor_rel})
    if not boundary:
        return None
    r, c = random.choice(boundary)
    patch[r, c] = int(connector_color)
    return patch


def _paint_legend_piece(
    mask: np.ndarray,
    body_color: int,
    legend_map: np.ndarray,
    anchor_colors: tuple[int, int],
    connector_colors: tuple[int, int],
    target_id: int,
    extra_anchor_rel: tuple[int, int],
) -> np.ndarray | None:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(body_color)

    for idx, (r, c) in enumerate(PALETTE_POS):
        rr, cc = divmod(idx, K)
        patch[r, c] = int(legend_map[rr, cc])
    for idx, (r, c) in enumerate(ANCHOR_POS):
        row_bit, _col_bit = QUADRANT_TO_BITS[idx]
        patch[r, c] = int(anchor_colors[row_bit])
    for idx, (r, c) in enumerate(CONNECTOR_POS):
        _row_bit, col_bit = QUADRANT_TO_BITS[idx]
        patch[r, c] = int(connector_colors[col_bit])

    row_bit, col_bit = QUADRANT_TO_BITS[target_id]
    patch[extra_anchor_rel] = int(anchor_colors[row_bit])

    forbidden = {(r, c) for r in range(PATTERN_SIZE) for c in range(PATTERN_SIZE)} | {extra_anchor_rel}
    boundary = _boundary_cells(mask, forbidden=forbidden)
    if not boundary:
        return None
    r, c = random.choice(boundary)
    patch[r, c] = int(connector_colors[col_bit])
    return patch


def _scatter_pieces(
    masks: list[np.ndarray],
    body_color: int,
    legend_map: np.ndarray,
    anchor_colors: tuple[int, int],
    connector_colors: tuple[int, int],
    target_ids: list[int],
    anchor_cells_abs: list[tuple[int, int]],
    side: int,
) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    order = random.sample(range(NUM_PIECES), NUM_PIECES)
    for pi in order:
        crop = _crop_mask(masks[pi])
        r0, c0, _r1, _c1 = bbox_coords(np.argwhere(masks[pi]))
        anchor_rel_base = (anchor_cells_abs[pi][0] - r0, anchor_cells_abs[pi][1] - c0)

        if pi == 0:
            oriented = crop
            anchor_rel = anchor_rel_base
            patch = _paint_legend_piece(
                oriented,
                body_color,
                legend_map,
                anchor_colors,
                connector_colors,
                target_ids[pi],
                anchor_rel,
            )
        else:
            oriented, anchor_rel = _random_orientation_with_anchor(crop, anchor_rel_base)
            row_bit, col_bit = QUADRANT_TO_BITS[target_ids[pi]]
            patch = _paint_simple_piece(
                oriented,
                body_color,
                anchor_rel,
                anchor_colors[row_bit],
                connector_colors[col_bit],
            )
        if patch is None:
            return None

        try:
            box = np.ones(oriented.shape, dtype=int)
            pr, pc = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        occupied[pr : pr + oriented.shape[0], pc : pc + oriented.shape[1]] = 1
        paste(grid, patch, pr, pc, transparent=0)
    return grid


def _expected_output(masks: list[np.ndarray], target_ids: list[int], legend_map: np.ndarray, side: int) -> np.ndarray:
    out = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = divmod(target_ids[pi], K)
        out[mask] = int(legend_map[rr, cc])
    return out


def _old_centroid_output(masks: list[np.ndarray], legend_map: np.ndarray, side: int) -> np.ndarray:
    out = np.zeros((side, side), dtype=int)
    for mask in masks:
        qid = _centroid_cell_id(mask, side)
        rr, cc = divmod(qid, K)
        out[mask] = int(legend_map[rr, cc])
    return out


def _gen_attempt(return_metrics: bool = False):
    side = random.choice(SIDE_CHOICES)
    min_area, max_area = _area_bounds(side)
    legend = _make_legend_mask(side, min_area, max_area)
    labels = _partition_rest(~legend)
    if labels is None:
        return None

    masks = [legend] + [labels == i for i in range(3)]
    areas = [int(m.sum()) for m in masks]
    if any(a < min_area or a > max_area for a in areas):
        return None
    if any(len(components_mask(m, connectivity=4)) != 1 or _has_hole(m) for m in masks):
        return None
    if not _forces_unique_square(masks, side):
        return None

    target_info = _choose_piece_targets(masks, side)
    if target_info is None:
        return None
    target_ids, anchor_cells_abs, centroid_ids = target_info

    colors = random.sample(range(1, 10), 9)
    body_color = int(colors[0])
    palette_colors = colors[1:5]
    anchor_colors = (int(colors[5]), int(colors[6]))
    connector_colors = (int(colors[7]), int(colors[8]))
    random.shuffle(palette_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    expected = _expected_output(masks, target_ids, legend_map, side)
    old_like = _old_centroid_output(masks, legend_map, side)
    if np.array_equal(expected, old_like):
        return None

    grid = _scatter_pieces(
        masks,
        body_color,
        legend_map,
        anchor_colors,
        connector_colors,
        target_ids,
        anchor_cells_abs,
        side,
    )
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    if not return_metrics:
        return grid

    metrics = {
        "target_ids": target_ids,
        "centroid_ids": centroid_ids,
        "mismatch_count": sum(int(a != b) for a, b in zip(target_ids, centroid_ids)),
        "anchor_colors": anchor_colors,
        "connector_colors": connector_colors,
    }
    return grid, metrics


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate f560132c combo input')
