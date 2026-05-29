# Combo variant of f560132c (index 23)
# combo_spec: D5/0+D8/1+D10/0
# Name: 线性色带+边缘图例+无框架识别

"""
design_analysis:
  concretization_and_trace:
    input_look:
      - 输入里仍有 4 个散落的单色拼块，它们在输出中要恰好拼成一个正方形。
      - 原题的图例是“某个拼块内部的 2x2 多色核心”；本组合改成“网格最外侧顶行或底行上的 1x4 小色带”。
      - 这条色带没有框、没有拉长色段、也不嵌在拼块里；它就是 4 个彼此不同的连续彩色像素。
      - 由于边缘色带不再兼作拼块，solver 还要从剩余 4 个单色拼块中找出谁应当在装配后固定到左上角；这里沿用 D8 的空间线索：离色带最近的拼块就是锚点拼块。
    behavior_trace:
      original:
        - 先在多色组件内部截取 2x2 图例。
        - 多色组件本身也是拼图块，并在 exact-cover 中固定到输出左上。
      combo:
        - 先在顶边或底边寻找唯一的“恰好 4 格、连续、互异、无框”的小色带。
        - 按从左到右读取为 [TL, TR, BL, BR]，再展开成 2x2 象限色表。
        - 删除这 4 个边缘像素后，网格内只剩 4 个单色拼块。
        - 取与该边缘行距离最小的唯一拼块作为旧图例拼块的替身，把它固定到输出左上，其余装配与原题一致。
      difference:
        - 图例的形态从 2D 色块变成 1D 小色带。
        - 图例的位置从拼块内部移到最外层边缘。
        - 图例识别方式从“多色组件”改成“边缘上唯一的 4 连续异色像素”。
  complexity_calibration:
    - 只允许顶边或底边两种边缘位置，避免四边全开导致样例太散，同时足够明显地体现“边缘图例”。
    - 色带长度固定为 4，且 4 色互异，保证人眼能稳定看出 1D 顺序与 2x2 象限的对应。
    - 不使用 D8 单变体里的长色段编码；若继续允许每色重复 2-3 格，会削弱 D10/0 “微小无框图例”的视觉特征。
    - 其他 3 个非锚点拼块被强制远离图例边缘，保证“最近拼块”线索唯一、稳定、可见。
  implicit_assumption_audit:
    original_main_dependencies:
      - "`components_mask(g != bg)` 原先只会得到 4 个拼块；现在先会包含边缘色带，所以必须先删除色带像素再取组件。"
      - "`k = sqrt(len(comps))` 原先可直接由组件数推出；现在要先从边缘色带长度 4 推出 k=2，再检查删除色带后的组件数是否为 4。"
      - "`max distinct colors` 原先可定位图例组件；现在所有拼块都应保持单色，这个假设完全失效。"
      - "`_extract_legend_map` 原先依赖组件 bbox 内存在 2x2 彩色窗；现在必须改成顶/底边上的精确 1x4 检测。"
      - "原题中图例组件天然就是装配锚点；现在锚点需要由与边缘图例的最近关系恢复。"
    new_main_assumptions:
      - 顶边和底边中恰有一边出现唯一的 4 格连续异色像素。
      - 删除该色带后恰有 4 个 4 连通拼块。
      - 存在唯一一个拼块到图例所在边的距离最小。
      - 色带顺序恒按左到右对应 TL,TR,BL,BR，不需要再推断旋转或镜像。
    generator_guarantees:
      - 只在顶边或底边放置一条长度恰为 4 的色带，其他边保持背景。
      - 图例拼块与色带之间固定留 1 格空隙，且其余拼块被排除在靠边保留带之外，故最近关系唯一。
      - 所有拼块保持单色，从而不会意外生成“旧题式多色图例块”。
  placement_legality:
    - 色带严格放在第 0 行或最后 1 行，且避开角落；因此不会越界，也不会与内部拼块连通。
    - 图例拼块放在距边缘 1 格的位置，并要求其列区间与 1x4 色带有可见重叠，保证视觉配对清楚。
    - 其他拼块用 `random_free_location` 放置，且在图例边缘附近预留整条禁放带，避免重叠和错误靠近。
  visual_observability:
    - 人类解题时首先会看到唯一触边的 4 彩色小条，而不是某个内部多色块。
    - 多个样例中顶边/底边位置会变化，但“边缘 1x4 小条 + 最近拼块 + 输出象限换色”的稳定规律保持不变。
    - 因为所有拼块在输入中都是单色，解题者能明显感受到“图例已从拼块里移走”这一融合变化。
  visual_cleanliness:
    - 图例边缘、最近拼块和其余拼块分区明确，没有叠加覆盖。
    - 色带只有 4 格，不会像长边框或长刻度那样抢占过多画面。
    - 输出仍是紧凑裁剪后的正方形复合体，和原题视觉终态一致，便于比较输入输出关系。
  color_inference:
    - 仍只使用 4 个主体色 + 4 个图例色，颜色总复杂度与原题一致。
    - 图例色的语义完全由可见顺序给出：左到右就是 TL,TR,BL,BR，不依赖颜色编号、面积或隐藏排序。
"""

import math
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    out = []
    for h in range(1, int(n**0.5) + 1):
        if n % h == 0:
            w = n // h
            out.append((h, w))
            if h != w:
                out.append((w, h))
    return out


def _normalize_coords(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, _ = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


def _dihedral_orientations(
    coords: tuple[tuple[int, int], ...],
) -> list[tuple[tuple[tuple[int, int], ...], int, int, int, bool]]:
    """
    Unique orientations under rotations + mirror (D4).
    Returned as (coords, h, w, rot_k, flipped), where the transform is:
      - if flipped: horizontal flip (mirror left-right)
      - then rot_k times 90° clockwise rotation
    coords are normalized to start at (0,0).
    """
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


def _parse_edge_strip(vals: np.ndarray, bg: int) -> tuple[int, int, np.ndarray] | None:
    idxs = [i for i, v in enumerate(vals) if int(v) != bg]
    if len(idxs) != 4:
        return None
    lo = min(idxs)
    hi = max(idxs)
    if hi - lo != 3:
        return None
    seq = np.asarray(vals[lo : hi + 1], dtype=int)
    if len(set(map(int, seq))) != 4:
        return None
    return int(lo), int(hi), seq.copy()


def _extract_edge_legend(g: np.ndarray, bg: int) -> tuple[str, int, int, np.ndarray]:
    """
    Find the unique 1x4 legend strip on the outermost top or bottom row.

    Returns (side, lo, hi, seq), where seq is read left-to-right and corresponds to
    [top-left, top-right, bottom-left, bottom-right].
    """
    h, _ = g.shape
    candidates: list[tuple[str, int, int, np.ndarray]] = []
    for side, vals in (("top", g[0, :]), ("bottom", g[h - 1, :])):
        parsed = _parse_edge_strip(np.asarray(vals, dtype=int), bg)
        if parsed is None:
            continue
        lo, hi, seq = parsed
        candidates.append((side, lo, hi, seq))
    assert len(candidates) == 1, "grid must contain exactly one valid edge legend strip"
    return candidates[0]


def _sequence_to_legend_map(seq: np.ndarray, k: int) -> np.ndarray:
    assert int(seq.size) == k * k, "legend strip must have k*k entries"
    flat = [int(x) for x in np.asarray(seq, dtype=int).ravel()]
    return np.asarray([[flat[0], flat[1]], [flat[2], flat[3]]], dtype=int)


def _fill_holes_in_component(
    g: np.ndarray, pts: np.ndarray, bg: int
) -> tuple[tuple[tuple[int, int], ...], int]:
    """
    Treat background 'holes' fully enclosed inside the component bbox as part of the piece.
    Returns (filled_pts, hole_count), where filled_pts is a sorted tuple of absolute coords.
    """
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


def _assemble_exact_cover(
    pieces: list[dict],
    legend_idx: int,
    legend_map: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Tile all pieces into a full rectangle (no holes), allowing rotations + reflections.
    Recolor each piece by quantizing its centroid into a k×k grid, using legend_map.
    """
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(h, w) for (h, w) in dims if h >= max_h and w >= max_w]
    assert len(dims) == 1, "piece areas and bounds must imply a unique output rectangle"

    for H, W in dims:
        nbits = H * W
        all_mask = (1 << nbits) - 1

        other_pis = [i for i in range(len(pieces)) if i != legend_idx]
        placements_other: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
        for pi in other_pis:
            opts: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
            for coords, ph, pw, _, _ in pieces[pi]["oris"]:
                for r0 in range(0, H - ph + 1):
                    for c0 in range(0, W - pw + 1):
                        m = 0
                        for dr, dc in coords:
                            idx = (int(r0) + int(dr)) * int(W) + (int(c0) + int(dc))
                            m |= 1 << idx
                        opts.append((m, int(r0), int(c0), coords))
            placements_other.append(opts)

        def solve_with_fixed_legend(
            legend_coords: tuple[tuple[int, int], ...],
        ) -> dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]] | None:
            legend_mask = 0
            for dr, dc in legend_coords:
                idx = int(dr) * int(W) + int(dc)
                legend_mask |= 1 << idx

            filtered: list[list[tuple[int, int, int, tuple[tuple[int, int], ...]]]] = []
            for opts in placements_other:
                filtered.append([pl for pl in opts if (pl[0] & legend_mask) == 0])

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
                for k2 in sorted(remaining):
                    opts_k = []
                    for pl in filtered[k2]:
                        m = pl[0]
                        if (m >> cell) & 1 and (m & used) == 0:
                            opts_k.append(pl)
                    if opts_k:
                        candidates.append((len(opts_k), k2, opts_k))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, k2, opts_k in candidates:
                    remaining.remove(k2)
                    pi = other_pis[k2]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(k2)
                return False

            if not search(legend_mask):
                return None
            return chosen

        chosen = None
        for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
            assert ph <= H and pw <= W, "legend orientation must fit the unique output rectangle"
            sol = solve_with_fixed_legend(coords)
            if sol is None:
                continue
            chosen = sol
            break
        assert chosen is not None, "exact-cover assembly failed: no valid legend orientation"

        piece_cell: dict[int, tuple[int, int]] = {}
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rs = [r0 + dr for dr, _ in coords]
            cs = [c0 + dc for _, dc in coords]
            cr = float(sum(rs)) / float(len(rs))
            cc = float(sum(cs)) / float(len(cs))
            rr = int(cr * float(k) / float(H))
            cc2 = int(cc * float(k) / float(W))
            rr = max(0, min(int(k) - 1, rr))
            cc2 = max(0, min(int(k) - 1, cc2))
            piece_cell[pi] = (rr, cc2)

        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            rr, cc2 = piece_cell[pi]
            col = int(legend_map[rr, cc2])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out

    raise AssertionError("unreachable rectangle assembly branch")


def _distance_to_side(pts: np.ndarray, side: str, shape: tuple[int, int]) -> int:
    h, _ = shape
    r0, _, r1, _ = bbox_coords(pts)
    if side == "top":
        return int(r0)
    if side == "bottom":
        return int(h - 1 - r1)
    raise ValueError(f"unknown side: {side}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, recoloring each from a tiny border legend strip.

    Input structure:
    - bg is the most common color.
    - Four monochrome non-bg sprites are scattered in the interior and must tile a square exactly.
    - A separate legend is not inside any sprite: it is a bare 1x4 strip of four distinct pixels on the outermost top or bottom row, with no frame around it.
    - The sprite closest to that edge strip plays the old legend piece's assembly role.

    Rule:
    - Find the unique 4-pixel border strip, read it left-to-right, and unpack the sequence as [TL, TR, BL, BR].
    - Remove the strip from the grid, then tile the remaining four pieces into the unique tight rectangle with rotations/reflections allowed.
    - Anchor the piece nearest the strip at the assembled top-left, and recolor each assembled piece by the legend color selected from the 2x2 quadrant reached by its centroid.

    Key insight:
    - The puzzle no longer marks the anchor with a multicolor sprite; instead the only reliable cue is the tiny unframed border strip, whose 1D order must be expanded back into the 2D quadrant palette before the jigsaw assembly step.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, _ = g.shape

    side, lo, hi, legend_seq = _extract_edge_legend(g, bg)
    k = int(round(len(legend_seq) ** 0.5))
    assert k * k == len(legend_seq), "legend strip length must be a perfect square"
    legend_map = _sequence_to_legend_map(legend_seq, k)

    piece_mask = g != bg
    if side == "top":
        piece_mask[0, lo : hi + 1] = False
    else:
        piece_mask[h - 1, lo : hi + 1] = False

    comps = components_mask(piece_mask, connectivity=4)
    assert comps, "grid must contain pieces after removing the edge legend"
    assert len(comps) == k * k, "piece count must match the decoded legend strip"

    dists = [_distance_to_side(pts, side, g.shape) for pts in comps]
    min_dist = min(dists)
    legend_candidates = [i for i, d in enumerate(dists) if d == min_dist]
    assert len(legend_candidates) == 1, "legend anchor piece must be uniquely closest to the edge strip"
    legend_idx = legend_candidates[0]

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        pieces.append({"oris": _dihedral_orientations(rel_norm), "area": len(filled_abs)})

    return _assemble_exact_cover(pieces, legend_idx, legend_map, k)


# --------------- generator (o5x) ---------------

# === CFG ===
SIDE_CHOICES = [8, 10, 12]
K = 2
NUM_PIECES = K * K
NUM_COLORS = 8
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
SCATTER_BORDER = 2
SCATTER_PADDING = 1
EDGE_SIDES = ("top", "bottom")
EDGE_MARGIN = 1
EDGE_STRIP_LEN = NUM_PIECES
LEGEND_GAP = 1
LEGEND_EXCLUSION_BAND = 5
GEN_RETRIES = 300


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


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k2 in range(4):
        rot = np.rot90(arr, k2)
        for oriented in (rot, np.fliplr(rot)):
            r0, c0, r1, c1 = bbox_coords(np.argwhere(oriented > 0))
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _paint_edge_strip(grid: np.ndarray, side: str, start: int, seq: np.ndarray) -> None:
    if side == "top":
        grid[0, start : start + EDGE_STRIP_LEN] = seq
    elif side == "bottom":
        grid[-1, start : start + EDGE_STRIP_LEN] = seq
    else:
        raise ValueError(f"unsupported edge side: {side}")


def _legend_piece_position(grid_size: int, edge_side: str, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    c_lo = 1
    c_hi = grid_size - w - 2
    if c_lo > c_hi:
        return None
    if edge_side == "top":
        r0 = LEGEND_GAP + 1
        if r0 + h >= grid_size - 1:
            return None
        return int(r0), random.randint(c_lo, c_hi)
    if edge_side == "bottom":
        r0 = grid_size - h - (LEGEND_GAP + 1)
        if r0 < 1:
            return None
        return int(r0), random.randint(c_lo, c_hi)
    return None


def _strip_start_candidates(grid_size: int, span: int, bbox_lo: int, bbox_hi: int) -> list[int]:
    lo = EDGE_MARGIN
    hi = grid_size - span - EDGE_MARGIN
    if lo > hi:
        return []
    starts = []
    for start in range(lo, hi + 1):
        end = start + span - 1
        if start <= bbox_hi and end >= bbox_lo:
            starts.append(int(start))
    return starts


def _reserve_band(occupied: np.ndarray, edge_side: str) -> None:
    if edge_side == "top":
        occupied[:LEGEND_EXCLUSION_BAND, :] = 1
    elif edge_side == "bottom":
        occupied[-LEGEND_EXCLUSION_BAND:, :] = 1
    else:
        raise ValueError(f"unsupported edge side: {edge_side}")


def _scatter_pieces(
    masks: list[np.ndarray], body_colors: list[int], legend_seq: np.ndarray, out_side: int
) -> np.ndarray | None:
    total = out_side * out_side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    edge_side = random.choice(EDGE_SIDES)
    legend_oriented = _random_orientation(_crop_mask(masks[0]))
    legend_pos = _legend_piece_position(grid_size, edge_side, legend_oriented.shape)
    if legend_pos is None:
        return None
    lr0, lc0 = legend_pos
    lh, lw = legend_oriented.shape

    starts = _strip_start_candidates(grid_size, EDGE_STRIP_LEN, lc0, lc0 + lw - 1)
    if not starts:
        return None
    strip_start = random.choice(starts)
    _paint_edge_strip(grid, edge_side, strip_start, legend_seq)
    if edge_side == "top":
        occupied[0, strip_start : strip_start + EDGE_STRIP_LEN] = 1
    else:
        occupied[-1, strip_start : strip_start + EDGE_STRIP_LEN] = 1

    legend_patch = np.zeros(legend_oriented.shape, dtype=int)
    legend_patch[legend_oriented] = body_colors[0]
    paste(grid, legend_patch, lr0, lc0, transparent=0)
    occupied[lr0 : lr0 + lh, lc0 : lc0 + lw] = 1

    _reserve_band(occupied, edge_side)
    occupied[lr0 : lr0 + lh, lc0 : lc0 + lw] = 1

    for pi in random.sample(range(1, NUM_PIECES), NUM_PIECES - 1):
        oriented = _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        try:
            box = np.ones(oriented.shape, dtype=int)
            r0, c0 = random_free_location(
                occupied,
                box,
                transparent=0,
                border=SCATTER_BORDER,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        occupied[r0 : r0 + oriented.shape[0], c0 : c0 + oriented.shape[1]] = 1
        paste(grid, patch, r0, c0, transparent=0)
    return grid


def _gen_attempt() -> np.ndarray | None:
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

    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    if len(set(qcells)) != NUM_PIECES:
        return None

    colors = random.sample(range(1, 10), NUM_COLORS)
    body_colors = colors[:NUM_PIECES]
    palette_colors = colors[NUM_PIECES:]
    random.shuffle(palette_colors)
    legend_seq = np.asarray(palette_colors, dtype=int)
    legend_map = _sequence_to_legend_map(legend_seq, K)

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        rr, cc = qcells[pi]
        expected[mask] = legend_map[rr, cc]

    grid = _scatter_pieces(masks, body_colors, legend_seq, side)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate f560132c combo input")
