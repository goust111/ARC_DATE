# Variant of f560132c
# Dimension: D5 — 符号指令：图例的编码方式与结构 (symbol_instructions)
# Change: 图例精灵包含2x2彩色像素嵌入在单色框架中，每个像素的位置（左上、右上、左下、右下）映射到复合体中对应象限的重着色颜色... → 图例为线性色带而非2x2矩阵

"""
design_analysis:
  concretization_and_trace:
    input_look:
      - 仍然是 4 个散落的 4 连通拼块，其中恰有 1 个图例块包含多种颜色。
      - 原题图例块内部是 2x2 彩色小方阵；本变体改成 4 格直线色带，可能是 1x4 水平条，也可能是 4x1 竖直条。
      - 色带四格的可见顺序固定为 TL, TR, BL, BR；若为水平条就按从左到右读取，若为竖直条就按从上到下读取。
      - 输出仍然是将 4 块恰好拼成正方形后，按每块质心落入的 2x2 象限进行统一重着色。
    behavior_trace:
      original:
        - solver 在图例 bbox 内滑动搜索 2x2 窗口，直接把窗口像素当作 [[TL,TR],[BL,BR]]。
      variant:
        - solver 改为搜索长度为 4 的水平/竖直窗口。
        - 找到色带后先读出线性序列 [c0,c1,c2,c3]。
        - 再显式展开成 [[c0,c1],[c2,c3]]，随后沿用原来的 exact-cover + centroid 象限着色。
      difference:
        - 变化只发生在“图例解码”环节；拼图装配、象限判定、输出裁剪逻辑不变。
  complexity_calibration:
    - 仍只保留两种可见编码档位：水平色带或竖直色带，避免引入过多自由度。
    - 每个样例里色带长度恒为 4，并且四种颜色互异，保证人眼能看到“线性顺序展开到 2x2”的稳定规律。
    - 若继续允许 2x2 方阵，就会退化回原题；因此 generator 只嵌入线性色带。
  implicit_assumption_audit:
    original_main_dependencies:
      - "_extract_legend_map(..., k) 依赖图例里存在 kxk 彩色块。变体下此假设失效，必须换成 1x(k*k) / (k*k)x1 搜索。"
      - "legend_map 直接按二维索引使用，原题的二维布局天然可读；变体下必须先做线性到二维的展开。"
      - "原 generator 不旋转图例块，因此 solver 读取到的图例朝向就是规范朝向。这个隐式前提原题中对 2x2 方阵几乎不可见，但对线性色带非常关键。"
    retained_assumptions:
      - 仍假设图例块在输入中保持规范朝向；否则线性色带在旋转/镜像后会产生顺序翻转，solver 需要额外恢复姿态，这已超出本次 D5 最小变更。
      - generator 通过不随机旋转图例块来保证该前提成立。
  placement_legality:
    - 色带只能放在图例块 bbox 的内部位置，且其外侧一圈仍属于该拼块，从而四个彩色像素与图例主体保持同一连通分量。
    - 水平条要求存在 3x6 全填充区域；竖直条要求存在 6x3 全填充区域。generator 只选择满足对应条件的图例形状和放置点。
    - 散落放置时继续使用 random_free_location，保证组件互不重叠、不中断、不过界。
  visual_observability:
    - 变体最显著的视觉差异就是图例不再是 2x2 色块，而是一条 4 格直线；这一点在 2-3 个样例上肉眼即可稳定观察。
    - 水平/竖直两种方向都会出现，促使解题者归纳“按可见阅读顺序展开”为 2x2 象限，而不是把方向当成无意义扰动。
  visual_cleanliness:
    - 仍然只有一层“主体色 + 内嵌图例色”的叠加，图例块内部结构简单，不会与其余散落拼块发生视觉缠绕。
    - 色带被限制在图例块内部而不是贴边，避免看起来像独立的小组件。
  color_inference:
    - 仍使用 4 个主体色 + 4 个图例色，总颜色复杂度与原题相同。
    - 四个图例色的排序完全由可见位置（左到右 / 上到下）决定，不使用颜色 id、面积、排序等不可见因素。
  post_change_main_assumptions:
    - solver 假设图例组件中最常见的非背景色是主体色；generator 通过“主体面积远大于 4 个图例像素”保证这一点。
    - solver 假设恰有一个不含主体色且拥有 4 个互异颜色的长度 4 窗口；generator 通过内部嵌入单条色带保证这一点。
"""

import math
import random
import sys
from collections import Counter

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


def _extract_legend_sequence(g: np.ndarray, pts: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Inside the legend component bbox, find a 1xN or Nx1 color strip.
    Preference: a strip made entirely of non-body colors with maximal color diversity.
    Ties go to topmost, then leftmost, then horizontal before vertical.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert seq_len > 0 and (hh >= seq_len or ww >= seq_len), "legend component must fit a linear color strip"

    comp_vals = [int(g[r, c]) for r, c in pts]
    body_color = Counter(comp_vals).most_common(1)[0][0]

    best_key = None
    best_seq = None
    for orient, rr, cc in _legend_strip_windows(sub, seq_len):
        if orient == "h":
            window = sub[rr, cc : cc + seq_len]
        else:
            window = sub[rr : rr + seq_len, cc]
        vals = [int(x) for x in window]
        distinct = len(set(vals))
        body_hits = sum(int(v == body_color) for v in vals)
        key = (-body_hits, distinct, -rr, -cc, int(orient == "h"))
        if best_key is None or key > best_key:
            best_key = key
            best_seq = vals

    assert best_seq is not None, "legend component must contain a valid linear color strip"
    assert len(set(best_seq)) == seq_len, "legend strip must expose four distinct colors"
    return np.asarray(best_seq, dtype=int)


def _legend_strip_windows(sub: np.ndarray, seq_len: int) -> list[tuple[str, int, int]]:
    hh, ww = sub.shape
    out: list[tuple[str, int, int]] = []
    if ww >= seq_len:
        for rr in range(hh):
            for cc in range(ww - seq_len + 1):
                out.append(("h", rr, cc))
    if hh >= seq_len:
        for rr in range(hh - seq_len + 1):
            for cc in range(ww):
                out.append(("v", rr, cc))
    return out


def _sequence_to_legend_map(seq: np.ndarray, k: int) -> np.ndarray:
    assert int(seq.size) == k * k, "linear legend sequence must have k*k entries"
    flat = [int(x) for x in np.asarray(seq, dtype=int).ravel()]
    return np.asarray([[flat[0], flat[1]], [flat[2], flat[3]]], dtype=int)


def _fill_holes_in_component(g: np.ndarray, pts: np.ndarray, bg: int) -> tuple[tuple[tuple[int, int], ...], int]:
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
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, recoloring each from a legend sprite's linear palette strip.

    Input structure:
    - bg is the most common color.
    - Four scattered non-bg sprites act as jigsaw pieces; one of them is the legend and has multiple colors.
    - The legend does not contain a 2x2 palette block anymore: instead it hides a 1x4 horizontal strip or 4x1 vertical strip of four distinct colors inside a monochrome frame.

    Rule:
    - Reassemble all pieces into the unique tight rectangle, allowing rotations and reflections for the tiling step.
    - Read the legend strip in visible order: left-to-right for a horizontal strip, top-to-bottom for a vertical strip.
    - Expand that linear order into quadrant colors [TL, TR, BL, BR], then recolor each assembled piece by the color of the quadrant containing its centroid.

    Key insight:
    - The only new reasoning step is decoding a 1D legend into the same 2x2 spatial palette the original puzzle used; once unfolded, the rest of the assembly logic is unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain at least one non-background component"

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps), "piece count must be a perfect square to determine k"

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, "legend component must be uniquely identified by distinct colors"
    legend_idx = legend_candidates[0]

    legend_seq = _extract_legend_sequence(g, comps[legend_idx], k * k)
    legend_map = _sequence_to_legend_map(legend_seq, k)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append({"oris": oris, "area": len(filled_abs)})

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
LEGEND_STRIP_LEN = NUM_PIECES
PARTITION_GROWTH_EXP = 1.45
PARTITION_FRONTIER_EXP = 0.2
PARTITION_SEED_JITTER = 0.2
INPUT_SIZE_LO = 20
INPUT_SIZE_HI = 26
DENSITY_LO = 0.15
DENSITY_HI = 0.25
SCATTER_BORDER = 1
SCATTER_PADDING = 1
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


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _legend_strip_spots(mask: np.ndarray, orientation: str | None = None) -> list[tuple[str, int, int]]:
    spots: list[tuple[str, int, int]] = []
    h, w = mask.shape

    if orientation in (None, "h") and h >= 3 and w >= LEGEND_STRIP_LEN + 2:
        for r in range(1, h - 1):
            for c in range(1, w - LEGEND_STRIP_LEN):
                if mask[r - 1 : r + 2, c - 1 : c + LEGEND_STRIP_LEN + 1].all():
                    spots.append(("h", r, c))

    if orientation in (None, "v") and h >= LEGEND_STRIP_LEN + 2 and w >= 3:
        for r in range(1, h - LEGEND_STRIP_LEN):
            for c in range(1, w - 1):
                if mask[r - 1 : r + LEGEND_STRIP_LEN + 1, c - 1 : c + 2].all():
                    spots.append(("v", r, c))

    return spots


def _make_legend_mask(side: int, min_area: int, max_area: int, strip_dir: str) -> np.ndarray:
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
                    if not (min_area <= area <= max_area):
                        continue
                    if not _legend_strip_spots(_crop_mask(mask), strip_dir):
                        continue
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


def _scatter_pieces(masks: list[np.ndarray], body_colors: list[int], legend_seq: np.ndarray, side: int) -> np.ndarray | None:
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    for pi in random.sample(range(NUM_PIECES), NUM_PIECES):
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        patch = np.zeros(oriented.shape, dtype=int)
        patch[oriented] = body_colors[pi]
        if pi == 0:
            spots = _legend_strip_spots(oriented)
            if not spots:
                return None
            orient, r, c = random.choice(spots)
            if orient == "h":
                patch[r, c : c + LEGEND_STRIP_LEN] = legend_seq
            else:
                patch[r : r + LEGEND_STRIP_LEN, c] = legend_seq
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
    strip_dir = random.choice(["h", "v"])
    legend = _make_legend_mask(side, min_area, max_area, strip_dir)
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
    palette_colors = colors[NUM_COLORS - NUM_PIECES :]
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
    raise RuntimeError("failed to generate f560132c D5 input")
