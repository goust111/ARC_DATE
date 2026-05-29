# Variant of f560132c
# Dimension: D1 — 匹配条件：形状匹配 vs 其他匹配逻辑 (match_cond)
# Change: s.shape matches d.region.shape — 每个单色精灵的形状恰好对应复合矩形中某个象限区域的形状... → 连接器颜色匹配

import math
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4, paste, random_free_location


# Design analysis for the D1 variant
# ----------------------------------
# variant_spec:
#   preserved_backbone:
#     - 仍然是 4 个 4-connected piece，允许 D4 变换，exact-cover 拼成唯一的 S*S 方形。
#     - 仍然有一个 legend piece；输入仍由 output-first generator 反推并用 main() 验证。
#   core_change:
#     old:
#       - piece 的输出颜色由装配后落入哪个 2*2 空间象限决定。
#       - 只要 exact-cover 唯一，piece→颜色 的对应可以从几何位置直接读出。
#     new:
#       - legend 内部不只含 2*2 palette，还含 4 个“象限连接器像素”。
#       - 每个 piece 自身也带 1 个边界连接器像素，颜色与 legend 某个象限连接器一致。
#       - 输出颜色不再按装配后的空间象限，而是按“piece 连接器颜色 → legend 对应象限”的映射决定。
#   concrete_pixels:
#     - 所有 piece 共用同一种 body color；这样 1(body)+4(palette)+4(connector)=9 个非背景色，正好落在 ARC 调色板内。
#     - legend 内嵌一个 4*4 微结构：
#         body body  cTR body
#         cTL  pTL  pTR body
#         body pBL  pBR cBR
#         body cBL  body body
#       其中 p** 为 palette 四格，c** 为与这四格一一对应的连接器颜色。
#     - legend 另外还有 1 个边界连接器像素，表示“legend 这块 piece 自己匹配哪个象限”。
#     - 其余 3 个 piece 也各有 1 个边界连接器像素。
#   behavior_trace:
#     - 原题在着色阶段：exact-cover 结束后，用 piece 质心量化到 2*2 网格，再从 legend_map 取色。
#     - 变体在着色阶段：exact-cover 结束后，先读每个 piece 的连接器颜色，再用 legend 的 connector->palette 映射取色。
#     - 差异点：同一块 piece 即使装配在右下角，只要连接器匹配的是左上象限，就应涂成左上 palette 颜色。
#   anti_degeneracy:
#     - generator 强制 piece 的 connector-assignment 与其真实空间象限至少有 2 处不一致。
#     - 因此旧版“按空间象限着色”的逻辑在这些样本上通常会给出不同输出，连接器变量真正起作用。
#   implicit_assumptions_audit:
#     - 原 main() 依赖“颜色只用于识别 legend，piece 的最终颜色来自空间位置”。这个前提已失效，必须移除 centroid 着色。
#     - 原 main() 默认每块 piece 可以看作单色形状；现在必须从 component 内部读出少量标记色，但这些标记不能破坏 component 连通性。
#     - 新 solver 额外假设：
#         1. legend 的 4*4 微结构唯一存在且可从 D4 方向中识别。
#         2. 非 legend piece 恰有 1 个非 body 的连接器颜色。
#         3. legend 除微结构中的 4 个 map-connector 外，恰有 1 个额外 connector 作为该 piece 自身标记。
#       generator 会显式构造并验证这些条件。
#   placement_legality:
#     - 连接器像素直接涂在 piece 已占据的边界单元上，不引入额外 protrusion，因此不会破坏 exact-cover 面积。
#     - legend 的微结构只嵌在 4*4 全填充子块内，避免 palette/connectors 跨出 piece。
#     - piece 散布仍使用 random_free_location + bbox padding，互不重叠且不越界。
#   visual_observability:
#     - 由于 body color 统一，解题者会更自然地把注意力放到少量 connector 色与 legend 内的 4 个 connector 色上。
#     - 输出中颜色排列不再总是与几何象限一致，从 2-3 组样例就能看出“看 connector，不看位置”。


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
GEN_RETRIES = 300

PATTERN_SIZE = 4
PALETTE_POS = ((1, 1), (1, 2), (2, 1), (2, 2))  # TL, TR, BL, BR
CONNECTOR_POS = ((1, 0), (0, 2), (3, 1), (2, 3))  # TL, TR, BL, BR


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
    norm, h, w = _normalize_coords(coords)
    rot = [(c, h - 1 - r) for r, c in norm]
    return _normalize_coords(rot)


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, w = _normalize_coords(coords)
    flp = [(r, w - 1 - c) for r, c in norm]
    return _normalize_coords(flp)


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


BODY_POS = tuple((r, c) for r in range(PATTERN_SIZE) for c in range(PATTERN_SIZE) if (r, c) not in set(PALETTE_POS) | set(CONNECTOR_POS))


def _extract_legend_signature(
    g: np.ndarray,
    pts: np.ndarray,
    bg: int,
) -> tuple[dict[int, int], int, int, np.ndarray]:
    """
    Find the 4x4 legend micro-pattern under any D4 orientation.
    Returns:
      - connector_color -> palette_color
      - legend piece's own connector color
      - shared body color
      - visual 2x2 legend map as it appears in the detected window
    """
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
            body = body_vals[0]
            if body == bg:
                continue
            pal_vals = [int(window[r, c]) for r, c in PALETTE_POS]
            conn_vals = [int(window[r, c]) for r, c in CONNECTOR_POS]
            if len(set(pal_vals)) != 4 or len(set(conn_vals)) != 4:
                continue
            if body in pal_vals or body in conn_vals:
                continue
            if set(pal_vals) & set(conn_vals):
                continue
            connector_to_palette = {int(conn_vals[i]): int(pal_vals[i]) for i in range(4)}
            abs_conn = {(r0 + rr + r, c0 + cc + c) for r, c in CONNECTOR_POS}
            legend_map = np.array(
                [
                    [int(window[PALETTE_POS[0]]), int(window[PALETTE_POS[1]])],
                    [int(window[PALETTE_POS[2]]), int(window[PALETTE_POS[3]])],
                ],
                dtype=int,
            )
            candidates.append((connector_to_palette, abs_conn, int(body), legend_map))

    assert len(candidates) == 1, 'legend must contain exactly one valid connector+palette micro-pattern'
    connector_to_palette, abs_conn, body_color, legend_map = candidates[0]
    connector_colors = set(connector_to_palette)

    extra = [(int(r), int(c)) for r, c in pts if (int(r), int(c)) not in abs_conn and int(g[r, c]) in connector_colors]
    assert len(extra) == 1, 'legend must have exactly one extra connector marker for the legend piece'
    legend_piece_connector = int(g[extra[0][0], extra[0][1]])
    return connector_to_palette, legend_piece_connector, body_color, legend_map


def _solve_exact_cover(
    pieces: list[dict],
    legend_idx: int,
) -> tuple[int, int, dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]]]:
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    assert len(dims) == 1, 'piece areas and bounds must imply a unique output rectangle'

    H, W = dims[0]
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
            for k in sorted(remaining):
                opts_k = []
                for pl in filtered[k]:
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

        if not search(legend_mask):
            return None
        return chosen

    chosen = None
    for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
        assert ph <= H and pw <= W, 'legend orientation must fit the unique output rectangle'
        sol = solve_with_fixed_legend(coords)
        if sol is not None:
            chosen = sol
            break
    assert chosen is not None, 'exact-cover assembly failed: no valid legend orientation'
    return int(H), int(W), chosen


def _render_solution(
    H: int,
    W: int,
    chosen: dict[int, tuple[int, int, int, tuple[tuple[int, int], ...]]],
    pieces: list[dict],
) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for pi, pl in chosen.items():
        _, r0, c0, coords = pl
        col = int(pieces[pi]["out_color"])
        for dr, dc in coords:
            out[r0 + dr, c0 + dc] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw pieces into a rectangle, then recolor each piece by connector-color lookup.

    Input structure:
    - Background plus 4 disconnected foreground pieces that still exact-cover a unique square under D4 moves.
    - One piece is the legend: inside it, a 2x2 palette is paired with 4 nearby connector pixels, and the legend piece also has its own extra connector mark.
    - Every non-legend piece is mostly one body color, with exactly one boundary connector pixel whose color matches one legend connector.

    Rule:
    - Reassemble all 4 pieces into the tight square, keeping the legend piece anchored at the square's top-left as in the original task family.
    - Read the legend's connector-to-palette correspondence from the embedded micro-pattern.
    - Recolor each assembled piece uniformly with the palette color selected by that piece's connector color, not by its final geometric quadrant.

    Key insight: The rectangle assembly is still geometric, but the color assignment is no longer spatial; the tiny connector colors are the actual matching signal.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, 'grid must contain at least one non-background component'

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps) == NUM_PIECES, 'this variant expects exactly 4 pieces'

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, 'legend component must be uniquely identified by distinct colors'
    legend_idx = legend_candidates[0]

    connector_to_palette, legend_piece_connector, body_color, _legend_map = _extract_legend_signature(g, comps[legend_idx], bg)

    pieces: list[dict] = []
    for i, pts in enumerate(comps):
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)

        if i == legend_idx:
            mark_color = legend_piece_connector
        else:
            extra_colors = {int(g[r, c]) for r, c in pts if int(g[r, c]) not in (bg, body_color)}
            assert len(extra_colors) == 1, 'non-legend piece must have exactly one connector color'
            mark_color = next(iter(extra_colors))

        assert mark_color in connector_to_palette, 'piece connector must match one legend connector'
        pieces.append(
            {
                "oris": oris,
                "area": len(filled_abs),
                "mark_color": int(mark_color),
                "out_color": int(connector_to_palette[int(mark_color)]),
            }
        )

    H, W, chosen = _solve_exact_cover(pieces, legend_idx)
    return _render_solution(H, W, chosen, pieces)


def _area_bounds(side):
    total = side * side
    return max(AREA_ABS_LO, math.ceil(AREA_FRAC_LO * total)), min(AREA_ABS_HI, math.floor(AREA_FRAC_HI * total))


def _has_hole(mask):
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


def _forces_unique_square(masks, side):
    bbs = [bbox_coords(np.argwhere(m)) for m in masks]
    max_h = max(r1 - r0 + 1 for r0, _, r1, _ in bbs)
    max_w = max(c1 - c0 + 1 for _, c0, _, c1 in bbs)
    dims = [(h, w) for h, w in _factor_pairs(side * side) if h >= max_h and w >= max_w]
    return dims == [(side, side)]


def _make_legend_mask(side, min_area, max_area):
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


def _partition_rest(rest):
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

    def add_frontier(i, r, c):
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


def _crop_mask(mask):
    r0, c0, r1, c1 = bbox_coords(np.argwhere(mask))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_orientation(mask):
    arr = mask.astype(int)
    opts = []
    seen = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        for oriented in (rot, np.fliplr(rot)):
            r0, c0, r1, c1 = bbox_coords(np.argwhere(oriented > 0))
            crop = oriented[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = (crop.shape, tuple(crop.ravel()))
            if key not in seen:
                seen.add(key)
                opts.append(crop)
    return random.choice(opts)


def _legend_block_spots(mask):
    spots = []
    h, w = mask.shape
    for r in range(0, h - PATTERN_SIZE + 1):
        for c in range(0, w - PATTERN_SIZE + 1):
            if mask[r : r + PATTERN_SIZE, c : c + PATTERN_SIZE].all():
                spots.append((r, c))
    return spots


def _boundary_cells(mask, forbidden: set[tuple[int, int]] | None = None):
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


def _paint_simple_piece(mask: np.ndarray, body_color: int, connector_color: int) -> np.ndarray | None:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(body_color)
    boundary = _boundary_cells(mask)
    if not boundary:
        return None
    r, c = random.choice(boundary)
    patch[r, c] = int(connector_color)
    return patch


def _paint_legend_piece(
    mask: np.ndarray,
    body_color: int,
    legend_map: np.ndarray,
    connector_colors: list[int],
    legend_piece_cell_id: int,
) -> np.ndarray | None:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(body_color)
    spots = _legend_block_spots(mask)
    random.shuffle(spots)
    for r0, c0 in spots:
        forbidden = {(r, c) for r in range(r0, r0 + PATTERN_SIZE) for c in range(c0, c0 + PATTERN_SIZE)}
        boundary = _boundary_cells(mask, forbidden=forbidden)
        if not boundary:
            continue

        for idx, (dr, dc) in enumerate(PALETTE_POS):
            rr = 0 if idx < 2 else 1
            cc = idx % 2
            patch[r0 + dr, c0 + dc] = int(legend_map[rr, cc])
        for idx, (dr, dc) in enumerate(CONNECTOR_POS):
            patch[r0 + dr, c0 + dc] = int(connector_colors[idx])

        mr, mc = random.choice(boundary)
        patch[mr, mc] = int(connector_colors[legend_piece_cell_id])
        return patch
    return None


def _scatter_pieces(masks, body_color, legend_map, connector_colors, piece_cell_ids, side):
    total = side * side
    nlo = max(INPUT_SIZE_LO, math.ceil(math.sqrt(total / DENSITY_HI)))
    nhi = min(INPUT_SIZE_HI, math.floor(math.sqrt(total / DENSITY_LO)))
    grid_size = random.randint(nlo, nhi)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    occupied = np.zeros_like(grid)

    order = random.sample(range(NUM_PIECES), NUM_PIECES)
    for pi in order:
        oriented = _crop_mask(masks[pi]) if pi == 0 else _random_orientation(_crop_mask(masks[pi]))
        if pi == 0:
            patch = _paint_legend_piece(oriented, body_color, legend_map, connector_colors, piece_cell_ids[pi])
        else:
            patch = _paint_simple_piece(oriented, body_color, connector_colors[piece_cell_ids[pi]])
        if patch is None:
            return None
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


def _cell_ids_from_masks(masks, side):
    ids = []
    for mask in masks:
        pts = np.argwhere(mask)
        rr = min(1, int(pts[:, 0].mean() * K / side))
        cc = min(1, int(pts[:, 1].mean() * K / side))
        ids.append(int(rr * K + cc))
    return ids


def _gen_attempt():
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

    spatial_ids = _cell_ids_from_masks(masks, side)
    if len(set(spatial_ids)) != NUM_PIECES:
        return None

    piece_cell_ids = list(range(NUM_PIECES))
    random.shuffle(piece_cell_ids)
    if sum(int(a != b) for a, b in zip(piece_cell_ids, spatial_ids)) < 2:
        return None

    colors = random.sample(range(1, 10), 9)
    body_color = int(colors[0])
    palette_colors = colors[1:5]
    connector_colors = colors[5:9]
    random.shuffle(palette_colors)
    random.shuffle(connector_colors)
    legend_map = np.array(palette_colors, dtype=int).reshape(K, K)

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        cid = piece_cell_ids[pi]
        rr, cc = divmod(cid, K)
        expected[mask] = int(legend_map[rr, cc])

    grid = _scatter_pieces(masks, body_color, legend_map, connector_colors, piece_cell_ids, side)
    if grid is None:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError('failed to generate f560132c input')
