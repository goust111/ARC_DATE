# Variant of f560132c
# Dimension: D14 — 图例颜色与精灵颜色的映射方式 (meta)
# Change: 图例2x2中的4种颜色与精灵的4种颜色完全不同——图例颜色是目标色，精灵颜色是源色，形成完全的重映射... → 颜色循环移位
#
# Design analysis
# - source_task: none
# - concretization:
#   - 保留原题的 4 个散落拼图块、唯一的多色 legend 块、以及“先拼成唯一方阵”的主流程。
#   - 变体把 legend 的 2x2 色块从“象限目标色表”改成“源色循环表”。
#   - 读取顺序固定为顺时针环：TL -> TR -> BR -> BL -> TL。
#   - 例子：若 legend 色块是 [[2, 5], [8, 7]]，则它定义 2->5, 5->7, 7->8, 8->2。
#     某个输入里本色为 7 的拼图块，拼装后应整体变成 8，而不是读取其所在象限的色块值。
# - behavior trace:
#   - 原逻辑：确定每块在成品方阵中的粗象限 cell，然后直接输出 legend_map[cell]。
#   - 新逻辑：仍需先确定每块是哪一块、放到哪里，但最终颜色来自 successor(piece_body_color)。
#   - 差异点：颜色由“位置 -> 色块值”变为“输入本色 -> legend 定义的后继色”。
# - complexity calibration:
#   - 仍然保留 D4 拼装搜索，避免把题目降成纯颜色替换。
#   - 只使用 4 个非背景色，并让它们同时充当 4 个拼图本色与 legend 循环色，保证循环关系可见。
#   - 生成器继续要求四块的 centroid 落在四个不同 2x2 象限，保持整体结构清晰。
# - implicit assumptions audited:
#   - 原 `_assemble_exact_cover()` 默认 `legend_map[cell]` 就是最终颜色；这个前提在变体里失效，必须显式携带每块的源色。
#   - 原 generator 假设 body 色与 palette 色完全分离；变体下若继续分离，就无法从样例中观察“按本色循环移位”，因此必须共用同一组 4 色。
#   - 原 solver 实际依赖 legend 在输入中不被旋转，因为它不会把提取出的 2x2 色块再跟随拼装朝向旋转；本变体也保留这个前提，
#     generator 继续只随机旋转非 legend 块。
# - placement legality:
#   - 沿用原散布逻辑，块之间留 padding，不重叠、不越界。
#   - legend 的 2x2 色块仍放在 legend 内部并被本体像素包围，solver 能稳定提取。
# - visual observability:
#   - 训练样例里，解题者能同时看到：每块的输入本色、legend 的 2x2 循环顺序、以及输出中每块变成后继色。
#   - 生成器拒绝与“原题按象限直接取色”过于接近的样本，要求至少 2 块的输出色和旧规则不同，避免退化。
# - cleanliness:
#   - 输入里只有 4 个主要组件，legend 仍是唯一多色组件，视觉层次与原题一致，不额外叠加标记。
# - new solver assumptions checked against generator:
#   - k 固定为 2；generator 只生成 4 块。
#   - 4 个拼图本色互异，且恰好等于 legend 2x2 中出现的 4 色；generator 保证这一点。
#   - legend 块的本体色是该组件的众数颜色；由于 legend 面积 >= 11 而色块只占 4 格，generator 保证众数唯一。

import math
import random

import numpy as np

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


def _extract_legend_map(g: np.ndarray, pts: np.ndarray, k: int) -> np.ndarray:
    """
    Inside the legend component's bounding box, find a k×k color map block (may include 0).
    We pick the k×k window with the maximum number of distinct colors; ties go to top-left.
    """
    r0, c0, r1, c1 = bbox_coords(pts)
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    assert k > 0 and hh >= k and ww >= k, 'legend component must fit k×k map'
    best = None
    best_rc = None
    for rr in range(hh - k + 1):
        for cc in range(ww - k + 1):
            window = sub[rr : rr + k, cc : cc + k]
            distinct = int(len(set(map(int, window.ravel()))))
            key = (distinct, -rr, -cc)
            if best is None or key > best:
                best = key
                best_rc = (rr, cc)
    assert best_rc is not None, 'legend component must contain a valid k×k color map window'
    rr, cc = best_rc
    return np.asarray(sub[rr : rr + k, cc : cc + k], dtype=int).copy()


def _extract_body_color(g: np.ndarray, pts: np.ndarray, bg: int) -> int:
    counts: dict[int, int] = {}
    for r, c in pts:
        col = int(g[r, c])
        if col == bg:
            continue
        counts[col] = counts.get(col, 0) + 1
    assert counts, 'component must contain non-background color'
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _cycle_successor_map(legend_map: np.ndarray) -> dict[int, int]:
    assert legend_map.shape == (2, 2), 'this variant expects a 2x2 legend cycle'
    cycle = [
        int(legend_map[0, 0]),
        int(legend_map[0, 1]),
        int(legend_map[1, 1]),
        int(legend_map[1, 0]),
    ]
    assert len(set(cycle)) == 4, 'legend cycle must contain four distinct colors'
    return {cycle[i]: cycle[(i + 1) % 4] for i in range(4)}


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
    successor_map: dict[int, int],
    k: int,
) -> np.ndarray:
    """
    Tile all pieces into a full rectangle (no holes), allowing rotations + reflections.
    Recolor each piece by shifting its original body color forward in the legend-defined cycle.
    """
    assert k == 2, 'this variant is only defined for four pieces / a 2x2 legend cycle'
    total = sum(int(p["area"]) for p in pieces)
    max_h = max(max(h for _, h, _, _, _ in p["oris"]) for p in pieces)
    max_w = max(max(w for _, _, w, _, _ in p["oris"]) for p in pieces)

    dims = _factor_pairs(total)
    dims = [(H, W) for (H, W) in dims if H >= max_h and W >= max_w]
    assert len(dims) == 1, 'piece areas and bounds must imply a unique output rectangle'

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
                for local_idx in sorted(remaining):
                    opts_k = []
                    for pl in filtered[local_idx]:
                        m = pl[0]
                        if (m >> cell) & 1 and (m & used) == 0:
                            opts_k.append(pl)
                    if opts_k:
                        candidates.append((len(opts_k), local_idx, opts_k))
                if not candidates:
                    return False

                candidates.sort(key=lambda x: (x[0], x[1]))
                for _, local_idx, opts_k in candidates:
                    remaining.remove(local_idx)
                    pi = other_pis[local_idx]
                    for pl in opts_k:
                        chosen[pi] = pl
                        if search(used | pl[0]):
                            return True
                    chosen.pop(pi, None)
                    remaining.add(local_idx)
                return False

            if not search(legend_mask):
                return None
            return chosen

        chosen = None
        for coords, ph, pw, _, _ in pieces[legend_idx]["oris"]:
            assert ph <= H and pw <= W, 'legend orientation must fit the unique output rectangle'
            sol = solve_with_fixed_legend(coords)
            if sol is None:
                continue
            chosen = sol
            break
        assert chosen is not None, 'exact-cover assembly failed: no valid legend orientation'

        out = np.zeros((H, W), dtype=int)
        for pi, pl in chosen.items():
            _, r0, c0, coords = pl
            col = int(successor_map[int(pieces[pi]["src_color"])])
            for dr, dc in coords:
                out[r0 + dr, c0 + dc] = col
        return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered polyomino pieces into a rectangle, then recolor each piece by advancing its own input color along the legend's cycle.

    Input structure:
    - bg is the most frequent color.
    - Four non-bg components are scattered across the canvas; three are solid-color pieces.
    - The remaining component is the legend piece: it is mostly one body color, but it contains an interior 2x2 block using the four piece-body colors exactly once.

    Rule:
    - Tile the four pieces into the unique tight rectangle, allowing rotations and reflections.
    - Read the legend's 2x2 block as a clockwise color loop: top-left -> top-right -> bottom-right -> bottom-left -> back to top-left.
    - After assembly, each piece is repainted to the next color in that loop relative to its original body color, rather than by directly copying the color under its assembled quadrant.

    Key insight:
    - The legend is no longer a quadrant-to-color lookup table. It is a permutation key over the four source colors, so you must preserve piece identity through the assembly and then apply the color cycle to each piece's original color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    assert comps, 'grid must contain at least one non-background component'

    k = int(round(len(comps) ** 0.5))
    assert k * k == len(comps), 'piece count must be a perfect square to determine k'

    comp_colors = []
    for pts in comps:
        cols = set(int(g[r, c]) for r, c in pts if int(g[r, c]) != bg)
        comp_colors.append(cols)

    max_distinct = max(len(cols) for cols in comp_colors)
    legend_candidates = [i for i, cols in enumerate(comp_colors) if len(cols) == max_distinct]
    assert len(legend_candidates) == 1, 'legend component must be uniquely identified by distinct colors'
    legend_idx = legend_candidates[0]
    legend_map = _extract_legend_map(g, comps[legend_idx], k)
    successor_map = _cycle_successor_map(legend_map)

    pieces: list[dict] = []
    for pts in comps:
        filled_abs, _ = _fill_holes_in_component(g, pts, bg)
        r0, c0, _, _ = bbox_coords(np.asarray(list(filled_abs)))
        rel = tuple((r - r0, c - c0) for r, c in filled_abs)
        rel_norm, _, _ = _normalize_coords(rel)
        oris = _dihedral_orientations(rel_norm)
        pieces.append({
            "oris": oris,
            "area": len(filled_abs),
            "src_color": _extract_body_color(g, pts, bg),
        })

    piece_colors = [int(p["src_color"]) for p in pieces]
    assert len(set(piece_colors)) == len(pieces), 'piece body colors must be distinct'
    assert set(piece_colors) == set(successor_map), 'legend cycle must use exactly the piece body colors'

    return _assemble_exact_cover(pieces, legend_idx, successor_map, k)


# --------------- generator (o5x) ---------------

# === CFG ===
SIDE_CHOICES = [8, 10, 12]
K = 2
NUM_PIECES = K * K
NUM_COLORS = NUM_PIECES
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
MIN_RULE_MISMATCH = 2


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


def _palette_spots(mask):
    spots = []
    h, w = mask.shape
    for r in range(1, h - K):
        for c in range(1, w - K):
            if mask[r : r + K, c : c + K].all() and mask[r - 1 : r + K + 1, c - 1 : c + K + 1].all():
                spots.append((r, c))
    return spots


def _legend_map_from_cycle(cycle_colors):
    return np.array(
        [
            [cycle_colors[0], cycle_colors[1]],
            [cycle_colors[3], cycle_colors[2]],
        ],
        dtype=int,
    )


def _successor_from_cycle(cycle_colors):
    return {int(cycle_colors[i]): int(cycle_colors[(i + 1) % NUM_PIECES]) for i in range(NUM_PIECES)}


def _old_rule_mismatch_count(body_colors, qcells, legend_map, successor_map):
    mismatch = 0
    for pi, src in enumerate(body_colors):
        new_col = int(successor_map[int(src)])
        old_col = int(legend_map[qcells[pi]])
        mismatch += int(new_col != old_col)
    return mismatch


def _scatter_pieces(masks, body_colors, legend_map, side):
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
            spots = _palette_spots(oriented)
            if not spots:
                return None
            r, c = random.choice(spots)
            patch[r : r + K, c : c + K] = legend_map
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

    qcells = []
    for mask in masks:
        pts = np.argwhere(mask)
        qcells.append((min(1, int(pts[:, 0].mean() * K / side)), min(1, int(pts[:, 1].mean() * K / side))))
    if len(set(qcells)) != NUM_PIECES:
        return None

    body_colors = random.sample(range(1, 10), NUM_COLORS)

    cycle_colors = body_colors.copy()
    found_cycle = None
    for _ in range(40):
        random.shuffle(cycle_colors)
        legend_map = _legend_map_from_cycle(cycle_colors)
        successor_map = _successor_from_cycle(cycle_colors)
        if _old_rule_mismatch_count(body_colors, qcells, legend_map, successor_map) >= MIN_RULE_MISMATCH:
            found_cycle = cycle_colors.copy()
            break
    if found_cycle is None:
        return None

    legend_map = _legend_map_from_cycle(found_cycle)
    successor_map = _successor_from_cycle(found_cycle)

    expected = np.zeros((side, side), dtype=int)
    for pi, mask in enumerate(masks):
        expected[mask] = successor_map[body_colors[pi]]

    grid = _scatter_pieces(masks, body_colors, legend_map, side)
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
