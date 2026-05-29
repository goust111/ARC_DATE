# Combo variant of 16b78196 (index 16)
# combo_spec: D4/1+D15/2
# Name: 分叉链+数字标记

from __future__ import annotations

from itertools import combinations
import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis for combo D4/1 + D15/2
# a) 具体化与行为追踪
#    - 原题: 若干散落单色精灵被插回中央横/竖杆的缺口，形成 1-2 条线性链。后继精灵只需
#      覆盖前一精灵 bbox 里的背景空腔并继续向外延伸。
#    - 融合后: 每个精灵都带有同一标记色的小点。1 点 = 根，2 点 = 中间节点，3 点 = 叶子。
#      这里只有一棵树: 根先堵住杆上的真缺口；2 点精灵再填根的空腔；随后它的 bbox 内出现
#      两个彼此分离的背景块，两个 3 点精灵分别填入，形成分叉。
#    - 具体例子: 先把 1-dot 根插进杆顶缺口；根 bbox 顶部保留一个长槽。2-dot 中间精灵只在
#      这条槽上落位，不压住根的实心像素。放下后，中间精灵 bbox 内出现一个 2x2 方槽和一个
#      L 形小槽。两个 3-dot 叶子虽然点数相同，但形状不同，只能分别插到左/右分支，最终得到
#      树状拼接。
# b) 复杂度校准
#    - 若保留原题“多条线性链”，再加分叉，搜索自由度会过高且样例不稳定；因此这里固定成
#      1 棵 4 片树: 1 个根、1 个中间节点、2 个叶子。这样 1/2/3 三种点数都稳定出现，并且
#      3-dot 会重复两次，肉眼能看出“同层多个叶子”这一新规律。
#    - 分叉结构仍保留原题的“前一层 bbox 空腔驱动下一层”的几何主线，不会退化成单纯按点数
#      排序即可完成的拼图；两个 3-dot 仍需靠 cavity 形状匹配决定左右去向。
# c) 隐式假设审查
#    - 原题 `pieces_raw = comps[1:]` 假设每片精灵是单色连通块；这里失效，因为标记点会把
#      单色连通块切碎。solver 改为: 先用单色组件找 bar，再用多色组件找整片 sprite。
#    - 原题把每片精灵当作纯色绘制；这里若继续这么做会擦掉点数标记，所以 solver 保存每个
#      相对坐标的真实颜色并按整片多色图案回写。
#    - 原题默认“只要覆盖当前 cavity 就可能是下一个 piece”；融合后这太弱。solver 先按
#      1/2/3 点分层，再要求 2-dot 之后必须产生两个 cavity，两个 3-dot 只能分别填其中一个。
#    - 原题大量依赖“固定条带宽度”约束。分叉后，叶子会从中间节点的左右窄槽分出，若继续把
#      叶子强行限制在同一整条 strip 内，合法分叉反而会被错误排除。因此 solver 只对
#      root->middle 继续沿用“覆盖前层 cavity 且接触前层”的原始几何逻辑；叶子阶段改成
#      “只允许占据 middle bbox 内的一个 cavity 分量，再向外伸出”。
# d) 放置合法性
#    - generator 仍是 output-first。先构造 bar 与完整树状输出，再把四个完整 sprite 散布到
#      bar 周围，要求不重叠、不越界、且与 bar/其他 sprite 不 4 邻接。
#    - 所有标记点都从 sprite 现有像素中选择，因此不会改变占据集合、连通性或 cavity 结构。
# e) 视觉可观察性
#    - 输入里共享标记色会反复出现，且 1/2/3 的数量关系稳定为 1,1,2，解题者能先看出点数
#      对应层级，再看到 2-dot 输出后分出两个分支。
#    - 规则不依赖颜色编号大小；标记色只通过“全局共享且以孤立小点形式出现”这一视觉模式被识别。
# f) 可视化整洁度
#    - 只引入一个额外语义色作为标记色，避免颜色过多。
#    - 分叉固定发生在 2-dot 节点，且两个叶子只在左右各占一个窄槽，不会在同一块局部里叠出
#      多层难分辨的纹理。
# g) 变化的可推断性
#    - 没有复杂颜色映射；真正新增的信息只有“点数=树层级”和“2-dot 后会分叉成两个 3-dot”。
#      这两点都能从少量样例稳定重复中被观察到。
# 新增 solver 假设与 generator 保证
#    - solver 假设存在 1 个 1-dot、1 个 2-dot、2 个 3-dot sprite；generator 固定如此生成。
#    - solver 假设 2-dot sprite 在正确放置后，其 bbox 内恰有两个背景连通块；generator 的
#      middle 模板专门保证这一点。
#    - solver 假设每个 3-dot 叶子在正确放置后，其与 middle bbox 的交集必须恰好等于某个
#      cavity；generator 让左叶对应 2x2 方槽，右叶对应 L 形槽来满足这一点。


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ROOT_TEMPLATE = np.array(
    [
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ],
    dtype=int,
)
MIDDLE_TEMPLATE = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ],
    dtype=int,
)
SHORT_LEAF_TEMPLATE = np.ones((4, 2), dtype=int)
TALL_LEAF_TEMPLATE = np.array(
    [
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 1],
    ],
    dtype=int,
)


def _component_sets(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(cells)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        comp = set()
        stack = [start]
        while stack:
            r, c = stack.pop()
            if (r, c) not in remaining:
                continue
            remaining.remove((r, c))
            comp.add((r, c))
            for dr, dc in DIR4:
                nb = (r + dr, c + dc)
                if nb in remaining:
                    stack.append(nb)
        comps.append(comp)
    return comps


def _template_cells(template: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    h, w = map(int, template.shape)
    return {
        (top + r, left + c)
        for r in range(h)
        for c in range(w)
        if int(template[r, c]) == 1
    }


def _shape_from_component(grid: np.ndarray, marker_color: int, coords: np.ndarray, bbox) -> dict:
    r0, c0, r1, c1 = map(int, bbox)
    rel = []
    rel_colors = []
    marker_count = 0
    body_hist: dict[int, int] = {}
    for r, c in coords:
        rr, cc = int(r), int(c)
        lr, lc = rr - r0, cc - c0
        color = int(grid[rr, cc])
        rel.append((lr, lc))
        rel_colors.append((lr, lc, color))
        if color == marker_color:
            marker_count += 1
        else:
            body_hist[color] = body_hist.get(color, 0) + 1
    body_color = max(body_hist.items(), key=lambda kv: kv[1])[0]
    return {
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "rel": rel,
        "rel_colors": rel_colors,
        "area": int(len(rel)),
        "solid": int(len(rel)) == int((r1 - r0 + 1) * (c1 - c0 + 1)),
        "order": int(marker_count),
        "body_color": int(body_color),
    }


def _infer_marker_color(mono_comps: list[dict], bar_color: int) -> int:
    stats: dict[int, list[int]] = {}
    for comp in mono_comps[1:]:
        color = int(comp["color"])
        if color == bar_color:
            continue
        stats.setdefault(color, []).append(int(comp["area"]))

    candidates = []
    for color, areas in stats.items():
        if areas and max(areas) == 1:
            candidates.append((len(areas), -sum(areas), -color, color))
    if candidates:
        candidates.sort(reverse=True)
        return int(candidates[0][3])

    fallback = []
    for color, areas in stats.items():
        fallback.append((-max(areas), len(areas), -sum(areas), color))
    fallback.sort(reverse=True)
    return int(fallback[0][3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble a marked branching chain:
    1-dot root -> 2-dot middle -> two 3-dot leaves.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    mono.sort(key=lambda d: -d["area"])
    assert mono

    bar = mono[0]
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1

    out = np.full((H, W), int(bg), dtype=int)
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    marker_color = _infer_marker_color(mono, bar_color)

    multicolor = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    multicolor.sort(key=lambda d: -d["area"])
    assert multicolor

    pieces = []
    for comp in multicolor[1:]:
        r0, c0 = map(int, comp["bbox"][:2])
        shape = _shape_from_component(g, marker_color, comp["coords"], comp["bbox"])
        pieces.append({"shape": shape, "orig": (r0, c0), "order": int(shape["order"])})

    by_order: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for i, piece in enumerate(pieces):
        if piece["order"] in by_order:
            by_order[piece["order"]].append(i)
    if len(by_order[1]) != 1 or len(by_order[2]) != 1 or len(by_order[3]) != 2:
        return out

    def bar_notches(mask, base_r, base_c):
        empty = ~mask
        h, w = map(int, empty.shape)
        notches = []
        for pts in components_mask(empty, connectivity=4):
            rr = pts[:, 0]
            cc = pts[:, 1]
            top = int((rr == 0).sum())
            bottom = int((rr == h - 1).sum())
            left = int((cc == 0).sum())
            right = int((cc == w - 1).sum())
            if top + bottom + left + right == 0:
                continue
            hole = {(base_r + int(r), base_c + int(c)) for r, c in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            notches.append({"hole": hole, "mouth_n": int(mouth_n), "side": side})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(shape, top, left):
        return {(top + lr, left + lc) for lr, lc in shape["rel"]}

    def paint_shape(out_grid, shape, top, left):
        for lr, lc, color in shape["rel_colors"]:
            out_grid[top + lr, left + lc] = int(color)

    def clear_shape(out_grid, shape, top, left):
        for lr, lc, _ in shape["rel_colors"]:
            out_grid[top + lr, left + lc] = int(bg)

    def placement_pts(shape, top, left, occ, must_cover=None):
        if top < 0 or left < 0 or top + shape["h"] > H or left + shape["w"] > W:
            return None
        pts = pixels(shape, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, shape):
        pos = set()
        for r_h, c_h in hole:
            for lr, lc in shape["rel"]:
                pos.add((r_h - lr, c_h - lc))
        return pos

    def cavity_components_for_placed_shape(out_grid, shape, top, left):
        cells = {
            (r, c)
            for r in range(top, top + shape["h"])
            for c in range(left, left + shape["w"])
            if out_grid[r, c] == bg
        }
        if not cells:
            return []
        comps = _component_sets(cells)
        comps.sort(key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp), len(comp)))
        return comps

    def bbox_cells(top, left, shape):
        return {
            (r, c)
            for r in range(top, top + shape["h"])
            for c in range(left, left + shape["w"])
        }

    root_idx = by_order[1][0]
    mid_idx = by_order[2][0]
    leaf_indices = tuple(by_order[3])

    root_cands = []
    root_shape = pieces[root_idx]["shape"]
    for notch_i, notch in enumerate(notches):
        for top, left in hole_positions(notch["hole"], root_shape):
            pts = placement_pts(root_shape, top, left, bar_occ, must_cover=notch["hole"])
            if pts is None:
                continue
            mv = abs(int(top) - int(pieces[root_idx]["orig"][0])) + abs(int(left) - int(pieces[root_idx]["orig"][1]))
            root_cands.append(
                (
                    (int(mv), -int(notch["mouth_n"])),
                    {"notch_i": notch_i, "top": int(top), "left": int(left), "pts": pts},
                )
            )
    root_cands.sort(key=lambda x: x[0])

    best_out = out.copy()
    best_score = (-1, -1)

    def update_best(out_grid, placed):
        nonlocal best_out, best_score
        score = (placed, -int((out_grid == bg).sum()))
        if score > best_score:
            best_score = score
            best_out = out_grid.copy()

    def try_leaves(out_grid, occ_set, middle_info, unused_leafs):
        if not unused_leafs:
            return not cavity_components_for_placed_shape(out_grid, middle_info["shape"], middle_info["top"], middle_info["left"])

        middle_bbox = bbox_cells(middle_info["top"], middle_info["left"], middle_info["shape"])
        cavities = cavity_components_for_placed_shape(out_grid, middle_info["shape"], middle_info["top"], middle_info["left"])
        if len(cavities) != len(unused_leafs):
            return False

        choice = None
        for cavity in cavities:
            moves = []
            for pi in unused_leafs:
                shape = pieces[pi]["shape"]
                for top, left in hole_positions(cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity)
                    if pts is None:
                        continue
                    if not touches(pts, middle_info["pts"]):
                        continue
                    if (pts & middle_bbox) != cavity:
                        continue
                    mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                    moves.append((mv, pi, top, left, pts))
            if not moves:
                return False
            moves.sort()
            if choice is None or len(moves) < len(choice[1]):
                choice = (cavity, moves)

        assert choice is not None
        _, moves = choice
        for _, pi, top, left, pts in moves[:12]:
            shape = pieces[pi]["shape"]
            paint_shape(out_grid, shape, top, left)
            update_best(out_grid, 3 + (len(leaf_indices) - len(unused_leafs)))
            occ2 = set(occ_set) | set(pts)
            rem2 = set(unused_leafs)
            rem2.remove(pi)
            if try_leaves(out_grid, occ2, middle_info, rem2):
                return True
            clear_shape(out_grid, shape, top, left)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color
        return False

    for _, root in root_cands[:40]:
        out0 = np.full((H, W), int(bg), dtype=int)
        for r, c in bar_occ:
            out0[r, c] = bar_color
        occ0 = set(bar_occ)

        paint_shape(out0, root_shape, root["top"], root["left"])
        occ0 |= root["pts"]
        update_best(out0, 1)

        root_cavities = cavity_components_for_placed_shape(out0, root_shape, root["top"], root["left"])
        if len(root_cavities) != 1:
            continue
        root_cavity = root_cavities[0]
        root_pts = set(root["pts"])

        mid_shape = pieces[mid_idx]["shape"]
        mid_moves = []
        for top, left in hole_positions(root_cavity, mid_shape):
            pts = placement_pts(mid_shape, top, left, occ0, must_cover=root_cavity)
            if pts is None:
                continue
            if not touches(pts, root_pts):
                continue
            mv = abs(int(top) - int(pieces[mid_idx]["orig"][0])) + abs(int(left) - int(pieces[mid_idx]["orig"][1]))
            mid_moves.append((mv, int(top), int(left), pts))
        mid_moves.sort()

        for _, mid_top, mid_left, mid_pts in mid_moves[:20]:
            paint_shape(out0, mid_shape, mid_top, mid_left)
            update_best(out0, 2)

            mid_cavities = cavity_components_for_placed_shape(out0, mid_shape, mid_top, mid_left)
            if len(mid_cavities) == 2:
                middle_info = {"shape": mid_shape, "top": mid_top, "left": mid_left, "pts": set(mid_pts)}
                if try_leaves(out0, occ0 | set(mid_pts), middle_info, set(leaf_indices)):
                    return out0

            clear_shape(out0, mid_shape, mid_top, mid_left)
            for r, c in bar_occ:
                out0[r, c] = bar_color

    return best_out


def _choose_marker_cells(cells: set[tuple[int, int]], count: int) -> set[tuple[int, int]] | None:
    cell_list = sorted(cells)
    cell_set = set(cell_list)
    rs = [r for r, _ in cell_list]
    cs = [c for _, c in cell_list]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0

    def degree(rc):
        r, c = rc
        return sum((r + dr, c + dc) in cell_set for dr, dc in DIR4)

    def edge_margin(rc):
        r, c = rc
        return min(r - r0, r1 - r, c - c0, c1 - c)

    candidates = [rc for rc in cell_list if degree(rc) >= 2]
    if len(candidates) < count:
        candidates = [rc for rc in cell_list if degree(rc) >= 1]
    if len(candidates) < count:
        candidates = list(cell_list)
    if len(candidates) < count:
        return None

    best = None
    best_score = None
    for combo in combinations(candidates, count):
        if any(abs(r1_ - r2_) + abs(c1_ - c2_) == 1 for (r1_, c1_), (r2_, c2_) in combinations(combo, 2)):
            continue
        score = (
            sum(degree(rc) for rc in combo),
            sum(edge_margin(rc) for rc in combo),
            -sum((rc[0] - cr) ** 2 + (rc[1] - cc) ** 2 for rc in combo),
        )
        if best_score is None or score > best_score:
            best_score = score
            best = combo
    if best is not None:
        return set(best)

    fallback = sorted(
        candidates,
        key=lambda rc: (degree(rc), edge_margin(rc), -((rc[0] - cr) ** 2 + (rc[1] - cc) ** 2)),
        reverse=True,
    )
    picked: list[tuple[int, int]] = []
    for rc in fallback:
        if all(abs(rc[0] - rr) + abs(rc[1] - cc2) != 1 for rr, cc2 in picked):
            picked.append(rc)
        if len(picked) == count:
            return set(picked)
    if len(fallback) >= count:
        return set(fallback[:count])
    return None


def _transform_coord(mode: str, rc: tuple[int, int], size: int) -> tuple[int, int]:
    r, c = rc
    n = size - 1
    if mode == "id":
        return r, c
    if mode == "rot90":
        return c, n - r
    if mode == "rot180":
        return n - r, n - c
    if mode == "rot270":
        return n - c, r
    if mode == "flipud":
        return n - r, c
    if mode == "fliplr":
        return r, n - c
    raise ValueError(mode)


def _transform_cells(mode: str, cells: set[tuple[int, int]], size: int) -> set[tuple[int, int]]:
    return {_transform_coord(mode, rc, size) for rc in cells}


def _scatter_piece(
    rng: random.Random,
    grid: np.ndarray,
    placed_occ: set[tuple[int, int]],
    rel_cells: list[tuple[int, int]],
    marker_rel: set[tuple[int, int]],
    body_color: int,
    marker_color: int,
    bar_bbox: tuple[int, int, int, int],
) -> bool:
    H, W = map(int, grid.shape)
    p_h = max(r for r, _ in rel_cells) + 1
    p_w = max(c for _, c in rel_cells) + 1
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar_bbox)
    horizontal = (bar_c1 - bar_c0 + 1) == W

    for _ in range(600):
        gap = rng.randint(2, 7)
        if horizontal:
            top = (bar_r0 - gap - p_h) if rng.random() < 0.5 else (bar_r1 + gap + 1)
            left = rng.randint(0, W - p_w)
        else:
            left = (bar_c0 - gap - p_w) if rng.random() < 0.5 else (bar_c1 + gap + 1)
            top = rng.randint(0, H - p_h)

        if top < 0 or left < 0 or top + p_h > H or left + p_w > W:
            continue
        pts = {(top + dr, left + dc) for dr, dc in rel_cells}
        if pts & placed_occ:
            continue
        if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
            continue

        for dr, dc in rel_cells:
            grid[top + dr, left + dc] = marker_color if (dr, dc) in marker_rel else body_color
        placed_occ |= pts
        return True
    return False


def _add_decoy_notches(
    rng: random.Random,
    bar_cells: set[tuple[int, int]],
    bar_r0: int,
    bar_r1: int,
    W: int,
    used_range: tuple[int, int],
) -> set[tuple[int, int]]:
    top_ranges = [used_range]
    bottom_ranges: list[tuple[int, int]] = []
    total_decoys = rng.randint(5, 6)
    holes: set[tuple[int, int]] = set()
    decoy_count = 0

    def free_on_side(ranges: list[tuple[int, int]], start: int, end: int) -> bool:
        for a, b in ranges:
            if not (end < a - 1 or start > b + 1):
                return False
        return True

    for _ in range(400):
        if decoy_count >= total_decoys:
            break
        side = rng.choice(["top", "bottom"])
        depth = rng.randint(1, 2)
        width = rng.randint(1, 4)
        start = rng.randint(1, W - width - 1)
        end = start + width - 1
        ranges = top_ranges if side == "top" else bottom_ranges
        if not free_on_side(ranges, start, end):
            continue
        ranges.append((start, end))
        for rr in range(depth):
            row = bar_r0 + rr if side == "top" else bar_r1 - rr
            for c in range(start, end + 1):
                holes.add((row, c))
        decoy_count += 1
    return holes


def _build_output_tree(rng: random.Random):
    H = W = 30
    bg = 0

    thickness = rng.randint(5, 7)
    center = int(rng.uniform(0.38, 0.52) * H)
    bar_r0 = max(7, center - thickness // 2)
    bar_r1 = min(H - 1, bar_r0 + thickness - 1)
    bar_r0 = bar_r1 - thickness + 1

    bar_color = rng.randint(1, 9)
    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    marker_color = avail_colors.pop()
    body_colors = [avail_colors.pop() for _ in range(4)]

    root_left = rng.randint(2, 23)
    root_top = bar_r0 - 2
    middle_top = root_top - 2
    middle_left = root_left + 1

    root_cells = _template_cells(ROOT_TEMPLATE, root_top, root_left)
    middle_cells = _template_cells(MIDDLE_TEMPLATE, middle_top, middle_left)
    left_leaf_top = middle_top - 2
    left_leaf_left = middle_left
    right_leaf_top = middle_top - 3
    right_leaf_left = middle_left + 3
    short_leaf_cells = _template_cells(SHORT_LEAF_TEMPLATE, left_leaf_top, left_leaf_left)
    tall_leaf_cells = _template_cells(TALL_LEAF_TEMPLATE, right_leaf_top, right_leaf_left)

    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(W)}
    used_notch = {(bar_r0, c) for c in range(root_left, root_left + 5)}
    decoy_holes = _add_decoy_notches(rng, bar_cells, bar_r0, bar_r1, W, (root_left, root_left + 4))
    bar_cells -= used_notch | decoy_holes

    piece_cells = [root_cells, middle_cells, short_leaf_cells, tall_leaf_cells]
    if any(any(r < 0 or r >= H or c < 0 or c >= W for r, c in cells) for cells in piece_cells):
        return None

    if used_notch != {rc for rc in root_cells if rc[0] == bar_r0}:
        return None
    if root_cells & middle_cells or root_cells & short_leaf_cells or root_cells & tall_leaf_cells:
        return None
    if middle_cells & short_leaf_cells or middle_cells & tall_leaf_cells or short_leaf_cells & tall_leaf_cells:
        return None

    pieces = []
    orders = [1, 2, 3, 3]
    for cells, body_color, order in zip(piece_cells, body_colors, orders):
        markers = _choose_marker_cells(cells, order)
        if markers is None:
            return None
        pieces.append(
            {
                "coords": set(cells),
                "marker_cells": set(markers),
                "body_color": int(body_color),
                "marker_color": int(marker_color),
                "order": int(order),
            }
        )

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for piece in pieces:
        for r, c in piece["coords"]:
            output[r, c] = piece["body_color"]
        for r, c in piece["marker_cells"]:
            output[r, c] = marker_color

    return {
        "output": output,
        "bar_cells": bar_cells,
        "bar_color": int(bar_color),
        "pieces": pieces,
        "marker_color": int(marker_color),
    }


def _apply_global_transform(rng: random.Random, spec: dict):
    mode = rng.choice(["id", "fliplr", "flipud", "rot90", "rot180", "rot270"])
    size = int(spec["output"].shape[0])
    bar_cells = _transform_cells(mode, set(spec["bar_cells"]), size)
    pieces = []
    for piece in spec["pieces"]:
        pieces.append(
            {
                "coords": _transform_cells(mode, set(piece["coords"]), size),
                "marker_cells": _transform_cells(mode, set(piece["marker_cells"]), size),
                "body_color": int(piece["body_color"]),
                "marker_color": int(piece["marker_color"]),
                "order": int(piece["order"]),
            }
        )
    output = np.full_like(spec["output"], 0)
    for r, c in bar_cells:
        output[r, c] = int(spec["bar_color"])
    for piece in pieces:
        for r, c in piece["coords"]:
            output[r, c] = int(piece["body_color"])
        for r, c in piece["marker_cells"]:
            output[r, c] = int(piece["marker_color"])
    return {"output": output, "bar_cells": bar_cells, "bar_color": int(spec["bar_color"]), "pieces": pieces, "marker_color": int(spec["marker_color"])}


def _gen_attempt(rng: random.Random | None = None):
    rng = rng or random.Random()
    built = _build_output_tree(rng)
    if built is None:
        return None
    spec = _apply_global_transform(rng, built)

    H, W = map(int, spec["output"].shape)
    input_grid = np.zeros((H, W), dtype=int)
    for r, c in spec["bar_cells"]:
        input_grid[r, c] = int(spec["bar_color"])

    placed_occ = set(spec["bar_cells"])
    rs = [r for r, _ in spec["bar_cells"]]
    cs = [c for _, c in spec["bar_cells"]]
    bar_bbox = (min(rs), min(cs), max(rs), max(cs))

    piece_order = list(spec["pieces"])
    rng.shuffle(piece_order)
    for piece in piece_order:
        coords = piece["coords"]
        prs = [r for r, _ in coords]
        pcs = [c for _, c in coords]
        r0, c0 = min(prs), min(pcs)
        rel = sorted((r - r0, c - c0) for r, c in coords)
        marker_rel = {(r - r0, c - c0) for r, c in piece["marker_cells"]}
        if not _scatter_piece(
            rng,
            input_grid,
            placed_occ,
            rel,
            marker_rel,
            int(piece["body_color"]),
            int(piece["marker_color"]),
            bar_bbox,
        ):
            return None

    return input_grid, spec["output"], spec


def generate_input() -> np.ndarray:
    for seed in range(300):
        rng = random.Random(10_000 + seed * 97)
        result = _gen_attempt(rng)
        if result is None:
            continue
        inp, out, _ = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
