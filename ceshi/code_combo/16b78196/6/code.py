# Combo variant of 16b78196 (index 6)
# combo_spec: D3/1+D10/2
# Name: 颜色标记缺口+高相似干扰
#
# Design analysis
# a) 具体化与行为追踪:
# - 原题: 中央杆是单色，所有缺口都是黑洞。solver 对每个散落精灵枚举是否能覆盖某个洞，
#   只靠几何覆盖 + 后续链式 cavity 填充来判断。
# - 融合后: 每个缺口的洞口内缘有颜色提示。对每条真实链，root 颜色会在两个缺口上出现:
#   一个是真缺口，另一个是同色干扰缺口；干扰缺口的归一化形状与真缺口只差 1-2 像素。
# - 小例子: 若红色 root 真缺口是 2x3 中缺 1 格的形状，则杆上还会出现另一个红色包边缺口，
#   例如同样 2x3 但多 1 格或少 1 格。颜色阶段先把候选从“所有洞”缩到“红色那两个洞”，
#   几何阶段再要求 root 放回杆框内时，覆盖到的洞像素集合必须与缺口完全相等，只留下真缺口。
# - 差异: 原题的 root 判断是“纯形状搜索”；本变体是“先看颜色，再做精确洞形校验”。
# b) 复杂度校准:
# - 每个有效颜色通常对应 2 个同色缺口(1 真 1 假)，把分支压到很小但保留真实歧义。
# - 干扰缺口只做 1-2 像素扰动，肉眼能看出“几乎一样”，不会退化成完全不同的随意小洞。
# - 其余 generic decoy 继续保留，但提示色不对应任何精灵，避免全局噪声过大。
# c) 隐式假设审查:
# - 原 `bar = comps[0]` 假设杆是最大单色块。现在包边彩色化，必须先取最大主体块，再吸收所有
#   与其 4 邻接的非背景小块作为 bar union。
# - 原 `bar_mask` 只含主体色。现在若不把提示边算进去，会把包边误判为散落精灵或新的洞。
# - 原 root 合法性只隐含要求 `hole <= pts`。在高相似干扰下，这种写法不够显式；这里新增
#   “piece 放回后位于 bar bbox 内的像素集合 == notch.hole”，把精确洞形匹配写成硬约束。
# - 原回溯恢复把整根杆都刷回单色。现在必须逐像素恢复 `bar_colors`，否则会抹掉彩色提示边。
# - 新假设: 同色提示缺口的数量不大；每个同色组里至少有 1 个洞与 piece 精确匹配。generator
#   明确构造为“每个 root 颜色正好对应 1 真 + 1 假”并验证 `main(input)==output`。
# d) 放置合法性:
# - output-first 先造 bar、真缺口和链，再额外 carve 相似干扰缺口，最后给每个缺口涂 hint ring。
# - 所有缺口都要求: 在 bar 内、不越界、不与已有缺口重叠或紧贴、去掉后 bar 主体仍 4 连通。
# - 散落精灵放置时把 bar union(主体+hint ring)都算作占用区，并禁止与已有对象 4 邻接。
# e) 视觉可观察性:
# - 人类样例里会稳定看到: 某个精灵颜色在杆上出现两处同色缺口，但只有其中一个能被该精灵精确嵌入；
#   输出里该链只会填进真缺口。这正对应“颜色提示先筛、形状再确认”。
# f) 可视化整洁度:
# - hint 只涂缺口内缘，不在同一区域叠多层标记。
# - 相似 decoy 放在同一侧但与其他缺口保留至少 1 格缓冲，避免 ring 黏连成大片噪声。
# g) 变化的可推断性:
# - 颜色只承担“相等/不相等”的提示，不依赖颜色编号大小。
# - 同色缺口成对出现，且形状差异非常小，足以让观察者从 2-4 组样例中归纳出两阶段规则。

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random as _random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _shape_from_coords(coords: np.ndarray, bbox):
    r0, c0, r1, c1 = map(int, bbox)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}


def _pixels(shape, top, left):
    return {(top + lr, left + lc) for lr, lc in shape["rel"]}


def _connected(cells):
    if len(cells) <= 1:
        return True
    cells = set(cells)
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _normalize(cells):
    cells = set(cells)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    return {(r - r0, c - c0) for r, c in cells}


def _shape_diff(cells_a, cells_b):
    return len(_normalize(cells_a) ^ _normalize(cells_b))


def _trim_rel(cells):
    return _normalize(cells)


def _hint_ring(hole, bar_cells):
    return {
        (r + dr, c + dc)
        for r, c in hole
        for dr, dc in DIR4
        if (r + dr, c + dc) in bar_cells
    }


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=4):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(max(20, target * 20)):
            if len(cells) >= target:
                break
            frontier = [
                nb
                for r, c in cells
                for dr, dc in DIR4
                if (nb := (r + dr, c + dc)) in available and nb not in cells
            ]
            if not frontier:
                break
            if must_span_axis is not None and span_range is not None:
                A, B = span_range
                if must_span_axis == "cols":
                    cmin = min(c for _, c in cells)
                    cmax = max(c for _, c in cells)
                    if cmin > A:
                        left_ext = [nb for nb in frontier if nb[1] < cmin]
                        if left_ext:
                            cells.add(rng.choice(left_ext))
                            continue
                    if cmax < B:
                        right_ext = [nb for nb in frontier if nb[1] > cmax]
                        if right_ext:
                            cells.add(rng.choice(right_ext))
                            continue
                else:
                    rmin = min(r for r, _ in cells)
                    rmax = max(r for r, _ in cells)
                    if rmin > A:
                        up_ext = [nb for nb in frontier if nb[0] < rmin]
                        if up_ext:
                            cells.add(rng.choice(up_ext))
                            continue
                    if rmax < B:
                        down_ext = [nb for nb in frontier if nb[0] > rmax]
                        if down_ext:
                            cells.add(rng.choice(down_ext))
                            continue
            cells.add(rng.choice(frontier))
        if len(cells) < target:
            continue

        while len(cells) > target:
            removable = []
            for rc in cells:
                if rc in seeds:
                    continue
                if must_span_axis is not None and span_range is not None:
                    A, B = span_range
                    if must_span_axis == "cols":
                        if rc[1] == A and sum(1 for _, c2 in cells if c2 == A) <= 1:
                            continue
                        if rc[1] == B and sum(1 for _, c2 in cells if c2 == B) <= 1:
                            continue
                    else:
                        if rc[0] == A and sum(1 for r2, _ in cells if r2 == A) <= 1:
                            continue
                        if rc[0] == B and sum(1 for r2, _ in cells if r2 == B) <= 1:
                            continue
                if _connected(cells - {rc}):
                    removable.append(rc)
            if not removable:
                break
            cells.remove(rng.choice(removable))
        if len(cells) != target:
            continue
        if must_span_axis is not None and span_range is not None:
            A, B = span_range
            if must_span_axis == "cols":
                if min(c for _, c in cells) != A or max(c for _, c in cells) != B:
                    continue
            else:
                if min(r for r, _ in cells) != A or max(r for r, _ in cells) != B:
                    continue
        if _connected(cells):
            return cells
    return None


def _collect_bar_and_pieces(g: np.ndarray, bg: int):
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    base = comps[0]
    bar_ids = {0}
    bar_occ = {(int(r), int(c)) for r, c in base["coords"]}

    changed = True
    while changed:
        changed = False
        for idx, comp in enumerate(comps[1:], start=1):
            if idx in bar_ids:
                continue
            coords = [(int(r), int(c)) for r, c in comp["coords"]]
            if any((r + dr, c + dc) in bar_occ for r, c in coords for dr, dc in DIR4):
                bar_ids.add(idx)
                bar_occ.update(coords)
                changed = True

    piece_comps = [comp for idx, comp in enumerate(comps) if idx not in bar_ids]
    bar_colors = {rc: int(g[rc]) for rc in bar_occ}
    rs = [r for r, _ in bar_occ]
    cs = [c for _, c in bar_occ]
    bar_bbox = (min(rs), min(cs), max(rs), max(cs))
    return base, bar_occ, bar_colors, bar_bbox, piece_comps


def _bar_notches(bar_mask, bar_r0, bar_c0, bar_occ, bar_colors, base_bar_color):
    empty = ~bar_mask
    h, w = map(int, empty.shape)
    notches = []
    for pts in components_mask(empty, connectivity=4):
        r = pts[:, 0]
        c = pts[:, 1]
        top = int((r == 0).sum())
        bottom = int((r == h - 1).sum())
        left = int((c == 0).sum())
        right = int((c == w - 1).sum())
        if top + bottom + left + right == 0:
            continue

        hole = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"

        rim_colors = [
            bar_colors[(r0 + dr, c0 + dc)]
            for r0, c0 in hole
            for dr, dc in DIR4
            if (r0 + dr, c0 + dc) in bar_occ and bar_colors[(r0 + dr, c0 + dc)] != base_bar_color
        ]
        hint_color = Counter(rim_colors).most_common(1)[0][0] if rim_colors else None
        notches.append({"axis": axis, "hole": hole, "mouth_n": mouth_n, "hint_color": hint_color, "side": side})
    return notches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Two-stage solver:
    1. use notch border color to narrow root-piece candidates,
    2. require exact notch-shape agreement inside the bar bbox.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)

    base_bar, bar_occ, bar_colors, bar_bbox, pieces_raw = _collect_bar_and_pieces(g, bg)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar_bbox)
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    base_bar_color = int(base_bar["color"])

    out = np.full((H, W), int(bg), dtype=int)
    for (r, c), col in bar_colors.items():
        out[r, c] = int(col)

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": _shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    notches = _bar_notches(bar_mask, bar_r0, bar_c0, bar_occ, bar_colors, base_bar_color)

    def placement_pts(shape, top, left, occ, must_cover=None, strip=None, exact_hole=None):
        if top < 0 or left < 0 or top + shape["h"] > H or left + shape["w"] > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or shape["w"] != B - A + 1:
                    return None
            else:
                if top != A or shape["h"] != B - A + 1:
                    return None
        pts = _pixels(shape, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        if exact_hole is not None:
            inside_bar = {
                (r, c)
                for r, c in pts
                if bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1
            }
            if inside_bar != exact_hole:
                return None
        return pts

    def aligned_positions(axis, A, cover, shape):
        if axis == "cols":
            left = A
            tops = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in shape["rel"]:
                    if rN == top + lr:
                        lefts.add(cN - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        return any((r + dr, c + dc) in other for r, c in pts for dr, dc in DIR4)

    def hole_positions(hole, shape):
        pos = set()
        for rH, cH in hole:
            for lr, lc in shape["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    root_cands = []
    for ni, notch in enumerate(notches):
        if notch["hint_color"] is None:
            continue
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            if piece["color"] != notch["hint_color"]:
                continue
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(
                    shape,
                    top,
                    left,
                    bar_occ,
                    must_cover=notch["hole"],
                    exact_hole=notch["hole"],
                )
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(notch["mouth_n"])),
                        {"ni": ni, "pi": pi, "top": int(top), "left": int(left), "axis": axis, "A": A, "B": B, "shape": shape},
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:20]]
    best_out = out.copy()
    best_key = None

    def restore_bar(grid_out):
        for (r, c), col in bar_colors.items():
            grid_out[r, c] = int(col)

    def search(out_grid, occ_set, chains, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        def gen_extend_moves(chain, remaining_set):
            axis = chain["axis"]
            A, B = int(chain["A"]), int(chain["B"])
            head = chain["head"]
            head_pts = _pixels(head["shape"], head["top"], head["left"])
            head_top, head_left = int(head["top"]), int(head["left"])
            head_h, head_w = int(head["shape"]["h"]), int(head["shape"]["w"])
            cavity = {
                (r, c)
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if out_grid[r, c] == bg
            }
            if not cavity:
                return []

            moves = []
            for pi in remaining_set:
                shape = pieces[pi]["shape"]
                for top, left in aligned_positions(axis, A, cavity, shape):
                    pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, A, B))
                    if pts is None or not touches(pts, head_pts):
                        continue
                    moves.append((pi, top, left, pts, shape))
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, shape in moves[:8]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": shape}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                restore_bar(out_grid)
        return False

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            restore_bar(out0)

            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = _pixels(root["shape"], root["top"], root["left"])
                if pts & occ0:
                    ok = False
                    break
                used_p.add(root["pi"])
                used_n.add(root["ni"])
                for r, c in pts:
                    out0[r, c] = int(pieces[root["pi"]]["color"])
                occ0 |= pts
                chains.append(
                    {
                        "axis": root["axis"],
                        "A": root["A"],
                        "B": root["B"],
                        "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                    }
                )
            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _side_touches(rel_cells, side):
    if side == "top":
        return any(r == 0 for r, _ in rel_cells)
    if side == "bottom":
        max_r = max(r for r, _ in rel_cells)
        return any(r == max_r for r, _ in rel_cells)
    if side == "left":
        return any(c == 0 for _, c in rel_cells)
    max_c = max(c for _, c in rel_cells)
    return any(c == max_c for _, c in rel_cells)


def _mutate_hole_rel(rng, hole, side):
    rel = _trim_rel(hole)
    rs = [r for r, _ in rel]
    cs = [c for _, c in rel]
    h, w = max(rs) + 1, max(cs) + 1
    rect = {(r, c) for r in range(h) for c in range(w)}

    def boundary_ok(cells):
        cells = set(cells)
        if not cells or not _connected(cells):
            return False
        return _side_touches(cells, side)

    addable = [rc for rc in rect if rc not in rel]
    removable = [rc for rc in rel if len(rel) > 1 and boundary_ok(rel - {rc})]
    cand_lists = []
    for rc in removable:
        cand_lists.append(rel - {rc})
    for rc in addable:
        new_cells = rel | {rc}
        if boundary_ok(new_cells):
            cand_lists.append(new_cells)
    for rem in removable:
        for add in addable:
            new_cells = (rel - {rem}) | {add}
            if boundary_ok(new_cells):
                cand_lists.append(new_cells)

    rng.shuffle(cand_lists)
    for cand in cand_lists:
        diff = len(_normalize(rel) ^ _normalize(cand))
        if 1 <= diff <= 2:
            return _trim_rel(cand)
    return None


def _neighbors_of(cells):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in DIR4}


def _place_hole_shape(rng, rel_shape, side, bar_r0, bar_c0, bar_r1, bar_c1, bar_cells, existing_holes):
    rel_shape = set(rel_shape)
    h = max(r for r, _ in rel_shape) + 1
    w = max(c for _, c in rel_shape) + 1
    all_holes = set().union(*existing_holes) if existing_holes else set()
    forbidden = all_holes | _neighbors_of(all_holes)

    for _ in range(200):
        if side == "top":
            top = bar_r0
            left = rng.randint(bar_c0, bar_c1 - w + 1)
        elif side == "bottom":
            top = bar_r1 - h + 1
            left = rng.randint(bar_c0, bar_c1 - w + 1)
        elif side == "left":
            top = rng.randint(bar_r0, bar_r1 - h + 1)
            left = bar_c0
        else:
            top = rng.randint(bar_r0, bar_r1 - h + 1)
            left = bar_c1 - w + 1

        hole = {(top + r, left + c) for r, c in rel_shape}
        if hole & forbidden:
            continue
        if not hole.issubset(bar_cells):
            continue
        if not _connected(bar_cells - hole):
            continue
        return hole
    return None


def _random_decoy_notch(rng, orientation, bar_r0, bar_c0, bar_r1, bar_c1, bar_cells, existing_holes):
    for _ in range(300):
        depth = rng.randint(1, 2)
        if orientation == "horizontal":
            side = rng.choice(["top", "bottom"])
            row0 = bar_r0 if side == "top" else bar_r1 - depth + 1
            row1 = row0 + depth - 1
            width = rng.randint(1, 4)
            left = rng.randint(bar_c0, bar_c1 - width + 1)
            avail = {(r, c) for r in range(row0, row1 + 1) for c in range(left, left + width)}
        else:
            side = rng.choice(["left", "right"])
            col0 = bar_c0 if side == "left" else bar_c1 - depth + 1
            col1 = col0 + depth - 1
            height = rng.randint(1, 4)
            top = rng.randint(bar_r0, bar_r1 - height + 1)
            avail = {(r, c) for r in range(top, top + height) for c in range(col0, col1 + 1)}

        all_holes = set().union(*existing_holes) if existing_holes else set()
        avail = {rc for rc in avail if rc in bar_cells and rc not in all_holes and rc not in _neighbors_of(all_holes)}
        if not avail:
            continue
        area = rng.randint(1, min(6, len(avail)))
        hole = _grow_connected(rng, avail, area, {rng.choice(list(avail))})
        if hole is None or not _connected(bar_cells - hole):
            continue
        return hole, side
    return None, None


def _gen_attempt():
    rng = _random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    bar_color = rng.randint(1, 9)
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(H - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = bar_r0, H - 1 - bar_r1
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(W - 1, bar_c0 + thickness - 1)
        bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        space_before, space_after = bar_c0, W - 1 - bar_c1

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    num_chains = rng.randint(1, 2)
    chain_lens = [3] if num_chains == 1 else [rng.randint(2, 3) for _ in range(2)]
    if num_chains == 2:
        while not (3 <= sum(chain_lens) <= 6):
            chain_lens = [rng.randint(2, 3) for _ in range(2)]

    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)
    color_idx = 0

    all_chain_pieces = []
    used_notches = []
    all_notch_holes = []
    occ_set = set(bar_cells)

    for chain_len in chain_lens:
        strip_width = rng.randint(4, 5)
        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next(
                (s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 2 + 3),
                None,
            )
            if side is None:
                return None
            sc0 = rng.randint(1, W - strip_width - 1)
            sc1 = sc0 + strip_width - 1
            sA, sB = sc0, sc1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next(
                (s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 2 + 3),
                None,
            )
            if side is None:
                return None
            sr0 = rng.randint(1, H - strip_width - 1)
            sr1 = sr0 + strip_width - 1
            sA, sB = sr0, sr1

        notch_depth = rng.randint(1, 2)
        max_area = min(4, notch_depth * strip_width)
        if max_area < 2:
            return None
        notch_area = rng.randint(2, max_area)

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sc0, sc1 + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(r, c) for r in range(sA, sB + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*all_notch_holes) if all_notch_holes else set()
        n_avail -= all_holes | _neighbors_of(all_holes)
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        all_notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            if color_idx >= len(avail_colors):
                return None
            piece_color = avail_colors[color_idx]
            color_idx += 1
            piece_area = rng.randint(8, 13)
            chain_dim = rng.randint(2, 3)

            if piece_idx == 0:
                if orientation == "horizontal":
                    if side == "top":
                        pr0, pr1 = max(0, bar_r0 - chain_dim), max(r for r, _ in notch)
                    else:
                        pr0, pr1 = min(r for r, _ in notch), min(H - 1, bar_r1 + chain_dim)
                    pc0, pc1 = sc0, sc1
                else:
                    if side == "left":
                        pc0, pc1 = max(0, bar_c0 - chain_dim), max(c for _, c in notch)
                    else:
                        pc0, pc1 = min(c for _, c in notch), min(W - 1, bar_c1 + chain_dim)
                    pr0, pr1 = sA, sB

                bbox_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
                target = max(max(8, int(bbox_area * 0.54)), min(piece_area, int(bbox_area * 0.80)))
                if target > bbox_area or target < 8:
                    return None
                avail = {
                    (r, c)
                    for r in range(pr0, pr1 + 1)
                    for c in range(pc0, pc1 + 1)
                    if (r, c) in notch or (r, c) not in occ_set
                }
                root = _grow_connected(
                    rng,
                    avail,
                    target,
                    notch,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=6,
                )
                if root is None:
                    return None
                if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in root):
                    return None
                piece_cells = root
            else:
                prev_coords = chain_pieces[-1]["coords"]
                prs = [r for r, _ in prev_coords]
                pcs = [c for _, c in prev_coords]
                prev_r0, prev_r1 = min(prs), max(prs)
                prev_c0, prev_c1 = min(pcs), max(pcs)

                if orientation == "horizontal":
                    cavity = {
                        (r, c)
                        for r in range(prev_r0, prev_r1 + 1)
                        for c in range(sA, sB + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    ext_rows = (
                        (max(0, prev_r0 - chain_dim), prev_r0 - 1)
                        if side == "top"
                        else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    )
                    ext = (
                        {(r, c) for r in range(ext_rows[0], ext_rows[1] + 1) for c in range(sA, sB + 1) if (r, c) not in occ_set}
                        if ext_rows[0] <= ext_rows[1]
                        else set()
                    )
                else:
                    cavity = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    ext_cols = (
                        (max(0, prev_c0 - chain_dim), prev_c0 - 1)
                        if side == "left"
                        else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    )
                    ext = (
                        {(r, c) for r in range(sA, sB + 1) for c in range(ext_cols[0], ext_cols[1] + 1) if (r, c) not in occ_set}
                        if ext_cols[0] <= ext_cols[1]
                        else set()
                    )

                if not cavity:
                    return None
                all_avail = cavity | ext
                target = min(13, max(len(cavity), min(piece_area, len(all_avail))))
                if len(cavity) > 13 or target < 8:
                    return None
                is_tail = piece_idx == chain_len - 1
                if is_tail:
                    target = min(13, len(all_avail))

                piece_cells = _grow_connected(
                    rng,
                    all_avail,
                    target,
                    cavity,
                    must_span_axis="cols" if orientation == "horizontal" else "rows",
                    span_range=(sA, sB),
                    retries=6,
                )
                if piece_cells is None or not cavity.issubset(piece_cells):
                    return None
                if is_tail:
                    prs2 = [r for r, _ in piece_cells]
                    pcs2 = [c for _, c in piece_cells]
                    t_r0, t_r1 = min(prs2), max(prs2)
                    t_c0, t_c1 = min(pcs2), max(pcs2)
                    if orientation == "horizontal":
                        remaining = {
                            (r, c)
                            for r in range(t_r0, t_r1 + 1)
                            for c in range(sA, sB + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    else:
                        remaining = {
                            (r, c)
                            for r in range(sA, sB + 1)
                            for c in range(t_c0, t_c1 + 1)
                            if (r, c) not in occ_set and (r, c) not in piece_cells
                        }
                    if remaining:
                        return None

            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append({"color": piece_color, "coords": piece_cells})
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)
        used_notches.append({"hole": notch, "root_color": chain_pieces[0]["color"], "side": side})

    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    similar_decoys = []
    for meta in used_notches:
        rel_variant = _mutate_hole_rel(rng, meta["hole"], meta["side"])
        if rel_variant is None:
            return None
        decoy_hole = _place_hole_shape(
            rng,
            rel_variant,
            meta["side"],
            bar_r0,
            bar_c0,
            bar_r1,
            bar_c1,
            bar_cells,
            all_notch_holes,
        )
        if decoy_hole is None:
            return None
        if not (1 <= _shape_diff(meta["hole"], decoy_hole) <= 2):
            return None
        bar_cells -= decoy_hole
        all_notch_holes.append(decoy_hole)
        similar_decoys.append({"hole": decoy_hole, "hint_color": meta["root_color"], "source_hole": meta["hole"]})

    min_generic = max(0, 6 - len(used_notches) - len(similar_decoys))
    max_generic = min(9 - len(used_notches) - len(similar_decoys), 7)
    if min_generic > max_generic:
        return None
    num_generic = rng.randint(min_generic, max_generic)
    generic_decoys = []
    for _ in range(num_generic):
        hole, side = _random_decoy_notch(rng, orientation, bar_r0, bar_c0, bar_r1, bar_c1, bar_cells, all_notch_holes)
        if hole is None:
            return None
        bar_cells -= hole
        all_notch_holes.append(hole)
        generic_decoys.append({"hole": hole, "side": side})

    total_notches = len(used_notches) + len(similar_decoys) + len(generic_decoys)
    if not (6 <= total_notches <= 9):
        return None

    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_cells:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for pts in components_mask(~bar_mask, connectivity=4):
        rr, cc = pts[:, 0], pts[:, 1]
        if not ((rr == 0).any() or (rr == bar_h - 1).any() or (cc == 0).any() or (cc == bar_w - 1).any()):
            return None

    piece_colors = {p["color"] for p in all_chain_pieces}
    decoy_palette = [c for c in range(1, 10) if c != bar_color and c not in piece_colors]
    if not decoy_palette:
        return None

    all_notches = []
    hint_occ = set()
    for meta in used_notches:
        ring = _hint_ring(meta["hole"], bar_cells)
        if not ring or ring & hint_occ:
            return None
        hint_occ |= ring
        all_notches.append({"hole": meta["hole"], "hint_cells": ring, "hint_color": meta["root_color"]})
    for meta in similar_decoys:
        ring = _hint_ring(meta["hole"], bar_cells)
        if not ring or ring & hint_occ:
            return None
        hint_occ |= ring
        all_notches.append({"hole": meta["hole"], "hint_cells": ring, "hint_color": meta["hint_color"]})
    for i, meta in enumerate(generic_decoys):
        ring = _hint_ring(meta["hole"], bar_cells)
        if not ring or ring & hint_occ:
            return None
        hint_occ |= ring
        all_notches.append({"hole": meta["hole"], "hint_cells": ring, "hint_color": decoy_palette[i % len(decoy_palette)]})

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for meta in all_notches:
        for r, c in meta["hint_cells"]:
            output[r, c] = meta["hint_color"]
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = piece["color"]

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color
    for meta in all_notches:
        for r, c in meta["hint_cells"]:
            input_grid[r, c] = meta["hint_color"]

    placed_occ = set(bar_cells) | set(hint_occ)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)
    for piece in pieces_order:
        coords = piece["coords"]
        color = piece["color"]
        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        p_r0, p_c0 = min(rs), min(cs)
        rel = [(r - p_r0, c - p_c0) for r, c in coords]
        p_h, p_w = max(rs) - p_r0 + 1, max(cs) - p_c0 + 1

        placed = False
        for _ in range(500):
            gap = rng.randint(2, 7)
            if orientation == "horizontal":
                top = (bar_r0 - gap - p_h) if rng.random() < 0.5 else (bar_r1 + gap + 1)
                left = rng.randint(0, max(0, W - p_w))
            else:
                left = (bar_c0 - gap - p_w) if rng.random() < 0.5 else (bar_c1 + gap + 1)
                top = rng.randint(0, max(0, H - p_h))
            if top < 0 or top + p_h > H or left < 0 or left + p_w > W:
                continue
            pts = {(top + dr, left + dc) for dr, dc in rel}
            if pts & placed_occ:
                continue
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in DIR4):
                continue
            for r, c in pts:
                input_grid[r, c] = color
            placed_occ |= pts
            placed = True
            break
        if not placed:
            return None

    return input_grid, output


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 500 attempts")


def fusion_metrics(grid: np.ndarray) -> dict:
    """
    Diagnostic helper for step 8:
    - same_color_groups: for each piece color, how many same-color hinted notches exist
    - similar_pairs: how many piece colors have a same-color distractor within shape diff 1..2
    """
    g = np.array(grid, dtype=int)
    bg = get_bg(g)
    base_bar, bar_occ, bar_colors, bar_bbox, pieces_raw = _collect_bar_and_pieces(g, bg)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar_bbox)
    base_bar_color = int(base_bar["color"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
    notches = _bar_notches(bar_mask, bar_r0, bar_c0, bar_occ, bar_colors, base_bar_color)
    piece_colors = sorted({int(p["color"]) for p in pieces_raw})

    same_color_groups = {}
    min_diff_per_color = {}
    similar_pairs = 0
    for color in piece_colors:
        color_notches = [n["hole"] for n in notches if n["hint_color"] == color]
        same_color_groups[color] = len(color_notches)
        if len(color_notches) < 2:
            min_diff_per_color[color] = None
            continue
        diffs = []
        for i in range(len(color_notches)):
            for j in range(i + 1, len(color_notches)):
                diffs.append(_shape_diff(color_notches[i], color_notches[j]))
        best = min(diffs) if diffs else None
        min_diff_per_color[color] = best
        if best is not None and 1 <= best <= 2:
            similar_pairs += 1
    return {
        "same_color_groups": same_color_groups,
        "min_diff_per_color": min_diff_per_color,
        "similar_pairs": similar_pairs,
        "piece_color_count": len(piece_colors),
        "notch_count": len(notches),
    }
