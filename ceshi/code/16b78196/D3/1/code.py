# Variant of 16b78196
# Dimension: D3 — 目标结构：中央杆与缺口的设计 (dst)
# Change: 全幅横杆或竖杆（厚度5-7），带6-9个边界缺口（面积1-10，深度1-2），其中1-2个为有效缺口，其余为干扰缺口... → 杆上缺口带颜色填充提示
#
# Design analysis
# - 关于参考任务 d35bdbdc:
#   当前仓库里没有它的 solver 代码文件；只能读到 evaluation JSON 与机制摘要
#   “border_color_match: s.border_color == d.center_color”。因此这里把它当作
#   “目标区域自带颜色提示，匹配时同时看几何和颜色”的基线机制，而不是照搬某个不存在的实现。
# - 具体化:
#   原题里，中央杆是纯色，缺口只是黑色洞，根精灵 purely by shape 匹配洞。
#   本变体里，每个缺口周围与洞 4 邻接的杆像素都会被统一改成某个提示色。
#   这个提示色在有效缺口上等于该链 root 精灵的颜色；在干扰缺口上使用不属于任何精灵的颜色。
#   因此 input 里可见的是：一根仍然横跨全幅/全高的杆，但缺口内缘不是杆色，而是彩色“包边”。
#   output 里这些彩色包边保留，链式拼接仍像原题一样向外生长。
# - 小例子追踪:
#   原逻辑: notch A 与 notch B 都可能被某个红色/蓝色精灵从形状上覆盖，solver 枚举 hole_positions，
#   只按 must_cover(hole) + strip 对齐 + 搜索延展决定。
#   新逻辑: 若 notch A 的包边是红色、notch B 的包边是绿/黄且不存在同色精灵，则 root 候选只允许
#   “piece.color == notch.hint_color”的组合进入枚举。于是根匹配从“看形状”升级为“先看颜色，再做几何放置”。
# - 复杂度校准:
#   没有引入新的移动规则，只给 root 匹配新增一个离散条件。提示色只有 1-2 个有效颜色 + 1-2 个重复使用的
#   干扰颜色，视觉上明显，但不会爆炸出大量颜色映射分支。
# - 原 solver 中与变体直接相关的隐式假设审查:
#   1. `bar = comps[0]` 假设中央杆是最大的单色连通块。现在杆不是纯色，此假设不再成立。
#      修正: 先取最大的单色块作为“杆主体”，再把所有与其 4 邻接的非背景小色块吸收到 bar union 中，
#      这些被吸收的块就是缺口彩色包边。
#   2. `bar_mask` 原先只由 bar_color 构成。现在若继续这样做，会把提示色误当成空洞或散落精灵。
#      修正: `bar_mask` 改为整根杆的并集掩码，包含主体色和提示色。
#   3. `out[r,c] = bar_color` / 回溯时恢复 `bar_color` 原先无害，因为杆内像素全同色。
#      现在会覆盖掉提示色，导致 output 错误。
#      修正: 保存 `bar_colors[(r,c)] = input[r,c]`，初始化和回溯都逐像素恢复原杆颜色。
#   4. `root_cands` 原先对每个 notch 和每个 piece 都尝试几何覆盖。
#      现在必须附加 `piece.color == notch.hint_color`，否则会把颜色提示完全忽略，退化回原题。
# - 放置合法性:
#   generator 仍然 output-first 造链，只在最后给每个缺口计算一圈与洞 4 邻接的杆像素作为 hint ring。
#   所有 hint ring 必须非空且互不重叠；散落精灵放置时仍把整根杆当作占用区，因此不会压到提示边。
# - 视觉可观察性:
#   人看到 2-4 组样例时，会稳定看到“真正会被插入的缺口，其边框颜色出现在某个散落精灵上；其他缺口的边框颜色
#   不会出现在任何精灵上”。这样颜色规则在少量样例里可归纳，而且链式几何规则仍然保留。
# - 颜色可推断性:
#   不按颜色数值大小做任何排序；颜色只承担“相等/不相等”的可见关系。干扰缺口颜色从“未被精灵使用”的颜色池中选，
#   防止出现需要从样例外推复杂颜色映射的情况。
# - 新 solver 的额外输入假设及 generator 保证:
#   1. 与杆主体 4 邻接的非背景分量都属于缺口包边，而不是散落精灵。generator 通过散落距离 >= 2 保证。
#   2. 每个缺口包边的非主体色是单一颜色。generator 对同一缺口的 hint ring 统一涂色保证。
#   3. 干扰缺口颜色不属于任何精灵颜色。generator 从剩余颜色池重复取值保证。

from __future__ import annotations

from collections import Counter
from itertools import combinations

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
    cells_set = set(cells)
    start = next(iter(cells_set))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb in cells_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells_set)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
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
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cm = min((c for _, c in cells), default=B)
                    cx = max((c for _, c in cells), default=A)
                    if cm > A:
                        cand = [nb for nb in frontier if nb[1] < cm]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cx < B:
                        cand = [nb for nb in frontier if nb[1] > cx]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                else:
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
                    if cm > A:
                        cand = [nb for nb in frontier if nb[0] < cm]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
                    if cx < B:
                        cand = [nb for nb in frontier if nb[0] > cx]
                        if cand:
                            cells.add(rng.choice(cand))
                            continue
            cells.add(rng.choice(frontier))
        if len(cells) < target:
            continue

        while len(cells) > target:
            removable = []
            for rc in cells:
                if rc in seeds:
                    continue
                if must_span_axis and span_range:
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
        if must_span_axis and span_range:
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

        hole_gl = {(bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts}
        side, mouth_n = max(
            [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
            key=lambda x: x[1],
        )
        axis = "cols" if side in ("top", "bottom") else "rows"

        rim_colors = [
            bar_colors[(r0 + dr, c0 + dc)]
            for r0, c0 in hole_gl
            for dr, dc in DIR4
            if (r0 + dr, c0 + dc) in bar_occ and bar_colors[(r0 + dr, c0 + dc)] != base_bar_color
        ]
        hint_color = None
        if rim_colors:
            hint_color = Counter(rim_colors).most_common(1)[0][0]

        notches.append(
            {"axis": axis, "hole": hole_gl, "mouth_n": mouth_n, "hint_color": hint_color, "side": side}
        )
    return notches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble sprites into the central bar exactly as in 16b78196, except that
    the usable notch is now color-coded by a hint ring on the bar itself.
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

    def placement_pts(shape, top, left, occ, must_cover=None, strip=None):
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
                pts = placement_pts(shape, top, left, bar_occ, must_cover=notch["hole"])
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


def _hint_ring(hole, bar_cells):
    return {
        (r + dr, c + dc)
        for r, c in hole
        for dr, dc in DIR4
        if (r + dr, c + dc) in bar_cells
    }


def _gen_attempt():
    import random as _random

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
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sc0, sc1 + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(r, c) for r in range(sA, sB + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*all_notch_holes) if all_notch_holes else set()
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
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
                    retries=5,
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
                    es = (
                        (max(0, prev_r0 - chain_dim), prev_r0 - 1)
                        if side == "top"
                        else (prev_r1 + 1, min(H - 1, prev_r1 + chain_dim))
                    )
                    ext = (
                        {(r, c) for r in range(es[0], es[1] + 1) for c in range(sA, sB + 1) if (r, c) not in occ_set}
                        if es[0] <= es[1]
                        else set()
                    )
                else:
                    cavity = {
                        (r, c)
                        for r in range(sA, sB + 1)
                        for c in range(prev_c0, prev_c1 + 1)
                        if (r, c) not in occ_set and (r, c) not in prev_coords
                    }
                    es = (
                        (max(0, prev_c0 - chain_dim), prev_c0 - 1)
                        if side == "left"
                        else (prev_c1 + 1, min(W - 1, prev_c1 + chain_dim))
                    )
                    ext = (
                        {(r, c) for r in range(sA, sB + 1) for c in range(es[0], es[1] + 1) if (r, c) not in occ_set}
                        if es[0] <= es[1]
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
                    retries=5,
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
        used_notches.append({"hole": notch, "root_color": chain_pieces[0]["color"]})

    if not (3 <= len(all_chain_pieces) <= 6):
        return None

    min_decoys = max(4, 6 - len(used_notches))
    max_decoys = min(7, 9 - len(used_notches))
    if min_decoys > max_decoys:
        return None
    num_decoys = rng.randint(min_decoys, max_decoys)
    decoy_notches = []
    decoys = 0
    for _ in range(500):
        if decoys >= num_decoys:
            break
        depth = rng.randint(1, 2)
        area = rng.randint(1, min(6, depth * 5))
        if orientation == "horizontal":
            ds = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if ds == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if ds == "top" else bar_r1
            dc0 = rng.randint(bar_c0, bar_c1)
            dc1 = min(bar_c1, dc0 + rng.randint(0, 4))
        else:
            ds = rng.choice(["left", "right"])
            dc0 = bar_c0 if ds == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if ds == "left" else bar_c1
            dr0 = rng.randint(bar_r0, bar_r1)
            dr1 = min(bar_r1, dr0 + rng.randint(0, 4))

        d_avail = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_holes = set().union(*all_notch_holes) if all_notch_holes else set()
        d_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in DIR4}
        if len(d_avail) < 1:
            continue
        area = min(area, len(d_avail))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None or not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        all_notch_holes.append(dn)
        decoy_notches.append({"hole": dn})
        decoys += 1
    if decoys < 4:
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
    for i, meta in enumerate(decoy_notches):
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
    for _ in range(400):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
