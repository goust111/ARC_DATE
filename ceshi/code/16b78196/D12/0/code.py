# Variant of 16b78196
# Dimension: D12 — 空腔填充逻辑 (meta)
# Change: 后续精灵必须完全填充前一精灵bbox在条带内的所有背景空腔，然后向外延伸... → 空腔部分填充

from __future__ import annotations

# Design analysis for the D12 variant
# ----------------------------------
# a) 具体化与行为追踪
# - 原题里，后一块精灵必须把前一块 bbox 条带里的黑色空腔全部补满，所以链条看起来像
#   “空洞被逐步完全塞住，然后继续向外长”。
# - 本变体保留中央横/竖杆、缺口、固定条带、root 精灵插入缺口这些结构，只改 continuation：
#   后一块精灵只需要覆盖前一块空腔的至少 50%，并继续沿同一条带向远离横/竖杆的方向延伸。
# - 这会在输出中留下可见的残余黑色孔洞。例子：
#   原逻辑：root bbox 里有 6 个黑格，第二块必须吃掉 6/6。
#   新逻辑：第二块只吃掉其中 3-5 个即可，同时必须有一部分伸到 root bbox 外侧。
#   这样输出里会留下 1-3 个 root 内部黑洞，链条仍明显朝外生长。
# - 受影响的环节只有两个：
#   1. solver 的 continuation 匹配：从 `must_cover=cavity` 改成 `coverage(cavity) >= ceil(len(cavity)/2)`。
#   2. generator 的 continuation 构造：不再把整块 cavity 当成必含 seed，而是构造“靠外侧的 cavity 子集”
#      作为必含 seed，并强制留下残余空腔，避免样本退化回原题。
#
# b) 复杂度校准
# - 如果只把 solver 的阈值放宽成 >=50%，但 generator 仍常常生成完整填充，那么视觉上几乎看不出变体。
# - 因此 generator 额外保证：
#   - 每个 continuation 步骤尽量让前一块 cavity 至少有 4 格。
#   - 实际覆盖严格小于 cavity 总量，通常覆盖 50%-75%，留下至少 1-2 格残余。
# - 这样既让新规则可见，又没有引入过多离散子规则；解题者仍只需把握一个核心规律：
#   “后一块只补前一块孔洞的一半以上，而不是补满。”
#
# c) main() 相关隐式假设审查
# - `placement_pts(..., must_cover=cavity)`：原来写死“全覆盖”。在变体中失效，必须替换成计数阈值。
# - `aligned_positions(axis, A, cavity, v)`：原来靠 cavity 枚举对齐位置。即使改成部分覆盖，这个枚举方式仍成立，
#   只是候选更多，所以需要额外用“至少半覆盖 + 接触前一块 + 向外延伸”来剪枝。
# - 原 solver 只要求 `touches(pts, head_pts)`，没有显式检查“向外延伸”。原题里因为 cavity 全覆盖较强，
#   这个缺失通常无害；变体放宽后，若不补上，容易出现一块只在 head bbox 里晃动却不真正向外生长的假解。
#   所以这里新增 `extends_outward(...)` 检查。
# - 回溯撤销时恢复 bar 像素的逻辑仍安全，因为 root 互不重叠、bar 颜色固定，D12 不改变这点。
#
# d) 放置合法性
# - generator 仍沿用 output-first：先造合法输出，再把所有精灵打散成输入。
# - continuation 的可用区域仍只来自：
#   1. 前一块 bbox 内条带中的部分 cavity 子集；
#   2. bbox 外、远离 bar 一侧的延伸区域。
# - 每块 piece 仍需 4 连通、8-13 格、不越界、不与 bar/其他 piece 重叠。
# - 条带宽度锁定规则不变：横杆链固定列范围，竖杆链固定行范围。
#
# e) 视觉可观察性
# - 核心可见信号是：后一块明明没有把前一块的黑洞全部补满，但它仍被视为正确后继，并继续向外长。
# - 为了让人类在 2-4 个例子里看出来，generator 尽量让 continuation 覆盖率落在 50%-75%，
#   而不是偶尔碰巧 100%。
# - 不依赖颜色数值做排序或控制；颜色只负责区分不同 piece。
#
# f) 可视化整洁度
# - 保持原题的大结构：一个大 bar + 少量链条 + 4-7 个 decoy 缺口。
# - 变体新增的黑洞只留在链条内部，不额外叠新标记，因此视觉仍干净。
# - tail piece 继续偏向做成“自身 bbox 没有新 cavity”的终止块，避免链条末端也满是洞而过乱。
#
# g) 颜色变化可推断性
# - 颜色规则不改：bar 一色，piece 各自单色，通常互异。
# - 变体推断完全依赖几何关系，不依赖额外颜色映射。
#
# h) 量化验证指标
# - 定义 continuation step 的 `coverage_ratio = covered_prev_cavity / prev_cavity_size`。
# - 本变体要求 generator 产生的大多数 continuation step 满足：
#   `0.5 <= coverage_ratio < 1.0`，并且 `prev_cavity_size >= 4`。
# - 这能直接检验样本是否真的发生了“部分填充”，而不是退化成原题的“全填充”。
#
# i) main() 新假设与 generator 对应保证
# - 新 solver 假设 continuation piece 必须：
#   1. 仍在 root 锁定的条带内；
#   2. 覆盖当前 head cavity 的至少一半；
#   3. 与 head 4 邻接；
#   4. 至少有一部分超出 head bbox，朝远离 bar 的方向延伸。
# - generator 通过构造 cover_subset 位于 cavity 的外侧前沿，并把 ext 限定为远离 bar 的外部区域，
#   来保证这些假设成立。

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import components_mask, find_components, get_bg


def _connected(cells):
    """Check 4-connectivity of a cell set."""
    if len(cells) <= 1:
        return True
    cells_set = set(cells)
    start = next(iter(cells_set))
    visited = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in cells_set and nb not in visited:
                visited.add(nb)
                stack.append(nb)
    return len(visited) == len(cells_set)


def _grow_connected(rng, available, target, seeds, must_span_axis=None, span_range=None, retries=3):
    """Grow a 4-connected shape of exactly *target* cells within *available*, starting from *seeds*."""
    d4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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
                for dr, dc in d4
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
                        left_gap = [nb for nb in frontier if nb[1] < cm]
                        if left_gap:
                            cells.add(rng.choice(left_gap))
                            continue
                    if cx < B:
                        right_gap = [nb for nb in frontier if nb[1] > cx]
                        if right_gap:
                            cells.add(rng.choice(right_gap))
                            continue
                else:
                    cm = min((r for r, _ in cells), default=B)
                    cx = max((r for r, _ in cells), default=A)
                    if cm > A:
                        up_gap = [nb for nb in frontier if nb[0] < cm]
                        if up_gap:
                            cells.add(rng.choice(up_gap))
                            continue
                    if cx < B:
                        down_gap = [nb for nb in frontier if nb[0] > cx]
                        if down_gap:
                            cells.add(rng.choice(down_gap))
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


def _bbox_of_cells(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _piece_cavity(piece_cells, occ_before, axis, A, B):
    """Background cells in the piece bbox, restricted to the chain strip."""
    r0, c0, r1, c1 = _bbox_of_cells(piece_cells)
    if axis == "cols":
        return {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(A, B + 1)
            if (r, c) not in occ_before and (r, c) not in piece_cells
        }
    return {
        (r, c)
        for r in range(A, B + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in occ_before and (r, c) not in piece_cells
    }


def _frontier_cells(cells, axis, sign):
    if axis == "cols":
        edge = min(r for r, _ in cells) if sign < 0 else max(r for r, _ in cells)
        return {rc for rc in cells if rc[0] == edge}
    edge = min(c for _, c in cells) if sign < 0 else max(c for _, c in cells)
    return {rc for rc in cells if rc[1] == edge}


def _partial_cover_subset(rng, cavity, cover_target, axis, sign):
    frontier = list(_frontier_cells(cavity, axis, sign))
    rng.shuffle(frontier)
    for seed in frontier[: min(len(frontier), 8)]:
        subset = _grow_connected(rng, cavity, cover_target, {seed}, retries=8)
        if subset is None:
            continue
        if subset & set(frontier):
            return subset
    return None


def _extension_cells(prev_coords, axis, sign, A, B, occ_set, H, W, chain_dim):
    r0, c0, r1, c1 = _bbox_of_cells(prev_coords)
    if axis == "cols":
        if sign < 0:
            es0, es1 = max(0, r0 - chain_dim), r0 - 1
        else:
            es0, es1 = r1 + 1, min(H - 1, r1 + chain_dim)
        if es0 > es1:
            return set()
        return {
            (r, c)
            for r in range(es0, es1 + 1)
            for c in range(A, B + 1)
            if (r, c) not in occ_set
        }
    if sign < 0:
        es0, es1 = max(0, c0 - chain_dim), c0 - 1
    else:
        es0, es1 = c1 + 1, min(W - 1, c1 + chain_dim)
    if es0 > es1:
        return set()
    return {
        (r, c)
        for r in range(A, B + 1)
        for c in range(es0, es1 + 1)
        if (r, c) not in occ_set
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble colored sprites into a central bar's notches, chaining successive pieces along fixed-width strips.

    D12 variant:
    - Root pieces still plug bar notches exactly.
    - Continuation pieces stay in the same strip and must cover at least half of the previous piece's
      strip-restricted cavity, not necessarily all of it.
    - They must still touch the previous piece and extend farther away from the bar.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    dir4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    bg = get_bg(g)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps.sort(key=lambda d: -d["area"])
    assert comps

    bar = comps[0]
    pieces_raw = comps[1:]

    out = np.full((H, W), int(bg), dtype=int)
    bar_r0, bar_c0, bar_r1, bar_c1 = map(int, bar["bbox"])
    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
    bar_color = int(bar["color"])
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        return {"rel": rel, "rel_set": set(rel), "h": h, "w": w}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append(
            {
                "color": int(p["color"]),
                "shape": shape_from_coords(p["coords"], p["bbox"]),
                "orig": (r0, c0),
            }
        )

    def bar_notches(mask, base_r0, base_c0):
        empty = ~mask
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
            hole_gl = {(base_r0 + int(rr), base_c0 + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(v, top, left):
        return {(top + lr, left + lc) for lr, lc in v["rel"]}

    def placement_pts(v, top, left, occ, must_cover=None, strip=None):
        if top < 0 or left < 0 or top + v["h"] > H or left + v["w"] > W:
            return None
        if strip is not None:
            axis, A, B = strip
            if axis == "cols":
                if left != A or v["w"] != B - A + 1:
                    return None
            else:
                if top != A or v["h"] != B - A + 1:
                    return None
        pts = pixels(v, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis, A, cover, v):
        if axis == "cols":
            left = A
            tops = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if cN == left + lc:
                        tops.add(rN - lr)
            for top in tops:
                yield top, left
        else:
            top = A
            lefts = set()
            for rN, cN in cover:
                for lr, lc in v["rel"]:
                    if rN == top + lr:
                        lefts.add(cN - lc)
            for left in lefts:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in dir4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, v):
        pos = set()
        for rH, cH in hole:
            for lr, lc in v["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    def grow_sign_for_root(root_pts, axis):
        if axis == "cols":
            return -1 if min(r for r, _ in root_pts) < bar_r0 else 1
        return -1 if min(c for _, c in root_pts) < bar_c0 else 1

    def extends_outward(chain, pts):
        head = chain["head"]
        sign = int(chain["grow_sign"])
        if chain["axis"] == "cols":
            head_min = int(head["top"])
            head_max = head_min + int(head["shape"]["h"]) - 1
            new_min = min(r for r, _ in pts)
            new_max = max(r for r, _ in pts)
            return new_min < head_min if sign < 0 else new_max > head_max
        head_min = int(head["left"])
        head_max = head_min + int(head["shape"]["w"]) - 1
        new_min = min(c for _, c in pts)
        new_max = max(c for _, c in pts)
        return new_min < head_min if sign < 0 else new_max > head_max

    def outward_gain(chain, pts):
        head = chain["head"]
        sign = int(chain["grow_sign"])
        if chain["axis"] == "cols":
            head_min = int(head["top"])
            head_max = head_min + int(head["shape"]["h"]) - 1
            new_min = min(r for r, _ in pts)
            new_max = max(r for r, _ in pts)
            return (head_min - new_min) if sign < 0 else (new_max - head_max)
        head_min = int(head["left"])
        head_max = head_min + int(head["shape"]["w"]) - 1
        new_min = min(c for _, c in pts)
        new_max = max(c for _, c in pts)
        return (head_min - new_min) if sign < 0 else (new_max - head_max)

    root_cands = []
    for ni, n in enumerate(notches):
        axis = n["axis"]
        for pi, p in enumerate(pieces):
            v = p["shape"]
            for top, left in hole_positions(n["hole"], v):
                pts = placement_pts(v, top, left, bar_occ, must_cover=n["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + v["w"] - 1)
                else:
                    A, B = int(top), int(top + v["h"] - 1)
                mv = abs(int(top) - int(p["orig"][0])) + abs(int(left) - int(p["orig"][1]))
                root_cands.append(
                    (
                        (int(mv), -int(n["mouth_n"])),
                        {
                            "ni": ni,
                            "pi": pi,
                            "top": int(top),
                            "left": int(left),
                            "axis": axis,
                            "A": A,
                            "B": B,
                            "shape": v,
                        },
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:20]]

    best_out = out.copy()
    best_key = None

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
            head_pts = pixels(head["shape"], head["top"], head["left"])
            head_top = int(head["top"])
            head_left = int(head["left"])
            head_h = int(head["shape"]["h"])
            head_w = int(head["shape"]["w"])

            cavity = {
                (r, c)
                for r in range(head_top, head_top + head_h)
                for c in range(head_left, head_left + head_w)
                if out_grid[r, c] == bg
            }
            if not cavity:
                return []
            need = (len(cavity) + 1) // 2

            moves = []
            for pi in remaining_set:
                v = pieces[pi]["shape"]
                seen = set()
                for top, left in aligned_positions(axis, A, cavity, v):
                    if (top, left) in seen:
                        continue
                    seen.add((top, left))
                    pts = placement_pts(v, top, left, occ_set, strip=(axis, A, B))
                    if pts is None:
                        continue
                    covered = len(pts & cavity)
                    if covered < need:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    if not extends_outward(chain, pts):
                        continue
                    mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                    score = (
                        int(covered == len(cavity)),
                        -covered,
                        -outward_gain(chain, pts),
                        mv,
                    )
                    moves.append((score, pi, int(top), int(left), pts, v))
            moves.sort(key=lambda item: item[0])
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for _score, pi, top, left, pts, v in moves[:10]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": v}
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                for r, c in pts:
                    out_grid[r, c] = int(bg)
                for r, c in bar_occ:
                    out_grid[r, c] = bar_color
        return False

    from itertools import combinations

    for k in range(1, min(3, len(cand_roots)) + 1):
        for roots_sel in combinations(range(len(cand_roots)), k):
            used_p = set()
            used_n = set()
            occ0 = set(bar_occ)
            out0 = np.full((H, W), int(bg), dtype=int)
            for r, c in bar_occ:
                out0[r, c] = bar_color

            chains = []
            ok = True
            for idx in roots_sel:
                root = cand_roots[idx]
                if root["pi"] in used_p or root["ni"] in used_n:
                    ok = False
                    break
                pts = pixels(root["shape"], root["top"], root["left"])
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
                        "grow_sign": grow_sign_for_root(pts, root["axis"]),
                        "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                    }
                )

            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _gen_attempt(return_meta=False):
    """Build one (input, output) pair; returns (input, output) or None."""
    import random as _random

    rng = _random.Random()
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    axis = "cols" if orientation == "horizontal" else "rows"
    bar_color = rng.randint(1, 9)
    thickness = rng.randint(5, 7)

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = bar_r0 + thickness - 1
        if bar_r1 >= H:
            bar_r1 = H - 1
            bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before, space_after = bar_r0, H - 1 - bar_r1
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = bar_c0 + thickness - 1
        if bar_c1 >= W:
            bar_c1 = W - 1
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
    notch_holes: list[set] = []
    occ_set = set(bar_cells)
    coverage_stats = []

    for chain_idx in range(num_chains):
        chain_len = chain_lens[chain_idx]
        strip_width = rng.randint(4, 5)

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next(
                (
                    s
                    for s in sides
                    if (space_before if s == "top" else space_after) >= chain_len * 2 + 4
                ),
                None,
            )
            if side is None:
                return None
            sc0 = rng.randint(1, W - strip_width - 1)
            sc1 = sc0 + strip_width - 1
            sA, sB = sc0, sc1
            grow_sign = -1 if side == "top" else 1
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next(
                (
                    s
                    for s in sides
                    if (space_before if s == "left" else space_after) >= chain_len * 2 + 4
                ),
                None,
            )
            if side is None:
                return None
            sr0 = rng.randint(1, H - strip_width - 1)
            sr1 = sr0 + strip_width - 1
            sA, sB = sr0, sr1
            grow_sign = -1 if side == "left" else 1

        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * strip_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(sA, sB + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(r, c) for r in range(sA, sB + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if len(n_avail) < notch_area:
            return None

        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))})
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        for piece_idx in range(chain_len):
            if color_idx >= len(avail_colors):
                return None
            piece_color = avail_colors[color_idx]
            color_idx += 1
            is_root = piece_idx == 0
            is_tail = piece_idx == chain_len - 1
            piece_cells = None

            if is_root:
                for _ in range(40):
                    piece_area = rng.randint(8, 13)
                    chain_dim = 3 if chain_len > 1 else rng.randint(2, 3)
                    if orientation == "horizontal":
                        if side == "top":
                            pr0, pr1 = max(0, bar_r0 - chain_dim), max(r for r, _ in notch)
                        else:
                            pr0, pr1 = min(r for r, _ in notch), min(H - 1, bar_r1 + chain_dim)
                        pc0, pc1 = sA, sB
                    else:
                        if side == "left":
                            pc0, pc1 = max(0, bar_c0 - chain_dim), max(c for _, c in notch)
                        else:
                            pc0, pc1 = min(c for _, c in notch), min(W - 1, bar_c1 + chain_dim)
                        pr0, pr1 = sA, sB

                    bbox_area = (pr1 - pr0 + 1) * (pc1 - pc0 + 1)
                    target_lo = max(8, int(np.ceil(bbox_area * 0.54)))
                    target_hi = min(13, max(target_lo, int(np.floor(bbox_area * 0.76))))
                    if target_lo > target_hi:
                        continue
                    target = min(target_hi, max(target_lo, piece_area))
                    avail = {
                        (r, c)
                        for r in range(pr0, pr1 + 1)
                        for c in range(pc0, pc1 + 1)
                        if (r, c) in notch or (r, c) not in occ_set
                    }
                    root = _grow_connected(rng, avail, target, notch, must_span_axis=axis, span_range=(sA, sB), retries=8)
                    if root is None:
                        continue
                    if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in root):
                        continue
                    if chain_len > 1:
                        cavity = _piece_cavity(root, occ_set, axis, sA, sB)
                        if len(cavity) < 4:
                            continue
                    piece_cells = root
                    break
            else:
                prev_coords = chain_pieces[-1]["coords"]
                # occ_set at this point excludes the current piece and includes the previous chain state only.
                prev_cavity = _piece_cavity(prev_coords, occ_set, axis, sA, sB)
                if len(prev_cavity) < 4:
                    return None
                cover_need = (len(prev_cavity) + 1) // 2
                if cover_need >= len(prev_cavity):
                    return None

                for _ in range(60):
                    chain_dim = rng.randint(2, 3)
                    ext = _extension_cells(prev_coords, axis, grow_sign, sA, sB, occ_set, H, W, chain_dim)
                    if not ext:
                        continue
                    cover_hi = min(len(prev_cavity) - 1, max(cover_need, int(np.ceil(len(prev_cavity) * 0.75))))
                    if cover_hi < cover_need:
                        continue
                    cover_target = rng.randint(cover_need, cover_hi)
                    cover_subset = _partial_cover_subset(rng, prev_cavity, cover_target, axis, grow_sign)
                    if cover_subset is None:
                        continue
                    all_avail = set(cover_subset) | set(ext)
                    target_lo = max(8, len(cover_subset) + 1)
                    target_hi = min(13, len(all_avail))
                    if target_lo > target_hi:
                        continue
                    target = target_hi if is_tail else rng.randint(target_lo, target_hi)
                    candidate = _grow_connected(rng, all_avail, target, cover_subset, must_span_axis=axis, span_range=(sA, sB), retries=10)
                    if candidate is None:
                        continue
                    covered = len(candidate & prev_cavity)
                    residual = len(prev_cavity) - covered
                    if covered < cover_need or residual <= 0:
                        continue
                    if axis == "cols":
                        if grow_sign < 0 and min(r for r, _ in candidate) >= min(r for r, _ in prev_coords):
                            continue
                        if grow_sign > 0 and max(r for r, _ in candidate) <= max(r for r, _ in prev_coords):
                            continue
                    else:
                        if grow_sign < 0 and min(c for _, c in candidate) >= min(c for _, c in prev_coords):
                            continue
                        if grow_sign > 0 and max(c for _, c in candidate) <= max(c for _, c in prev_coords):
                            continue
                    next_cavity = _piece_cavity(candidate, occ_set | candidate, axis, sA, sB)
                    if not is_tail and len(next_cavity) < 4:
                        continue
                    if is_tail and len(next_cavity) > 1:
                        continue
                    piece_cells = candidate
                    coverage_stats.append(
                        {
                            "covered": covered,
                            "cavity": len(prev_cavity),
                            "residual": residual,
                            "ratio": covered / len(prev_cavity),
                        }
                    )
                    break

            if piece_cells is None:
                return None
            if not (8 <= len(piece_cells) <= 13):
                return None
            chain_pieces.append({"color": piece_color, "coords": piece_cells})
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)

    if not (3 <= len(all_chain_pieces) <= 6):
        return None
    if not coverage_stats or not all(0.5 <= stat["ratio"] < 1.0 for stat in coverage_stats):
        return None

    num_decoys = rng.randint(4, 7)
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
        all_h = set().union(*notch_holes) if notch_holes else set()
        d_avail -= all_h | {(r + dr, c + dc) for r, c in all_h for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if len(d_avail) < 1:
            continue
        area = min(area, len(d_avail))
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))})
        if dn is None:
            continue
        if not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(dn)
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

    output = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        output[r, c] = bar_color
    for p in all_chain_pieces:
        for r, c in p["coords"]:
            output[r, c] = p["color"]

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color

    placed_occ = set(bar_cells)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)

    for p in pieces_order:
        coords = p["coords"]
        color = p["color"]
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
            if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                continue
            for r, c in pts:
                input_grid[r, c] = color
            placed_occ |= pts
            placed = True
            break
        if not placed:
            return None

    if return_meta:
        return input_grid, output, {"coverage_stats": coverage_stats, "num_chains": num_chains, "chain_lens": chain_lens}
    return input_grid, output


def generate_input():
    """Generate a valid input grid for the D12 partial-cavity variant."""
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
