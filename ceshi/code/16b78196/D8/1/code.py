# Variant of 16b78196
# Dimension: D8 — 条带宽度约束 (meta)
# Change: 每条链所有精灵共享固定条带宽度（4-5格），条带轴与杆跨越轴一致... → 条带宽度逐块递减

from __future__ import annotations

import numpy as np

from arc_utils import components_mask, find_components, get_bg

# Design analysis for the D8 variant
# a) Concrete behaviour:
#    原题里，同一链上的所有块都锁定在同一组列/行上，例如宽度 5 的条带会一直保持 5。
#    这个变体保留“根块塞进缺口，后续块填充前一块 bbox 内的空腔并继续远离横杆/竖杆生长”的主逻辑，
#    但把条带改成逐块收缩：宽度序列只能是 5→4→3 或 4→3→2。
#    具体像素上，上一块 bbox 里留出的黑色空腔只出现在“下一条更窄的子条带”中，被缩掉的那一列/行必须被上一块填实。
#    例：水平杆上方一条链，根块跨列 10..14；若第二块宽度要减到 4，则根块 bbox 内的黑色空腔只能跨列 10..13
#    或 11..14，另一侧那一整列在根块中必须是实心。solver 的关键差异就发生在“延伸块候选”的条带推断上：
#    原逻辑直接沿用根块的 A..B；现在必须先从空腔的列/行跨度恢复下一块的新 A..B。
# b) Complexity calibration:
#    只允许每步收缩 1 格，保持自由度很小；同时随机保留“收缩到起始侧”或“收缩到末端侧”两种锥形，
#    这样视觉上明显是 taper，但不会引入难以枚举的大量分支。链长仍为 2-3，因而每个样例里常见 4→3、5→4
#    或 5→4→3 这类重复模式，人眼可以从少量样例归纳。
# c) Hidden assumptions audited in main():
#    - 原 `placement_pts(..., strip=(axis,A,B))` 假定所有后续块都与根块等宽；变体下不成立，必须改成用当前头块空腔的跨度。
#    - 原 `aligned_positions(axis, A, cavity, v)` 假定 A 永远不变；现在 A 可能因收缩方向改变而右移/下移 1。
#    - 原 search 回溯只恢复 `head`；变体里链状态还包含会变化的 `A,B`，回溯时必须一起恢复，否则下一次分支会沿用错误宽度。
#    - 原代码没有显式要求“继续向远离杆的一侧延伸”；原 generator 里这个条件由构造天然保证。变体继续依赖这个性质，
#      因此 solver 现在显式检查候选块必须在生长方向上越过上一块 bbox。
# d) Placement legality:
#    generator 仍采用 output-first。构造每块时只在预定矩形中生长，所有 seeds 都在 available 内；条带收缩后，
#    被丢弃的一列/行会被整条 seed stripe 填满，从而不会在旧宽度位置残留黑色空腔。散射输入时仍禁止与杆或别的块重叠/接触。
# e) Visual observability:
#    变体的关键线索直接体现在 output 的链轮廓：每往外一块，横向/纵向宽度缩 1，形成单侧锥形收敛。这个特征不依赖颜色值。
# f) Visual cleanliness:
#    仍然只有一根大杆、1-2 条链和若干诱饵缺口；没有额外标记层。收缩只发生在条带边界，链内部结构比原题更规整，不会更乱。
# g) Color reasoning:
#    颜色仍只用于区分块，不参与排序或几何决策。每块单色且通常不同色，解题者只需要看形状与位置。
# Post-review of new solver assumptions:
#    solver 现在假设：上一块 bbox 内的黑色空腔在条带轴上的跨度恰好比上一块窄 1，并且这组空腔能与下一块原样匹配。
#    generator 通过显式构造 cavity span、强制 dropped stripe 填实、并在输出验证前调用 main(input) 来保证这些假设成立。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble colored sprites into a central bar's notches, chaining successive pieces along shrinking strips.

    Compared with the base task, each chain is tapered:
    - the root piece sets the initial strip width (4-5),
    - every continuation piece must use the bg cavity inside the previous piece's bbox,
    - that cavity spans a strip exactly 1 cell narrower than the previous piece.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
    bar_occ = set((int(r), int(c)) for r, c in bar["coords"])
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
    for r, c in bar_occ:
        out[r, c] = bar_color

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        rel_set = set(rel)
        return {"rel": rel, "rel_set": rel_set, "h": h, "w": w}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def bar_notches(bar_mask, bar_r0, bar_c0):
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

            hole_gl = set((bar_r0 + int(rr), bar_c0 + int(cc)) for rr, cc in pts)
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            grow_sign = -1 if side in ("top", "left") else 1
            notches.append({"axis": axis, "hole": hole_gl, "mouth_n": mouth_n, "side": side, "grow_sign": grow_sign})
        return notches

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    def pixels(v, top, left):
        return set((top + lr, left + lc) for lr, lc in v["rel"])

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
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, v):
        pos = set()
        for rH, cH in hole:
            for lr, lc in v["rel"]:
                pos.add((rH - lr, cH - lc))
        return pos

    def span_of_cells(axis, pts):
        vals = [c for _, c in pts] if axis == "cols" else [r for r, _ in pts]
        return int(min(vals)), int(max(vals))

    def extends_outward(pts, head, axis, grow_sign):
        top, left = int(head["top"]), int(head["left"])
        h, w = int(head["shape"]["h"]), int(head["shape"]["w"])
        if axis == "cols":
            edge = top if grow_sign < 0 else top + h - 1
            return any((r < edge) if grow_sign < 0 else (r > edge) for r, _ in pts)
        edge = left if grow_sign < 0 else left + w - 1
        return any((c < edge) if grow_sign < 0 else (c > edge) for _, c in pts)

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
                            "grow_sign": int(n["grow_sign"]),
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
            cur_A, cur_B = int(chain["A"]), int(chain["B"])
            cur_width = cur_B - cur_A + 1
            if cur_width <= 1:
                return []

            head = chain["head"]
            head_pts = pixels(head["shape"], head["top"], head["left"])
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

            next_A, next_B = span_of_cells(axis, cavity)
            if next_B - next_A + 1 != cur_width - 1:
                return []

            moves = []
            for pi in remaining_set:
                v = pieces[pi]["shape"]
                for top, left in aligned_positions(axis, next_A, cavity, v):
                    pts = placement_pts(v, top, left, occ_set, must_cover=cavity, strip=(axis, next_A, next_B))
                    if pts is None:
                        continue
                    if not touches(pts, head_pts):
                        continue
                    if not extends_outward(pts, head, axis, chain["grow_sign"]):
                        continue
                    moves.append((pi, top, left, pts, v, next_A, next_B))
            return moves

        for ch in chains:
            moves = gen_extend_moves(ch, remaining)
            if not moves:
                continue
            for pi, top, left, pts, v, next_A, next_B in moves[:8]:
                for r, c in pts:
                    out_grid[r, c] = int(pieces[pi]["color"])
                old_head = ch["head"]
                old_A, old_B = ch["A"], ch["B"]
                ch["head"] = {"top": int(top), "left": int(left), "shape": v}
                ch["A"], ch["B"] = int(next_A), int(next_B)
                occ2 = set(occ_set) | set(pts)
                rem2 = set(remaining)
                rem2.remove(pi)
                if search(out_grid, occ2, chains, rem2):
                    return True
                ch["head"] = old_head
                ch["A"], ch["B"] = old_A, old_B
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
                        "grow_sign": root["grow_sign"],
                        "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
                    }
                )

            if not ok:
                continue
            remaining = set(range(len(pieces))) - used_p
            search(out0, occ0, chains, remaining)

    return best_out


def _connected(cells):
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
    _D4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for __ in range(target * 20):
            if len(cells) >= target:
                break
            frontier = [nb for r, c in cells for dr, dc in _D4 if (nb := (r + dr, c + dc)) in available and nb not in cells]
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cm, cx = min((c for _, c in cells), default=B), max((c for _, c in cells), default=A)
                    if cm > A:
                        p = [nb for nb in frontier if nb[1] < cm]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                    if cx < B:
                        p = [nb for nb in frontier if nb[1] > cx]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                else:
                    cm, cx = min((r for r, _ in cells), default=B), max((r for r, _ in cells), default=A)
                    if cm > A:
                        p = [nb for nb in frontier if nb[0] < cm]
                        if p:
                            cells.add(rng.choice(p))
                            continue
                    if cx < B:
                        p = [nb for nb in frontier if nb[0] > cx]
                        if p:
                            cells.add(rng.choice(p))
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


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _span_of_cells(axis, pts):
    vals = [c for _, c in pts] if axis == "cols" else [r for r, _ in pts]
    return min(vals), max(vals)


def _shrink_span(A, B, keep_low_side):
    if B - A + 1 <= 1:
        return None
    if keep_low_side:
        return A, B - 1
    return A + 1, B


def _cavity_in_bbox(coords, occ_set):
    r0, c0, r1, c1 = _bbox(coords)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in occ_set}


def _outward_boundary_value(orientation, side, bbox):
    r0, c0, r1, c1 = bbox
    if orientation == "horizontal":
        return r0 if side == "top" else r1
    return c0 if side == "left" else c1


def _touches_outward_face(cavity, orientation, side, bbox):
    edge = _outward_boundary_value(orientation, side, bbox)
    if orientation == "horizontal":
        return any(r == edge for r, _ in cavity)
    return any(c == edge for _, c in cavity)


def _dropped_stripe_cells(orientation, bbox, cur_A, cur_B, next_A, next_B):
    r0, c0, r1, c1 = bbox
    if orientation == "horizontal":
        if next_A == cur_A:
            drop_col = cur_B
        else:
            drop_col = cur_A
        return {(r, drop_col) for r in range(r0, r1 + 1) if c0 <= drop_col <= c1}
    if next_A == cur_A:
        drop_row = cur_B
    else:
        drop_row = cur_A
    return {(drop_row, c) for c in range(c0, c1 + 1) if r0 <= drop_row <= r1}


def _planned_bbox(orientation, side, cur_A, cur_B, inner_lo, inner_hi, ext_depth, H, W):
    if orientation == "horizontal":
        if side == "top":
            r0, r1 = max(0, inner_lo - ext_depth), inner_hi
        else:
            r0, r1 = inner_lo, min(H - 1, inner_hi + ext_depth)
        return r0, cur_A, r1, cur_B
    if side == "left":
        c0, c1 = max(0, inner_lo - ext_depth), inner_hi
    else:
        c0, c1 = inner_lo, min(W - 1, inner_hi + ext_depth)
    return cur_A, c0, cur_B, c1


def _build_piece(rng, available, seeds, axis, span_range, area_lo, area_hi, retries=40):
    seeds = set(seeds)
    available = set(available)
    if not seeds or not seeds.issubset(available):
        return None
    lo = max(area_lo, len(seeds))
    hi = min(area_hi, len(available))
    if lo > hi:
        return None
    targets = list(range(lo, hi + 1))
    rng.shuffle(targets)
    for target in targets:
        piece = _grow_connected(rng, available, target, seeds, must_span_axis=axis, span_range=span_range, retries=retries)
        if piece is not None:
            return piece
    return None


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
    bar_cells = set((r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1))

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

    for chain_len in chain_lens:
        root_width = rng.randint(4, 5)

        if orientation == "horizontal":
            sides = ["top", "bottom"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "top" else space_after) >= chain_len * 3 + 1), None)
            if side is None:
                return None
            cur_A = rng.randint(1, W - root_width - 1)
            cur_B = cur_A + root_width - 1
            axis = "cols"
        else:
            sides = ["left", "right"]
            rng.shuffle(sides)
            side = next((s for s in sides if (space_before if s == "left" else space_after) >= chain_len * 3 + 1), None)
            if side is None:
                return None
            cur_A = rng.randint(1, H - root_width - 1)
            cur_B = cur_A + root_width - 1
            axis = "rows"

        keep_low_side = rng.choice([True, False])
        width_seq = [root_width - i for i in range(chain_len)]
        if width_seq[-1] < 2:
            return None
        spans = []
        A, B = cur_A, cur_B
        spans.append((A, B))
        for _ in range(1, chain_len):
            nxt = _shrink_span(A, B, keep_low_side)
            if nxt is None:
                return None
            A, B = nxt
            spans.append((A, B))

        notch_depth = rng.randint(1, 2)
        notch_area = rng.randint(1, min(4, notch_depth * root_width))

        if orientation == "horizontal":
            nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
            nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
            n_avail = {(r, c) for r in range(nr0, nr1 + 1) for c in range(spans[0][0], spans[0][1] + 1) if (r, c) in bar_cells}
        else:
            nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
            nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
            n_avail = {(r, c) for r in range(spans[0][0], spans[0][1] + 1) for c in range(nc0, nc1 + 1) if (r, c) in bar_cells}

        all_holes = set().union(*notch_holes) if notch_holes else set()
        n_avail -= all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        if len(n_avail) < notch_area:
            return None
        notch = _grow_connected(rng, n_avail, notch_area, {rng.choice(list(n_avail))}, retries=8)
        if notch is None:
            return None
        bar_cells -= notch
        notch_holes.append(notch)

        chain_pieces = []
        prev_cavity = None
        for piece_idx in range(chain_len):
            if color_idx >= len(avail_colors):
                return None
            piece_color = avail_colors[color_idx]
            color_idx += 1

            cur_A, cur_B = spans[piece_idx]
            is_root = piece_idx == 0
            is_tail = piece_idx == chain_len - 1
            next_span = spans[piece_idx + 1] if not is_tail else None

            piece_cells = None
            for _attempt in range(80):
                ext_depth = rng.randint(2, 4)
                if is_root:
                    if orientation == "horizontal":
                        inner_lo = inner_hi = max(r for r, _ in notch) if side == "top" else min(r for r, _ in notch)
                    else:
                        inner_lo = inner_hi = max(c for _, c in notch) if side == "left" else min(c for _, c in notch)
                    bbox = _planned_bbox(orientation, side, cur_A, cur_B, inner_lo, inner_hi, ext_depth, H, W)
                    r0, c0, r1, c1 = bbox
                    available = {
                        (r, c)
                        for r in range(r0, r1 + 1)
                        for c in range(c0, c1 + 1)
                        if (r, c) in notch or (r, c) not in occ_set
                    }
                    seeds = set(notch)
                else:
                    cavity = prev_cavity
                    if not cavity or not _connected(cavity):
                        return None
                    if orientation == "horizontal":
                        inner_lo = min(r for r, _ in cavity)
                        inner_hi = max(r for r, _ in cavity)
                    else:
                        inner_lo = min(c for _, c in cavity)
                        inner_hi = max(c for _, c in cavity)
                    bbox = _planned_bbox(orientation, side, cur_A, cur_B, inner_lo, inner_hi, ext_depth, H, W)
                    r0, c0, r1, c1 = bbox
                    available = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in occ_set}
                    seeds = set(cavity)

                if next_span is not None:
                    next_A, next_B = next_span
                    seeds |= _dropped_stripe_cells(orientation, bbox, cur_A, cur_B, next_A, next_B) & available

                bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if bbox_area < 8:
                    continue

                if is_tail:
                    if not (8 <= len(available) <= 13):
                        continue
                    piece = set(available)
                else:
                    max_fill = min(13, int(np.floor(bbox_area * 0.80)))
                    min_fill = max(8, int(np.ceil(bbox_area * 0.54)))
                    if max_fill >= bbox_area:
                        max_fill = bbox_area - 1
                    if max_fill < min_fill:
                        continue
                    piece = _build_piece(rng, available, seeds, axis, (cur_A, cur_B), min_fill, max_fill, retries=8)
                    if piece is None:
                        continue

                if not (8 <= len(piece) <= 13):
                    continue
                if not _connected(piece):
                    continue

                occ_after = occ_set | piece
                if is_root and not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in piece):
                    continue

                if next_span is not None:
                    cavity2 = _cavity_in_bbox(piece, occ_after)
                    if not cavity2 or not _connected(cavity2):
                        continue
                    if _span_of_cells(axis, cavity2) != next_span:
                        continue
                    if not _touches_outward_face(cavity2, orientation, side, _bbox(piece)):
                        continue
                    prev_cavity_candidate = cavity2
                else:
                    cavity2 = _cavity_in_bbox(piece, occ_after)
                    if cavity2:
                        continue
                    prev_cavity_candidate = None

                piece_cells = piece
                prev_cavity = prev_cavity_candidate
                break

            if piece_cells is None:
                return None

            chain_pieces.append({"color": piece_color, "coords": piece_cells})
            occ_set |= piece_cells

        all_chain_pieces.extend(chain_pieces)

    if not (3 <= len(all_chain_pieces) <= 6):
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
        dn = _grow_connected(rng, d_avail, area, {rng.choice(list(d_avail))}, retries=8)
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

    return input_grid, output


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is None:
            continue
        inp, out = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 300 attempts")
