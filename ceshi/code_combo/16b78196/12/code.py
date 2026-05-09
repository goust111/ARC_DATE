# Combo variant of 16b78196 (index 12)
# combo_spec: D4/3+D8/1
# Name: 单条长链+收缩条带

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg

# Design analysis for the combo variant
# a) Concrete behaviour:
#    原题允许 1-2 条独立短链，每条链 2-3 块，且同链条带宽度固定。这个融合版把两件事同时改掉：
#    只保留 1 条长链，并且每往外一块条带宽度缩 1。一个具体例子是水平横杆下方的 5 块链：
#    宽度序列可能是 6→5→4→3→2。根块先把某个缺口补上；它的 bbox 外侧留出一块宽 5 的黑色空腔；
#    第二块必须正好填满这块空腔，并在更外侧留下宽 4 的空腔；之后重复，直到尾块宽 2 且不再留空腔。
#    与原题相比，solver 不再需要在多个缺口之间交替延伸，而是必须连续追踪更深的单链依赖；同时不能把条带
#    当成常量，而要从每一步的 cavity 跨度恢复“下一块的新宽度和新位置”。
# b) Complexity calibration:
#    长链长度控制在 4-6，宽度每步只缩 1，收缩侧只允许“保留低端”或“保留高端”两类，因此视觉上能稳定形成
#    单侧锥形，但自由度仍然很低。为了让 6 块链在 30x30 网格中物理可实现，generator 把尾宽固定为 2，
#    因而根宽自然变成 `chain_len + 1`，即 5/6/7。这样每个样例都显式出现长链和 taper，而不会退化成原题的
#    4-5 宽短链。
# c) Hidden assumptions audited in main():
#    - 原代码默认可以同时放多个 root；这里改为单 root 搜索，否则会把 decoy notch 误当第二条链起点。
#    - 原 `placement_pts(..., strip=(axis,A,B))` 假定后续块与 root 同宽；这里改为每步先从 cavity 求 `next_A,next_B`。
#    - 原 `aligned_positions(axis, A, cavity, v)` 默认 A 不变；现在 A 可能因收缩而右移/下移 1，所以必须跟随 cavity。
#    - 原 search 只需回溯 `head`；这里链状态里的 `A,B` 也会逐步缩窄，回溯时必须一起恢复。
#    - 原 solver 没有显式要求“继续远离杆延伸”；原 generator 靠构造保证。长链场景里这条约束更重要，否则某块可能
#      仅仅覆盖 cavity 而没有新外延，导致链深度被错误压扁，所以 solver 额外检查候选块必须越过上一块外侧边界。
# d) Placement legality:
#    generator 仍然采用 output-first。先在输出里构造 bar、1 个真实 notch、4-7 个 decoy notch 和 1 条 4-6 块链，
#    再把每块原样散射到输入。所有 piece 均保持原朝向、不旋转；散射时禁止与 bar 或其他 piece 重叠或 4 邻接。
# e) Visual observability:
#    输出里只有 1 条真实链，因此人眼会看到“唯一能连续嵌套并逐步变窄的一串块”。由于每块宽度严格减 1，
#    4-6 个样本里会重复出现 5→4→3→2、6→5→4→3→2 或 7→6→5→4→3→2 这类可见模式，不需要读取颜色值。
# f) Visual cleanliness:
#    仍然只有一根中心杆、若干边界缺口和一串链。decoy notch 只留在 bar 内部边缘，不再引入第二条链，因此
#    画面比原题更集中。链块本身是“矩形减去更窄的外侧 cavity”的阶梯形，层次清楚，不会在同一区域多重叠压。
# g) Color reasoning:
#    颜色只用于区分不同 piece，不编码顺序。顺序由几何关系决定：谁能填前一块的 cavity，谁就是下一块。
# Post-review of new solver assumptions:
#    solver 现在额外假设：
#    1. 输入中只有一条真实链，且所有小块都属于该链；
#    2. 每一步 cavity 在条带轴上的跨度恰好比当前块窄 1；
#    3. 每一步 cavity 都贴着当前块的外侧边界，因此下一块一定还能向外再长一段。
#    generator 通过单链构造、固定宽度递减、以及“新 cavity 总放在外侧边界”这三点显式保证这些假设成立。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered sprites into a single long tapered chain growing from one bar notch.

    This combo variant differs from the base task in two coupled ways:
    - exactly one chain is real, but it is longer (4-6 pieces),
    - each continuation uses a strip that is 1 cell narrower than the previous piece.
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
    bar_occ = {(int(r), int(c)) for r, c in bar["coords"]}
    bar_mask = np.zeros((bar_h, bar_w), dtype=bool)
    for r, c in bar_occ:
        bar_mask[r - bar_r0, c - bar_c0] = True
        out[r, c] = bar_color

    def shape_from_coords(coords: np.ndarray, bbox):
        r0, c0, r1, c1 = map(int, bbox)
        rel = [(int(r - r0), int(c - c0)) for r, c in coords]
        return {"rel": rel, "h": int(r1 - r0 + 1), "w": int(c1 - c0 + 1)}

    pieces = []
    for p in pieces_raw:
        r0, c0 = map(int, p["bbox"][:2])
        pieces.append({"color": int(p["color"]), "shape": shape_from_coords(p["coords"], p["bbox"]), "orig": (r0, c0)})

    def bar_notches(mask, top, left):
        empty = ~mask
        h, w = empty.shape
        notches = []
        for pts in components_mask(empty, connectivity=4):
            r = pts[:, 0]
            c = pts[:, 1]
            top_n = int((r == 0).sum())
            bot_n = int((r == h - 1).sum())
            left_n = int((c == 0).sum())
            right_n = int((c == w - 1).sum())
            if top_n + bot_n + left_n + right_n == 0:
                continue
            hole = {(top + int(rr), left + int(cc)) for rr, cc in pts}
            side, mouth_n = max(
                [(name, cnt) for name, cnt in [("top", top_n), ("bottom", bot_n), ("left", left_n), ("right", right_n)] if cnt > 0],
                key=lambda x: x[1],
            )
            axis = "cols" if side in ("top", "bottom") else "rows"
            grow_sign = -1 if side in ("top", "left") else 1
            notches.append({"axis": axis, "hole": hole, "mouth_n": mouth_n, "side": side, "grow_sign": grow_sign})
        return notches

    def pixels(shape, top, left):
        return {(top + dr, left + dc) for dr, dc in shape["rel"]}

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
        pts = pixels(shape, top, left)
        if pts & occ:
            return None
        if must_cover is not None and not (must_cover <= pts):
            return None
        return pts

    def aligned_positions(axis, A, cover, shape):
        if axis == "cols":
            left = A
            seen = set()
            for r_need, c_need in cover:
                for lr, lc in shape["rel"]:
                    if c_need == left + lc:
                        seen.add(r_need - lr)
            for top in seen:
                yield top, left
        else:
            top = A
            seen = set()
            for r_need, c_need in cover:
                for lr, lc in shape["rel"]:
                    if r_need == top + lr:
                        seen.add(c_need - lc)
            for left in seen:
                yield top, left

    def touches(pts, other):
        for r, c in pts:
            for dr, dc in DIR4:
                if (r + dr, c + dc) in other:
                    return True
        return False

    def hole_positions(hole, shape):
        pos = set()
        for r_hole, c_hole in hole:
            for lr, lc in shape["rel"]:
                pos.add((r_hole - lr, c_hole - lc))
        return pos

    def span_of_cells(axis, pts):
        vals = [c for _, c in pts] if axis == "cols" else [r for r, _ in pts]
        return int(min(vals)), int(max(vals))

    def extends_outward(pts, head, axis, grow_sign):
        top, left = int(head["top"]), int(head["left"])
        hh, hw = int(head["shape"]["h"]), int(head["shape"]["w"])
        if axis == "cols":
            edge = top if grow_sign < 0 else top + hh - 1
            return any((r < edge) if grow_sign < 0 else (r > edge) for r, _ in pts)
        edge = left if grow_sign < 0 else left + hw - 1
        return any((c < edge) if grow_sign < 0 else (c > edge) for _, c in pts)

    notches = bar_notches(bar_mask, bar_r0, bar_c0)

    root_cands = []
    for ni, notch in enumerate(notches):
        axis = notch["axis"]
        for pi, piece in enumerate(pieces):
            shape = piece["shape"]
            for top, left in hole_positions(notch["hole"], shape):
                pts = placement_pts(shape, top, left, bar_occ, must_cover=notch["hole"])
                if pts is None:
                    continue
                if axis == "cols":
                    A, B = int(left), int(left + shape["w"] - 1)
                    width = int(shape["w"])
                else:
                    A, B = int(top), int(top + shape["h"] - 1)
                    width = int(shape["h"])
                mv = abs(int(top) - int(piece["orig"][0])) + abs(int(left) - int(piece["orig"][1]))
                root_cands.append(
                    (
                        (-width, int(mv), -int(notch["mouth_n"])),
                        {
                            "ni": ni,
                            "pi": pi,
                            "top": int(top),
                            "left": int(left),
                            "axis": axis,
                            "A": A,
                            "B": B,
                            "grow_sign": int(notch["grow_sign"]),
                            "shape": shape,
                        },
                    )
                )

    root_cands.sort(key=lambda x: x[0])
    cand_roots = [pair[1] for pair in root_cands[:40]]

    best_out = out.copy()
    best_key = None

    def search(out_grid, occ_set, chain, remaining):
        nonlocal best_out, best_key
        placed_cnt = len(pieces) - len(remaining)
        key = (placed_cnt, -len(remaining), -len(occ_set))
        if best_key is None or key > best_key:
            best_key = key
            best_out = out_grid.copy()
        if not remaining:
            return True

        axis = chain["axis"]
        cur_A, cur_B = int(chain["A"]), int(chain["B"])
        cur_width = cur_B - cur_A + 1
        if cur_width <= 1:
            return False

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
            return False

        next_A, next_B = span_of_cells(axis, cavity)
        if next_B - next_A + 1 != cur_width - 1:
            return False

        moves = []
        for pi in remaining:
            shape = pieces[pi]["shape"]
            needed_dim = int(shape["w"] if axis == "cols" else shape["h"])
            if needed_dim != next_B - next_A + 1:
                continue
            for top, left in aligned_positions(axis, next_A, cavity, shape):
                pts = placement_pts(shape, top, left, occ_set, must_cover=cavity, strip=(axis, next_A, next_B))
                if pts is None:
                    continue
                if not touches(pts, head_pts):
                    continue
                if not extends_outward(pts, head, axis, chain["grow_sign"]):
                    continue
                mv = abs(int(top) - int(pieces[pi]["orig"][0])) + abs(int(left) - int(pieces[pi]["orig"][1]))
                moves.append((mv, -len(pts), pi, int(top), int(left), pts, shape))
        moves.sort(key=lambda x: x[:2])

        for _, _, pi, top, left, pts, shape in moves[:10]:
            for r, c in pts:
                out_grid[r, c] = int(pieces[pi]["color"])
            old_head = chain["head"]
            old_A, old_B = chain["A"], chain["B"]
            chain["head"] = {"top": int(top), "left": int(left), "shape": shape}
            chain["A"], chain["B"] = int(next_A), int(next_B)
            occ2 = set(occ_set) | set(pts)
            rem2 = set(remaining)
            rem2.remove(pi)
            if search(out_grid, occ2, chain, rem2):
                return True
            chain["head"] = old_head
            chain["A"], chain["B"] = old_A, old_B
            for r, c in pts:
                out_grid[r, c] = int(bg)
            for r, c in bar_occ:
                out_grid[r, c] = bar_color
        return False

    for root in cand_roots:
        occ0 = set(bar_occ)
        out0 = np.full((H, W), int(bg), dtype=int)
        for r, c in bar_occ:
            out0[r, c] = bar_color
        pts = pixels(root["shape"], root["top"], root["left"])
        if pts & occ0:
            continue
        for r, c in pts:
            out0[r, c] = int(pieces[root["pi"]]["color"])
        occ0 |= pts
        chain = {
            "axis": root["axis"],
            "A": root["A"],
            "B": root["B"],
            "grow_sign": root["grow_sign"],
            "head": {"top": root["top"], "left": root["left"], "shape": root["shape"]},
        }
        remaining = set(range(len(pieces))) - {root["pi"]}
        if search(out0, occ0, chain, remaining):
            return out0

    return best_out


def _connected(cells: set[tuple[int, int]]) -> bool:
    if len(cells) <= 1:
        return True
    cells = set(cells)
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def _grow_connected(
    rng: random.Random,
    available: set[tuple[int, int]],
    target: int,
    seeds: set[tuple[int, int]],
    must_span_axis: str | None = None,
    span_range: tuple[int, int] | None = None,
    retries: int = 6,
) -> set[tuple[int, int]] | None:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    available = set(available)
    seeds = set(seeds)
    for _ in range(retries):
        cells = set(seeds)
        if not cells.issubset(available):
            return None
        for _step in range(max(20, target * 20)):
            if len(cells) >= target:
                break
            frontier = [nb for r, c in cells for dr, dc in dirs if (nb := (r + dr, c + dc)) in available and nb not in cells]
            if not frontier:
                break
            if must_span_axis and span_range:
                A, B = span_range
                if must_span_axis == "cols":
                    cur_lo = min((c for _, c in cells), default=B)
                    cur_hi = max((c for _, c in cells), default=A)
                    if cur_lo > A:
                        low = [nb for nb in frontier if nb[1] < cur_lo]
                        if low:
                            cells.add(rng.choice(low))
                            continue
                    if cur_hi < B:
                        high = [nb for nb in frontier if nb[1] > cur_hi]
                        if high:
                            cells.add(rng.choice(high))
                            continue
                else:
                    cur_lo = min((r for r, _ in cells), default=B)
                    cur_hi = max((r for r, _ in cells), default=A)
                    if cur_lo > A:
                        low = [nb for nb in frontier if nb[0] < cur_lo]
                        if low:
                            cells.add(rng.choice(low))
                            continue
                    if cur_hi < B:
                        high = [nb for nb in frontier if nb[0] > cur_hi]
                        if high:
                            cells.add(rng.choice(high))
                            continue
            cells.add(rng.choice(frontier))
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


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _span_of_cells(axis: str, pts: set[tuple[int, int]]) -> tuple[int, int]:
    vals = [c for _, c in pts] if axis == "cols" else [r for r, _ in pts]
    return min(vals), max(vals)


def _rect_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    if r0 > r1 or c0 > c1:
        return set()
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _cavity_in_bbox(coords: set[tuple[int, int]], occ_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in occ_set}


def _shrink_span(A: int, B: int, keep_low_side: bool) -> tuple[int, int] | None:
    if B - A + 1 <= 1:
        return None
    if keep_low_side:
        return A, B - 1
    return A + 1, B


def _pick_depth_plan(
    rng: random.Random,
    widths: list[int],
    notch_area_lo: int,
    notch_area_hi: int,
    max_space: int,
) -> tuple[int, list[int], list[int]] | None:
    n = len(widths)
    for _ in range(2000):
        notch_area = rng.randint(notch_area_lo, notch_area_hi)
        cavity_depths: list[int] = []
        outside_depths: list[int] = []

        d0 = rng.randint(1, 2)
        o0_choices = [o for o in range(2, 5) if 8 <= notch_area + widths[0] * o - widths[1] * d0 <= 13]
        if not o0_choices:
            continue
        o0 = rng.choice(o0_choices)
        cavity_depths.append(d0)
        outside_depths.append(o0)

        prev_d = d0
        ok = True
        for idx in range(1, n - 1):
            w = widths[idx]
            next_w = widths[idx + 1]
            choices = []
            for d_next in range(1, 4):
                for out_d in range(1, 6):
                    # Keep the new cavity entirely in the freshly added outer band.
                    # Otherwise the next piece would only partially cover the previous cavity,
                    # which breaks the observable "fill previous hole, then extend outward" rule.
                    if d_next > out_d:
                        continue
                    area = w * (prev_d + out_d) - next_w * d_next
                    if 8 <= area <= 13:
                        choices.append((d_next, out_d))
            if not choices:
                ok = False
                break
            d_next, out_d = rng.choice(choices)
            cavity_depths.append(d_next)
            outside_depths.append(out_d)
            prev_d = d_next
        if not ok:
            continue

        tail_choices = [out_d for out_d in range(1, 7) if 8 <= widths[-1] * (prev_d + out_d) <= 13]
        if not tail_choices:
            continue
        outside_depths.append(rng.choice(tail_choices))
        if sum(outside_depths) > max_space:
            continue
        return notch_area, outside_depths, cavity_depths
    return None


def _boundary_cells(side: str, cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if side == "top":
        edge = min(r for r, _ in cells)
        return [rc for rc in cells if rc[0] == edge]
    if side == "bottom":
        edge = max(r for r, _ in cells)
        return [rc for rc in cells if rc[0] == edge]
    if side == "left":
        edge = min(c for _, c in cells)
        return [rc for rc in cells if rc[1] == edge]
    edge = max(c for _, c in cells)
    return [rc for rc in cells if rc[1] == edge]


def _build_root_piece(
    orientation: str,
    side: str,
    notch: set[tuple[int, int]],
    cur_span: tuple[int, int],
    next_span: tuple[int, int],
    outside_depth: int,
    cavity_depth: int,
    bar_bbox: tuple[int, int, int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    bar_r0, bar_c0, bar_r1, bar_c1 = bar_bbox
    cur_A, cur_B = cur_span
    next_A, next_B = next_span
    if orientation == "horizontal":
        if side == "top":
            outside = _rect_cells(bar_r0 - outside_depth, cur_A, bar_r0 - 1, cur_B)
            cavity = _rect_cells(bar_r0 - outside_depth, next_A, bar_r0 - outside_depth + cavity_depth - 1, next_B)
        else:
            outside = _rect_cells(bar_r1 + 1, cur_A, bar_r1 + outside_depth, cur_B)
            cavity = _rect_cells(bar_r1 + outside_depth - cavity_depth + 1, next_A, bar_r1 + outside_depth, next_B)
    else:
        if side == "left":
            outside = _rect_cells(cur_A, bar_c0 - outside_depth, cur_B, bar_c0 - 1)
            cavity = _rect_cells(next_A, bar_c0 - outside_depth, next_B, bar_c0 - outside_depth + cavity_depth - 1)
        else:
            outside = _rect_cells(cur_A, bar_c1 + 1, cur_B, bar_c1 + outside_depth)
            cavity = _rect_cells(next_A, bar_c1 + outside_depth - cavity_depth + 1, next_B, bar_c1 + outside_depth)
    piece = (outside - cavity) | set(notch)
    return piece, cavity


def _build_mid_piece(
    orientation: str,
    side: str,
    prev_cavity: set[tuple[int, int]],
    cur_span: tuple[int, int],
    next_span: tuple[int, int],
    outside_depth: int,
    cavity_depth: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cur_A, cur_B = cur_span
    next_A, next_B = next_span
    pr0, pc0, pr1, pc1 = _bbox(prev_cavity)
    if orientation == "horizontal":
        if side == "top":
            full = _rect_cells(pr0 - outside_depth, cur_A, pr1, cur_B)
            cavity = _rect_cells(pr0 - outside_depth, next_A, pr0 - outside_depth + cavity_depth - 1, next_B)
        else:
            full = _rect_cells(pr0, cur_A, pr1 + outside_depth, cur_B)
            cavity = _rect_cells(pr1 + outside_depth - cavity_depth + 1, next_A, pr1 + outside_depth, next_B)
    else:
        if side == "left":
            full = _rect_cells(cur_A, pc0 - outside_depth, cur_B, pc1)
            cavity = _rect_cells(next_A, pc0 - outside_depth, next_B, pc0 - outside_depth + cavity_depth - 1)
        else:
            full = _rect_cells(cur_A, pc0, cur_B, pc1 + outside_depth)
            cavity = _rect_cells(next_A, pc1 + outside_depth - cavity_depth + 1, next_B, pc1 + outside_depth)
    return full - cavity, cavity


def _build_tail_piece(
    orientation: str,
    side: str,
    prev_cavity: set[tuple[int, int]],
    cur_span: tuple[int, int],
    outside_depth: int,
) -> set[tuple[int, int]]:
    cur_A, cur_B = cur_span
    pr0, pc0, pr1, pc1 = _bbox(prev_cavity)
    if orientation == "horizontal":
        if side == "top":
            return _rect_cells(pr0 - outside_depth, cur_A, pr1, cur_B)
        return _rect_cells(pr0, cur_A, pr1 + outside_depth, cur_B)
    if side == "left":
        return _rect_cells(cur_A, pc0 - outside_depth, cur_B, pc1)
    return _rect_cells(cur_A, pc0, cur_B, pc1 + outside_depth)


def _validate_piece_and_cavity(
    piece: set[tuple[int, int]],
    occ_before: set[tuple[int, int]],
    planned_cavity: set[tuple[int, int]] | None,
    axis: str,
    next_span: tuple[int, int] | None,
) -> bool:
    if not (8 <= len(piece) <= 13):
        return False
    if not _connected(piece):
        return False
    if piece & occ_before:
        return False
    occ_after = set(occ_before) | set(piece)
    cavity = _cavity_in_bbox(piece, occ_after)
    if planned_cavity is None:
        return not cavity
    if cavity != planned_cavity:
        return False
    if not cavity or not _connected(cavity):
        return False
    return _span_of_cells(axis, cavity) == next_span


def _scatter_piece(
    rng: random.Random,
    grid: np.ndarray,
    placed_occ: set[tuple[int, int]],
    coords: set[tuple[int, int]],
    color: int,
    orientation: str,
    bar_r0: int,
    bar_r1: int,
    bar_c0: int,
    bar_c1: int,
) -> bool:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    p_r0, p_c0 = min(rs), min(cs)
    rel = [(r - p_r0, c - p_c0) for r, c in coords]
    p_h, p_w = max(rs) - p_r0 + 1, max(cs) - p_c0 + 1
    for _ in range(500):
        gap = rng.randint(2, 7)
        if orientation == "horizontal":
            top = bar_r0 - gap - p_h if rng.random() < 0.5 else bar_r1 + gap + 1
            left = rng.randint(0, max(0, grid.shape[1] - p_w))
        else:
            left = bar_c0 - gap - p_w if rng.random() < 0.5 else bar_c1 + gap + 1
            top = rng.randint(0, max(0, grid.shape[0] - p_h))
        if top < 0 or left < 0 or top + p_h > grid.shape[0] or left + p_w > grid.shape[1]:
            continue
        pts = {(top + dr, left + dc) for dr, dc in rel}
        if pts & placed_occ:
            continue
        if any((r + dr, c + dc) in placed_occ for r, c in pts for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            continue
        for r, c in pts:
            grid[r, c] = int(color)
        placed_occ |= pts
        return True
    return False


def _gen_attempt(seed: int | None = None):
    rng = random.Random(seed)
    H, W = 30, 30
    bg = 0

    orientation = rng.choice(["horizontal", "vertical"])
    thickness = rng.randint(5, 7)
    bar_color = rng.randint(1, 9)

    if orientation == "horizontal":
        center = int(rng.uniform(0.35, 0.53) * H)
        bar_r0 = max(0, center - thickness // 2)
        bar_r1 = min(H - 1, bar_r0 + thickness - 1)
        bar_r0 = bar_r1 - thickness + 1
        bar_c0, bar_c1 = 0, W - 1
        space_before = bar_r0
        space_after = H - 1 - bar_r1
    else:
        center = int(rng.uniform(0.35, 0.53) * W)
        bar_c0 = max(0, center - thickness // 2)
        bar_c1 = min(W - 1, bar_c0 + thickness - 1)
        bar_c0 = bar_c1 - thickness + 1
        bar_r0, bar_r1 = 0, H - 1
        space_before = bar_c0
        space_after = W - 1 - bar_c1

    bar_cells = {(r, c) for r in range(bar_r0, bar_r1 + 1) for c in range(bar_c0, bar_c1 + 1)}

    chain_len = rng.randint(4, 6)
    widths = list(range(chain_len + 1, 1, -1))
    keep_low_side = rng.choice([True, False])

    if orientation == "horizontal":
        axis = "cols"
        side_options = []
        if space_before >= 10:
            side_options.append(("top", space_before))
        if space_after >= 10:
            side_options.append(("bottom", space_after))
    else:
        axis = "rows"
        side_options = []
        if space_before >= 10:
            side_options.append(("left", space_before))
        if space_after >= 10:
            side_options.append(("right", space_after))
    if not side_options:
        return None
    side_options.sort(key=lambda x: -x[1])
    side, max_space = side_options[0]

    if orientation == "horizontal":
        root_A = rng.randint(1, W - widths[0] - 1)
    else:
        root_A = rng.randint(1, H - widths[0] - 1)
    root_B = root_A + widths[0] - 1

    spans = []
    A, B = root_A, root_B
    spans.append((A, B))
    for _ in range(1, chain_len):
        nxt = _shrink_span(A, B, keep_low_side)
        if nxt is None:
            return None
        A, B = nxt
        spans.append((A, B))

    notch_depth = rng.randint(1, 2)
    notch_area_hi = min(4, notch_depth * widths[0])
    plan = _pick_depth_plan(rng, widths, 2, notch_area_hi, max_space)
    if plan is None:
        return None
    notch_area, outside_depths, cavity_depths = plan

    if orientation == "horizontal":
        nr0 = bar_r0 if side == "top" else bar_r1 - notch_depth + 1
        nr1 = bar_r0 + notch_depth - 1 if side == "top" else bar_r1
        notch_box = {(r, c) for r in range(nr0, nr1 + 1) for c in range(spans[0][0], spans[0][1] + 1)}
    else:
        nc0 = bar_c0 if side == "left" else bar_c1 - notch_depth + 1
        nc1 = bar_c0 + notch_depth - 1 if side == "left" else bar_c1
        notch_box = {(r, c) for r in range(spans[0][0], spans[0][1] + 1) for c in range(nc0, nc1 + 1)}

    seed_choices = _boundary_cells(side, notch_box)
    notch = _grow_connected(rng, notch_box, notch_area, {rng.choice(seed_choices)}, retries=12)
    if notch is None:
        return None
    bar_cells -= notch
    notch_holes = [set(notch)]

    occ_set = set(bar_cells)
    all_chain_pieces = []
    avail_colors = [c for c in range(1, 10) if c != bar_color]
    rng.shuffle(avail_colors)

    prev_cavity = None
    for idx in range(chain_len):
        cur_span = spans[idx]
        color = avail_colors[idx]
        if idx == 0:
            piece, cavity = _build_root_piece(
                orientation,
                side,
                notch,
                cur_span,
                spans[idx + 1],
                outside_depths[idx],
                cavity_depths[idx],
                (bar_r0, bar_c0, bar_r1, bar_c1),
            )
            if not _validate_piece_and_cavity(piece, occ_set, cavity, axis, spans[idx + 1]):
                return None
            if not any(not (bar_r0 <= r <= bar_r1 and bar_c0 <= c <= bar_c1) for r, c in piece):
                return None
            prev_cavity = cavity
        elif idx < chain_len - 1:
            piece, cavity = _build_mid_piece(
                orientation,
                side,
                prev_cavity,
                cur_span,
                spans[idx + 1],
                outside_depths[idx],
                cavity_depths[idx],
            )
            if not _validate_piece_and_cavity(piece, occ_set, cavity, axis, spans[idx + 1]):
                return None
            prev_cavity = cavity
        else:
            piece = _build_tail_piece(orientation, side, prev_cavity, cur_span, outside_depths[idx])
            if not _validate_piece_and_cavity(piece, occ_set, None, axis, None):
                return None
        all_chain_pieces.append({"color": color, "coords": set(piece)})
        occ_set |= set(piece)

    if not (4 <= len(all_chain_pieces) <= 6):
        return None

    protected_bar = set()
    for piece in all_chain_pieces:
        pr0, pc0, pr1, pc1 = _bbox(piece["coords"])
        rr0 = max(bar_r0, pr0)
        rr1 = min(bar_r1, pr1)
        cc0 = max(bar_c0, pc0)
        cc1 = min(bar_c1, pc1)
        protected_bar |= _rect_cells(rr0, cc0, rr1, cc1)

    num_decoys = rng.randint(4, 7)
    decoys = 0
    for _ in range(700):
        if decoys >= num_decoys:
            break
        depth = rng.randint(1, 2)
        area = rng.randint(1, min(6, depth * 5))
        if orientation == "horizontal":
            dside = rng.choice(["top", "bottom"])
            dr0 = bar_r0 if dside == "top" else bar_r1 - depth + 1
            dr1 = bar_r0 + depth - 1 if dside == "top" else bar_r1
            dc0 = rng.randint(bar_c0, bar_c1)
            dc1 = min(bar_c1, dc0 + rng.randint(0, 4))
            region = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        else:
            dside = rng.choice(["left", "right"])
            dc0 = bar_c0 if dside == "left" else bar_c1 - depth + 1
            dc1 = bar_c0 + depth - 1 if dside == "left" else bar_c1
            dr0 = rng.randint(bar_r0, bar_r1)
            dr1 = min(bar_r1, dr0 + rng.randint(0, 4))
            region = {(r, c) for r in range(dr0, dr1 + 1) for c in range(dc0, dc1 + 1) if (r, c) in bar_cells}
        all_holes = set().union(*notch_holes) if notch_holes else set()
        blocked = all_holes | {(r + dr, c + dc) for r, c in all_holes for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]}
        region -= blocked
        region -= protected_bar
        if len(region) < 1:
            continue
        seed_choices = _boundary_cells(dside, region)
        if not seed_choices:
            continue
        area = min(area, len(region))
        dn = _grow_connected(rng, region, area, {rng.choice(seed_choices)}, retries=10)
        if dn is None:
            continue
        if not _connected(bar_cells - dn):
            continue
        bar_cells -= dn
        notch_holes.append(dn)
        decoys += 1
    if decoys < 4:
        return None

    bar_h, bar_w = bar_r1 - bar_r0 + 1, bar_c1 - bar_c0 + 1
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
    for piece in all_chain_pieces:
        for r, c in piece["coords"]:
            output[r, c] = piece["color"]

    input_grid = np.full((H, W), bg, dtype=int)
    for r, c in bar_cells:
        input_grid[r, c] = bar_color

    placed_occ = set(bar_cells)
    pieces_order = list(all_chain_pieces)
    rng.shuffle(pieces_order)
    for piece in pieces_order:
        if not _scatter_piece(rng, input_grid, placed_occ, piece["coords"], piece["color"], orientation, bar_r0, bar_r1, bar_c0, bar_c1):
            return None

    meta = {
        "chain_len": chain_len,
        "widths": widths,
        "outside_depths": outside_depths,
        "cavity_depths": cavity_depths,
        "side": side,
        "orientation": orientation,
    }
    if not np.array_equal(main(input_grid), output):
        return None
    return input_grid, output, meta


def generate_input() -> np.ndarray:
    for attempt in range(400):
        seed = attempt * 10007 + int(np.random.randint(0, 1_000_000))
        result = _gen_attempt(seed)
        if result is None:
            continue
        inp, out, _meta = result
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("generate_input: failed after 400 attempts")
