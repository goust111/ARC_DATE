# Combo variant of cbebaa4b (index 19)
# combo_spec: D4/1+D5/0
# Name: 擦除锚点 + 保留源精灵

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste


# Design analysis for combo D4/1 + D5/0
# -------------------------------------
# a) 具体化与行为追踪
#    - 画面仍由一个固定黄色锚点精灵和若干带红色接口的彩色精灵组成。输入保持原题散布形态：
#      锚点留在中央附近，其他精灵分散在外围。
#    - 原题输出：把所有非锚点精灵平移到锚点附近，红接口两两重合，最后只剩完整组装体。
#    - D4/1 改动：组装完成后，把黄色锚点块擦成背景，留下一个清晰黑洞。
#    - D5/0 改动：非锚点精灵不是“搬走后消失”，而是“原地保留，同时再复制一份去参与组装”。
#    - 小例子：
#        1. 顶部蓝色条块原本散落在左上角，锚点位于中央。
#        2. 原题会把蓝色条块移动到中央结构里，左上角变黑。
#        3. 本融合变体会在中央生成同一蓝色条块的组装副本，但左上角原块保持不变。
#        4. 组装完毕后，中央黄色 3x3 块被擦掉，所以输出是“外围保留散布副本 + 中央无黄洞环”。
#
# b) 复杂度校准
#    - 只做 keep_source 而不控制布局，源副本可能压到组装区，肉眼会看成局部加粗，不易推断复制语义。
#    - 只做 erase_anchor 而继续用树状装配，去锚后可能断裂成多段，视觉上也不稳定。
#    - 因此生成器改为固定 5 片环状装配：擦锚后仍保持 4 连通，并把完整装配区整体保留为散射禁区，
#      迫使所有源副本远离中央结构。这样两项变化都会在每个样例里稳定可见。
#
# c) 隐式假设审查（按 main 的关键逻辑）
#    - `_extract_pieces` 假设输入中的每个精灵都是独立 4 连通分量；生成器用 >=2 的 Chebyshev 间隔保证。
#    - 锚点识别依赖“恰有一个分量含黄色 4”；融合后仍成立，因为源副本不复制锚点。
#    - `_try_place` 假设除了红-red 重叠外不允许前景冲突；生成器仍只用红接口做重合。
#    - DFS 假设所有红接口最终都恰好配对两次；装配器继续显式验证 red count。
#    - 新假设 1：最终输出需要同时容纳源副本和组装体；生成器通过把整块组装足迹加入禁区来保证二者不重叠。
#    - 新假设 2：擦锚后中央空洞必须仍然可见；生成器显式检查黄色块四侧都还有非背景邻接。
#
# d) 放置合法性
#    - 先在局部坐标系里构造环状装配，再把整套结构平移到网格中央附近，确保不越界。
#    - 散布时以“完整装配足迹 + 已放输入精灵”为占用图调用 `random_free_location`，因此每个源精灵都：
#      1. 不越界；2. 不与输入其他精灵重叠；3. 不落进未来输出的中央组装区。
#
# e) 视觉可观察性
#    - 求解者在 2-3 个样例里能直接看到：输出同时包含“输入里原封不动留下的散布精灵”与“中央去黄后的组装环”。
#    - 规则不依赖颜色编号排序、隐藏状态或不可见计数，只依赖可见的接口重合与位置保留。
#
# f) 可视化整洁度
#    - 中央组装体被收缩为较紧凑的 13x13 左右区域，并尽量居中。
#    - 源副本被强制散在外围，避免多层叠压；颜色分工也保持原题风格：黄色只属于辅助锚点，红色只属于接口。
#
# g) 变化的可推断性
#    - 本融合不引入新颜色映射，只改变“结果中保留哪些对象”。
#    - 因为颜色语义未变，求解者更容易把注意力集中在两个稳定现象上：
#      “黄色中心最终消失”与“散布精灵没有消失而是被复制到中央”。
#
# New main assumptions and generator guarantees
#    - main() 仍然只根据输入做唯一装配搜索；它不依赖源副本和中央结构在输出中必须分离。
#    - generator 额外保证更强条件：源副本与中央去锚结构完全不重叠，且去锚后中央结构仍单连通、孔洞清晰。


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        reds = np.argwhere(arr == 2)
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "reds": [(int(r), int(c)) for r, c in reds],
                "shape": (int(arr.shape[0]), int(arr.shape[1])),
                "area": int(np.sum(arr != bg)),
            }
        )
    return pieces


def _try_place(
    canvas: np.ndarray, arr: np.ndarray, top: int, left: int, bg: int
) -> np.ndarray | None:
    h, w = canvas.shape
    ph, pw = arr.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return None

    sub = canvas[top : top + ph, left : left + pw]
    mask = arr != bg
    if not np.any(mask):
        return None

    conflict = mask & (sub != bg) & ~(((arr == 2) & (sub == 2)) | (arr == sub))
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    out_sub[mask] = arr[mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def _erase_anchor_block(arr: np.ndarray, bg: int) -> np.ndarray:
    out = arr.copy()
    out[out == 4] = bg
    return out


def _solve_assembly(grid: np.ndarray) -> tuple[np.ndarray, dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = 0

    pieces = _extract_pieces(g, bg)
    assert pieces, "input must have at least one colored piece"

    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1, "input must have exactly one yellow (4) anchor piece"
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]

    canvas = np.full((h, w), bg, dtype=int)
    ar0, ac0, _, _ = anchor["bbox"]
    placed = _try_place(canvas, anchor["arr"], ar0, ac0, bg)
    assert placed is not None, "anchor piece must fit on canvas at its original position"

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        key = (ar0 + rr, ac0 + cc)
        red_counts[key] = red_counts.get(key, 0) + 1

    solutions: list[np.ndarray] = []

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                solutions.append(cur)
            return

        if len(rc) > target_unique_red:
            return

        unpaired = [k for k, v in rc.items() if v == 1]
        if not unpaired:
            return

        rc1 = {k for k, v in rc.items() if v == 1}

        best_moves = None
        for tr, tc in unpaired:
            moves = []
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for pr, pc in p["reds"]:
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue

                    bad = False
                    overlap = 0
                    for rr, cc in p["reds"]:
                        gr, gc = top + rr, left + cc
                        prev = rc.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if (gr, gc) in rc1:
                            overlap += 1
                    if bad:
                        continue
                    moves.append((idx, top, left, overlap))

            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in p["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, red_counts)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return solutions[0], anchor, rest


def _compose_output(assembled_erased: np.ndarray, source_pieces: list[dict], bg: int) -> np.ndarray:
    out = assembled_erased.copy()
    for p in source_pieces:
        r0, c0, _r1, _c1 = p["bbox"]
        paste(out, p["arr"], r0, c0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the scattered connector sprites into the solved assembly, then erase only the yellow helper from that assembled copy.

    Input structure:
    - bg = black (0). One anchor piece containing a yellow block stays fixed, while the other colored connector sprites are scattered away from it.
    - Exactly one piece contains yellow (4); every piece exposes red (2) connector pixels on its boundary.
    - The non-anchor sprites are separate connected components in the input.

    Rule:
    - Reconstruct the unique translation-only assembly by overlapping red connector pixels pairwise while keeping the anchor at its input location.
    - After the full structure is assembled, repaint the yellow anchor block to background but keep the red connector overlaps.
    - Also keep every non-anchor sprite at its original input location, so the final output is the union of the source sprites and the de-anchored assembled copy.

    Key insight: The red pixels still define the same jigsaw assembly as the source task, but the result is now "copy and assemble": the scattered sprites do not disappear, and the yellow center is only a temporary jig that is removed at the end.
    """
    full, _anchor, rest = _solve_assembly(grid)
    erased = _erase_anchor_block(full, 0)
    return _compose_output(erased, rest, 0)


# --------------- generator (combo D4/1 + D5/0) ---------------

GRID_H_RANGE = (24, 26)
GRID_W_RANGE = (24, 26)
MAX_ASSEMBLY_DIM = 15

PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
SCATTER_PADDING = 2
ASSEMBLY_CLEARANCE = 1
GEN_RETRY_LIMIT = 1600

ANCHOR_BLOCK_CHOICES = [(3, 3)]
BAR_SKIP_PROB = 0.35

SIDES = ("top", "right", "bottom", "left")
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind: str, h: int, w: int, color: int, top: int = 0, left: int = 0) -> dict:
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "sides": set(),
        "ports": {},
        "trace_sides": set(),
    }


def _port_cells(node: dict, side: str) -> list[tuple[int, int]]:
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _bbox_of(node: dict) -> tuple[int, int, int, int]:
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _union_bbox(nodes: list[dict]) -> tuple[int, int, int, int]:
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(
    arr: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int, order: int
) -> None:
    r, c = start
    er, ec = end
    axes = (
        [(r, er, c, True), (c, ec, er, False)]
        if order == 0
        else [(c, ec, r, False), (r, er, ec, True)]
    )
    for a_start, a_end, fixed, is_row in axes:
        step = 1 if a_end >= a_start else -1
        for v in range(a_start, a_end + step, step):
            rr, cc = (v, fixed) if is_row else (fixed, v)
            if arr[rr, cc] != 2:
                arr[rr, cc] = color


def _paint_bar(arr: np.ndarray, side: str, color: int, used: set[str]) -> None:
    h, w = arr.shape
    is_horizontal = side in ("top", "bottom")
    fixed = 0 if side in ("top", "left") else (h - 1 if side == "bottom" else w - 1)
    span = w if is_horizontal else h
    lo_conflict = "left" if is_horizontal else "top"
    hi_conflict = "right" if is_horizontal else "bottom"
    for i in range(span):
        if (i == 0 and lo_conflict in used) or (i == span - 1 and hi_conflict in used):
            continue
        if is_horizontal:
            arr[fixed, i] = color
        else:
            arr[i, fixed] = color


def _bar_endpoint(shape: tuple[int, int], side: str, hub: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node: dict) -> np.ndarray | None:
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = set(node.get("trace_sides", set()))
    unused_sides = [s for s in SIDES if s not in used]
    if len(unused_sides) > 1 and random.random() < BAR_SKIP_PROB:
        unused_sides.remove(random.choice(unused_sides))
    bar_sides.update(unused_sides)
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in [s for s in bar_sides if s not in used]:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != 2 * len(used):
        return None
    return arr


def _make_anchor(node: dict) -> np.ndarray:
    bh, bw = node["block"]
    arr = np.zeros(node["shape"], dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        for r, c in _port_cells(node, side):
            arr[r, c] = 2
    return arr


def _place_node_on_canvas(out: np.ndarray, node: dict, sr: int, sc: int) -> dict | None:
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    red_counts = {}
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]
    conflicts = (cur_vals != 0) & ~((cur_vals == 2) & (vals == 2))
    if np.any(conflicts):
        return None
    red_mask = vals == 2
    for gr, gc in zip(g_rows[red_mask], g_cols[red_mask]):
        key = (int(gr), int(gc))
        red_counts[key] = red_counts.get(key, 0) + 1
    paint_mask = cur_vals == 0
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return red_counts


def _has_visible_hole(erased: np.ndarray, anchor: dict) -> bool:
    top, left = anchor["pos"]
    bh, bw = anchor["block"]
    r0, c0 = top + 1, left + 1
    r1, c1 = r0 + bh - 1, c0 + bw - 1
    if not np.all(erased[r0 : r1 + 1, c0 : c1 + 1] == 0):
        return False
    above = erased[r0 - 1, c0 : c1 + 1]
    below = erased[r1 + 1, c0 : c1 + 1]
    left_col = erased[r0 : r1 + 1, c0 - 1]
    right_col = erased[r0 : r1 + 1, c1 + 1]
    return (
        np.any(above != 0)
        and np.any(below != 0)
        and np.any(left_col != 0)
        and np.any(right_col != 0)
    )


def _build_ring_nodes() -> list[dict] | None:
    bh, bw = ANCHOR_BLOCK_CHOICES[0]
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, top=0, left=0)
    anchor["block"] = (bh, bw)
    anchor["sides"] = {"top"}
    anchor["ports"]["top"] = random.randint(1, 2)

    colors = random.sample(PIECE_COLORS, 5)
    tm = _new_node("piece", 4, 5, colors[0], top=-3, left=0)
    tl = _new_node("piece", 8, 5, colors[1], top=-3, left=-4)
    bl = _new_node("piece", random.randint(5, 6), 7, colors[2], top=4, left=-3)
    br = _new_node("piece", random.randint(5, 6), 6, colors[3], top=4, left=3)
    tr = _new_node("piece", 8, 5, colors[4], top=-3, left=4)

    tm["sides"] = {"bottom", "left", "right"}
    tl["sides"] = {"right", "bottom"}
    bl["sides"] = {"top", "right"}
    br["sides"] = {"left", "top"}
    tr["sides"] = {"left", "bottom"}

    tm["trace_sides"] = {"bottom"}
    tl["trace_sides"] = {"right"}
    bl["trace_sides"] = {"top"}
    br["trace_sides"] = {"top"}
    tr["trace_sides"] = {"left"}

    tm["ports"] = {"bottom": anchor["ports"]["top"], "left": 1, "right": 1}
    tl["ports"] = {"right": 1, "bottom": 2}
    bl["ports"] = {"top": 1, "right": 2}
    br["ports"] = {"left": 2, "top": 2}
    tr["ports"] = {"left": 1, "bottom": 1}

    nodes = [anchor, tm, tl, bl, br, tr]
    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > MAX_ASSEMBLY_DIM or (c1 - c0 + 1) > MAX_ASSEMBLY_DIM:
        return None
    return nodes


def _centered_shift(nodes: list[dict], h: int, w: int) -> tuple[int, int]:
    r0, c0, r1, c1 = _union_bbox(nodes)
    ah = r1 - r0 + 1
    aw = c1 - c0 + 1
    sr_min, sr_max = -r0, h - 1 - r1
    sc_min, sc_max = -c0, w - 1 - c1
    base_sr = (h - ah) // 2 - r0
    base_sc = (w - aw) // 2 - c0
    sr_choices = [base_sr + d for d in (-1, 0, 1) if sr_min <= base_sr + d <= sr_max]
    sc_choices = [base_sc + d for d in (-1, 0, 1) if sc_min <= base_sc + d <= sc_max]
    if not sr_choices:
        sr_choices = [base_sr]
    if not sc_choices:
        sc_choices = [base_sc]
    return random.choice(sr_choices), random.choice(sc_choices)


def _assemble(nodes: list[dict], h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None

    sr, sc = _centered_shift(nodes, h, w)
    full = np.zeros((h, w), dtype=int)
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(full, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            all_red_counts[k] = all_red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(full != 0, connectivity=4)) != 1:
        return None

    erased = _erase_anchor_block(full, 0)
    if len(components_mask(erased != 0, connectivity=4)) != 1:
        return None
    if not _has_visible_hole(erased, nodes[0]):
        return None

    return full, erased


def _scatter(nodes: list[dict], full: np.ndarray) -> np.ndarray | None:
    h, w = full.shape
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    occupied = inp != 0
    reserved = full != 0

    rest = list(nodes[1:])
    random.shuffle(rest)
    for node in rest:
        loc = _find_scatter_location(occupied, reserved, node["arr"])
        if loc is None:
            return None
        r, c = loc
        node["scatter_pos"] = (int(r), int(c))
        paste(inp, node["arr"], r, c, transparent=0)
        patch_mask = node["arr"] != 0
        ph, pw = node["arr"].shape
        occupied[r : r + ph, c : c + pw] |= patch_mask
    return inp


def _dilate_chebyshev(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - radius)
        r1 = min(h, int(r) + radius + 1)
        c0 = max(0, int(c) - radius)
        c1 = min(w, int(c) + radius + 1)
        out[r0:r1, c0:c1] = True
    return out


def _find_scatter_location(
    occupied: np.ndarray, reserved: np.ndarray, patch: np.ndarray
) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = patch.shape
    patch_mask = patch != 0
    sprite_block = _dilate_chebyshev(occupied, SCATTER_PADDING)
    assembly_block = _dilate_chebyshev(reserved, ASSEMBLY_CLEARANCE)
    candidates = []
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            sprite_hit = sprite_block[r : r + ph, c : c + pw]
            assembly_hit = assembly_block[r : r + ph, c : c + pw]
            if np.any((sprite_hit | assembly_hit) & patch_mask):
                continue
            candidates.append((int(r), int(c)))
    if not candidates:
        return None
    return random.choice(candidates)


def _expected_output_from_nodes(erased: np.ndarray, nodes: list[dict]) -> np.ndarray:
    out = erased.copy()
    for node in nodes[1:]:
        paste(out, node["arr"], *node["scatter_pos"], transparent=0)
    return out


def _source_mask(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    pieces = _extract_pieces(grid, 0)
    anchors = [p for p in pieces if np.any(p["arr"] == 4)]
    assert len(anchors) == 1
    anchor = anchors[0]
    rest = [p for p in pieces if p is not anchor]
    mask = np.zeros(grid.shape, dtype=bool)
    for p in rest:
        r0, c0, _r1, _c1 = p["bbox"]
        ph, pw = p["arr"].shape
        mask[r0 : r0 + ph, c0 : c0 + pw] |= p["arr"] != 0
    return mask, rest


def _combo_visible(inp: np.ndarray, out: np.ndarray) -> bool:
    source_mask, _rest = _source_mask(inp)
    source_pixels = int(np.sum(source_mask))
    kept_pixels = int(np.sum(source_mask & (inp == out) & (inp != 0)))
    added_pixels = int(np.sum((out != 0) & ~source_mask))
    overlap_pixels = int(np.sum(source_mask & (out != inp) & (out != 0)))
    return (
        source_pixels >= 24
        and kept_pixels == source_pixels
        and added_pixels >= 24
        and overlap_pixels == 0
    )


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    nodes = _build_ring_nodes()
    if nodes is None:
        return None

    assembled = _assemble(nodes, h, w)
    if assembled is None:
        return None
    full, erased = assembled

    inp = _scatter(nodes, full)
    if inp is None:
        return None

    expected = _expected_output_from_nodes(erased, nodes)
    if not _combo_visible(inp, expected):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
