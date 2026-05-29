# Variant of cbebaa4b
# Dimension: D1 — 连接器匹配条件 (match_cond)
# Change: 精灵的红色像素与目标（中央黄色方块或已连接精灵）的红色像素对齐，形成连接... → 连接器互补色匹配

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# ---
# variant_spec:
#   visual_form:
#     - 连接器不再是单一红色，而是仅使用 blue(1) 和 red(2) 两种像素。
#     - 每个连接边仍然由一对相邻边界像素组成；同一侧的两个像素颜色相同，避免额外模式维度。
#     - 非连接器主体颜色改为 {3,5,6,7,8,9}，彻底避开 blue(1)，这样解题者能直接把 1/2 看成“端口色”而不是主体色。
#     - 输出中每个接缝位置只显示先在装配体上的那个连接器颜色；后贴上的互补色像素被遮住，但计数上仍然算作成功配对。
#   behavior_trace:
#     old_case:
#       - 旧规则里，装配体有一个 red(2) 未配对端口，候选碎片也必须拿 red(2) 对齐它。
#       - 重合后该坐标仍显示 red(2)，计数从 1 变为 2。
#     new_case:
#       - 新规则里，若装配体该端口是 red(2)，候选碎片必须拿 blue(1) 去重合；若装配体是 blue(1)，则候选必须是 red(2)。
#       - 重合后画布保留装配体已有颜色，例如 red(2)；隐藏的是后来贴上的 blue(1)。
#       - 因此 solver 不能再只按“同色重合”试位，而必须同时追踪：
#         1. 当前可继续匹配的未配对端口坐标；
#         2. 这些坐标当前可见的连接器颜色；
#         3. 新碎片每个端口自己的颜色。
#   complexity_calibration:
#     - 只使用一组固定互补对 1↔2，而不是多组任意配色，避免规则自由度过大导致 2-3 组样例难归纳。
#     - 每题有 4-6 条连接边；生成器强制整张图里两种连接器颜色都出现，否则重试，保证样例内有重复支撑。
#     - 树结构、平移装配、anchor 保持不变，变体的视觉差异集中在“蓝红混合端口”这一处。
#   hidden_assumptions_review:
#     - 原 _extract_pieces 只搜 arr == 2，隐含“所有端口都是红色”；现在必须改成搜 {1,2}。
#     - 原 _try_place 允许 red-on-red 重叠，且会直接写回 arr；现在必须只允许 1↔2 重叠，并保留旧画布颜色，不能让后贴碎片覆盖已存在端口色。
#     - 原 dfs 用 rc[count==1] 就足够，因为所有未配对端口颜色相同；现在 rc 之外还必须维护 visible_color[coord]。
#     - 原候选位移由目标 red 与碎片 red 对齐得到；现在只能由目标色的互补端口触发该位移。
#   placement_legality:
#     - 生成器仍先长树、再渲染 assembled output、最后 scatter。
#     - 每条边在父节点侧随机选 1 或 2，子节点入边自动取互补色，几何位置仍按同一对局部坐标重合。
#     - assembled 渲染时只允许互补连接器重叠；任何主体重叠、同色连接器重叠、越界都视为非法。
#   visual_observability:
#     - 主体不再使用颜色 1，所以 blue 只会出现在连接器上，规则线索干净。
#     - 输出里会同时出现红接缝和蓝接缝，而输入中每个碎片外露端口也会混合两色；观察几组样例即可看出“不是同色拼接”。
#   neatness:
#     - 仍保持单连通、低重叠层次、anchor 不移动；变体没有新增标注或局部噪声。
#   color_inference:
#     - 颜色变化不是任意映射，而是唯一固定互补对 1↔2。
#     - 生成器拒绝只出现单色端口的样本，避免规则退化成“看起来和原题一样，只是把 red 改成 blue”。


CONNECTOR_COLORS = (1, 2)
COMP_COLOR = {1: 2, 2: 1}


def _is_connector(arr: np.ndarray) -> np.ndarray:
    return np.isin(arr, CONNECTOR_COLORS)


def _complement_grid(arr: np.ndarray) -> np.ndarray:
    out = np.full(arr.shape, -1, dtype=int)
    out[arr == 1] = 2
    out[arr == 2] = 1
    return out


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        ports = np.argwhere(_is_connector(arr))
        pieces.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "arr": arr,
                "ports": [(int(r), int(c), int(arr[r, c])) for r, c in ports],
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

    arr_conn = _is_connector(arr)
    sub_conn = _is_connector(sub)
    comp_match = arr_conn & sub_conn & (sub == _complement_grid(arr))
    overlap = mask & (sub != bg)
    conflict = overlap & ~comp_match
    if np.any(conflict):
        return None

    out = canvas.copy()
    out_sub = out[top : top + ph, left : left + pw]
    paint_mask = mask & (sub == bg)
    out_sub[paint_mask] = arr[paint_mask]
    out[top : top + ph, left : left + pw] = out_sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around the fixed yellow anchor by matching complementary connector colors.

    Input structure:
    - bg = black (0). Multiple colored sprites are scattered on the grid.
    - Exactly one sprite is the anchor: it contains a yellow (4) rectangular core.
    - Connectors are boundary pixels colored blue (1) or red (2); body colors avoid blue so these ports are visually distinct.

    Rule:
    - The anchor stays where it is; every other sprite is translated without rotation.
    - A sprite may attach only when one of its connector pixels lands on an unpaired connector of the opposite color already present in the growing assembly.
    - Each connector pixel is used exactly once, so every final joint is formed by one blue pixel and one red pixel sharing a coordinate.

    Key insight: The search state is no longer “where are the red pegs”, but “which connector coordinates are still open, and what visible color is currently sitting there”; only a complementary-colored port can close that slot.
    """
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

    total_ports = int(sum(len(p["ports"]) for p in pieces))
    assert total_ports % 2 == 0, "total connector pixels must be even for complete pairing"
    target_unique_ports = total_ports // 2

    port_counts: dict[tuple[int, int], int] = {}
    port_colors: dict[tuple[int, int], int] = {}
    for rr, cc, color in anchor["ports"]:
        key = (ar0 + rr, ac0 + cc)
        port_counts[key] = port_counts.get(key, 0) + 1
        port_colors[key] = color

    solutions: list[np.ndarray] = []

    def dfs(
        cur: np.ndarray,
        remaining: list[dict],
        counts: dict[tuple[int, int], int],
        colors: dict[tuple[int, int], int],
    ) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(counts) == target_unique_ports and all(v == 2 for v in counts.values()):
                solutions.append(cur)
            return

        if len(counts) > target_unique_ports:
            return

        unpaired = [k for k, v in counts.items() if v == 1]
        if not unpaired:
            return
        open_slots = {k for k, v in counts.items() if v == 1}

        best_moves = None
        for target in unpaired:
            moves = []
            tr, tc = target
            needed = COMP_COLOR[colors[target]]
            for idx, p in enumerate(remaining):
                ph, pw = p["shape"]
                for pr, pc, color in p["ports"]:
                    if color != needed:
                        continue
                    top = tr - pr
                    left = tc - pc
                    if top < 0 or left < 0 or top + ph > h or left + pw > w:
                        continue
                    bad = False
                    overlap = 0
                    for rr, cc, pcolor in p["ports"]:
                        gr, gc = top + rr, left + cc
                        prev = counts.get((gr, gc), 0)
                        if prev >= 2:
                            bad = True
                            break
                        if prev == 1:
                            if colors[(gr, gc)] != COMP_COLOR[pcolor]:
                                bad = True
                                break
                            if (gr, gc) in open_slots:
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

        assert best_moves is not None, "there must be at least one feasible complementary match"

        dedup: dict[tuple[int, int, int], int] = {}
        for idx, top, left, ov in best_moves:
            key = (idx, top, left)
            dedup[key] = max(dedup.get(key, 0), ov)
        moves2 = [(idx, top, left, ov) for (idx, top, left), ov in dedup.items()]
        moves2.sort(key=lambda x: (-x[3], -remaining[x[0]]["area"], x[1], x[2], x[0]))

        for idx, top, left, _ov in moves2:
            p = remaining[idx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_counts = dict(counts)
            next_colors = dict(colors)
            for rr, cc, pcolor in p["ports"]:
                gr, gc = top + rr, left + cc
                key = (gr, gc)
                prev = next_counts.get(key, 0)
                if prev == 0:
                    next_counts[key] = 1
                    next_colors[key] = pcolor
                else:
                    next_counts[key] = prev + 1

            nxt = _try_place(cur, p["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_counts, next_colors)
            if len(solutions) >= 2:
                return

    dfs(placed, rest, port_counts, port_colors)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (o5x) ---------------

# ── Tunable configuration ──────────────────────────────────────────────
# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 300

# ── Side geometry lookups ──────────────────────────────────────────────
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "sides": set(),
        "ports": {},
        "port_colors": {},
    }


def _ensure_port(node, side):
    if side in node["ports"]:
        return
    h, w = node["shape"]
    span = h if side in ("left", "right") else w
    node["ports"][side] = random.randint(1, span - 3)


def _set_port_color(node, side, color=None):
    if side in node["port_colors"]:
        return node["port_colors"][side]
    node["port_colors"][side] = random.choice(CONNECTOR_COLORS) if color is None else int(color)
    return node["port_colors"][side]


def _port_cells(node, side):
    h, w = node["shape"]
    k = node["ports"][side]
    if side == "top":
        return [(0, k), (0, k + 1)]
    if side == "bottom":
        return [(h - 1, k), (h - 1, k + 1)]
    if side == "left":
        return [(k, 0), (k + 1, 0)]
    return [(k, w - 1), (k + 1, w - 1)]


def _global_port(node, side):
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, side)]


def _bbox_of(node):
    h, w = node["shape"]
    return (node["top"], node["left"], node["top"] + h - 1, node["left"] + w - 1)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _union_bbox(nodes):
    boxes = [_bbox_of(n) for n in nodes]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _draw_path(arr, start, end, color, order):
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
            if arr[rr, cc] not in CONNECTOR_COLORS:
                arr[rr, cc] = color


def _paint_bar(arr, side, color, used):
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


def _bar_endpoint(shape, side, hub):
    h, w = shape
    hr, hc = hub
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        return (row, min(max(hc, 1), w - 2))
    col = 0 if side == "left" else w - 1
    return (min(max(hr, 1), h - 2), col)


def _make_piece(node):
    h, w = node["shape"]
    color = node["color"]
    used = set(node["sides"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    bar_sides = [s for s in SIDES if s not in used]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for side in used:
        dr, dc = INWARD[side]
        port_color = node["port_colors"][side]
        for r, c in _port_cells(node, side):
            arr[r, c] = port_color
            rr, cc = r + dr, c + dc
            arr[rr, cc] = color
            terminals.append((rr, cc))

    arr[hub] = color
    order = random.randint(0, 1)
    for t in terminals:
        _draw_path(arr, t, hub, color, order)
    for side in bar_sides:
        _draw_path(arr, hub, _bar_endpoint((h, w), side, hub), color, 1 - order)

    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(_is_connector(arr))) != 2 * len(used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for side in node["sides"]:
        port_color = node["port_colors"][side]
        for r, c in _port_cells(node, side):
            arr[r, c] = port_color
    return arr


def _candidate_fits(nodes, parent_i, child):
    cb = _bbox_of(child)
    if not _rects_overlap(cb, _bbox_of(nodes[parent_i])):
        return False
    for i, other in enumerate(nodes):
        if i == parent_i:
            continue
        if _rects_overlap(cb, _bbox_of(other)):
            return False
    r0, c0, r1, c1 = _union_bbox(nodes + [child])
    return (r1 - r0 + 1) <= MAX_ASSEMBLY_DIM and (c1 - c0 + 1) <= MAX_ASSEMBLY_DIM


def _grow_tree():
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, n in enumerate(nodes) if len(n["sides"]) < n["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                sides = [s for s in SIDES if s not in parent["sides"]]
                random.shuffle(sides)
                for side in sides:
                    child = _new_node("piece", ph, pw, colors[i])
                    incoming = OPPOSITE[side]
                    _ensure_port(parent, side)
                    _ensure_port(child, incoming)
                    parent_color = _set_port_color(parent, side)
                    _set_port_color(child, incoming, COMP_COLOR[parent_color])
                    pr, pc = _global_port(parent, side)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    child["top"] = pr - cr
                    child["left"] = pc - cc
                    child["sides"].add(incoming)
                    if not _candidate_fits(nodes, parent_i, child):
                        continue
                    parent["sides"].add(side)
                    nodes.append(child)
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    edge_colors = [node["port_colors"][side] for node in nodes for side in node["sides"]]
    if not ({1, 2} <= set(edge_colors)):
        return None
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    arr = node["arr"]
    top, left = node["top"] + sr, node["left"] + sc
    node["pos"] = (top, left)
    port_counts = {}
    mask = arr != 0
    rows, cols = np.nonzero(mask)
    g_rows, g_cols = rows + top, cols + left
    vals = arr[rows, cols]
    cur_vals = out[g_rows, g_cols]
    vals_conn = np.isin(vals, CONNECTOR_COLORS)
    cur_conn = np.isin(cur_vals, CONNECTOR_COLORS)
    comp_vals = np.array([COMP_COLOR.get(int(v), -1) for v in vals], dtype=int)
    legal_overlap = (cur_vals != 0) & vals_conn & cur_conn & (cur_vals == comp_vals)
    conflicts = (cur_vals != 0) & ~legal_overlap
    if np.any(conflicts):
        return None
    for gr, gc, is_port in zip(g_rows, g_cols, vals_conn):
        if not is_port:
            continue
        key = (int(gr), int(gc))
        port_counts[key] = port_counts.get(key, 0) + 1
    paint_mask = cur_vals == 0
    out[g_rows[paint_mask], g_cols[paint_mask]] = vals[paint_mask]
    return port_counts


def _assemble(nodes, h, w):
    for node in nodes:
        node["arr"] = _make_anchor(node) if node["kind"] == "anchor" else _make_piece(node)
        if node["arr"] is None:
            return None

    r0, c0, r1, c1 = _union_bbox(nodes)
    if (r1 - r0 + 1) > h or (c1 - c0 + 1) > w:
        return None
    sr = random.randint(-r0, h - 1 - r1)
    sc = random.randint(-c0, w - 1 - c1)

    out = np.zeros((h, w), dtype=int)
    all_port_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for key, val in rc.items():
            all_port_counts[key] = all_port_counts.get(key, 0) + val

    total_ports = sum(int(np.sum(_is_connector(n["arr"]))) for n in nodes)
    if len(all_port_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_port_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    if not ({1, 2} <= set(np.unique(out))):
        return None
    return out


def _scatter(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        try:
            r, c = random_free_location(
                inp,
                node["arr"],
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, node["arr"], r, c, transparent=0)
    return inp


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree()
    if nodes is None:
        return None
    out = _assemble(nodes, h, w)
    if out is None:
        return None
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(solved, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b input")
