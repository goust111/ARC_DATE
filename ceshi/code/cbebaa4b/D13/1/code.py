# Variant of cbebaa4b
# Dimension: D13 — 连接的空间约束（对齐方向） (meta)
# Change: 连接器像素在四个基本方向（上下左右）对齐即可连接... → 允许对角线方向连接

import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# a) 具体化与行为追踪
#    - 原题里每条连接边都由 2 个红像素组成，并且这 2 个红像素总是放在上下左右某一条边上。
#      因此当 child 接到 parent 上时，child 的包围盒只会向上/下/左/右延伸。
#    - 本变体保留“红连接器重合后完成拼装”的机制，但新增四种角落对角端口 `ne/nw/se/sw`。
#      为了让 piece 在精确重合后仍保持 4 连通且不产生非红冲突，对角端口采用 3 红像素的 L 形角口；
#      正交端口仍保持原来的 2 红像素直线段。
#    - 小例子：若某 piece 的 `ne` 角口占据其右上角 2x2 小窗里的三个边界格，而另一块 piece
#      的 `sw` 角口在自己的左下角也占据同构的三个格，则把这两个角口平移到完全重合后，
#      child 会整体落在 parent 的右上方，形成肉眼可见的 45 度延伸。
# b) 复杂度校准
#    - 若只是“允许对角方向”但仍复用直线侧边端口，solver 实际无需改动，且样例也不一定能看出变化。
#      这里把对角连接具体化为角落 L 形红端口，使视觉差异稳定可见。
#    - 同时不引入旋转、镜像或颜色规则变化；每个端口仍只占 2 个红像素，节点度数上限仍保持很小，
#      避免搜索空间失控。
# c) 原 main() 的隐式假设审查
#    - `_extract_pieces`: 只假设 piece 是 4 连通组件。对角端口仍通过内部彩色路径接回主体，所以成立。
#    - `red_counts` / `target_unique_red`: 只依赖“每枚红像素最终恰好与另一枚红像素配对重合”，
#      与方向无关，因此成立。
#    - 原 DFS 中 `top = tr - pr, left = tc - pc` 只是在“某枚红像素重合”的层面枚举平移，
#      它并不直接假设方向，但没有验证这是否是一整组连接器的匹配。随着对角端口加入，
#      单红像素重合会产生更多伪候选，因此 solver 需要提升到“至少一整组红连接器同时重合”。
#    - `_try_place`: 仍只允许红对红重叠（或同色像素重叠）。本变体没有改变这一点。
# d) 放置合法性
#    - 生成器继续采用 output-first：先在局部坐标中长出连接树，再整体随机平移进画布，
#      最后固定 anchor、把其余 piece 散开生成输入。
#    - 对角端口只放在非 anchor piece 上，并使用固定角落几何；正交端口仍使用避开角落的随机边缘偏移，
#      因此两类端口不会争用同一格。
#    - `_candidate_fits` 仍负责限制 child 只与 parent 的包围盒相交，并且总装配尺寸不超界；
#      真正的像素级冲突则交给 `_assemble` 再验证。
# e) 视觉可观察性
#    - 每个样例强制至少出现 1 条对角连接边，所以输出不会退化回纯上下左右拼接。
#    - 颜色不参与排序、映射或逻辑分支；颜色仅用于区分不同 piece。
# f) 可视化整洁度
#    - 正交端口仍是 2 红像素，对角端口是 3 红像素角口；piece 内部仍通过单色路径接回中心，
#      避免多层标记叠加。
#    - 对角端口只落在四个角附近，视觉上可以直接看出某块是“从角落斜着接出去”的。
# g) 颜色变化可推断性
#    - 本变体没有新增颜色变换。颜色集合和原题保持一致。
# h) 变体量化指标
#    - 量化指标定义为 assembled 结构中的对角连接边数（生成树里 `ne/nw/se/sw` 的边数）。
#      生成器保证该值 >= 1，并在测试后统计其分布以验证没有退化。
# i) main() 的新隐式假设与 generator 保证
#    - solver 现在显式假设“合法端口要么是 2 红像素直线边口，要么是 3 红像素角口”。
#      generator 只生成这 8 类端口，因此满足。
#    - solver 现在要求新放置的 piece 至少完整吃掉一组未配对端口；generator 的每次树连接本来就是
#      通过完整端口重合实现的，因此满足。


ORTHO_DIRS = ("top", "right", "bottom", "left")
DIAG_DIRS = ("ne", "se", "sw", "nw")
DIRS = ORTHO_DIRS + DIAG_DIRS
SIDES = ORTHO_DIRS
OPPOSITE = {
    "top": "bottom",
    "right": "left",
    "bottom": "top",
    "left": "right",
    "ne": "sw",
    "se": "nw",
    "sw": "ne",
    "nw": "se",
}
INWARD = {"top": (1, 0), "bottom": (-1, 0), "left": (0, 1), "right": (0, -1)}


def _diag_cells(shape, direction):
    h, w = shape
    if direction == "ne":
        return [(0, w - 2), (0, w - 1), (1, w - 1)]
    if direction == "nw":
        return [(0, 0), (0, 1), (1, 1)]
    if direction == "se":
        return [(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 1)]
    return [(h - 2, 0), (h - 2, 1), (h - 1, 1)]


def _connector_size(direction):
    return 2 if direction in ORTHO_DIRS else 3


def _extract_connectors(arr: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Infer straight edge connectors and diagonal corner connectors from a piece bbox."""
    h, w = arr.shape
    reds = {tuple(map(int, rc)) for rc in np.argwhere(arr == 2)}
    found = []
    seen: set[frozenset[tuple[int, int]]] = set()
    claimed: set[tuple[int, int]] = set()

    def add(cells: list[tuple[int, int]], *, reserve: bool) -> None:
        key = frozenset(cells)
        if len(key) != len(cells) or key in seen:
            return
        if not reserve and any(cell in claimed for cell in cells):
            return
        if all(cell in reds for cell in cells):
            seen.add(key)
            found.append(tuple(sorted(key)))
            if reserve:
                claimed.update(cells)

    for direction in DIAG_DIRS:
        add(_diag_cells((h, w), direction), reserve=True)

    for c in range(max(0, w - 1)):
        add([(0, c), (0, c + 1)], reserve=False)
        add([(h - 1, c), (h - 1, c + 1)], reserve=False)
    for r in range(max(0, h - 1)):
        add([(r, 0), (r + 1, 0)], reserve=False)
        add([(r, w - 1), (r + 1, w - 1)], reserve=False)
    return found


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
                "connectors": _extract_connectors(arr),
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered sprites around the fixed yellow anchor by matching complete red connector groups.

    Input structure:
    - The background is black (0) and several colored sprites are scattered apart.
    - Exactly one sprite contains the yellow (4) anchor block and it remains at its input location.
    - Red (2) connectors can appear as 2-pixel straight edge ports or as 3-pixel diagonal corner elbows.

    Rule:
    - Translate the non-anchor sprites one at a time onto the growing assembly; rotations are never used.
    - A move is legal only when at least one whole red connector group from the moved sprite lands on currently unpaired red pixels of the assembly.
    - Straight side connectors and 45-degree corner connectors are both allowed, but every red pixel must still end up paired exactly once.

    Key insight: The puzzle is governed by complete red connector groups rather than by color shape matching; once entire ports are matched, the unique assembled structure is determined even when some attachments extend diagonally.
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

    total_ports = int(sum(len(p["reds"]) for p in pieces))
    assert total_ports % 2 == 0, "total red connector ports must be even for complete pairing"
    target_unique_red = total_ports // 2

    red_counts: dict[tuple[int, int], int] = {}
    for rr, cc in anchor["reds"]:
        red_counts[(ar0 + rr, ac0 + cc)] = red_counts.get((ar0 + rr, ac0 + cc), 0) + 1

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
        for t in unpaired:
            moves = []
            tr, tc = t
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

                    full_matches = 0
                    for conn in p["connectors"]:
                        if all((top + rr, left + cc) in rc1 for rr, cc in conn):
                            full_matches += 1
                    if full_matches == 0:
                        continue

                    moves.append((idx, top, left, full_matches, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        dedup: dict[tuple[int, int, int], tuple[int, int]] = {}
        for idx, top, left, full_matches, overlap in best_moves:
            k = (idx, top, left)
            prev = dedup.get(k, (0, 0))
            dedup[k] = max(prev, (full_matches, overlap))
        moves2 = [
            (idx, top, left, score[0], score[1])
            for (idx, top, left), score in dedup.items()
        ]
        moves2.sort(
            key=lambda x: (-x[3], -x[4], -remaining[x[0]]["area"], x[1], x[2], x[0])
        )

        for idx, top, left, _full_matches, _overlap in moves2:
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
    return solutions[0]


# --------------- generator (o5x) ---------------

# Canvas
GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

# Pieces
PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 12
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35

# Anchor
ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1, 2]

# Scatter
SCATTER_PADDING = 2

# Generation
GEN_RETRY_LIMIT = 300
MIN_DIAGONAL_EDGES = 1
DIAGONAL_DIR_PROB = 0.55


def _new_node(kind, h, w, color, top=0, left=0, cap=PIECE_CAP):
    return {
        "kind": kind,
        "shape": (int(h), int(w)),
        "color": int(color),
        "top": int(top),
        "left": int(left),
        "cap": int(cap),
        "dirs": set(),
        "ports": {},
    }


def _ensure_port(node, direction):
    """Lazily assign a random straight-side port offset; diagonal ports are fixed."""
    if direction in node["ports"] or direction in DIAG_DIRS:
        return
    h, w = node["shape"]
    span = h if direction in ("left", "right") else w
    node["ports"][direction] = random.randint(1, span - 3)


def _diag_terminal(shape, direction):
    h, w = shape
    if direction == "ne":
        return (0, w - 3)
    if direction == "nw":
        return (1, 2)
    if direction == "se":
        return (h - 2, w - 3)
    return (h - 1, 2)


def _port_cells(node, direction):
    """Return the local connector cells for a given direction."""
    h, w = node["shape"]
    if direction == "top":
        k = node["ports"][direction]
        return [(0, k), (0, k + 1)]
    if direction == "bottom":
        k = node["ports"][direction]
        return [(h - 1, k), (h - 1, k + 1)]
    if direction == "left":
        k = node["ports"][direction]
        return [(k, 0), (k + 1, 0)]
    if direction == "right":
        k = node["ports"][direction]
        return [(k, w - 1), (k + 1, w - 1)]
    return _diag_cells((h, w), direction)


def _global_port(node, direction):
    tr, tc = node["top"], node["left"]
    return [(tr + r, tc + c) for r, c in _port_cells(node, direction)]


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
    """Draw an L-shaped path from *start* to *end*, preserving red (2) pixels."""
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


def _paint_bar(arr, side, color, used):
    """Paint a decorative bar along a straight side, skipping corners claimed by used sides."""
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


def _terminals_for_direction(shape, direction, cells):
    if direction in ORTHO_DIRS:
        dr, dc = INWARD[direction]
        return [(r + dr, c + dc) for r, c in cells]
    return [_diag_terminal(shape, direction)]


def _make_piece(node):
    """Render a piece sprite array with straight or diagonal red connectors and decorative bars."""
    h, w = node["shape"]
    color = node["color"]
    used = set(node["dirs"])
    arr = np.zeros((h, w), dtype=int)
    hub = (random.randint(1, h - 2), random.randint(1, w - 2))

    blocked = set()
    if "ne" in used:
        blocked.update(("top", "right"))
    if "nw" in used:
        blocked.update(("top", "left"))
    if "se" in used:
        blocked.update(("bottom", "right"))
    if "sw" in used:
        blocked.update(("bottom", "left"))
    bar_sides = [s for s in SIDES if s not in used and s not in blocked]
    if len(bar_sides) > 1 and random.random() < BAR_SKIP_PROB:
        bar_sides.remove(random.choice(bar_sides))
    for side in bar_sides:
        _paint_bar(arr, side, color, used)

    terminals = []
    for direction in used:
        cells = _port_cells(node, direction)
        for r, c in cells:
            arr[r, c] = 2
        for rr, cc in _terminals_for_direction((h, w), direction, cells):
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
    if int(np.sum(arr == 2)) != sum(_connector_size(d) for d in used):
        return None
    return arr


def _make_anchor(node):
    bh, bw = node["block"]
    h, w = node["shape"]
    arr = np.zeros((h, w), dtype=int)
    arr[1 : 1 + bh, 1 : 1 + bw] = 4
    for direction in node["dirs"]:
        for r, c in _port_cells(node, direction):
            arr[r, c] = 2
    return arr


def _candidate_fits(nodes, parent_i, child):
    """Check child overlaps only its parent and total assembly stays within bounds."""
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


def _ordered_dirs(available, diag_needed):
    diag = [d for d in available if d in DIAG_DIRS]
    ortho = [d for d in available if d in ORTHO_DIRS]
    random.shuffle(diag)
    random.shuffle(ortho)
    if diag_needed or random.random() < DIAGONAL_DIR_PROB:
        return diag + ortho
    return ortho + diag


def _grow_tree():
    """Build a tree of connected nodes (1 anchor + N pieces) in local coordinates."""
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)
    diag_edges = 0

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parents = [j for j, n in enumerate(nodes) if len(n["dirs"]) < n["cap"]]
            random.shuffle(parents)
            for parent_i in parents:
                parent = nodes[parent_i]
                allowed_dirs = ORTHO_DIRS if parent["kind"] == "anchor" else DIRS
                available = [d for d in allowed_dirs if d not in parent["dirs"]]
                for direction in _ordered_dirs(available, diag_edges < MIN_DIAGONAL_EDGES):
                    child = _new_node("piece", ph, pw, colors[i])
                    incoming = OPPOSITE[direction]
                    _ensure_port(parent, direction)
                    _ensure_port(child, incoming)
                    pr, pc = _global_port(parent, direction)[0]
                    cr, cc = _port_cells(child, incoming)[0]
                    child["top"] = pr - cr
                    child["left"] = pc - cc
                    child["dirs"].add(incoming)
                    if not _candidate_fits(nodes, parent_i, child):
                        continue
                    parent["dirs"].add(direction)
                    nodes.append(child)
                    if direction in DIAG_DIRS:
                        diag_edges += 1
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    if diag_edges < MIN_DIAGONAL_EDGES:
        return None
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
    """Stamp a single node's sprite onto *out*, tracking red overlaps."""
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


def _assemble(nodes, h, w):
    """Render all nodes, shift into grid, and validate the assembled output."""
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
    all_red_counts = {}
    for node in nodes:
        rc = _place_node_on_canvas(out, node, sr, sc)
        if rc is None:
            return None
        for k, v in rc.items():
            all_red_counts[k] = all_red_counts.get(k, 0) + v

    total_ports = sum(int(np.sum(n["arr"] == 2)) for n in nodes)
    if len(all_red_counts) * 2 != total_ports:
        return None
    if any(v != 2 for v in all_red_counts.values()):
        return None
    if len(components_mask(out != 0, connectivity=4)) != 1:
        return None
    return out


def _scatter(nodes, h, w):
    """Place anchor at its assembled position, then scatter remaining pieces."""
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
