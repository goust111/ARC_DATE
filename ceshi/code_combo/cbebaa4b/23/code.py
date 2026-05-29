# Combo variant of cbebaa4b (index 23)
# combo_spec: D1/2+D13/1
# Name: 旋转翻转匹配 + 对角线连接
#
# Design analysis
# ----------------
# a) concretization_and_trace:
# - 保留原题的 output-first 组装框架：输入里有一个固定黄锚点，其他彩色 piece 散落；输出是把它们
#   重新拼回一棵连通树。
# - D13/1 把“连接方向”从四联通边口扩展到 8 个方向。这里将对角连接具体化为 3 红像素的 L 形角口，
#   正交连接仍是 2 红像素直线边口。这样 piece 真正可以沿 45 度方向长出去，肉眼可见。
# - D1/2 让散落输入中的非锚点 piece 可以处于任意二面体朝向。solver 不能只平移 raw piece，而要先
#   在 8 个旋转/翻转姿态里找出哪一个姿态能让完整连接器组重合，再决定平移量。
# - 小例子：原题里一个 leaf 若在左边有 2 红像素边口，它只能平移到装配体某个未配对边口上。融合后，
#   该 leaf 可能在输入里先被翻转再旋转，导致这组红像素跑到右上角；与此同时它要连接的目标也可能是
#   一个 3 红像素对角角口。solver 必须先试出正确姿态，再要求整个 3 红像素角口一次性吃掉，而不是
#   只让其中某个红点碰上就算合法。
#
# b) complexity_calibration:
# - 两个自由度叠加后，若仍允许任意树分叉，候选状态会暴涨并更容易出现对称歧义。这里把连接图限制为
#   一条链：锚点只连向第一个 piece，之后 piece 串成 path。难点集中在“姿态 + 方向”的联合搜索上，
#   而不是枝杈排列。
# - 每个非锚点 piece 使用唯一颜色，并额外强制一个不对称内部 tag；只有拥有至少 4 个唯一二面体变体
#   的 piece 才会被接受。这样旋转/翻转对人类是可见的，不会退化成对称块。
# - 生成器强制至少一条 assembled 边是对角连接，并强制散落输入里至少一个 piece 带 flip、至少一个
#   piece 带非零旋转，从而避免融合规则只在代码里存在、样例里却不出现。
#
# c) implicit_assumption_audit_of_original_main:
# - 原 `shape/reds` 默认每个 piece 只有一个固定包围盒和固定红点坐标。融合后，solver 必须对每个
#   piece 预先枚举唯一 dihedral variants，并在 DFS 中使用 variant 自己的 bbox、reds、connectors。
# - 原 `top = tr - pr, left = tc - pc` 只保证“某个红点重合”。对角口加入后，这会放大伪候选，所以
#   新 solver 额外要求：至少有一整组连接器 fully matched，并且当前被选中的未配对红点属于该整组。
# - 原 move dedup 只按 `(piece, top, left)`；现在姿态影响真实状态，所以 dedup key 必须包含 variant。
# - 原 `_try_place` 的合法性假设仍成立：只能 red-on-red 重叠，非红不能冲突。融合变体没有改变这条。
#
# d) placement_legality:
# - 生成器仍先在局部坐标里长出装配树，再整体随机平移进画布，再保持锚点原位、把其余 piece 散开。
# - `_candidate_fits` 只允许 child 的 bbox 与 parent 相交，且不与其他节点 bbox 相交；随后 `_assemble`
#   再做像素级冲突验证和红口计数验证，保证无越界、无非法重叠。
# - 对角口使用固定角几何，正交口使用避开角落的边缘偏移，两类端口不会争用同一组边界格。
#
# e) visual_observability:
# - 人类看到样例时，能直接观察到两个变化：输入中的 piece 朝向和输出不同；输出里至少有一段拼接是
#   斜向延伸的，因为端口是角落 L 形红口。
# - 没有引入颜色映射、数值排序或隐藏标记；颜色只用于区分 piece，本体结构和端口几何提供全部线索。
#
# f) visual_tidiness:
# - 正交口固定 2 红像素，对角口固定 3 红像素，内部主体都通过单色路径回到中心，不叠加额外标注层。
# - 散布时继续使用 Chebyshev padding=2，保证输入组件分离清楚；assembled 输出仍是一整个 4 连通体。
#
# g) behavior_metric_and_non_degeneracy:
# - 量化指标定义为 `diag_edge_count` 与散落朝向指标 `flip_piece_count/rotated_piece_count`。
# - 生成器保证 `diag_edge_count >= 1`、`flip_piece_count >= 1`、`rotated_piece_count >= 1`。若这些指标
#   不成立则直接丢弃样本，因此不会退化回“纯平移 + 纯正交连接”的原题行为。
#
# h) new_solver_assumptions_and_generator_guarantees:
# - solver 假设合法端口只来自 8 类规范边界几何：4 个正交 2 红像素边口，4 个对角 3 红像素角口。
#   generator 只生成这 8 类端口。
# - solver 假设每个 movable piece 至少有一个非恒等朝向，并且完整解唯一。generator 通过最少变体数、
#   强制非恒等散布和 `main()` 回验来保证。
# - solver 假设锚点不旋转且留在输入原位。generator 始终按该方式构造输入。

import os
import random
import sys

import numpy as np

_ROOT = "/home/lixinyu/snap/new"
_SOLVERS = os.path.join(_ROOT, "solvers")
if _SOLVERS not in sys.path:
    sys.path.append(_SOLVERS)

from arc_utils import bbox_coords, components_mask, paste, random_free_location


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


def _diag_cells(shape: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    h, w = shape
    if direction == "ne":
        return [(0, w - 2), (0, w - 1), (1, w - 1)]
    if direction == "nw":
        return [(0, 0), (0, 1), (1, 1)]
    if direction == "se":
        return [(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 1)]
    return [(h - 2, 0), (h - 2, 1), (h - 1, 1)]


def _connector_size(direction: str) -> int:
    return 2 if direction in ORTHO_DIRS else 3


def _extract_connectors(arr: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    """Infer canonical red connector groups from the piece boundary geometry."""
    h, w = arr.shape
    reds = {tuple(map(int, rc)) for rc in np.argwhere(arr == 2)}
    found: list[tuple[tuple[int, int], ...]] = []
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


def _piece_record(arr: np.ndarray, bg: int) -> dict:
    reds = np.argwhere(arr == 2)
    return {
        "arr": arr.copy(),
        "reds": [(int(r), int(c)) for r, c in reds],
        "connectors": _extract_connectors(arr),
        "shape": (int(arr.shape[0]), int(arr.shape[1])),
        "area": int(np.sum(arr != bg)),
    }


def _variant_records(arr: np.ndarray, bg: int) -> list[dict]:
    """Enumerate all unique dihedral variants of a piece."""
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        for is_flip in (False, True):
            cur = np.fliplr(rot) if is_flip else rot
            key = (tuple(cur.shape), cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            rec = _piece_record(cur, bg)
            rec["is_flip"] = bool(is_flip)
            rec["rot_k"] = int(k)
            rec["code"] = "id" if (k == 0 and not is_flip) else (
                f"fr{90 * k}" if is_flip else f"r{90 * k}"
            )
            variants.append(rec)
    return variants


def _extract_pieces(g: np.ndarray, bg: int) -> list[dict]:
    comps = components_mask(g != bg, connectivity=4)
    pieces = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        piece = _piece_record(arr, bg)
        piece["bbox"] = (int(r0), int(c0), int(r1), int(c1))
        pieces.append(piece)
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
    Reassemble scattered colored sprites around a fixed yellow anchor by matching full red connector groups after reorientation.

    Input structure:
    - The background is black (0), with one anchor sprite and several non-anchor sprites scattered apart.
    - The anchor contains the yellow (4) block and stays at its input position.
    - Movable sprites may already be rotated or flipped, and their red (2) connectors can be either straight 2-pixel edge ports or 3-pixel diagonal corner elbows.

    Rule:
    - Keep the anchor fixed, then attach the other sprites one at a time onto the growing assembly.
    - Before translating a movable sprite, you may choose any unique rotation/flip variant of that sprite.
    - A move is legal only if it creates at least one complete red connector-group overlap on currently unpaired red pixels; every red pixel must end in exactly one red-on-red pair.

    Key insight: The puzzle is no longer solved by translation alone. You must jointly infer the correct dihedral pose of each piece and the correct orthogonal or diagonal port it uses, because valid attachments are defined by whole connector groups rather than isolated red pixels.
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
    for p in rest:
        p["variants"] = _variant_records(p["arr"], bg)
        assert p["variants"], "each movable piece must have at least one orientation variant"

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

    def _push_solution(arr: np.ndarray) -> None:
        if not any(np.array_equal(arr, prev) for prev in solutions):
            solutions.append(arr.copy())

    def dfs(cur: np.ndarray, remaining: list[dict], rc: dict[tuple[int, int], int]) -> None:
        if len(solutions) >= 2:
            return

        if not remaining:
            if len(rc) == target_unique_red and all(v == 2 for v in rc.values()):
                _push_solution(cur)
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
                for vidx, v in enumerate(p["variants"]):
                    ph, pw = v["shape"]
                    for pr, pc in v["reds"]:
                        top = tr - pr
                        left = tc - pc
                        if top < 0 or left < 0 or top + ph > h or left + pw > w:
                            continue

                        bad = False
                        overlap = 0
                        for rr, cc in v["reds"]:
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
                        target_matched = False
                        for conn in v["connectors"]:
                            placed_conn = [(top + rr, left + cc) for rr, cc in conn]
                            if all(cell in rc1 for cell in placed_conn):
                                full_matches += 1
                                if (pr, pc) in conn:
                                    target_matched = True
                        if full_matches == 0 or not target_matched:
                            continue

                        moves.append((idx, vidx, top, left, full_matches, overlap))
            if not moves:
                return
            if best_moves is None or len(moves) < len(best_moves):
                best_moves = moves
                if len(best_moves) == 1:
                    break

        assert best_moves is not None, "there must be at least one feasible connector move"

        dedup: dict[tuple[int, int, int, int], tuple[int, int]] = {}
        for idx, vidx, top, left, full_matches, overlap in best_moves:
            key = (idx, vidx, top, left)
            prev = dedup.get(key, (0, 0))
            dedup[key] = max(prev, (full_matches, overlap))
        moves2 = [
            (idx, vidx, top, left, score[0], score[1])
            for (idx, vidx, top, left), score in dedup.items()
        ]
        moves2.sort(
            key=lambda x: (
                -x[4],
                -x[5],
                -remaining[x[0]]["area"],
                x[2],
                x[3],
                x[0],
                x[1],
            )
        )

        for idx, vidx, top, left, _full_matches, _overlap in moves2:
            p = remaining[idx]
            v = p["variants"][vidx]
            next_remaining = remaining[:idx] + remaining[idx + 1 :]

            next_rc = dict(rc)
            for rr, cc in v["reds"]:
                gr, gc = top + rr, left + cc
                next_rc[(gr, gc)] = next_rc.get((gr, gc), 0) + 1

            nxt = _try_place(cur, v["arr"], top, left, bg)
            if nxt is None:
                continue
            dfs(nxt, next_remaining, next_rc)
            if len(solutions) >= 2:
                return

    global LAST_SOLUTION_COUNT
    dfs(placed, rest, red_counts)
    LAST_SOLUTION_COUNT = len(solutions)
    assert len(solutions) == 1, "jigsaw assembly must have exactly one valid solution"
    return solutions[0]


# --------------- generator (combo) ---------------

GRID_H_RANGE = (22, 26)
GRID_W_RANGE = (22, 26)
MAX_ASSEMBLY_DIM = 24

PIECE_COUNT_RANGE = (4, 6)
PIECE_H_RANGE = (4, 7)
PIECE_W_RANGE = (4, 8)
PIECE_DIM_CANDIDATES = 14
PIECE_COLORS = [1, 3, 5, 6, 7, 8, 9]
PIECE_CAP = 2
BAR_SKIP_PROB = 0.35
MIN_UNIQUE_VARIANTS = 4

ANCHOR_BLOCK_CHOICES = [(3, 3), (4, 5)]
ANCHOR_CAP_CHOICES = [1]

SCATTER_PADDING = 2
GEN_RETRY_LIMIT = 500
MIN_DIAGONAL_EDGES = 1
DIAGONAL_DIR_PROB = 0.6

LAST_GEN_INFO: dict = {}
LAST_SOLUTION_COUNT = 0


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
        "incoming_dir": None,
    }


def _ensure_port(node, direction):
    if direction in node["ports"] or direction in DIAG_DIRS:
        return
    h, w = node["shape"]
    span = h if direction in ("left", "right") else w
    node["ports"][direction] = random.randint(1, span - 3)


def _diag_terminal(shape: tuple[int, int], direction: str) -> tuple[int, int]:
    h, w = shape
    if direction == "ne":
        return (0, w - 3)
    if direction == "nw":
        return (1, 2)
    if direction == "se":
        return (h - 2, w - 3)
    return (h - 1, 2)


def _port_cells(node, direction):
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


def _tag_points(h: int, w: int) -> list[tuple[int, int]]:
    raw = [
        (1, 1),
        (1, w - 2),
        (h - 2, 1),
        (h - 2, w - 2),
        (1, 2),
        (2, w - 2),
        (h - 2, w - 3),
        (h - 3, 1),
    ]
    out = []
    seen = set()
    for r, c in raw:
        if 0 <= r < h and 0 <= c < w and (r, c) not in seen:
            seen.add((r, c))
            out.append((r, c))
    return out


def _make_piece(node):
    """Render an asymmetric piece with straight/diagonal connectors."""
    h, w = node["shape"]
    color = node["color"]
    used = set(node["dirs"])
    arr = np.zeros((h, w), dtype=int)

    core_h = 2 if h <= 5 else 3
    core_w = 2 if w <= 6 else 3
    core_r0 = random.randint(1, h - core_h - 1)
    core_c0 = random.randint(1, w - core_w - 1)
    arr[core_r0 : core_r0 + core_h, core_c0 : core_c0 + core_w] = color
    hub = (core_r0 + core_h // 2, core_c0 + core_w // 2)

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

    tag_choices = _tag_points(h, w)
    random.shuffle(tag_choices)
    chosen_tag = None
    for tag in tag_choices:
        if tag == hub:
            continue
        trial = arr.copy()
        trial[tag] = color
        _draw_path(trial, hub, tag, color, random.randint(0, 1))
        if len(components_mask(trial != 0, connectivity=4)) != 1:
            continue
        if int(np.sum(trial == 2)) != sum(_connector_size(d) for d in used):
            continue
        if len(_variant_records(trial, 0)) < MIN_UNIQUE_VARIANTS:
            continue
        arr = trial
        chosen_tag = tag
        break
    if chosen_tag is None:
        return None

    node["tag"] = chosen_tag
    if len(components_mask(arr != 0, connectivity=4)) != 1:
        return None
    if int(np.sum(arr == 2)) != sum(_connector_size(d) for d in used):
        return None
    if len(_variant_records(arr, 0)) < MIN_UNIQUE_VARIANTS:
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
    """Build a path-shaped connection tree in local coordinates."""
    bh, bw = random.choice(ANCHOR_BLOCK_CHOICES)
    anchor = _new_node("anchor", bh + 2, bw + 2, 4, cap=random.choice(ANCHOR_CAP_CHOICES))
    anchor["block"] = (bh, bw)
    nodes = [anchor]
    n_pieces = random.randint(*PIECE_COUNT_RANGE)
    colors = random.sample(PIECE_COLORS, n_pieces)
    tail_i = 0
    diag_edges = 0

    for i in range(n_pieces):
        placed = False
        dims = [
            (random.randint(*PIECE_H_RANGE), random.randint(*PIECE_W_RANGE))
            for _ in range(PIECE_DIM_CANDIDATES)
        ]
        random.shuffle(dims)
        for ph, pw in dims:
            parent = nodes[tail_i]
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
                child["incoming_dir"] = incoming
                if not _candidate_fits(nodes, tail_i, child):
                    continue
                parent["dirs"].add(direction)
                nodes.append(child)
                tail_i = len(nodes) - 1
                if direction in DIAG_DIRS:
                    diag_edges += 1
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if diag_edges < MIN_DIAGONAL_EDGES:
        return None
    return nodes


def _place_node_on_canvas(out, node, sr, sc):
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


def _choose_scatter_variants(nodes):
    piece_infos = []
    for node in nodes[1:]:
        variants = _variant_records(node["arr"], 0)
        non_identity = [v for v in variants if not np.array_equal(v["arr"], node["arr"])]
        flip_choices = [v for v in non_identity if v["is_flip"]]
        rot_choices = [v for v in non_identity if v["rot_k"] != 0]
        if not non_identity or not flip_choices or not rot_choices:
            return False
        piece_infos.append((node, variants, non_identity, flip_choices, rot_choices))

    if not piece_infos:
        return False

    for node, variants, non_identity, _flip_choices, _rot_choices in piece_infos:
        node["all_variants"] = variants
        chosen = random.choice(non_identity)
        node["scatter_arr"] = chosen["arr"]
        node["scatter_variant"] = chosen["code"]
        node["scatter_is_flip"] = bool(chosen["is_flip"])
        node["scatter_rot_k"] = int(chosen["rot_k"])

    if not any(node.get("scatter_is_flip") for node, *_ in piece_infos):
        node, _variants, _non_identity, flip_choices, _rot_choices = random.choice(piece_infos)
        chosen = random.choice(flip_choices)
        node["scatter_arr"] = chosen["arr"]
        node["scatter_variant"] = chosen["code"]
        node["scatter_is_flip"] = bool(chosen["is_flip"])
        node["scatter_rot_k"] = int(chosen["rot_k"])

    if not any(node.get("scatter_rot_k", 0) != 0 for node, *_ in piece_infos):
        candidates = [(node, rot_choices) for node, _v, _n, _f, rot_choices in piece_infos]
        node, rot_choices = random.choice(candidates)
        chosen = random.choice(rot_choices)
        node["scatter_arr"] = chosen["arr"]
        node["scatter_variant"] = chosen["code"]
        node["scatter_is_flip"] = bool(chosen["is_flip"])
        node["scatter_rot_k"] = int(chosen["rot_k"])

    return True


def _scatter(nodes, h, w):
    inp = np.zeros((h, w), dtype=int)
    anchor = nodes[0]
    paste(inp, anchor["arr"], *anchor["pos"], transparent=0)
    rest = nodes[1:]
    random.shuffle(rest)
    for node in rest:
        arr = node["scatter_arr"]
        try:
            r, c = random_free_location(
                inp,
                arr,
                transparent=0,
                padding=SCATTER_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, arr, r, c, transparent=0)
    return inp


def _record_last_gen_info(nodes):
    global LAST_GEN_INFO
    piece_nodes = nodes[1:]
    LAST_GEN_INFO = {
        "piece_count": len(piece_nodes),
        "changed_piece_count": int(
            sum(1 for n in piece_nodes if n.get("scatter_variant") not in (None, "id"))
        ),
        "flip_piece_count": int(sum(1 for n in piece_nodes if n.get("scatter_is_flip"))),
        "rotated_piece_count": int(sum(1 for n in piece_nodes if n.get("scatter_rot_k", 0) != 0)),
        "diag_edge_count": int(sum(1 for n in piece_nodes if n.get("incoming_dir") in DIAG_DIRS)),
        "variant_codes": [n.get("scatter_variant") for n in piece_nodes],
        "unique_variant_counts": [len(n.get("all_variants", [])) for n in piece_nodes],
        "solution_count": int(LAST_SOLUTION_COUNT),
    }


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    nodes = _grow_tree()
    if nodes is None:
        return None
    assembled = _assemble(nodes, h, w)
    if assembled is None:
        return None
    if not _choose_scatter_variants(nodes):
        return None
    inp = _scatter(nodes, h, w)
    if inp is None:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, assembled):
        return None
    _record_last_gen_info(nodes)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cbebaa4b combo input")
