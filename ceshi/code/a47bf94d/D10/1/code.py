# Variant of a47bf94d
# Dimension: D10 — 通道排列逻辑（置换结构） (meta)
# Change: 管道实现通道的非恒等置换（permutation），每个源端恰好连接一个目的端（1:1映射），置换至少包含一个交叉... → 条件置换（基于token属性）

# Design analysis
# ===============
# spec_variant:
#   core_change:
#     original: 所有 token 都沿同一套管道置换走；只要找到端点和远端，solid/X 的差别只影响发射出的形状。
#     variant: 同一张管道图提供两套路由解释：
#       - solid token: 在竖直主干遇到横向 branch 时，跟着 branch 拐弯，因此沿“移动的线”走。
#       - X token: 尽量保持当前竖直槽位直行，只有直行被堵住时才转向，因此沿“固定槽位”走。
#     effect: 同一批 crossover 行会同时定义两个置换：
#       - solid 使用非恒等置换 perm_s
#       - X 使用恒等置换 perm_x = identity
#   concrete_pixel_behavior:
#     - 仍然保留原题三种 3x3 token：solid、X、diamond。
#     - crossover 行的像素外观仍是原题风格：一条横向 8 线穿过若干竖向列，through-wire 周围带 5 标记。
#     - 当 token 是 solid 时，它在 src 列进入 crossover 行后会拐入横线，直到 dst 列再继续朝对侧前进。
#     - 当 token 是 X 时，它在同一位置会继续沿当前列直走，把横线视为“给别的 token 用的支路”。
#   small_trace_example:
#     setup:
#       - 列顺序 [0,1,2,3]
#       - crossover 把列 0 的 wire 搬到列 2，上方和下方都还有其它竖线通过。
#     original_logic:
#       - solid 和 X 都从列 0 出发，都会被 tracer 带到列 2，因此都去底部第 2 列。
#     variant_logic:
#       - solid 到达该 crossover 时，从竖线拐进横线，最终去底部第 2 列。
#       - X 到达同一行时保持直行，最终去底部第 0 列。
#     difference:
#       - 变体不再是“看到了某根管就只有唯一 far endpoint”，而是“同一个 junction 对两类 token 的出边不同”。
#   complexity_calibration:
#     - 只允许两种路由类别（solid / X），避免做多档条件导致样例不足以归纳。
#     - X 置换固定为 identity，solid 置换保持原题式 crossed permutation，差异大且肉眼可见。
#     - 每个样例强制同时出现 solid 与 X，避免退化为原题或退化为纯直通题。
#   implicit_assumption_audit:
#     - 原 main 的 `_trace_wire` 默认“优先直行，必要时转弯”，这等价于所有 token 共用一套路径选择；变体中不再成立。
#     - 原 generator 默认所有颜色共享同一 bottom 目标槽集合 `perm[i]`；变体混入 identity 路由后若随意选 X 通道，会与 solid 目标重叠。
#     - 原 output 构造固定为“顶部全是 diamond、底部全是 X”；变体里两侧都可能混有 diamond/X，取决于每个 token 的输入侧与类型。
#   legality_constraints:
#     - 为避免 solid 目标槽与 X 目标槽碰撞，只把 X 通道选成 `perm` 的整条 cycle 的并集；于是 solid 通道的 `perm` 像与 X 通道的 identity 槽天然不相交。
#     - permutation 只从少量模板中选，保证至少两条非平凡 cycle，且可视上确实有跨列交叉。
#     - color-9 矩形仍可嵌入，但禁止覆盖 crossover 行；这样条件分流发生在清晰可见的 8/5 junction，而 9 只承担“直通盒子”。
#   observability:
#     - 训练样例里会重复出现：solid 颜色去到被交换后的列，X 颜色留在原列。
#     - 由于两类 token 形状本身可见，解题者可以把“哪一类走 crossed perm、哪一类走 identity”从 2-4 例中归纳出来。
#   cleanliness:
#     - 不引入新的颜色语义；仍只用原题已有的 5/8/9 线网和 1/2/3/4/6/7 token 色。
#     - 不让两类输出 token 落到同一 3x3 中心，避免遮挡造成画面混乱。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

_DIAMOND = _DIAMOND_OFF
_X = _X_OFF
_SOLID = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg or np.any(sub != col):
                continue
            maximal = True
            for rr in range(r0, r0 + 3):
                for cc in range(c0, c0 + 3):
                    for nr, nc in neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                            if int(g[nr, nc]) == col:
                                maximal = False
                                break
                    if not maximal:
                        break
                if not maximal:
                    break
            if maximal:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_x_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for isolated 3x3 X patterns."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _X_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = (col, (r0, c0))
    return found


def _find_diamond_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], int]:
    """Return {center: color} for isolated 3x3 diamonds (4-neigh, empty center)."""
    h, w = g.shape
    found: dict[tuple[int, int], int] = {}

    mask = np.zeros((3, 3), dtype=bool)
    for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        mask[rr, cc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            vals = {int(sub[rr, cc]) for rr, cc in [(0, 1), (1, 0), (1, 2), (2, 1)]}
            if len(vals) != 1:
                continue
            col = vals.pop()
            if col == bg or int(sub[1, 1]) != bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue
            isolated = True
            for dr0, dc0 in _DIAMOND_OFF:
                rr, cc = r0 + 1 + dr0, c0 + 1 + dc0
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(r0 + 1, c0 + 1)] = col
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _component_degrees(comp: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg = {p: 0 for p in comp}
    for r, c in comp:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in comp:
                deg[(r, c)] += 1
    return deg


def _extract_crossover_rows(
    comp: set[tuple[int, int]],
    g: np.ndarray,
    slot_cols: list[int],
) -> list[tuple[int, int, int]]:
    """Return ordered crossover rows as (row, src_col, dst_col)."""
    rows = sorted({r for r, _ in comp})
    slot_set = set(slot_cols)
    crossovers: list[tuple[int, int, int]] = []
    for r in rows:
        cols = sorted(
            c for rr, c in comp
            if rr == r and int(g[rr, c]) in (5, 8)
        )
        if len(cols) < 2:
            continue
        run_start = 0
        while run_start < len(cols):
            run_end = run_start
            while run_end + 1 < len(cols) and cols[run_end + 1] == cols[run_end] + 1:
                run_end += 1
            run = cols[run_start:run_end + 1]
            if len(run) >= 2:
                left, right = run[0], run[-1]
                if left in slot_set or right in slot_set:
                    left_src = (r - 1, left) in comp and (r + 1, left) not in comp
                    right_src = (r - 1, right) in comp and (r + 1, right) not in comp
                    if left_src ^ right_src:
                        src_col = left if left_src else right
                        dst_col = right if left_src else left
                        crossovers.append((r, src_col, dst_col))
            run_start = run_end + 1
    return sorted(crossovers)


def _infer_component_permutation(
    comp: set[tuple[int, int]],
    g: np.ndarray,
) -> tuple[list[int], dict[int, int], dict[int, int], int, int] | None:
    """Infer slot columns plus solid-routing permutation for one generated wire component."""
    deg = _component_degrees(comp)
    endpoints = [p for p in comp if deg[p] <= 1]
    if not endpoints:
        return None

    top_row = min(r for r, _ in endpoints)
    bot_row = max(r for r, _ in endpoints)
    top_cols = sorted(c for r, c in endpoints if r == top_row)
    bot_cols = sorted(c for r, c in endpoints if r == bot_row)
    if not top_cols or top_cols != bot_cols:
        return None

    slot_cols = top_cols
    col_to_slot = {c: i for i, c in enumerate(slot_cols)}
    crossovers = _extract_crossover_rows(comp, g, slot_cols)

    slot_wire: list[int | None] = list(range(len(slot_cols)))
    bypass_wire: int | None = None
    for _, src_col, dst_col in crossovers:
        src_slot = col_to_slot.get(src_col, -1)
        dst_slot = col_to_slot.get(dst_col, -1)
        if src_slot == -1:
            if bypass_wire is None or dst_slot == -1:
                return None
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

    if bypass_wire is not None or any(v is None for v in slot_wire):
        return None

    inv_perm = {dst: int(src) for dst, src in enumerate(slot_wire)}
    perm = {src: dst for dst, src in inv_perm.items()}
    return slot_cols, perm, inv_perm, top_row, bot_row


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and the wire cells touching each face."""
    mask9 = g == 9
    if not mask9.any():
        return []
    h, w = g.shape
    rects = []
    for comp9 in components_mask(mask9, connectivity=4):
        cells = set((int(r), int(c)) for r, c in comp9)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        faces: dict[str, list[tuple[int, int]]] = {}
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) not in cells and 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) in (5, 8):
                    face = "N" if nr < r0 else "S" if nr > r1 else "W" if nc < c0 else "E"
                    faces.setdefault(face, []).append((nr, nc))
        rects.append((cells, faces))
    return rects


_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _choose_exit_from_rect(
    faces9: dict[str, list[tuple[int, int]]],
    cur: tuple[int, int],
    d: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    entry_face = _DIR2FACE[d]
    opp = _OPP[entry_face]
    if opp in faces9:
        exit_face = opp
    else:
        others = [f for f in faces9 if f != entry_face]
        assert len(others) == 1
        exit_face = others[0]
    pts = faces9[exit_face]
    er, ec = cur
    if entry_face in ("N", "S"):
        exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
    else:
        exit_cell = min(pts, key=lambda p: abs(p[0] - er))
    return exit_cell, _FACE2DIR[exit_face]


def _pick_step(
    opts: list[tuple[tuple[int, int], int, int]],
    d: tuple[int, int],
    flow_dir: tuple[int, int],
    token_kind: str,
) -> tuple[tuple[int, int], int, int]:
    rev = (-d[0], -d[1])
    opts2 = [o for o in opts if (o[1], o[2]) != rev] or opts
    straight = [o for o in opts2 if (o[1], o[2]) == d]
    perp = [o for o in opts2 if o[1] * d[0] + o[2] * d[1] == 0]
    flow = [o for o in opts2 if (o[1], o[2]) == flow_dir]

    if token_kind == "solid":
        if d[0] != 0:
            if perp:
                return sorted(perp, key=lambda x: (x[1], x[2]))[0]
            if straight:
                return straight[0]
        else:
            if straight:
                return straight[0]
            if flow:
                return flow[0]
            if perp:
                return sorted(perp, key=lambda x: (x[1], x[2]))[0]
    else:
        if straight:
            return straight[0]
        if flow:
            return flow[0]
        if perp:
            return sorted(perp, key=lambda x: (x[1], x[2]))[0]

    return opts2[0]


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
    token_kind: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Conditional wire following.

    - solid tokens turn onto horizontal swap rows when coming from a vertical segment.
    - X tokens keep their current straight direction whenever possible.
    - color-9 rectangles remain passive pass-through boxes.
    """
    cur, d, prev = start, dir0, None
    flow_dir = dir0
    visited: set[tuple[tuple[int, int], tuple[int, int], str]] = set()
    for _ in range(h * w * 4):
        state = (cur, d, token_kind)
        if state in visited:
            assert False, "wire trace looped"
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                cur, d = _choose_exit_from_rect(faces9, cur, d)
                prev = None
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

        p, dr, dc = _pick_step(opts, d, flow_dir, token_kind)
        if p == prev:
            return cur, d

        prev = cur
        cur = p
        d = (dr, dc)

    assert False, "wire trace exceeded step limit"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each token by combining its shape transform with a type-dependent routing rule.

    Input structure:
    - Background cells stay empty; non-background cells form 3x3 tokens plus a cyan/grey/maroon pipe network whose junction rows can be read in two ways.

    Rule:
    - A solid 3x3 block always collapses in place to a diamond, then sends an X to the opposite side by following the moving wire through every swap row.
    - An X token stays in place, then sends a diamond to the opposite side by staying in its current straight slot whenever a swap row offers both straight and turning options.

    Key insight:
    - The pipes do not encode one universal permutation anymore: solid tokens follow the permuted wire identity, while X tokens follow the straight slot identity, so token type chooses the routing permutation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)
    diamonds = _find_diamond_3x3(g, bg)

    track_mask = g != bg
    for (_, _), (_, (r0, c0)) in solids.items():
        track_mask[r0:r0 + 3, c0:c0 + 3] = False
    for cr, cc in xs:
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False
    for cr, cc in diamonds:
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0:r0 + 3, c0:c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue
        inferred = _infer_component_permutation(comp, g)
        if inferred is None:
            continue
        slot_cols, perm, inv_perm, top_ep_row, bot_ep_row = inferred
        col_to_slot = {c: i for i, c in enumerate(slot_cols)}

        endpoints = [p for p, d in _component_degrees(comp).items() if d <= 1]
        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                token_kind = None
                color = None
                if cen in solids:
                    token_kind = "solid"
                    color = solids[cen][0]
                elif cen in xs:
                    token_kind = "x"
                    color = xs[cen][0]
                if token_kind is None:
                    continue

                if ec not in col_to_slot:
                    continue
                src_slot = col_to_slot[ec]
                if er == top_ep_row:
                    dest_slot = perm[src_slot] if token_kind == "solid" else src_slot
                    dest = (bot_ep_row + 2, slot_cols[dest_slot])
                elif er == bot_ep_row:
                    dest_slot = inv_perm[src_slot] if token_kind == "solid" else src_slot
                    dest = (top_ep_row - 2, slot_cols[dest_slot])
                else:
                    continue
                stamp = _X_OFF if token_kind == "solid" else _DIAMOND_OFF
                for dr, dc in stamp:
                    rr, cc2 = dest[0] + dr, dest[1] + dc
                    if 0 <= rr < h and 0 <= cc2 < w:
                        out[rr, cc2] = color

    return out


# --------------- generator ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [12, 42, 28, 18]
RECT_WIRE_COUNT_WEIGHTS = [55, 35, 10]
RECT_PARALLEL_PROB = 0.8
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 6
RECT_MARGIN_GAP = 1

FAR_DIAMOND_PROB = 0.33
GEN_MAX_RETRIES = 400

PERMUTATION_LIBRARY = {
    4: [
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ],
    5: [
        [2, 3, 0, 1, 4],
        [3, 4, 2, 0, 1],
        [2, 0, 1, 4, 3],
    ],
}


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into crossover moves using a bypass column."""
    n = len(perm)
    visited = [False] * n
    crossovers = []
    for start in range(n):
        if visited[start] or perm[start] == start:
            visited[start] = True
            continue
        cycle = []
        i = start
        while not visited[i]:
            visited[i] = True
            cycle.append(i)
            i = perm[i]
        if len(cycle) < 2:
            continue
        crossovers.append((cycle[0], -1))
        for k in range(len(cycle) - 1, 0, -1):
            crossovers.append((cycle[k], cycle[(k + 1) % len(cycle)]))
        crossovers.append((-1, cycle[1]))
    return crossovers


def _perm_cycles(perm: list[int]) -> list[list[int]]:
    n = len(perm)
    seen = [False] * n
    cycles = []
    for i in range(n):
        if seen[i]:
            continue
        cyc = []
        j = i
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = perm[j]
        cycles.append(cyc)
    return cycles


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _span_hits_xover(rect_r0: int, rect_r1: int, xover_rows: list[int]) -> bool:
    return any(rect_r0 <= xr <= rect_r1 for xr in xover_rows)


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    xover_rows: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    n: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Attempt to place one passive pass-through 9-rect away from conditional swap rows."""
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]

    if target_wires >= 2 and random.random() < RECT_PARALLEL_PROB and n >= 2:
        mode = "parallel"
    else:
        mode = "single"

    if mode == "parallel":
        j0 = random.randint(0, n - 2)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)
    else:
        j0 = random.randint(0, n - 1)
        c_lo = c_hi = ch_cols[j0]
        margin_l, margin_r = random.randint(0, 2), random.randint(0, 2)
        if margin_l + margin_r < 1:
            margin_l = 1 if random.random() < 0.5 else 0
            margin_r = 1 - margin_l
    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r
    if rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH or rect_c0 < 0 or rect_c1 >= w:
        return None

    rect_h_max = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 4)
    if rect_h_max < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], rect_h_max)
    r_start = random.randint(top_ep_row + 2, bot_ep_row - rect_h - 1)
    rect_r0, rect_r1 = r_start, r_start + rect_h - 1
    if _span_hits_xover(rect_r0, rect_r1, xover_rows):
        return None

    wire_cols = sorted(c for c in ch_cols if rect_c0 <= c <= rect_c1)
    if not wire_cols:
        return None
    for c in wire_cols:
        if not np.all(grid[rect_r0:rect_r1 + 1, c] == 8):
            return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    if rect_r0 <= top_ep_row or rect_r1 >= bot_ep_row:
        return None
    for c in wire_cols:
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
            return None

    saved = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1].copy()
    grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = 9

    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(rect_c0, rect_c1 + 1):
        if rect_r0 - 1 >= 0 and grid[rect_r0 - 1, c] in (5, 8):
            n_cols.add(c)
        if rect_r1 + 1 < h and grid[rect_r1 + 1, c] in (5, 8):
            s_cols.add(c)
    for r in range(rect_r0, rect_r1 + 1):
        if rect_c0 - 1 >= 0 and grid[r, rect_c0 - 1] in (5, 8):
            w_rows.add(r)
        if rect_c1 + 1 < w and grid[r, rect_c1 + 1] in (5, 8):
            e_rows.add(r)
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows)
    if total_wires < 1 or total_wires > 3:
        grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _build_wire_network(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None]:
    """Draw the union wire network. slot_wire tracks the moving-wire identity per column slot."""
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None

    col_seg_start: dict[int, int] = {}
    for j in range(n):
        col_seg_start[ch_cols[j]] = top_ep_row
        grid[top_ep_row, ch_cols[j]] = 8

    for xover_idx, (src_slot, dst_slot) in enumerate(crossovers):
        xrow = xover_rows[xover_idx]
        if src_slot == -1:
            src_col, dst_col = bypass_col, ch_cols[dst_slot]
        elif dst_slot == -1:
            src_col, dst_col = ch_cols[src_slot], bypass_col
        else:
            src_col, dst_col = ch_cols[src_slot], ch_cols[dst_slot]

        active_cols = {ch_cols[j] for j in range(n) if slot_wire[j] is not None}
        if bypass_wire is not None:
            active_cols.add(bypass_col)

        for c in active_cols:
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= xrow - 1:
                    grid[start:xrow, c] = 8

        c_min, c_max = min(src_col, dst_col), max(src_col, dst_col)
        grid[xrow, c_min:c_max + 1] = 8
        for c in active_cols:
            grid[xrow, c] = 8

        for c in active_cols:
            if c == src_col or c == dst_col:
                continue
            if c_min < c < c_max:
                if c - 1 >= c_min:
                    grid[xrow, c - 1] = 5
                if c + 1 <= c_max:
                    grid[xrow, c + 1] = 5

        new_seg_start = {c: xrow for c in active_cols if c != src_col}
        new_seg_start[dst_col] = xrow
        col_seg_start = new_seg_start

        if src_slot == -1:
            slot_wire[dst_slot] = bypass_wire
            bypass_wire = None
        elif dst_slot == -1:
            bypass_wire = slot_wire[src_slot]
            slot_wire[src_slot] = None
        else:
            slot_wire[dst_slot] = slot_wire[src_slot]
            slot_wire[src_slot] = None

    for j in range(n):
        if slot_wire[j] is not None:
            c = ch_cols[j]
            if c in col_seg_start:
                start = col_seg_start[c] + 1
                if start <= bot_ep_row:
                    grid[start:bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _choose_perm_and_xchannels(n: int) -> tuple[list[int], set[int]]:
    candidates = []
    for perm in PERMUTATION_LIBRARY[n]:
        cycles = _perm_cycles(perm)
        nontrivial = [cyc for cyc in cycles if len(cyc) > 1]
        if len(nontrivial) >= 2:
            candidates.append((perm, nontrivial))
    perm, nontrivial = random.choice(candidates)
    take = random.randint(1, len(nontrivial) - 1)
    chosen_cycles = random.sample(nontrivial, take)
    x_channels = {idx for cyc in chosen_cycles for idx in cyc}
    assert x_channels and len(x_channels) < n
    return list(perm), x_channels


def _build_token_specs(
    n: int,
    x_channels: set[int],
) -> dict[int, dict[str, str]]:
    specs: dict[int, dict[str, str]] = {}
    for i in range(n):
        kind = "x" if i in x_channels else "solid"
        side = random.choice(["top", "bottom"])
        specs[i] = {"kind": kind, "side": side}
    return specs


def _derive_input(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    x_channels: set[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> tuple[np.ndarray, dict[int, dict[str, str]]]:
    inp = grid.copy()
    specs = _build_token_specs(n, x_channels)
    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_solid_center = (bot_token_row, ch_cols[perm[i]])
        bot_x_center = (bot_token_row, ch_cols[i])

        if specs[i]["kind"] == "x":
            near = top_center if specs[i]["side"] == "top" else bot_x_center
            far = bot_x_center if specs[i]["side"] == "top" else top_center
            _stamp_token(inp, near[0], near[1], col, _X)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND)
        else:
            side_center = top_center if specs[i]["side"] == "top" else bot_solid_center
            _stamp_token(inp, side_center[0], side_center[1], col, _SOLID)
    return inp, specs


def _build_expected_output(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
    specs: dict[int, dict[str, str]],
) -> np.ndarray:
    out = grid.copy()
    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_solid_center = (bot_token_row, ch_cols[perm[i]])
        bot_x_center = (bot_token_row, ch_cols[i])
        kind = specs[i]["kind"]
        side = specs[i]["side"]

        if kind == "x":
            near = top_center if side == "top" else bot_x_center
            far = bot_x_center if side == "top" else top_center
            _stamp_token(out, near[0], near[1], col, _X)
            _stamp_token(out, far[0], far[1], col, _DIAMOND)
        else:
            near = top_center if side == "top" else bot_solid_center
            far = bot_solid_center if side == "top" else top_center
            _stamp_token(out, near[0], near[1], col, _DIAMOND)
            _stamp_token(out, far[0], far[1], col, _X)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET

    w = max(bypass_col + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n)
    perm, x_channels = _choose_perm_and_xchannels(n)

    crossovers = _decompose_to_crossovers(perm)
    if not crossovers:
        return None
    n_xovers = len(crossovers)

    min_wire_rows = n_xovers * 2 + 2
    h_min = top_ep_row + min_wire_rows + top_token_row + 3
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_wire_rows:
        return None

    avail = bot_ep_row - top_ep_row - 2
    step = max(2, avail // (n_xovers + 1))
    xover_rows = [top_ep_row + 1 + step * (s + 1) for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        xover_rows = [top_ep_row + 2 + s * 2 for s in range(n_xovers)]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((h, w), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid, n, ch_cols, bypass_col, top_ep_row, bot_ep_row, crossovers, xover_rows
    )

    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, xover_rows, top_ep_row, bot_ep_row, n, placed_rects)
        if result is not None:
            placed_rects.append(result)

    inp, specs = _derive_input(grid, n, ch_cols, perm, x_channels, token_colors, top_token_row, bot_token_row)
    output = _build_expected_output(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row, specs)

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
