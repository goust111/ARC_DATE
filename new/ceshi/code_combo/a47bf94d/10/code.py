# Combo variant of a47bf94d (index 10)
# combo_spec: D8/1+D10/0
# Name: 阻断过滤+循环置换

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# -----------------------------------------------------------------------------
# Design analysis for combo D8/1 + D10/0
#
# concrete_behavior:
# - 原题里，每个顶/底端点外侧放一个 token；solid 会在原位裂成 diamond，并把互补的 X 送到线路另一端，
#   X 则保持原状并把 diamond 送到另一端。9 号矩形只是无条件穿线。
# - 融合后保留 token 形状语义，但每条通道在靠近顶部端点的竖直主干上都嵌入一个 9 号矩形过滤器。
#   过滤器若面积为偶数，token 可以穿过；若面积为奇数，token 在此被拦下，不再向远端发送互补形状。
# - 路由不再是任意 permutation，而是严格的一整个 cycle。视觉上看，顶部颜色序列沿管道整体轮转到底部；
#   但只有通过偶面积过滤器的那些 token 才会真的在底部出现互补形状。
#
# behavior_trace:
# - 具体小例子：4 个顶部 token 颜色按列为 A,B,C,D；底部线路顺序是 A->C, C->D, D->B, B->A，
#   这是一整个 4-cycle。若 A 列顶部过滤块是 3x3，B/C/D 分别是 2x3/4x3/2x2：
#   1. 原题：A/B/C/D 都会沿线抵达各自 cycle 目标列。
#   2. 融合后：A 在进入 3x3 过滤块时被拦下，只保留顶部原地变化；B/C/D 通过偶面积块，仍到达底部目标列。
#   3. 差异：输出底部只缺失 A 的互补形状，而且这个缺失并不是“没有线路”，而是“线路存在但被奇面积 9 块阻断”。
#
# complexity_calibration:
# - 如果让 9 块随机覆盖多根线、且一个 token 可能连续穿过多个过滤块，那么“到底被哪一块决定了命运”会变得难观察。
#   因此 generator 约束为“每个源通道恰好一个单线过滤块，且位于第一次 crossover 之前”，这样每个 token 的阻断原因唯一。
# - 过滤尺寸只用少量档位：允许的偶面积主要是 2x2 / 2x3 / 4x3，阻断的奇面积主要是 3x3 / 5x3。
#   这样单个样例内就能同时出现两种 odd-area 阻断块，避免规则退化成“只有 3x3 会拦截”。
#
# implicit_assumption_audit:
# - 原 _trace_wire() 假设进入 9 块一定要穿出；这里必须把“进入奇面积 9 块可能直接终止”编码进 trace 返回值，
#   否则 solver 会错误地给被阻断的 token 生成远端互补形状。
# - 原 generator 允许 0-3 个 9 块且它们与通道关系松散；这里新 solver 默认“每个 token 的路径上至多出现一个决定性的过滤块”，
#   因此 generator 必须显式保证每列顶部各放一个、且第一次 crossover 之前再无其他 9 块。
# - 原 solver 扫描所有 degree-1 端点，并假定端点外 2 格若有 token，则存在唯一远端。融合后这个假设只对“偶面积或无过滤块”成立；
#   对奇面积块，远端不存在，所以 trace 需要允许 blocked=True。
# - 原 permutation 可任意分解成多个 cycle；融合后若还允许多个 cycle，人类可能把“某些底部缺失”误判成“另一个独立回路没有源”。
#   因此 generator 必须生成单一 cycle，避免过滤规则和拓扑规则互相混淆。
#
# placement_legality:
# - 9 块只放在顶部竖段，矩形完全位于第一次 crossover 之上，且只覆盖一根竖线，这样它只会有 N/S 两个面接线。
# - 宽度只用 2 或 3，配合 4 列间距可保证相邻过滤块不重叠；高度只用 2-5，且第一次 crossover 延后到更下方，
#   从而 5x3 这种第二类奇面积块也能完整放下。
# - token 中心仍固定在端点外 2 格，因此 solver 的端点->token 匹配规则保持合法。
#
# observability_and_cleanliness:
# - 单一 cycle 通过整条线路拓扑可见；过滤规则通过“同样有线路，但有些底部位置始终不出现互补形状”可见。
# - 颜色值本身不参与任何排序或奇偶判断；决定通行的是可见的矩形尺寸。
# - 同一区域只叠加一种语义：顶部是源 token，顶部竖段是过滤块，中部是 crossover，底部是潜在目的 token，视觉层次清晰。
# -----------------------------------------------------------------------------

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col in (bg, 5, 8, 9) or np.any(sub != col):
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
            if col in (bg, 5, 8, 9):
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


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    """Find color-9 rectangular blocks and expose per-cell lookup."""
    mask9 = g == 9
    if not mask9.any():
        return [], {}
    h, w = g.shape
    rects: list[dict] = []
    lookup: dict[tuple[int, int], dict] = {}
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
        rect = {
            "cells": cells,
            "faces": faces,
            "area": len(cells),
            "allow": (len(cells) % 2 == 0),
            "bbox": (r0, c0, r1, c1),
        }
        rects.append(rect)
        for cell in cells:
            lookup[cell] = rect
    return rects, lookup


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rect_lookup: dict[tuple[int, int], dict],
) -> tuple[tuple[int, int] | None, tuple[int, int] | None, bool]:
    """
    Greedily follow a wire component.

    Returns (end_cell, outgoing_dir, blocked). If an odd-area 9-rect is entered,
    the token is blocked and the far endpoint is omitted.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped in filtered pipeline")
        visited.add(state)

        rect = rect_lookup.get(cur)
        if rect is not None:
            if not rect["allow"]:
                return None, None, True
            if d not in _DIR2FACE:
                raise AssertionError("unexpected non-orthogonal entry into 9-rect")
            entry_face = _DIR2FACE[d]
            opp = _OPP[entry_face]
            faces = rect["faces"]
            if opp in faces:
                exit_face = opp
            else:
                others = [face for face in faces if face != entry_face]
                if len(others) != 1:
                    raise AssertionError("ambiguous even-area 9-rect exit")
                exit_face = others[0]
            pts = faces[exit_face]
            er, ec = cur
            if entry_face in ("N", "S"):
                exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
            else:
                exit_cell = min(pts, key=lambda p: abs(p[0] - er))
            cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, False

        rev = (-d[0], -d[1])
        nonrev = [opt for opt in opts if (opt[1], opt[2]) != rev]
        opts2 = nonrev if nonrev else opts
        straight = [opt for opt in opts2 if (opt[1], opt[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [opt for opt in opts2 if opt[1] * d[0] + opt[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
            else:
                p, dr, dc = opts2[0]

        if p == prev:
            return cur, d, False

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_token(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[cr + dr, cc + dc] = color


def _clear_solid(grid: np.ndarray, r0: int, c0: int, bg: int) -> None:
    grid[r0:r0 + 3, c0:c0 + 3] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through a cyclic wire network whose color-9 filters may stop transmission.

    Input structure:
    - The background is the dominant color. Non-background cells form one orthogonal wire network made of 8/5,
      plus filled color-9 rectangles embedded in vertical wire segments, plus 3x3 tokens attached two cells outside endpoints.
    - Tokens are either solid 3x3 blocks or X-patterns. Each 9-rectangle is a filled rectangle whose visible dimensions
      determine whether it is an even-area pass filter or an odd-area blocker.

    Rule:
    - Every solid token always decomposes locally into a diamond at the same center; every X token stays as an X.
    - A token also tries to send its complementary shape to the far endpoint determined by the wire permutation.
      Passing through an even-area 9-rectangle is allowed; entering an odd-area 9-rectangle stops the transmission,
      so no remote complementary shape is created.

    Key insight:
    - Solve the wiring exactly as in the original family, but treat each color-9 rectangle as a visible gate whose area parity
      decides whether the token continues around the single cycle or dies at the gate.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0:r0 + 3, c0:c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    _rects, rect_lookup = _find_9rects(g)

    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        _clear_solid(out, r0, c0, bg)
        _stamp_token(out, (cr, cc), col, _DIAMOND_OFF)

    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

        comp_colors = {int(g[r, c]) for r, c in comp}
        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]
        if not endpoints:
            continue

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N4:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, _ = solids[cen]
                    if col in comp_colors:
                        continue
                    end, dd, blocked = _trace_wire(comp, ep, (-odr, -odc), h, w, rect_lookup)
                    if blocked:
                        continue
                    assert end is not None and dd is not None
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    _stamp_token(out, dest, col, _X_OFF)

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    end, dd, blocked = _trace_wire(comp, ep, (-odr, -odc), h, w, rect_lookup)
                    if blocked:
                        continue
                    assert end is not None and dd is not None
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    _stamp_token(out, dest, col, _DIAMOND_OFF)

    return out


# --------------- generator (filtered_cycle) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = 4
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW = 2
TOP_EP_ROW = TOKEN_ROW + 2
GRID_H = 28
GRID_W_MIN = 22
GRID_W_MAX = 26
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GEN_MAX_RETRIES = 300

RECT_TOP_ROW = 5
XOVER_BASE_ROW = 11
EVEN_DIMS = [(2, 2), (2, 3), (4, 3)]
ODD_DIMS = [(3, 3), (5, 3)]
X_INPUT_COUNT_CHOICES = [1, 2]
X_INPUT_COUNT_WEIGHTS = [70, 30]


def _random_single_cycle(n: int) -> list[int]:
    order = list(range(n))
    random.shuffle(order)
    perm = [0] * n
    for idx, src in enumerate(order):
        perm[src] = order[(idx + 1) % n]
    return perm


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a permutation into sequential bypass-assisted crossover moves."""
    n = len(perm)
    visited = [False] * n
    crossovers: list[tuple[int, int]] = []
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
    """Draw the orthogonal wire network; same semantics as the original generator."""
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


def _rect_span_for_channel(c: int, rect_w: int) -> tuple[int, int]:
    if rect_w == 3:
        return c - 1, c + 1
    if random.random() < 0.5:
        return c - 1, c
    return c, c + 1


def _place_filter_rects(
    grid: np.ndarray,
    ch_cols: list[int],
    allowed_mask: list[bool],
) -> list[dict]:
    allowed_dims = random.sample(EVEN_DIMS, k=min(sum(allowed_mask), len(EVEN_DIMS)))
    blocked_dims = random.sample(ODD_DIMS, k=min(len(ch_cols) - sum(allowed_mask), len(ODD_DIMS)))
    while len(allowed_dims) < sum(allowed_mask):
        allowed_dims.append(random.choice(EVEN_DIMS))
    while len(blocked_dims) < len(ch_cols) - sum(allowed_mask):
        blocked_dims.append(random.choice(ODD_DIMS))

    rect_specs = []
    allow_idx = 0
    block_idx = 0
    for idx, c in enumerate(ch_cols):
        if allowed_mask[idx]:
            rect_h, rect_w = allowed_dims[allow_idx]
            allow_idx += 1
        else:
            rect_h, rect_w = blocked_dims[block_idx]
            block_idx += 1
        r0 = RECT_TOP_ROW
        r1 = r0 + rect_h - 1
        c0, c1 = _rect_span_for_channel(c, rect_w)
        grid[r0:r1 + 1, c0:c1 + 1] = 9
        rect_specs.append(
            {
                "channel": idx,
                "dims": (rect_h, rect_w),
                "allow": allowed_mask[idx],
                "bbox": (r0, c0, r1, c1),
            }
        )
    return rect_specs


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(CHANNEL_COUNTS)
    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * CHANNEL_SPACING for i in range(n)]
    bypass_col = ch_cols[-1] + BYPASS_COL_OFFSET
    width = max(bypass_col + 3, GRID_W_MIN)
    if width > GRID_W_MAX:
        return None

    top_token_row = TOKEN_ROW
    top_ep_row = TOP_EP_ROW
    bot_token_row = GRID_H - top_token_row - 1
    bot_ep_row = bot_token_row - 2

    perm = _random_single_cycle(n)
    crossovers = _decompose_to_crossovers(perm)
    if len(crossovers) != n + 1:
        return None
    xover_rows = [XOVER_BASE_ROW + 2 * i for i in range(len(crossovers))]
    if xover_rows[-1] >= bot_ep_row:
        return None

    grid = np.zeros((GRID_H, width), dtype=int)
    slot_wire, bypass_wire = _build_wire_network(
        grid,
        n,
        ch_cols,
        bypass_col,
        top_ep_row,
        bot_ep_row,
        crossovers,
        xover_rows,
    )
    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    if any(slot_wire[j] != inv_perm[j] for j in range(n)) or bypass_wire is not None:
        return None

    blocked_count = min(2, n - 2)
    blocked_channels = set(random.sample(range(n), blocked_count))
    allowed_mask = [i not in blocked_channels for i in range(n)]
    if all(allowed_mask) or not any(allowed_mask):
        return None
    rect_specs = _place_filter_rects(grid, ch_cols, allowed_mask)
    if any(spec["bbox"][2] >= xover_rows[0] for spec in rect_specs):
        return None

    token_colors = random.sample(TOKEN_COLORS_POOL, n)
    n_x = random.choices(X_INPUT_COUNT_CHOICES, weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    inp = grid.copy()
    out = grid.copy()
    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            _stamp_token(inp, top_center, col, _X_OFF)
            _stamp_token(out, top_center, col, _X_OFF)
            if allowed_mask[i]:
                _stamp_token(out, bot_center, col, _DIAMOND_OFF)
        else:
            _stamp_token(inp, top_center, col, _SOLID_OFF)
            _stamp_token(out, top_center, col, _DIAMOND_OFF)
            if allowed_mask[i]:
                _stamp_token(out, bot_center, col, _X_OFF)

    try:
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
