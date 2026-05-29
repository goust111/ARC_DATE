# Variant of a47bf94d
# Dimension: D11 — 通道数量与空间布局 (meta)
# Change: 4-5个通道，均匀列间距4-5，顶部和底部放置token行，管道竖直为主加横向交叉段... → 四边环形布局

import random
import itertools

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the D11 variant
# spec_variant:
#   core_change:
#     old: 只有一组上下对置的竖直通道，token 只出现在 top/bottom 两边
#     new: 通道拆成两组叠加在同一网格里
#       - 竖直组: top <-> bottom
#       - 水平组: left <-> right
#     result: 输入和输出的四条边都会出现 token，中心区域出现大量正交交会
#   concrete_objects:
#     - token 形状仍然只用 solid / X / diamond，颜色规则不变
#     - 每个 token 中心仍与自己的 wire endpoint 相距 2 格，因此原 solver 的“从端点向外两格找 token”
#       这个核心识别方式可以保留
#     - 竖直组沿列做 permutation routing，水平组沿行做 permutation routing；两组只允许正交穿越，
#       不允许 T 形分叉，也不允许在 turn cell 上叠加另一组通道
#   behavior_trace:
#     old_example:
#       - top 某列有 solid，solver 先把它原地改成 diamond，再沿该列/横向 crossover trace 到 bottom 端点，
#         在 far end 写一个 X
#     new_example:
#       - top 某列仍可有 solid，行为同旧题
#       - 同时 right 某行也可有 X，solver 从 right 端点向左 trace 同一张大网络中的水平线路，
#         在 left 端补一个 diamond
#       - 二者会在中心十字口相交，但 trace 进入交点时总是优先直行，所以 vertical token 不会误拐到
#         horizontal channel，horizontal token 也不会误拐到 vertical channel
#   complexity_calibration:
#     - 总通道数仍为 4-5，避免自由度暴涨
#     - 两组通道数固定为 2+2 或 3+2，保证四边都有 token，同时仍能在 22-26 尺寸内稳定摆放
#     - 每组 permutation 限制为“至多 3 个 crossover move”的非恒等置换，避免布局不下，也避免过于简单
#   implicit_assumptions_checked:
#     - main() 假设 token 中心距端点 2 格: 新生成器继续严格满足
#     - main() 依赖 wire component 没有需要猜测的 T-junction: 新生成器只产生 straight crossing 和 L-turn
#     - _trace_wire 在十字口依赖“直行优先”: 新布局中的组间交会全部是 straight crossing
#     - main() 没有识别 diamond；原题靠“diamond 留在 track component 中”抑制重复发射。
#       新生成器继续只在 X-channel 的 far end 预放 diamond，因此这个旧机制仍然成立
#   legality:
#     - vertical crossover rows 放在所有 horizontal lane rows 之上
#     - horizontal crossover cols 放在所有 vertical lane cols 之左
#     - 因此 turn cells 不会与另一组通道重叠，组间共享的只有 straight-through 交点
#     - 9-rect 只允许覆盖 straight wire 区域，不允许覆盖 turn cells
#   observability:
#     - 输入中强制每组至少 1 个 solid 和 1 个 X，所以 top/bottom/left/right 四条边都实际可见 token
#     - 输出中 top+left 为 diamond 端，bottom+right 为 X 端，结构上能从少量样例中看出
#   cleanliness:
#     - 通道数仍小，交会集中在中心区域，边缘只放 token，不把 token 和 9-rect 混在一起

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H = 26
GRID_W = 26
TOP_TOKEN_ROW = 2
LEFT_TOKEN_COL = 2
GEN_MAX_RETRIES = 400
EP_TOKEN_GAP = 2

RECT_COUNT_WEIGHTS = [15, 45, 28, 12]
RECT_RETRY_FACTOR = 10
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 7)
RECT_MARGIN_GAP = 1
FAR_DIAMOND_PROB = 0.33

LANE_TEMPLATES = {
    2: [[13, 17], [13, 18], [14, 18]],
    3: [[13, 17, 20], [13, 18, 20]],
}

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
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
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
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


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangles and the wire-touching cells adjacent to each face."""
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


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow one wire track greedily, interpreting color-9 rectangles as pass-through junctions.

    Straight continuation wins at ordinary crossings, which is why the generator must avoid T-junctions.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped")
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [f for f in faces9 if f != entry_face]
                    if len(others) != 1:
                        raise AssertionError("invalid 9-rect routing")
                    exit_face = others[0]
                pts = faces9[exit_face]
                er, ec = cur
                if entry_face in ("N", "S"):
                    exit_cell = min(pts, key=lambda p: abs(p[1] - ec))
                else:
                    exit_cell = min(pts, key=lambda p: abs(p[0] - er))
                cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
                jumped = True
                break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [o for o in opts2 if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [o for o in opts2 if o[1] * d[0] + o[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
            else:
                p, dr, dc = opts2[0]

        if p == prev:
            return cur, d

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route edge tokens through an orthogonal wire network and materialize each token's complementary shape.

    Input structure:
    - Background cells are empty. Non-background cells form wire tracks plus 3x3 tokens placed two cells outside wire endpoints.
    - In this variant the endpoints can live on all four sides of the board, not just top and bottom.

    Rule:
    - A solid 3x3 token collapses in place into a diamond.
    - Any endpoint token also sends its complementary pattern to the far end of the same routed wire: solid sends X, X sends diamond.

    Key insight:
    - The solver does not need side-specific logic. It only needs to find endpoints, look two cells outward for a token center,
      and trace the wire while always continuing straight through 4-way crossings and through color-9 pass-through rectangles.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for cr, cc in xs.keys():
        for dr, dc in _X_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                track_mask[rr, cc2] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        for dr, dc in _DIAMOND_OFF:
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < h and 0 <= cc2 < w:
                out[rr, cc2] = col

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
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col
                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    end, dd = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
    """Decompose a non-identity permutation into crossover moves; -1 denotes the bypass lane."""
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


def _sample_perm(n: int, max_crossovers: int = 3) -> tuple[list[int], list[tuple[int, int]]]:
    perms = []
    base = list(range(n))
    for perm in itertools.permutations(base):
        perm = list(perm)
        if perm == base:
            continue
        crossovers = _decompose_to_crossovers(perm)
        if len(crossovers) <= max_crossovers:
            perms.append((perm, crossovers))
    if not perms:
        raise AssertionError("no valid permutation")
    return random.choice(perms)


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _build_wire_network(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    bypass_col: int,
    top_ep_row: int,
    bot_ep_row: int,
    crossovers: list[tuple[int, int]],
    xover_rows: list[int],
) -> tuple[list[int | None], int | None, set[tuple[int, int]]]:
    """Original top-to-bottom permutation router, plus the set of cells used by crossover turns."""
    slot_wire: list[int | None] = list(range(n))
    bypass_wire: int | None = None
    col_seg_start: dict[int, int] = {}
    protected_turns: set[tuple[int, int]] = set()

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
        grid[xrow, c_min : c_max + 1] = 8
        protected_turns.update((xrow, c) for c in range(c_min, c_max + 1))

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
                    grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire, protected_turns


def _choose_x_sets(nv: int, nh: int) -> tuple[set[int], set[int]]:
    xv = {random.randrange(nv)}
    xh = {random.randrange(nh)}
    if nv >= 3 and random.random() < 0.35:
        xv.add(random.choice([i for i in range(nv) if i not in xv]))
    elif nh >= 3 and random.random() < 0.35:
        xh.add(random.choice([i for i in range(nh) if i not in xh]))
    return xv, xh


def _transposed_turns(turns_t: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(c, r) for r, c in turns_t}


def _rect_contacts(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> tuple[set[int], set[int], set[int], set[int]]:
    h, w = grid.shape
    n_cols, s_cols, w_rows, e_rows = set(), set(), set(), set()
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0 and grid[r0 - 1, c] in (5, 8):
            n_cols.add(c)
        if r1 + 1 < h and grid[r1 + 1, c] in (5, 8):
            s_cols.add(c)
    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0 and grid[r, c0 - 1] in (5, 8):
            w_rows.add(r)
        if c1 + 1 < w and grid[r, c1 + 1] in (5, 8):
            e_rows.add(r)
    return n_cols, s_cols, w_rows, e_rows


def _try_place_9rect(
    grid: np.ndarray,
    placed_rects: list[tuple[int, int, int, int]],
    protected_turns: set[tuple[int, int]],
    endpoint_bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    top_ep_row, bot_ep_row, left_ep_col, right_ep_col = endpoint_bounds
    rect_h = random.randint(*RECT_H_RANGE)
    rect_w = random.randint(*RECT_W_RANGE)

    r0_min = top_ep_row + 1
    r0_max = bot_ep_row - rect_h
    c0_min = left_ep_col + 1
    c0_max = right_ep_col - rect_w
    if r0_min > r0_max or c0_min > c0_max:
        return None

    r0 = random.randint(r0_min, r0_max)
    c0 = random.randint(c0_min, c0_max)
    r1, c1 = r0 + rect_h - 1, c0 + rect_w - 1

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.any(sub == 8):
        return None
    if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
        return None
    if any((r, c) in protected_turns for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if r0 <= pr1 + gap and pr0 <= r1 + gap and c0 <= pc1 + gap and pc0 <= c1 + gap:
            return None

    saved = sub.copy()
    grid[r0 : r1 + 1, c0 : c1 + 1] = 9
    n_cols, s_cols, w_rows, e_rows = _rect_contacts(grid, r0, c0, r1, c1)
    total_wires = len(n_cols & s_cols) + len(w_rows & e_rows)
    valid = (
        total_wires >= 1
        and total_wires <= 3
        and n_cols == s_cols
        and w_rows == e_rows
    )
    if not valid:
        grid[r0 : r1 + 1, c0 : c1 + 1] = saved
        return None
    return (r0, c0, r1, c1)


def _derive_input(
    grid: np.ndarray,
    top_centers: list[tuple[int, int]],
    bottom_centers: list[tuple[int, int]],
    left_centers: list[tuple[int, int]],
    right_centers: list[tuple[int, int]],
    v_colors: list[int],
    h_colors: list[int],
    x_vertical: set[int],
    x_horizontal: set[int],
) -> np.ndarray:
    inp = grid.copy()
    for i, col in enumerate(v_colors):
        if i in x_vertical:
            _stamp_token(inp, bottom_centers[i][0], bottom_centers[i][1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, top_centers[i][0], top_centers[i][1], col, _DIAMOND_OFF)
        else:
            _stamp_token(inp, top_centers[i][0], top_centers[i][1], col, _SOLID_OFF)

    for i, col in enumerate(h_colors):
        if i in x_horizontal:
            _stamp_token(inp, right_centers[i][0], right_centers[i][1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, left_centers[i][0], left_centers[i][1], col, _DIAMOND_OFF)
        else:
            _stamp_token(inp, left_centers[i][0], left_centers[i][1], col, _SOLID_OFF)
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_total = random.choice([4, 5])
    if n_total == 4:
        nv, nh = 2, 2
    else:
        nv, nh = random.choice([(2, 3), (3, 2)])

    top_token_row = TOP_TOKEN_ROW
    left_token_col = LEFT_TOKEN_COL
    top_ep_row = top_token_row + EP_TOKEN_GAP
    left_ep_col = left_token_col + EP_TOKEN_GAP
    h = GRID_H
    w = GRID_W
    bottom_token_row = h - top_token_row - 1
    right_token_col = w - left_token_col - 1
    bot_ep_row = bottom_token_row - EP_TOKEN_GAP
    right_ep_col = right_token_col - EP_TOKEN_GAP

    perm_v, crossovers_v = _sample_perm(nv, max_crossovers=3)
    perm_h, crossovers_h = _sample_perm(nh, max_crossovers=3)
    n_xv = len(crossovers_v)
    n_xh = len(crossovers_h)
    h_rows = random.choice(LANE_TEMPLATES[nh]).copy()
    v_cols = random.choice(LANE_TEMPLATES[nv]).copy()

    xover_rows_v = [top_ep_row + 2 + 2 * i for i in range(n_xv)]
    xover_cols_h = [left_ep_col + 2 + 2 * i for i in range(n_xh)]
    if n_xv and xover_rows_v[-1] >= min(h_rows):
        return None
    if n_xh and xover_cols_h[-1] >= min(v_cols):
        return None

    v_bypass_col = v_cols[0] - 2
    h_bypass_row = h_rows[0] - 2
    if v_bypass_col <= xover_cols_h[-1] or h_bypass_row <= xover_rows_v[-1]:
        return None
    if v_cols[-1] >= right_ep_col or h_rows[-1] >= bot_ep_row:
        return None

    token_colors = random.sample(TOKEN_COLORS_POOL, n_total)
    v_colors = token_colors[:nv]
    h_colors = token_colors[nv:]
    x_vertical, x_horizontal = _choose_x_sets(nv, nh)

    grid = np.zeros((h, w), dtype=int)
    slot_wire_v, bypass_v, turns_v = _build_wire_network(
        grid, nv, v_cols, v_bypass_col, top_ep_row, bot_ep_row, crossovers_v, xover_rows_v
    )
    slot_wire_h, bypass_h, turns_h_t = _build_wire_network(
        grid.T, nh, h_rows, h_bypass_row, left_ep_col, right_ep_col, crossovers_h, xover_cols_h
    )

    inv_perm_v = [0] * nv
    for i in range(nv):
        inv_perm_v[perm_v[i]] = i
    inv_perm_h = [0] * nh
    for i in range(nh):
        inv_perm_h[perm_h[i]] = i
    if any(slot_wire_v[j] != inv_perm_v[j] for j in range(nv)) or bypass_v is not None:
        return None
    if any(slot_wire_h[j] != inv_perm_h[j] for j in range(nh)) or bypass_h is not None:
        return None

    top_centers = [(top_token_row, c) for c in v_cols]
    bottom_centers = [(bottom_token_row, v_cols[perm_v[i]]) for i in range(nv)]
    left_centers = [(r, left_token_col) for r in h_rows]
    right_centers = [(h_rows[perm_h[i]], right_token_col) for i in range(nh)]

    protected_turns = turns_v | _transposed_turns(turns_h_t)
    placed_rects: list[tuple[int, int, int, int]] = []
    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    endpoint_bounds = (top_ep_row, bot_ep_row, left_ep_col, right_ep_col)
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        rect = _try_place_9rect(grid, placed_rects, protected_turns, endpoint_bounds)
        if rect is not None:
            placed_rects.append(rect)

    output = grid.copy()
    for i, col in enumerate(v_colors):
        _stamp_token(output, top_centers[i][0], top_centers[i][1], col, _DIAMOND_OFF)
        _stamp_token(output, bottom_centers[i][0], bottom_centers[i][1], col, _X_OFF)
    for i, col in enumerate(h_colors):
        _stamp_token(output, left_centers[i][0], left_centers[i][1], col, _DIAMOND_OFF)
        _stamp_token(output, right_centers[i][0], right_centers[i][1], col, _X_OFF)

    inp = _derive_input(
        grid,
        top_centers,
        bottom_centers,
        left_centers,
        right_centers,
        v_colors,
        h_colors,
        x_vertical,
        x_horizontal,
    )

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
