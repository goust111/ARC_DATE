# Combo variant of a47bf94d (index 4)
# combo_spec: D1/1+D20/2
# Name: 对角路径+密集网格管道

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# concrete_behavior:
# - 保留原题的 token 语义：solid 3x3 会在原地裂解为 diamond；solid/X 都会把互补形状送到管道另一端。
# - 融合改动不再使用“少量竖线 + 1-2 个稀疏换轨点”。这里把主网络做成 3 层交错的对角交换带：
#   若通道列是 [c0,c1,c2,c3,c4]，第 1 层交换 (c0,c1),(c2,c3)，第 2 层交换 (c1,c2),(c3,c4)，
#   第 3 层再交换 (c0,c1),(c2,c3)；视觉上形成连续的棋盘式 X/X/X 晶格。
# - 具体追踪例子：假设颜色 2 的 solid 在顶部 c0。原题会沿单条正交线或少量横向 crossover 走到底。
#   这里它先沿竖线下行，到第 1 层被迫转入对角线到 c1；第 2 层若当前列参与交换，又继续沿另一条对角线到 c2；
#   第 3 层再根据所在列决定是否回切。最终底部端点由“跨过哪些斜带”决定，而不是单次水平换轨决定。
#
# behavior_trace:
# - 原题 solver 的关键假设是 4 邻接管道；D1 单变体把它放宽到 8 邻接并在单个 diagonal X 交叉处保持“直行优先”。
# - 这次融合进一步放大该机制：同一个 connected component 里重复出现多层 diagonal X，且层与层交错覆盖不同相邻列。
#   因而 token 的远端匹配必须沿 8 邻接持续追踪，不能把整块密集网当作任意可达的无向图胡乱连。
# - 在局部节点上仍采用统一规则：若当前方向还能直行，就继续；否则在不反向的邻居里选与当前方向点积最大的那条。
#   这样从竖直段进入交换带时会顺滑转入对角段，在 X 交点又会沿当前斜率穿过，而不是横跳到另一条线。
#
# complexity_calibration:
# - 如果允许任意列对之间自由连线，密集网会退化为难以读懂的大杂烩；如果只保留 1 个 band，又看不出 D20/2 的密集网特征。
# - 因此 generator 固定为 3 层、相邻列交换、奇偶层交错的少量档位。这样每个样例里会重复出现相同结构，
#   人眼能从 2-3 个示例总结“顺着斜带连续追踪”的规律，同时网络明显比原题更密。
# - 通道间距固定为 4。这样既能容纳 45 度对角线，又能在 26x26 以内放下 3 层交换带、上下 token 和偶发 9-rect。
#
# implicit_assumption_audit:
# - 原 main() 用 4 邻接做 connected components 和 degree，融合后必须统一改成 8 邻接，否则 diagonal band 会被拆断。
# - 原 endpoint→token 搜索只枚举 4 个方向；这里保留 token 在上下两侧，但 solver 仍改成枚举 8 个方向，
#   避免对角终端或后续预放置形状导致漏检。
# - 原 _trace_wire() 的转向逻辑默认“直行/垂直转弯”足够；密集网里需要允许 vertical↔diagonal 的连续切换，
#   所以 fallback 必须按点积选最接近当前方向的邻居，而不是只看正交转弯。
# - 新增的密集晶格会自然生成许多 8 色的局部 X 图案；如果 token 检测仍把任意颜色都当候选，就会误把管道识别成 X-token。
#   因此 solver 额外要求 token 颜色不能是 {5,8,9}，而 generator 也只从 {1,2,3,4,6,7} 取 token 颜色来满足这一点。
# - 9-rect 仍沿 N/S/E/W 面进出，因此 generator 只把它们放在单根竖直直通线的局部窗口上。
#   这保证进入 9-rect 时方向仍是正交的，不会引入“斜着进矩形”的新未定义情况。
#
# placement_legality:
# - 交换带只连接相邻列，且同一层只交换不相邻的 pair，避免同层内多条斜线互相擦边混淆。
# - token 中心仍与端点保持 2 格轴向距离，所以 solver 的 “dest = endpoint + 2 * final_dir” 仍成立。
# - 9-rect 放置前会检查：候选区域只能覆盖 0/8；矩形上下必须保留 8 作为入口出口；放入 9 后统计面连接数必须恰好是 1 根线。
# - generator 采用 output-first，自检 `main(inp) == output`；因此任何局部布局若破坏了 solver 的隐式假设，都会被回退重试。
#
# visual_observability_and_cleanliness:
# - 肉眼可见的主变化是整片 cyan 管道出现重复 X 形对角晶格，而不是少量稀疏横线。这个差异在 1 个样例里就很明显。
# - 颜色不参与排序、优先级或路由判定；颜色只标识 token 身份，人类观察者可以纯靠几何追踪推断对应关系。
# - 9-rect 只偶尔嵌在单根竖直线上，不会和斜带大面积重叠；因此网格虽然更密，但 token、rect、主通路仍能分辨。
# -----------------------------------------------------------------------------

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_N8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID_OFF = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]
_PIPE_COLORS = {5, 8, 9}  # dense lattice creates accidental X motifs in pipe colors; exclude them from token detection


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    """Return {center: (color, (r0,c0))} for maximal 3x3 solid blocks."""
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0:r0 + 3, c0:c0 + 3]
            col = int(sub[0, 0])
            if col == bg or col in _PIPE_COLORS or np.any(sub != col):
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
            if col == bg or col in _PIPE_COLORS:
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
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _segment_cells(a: tuple[int, int], b: tuple[int, int]):
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    steps = max(abs(dr), abs(dc))
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc)
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _draw_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int = 8) -> None:
    for rr, cc in _segment_cells(a, b):
        grid[rr, cc] = color


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
    """Find color-9 rectangular blocks and their wire connections per face."""
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


def _trace_wire(
    comp: set[tuple[int, int]],
    start: tuple[int, int],
    dir0: tuple[int, int],
    h: int,
    w: int,
    rects_9: list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow an 8-connected wire with 9-rect pass-through.

    Dense diagonal lattices create repeated X-junctions. The traversal rule is:
    keep the current direction when possible; otherwise choose the non-reversing
    neighbor whose direction has the largest dot product with the current one.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 8):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped in dense diagonal lattice")
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                if d not in _DIR2FACE:
                    raise AssertionError("diagonal entry into 9-rect is unsupported")
                entry_face = _DIR2FACE[d]
                opp = _OPP[entry_face]
                if opp in faces9:
                    exit_face = opp
                else:
                    others = [f for f in faces9 if f != entry_face]
                    if len(others) != 1:
                        raise AssertionError("ambiguous 9-rect exit")
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
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors8(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [o for o in opts2 if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            p, dr, dc = max(
                opts2,
                key=lambda o: (o[1] * d[0] + o[2] * d[1], -(abs(o[1]) + abs(o[2])), -o[0][0], -o[0][1]),
            )

        if p == prev:
            return cur, d

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route colored tokens through a dense orthogonal/diagonal lattice and emit their complementary 3x3 patterns.

    Input structure:
    - The background is the most frequent color. Non-background cells form one cyan wire component with repeated
      diagonal swap bands, optional maroon pass-through rectangles on straight vertical subsegments, and colored
      3x3 tokens placed two cells outside top or bottom endpoints.
    - Pipes are 8-connected: the same component mixes vertical runs with many 45-degree diagonal X junctions.

    Rule:
    - Every solid 3x3 block collapses in place into a diamond (the four cardinal cells around its center).
    - Any solid or X token at one endpoint sends its complementary shape to the far endpoint reached by tracing
      the lattice: solid emits X, X emits diamond. Existing diamonds stay unchanged.
    - A maroon rectangle still acts as a chamber that passes a wire straight through from one face to the opposite
      face, or through the unique other face when the rectangle encodes a forced turn.

    Key insight:
    - The token transformation is still the same shape cycle as the original task. The fused difficulty is that the
      pipe is now a dense checkerboard-like lattice, so the far endpoint is found only by 8-neighbor tracing that
      preserves the current travel direction across diagonal crossings rather than treating the whole network as
      an arbitrary connected blob.
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
    rects_9 = _find_9rects(g)
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

        comp_colors = {int(g[r, c]) for r, c in comp}
        deg = {p: 0 for p in comp}
        for r, c in comp:
            for nr, nc in neighbors8(r, c):
                if (nr, nc) in comp:
                    deg[(r, c)] += 1
        endpoints = [p for p in comp if deg[p] <= 1]
        if not endpoints:
            continue

        for ep in endpoints:
            er, ec = ep
            for odr, odc in _N8:
                cen = (er + 2 * odr, ec + 2 * odc)
                if not (0 <= cen[0] < h and 0 <= cen[1] < w):
                    continue

                if cen in solids:
                    col, _ = solids[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _X_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

                elif cen in xs:
                    col, _ = xs[cen]
                    if col in comp_colors:
                        continue
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


# --------------- generator (dense diagonal lattice variant) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = 4
LEFT_MARGIN_RANGE = (2, 3)
TOP_TOKEN_ROW = 2
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H = 26
GRID_W_MIN = 24
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

BAND_COUNT = 3
BAND_START_OFFSET = 2
BAND_GAP = 1

RECT_COUNT_WEIGHTS = [35, 45, 20]
RECT_H_RANGE = (2, 3)
RECT_MAX_WIDTH = 3
RECT_RETRY_FACTOR = 12
RECT_MARGIN_GAP = 1

# Keep the X-input branch visible across small ARC training sets instead of
# over-concentrating on the all-solid case.
X_INPUT_COUNT_WEIGHTS = [30, 45, 25]
FAR_DIAMOND_PROB = 0.33

GEN_MAX_RETRIES = 400


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _make_full_phases(n: int, band_count: int, start_parity: int) -> tuple[list[list[int]], list[int]]:
    phases: list[list[int]] = []
    order = list(range(n))
    for band_idx in range(band_count):
        parity = (start_parity + band_idx) % 2
        pairs = list(range(parity, n - 1, 2))
        phases.append(pairs)
        for j in pairs:
            order[j], order[j + 1] = order[j + 1], order[j]

    perm = [0] * n
    for slot, wire_id in enumerate(order):
        perm[wire_id] = slot
    return phases, perm


def _build_wire_network(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    phases: list[list[int]],
    band_starts: list[int],
    spacing: int,
) -> None:
    current_row = top_ep_row
    n = len(ch_cols)

    for c in ch_cols:
        grid[top_ep_row, c] = 8

    for phase_pairs, band_r0 in zip(phases, band_starts):
        band_r1 = band_r0 + spacing
        for c in ch_cols:
            _draw_segment(grid, (current_row, c), (band_r0, c), 8)

        swapped_slots = set()
        for j in phase_pairs:
            swapped_slots.add(j)
            swapped_slots.add(j + 1)
            c0, c1 = ch_cols[j], ch_cols[j + 1]
            _draw_segment(grid, (band_r0, c0), (band_r1, c1), 8)
            _draw_segment(grid, (band_r0, c1), (band_r1, c0), 8)

        for j in range(n):
            if j not in swapped_slots:
                c = ch_cols[j]
                _draw_segment(grid, (band_r0, c), (band_r1, c), 8)

        current_row = band_r1

    for c in ch_cols:
        _draw_segment(grid, (current_row, c), (bot_ep_row, c), 8)


def _rect_wire_count(grid: np.ndarray, rect_r0: int, rect_c0: int, rect_r1: int, rect_c1: int) -> int:
    h, w = grid.shape
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
    return len(n_cols & s_cols) + len(w_rows & e_rows)


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    phases: list[list[int]],
    band_starts: list[int],
    spacing: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape

    candidates: list[tuple[int, int]] = []
    for band_idx, phase_pairs in enumerate(phases):
        swapped = set()
        for j in phase_pairs:
            swapped.add(j)
            swapped.add(j + 1)
        for slot in range(len(ch_cols)):
            if slot not in swapped:
                candidates.append((band_idx, slot))
    if not candidates:
        return None

    band_idx, slot = random.choice(candidates)
    col = ch_cols[slot]
    band_r0 = band_starts[band_idx]
    band_r1 = band_r0 + spacing
    max_h = min(RECT_H_RANGE[1], band_r1 - band_r0 - 1)
    if max_h < RECT_H_RANGE[0]:
        return None

    rect_h = random.randint(RECT_H_RANGE[0], max_h)
    rect_r0 = random.randint(band_r0 + 1, band_r1 - rect_h)
    rect_r1 = rect_r0 + rect_h - 1

    margin_l = random.randint(0, 1)
    margin_r = random.randint(0, 1)
    rect_c0 = col - margin_l
    rect_c1 = col + margin_r
    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
        return None

    if grid[rect_r0 - 1, col] != 8 or grid[rect_r1 + 1, col] != 8:
        return None
    if not np.all(grid[rect_r0:rect_r1 + 1, col] == 8):
        return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 8)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    saved = sub.copy()
    grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = 9
    total_wires = _rect_wire_count(grid, rect_r0, rect_c0, rect_r1, rect_c1)
    if total_wires != 1:
        grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = saved
        return None
    return (rect_r0, rect_c0, rect_r1, rect_c1)


def _derive_input(
    grid: np.ndarray,
    n: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n), min(n_x, n)))

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            _stamp_token(inp, bot_center[0], bot_center[1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, top_center[0], top_center[1], col, _DIAMOND_OFF)
        else:
            _stamp_token(inp, top_center[0], top_center[1], col, _SOLID_OFF)

    return inp


def _gen_attempt():
    n = random.choice(CHANNEL_COUNTS)
    spacing = CHANNEL_SPACING
    top_token_row = TOP_TOKEN_ROW
    top_ep_row = top_token_row + EP_TOKEN_GAP

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    w = max(ch_cols[-1] + 4, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    h = GRID_H
    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP

    band_starts = [top_ep_row + BAND_START_OFFSET + i * (spacing + BAND_GAP) for i in range(BAND_COUNT)]
    if band_starts[-1] + spacing >= bot_ep_row:
        return None

    token_colors = random.sample(TOKEN_COLORS_POOL, n)
    phases, perm = _make_full_phases(n, BAND_COUNT, random.choice([0, 1]))

    grid = np.zeros((h, w), dtype=int)
    _build_wire_network(grid, ch_cols, top_ep_row, bot_ep_row, phases, band_starts, spacing)

    n_rects = random.choices([0, 1, 2], weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        placed = _try_place_9rect(grid, ch_cols, phases, band_starts, spacing, placed_rects)
        if placed is not None:
            placed_rects.append(placed)

    output = grid.copy()
    for i in range(n):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND_OFF)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X_OFF)

    inp = _derive_input(grid, n, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
