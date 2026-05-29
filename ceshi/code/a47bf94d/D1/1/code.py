# Variant of a47bf94d
# Dimension: D1 — 管道路径追踪方式 (path)
# Change: 管道追踪（pipeline_trace）——沿青色(color 8)正交管道线段从一个端点追踪到另一个端点，遇到color-9矩形块时穿过
#         → 对角线混合管道：主干仍可含正交段，但换轨区改为45度对角线段，求解时需要在8邻接图上沿“直行优先”的混合路径追踪。

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# -----------------------------------------------------------------------------
# Design analysis for the D1 variant
#
# concrete_behavior:
# - 仍然保留原题的 token 语义：solid 3x3 在原地裂解成 diamond；solid/X 都会把互补形状送到同色管道另一端。
# - 可见变化集中在“管道怎么连”：不再使用纯正交的水平 crossover，而是使用两条相邻竖管之间的 45 度对角交换带。
#   每个交换带跨越一对相邻列：左线斜着到右列，右线斜着到左列，在带中央形成一个 X 形交点。
# - 这样，输入里会同时出现竖直 8 管段、对角 8 管段、以及嵌在竖直区段上的 9 矩形块。输出里的 token 放置规则不变。
#
# behavior_trace:
# - 原逻辑在 crossover 行做一条水平线；trace 只看 4 邻接，靠“直行优先 + 5 标记”避免在十字交叉处转错。
# - 新逻辑在 swap band 中画两条对角线；trace 改为看 8 邻接，并在 X 交点继续沿当前方向直行。
#   例如列 c 与 c+4 的相邻两条线在 band 顶部从 (r,c)/(r,c+4) 出发，5 行后到达 (r+4,c+4)/(r+4,c)。
#   中点只有一个共享格，进入方向若是 (1,1) 就继续到 (1,1)，若是 (1,-1) 就继续到 (1,-1)。
# - 差异的本质是：原题的“继续直走”发生在正交十字口；此变体的“继续直走”发生在对角 X 形交点。
#
# complexity_calibration:
# - 如果让任意 diagonal 与 vertical 在同一 band 中相交，会引入额外歧义；因此 generator 只允许“相邻列交换”，
#   且同一 band 内只做互不相邻的若干对交换，避免需要新的隐藏标记。
# - 两个 band 足以产生多条对角段和非恒等 permutation，同时仍能放下 9-rect 和 token，视觉上能明显看出不是原题。
#
# implicit_assumption_audit:
# - 原 main() 默认 track 用 4 邻接分量；这里必须改成 8 邻接，否则对角段会被拆断。
# - 原 endpoint→token 的搜索只枚举 4 个方向；这里虽然 token 仍布在上下两侧，但 solver 也要支持 8 个方向，
#   否则若后续有对角端点布局会直接漏检。
# - 原 _trace_wire() 的“perpendicular fallback”基于点积为 0 的 4 向转弯；这里改成 8 向的最大点积转向，
#   才能从竖直段顺滑进入对角段，也能在对角段末端回到竖直段。
# - 9-rect 的穿越规则仍依赖“从 N/S/E/W 某一面进，从对面出”；因此 generator 只把 9-rect 放在竖直区段，
#   保证进入矩形时方向仍是正交的，不会触发新的未定义情形。
#
# placement_legality:
# - band 之间留出纯竖直空档，9-rect 只放在这些空档里，且要求矩形覆盖的每根竖线在矩形上下都有 8 作为入口/出口。
# - token 仍与端点保持 2 格距离，所以 solver 的 destination = endpoint + 2 * dir 仍然成立。
# - 预放置的 far diamond 仍允许贴在端点外侧；这会把该端 degree 从 1 提高到 2。solver 依靠“source 端仍是 degree-1，
#   且该颜色已经出现在 component 里时不再重复发射”这一机制保持正确。
#
# observability_and_cleanliness:
# - 交换带是整行可见的斜线，而不是内部状态变化；2-3 个样例足以看出“线在斜着换列”。
# - 没有引入基于颜色 id 的排序或分支；颜色只标识 token 身份，不参与路由决策。
# - 同一 band 内只出现不相交的相邻交换，视觉上是一组清晰的 X，而不是大面积杂乱的折线网。
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
    Follow an 8-connected pipe with 9-rect pass-through.

    At an X-shaped diagonal crossing, continue in the current direction if possible.
    At a vertical/diagonal joint, choose the non-reversing neighbor with the largest
    dot product against the current direction.
    """
    cur, d, prev = start, dir0, None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 8):
        state = (cur, d)
        if state in visited:
            raise AssertionError("wire trace looped in diagonal pipeline")
        visited.add(state)

        jumped = False
        for cells9, faces9 in rects_9:
            if cur in cells9:
                if d not in _DIR2FACE:
                    raise AssertionError("diagonal entry into 9-rect is unsupported in this variant")
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
    Route colored tokens through mixed orthogonal/diagonal pipes and emit their complementary 3x3 forms.

    Input structure:
    - The background is the most common color. Non-background cells form cyan pipe networks, optional maroon
      rectangles embedded in straight pipe runs, and colored 3x3 tokens placed two steps outside pipe endpoints.
    - Pipes are now 8-connected: long vertical stretches are linked by 45-degree diagonal swap bands.

    Rule:
    - Every solid 3x3 block collapses in place into a diamond (the four cardinal cells around its center).
    - Any solid or X token at one pipe endpoint sends its complementary shape to the opposite endpoint:
      solid emits X, X emits diamond. Existing diamonds stay as they are.
    - Maroon rectangles still act as pass-through chambers: a pipe entering on one face exits from the opposite face
      (or the unique other face when the rectangle encodes a turn).

    Key insight:
    - The token logic is unchanged; the new challenge is following the pipe itself. The correct far endpoint is no
      longer recoverable by 4-neighbor tracing, because swap bands use diagonal X-crossings that must be traversed
      by continuing in the current direction across an 8-connected path.
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


# --------------- generator (diagonal swap-band variant) ---------------

CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

BAND_COUNT = 2
BAND_START_OFFSET = 1
BAND_GAP = 3

RECT_COUNT_WEIGHTS = [20, 45, 25, 10]
RECT_WIRE_COUNT_WEIGHTS = [55, 35, 10]
RECT_H_RANGE = (2, 4)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 8
RECT_MARGIN_GAP = 1

X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

GEN_MAX_RETRIES = 400


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _vertical_windows(top_ep_row: int, bot_ep_row: int, band_starts: list[int], spacing: int) -> list[tuple[int, int]]:
    windows = []
    first_start = band_starts[0]
    if first_start - 2 >= top_ep_row:
        windows.append((top_ep_row, first_start - 2))
    for i in range(len(band_starts) - 1):
        prev_end = band_starts[i] + spacing
        next_start = band_starts[i + 1]
        if next_start - 2 >= prev_end + 2:
            windows.append((prev_end + 2, next_start - 2))
    last_end = band_starts[-1] + spacing
    if bot_ep_row >= last_end + 2:
        windows.append((last_end + 2, bot_ep_row))
    return windows


def _choose_phases(n: int) -> tuple[list[list[int]], list[int]]:
    while True:
        start_parity = random.choice([0, 1])
        parities = [start_parity, 1 - start_parity]
        phases: list[list[int]] = []
        order = list(range(n))
        for parity in parities:
            eligible = list(range(parity, n - 1, 2))
            picks = [j for j in eligible if random.random() < 0.75]
            if not picks:
                picks = [random.choice(eligible)]
            phases.append(sorted(picks))
            for j in picks:
                order[j], order[j + 1] = order[j + 1], order[j]

        perm = [0] * n
        for slot, wire_id in enumerate(order):
            perm[wire_id] = slot
        moved = sum(i != perm[i] for i in range(n))
        if moved >= 2:
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


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    windows: list[tuple[int, int]],
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    target_wires = random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0]
    viable_windows = [win for win in windows if win[1] - win[0] + 1 >= RECT_H_RANGE[0]]
    if not viable_windows:
        return None
    r_lo, r_hi = random.choice(viable_windows)

    wire_span = min(target_wires, len(ch_cols))
    j0 = random.randint(0, len(ch_cols) - wire_span)
    wire_cols = ch_cols[j0:j0 + wire_span]

    rect_h = random.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], r_hi - r_lo + 1))
    rect_r0 = random.randint(r_lo, r_hi - rect_h + 1)
    rect_r1 = rect_r0 + rect_h - 1

    margin_l = random.randint(0, 1 if wire_span > 1 else 2)
    margin_r = random.randint(0, 1 if wire_span > 1 else 2)
    rect_c0 = wire_cols[0] - margin_l
    rect_c1 = wire_cols[-1] + margin_r
    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
        return None

    for c in wire_cols:
        if not np.all(grid[rect_r0:rect_r1 + 1, c] == 8):
            return None
        if rect_r0 - 1 < 0 or rect_r1 + 1 >= h:
            return None
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
            return None

    sub = grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 8)):
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    grid[rect_r0:rect_r1 + 1, rect_c0:rect_c1 + 1] = 9
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
            # To match the fixed output orientation (diamond on top, X on bottom),
            # the active X must start at the bottom endpoint.
            _stamp_token(inp, bot_center[0], bot_center[1], col, _X_OFF)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, top_center[0], top_center[1], col, _DIAMOND_OFF)
        else:
            # Likewise, a solid source must begin on the top side so it collapses
            # there and emits its X to the bottom side.
            _stamp_token(inp, top_center[0], top_center[1], col, _SOLID_OFF)

    return inp


def _gen_attempt():
    n = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)
    top_token_row = random.choice(TOKEN_ROW_OPTIONS)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n)]
    w = max(ch_cols[-1] + 3, GRID_W_MIN)
    if w > GRID_W_MAX:
        return None

    top_ep_row = top_token_row + EP_TOKEN_GAP
    band_starts = [top_ep_row + BAND_START_OFFSET + i * (spacing + BAND_GAP) for i in range(BAND_COUNT)]
    last_band_end = band_starts[-1] + spacing
    h_min = last_band_end + top_token_row + EP_TOKEN_GAP + 2
    if h_min > GRID_H_RANGE[1]:
        return None
    h = random.randint(max(GRID_H_RANGE[0], h_min, 24), GRID_H_RANGE[1])

    bot_token_row = h - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row <= last_band_end:
        return None

    token_colors = random.sample(TOKEN_COLORS_POOL, n)
    phases, perm = _choose_phases(n)

    grid = np.zeros((h, w), dtype=int)
    _build_wire_network(grid, ch_cols, top_ep_row, bot_ep_row, phases, band_starts, spacing)

    windows = _vertical_windows(top_ep_row, bot_ep_row, band_starts, spacing)
    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        placed = _try_place_9rect(grid, ch_cols, windows, placed_rects)
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
