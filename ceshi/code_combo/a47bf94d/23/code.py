# Combo variant of a47bf94d (index 23)
# combo_spec: D4/0+D5/1+D9/2
# Name: 逆向分解+颜色偏移补全+颜色决定分解

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

# Design analysis for the fused variant
# a) 具体化与行为追踪
#    - 我把“颜色决定分解方式”落实成两类源 token 颜色，并且每个样本里都重复出现：
#      color 1 的 solid 会被反向拆成“本地留 X、远端送 diamond”；color 6 的 solid 会被反向拆成
#      “本地留右折 L、远端送左折 L”。这里的“反向”对应 D4：留在原位的是互补半边，而不是原题里
#      送走的那一半。
#    - 一个具体轨迹：若顶部有 color-1 solid，连到某条通道并沿路穿过 2 个 9-rect，那么输出顶部中心
#      变成同色 X；底部远端出现 color 3 的 diamond。若顶部有 color-6 的右折 L，沿路只穿过 1 个
#      9-rect，那么它本地不变，远端补出 color 7 的左折 L。
#    - 相比原题，改动落在三个环节：solid 的原地残留形状变了；不同颜色的 solid / residue 选择不同
#      的“残留 vs 传输”图元对；远端颜色不再等于源颜色，而是等于 `source_color + rect_pass_count`。
# b) 复杂度校准
#    - 为了让“颜色决定分解”在少量样例里可见，我不再要求每条通道颜色互异，而是固定 4 条通道中有
#      两条 color 1、两条 color 6。这样同色规则在单个样例内就有重复支撑。
#    - 为了让“颜色偏移补全”可见，每个样例强制出现四种路径角色：color 1 的 0-rect 与 2-rect 各一条，
#      color 6 的 0-rect 与 1-rect 各一条。这样每个样例都同时包含“不变色”和“变色”。
#    - 只使用两类分解模式，避免把颜色到形状的映射做得过宽；否则人类在 2-4 个训练对里很难归纳。
# c) 隐式假设审查
#    - 原 solver 假设只有 solid 与 X 需要从轨道 mask 中抠除；融合后如果远端预放 diamond 或左折 L，
#      不先移除就会被误当成导线支叉，所以这里显式检测并抠除 diamond / 左折 L / 右折 L。
#    - 原 solver 假设 solid 的原地残留与远端补形固定为 diamond / X；融合后这个前提失效，必须把
#      “源颜色 -> (本地残留, 远端补形)”抽成映射，并在 solid 分支与 residue 分支共用。
#    - 原 solver 默认远端颜色等于源颜色；融合后远端写入必须改成 `col + rect_passes`。为了不碰到
#      线色 5/8/9，generator 只用源颜色 1 和 6，并受控地产生 0/1/2 次穿越。
#    - 原 generator 对 9-rect 的尺寸较自由，但这里需要额外规避一个老隐患：如果 9-rect 恰好是 3x3，
#      `_find_solid_3x3` 会把它误判成 token。因此受控矩形固定成宽 3、高 2 或 4 的单线通行块。
# d) 放置合法性
#    - token 仍然放在 degree-1 端点外 2 格的 3x3 框中；通道列、上下 token 行、交叉行都沿用原题几何，
#      所以不会越界。
#    - 受控 9-rect 只放在确定为纯竖直单线段的带状区域，并做 1 格间隔检查，不与 token、交叉线或其他
#      9-rect 重叠。
#    - 本变体只在顶部放源 token，底部最多预放“已经补完”的目标图元，避免同一端点同时出现两套可被
#      解释为源的结构。
# e) 视觉可观察性
#    - 解题者在单个样例里就能看到：同为 color 1 的两条通道都留下 X，而同为 color 6 的两条通道都
#      留下右折 L；这让“颜色决定分解方式”是肉眼可见的。
#    - 同时还能比较每种颜色的“穿过 0 个 rect”与“穿过若干 rect”的远端颜色差异，推断颜色偏移依赖
#      于 9-rect 数量而不是位置或形状本身。
# f) 可视化整洁度
#    - 新增的 L 形仍然局限在原有 3x3 token 框里，没有引入更大物体；远端补形和本地残留都与导线分开
#      两格，读图时不会和线网混在一起。
#    - 9-rect 仍然是规则矩形，而且只嵌在竖直线段里，便于人眼数出“这条路径经过了几个 maroon 块”。
# g) 变化的可推断性
#    - 颜色变化只发生在远端补形，且始终与“经过几个 9-rect”同步出现，不依赖隐藏排序、坐标奇偶等人类
#      不可视因素。
#    - 源颜色只用两种，并且每种颜色都在样例内重复，避免把“颜色决定分解”退化成四条通道各不相同的孤例。
# 9) 新 main() 的输入假设与 generator 保证
#    - 假设所有 solid 源颜色都来自 {1, 6}；generator 固定使用这两个源颜色。
#    - 假设若出现 residue 源，则 color 1 只会以 X 出现，color 6 只会以右折 L 出现；generator 保证
#      不制造错配的 residue。
#    - 假设远端预放图元只会是目标补形（diamond 或左折 L），不会额外出现可被当作源 residue 的 X /
#      右折 L；generator 只按这个约束预放。
#    - 假设每条活动路径的 rect 计数由 0/1/2 组成，并且不会把远端颜色推到 5/8/9；generator 通过
#      固定源颜色和受控放置矩形来满足这一点。

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_L_LEFT_OFF = [(-1, -1), (0, -1), (1, -1), (1, 0)]
_L_RIGHT_OFF = [(-1, 1), (0, 1), (1, 1), (1, 0)]
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

SOURCE_X_COLOR = 1
SOURCE_L_COLOR = 6

CHANNEL_COUNTS = [4]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
BYPASS_COL_OFFSET = 2
TOKEN_ROW_OPTIONS = [2, 3]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

PREPLACE_REMOTE_PROB = 0.4
GEN_MAX_RETRIES = 300
RECT_MARGIN_GAP = 1
RECT_HEIGHT_OPTIONS = [2, 4]

_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
_DIR2FACE = {(1, 0): "N", (-1, 0): "S", (0, 1): "W", (0, -1): "E"}
_FACE2DIR = {"S": (1, 0), "N": (-1, 0), "E": (0, 1), "W": (0, -1)}


def _source_shape_rule(color: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if color == SOURCE_X_COLOR:
        return _X_OFF, _DIAMOND_OFF
    if color == SOURCE_L_COLOR:
        return _L_RIGHT_OFF, _L_LEFT_OFF
    return None


def _find_solid_3x3(g: np.ndarray, bg: int) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
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


def _find_pattern_3x3(
    g: np.ndarray,
    bg: int,
    offsets: list[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, tuple[int, int]]]:
    h, w = g.shape
    found: dict[tuple[int, int], tuple[int, tuple[int, int]]] = {}
    mask = np.zeros((3, 3), dtype=bool)
    for dr, dc in offsets:
        mask[1 + dr, 1 + dc] = True

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sub = g[r0 : r0 + 3, c0 : c0 + 3]
            vals = {int(sub[rr, cc]) for rr in range(3) for cc in range(3) if mask[rr, cc]}
            if len(vals) != 1:
                continue
            col = vals.pop()
            if col == bg:
                continue
            if np.any(sub[mask] != col) or np.any(sub[~mask] != bg):
                continue

            isolated = True
            cr, cc = r0 + 1, c0 + 1
            for dr, dc in offsets:
                rr, cc2 = cr + dr, cc + dc
                for nr, nc in neighbors4(rr, cc2):
                    if 0 <= nr < h and 0 <= nc < w and not (r0 <= nr < r0 + 3 and c0 <= nc < c0 + 3):
                        if int(g[nr, nc]) == col:
                            isolated = False
                            break
                if not isolated:
                    break
            if isolated:
                found[(cr, cc)] = (col, (r0, c0))
    return found


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=4)]


def _find_9rects(g: np.ndarray) -> list[tuple[set[tuple[int, int]], dict[str, list[tuple[int, int]]]]]:
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
) -> tuple[tuple[int, int], tuple[int, int], int]:
    cur, d, prev = start, dir0, None
    rect_passes = 0
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped")
        visited.add((cur, d))

        jumped = False
        for cells9, faces9 in rects_9:
            if cur not in cells9:
                continue
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
            cur, d, prev = exit_cell, _FACE2DIR[exit_face], None
            rect_passes += 1
            jumped = True
            break
        if jumped:
            continue

        r, c = cur
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d, rect_passes

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        if not nonrev:
            return cur, d, rect_passes

        straight = [o for o in nonrev if (o[1], o[2]) == d]
        if straight:
            p, dr, dc = straight[0]
        else:
            perp = [o for o in nonrev if o[1] * d[0] + o[2] * d[1] == 0]
            if perp:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
            else:
                p, dr, dc = nonrev[0]

        if p == prev:
            return cur, d, rect_passes
        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _stamp_shape(grid: np.ndarray, center: tuple[int, int], color: int, offsets: list[tuple[int, int]]) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[cr + dr, cc + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route endpoint tokens through the wire network, but let the source color choose which reversed split is used.

    Input structure:
    - The dominant background surrounds one orthogonal wire network made of colors 8/5/9, with 3x3 tokens sitting
      two cells away from endpoints. Source tokens are solid 3x3 blocks or already-split residue shapes.
    - Color 1 tokens use the X/diamond family; color 6 tokens use a mirrored-L pair. Some far endpoints may
      already contain the remote target shape from an earlier completion.

    Rule:
    - A solid source token decomposes according to its color, but with the split reversed: the complementary residue
      stays in place, while the other half travels to the opposite endpoint.
    - A residue source token sends the same remote half as its solid counterpart would. The transmitted shape is
      recolored to `source_color + number_of_9_rectangles_crossed`.

    Key insight:
    - The wire tracing itself is unchanged; the task is to infer two coupled controls: source color decides which
      residue/complement pair is active, and 9-rect count decides how much the remote color is shifted.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_pattern_3x3(g, bg, _X_OFF)
    diamonds = _find_pattern_3x3(g, bg, _DIAMOND_OFF)
    l_lefts = _find_pattern_3x3(g, bg, _L_LEFT_OFF)
    l_rights = _find_pattern_3x3(g, bg, _L_RIGHT_OFF)

    track_mask = g != bg
    for _, (_, (r0, c0)) in solids.items():
        track_mask[r0 : r0 + 3, c0 : c0 + 3] = False
    for found, offsets in (
        (xs, _X_OFF),
        (diamonds, _DIAMOND_OFF),
        (l_lefts, _L_LEFT_OFF),
        (l_rights, _L_RIGHT_OFF),
    ):
        for (cr, cc) in found.keys():
            for dr, dc in offsets:
                track_mask[cr + dr, cc + dc] = False

    comps = _components(track_mask)
    rects_9 = _find_9rects(g)
    out = g.copy()

    for (cr, cc), (col, (r0, c0)) in solids.items():
        rule = _source_shape_rule(col)
        if rule is None:
            continue
        local_offsets, _remote_offsets = rule
        out[r0 : r0 + 3, c0 : c0 + 3] = bg
        _stamp_shape(out, (cr, cc), col, local_offsets)

    for pts in comps:
        comp = set(pts)
        if len(comp) < 4:
            continue

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

                col = None
                remote_offsets = None
                if cen in solids:
                    col, _ = solids[cen]
                    rule = _source_shape_rule(col)
                    if rule is not None:
                        _local_offsets, remote_offsets = rule
                elif cen in xs:
                    col, _ = xs[cen]
                    rule = _source_shape_rule(col)
                    if rule is not None and rule[0] == _X_OFF:
                        _local_offsets, remote_offsets = rule
                elif cen in l_rights:
                    col, _ = l_rights[cen]
                    rule = _source_shape_rule(col)
                    if rule is not None and rule[0] == _L_RIGHT_OFF:
                        _local_offsets, remote_offsets = rule

                if col is None or remote_offsets is None:
                    continue

                end, dd, rect_passes = _trace_wire(comp, ep, (-odr, -odc), h, w, rects_9)
                dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                shifted = col + rect_passes
                _stamp_shape(out, dest, shifted, remote_offsets)

    return out


def _decompose_to_crossovers(perm: list[int]) -> list[tuple[int, int]]:
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
        grid[xrow, c_min : c_max + 1] = 8
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
        if slot_wire[j] is None:
            continue
        c = ch_cols[j]
        if c in col_seg_start:
            start = col_seg_start[c] + 1
            if start <= bot_ep_row:
                grid[start : bot_ep_row + 1, c] = 8

    return slot_wire, bypass_wire


def _place_vertical_rect(
    grid: np.ndarray,
    col: int,
    row_lo: int,
    row_hi: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    for _ in range(30):
        rect_h = random.choice(RECT_HEIGHT_OPTIONS)
        if row_hi - row_lo + 1 < rect_h:
            return None
        r0 = random.randint(row_lo, row_hi - rect_h + 1)
        r1 = r0 + rect_h - 1
        c0, c1 = col - 1, col + 1
        if r0 <= 0 or r1 >= h - 1 or c0 < 0 or c1 >= w:
            continue
        if grid[r0 - 1, col] != 8 or grid[r1 + 1, col] != 8:
            continue
        if not np.all(grid[r0 : r1 + 1, col] == 8):
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((sub == 0) | (sub == 5) | (sub == 8)):
            continue

        clash = False
        for pr0, pc0, pr1, pc1 in placed_rects:
            if r0 <= pr1 + RECT_MARGIN_GAP and pr0 <= r1 + RECT_MARGIN_GAP and c0 <= pc1 + RECT_MARGIN_GAP and pc0 <= c1 + RECT_MARGIN_GAP:
                clash = True
                break
        if clash:
            continue

        grid[r0 : r1 + 1, c0 : c1 + 1] = 9
        rect = (r0, c0, r1, c1)
        placed_rects.append(rect)
        return rect
    return None


def _trace_channel_info(grid: np.ndarray, ch_cols: list[int], top_ep_row: int) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rects_9 = _find_9rects(g)
    comps = _components(g != 0)
    dest_cols = []
    rect_counts = []
    for c in ch_cols:
        start = (top_ep_row, c)
        comp = None
        for pts in comps:
            cand = set(pts)
            if start in cand:
                comp = cand
                break
        assert comp is not None
        end, _dd, rect_passes = _trace_wire(comp, start, (1, 0), h, w, rects_9)
        dest_cols.append(end[1])
        rect_counts.append(rect_passes)
    return dest_cols, rect_counts


def _build_expected_output(
    grid: np.ndarray,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    rect_counts: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    out = grid.copy()
    for i, col in enumerate(token_colors):
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        local_offsets, remote_offsets = _source_shape_rule(col)
        _stamp_shape(out, top_center, col, local_offsets)
        _stamp_shape(out, bot_center, col + rect_counts[i], remote_offsets)
    return out


def _derive_input(
    grid: np.ndarray,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    rect_counts: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> tuple[np.ndarray, dict[str, object]]:
    inp = grid.copy()
    n = len(ch_cols)
    color_to_indices = {
        SOURCE_X_COLOR: [i for i, col in enumerate(token_colors) if col == SOURCE_X_COLOR],
        SOURCE_L_COLOR: [i for i, col in enumerate(token_colors) if col == SOURCE_L_COLOR],
    }
    residue_indices = {
        SOURCE_X_COLOR: random.choice(color_to_indices[SOURCE_X_COLOR]),
        SOURCE_L_COLOR: random.choice(color_to_indices[SOURCE_L_COLOR]),
    }
    preplaced_channels: list[int] = []

    for i in range(n):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])
        local_offsets, remote_offsets = _source_shape_rule(col)

        if residue_indices[col] == i:
            _stamp_shape(inp, top_center, col, local_offsets)
            if rect_counts[i] > 0 and random.random() < PREPLACE_REMOTE_PROB:
                _stamp_shape(inp, bot_center, col + rect_counts[i], remote_offsets)
                preplaced_channels.append(i)
        else:
            _stamp_shape(inp, top_center, col, _SOLID_OFF)

    meta = {
        "residue_channels": residue_indices,
        "preplaced_channels": preplaced_channels,
        "rect_counts": rect_counts,
        "token_colors": token_colors,
    }
    return inp, meta


def _gen_attempt(return_meta: bool = False):
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

    while True:
        perm = list(range(n))
        random.shuffle(perm)
        if perm != list(range(n)):
            break

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

    top_band = (top_ep_row + 1, xover_rows[0] - 1)
    bot_band = (xover_rows[-1] + 1, bot_ep_row - 1)
    if top_band[1] - top_band[0] + 1 < min(RECT_HEIGHT_OPTIONS):
        return None
    if bot_band[1] - bot_band[0] + 1 < min(RECT_HEIGHT_OPTIONS):
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

    token_colors = [SOURCE_X_COLOR, SOURCE_X_COLOR, SOURCE_L_COLOR, SOURCE_L_COLOR]
    random.shuffle(token_colors)
    x_channels = [i for i, col in enumerate(token_colors) if col == SOURCE_X_COLOR]
    l_channels = [i for i, col in enumerate(token_colors) if col == SOURCE_L_COLOR]
    double_src = random.choice(x_channels)
    single_src = random.choice(l_channels)

    placed_rects: list[tuple[int, int, int, int]] = []
    if _place_vertical_rect(grid, ch_cols[double_src], top_band[0], top_band[1], placed_rects) is None:
        return None
    if _place_vertical_rect(grid, ch_cols[perm[double_src]], bot_band[0], bot_band[1], placed_rects) is None:
        return None
    if _place_vertical_rect(grid, ch_cols[single_src], top_band[0], top_band[1], placed_rects) is None:
        return None

    traced_dest_cols, traced_counts = _trace_channel_info(grid, ch_cols, top_ep_row)
    if traced_dest_cols != [ch_cols[p] for p in perm]:
        return None

    expected_counts = [0] * n
    expected_counts[double_src] = 2
    expected_counts[single_src] = 1
    if traced_counts != expected_counts:
        return None

    expected = _build_expected_output(grid, ch_cols, perm, token_colors, expected_counts, top_token_row, bot_token_row)
    inp, meta = _derive_input(grid, ch_cols, perm, token_colors, expected_counts, top_token_row, bot_token_row)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None

    if return_meta:
        meta.update(
            {
                "perm": perm,
                "expected_counts": expected_counts,
                "double_src": double_src,
                "single_src": single_src,
                "top_token_row": top_token_row,
                "bot_token_row": bot_token_row,
                "ch_cols": ch_cols,
            }
        )
        return inp, meta
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
