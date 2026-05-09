# Combo variant of a47bf94d (index 15)
# combo_spec: D2/1+D19/2
# Name: 无标记优先规则+标记位置偏移
#
# Design analysis for combo D2/1 + D19/2
# =======================================
# a) 具体化与行为追踪
# - 原题里，真正的十字交叉会在穿行方向的前后各 1 格放一对灰色 5，显式提示“直行不要转弯”。
# - D2/1 去掉这些近邻标记后，裸露的 8 色 '+' 只能靠运动方向来解读：若前方仍有线段，就继续直行。
# - 本融合进一步把少量灰色 5 保留下来，但不再放在交叉旁边；它们改为出现在需要离开竖直主干的 T 形结点同列、前后 2 或 3 格。
# - 一个具体小例子：某条交换横线从第 1 列通道连到第 4 列通道。第 1 列和第 4 列的结点本地都同时有“直行向下”和“拐入横线”两种可能。
#   原题：靠紧邻结点的 5 提示穿行直走；当前融合：大多数内部 '+' 没任何标记且必须直走，但交换横线两端的竖直主干在远处各有一对 5，
#   说明这里不是继续直落，而是离开主干拐入或拐出横线。内部被横线跨过的中间通道仍是裸 '+'，因此依然按直行优先穿过。
# - 差异：输入里同时存在“无标记直行”的裸 '+' 与“远距成对 5 触发转弯”的少数结点，两者必须协同才能恢复通道置换。
#
# b) 复杂度校准
# - 保留 4-5 条通道、非恒等置换、0-3 个 9 号矩形宽接合体，避免图像退化成单次简单弯折。
# - 每个样例强制至少有一条跨越至少 3 个通道中心的交换横线，从而保证至少出现一个裸露的内部 '+'，否则 D2/1 不可观察。
# - 每条交换横线只在两个端点放偏移标记；标记类型保持单一，避免出现多种 5 编码让少量样例无法归纳。
#
# c) 隐式假设审查
# - `_trace_wire` 原先默认“能直走就直走”，这在裸 '+' 仍然成立；但现在必须先检测“当前是被偏移标记包夹的结点且只有一个可转入的侧枝”，
#   否则会错过应当从竖直主干拐入横线的位置。
# - `track_mask` 不能再把 5 当成噪声，因为偏移标记本身仍落在线路上，只是颜色改为 5；因此线路颜色集合恢复为 {5,8,9}。
# - `_find_9rects` 必须把 5/8 都视作矩形外侧的有效连线接触色，因为矩形边上可能正好贴着偏移标记。
# - 生成器不能再依赖“source 列下方消失、dst 列上方消失”的原始 crossover 几何；本实现改成所有竖直主干全程保留，再用带标记的横线端点完成交换。
#
# d) 放置合法性
# - token 仍与端点相隔 2 格，通道间距仍为 4-5，因此 3x3 token 不会互相重叠。
# - 交换行只放在 token 区之外；每个结点的偏移距离只允许 2 或 3，且其两端标记不能落到另一条交换行上。
# - 9 号矩形避开所有交换行与标记单元，只覆盖纯 8 线段，因此不会吞掉关键线索，也不会改变转弯结点的局部几何。
#
# e) 视觉可观察性
# - 人类在 2-4 个样例里可以看到：同样的横向跨接结构里，中间的裸 '+' 没有任何灰色辅助，说明它们按默认直行；只有少数列的远距成对 5
#   稳定地围住某个竖直结点，且这些结点恰好对应 token 真实发生换道的位置。
# - 标记距离只取 2 或 3 两档，重复出现足够多，便于从少量样例中归纳“灰色不是普通线，而是远处的转弯提示”。
#
# f) 可视化整洁度
# - 交换横线只在单独若干行出现，标记仅沿其两端列分布，避免同一区域既有矩形、又有近邻标记、又有多层横线叠压。
# - 9 号矩形不覆盖交换行，所以“裸 '+' 负责直行、远距 5 负责转弯”的视觉语义不会被 9 号块打乱。
#
# g) 变化的可推断性
# - 本融合不引入新的颜色映射；颜色仍仅用于配对 token。
# - 新增规则完全由结构线索决定：有没有灰色、灰色离结点多远、灰色是否成对包夹竖直主干，均可直接目视。

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_VERT = [(1, 0), (-1, 0)]
_HORIZ = [(0, 1), (0, -1)]
_WIRE_COLORS = {5, 8, 9}

_DIAMOND_OFF = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X_OFF = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]


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
    """Find color-9 rectangular blocks and record which wire cells touch each face."""
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


def _has_offset_turn_marker(g: np.ndarray, cell: tuple[int, int], d: tuple[int, int]) -> bool:
    """A turn instruction exists when the current axis has matching 5-cells at distance 2 or 3."""
    r, c = cell
    for k in (2, 3):
        r1, c1 = r - k * d[0], c - k * d[1]
        r2, c2 = r + k * d[0], c + k * d[1]
        if (
            0 <= r1 < g.shape[0]
            and 0 <= c1 < g.shape[1]
            and 0 <= r2 < g.shape[0]
            and 0 <= c2 < g.shape[1]
            and int(g[r1, c1]) == 5
            and int(g[r2, c2]) == 5
        ):
            return True
    return False


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
    g: np.ndarray,
    use_offset_turns: bool,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Follow one wire component with straight-priority unless a marked T-junction overrides it.

    - Unmarked '+' crossings: continue straight whenever possible.
    - Marked T-junctions: if the current axis is bracketed by offset 5-cells and there is exactly one side branch,
      leave the main axis and take that turn instead of continuing straight.
    - Color-9 rectangles still act as wide pass-through junctions.
    """
    cur, d, prev = start, dir0, None
    carry_dir: tuple[int, int] | None = None
    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for _ in range(h * w * 4):
        if (cur, d) in visited:
            raise AssertionError("wire trace looped unexpectedly")
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
        opts = [((nr, nc), nr - r, nc - c) for nr, nc in neighbors4(r, c) if (nr, nc) in comp]
        if not opts:
            return cur, d

        rev = (-d[0], -d[1])
        nonrev = [o for o in opts if (o[1], o[2]) != rev]
        opts2 = nonrev if nonrev else opts

        straight = [o for o in opts2 if (o[1], o[2]) == d]
        perp = [o for o in opts2 if o[1] * d[0] + o[2] * d[1] == 0]

        if use_offset_turns and straight and len(perp) == 1 and _has_offset_turn_marker(g, cur, d):
            p, dr, dc = perp[0]
            carry_dir = d
        elif straight:
            p, dr, dc = straight[0]
        elif carry_dir is not None and len(perp) == 2:
            match = [o for o in perp if (o[1], o[2]) == carry_dir]
            if match:
                p, dr, dc = match[0]
                carry_dir = None
            else:
                p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
                carry_dir = None
        elif perp:
            p, dr, dc = sorted(perp, key=lambda x: (x[1], x[2]))[0]
        else:
            p, dr, dc = opts2[0]

        if p == prev:
            return cur, d

        prev = cur
        cur = p
        d = (dr, dc)

    raise AssertionError("wire trace exceeded step limit")


def _solve(grid: np.ndarray, *, use_offset_turns: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    solids = _find_solid_3x3(g, bg)
    xs = _find_x_3x3(g, bg)

    track_mask = np.isin(g, list(_WIRE_COLORS))
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
                    dir_in = (-odr, -odc)
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9, g, use_offset_turns)
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
                    end, dd = _trace_wire(comp, ep, dir_in, h, w, rects_9, g, use_offset_turns)
                    dest = (end[0] + 2 * dd[0], end[1] + 2 * dd[1])
                    for dr, dc in _DIAMOND_OFF:
                        rr, cc2 = dest[0] + dr, dest[1] + dc
                        if 0 <= rr < h and 0 <= cc2 < w:
                            out[rr, cc2] = col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route colored 3x3 tokens through a mostly unmarked wire network and rebuild the complementary token at the far endpoint.

    Input structure:
    - The dominant color is background. Other cells form teal/grey wire paths, optional maroon rectangular pass-through blocks, and colored 3x3 tokens placed two cells away from degree-1 wire endpoints.
    - Most crossings are plain cyan '+' junctions with no nearby marker cells. A small number of vertical stem cells instead have a matched pair of grey 5-cells placed 2 or 3 steps above and below the branch point.

    Rule:
    - A solid 3x3 block collapses in place into a diamond on its four cardinal neighbors and emits an X to the remote endpoint.
    - An X token stays in place and emits a diamond to the remote endpoint.
    - While tracing the wire, bare intersections obey straight-ahead priority. The only exception is a junction whose current travel axis is bracketed by one of those offset grey pairs: there, leave the vertical stem and take the unique side branch.

    Key insight:
    - Grey 5-cells are no longer local “go straight” brackets. Instead, the absence of markers means “use default straight-through behavior,” and the rare far-away symmetric 5 pair marks the specific stem where that default must be overridden into a turn.
    """
    return _solve(grid, use_offset_turns=True)


def _solve_without_offset_turns(grid: np.ndarray) -> np.ndarray:
    """Diagnostic helper used to confirm generated samples do not collapse back to the pure D2/1 behavior."""
    return _solve(grid, use_offset_turns=False)


# --- Token shape offsets (relative to 3x3 center) ---
_DIAMOND = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_X = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]
_SOLID = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

# --- Layout configuration ---
CHANNEL_COUNTS = [4, 5]
CHANNEL_SPACING = [4, 5]
LEFT_MARGIN_RANGE = (2, 3)
TOKEN_ROW_OPTIONS = [2, 3]
TOKEN_COLORS_POOL = [1, 2, 3, 4, 6, 7]
GRID_H_RANGE = (20, 26)
GRID_W_MIN = 22
GRID_W_MAX = 26
EP_TOKEN_GAP = 2

# --- Swap-row / marker configuration ---
MIN_SWAP_ROW_GAP = 3
MARKER_OFFSET_CHOICES = [2, 3]

# --- 9-rect placement ---
RECT_COUNT_WEIGHTS = [10, 40, 30, 20]
RECT_WIRE_COUNT_WEIGHTS = [50, 38, 12]
RECT_H_RANGE = (2, 5)
RECT_MAX_WIDTH = 7
RECT_RETRY_FACTOR = 5
RECT_MARGIN_GAP = 1

# --- Input derivation ---
X_INPUT_COUNT_WEIGHTS = [60, 30, 10]
FAR_DIAMOND_PROB = 0.33

# --- Generation retry ---
GEN_MAX_RETRIES = 400


def _stamp_token(grid: np.ndarray, center_r: int, center_c: int, color: int, offsets: list[tuple[int, int]]) -> None:
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = color


def _apply_swaps(n_channels: int, swaps: list[tuple[int, int]]) -> list[int]:
    cur = list(range(n_channels))
    for a, b in swaps:
        cur[a], cur[b] = cur[b], cur[a]
    return cur


def _decompose_to_swaps(final_slots: list[int]) -> list[tuple[int, int]]:
    """Transform [0..N-1] into final_slots using direct transpositions."""
    cur = list(range(len(final_slots)))
    swaps: list[tuple[int, int]] = []
    for j, want in enumerate(final_slots):
        if cur[j] == want:
            continue
        k = cur.index(want, j + 1)
        swaps.append((j, k))
        cur[j], cur[k] = cur[k], cur[j]
    return swaps


def _choose_swap_rows(top_ep_row: int, bot_ep_row: int, n_swaps: int) -> tuple[list[int], list[int]] | tuple[None, None]:
    candidates = [r for r in range(top_ep_row + 3, bot_ep_row - 2)]
    if len(candidates) < n_swaps:
        return None, None
    for _ in range(200):
        rows = sorted(random.sample(candidates, n_swaps))
        if any(b - a < MIN_SWAP_ROW_GAP for a, b in zip(rows, rows[1:])):
            continue
        rowset = set(rows)
        feasible_per_row: list[list[int]] = []
        ok = True
        for r in rows:
            feasible = [
                d
                for d in MARKER_OFFSET_CHOICES
                if top_ep_row + 1 <= r - d
                and r + d <= bot_ep_row - 1
                and (r - d) not in rowset
                and (r + d) not in rowset
            ]
            if not feasible:
                ok = False
                break
            feasible_per_row.append(feasible)
        if ok:
            # Keep distance-3 markers visible often enough that the shifted-placement rule
            # can be inferred from a small sample set rather than appearing as a rare edge case.
            need_three = any(3 in feasible for feasible in feasible_per_row)
            dists = []
            used_three = False
            for feasible in feasible_per_row:
                if 3 in feasible and 2 in feasible:
                    d = random.choices([3, 2], weights=[3, 2])[0]
                elif 3 in feasible:
                    d = 3
                else:
                    d = 2
                used_three |= d == 3
                dists.append(d)
            if need_three and not used_three:
                continue
            return rows, dists
    return None, None


def _reserved_swap_cells(
    ch_cols: list[int],
    swaps: list[tuple[int, int]],
    swap_rows: list[int],
    marker_dists: list[int],
) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for (a, b), row, dist in zip(swaps, swap_rows, marker_dists):
        c0, c1 = sorted((ch_cols[a], ch_cols[b]))
        for c in range(c0, c1 + 1):
            reserved.add((row, c))
        for slot in (a, b):
            c = ch_cols[slot]
            reserved.add((row - dist, c))
            reserved.add((row + dist, c))
    return reserved


def _build_wire_network(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    swaps: list[tuple[int, int]],
    swap_rows: list[int],
) -> None:
    for c in ch_cols:
        grid[top_ep_row : bot_ep_row + 1, c] = 8
    for (a, b), row in zip(swaps, swap_rows):
        c0, c1 = sorted((ch_cols[a], ch_cols[b]))
        grid[row, c0 : c1 + 1] = 8


def _place_offset_markers(
    grid: np.ndarray,
    ch_cols: list[int],
    swaps: list[tuple[int, int]],
    swap_rows: list[int],
    marker_dists: list[int],
) -> None:
    for (a, b), row, dist in zip(swaps, swap_rows, marker_dists):
        for slot in (a, b):
            c = ch_cols[slot]
            grid[row - dist, c] = 5
            grid[row + dist, c] = 5


def _count_bare_crossings(grid: np.ndarray) -> int:
    h, w = grid.shape
    total = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != 8:
                continue
            if (
                int(grid[r - 1, c]) in _WIRE_COLORS
                and int(grid[r + 1, c]) in _WIRE_COLORS
                and int(grid[r, c - 1]) in _WIRE_COLORS
                and int(grid[r, c + 1]) in _WIRE_COLORS
            ):
                total += 1
    return total


def _count_marked_turn_sites(grid: np.ndarray) -> int:
    h, w = grid.shape
    total = 0
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) not in (5, 8):
                continue
            if int(grid[r - 1, c]) not in _WIRE_COLORS or int(grid[r + 1, c]) not in _WIRE_COLORS:
                continue
            horiz = [int(grid[r, c - 1]) in _WIRE_COLORS, int(grid[r, c + 1]) in _WIRE_COLORS]
            if sum(horiz) != 1:
                continue
            if _has_offset_turn_marker(grid, (r, c), (1, 0)):
                total += 1
    return total


def _try_place_9rect(
    grid: np.ndarray,
    ch_cols: list[int],
    top_ep_row: int,
    bot_ep_row: int,
    reserved: set[tuple[int, int]],
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    n_channels = len(ch_cols)
    wire_count = min(random.choices([1, 2, 3], weights=RECT_WIRE_COUNT_WEIGHTS)[0], n_channels)

    if wire_count == 1:
        j0 = random.randint(0, n_channels - 1)
        c_lo = c_hi = ch_cols[j0]
        margin_l, margin_r = random.randint(0, 2), random.randint(0, 2)
        if margin_l + margin_r < 1:
            margin_l = 1
    else:
        j0 = random.randint(0, n_channels - wire_count)
        c_lo, c_hi = ch_cols[j0], ch_cols[j0 + wire_count - 1]
        margin_l, margin_r = random.randint(0, 1), random.randint(0, 1)

    rect_c0, rect_c1 = c_lo - margin_l, c_hi + margin_r
    if rect_c0 < 0 or rect_c1 >= w or rect_c1 - rect_c0 + 1 > RECT_MAX_WIDTH:
        return None

    max_h = min(RECT_H_RANGE[1], bot_ep_row - top_ep_row - 3)
    if max_h < RECT_H_RANGE[0]:
        return None
    rect_h = random.randint(RECT_H_RANGE[0], max_h)
    rect_r0 = random.randint(top_ep_row + 1, bot_ep_row - rect_h)
    rect_r1 = rect_r0 + rect_h - 1

    wire_cols = [c for c in ch_cols if rect_c0 <= c <= rect_c1]
    if len(wire_cols) != wire_count:
        return None

    gap = RECT_MARGIN_GAP
    for pr0, pc0, pr1, pc1 in placed_rects:
        if rect_r0 <= pr1 + gap and pr0 <= rect_r1 + gap and rect_c0 <= pc1 + gap and pc0 <= rect_c1 + gap:
            return None

    for rr in range(rect_r0, rect_r1 + 1):
        for cc in range(rect_c0, rect_c1 + 1):
            if (rr, cc) in reserved:
                return None

    for c in wire_cols:
        if not np.all(grid[rect_r0 : rect_r1 + 1, c] == 8):
            return None
        if grid[rect_r0 - 1, c] != 8 or grid[rect_r1 + 1, c] != 8:
            return None

    sub = grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
    if not np.all((sub == 0) | (sub == 8)):
        return None

    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = 9
    return rect_r0, rect_c0, rect_r1, rect_c1


def _derive_input(
    grid: np.ndarray,
    n_channels: int,
    ch_cols: list[int],
    perm: list[int],
    token_colors: list[int],
    top_token_row: int,
    bot_token_row: int,
) -> np.ndarray:
    inp = grid.copy()
    n_x = random.choices([0, 1, 2], weights=X_INPUT_COUNT_WEIGHTS)[0]
    x_channels = set(random.sample(range(n_channels), min(n_x, n_channels)))

    for i in range(n_channels):
        col = token_colors[i]
        top_center = (top_token_row, ch_cols[i])
        bot_center = (bot_token_row, ch_cols[perm[i]])

        if i in x_channels:
            if random.choice([True, False]):
                near, far = top_center, bot_center
            else:
                near, far = bot_center, top_center
            _stamp_token(inp, near[0], near[1], col, _X)
            if random.random() < FAR_DIAMOND_PROB:
                _stamp_token(inp, far[0], far[1], col, _DIAMOND)
        else:
            side = random.choice([top_center, bot_center])
            _stamp_token(inp, side[0], side[1], col, _SOLID)
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_channels = random.choice(CHANNEL_COUNTS)
    spacing = random.choice(CHANNEL_SPACING)

    left_margin = random.randint(*LEFT_MARGIN_RANGE)
    first_col = left_margin + 1
    ch_cols = [first_col + i * spacing for i in range(n_channels)]

    width = max(ch_cols[-1] + 3, GRID_W_MIN)
    if width > GRID_W_MAX:
        return None

    top_token_row = random.choice(TOKEN_ROW_OPTIONS)
    top_ep_row = top_token_row + EP_TOKEN_GAP
    token_colors = random.sample(TOKEN_COLORS_POOL, n_channels)

    for _ in range(60):
        perm = list(range(n_channels))
        random.shuffle(perm)
        if perm == list(range(n_channels)):
            continue
        inv_perm = [0] * n_channels
        for i, dst in enumerate(perm):
            inv_perm[dst] = i
        swaps = _decompose_to_swaps(inv_perm)
        if swaps and any(abs(a - b) >= 2 for a, b in swaps):
            break
    else:
        return None

    min_span = 2 * 3 + (len(swaps) - 1) * MIN_SWAP_ROW_GAP
    min_height = max(GRID_H_RANGE[0], min_span + 2 * top_token_row + 5)
    if min_height > GRID_H_RANGE[1]:
        return None
    height = random.randint(min_height, GRID_H_RANGE[1])

    bot_token_row = height - top_token_row - 1
    bot_ep_row = bot_token_row - EP_TOKEN_GAP
    if bot_ep_row - top_ep_row < min_span:
        return None

    swap_rows, marker_dists = _choose_swap_rows(top_ep_row, bot_ep_row, len(swaps))
    if swap_rows is None or marker_dists is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    _build_wire_network(grid, ch_cols, top_ep_row, bot_ep_row, swaps, swap_rows)
    reserved = _reserved_swap_cells(ch_cols, swaps, swap_rows, marker_dists)

    if _count_bare_crossings(grid) < 1:
        return None

    n_rects = random.choices(range(4), weights=RECT_COUNT_WEIGHTS)[0]
    placed_rects: list[tuple[int, int, int, int]] = []
    for _ in range(n_rects * RECT_RETRY_FACTOR):
        if len(placed_rects) >= n_rects:
            break
        result = _try_place_9rect(grid, ch_cols, top_ep_row, bot_ep_row, reserved, placed_rects)
        if result is not None:
            placed_rects.append(result)

    _place_offset_markers(grid, ch_cols, swaps, swap_rows, marker_dists)

    if _count_bare_crossings(grid) < 1:
        return None
    if _count_marked_turn_sites(grid) != 2 * len(swaps):
        return None

    if _apply_swaps(n_channels, swaps) != inv_perm:
        return None

    output = grid.copy()
    for i in range(n_channels):
        _stamp_token(output, top_token_row, ch_cols[i], token_colors[i], _DIAMOND)
        _stamp_token(output, bot_token_row, ch_cols[perm[i]], token_colors[i], _X)

    inp = _derive_input(grid, n_channels, ch_cols, perm, token_colors, top_token_row, bot_token_row)
    try:
        if not np.array_equal(main(inp), output):
            return None
        if np.array_equal(_solve_without_offset_turns(inp), output):
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_MAX_RETRIES} attempts")
