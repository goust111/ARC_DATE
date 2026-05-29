# Combo variant of 332f06d7 (index 35)
# combo_spec: D10/0+D12/1+D17/1
# Name: 多体顺序移动+环形通道+一对多目标
#
# Design analysis
# ===============
# a) Concrete behavior / trace
#    原题只有一个 mover 和一个目标，蓝色通道也是单链，所以“走到目标”与“沿唯一通道一直前进”几乎是同一件事。
#    这里把三项改动融合成一个可见、可推断的规则：
#    - 只有一个黑色 mover，避免同时叠加“多 mover 顺序”和“多目标顺序”两种隐藏序列。
#    - 输入里有 3 个不同颜色的 n×n 目标方块，它们各自还有一个同色 1×1 图例点。
#      图例排成一行或一列，给出目标访问顺序；不能依赖颜色数值做不可见排序。
#    - 蓝色通道是一个矩形环，mover 从左侧支路进入，3 个目标挂在环的其它支路上。
#    小例子：
#    - 图例顺序是 [黄, 红, 灰]。
#    - mover 先去黄色目标；到达后继续移动。
#    - 接着它尝试去红色目标。若红色支路被卡点缩窄到小于 n，则它不会绕去灰色目标，
#      而是停在“尽量接近红色支路末端”的最后可达位置；灰色目标保持不变。
#    - 若红色支路不堵，则 mover 去完黄色后再去红色，再继续去灰色并停在灰色上。
#    与原题差异：
#    - 不是单次移动，而是按图例顺序的多步目标序列。
#    - 不是单一路径，而是环路；每一步都允许顺时针/逆时针择短路。
#    - 不是 1:1，而是 1:many；已经访问过的目标会被重新还原为蓝色河道。
#
# b) Complexity calibration
#    - 固定为 3 个目标，既能稳定体现 1:many，也能让“第二个目标被卡住、第三个几何上仍可达”
#      这种顺序依赖稳定出现。
#    - n 限制为 2 或 3，避免 1×1 图例和大方块混淆。
#    - 目标颜色只用 3 种，且每种都在“大目标 + 图例点”上重复出现，可视支撑足够。
#    - 环结构固定为一个粗矩形环加 4 条短支路，自由度适中，生成成功率高且视觉稳定。
#
# c) Implicit assumption audit
#    原 main() 的关键假设在融合后都失效了：
#    - “同尺寸方块里按颜色排序取前两个”失效：现在有 1 个 mover + 3 个不同颜色目标。
#    - “目标只有一个坐标”失效：现在有有序目标列表。
#    - “所有同色像素都可通行”失效：目标颜色还出现在 1×1 图例里，图例绝不能当河道。
#    - “stuck 时取离起点最远的位置”失效：在环上那会把 mover 放到与当前目标无关的一侧。
#    新 solver 因此显式保证：
#    - 只把 n×n 大目标组件当作可通行目标，不把图例点纳入 BFS。
#    - 目标顺序只从图例读出，不依赖颜色值。
#    - 每次只追当前目标；若当前目标不可达，则按“离当前目标最近，若并列则 BFS 更远”选停点。
#
# d) Placement legality
#    - 生成器先画环和支路，再把 mover / targets 放到互不重叠的 n×n 支路末端。
#    - 图例只放在背景像素上，并要求四邻也为空背景，避免和河道或同色大方块连成一片。
#    - blocked 样例只在第二个目标支路上打卡点，保证第一个目标可达、第二个不可达、
#      第三个支路仍然保持几何通达，从而真正触发“按序停下”的融合逻辑。
#
# e) Visual observability
#    - 人类可以直接看到：一个黑方块、三个彩色目标、一个环形蓝河道、以及同色图例顺序。
#    - reachable 样例中，前两个目标消失、黑方块最终停在第三个目标上。
#    - blocked 样例中，第一个目标消失，黑方块停在第二个目标支路附近，而第三个目标仍保留，
#      说明“不是去所有能到的目标”，而是“按图例顺序逐个尝试”。
#
# f) Visual cleanliness
#    - 只有 4 层：背景、蓝河道、一个黑 mover、三个彩色目标加对应图例点。
#    - 图例和主结构分开摆放，不与环或支路重叠。
#    - 目标分散在环的三个方位，既清楚显示环，也避免挤成一团。
#
# g) New solver assumptions and generator guarantees
#    - 恰有一个 n>1 的 mover 颜色没有同色图例点；生成器固定为黑色 mover。
#    - 恰有 3 个目标颜色各自同时出现为 “1 个 n×n 方块 + 1 个 1×1 图例”；生成器强制如此构造。
#    - 图例点共线；生成器保证横向或纵向放置。
#    - start-reachable placement 图确实含环；生成器用 BFS 图上的环检测做校验。

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


_BG = 3
_RIVER = 1
_MOVER = 0
_TARGET_POOL = [2, 4, 5, 6, 7, 8, 9]
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _solid_square_components(g: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(col)).all()):
            continue
        squares.append(
            {
                "col": col,
                "r0": int(b[0]),
                "c0": int(b[1]),
                "n": int(bbox_h(b)),
                "area": int(comp["area"]),
            }
        )
    return squares


def _find_large_square(g: np.ndarray, bg: int, color: int, n: int) -> tuple[int, int, np.ndarray]:
    matches: list[tuple[int, int, np.ndarray]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[color], return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != n or bbox_w(b) != n or int(comp["area"]) != n * n:
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(color)).all()):
            continue
        mask = np.zeros_like(g, dtype=bool)
        mask[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1] = True
        matches.append((int(b[0]), int(b[1]), mask))
    assert len(matches) == 1
    return matches[0]


def _legend_order(g: np.ndarray, bg: int, target_colors: set[int]) -> list[int]:
    singles: list[tuple[int, int, int]] = []
    for col in target_colors:
        comps = [
            comp
            for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[col], return_format="dict")
            if int(comp["area"]) == 1
        ]
        assert len(comps) == 1
        r, c = map(int, comps[0]["coords"][0])
        singles.append((r, c, int(col)))
    same_row = len({r for r, _c, _col in singles}) == 1
    same_col = len({c for _r, c, _col in singles}) == 1
    assert same_row or same_col
    if same_row:
        singles.sort(key=lambda item: item[1])
    else:
        singles.sort(key=lambda item: item[0])
    return [col for _r, _c, col in singles]


def _infer_setup(g: np.ndarray) -> tuple[int, int, int, dict, list[dict]]:
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    squares = _solid_square_components(g, bg)
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        if int(sq["n"]) > 1:
            by_n[int(sq["n"])].append(sq)

    candidates = []
    for nside, group in by_n.items():
        if len(group) < 4:
            continue
        by_color: dict[int, list[dict]] = defaultdict(list)
        for sq in group:
            by_color[int(sq["col"])].append(dict(sq))
        if any(len(parts) != 1 for parts in by_color.values()):
            continue
        target_colors: list[int] = []
        mover_colors: list[int] = []
        valid = True
        for col in by_color:
            singles = [
                comp
                for comp in find_components(
                    g, bg=bg, connectivity=4, monochromatic=True, colors=[col], return_format="dict"
                )
                if int(comp["area"]) == 1
            ]
            if len(singles) == 1:
                target_colors.append(int(col))
            elif len(singles) == 0:
                mover_colors.append(int(col))
            else:
                valid = False
                break
        if not valid or len(mover_colors) != 1 or len(target_colors) < 2:
            continue
        order = _legend_order(g, bg, set(target_colors))
        if sorted(order) != sorted(target_colors):
            continue
        candidates.append((len(group), int(nside), mover_colors[0], order, by_color))

    assert candidates
    _count, n, mover_col, order, by_color = max(candidates, key=lambda item: (item[0], item[1]))
    mover = dict(by_color[int(mover_col)][0])
    targets = [dict(by_color[int(col)][0]) for col in order]

    counts = Counter(map(int, g.flatten()))
    river = None
    banned = {bg, int(mover_col)} | {int(t["col"]) for t in targets}
    for col, _ct in counts.most_common():
        if int(col) not in banned:
            river = int(col)
            break
    assert river is not None
    return bg, int(river), int(n), mover, targets


def _compute_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _choose_stuck_target(dist: np.ndarray, dr: int, dc: int) -> tuple[int, int]:
    best_key = None
    best_rc = None
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) < 0:
                continue
            key = (abs(r - dr) + abs(c - dc), -int(dist[r, c]), abs(r - dr), abs(c - dc), r, c)
            if best_key is None or key < best_key:
                best_key = key
                best_rc = (int(r), int(c))
    assert best_rc is not None
    return best_rc


def _build_traversable(
    out: np.ndarray,
    bg: int,
    n: int,
    river: int,
    mover_col: int,
    remaining_targets: list[dict],
) -> tuple[int, int, np.ndarray, np.ndarray]:
    sr, sc, mover_mask = _find_large_square(out, bg, mover_col, n)
    trav = (out == int(river)) | mover_mask
    for target in remaining_targets:
        _rr, _cc, mask = _find_large_square(out, bg, int(target["col"]), n)
        trav |= mask
    feasible = _compute_feasible(trav, n)
    assert bool(feasible[sr, sc])
    return sr, sc, mover_mask, feasible


def _move_to_target(
    out: np.ndarray,
    bg: int,
    n: int,
    river: int,
    mover_col: int,
    remaining_targets: list[dict],
    current_target: dict,
) -> tuple[bool, tuple[int, int], tuple[int, int], np.ndarray]:
    sr, sc, mover_mask, feasible = _build_traversable(out, bg, n, river, mover_col, remaining_targets)
    dist = _bfs_placement(feasible, sr, sc)
    dr, dc = int(current_target["r0"]), int(current_target["c0"])
    reachable = 0 <= dr < feasible.shape[0] and 0 <= dc < feasible.shape[1] and feasible[dr, dc] and dist[dr, dc] >= 0
    if reachable:
        tr, tc = dr, dc
    else:
        tr, tc = _choose_stuck_target(dist, dr, dc)
    out[mover_mask] = int(river)
    out[tr : tr + n, tc : tc + n] = int(mover_col)
    return bool(reachable), (int(sr), int(sc)), (int(tr), int(tc)), dist


def _solve_details(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg, river, n, mover, targets = _infer_setup(g)
    out = g.copy()
    steps = []
    reached_count = 0
    final_pos = (int(mover["r0"]), int(mover["c0"]))
    for idx, target in enumerate(targets):
        reached, start_pos, end_pos, dist = _move_to_target(
            out,
            bg,
            n,
            river,
            int(mover["col"]),
            targets[idx:],
            target,
        )
        steps.append(
            {
                "target_col": int(target["col"]),
                "target_pos": (int(target["r0"]), int(target["c0"])),
                "start_pos": start_pos,
                "end_pos": end_pos,
                "reached": bool(reached),
                "dist": dist,
            }
        )
        final_pos = end_pos
        if not reached:
            break
        reached_count += 1
    return {
        "out": out,
        "bg": int(bg),
        "river": int(river),
        "n": int(n),
        "mover": mover,
        "targets": targets,
        "steps": steps,
        "reached_count": int(reached_count),
        "final_pos": final_pos,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - One solid mover square travels in a ring-shaped river with several colored target squares.
    - Target visit order is shown by same-colored 1x1 legend cells on one line.
    - For each target in legend order, the mover uses shortest-path reachability on the ring.
    - If the current target is reachable, the mover lands on it and then continues toward the next one.
    - If the current target is blocked by a too-narrow branch, the mover stops at the reachable placement
      closest to that target and later targets remain untouched.
    """
    return _solve_details(np.array(grid, dtype=int))["out"]


def _component_has_cycle(feasible: np.ndarray, sr: int, sc: int) -> bool:
    seen = {(sr, sc)}
    q = deque([(sr, sc)])
    nodes = 0
    edges2 = 0
    h2, w2 = feasible.shape
    while q:
        r, c = q.popleft()
        nodes += 1
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc]:
                edges2 += 1
                if (rr, cc) not in seen:
                    seen.add((rr, cc))
                    q.append((rr, cc))
    return (edges2 // 2) >= nodes


def _initial_target_reachability(grid: np.ndarray) -> tuple[list[bool], bool]:
    g = np.array(grid, dtype=int)
    bg, river, n, mover, targets = _infer_setup(g)
    sr, sc, mover_mask = _find_large_square(g, bg, int(mover["col"]), n)
    trav = (g == int(river)) | mover_mask
    for target in targets:
        _rr, _cc, mask = _find_large_square(g, bg, int(target["col"]), n)
        trav |= mask
    feasible = _compute_feasible(trav, n)
    dist = _bfs_placement(feasible, sr, sc)
    reachability = []
    for target in targets:
        tr, tc = int(target["r0"]), int(target["c0"])
        reachability.append(bool(feasible[tr, tc] and dist[tr, tc] >= 0))
    return reachability, _component_has_cycle(feasible, sr, sc)


def _combo_metrics(grid: np.ndarray) -> dict:
    details = _solve_details(grid)
    initial_reachability, has_cycle = _initial_target_reachability(grid)
    later_reachable_after_block = False
    if details["reached_count"] < len(details["targets"]):
        blocked_idx = int(details["reached_count"])
        later_reachable_after_block = any(initial_reachability[blocked_idx + 1 :])
    return {
        "target_count": len(details["targets"]),
        "has_cycle": bool(has_cycle),
        "ordered_reached_count": int(details["reached_count"]),
        "individually_reachable_count": int(sum(initial_reachability)),
        "order_blocked_later_reachable": bool(later_reachable_after_block),
        "final_pos": details["final_pos"],
    }


def _compute_offsets(G: int, n: int, count: int) -> list[int] | None:
    if count == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (count - 1)
    return [int(round(first + i * step)) for i in range(count)]


def _rect_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        out.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        out.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        out.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        out.append((r, c0))
    return out


def _path_edges(path: list[tuple[int, int]], close_cycle: bool = False) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    if close_cycle and len(path) >= 2:
        edges.append((path[-1], path[0]))
    return edges


def _render_graph(
    grid: np.ndarray,
    nodes: set[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> dict[tuple[int, int], tuple[int, int]]:
    pixel_pos: dict[tuple[int, int], tuple[int, int]] = {}
    for cr, cc in nodes:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos[(cr, cc)] = (pr, pc)
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for a, b in edges:
        pr1, pc1 = pixel_pos[a]
        pr2, pc2 = pixel_pos[b]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
    return pixel_pos


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int) -> None:
    H, W = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    narrow_w = random.randint(1, n - 1)
    remove = n - narrow_w
    side = random.randint(0, 1)
    if pr1 == pr2:
        c_lo = min(pc1, pc2) + 1
        c_hi = max(pc1, pc2) + n - 1
        span = max(1, c_hi - c_lo + 1)
        cut_len = min(span, max(1, 2 * remove + 1))
        c_start = c_lo + (span - cut_len) // 2
        for c in range(c_start, c_start + cut_len):
            for k in range(remove):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < H and 0 <= c < W:
                    grid[rr, c] = _BG
    else:
        r_lo = min(pr1, pr2) + 1
        r_hi = max(pr1, pr2) + n - 1
        span = max(1, r_hi - r_lo + 1)
        cut_len = min(span, max(1, 2 * remove + 1))
        r_start = r_lo + (span - cut_len) // 2
        for r in range(r_start, r_start + cut_len):
            for k in range(remove):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= r < H and 0 <= cc < W:
                    grid[r, cc] = _BG


def _slot_clear_for_singleton(grid: np.ndarray, r: int, c: int) -> bool:
    H, W = grid.shape
    if grid[r, c] != _BG:
        return False
    for dr, dc in _DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and grid[rr, cc] != _BG:
            return False
    return True


def _place_legend(grid: np.ndarray, colors: list[int]) -> bool:
    H, W = grid.shape
    need = 2 * len(colors) - 1
    rows = [0, H - 1, 1, H - 2]
    cols = [0, W - 1, 1, W - 2]
    for r in rows:
        if not (0 <= r < H):
            continue
        for start in range(0, W - need + 1):
            slots = [(r, start + 2 * i) for i in range(len(colors))]
            if all(_slot_clear_for_singleton(grid, rr, cc) for rr, cc in slots):
                for (rr, cc), col in zip(slots, colors):
                    grid[rr, cc] = int(col)
                return True
    for c in cols:
        if not (0 <= c < W):
            continue
        for start in range(0, H - need + 1):
            slots = [(start + 2 * i, c) for i in range(len(colors))]
            if all(_slot_clear_for_singleton(grid, rr, cc) for rr, cc in slots):
                for (rr, cc), col in zip(slots, colors):
                    grid[rr, cc] = int(col)
                return True
    return False


def _choose_layout(n: int) -> tuple[int, list[int], list[int]] | None:
    choices = []
    for G in range(max(14, 5 * n), 21):
        offsets = _compute_offsets(G, n, 5)
        if offsets is None:
            continue
        if any(offsets[i + 1] - offsets[i] <= n for i in range(4)):
            continue
        choices.append((G, offsets))
    if not choices:
        return None
    G, offsets = random.choice(choices)
    return G, offsets, list(offsets)


def _gen_attempt() -> np.ndarray | None:
    reachable = random.random() < 0.5
    n = random.choice([2, 3])
    layout = _choose_layout(n)
    if layout is None:
        return None
    G, offsets_r, offsets_c = layout
    grid = np.full((G, G), _BG, dtype=int)

    loop_path = _rect_cycle(1, 1, 3, 3)
    loop_edges = _path_edges(loop_path, close_cycle=True)
    start_attach = (2, 1)
    start_node = (2, 0)
    ordered_specs = [
        {"attach": (1, 2), "end": (0, 2)},
        {"attach": (3, 2), "end": (4, 2)},
        {"attach": (2, 3), "end": (2, 4)},
    ]
    nodes = set(loop_path)
    nodes.add(start_node)
    for spec in ordered_specs:
        nodes.add(spec["end"])
    edges = list(loop_edges)
    edges.append((start_node, start_attach))
    for spec in ordered_specs:
        edges.append((spec["attach"], spec["end"]))

    pixel_pos = _render_graph(grid, nodes, edges, n, offsets_r, offsets_c)
    mover_r, mover_c = pixel_pos[start_node]
    target_positions = [pixel_pos[spec["end"]] for spec in ordered_specs]

    if not reachable:
        blocked = ordered_specs[1]
        _add_chokepoint(grid, pixel_pos[blocked["attach"]], pixel_pos[blocked["end"]], n)

    target_colors = random.sample(_TARGET_POOL, 3)
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = int(_MOVER)
    for color, (rr, cc) in zip(target_colors, target_positions):
        grid[rr : rr + n, cc : cc + n] = int(color)

    if not _place_legend(grid, target_colors):
        return None

    try:
        out = main(grid)
        details = _solve_details(grid)
        metrics = _combo_metrics(grid)
    except Exception:
        return None

    if not np.array_equal(out, details["out"]):
        return None
    if not metrics["has_cycle"] or metrics["target_count"] != 3:
        return None
    if reachable:
        if metrics["ordered_reached_count"] != 3:
            return None
        if not np.array_equal(details["final_pos"], target_positions[-1]):
            return None
    else:
        if metrics["ordered_reached_count"] != 1:
            return None
        if metrics["individually_reachable_count"] < 2:
            return None
        if not metrics["order_blocked_later_reachable"]:
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input after 400 attempts")
