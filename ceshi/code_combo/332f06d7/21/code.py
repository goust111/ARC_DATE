# Combo variant of 332f06d7 (index 21)
# combo_spec: D10/3+D17/0
# Name: 双向移动+多对一竞争

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
spec_variant:
  input_scene:
    - 绿色背景(3)包围一条蓝色蜿蜒河道(1)
    - 1 个红色 n×n 目标方块贴着外边界
    - 2-3 个不同颜色的 n×n mover 方块位于同一条河道内部，尺寸与目标一致
    - 所有 mover 都面向同一个红色目标，且离目标的几何距离不同
  output_scene:
    - 红色目标消失；它的起点被蓝河道回填
    - 所有 mover 同时向目标推进，推进的总时长由“最先与目标相遇”的赢家决定
    - 赢家不是走到原红块位置，而是和移动中的目标在中途相遇，并占据该相遇点
    - 其余 mover 也会在同一个全局时刻停下；若终点冲突，则在各自路径上后退到最近的空位

  small_example_trace:
    - 原题：只有一个黑块，若能到达红块，则黑块直接覆盖红块原位
    - D10/3：黑块与红块双向内收，停在中段，不再固定落到红块原位
    - D17/0：多个 mover 竞争一个目标，但原本只有赢家移动
    - 融合后：先比较每个 mover 到目标的最短路径长度 D，赢家是 race_time=ceil(D/2) 最小者
      之后所有 mover 都按这个同一 race_time 前进，因此输出里会看到：
      1) 赢家在河道中部吃掉目标
      2) 输家也明显前进，而不是静止
      3) 赢家终点通常早于红块原起点，说明目标确实也在移动

b) Complexity calibration
  - 只保留 1 个目标和 2-3 个 mover，避免 many:1 场景爆炸。
  - 生成器强制唯一赢家，避免依赖隐藏 tie-break。
  - 强制 race_time>=2、至少两个 mover 发生位移，避免退化成“原题 + 多余装饰”。
  - 目标始终贴边，mover 始终在内部，角色区分稳定可见。

c) Hidden-assumption audit against the source solver
  - 原题 `cand.sort(...); mover=cand[0]; dest=cand[1]` 假设 relevant square 只有两个：这里失效。
    新版本必须找“唯一贴边的同尺寸方块”作为目标，其余同尺寸内陆方块都是 mover。
  - 原题只做一次 BFS：这里每个 mover 都要做一次，因为要比较谁先追上移动中的目标。
  - 原题 reachable 分支把 mover 放到红块原位：这里失效，赢家终点改为 shortest path 上的中途会合点。
  - 原题只回填一个起点：这里必须回填所有 mover 起点和目标起点。
  - 新增假设：训练输入满足“至少一个 mover 可达目标”，且常规样本中所有 mover 都可达。
    生成器会构造并验证这个前提。

d) Placement legality
  - 所有活动方块都保持同尺寸 n×n 实心方块。
  - 生成器只从合法的河道 placement 位置取 mover/target 起点。
  - mover 之间使用至少数个 coarse 步长的间隔，避免输入重叠与组件粘连。
  - 输出使用同一套 feasible n×n placement 语义；若多个 mover 的暂定终点冲突，solver 会回退。

e) Visual observability
  - 人类可直接看到“多个彩色方块争抢一个贴边红块”。
  - 若只看原题，赢家应落到红块原位；本变体中赢家明显停在中途，说明目标也在动。
  - 输家同样前进，说明不是“最近者独动”，而是共享同一竞争时间轴。
  - 赢家只由几何距离决定，生成器避免并列，防止解题者被隐藏排序误导。

f) Visual cleanliness
  - 场景仍然只有背景 / 河道 / 多个 mover / 一个目标 四层语义。
  - mover 数量上限 3，颜色不同，且沿主河道排布，输出不会过度杂乱。
  - 终点冲突通过回退处理，避免多个方块重叠到同一块区域。

g) Color inference
  - 颜色只用于区分不同 mover，不参与赢家判定。
  - 目标保持固定红色，结构线索仍然来自“谁更近、谁在中途吃掉目标”。

New solver-side assumptions introduced here
------------------------------------------
  - 恰有一个 relevant n×n 方块贴边且为目标；生成器保证。
  - 至少 2 个 interior mover 与目标同尺寸；生成器保证。
  - 常规样本里 shortest path winner 唯一；生成器显式过滤平局。
  - collision handling 仅需在 final-placement 层面回退，不需要完整物理动画；
    生成器保持主河道顺序清晰，使该判定稳定。
"""


_BG, _RIVER, _DEST = 3, 1, 2
_MOVER_COLORS = [0, 4, 5, 6, 7, 8, 9]
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _touches_edge(r0: int, c0: int, n: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + n == H or c0 + n == W


def _find_square_components(g: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})
    return squares


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares = _find_square_components(g, bg)
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    candidates: list[tuple[int, int]] = []
    for n, same in by_n.items():
        boundary = [sq for sq in same if _touches_edge(int(sq["r0"]), int(sq["c0"]), n, H, W)]
        if len(boundary) == 1 and int(boundary[0]["col"]) == _DEST and len(same) >= 3:
            candidates.append((len(same), int(n)))
    assert candidates

    n = max(candidates)[1]
    same = list(by_n[n])
    dest = [sq for sq in same if _touches_edge(int(sq["r0"]), int(sq["c0"]), n, H, W)][0]
    movers = [sq for sq in same if (int(sq["r0"]), int(sq["c0"])) != (int(dest["r0"]), int(dest["c0"]))]
    assert len(movers) >= 2

    excluded_colors = {int(sq["col"]) for sq in same}
    river = None
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) == bg or int(col) in excluded_colors:
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == river)
    for sq in same:
        r0, c0 = int(sq["r0"]), int(sq["c0"])
        trav[r0 : r0 + n, c0 : c0 + n] = True

    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True

    dr, dc = int(dest["r0"]), int(dest["c0"])
    assert bool(feasible[dr, dc])
    for mv in movers:
        assert bool(feasible[int(mv["r0"]), int(mv["c0"])])

    return {
        "grid": g,
        "H": H,
        "W": W,
        "bg": bg,
        "river": river,
        "n": int(n),
        "dest": dest,
        "movers": movers,
        "feasible": feasible,
    }


def _bfs_with_parent(
    feasible: np.ndarray, start: tuple[int, int]
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    h2, w2 = feasible.shape
    sr, sc = start
    assert 0 <= sr < h2 and 0 <= sc < w2 and bool(feasible[sr, sc])

    dist = -np.ones((h2, w2), dtype=int)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                parent[(rr, cc)] = (r, c)
                q.append((rr, cc))
    return dist, parent


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None], end: tuple[int, int]
) -> list[tuple[int, int]]:
    path = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _solve_fusion(grid: np.ndarray, return_meta: bool = False):
    scene = _extract_scene(grid)
    g = scene["grid"]
    n = int(scene["n"])
    river = int(scene["river"])
    dest = scene["dest"]
    dr, dc = int(dest["r0"]), int(dest["c0"])
    feasible = scene["feasible"]

    profiles: list[dict] = []
    for mover in scene["movers"]:
        sr, sc = int(mover["r0"]), int(mover["c0"])
        dist, parent = _bfs_with_parent(feasible, (sr, sc))
        dest_dist = int(dist[dr, dc]) if 0 <= dr < dist.shape[0] and 0 <= dc < dist.shape[1] else -1
        path = _reconstruct_path(parent, (dr, dc)) if dest_dist >= 0 else None
        race_time = (dest_dist + 1) // 2 if dest_dist >= 0 else None
        profiles.append(
            {
                "mover": mover,
                "start": (sr, sc),
                "dist": dist,
                "dest_dist": dest_dist,
                "path": path,
                "race_time": race_time,
            }
        )

    reachable = [p for p in profiles if int(p["dest_dist"]) >= 0]
    assert reachable

    winner = min(
        reachable,
        key=lambda p: (
            int(p["race_time"]),
            int(p["dest_dist"]),
            int(p["start"][0]),
            int(p["start"][1]),
        ),
    )
    global_time = int(winner["race_time"])

    placements: list[dict] = []
    reserved: set[tuple[int, int]] = set()

    winner_final = tuple(winner["path"][global_time])
    reserved.add(winner_final)
    placements.append(
        {
            "mover": winner["mover"],
            "start": tuple(winner["start"]),
            "final": winner_final,
            "steps": global_time,
            "is_winner": True,
            "dest_dist": int(winner["dest_dist"]),
        }
    )

    reachable_losers = [p for p in reachable if p is not winner]
    reachable_losers.sort(key=lambda p: (int(p["dest_dist"]), int(p["start"][0]), int(p["start"][1])))
    for prof in reachable_losers:
        cap = min(global_time, max(0, int(prof["dest_dist"]) - 1))
        step = int(cap)
        while step > 0 and tuple(prof["path"][step]) in reserved:
            step -= 1
        final = tuple(prof["path"][step])
        assert final not in reserved
        reserved.add(final)
        placements.append(
            {
                "mover": prof["mover"],
                "start": tuple(prof["start"]),
                "final": final,
                "steps": int(step),
                "is_winner": False,
                "dest_dist": int(prof["dest_dist"]),
            }
        )

    unreachable = [p for p in profiles if int(p["dest_dist"]) < 0]
    for prof in unreachable:
        final = tuple(prof["start"])
        assert final not in reserved
        reserved.add(final)
        placements.append(
            {
                "mover": prof["mover"],
                "start": tuple(prof["start"]),
                "final": final,
                "steps": 0,
                "is_winner": False,
                "dest_dist": -1,
            }
        )

    out = g.copy()
    out[dr : dr + n, dc : dc + n] = river
    for info in placements:
        sr, sc = info["start"]
        out[sr : sr + n, sc : sc + n] = river
    for info in placements:
        fr, fc = info["final"]
        out[fr : fr + n, fc : fc + n] = int(info["mover"]["col"])

    if return_meta:
        reachable_sorted = sorted(
            [(int(p["race_time"]), int(p["dest_dist"]), tuple(p["start"])) for p in reachable]
        )
        return out, {
            "scene": scene,
            "profiles": profiles,
            "placements": placements,
            "winner_start": tuple(winner["start"]),
            "winner_color": int(winner["mover"]["col"]),
            "winner_dist": int(winner["dest_dist"]),
            "winner_final": winner_final,
            "dest_start": (dr, dc),
            "race_time": global_time,
            "moved_count": sum(int(p["steps"]) > 0 for p in placements),
            "all_reachable": len(reachable) == len(profiles),
            "reachable_race_keys": reachable_sorted,
            "num_movers": len(profiles),
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Multiple same-sized movers compete for one boundary target on a winding river.

    Each mover gets its own shortest path to the target on the feasible n×n placement
    graph. The mover with the smallest bidirectional meeting time ceil(D/2) wins: the
    target is treated as moving toward the winner, so the winner stops at a midpoint
    meeting cell rather than at the target's original position. All other movers also
    advance for that same global amount of time and back off one step if their final
    cell would collide with an already claimed meeting position.
    """
    return _solve_fusion(grid)


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= total * 0.7:
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                break
            neighbors = []
            for dr, dc in _DIR4:
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            valid = []
            for nb in neighbors:
                if nb == end:
                    valid.append(nb)
                    continue
                test_visited = visited | {nb}
                seen = {end}
                q = deque([end])
                while q:
                    cr, cc = q.popleft()
                    for dr2, dc2 in _DIR4:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4:
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) not in test_visited and (ar, ac) in seen:
                        valid.append(nb)
                        break
            if not valid:
                break
            random.shuffle(valid)
            nxt = valid[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        if best_path and best_len >= total * 0.85:
            break
    return best_path


def _render_channel(
    grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
    return pixel_pos


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[int, int] | None:
    H, W = grid.shape
    candidates = []
    for direction in ("top", "bottom", "left", "right"):
        if direction == "top":
            dist = pr
            clear = all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n) if c < W)
            if clear:
                candidates.append((dist, 0, pc))
        elif direction == "bottom":
            dist = H - (pr + n)
            clear = all(
                grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n) if c < W
            )
            if clear:
                candidates.append((dist, H - n, pc))
        elif direction == "left":
            dist = pc
            clear = all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H)
            if clear:
                candidates.append((dist, pr, 0))
        else:
            dist = W - (pc + n)
            clear = all(
                grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W) if r < H
            )
            if clear:
                candidates.append((dist, pr, W - n))
    if not candidates:
        return None
    candidates.sort()
    nr, nc = candidates[0][1], candidates[0][2]
    if nr != pr:
        r_lo, r_hi = min(nr, pr), max(nr, pr) + n
        grid[r_lo:r_hi, pc : pc + n] = _RIVER
    if nc != pc:
        c_lo, c_hi = min(nc, pc), max(nc, pc) + n
        grid[pr : pr + n, c_lo:c_hi] = _RIVER
    return int(nr), int(nc)


def _choose_mover_indices(pixel_pos: list[tuple[int, int]], n: int, H: int, W: int) -> list[int] | None:
    interior = [i for i, (r, c) in enumerate(pixel_pos[:-2]) if not _touches_edge(r, c, n, H, W)]
    if len(interior) < 2:
        return None

    target_num = 3 if len(interior) >= 8 and random.random() < 0.65 else 2
    lo = max(2, len(pixel_pos) // 3)
    hi = len(pixel_pos) - 4
    winner_candidates = [i for i in interior if lo <= i <= hi]
    random.shuffle(winner_candidates)

    for winner_idx in winner_candidates:
        earlier = [i for i in interior if i <= winner_idx - 3]
        if len(earlier) < 1:
            continue
        random.shuffle(earlier)
        chosen = [winner_idx]
        for idx in earlier:
            if all(abs(idx - j) >= 3 for j in chosen):
                chosen.append(idx)
            if len(chosen) >= target_num:
                break
        if len(chosen) >= 2:
            return sorted(chosen)
    return None


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([1, 2, 3, 4])
    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = n + 1
    max_coarse = max(4, (G - 2) // min_spacing)
    coarse_dim = random.randint(4, min(6, max_coarse))
    offsets_r = _compute_offsets(G, n, coarse_dim)
    offsets_c = _compute_offsets(G, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None
    if any(offsets_r[i + 1] - offsets_r[i] < n for i in range(len(offsets_r) - 1)):
        return None
    if any(offsets_c[i + 1] - offsets_c[i] < n for i in range(len(offsets_c) - 1)):
        return None

    edge_cells = set()
    for c in range(coarse_dim):
        edge_cells.add((0, c))
        edge_cells.add((coarse_dim - 1, c))
    for r in range(1, coarse_dim - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_dim - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_dim, coarse_dim, start_coarse, end_coarse)
    if walk is None or len(walk) < 7:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_dim or len(set(cc for _, cc in walk)) < coarse_dim:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    area_ratio = float(np.mean(grid == _RIVER))
    if not (0.18 <= area_ratio <= 0.62):
        return None

    dest_r, dest_c = pixel_pos[-1]
    extended = _extend_to_edge(grid, dest_r, dest_c, n)
    if extended is None:
        return None
    dest_r, dest_c = extended
    if not _touches_edge(dest_r, dest_c, n, H, W):
        return None

    mover_indices = _choose_mover_indices(pixel_pos, n, H, W)
    if mover_indices is None:
        return None
    winner_idx = max(mover_indices)

    mover_colors = random.sample(_MOVER_COLORS, len(mover_indices))
    for idx, col in zip(mover_indices, mover_colors):
        r0, c0 = pixel_pos[idx]
        if _touches_edge(r0, c0, n, H, W):
            return None
        grid[r0 : r0 + n, c0 : c0 + n] = int(col)
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    try:
        expected, meta = _solve_fusion(grid, return_meta=True)
    except Exception:
        return None

    if int(meta["num_movers"]) != len(mover_indices):
        return None
    if not bool(meta["all_reachable"]):
        return None
    if tuple(meta["winner_start"]) != tuple(pixel_pos[winner_idx]):
        return None
    if int(meta["race_time"]) < 2:
        return None
    if tuple(meta["winner_final"]) == tuple(meta["dest_start"]):
        return None
    if _touches_edge(int(meta["winner_final"][0]), int(meta["winner_final"][1]), n, H, W):
        return None

    move_steps = [int(p["steps"]) for p in meta["placements"]]
    if sum(step > 0 for step in move_steps) < 2:
        return None
    if max(move_steps) < 2:
        return None

    race_keys = list(meta["reachable_race_keys"])
    if len(race_keys) >= 2 and race_keys[0][:2] == race_keys[1][:2]:
        return None

    finals = [tuple(p["final"]) for p in meta["placements"]]
    if len(set(finals)) != len(finals):
        return None

    out = main(grid)
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
