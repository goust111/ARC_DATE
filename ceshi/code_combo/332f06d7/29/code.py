# Combo variant of 332f06d7 (index 29)
# combo_spec: D3/1+D10/1
# Name: 间歇轨迹+移动体分裂
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 输入保留原题的可视角色：绿色背景、蓝色河道、黑色 mover、红色目标。
# - 但几何改成固定的分裂场景：一个 5x5 黑方块沿 5 像素宽干道进入 5x5 分叉室，
#   然后在分叉室顶部裂成两个 2x2 子方块，分别进入左、右两条 2 像素宽分支。
# - 左右分支末端各有一个 2x2 红色目标；若分支上出现 1 像素卡点，该子方块会停在
#   最后一个可行 2x2 放置位置，否则覆盖该红块。
# - 与 D10 单变体相比，输出不再只有两个最终 2x2 黑块。每个子方块沿自己的实际路径
#   还会留下间歇轨迹：左分支周期固定为 3，右分支周期固定为 4。
# - 具体例子：左子方块路径 placements 为 p0..p10，则在 p3、p6、p9 留下 2x2 黑印记；
#   右子方块路径 q0..q9，则在 q4、q8 留下 2x2 黑印记。终点本身不是轨迹印记，因为
#   它本来就由最终子方块占据。
#
# b) 复杂度校准
# - 仍只使用 4 种颜色，避免颜色映射歧义。
# - 大小关系固定为 5x5 -> 两个 2x2，保证“分裂”在视觉上稳定、明显。
# - 轨迹周期只取两个值 3 和 4，且固定绑定到左/右分支，避免自由度过多。
# - generator 强制两条分支都足够长，且总轨迹印记数至少为 3，从而肉眼能看到“分裂”
#   和“虚线”两种变化同时出现。
#
# c) 隐式假设审查
# - 原题 `main()` 假设 mover 与 destination 同尺寸；这里不成立，必须先检测唯一最大黑块，
#   再检测两个较小且同尺寸的红块。
# - 原题只需要一个 BFS 和一个终点；这里不成立，需要：
#   1. 5x5 大方块的可达覆盖区；
#   2. 去掉该覆盖区后得到两条独立分支；
#   3. 每条分支各自做 2x2 BFS，并记录 parent 恢复实际路径。
# - D3 的轨迹规则要求显式路径；因此分支 BFS 必须保留 parent，不能只保留 dist。
# - 原题默认输出里除终点外没有额外黑块；这里不成立，输出会新增多个 2x2 轨迹印记。
#
# d) 放置合法性
# - 干道用 5x5 top-left 稠密路径绘制，分支用 2x2 top-left 稠密路径绘制，确保 solver 的
#   placement-BFS 与 generator 的几何完全一致。
# - 两条 2 宽分支在分叉室上方由 1 列背景隔开，去掉大方块可达覆盖区后必定拆成两个像素组件。
# - mover、targets、轨迹都不越界；红目标不放在角落；卡点只削成 1 宽，不会破坏整条河道的
#   可视连通性。
#
# e) 视觉可观察性
# - 人类从样例能直接看出：
#   1. 一个大黑块消失，变成两个小黑块；
#   2. 左右两条分支各自沿路出现虚线印记；
#   3. 某些样例里某个子方块会停在卡点前，红目标保留。
# - 轨迹密度差异不依赖颜色编号，而依赖可见的左右分支角色：左边更密，右边更疏。
#
# f) 可视化整洁度
# - 不引入新装饰色，不在同一区域叠多层标记。
# - 轨迹只画在 2x2 合法放置窗口上，且左右分支分离，避免两条虚线纠缠成黑团。
# - generator 过滤掉轨迹过少、或融合版输出与“纯分裂版”完全相同的样例。
#
# g) 变化的可推断性
# - 没有隐藏的数值排序规则。颜色角色固定且可见：黑=移动体/轨迹，红=目标，蓝=河道，绿=背景。
# - 左右两条分支持续重复同一规律：左边每 3 步留下一个 2x2 黑印记，右边每 4 步留下一个。
#   这在 2-4 个样例中是可归纳的稳定结构。
#
# New solver assumptions enforced by the generator
# ------------------------------------------------
# - 输入中恰有一个最大的实心方块作为 mover，且其尺寸固定为 5。
# - 输入中恰有两个相同尺寸的较小红方块作为分裂后的目标，尺寸固定为 2。
# - 去掉 5x5 mover 的可达覆盖后，剩余可 traversable 像素正好分成两个含目标的分支组件。
# - 左右分支可按目标的列坐标稳定区分，并分别绑定周期 3 / 4。
# - 每条分支从分叉口到最终停点的路径足够长，至少留下一个轨迹印记，且融合输出必须不同于
#   不带轨迹的纯 D10 输出。

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_BIG, _SMALL = 5, 2
_PERIODS = (3, 4)


def _solid_square_components(grid: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        box = comp["bbox"]
        if bbox_h(box) != bbox_w(box):
            continue
        r0, c0, r1, c1 = map(int, box)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == color).all()):
            continue
        squares.append({"color": color, "r0": r0, "c0": c0, "n": int(bbox_h(box))})
    return squares


def _bool_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            comp = np.zeros((h, w), dtype=bool)
            q = deque([(r, c)])
            seen[r, c] = True
            comp[r, c] = True
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        comp[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _feasible_windows(trav: np.ndarray, size: int) -> np.ndarray:
    h, w = trav.shape
    h2, w2 = h - size + 1, w - size + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + size, c : c + size].all())
    return feasible


def _bfs_with_parent(mask: np.ndarray, seeds: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    dist = -np.ones((h, w), dtype=int)
    parent = -np.ones((h, w, 2), dtype=int)
    q = deque()
    for r, c in seeds:
        if 0 <= r < h and 0 <= c < w and bool(mask[r, c]) and int(dist[r, c]) == -1:
            dist[r, c] = 0
            parent[r, c] = (r, c)
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c]) + 1
                parent[rr, cc] = (r, c)
                q.append((rr, cc))
    return dist, parent


def _reachable_cover(reach: np.ndarray, size: int, shape: tuple[int, int]) -> np.ndarray:
    cover = np.zeros(shape, dtype=bool)
    for r in range(reach.shape[0]):
        for c in range(reach.shape[1]):
            if int(reach[r, c]) >= 0:
                cover[r : r + size, c : c + size] = True
    return cover


def _dense_path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert points
    path = [points[0]]
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for c in range(c0 + step, c1 + step, step):
                path.append((r0, c))
        else:
            assert c0 == c1
            step = 1 if r1 > r0 else -1
            for r in range(r0 + step, r1 + step, step):
                path.append((r, c0))
    return path


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], size: int, color: int) -> None:
    for r, c in path:
        grid[r : r + size, c : c + size] = color


def _narrow_vertical_corridor(
    grid: np.ndarray,
    c: int,
    top_r: int,
    root_r: int,
    width: int,
    min_open_steps: int,
) -> bool:
    assert width == 2
    cut_min = top_r + 1
    cut_max = root_r - min_open_steps
    if cut_max < cut_min:
        return False
    cut_r = random.randint(cut_min, cut_max)
    length = min(2, root_r - cut_r)
    if length < 1:
        return False
    keep_right = bool(random.getrandbits(1))
    if keep_right:
        grid[cut_r : cut_r + length, c : c + width - 1] = _BG
    else:
        grid[cut_r : cut_r + length, c + 1 : c + width] = _BG
    return True


def _pick_roles(grid: np.ndarray, bg: int) -> tuple[dict, list[dict], int]:
    squares = _solid_square_components(grid, bg)
    assert squares

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    mover_n = max(by_n)
    assert len(by_n[mover_n]) == 1
    mover = by_n[mover_n][0]

    child_sizes = [n for n, items in by_n.items() if n < mover_n and len(items) >= 2]
    assert child_sizes
    child_n = max(child_sizes)
    dests = sorted(by_n[child_n], key=lambda s: (int(s["c0"]), int(s["r0"])))[:2]
    assert len(dests) == 2
    return mover, dests, child_n


def _river_color(grid: np.ndarray, bg: int, mover: dict, dests: list[dict]) -> int:
    excluded = {int(bg), int(mover["color"])} | {int(d["color"]) for d in dests}
    counts = Counter(map(int, grid.flatten()))
    for color, _count in counts.most_common():
        if int(color) not in excluded:
            return int(color)
    raise AssertionError("river color not found")


def _arm_components(trav: np.ndarray, big_cover: np.ndarray, dests: list[dict], child_n: int) -> list[np.ndarray]:
    small_only = trav & ~big_cover
    comps = _bool_components(small_only)
    arms: list[np.ndarray] = []
    for dest in dests:
        r0, c0 = int(dest["r0"]), int(dest["c0"])
        matches = [comp for comp in comps if bool(comp[r0 : r0 + child_n, c0 : c0 + child_n].all())]
        assert len(matches) == 1
        arms.append(matches[0])
    return arms


def _branch_state(
    trav: np.ndarray,
    big_cover: np.ndarray,
    arm: np.ndarray,
    child_n: int,
    dest: dict,
    period: int,
) -> dict:
    small_feasible = _feasible_windows(trav, child_n)
    h2, w2 = small_feasible.shape
    branch_feasible = np.zeros((h2, w2), dtype=bool)
    seeds: list[tuple[int, int]] = []

    for r in range(h2):
        for c in range(w2):
            if not bool(small_feasible[r, c]):
                continue
            if not bool(arm[r : r + child_n, c : c + child_n].any()):
                continue
            allowed_patch = (arm | big_cover)[r : r + child_n, c : c + child_n]
            if not bool(allowed_patch.all()):
                continue
            branch_feasible[r, c] = True
            if bool(big_cover[r : r + child_n, c : c + child_n].any()):
                seeds.append((r, c))

    assert seeds
    dist, parent = _bfs_with_parent(branch_feasible, seeds)

    dr, dc = int(dest["r0"]), int(dest["c0"])
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(branch_feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        tr, tc = dr, dc
        reached = True
    else:
        best = -1
        tr, tc = seeds[0]
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c
        reached = False

    path = [(int(tr), int(tc))]
    cr, cc = int(tr), int(tc)
    while True:
        pr, pc = parent[cr, cc]
        assert pr >= 0 and pc >= 0
        if (int(pr), int(pc)) == (cr, cc):
            break
        path.append((int(pr), int(pc)))
        cr, cc = int(pr), int(pc)
    path.reverse()

    trail_positions = [path[i] for i in range(period, len(path) - 1, period)]
    return {
        "dest": dest,
        "period": int(period),
        "branch_feasible": branch_feasible,
        "dist": dist,
        "parent": parent,
        "path": path,
        "trail_positions": trail_positions,
        "tr": int(tr),
        "tc": int(tc),
        "reached": reached,
    }


def _scene_state(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    mover, dests, child_n = _pick_roles(g, bg)
    river = _river_color(g, bg, mover, dests)

    mover_color = int(mover["color"])
    dest_colors = {int(d["color"]) for d in dests}
    trav = (g == river) | (g == mover_color)
    for color in dest_colors:
        trav |= g == color

    big_n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    big_feasible = _feasible_windows(trav, big_n)
    assert bool(big_feasible[sr, sc])
    big_reach, _big_parent = _bfs_with_parent(big_feasible, [(sr, sc)])
    big_cover = _reachable_cover(big_reach, big_n, (h, w))

    arms = _arm_components(trav, big_cover, dests, child_n)
    branches = []
    for idx, (arm, dest) in enumerate(zip(arms, dests)):
        branches.append(_branch_state(trav, big_cover, arm, child_n, dest, _PERIODS[idx]))

    return {
        "grid": g,
        "bg": bg,
        "river": river,
        "mover": mover,
        "dests": dests,
        "child_n": int(child_n),
        "trav": trav,
        "big_cover": big_cover,
        "branches": branches,
    }


def _render_output(state: dict, *, with_trails: bool) -> np.ndarray:
    g = state["grid"]
    river = int(state["river"])
    mover_color = int(state["mover"]["color"])
    big_n = int(state["mover"]["n"])
    sr, sc = int(state["mover"]["r0"]), int(state["mover"]["c0"])
    child_n = int(state["child_n"])

    out = g.copy()
    out[sr : sr + big_n, sc : sc + big_n] = river
    for branch in state["branches"]:
        if with_trails:
            for rr, cc in branch["trail_positions"]:
                out[rr : rr + child_n, cc : cc + child_n] = mover_color
        tr, tc = int(branch["tr"]), int(branch["tc"])
        out[tr : tr + child_n, tc : tc + child_n] = mover_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    state = _scene_state(np.array(grid, dtype=int))
    return _render_output(state, with_trails=True)


def _analyze_instance(grid: np.ndarray) -> dict:
    state = _scene_state(grid)
    combo_out = _render_output(state, with_trails=True)
    split_only_out = _render_output(state, with_trails=False)
    return {
        "state": state,
        "combo_out": combo_out,
        "split_only_out": split_only_out,
        "arm_count": len(state["branches"]),
        "stamp_counts": [len(branch["trail_positions"]) for branch in state["branches"]],
        "reached_children": sum(1 for branch in state["branches"] if branch["reached"]),
        "blocked_children": sum(1 for branch in state["branches"] if not branch["reached"]),
        "path_lengths": [len(branch["path"]) - 1 for branch in state["branches"]],
        "outputs_differ_from_split_only": not np.array_equal(combo_out, split_only_out),
    }


def _gen_attempt() -> np.ndarray | None:
    big_n, child_n = _BIG, _SMALL
    gsize = random.randint(19, 20)
    grid = np.full((gsize, gsize), _BG, dtype=int)

    jr = random.randint(8, gsize - big_n - 3)
    jc = random.randint(6, gsize - big_n - 7)
    left_root = (jr, jc)
    right_root = (jr, jc + big_n - child_n)

    top1 = random.randint(1, jr - 5)
    top2 = random.randint(1, jr - 5)
    end1c = random.randint(1, jc - 3)
    end2c = random.randint(right_root[1] + 3, gsize - child_n - 2)

    sr = gsize - big_n - random.randint(1, 2)
    sc = random.randint(1, min(3, jc - 4))
    midc = random.randint(sc + 2, jc - 2)

    trunk = _dense_path([(sr, sc), (sr, midc), (jr, midc), (jr, jc)])
    left_path = _dense_path([left_root, (top1, jc), (top1, end1c)])
    right_path = _dense_path([right_root, (top2, right_root[1]), (top2, end2c)])

    if len(left_path) - 1 < 8 or len(right_path) - 1 < 9:
        return None

    _paint_path(grid, trunk, big_n, _RIVER)
    _paint_path(grid, left_path, child_n, _RIVER)
    _paint_path(grid, right_path, child_n, _RIVER)

    block_case = random.choice(["none", "left", "right", "both"])
    if block_case in ("left", "both"):
        if not _narrow_vertical_corridor(grid, jc, top1, jr, child_n, min_open_steps=5):
            return None
    if block_case in ("right", "both"):
        if not _narrow_vertical_corridor(grid, right_root[1], top2, jr, child_n, min_open_steps=6):
            return None

    grid[sr : sr + big_n, sc : sc + big_n] = _MOVER
    grid[top1 : top1 + child_n, end1c : end1c + child_n] = _DEST
    grid[top2 : top2 + child_n, end2c : end2c + child_n] = _DEST

    try:
        metrics = _analyze_instance(grid)
    except Exception:
        return None

    if metrics["arm_count"] != 2:
        return None
    if metrics["stamp_counts"][0] < 2 or metrics["stamp_counts"][1] < 1:
        return None
    if sum(metrics["stamp_counts"]) < 3:
        return None
    if not metrics["outputs_differ_from_split_only"]:
        return None

    combo_out = metrics["combo_out"]
    if not np.array_equal(combo_out, main(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo input after 600 attempts")
