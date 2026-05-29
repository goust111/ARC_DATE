# Combo variant of 7b5033c1 (index 12)
# combo_spec: D4/3+D7/1
# Name: 双色路径+噪声色块

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.insert(0, str(Path("/home/lixinyu/snap/new/arc2_solver_select")))
    from arc_utils import get_bg


# Design analysis for the combo variant:
# a) 具体化与行为追踪:
#    原题里，所有非背景像素都属于同一条多色蛇，`main()` 直接把整张图的非背景像素当作一条链来找端点。
#    本融合版里，输入由两部分组成:
#    - 一条 20-30 像素的 4 连通蛇，沿路径顺序只使用两种颜色，并按连续色段 A-B-A-B... 交替。
#    - 2-4 个独立噪声色块，每块 2-3 像素，颜色从这两种路径色里抽取，因此颜色不能用来排除噪声。
#    具体例子: 假设蛇的路径色顺序为 [2,2,2,8,8,2,2,8,8,8]，左上角再放一个颜色为 2 的 3 像素小 L 块。
#    原题代码会把蛇和噪声一起视为前景，`ends` 会混入噪声端点，随后从全局最左上的噪声端点起步，
#    最终只走出 2-3 个像素，输出错误。融合后必须先按 4 连通分量拆开，再找同时满足
#    “几何上是长链”与“颜色上是双色交替分段”的那个分量，最后仍按该链的左上端点方向输出颜色列。
# b) 复杂度校准:
#    仅仅把路径改成双色会太弱，因为如果没有重复分段，人类只会看到“一条双色蛇”。因此生成器强制 4-6 段，
#    每段 2-9 像素，总长 20-30，保证 A/B 两色在样例中多次重复，肉眼能看出“同两色反复交替”的结构。
#    仅仅加噪声也会太弱，如果噪声总在蛇下方或颜色不同，解题者仍可沿用原题直觉。这里噪声颜色与路径重合，
#    并强制至少一个噪声块位于蛇左上方，让“先剔除干扰分量”在样例里稳定可见。
# c) 隐式假设审查:
#    原 `mask = g != bg` 假设所有前景都属于目标蛇；融合后失效，因为噪声也前景。
#    原 `pts = argwhere(mask)` 后的 `deg` 统计假设全局只有一条链；融合后会得到多个独立分量。
#    原 `start = min(ends)` 假设全局最左上端点就是答案起点；融合后这个端点可能来自噪声块。
#    原跟踪循环还依赖“当前像素未访问邻居唯一”；该前提只对目标蛇成立，不对整个前景集合成立。
#    新 solver 因此改成:
#    1) 先做分量分解。
#    2) 在每个分量内部检查是否为简单链。
#    3) 只接受颜色序列满足“恰有两色、3-6 个连续色段、色段颜色交替、每段长 2-9”的链分量。
#    4) 在候选里取最大的那条。这样 D4 的颜色约束和 D7 的分量约束在同一选择步骤里协同工作。
# d) 放置合法性:
#    路径生长时始终限制在内圈，且新增像素除前驱外不能与已有路径产生额外 4 邻接，从而保持简单链。
#    噪声块放置时检查越界、重叠，并使用 8 邻域缓冲避免与路径或其他噪声贴边，保证视觉上清爽且分量分离。
# e) 视觉可观察性:
#    解题者在 2-4 个样例里会反复看到:
#    - 图里只有两种前景颜色，但输出不是把所有同色像素聚合，而是沿某条蛇按顺序拉直。
#    - 左上角常有同色小块，却从不出现在输出里。
#    - 蛇身颜色不是随机散点，而是长短不等的 A/B/A/B 连续段。
#    这些现象共同指向“忽略小块，只读长蛇”的规则。
# f) 可视化整洁度:
#    生成器让目标蛇从中上区域开始生长，锚定噪声块固定在更左上的区域，其余噪声块也保留空隙；
#    因此图面上不会出现多个对象粘连成团，也不会靠颜色数堆出混乱观感。
# g) 颜色变化可推断性:
#    颜色变化没有引入隐藏映射，只是把原来的 3-5 色分段压缩成两色交替分段。输出仍然逐像素保留原色；
#    人类只需推断“沿路径顺序读取”，无需猜测颜色变换函数。
# 新 solver 的额外输入假设审查:
#    - 存在唯一一个分量同时满足“简单链 + 双色交替分段”；`generate_input()` 直接构造并验证该条件。
#    - 该目标分量长度显著大于噪声块；生成器固定蛇长 20-30、噪声块长 2-3。
#    - 噪声块即使使用路径同色，也不会自己形成 3-6 段双色交替序列；生成器把每个噪声块做成单色。


Coord = tuple[int, int]
NEI4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NOISE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _connected_components(mask: np.ndarray) -> list[list[Coord]]:
    """Return 4-connected foreground components."""
    H, W = mask.shape
    seen: set[Coord] = set()
    comps: list[list[Coord]] = []
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[Coord] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in NEI4:
                nr, nc = cr + dr, cc + dc
                nxt = (nr, nc)
                if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _chain_order(comp: list[Coord]) -> list[Coord] | None:
    """Trace a component from its top-left endpoint if it is a simple chain."""
    pset = set(comp)
    deg = {
        cell: sum((cell[0] + dr, cell[1] + dc) in pset for dr, dc in NEI4)
        for cell in comp
    }
    ends = sorted(cell for cell in comp if deg[cell] == 1)
    if len(ends) != 2:
        return None
    if any(deg[cell] not in (1, 2) for cell in comp):
        return None

    order: list[Coord] = []
    prev: Coord | None = None
    cur = ends[0]
    for _ in range(len(comp)):
        order.append(cur)
        nbrs = []
        for dr, dc in NEI4:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in pset and nxt != prev:
                nbrs.append(nxt)
        if not nbrs:
            break
        if len(nbrs) != 1:
            return None
        prev, cur = cur, nbrs[0]

    return order if len(order) == len(comp) else None


def _color_runs(colors: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for col in colors:
        if runs and runs[-1][0] == col:
            runs[-1] = (col, runs[-1][1] + 1)
        else:
            runs.append((col, 1))
    return runs


def _matches_two_color_segment_rule(colors: list[int]) -> bool:
    """Check the fused D4 structure on a traced chain."""
    palette = sorted(set(colors))
    if len(palette) != 2:
        return False

    runs = _color_runs(colors)
    if not (3 <= len(runs) <= 6):
        return False
    if any(not (2 <= run_len <= 9) for _, run_len in runs):
        return False
    if any(runs[i][0] == runs[i - 1][0] for i in range(1, len(runs))):
        return False
    return True


def _select_snake(g: np.ndarray, comps: list[list[Coord]]) -> list[int]:
    """Pick the unique large chain component whose colors follow the two-color segment rule."""
    candidates: list[tuple[int, list[int]]] = []
    fallback_chains: list[tuple[int, list[int]]] = []

    for comp in comps:
        order = _chain_order(comp)
        if order is None:
            continue
        colors = [int(g[r, c]) for r, c in order]
        fallback_chains.append((len(comp), colors))
        if _matches_two_color_segment_rule(colors):
            candidates.append((len(comp), colors))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    if fallback_chains:
        return max(fallback_chains, key=lambda item: item[0])[1]

    largest = max(comps, key=len)
    return [int(g[r, c]) for r, c in sorted(largest)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule:
    - Foreground may contain detached 2-3 pixel noise blobs in the same two colors as the path.
    - The real object is the unique long chain whose traced colors form alternating two-color runs.
    - Output that chain's colors, read from its top-left endpoint, as an (N x 1) column.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = _connected_components(g != bg)
    if not comps:
        return np.zeros((0, 1), dtype=int)
    colors = _select_snake(g, comps)
    return np.array(colors, dtype=int).reshape((-1, 1))


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    if num_parts * min_len > total or num_parts * max_len < total:
        return None
    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i in range(num_parts) if parts[i] < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def _try_extend(
    path_set: set[Coord],
    last: Coord,
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[Coord]:
    """Extend a simple path without creating overlaps or extra 4-neighbors."""
    segment: list[Coord] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in path_set:
            break
        pred = (r + dr * (i - 1), c + dc * (i - 1))
        if any((nr + ddr, nc + ddc) in path_set and (nr + ddr, nc + ddc) != pred for ddr, ddc in NEI4):
            break
        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _grow_path(H: int, W: int, target_len: int) -> list[Coord] | None:
    for _ in range(240):
        start = (random.randint(3, 6), random.randint(4, W - 5))
        path = [start]
        path_set = {start}
        turns = 0
        is_vertical = True
        h_dir = random.choice([-1, 1])

        while len(path) < target_len:
            run_len = min(random.randint(1, 5), target_len - len(path))
            last = path[-1]
            segment: list[Coord] = []

            if is_vertical:
                dirs = [1, -1] if random.random() < 0.75 else [-1, 1]
                for v_dir in dirs:
                    segment = _try_extend(path_set, last, v_dir, 0, run_len, H, W)
                    if segment:
                        break
            else:
                for cand in (h_dir, -h_dir):
                    segment = _try_extend(path_set, last, 0, cand, run_len, H, W)
                    if segment:
                        h_dir = -cand
                        break

            if not segment:
                break

            path.extend(segment)
            turns += 1
            is_vertical = not is_vertical

        if 20 <= len(path) <= 30 and 7 <= turns <= 11:
            return path
    return None


def _orient_path_top_left(path: list[Coord]) -> list[Coord]:
    return path if path[0] <= path[-1] else list(reversed(path))


def _path_colors(total: int, palette: list[int]) -> list[int] | None:
    num_segments = random.randint(4, 6)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=9)
    if seg_lengths is None:
        return None
    first = random.choice([0, 1])
    seg_colors = [palette[(first + i) % 2] for i in range(num_segments)]
    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)
    return colors


def _blocked8(cells: list[Coord], H: int, W: int) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out.add((nr, nc))
    return out


def _can_place_blob(cells: list[Coord], blocked: set[Coord], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W and (r, c) not in blocked for r, c in cells)


def _place_blob(
    grid: np.ndarray,
    blocked: set[Coord],
    shape: tuple[tuple[int, int], ...],
    base: Coord,
    color: int,
) -> tuple[bool, set[Coord]]:
    H, W = grid.shape
    cells = [(base[0] + dr, base[1] + dc) for dr, dc in shape]
    if not _can_place_blob(cells, blocked, H, W):
        return False, blocked
    for r, c in cells:
        grid[r, c] = color
    blocked = blocked | _blocked8(cells, H, W)
    return True, blocked


def _place_noise_blobs(grid: np.ndarray, snake_cells: list[Coord], palette: list[int]) -> bool:
    """Place detached single-color noise blobs; one is forced into the top-left region."""
    H, W = grid.shape
    blocked = _blocked8(snake_cells, H, W)

    anchor_shape = random.choice(NOISE_SHAPES[2:])
    anchor_color = random.choice(palette)
    anchor_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(anchor_positions)
    placed, blocked_after = False, blocked
    for base in anchor_positions:
        placed, blocked_after = _place_blob(grid, blocked, anchor_shape, base, anchor_color)
        if placed:
            blocked = blocked_after
            break
    if not placed:
        return False

    num_extra = random.randint(1, 3)
    for _ in range(num_extra):
        ok = False
        for _ in range(120):
            shape = random.choice(NOISE_SHAPES)
            max_r = max(dr for dr, _ in shape)
            max_c = max(dc for _, dc in shape)
            base_r = random.randint(0, H - max_r - 1)
            base_c = random.randint(0, W - max_c - 1)
            color = random.choice(palette)
            ok, blocked_after = _place_blob(grid, blocked, shape, (base_r, base_c), color)
            if ok:
                blocked = blocked_after
                break
        if not ok:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    palette = random.sample([c for c in range(10) if c != bg], 2)

    target_len = random.randint(20, 30)
    raw_path = _grow_path(H, W, target_len)
    if raw_path is None:
        return None
    path = _orient_path_top_left(raw_path)
    colors = _path_colors(len(path), palette)
    if colors is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    if not _place_noise_blobs(grid, path, palette):
        return None

    comps = _connected_components(grid != bg)
    comp_sizes = sorted((len(comp) for comp in comps), reverse=True)
    if len(comp_sizes) < 3 or comp_sizes[0] != len(path) or comp_sizes[1] > 3:
        return None

    snake_colors = main(grid)
    expected = np.array(colors, dtype=int).reshape((-1, 1))
    if not np.array_equal(snake_colors, expected):
        return None

    # Anti-degeneracy check: the original clean-task solver must fail on this fused input.
    all_pts = [(int(r), int(c)) for r, c in np.argwhere(grid != bg)]
    pset = set(all_pts)
    deg = {
        cell: sum((cell[0] + dr, cell[1] + dc) in pset for dr, dc in NEI4)
        for cell in all_pts
    }
    old_ends = [cell for cell in all_pts if deg[cell] == 1]
    if old_ends:
        cur = min(old_ends)
        prev: Coord | None = None
        old_colors: list[int] = []
        for _ in range(len(all_pts)):
            old_colors.append(int(grid[cur]))
            nbrs = []
            for dr, dc in NEI4:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in pset and nxt != prev:
                    nbrs.append(nxt)
            if not nbrs:
                break
            cur, prev = nbrs[0], cur
        old_out = np.array(old_colors, dtype=int).reshape((-1, 1))
        if np.array_equal(old_out, expected):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input after 600 attempts")
