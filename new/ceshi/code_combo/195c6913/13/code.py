# Combo variant of 195c6913 (index 13)
# combo_spec: D1/3+D3/4
# Name: 螺旋路径+距离决定颜色
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题输入里有三类显眼对象：左上模板 2x2 色块、单像素 marker、远处 2x2 终止块。
#      这里保留这三类对象，但把原来的“粗对角带 + 碰撞转弯”改成“一条细的矩形螺旋走廊”。
#    - 小例子：若模板是 [A,B,C]，原题从 marker 出发会沿直线前进，按步数循环
#      B,C,A,B...；撞到非 path 后才按 CCW/CW 规则换方向。
#      融合后 marker 放在螺旋端点，粒子不再等待碰撞决策，而是沿唯一可继续的螺旋走廊一直走。
#      每个访问格的颜色不看“这是第几步”，而看它到 marker 的曼哈顿距离 d，并写成
#      pattern[d % 3]。因此相邻圈中与起点等距的格子会共享颜色，出现环带。
#    - 与原题差异：
#      1. 运动控制从“射线 + 碰撞时交替转向”变成“由螺旋几何直接决定的连续转弯”。
#      2. 颜色控制从“步进循环”变成“几何距离分层”。
#
# b) 复杂度校准
#    - 为了让 D1 可见，path 不能只是厚色块内部的一条隐式顺序；所以生成器直接造 1 格宽、
#      无分叉的螺旋走廊，肉眼能立即看出“这是一条螺旋路”。
#    - 为了让 D3 可见，生成器固定使用 outward 版本：marker 放在螺旋内端。这样距离层围绕
#      marker 向外展开，真的会形成圈层色带；若从外角往里走，曼哈顿距离更像对角渐变，
#      反而削弱融合效果。
#    - 模板色数限制在 3-4，且互不重复，避免“颜色太多难归纳”或“重复太多看不出距离层”。
#
# c) 隐式假设审查
#    - 原 solver 的 path 假设是“单个粗带 + 进入 path 后按当前方向直走，撞墙再转”，这些前提
#      在螺旋走廊里全部失效，所以相关的初始方向优先级、碰撞计数、交替左/右转逻辑都必须删除。
#    - 原代码还假设“颜色索引 = 已走步数 mod K”；融合后必须改成
#      `曼哈顿距离(start, cell) mod K`，否则会退化回原题的循环色条。
#    - 新 solver 的关键新假设是：path_color 形成一条 4 连通、每个格子度数 <= 2 的简单路径，
#      marker 覆盖其中一个端点。生成器会显式保证这一点。
#
# d) 放置合法性
#    - 模板仍固定放在左上安全区域；螺旋矩形整体放到其下方；终止块放在右下留白区，三者互不重叠。
#    - marker 覆盖螺旋内端点，因此一定在网格内，且只与一格 path 相邻，不会出现分叉或歧义起点。
#    - 终止块与 path 至少隔 1 格背景，擦除后不会和轨迹混在一起。
#
# e) 视觉可观察性
#    - 输入里直接能看到一条明显的螺旋走廊和位于其中心附近的 marker，说明“运动路径”已改变。
#    - 输出里相隔一圈的若干格会复现同色，形成同心式色带；这和原题那种沿路逐步轮换的条纹不同。
#    - 颜色映射只依赖可见几何距离，不依赖颜色 id 或不可见排序。
#
# f) 可视化整洁度
#    - 全图只有一条单线螺旋，没有多条轨迹叠涂，也没有原题的碰撞点噪声，读图更干净。
#    - 模板块和终止块在输出中被统一擦除，避免保留无关残留。
#
# g) 变化的可推断性
#    - 生成器要求至少 6 个不同距离层、至少 6 次同向转弯，并要求“距离着色”与“按步循环着色”
#      在同一样本上有足够多的不一致格，防止退化成旧规则外观。
#    - 还要求某些距离值在路径上出现多次，这样人眼能看到“同一距离 -> 同一颜色”的重复证据。

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import label_mask

DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    H, W = shape
    out: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < H and 0 <= cc < W:
            out.append((int(rr), int(cc)))
    return out


def _find_template(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    H, W = g.shape
    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            rc = (int(r), int(c))
            if rc in seen:
                continue
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color):
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                pattern.append(v)
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))
    return pattern, pat_cells


def _find_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pattern: list[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                return int(v), (int(r), int(c))
    return None, None


def _find_markers(g: np.ndarray, path_color: int, pattern: list[int]) -> list[tuple[int, int, int]]:
    H, W = g.shape
    pat_set = set(map(int, pattern))
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for rr, cc in _neighbors4((H, W), r, c):
                if int(g[rr, cc]) == v:
                    same += 1
                if int(g[rr, cc]) == int(path_color):
                    has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))
    return markers


def _assert_simple_path(mask: np.ndarray) -> None:
    _, n_comp = label_mask(mask, connectivity=4)
    assert n_comp == 1, f"path_color must form one component, got {n_comp}"
    H, W = mask.shape
    endpoint_count = 0
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]):
                continue
            deg = sum(1 for rr, cc in _neighbors4((H, W), r, c) if bool(mask[rr, cc]))
            assert deg in (1, 2), f"path cell ({r},{c}) has degree {deg}, expected 1 or 2"
            if deg == 1:
                endpoint_count += 1
    assert endpoint_count == 2, f"path must have exactly 2 endpoints, got {endpoint_count}"


def _recover_spiral_walk(
    g: np.ndarray, path_color: int, marker: tuple[int, int, int]
) -> list[tuple[int, int]]:
    H, W = g.shape
    sr, sc, _ = marker
    start_neighbors = [
        (rr, cc) for rr, cc in _neighbors4((H, W), sr, sc) if int(g[rr, cc]) == int(path_color)
    ]
    assert len(start_neighbors) == 1, (
        f"marker at ({sr},{sc}) must touch exactly one path cell, got {len(start_neighbors)}"
    )

    total_path_cells = int(np.sum(g == path_color))
    prev = (int(sr), int(sc))
    cur = start_neighbors[0]
    visited: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []

    while True:
        assert cur not in visited, f"path loop detected at {cur}"
        visited.add(cur)
        ordered.append(cur)

        next_cells = [
            (rr, cc)
            for rr, cc in _neighbors4((H, W), cur[0], cur[1])
            if int(g[rr, cc]) == int(path_color) and (rr, cc) != prev
        ]
        next_unseen = [cell for cell in next_cells if cell not in visited]
        if not next_unseen:
            break
        assert len(next_unseen) == 1, f"path branches at {cur}: {next_unseen}"
        prev, cur = cur, next_unseen[0]

    assert len(ordered) == total_path_cells, (
        f"recovered {len(ordered)} path cells, expected {total_path_cells}"
    )
    return ordered


def _turn_signs(points: list[tuple[int, int]]) -> list[int]:
    dirs = [
        (int(r2 - r1), int(c2 - c1))
        for (r1, c1), (r2, c2) in zip(points, points[1:])
    ]
    signs: list[int] = []
    for (dr1, dc1), (dr2, dc2) in zip(dirs, dirs[1:]):
        if (dr1, dc1) == (dr2, dc2):
            continue
        cross = int(dr1 * dc2 - dc1 * dr2)
        assert cross != 0, "path must stay axis-aligned"
        signs.append(cross)
    return signs


def _scene_from_grid(
    grid: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
    int,
    list[int],
    list[tuple[int, int]],
    int | None,
    tuple[int, int] | None,
    tuple[int, int, int],
    list[tuple[int, int]],
]:
    g = np.asarray(grid, dtype=int)
    template_bg = int(g[0, 0])
    counts = Counter(map(int, g.flatten()))
    non_bg = [(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)]
    assert non_bg, "grid must contain non-background colors"
    path_color = int(max(non_bg)[1])

    pattern, pat_cells = _find_template(g, template_bg, path_color)
    assert len(pattern) >= 2, "need at least 2 template colors"
    end_color, end_pos = _find_end_block(g, template_bg, path_color, pattern)
    markers = _find_markers(g, path_color, pattern)
    assert len(markers) == 1, f"expected exactly 1 marker, got {len(markers)}"

    _assert_simple_path(g == path_color)
    path = _recover_spiral_walk(g, path_color, markers[0])
    turn_signs = _turn_signs([(markers[0][0], markers[0][1]), *path])
    assert len(turn_signs) >= 3, "spiral must turn several times"
    assert len(set(turn_signs)) == 1, f"spiral turns must keep one handedness, got {set(turn_signs)}"

    return g, template_bg, path_color, pattern, pat_cells, end_color, end_pos, markers[0], path


def _paint_distance_spiral(
    g: np.ndarray,
    template_bg: int,
    pattern: list[int],
    pat_cells: list[tuple[int, int]],
    end_pos: tuple[int, int] | None,
    marker: tuple[int, int, int],
    path: list[tuple[int, int]],
) -> np.ndarray:
    out = g.copy()
    for r, c in pat_cells:
        out[r, c] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[er : er + 2, ec : ec + 2] = int(template_bg)

    sr, sc, _ = marker
    k = len(pattern)
    for r, c in path:
        dist = abs(int(r) - int(sr)) + abs(int(c) - int(sc))
        out[r, c] = int(pattern[dist % k])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow a visible spiral corridor and recolor it by Manhattan distance from the start marker.

    Input structure:
    - The top-left row of 2x2 monochrome blocks still defines the usable palette.
    - One isolated palette-colored marker sits on a spiral endpoint, touching exactly one path cell.
    - A separate 2x2 monochrome block of another color remains present as a removable terminal cue.

    Rule:
    - Erase the template blocks and the separate 2x2 terminal block back to the template background.
    - Start from the marker and traverse the unique 1-cell-wide spiral path until the far endpoint.
    - For each visited path cell, compute its Manhattan distance to the marker and paint
      `pattern[distance % len(pattern)]`.
    - The marker itself already represents distance 0, so it is kept unchanged.

    Key insight: Motion is no longer produced by straight rays and collision-time left/right decisions;
    the spiral geometry itself determines the walk, and color depends on geometric distance from the
    start rather than on path step order.
    """
    (
        g,
        template_bg,
        _path_color,
        pattern,
        pat_cells,
        _end_color,
        end_pos,
        marker,
        path,
    ) = _scene_from_grid(grid)
    return _paint_distance_spiral(g, template_bg, pattern, pat_cells, end_pos, marker, path)


def _spiral_path_tl(height: int, width: int, clockwise: bool) -> list[tuple[int, int]]:
    assert height >= 5 and width >= 5
    if clockwise:
        top, left, bottom, right = 0, 0, height - 1, width - 1
        r, c = top, left
        path = [(int(r), int(c))]
        while True:
            for nc in range(c + 1, right + 1):
                path.append((int(r), int(nc)))
            c = right
            top += 2
            if top > bottom or left > right:
                break
            for nr in range(r + 1, bottom + 1):
                path.append((int(nr), int(c)))
            r = bottom
            right -= 2
            if top > bottom or left > right:
                break
            for nc in range(c - 1, left - 1, -1):
                path.append((int(r), int(nc)))
            c = left
            bottom -= 2
            if top > bottom or left > right:
                break
            for nr in range(r - 1, top - 1, -1):
                path.append((int(nr), int(c)))
            r = top
            left += 2
            if top > bottom or left > right:
                break
        return path

    top, left, bottom, right = 0, 0, height - 1, width - 1
    r, c = top, left
    path = [(int(r), int(c))]
    while True:
        for nr in range(r + 1, bottom + 1):
            path.append((int(nr), int(c)))
        r = bottom
        left += 2
        if top > bottom or left > right:
            break
        for nc in range(c + 1, right + 1):
            path.append((int(r), int(nc)))
        c = right
        bottom -= 2
        if top > bottom or left > right:
            break
        for nr in range(r - 1, top - 1, -1):
            path.append((int(nr), int(c)))
        r = top
        right -= 2
        if top > bottom or left > right:
            break
        for nc in range(c - 1, left - 1, -1):
            path.append((int(r), int(nc)))
        c = left
        top += 2
        if top > bottom or left > right:
            break
    return path


def _fusion_metrics(
    marker: tuple[int, int],
    path: list[tuple[int, int]],
    palette_len: int,
) -> dict[str, int]:
    dists = [abs(int(r) - int(marker[0])) + abs(int(c) - int(marker[1])) for r, c in path]
    turns = _turn_signs([marker, *path])
    mismatch_vs_step_cycle = sum(
        1 for step, dist in enumerate(dists, start=1) if (step % palette_len) != (dist % palette_len)
    )
    max_repeat = max(Counter(dists).values()) if dists else 0
    return {
        "path_len": len(path),
        "turns": len(turns),
        "distinct_distances": len(set(dists)),
        "max_distance": max(dists) if dists else 0,
        "max_repeat_per_distance": int(max_repeat),
        "mismatch_vs_step_cycle": int(mismatch_vs_step_cycle),
    }


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(22, 30)
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(3, 4)
    pattern = random.sample(palette[3:], n_blocks)

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    if block_cols[-1] + 1 >= W:
        return None

    top = random.randint(5, 7)
    max_height = H - top - 4
    max_width = W - 5
    if max_height < 11 or max_width < 13:
        return None
    height = random.randint(11, max_height)
    width = random.randint(13, max_width)
    clockwise = bool(random.getrandbits(1))

    local_path = _spiral_path_tl(height, width, clockwise)
    path = [(int(top + r), int(c)) for r, c in local_path]

    g = np.full((H, W), bg, dtype=int)
    for r, c in path:
        g[r, c] = int(path_color)

    for bc, col in zip(block_cols, pattern):
        if np.any(g[template_row : template_row + 2, bc : bc + 2] != bg):
            return None
        g[template_row : template_row + 2, bc : bc + 2] = int(col)

    er, ec = H - 3, W - 3
    if np.any(g[er : er + 2, ec : ec + 2] != bg):
        return None
    r0 = max(0, er - 1)
    c0 = max(0, ec - 1)
    if np.any(g[r0 : er + 3, c0 : ec + 3] == path_color):
        return None
    g[er : er + 2, ec : ec + 2] = int(end_color)

    marker = path[-1]
    g[marker] = int(pattern[0])

    path_mask = g == path_color
    try:
        _assert_simple_path(path_mask)
    except Exception:
        return None

    metrics = _fusion_metrics(marker, path[:-1][::-1], len(pattern))
    # `path[:-1][::-1]` are the actual visited path_color cells starting from the marker's only neighbor.
    if metrics["turns"] < 6:
        return None
    if metrics["distinct_distances"] < 6:
        return None
    if metrics["max_repeat_per_distance"] < 3:
        return None
    if metrics["mismatch_vs_step_cycle"] < max(8, metrics["path_len"] // 5):
        return None

    # Oracle + regression check through the solver.
    try:
        out = main(g)
    except Exception:
        return None

    if int(np.sum(out == path_color)) >= int(np.sum(g == path_color)):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")
