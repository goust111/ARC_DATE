# Combo variant of d35bdbdc (index 19)
# combo_spec: D7/1+D8/2
# Name: 环形路径+路径着色为目标边框色

"""
Design analysis:
  variant_goal:
    old:
      - 原题只有一条开放灰路径, 依靠两个端点各自选出 2 个目标精灵
      - 输出里灰路径保持灰色不变
    new:
      - 灰路径改成单个闭合环, 不再存在端点
      - 所有通过合法端口接到环上的精灵都是目标精灵
      - 输出里整条环不再保留灰色, 而是按“离哪个目标端口最近”染成该目标的边框色
  concrete_visualization:
    objects:
      - 10x10 网格, 背景 0, 输入中的环颜色固定为灰色 5
      - 所有精灵仍是同形状的 3x3 bordered sprite, 形状为 box 或 plus
      - 每个精灵只有两种颜色: 多像素 border_color + 唯一 center_color
    fused_layout:
      - 生成器固定构造中央矩形环, 环长 20
      - 四个目标精灵固定分布在环的上/右/下/左四侧, 每个都有一个唯一合法端口连到环
      - 其它 source / distractor 只放四角安全槽位, 避免意外接环
    output_effect:
      - 四个接环目标都保留, 非目标精灵全部删除
      - 每个目标仍按“目标原中心色 -> source.border -> source.center”更新中心
      - 更新完成后, 环上的每个像素改成最近目标端口对应的边框色, 形成 4 段彩色环
  behavior_trace:
    small_example:
      - 假设上/右/下/左四个目标的边框色依次是蓝/红/黄/青
      - 输入中中央是一圈灰环, 四个目标中心色则重复引用 1-2 个 source 的边框色
      - 原题逻辑会先找两个端点, 这里会直接失效, 因为闭环没有端点
      - 新逻辑先枚举所有“中心外隔一格且中间是本精灵边框色”的灰端口, 得到 4 个目标
      - 然后对每个目标做颜色查表, 最后把环按最近端口切成 4 个连续色段
    difference_point:
      - 目标选择从“两个端点”变成“整条环上的所有合法端口”
      - 路径处理从“保留灰色”变成“输出中完全吸收到目标边框色里”
  complexity_calibration:
    - 目标数固定为 4, 明显区别于原题的 2 个目标, 也比 3 个目标更稳定
    - source 只保留 1-2 个, 让多个目标重复引用同一 source.border, 人眼容易看到“查表复制”
    - 环长固定 20 且四个端口在环上的间距都是 5, 每段最近目标唯一, 避免隐藏 tie-break
  implicit_assumption_review:
    old_code_dependencies:
      - 原 solver 假设灰路径有两个端点
      - 原 solver 只需检查端点是否是某个 sprite 的 port
      - 原 generator 只需要构造两端口之间的一条开放 trail
    fused_replacement:
      - solver 改为验证灰像素构成单个 4 邻域简单闭环
      - 目标识别改为“凡是存在灰环端口的精灵都是目标”
      - 路径着色依赖环的循环顺序, 因此需要显式恢复环顺序并按环距离着色
    new_assumptions:
      - 每个目标恰有一个可见端口连到环
      - 四个目标全部存在, 且端口刚好把 20 格环分成 4 段等长奇间隔, 因而最近目标唯一
      - source.border 不会出现在目标边框里, 所以颜色查表时可明确过滤非目标 source
    generator_guarantees:
      - 四侧目标使用预定义槽位, 端口唯一且合法
      - 角落 extras 不会与环重叠, 也不会形成新的合法端口
      - 清理非目标后再着色, 避免把已着色环误删为普通前景
  placement_legality:
    - 中央环与四侧目标之间始终隔着恰好一个边框像素, 对 box / plus 都合法
    - 角落安全槽位之间互不重叠, 放置时仍做 overlap 和 4 邻接检查
    - 所有 patch 即使贴边裁切, 中心仍在网格内, 满足 bordered sprite 检测条件
  observability:
    - 输入里最显眼的新结构是“灰色闭环 + 四个接环目标”
    - 输出里最显眼的新结构是“灰环完全消失, 改成 4 段目标边框色的彩色环”
    - 中心色转移仍保留原题主线, 环着色则提供第二条强视觉线索, 两者彼此支撑
  cleanliness:
    - 环固定在中部, 目标固定在四侧, sources/distractors 收到四角, 区域分层明确
    - 环颜色只使用 surviving target 的边框色, 不引入额外无来源颜色
  inferability_of_color_change:
    - 颜色变化不是任意映射, 而是严格由“最近的接环目标”决定
    - 四条连续色段都直接贴着对应目标, 人眼能从少量样例中归纳出吸附关系
"""

from collections import deque
import random

import numpy as np

from arc_utils import components_mask, neighbors4, paste


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
RING_BOUNDS = (2, 2, 7, 7)
NUM_DEST = 4
NUM_SOURCE_CHOICES = [1, 2]
DISTRACTOR_CHOICES = [0, 1]
GEN_RETRIES = 400

DEST_SLOTS = [
    {"name": "top", "center": (0, 4), "port": (2, 4), "r0": -1, "c0": 3},
    {"name": "right", "center": (4, 9), "port": (4, 7), "r0": 3, "c0": 8},
    {"name": "bottom", "center": (9, 5), "port": (7, 5), "r0": 8, "c0": 4},
    {"name": "left", "center": (5, 0), "port": (5, 2), "r0": 4, "c0": -1},
]

EXTRA_SLOTS = [
    (-1, 0),
    (0, -1),
    (-1, 7),
    (0, 8),
    (8, 0),
    (7, -1),
    (8, 7),
    (7, 8),
]


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
    Clipped edge sprites are allowed.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color in (bg, grey) or center_color == grey:
            continue

        center_pos = None
        for r, c in pts:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
        if center_pos is None:
            continue

        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _grey_cells(g: np.ndarray, grey: int) -> set[tuple[int, int]]:
    return set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))


def _ordered_loop_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Recover the cyclic order of a simple 4-connected loop."""
    assert cells, "expected a non-empty grey loop"

    adj = {
        cell: [nb for nb in neighbors4(*cell) if nb in cells]
        for cell in cells
    }
    assert all(len(nbs) == 2 for nbs in adj.values()), "expected every loop cell to have 4-degree 2"

    start = min(cells)
    seen = {start}
    ordered = [start]
    prev = None
    cur = start

    while True:
        nxts = [nb for nb in adj[cur] if nb != prev]
        if prev is None:
            nxt = min(nxts)
        else:
            nxt = nxts[0]
        if nxt == start:
            break
        assert nxt not in seen, "grey loop self-intersection is not supported"
        ordered.append(nxt)
        seen.add(nxt)
        prev, cur = cur, nxt

    assert len(ordered) == len(cells), "grey cells must form one simple loop"
    return ordered


def _grey_ports_for_sprite(
    sprite: dict, g: np.ndarray, grey_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    A grey cell is a port for this sprite if:
      - it is exactly 2 steps away from the sprite center in a cardinal direction
      - the intermediate cell is the sprite border color
    """
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        if not (0 <= ir < g.shape[0] and 0 <= ic < g.shape[1]):
            continue
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in grey_cells:
            continue
        if int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def _paint_loop_by_nearest_target(
    out: np.ndarray,
    ordered_loop: list[tuple[int, int]],
    dest_info: list[dict],
) -> None:
    """Color each loop cell by the border of the destination whose port is closest along the loop."""
    n = len(ordered_loop)
    index_of = {cell: i for i, cell in enumerate(ordered_loop)}
    port_indices = [index_of[tuple(info["port"])] for info in dest_info]

    for i, (r, c) in enumerate(ordered_loop):
        distances = [
            min((i - pj) % n, (pj - i) % n)
            for pj in port_indices
        ]
        best = min(range(len(dest_info)), key=lambda j: distances[j])
        assert distances.count(distances[best]) == 1, "expected a unique nearest destination for each loop cell"
        out[r, c] = int(dest_info[best]["sprite"]["border"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors into every sprite attached to a grey loop, then absorb the loop into their border colors.

    Input structure:
    - The grid contains background, a grey (5) closed loop, and several bordered sprites whose single center pixel
      differs from the sprite border color.
    - Any sprite with a valid 2-step port touching the grey loop is a destination; in this fused variant the loop
      connects all kept targets rather than only two endpoint targets.

    Rule:
    - Detect every destination sprite attached to the grey loop through the standard port geometry.
    - For each destination, use its original center color as a lookup key: find the non-destination source sprite
      whose border color matches that key, and copy that source center color into the destination center.
    - Delete every sprite not attached to the loop and any leftover foreground not connected to the kept targets.
    - Finally, traverse the loop in cyclic order and recolor each loop cell to the border color of the destination
      whose port is nearest along the loop.

    Key insight:
    - Two changes must be fused, not handled independently: the closed-loop topology decides that all loop-attached
      sprites survive, and that same loop geometry also decides how the output path is partitioned into target-colored
      segments by nearest attached port.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite"

    grey_cells = _grey_cells(g, grey)
    ordered_loop = _ordered_loop_cells(grey_cells)

    dest_info: list[dict] = []
    for sp in sprites:
        ports = _grey_ports_for_sprite(sp, g, grey_cells)
        if ports:
            assert len(ports) == 1, "expected each destination to expose exactly one loop port"
            dest_info.append({"sprite": sp, "port": ports[0]})

    assert len(dest_info) >= 3, "expected multiple loop-connected destinations"
    dest_ids = {id(info["sprite"]) for info in dest_info}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for info in dest_info:
        for r, c in info["sprite"]["pts"]:
            if out[r, c] not in (bg, grey) and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    for info in dest_info:
        dest = info["sprite"]
        want_border = int(dest["center"])
        srcs = [sp for sp in by_border.get(want_border, []) if id(sp) not in dest_ids]
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination lookup should match exactly one non-destination source"
        dr, dc = dest["center_pos"]
        out[dr, dc] = int(srcs[0]["center"])

    _paint_loop_by_nearest_target(out, ordered_loop, dest_info)
    return out


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch with center at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check overlap or 4-touching another non-bg, non-grey object."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in neighbors4(rr, cc):
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _rect_ring_cells(bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bounds
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        cells.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cells.append((r, c0))
    return cells


def _assign_colors(num_sources: int, num_distractors: int) -> dict | None:
    total = NUM_DEST + num_sources + num_distractors
    palette = [c for c in range(1, 10) if c != GREY]
    if total > len(palette):
        return None

    border_colors = random.sample(palette, total)
    dest_borders = border_colors[:NUM_DEST]
    source_borders = border_colors[NUM_DEST:NUM_DEST + num_sources]
    distractor_borders = border_colors[NUM_DEST + num_sources:]

    center_pool = [c for c in palette if c not in border_colors]
    if len(center_pool) < num_sources:
        return None
    source_centers = random.sample(center_pool, num_sources)
    src_center_by_border = dict(zip(source_borders, source_centers, strict=True))

    if num_sources == 1:
        dest_centers = [source_borders[0]] * NUM_DEST
    else:
        a, b = source_borders
        dest_centers = random.choice(
            [
                [a, b, a, b],
                [a, a, b, b],
                [a, b, b, a],
            ]
        )

    output_dest_centers = [src_center_by_border[c] for c in dest_centers]

    distractor_centers = []
    reserved = set(source_centers)
    for bc in distractor_borders:
        choices = [c for c in palette if c != bc and c not in reserved]
        if not choices:
            choices = [c for c in palette if c != bc]
        chosen = random.choice(choices)
        distractor_centers.append(chosen)
        reserved.add(chosen)

    return {
        "dest_borders": dest_borders,
        "source_borders": source_borders,
        "distractor_borders": distractor_borders,
        "dest_centers": dest_centers,
        "source_centers": source_centers,
        "distractor_centers": distractor_centers,
        "output_dest_centers": output_dest_centers,
    }


def _grey_port_exists_for_center(
    grid: np.ndarray, center: tuple[int, int], border_color: int,
) -> bool:
    grey_cells = _grey_cells(grid, GREY)
    dummy = {"center_pos": center, "border": border_color}
    return bool(_grey_ports_for_sprite(dummy, grid, grey_cells))


def _place_destinations(output_grid: np.ndarray, shape_type: str, colors: dict) -> list[dict] | None:
    destinations: list[dict] = []
    slots = DEST_SLOTS[:]
    random.shuffle(slots)
    for i, slot in enumerate(slots):
        patch = _make_sprite_patch(shape_type, colors["dest_borders"][i], colors["output_dest_centers"][i])
        if _overlaps(output_grid, patch, slot["r0"], slot["c0"]):
            return None
        _place_sprite(output_grid, patch, slot["r0"], slot["c0"])
        destinations.append(
            {
                "slot": slot,
                "border": colors["dest_borders"][i],
                "input_center": colors["dest_centers"][i],
                "output_center": colors["output_dest_centers"][i],
            }
        )
    return destinations


def _place_extras(
    input_grid: np.ndarray,
    shape_type: str,
    sprite_specs: list[tuple[int, int]],
) -> bool:
    slots = EXTRA_SLOTS[:]
    random.shuffle(slots)
    used: set[tuple[int, int]] = set()
    for border_color, center_color in sprite_specs:
        patch = _make_sprite_patch(shape_type, border_color, center_color)
        placed = False
        for r0, c0 in slots:
            if (r0, c0) in used or _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            center = (r0 + 1, c0 + 1)
            if _grey_port_exists_for_center(input_grid, center, border_color):
                _erase_sprite(input_grid, patch, r0, c0)
                continue
            used.add((r0, c0))
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    num_sources = random.choice(NUM_SOURCE_CHOICES)
    num_distractors = 0 if num_sources == 2 else random.choice(DISTRACTOR_CHOICES)

    colors = _assign_colors(num_sources, num_distractors)
    if colors is None:
        return None

    grey_output = np.zeros((GRID_H, GRID_W), dtype=int)
    destinations = _place_destinations(grey_output, shape_type, colors)
    if destinations is None:
        return None

    ring_cells = _rect_ring_cells(RING_BOUNDS)
    for r, c in ring_cells:
        if grey_output[r, c] != BG:
            return None
        grey_output[r, c] = GREY

    grey_cells = _grey_cells(grey_output, GREY)
    ordered_loop = _ordered_loop_cells(grey_cells)

    dest_info = []
    for dest in destinations:
        center = dest["slot"]["center"]
        border = dest["border"]
        if not _grey_port_exists_for_center(grey_output, center, border):
            return None
        dummy = {"center_pos": center, "border": border}
        ports = _grey_ports_for_sprite(dummy, grey_output, grey_cells)
        if len(ports) != 1 or ports[0] != dest["slot"]["port"]:
            return None
        dest_info.append(
            {
                "sprite": {"border": border},
                "port": dest["slot"]["port"],
            }
        )

    output_grid = grey_output.copy()
    _paint_loop_by_nearest_target(output_grid, ordered_loop, dest_info)

    input_grid = grey_output.copy()
    for dest in destinations:
        cr, cc = dest["slot"]["center"]
        input_grid[cr, cc] = dest["input_center"]

    sprite_specs = list(zip(colors["source_borders"], colors["source_centers"], strict=True))
    sprite_specs.extend(zip(colors["distractor_borders"], colors["distractor_centers"], strict=True))
    random.shuffle(sprite_specs)
    if not _place_extras(input_grid, shape_type, sprite_specs):
        return None

    try:
        out = main(input_grid)
    except Exception:
        return None
    return input_grid if np.array_equal(out, output_grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
