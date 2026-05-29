# Variant of d35bdbdc
# Dimension: D7 — 灰色路径结构 (meta)
# Change: 灰色像素构成连通路径，端点连接到目标精灵的中心位置... → 环形路径

"""
design_analysis:
  variant_goal:
    dimension: D7
    old: 灰色像素是一条开放的一笔画路径, 只有两个端点各自选出一个目标精灵
    new: 灰色像素改成单个闭合环, 所有通过合法端口接到这个环上的精灵都是目标精灵
  concrete_visualization:
    objects:
      - 10x10 网格, 背景 0, 灰环颜色 5
      - 若干 3x3 有边框精灵, 形状仍然统一为 box 或 plus
      - 每个精灵仍然只有两种颜色: 多像素的 border_color 与唯一中心色 center_color
    ring:
      - 生成器固定构造一个位于中部的细矩形环, 在 8 邻域下连成单个闭合灰色分量
      - 环上连接 3 或 4 个目标精灵, 目标精灵分布在环四周
      - 合法端口规则保持不变: 端口灰格必须与精灵中心曼哈顿距离 2, 中间一格必须是该精灵边框色
    input_to_output_effect:
      - 输入中所有接环目标都保留
      - 每个目标用自己的原中心色去查找某个 source.border
      - 输出把目标中心改成对应 source.center
      - 所有不接环的 source / distractor 精灵被删除
  behavior_trace:
    original_local_step:
      - 旧代码先找灰路径两个端点
      - 再由两个端点各自匹配一个目标精灵
    variant_local_step:
      - 新代码不再找端点, 而是枚举每个精灵是否存在灰色合法端口
      - 只要某精灵至少有一个端口落在灰环上, 它就是目标
    example:
      - 若灰色是中央矩形环, 上/右/下三个精灵分别各有一个端口落在环上
      - 旧逻辑会失败, 因为不存在度为 1 的端点
      - 新逻辑会把这三个精灵全部识别为目标, 并同时执行颜色转移
  complexity_calibration:
    - 目标精灵数提升到 3-4 个, 明显区别于原题固定 2 个目标
    - source 数量只保留 1-2 个, 且多个目标会重复引用同一个 source, 让“查表复制”在人眼上有重复支撑
    - 目标中心色只从 1-2 个 source.border 中取值, 避免颜色链过于分散
  implicit_assumption_review:
    old_dependencies:
      - _find_trail_endpoints 假设灰路径存在两个端点
      - _ports_for_sprite(..., eps) 假设只需要检查端点集合
      - main() 中的二分匹配与 used 集合都写死为两个目标
    new_replacement:
      - 改为检测灰色是否为单个无端点分量
      - 端口检查从“端点属于 eps”改为“端口属于全部灰格”
      - 目标集合改为所有有灰端口的精灵, 不再做二分匹配
    new_generator_guarantees:
      - 额外精灵只放在远离灰环的安全槽位, 不会意外拥有灰端口
      - 每个目标恰有一个可见端口连到灰环, 所以目标集合清晰
      - 每个目标中心色都对应恰好一个非目标 source.border
  placement_legality:
    destinations:
      - 目标精灵只使用环四侧的预定义槽位, 保证与灰环之间留出正好一格边框距离
      - 这些槽位对 box / plus 两种形状都合法, 且不会互相重叠
    extras:
      - source / distractor 只放在四角附近的安全槽位
      - 放置时仍检查不重叠, 不 4 邻接其他非灰对象, 不与灰环形成合法端口
  observability:
    - 人类在 2-3 个样例里能直接看到: 灰色不再是线段而是封闭环
    - 输出里保留的不再只是两个端点相连对象, 而是环周围所有接入对象
    - 多个目标同时变色, 且目标原中心色重复指向少量 source.border, 规律可见
  cleanliness:
    - 灰环固定在中央, 目标在四周, source/distractor 在角落, 区域分层清楚
    - 不使用基于颜色 id 的排序或不可见 tie-break 规则
"""

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
RING_BOUNDS = (2, 2, 7, 7)
NUM_DEST_CHOICES = [3, 4]
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
      - center color appears exactly once (its location is the 'center')
    This also handles clipped/partial borders (e.g., near the grid edge).
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
        if border_color == bg or border_color == grey or center_color == grey:
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


def _is_single_loop_component(cells: set[tuple[int, int]]) -> bool:
    """A permissive loop check: one 8-connected component with no degree-1 cells."""
    if not cells:
        return False

    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    if seen != cells:
        return False

    for r, c in cells:
        deg = sum(1 for nr, nc in neighbors8(r, c) if (nr, nc) in cells)
        if deg < 2:
            return False
    return True


def _grey_ports_for_sprite(
    sprite: dict, g: np.ndarray, grey_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    A grey cell is a port for this sprite if:
      - it is exactly 2 steps away from the sprite center in a cardinal direction
      - the intermediate cell is the sprite border color
    """
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        if not (0 <= ir < g.shape[0] and 0 <= ic < g.shape[1]):
            continue
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in grey_cells:
            continue
        if int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors into every bordered sprite attached to the grey loop, keeping only loop-attached sprites.

    Input structure:
    - The grid contains background, a grey (5) loop, and several small bordered sprites with one unique center pixel.
    - Grey pixels form a single closed 8-connected ring; any sprite whose valid port touches that ring is a destination.

    Rule:
    - Collect every destination sprite connected to the grey ring through the standard 2-step port geometry.
    - For each destination, find the source sprite whose border color equals the destination's original center color, then copy that source center color into the destination center.
    - Delete all non-destination sprites; the grey ring and all ring-connected destinations remain.

    Key insight:
    - The topology change matters: target selection is no longer about two endpoints, but about loop membership. The original destination center color still acts as the lookup key for the source border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    grey_cells = _grey_cells(g, grey)
    assert _is_single_loop_component(grey_cells), "expected one closed grey loop"

    dest = [sp for sp in sprites if _grey_ports_for_sprite(sp, g, grey_cells)]
    assert len(dest) >= 3, "expected at least three loop-connected destination sprites"
    dest_ids = {id(sp) for sp in dest}

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
    for d in dest:
        for r, c in d["pts"]:
            if out[r, c] != bg and out[r, c] != grey and not keep[r, c]:
                keep[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                if keep[nr, nc]:
                    continue
                if out[nr, nc] == bg or out[nr, nc] == grey:
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    for d in dest:
        want_border = int(d["center"])
        srcs = [sp for sp in by_border.get(want_border, []) if id(sp) not in dest_ids]
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination lookup should match exactly one non-destination source"
        dr, dc = d["center_pos"]
        out[dr, dc] = int(srcs[0]["center"])

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


def _assign_colors(num_dest: int, num_sources: int, num_distractors: int) -> dict | None:
    total = num_dest + num_sources + num_distractors
    palette = [c for c in range(1, 10) if c != GREY]
    if total > len(palette):
        return None
    border_colors = random.sample(palette, total)

    dest_borders = border_colors[:num_dest]
    source_borders = border_colors[num_dest:num_dest + num_sources]
    distractor_borders = border_colors[num_dest + num_sources:]

    center_pool = [c for c in palette if c not in border_colors]
    if len(center_pool) < num_sources:
        return None
    source_centers = random.sample(center_pool, num_sources)
    src_center_by_border = dict(zip(source_borders, source_centers, strict=True))

    if num_sources == 1:
        dest_centers = [source_borders[0]] * num_dest
    else:
        a, b = source_borders
        patterns = {
            3: [[a, b, a], [a, a, b], [a, b, b]],
            4: [[a, b, a, b], [a, a, b, b], [a, b, b, a]],
        }
        dest_centers = random.choice(patterns[num_dest])

    output_dest_centers = [src_center_by_border[c] for c in dest_centers]

    distractor_centers = []
    reserved = set(source_centers)
    for bc in distractor_borders:
        choices = [c for c in palette if c != GREY and c != bc and c not in reserved]
        if not choices:
            choices = [c for c in palette if c != GREY and c != bc]
        chosen = random.choice(choices)
        distractor_centers.append(chosen)
        reserved.add(chosen)

    return {
        "border_colors": border_colors,
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


def _place_destinations(
    output_grid: np.ndarray, shape_type: str, colors: dict, num_dest: int,
) -> list[dict] | None:
    slots = random.sample(DEST_SLOTS, num_dest)
    destinations: list[dict] = []
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
    input_grid: np.ndarray, shape_type: str, sprite_specs: list[tuple[int, int]],
) -> bool:
    slots = EXTRA_SLOTS[:]
    random.shuffle(slots)
    used: set[tuple[int, int]] = set()
    for border_color, center_color in sprite_specs:
        patch = _make_sprite_patch(shape_type, border_color, center_color)
        placed = False
        for r0, c0 in slots:
            if (r0, c0) in used:
                continue
            if _overlaps(input_grid, patch, r0, c0):
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
    num_dest = random.choice(NUM_DEST_CHOICES)
    num_sources = random.choice(NUM_SOURCE_CHOICES)
    if num_dest == 4 and num_sources == 2:
        num_distractors = 0
    else:
        num_distractors = random.choice(DISTRACTOR_CHOICES)

    colors = _assign_colors(num_dest, num_sources, num_distractors)
    if colors is None:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    destinations = _place_destinations(output_grid, shape_type, colors, num_dest)
    if destinations is None:
        return None

    ring_cells = _rect_ring_cells(RING_BOUNDS)
    for r, c in ring_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    grey_cells = _grey_cells(output_grid, GREY)
    if not _is_single_loop_component(grey_cells):
        return None

    # Each designated destination must have a port on the ring; non-designated slots stay absent.
    for dest in destinations:
        if dest["slot"]["port"] not in grey_cells:
            return None
        if not _grey_port_exists_for_center(output_grid, dest["slot"]["center"], dest["border"]):
            return None

    input_grid = output_grid.copy()
    for dest in destinations:
        cr, cc = dest["slot"]["center"]
        input_grid[cr, cc] = dest["input_center"]

    sprite_specs = []
    sprite_specs.extend(zip(colors["source_borders"], colors["source_centers"], strict=True))
    sprite_specs.extend(zip(colors["distractor_borders"], colors["distractor_centers"], strict=True))
    sprite_specs = list(sprite_specs)
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
