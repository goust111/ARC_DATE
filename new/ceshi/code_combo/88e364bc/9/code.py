# Combo variant of 88e364bc (index 9)
# combo_spec: D5/0+D18/0
# Name: 2x2色块+小网格

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, neighbors4

"""
Design analysis for the D5/0 + D18/0 fused variant.

核心保留:
- 仍然是“墙色匹配边界图例 -> 图例里 center / indicator 的相对位置给出 8 个方向 -> 黄色对象滑向远端墙”。

融合改动:
- D5/0: 黄色移动对象不再是单像素，而是刚性的 2x2 黄色块。
- D18/0: 网格缩小到 10x10-12x12；图例也收紧为统一 5x5，以便在小画布上仍能和封闭区域同时出现。

可观察性策略:
- 仅把输入里的黄色换成 2x2 还不够，因为在规则矩形腔室里，逐像素滑动和刚体滑动可能看起来一样。
- 因此 generator 使用带缺口的小腔室，让 2x2 刚体会整体提前停下，而旧的逐像素规则会把四个黄点剪切开。
- 生成样例时显式拒绝“刚体输出 == 旧逐像素输出”的输入，保证 D5 变化在 2-4 组样例内稳定可见。

隐式假设审查:
- legend 仍是边界上的实心矩形，且恰含三色；solver 的 legend 解析可以沿用原题质心思路。
- 黄色组件现在必须被识别成若干个独立的 2x2 连通块，不能再按单像素列表处理。
- 2x2 对角滑动时仍需要整块通过两个正交中间位置，避免“切角”。
- 小网格里如果还沿用大图的 6x7 对角图例，几乎没有空间放区域，所以这里把所有方向都压成 5x5 紧凑三色图例；
  方向语义仍完全来自 center->indicator 的质心符号，而不是颜色编号或硬编码箭头类别。
"""


MOVER = 4
SPRITE_SIZE = 2

GRID_CHOICES = [10, 11, 12]
GEN_RETRIES = 300

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ALL_DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]

BASE_RIGHT_MASK = np.array(
    [
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
    dtype=bool,
)

BASE_DOWN_RIGHT_MASK = np.array(
    [
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ],
    dtype=bool,
)

LEGEND_CENTER_CELLS = [(1, 1), (1, 2), (2, 1), (2, 2)]
LEGEND_INDICATOR_CELLS = {
    (0, 1): [(1, 3), (2, 3)],
    (0, -1): [(1, 0), (2, 0)],
    (1, 0): [(3, 1), (3, 2)],
    (-1, 0): [(0, 1), (0, 2)],
    (1, 1): [(3, 3), (4, 4)],
    (1, -1): [(3, 0), (4, 1)],
    (-1, 1): [(0, 3), (1, 4)],
    (-1, -1): [(0, 0), (1, 0)],
}


def _projection_key(p: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int]:
    return (dr * p[0] + dc * p[1], p[0], p[1])


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _legend_patch(fill: int, dr: int, dc: int) -> np.ndarray:
    patch = np.full((5, 5), int(fill), dtype=int)
    for r, c in LEGEND_CENTER_CELLS:
        patch[r, c] = 1
    for r, c in LEGEND_INDICATOR_CELLS[(dr, dc)]:
        patch[r, c] = 2
    return patch


def _legend_patch_small(fill: int, dr: int, dc: int) -> np.ndarray:
    patch = np.full((4, 4), int(fill), dtype=int)
    for r, c in LEGEND_CENTER_CELLS:
        patch[r, c] = 1
    small_indicator = {
        (0, 1): [(1, 3), (2, 3)],
        (0, -1): [(1, 0), (2, 0)],
        (1, 0): [(3, 1), (3, 2)],
        (-1, 0): [(0, 1), (0, 2)],
        (1, 1): [(2, 3), (3, 3)],
        (1, -1): [(2, 0), (3, 0)],
        (-1, 1): [(0, 3), (1, 3)],
        (-1, -1): [(0, 0), (1, 0)],
    }
    for r, c in small_indicator[(dr, dc)]:
        patch[r, c] = 2
    return patch


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            value = int(g[r, c])
            counts[value] = counts.get(value, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        if len(others) != 2:
            continue
        others.sort()
        indicator_col = others[0][1]
        center_col = others[1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indicator = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indicator:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indicator, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))

    return {color: direction for color, (_, direction) in best.items()}


def _block_cells(anchor: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = anchor
    return [(r + dr, c + dc) for dr in range(SPRITE_SIZE) for dc in range(SPRITE_SIZE)]


def _block_cell_set(anchor: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_block_cells(anchor))


def _anchor_fits(region_set: set[tuple[int, int]], anchor: tuple[int, int]) -> bool:
    return all(cell in region_set for cell in _block_cells(anchor))


def _block_step_valid(
    region_set: set[tuple[int, int]],
    anchor: tuple[int, int],
    dr: int,
    dc: int,
) -> bool:
    r, c = anchor
    nxt = (r + dr, c + dc)
    if not _anchor_fits(region_set, nxt):
        return False
    if dr != 0 and dc != 0:
        if not _anchor_fits(region_set, (r + dr, c)):
            return False
        if not _anchor_fits(region_set, (r, c + dc)):
            return False
    return True


def _extract_block_anchors(
    g: np.ndarray,
    region_set: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    yellow = {(int(r), int(c)) for r, c in region_set if int(g[r, c]) == MOVER}
    anchors: list[tuple[int, int]] = []
    while yellow:
        start = yellow.pop()
        stack = [start]
        comp = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in neighbors4(r, c):
                nxt = (int(nr), int(nc))
                if nxt in yellow:
                    yellow.remove(nxt)
                    stack.append(nxt)
                    comp.add(nxt)
        if len(comp) != 4:
            raise ValueError("yellow component is not a filled 2x2 block")
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        expected = _block_cell_set((r0, c0))
        if comp != expected:
            raise ValueError("yellow component is not a filled 2x2 block")
        anchors.append((r0, c0))
    return anchors


def _slide_blocks(
    region_set: set[tuple[int, int]],
    anchors: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> list[tuple[int, int]]:
    if not anchors:
        return []
    anchors_sorted = sorted(anchors, key=lambda p: _projection_key(p, dr, dc), reverse=True)
    occupied_cells: set[tuple[int, int]] = set()
    out_anchors: list[tuple[int, int]] = []
    for anchor0 in anchors_sorted:
        anchor = anchor0
        while True:
            if not _block_step_valid(region_set, anchor, dr, dc):
                break
            nxt = (anchor[0] + dr, anchor[1] + dc)
            if _block_cell_set(nxt) & occupied_cells:
                break
            anchor = nxt
        occupied_cells.update(_block_cells(anchor))
        out_anchors.append(anchor)
    return out_anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide rigid 2x2 yellow blocks inside enclosed wall regions using compact border legends.

    Input structure:
    - Black is the background.
    - A single wall color forms one or more enclosed cavities.
    - Border legends are 5x5 filled rectangles that reuse that wall color and contain:
      a 4-pixel center marker (color 1) and a 2-pixel indicator marker (color 2).
    - Yellow movers are now full 2x2 blocks.

    Rule:
    - Match each enclosed cavity to the legend whose fill color equals the cavity wall color.
    - Read the sign of centroid(indicator) - centroid(center) to get one of 8 directions.
    - Move every yellow 2x2 block rigidly in that direction until the whole block would hit
      a wall, violate diagonal corner passage, or overlap a previously stopped 2x2 block.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        region_set = {(int(r), int(c)) for r, c in pts}
        if not any(int(g[r, c]) == MOVER for r, c in region_set):
            continue

        border_counts: dict[int, int] = {}
        for r, c in region_set:
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in region_set:
                    continue
                value = int(g[nr, nc])
                if value == bg or value == MOVER:
                    continue
                border_counts[value] = border_counts.get(value, 0) + 1
        if not border_counts:
            continue

        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        anchors = _extract_block_anchors(g, region_set)
        for anchor in anchors:
            for r, c in _block_cells(anchor):
                out[r, c] = bg
        for anchor in _slide_blocks(region_set, anchors, dr, dc):
            for r, c in _block_cells(anchor):
                out[r, c] = MOVER

    return out


def _transform_cardinal_mask(direction: tuple[int, int]) -> np.ndarray:
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(BASE_RIGHT_MASK, rot[direction])


def _transform_diagonal_mask(direction: tuple[int, int]) -> np.ndarray:
    if direction == (1, 1):
        return BASE_DOWN_RIGHT_MASK.copy()
    if direction == (-1, 1):
        return np.flipud(BASE_DOWN_RIGHT_MASK)
    if direction == (1, -1):
        return np.fliplr(BASE_DOWN_RIGHT_MASK)
    if direction == (-1, -1):
        return np.rot90(BASE_DOWN_RIGHT_MASK, 2)
    raise ValueError(direction)


def _interior_mask_for_direction(direction: tuple[int, int]) -> np.ndarray:
    if abs(direction[0]) + abs(direction[1]) == 1:
        return _transform_cardinal_mask(direction)
    return _transform_diagonal_mask(direction)


def _legacy_step_valid(
    region_set: set[tuple[int, int]],
    point: tuple[int, int],
    dr: int,
    dc: int,
) -> bool:
    r, c = point
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set:
            return False
        if (r, c + dc) not in region_set:
            return False
    return True


def _legacy_slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
) -> set[tuple[int, int]]:
    pts_sorted = sorted(points, key=lambda p: _projection_key(p, dr, dc), reverse=True)
    occupied: set[tuple[int, int]] = set()
    out_pts: set[tuple[int, int]] = set()
    for r0, c0 in pts_sorted:
        r, c = r0, c0
        while True:
            if not _legacy_step_valid(region_set, (r, c), dr, dc):
                break
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.add((r, c))
    return out_pts


def _room_patch(mask: np.ndarray, wall_color: int) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    ih, iw = map(int, mask.shape)
    room = np.full((ih + 2, iw + 2), int(wall_color), dtype=int)
    inner = room[1 : 1 + ih, 1 : 1 + iw]
    inner[mask] = 0
    region_local = {(int(r + 1), int(c + 1)) for r, c in np.argwhere(mask)}
    occupied_local = {(r, c) for r in range(room.shape[0]) for c in range(room.shape[1]) if int(room[r, c]) != 0}
    return room, region_local, occupied_local


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _touches_patch(
    occupied_cells: set[tuple[int, int]],
    patch_cells: set[tuple[int, int]],
) -> bool:
    if occupied_cells & patch_cells:
        return True
    for r, c in occupied_cells:
        for nr, nc in neighbors4(r, c):
            if (int(nr), int(nc)) in patch_cells:
                return True
    return False


def _room_positions(
    h: int,
    w: int,
    room_shape: tuple[int, int],
    patch_cells: set[tuple[int, int]],
    room_occupied_local: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    rh, rw = room_shape
    candidates: list[tuple[int, int, int]] = []
    patch_center_r = sum(r for r, _ in patch_cells) / len(patch_cells)
    patch_center_c = sum(c for _, c in patch_cells) / len(patch_cells)
    for r0 in range(h - rh + 1):
        for c0 in range(w - rw + 1):
            if any(r0 <= pr < r0 + rh and c0 <= pc < c0 + rw for pr, pc in patch_cells):
                continue
            room_cells = _shift_cells(room_occupied_local, r0, c0)
            if _touches_patch(room_cells, patch_cells):
                continue
            room_center_r = r0 + (rh - 1) / 2.0
            room_center_c = c0 + (rw - 1) / 2.0
            dist = abs(room_center_r - patch_center_r) + abs(room_center_c - patch_center_c)
            candidates.append((int(dist * 100), r0, c0))
    rng.shuffle(candidates)
    candidates.sort(reverse=True)
    return [(r0, c0) for _dist, r0, c0 in candidates[: max(1, min(8, len(candidates)))]]


def _local_case_options(
    mask: np.ndarray,
    direction: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    region_local = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in sorted(region_local):
        anchor = (r, c)
        if not _anchor_fits(region_local, anchor):
            continue
        final_anchor = _slide_blocks(region_local, [anchor], direction[0], direction[1])[0]
        if final_anchor == anchor:
            continue
        rigid_out = _block_cell_set(final_anchor)
        legacy_out = _legacy_slide_points(region_local, _block_cells(anchor), direction[0], direction[1])
        if rigid_out == legacy_out:
            continue
        options.append((anchor, final_anchor))
    return options


def _build_case(
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    size = rng.choice(GRID_CHOICES)
    direction_pool = CARDINAL_DIRS if size == 10 else ALL_DIRS
    direction = rng.choice(direction_pool)
    wall_color = rng.choice(WALL_COLOR_CHOICES)
    corner = rng.choice(CORNERS)

    mask = _interior_mask_for_direction(direction)
    options = _local_case_options(mask, direction)
    if not options:
        return None
    input_anchor_local, final_anchor_local = rng.choice(options)

    legend = (
        _legend_patch_small(wall_color, direction[0], direction[1])
        if size == 10
        else _legend_patch(wall_color, direction[0], direction[1])
    )
    grid = np.zeros((size, size), dtype=int)
    lr, lc = _corner_origin(corner, size, size, legend.shape[0], legend.shape[1])
    grid[lr : lr + legend.shape[0], lc : lc + legend.shape[1]] = legend
    patch_cells = {(lr + r, lc + c) for r in range(legend.shape[0]) for c in range(legend.shape[1])}

    room, region_local, room_occupied_local = _room_patch(mask, wall_color)
    positions = _room_positions(size, size, room.shape, patch_cells, room_occupied_local, rng)
    if not positions:
        return None
    rr, rc = rng.choice(positions)
    room_mask = room != 0
    grid_slice = grid[rr : rr + room.shape[0], rc : rc + room.shape[1]]
    grid_slice[room_mask] = room[room_mask]

    input_anchor = (rr + input_anchor_local[0] + 1, rc + input_anchor_local[1] + 1)
    final_anchor = (rr + final_anchor_local[0] + 1, rc + final_anchor_local[1] + 1)
    for r, c in _block_cells(input_anchor):
        grid[r, c] = MOVER

    expected = grid.copy()
    for r, c in _block_cells(input_anchor):
        expected[r, c] = 0
    for r, c in _block_cells(final_anchor):
        expected[r, c] = MOVER

    if np.array_equal(grid, expected):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if int(get_bg(grid)) != 0:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = _build_case(rng)
        if built is None:
            continue
        return built[0]
    raise RuntimeError("generate_input: exceeded retries")
