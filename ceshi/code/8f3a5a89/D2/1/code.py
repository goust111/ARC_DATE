# Variant of 8f3a5a89
# Dimension: D2 — 内边界涂色规则 (trajectory)
# Change: inner_border——可达区域中，位于网格边缘、8-邻接结构墙、或4-邻接不可达像素的背景格子涂橙色，形成1像素宽... → 双层边界（内外各一圈）

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis for the D2 variant
# ----------------------------------
# spec:
#   preserved_core:
#     - 背景色仍为 8，结构墙仍为 1，角落标记仍为单个 6，结构墙/装饰墙的判定方式不变。
#     - 从左下角标记出发，第一次洪填把所有墙都当障碍；第二次洪填允许穿过非结构墙装饰块。
#     - 不可达侧的装饰墙仍会被擦成背景。
#   variant_core:
#     - 原题只在“可达区域一侧”给边界上色；变体要求在“不可达区域一侧”再补一圈第二颜色。
#     - 具体化后，输出包含两条不同颜色的细边界：
#       1. 橙色内边界：与原题完全相同，画在 reachable 的背景格上。
#       2. 绿色外边界：画在 unreachable 的背景格上，只要该格 8 邻接结构墙，即视作墙外侧边界。
#   behavior_trace:
#     - 例：若有一条从顶到底的蓝色竖墙把左侧 marker 区和右侧口袋区分开，
#       原题只会把竖墙左边的 reachable 背景格涂橙；
#       新题会继续把竖墙右边的 unreachable 背景格涂绿。
#     - 差异点在于：同一堵结构墙两侧都被显式编码，求解者必须同时辨认“墙内侧”和“墙外侧”。
#   complexity_calibration:
#     - 外边界颜色只新增 1 种，避免颜色映射过多。
#     - 仍保留 1-4 个结构墙部件与 0-3 个装饰块，样例不会退化成单一模板。
#     - generator 强制外边界像素数达到下限，避免输出肉眼上与原题几乎一致。
#   implicit_assumption_audit:
#     - 原 solver 中“只给 reachable 背景格染色”的循环依赖旧前提：边界只有单侧需要显示。
#       变体下这个前提失效，所以必须额外扫描 unreachable 背景格。
#     - 原代码按 `out[r,c] == bg` 限制染色，避免覆盖 marker / wall / 可见装饰墙；这个前提仍成立。
#     - 原代码通过 `walls` 而非全部 wall-color 像素定义边界；变体仍必须沿用，否则装饰墙会错误地产生绿色外边界。
#   placement_legality:
#     - 可见装饰墙只允许放在最终输出仍为背景的 reachable 深处，避免覆盖橙边界。
#     - 隐藏装饰墙只允许放在 unreachable 深处，且避开最终绿色外边界，避免把双层边界伪装成“擦掉墙后变色”。
#   observability:
#     - 人类从 2-3 个样例中可直接看到：蓝墙一侧是橙色，另一侧多出绿色。
#     - 如果结构墙只贴边但没有困住任何 unreachable 背景区，绿色就不会出现，样例会退化回原题；
#       因此 generator 显式拒绝这种布局。
#   cleanliness:
#     - 双色边界都只画在背景格上，不与 marker、结构墙或可见装饰墙重叠。
#     - 绿色不沿整张外框乱涂，只出现在结构墙的不可达一侧，视觉上仍像“墙的另一面”。


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _identify_marker_and_wall(g: np.ndarray, bg: int) -> tuple[tuple[int, int, int], int]:
    counts = Counter(int(v) for v in g.ravel())
    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    _mr, _mc, marker_col = marker
    other_cols = [c for c in counts if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])
    return marker, wall_col


def _identify_structural_walls(g: np.ndarray, wall_col: int, reach0: set[tuple[int, int]]) -> np.ndarray:
    h, w = g.shape
    walls = np.zeros((h, w), dtype=bool)
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            for r, c in comp:
                walls[r, c] = True
    return walls


def _compute_border_masks(walls: np.ndarray, reach: np.ndarray, bg_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = walls.shape
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_walls8 = _dilate8(walls)
    near_nonreach4 = _dilate4(~reach)

    inner = reach & bg_mask & (edge | near_walls8 | near_nonreach4)

    # For the unreachable side there is no "outside-of-grid cell" to paint, so the
    # second ring is defined only by structural-wall adjacency on the unreachable side.
    outer = (~reach) & bg_mask & near_walls8
    return inner, outer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a two-color boundary on both sides of the region split by structural walls.

    Input structure:
    - The dominant background color fills most cells; blue wall-colored cells contain both real barriers and small decorative fragments.
    - A unique one-pixel marker sits in a corner and identifies which side of the wall system is the reachable side.
    - True structural walls are exactly the wall components that touch the frame and border the marker-reachable area.

    Rule:
    - Flood from the marker corner, first treating every wall-colored cell as solid, then allowing passage through non-structural wall decorations.
    - Erase hidden decorative wall fragments that lie on the unreachable side of the structural barriers.
    - Paint orange on the reachable-side inner boundary as before, and also paint green on unreachable background cells directly lining the far side of those same structural walls.

    Key insight: The same wall identification logic from the original task still decides what is a real separator; the variant only changes the rendering by exposing both sides of that separator with two different boundary colors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    marker, wall_col = _identify_marker_and_wall(g, bg)
    mr, mc, marker_col = marker

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls = _identify_structural_walls(g, wall_col, reach0)

    reach_set = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and not walls[r, c]),
        h,
        w,
    )
    reach = np.zeros((h, w), dtype=bool)
    for r, c in reach_set:
        reach[r, c] = True

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and not walls[r, c] and not reach[r, c]:
                out[r, c] = bg

    bg_mask = out == bg
    inner_mask, outer_mask = _compute_border_masks(walls, reach, bg_mask)
    out[inner_mask] = FILL
    out[outer_mask] = OUTER_FILL
    return out


# ---- Palette (task colors) ----
BG = 8
WALL = 1
MARKER = 6
FILL = 7
OUTER_FILL = 3

# ---- Canvas size ----
GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

# ---- Structural walls ----
NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

# ---- Reachable region gate ----
REACH_RATIO_RANGE = (0.38, 0.87)
OUTER_MIN_PIXELS = 4

# ---- Decorations ----
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

# ---- Outer retry budget ----
GEN_ATTEMPTS = 600


def _shape_vertical_line(h, w, rng):
    c = rng.randint(2, w - 2)
    mode = rng.choice(["top", "bottom", "full"])
    if mode == "full":
        r0, r1 = 0, h - 1
    elif mode == "top":
        r0, r1 = 0, rng.randint(max(1, h // 3), h - 2)
    else:
        r0, r1 = rng.randint(1, h - max(2, h // 3)), h - 1
    return [(r, c) for r in range(r0, r1 + 1)]


def _shape_horizontal_line(h, w, rng):
    r = rng.randint(1, h - 2)
    if rng.choice(["left", "right"]) == "left":
        c0, c1 = 0, rng.randint(max(1, w // 3), w - 2)
    else:
        c0, c1 = rng.randint(1, w - max(2, w // 3)), w - 1
    return [(r, c) for c in range(c0, c1 + 1)]


def _shape_rectangle(h, w, rng):
    rh = rng.randint(2, max(2, h // RECT_MAX_DIM_DIV))
    rw = rng.randint(2, max(2, w // RECT_MAX_DIM_DIV))
    anchor = rng.choice(["top", "right", "tr", "top_mid", "right_mid", "bottom_right"])
    if anchor == "top":
        r0, c0 = 0, rng.randint(1, max(1, w - rw - 1))
    elif anchor == "right":
        r0, c0 = rng.randint(1, max(1, h - rh - 1)), w - rw
    elif anchor == "tr":
        r0, c0 = 0, w - rw
    elif anchor == "top_mid":
        r0, c0 = 0, rng.randint(2, max(2, w - rw - 2))
    elif anchor == "right_mid":
        r0, c0 = rng.randint(2, max(2, h - rh - 2)), w - rw
    else:
        r0, c0 = h - rh, w - rw
    return [(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)]


def _shape_left_diamond(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 3:
        return None
    size = rng.randint(3, max_size)
    r_center = rng.randint(size, h - size - 1)
    top = r_center - (size - 1)
    cells = []
    for i in range(size):
        cells.extend((top + i, c) for c in range(i + 1))
    for i in range(size - 1):
        cells.extend((top + size + i, c) for c in range(size - 1 - i))
    return cells


def _shape_top_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // TRI_MIN_DIM_DIV, w // TRI_MIN_DIM_DIV)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    c_start = rng.randint(1, max(1, w - 2 * size))
    cells = []
    for i in range(size):
        cells.extend((i, c) for c in range(c_start, c_start + size - i))
    return cells


def _shape_right_triangle(h, w, rng):
    max_size = min(TRI_MAX_SIZE, h // 2, w // 2)
    if max_size < 2:
        return None
    size = rng.randint(2, max_size)
    r_start = rng.randint(0, max(0, h - size - 1))
    cells = []
    for i in range(size):
        length = i + 1
        cells.extend((r_start + i, c) for c in range(w - length, w))
    return cells


SHAPE_FACTORIES = [
    _shape_vertical_line,
    _shape_horizontal_line,
    _shape_rectangle,
    _shape_left_diamond,
    _shape_top_triangle,
    _shape_right_triangle,
]


def _place_structural_walls(h, w, marker_rc, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells or any(structural[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    return structural if placed > 0 else None


def _validate_structural(structural: np.ndarray, reach: np.ndarray, h: int, w: int) -> bool:
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER
    inner_mask, outer_mask = _compute_border_masks(structural, reach, out == BG)
    out[inner_mask] = FILL
    out[outer_mask] = OUTER_FILL
    return out, inner_mask, outer_mask


def _place_decor(allowed_mask: np.ndarray, count: int, rng) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed
    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=DECOR_BORDER,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None
    if not _validate_structural(structural, reach, h, w):
        return None

    out, inner_mask, outer_mask = _paint_output(structural, reach, marker_rc)
    if not inner_mask.any():
        return None
    if int(outer_mask.sum()) < OUTER_MIN_PIXELS:
        return None
    if len(components_mask(inner_mask, connectivity=4)) != 1:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural) & (~outer_mask)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    out_final = out.copy()
    out_final[visible_placed] = WALL

    inp = out_final.copy()
    inp[(inp == FILL) | (inp == OUTER_FILL)] = BG
    inp[hidden_placed] = WALL

    solved = main(inp)
    if not np.array_equal(solved, out_final):
        return None
    if np.count_nonzero(solved == OUTER_FILL) < OUTER_MIN_PIXELS:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D2 dual-border variant")
