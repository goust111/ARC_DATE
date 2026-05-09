# Variant of 8f3a5a89
# Dimension: D10 — 多步处理/迭代逻辑 (multi_step)
# Change: 两步洪填——第一次洪填（仅穿过背景和标记）确定reach0，用于识别结构墙；第二次洪填（还可穿过装饰墙）确定最终可达区域... → 先涂边界再擦除装饰

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis:
# spec_variant:
#   core_change:
#     summary: |
#       变体把“边界绘制”和“装饰擦除”的顺序反过来。输入里仍然只有一种蓝色墙，
#       但输出不再只沿结构墙画橙边；可达装饰墙也会保留自己的橙色轮廓。
#     concrete_behavior:
#       input: |
#         左下角仍有单个品红标记；蓝色既包含触边结构墙，也包含内部装饰墙。
#         装饰墙分成两类：可达装饰位于标记侧区域中；隐藏装饰位于结构墙另一侧。
#       output: |
#         先把所有蓝色墙周围可能形成的橙色边界预涂出来，再识别哪些蓝色块其实是
#         不可达装饰，并把这些蓝色块以及围着它们的错误橙边一起擦掉。
#         因此保留下来的橙色包含两种来源：
#         1) 结构墙与网格边缘围成的大区域内边界；
#         2) 可达装饰墙周围的局部橙色轮廓。
#       mini_trace: |
#         原题中，一个可达内部蓝点最终只会作为蓝点保留，不会在周围产生成环橙边，
#         因为第二次洪填把它当成可穿过对象，最后只按结构墙画边。
#         本变体中，预涂步骤先把它也当作普通墙来描边；后续清理只删除“不可达装饰”
#         对应的蓝块和橙边，所以可达装饰周围的橙边会留在最终输出里。
#   complexity_calibration:
#     visible_difference: |
#       为避免退化回原题，生成器强制大多数样本同时出现至少一个可达装饰和至少一个
#       隐藏装饰。这样输出里既能看到“保留下来的装饰轮廓”，又能看到“被清掉的假轮廓”。
#     freedom_control: |
#       装饰仍限制为 1xk / kx1 / 2x2 小块，数量保持少量，避免规则依赖过多形态。
#   implicit_assumption_review:
#     - 第一次洪填仍假设标记在角上且唯一，因此可以稳定得到 marker-side 的 bg 区域 reach0。
#     - “结构墙 = 触边且与 reach0 4 邻接”的判定保留不变，所以隐藏装饰绝不能触边。
#     - 预涂会把结构墙两侧以及隐藏装饰周围都染上橙色，因此后续必须清除所有不在最终 reach
#       内的橙色，否则会残留错误边界。
#     - 最终保留橙色时使用“仍然存活的蓝墙”而不是仅结构墙；否则可达装饰轮廓会被误删。
#   placement_legality:
#     - 结构墙仍由触边 4 连通图形组成，且不覆盖左下角标记。
#     - 可达装饰放在 final reach 的内部、非触边、且不与结构墙 4 连通的位置，避免被误判为结构墙。
#     - 隐藏装饰放在 ~reach 的内部、非触边位置，确保 solver 会把它擦掉。
#   observability:
#     - 人类可直接看到：输入中所有蓝色看起来同色，但输出里只有某些内部蓝块旁边会保留橙边，
#       另一些蓝块则连同橙边一起消失；这迫使解题者区分“可达装饰”和“隐藏装饰”。
#   visual_cleanliness:
#     - 装饰数量少且与结构墙保持间隔，避免大片蓝橙重叠。
#     - 隐藏装饰规模小，避免把不可达侧染成大面积噪声。
#   color_inference:
#     - 没有新增颜色或按颜色 id 排序；颜色语义仍是 bg / wall / marker / fill 四类，
#       变化完全由空间可达性决定。


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
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not reach[nr, nc]:
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


def _find_marker_and_wall(g: np.ndarray, bg: int) -> tuple[tuple[int, int, int], int]:
    h, w = g.shape
    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    other_cols = [c for c in counts.keys() if c not in (bg, marker[2])]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])
    return marker, wall_col


def _structural_walls(g: np.ndarray, wall_col: int, reach0: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = g.shape
    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)
    return walls


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pre-paint orange borders around all wall-colored pixels, then erase the ones tied to hidden decorations.

    Input structure:
    - The background is the most common color, the start marker is a unique corner pixel, and blue pixels play two roles.
    - Border-touching blue components next to the marker-side area are structural walls; interior blue components are decorations.
    - Some decorations lie in the marker-reachable side, while others sit behind structural walls and should disappear.

    Rule:
    - First flood from the marker with every blue pixel treated as blocking, and use that coarse separation to pre-paint orange around all blue walls.
    - Then re-identify which blue components are truly structural, flood again while allowing non-structural blue pixels to be traversed, and erase hidden decorations.
    - Finally keep only the orange cells that still line the final reachable region or touch surviving blue walls, removing the extra orange tied to erased decorations.

    Key insight: The first orange pass is intentionally over-inclusive; the real job is to recognize which wall-colored pixels survive the reachability test and prune their bogus border artifacts.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    fill_col = 7

    out = g.copy()
    marker, wall_col = _find_marker_and_wall(g, bg)
    mr, mc, marker_col = marker

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )
    reach0_mask = np.zeros((h, w), dtype=bool)
    for r, c in reach0:
        reach0_mask[r, c] = True

    wall_mask = g == wall_col
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) != bg:
                continue
            near_any_wall = any(
                0 <= nr < h and 0 <= nc < w and wall_mask[nr, nc]
                for nr, nc in neighbors8(r, c)
            )
            on_initial_contour = (r, c) in reach0 and (
                r == 0 or c == 0 or r == h - 1 or c == w - 1
                or any(
                    not (0 <= nr < h and 0 <= nc < w) or (nr, nc) not in reach0
                    for nr, nc in neighbors4(r, c)
                )
            )
            if near_any_wall or on_initial_contour:
                out[r, c] = fill_col

    walls = _structural_walls(g, wall_col, reach0)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col, fill_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )
    reach_mask = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        reach_mask[r, c] = True

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    surviving_walls = {(r, c) for r in range(h) for c in range(w) if int(out[r, c]) == wall_col}
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) != fill_col:
                continue
            if (r, c) not in reach:
                out[r, c] = bg
                continue
            keep = False
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                keep = True
            elif any((nr, nc) in surviving_walls for nr, nc in neighbors8(r, c)):
                keep = True
            elif any(
                0 <= nr < h and 0 <= nc < w and not reach_mask[nr, nc]
                for nr, nc in neighbors4(r, c)
            ):
                keep = True
            if not keep:
                out[r, c] = bg

    return out


# ---- Palette ----
BG = 8
WALL = 1
MARKER = 6
FILL = 7

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

# ---- Decorations ----
NUM_VISIBLE_DECOR_RANGE = (1, 2)
NUM_HIDDEN_DECOR_RANGE = (1, 2)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

# ---- Outer retry budget ----
GEN_ATTEMPTS = 500


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


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _place_structural_walls(h: int, w: int, marker_rc: tuple[int, int], rng) -> np.ndarray | None:
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


def _paint_variant_output(
    structural: np.ndarray,
    visible_decor: np.ndarray,
    reach: np.ndarray,
    marker_rc: tuple[int, int],
) -> np.ndarray:
    h, w = structural.shape
    walls = structural | visible_decor
    out = np.full((h, w), BG, dtype=int)
    out[walls] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_wall = _dilate8(walls)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_wall | near_unreach)
    out[fill] = FILL
    return out


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
            placed[r0:r0 + ph, c0:c0 + pw] = True
            break
        else:
            return None
    return placed


def _variant_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    visible_halo = 0
    hidden_removed = 0
    for comp_arr in components_mask(inp == WALL, connectivity=4):
        comp = {(int(r), int(c)) for r, c in comp_arr}
        touches_border = any(r == 0 or c == 0 or r == inp.shape[0] - 1 or c == inp.shape[1] - 1 for r, c in comp)
        out_has_wall = any(out[r, c] == WALL for r, c in comp)
        if touches_border:
            continue
        if out_has_wall:
            halo = 0
            for r, c in comp:
                halo += sum(
                    1
                    for nr, nc in neighbors8(r, c)
                    if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and out[nr, nc] == FILL
                )
            if halo > 0:
                visible_halo += 1
        else:
            hidden_removed += 1
    return {"visible_halo_components": visible_halo, "hidden_removed_components": hidden_removed}


def _gen_attempt() -> np.ndarray | None:
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

    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    visible_allowed = reach & ~structural & ~edge
    visible_allowed &= ~_dilate4(structural)
    visible_allowed[marker_rc] = False

    hidden_allowed = ~reach & ~structural & ~edge

    n_visible = rng.randint(*NUM_VISIBLE_DECOR_RANGE)
    n_hidden = rng.randint(*NUM_HIDDEN_DECOR_RANGE)
    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    if not visible_placed.any() or not hidden_placed.any():
        return None

    out = _paint_variant_output(structural, visible_placed, reach, marker_rc)
    inp = out.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    metrics = _variant_metrics(inp, out)
    if metrics["visible_halo_components"] < 1 or metrics["hidden_removed_components"] < 1:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D10 variant")
