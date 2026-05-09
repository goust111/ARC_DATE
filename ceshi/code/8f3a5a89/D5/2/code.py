# Variant of 8f3a5a89
# Dimension: D5 — 结构墙识别规则 (obstacle)
# Change: 结构墙 = 墙色连通分量中满足：(1) 触碰网格边缘 (2) 4-邻接可达区域（第一次洪填的reach0）的分量；非结构墙视为装饰
#         -> 墙壁颜色多样化: 结构墙使用显式障碍色, 装饰墙使用另一种颜色, 且可选第三色作为同功能干扰色
#
# Design analysis for the D5 variant
# ==================================
# a) Concretization and behavior trace
# - Input keeps the original chamber layout: the marker is still a unique magenta
#   pixel in the bottom-left corner, and orange is still painted as a 1-cell inner
#   border in the output.
# - The changed part is wall semantics:
#   * structural walls are always drawn with one explicit "blocking" color;
#   * decoration walls are drawn with a different color;
#   * some examples also contain a third color that behaves exactly like decoration.
# - Small trace:
#   * Original solver: flood 0 treats every wall-colored pixel as blocked, then keeps
#     only border-touching/reach-adjacent components as true walls.
#   * New solver: first identify the unique non-marker color that reaches the frame;
#     that color alone blocks motion. Decorative colors are passable from the start.
#   * Difference: a red or maroon patch that sits in the reachable region no longer
#     creates a temporary obstacle, and a hidden red/maroon patch is removed purely
#     because it is on the unreachable side of the blue barrier.
#
# b) Complexity calibration
# - The variant is made visible by guaranteeing at least one decorative patch in every
#   generated input, and by introducing the third color in only a subset of samples.
# - The color role system stays small: one blocking color, one main decorative color,
#   one optional decorative alias. This is enough to make the rule visible without
#   creating a large palette-mapping problem.
#
# c) Old assumptions that must change
# - "Most frequent non-bg non-marker color is the wall color" is no longer valid once
#   blue, red, and maroon can coexist.
# - "All wall-colored pixels are blocked in the first flood" is no longer valid because
#   decorative colors must be traversable.
# - "Any wall-colored hidden pixel should be erased" must be generalized to all
#   decorative colors, while the blocking color must never be erased.
#
# d) Placement legality
# - Structural components are generated exactly as before: 4-connected, border-touching,
#   non-overlapping, and never covering the marker.
# - Decorative patches are placed only on allowed cells, with border and padding
#   constraints, so they do not overlap each other or touch the grid frame.
# - Hidden decorative patches are only placed outside the structural reach region, so
#   the solver has a meaningful erase/not-erase distinction.
#
# e) Visual observability
# - A human can now see that blue pieces behave like rigid barriers while red/maroon
#   pieces do not. The examples still show the same orange-border effect, but the
#   reason a patch is kept or erased is now visually tied to color.
#
# f) Visual cleanliness
# - Decorative patches remain small and isolated, and they are never drawn on top of
#   orange border cells. This avoids ambiguous overlaps between barrier, border, and
#   decoration.
#
# g) Color-change inferability
# - The third color is not a new behavior; it is an alias of the decorative role.
#   Generator samples with and without that alias let training examples demonstrate
#   that "non-blue wall colors are cosmetic" while keeping the mapping simple enough
#   to infer from a few examples.
#
# h) Quantitative anti-degeneration target
# - Generator validation later checks that inputs frequently contain decorative pixels
#   in colors different from the structural color, and often contain the third color.
#   This ensures the dataset is actually exercising the new rule rather than falling
#   back to single-color walls.
#
# i) New solver assumptions and generator guarantees
# - Solver assumes exactly one non-marker, non-background color touches the frame.
#   Generator guarantees this by forbidding decorative colors from the border.
# - Solver assumes every decorative color is passable. Generator guarantees decorative
#   patches are never required to separate regions.
# - Solver assumes the corner marker is still unique and isolated by color count.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


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


def _infer_role_colors(g: np.ndarray, bg: int, marker_col: int) -> tuple[int, tuple[int, ...]]:
    counts = Counter(int(v) for v in g.ravel())
    other_cols = [c for c in counts if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found"

    h, w = g.shape
    border_cols = set()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            v = int(g[r, c])
            if v in other_cols:
                border_cols.add(v)
    assert len(border_cols) == 1, "expected exactly one explicit border wall color"

    structural_col = next(iter(border_cols))
    decor_cols = tuple(c for c in other_cols if c != structural_col)
    return structural_col, decor_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw an orange inner rim around the chamber reachable from the corner marker.

    Input structure:
    - The background is the dominant color, and a unique marker pixel sits on one corner.
    - One non-marker color is reserved for the real barrier pieces, and those pieces are
      the only non-background objects allowed to touch the grid frame.
    - One or two other non-marker colors form decorative wall fragments that may lie
      inside the reachable side or hide behind the barrier.

    Rule:
    - Identify the blocking wall color from the frame-touching non-marker pixels.
    - Flood-fill from the marker through background and decorative colors, blocking only
      on the structural wall color.
    - Erase decorative fragments that are not reached from the marker side.
    - Paint orange on every reachable background cell that lies on the grid edge, sits
      next to a structural wall, or borders an unreached cell.

    Key insight:
    - In this variant obstacle role is encoded by color, not by testing each wall
      component's position after a preliminary flood.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    structural_col, decor_cols = _infer_role_colors(g, bg, marker_col)
    decor_set = set(decor_cols)
    structural_walls = {
        (int(r), int(c))
        for r, c in np.argwhere(g == structural_col)
    }

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) == bg or int(g[r, c]) == marker_col or int(g[r, c]) in decor_set,
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) in decor_set and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((nr, nc) in structural_walls for nr, nc in neighbors8(r, c)):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = fill_col

    return out


BG = 8
STRUCT = 1
DECOR_MAIN = 2
DECOR_ALIAS = 9
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

NUM_STRUCTS_RANGE = (1, 4)
STRUCT_SHAPE_RETRIES = 8
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

REACH_RATIO_RANGE = (0.38, 0.87)

NUM_DECOR_RANGE = (1, 4)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1
DECOR_ALIAS_PROB = 0.45

GEN_ATTEMPTS = 300


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


def _flood_mask(start, blocked, h, w):
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


def _touches_border(mask):
    return mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()


def _dilate4(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask):
    m = _dilate4(mask)
    m[1:, 1:] |= mask[:-1, :-1]
    m[1:, :-1] |= mask[:-1, 1:]
    m[:-1, 1:] |= mask[1:, :-1]
    m[:-1, :-1] |= mask[1:, 1:]
    return m


def _place_structural_walls(h, w, marker_rc, rng):
    num = rng.randint(*NUM_STRUCTS_RANGE)
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _ in range(STRUCT_SHAPE_RETRIES):
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


def _validate_structural(structural, reach, h, w):
    reach_nbr = _dilate4(reach)
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
    return True


def _paint_output(structural, reach, marker_rc):
    out = np.full(structural.shape, BG, dtype=int)
    out[structural] = STRUCT
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL
    return out


def _place_decor_patches(allowed_mask, count, rng):
    h, w = allowed_mask.shape
    occupied = np.zeros((h, w), dtype=bool)
    patches = []
    if count == 0:
        return patches

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
            canvas = np.where(allowed_mask & ~occupied, 0, 9)
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

            patch_mask = np.zeros((h, w), dtype=bool)
            patch_mask[r0:r0 + ph, c0:c0 + pw] = True
            occupied |= patch_mask
            patches.append(patch_mask)
            break
        else:
            return None

    return patches


def _assign_decor_colors(count, rng):
    assert count >= 1
    use_alias = count >= 2 and rng.random() < DECOR_ALIAS_PROB
    colors = [DECOR_MAIN] * count
    if use_alias:
        alias_count = rng.randint(1, count - 1)
        alias_ix = set(rng.sample(range(count), alias_count))
        for idx in alias_ix:
            colors[idx] = DECOR_ALIAS
    return colors


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    if not _validate_structural(structural, reach, h, w):
        return None

    out = _paint_output(structural, reach, marker_rc)
    fill_mask = out == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    if n_decor == 1 and rng.random() < DECOR_ALIAS_PROB:
        n_decor = 2
    if n_decor == 1:
        n_visible = rng.randint(0, 1)
    else:
        branch_roll = rng.random()
        if branch_roll < 0.5:
            n_visible = rng.randint(1, n_decor - 1)
        elif branch_roll < 0.75:
            n_visible = 0
        else:
            n_visible = n_decor
    n_hidden = n_decor - n_visible

    visible_allowed = reach & (out == BG)
    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    visible_patches = _place_decor_patches(visible_allowed, n_visible, rng)
    if visible_patches is None:
        return None
    hidden_patches = _place_decor_patches(hidden_allowed, n_hidden, rng)
    if hidden_patches is None:
        return None

    all_patches = visible_patches + hidden_patches
    if len(all_patches) != n_decor:
        return None
    decor_colors = _assign_decor_colors(len(all_patches), rng)

    out_final = out.copy()
    for patch_mask, color in zip(visible_patches, decor_colors[:n_visible]):
        out_final[patch_mask] = color

    inp = out_final.copy()
    inp[inp == FILL] = BG
    for patch_mask, color in zip(hidden_patches, decor_colors[n_visible:]):
        inp[patch_mask] = color

    if not np.array_equal(main(inp), out_final):
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D5 variant")
