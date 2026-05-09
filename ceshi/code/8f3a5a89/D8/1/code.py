# Variant of 8f3a5a89
# Dimension: D8 — 洪填连通性 (path)
# Change: 4-连通洪填（只通过上下左右邻居扩展），8-邻接用于判断结构墙附近的边界像素... → 4-连通洪填但边界判定用4-邻接
#
# Design analysis:
#   concretization:
#     - 输入仍然是蓝墙 / 青底 / 左下角品红单像素标记。
#     - solver 仍先用 4-连通洪填找出从标记可达的一侧，再删除不可达侧的装饰墙。
#     - 唯一核心变化是橙色内边界的“靠墙”判定：原题把与结构墙对角接触的可达底色也算边界，
#       本变体只把与结构墙上下左右相邻的可达底色算边界。
#   behavior_trace:
#     - 例：结构墙做成一条从上边界出发、向右台阶式延伸到右边界的 1 像素厚路径。
#     - 在某个台阶拐点 `(r, c)` 的左下角 `(r+1, c-1)`，该格与墙只对角接触。
#     - 原题：`neighbors8` 命中 `(r, c)`，所以 `(r+1, c-1)` 会涂成橙色。
#     - 本变体：`neighbors4` 看不到 `(r, c)`，且该格四邻都仍在可达区，于是保持青色。
#     - 差异正好落在这些“对角擦肩而过”的像素上，边界环会比原题更稀疏。
#   complexity_calibration:
#     - 若继续随机放直线/矩形，很多样本里新旧规则会完全等价，看不出 D8 变化。
#     - 因此生成器强制包含 1 个连接上边和右边的台阶结构墙，并让它产生至少 2 个对角见证点；
#       同时保留 0-3 个额外结构墙和 0-3 个装饰墙，避免任务被简化成单一模板。
#   implicit_assumptions_review:
#     - 原 `main()` 中唯一依赖旧 D8 的代码是 `neighbors8(r, c)` 的靠墙判定；洪填本身已是 4-连通，
#       结构墙识别也基于 4-连通，因此其余逻辑可以保持。
#     - 但原 generator 的 `_paint_output()` 用 `_dilate8(structural)` 生成橙边，必须同步改成 `_dilate4`。
#     - 原 generator 还假设“只要 solver 可复原就够了”；对本变体不够，因为很多输入会让新旧输出相同。
#       现在额外要求同一输入在旧 8 邻接规则下至少多出 2 个橙色像素。
#     - 原 generator 额外要求橙边本身是单一 4-连通环；这个前提依赖 8 邻接靠墙判定，
#       在本变体里会错误过滤掉合法样本，因此必须删除。
#   placement_legality:
#     - 台阶墙严格 4-连通、只占 1 像素厚、不覆盖左下角标记，并始终触碰上边和右边。
#     - 其左下对角见证点周围做 8 邻域保护，禁止额外结构墙或可见装饰覆盖，避免变体信号被破坏。
#     - 隐藏装饰仍限制为不触边，防止被 solver 误认成结构墙。
#   visual_observability:
#     - 2-4 个样例中，人能直接看到：有些靠近墙角的青色格在原题直觉下像“应该涂边”，但输出故意不涂，
#       而靠边界、靠正交墙面、靠不可达区的橙边仍照常出现。
#     - 台阶墙提供重复的同类证据，不依赖颜色 id 或不可见元数据。
#   cleanliness:
#     - 仍只用 4 种颜色，墙/标记/边界职责不变，输出不会叠多层信息。
#   post-change_main_assumptions:
#     - `main()` 仍假设存在唯一角落单像素标记、墙色是最常见非背景非标记色、结构墙是触边且贴近首次可达区的
#       4-连通蓝色分量。generator 保持这些前提成立。

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, random_free_location


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


def _solve(grid: np.ndarray, wall_neighbor_mode: int) -> np.ndarray:
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
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    wall_neighbors = neighbors4 if wall_neighbor_mode == 4 else _neighbors8
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((nr, nc) in walls for nr, nc in wall_neighbors(r, c)):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = fill_col

    return out


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a 1-cell inner outline on the side of the maze that the corner marker can reach.

    Input structure:
    - The background is the dominant color, with one unique corner marker pixel in magenta.
    - Blue pixels split into border-anchored structural walls and smaller decorative wall fragments.
    - The marker starts in the bottom-left corner and the flood still moves only in 4 directions.

    Rule:
    - First find the region reachable from the marker when every blue pixel blocks movement.
    - Keep only those border-touching blue components that truly face that reachable side as structural walls,
      then erase decorative blue pixels stranded on the hidden side.
    - Paint orange on reachable background cells that lie on the grid edge, touch a structural wall in 4-neighborhood,
      or sit next to a non-reachable cell in 4-neighborhood.

    Key insight:
    - The reachable region is unchanged from the original task; only the wall-adjacency test changes.
    - A diagonal brush with a wall corner no longer counts as "near the wall", so some corner-touching cells stay background.
    """
    return _solve(grid, wall_neighbor_mode=4)


BG = 8
WALL = 1
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

NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

MIN_VARIANT_DIFF = 2
GEN_ATTEMPTS = 400


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
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _shape_top_right_stair(h, w, rng):
    max_turns = min(4, h - 4, w - 4)
    if max_turns < 2:
        return None
    turns = rng.randint(2, max_turns)
    rows = sorted(rng.sample(list(range(2, h - 2)), turns))
    cols = sorted(rng.sample(list(range(2, w - 2)), turns))

    cells: set[tuple[int, int]] = set()
    for r in range(0, rows[0] + 1):
        cells.add((r, cols[0]))
    for i in range(turns - 1):
        for c in range(cols[i], cols[i + 1] + 1):
            cells.add((rows[i], c))
        for r in range(rows[i], rows[i + 1] + 1):
            cells.add((r, cols[i + 1]))
    for c in range(cols[-1], w):
        cells.add((rows[-1], c))

    witnesses = [(rows[i] + 1, cols[i] - 1) for i in range(turns)]
    return sorted(cells), witnesses


def _place_structural_walls(h, w, marker_rc, rng):
    total = rng.randint(*NUM_STRUCTS_RANGE)
    base = _shape_top_right_stair(h, w, rng)
    if base is None:
        return None, None
    base_cells, witnesses = base

    structural = np.zeros((h, w), dtype=bool)
    for r, c in base_cells:
        if (r, c) == marker_rc:
            return None, None
        structural[r, c] = True

    witness_mask = np.zeros((h, w), dtype=bool)
    for r, c in witnesses:
        witness_mask[r, c] = True
    protected = _dilate8(witness_mask)
    protected[marker_rc] = True

    for _ in range(total - 1):
        placed = False
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells:
                continue
            if any(structural[r, c] or protected[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed = True
            break
        if not placed:
            continue

    return structural, witness_mask


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


def _paint_output(structural, reach, marker_rc, wall_neighbor_mode):
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True

    near_struct = _dilate4(structural) if wall_neighbor_mode == 4 else _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL
    return out


def _place_decor(allowed_mask, count, rng):
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

    structural, witness_mask = _place_structural_walls(h, w, marker_rc, rng)
    if structural is None or witness_mask is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    reach_frac = reach.mean()
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None
    if not _validate_structural(structural, reach, h, w):
        return None

    witness_live = witness_mask & reach & (~_dilate4(structural)) & (~_dilate4(~reach))
    if witness_live.sum() < MIN_VARIANT_DIFF:
        return None

    out = _paint_output(structural, reach, marker_rc, wall_neighbor_mode=4)
    if not (out == FILL).any():
        return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    protected_visible = _dilate8(witness_live)
    visible_allowed = reach & (out == BG) & (~protected_visible)
    hidden_allowed = (~reach) & (~structural)
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
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    if not np.array_equal(main(inp), out_final):
        return None

    legacy_out = _solve(inp, wall_neighbor_mode=8)
    if int(np.count_nonzero(legacy_out != out_final)) < MIN_VARIANT_DIFF:
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D8 variant")
