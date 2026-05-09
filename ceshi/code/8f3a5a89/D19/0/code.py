# Variant of 8f3a5a89
# Dimension: D19 — 符号编码/图例机制 (symbol_instructions)
# Change: 无显式图例——所有规则从输入输出对中隐式推断... → 网格角落放置颜色映射图例

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# Design analysis
# - concrete_form:
#   输入右上角放一个 2 列竖直图例，每行是 [墙色, 边界色]；图例下方与左侧各留 1 格缓冲，
#   这个右上角小盒子完全不参与主谜面。主谜面仍然是“左下角标记 + 若干边界墙 + 可达区 + 装饰墙”。
# - behavior_trace:
#   原题里所有结构墙都默认同一种颜色，solver 直接把可达区内边界全部刷成固定橙色。
#   现在例如图例是 [蓝->橙, 红->黄] 时，靠蓝墙的那段内边界刷橙，靠红墙的那段刷黄；
#   与墙不直接接触但属于同一条边界轮廓的边缘段，会沿边界从最近的墙色种子继续传播同色。
# - complexity_calibration:
#   每题固定 2-3 条图例映射，且每个墙色都至少在一个结构墙分量上出现，并在输出中贡献多格边界色，
#   避免“图例存在但只用到 1 种颜色”的退化。
# - implicit_assumptions_audit:
#   原代码依赖“最常见的非背景非标记色就是墙色”，加入图例和多墙色后这一点失效，因此必须改为先读图例。
#   原代码也会在四角里找唯一非背景角像素；现在右上角被图例占据，所以必须跳过受保护角。
#   原代码把 fill_col 固定写死为 7；现在必须按图例给不同墙段映射到不同边界色。
# - placement_legality:
#   生成器先保留右上角保护盒，再放结构墙；所有结构墙和装饰墙都不得进入保护盒、不得重叠。
#   隐藏装饰墙仍然禁止触边，避免被 solver 误判成结构墙。
# - visual_observability:
#   图例颜色和输出边界颜色一一对应，且每个颜色在输出里都有重复边界像素支撑；人可以从 2-3 组样例看出
#   “先读角落图例，再根据墙色给内边界上色”。
# - visual_cleanliness:
#   图例与主体之间留出缓冲带，主体内部的可见装饰只放在最终不会被刷边界色的背景内区，减少重叠干扰。
# - color_inference:
#   每题只用 2-3 对映射，墙色集合与边界色集合互不重叠，也都避开背景色 8 和标记色 6，
#   让颜色变化完全由图例显式给出而不是靠颜色 id 猜测。


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


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    reach = np.zeros((h, w), dtype=bool)
    if blocked[start] or not active_mask[start]:
        return reach
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and active_mask[nr, nc] and not blocked[nr, nc] and not reach[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


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


def _find_legend(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid must be large enough for legend protection"
    lc = w - 2

    entries: list[tuple[int, int]] = []
    for r in range(h):
        left = int(g[r, lc])
        right = int(g[r, lc + 1])
        if left == bg and right == bg:
            break
        assert left != bg and right != bg, "legend rows must be complete color pairs"
        entries.append((left, right))

    assert entries, "top-right legend is missing"
    assert len(entries) < h, "legend must terminate before bottom edge"
    assert int(g[len(entries), lc]) == bg and int(g[len(entries), lc + 1]) == bg, "legend needs a blank separator row"

    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[: len(entries), lc : lc + 2] = True

    protected = np.zeros_like(g, dtype=bool)
    protected[: len(entries) + 1, w - 3 :] = True
    return entries, legend_mask, protected


def _pick_wall_color(adj_cols: list[int], wall_order: dict[int, int]) -> int:
    counts = Counter(adj_cols)
    return min(counts, key=lambda col: (-counts[col], wall_order[col]))


def _compute_candidate_boundary(
    out: np.ndarray,
    reach: set[tuple[int, int]],
    walls: set[tuple[int, int]],
    active_mask: np.ndarray,
) -> np.ndarray:
    h, w = out.shape
    candidate = np.zeros((h, w), dtype=bool)
    for r, c in reach:
        if not active_mask[r, c] or int(out[r, c]) != BG:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            candidate[r, c] = True
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            candidate[r, c] = True
            continue
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and active_mask[nr, nc] and (nr, nc) not in reach:
                candidate[r, c] = True
                break
    return candidate


def _paint_boundary_by_legend(
    out: np.ndarray,
    candidate: np.ndarray,
    walls: set[tuple[int, int]],
    wall_at: dict[tuple[int, int], int],
    wall_to_fill: dict[int, int],
    wall_order: dict[int, int],
) -> np.ndarray:
    seeds: list[tuple[int, int, int, int]] = []
    h, w = out.shape
    for r, c in np.argwhere(candidate):
        adj_cols = [wall_at[(nr, nc)] for nr, nc in neighbors8(int(r), int(c)) if (nr, nc) in walls]
        if adj_cols:
            wall_col = _pick_wall_color(adj_cols, wall_order)
            seeds.append((wall_order[wall_col], int(r), int(c), wall_col))

    assert seeds, "boundary must touch at least one structural wall"

    label = np.full((h, w), -1, dtype=int)
    q = deque()
    for _, r, c, wall_col in sorted(seeds):
        if label[r, c] == -1:
            label[r, c] = wall_col
            q.append((r, c, wall_col))

    while q:
        r, c, wall_col = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and candidate[nr, nc] and label[nr, nc] == -1:
                label[nr, nc] = wall_col
                q.append((nr, nc, wall_col))

    assert not ((label == -1) & candidate).any(), "every boundary pixel must inherit some legend color"

    painted = out.copy()
    for r, c in np.argwhere(candidate):
        wall_col = int(label[r, c])
        painted[r, c] = wall_to_fill[wall_col]
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint a color-coded inner border around the region reachable from a corner marker.

    Input structure:
    - Background is the dominant color, and the bottom-left corner still holds a unique single-pixel marker.
    - Border-touching wall components separate reachable space from hidden space, but wall colors are no longer implicit.
    - A protected legend sits in the top-right corner: each row is a (wall color, boundary color) pair.

    Rule:
    - Read the legend first to learn which wall colors are meaningful and which fill color each one commands.
    - Flood-fill from the marker through background plus non-structural decorations; border-touching legend-listed walls remain barriers.
    - Delete unreachable decorative wall pixels, then paint the reachable inner boundary using the fill color paired with the nearby structural wall color.
    - Boundary cells that are only on the grid-edge part of the same contour inherit the nearest legend-colored wall segment along that contour.

    Key insight: The hard part is no longer “which fill color is fixed”, but “which corner pixels are instructions”; once the legend is carved out, the old structural-vs-decoration logic still applies.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_entries, legend_mask, protected = _find_legend(g, bg)
    active_mask = ~protected

    wall_to_fill = {wall: fill for wall, fill in legend_entries}
    wall_cols = [wall for wall, _ in legend_entries]
    wall_order = {wall: i for i, wall in enumerate(wall_cols)}
    out = g.copy()

    counts = Counter(int(v) for v in g[active_mask].ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        if protected[r, c]:
            continue
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique active corner marker found"
    mr, mc, marker_col = marker

    reach0 = _flood(
        (mr, mc),
        lambda r, c: active_mask[r, c] and int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    wall_at: dict[tuple[int, int], int] = {}
    for wall_col in wall_cols:
        for comp_arr in components_mask((g == wall_col) & active_mask, connectivity=4):
            comp = [(int(r), int(c)) for r, c in comp_arr]
            touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
            if not touches_border:
                continue
            adjacent_to_reach0 = any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp)
            if adjacent_to_reach0:
                for cell in comp:
                    walls.add(cell)
                    wall_at[cell] = wall_col

    reach = _flood(
        (mr, mc),
        lambda r, c: active_mask[r, c]
        and (
            int(g[r, c]) in (bg, marker_col)
            or (int(g[r, c]) in wall_to_fill and (r, c) not in walls)
        ),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if not active_mask[r, c]:
                continue
            if int(out[r, c]) in wall_to_fill and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    candidate = _compute_candidate_boundary(out, reach, walls, active_mask)
    out = _paint_boundary_by_legend(out, candidate, walls, wall_at, wall_to_fill, wall_order)
    out[legend_mask] = g[legend_mask]
    return out


# ---- Palette anchors ----
BG = 8
MARKER = 6
PALETTE = [0, 1, 2, 3, 4, 5, 7, 9]

# ---- Canvas size ----
GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)

# ---- Legend ----
LEGEND_ENTRY_RANGE = (2, 3)

# ---- Structural walls ----
NUM_STRUCTS_RANGE = (2, 4)
STRUCT_SHAPE_RETRIES = 12
RECT_MAX_DIM_DIV = 3
TRI_MAX_SIZE = 6
TRI_MIN_DIM_DIV = 3

# ---- Reachable region gate ----
REACH_RATIO_RANGE = (0.30, 0.90)

# ---- Decorations ----
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 4)
DECOR_BORDER = 1
DECOR_PADDING = 1

# ---- Output observability gates ----
MIN_FILL_PIXELS_PER_ENTRY = 4
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


def _choose_legend(rng) -> list[tuple[int, int]]:
    n = rng.randint(*LEGEND_ENTRY_RANGE)
    cols = rng.sample(PALETTE, 2 * n)
    walls = cols[:n]
    fills = cols[n:]
    rng.shuffle(walls)
    rng.shuffle(fills)
    return list(zip(walls, fills))


def _paint_legend(base: np.ndarray, entries: list[tuple[int, int]]) -> np.ndarray:
    g = base.copy()
    h, w = g.shape
    lc = w - 2
    for r, (wall, fill) in enumerate(entries):
        g[r, lc] = wall
        g[r, lc + 1] = fill
    return g


def _place_structural_walls(h, w, marker_rc, protected, min_components, rng):
    num = rng.randint(max(NUM_STRUCTS_RANGE[0], min_components), NUM_STRUCTS_RANGE[1])
    structural = np.zeros((h, w), dtype=bool)
    placed = 0
    for _ in range(num):
        for _try in range(STRUCT_SHAPE_RETRIES):
            cells = rng.choice(SHAPE_FACTORIES)(h, w, rng)
            if not cells:
                continue
            if marker_rc in cells:
                continue
            if any(protected[r, c] for r, c in cells):
                continue
            if any(structural[r, c] for r, c in cells):
                continue
            sep = _dilate8(structural)
            if any(sep[r, c] for r, c in cells):
                continue
            for r, c in cells:
                structural[r, c] = True
            placed += 1
            break
    if placed < min_components:
        return None
    comps = components_mask(structural, connectivity=4)
    if len(comps) < min_components:
        return None
    return structural


def _validate_structural(structural, reach, active_mask):
    reach_nbr = _dilate4(reach)
    h, w = structural.shape
    for comp_arr in components_mask(structural, connectivity=4):
        cmask = np.zeros((h, w), dtype=bool)
        cmask[comp_arr[:, 0], comp_arr[:, 1]] = True
        if not _touches_border(cmask):
            return False
        if not (cmask & reach_nbr).any():
            return False
        if (cmask & (~active_mask)).any():
            return False
    return True


def _color_structural_components(structural: np.ndarray, wall_cols: list[int], rng) -> np.ndarray:
    h, w = structural.shape
    colored = np.full((h, w), BG, dtype=int)
    comps = [comp_arr for comp_arr in components_mask(structural, connectivity=4)]
    if len(comps) < len(wall_cols):
        raise ValueError("not enough structural components for all legend walls")

    comp_ids = list(range(len(comps)))
    rng.shuffle(comp_ids)
    assigned_cols = wall_cols[:]
    while len(assigned_cols) < len(comps):
        assigned_cols.append(rng.choice(wall_cols))
    rng.shuffle(assigned_cols)
    for forced_color, idx in zip(wall_cols, comp_ids):
        assigned_cols[idx] = forced_color

    for comp_arr, wall_col in zip(comps, assigned_cols):
        colored[comp_arr[:, 0], comp_arr[:, 1]] = wall_col
    return colored


def _place_decor_patches(allowed_mask: np.ndarray, count: int, rng):
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    patches: list[np.ndarray] = []
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
            mask = np.zeros((h, w), dtype=bool)
            mask[r0 : r0 + ph, c0 : c0 + pw] = True
            placed |= mask
            patches.append(mask)
            break
        else:
            return None
    return patches


def _gen_attempt():
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    legend_entries = _choose_legend(rng)
    blank = np.full((h, w), BG, dtype=int)
    legend_grid = _paint_legend(blank, legend_entries)
    _, legend_mask, protected = _find_legend(legend_grid, BG)
    active_mask = ~protected

    structural = _place_structural_walls(h, w, marker_rc, protected, len(legend_entries), rng)
    if structural is None or structural[marker_rc]:
        return None

    reach = _flood_mask(marker_rc, structural | protected, active_mask)
    reach_frac = float(reach[active_mask].mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None
    if not _validate_structural(structural, reach, active_mask):
        return None

    structural_colored = _color_structural_components(structural, [wall for wall, _ in legend_entries], rng)
    base_inp = np.full((h, w), BG, dtype=int)
    base_inp[legend_mask] = legend_grid[legend_mask]
    base_inp[structural] = structural_colored[structural]
    base_inp[marker_rc] = MARKER

    base_out = main(base_inp)
    fill_cols = [fill for _, fill in legend_entries]
    fill_union = np.isin(base_out, fill_cols) & active_mask
    if not fill_union.any() or len(components_mask(fill_union, connectivity=4)) != 1:
        return None
    if any(int(((base_out == fill) & active_mask).sum()) < MIN_FILL_PIXELS_PER_ENTRY for fill in fill_cols):
        return None

    # Reject ambiguous seed cells so the wall->boundary mapping stays easy to see.
    reach_after = _flood(
        marker_rc,
        lambda r, c: active_mask[r, c] and int(base_inp[r, c]) in (BG, MARKER),
        h,
        w,
    )
    walls: set[tuple[int, int]] = set()
    wall_at: dict[tuple[int, int], int] = {}
    for wall_col, _fill in legend_entries:
        for comp_arr in components_mask((base_inp == wall_col) & active_mask, connectivity=4):
            comp = [(int(r), int(c)) for r, c in comp_arr]
            touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
            adjacent = any(any((nr, nc) in reach_after for nr, nc in neighbors4(r, c)) for r, c in comp)
            if touches_border and adjacent:
                for cell in comp:
                    walls.add(cell)
                    wall_at[cell] = wall_col
    candidate = _compute_candidate_boundary(base_inp.copy(), set(map(tuple, np.argwhere(reach))), walls, active_mask)
    for r, c in np.argwhere(candidate):
        adj_cols = {wall_at[(nr, nc)] for nr, nc in neighbors8(int(r), int(c)) if (nr, nc) in walls}
        if len(adj_cols) > 1:
            return None

    n_decor = rng.randint(*NUM_DECOR_RANGE)
    n_visible = rng.randint(0, n_decor)
    n_hidden = n_decor - n_visible

    visible_allowed = active_mask & (base_out == BG)
    hidden_allowed = active_mask & (~reach) & (~structural)
    hidden_allowed[0, :] = hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = hidden_allowed[:, -1] = False

    visible_patches = _place_decor_patches(visible_allowed, n_visible, rng)
    if visible_patches is None:
        return None
    hidden_patches = _place_decor_patches(hidden_allowed, n_hidden, rng)
    if hidden_patches is None:
        return None

    expected_out = base_out.copy()
    inp = base_inp.copy()
    wall_choices = [wall for wall, _ in legend_entries]

    for patch in visible_patches:
        wall_col = rng.choice(wall_choices)
        inp[patch] = wall_col
        expected_out[patch] = wall_col

    for patch in hidden_patches:
        wall_col = rng.choice(wall_choices)
        inp[patch] = wall_col

    if not np.array_equal(main(inp), expected_out):
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
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D19")
