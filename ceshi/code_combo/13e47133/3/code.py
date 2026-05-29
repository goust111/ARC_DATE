# Combo variant of 13e47133 (index 3)
# combo_spec: D1/2+D4/1
# Name: BFS扩散+碰撞停止

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    原题在每个墙围成的房间里读取“边界深度 -> 颜色”的调色板，然后按切比雪夫深度
#    重复涂成矩形环。这里两处同时改动：
#    - D1/2：孤立彩色像素不再是调色板槽位，而是 4 邻域多源 BFS 的种子。
#    - D4/1：房间里额外出现的多像素色块是阻挡物，波前不能穿过它们。
#    小例子：一个 8x8 房间里，左上/右上/左下有三个单像素种子，中部偏右有一个 2x3
#    阻挡块。原题会得到整齐的矩形框；仅 BFS 会得到三块曼哈顿 Voronoi；融合后则是
#    “绕障碍的 Voronoi”，右上的颜色不能直穿阻挡块，部分区域改由其余种子绕行占领。
# b) Complexity calibration
#    只做 BFS 而没有障碍时，视觉上容易退化成普通距离分区；只做障碍而种子太少时，
#    则只是单色扩散被挖洞。生成器因此强制：
#    - 至少 1 个区域含 3 个种子，保证竞争边界明显；
#    - 至少 1 个带阻挡的区域同时也是多种子竞争区域；
#    - 阻挡物只取少量紧凑矩形块，避免无结构噪声。
# c) Implicit assumptions audit against the original main()
#    - 原 `_boundary_distance(comp_mask)` 用来决定所有输出色；融合后该前提失效，
#      只保留给生成阶段做“不要退化回原题/D4-only”的对照指标。
#    - 原代码把所有非 base 像素都当作调色板标记；融合后必须先区分：
#      单像素连通分量 = 种子，多像素连通分量 = 阻挡物，否则会把障碍误当颜色线索。
#    - 原 `palette[d % m]` 会无条件覆盖整个区域；融合后必须显式维护可达性，
#      只能在非阻挡自由格上进行 BFS，且未被任何种子到达的格子保留 base。
# d) Placement legality
#    生成器仍通过递归矩形切分得到 3-6 个房间。种子只放在模板锚点上，阻挡块只放在
#    房间内部、距边界至少 1 格的位置，不与墙、其他阻挡物或种子重叠。
# e) Visual observability
#    解题者可以直接看到三类角色：墙、孤立单像素种子、紧凑多像素阻挡块。输出中的
#    颜色边界会围绕阻挡物弯折，因此可以从少量样例归纳出“多源扩散 + 障碍绕行/阻断”。
# f) Visual tidiness
#    每个房间至多一个阻挡块，种子位置从少量模板中选取，区域数量也受限，避免一张图
#    里同时出现过多交叠机制。
# g) Inference of changes
#    颜色编号本身不参与任何隐藏排序。颜色只表示“哪一个种子”以及“哪一块是障碍”；
#    推理依赖的是空间结构：单像素 vs 多像素、距离竞争、以及被阻挡后的绕行结果。


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _cc4(mask: np.ndarray):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Legacy helper for generator-side anti-degeneration checks.
    Compute 8-neigh (Chebyshev) distance to the component boundary.
    """
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _classify_region_colors(g: np.ndarray, comp_mask: np.ndarray):
    base = _mode(g[comp_mask])
    seed_cells: list[tuple[int, int, int]] = []
    blocker_mask = np.zeros_like(comp_mask, dtype=bool)

    for color in np.unique(g[comp_mask]):
        color = int(color)
        if color == base:
            continue
        for pts in _cc4(comp_mask & (g == color)):
            if len(pts) == 1:
                r, c = map(int, pts[0])
                seed_cells.append((r, c, color))
            else:
                for r, c in pts:
                    blocker_mask[int(r), int(c)] = True

    seed_cells.sort(key=lambda x: (x[0], x[1], x[2]))
    return int(base), seed_cells, blocker_mask


def _solve_bfs(grid: np.ndarray, *, collision_stop: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall

    for pts_arr in _cc4(free):
        pts = sorted((int(r), int(c)) for r, c in pts_arr)
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        base, seeds, blocker_mask = _classify_region_colors(g, comp_mask)
        walkable = comp_mask & ~blocker_mask if collision_stop else comp_mask

        for r, c in pts:
            if blocker_mask[r, c]:
                out[r, c] = int(g[r, c])
            else:
                out[r, c] = int(base)

        if not seeds:
            continue

        owner = np.full((h, w), -1, dtype=int)
        q: deque[tuple[int, int]] = deque()

        for idx, (r, c, color) in enumerate(seeds):
            if collision_stop and blocker_mask[r, c]:
                continue
            owner[r, c] = idx
            out[r, c] = int(color)
            q.append((r, c))

        while q:
            r, c = q.popleft()
            idx = int(owner[r, c])
            color = int(seeds[idx][2])
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and walkable[rr, cc] and owner[rr, cc] == -1:
                    owner[rr, cc] = idx
                    out[rr, cc] = color
                    q.append((rr, cc))

    return out


def _build_palette(seed_cells: list[tuple[int, int, int]], base: int, dist: np.ndarray) -> list[int]:
    dist_to_color: dict[int, int] = {}
    for r, c, color in sorted(seed_cells, key=lambda x: (int(dist[x[0], x[1]]), x[0], x[1])):
        d = int(dist[r, c])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = int(color)

    if not dist_to_color:
        return [int(base)]

    palette: list[int] = []
    depths = sorted(dist_to_color)
    if depths[0] > 0:
        palette.append(int(base))
    for d in depths:
        palette.append(int(dist_to_color[d]))
    return palette


def _solve_palette_collision(grid: np.ndarray) -> np.ndarray:
    """
    D4-only style baseline for generator checks: palette bands with blockers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall

    for pts_arr in _cc4(free):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        base, seed_cells, blocker_mask = _classify_region_colors(g, comp_mask)
        dist = _boundary_distance(comp_mask)
        palette = _build_palette(seed_cells, base, dist)

        rows, cols = np.where(comp_mask)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        local_comp = comp_mask[r0 : r1 + 1, c0 : c1 + 1]
        local_block = blocker_mask[r0 : r1 + 1, c0 : c1 + 1]
        solid = (~local_comp) | local_block

        bh, bw = solid.shape
        row_pref = np.zeros((bh, bw + 1), dtype=int)
        row_pref[:, 1:] = np.cumsum(solid.astype(int), axis=1)
        col_pref = np.zeros((bh + 1, bw), dtype=int)
        col_pref[1:, :] = np.cumsum(solid.astype(int), axis=0)
        m = len(palette)

        for r, c in pts:
            if blocker_mask[r, c]:
                continue
            lr, lc = r - r0, c - c0
            depths: list[int] = []
            if col_pref[lr, lc] == 0:
                depths.append(lr)
            if (col_pref[bh, lc] - col_pref[lr + 1, lc]) == 0:
                depths.append(bh - 1 - lr)
            if row_pref[lr, lc] == 0:
                depths.append(lc)
            if (row_pref[lr, bw] - row_pref[lr, lc + 1]) == 0:
                depths.append(bw - 1 - lc)

            if depths:
                out[r, c] = int(palette[min(depths) % m])
            else:
                out[r, c] = int(base)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand colored seeds through wall-bounded rooms while interior blocker patches stop the waves.

    Input structure:
    - One wall color partitions the grid into several 4-connected rooms.
    - Inside each room, most cells share a base color.
    - Isolated single-pixel non-base cells are colored seeds, while compact multi-pixel non-base patches are blockers.

    Rule:
    - In each room, launch a simultaneous 4-neighbor BFS from all single-pixel seeds.
    - A wave may claim only non-wall, non-blocker cells; when a frontier would step into a blocker patch, that branch stops there.
    - If multiple seeds could reach a cell at the same BFS time, the seed earlier in reading order wins.
    - Blocker cells themselves remain unchanged, and any free cell unreachable from every seed stays in the room's base color.

    Key insight: The informative colored pixels are no longer a boundary-depth palette; they are competing sources whose Voronoi-like territories become bent and clipped by obstacle patches.
    """
    return _solve_bfs(grid, collision_stop=True)


# --------------- generator ---------------

# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers / seeds ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 4, 5]  # 0/1/2/3
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 7

# ---- Blockers ----
MIN_BLOCKER_REGIONS = 1
MAX_BLOCKER_REGIONS = 3
BLOCKER_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3)]

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 500

ANCHOR_PATTERNS = {
    1: [
        ("tl",),
        ("tr",),
        ("bl",),
        ("br",),
        ("c",),
    ],
    2: [
        ("tl", "br"),
        ("tr", "bl"),
        ("ml", "mr"),
        ("tm", "bm"),
        ("tl", "mr"),
        ("tr", "ml"),
    ],
    3: [
        ("tl", "tr", "bl"),
        ("tl", "tr", "br"),
        ("tl", "bl", "br"),
        ("tr", "bl", "br"),
        ("tl", "c", "br"),
        ("tr", "c", "bl"),
    ],
}


def _place_walls(H: int, W: int, rng: random.Random):
    wall_mask = np.zeros((H, W), dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)

    for _ in range(n_splits):
        candidates = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            can_h = (r1 - r0 + 1) >= SPLITTABLE_LEN
            can_v = (c1 - c0 + 1) >= SPLITTABLE_LEN
            if can_h or can_v:
                candidates.append((i, can_h, can_v))
        if not candidates:
            break

        i, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(i)
        axes = ("h",) * can_h + ("v",) * can_v
        axis = rng.choice(axes)

        if axis == "h":
            rr = rng.randint(r0 + MIN_REGION_DIM, r1 - MIN_REGION_DIM)
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _anchor_pos(rect, name: str) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    rm = (r0 + r1) // 2
    cm = (c0 + c1) // 2
    if name == "tl":
        return r0, c0
    if name == "tr":
        return r0, c1
    if name == "bl":
        return r1, c0
    if name == "br":
        return r1, c1
    if name == "tm":
        return r0, cm
    if name == "bm":
        return r1, cm
    if name == "ml":
        return rm, c0
    if name == "mr":
        return rm, c1
    if name == "c":
        return rm, cm
    raise ValueError(name)


def _valid_pattern(rect, pattern: tuple[str, ...]) -> bool:
    pts = [_anchor_pos(rect, name) for name in pattern]
    if len(set(pts)) != len(pts):
        return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1]) < 2:
                return False
    return True


def _choose_markers(rect, marker_pool: list[int], rng: random.Random, forced_n: int | None = None):
    n = forced_n
    if n is None:
        n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(int(n), MARKERS_PER_REGION_MAX)
    if n == 0:
        return []

    candidates = [p for p in ANCHOR_PATTERNS[n] if _valid_pattern(rect, p)]
    if not candidates:
        return []

    pattern = rng.choice(candidates)
    colors = rng.sample(marker_pool, n)
    markers = []
    for color, name in zip(colors, pattern):
        rr, cc = _anchor_pos(rect, name)
        markers.append((int(color), int(rr), int(cc)))
    return markers


def _pick_colors(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    blocker = rng.choice(colors)
    colors.remove(blocker)
    return bg, wall, blocker, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    if not rects:
        return None, None, None

    focus_candidates = [
        idx for idx, (r0, c0, r1, c1) in enumerate(rects)
        if (r1 - r0 + 1) >= 7 and (c1 - c0 + 1) >= 7
    ]
    if not focus_candidates:
        return None, None, None
    focus_idx = rng.choice(focus_candidates)

    used = set()
    markers_by_rect: dict[tuple[int, int, int, int], list[tuple[int, int, int]]] = {}

    for idx, rect in enumerate(rects):
        forced_n = 3 if idx == focus_idx else None
        markers = _choose_markers(rect, marker_pool, rng, forced_n=forced_n)
        if idx == focus_idx and len(markers) < 3:
            return None, None, None
        markers_by_rect[rect] = markers
        for col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None, None
            if grid[rr, cc] != bg:
                return None, None, None
            grid[rr, cc] = col
            used.add(int(col))

    return used, markers_by_rect, focus_idx


def _interior_blocker_candidates(rect):
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    out = []
    for bh, bw in BLOCKER_SHAPES:
        if bh > h - 2 or bw > w - 2:
            continue
        rr_lo = r0 + 1
        rr_hi = r1 - bh
        cc_lo = c0 + 1
        cc_hi = c1 - bw
        if rr_lo > rr_hi or cc_lo > cc_hi:
            continue
        for rr in range(rr_lo, rr_hi + 1):
            for cc in range(cc_lo, cc_hi + 1):
                out.append((rr, cc, rr + bh - 1, cc + bw - 1))
    return out


def _candidate_score(candidate, rect):
    rr0, cc0, rr1, cc1 = candidate
    r0, c0, r1, c1 = rect
    room_cr = (r0 + r1) / 2.0
    room_cc = (c0 + c1) / 2.0
    cand_cr = (rr0 + rr1) / 2.0
    cand_cc = (cc0 + cc1) / 2.0
    return abs(cand_cr - room_cr) + abs(cand_cc - room_cc)


def _stamp_blocker_in_rect(grid, rect, bg, blocker, markers, rng, *, central: bool):
    candidates = _interior_blocker_candidates(rect)
    if not candidates:
        return False

    markers_set = {(rr, cc) for _color, rr, cc in markers}
    filtered = []
    for candidate in candidates:
        rr0, cc0, rr1, cc1 = candidate
        ok = True
        for rr in range(rr0, rr1 + 1):
            for cc in range(cc0, cc1 + 1):
                if grid[rr, cc] != bg or (rr, cc) in markers_set:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        # Keep a visible gap between seeds and blockers so both roles are legible.
        too_close = False
        for _color, mr, mc in markers:
            if rr0 - 1 <= mr <= rr1 + 1 and cc0 - 1 <= mc <= cc1 + 1:
                too_close = True
                break
        if too_close:
            continue
        filtered.append(candidate)

    if not filtered:
        return False

    filtered.sort(key=lambda cand: _candidate_score(cand, rect))
    if central:
        top = filtered[: max(1, min(8, len(filtered)))]
        candidate = rng.choice(top)
    else:
        candidate = rng.choice(filtered[: max(1, min(20, len(filtered)))])

    rr0, cc0, rr1, cc1 = candidate
    grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = blocker
    return True


def _stamp_blockers(grid, rects, bg, blocker, markers_by_rect, focus_idx, rng):
    placed = 0
    active_rects: set[int] = set()

    focus_rect = rects[focus_idx]
    if not _stamp_blocker_in_rect(
        grid,
        focus_rect,
        bg,
        blocker,
        markers_by_rect[focus_rect],
        rng,
        central=True,
    ):
        return None
    placed += 1
    active_rects.add(focus_idx)

    extra_candidates = [
        idx for idx, rect in enumerate(rects)
        if idx != focus_idx and (rect[2] - rect[0] + 1) >= 5 and (rect[3] - rect[1] + 1) >= 5
    ]
    rng.shuffle(extra_candidates)
    target = rng.randint(MIN_BLOCKER_REGIONS, min(MAX_BLOCKER_REGIONS, 1 + len(extra_candidates)))

    for idx in extra_candidates:
        if placed >= target:
            break
        rect = rects[idx]
        if _stamp_blocker_in_rect(
            grid,
            rect,
            bg,
            blocker,
            markers_by_rect[rect],
            rng,
            central=False,
        ):
            placed += 1
            active_rects.add(idx)

    return {"placed": placed, "rect_ids": active_rects}


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _competitive_region_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    free = g != wall

    count = 0
    h, w = g.shape
    for pts_arr in _cc4(free):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts_arr:
            comp_mask[int(r), int(c)] = True
        _base, seeds, _blocker_mask = _classify_region_colors(g, comp_mask)
        if len(seeds) >= 2:
            count += 1
    return count


def _variant_metrics(grid: np.ndarray):
    combo_out = main(grid)
    plain_bfs_out = _solve_bfs(grid, collision_stop=False)
    palette_stop_out = _solve_palette_collision(grid)

    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    free = g != wall
    h, w = g.shape

    blocker_cells = 0
    active_blocker_regions = 0
    competitive_regions = 0
    competitive_blocker_regions = 0
    tri_seed_blocker_regions = 0
    for pts_arr in _cc4(free):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts_arr:
            comp_mask[int(r), int(c)] = True
        _base, seeds, blocker_mask = _classify_region_colors(g, comp_mask)
        blocker_cells += int(blocker_mask.sum())
        if len(seeds) >= 2:
            competitive_regions += 1
        if blocker_mask.any():
            region_diff = 0
            for r, c in pts_arr:
                r = int(r)
                c = int(c)
                if int(combo_out[r, c]) != int(plain_bfs_out[r, c]):
                    region_diff += 1
            if region_diff > 0:
                active_blocker_regions += 1
                if len(seeds) >= 2:
                    competitive_blocker_regions += 1
                if len(seeds) >= 3:
                    tri_seed_blocker_regions += 1

    return {
        "combo_changed": int(np.count_nonzero(combo_out != g)),
        "blocker_cells": int(blocker_cells),
        "competitive_regions": int(competitive_regions),
        "active_blocker_regions": int(active_blocker_regions),
        "competitive_blocker_regions": int(competitive_blocker_regions),
        "tri_seed_blocker_regions": int(tri_seed_blocker_regions),
        "plain_bfs_diff": int(np.count_nonzero(combo_out != plain_bfs_out)),
        "palette_stop_diff": int(np.count_nonzero(combo_out != palette_stop_out)),
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, blocker, marker_pool = _pick_colors(rng)
    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, markers_by_rect, focus_idx = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None

    blocker_info = _stamp_blockers(grid, rects, bg, blocker, markers_by_rect, focus_idx, rng)
    if blocker_info is None or blocker_info["placed"] < MIN_BLOCKER_REGIONS:
        return None
    if not _validate_color_order(grid, bg, wall):
        return None
    if _competitive_region_count(grid) == 0:
        return None

    focus_rect = rects[focus_idx]
    focus_markers = markers_by_rect[focus_rect]
    if len(focus_markers) < 3:
        return None

    try:
        out = main(grid)
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if metrics["blocker_cells"] < 4:
        return None
    if metrics["competitive_regions"] < 1:
        return None
    if metrics["active_blocker_regions"] < 1:
        return None
    if metrics["competitive_blocker_regions"] < 1:
        return None
    if metrics["tri_seed_blocker_regions"] < 1:
        return None
    if metrics["plain_bfs_diff"] < 8:
        return None
    if metrics["palette_stop_diff"] < 12:
        return None
    if metrics["combo_changed"] < 14:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
