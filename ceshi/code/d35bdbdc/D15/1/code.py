# Variant of d35bdbdc
# Dimension: D15 — 符号编码与图例机制 (symbol_instructions)
# Change: 无符号编码（颜色匹配规则由示例隐含给出）... → 路径上的颜色提示像素
#
# Design analysis
# reference_baseline:
#   source_task: 3e6067c3
#   borrowed_idea:
#     - 参考题把“控制信息”做成可见的彩色像素序列，颜色本身不是被变换对象，而是指令。
#     - 本变体沿用这种复杂度：不引入额外文本/图例区，而把提示像素嵌到原题已有的灰色路径里。
# a_concretization_and_trace:
#   input_look:
#     - 仍然有 4-5 个 3x3 bordered sprite，灰色细路径连接其中两个目标精灵。
#     - 路径上额外嵌入 2 个非灰色单像素提示点，分别靠近路径两端，但不占用端点本身。
#     - 每个提示点颜色 = 某个源精灵的 border color。
#   output_look:
#     - 两个与路径端点相连的目标精灵被保留。
#     - 它们中心色改成提示点所指定源精灵的中心色。
#     - 其他精灵被删除；路径和提示点都保留。
#   behavior_trace:
#     - 原题：目标中心色 4 -> 查 border=4 的源 -> 把该源中心色写入目标。
#     - 新题：路径端附近提示点颜色 4 -> 查 border=4 的源 -> 把该源中心色写入该端目标。
#     - 差异：lookup key 从“目标自身中心”移到“路径上的可见符号”。
# b_complexity_calibration:
#   choices:
#     - 每题固定 2 个提示点，各自服务一个路径端，避免自由度过高。
#     - 仍允许 1 或 2 个源；若只有 1 个源，则两个提示点同色，形成重复支撑。
#   anti_trivialization:
#     - 目标输入中心色明确避开所有源 border 色，防止旧规则仍然碰巧成立。
#     - 提示点不放在端点，避免它们只是“换色端点”而不是独立可读的指令。
# c_implicit_assumption_audit_on_original_main:
#   affected_lines:
#     - 原 `_find_trail_endpoints` 假设路径只有灰色；现在必须把彩色提示点也纳入 trail。
#     - 原 `_detect_bordered_sprites` 用 `(g != bg) & (g != grey)` 找组件；现在会看到提示点，
#       所以生成器必须保证提示点是与 sprite 4-分离的单像素组件，solver 再把它们忽略为非-sprite。
#     - 原删除逻辑会删除所有非 bg/grey 且非目标连通块；现在若不特殊保留，提示点会被误删。
#     - 原颜色转移以 `dest.center` 为 key；现在必须改为“从路径端朝中间读到的第一个提示色”。
# d_placement_legality:
#   guarantees_needed:
#     - 提示点必须位于路径内部单元，且与任意 sprite 非路径像素不 4-接触。
#     - 路径端点仍是合法 port cell：距目标中心曼哈顿距离 2，夹着目标 border。
#     - 放置源/干扰精灵时也必须避开提示点，否则 boolean component 会把它们并到一起。
# e_visual_observability:
#   why_human_can_see_it:
#     - 输入里目标中心色不再暗示来源；真正的线索改成路径上的彩色“路标”。
#     - 两端各一个提示点，位置上天然和对应目标绑定，2-3 个样例就能看出“看路径色，不看目标中心”。
# f_visual_cleanliness:
#   cleanup:
#     - 提示点数量固定为 2，不在同一区域叠太多标记。
#     - 仍沿用单条细路径和小 sprite，画面结构与原题一致，只增加最低限度的说明符号。
# g_color_inference:
#   constraints:
#     - 提示色直接复用源 sprite 的 border 色，映射关系是“同色引用”，不是隐式重编码。
#     - 目标原始中心色避开这些 border 色，减少错误归纳。
# h_generator_oracle:
#   approach:
#     - 继续 output-first：先建好最终保留的两个目标与带提示的路径，再反推输入的目标中心色并添加源/干扰。
#     - 每个样本都用新 `main()` 回放校验。
# i_new_main_assumptions_to_recheck:
#   assumptions:
#     - 提示点是单像素且互不相邻。
#     - 从每个端点向路径内部至少能读到一个提示点。
#     - 每个提示色最多对应一个源 border；生成器通过 border 全异保证这一点。

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_path_cells(
    g: np.ndarray, bg: int, grey: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Path = grey cells plus isolated colored hint pixels embedded in the trail.
    Hint pixels are single-cell non-bg/non-grey components with at least 2 grey
    neighbors under 8-connectivity.
    """
    h, w = g.shape
    grey_cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    mask = (g != bg) & (g != grey)
    hint_cells: set[tuple[int, int]] = set()
    for pts in components_mask(mask, connectivity=4):
        if len(pts) != 1:
            continue
        r, c = map(int, pts[0])
        grey_nbs = 0
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == grey:
                grey_nbs += 1
        if grey_nbs >= 2:
            hint_cells.add((r, c))

    path_cells = grey_cells | hint_cells
    assert path_cells, "expected a path component"
    return path_cells, hint_cells


def _trace_trail(
    g: np.ndarray, path_cells: set[tuple[int, int]], sprites: list[dict],
) -> list[tuple[int, int]]:
    """Trace the mixed grey+hint trail as one ordered endpoint-to-endpoint path."""
    h, w = g.shape
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in path_cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in path_cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in path_cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == b:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(path_cells))
    goal_cells = {deg1[1]} if len(deg1) >= 2 else port_cells
    n = len(path_cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == n:
            return [cur] if cur in goal_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                result = dfs(nb, visited | {nb})
                if result is not None:
                    return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace mixed trail"
    return path


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


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A trail endpoint (er,ec) is a port for this sprite if:
      - it's exactly 2 steps away from the sprite center (Manhattan distance 2),
      - the intermediate cell (1 step away from center toward endpoint) is the sprite border color.
    """
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def _first_hint_color(path: list[tuple[int, int]], g: np.ndarray, grey: int) -> int | None:
    """Read the first non-grey trail cell encountered along the ordered path."""
    for r, c in path:
        col = int(g[r, c])
        if col != grey:
            return col
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer source center colors into the two path-connected targets using color hints embedded in the trail.

    Input structure:
    - Background, a thin trail whose main body is grey (5), and several small bordered sprites with one distinct center pixel.
    - The trail still connects exactly two destination sprites by endpoint ports, but now the trail also contains a few colored hint pixels inside it.

    Rule:
    - Find the two destination sprites touched by the trail endpoints, same as in the base task.
    - For each endpoint, walk inward along the trail and read the first colored hint pixel; that hint color names the border color of the source sprite to consult.
    - Copy that source sprite's center color into the connected destination's center, then delete all non-destination sprites while preserving the trail and its hints.

    Key insight: The lookup key is no longer hidden in the destination center; it is explicitly written on the path, so the path serves as both connector and legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    path_cells, _hint_cells = _find_path_cells(g, bg, grey)
    ordered_path = _trace_trail(g, path_cells, sprites)
    te1, te2 = ordered_path[0], ordered_path[-1]
    trail_eps = {te1, te2}

    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    # Bipartite match: fewer-options-first; tiebreak by sprite farthest from OTHER endpoint
    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[te1]) <= len(opts[te2]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = eps_pair[i]
        other = eps_pair[1 - i]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 dest sprites from trail endpoints"
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
    path_mask = np.zeros(out.shape, dtype=bool)
    for r, c in path_cells:
        path_mask[r, c] = True
    drop = (out != bg) & (~path_mask) & (~keep)
    out[drop] = bg

    hint_borders = [
        _first_hint_color(ordered_path[1:-1], g, grey),
        _first_hint_color(list(reversed(ordered_path))[1:-1], g, grey),
    ]
    assert all(h is not None for h in hint_borders), "expected a readable hint from each trail endpoint"

    for d, want_border in zip(dest, hint_borders, strict=False):
        srcs = by_border.get(int(want_border), [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each hint color should match at most one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_H, GRID_W = 10, 10                     # canvas size
BG, GREY = 0, 5                             # background and trail body colors
SPRITE_COUNTS = [4, 5]                      # number of sprites per example
SOURCE_COUNTS = [1, 2]                      # possible number of source sprites
INTERIOR_DEST_WEIGHTS = [3, 1]              # weights for [0, 1] interior dests
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24         # trail length bounds
MIN_PORT_DIST_FRAC = 3                      # min_port_dist = max(3, target_len // this)
MAX_THICKNESS = 3                           # max 8-neighbor degree for path cells
PATH_WANDER_AWAY_P = 0.5                    # P(choose away-moves) when budget is large
PATH_WANDER_DIAG_P = 0.6                    # P(prefer diagonal) in wander/moderate phases
PATH_MODERATE_RANDOM_P = 0.3                # P(random move) in moderate-budget phase
GEN_RETRIES = 300                           # outer retry limit


def _make_sprite_patch(shape_type, border_color, center_color):
    """Return 3x3 patch for a box or plus. Center is at (1,1)."""
    if shape_type == "box":
        p = np.full((3, 3), border_color, dtype=int)
    else:  # plus
        p = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p[1 + dr, 1 + dc] = border_color
    p[1, 1] = center_color
    return p


def _place_sprite(grid, patch, r0, c0):
    """Place patch cells on grid, skipping transparent (0) and out-of-bounds."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid, patch, r0, c0):
    """Undo a _place_sprite by zeroing non-transparent patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != 0:
                grid[rr, cc] = BG


def _overlaps(grid, patch, r0, c0):
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != 0:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if grid[rr, cc] != 0:
                        return True
                    patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions():
    """All valid (r0,c0) for a 3x3 patch where center stays on grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0, c0):
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r, c, occupied):
    """Check if adding (r,c) would push any cell's 8-degree above MAX_THICKNESS."""
    nbs = [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and (r + dr, c + dc) in occupied
    ]
    if len(nbs) > MAX_THICKNESS:
        return True
    for nr, nc in nbs:
        deg = sum(
            1
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (nr + dr, nc + dc) in occupied
        )
        if deg + 1 > MAX_THICKNESS:
            return True
    return False


def _gen_winding_path(grid, start, end, target_len):
    """Generate a winding 8-connected grey path."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(m):
        return abs(m[0] - end[0]) + abs(m[1] - end[1])

    for _ in range(target_len + 20):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if (
                    0 <= nr < GRID_H
                    and 0 <= nc < GRID_W
                    and (nr, nc) not in visited
                    and grid[nr, nc] == 0
                    and not _too_thick(nr, nc, visited)
                ):
                    moves.append((nr, nc, dr != 0 and dc != 0))

        if not moves:
            return None

        dist = abs(cr - end[0]) + abs(cc - end[1])
        budget = target_len - len(path)
        diag = [m for m in moves if m[2]]
        away = [m for m in moves if score(m) > dist]

        if budget > dist + 4:
            if away and random.random() < PATH_WANDER_AWAY_P:
                pool = away
            elif diag and random.random() < PATH_WANDER_DIAG_P:
                pool = diag
            else:
                pool = moves
            chosen = random.choice(pool)
        elif budget > dist + 1:
            pool = diag if diag and random.random() < PATH_WANDER_DIAG_P else moves
            if random.random() < PATH_MODERATE_RANDOM_P:
                chosen = random.choice(pool)
            else:
                chosen = min(pool, key=score)
        else:
            best = sorted(moves, key=score)
            top = [m for m in best if score(m) <= score(best[0]) + 1]
            dtop = [m for m in top if m[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if not _too_thick(end[0], end[1], visited):
            path.append(end)
        else:
            return None

    return path


def _is_8_connected(cells):
    """Check 8-connectivity of a set of cells via BFS."""
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
    return len(visited) == len(cell_set)


def _find_port_dir(grid, cr, cc, bc):
    """Find a valid port direction for a sprite centered at (cr,cc) with border color bc."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == 0:
            return (er, ec)
    return None


def _choose_color(exclude: set[int], preferred_unused: set[int]) -> int | None:
    palette = [c for c in range(1, 10) if c != GREY and c not in exclude]
    if not palette:
        return None
    fresh = [c for c in palette if c not in preferred_unused]
    return random.choice(fresh if fresh else palette)


def _assign_colors(border_colors, num_sprites):
    """Assign roles and pick source/hint/destination colors for the path-instruction variant."""
    remaining = list(range(2, num_sprites))
    random.shuffle(remaining)
    all_borders = set(border_colors)
    num_sources = random.choice(SOURCE_COUNTS) if len(remaining) >= 2 else max(1, len(remaining))
    if num_sources == 0:
        return None
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    used_nonborder: set[int] = set()
    source_center_colors = []
    for si in source_indices:
        forbidden = {BG, GREY, border_colors[si], border_colors[0], border_colors[1]}
        col = _choose_color(forbidden, used_nonborder)
        if col is None:
            return None
        source_center_colors.append(col)
        used_nonborder.add(col)

    if num_sources == 1:
        hint_border_colors = [border_colors[source_indices[0]], border_colors[source_indices[0]]]
        output_dest_centers = [source_center_colors[0], source_center_colors[0]]
    else:
        hint_border_colors = [border_colors[source_indices[0]], border_colors[source_indices[1]]]
        output_dest_centers = source_center_colors[:2]

    dest_center_colors = []
    for i in range(2):
        forbidden = {
            BG,
            GREY,
            output_dest_centers[i],
            *hint_border_colors,
            *all_borders,
        }
        col = _choose_color(forbidden, used_nonborder)
        if col is None:
            return None
        dest_center_colors.append(col)
        used_nonborder.add(col)

    distractor_center_colors = []
    for di in distractor_indices:
        forbidden = {BG, GREY, border_colors[di], *hint_border_colors}
        col = _choose_color(forbidden, set())
        if col is None:
            return None
        distractor_center_colors.append(col)

    return dict(
        source_indices=source_indices,
        distractor_indices=distractor_indices,
        hint_border_colors=hint_border_colors,
        source_center_colors=source_center_colors,
        dest_center_colors=dest_center_colors,
        distractor_center_colors=distractor_center_colors,
        output_dest_centers=output_dest_centers,
    )


def _place_dests(output_grid, shape_type, border_colors, output_dest_centers, min_port_dist):
    """Place 2 destination sprites on the grid. Returns center positions or None."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions = []
    first_port = None

    for i in range(2):
        bc = border_colors[i]
        patch = _make_sprite_patch(shape_type, bc, output_dest_centers[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc_pos = r0 + 1, c0 + 1

            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc_pos, bc)
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue

            if i == 1:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue

            dest_center_positions.append((cr, cc_pos))
            if i == 0:
                first_port = port
            placed = True
            break

        if not placed:
            return None

    return dest_center_positions


def _hint_legal(cell, grid, path_set):
    """Hint cells must stay visually separate from sprite pixels except through the trail."""
    r, c = cell
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
            if (nr, nc) in path_set:
                continue
            if int(grid[nr, nc]) != BG:
                return False
    return True


def _choose_hint_indices(path_cells, grid):
    """Pick one safe interior path cell near each endpoint for the two instruction pixels."""
    n = len(path_cells)
    if n < 7:
        return None
    path_set = set(path_cells)
    left_choices = [
        i
        for i in range(2, min(5, n - 2))
        if _hint_legal(path_cells[i], grid, path_set)
    ]
    if not left_choices:
        return None
    left_idx = random.choice(left_choices)

    right_choices = [
        i
        for i in range(max(n - 5, left_idx + 3), n - 2)
        if _hint_legal(path_cells[i], grid, path_set)
    ]
    if not right_choices:
        return None
    right_idx = random.choice(right_choices)
    return left_idx, right_idx


def _gen_attempt():
    shape_type = random.choice(["box", "plus"])
    palette = [c for c in range(1, 10) if c != GREY]
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(palette, num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(output_grid, shape_type, border_colors, colors["output_dest_centers"], min_port_dist)
    if result is None:
        return None
    dest_center_positions = result

    port_cells = []
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        port = _find_port_dir(output_grid, cr, cc_pos, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(output_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    hint_indices = _choose_hint_indices(path_cells, output_grid)
    if hint_indices is None:
        return None
    output_grid[path_cells[hint_indices[0]]] = colors["hint_border_colors"][0]
    output_grid[path_cells[hint_indices[1]]] = colors["hint_border_colors"][1]

    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        input_grid[cr, cc_pos] = colors["dest_center_colors"][i]

    sprites_to_place = (
        [(border_colors[si], colors["source_center_colors"][j]) for j, si in enumerate(colors["source_indices"])]
        + [(border_colors[di], colors["distractor_center_colors"][j]) for j, di in enumerate(colors["distractor_indices"])]
    )
    all_pos = _sprite_positions()
    for bc, cc_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, bc, cc_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if not _overlaps(input_grid, patch, r0, c0):
                _place_sprite(input_grid, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(input_grid)
        if np.array_equal(out, output_grid):
            return input_grid
    except Exception:
        pass
    return None


def generate_input():
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
