# Variant of d35bdbdc
# Dimension: D9 — 输出网格尺寸 (grid_shape)
# Change: 输出网格与输入尺寸相同... → 仅输出目标精灵区域
#
# Design analysis
# ---
# source_task_baseline:
#   reference_solver: a251c730
#   relevant_mechanism:
#     - 参考任务的核心 D9 动作是“先按原逻辑找出真正保留的目标区域，再只返回那个区域自己的局部画布”。
#     - 这里不需要像 a251c730 那样重建一个新矩形框，只需要在原题完成目标识别与颜色转移后，
#       把其余内容删除，再对两个目标精灵的可见像素做一次最小外接矩形裁剪即可。
# concrete_behavior:
#   input:
#     - 仍然是 10x10 全尺寸网格。
#     - 含一条灰色(5)单笔路径，路径两端通过 port 接到两个目标精灵。
#     - 还有 1-2 个 source 精灵和 0-2 个 distractor 精灵；它们只在输入中出现。
#   output:
#     - 不再保留整张 10x10。
#     - 只保留两个目标精灵在最终状态下的联合最小外接矩形。
#     - 灰路径被完全删除；非目标精灵也被删除。
#     - 两个目标精灵之间如果存在空白列/行，会作为背景保留在裁剪后的局部画布中。
# behavior_trace:
#   original_flow:
#     - 找到两端点对应的目标精灵。
#     - 用目标中心色去查 source.border，并把 source.center 写回目标中心。
#     - 删除非目标，但整张 10x10 和灰路径仍保留。
#   variant_flow:
#     - 前两步完全相同。
#     - 之后把灰路径也抹成背景。
#     - 只对剩余目标精灵像素求 bbox，并返回该 bbox 子数组。
#   concrete_example:
#     - 若两个目标分别位于原网格的左上和中右部，原题输出仍是 10x10，能看到灰路径。
#     - 变体输出只截取这两个精灵覆盖到的 6x8 或 7x9 局部区域，路径消失。
# complexity_calibration:
#   - 不能把两个目标长期固定成相邻位置，否则裁剪几乎总是 3x6/3x5，规则过于单薄。
#   - 也不能让 bbox 经常退化回 10x10，否则 D9 几乎不可见。
#   - 因此 generator 只额外约束“目标联合 bbox 不能等于整张输入尺寸”，其余放置逻辑保持原题风格。
# implicit_assumption_audit:
#   - _detect_bordered_sprites / _find_trail_endpoints / 端点到目标的匹配逻辑都只依赖输入结构，不依赖输出尺寸，保持不变。
#   - 原 main 中 `out = g.copy()` 之后默认返回整张图；这正是 D9 要改的地方。
#   - 原逻辑允许灰路径保留；变体中如果直接 crop `out != bg` 会把灰路径一起带进 bbox，所以必须先把灰色改回背景。
#   - 原逻辑删除非目标后，仍可能残留与目标 4 连通的非灰杂质；keep/drop 清理流程仍有必要，不能删。
# placement_legality:
#   - 输入中的所有 sprite 仍使用原放置约束：不重叠、不 4-touch、端口合法、路径可生成。
#   - 新增合法性约束只有一个：最终两个目标精灵的联合 bbox 不能覆盖整张 10x10。
# visual_observability:
#   - 解题者仍能从输入看出“灰路径只指向两个目标，其他精灵消失，中心色发生查表转移”。
#   - 与原题相比，最显眼的新现象是输出尺寸收缩到目标区域，且输出中完全没有灰路径。
# neatness:
#   - 输出只含两个小精灵及其之间必要的背景，视觉上比原题更聚焦。
# color_inference:
#   - 颜色规则未新增任何不可见排序；仍只使用“目标中心色 == source 边框色”这个可见链条。

import random
from collections import deque

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the 2 endpoints of the grey one-stroke trail under 8-connectivity."""
    h, w = g.shape
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [c for c, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == b:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    n = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset) -> list | None:
        if len(visited) == n:
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                result = dfs(nb, visited | {nb})
                if result is not None:
                    return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace grey cells as one-stroke trail"
    return path[0], path[-1]


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
    A grey endpoint (er,ec) is a port for this sprite if:
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


def _crop_target_region(g: np.ndarray, bg: int, grey: int) -> np.ndarray:
    """Remove grey path, then crop to the bbox of the remaining non-bg target pixels."""
    cropped = g.copy()
    cropped[cropped == grey] = bg
    coords = np.argwhere(cropped != bg)
    assert coords.size > 0, "expected target pixels to remain after removing grey path"
    r0, c0, r1, c1 = bbox_coords(coords)
    return cropped[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors between trail-selected bordered sprites, then crop to them.

    Input structure:
    - A full-size canvas contains several bordered sprites plus a grey (5) one-stroke trail.
    - The trail's two endpoints connect, via sprite-colored port cells, to exactly two destination sprites.

    Rule:
    - Use the two trail endpoints to identify the destination sprites.
    - For each destination, look up the sprite whose border color matches that destination's original center color, and copy that source center color into the destination center.
    - Delete every non-destination sprite, erase the grey trail, then return only the minimal bounding rectangle covering the remaining destination sprites.

    Key insight:
    - The path still tells you which sprites matter, and each destination center still acts as a color key into another sprite's border; the new twist is that the final answer is the cropped destination region, not the original whole grid.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}

    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

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
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return _crop_target_region(out, bg, grey)


# --------------- generator ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 300


def _make_sprite_patch(shape_type, border_color, center_color):
    """Return 3x3 patch for a box or plus. Center is at (1,1)."""
    if shape_type == "box":
        p = np.full((3, 3), border_color, dtype=int)
    else:
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
    """Generate a winding 8-connected grey path with no 2x2 blocks."""
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


def _assign_colors(border_colors, num_sprites):
    """Assign roles (dest/source/distractor) and pick center colors for all sprites."""
    remaining = list(range(2, num_sprites))
    num_sources = random.choice(SOURCE_COUNTS) if len(remaining) >= 2 else max(1, len(remaining))
    if num_sources == 0:
        return None
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    if num_sources == 1:
        dest_center_colors = [border_colors[source_indices[0]]] * 2
    else:
        dest_center_colors = [border_colors[si] for si in source_indices]

    source_center_colors = []
    for j, si in enumerate(source_indices):
        forbidden = {BG, GREY, border_colors[si]}
        if num_sources == 1:
            forbidden.update([border_colors[0], border_colors[1]])
        else:
            forbidden.add(border_colors[j])
        candidates = [c for c in range(1, 10) if c not in forbidden]
        if not candidates:
            return None
        source_center_colors.append(random.choice(candidates))

    distractor_center_colors = [
        random.choice([c for c in range(1, 10) if c != GREY and c != border_colors[di]])
        for di in distractor_indices
    ]

    output_dest_centers = (source_center_colors * 2)[:2] if num_sources == 1 else source_center_colors

    return dict(
        source_indices=source_indices,
        distractor_indices=distractor_indices,
        dest_center_colors=dest_center_colors,
        source_center_colors=source_center_colors,
        distractor_center_colors=distractor_center_colors,
        output_dest_centers=output_dest_centers,
    )


def _crop_is_observable(grid: np.ndarray) -> bool:
    """D9 should shrink at least one dimension; reject full-size target bboxes."""
    cropped = _crop_target_region(grid, BG, GREY)
    return cropped.shape != (GRID_H, GRID_W)


def _place_dests(output_grid, shape_type, border_colors, output_dest_centers, min_port_dist):
    """Place 2 destination sprites on the grid. Returns center positions or None."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions = []
    placed_info: list[tuple[np.ndarray, int, int]] = []
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
                if dist < min_port_dist or not _crop_is_observable(output_grid):
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue

            dest_center_positions.append((cr, cc_pos))
            placed_info.append((patch, r0, c0))
            if i == 0:
                first_port = port
            placed = True
            break

        if not placed:
            for patch_prev, r0_prev, c0_prev in reversed(placed_info):
                _erase_sprite(output_grid, patch_prev, r0_prev, c0_prev)
            return None

    return dest_center_positions


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

    solved_full = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(solved_full, shape_type, border_colors, colors["output_dest_centers"], min_port_dist)
    if result is None:
        return None
    dest_center_positions = result

    port_cells = []
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        port = _find_port_dir(solved_full, cr, cc_pos, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(solved_full, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if solved_full[r, c] != BG:
            return None
        solved_full[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and solved_full[nr, nc] == GREY
        )
        if deg > 1:
            return None

    expected = _crop_target_region(solved_full, BG, GREY)
    if expected.shape == (GRID_H, GRID_W):
        return None

    input_grid = solved_full.copy()
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        input_grid[cr, cc_pos] = colors["dest_center_colors"][i]

    sprites_to_place = (
        [(border_colors[si], colors["source_center_colors"][j]) for j, si in enumerate(colors["source_indices"])]
        + [
            (border_colors[di], colors["distractor_center_colors"][j])
            for j, di in enumerate(colors["distractor_indices"])
        ]
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
        if np.array_equal(out, expected):
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
