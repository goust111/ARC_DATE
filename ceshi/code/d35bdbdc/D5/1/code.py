# Variant of d35bdbdc
# Dimension: D5 — 属性转移方向 (transform)
# Change: s.center_color -> d.center_color（将源的中心色复制到目标的中心位置）... → 双向属性交换

# Design analysis
# spec_variant:
#   a_concretization:
#     - 输入仍是 10x10 网格，包含 2 个被灰色单笔路径端点接触的 destination 精灵，
#       以及 2 个独立的 source 精灵，外加 0-1 个 distractor。
#     - 每个精灵仍是同形的 3x3 box 或 plus，border_color 唯一，center_color 单像素。
#     - destination 的中心色仍然不是“最终颜色”，而是一个 lookup key：
#       它等于某个 source 的 border_color。
#     - 原题在这里执行的是：
#       dest.center <- source.center，然后 source 整体被删除。
#     - 变体要求的是：
#       同一对 (dest, source) 做同步交换，既有
#       dest.center <- source.center，也有 source.center <- dest.center。
#       由于 dest.center 本来就等于 source.border，交换后 source 会变成纯边框色实心精灵。
#     - 这一步如果仍删除 source，则交换无法在输出中观察到，所以输出必须保留参与交换的 source，
#       只删除 distractor 和额外杂质组件。这是为 D5 可观察性做的最小适配。
#   b_complexity_calibration:
#     - 原题允许 1 个 source 服务 2 个 destination；对“复制”成立，但对“交换”不成立，
#       因为一个 source 的中心色不能同时与两个不同目标交换。
#     - 因此生成器固定为 2 个 source 且一对一配对；这保留原题的颜色链难点，
#       同时避免多对一歧义。
#     - 为了让交换肉眼可见，source.center 取自“非任何 border 的颜色池”，
#       所以 destination 输出中心一定改变，source 输出也一定从异色中心变为纯 border。
#   c_hidden_assumptions_review:
#     - 原 main() 的 by_border 查找仍可复用，因为颜色链定义没变。
#     - 原 main() 先删除全部非 destination，再写 destination 中心；
#       这依赖“source 输出不需要存在”。变体下这会直接抹掉交换另一端，因此必须改成：
#       先确定 swap pair，再只删除未参与 pair 的 sprite。
#     - 原 main() 逐个改 destination 不会互相覆盖；变体下若直接原地写 source/dest，
#       会引入顺序依赖。所以要先用输入态记录 pair，再按“同步赋值”写回输出。
#     - 交换语义还额外要求 source 不得被两个 destination 复用；solver 显式断言这一点，
#       generator 通过一对一配对构造保证成立。
#   d_placement_legality:
#     - destination 的端口、灰色路径、边缘/内部放置规则保持原样，保证 solver 的 endpoint 选择逻辑不变。
#     - output-first 改成：先放 2 个 destination 和灰路径，再放 2 个交换后的 source；
#       这样路径有最大的可用空间，成功率更高。
#     - input 由 output 回退而来：把 destination/source 的中心恢复成交换前颜色，再额外放 distractor。
#     - 所有精灵仍通过 _overlaps() 防止重叠和 4 邻接粘连，保证 solver 的组件检测稳定。
#   e_visual_observability:
#     - 人类可见线索是“灰路径只指定哪两个是 destination；destination 中心颜色指向远处 source 的 border；
#       输出里 destination 得到 source 的中心色，而远处 source 的中心同时消失，变成纯色”。
#     - 只用可见的颜色和空间关系，不用颜色 id 排序或任何不可见元数据。
#   f_cleanliness:
#     - 仍限制总 sprite 数 4-5，且 distractor 最多 1 个，避免 10x10 画面过满。
#     - source 输出会变成单色 box/plus，但仍是规则形状，和灰路径/背景区分清楚。
#   g_color_inference:
#     - border 颜色全异；2 个 destination 的输入中心色分别等于 2 个 source 的 border。
#     - 2 个 source 的输入中心色互异且不属于任何 border 颜色，因此 swap 后的变化在 2-3 个样例中可归纳。
#   i_new_solver_assumptions:
#     - solver 假设每个 destination 最多匹配一个 source，且参与交换的 source 彼此不同。
#     - generator 保证 destination 和 matched source 是分离精灵、颜色链一对一、且每对交换都会真正改色。

import numpy as np

from arc_utils import neighbors4, neighbors8, components_mask, paste
import random
from collections import deque


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


def _select_destinations(
    sprites: list[dict], g: np.ndarray, trail_eps: set[tuple[int, int]], eps_pair: list[tuple[int, int]],
) -> list[dict]:
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[eps_pair[0]]) <= len(opts[eps_pair[1]]) else [1, 0]
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
    return dest


def _match_swap_pairs(sprites: list[dict], dest: list[dict]) -> list[tuple[dict, dict]]:
    """Pair each destination with the unique non-destination source named by its center color."""
    dest_ids = {id(sp) for sp in dest}
    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        by_border.setdefault(int(sp["border"]), []).append(sp)

    pairs: list[tuple[dict, dict]] = []
    used_sources: set[int] = set()
    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match exactly one non-destination source border"
        src = srcs[0]
        sid = id(src)
        assert sid not in used_sources, "swap variant requires one-to-one destination/source matches"
        used_sources.add(sid)
        pairs.append((d, src))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap center colors between trail-selected destinations and their matched lookup sources.

    Input structure:
    - The grid contains background, one grey (5) one-stroke trail, and several small bordered sprites with a unique center pixel.
    - The trail endpoints identify exactly two destination sprites through 2-step ports; other sprites are detached lookup sources or distractors.

    Rule:
    - Read each destination's original center color as a key for a non-destination source whose border color matches it.
    - Keep the two destinations, keep the matched sources, and simultaneously exchange each paired destination/source center color.
    - Delete every other sprite and any leftover non-sprite foreground component; preserve the grey trail.

    Key insight:
    - A destination center is still a pointer into border colors elsewhere, but the action is no longer one-way filling.
    - Because the matched source also survives, its contrasting center disappears into its own border color after the swap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    dest = _select_destinations(sprites, g, trail_eps, eps_pair)
    pairs = _match_swap_pairs(sprites, dest)

    keep_ids = {id(sp) for sp in dest}
    for _d, src in pairs:
        keep_ids.add(id(src))

    out = g.copy()

    for sp in sprites:
        if id(sp) in keep_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for sp in sprites:
        if id(sp) not in keep_ids:
            continue
        for r, c in sp["pts"]:
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

    updates: dict[tuple[int, int], int] = {}
    for d, src in pairs:
        dr, dc = d["center_pos"]
        sr, sc = src["center_pos"]
        updates[(dr, dc)] = int(src["center"])
        updates[(sr, sc)] = int(d["center"])
    for (r, c), color in updates.items():
        out[r, c] = color

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
SOURCE_COUNT = 2
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400


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
    """Assign destination/source/distractor roles and all pre/post-swap center colors."""
    if num_sprites < 2 + SOURCE_COUNT:
        return None

    source_indices = list(range(2, 2 + SOURCE_COUNT))
    distractor_indices = list(range(2 + SOURCE_COUNT, num_sprites))
    pair_source_indices = list(source_indices)
    random.shuffle(pair_source_indices)

    dest_center_colors = [border_colors[si] for si in pair_source_indices]

    source_center_pool = [c for c in range(1, 10) if c not in {BG, GREY} and c not in border_colors]
    if len(source_center_pool) < SOURCE_COUNT:
        return None
    random.shuffle(source_center_pool)

    source_center_by_index: dict[int, int] = {}
    for j, si in enumerate(source_indices):
        source_center_by_index[si] = source_center_pool[j]

    distractor_center_colors = []
    for di in distractor_indices:
        candidates = [c for c in range(1, 10) if c not in {BG, GREY, border_colors[di]}]
        distractor_center_colors.append(random.choice(candidates))

    output_dest_centers = [source_center_by_index[si] for si in pair_source_indices]
    output_source_centers = {si: border_colors[si] for si in source_indices}

    return dict(
        source_indices=source_indices,
        distractor_indices=distractor_indices,
        pair_source_indices=pair_source_indices,
        dest_center_colors=dest_center_colors,
        source_center_by_index=source_center_by_index,
        distractor_center_colors=distractor_center_colors,
        output_dest_centers=output_dest_centers,
        output_source_centers=output_source_centers,
    )


def _place_dests(output_grid, shape_type, border_colors, output_dest_centers, min_port_dist):
    """Place 2 destination sprites on the grid. Returns destination center positions or None."""
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


def _place_free_sprite(grid, patch):
    """Place a sprite anywhere legal; return its center position or None."""
    pool = _sprite_positions()
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(grid, patch, r0, c0):
            continue
        _place_sprite(grid, patch, r0, c0)
        return (r0 + 1, c0 + 1)
    return None


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
    dest_center_positions = _place_dests(
        output_grid, shape_type, border_colors, colors["output_dest_centers"], min_port_dist
    )
    if dest_center_positions is None:
        return None

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

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    source_center_positions: dict[int, tuple[int, int]] = {}
    for si in colors["source_indices"]:
        patch = _make_sprite_patch(shape_type, border_colors[si], colors["output_source_centers"][si])
        pos = _place_free_sprite(output_grid, patch)
        if pos is None:
            return None
        source_center_positions[si] = pos

    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        input_grid[cr, cc_pos] = colors["dest_center_colors"][i]
    for si in colors["source_indices"]:
        sr, sc = source_center_positions[si]
        input_grid[sr, sc] = colors["source_center_by_index"][si]

    for j, di in enumerate(colors["distractor_indices"]):
        patch = _make_sprite_patch(shape_type, border_colors[di], colors["distractor_center_colors"][j])
        if _place_free_sprite(input_grid, patch) is None:
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
