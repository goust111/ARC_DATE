# Variant of d35bdbdc
# Dimension: D8 — 路径在输出中的处理 (trajectory)
# Change: 灰色路径在输出中保留（未被删除或修改）... → 路径重新着色为转移的颜色
#
# Design analysis:
# - concretization:
#   输入仍然是两个端点精灵 + 一条灰色单笔轨迹 + 若干源/干扰精灵。输出里仍只保留
#   端点精灵，但轨迹不再统一保留为 5；它沿路径顺序从两端向中间分别染成两个
#   “真正被写入目标中心”的颜色。例：若有序路径 p0..p8，左端目标收到 7，右端
#   目标收到 2，则原题会输出 [5,5,5,5,5,5,5,5,5]，本变体输出
#   [7,7,7,7,7,2,2,2,2]。
# - complexity calibration:
#   仍保留原题的 10x10、4-5 个精灵、1-2 个源的规模；路径只允许最多两种输出色，
#   不引入额外匹配自由度。为保证双色流动可见，generator 在双源情形下强制两个源中心
#   色不同，因此路径分段通常肉眼可见，但单源样例仍合法并保留。
# - implicit assumptions checked:
#   原 solver 的清理逻辑把 “grey” 当成免删通道；若先把路径改色，再跑 drop/keep，
#   路径会被误删，因为它与目标精灵并不 4 连通。所以必须先完成原有删减，再在最后
#   按有序路径重着色。另一个旧前提是 “只需端点，不需路径顺序”；D8 变体失效，
#   因为中点分色依赖整条轨迹的顺序。
# - placement legality:
#   端点仍通过 port 与路径相接：路径端点距目标中心曼哈顿 2，夹着一格边框色。
#   generator 先放目标、再生成路径、再从输出反推输入，确保路径不与精灵重叠，
#   且 solver 的端点识别与颜色匹配假设始终成立。
# - observability / cleanliness:
#   路径长度仍是 10-24，足以让“颜色流淌”明显可见；路径只用 1-2 种颜色，且这两种
#   颜色正好等于两个保留目标中心的最终颜色，解题者能从少量样例中看出路径颜色来自
#   端点传色，而不是任意映射。
# - new solver assumptions audited:
#   `main()` 现在额外假设灰路径能被恢复成一条覆盖全部灰格的有序单笔路径；generator
#   继续用单笔 winding trail 构造来保证这一点。它还假设每个目标的匹配源至多一个；
#   generator 继续保证 border_color 唯一。

import numpy as np

from arc_utils import neighbors4, neighbors8, components_mask, paste
import random
from collections import deque


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _trail_port_cells(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> set[tuple[int, int]]:
    """Grey cells that can serve as valid endpoint ports for any sprite."""
    h, w = g.shape
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    ports: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == b:
                ports.add((er, ec))
    return ports


def _trace_trail_path(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> list[tuple[int, int]]:
    """Recover the grey one-stroke trail as an ordered 8-connected path."""
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    assert cells, "expected at least one grey trail cell"
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    port_cells = _trail_port_cells(g, grey, sprites)
    start = deg1[0] if deg1 else next(iter(cells))
    fixed_end = deg1[1] if len(deg1) >= 2 else None
    n = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == n:
            if fixed_end is not None:
                return [cur] if cur == fixed_end else None
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb not in visited:
                result = dfs(nb, visited | {nb})
                if result is not None:
                    return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace grey cells as one-stroke trail"
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


def _paint_flow_trail(
    out: np.ndarray,
    path: list[tuple[int, int]],
    start_color: int | None,
    end_color: int | None,
    grey: int,
) -> None:
    """Recolor the ordered trail from both ends toward the middle."""
    n = len(path)
    for idx, (r, c) in enumerate(path):
        if idx <= n - 1 - idx:
            out[r, c] = grey if start_color is None else int(start_color)
        else:
            out[r, c] = grey if end_color is None else int(end_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors between bordered sprites matched by color, then let those colors flow through the trail.

    Input structure:
    - Background, a grey (5) one-stroke trail, and several small bordered sprites whose border color surrounds one distinct center pixel.
    - The trail touches exactly two destination sprites through valid ports at its two ends; other sprites serve as lookup sources or distractors.

    Rule:
    - Use the two trail endpoints to identify the two destination sprites.
    - For each destination, find the source sprite whose border color matches that destination's original center color, and copy the source center color into the destination center.
    - Remove all non-destination sprites and other leftover non-grey components, then repaint the trail so each half takes the transferred color of the destination it reaches.

    Key insight:
    - A destination center is first a lookup key into another sprite's border color, and the ordered trail then turns that hidden transfer into a visible color flow from both ends toward the middle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    trail_path = _trace_trail_path(g, grey, sprites)
    te1, te2 = trail_path[0], trail_path[-1]
    trail_eps = {te1, te2}

    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)]
            for ep in eps_pair}

    # Bipartite match: fewer-options-first; tiebreak by sprite farthest from OTHER endpoint
    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[te1]) <= len(opts[te2]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = eps_pair[i]
        other = eps_pair[1 - i]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        best = max(cands, key=lambda sp: abs(sp["center_pos"][0] - other[0])
                   + abs(sp["center_pos"][1] - other[1]))
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

    # Keep grey intact during cleanup; otherwise recolored trail cells would be dropped here.
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

    transferred: list[int | None] = [None, None]
    for i, d in enumerate(dest):
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one source border"
        src = srcs[0]
        new_color = int(src["center"])
        dr, dc = d["center_pos"]
        out[dr, dc] = new_color
        transferred[i] = new_color

    _paint_flow_trail(out, trail_path, transferred[0], transferred[1], grey)
    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_H, GRID_W = 10, 10                    # canvas size
BG, GREY = 0, 5                            # background and path colors
SPRITE_COUNTS = [4, 5]                     # number of sprites per example
SOURCE_COUNTS = [1, 2]                     # possible number of source sprites
INTERIOR_DEST_WEIGHTS = [3, 1]             # weights for [0, 1] interior dests
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24        # trail length bounds
MIN_PORT_DIST_FRAC = 3                     # min_port_dist = max(3, target_len // this)
MAX_THICKNESS = 3                          # max 8-neighbor degree for path cells
PATH_WANDER_AWAY_P = 0.5                   # P(choose away-moves) when budget is large
PATH_WANDER_DIAG_P = 0.6                   # P(prefer diagonal) in wander/moderate phases
PATH_MODERATE_RANDOM_P = 0.3               # P(random move) in moderate-budget phase
GEN_RETRIES = 300                          # outer retry limit


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
    return [(r0, c0) for r0 in range(-1, GRID_H) for c0 in range(-1, GRID_W)
            if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W]


def _is_edge(r0, c0):
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r, c, occupied):
    """Check if adding (r,c) would push any cell's 8-degree above MAX_THICKNESS."""
    nbs = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
           if (dr or dc) and (r + dr, c + dc) in occupied]
    if len(nbs) > MAX_THICKNESS:
        return True
    for nr, nc in nbs:
        deg = sum(1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr or dc) and (nr + dr, nc + dc) in occupied)
        if deg + 1 > MAX_THICKNESS:
            return True
    return False


def _gen_winding_path(grid, start, end, target_len):
    """Generate a winding 8-connected trail with no thick clusters."""
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
                if (0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) not in visited
                        and grid[nr, nc] == 0 and not _too_thick(nr, nc, visited)):
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
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W
                and 0 <= er < GRID_H and 0 <= ec < GRID_W):
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
    used_source_centers: set[int] = set()
    for j, si in enumerate(source_indices):
        forbidden = {BG, GREY, border_colors[si], *used_source_centers}
        if num_sources == 1:
            forbidden.update([border_colors[0], border_colors[1]])
        else:
            forbidden.add(border_colors[j])
        candidates = [c for c in range(1, 10) if c not in forbidden]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        source_center_colors.append(chosen)
        used_source_centers.add(chosen)

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


def _paint_output_trail(grid: np.ndarray, path_cells: list[tuple[int, int]], endpoint_colors: list[int]) -> None:
    """Paint the trail from both endpoints toward the middle using transferred colors."""
    n = len(path_cells)
    for idx, (r, c) in enumerate(path_cells):
        grid[r, c] = endpoint_colors[0] if idx <= n - 1 - idx else endpoint_colors[1]


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

    base_output = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(base_output, shape_type, border_colors,
                          colors["output_dest_centers"], min_port_dist)
    if result is None:
        return None
    dest_center_positions = result

    port_cells = []
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        port = _find_port_dir(base_output, cr, cc_pos, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(base_output, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    path_set = set(path_cells)
    for pr, pc in port_cells:
        deg = sum(1 for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
                  if (nr, nc) in path_set)
        if deg > 1:
            return None

    input_grid = base_output.copy()
    for r, c in path_cells:
        if input_grid[r, c] != BG:
            return None
        input_grid[r, c] = GREY
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        input_grid[cr, cc_pos] = colors["dest_center_colors"][i]

    sprites_to_place = (
        [(border_colors[si], colors["source_center_colors"][j])
         for j, si in enumerate(colors["source_indices"])] +
        [(border_colors[di], colors["distractor_center_colors"][j])
         for j, di in enumerate(colors["distractor_indices"])]
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

    output_grid = base_output.copy()
    _paint_output_trail(output_grid, path_cells, colors["output_dest_centers"])

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
