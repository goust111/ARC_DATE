# Variant of d35bdbdc
# Dimension: D15 — 符号编码与图例机制 (symbol_instructions)
# Change: 无符号编码（颜色匹配规则由示例隐含给出）... → 网格边缘的颜色图例
#
# Design analysis
# - Concrete form:
#   The original 10x10 play logic is kept, but the canvas becomes 12x10 so the top
#   two rows can host a legend strip without shrinking the old scene. Each legend
#   column is a visible pair: row 0 stores a sprite border color, row 1 stores the
#   symbol that encodes that border. The grey path and bordered sprites stay in the
#   lower 10x10 playfield.
# - Behavior trace:
#   Old task: destination.center == source.border, so the center color itself was the
#   lookup key. New task: destination.center is only a symbol; the solver must first
#   decode symbol -> border from the legend, then use that border to find the source
#   sprite and copy its center color.
# - Complexity calibration:
#   The legend always has 2-3 columns, not just one, so the symbolic mechanism is
#   visually present in every example. Every legend symbol is distinct and never equal
#   to any sprite border color, preventing accidental collapse back to the old rule.
# - Old assumptions that break:
#   1. Border-based background detection is no longer safe because the legend occupies
#      the outer edge, so background is now the global mode.
#   2. Destination centers can no longer be matched directly against sprite borders;
#      they must be decoded through the legend first.
#   3. "Delete every non-grey/non-bg component except the two destinations" would also
#      delete the legend, so the legend needs an explicit keep mask.
#   4. The legend touches the playfield edge, so sprite detection excludes legend cells
#      to avoid contaminating connected components.
# - Placement legality:
#   Rows 0-1 are reserved for the legend and grey path cells are forbidden there.
#   Sprites are placed only in rows 2-11, with the same non-overlap and non-touching
#   constraints as the original generator. Ports and the winding path are also limited
#   to the lower playfield, so the legend stays visually separate.
# - Visual observability:
#   In the input, the two destination centers show symbol colors that do not appear as
#   sprite borders. A human therefore has to notice the top legend to understand why
#   those centers map to particular source sprites. In the output, the legend remains
#   unchanged while only the two connected destinations receive transferred centers.

import numpy as np

from arc_utils import neighbors4, neighbors8, components_mask, paste
import random
from collections import deque


def _bg_mode(g: np.ndarray) -> int:
    """Most common color in the full grid."""
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[cnts.argmax()])


def _legend_info(
    g: np.ndarray, bg: int, grey: int,
) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    """Read the 2-row top-edge legend as border->symbol and symbol->border maps."""
    assert g.shape[0] >= 2, "legend requires at least 2 rows"
    cols = [
        int(c) for c in range(g.shape[1])
        if int(g[0, c]) not in (bg, grey) and int(g[1, c]) not in (bg, grey)
    ]
    assert len(cols) >= 2, "expected a multi-column legend on the top edge"

    border_to_symbol: dict[int, int] = {}
    symbol_to_border: dict[int, int] = {}
    mask = np.zeros(g.shape, dtype=bool)
    for c in cols:
        border = int(g[0, c])
        symbol = int(g[1, c])
        assert border not in border_to_symbol, "legend border colors must be unique"
        assert symbol not in symbol_to_border, "legend symbols must be unique"
        border_to_symbol[border] = symbol
        symbol_to_border[symbol] = border
        mask[0, c] = True
        mask[1, c] = True
    return border_to_symbol, symbol_to_border, mask


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


def _detect_bordered_sprites(
    g: np.ndarray, bg: int, grey: int, reserved: np.ndarray,
) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey/reserved) with:
      - exactly 2 colors,
      - one singleton center color,
      - one repeated border color.
    """
    mask = (g != bg) & (g != grey) & (~reserved)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_candidates = [c for c in uniq if counts[c] == 1]
        if len(center_candidates) != 1:
            continue
        center_color = center_candidates[0]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode destination-center symbols through an edge legend, then transfer source centers.

    Input structure:
    - The top edge contains a 2-row legend strip: each legend column pairs a sprite border color
      on row 0 with its symbol color on row 1.
    - Below the legend is the original scene: a grey one-stroke trail and several small bordered
      sprites, where the two sprites touched by trail endpoints are the destinations.

    Rule:
    - Identify the two destination sprites from the trail endpoints exactly as in the original task.
    - For each destination, read its center as a symbol, decode symbol -> border using the legend,
      then find the unique sprite whose border has that decoded color and copy that sprite's center
      into the destination center.
    - Delete every non-destination sprite, but keep the legend and the grey trail unchanged.

    Key insight:
    - Destination centers no longer name their source border directly; they are symbolic tokens.
      The top legend is the instruction key that converts those tokens back into actionable border colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_mode(g)
    grey = 5

    _border_to_symbol, symbol_to_border, legend_mask = _legend_info(g, bg, grey)
    sprites = _detect_bordered_sprites(g, bg, grey, legend_mask)
    assert sprites, "expected at least one bordered sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}

    eps_pair = [te1, te2]
    opts = {
        ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)]
        for ep in eps_pair
    }

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

    keep = legend_mask.copy()
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
        symbol = int(d["center"])
        want_border = symbol_to_border.get(symbol)
        if want_border is None:
            continue
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each decoded border should match at most one sprite"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_H, GRID_W = 12, 10                   # 10x10 playfield + 2 legend rows
PLAY_R0 = 2                              # rows [0,1] reserved for the legend
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
    H, W = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < H and 0 <= cc < W and patch[dr, dc] != 0:
                grid[rr, cc] = BG


def _overlaps(grid, patch, r0, c0):
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
    H, W = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != 0:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < H and 0 <= cc < W:
                    if grid[rr, cc] != 0:
                        return True
                    patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < H and 0 <= nc < W:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions():
    """All valid (r0,c0) for a 3x3 patch whose center stays in the playfield."""
    return [
        (r0, c0)
        for r0 in range(PLAY_R0, GRID_H)
        for c0 in range(-1, GRID_W)
        if PLAY_R0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0, c0):
    return r0 <= PLAY_R0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


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
    """Generate a winding 8-connected grey path with no 2x2 blocks in the playfield."""
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
                if not (PLAY_R0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if (nr, nc) in visited or grid[nr, nc] != 0 or _too_thick(nr, nc, visited):
                    continue
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
        if PLAY_R0 <= end[0] < GRID_H and not _too_thick(end[0], end[1], visited):
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
        if not (PLAY_R0 <= ir < GRID_H and 0 <= ir < GRID_H and 0 <= ic < GRID_W):
            continue
        if not (PLAY_R0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == 0:
            return (er, ec)
    return None


def _assign_colors(border_colors, num_sprites):
    """Assign destination symbols, source centers, distractor centers, and legend entries."""
    non_dest = list(range(2, num_sprites))
    num_sources = min(random.choice(SOURCE_COUNTS), len(non_dest))
    source_indices = random.sample(non_dest, num_sources)
    distractor_indices = [i for i in non_dest if i not in source_indices]

    token_pool = [c for c in range(1, 10) if c not in border_colors and c != GREY]
    if len(token_pool) < len(non_dest):
        return None
    token_choices = random.sample(token_pool, len(non_dest))
    symbol_by_index = {idx: token_choices[j] for j, idx in enumerate(non_dest)}

    if num_sources == 1:
        dest_source_indices = [source_indices[0], source_indices[0]]
    else:
        shuffled = source_indices[:]
        random.shuffle(shuffled)
        dest_source_indices = shuffled[:2]
    dest_center_colors = [symbol_by_index[idx] for idx in dest_source_indices]

    used_source_centers: set[int] = set()
    source_center_by_index: dict[int, int] = {}
    for si in source_indices:
        forbidden = {BG, GREY, border_colors[si], symbol_by_index[si]}
        candidates = [c for c in range(1, 10) if c not in forbidden and c not in used_source_centers]
        if not candidates:
            candidates = [c for c in range(1, 10) if c not in forbidden]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        source_center_by_index[si] = chosen
        used_source_centers.add(chosen)

    distractor_center_by_index: dict[int, int] = {}
    for di in distractor_indices:
        forbidden = {BG, GREY, border_colors[di]}
        candidates = [c for c in range(1, 10) if c not in forbidden]
        if not candidates:
            return None
        distractor_center_by_index[di] = random.choice(candidates)

    output_dest_centers = [source_center_by_index[idx] for idx in dest_source_indices]

    return dict(
        source_indices=source_indices,
        distractor_indices=distractor_indices,
        dest_source_indices=dest_source_indices,
        dest_center_colors=dest_center_colors,
        source_center_by_index=source_center_by_index,
        distractor_center_by_index=distractor_center_by_index,
        symbol_by_index=symbol_by_index,
        output_dest_centers=output_dest_centers,
        legend_entries=[(border_colors[idx], symbol_by_index[idx]) for idx in non_dest],
    )


def _place_legend(grid, legend_entries):
    """Place a contiguous 2-row legend strip on the top edge."""
    order = legend_entries[:]
    random.shuffle(order)
    width = len(order)
    c0 = random.randint(0, GRID_W - width)
    for j, (border_color, symbol_color) in enumerate(order):
        grid[0, c0 + j] = border_color
        grid[1, c0 + j] = symbol_color


def _place_dests(output_grid, shape_type, border_colors, output_dest_centers, min_port_dist):
    """Place 2 destination sprites on the grid. Returns their center positions or None."""
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
    _place_legend(output_grid, colors["legend_entries"])
    result = _place_dests(
        output_grid,
        shape_type,
        border_colors,
        colors["output_dest_centers"],
        min_port_dist,
    )
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

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if PLAY_R0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        input_grid[cr, cc_pos] = colors["dest_center_colors"][i]

    sprites_to_place = [
        (border_colors[idx], colors["source_center_by_index"][idx]) for idx in colors["source_indices"]
    ] + [
        (border_colors[idx], colors["distractor_center_by_index"][idx]) for idx in colors["distractor_indices"]
    ]
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
