# Combo variant of d35bdbdc (index 36)
# combo_spec: D8/0+D15/0
# Name: 颜色图例+路径删除
#
# Design analysis
# - Concrete form:
#   The canvas expands to 12x10. Rows 0-1 hold a 2-row legend strip; rows 2-11 hold
#   the old 10x10 playfield. Each legend column is a visible pair: top pixel = source
#   border color, bottom pixel = symbol used inside destination centers. The input still
#   contains a grey winding path linking two destination sprites by ports, but the output
#   removes that path entirely and keeps only the legend plus the transformed destinations.
# - Behavior trace:
#   Original task: destination.center directly named a source border, so the solver could
#   match colors immediately and the grey path remained visible in the output. Fused task:
#   the path still identifies which two sprites are the destinations, but destination
#   centers now carry symbols rather than direct border colors. The solver must read the
#   edge legend, decode symbol -> border, copy the matching source center into the
#   destination, delete all non-destination sprites, and finally delete every grey path
#   cell as temporary scaffolding.
# - Complexity calibration:
#   The legend always has 2-3 columns because there are always 2-3 non-destination
#   sprites. Symbol colors are chosen outside the set of all sprite border colors, so the
#   mapping never collapses back to the original direct color lookup. The path is always
#   present in the input with length 10-24, so path deletion is visually obvious in every
#   output rather than being an occasional edge case.
# - Old assumptions that break:
#   1. Background can no longer be read from the outer border because the legend occupies
#      that border; use the global mode instead.
#   2. Sprite detection must exclude legend cells or the top strip would be misread as
#      tiny sprite components.
#   3. Destination.center is no longer the lookup key itself; it must be decoded through
#      the legend before matching against source borders.
#   4. Keeping grey cells in the output is wrong for this combo; all grey cells must be
#      scrubbed after the transfer.
#   5. "Keep everything connected to destinations except bg" would incorrectly retain the
#      path; the keep-mask must preserve only legend pixels and destination sprite pixels.
# - Placement legality:
#   Rows 0-1 are reserved for the legend, so both sprites and grey path are confined to
#   rows 2-11. Sprite placement still forbids overlap and 4-touching with other sprites.
#   Port cells are required to sit in the playfield, and the winding path generator avoids
#   occupied cells plus over-thick 8-neighborhood clusters.
# - Visual observability:
#   Across examples, the input shows destination centers in colors that do not appear as
#   any sprite border, forcing a human to consult the top legend. The output makes the
#   path deletion salient because all grey cells vanish while the legend remains. This
#   yields a clean two-step visual story: path chooses the recipients, legend chooses the
#   donors.
# - Neatness / inferability:
#   The legend is isolated in its own strip, the playfield stays uncluttered, and only two
#   sprites survive in the output. Color changes are supported by a small repeated mapping
#   set (2-3 legend pairs), so a human can infer the decoding rule from 2-3 examples.

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_mode(g: np.ndarray) -> int:
    """Most common color in the full grid."""
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[cnts.argmax()])


def _legend_info(
    g: np.ndarray, bg: int, grey: int,
) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    """Read the top-edge 2-row legend as border->symbol and symbol->border maps."""
    assert g.shape[0] >= 2, "legend requires at least 2 rows"
    cols = [
        c
        for c in range(g.shape[1])
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
    assert cells, "expected a grey path"
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
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

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
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
      - it is exactly 2 steps away from the sprite center in a cardinal direction,
      - the intermediate cell has the sprite's border color.
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
    Decode edge-legend symbols to choose source sprites, then erase the guiding path.

    Input structure:
    - The top two rows contain a legend strip whose columns pair source border colors
      with symbolic token colors.
    - The lower playfield contains several bordered sprites plus a grey one-stroke path.
      The path endpoints attach to exactly two destination sprites through valid ports.

    Rule:
    - Use the path endpoints to identify the two destination sprites.
    - For each destination, treat its center color as a symbol, decode symbol -> border
      through the legend, find the unique sprite with that border color, and copy that
      sprite's center color into the destination center.
    - Remove every non-destination sprite and also remove the entire grey path; only the
      legend and the transformed destination sprites remain.

    Key insight:
    - The path only tells you which sprites survive; it does not tell you what color they
      should receive. That second step comes from the edge legend, so solving requires
      combining transient geometric routing with persistent symbolic lookup.
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
        assert cands, "each trail endpoint must identify a destination sprite"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 destination sprites"
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
    for d in dest:
        for r, c in d["pts"]:
            keep[r, c] = True
    drop = (out != bg) & (~keep) & (out != grey)
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

    out[out == grey] = bg
    return out


# --------------- generator (c) ---------------

# Tunable generation parameters
GRID_H, GRID_W = 12, 10
PLAY_R0 = 2
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


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch with center at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place a patch on the grid, treating BG as transparent."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a patch placement by clearing every non-transparent patch cell."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and int(patch[dr, dc]) != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlaps and 4-touching against existing non-bg/non-grey cells."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if int(patch[dr, dc]) == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if int(grid[rr, cc]) != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All valid anchors whose center stays inside the playfield."""
    return [
        (r0, c0)
        for r0 in range(PLAY_R0, GRID_H)
        for c0 in range(-1, GRID_W)
        if PLAY_R0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= PLAY_R0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check whether adding one path cell would create an overly thick 8-neighborhood."""
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


def _gen_winding_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int,
) -> list[tuple[int, int]] | None:
    """Generate a winding 8-connected grey path within the playfield."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(cell: tuple[int, int, bool]) -> int:
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

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
                if (nr, nc) in visited or int(grid[nr, nc]) != BG:
                    continue
                if _too_thick(nr, nc, visited):
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


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
    """Check 8-connectivity of a cell set via BFS."""
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
    return len(seen) == len(cell_set)


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, bc: int) -> tuple[int, int] | None:
    """Find a valid port cell for the sprite centered at (cr,cc)."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (PLAY_R0 <= ir < GRID_H and 0 <= ic < GRID_W):
            continue
        if not (PLAY_R0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == bc and int(grid[er, ec]) == BG:
            return (er, ec)
    return None


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    """Assign source/distractor roles, legend symbols, and destination decode targets."""
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
        dest_source_indices = random.sample(source_indices, 2)
    dest_center_symbols = [symbol_by_index[idx] for idx in dest_source_indices]

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
    legend_entries = [(border_colors[idx], symbol_by_index[idx]) for idx in non_dest]

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_source_indices": dest_source_indices,
        "dest_center_symbols": dest_center_symbols,
        "source_center_by_index": source_center_by_index,
        "distractor_center_by_index": distractor_center_by_index,
        "symbol_by_index": symbol_by_index,
        "output_dest_centers": output_dest_centers,
        "legend_entries": legend_entries,
    }


def _place_legend(grid: np.ndarray, legend_entries: list[tuple[int, int]]) -> None:
    """Place a contiguous 2-row legend strip along the top edge."""
    order = legend_entries[:]
    random.shuffle(order)
    width = len(order)
    c0 = random.randint(0, GRID_W - width)
    for j, (border_color, symbol_color) in enumerate(order):
        grid[0, c0 + j] = border_color
        grid[1, c0 + j] = symbol_color


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    """Place the two destination sprites and ensure their ports can be linked."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions: list[tuple[int, int]] = []
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
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc, bc)
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue
            if i == 1:
                assert first_port is not None
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue
            dest_center_positions.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break

        if not placed:
            return None

    return dest_center_positions


def _gen_attempt() -> np.ndarray | None:
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
    dest_center_positions = _place_dests(
        output_grid,
        shape_type,
        border_colors,
        colors["output_dest_centers"],
        min_port_dist,
    )
    if dest_center_positions is None:
        return None

    port_cells = []
    for i in range(2):
        cr, cc = dest_center_positions[i]
        port = _find_port_dir(output_grid, cr, cc, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(output_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    path_grid = output_grid.copy()
    for r, c in path_cells:
        if int(path_grid[r, c]) != BG:
            return None
        path_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if PLAY_R0 <= nr < GRID_H and 0 <= nc < GRID_W and int(path_grid[nr, nc]) == GREY
        )
        if deg > 1:
            return None

    input_grid = output_grid.copy()
    for r, c in path_cells:
        input_grid[r, c] = GREY
    for i in range(2):
        cr, cc = dest_center_positions[i]
        input_grid[cr, cc] = colors["dest_center_symbols"][i]

    sprites_to_place = [
        (border_colors[idx], colors["source_center_by_index"][idx])
        for idx in colors["source_indices"]
    ] + [
        (border_colors[idx], colors["distractor_center_by_index"][idx])
        for idx in colors["distractor_indices"]
    ]

    all_pos = _sprite_positions()
    for bc, center_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, bc, center_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
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


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
