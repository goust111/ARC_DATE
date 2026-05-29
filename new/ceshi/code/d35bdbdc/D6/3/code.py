# Variant of d35bdbdc
# Dimension: D6 — 未连接精灵的处理 (transform)
# Change: 删除所有未连接到灰色路径的精灵（清除干扰项）... → 未连接精灵重新着色为灰色
#
# Design analysis for this variant
# - Concrete behavior:
#   Input still contains one grey one-stroke path plus 4-5 small bordered sprites.
#   Exactly two sprites are selected by the two path endpoints. Their center colors are lookup
#   keys into source-sprite border colors, exactly as in the original task.
#   The visible change is only after that lookup step: every non-destination bordered sprite keeps
#   its original footprint but all of its non-background pixels become grey (5). The sprite is not
#   erased; it becomes a grey silhouette. Extra non-sprite debris, if present, is still removed.
# - Trace of the changed stage:
#   Original: a source sprite with border 6 and center 2 that is not endpoint-connected is turned
#   into all background cells in the output.
#   Variant: the same source sprite stays in place, but every visible cell in that clipped 3x3
#   shape becomes grey. Destination-center transfer still uses the source's original input colors.
#   So the difference is isolated to the cleanup stage, not to endpoint detection or color lookup.
# - Complexity calibration:
#   The recolored silhouettes are large enough to be obvious because every instance has at least one
#   non-destination source sprite. The generator preserves 4-5 total sprites, 1-2 sources, and the
#   same path complexity, so the puzzle still requires the original destination/source reasoning.
#   To make the D6 change visually legible, non-destination sprites are preferentially placed next
#   to existing grey cells when legal, so many outputs show them visually absorbed into the path.
# - Implicit-assumption audit for main():
#   `_detect_bordered_sprites` depends on non-grey sprite colors in the input; this still holds
#   because recoloring happens only in the output, never in the generated input.
#   `_find_trail_endpoints`, `_ports_for_sprite`, and the destination bipartite match depend only on
#   the input path and original sprite geometry; these remain unchanged.
#   The original cleanup loop assumed "non-destination sprites should vanish", so assigning `bg`
#   there would now be wrong. The safe replacement is recoloring those sprite pixels to `grey`
#   before the residual-debris cleanup, because the later drop-mask already excludes grey cells.
#   The destination-center update must still run after recoloring, otherwise recoloring destination
#   sources would corrupt the lookup information. We therefore keep the lookup table from the input.
# - Placement legality:
#   The generator still places destinations first, then the path, then the colored source/distractor
#   sprites on the input. For each such placement, it mirrors the same footprint into the expected
#   output using a grey-only patch. This preserves clipping, avoids overlap, and guarantees that the
#   new solver sees legal inputs while the oracle output contains the intended grey silhouettes.
# - Visual observability / cleanliness:
#   The only changed regions are whole non-destination sprite shapes, so the recolor is easy to see
#   across 2-4 examples. No hidden ordering by color id is introduced. Shapes stay separated from
#   other colored sprites, and only sometimes touch grey, which keeps the grid readable while still
#   demonstrating the "absorbed into the network" idea.

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


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

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    # Collect grey cells that are valid port cells for any sprite.
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


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, eps: set[tuple[int, int]],
) -> list[tuple[int, int]]:
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
    Transfer colors into the endpoint-linked sprites, then absorb all other sprites into grey.

    Input structure:
    - The canvas has a background color, a grey path (color 5), and several tiny bordered sprites.
    - Each sprite has one repeated border color and one single-pixel center color; the grey path is a
      one-stroke 8-connected trail whose two endpoints touch the two destination sprites via ports.

    Rule:
    - Use the two grey trail endpoints to identify the two destination sprites.
    - For each destination, use its original center color as a key to find the unique source sprite
      whose border color matches that key, then copy the source's center color into the destination.
    - Every non-destination bordered sprite keeps its footprint but is recolored entirely to grey.
    - Any leftover non-sprite colored debris is removed; the grey trail, grey recolored sprites, and
      the two destinations remain.

    Key insight:
    - The endpoint-connected sprites are still the only ones that participate in the color-transfer
      lookup. All remaining sprites are not deleted anymore; they are converted into grey silhouettes
      so they visually merge with the same network color as the path.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    # Trail endpoints determine the 2 destination sprites (no color tiebreaking).
    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}

    eps_pair = [te1, te2]
    opts = {
        ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)]
        for ep in eps_pair
    }

    # Bipartite match: fewer-options-first; tiebreak by sprite farthest from OTHER endpoint.
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

    # Build border->sprites lookup from the original input attributes before recoloring anything.
    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    # Variant D6: non-destination sprites are absorbed into grey instead of being erased.
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = grey

    # Preserve destination sprites and any non-grey fragments attached to them; delete other debris.
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

    # Update destination centers by matching their original center color to a source sprite border.
    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_H, GRID_W = 10, 10                   # canvas size
BG, GREY = 0, 5                           # background and path colors
SPRITE_COUNTS = [4, 5]                    # number of sprites per example
SOURCE_COUNTS = [1, 2]                    # possible number of source sprites
INTERIOR_DEST_WEIGHTS = [3, 1]            # weights for [0, 1] interior dests
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24       # grey trail length bounds
MIN_PORT_DIST_FRAC = 3                    # min_port_dist = max(3, target_len // this)
MAX_THICKNESS = 3                         # max 8-neighbor degree for path cells
PATH_WANDER_AWAY_P = 0.5                  # P(choose away-moves) when budget is large
PATH_WANDER_DIAG_P = 0.6                  # P(prefer diagonal) in wander/moderate phases
PATH_MODERATE_RANDOM_P = 0.3              # P(random move) in moderate-budget phase
GEN_RETRIES = 300                         # outer retry limit


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


def _greyify_patch(patch: np.ndarray) -> np.ndarray:
    """Recolor every non-background patch cell to grey, preserving the patch footprint."""
    return np.where(patch != BG, GREY, BG).astype(int)


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


def _touches_grey(grid, patch, r0, c0):
    """Whether the patch would sit 4-adjacent to at least one existing grey cell."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != 0:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
                    patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
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

    def score(move):
        return abs(move[0] - end[0]) + abs(move[1] - end[1])

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
        diag = [move for move in moves if move[2]]
        away = [move for move in moves if score(move) > dist]

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
            top = [move for move in best if score(move) <= score(best[0]) + 1]
            dtop = [move for move in top if move[2]]
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
        if not (
            0 <= ir < GRID_H
            and 0 <= ic < GRID_W
            and 0 <= er < GRID_H
            and 0 <= ec < GRID_W
        ):
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

    # Dest center colors = source border colors (lookup key).
    if num_sources == 1:
        dest_center_colors = [border_colors[source_indices[0]]] * 2
    else:
        dest_center_colors = [border_colors[si] for si in source_indices]

    # Source center colors: must avoid bg, grey, own border, and matched dest's border.
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


def _try_place_non_dest_pair(input_grid, output_grid, patch, grey_patch, positions):
    """
    Place a colored source/distractor on the input and its grey counterpart on the output.
    Prefer positions that touch existing grey cells so the D6 recolor is visually obvious.
    """
    touch_pool = []
    other_pool = []
    for r0, c0 in positions:
        if _overlaps(input_grid, patch, r0, c0):
            continue
        if _touches_grey(input_grid, patch, r0, c0):
            touch_pool.append((r0, c0))
        else:
            other_pool.append((r0, c0))

    for pool in (touch_pool, other_pool):
        random.shuffle(pool)
        for r0, c0 in pool:
            _place_sprite(input_grid, patch, r0, c0)
            _place_sprite(output_grid, grey_patch, r0, c0)
            return True
    return False


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
    result = _place_dests(
        output_grid, shape_type, border_colors, colors["output_dest_centers"], min_port_dist
    )
    if result is None:
        return None
    dest_center_positions = result

    # Find port cells for grey path endpoints.
    port_cells = []
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        port = _find_port_dir(output_grid, cr, cc_pos, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    # Generate grey path.
    path_cells = _gen_winding_path(output_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    # Verify port cells are trail endpoints (4-degree <= 1 within grey).
    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    # Derive input from output: restore original destination center colors.
    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        input_grid[cr, cc_pos] = colors["dest_center_colors"][i]

    # Place source and distractor sprites on input, while mirroring their grey silhouette on output.
    sprites_to_place = (
        [(border_colors[si], colors["source_center_colors"][j]) for j, si in enumerate(colors["source_indices"])]
        + [
            (border_colors[di], colors["distractor_center_colors"][j])
            for j, di in enumerate(colors["distractor_indices"])
        ]
    )
    positions = _sprite_positions()
    random.shuffle(positions)
    for bc, cc_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, bc, cc_color)
        grey_patch = _greyify_patch(patch)
        if not _try_place_non_dest_pair(input_grid, output_grid, patch, grey_patch, positions):
            return None

    # Oracle check.
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
