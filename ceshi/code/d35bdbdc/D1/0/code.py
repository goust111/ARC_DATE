# Variant of d35bdbdc
# Dimension: D1 — 匹配条件：边框色与中心色的对应关系 (match_cond)
# Change: s.border_color == d.center_color（源精灵的边框色等于目标精灵的中心色）... → 中心色对中心色匹配
#
# Design analysis
# - Concrete behavior: the grid still contains two path-connected destination sprites plus
#   1-2 detached source sprites and optional distractors. The grey trail still selects the
#   two destinations by endpoint ports. The changed step is the color lookup: a destination
#   now searches detached sprites by shared center color, then writes the matched source's
#   border color into the destination center. This extra payload swap is the minimal
#   supporting change needed to avoid the center-to-center match collapsing into a no-op.
# - Example trace: original logic could use dest.center=7 to find a source with border=7
#   and then copy that source center, say 3, into the destination. In this variant,
#   dest.center=3 finds the source whose center is also 3, and the destination becomes that
#   source's border, say 7. The lookup key moved from border to center, and the written
#   value moved to the other visible source attribute so the change remains observable.
# - Complexity calibration: source center colors are unique lookup keys unless there is a
#   deliberate one-source/two-destination case, where both destinations share that same key.
#   Distractor centers are forbidden from reusing any lookup key, preventing ambiguous
#   multi-source matches. Source center colors are also kept outside the set of border colors,
#   so the new samples do not silently satisfy the old border-based rule.
# - Old assumptions that break: the original `by_border` table could safely include the
#   destinations themselves because a destination's own border never served as its lookup key.
#   After the D1 change, indexing all sprites by center would make every destination match
#   itself immediately. The solver therefore must build the lookup table only from non-
#   destination sprites.
# - Placement legality: sprite geometry, port detection, path generation, clipping, and the
#   output-first oracle loop are unchanged. The generator still constructs the final output,
#   restores input-only destination centers, then places detached sprites only where they do
#   not overlap or 4-touch existing colored cells.
# - Visual observability and cleanliness: the two destination sprites remain visually singled
#   out by the grey trail, while detached sprites expose both their border and center colors.
#   Because the input destination centers repeat source center colors and the outputs switch to
#   detached border colors, a human can see the center-based pairing from a few examples.
# - New solver assumptions checked against the generator: each destination has at most one
#   non-destination source sharing its center color; unmatched destinations may stay unchanged,
#   but generated tasks always provide a match; and repeated destination centers occur only in
#   the one-source branch, where both legitimately map to the same source border.

import numpy as np

from arc_utils import neighbors4, neighbors8, components_mask
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

    # Collect grey cells that are valid port cells for any sprite
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve path-selected destination sprites by center-color matching and keep only those destinations.

    Input structure:
    - The grid contains background, a grey (5) one-stroke trail, and several small bordered sprites.
    - Each sprite has a repeated border color plus one distinct center pixel, and the trail endpoints touch exactly two destination sprites via valid ports.

    Rule:
    - Use the two trail endpoints to identify the destination sprites.
    - For each destination, find the detached source sprite whose center color matches the destination's original center color.
    - Replace the destination center with that matched source's border color.
    - Remove every non-destination sprite; the grey trail and the two destinations remain.

    Key insight: The destination center is now a direct equality key on other sprite centers, while the visible output color comes from the matched source border.
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

    # Center-based lookup must exclude destinations, or each destination would match itself.
    by_center: dict[int, list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        by_center.setdefault(int(sp["center"]), []).append(sp)

    out = g.copy()

    # Delete all non-destination sprites completely.
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    # Also delete any remaining non-grey, non-bg pixels not connected to a destination sprite.
    # (Some instances contain extra non-sprite components that are removed in the output.)
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

    # Update destination centers by matching their original center color to a detached source center.
    for d in dest:
        want_center = int(d["center"])
        srcs = by_center.get(want_center, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination center color should match at most one detached source"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["border"])

    return out

# --------------- generator (c) ---------------

from arc_utils import paste

# ── CFG: tunable generation parameters ──────────────────────────────
GRID_H, GRID_W = 10, 10                    # canvas size
BG, GREY = 0, 5                            # background and path colors
SPRITE_COUNTS = [4, 5]                      # number of sprites per example
SOURCE_COUNTS = [1, 2]                      # possible number of source sprites
INTERIOR_DEST_WEIGHTS = [3, 1]              # weights for [0, 1] interior dests
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24        # grey trail length bounds
MIN_PORT_DIST_FRAC = 3                      # min_port_dist = max(3, target_len // this)
MAX_THICKNESS = 3                           # max 8-neighbor degree for path cells
PATH_WANDER_AWAY_P = 0.5                    # P(choose away-moves) when budget is large
PATH_WANDER_DIAG_P = 0.6                    # P(prefer diagonal) in wander/moderate phases
PATH_MODERATE_RANDOM_P = 0.3               # P(random move) in moderate-budget phase
GEN_RETRIES = 300                           # outer retry limit


def _make_sprite_patch(shape_type, border_color, center_color):
    """Return 3x3 patch for a box or plus. Center is at (1,1)."""
    if shape_type == 'box':
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
            # Plenty of budget: strongly prefer wandering away or sideways
            if away and random.random() < PATH_WANDER_AWAY_P:
                pool = away
            elif diag and random.random() < PATH_WANDER_DIAG_P:
                pool = diag
            else:
                pool = moves
            chosen = random.choice(pool)
        elif budget > dist + 1:
            # Some budget: mix of diagonal and toward-target
            pool = diag if diag and random.random() < PATH_WANDER_DIAG_P else moves
            if random.random() < PATH_MODERATE_RANDOM_P:
                chosen = random.choice(pool)
            else:
                chosen = min(pool, key=score)
        else:
            # Tight: head toward target, prefer diagonal
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
    random.shuffle(remaining)
    num_sources = random.choice(SOURCE_COUNTS) if len(remaining) >= 2 else max(1, len(remaining))
    if num_sources == 0:
        return None
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    non_border_palette = [c for c in range(1, 10) if c not in border_colors and c != GREY]
    if len(non_border_palette) < num_sources:
        return None

    # Source centers supply the lookup keys; keeping them off border colors avoids old-rule collisions.
    source_center_colors = random.sample(non_border_palette, num_sources)

    # Destination centers are the lookup keys seen in the input.
    if num_sources == 1:
        dest_center_colors = [source_center_colors[0]] * 2
        output_dest_centers = [border_colors[source_indices[0]]] * 2
    else:
        dest_center_colors = list(source_center_colors)
        output_dest_centers = [border_colors[si] for si in source_indices]

    distractor_center_colors = []
    for di in distractor_indices:
        forbidden = {BG, GREY, border_colors[di], *source_center_colors}
        candidates = [c for c in range(1, 10) if c not in forbidden]
        if not candidates:
            return None
        distractor_center_colors.append(random.choice(candidates))

    return dict(
        source_indices=source_indices, distractor_indices=distractor_indices,
        dest_center_colors=dest_center_colors, source_center_colors=source_center_colors,
        distractor_center_colors=distractor_center_colors, output_dest_centers=output_dest_centers,
    )


def _place_dests(output_grid, shape_type, border_colors, output_dest_centers, min_port_dist):
    """Place 2 destination sprites on the grid. Returns (center_positions, first_port) or None."""
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

            # Temporarily place to check port availability
            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc_pos, bc)
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue

            # Second dest must be far enough from first port
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
    shape_type = random.choice(['box', 'plus'])
    palette = [c for c in range(1, 10) if c != GREY]
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(palette, num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(output_grid, shape_type, border_colors,
                          colors['output_dest_centers'], min_port_dist)
    if result is None:
        return None
    dest_center_positions = result

    # Find port cells for grey path endpoints
    port_cells = []
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        port = _find_port_dir(output_grid, cr, cc_pos, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    # Generate grey path
    path_cells = _gen_winding_path(output_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    # Verify port cells are trail endpoints (4-degree <= 1 within grey)
    for pr, pc in port_cells:
        deg = sum(1 for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
                  if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY)
        if deg > 1:
            return None

    # Derive input from output: restore original dest center colors
    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        input_grid[cr, cc_pos] = colors['dest_center_colors'][i]

    # Place source and distractor sprites on input grid
    sprites_to_place = (
        [(border_colors[si], colors['source_center_colors'][j])
         for j, si in enumerate(colors['source_indices'])] +
        [(border_colors[di], colors['distractor_center_colors'][j])
         for j, di in enumerate(colors['distractor_indices'])]
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

    # Oracle check
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
