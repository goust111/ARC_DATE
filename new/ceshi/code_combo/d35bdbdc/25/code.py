# Combo variant of d35bdbdc (index 25)
# combo_spec: D1/1+D12/1
# Name: 按颜色分区聚集+边框色对边框色匹配

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste

# Design analysis for the combo variant
# a) Concreteization + behavior trace:
#    Use two repeated border colors, one for each destination-source pair. A typical input
#    has a left-half cluster with border 2 (destination on the grey path, source nearby but
#    disconnected) and a right-half cluster with border 7. The destination centers might be
#    3 and 4, which never appear as borders anywhere. Under the original task, the left
#    destination would look for a source whose border is 3 and would find nothing; under
#    this combo it instead looks for another sprite whose border is also 2, copies that
#    sprite's center, and the same happens independently in the right cluster. The grey
#    trail still decides which two sprites survive, but the color lookup and the spatial
#    grouping now point to the same pairing.
# b) Complexity calibration:
#    The scene keeps exactly two meaningful repeated-border pairs plus an optional unique
#    distractor. That is enough repetition for a solver to notice "same outline color means
#    same group" without creating many-to-one ambiguities. Destination centers are chosen
#    outside the border palette so the old border-vs-center rule has no support.
# c) Hidden assumptions from the original main():
#    The original lookup could safely index all sprites by border because the key was the
#    destination center color, not the destination border; here that would self-match every
#    destination. The solver must therefore exclude destination sprites from the border
#    lookup. The endpoint-to-destination logic also assumes each grey endpoint belongs to at
#    least one valid port and that the chosen destination is unique after the farthest-from-
#    other-endpoint tiebreak; the generator enforces this by forbidding every non-destination
#    sprite from exposing either trail endpoint as one of its own ports.
# d) Placement legality:
#    Generation stays output-first. Two destinations are placed in opposite horizontal
#    regions with inward-facing ports, then the grey path is drawn between those ports.
#    Input restoration only changes the destination centers and adds disconnected sprites in
#    empty cells, with overlap, 4-touching, and endpoint-port conflicts rejected.
# e) Visual observability:
#    The main visible cue is that each path-connected sprite has a nearby disconnected
#    partner with the same border color in the same half of the board. Since destination
#    centers are not border colors, the repeated outlines are the only plausible matching
#    signal, and the clustering makes that signal local rather than abstract.
# f) Visual cleanliness:
#    The two meaningful pairs are separated into left and right regions, while the grey path
#    travels between them. This keeps the repeated-border evidence legible and avoids mixing
#    several unrelated objects inside the same small area.
# g) Change inferability:
#    Only 2-3 border colors appear as outlines in any sample, and only the two repeated
#    ones participate in transfer. The optional distractor uses a unique border so it reads
#    as removable noise rather than a competing match candidate. The generator also measures
#    success against the updated solver so invalid or ambiguous layouts are discarded.


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
    Transfer center colors between bordered sprites by same-border matching, keeping only the two path-connected destinations.

    Input structure:
    - The grid contains background, a grey (5) one-stroke trail, and several 3x3 bordered sprites with one non-border center pixel each.
    - Exactly two sprites touch the two trail endpoints through valid ports; these are the destinations that remain in the output.
    - Other sprites are disconnected from the trail, and examples are generated so sprites sharing a border color are clustered in the same broad region.

    Rule:
    - Identify the two destination sprites from the grey trail endpoints.
    - For each destination, find the non-destination sprite whose border color is identical to the destination's border color.
    - Replace the destination center with that matched source's center color.
    - Delete every non-destination sprite and keep the grey trail plus the two destinations.

    Key insight:
    - The grey trail tells you which sprites survive, while the repeated outline color tells you which disconnected neighbor supplies the surviving sprite's new center; the spatial clustering reinforces that same-border pairing.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
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
        assert cands, "each trail endpoint should map to at least one sprite"
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
        if id(sp) in dest_ids:
            continue
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
        srcs = by_border.get(int(d["border"]), [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination border should match exactly one non-destination source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return 3x3 patch for a box or plus. Center is at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent (0) and out-of-bounds."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a _place_sprite by zeroing non-transparent patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != BG:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if grid[rr, cc] != BG:
                        return True
                    patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All valid (r0,c0) for a 3x3 patch where center stays on grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
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


def _gen_winding_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int
) -> list[tuple[int, int]] | None:
    """Generate a winding 8-connected grey path with no 2x2 blocks."""
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
                if (
                    0 <= nr < GRID_H
                    and 0 <= nc < GRID_W
                    and (nr, nc) not in visited
                    and grid[nr, nc] == BG
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


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
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


def _directed_port(grid: np.ndarray, cr: int, cc: int, bc: int, direction: str) -> tuple[int, int] | None:
    """Return the port cell in the requested direction if it is valid and empty."""
    dirs = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }
    dr, dc = dirs[direction]
    ir, ic = cr + dr, cc + dc
    er, ec = cr + 2 * dr, cc + 2 * dc
    if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
        return None
    if int(grid[ir, ic]) != int(bc) or int(grid[er, ec]) != BG:
        return None
    return (er, ec)


def _ports_from_center(grid: np.ndarray, cr: int, cc: int, bc: int) -> set[tuple[int, int]]:
    """All potential port cells for a placed sprite centered at (cr,cc)."""
    ports: set[tuple[int, int]] = set()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == bc:
            ports.add((er, ec))
    return ports


def _positions_in_region(side: str, *, for_dest: bool) -> list[tuple[int, int]]:
    """Candidate anchor positions whose centers lie in a broad left/right region."""
    if side == "left":
        allowed_cols = {1, 2} if for_dest else {1, 2, 3}
    else:
        allowed_cols = {7, 8} if for_dest else {6, 7, 8}
    positions = []
    for r0, c0 in _sprite_positions():
        cr, cc = r0 + 1, c0 + 1
        if cc in allowed_cols:
            positions.append((r0, c0))
    return positions


def _assign_variant_colors(dest_borders: list[int], distractor_borders: list[int]) -> dict | None:
    """Pick centers so border-to-border matching is necessary and old matching fails."""
    used_borders = set(dest_borders) | set(distractor_borders)
    available = [c for c in range(1, 10) if c != GREY and c not in used_borders]
    needed = 4 + len(distractor_borders)
    if len(available) < needed:
        return None

    random.shuffle(available)
    return {
        "source_center_colors": available[:2],
        "dest_center_colors": available[2:4],
        "distractor_center_colors": available[4:needed],
    }


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    dest_borders: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place one destination in each half, with inward-facing ports for the grey trail."""
    placements: list[tuple[int, int]] = []
    ports: list[tuple[int, int]] = []
    specs = [("left", "right"), ("right", "left")]

    first_port = None
    first_row = None
    for i, (region, port_dir) in enumerate(specs):
        patch = _make_sprite_patch(shape_type, dest_borders[i], output_dest_centers[i])
        edge_pos = [p for p in _positions_in_region(region, for_dest=True) if _is_edge(p[0], p[1])]
        int_pos = [p for p in _positions_in_region(region, for_dest=True) if not _is_edge(p[0], p[1])]
        num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]
        pool = list(int_pos if i < num_interior_dest else edge_pos or int_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            if first_row is not None and abs(cr - first_row) < 2:
                continue
            _place_sprite(output_grid, patch, r0, c0)
            port = _directed_port(output_grid, cr, cc, dest_borders[i], port_dir)
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue
            if first_port is not None:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue
            placements.append((cr, cc))
            ports.append(port)
            if first_port is None:
                first_port = port
                first_row = cr
            placed = True
            break

        if not placed:
            return None

    return placements, ports


def _place_clustered_sprite(
    input_grid: np.ndarray,
    shape_type: str,
    trail_eps: set[tuple[int, int]],
    border_color: int,
    center_color: int,
    side: str,
    ref_center: tuple[int, int],
) -> tuple[int, int] | None:
    """Place a disconnected sprite in the same half and near its destination partner."""
    patch = _make_sprite_patch(shape_type, border_color, center_color)

    def center_dist(pos: tuple[int, int]) -> int:
        cr, cc = pos[0] + 1, pos[1] + 1
        return abs(cr - ref_center[0]) + abs(cc - ref_center[1])

    base = _positions_in_region(side, for_dest=False)
    close = [
        pos
        for pos in base
        if 3 <= center_dist(pos) <= 5
        and abs((pos[0] + 1) - ref_center[0]) <= 3
        and abs((pos[1] + 1) - ref_center[1]) <= 2
    ]
    medium = [
        pos
        for pos in base
        if 3 <= center_dist(pos) <= 6
        and abs((pos[0] + 1) - ref_center[0]) <= 4
        and abs((pos[1] + 1) - ref_center[1]) <= 3
    ]
    pools = [close, medium]

    for pool in pools:
        shuffled = list(pool)
        random.shuffle(shuffled)
        shuffled.sort(key=center_dist)
        for r0, c0 in shuffled:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            cr, cc = r0 + 1, c0 + 1
            if trail_eps & _ports_from_center(input_grid, cr, cc, border_color):
                _erase_sprite(input_grid, patch, r0, c0)
                continue
            return (cr, cc)
    return None


def _place_distractor(
    input_grid: np.ndarray,
    shape_type: str,
    trail_eps: set[tuple[int, int]],
    border_color: int,
    center_color: int,
    dest_centers: list[tuple[int, int]],
) -> bool:
    """Place an optional unique-border distractor away from the two clustered pairs."""
    patch = _make_sprite_patch(shape_type, border_color, center_color)
    all_pos = _sprite_positions()

    def min_dest_dist(pos: tuple[int, int]) -> int:
        cr, cc = pos[0] + 1, pos[1] + 1
        return min(abs(cr - dr) + abs(cc - dc) for dr, dc in dest_centers)

    far = [pos for pos in all_pos if min_dest_dist(pos) >= 5]
    pools = [far, all_pos]

    for pool in pools:
        shuffled = list(pool)
        random.shuffle(shuffled)
        shuffled.sort(key=min_dest_dist, reverse=True)
        for r0, c0 in shuffled:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            cr, cc = r0 + 1, c0 + 1
            if trail_eps & _ports_from_center(input_grid, cr, cc, border_color):
                _erase_sprite(input_grid, patch, r0, c0)
                continue
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    num_sprites = random.choice(SPRITE_COUNTS)
    distractor_count = num_sprites - 4

    palette = [c for c in range(1, 10) if c != GREY]
    dest_borders = random.sample(palette, 2)
    remaining_borders = [c for c in palette if c not in dest_borders]
    distractor_borders = random.sample(remaining_borders, distractor_count)

    colors = _assign_variant_colors(dest_borders, distractor_borders)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(
        output_grid,
        shape_type,
        dest_borders,
        colors["source_center_colors"],
        min_port_dist,
    )
    if result is None:
        return None
    dest_center_positions, port_cells = result

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

    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc = dest_center_positions[i]
        input_grid[cr, cc] = colors["dest_center_colors"][i]

    source0 = _place_clustered_sprite(
        input_grid,
        shape_type,
        set(port_cells),
        dest_borders[0],
        colors["source_center_colors"][0],
        "left",
        dest_center_positions[0],
    )
    if source0 is None:
        return None
    source1 = _place_clustered_sprite(
        input_grid,
        shape_type,
        set(port_cells),
        dest_borders[1],
        colors["source_center_colors"][1],
        "right",
        dest_center_positions[1],
    )
    if source1 is None:
        return None

    same0 = abs(source0[0] - dest_center_positions[0][0]) + abs(source0[1] - dest_center_positions[0][1])
    same1 = abs(source1[0] - dest_center_positions[1][0]) + abs(source1[1] - dest_center_positions[1][1])
    cross0 = abs(source1[0] - dest_center_positions[0][0]) + abs(source1[1] - dest_center_positions[0][1])
    cross1 = abs(source0[0] - dest_center_positions[1][0]) + abs(source0[1] - dest_center_positions[1][1])
    if not (same0 < cross0 and same1 < cross1):
        return None

    for i in range(distractor_count):
        if not _place_distractor(
            input_grid,
            shape_type,
            set(port_cells),
            distractor_borders[i],
            colors["distractor_center_colors"][i],
            dest_center_positions,
        ):
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
