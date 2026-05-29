# Variant of d35bdbdc
# Dimension: D6 — 未连接精灵的处理 (transform)
# Change: 删除所有未连接到灰色路径的精灵（清除干扰项）... → 保留未连接精灵

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste

# Design analysis for the D6 variant
# - a) Concrete behavior:
#   Input still contains one grey trail and 4-5 two-color bordered sprites. Exactly
#   two sprites are connected to the trail endpoints through 2-step ports and are the
#   only sprites whose center colors may change. In the original task, every other
#   sprite vanished from the output. In this variant, those unconnected sprites stay
#   in place with the same border and center colors. Example trace: if the left
#   endpoint touches a red-bordered sprite whose center is blue, and some separate
#   green-bordered sprite has center yellow, then the red-bordered destination
#   receives yellow at its center because blue points to the green border. Any other
#   sprites remain visible in the output instead of being erased.
# - b) Complexity calibration:
#   The generator still uses 4-5 sprites with unique border colors, so there are
#   always at least two non-destination sprites to preserve. This makes the inversion
#   of D6 visually obvious: outputs contain both changed destinations and unchanged
#   bystanders. The color-chain freedom remains limited to 1-2 sources, matching the
#   original problem and avoiding ambiguous many-to-many lookup behavior.
# - c) Hidden assumptions from the original main():
#   The original code depended on "all non-dest material must disappear" in two
#   places: the loop that blanked non-destination sprite cells, and the flood-fill
#   that deleted any remaining non-grey/non-bg cells not connected to a destination.
#   Those assumptions are invalid here, so both removal stages must be skipped.
#   Endpoint detection, destination matching, and border->source lookup remain valid
#   because D6 changes only post-selection cleanup, not how destinations are found.
# - d) Placement legality:
#   Non-destination sprites are placed after the grey path is drawn so the generator
#   can reject any position that creates a valid grey port. That guarantees exactly
#   the two intended destinations are trail-connected. Overlap and 4-touch checks are
#   preserved so sprites remain separable and clipped-edge handling still matches the
#   solver's component detector.
# - e) Visual observability:
#   The new rule is inferable from a few examples because the same transfer action
#   happens at the endpoint-connected sprites while the other sprites now persist
#   unchanged. Nothing relies on hidden sorting by numeric color id; the only visible
#   cue is trail connectivity.
# - f) Visual cleanliness:
#   The grid keeps the original sparse layout: a thin single grey trail and several
#   small sprites that do not overlap or merge. Rejecting extra grey ports prevents
#   confusing cases where a bystander sprite appears half-connected to the trail.
# - g) Color-change inferability:
#   Only the two destination centers change, and their replacement colors are still
#   explained structurally by the matching source borders. Preserved non-destination
#   sprites provide stronger evidence for "only connected sprites change" without
#   adding extra color transformations.


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
        border = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == border:
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
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors between trail-connected bordered sprites while keeping bystanders intact.

    Input structure:
    - The grid contains background, one grey (5) one-stroke trail, and several small bordered sprites.
    - Each sprite uses exactly two colors: a repeated border color and one center pixel of a different color.
    - Only the two sprites touched by the grey trail's endpoints through valid ports are trail-connected destinations.

    Rule:
    - Determine the two destination sprites from the trail endpoints.
    - For each destination, treat its original center color as a lookup key for a source sprite whose border matches that color.
    - Replace the destination center with the matched source center color.
    - Leave every non-destination sprite exactly as it was; only the connected destinations change.

    Key insight:
    - The grey trail does not mark sprites to delete anymore; it marks the only sprites allowed to transform.
      Unconnected sprites remain as evidence that endpoint connectivity, not mere sprite existence, controls the update.
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

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

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

    def score(move: tuple[int, int, bool]) -> int:
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
                    and grid[nr, nc] == BG
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


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, border_color: int) -> tuple[int, int] | None:
    """Find a valid port direction for a sprite centered at (cr,cc) with border color."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == border_color and grid[er, ec] == BG:
            return (er, ec)
    return None


def _has_grey_port(grid: np.ndarray, cr: int, cc: int, border_color: int) -> bool:
    """True iff the sprite centered at (cr,cc) connects to any grey cell via a valid port."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == border_color and grid[er, ec] == GREY:
            return True
    return False


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
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
        random.choice([c for c in range(1, 10) if c not in {GREY, border_colors[di]}])
        for di in distractor_indices
    ]

    output_dest_centers = (source_center_colors * 2)[:2] if num_sources == 1 else source_center_colors

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
        "output_dest_centers": output_dest_centers,
    }


def _place_dests(
    grid: np.ndarray, shape_type: str, border_colors: list[int], output_dest_centers: list[int], min_port_dist: int
) -> list[tuple[int, int]] | None:
    """Place 2 destination sprites on the grid. Returns their center positions."""
    all_pos = _sprite_positions()
    edge_pos = [pos for pos in all_pos if _is_edge(pos[0], pos[1])]
    int_pos = [pos for pos in all_pos if not _is_edge(pos[0], pos[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions: list[tuple[int, int]] = []
    first_port = None

    for i in range(2):
        border = border_colors[i]
        patch = _make_sprite_patch(shape_type, border, output_dest_centers[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(grid, patch, r0, c0)
            port = _find_port_dir(grid, cr, cc, border)
            if port is None:
                _erase_sprite(grid, patch, r0, c0)
                continue
            if i == 1:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(grid, patch, r0, c0)
                    continue
            dest_center_positions.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break

        if not placed:
            return None

    return dest_center_positions


def _place_preserved_sprites(
    output_grid: np.ndarray,
    input_grid: np.ndarray,
    shape_type: str,
    sprites_to_place: list[tuple[int, int]],
) -> bool:
    """Place source/distractor sprites identically in input and output, rejecting grey ports."""
    all_pos = _sprite_positions()
    for border_color, center_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, border_color, center_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            _place_sprite(output_grid, patch, r0, c0)
            cr, cc = r0 + 1, c0 + 1
            if _has_grey_port(output_grid, cr, cc, border_color):
                _erase_sprite(output_grid, patch, r0, c0)
                continue
            _place_sprite(input_grid, patch, r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


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
    dest_center_positions = _place_dests(
        output_grid, shape_type, border_colors, colors["output_dest_centers"], min_port_dist
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

    sprites_to_place = [
        (border_colors[si], colors["source_center_colors"][j])
        for j, si in enumerate(colors["source_indices"])
    ] + [
        (border_colors[di], colors["distractor_center_colors"][j])
        for j, di in enumerate(colors["distractor_indices"])
    ]

    if not _place_preserved_sprites(output_grid, input_grid, shape_type, sprites_to_place):
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
