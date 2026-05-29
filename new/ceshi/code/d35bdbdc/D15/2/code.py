# Variant of d35bdbdc
# Dimension: D15 — 符号编码与图例机制 (symbol_instructions)
# Change: 无符号编码（颜色匹配规则由示例隐含给出）... → 精灵内部的方向箭头标记

"""
Design analysis for the D15 variant

a) Concrete behavior and flow change
   - Keep the original scaffold: a 10x10 grid, grey one-stroke path, and two path-connected
     destination sprites that survive in the output.
   - Change only the source-selection mechanism. The old task used destination center colors as
     hidden lookup keys into source border colors. The variant removes that color-chain and puts
     an explicit 3-pixel arrow inside each source sprite.
   - All sprites remain 3x3 bordered boxes. Destinations/distractors keep a single interior pixel.
     Sources replace the single interior pixel with a 3-pixel arrowhead made from the transfer color.
   - Example trace:
       old: dest.center == source.border -> copy source.center into dest.center
       new: source.arrow points right, exactly one destination lies to the right of that source ->
            copy source.arrow color into that destination center
     The path still tells us which two sprites are destinations; the arrow only decides source->target.

b) Complexity calibration
   - The arrow alphabet is limited to four global directions: up/down/left/right.
   - Each sample contains exactly two arrow-bearing sources so the symbol is repeated and decodable.
   - Destination input-center colors are chosen to avoid the original border-match mechanism, so the
     new symbolic rule is necessary rather than cosmetic.

c) Hidden assumptions from the old solver that no longer hold
   - Old `_detect_bordered_sprites` assumed the inner color appears exactly once. That breaks for
     source sprites because the transfer color now occupies three cells.
   - Old update logic used `dest.center` as a lookup key. That assumption is removed entirely.
   - Port detection assumed the four cardinal neighbors of the geometric center are border-colored.
     The arrow therefore uses only the center and corners, leaving cardinal border cells untouched.

d) Placement legality
   - Destination sprites may still be clipped at edges, matching the original visual language.
   - Arrow sources must be fully visible so their 3-pixel symbols are never clipped or ambiguous.
   - Source placement is filtered so each source has exactly one destination in its encoded direction.
   - Overlap/4-touch constraints and path validity checks are preserved.

e) Visual observability
   - The new rule is visible without reading color ids: two surviving destinations are still marked
     by the grey path, while the deleted source sprites visibly contain arrowheads that point toward
     the destination they affect.
   - Input destination center colors are not reliable lookup keys anymore, so examples force the
     solver to use the arrows.

f) Visual cleanliness
   - Only box sprites are used in this variant. That keeps the arrow symbols readable and avoids
     mixing shape differences with the new symbolic mechanism.
   - Arrows are 3-pixel corner-and-center motifs (`^`, `v`, `<`, `>`) inside a 3x3 frame, which is
     legible without adding extra layers or larger objects.

g) Color-change inferability
   - The color transferred to a destination is simply the arrow color shown inside the matching source.
   - No hidden ordering by color id is used. Mapping is driven by visible geometry only.

h) New solver assumptions and generator guarantees
   - Assumption: every arrow source is a full 3x3 box with one of four known 3-pixel patterns.
     Guarantee: generator places all sources away from edges and uses only those patterns.
   - Assumption: for each source, exactly one of the two destinations lies in the encoded global
     direction cone. Guarantee: generator filters source placements with the same direction test
     used by the solver.
"""

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


ARROW_PATTERNS = {
    "up": {(-1, -1), (-1, 1), (0, 0)},
    "down": {(0, 0), (1, -1), (1, 1)},
    "left": {(-1, -1), (0, 0), (1, -1)},
    "right": {(-1, 1), (0, 0), (1, 1)},
}


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
    Detect 2-color bordered sprites where the majority color is the border.

    Destination/distractor sprites have 1 inner pixel.
    Source sprites have a 3-pixel inner arrow pattern.
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
        border_color = max(counts, key=counts.get)
        inner_color = [c for c in uniq if c != border_color][0]
        if counts[border_color] <= counts[inner_color]:
            continue
        if border_color == bg or border_color == grey or inner_color == grey:
            continue

        pts_arr = np.array(pts, dtype=int)
        r0, c0 = pts_arr.min(axis=0)
        r1, c1 = pts_arr.max(axis=0)
        center_pos = ((int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2)
        inner_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == inner_color]

        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "inner": int(inner_color),
                "inner_pts": inner_pts,
                "center_pos": center_pos,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint (er,ec) is a port for this sprite if:
      - it is 2 steps from the sprite center along a cardinal direction,
      - the intermediate cell is the border color.
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


def _decode_arrow(sprite: dict) -> str | None:
    """Return the arrow direction for a full 3x3 source sprite, else None."""
    r0, c0, r1, c1 = sprite["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return None
    cr, cc = sprite["center_pos"]
    rel = {(r - cr, c - cc) for r, c in sprite["inner_pts"]}
    for direction, pattern in ARROW_PATTERNS.items():
        if rel == pattern:
            return direction
    return None


def _direction_to(src: tuple[int, int], dst: tuple[int, int]) -> str | None:
    """Dominant-axis direction from src to dst. Ties are treated as ambiguous."""
    dr = int(dst[0] - src[0])
    dc = int(dst[1] - src[1])
    if dr == 0 and dc == 0:
        return None
    if abs(dr) == abs(dc):
        return None
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer source arrow colors into the two path-connected bordered sprites, keeping only those destinations.

    Input structure:
    - The grid contains background, a grey (5) one-stroke trail, two destination bordered boxes attached to the
      trail endpoints, and additional bordered boxes away from the trail.
    - Non-destination boxes are either plain distractors or full 3x3 source boxes whose interior 3-pixel arrow
      points in one of the four global directions.

    Rule:
    - First find the two destination sprites from the grey trail endpoints exactly as in the original task.
    - For every arrow-bearing source sprite, decode its pointing direction and identify which destination lies in
      that direction from the source. Copy the source arrow color into that destination center pixel.
    - Delete every non-destination sprite and any extra non-grey, non-background components.

    Key insight:
    - The destination set is still selected by the grey path, but the source-to-target mapping is no longer hidden
      in color equality. It is written explicitly as a tiny directional symbol inside each source sprite.
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
    assert all(d is not None for d in dest), "expected exactly 2 destination sprites from trail endpoints"
    dest_ids = {id(sp) for sp in dest}

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
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    transfers: dict[int, int] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        direction = _decode_arrow(sp)
        if direction is None:
            continue
        cands = [d for d in dest if _direction_to(sp["center_pos"], d["center_pos"]) == direction]
        assert len(cands) == 1, "each source arrow should identify exactly one destination"
        target = cands[0]
        transfers[id(target)] = int(sp["inner"])

    for d in dest:
        new_color = transfers.get(id(d))
        if new_color is None:
            continue
        dr, dc = d["center_pos"]
        out[dr, dc] = new_color

    return out


# --------------- generator ---------------

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
GEN_RETRIES = 500


def _make_plain_patch(border_color: int, center_color: int) -> np.ndarray:
    p = np.full((3, 3), border_color, dtype=int)
    p[1, 1] = center_color
    return p


def _make_arrow_patch(border_color: int, arrow_color: int, direction: str) -> np.ndarray:
    p = np.full((3, 3), border_color, dtype=int)
    for dr, dc in ARROW_PATTERNS[direction]:
        p[1 + dr, 1 + dc] = arrow_color
    return p


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
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
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _full_box_positions() -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in range(GRID_H - 2) for c0 in range(GRID_W - 2)]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
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
    path = [start]
    visited = {start}
    cr, cc = start

    def score(m: tuple[int, int, bool]) -> int:
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


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, bc: int) -> tuple[int, int] | None:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == BG:
            return (er, ec)
    return None


def _pick_nonborder_colors(border_colors: list[int], count: int) -> list[int] | None:
    candidates = [c for c in range(1, 10) if c != GREY and c not in border_colors]
    if len(candidates) < count:
        return None
    return random.sample(candidates, count)


def _place_dests(
    output_grid: np.ndarray, border_colors: list[int], output_dest_centers: list[int], min_port_dist: int
) -> list[tuple[int, int]] | None:
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions: list[tuple[int, int]] = []
    first_port = None

    for i in range(2):
        bc = border_colors[i]
        patch = _make_plain_patch(bc, output_dest_centers[i])
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


def _place_arrow_source(
    grid: np.ndarray,
    border_color: int,
    arrow_color: int,
    target_idx: int,
    dest_centers: list[tuple[int, int]],
) -> bool:
    candidates = []
    for r0, c0 in _full_box_positions():
        src_center = (r0 + 1, c0 + 1)
        dirs = [_direction_to(src_center, dc) for dc in dest_centers]
        target_dir = dirs[target_idx]
        if target_dir is None or dirs[1 - target_idx] == target_dir:
            continue
        patch = _make_arrow_patch(border_color, arrow_color, target_dir)
        if _overlaps(grid, patch, r0, c0):
            continue
        candidates.append((r0, c0, patch))

    random.shuffle(candidates)
    if not candidates:
        return False
    r0, c0, patch = candidates[0]
    _place_sprite(grid, patch, r0, c0)
    return True


def _place_distractor(grid: np.ndarray, border_color: int, center_color: int) -> bool:
    patch = _make_plain_patch(border_color, center_color)
    pool = _sprite_positions()
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(grid, patch, r0, c0):
            continue
        _place_sprite(grid, patch, r0, c0)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    num_sprites = random.choice(SPRITE_COUNTS)
    num_distractors = num_sprites - 4
    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, num_sprites)

    source_colors = _pick_nonborder_colors(border_colors, 2)
    if source_colors is None:
        return None

    # Make input destination centers visibly different from their eventual outputs.
    dest_input_centers = [source_colors[1], source_colors[0]]

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_centers = _place_dests(output_grid, border_colors, source_colors, min_port_dist)
    if dest_centers is None:
        return None

    port_cells = []
    for i in range(2):
        cr, cc = dest_centers[i]
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
        cr, cc = dest_centers[i]
        input_grid[cr, cc] = dest_input_centers[i]

    if not _place_arrow_source(input_grid, border_colors[2], source_colors[0], 0, dest_centers):
        return None
    if not _place_arrow_source(input_grid, border_colors[3], source_colors[1], 1, dest_centers):
        return None

    for i in range(num_distractors):
        bc = border_colors[4 + i]
        choices = [c for c in range(1, 10) if c not in (GREY, bc)]
        if not choices:
            return None
        if not _place_distractor(input_grid, bc, random.choice(choices)):
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
