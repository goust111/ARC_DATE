# Variant of d35bdbdc
# Dimension: D3 — 源对象类型 (src)
# Change: 带中心色和边框色的精灵（有两层颜色结构：外层边框色、内层中心色）... → 三层嵌套色精灵

import numpy as np

from arc_utils import neighbors4, neighbors8, components_mask, paste
import random
from collections import deque


# Design analysis
# ---------------
# a) Concrete variant:
#    Each sprite is now a clipped 5x5 filled square with 3 visible layers:
#      - outer ring / border color on Chebyshev distance 2
#      - middle ring color on Chebyshev distance 1
#      - one-pixel core color at the center
#    The grey trail still selects exactly two destination sprites through endpoint ports,
#    but the lookup key moves from "destination core" to "destination middle ring".
#    Small trace:
#      original: dest(center=7) -> find source(border=7) -> write source.center into dest.center
#      variant : dest(middle=7, core=4) -> find source(border=7, middle=3, core=8)
#                -> write source.middle into all dest middle-ring cells and source.core into dest core
#    The visible difference is larger than a single-pixel update: the destination interior becomes
#    a copied two-layer payload, while the outer border remains fixed.
#
# b) Complexity calibration:
#    A 3x3 sprite cannot show three genuinely nested visible layers. A 5x5 filled square is the
#    smallest clean layout that makes the D3 change observable. To keep 10x10 grids feasible, the
#    sprite count drops from 4-5 to 3-4 and placements are biased toward edges so clipped examples
#    still fit.
#
# c) Old implicit assumptions that no longer hold:
#    - "component has exactly 2 colors" is false now; sprite detection must validate geometry.
#    - "rarest color is the only interior payload" is false now; there are two interior layers.
#    - "port intermediate cell is border" is false now; the first step is middle, second is border.
#    - "only one center pixel needs repainting" is false now; the whole middle ring plus core change.
#
# d) Placement legality:
#    Generator uses full clipped 5x5 rectangles, forbids overlap, and forbids placing non-destination
#    sprites where a grey endpoint would also be a valid port for them. This guarantees the solver's
#    endpoint->destination matching assumptions remain true.
#
# e) Visual observability:
#    The lookup relation is visible because each destination middle-ring color equals some source
#    border color in the input, and the output shows the source's entire inner payload copied into
#    the destination. No ordering or transformation uses hidden color-id arithmetic.
#
# f) Visual cleanliness:
#    Sprites stay separated as rectangular/clipped blocks, grey is the only path color, and copied
#    payloads are confined inside destination borders. The result remains easy to parse by eye.
#
# g) Color change inference:
#    Each example uses only a few sprites and 1-2 sources, so the repeated border<->middle keying
#    pattern is visible across the pair. The generator also avoids "no-op" transfers by ensuring the
#    destination core differs from the source core in the input.


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [3, 4]
SOURCE_COUNTS = [1, 2]
INTERIOR_DEST_WEIGHTS = [4, 1]  # weights for [0, 1] interior destinations
PATH_LEN_MIN, PATH_LEN_MAX = 8, 18
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _visible_square_cells(center_pos: tuple[int, int], radius: int = 2) -> set[tuple[int, int]]:
    cr, cc = center_pos
    r0 = max(0, cr - radius)
    r1 = min(GRID_H - 1, cr + radius)
    c0 = max(0, cc - radius)
    c1 = min(GRID_W - 1, cc + radius)
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _ring_cells(center_pos: tuple[int, int], ring: int) -> list[tuple[int, int]]:
    cr, cc = center_pos
    pts: list[tuple[int, int]] = []
    for r, c in _visible_square_cells(center_pos, radius=2):
        dist = max(abs(r - cr), abs(c - cc))
        if dist == ring:
            pts.append((r, c))
    return pts


def _detect_nested_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Nested sprite = clipped 5x5 filled square with three color layers:
      - border ring at Chebyshev distance 2
      - middle ring at Chebyshev distance 1
      - unique core at distance 0
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts_arr in comps:
        pts = [(int(r), int(c)) for r, c in pts_arr]
        pt_set = set(pts)
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 3:
            continue
        counts = {c: cols.count(c) for c in uniq}
        core_candidates = [p for p in pts if counts[int(g[p[0], p[1]])] == 1]
        for center_pos in core_candidates:
            cr, cc = center_pos
            if pt_set != _visible_square_cells(center_pos, radius=2):
                continue
            middle_pts = _ring_cells(center_pos, ring=1)
            border_pts = _ring_cells(center_pos, ring=2)
            if not middle_pts or not border_pts:
                continue
            middle_colors = {int(g[r, c]) for r, c in middle_pts}
            border_colors = {int(g[r, c]) for r, c in border_pts}
            core_color = int(g[cr, cc])
            if len(middle_colors) != 1 or len(border_colors) != 1:
                continue
            middle_color = next(iter(middle_colors))
            border_color = next(iter(border_colors))
            if len({border_color, middle_color, core_color}) != 3:
                continue
            if bg in {border_color, middle_color, core_color} or grey in {
                border_color, middle_color, core_color,
            }:
                continue
            sprites.append(
                {
                    "pts": pts,
                    "border": border_color,
                    "middle": middle_color,
                    "core": core_color,
                    "center_pos": center_pos,
                    "middle_pts": middle_pts,
                    "border_pts": border_pts,
                }
            )
            break
    return sprites


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, eps: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    A grey endpoint is a valid port if it is 3 steps away from the core in a cardinal
    direction, with middle-color then border-color cells between the core and endpoint.
    """
    cr, cc = sprite["center_pos"]
    middle = int(sprite["middle"])
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        mr, mc = cr + dr, cc + dc
        br, bc = cr + 2 * dr, cc + 2 * dc
        er, ec = cr + 3 * dr, cc + 3 * dc
        if (er, ec) not in eps:
            continue
        if not (0 <= mr < g.shape[0] and 0 <= mc < g.shape[1]):
            continue
        if not (0 <= br < g.shape[0] and 0 <= bc < g.shape[1]):
            continue
        if int(g[mr, mc]) == middle and int(g[br, bc]) == border:
            ports.append((er, ec))
    return ports


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two endpoints of the grey one-stroke trail under 8-connectivity."""
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
        middle = int(sp["middle"])
        border = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            mr, mc = cr + dr, cc + dc
            br, bc = cr + 2 * dr, cc + 2 * dc
            er, ec = cr + 3 * dr, cc + 3 * dc
            if not (0 <= mr < h and 0 <= mc < w and 0 <= br < h and 0 <= bc < w):
                continue
            if (er, ec) in cells and int(g[mr, mc]) == middle and int(g[br, bc]) == border:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    total = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == total:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer nested inner payloads between layered sprites, keeping only trail-connected targets.

    Input structure:
    - The grid contains background, a grey (5) one-stroke path, and several clipped 5x5 nested-square sprites.
    - Each sprite has three layers: an outer border ring, a middle ring, and a single core pixel.
    - The two grey trail endpoints sit just beyond the border of the two destination sprites.

    Rule:
    - Use the grey trail endpoints to identify the two destination sprites.
    - For each destination, treat its original middle-ring color as a lookup key for a source sprite whose border color matches it.
    - Copy the source sprite's middle-ring color into the destination middle ring, and copy the source core color into the destination core.
    - Remove every non-destination sprite and any stray non-grey colored components; keep the grey trail and destination borders.

    Key insight:
    - A destination sprite's visible inner ring is not its final content; it is an address that points to another sprite's border.
    - The source contributes a two-layer payload (middle ring + core), so solving requires tracking two interior attributes instead of a single center pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_nested_sprites(g, bg, grey)
    assert sprites, "expected at least one nested sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    dest: list[dict | None] = [None, None]
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
    dest_sprites = [d for d in dest if d is not None]
    dest_ids = {id(sp) for sp in dest_sprites}

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
    for d in dest_sprites:
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

    for d in dest_sprites:
        want_border = int(d["middle"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination middle color should match at most one source border"
        src = srcs[0]
        for r, c in d["middle_pts"]:
            out[r, c] = int(src["middle"])
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["core"])

    return out


def _make_sprite_patch(border_color: int, middle_color: int, core_color: int) -> np.ndarray:
    p = np.full((5, 5), border_color, dtype=int)
    p[1:4, 1:4] = middle_color
    p[2, 2] = core_color
    return p


def _place_sprite(grid: np.ndarray, patch: np.ndarray, center_pos: tuple[int, int]) -> None:
    cr, cc = center_pos
    paste(grid, patch, cr - 2, cc - 2, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, center_pos: tuple[int, int]) -> None:
    cr, cc = center_pos
    for dr in range(5):
        for dc in range(5):
            rr, cc2 = cr - 2 + dr, cc - 2 + dc
            if 0 <= rr < GRID_H and 0 <= cc2 < GRID_W and patch[dr, dc] != BG:
                grid[rr, cc2] = BG


def _overlaps(grid: np.ndarray, center_pos: tuple[int, int]) -> bool:
    patch_cells = _visible_square_cells(center_pos, radius=2)
    for rr, cc in patch_cells:
        if grid[rr, cc] != BG:
            return True
    for rr, cc in patch_cells:
        for nr, nc in neighbors4(rr, cc):
            if (nr, nc) not in patch_cells and 0 <= nr < GRID_H and 0 <= nc < GRID_W:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    return [(r, c) for r in range(GRID_H) for c in range(GRID_W)]


def _is_edge(center_pos: tuple[int, int]) -> bool:
    cr, cc = center_pos
    return min(cr, cc, GRID_H - 1 - cr, GRID_W - 1 - cc) <= 1


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    nbs = [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
           if (dr or dc) and (r + dr, c + dc) in occupied]
    if len(nbs) > MAX_THICKNESS:
        return True
    for nr, nc in nbs:
        deg = sum(
            1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and (nr + dr, nc + dc) in occupied
        )
        if deg + 1 > MAX_THICKNESS:
            return True
    return False


def _gen_winding_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int,
) -> list[tuple[int, int]] | None:
    path = [start]
    visited = {start}
    cr, cc = start

    def score(m: tuple[int, int, bool]) -> int:
        return abs(m[0] - end[0]) + abs(m[1] - end[1])

    for _ in range(target_len + 20):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves: list[tuple[int, int, bool]] = []
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


def _find_port_dir(
    grid: np.ndarray, center_pos: tuple[int, int], border_color: int, middle_color: int,
) -> tuple[int, int] | None:
    cr, cc = center_pos
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        mr, mc = cr + dr, cc + dc
        br, bc = cr + 2 * dr, cc + 2 * dc
        er, ec = cr + 3 * dr, cc + 3 * dc
        if not (0 <= mr < GRID_H and 0 <= mc < GRID_W):
            continue
        if not (0 <= br < GRID_H and 0 <= bc < GRID_W):
            continue
        if not (0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[mr, mc] == middle_color and grid[br, bc] == border_color and grid[er, ec] == BG:
            return (er, ec)
    return None


def _source_payload_positions(center_pos: tuple[int, int], endpoint: tuple[int, int]) -> bool:
    cr, cc = center_pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if (cr + 3 * dr, cc + 3 * dc) == endpoint:
            mr, mc = cr + dr, cc + dc
            br, bc = cr + 2 * dr, cc + 2 * dc
            if 0 <= mr < GRID_H and 0 <= mc < GRID_W and 0 <= br < GRID_H and 0 <= bc < GRID_W:
                return True
    return False


def _conflicts_with_endpoints(
    center_pos: tuple[int, int], endpoints: list[tuple[int, int]],
) -> bool:
    return any(_source_payload_positions(center_pos, ep) for ep in endpoints)


def _pick_color(candidates: list[int], forbidden: set[int]) -> int:
    options = [c for c in candidates if c not in forbidden]
    assert options, "no color candidates left"
    return random.choice(options)


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    palette = [c for c in range(1, 10) if c != GREY]
    remaining = list(range(2, num_sprites))
    random.shuffle(remaining)
    num_sources = min(random.choice(SOURCE_COUNTS), len(remaining))
    if num_sources == 0:
        return None
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    if num_sources == 1:
        matched_sources = [source_indices[0], source_indices[0]]
    else:
        matched_sources = source_indices[:2]

    source_payloads: dict[int, tuple[int, int]] = {}
    used_payloads: set[tuple[int, int]] = set()
    for si in source_indices:
        middle = _pick_color(palette, {BG, GREY, border_colors[si]})
        core = _pick_color(palette, {BG, GREY, border_colors[si], middle})
        if (middle, core) in used_payloads:
            options = [
                (m, k) for m in palette for k in palette
                if m not in {BG, GREY, border_colors[si]}
                and k not in {BG, GREY, border_colors[si], m}
                and (m, k) not in used_payloads
            ]
            if options:
                middle, core = random.choice(options)
        source_payloads[si] = (middle, core)
        used_payloads.add((middle, core))

    dest_input_middle = [border_colors[si] for si in matched_sources]
    dest_input_core: list[int] = []
    output_payloads: list[tuple[int, int]] = []
    for i, si in enumerate(matched_sources):
        src_middle, src_core = source_payloads[si]
        output_payloads.append((src_middle, src_core))
        forbidden = {BG, GREY, border_colors[i], dest_input_middle[i], src_core}
        dest_input_core.append(_pick_color(palette, forbidden))

    distractor_payloads: dict[int, tuple[int, int]] = {}
    for di in distractor_indices:
        middle = _pick_color(palette, {BG, GREY, border_colors[di]})
        core = _pick_color(palette, {BG, GREY, border_colors[di], middle})
        distractor_payloads[di] = (middle, core)

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "matched_sources": matched_sources,
        "source_payloads": source_payloads,
        "distractor_payloads": distractor_payloads,
        "dest_input_middle": dest_input_middle,
        "dest_input_core": dest_input_core,
        "output_payloads": output_payloads,
    }


def _place_dests(
    output_grid: np.ndarray, border_colors: list[int], output_payloads: list[tuple[int, int]], min_port_dist: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p)]
    int_pos = [p for p in all_pos if not _is_edge(p)]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_centers: list[tuple[int, int]] = []
    ports: list[tuple[int, int]] = []
    first_port: tuple[int, int] | None = None

    for i in range(2):
        border = border_colors[i]
        middle, core = output_payloads[i]
        patch = _make_sprite_patch(border, middle, core)
        if i < num_interior_dest:
            pool = list(int_pos)
            random.shuffle(pool)
            pool.extend(random.sample(edge_pos, len(edge_pos)))
        else:
            pool = list(edge_pos)
            random.shuffle(pool)
            pool.extend(random.sample(int_pos, len(int_pos)))

        placed = False
        for center_pos in pool:
            if _overlaps(output_grid, center_pos):
                continue
            _place_sprite(output_grid, patch, center_pos)
            port = _find_port_dir(output_grid, center_pos, border, middle)
            if port is None:
                _erase_sprite(output_grid, patch, center_pos)
                continue
            if i == 1 and first_port is not None:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, center_pos)
                    continue
            dest_centers.append(center_pos)
            ports.append(port)
            if i == 0:
                first_port = port
            placed = True
            break
        if not placed:
            return None

    return dest_centers, ports


def _paint_dest_input_interior(
    grid: np.ndarray, center_pos: tuple[int, int], middle_color: int, core_color: int,
) -> None:
    for r, c in _ring_cells(center_pos, ring=1):
        grid[r, c] = middle_color
    cr, cc = center_pos
    grid[cr, cc] = core_color


def _place_extra_sprites(
    input_grid: np.ndarray,
    sprites_to_place: list[tuple[int, int, int]],
    endpoints: list[tuple[int, int]],
) -> bool:
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p)]
    int_pos = [p for p in all_pos if not _is_edge(p)]
    for border, middle, core in sprites_to_place:
        patch = _make_sprite_patch(border, middle, core)
        pool = list(edge_pos)
        random.shuffle(pool)
        tail = list(int_pos)
        random.shuffle(tail)
        pool.extend(tail)

        placed = False
        for center_pos in pool:
            if _overlaps(input_grid, center_pos):
                continue
            if _conflicts_with_endpoints(center_pos, endpoints):
                continue
            _place_sprite(input_grid, patch, center_pos)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    palette = [c for c in range(1, 10) if c != GREY]
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(palette, num_sprites)
    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    placed = _place_dests(output_grid, border_colors, colors["output_payloads"], min_port_dist)
    if placed is None:
        return None
    dest_centers, port_cells = placed

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
        deg8 = sum(
            1
            for nr, nc in neighbors8(pr, pc)
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg8 != 1:
            return None

    input_grid = output_grid.copy()
    for i in range(2):
        _paint_dest_input_interior(
            input_grid,
            dest_centers[i],
            colors["dest_input_middle"][i],
            colors["dest_input_core"][i],
        )

    sprites_to_place: list[tuple[int, int, int]] = []
    for si in colors["source_indices"]:
        middle, core = colors["source_payloads"][si]
        sprites_to_place.append((border_colors[si], middle, core))
    for di in colors["distractor_indices"]:
        middle, core = colors["distractor_payloads"][di]
        sprites_to_place.append((border_colors[di], middle, core))
    random.shuffle(sprites_to_place)

    if not _place_extra_sprites(input_grid, sprites_to_place, port_cells):
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
