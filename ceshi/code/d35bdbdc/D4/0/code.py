# Variant of d35bdbdc
# Dimension: D4 — 目标对象选取 (dst)
# Change: 通过灰色路径端点连接的带边框精灵为目标精灵(d)... → 网格边缘上的精灵为目标

# Design analysis for the D4 variant
# a) Concrete behavior trace
#    - The primitive objects stay the same: 3x3 bordered sprites (box or plus), a grey
#      winding trail, and color lookup via border/center pairs.
#    - Original task: the two sprites attached to the grey trail endpoints are preserved
#      and recolored; everything else is deleted.
#    - Variant: the preserved sprites are instead the sprites touching the outer grid
#      boundary. Grey-path attachment is now irrelevant for destination selection.
#    - Example trace:
#      * Input contains two edge sprites E1/E2 and two interior sprites S1/S2. The grey
#        trail endpoints touch S1/S2 through valid ports.
#      * Old logic would keep S1/S2 because it starts from trail endpoints.
#      * New logic keeps E1/E2 because they touch the frame. Their center colors still
#        point to source border colors, so recoloring uses the same lookup mechanism as
#        before, but the kept objects come from a different spatial test.
#      * Therefore the path can visibly point at disappearing interior sprites while the
#        surviving recolored sprites sit on the border.
# b) Complexity calibration
#    - To keep the rule legible, each instance has exactly two edge-touching sprites and
#      exactly two interior source sprites. An optional extra interior distractor keeps
#      variety without creating branching ambiguity.
#    - This avoids the degenerate case where "edge sprites" and "path-endpoint sprites"
#      are the same two objects, which would make the variant visually indistinguishable
#      from the original task.
# c) Implicit-assumption audit against the original main()
#    - `_find_trail_endpoints(...)` and the bipartite endpoint-to-sprite match depended on
#      the old assumption that each destination is identified by a grey endpoint port.
#      That is no longer valid and is removed from the destination selection path.
#    - The border->source lookup remains valid because border colors are still unique.
#    - The deletion flood-fill from destination pixels remains valid, but now the seed set
#      must be all edge-touching sprites instead of endpoint-connected sprites.
#    - `_bg_boundary(...)` still assumes background dominates the outer frame, so the
#      generator keeps the grey path away from the frame and limits edge sprites to two.
# d) Placement legality
#    - Generator enforces:
#      * exactly two edge-touching target sprites,
#      * exactly two interior source sprites with valid grey ports,
#      * no overlap and no 4-touch between non-grey sprites,
#      * grey path endpoints attach only to the interior source sprites,
#      * chosen source ports do not coincide with any valid port of an edge target.
#    - This guarantees `main()` sees the intended roles and that the grid remains parseable.
# e) Visual observability
#    - The output always keeps the border sprites and deletes the path-linked interior
#      sprites. Across examples, that makes the D4 change visible without reading ids or
#      relying on hidden tie-break rules.
#    - The grey path is preserved as a misleading but consistent distractor, so humans can
#      compare "what the path touches" versus "what survives".
# f) Visual cleanliness
#    - The path is generated only after reserving the interior source sprites, so it winds
#      between them without overlapping sprites.
#    - Edge targets stay separated and interior sprites are spaced enough that the scene
#      reads as four small objects plus one thin trail, not as a merged cluster.
# g) Color-change inferability
#    - Only the center pixels of the two kept edge sprites change.
#    - Each target center color equals one interior source border color, so the lookup
#      signal is structural and repeated twice per grid. Colors are sampled from the ARC
#      palette but the rule never depends on ordering by color id.

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
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    # Fallback kept from the seed: if the trail is a single one-stroke cluster without
    # explicit degree-1 endpoints, accept port cells as its logical ends.
    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        b = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if 0 <= er < g.shape[0] and 0 <= ec < g.shape[1] and (er, ec) in cells:
                if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
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


def _touches_grid_edge(sprite: dict, shape: tuple[int, int]) -> bool:
    """A sprite is a destination in this variant iff any of its cells touches the frame."""
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in sprite["pts"])


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
    Transfer center colors into the bordered sprites that touch the outer frame.

    Input structure:
    - The grid contains a grey (5) winding trail plus several small bordered sprites, each
      made from one border color and one single center pixel of another color.
    - Some sprites touch the grid boundary; others sit in the interior and may connect to
      the grey trail through old-style ports, but that path relation is not what chooses
      the destinations in this variant.

    Rule:
    - Destination sprites are exactly the bordered sprites that touch any edge of the grid.
    - For each destination, use its original center color as a lookup key: find the sprite
      whose border color matches that key, then copy that source sprite's center color into
      the destination center.
    - Delete every non-destination sprite; preserve the grey trail and the edge-touching
      destinations.

    Key insight: The color-transfer mechanism is unchanged, but the "which sprites survive"
    decision comes purely from boundary contact, not from the trail endpoints.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    dest = [sp for sp in sprites if _touches_grid_edge(sp, g.shape)]
    assert dest, "expected at least one edge-touching destination sprite"
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
TARGET_COUNT = 2
SOURCE_COUNT = 2
EXTRA_DISTRACTOR_WEIGHTS = [2, 1]  # weights for [0, 1] extra interior distractors
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch with its center at local coord (1,1)."""
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
    """Undo a placement by clearing all non-transparent patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check overlap or 4-touch against existing non-bg, non-grey cells."""
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
    """All anchors whose center cell stays inside the 10x10 grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge_anchor(r0: int, c0: int) -> bool:
    """Whether a 3x3 sprite anchored at (r0,c0) touches the frame after clipping."""
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check whether adding (r,c) would create an overly thick 8-neighbor cluster."""
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
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    target_len: int,
) -> list[tuple[int, int]] | None:
    """Generate a winding 8-connected grey path between two reserved port cells."""
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
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if (nr, nc) in visited:
                    continue
                if grid[nr, nc] != BG:
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
        if grid[end[0], end[1]] != BG:
            return None
        if _too_thick(end[0], end[1], visited):
            return None
        path.append(end)

    return path


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
    """Check 8-connectivity of a cell set via BFS."""
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


def _possible_port_cells(center_pos: tuple[int, int], border_color: int, grid: np.ndarray) -> list[tuple[int, int]]:
    """All geometric port cells reachable from a sprite center through its border."""
    cr, cc = center_pos
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == border_color:
            ports.append((er, ec))
    return ports


def _available_ports(center_pos: tuple[int, int], border_color: int, grid: np.ndarray) -> list[tuple[int, int]]:
    """Valid port cells whose endpoint is empty on the working grid."""
    return [p for p in _possible_port_cells(center_pos, border_color, grid) if int(grid[p]) == BG]


def _place_edge_targets(
    grid: np.ndarray,
    shape_type: str,
    target_borders: list[int],
    output_target_centers: list[int],
) -> list[dict] | None:
    """Place exactly two destinations on the frame and keep them well separated."""
    edge_positions = [pos for pos in _sprite_positions() if _is_edge_anchor(pos[0], pos[1])]
    random.shuffle(edge_positions)
    placed: list[dict] = []
    for i in range(TARGET_COUNT):
        patch = _make_sprite_patch(shape_type, target_borders[i], output_target_centers[i])
        found = False
        for r0, c0 in edge_positions:
            if _overlaps(grid, patch, r0, c0):
                continue
            center_pos = (r0 + 1, c0 + 1)
            if any(abs(center_pos[0] - sp["center_pos"][0]) + abs(center_pos[1] - sp["center_pos"][1]) < 5 for sp in placed):
                continue
            _place_sprite(grid, patch, r0, c0)
            placed.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "center_pos": center_pos,
                    "border": target_borders[i],
                    "center": output_target_centers[i],
                }
            )
            found = True
            break
        if not found:
            return None
    return placed


def _place_interior_sources(
    grid: np.ndarray,
    shape_type: str,
    source_borders: list[int],
    source_centers: list[int],
    forbidden_ports: set[tuple[int, int]],
    min_port_dist: int,
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    """Place two interior sprites whose ports will become the grey trail endpoints."""
    interior_positions = [pos for pos in _sprite_positions() if not _is_edge_anchor(pos[0], pos[1])]
    random.shuffle(interior_positions)
    placed: list[dict] = []
    chosen_ports: list[tuple[int, int]] = []

    for i in range(SOURCE_COUNT):
        patch = _make_sprite_patch(shape_type, source_borders[i], source_centers[i])
        found = False
        for r0, c0 in interior_positions:
            if _overlaps(grid, patch, r0, c0):
                continue
            center_pos = (r0 + 1, c0 + 1)
            if any(abs(center_pos[0] - sp["center_pos"][0]) + abs(center_pos[1] - sp["center_pos"][1]) < 4 for sp in placed):
                continue
            _place_sprite(grid, patch, r0, c0)
            ports = [p for p in _available_ports(center_pos, source_borders[i], grid) if p not in forbidden_ports]
            if i == 1:
                ports = [p for p in ports if abs(p[0] - chosen_ports[0][0]) + abs(p[1] - chosen_ports[0][1]) >= min_port_dist]
            if not ports:
                _erase_sprite(grid, patch, r0, c0)
                continue
            chosen_port = random.choice(ports)
            placed.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "center_pos": center_pos,
                    "border": source_borders[i],
                    "center": source_centers[i],
                }
            )
            chosen_ports.append(chosen_port)
            found = True
            break
        if not found:
            return None

    return placed, chosen_ports


def _place_extra_distractor(
    grid: np.ndarray,
    shape_type: str,
    border_color: int,
    center_color: int,
) -> bool:
    """Place one extra interior non-target sprite on the input grid only."""
    patch = _make_sprite_patch(shape_type, border_color, center_color)
    interior_positions = [pos for pos in _sprite_positions() if not _is_edge_anchor(pos[0], pos[1])]
    random.shuffle(interior_positions)
    for r0, c0 in interior_positions:
        if not _overlaps(grid, patch, r0, c0):
            _place_sprite(grid, patch, r0, c0)
            return True
    return False


def _assign_colors(border_colors: list[int]) -> dict | None:
    """Assign 2 targets, 2 sources, and optionally 1 distractor without color-id ordering tricks."""
    target_borders = border_colors[:TARGET_COUNT]
    source_borders = border_colors[TARGET_COUNT:TARGET_COUNT + SOURCE_COUNT]
    distractor_borders = border_colors[TARGET_COUNT + SOURCE_COUNT:]

    mapping = [0, 1]
    random.shuffle(mapping)
    target_input_centers = [source_borders[mapping[i]] for i in range(TARGET_COUNT)]

    forbidden = {BG, GREY, *border_colors}
    source_center_pool = [c for c in range(1, 10) if c not in forbidden]
    if len(source_center_pool) < SOURCE_COUNT:
        return None
    source_centers = random.sample(source_center_pool, SOURCE_COUNT)
    output_target_centers = [source_centers[mapping[i]] for i in range(TARGET_COUNT)]

    distractor_centers = []
    used_centers = set(source_centers) | set(target_input_centers)
    for border in distractor_borders:
        candidates = [c for c in range(1, 10) if c not in {BG, GREY, border} and c not in used_centers]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        distractor_centers.append(chosen)
        used_centers.add(chosen)

    return {
        "target_borders": target_borders,
        "source_borders": source_borders,
        "distractor_borders": distractor_borders,
        "target_input_centers": target_input_centers,
        "source_centers": source_centers,
        "output_target_centers": output_target_centers,
        "distractor_centers": distractor_centers,
    }


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    extra_distractor_count = random.choices([0, 1], weights=EXTRA_DISTRACTOR_WEIGHTS)[0]
    num_sprites = TARGET_COUNT + SOURCE_COUNT + extra_distractor_count
    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, num_sprites)

    colors = _assign_colors(border_colors)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    targets = _place_edge_targets(
        output_grid,
        shape_type,
        colors["target_borders"],
        colors["output_target_centers"],
    )
    if targets is None:
        return None

    forbidden_target_ports: set[tuple[int, int]] = set()
    for target in targets:
        forbidden_target_ports.update(_possible_port_cells(target["center_pos"], target["border"], output_grid))

    planning_grid = output_grid.copy()
    source_result = _place_interior_sources(
        planning_grid,
        shape_type,
        colors["source_borders"],
        colors["source_centers"],
        forbidden_target_ports,
        min_port_dist,
    )
    if source_result is None:
        return None
    sources, port_cells = source_result

    path_cells = _gen_winding_path(planning_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if planning_grid[r, c] != BG:
            return None
        planning_grid[r, c] = GREY
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
    for i, target in enumerate(targets):
        r0, c0 = target["r0"], target["c0"]
        patch = _make_sprite_patch(shape_type, target["border"], colors["target_input_centers"][i])
        _erase_sprite(input_grid, _make_sprite_patch(shape_type, target["border"], target["center"]), r0, c0)
        _place_sprite(input_grid, patch, r0, c0)

    for source in sources:
        patch = _make_sprite_patch(shape_type, source["border"], source["center"])
        _place_sprite(input_grid, patch, source["r0"], source["c0"])

    for border, center in zip(colors["distractor_borders"], colors["distractor_centers"], strict=False):
        if not _place_extra_distractor(input_grid, shape_type, border, center):
            return None

    # Anti-degeneracy check for the D4 variant: the grey endpoints should indicate
    # two interior sprites only, so the path-based original rule would disagree.
    bg = _bg_boundary(input_grid)
    sprites = _detect_bordered_sprites(input_grid, bg, GREY)
    te1, te2 = _find_trail_endpoints(input_grid, GREY, sprites)
    eps = {te1, te2}
    endpoint_linked = [sp for sp in sprites if any(ep in _ports_for_sprite(sp, input_grid, eps) for ep in eps)]
    if len(endpoint_linked) != SOURCE_COUNT:
        return None
    if any(_touches_grid_edge(sp, input_grid.shape) for sp in endpoint_linked):
        return None

    try:
        out = main(input_grid)
        if not np.array_equal(out, output_grid):
            return None
    except Exception:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
