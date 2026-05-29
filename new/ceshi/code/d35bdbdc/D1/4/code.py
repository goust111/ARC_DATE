# Variant of d35bdbdc
# Dimension: D1 — 匹配条件：边框色与中心色的对应关系 (match_cond)
# Change: s.border_color == d.center_color（源精灵的边框色等于目标精灵的中心色）... → 尺寸排序匹配

"""
design_analysis:
  variant_goal:
    keep:
      - 10x10 canvas, grey one-stroke trail, bordered sprites, endpoint-based destination detection,
        and deletion of every sprite not attached to the grey path.
      - Sprite appearance stays in the same visual family: 3x3 box or plus, optionally clipped by the grid edge.
    change:
      - Replace the original color-chain lookup with a size-ordering lookup.
      - The two non-destination sprites are the source set. Rank sources by visible area, rank the two
        trail-connected destinations by visible area, and transfer source center colors by rank:
        largest source -> largest destination, smaller source -> smaller destination.

  a_specificization_and_trace:
    concrete_scene:
      - Every example has exactly four bordered sprites: two destinations touched by the grey trail
        endpoints and two detached sources elsewhere in the grid.
      - All sprites share one shape type within an example: either box or plus.
      - Different visible areas come only from clipping against edges/corners:
        box areas are in {4, 6, 9}; plus areas are in {3, 4, 5}.
      - The two destinations always have different areas, and the two sources also have different areas.
    traced_step:
      original_logic:
        - If a destination center was color 7, the solver searched for a source whose border color was 7.
      new_logic:
        - Suppose destination areas are 9 and 6, and source areas are 6 and 4.
        - The area-9 destination receives the center color of the area-6 source.
        - The area-6 destination receives the center color of the area-4 source.
      difference:
        - Destination input-center colors no longer encode the mapping key.
        - The mapping depends on relative size inside each role set, so the solver must detect areas and sort.

  b_complexity_calibration:
    why_not_more_sources_or_distractors:
      - Once color lookup is removed, unmarked extra bordered sprites would make the source set ambiguous.
      - Fixing the scene to exactly two sources and two destinations is the minimal adaptation that makes
        size ordering decisive while preserving the original endpoint-selection skeleton.
    visibility:
      - The area classes are coarse and countable by eye, not hidden numeric metadata.
      - Old color-chain behavior is actively disabled in generation by ensuring destination input-center
        colors are never equal to any border color.

  c_implicit_assumption_audit:
    original_dependency_points:
      - The original `by_border` table and `want_border = d["center"]` assumed a destination center color
        is a lookup key. That is invalid here and must be removed.
      - Deleting non-destination sprites before recoloring remains safe because the solver stores each
        source's original center color in the sprite metadata.
      - Destination detection by trail endpoints does not rely on the old D1 property and can remain unchanged.
    new_assumptions:
      - Exactly two non-destination bordered sprites exist, and they are the full source set.
      - Source visible areas are distinct from each other; destination visible areas are distinct from each other.
      - Generator must prevent a detached source from also forming an endpoint port, otherwise destination
        selection could become ambiguous.

  d_placement_legality:
    guarantees:
      - Candidate positions are filtered by visible area class induced by clipping.
      - `_overlaps` still prevents overlap and side-touching between colored sprite components.
      - Destinations are placed first so valid ports and a legal grey trail can be constructed.
      - Sources are placed after the trail and are rejected if they would also claim a trail endpoint as a port.

  e_visual_observability:
    human_view:
      - Across 2-4 examples, the solver can compare the larger/smaller detached sprite with the larger/smaller
        trail-connected sprite and see the center-color transfer follow area rank.
      - Because destination input-center colors are arbitrary non-border colors, color equality cannot explain
        the transformation; size is the only stable cue.

  f_cleanliness:
    layout:
      - Keep the original sparse composition: four small sprites plus one grey trail.
      - No extra colored noise components are added, since they would not support the new rule and would reduce clarity.

  g_color_change_inferability:
    palette_use:
      - The eight non-grey colors are partitioned cleanly into four border colors, two source-center colors,
        and two destination input-center colors.
      - Output colors on destinations come only from source centers, so the observed recoloring remains simple.
"""

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
                "area": int(len(pts)),
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


def _rank_by_area_desc(sprites: list[dict]) -> list[dict]:
    """Largest visible sprite first; deterministic tie-breaks keep behavior stable."""
    return sorted(
        sprites,
        key=lambda sp: (-int(sp["area"]), int(sp["center_pos"][0]), int(sp["center_pos"][1]), int(sp["border"])),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors between bordered sprites matched by visible-area rank, keeping only path-connected ones.

    Input structure:
    - A background, a grey (5) winding trail, and four bordered sprites from one shape family (box or plus).
    - Two sprites are touched by the grey trail endpoints through ports; these are the destinations.
    - The other two bordered sprites are detached sources, and visible sprite size may differ because edge clipping removes cells.

    Rule:
    - Use the two grey trail endpoints to identify the two destination sprites exactly as in the base task.
    - Delete every bordered sprite that is not a destination.
    - Rank the two destinations by visible area and rank the two detached sources by visible area, both from largest to smallest.
    - Replace each destination center with the center color of the source at the same size rank.

    Key insight: Destination selection still comes from the grey trail, but the recoloring key is no longer a color chain; it is the relative size order among the detached sources and among the trail-connected destinations.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert len(sprites) >= 4, "expected at least four bordered sprites in the grid"

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
    dest_ids = {id(sp) for sp in dest}

    sources = [sp for sp in sprites if id(sp) not in dest_ids]
    assert len(sources) == len(dest) == 2, "variant expects exactly two detached source sprites"

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

    ranked_dest = _rank_by_area_desc(dest)
    ranked_sources = _rank_by_area_desc(sources)
    for d, src in zip(ranked_dest, ranked_sources, strict=True):
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


# --------------- generator (D1 size-ordering variant) ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
TOTAL_SPRITES = 4
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
        p = np.full((3, 3), border_color, dtype=int)
    else:
        p = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p[1 + dr, 1 + dc] = border_color
    p[1, 1] = center_color
    return p


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent (0) and out-of-bounds."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a _place_sprite by zeroing non-transparent patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != 0:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
    h, w = grid.shape
    patch_cells: set[tuple[int, int]] = set()
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


def _sprite_positions() -> list[tuple[int, int]]:
    """All valid (r0,c0) for a 3x3 patch where center stays on grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _visible_area(patch: np.ndarray, r0: int, c0: int) -> int:
    area = 0
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                area += 1
    return int(area)


def _positions_by_area(shape_type: str) -> dict[int, list[tuple[int, int]]]:
    """Group all candidate top-left positions by visible clipped area."""
    probe = _make_sprite_patch(shape_type, 1, 2)
    out: dict[int, list[tuple[int, int]]] = {}
    for r0, c0 in _sprite_positions():
        area = _visible_area(probe, r0, c0)
        out.setdefault(area, []).append((r0, c0))
    return out


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


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, bc: int) -> tuple[int, int] | None:
    """Find a valid port direction for a sprite centered at (cr,cc) with border color bc."""
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


def _sprite_claims_endpoint(cr: int, cc: int, bc: int, endpoints: set[tuple[int, int]], grid: np.ndarray) -> bool:
    """Reject source placements that would also look attached to a trail endpoint."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) in endpoints and 0 <= ir < GRID_H and 0 <= ic < GRID_W and grid[ir, ic] == bc:
            return True
    return False


def _assign_variant_palette() -> dict[str, list[int]]:
    """
    Use all eight non-grey colors exactly once:
      - 4 border colors
      - 2 source-center colors
      - 2 destination input-center colors
    This guarantees the old color-chain key is absent.
    """
    palette = [c for c in range(1, 10) if c != GREY]
    shuffled = random.sample(palette, len(palette))
    return {
        "dest_borders": shuffled[:2],
        "source_borders": shuffled[2:4],
        "source_centers": shuffled[4:6],
        "dest_input_centers": shuffled[6:8],
    }


def _choose_area_specs(shape_type: str) -> tuple[list[int], list[int]]:
    area_levels = sorted(_positions_by_area(shape_type))
    dest_areas = random.sample(area_levels, 2)
    source_areas = random.sample(area_levels, 2)
    return dest_areas, source_areas


def _place_dests(
    output_grid: np.ndarray, shape_type: str, dest_specs: list[dict], min_port_dist: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place 2 destination sprites with chosen visible areas and return centers and ports."""
    positions_by_area = _positions_by_area(shape_type)
    order = sorted(range(2), key=lambda i: len(positions_by_area[int(dest_specs[i]["area"])]))
    center_positions: list[tuple[int, int] | None] = [None, None]
    port_cells: list[tuple[int, int] | None] = [None, None]

    for idx in order:
        spec = dest_specs[idx]
        patch = _make_sprite_patch(shape_type, int(spec["border"]), int(spec["output_center"]))
        pool = list(positions_by_area[int(spec["area"])])
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            _place_sprite(output_grid, patch, r0, c0)
            cr, cc = r0 + 1, c0 + 1
            port = _find_port_dir(output_grid, cr, cc, int(spec["border"]))
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue

            existing_ports = [p for p in port_cells if p is not None]
            if existing_ports:
                dist = min(abs(port[0] - p[0]) + abs(port[1] - p[1]) for p in existing_ports)
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue

            center_positions[idx] = (cr, cc)
            port_cells[idx] = port
            placed = True
            break

        if not placed:
            return None

    return center_positions, port_cells  # type: ignore[return-value]


def _place_sources(
    input_grid: np.ndarray,
    shape_type: str,
    source_specs: list[dict],
    endpoint_ports: set[tuple[int, int]],
) -> bool:
    """Place the two detached sources without letting them also attach to trail endpoints."""
    positions_by_area = _positions_by_area(shape_type)
    order = sorted(range(2), key=lambda i: len(positions_by_area[int(source_specs[i]["area"])]))

    for idx in order:
        spec = source_specs[idx]
        patch = _make_sprite_patch(shape_type, int(spec["border"]), int(spec["center"]))
        pool = list(positions_by_area[int(spec["area"])])
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            cr, cc = r0 + 1, c0 + 1
            if _sprite_claims_endpoint(cr, cc, int(spec["border"]), endpoint_ports, input_grid):
                _erase_sprite(input_grid, patch, r0, c0)
                continue
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    palette = _assign_variant_palette()
    dest_areas, source_areas = _choose_area_specs(shape_type)

    source_specs = [
        {"area": int(source_areas[i]), "border": int(palette["source_borders"][i]), "center": int(palette["source_centers"][i])}
        for i in range(2)
    ]
    dest_specs = [
        {"area": int(dest_areas[i]), "border": int(palette["dest_borders"][i]), "input_center": int(palette["dest_input_centers"][i])}
        for i in range(2)
    ]

    ranked_sources = sorted(source_specs, key=lambda sp: (-int(sp["area"]), int(sp["border"])))
    ranked_dest_idx = sorted(range(2), key=lambda i: (-int(dest_specs[i]["area"]), int(dest_specs[i]["border"])))
    for dest_idx, src in zip(ranked_dest_idx, ranked_sources, strict=True):
        dest_specs[dest_idx]["output_center"] = int(src["center"])

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    placed = _place_dests(output_grid, shape_type, dest_specs, min_port_dist)
    if placed is None:
        return None
    dest_center_positions, port_cells = placed

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
        input_grid[cr, cc] = int(dest_specs[i]["input_center"])

    if not _place_sources(input_grid, shape_type, source_specs, set(port_cells)):
        return None

    if len(_detect_bordered_sprites(input_grid, BG, GREY)) != TOTAL_SPRITES:
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
