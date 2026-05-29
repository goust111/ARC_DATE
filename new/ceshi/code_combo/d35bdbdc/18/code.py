# Combo variant of d35bdbdc (index 18)
# combo_spec: D4/1+D15/1
# Name: 标记像素指示目标+路径上颜色提示
#
# Design analysis
# a) Concrete form and behavior trace
# - Keep the original 10x10 canvas, bordered 3x3 sprites, and a thin grey trail so the puzzle still
#   reads as a d35bdbdc-family scene.
# - D4/1 changes target selection: two red singleton markers sit on old-style sprite port cells and
#   mark the only two destination sprites.
# - D15/1 changes the transferred color key: the grey trail contains two interior colored hint pixels,
#   one readable from each trail endpoint; each hint color names a source sprite by border color.
# - To make the two signals cooperate rather than duplicate each other, the trail no longer touches any
#   sprite port. Instead, each trail endpoint sits one step beyond a marker, so the local chain is:
#   destination sprite -> marker pixel -> adjacent trail endpoint -> first interior hint on that side.
# - Small trace:
#   original: grey endpoint chooses destination, destination center chooses source.
#   combo: red marker chooses destination, nearby trail end chooses which side to read, first path hint
#   from that side chooses source.
#
# b) Complexity calibration
# - Keep exactly 2 markers and exactly 2 readable path hints. This makes both new control variables
#   visible without exploding the combinatorics.
# - Marker color 2 is reserved and never used inside sprites. Hint colors are re-used source border
#   colors, so the path acts as an explicit legend instead of introducing a second encoding table.
# - The two destination input center colors are forced not to equal any sprite border color, so the old
#   destination-center lookup cannot accidentally keep working.
#
# c) Hidden-assumption audit against the original main()
# - Original sprite detection ignored only bg/grey. That would merge a red marker touching a sprite into
#   the sprite component, so marker color must be excluded during sprite detection.
# - Original destination selection relied on grey trail endpoints being sprite ports. In the combo they
#   are intentionally not ports; target ownership must be recovered from marker pixels instead.
# - Original lookup key came from destination.center. In the combo that field is decoupled from the
#   source and the key must be read from the path side associated with the marker.
# - Original cleanup removed all non-grey/non-bg non-destination pixels. The combo must preserve colored
#   hint pixels because they are part of the trail, while still removing the red markers.
#
# d) Placement legality
# - Each destination must have one outward-facing ray of length 3 from its center:
#   border at +1, marker at +2, trail endpoint at +3. This guarantees a visible marker-to-trail link.
# - The trail is generated only on background cells and is blocked from every sprite port cell, so it
#   cannot regress to the original endpoint-selection mechanism.
# - Hints are placed on interior trail cells that stay 4-separated from sprite bodies and markers.
#
# e) Visual observability
# - Across examples, the surviving sprites are the ones tagged by red pixels, not the ones nearest the
#   grey path. The path merely runs near those markers.
# - The transferred color is now visibly written on the path near each end, while destination centers no
#   longer match any source border. A human can therefore see that both cues are needed.
#
# f) Visual cleanliness
# - Markers are singleton pixels and hints are only two singleton trail cells, so the scene remains easy
#   to parse on a 10x10 grid.
# - The trail touches the rest of the scene only through its endpoints next to markers, avoiding dense
#   stacks of overlapping cues in the same cells.
#
# g) Color-change inferability
# - Hint color equals source border color directly; no hidden remapping is introduced.
# - Destination input centers are chosen from colors unused as borders, so the examples consistently
#   teach "read the path hint for the lookup key" instead of "inspect the destination center."
#
# h) Quantitative anti-degeneration targets
# - Every generated sample should have 0 grey trail endpoints that are valid sprite ports.
# - Every generated sample should have 2 destinations whose input centers do not match the hint-derived
#   source border, proving that D15 is active.
# - Every generated sample should have exactly 2 markers, each owning exactly 1 destination sprite and
#   exactly 1 adjacent trail endpoint, proving that D4 is active.
#
# i) New solver assumptions and generator guarantees
# - Assumption: marker color 2 is reserved for singleton markers. Guarantee: sprite palette excludes 2.
# - Assumption: trail hints are singleton colored cells embedded in the trail. Guarantee: each hint is an
#   interior trail cell with at least 2 grey neighbors and no 4-touching sprite pixels.
# - Assumption: each marker is adjacent to exactly one trail endpoint and marks exactly one sprite.
#   Guarantee: destination placement chooses explicit (marker, endpoint) pairs and rejects ambiguity.

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int, marker: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey/marker) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once (its location is the 'center')
    This also handles clipped/partial borders (e.g., near the grid edge).
    """
    mask = (g != bg) & (g != grey) & (g != marker)
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
        if border_color in (bg, grey, marker) or center_color in (grey, marker):
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
                "pts": [(int(r), int(c)) for r, c in pts],
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _find_marker_pixels(g: np.ndarray, marker: int) -> set[tuple[int, int]]:
    """Markers are singleton 4-connected components of the reserved marker color."""
    out: set[tuple[int, int]] = set()
    for pts in components_mask(g == marker, connectivity=4):
        if len(pts) != 1:
            continue
        r, c = pts[0]
        out.add((int(r), int(c)))
    return out


def _find_path_cells(
    g: np.ndarray, bg: int, grey: int, marker: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Path = grey cells plus isolated colored hint pixels embedded in the trail.
    Hint pixels are single-cell non-bg/non-grey/non-marker components with at least
    2 grey neighbors under 8-connectivity.
    """
    h, w = g.shape
    grey_cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    mask = (g != bg) & (g != grey) & (g != marker)
    hint_cells: set[tuple[int, int]] = set()
    for pts in components_mask(mask, connectivity=4):
        if len(pts) != 1:
            continue
        r, c = map(int, pts[0])
        grey_nbs = 0
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == grey:
                grey_nbs += 1
        if grey_nbs >= 2:
            hint_cells.add((r, c))

    path_cells = grey_cells | hint_cells
    assert path_cells, "expected a path component"
    return path_cells, hint_cells


def _trace_trail(path_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Trace the mixed grey+hint trail as one ordered endpoint-to-endpoint path."""
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in path_cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in path_cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    assert len(deg1) >= 2, "expected a trail with two endpoints"
    start, goal = deg1[0], deg1[1]
    n = len(path_cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(visited) == n:
            return [cur] if cur == goal else None
        for nb in adj[cur]:
            if nb not in visited:
                result = dfs(nb, visited | {nb})
                if result is not None:
                    return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace mixed trail"
    return path


def _marker_ports_for_sprite(
    sprite: dict,
    g: np.ndarray,
    marker_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    A marker cell (er,ec) marks this sprite iff:
      - it is exactly 2 cardinal steps away from the sprite center,
      - the intermediate cell has the sprite border color.
    """
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in marker_cells:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def _assign_marked_destinations(
    sprites: list[dict],
    marker_cells: set[tuple[int, int]],
    g: np.ndarray,
) -> dict[tuple[int, int], dict]:
    """Match each marker pixel to exactly one destination sprite."""
    markers = list(marker_cells)
    opts = {m: [sp for sp in sprites if m in _marker_ports_for_sprite(sp, g, {m})] for m in markers}

    marker_to_dest: dict[tuple[int, int], dict] = {}
    used: set[int] = set()
    order = sorted(markers, key=lambda m: len(opts[m]))
    for marker in order:
        others = [m for m in markers if m != marker]
        other = others[0] if others else marker
        cands = [sp for sp in opts[marker] if id(sp) not in used]
        assert cands, "expected each marker to own a destination sprite"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        marker_to_dest[marker] = best
        used.add(id(best))

    assert len(marker_to_dest) == 2, "expected exactly 2 marked destination sprites"
    return marker_to_dest


def _endpoint_for_marker(
    marker: tuple[int, int],
    trail_eps: set[tuple[int, int]],
) -> tuple[int, int]:
    """Each marker should be 4-adjacent to exactly one trail endpoint."""
    hits = [(nr, nc) for nr, nc in neighbors4(*marker) if (nr, nc) in trail_eps]
    assert len(hits) == 1, "expected each marker to touch exactly one trail endpoint"
    return hits[0]


def _first_hint_color(path: list[tuple[int, int]], g: np.ndarray, grey: int) -> int | None:
    """Read the first non-grey trail cell encountered along the ordered path."""
    for r, c in path:
        col = int(g[r, c])
        if col != grey:
            return col
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer source center colors into the two marker-selected targets using color hints embedded in a nearby trail.

    Input structure:
    - Background, a thin trail whose main body is grey (5), two red singleton marker pixels (2), and several small bordered sprites with one distinct center pixel.
    - A red marker indicates a destination sprite by occupying that sprite's old-style port cell: two cardinal steps from the sprite center with the intervening cell equal to the sprite border color.
    - The trail does not touch sprite ports. Instead, each trail endpoint sits adjacent to exactly one marker, and the trail also contains one colored hint pixel readable from each end.

    Rule:
    - The two sprites indicated by the red markers are the destination sprites.
    - For each destination, take the trail endpoint adjacent to its marker, walk inward along the trail, and read the first non-grey hint color; that color names the border color of the source sprite to consult.
    - Replace the destination center with that source center color, remove the markers and every non-destination sprite, and preserve the trail together with its colored hint pixels.

    Key insight: The markers answer "which sprites matter" while the path hints answer "which source color to copy"; neither signal alone is sufficient because the trail no longer selects targets and the destination centers no longer encode the lookup key.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5
    marker = 2

    sprites = _detect_bordered_sprites(g, bg, grey, marker)
    assert sprites, "expected at least one bordered sprite in the grid"

    marker_cells = _find_marker_pixels(g, marker)
    assert len(marker_cells) == 2, "expected exactly two singleton marker pixels"

    path_cells, _hint_cells = _find_path_cells(g, bg, grey, marker)
    ordered_path = _trace_trail(path_cells)
    trail_eps = {ordered_path[0], ordered_path[-1]}

    marker_to_dest = _assign_marked_destinations(sprites, marker_cells, g)
    dest = list(marker_to_dest.values())
    dest_ids = {id(sp) for sp in dest}

    endpoint_hint = {
        ordered_path[0]: _first_hint_color(ordered_path[1:-1], g, grey),
        ordered_path[-1]: _first_hint_color(list(reversed(ordered_path))[1:-1], g, grey),
    }
    assert all(col is not None for col in endpoint_hint.values()), "expected a readable hint from each trail endpoint"

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    for r, c in marker_cells:
        out[r, c] = bg

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

    path_mask = np.zeros(out.shape, dtype=bool)
    for r, c in path_cells:
        path_mask[r, c] = True
    drop = (out != bg) & (~path_mask) & (~keep)
    out[drop] = bg

    for marker_cell, d in marker_to_dest.items():
        endpoint = _endpoint_for_marker(marker_cell, trail_eps)
        want_border = int(endpoint_hint[endpoint])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each hint color should match at most one source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 10, 10
BG, GREY, MARKER = 0, 5, 2
SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_ENDPOINT_DIST_FRAC = 3
DEST_MIN_CENTER_DIST = 4
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 500
SPRITE_PALETTE = [1, 3, 4, 6, 7, 8, 9]


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
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid, patch, r0, c0):
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


def _marker_endpoint_options(grid, cr, cc, bc):
    """All free (marker_cell, trail_endpoint) rays available for this sprite."""
    opts = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        mr, mc = cr + 2 * dr, cc + 2 * dc
        er, ec = cr + 3 * dr, cc + 3 * dc
        if not (
            0 <= ir < GRID_H
            and 0 <= ic < GRID_W
            and 0 <= mr < GRID_H
            and 0 <= mc < GRID_W
            and 0 <= er < GRID_H
            and 0 <= ec < GRID_W
        ):
            continue
        if grid[ir, ic] != bc or grid[mr, mc] != BG or grid[er, ec] != BG:
            continue
        ok = True
        for nr, nc in neighbors4(er, ec):
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and (nr, nc) != (mr, mc):
                if grid[nr, nc] != BG:
                    ok = False
                    break
        if ok:
            opts.append(((mr, mc), (er, ec)))
    return opts


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
    """Generate a winding 8-connected grey path."""
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
        if not _too_thick(end[0], end[1], visited) and grid[end[0], end[1]] == BG:
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


def _choose_color(candidates, used):
    fresh = [c for c in candidates if c not in used]
    pool = fresh if fresh else list(candidates)
    return random.choice(pool) if pool else None


def _assign_colors(border_colors, num_sprites):
    """Assign roles and colors for the fused marker+path-hint variant."""
    remaining = list(range(2, num_sprites))
    random.shuffle(remaining)
    num_sources = random.choice(SOURCE_COUNTS) if len(remaining) >= 2 else max(1, len(remaining))
    if num_sources == 0:
        return None

    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    dest_center_pool = [c for c in SPRITE_PALETTE if c not in border_colors]
    if len(dest_center_pool) < 2:
        return None
    random.shuffle(dest_center_pool)
    dest_center_colors = dest_center_pool[:2]

    if num_sources == 1:
        dest_to_source = [source_indices[0], source_indices[0]]
    else:
        dest_to_source = random.sample(source_indices, 2)

    used_source_centers: set[int] = set()
    source_center_by_index: dict[int, int] = {}
    for si in source_indices:
        forbidden = {border_colors[si], *dest_center_colors}
        candidates = [c for c in SPRITE_PALETTE if c not in forbidden]
        col = _choose_color(candidates, used_source_centers)
        if col is None:
            return None
        source_center_by_index[si] = int(col)
        used_source_centers.add(int(col))

    hint_border_colors = [border_colors[si] for si in dest_to_source]
    output_dest_centers = [source_center_by_index[si] for si in dest_to_source]

    distractor_center_colors = []
    for di in distractor_indices:
        candidates = [c for c in SPRITE_PALETTE if c != border_colors[di]]
        col = _choose_color(candidates, set())
        if col is None:
            return None
        distractor_center_colors.append(int(col))

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "hint_border_colors": hint_border_colors,
        "source_center_by_index": source_center_by_index,
        "output_dest_centers": output_dest_centers,
        "distractor_center_colors": distractor_center_colors,
    }


def _place_dests(output_grid, shape_type, border_colors, output_dest_centers, min_endpoint_dist):
    """Place 2 destination sprites and choose one (marker, endpoint) ray for each."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_centers = []
    marker_cells = []
    endpoint_cells = []
    dest_records = []

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
            if dest_centers:
                prev_r, prev_c = dest_centers[0]
                if abs(cr - prev_r) + abs(cc - prev_c) < DEST_MIN_CENTER_DIST:
                    continue

            _place_sprite(output_grid, patch, r0, c0)
            options = _marker_endpoint_options(output_grid, cr, cc, bc)
            if i == 1:
                options = [
                    (m, e)
                    for m, e in options
                    if abs(e[0] - endpoint_cells[0][0]) + abs(e[1] - endpoint_cells[0][1]) >= min_endpoint_dist
                ]
            if not options:
                _erase_sprite(output_grid, patch, r0, c0)
                continue

            marker_cell, endpoint_cell = random.choice(options)
            dest_centers.append((cr, cc))
            marker_cells.append(marker_cell)
            endpoint_cells.append(endpoint_cell)
            dest_records.append({"center_pos": (cr, cc), "border": int(bc)})
            placed = True
            break

        if not placed:
            return None

    return dest_centers, marker_cells, endpoint_cells, dest_records


def _sprite_port_cells(sprite_records, grid):
    """All old-style port cells for all placed sprites."""
    ports = set()
    for sp in sprite_records:
        cr, cc = sp["center_pos"]
        bc = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
                continue
            if int(grid[ir, ic]) == bc:
                ports.add((er, ec))
    return ports


def _trail_endpoint_neighbors(marker_cells, endpoint_cells):
    pairs = {}
    for marker_cell in marker_cells:
        hits = [ep for ep in endpoint_cells if abs(ep[0] - marker_cell[0]) + abs(ep[1] - marker_cell[1]) == 1]
        if len(hits) != 1:
            return None
        pairs[marker_cell] = hits[0]
    return pairs


def _hint_legal(cell, grid, path_set):
    """Hint cells must stay visually separate from non-path colored objects."""
    r, c = cell
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
            if (nr, nc) in path_set:
                continue
            if int(grid[nr, nc]) != BG:
                return False
    return True


def _choose_hint_indices(path_cells, grid):
    """Pick one safe interior path cell near each endpoint for the two instruction pixels."""
    n = len(path_cells)
    if n < 7:
        return None
    path_set = set(path_cells)

    left_choices = [i for i in range(2, min(5, n - 2)) if _hint_legal(path_cells[i], grid, path_set)]
    if not left_choices:
        return None
    left_idx = random.choice(left_choices)

    right_choices = [
        i
        for i in range(max(n - 5, left_idx + 3), n - 2)
        if _hint_legal(path_cells[i], grid, path_set)
    ]
    if not right_choices:
        return None
    right_idx = random.choice(right_choices)
    return left_idx, right_idx


def _build_blocked_grid(input_grid, sprite_records, marker_cells, endpoint_cells):
    """Block sprite ports and all marker-adjacent cells except the chosen trail endpoints."""
    blocked = input_grid.copy()

    for r, c in _sprite_port_cells(sprite_records, input_grid):
        if blocked[r, c] == BG:
            blocked[r, c] = 9

    marker_to_endpoint = _trail_endpoint_neighbors(marker_cells, endpoint_cells)
    if marker_to_endpoint is None:
        return None

    for marker_cell, endpoint_cell in marker_to_endpoint.items():
        for nr, nc in neighbors4(*marker_cell):
            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                continue
            if (nr, nc) == endpoint_cell:
                continue
            if blocked[nr, nc] == BG:
                blocked[nr, nc] = 9

    return blocked


def _gen_attempt():
    shape_type = random.choice(["box", "plus"])
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(SPRITE_PALETTE, num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_endpoint_dist = max(3, target_len // MIN_ENDPOINT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(output_grid, shape_type, border_colors, colors["output_dest_centers"], min_endpoint_dist)
    if result is None:
        return None
    dest_center_positions, marker_cells, endpoint_cells, sprite_records = result

    input_grid = output_grid.copy()
    for i, (cr, cc) in enumerate(dest_center_positions):
        input_grid[cr, cc] = colors["dest_center_colors"][i]
    for mr, mc in marker_cells:
        if input_grid[mr, mc] != BG:
            return None
        input_grid[mr, mc] = MARKER

    sprites_to_place = (
        [
            (border_colors[si], colors["source_center_by_index"][si])
            for si in colors["source_indices"]
        ]
        + [
            (border_colors[di], colors["distractor_center_colors"][j])
            for j, di in enumerate(colors["distractor_indices"])
        ]
    )

    all_pos = _sprite_positions()
    for bc, cc_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, bc, cc_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            sprite_records.append({"center_pos": (r0 + 1, c0 + 1), "border": int(bc)})
            placed = True
            break
        if not placed:
            return None

    blocked = _build_blocked_grid(input_grid, sprite_records, marker_cells, endpoint_cells)
    if blocked is None:
        return None

    path_cells = _gen_winding_path(blocked, endpoint_cells[0], endpoint_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    hint_indices = _choose_hint_indices(path_cells, input_grid)
    if hint_indices is None:
        return None

    path_colors = {cell: GREY for cell in path_cells}
    path_colors[path_cells[hint_indices[0]]] = colors["hint_border_colors"][0]
    path_colors[path_cells[hint_indices[1]]] = colors["hint_border_colors"][1]

    for (r, c), col in path_colors.items():
        if input_grid[r, c] != BG or output_grid[r, c] != BG:
            return None
        input_grid[r, c] = col
        output_grid[r, c] = col

    marker_to_endpoint = _trail_endpoint_neighbors(marker_cells, endpoint_cells)
    if marker_to_endpoint is None:
        return None

    trail_ports = _sprite_port_cells(sprite_records, input_grid)
    grey_endpoint_port_count = sum(1 for cell in [path_cells[0], path_cells[-1]] if cell in trail_ports)
    if grey_endpoint_port_count != 0:
        return None

    for marker_cell in marker_cells:
        owners = 0
        for sp in _detect_bordered_sprites(input_grid, BG, GREY, MARKER):
            if marker_cell in _marker_ports_for_sprite(sp, input_grid, {marker_cell}):
                owners += 1
        if owners != 1:
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
