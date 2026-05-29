# Variant of d35bdbdc
# Dimension: D4 — 目标对象选取 (dst)
# Change: 通过灰色路径端点连接的带边框精灵为目标精灵(d)... → 标记像素指示的精灵为目标
#
# Design analysis
# a) Concrete form and behavior trace
# - Keep the original 10x10 canvas, bordered 3x3 sprites, and a grey winding trail so the task still
#   looks like d35bdbdc at a glance.
# - Replace "grey endpoint chooses the destination sprite" with a symbolic cue: exactly two red
#   singleton pixels mark the two destination sprites.
# - A marker sits on a sprite's old-style port cell: two cardinal steps from the sprite center, with
#   the intervening cell equal to that sprite's border color. This makes the cue visually local and
#   easy to read.
# - Small trace:
#   original: grey endpoint -> find sprite owning that port -> that sprite is a destination.
#   variant: red singleton -> find sprite owning that marked port -> that sprite is a destination.
#   unchanged after selection: destination.center is still a lookup key into source.border, and the
#   matched source.center is copied into the destination center.
# - The grey trail remains in the scene but is generated away from all sprite-port cells, so the old
#   endpoint-based rule is visibly inapplicable on these instances.
#
# b) Complexity calibration
# - Marker detection is intentionally simple: red singletons only, exactly two of them, one per
#   destination. This is enough to make D4 visible without introducing too many marker subtypes.
# - Red is reserved for markers and excluded from sprite border/center palettes; otherwise a red sprite
#   could be confused with the marker signal.
# - Grey remains reserved for the trail, preserving the original indirect color-transfer structure.
#
# c) Hidden-assumption audit against the original main()
# - Original sprite detection ignored only bg/grey because grey port cells were excluded automatically.
#   With red markers touching sprite borders, that would merge marker+sprite into a 3-color component.
#   Fix: bordered-sprite detection must also exclude the reserved marker color.
# - Original destination selection depended on `_find_trail_endpoints` and a bipartite assignment from
#   endpoints to sprite ports. That entire dependency is invalid once the trail is no longer a selector.
#   Fix: select destinations directly from marker-owned port cells.
# - Original cleanup preserved grey and destination components while deleting everything else. That still
#   works, but marker pixels must be erased explicitly because they are not part of any sprite.
#
# d) Placement legality
# - Destination placement must guarantee at least one in-bounds marker port cell per destination.
# - Markers are placed before non-destination sprites so later placements cannot overlap or 4-touch them.
# - The grey trail is generated only on background cells and is blocked from all sprite-port cells; this
#   prevents accidental regression to the old endpoint cue.
#
# e) Visual observability
# - Across 2-4 examples, the human-visible invariant is that the only sprites surviving are the two with
#   adjacent red singleton cues, not the ones nearest the grey trail.
# - Because the trail remains but does not attach to sprite ports, the change from "connectivity cue" to
#   "symbol cue" is visible, not merely latent.
#
# f) Visual cleanliness
# - Markers are isolated single pixels, not bars or multi-pixel glyphs, so they do not clutter the small
#   10x10 board.
# - The generator keeps marker cells separated and avoids placing the trail on any valid sprite-port cell.
#
# g) Color-change inferability
# - The color-transfer rule is unchanged from the original task family: a destination's input center color
#   names the source by border color, and the source center becomes the destination output center.
# - Since D4 is the only conceptual change, keeping this mapping intact makes the variant learnable from
#   a few examples without adding a second new abstraction at the same time.
#
# h) Quantitative anti-degeneration target
# - Every generated sample must have exactly 2 marked destination sprites.
# - Every generated sample must have 0 grey path endpoints that are valid sprite ports. This is the key
#   metric proving the new control variable is the marker mechanism rather than the old endpoint rule.
#
# i) New solver assumptions and generator guarantees
# - Assumption: marker color 2 is reserved for singleton markers and does not appear inside sprites.
#   Guarantee: generator draws sprite borders/centers from {1,3,4,6,7,8,9}.
# - Assumption: each of the two destination sprites has at least one unique marker-owned port.
#   Guarantee: generator chooses destination placements together with concrete marker cells.
# - Assumption: the grey trail is present but irrelevant to destination choice.
#   Guarantee: generator blocks the trail from every sprite-port cell, so old endpoint logic has no
#   valid destination to latch onto.

import numpy as np

from arc_utils import neighbors4, neighbors8, components_mask, paste
import random
from collections import deque


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
    """Red markers are singleton 4-connected components of the marker color."""
    out: set[tuple[int, int]] = set()
    for pts in components_mask(g == marker, connectivity=4):
        if len(pts) != 1:
            continue
        r, c = pts[0]
        out.add((int(r), int(c)))
    return out


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors between bordered sprites selected by nearby marker pixels.

    Input structure:
    - Background, a grey (5) winding trail, two red singleton marker pixels (2), and several small
      bordered sprites. Each sprite has one border color and one distinct center pixel.
    - A marker indicates a destination sprite by sitting on a port cell: two cardinal steps from that
      sprite's center, with the intervening cell equal to the sprite's border color.
    - The grey trail is present as a separate preserved structure, but it no longer chooses the targets.

    Rule:
    - The two sprites indicated by the red marker pixels are the destination sprites.
    - For each destination, find the source sprite whose border color equals that destination's original
      center color, then replace the destination center with the source center color.
    - Remove the marker pixels and every non-destination sprite; keep the grey trail and the destination
      sprite bodies.

    Key insight: The red singleton markers replace the old path-endpoint cue; once the marked sprites are
    identified, the original border-color lookup and center-color transfer rule stays the same.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5
    marker = 2

    sprites = _detect_bordered_sprites(g, bg, grey, marker)
    assert sprites, "expected at least one bordered sprite in the grid"

    marker_cells = _find_marker_pixels(g, marker)
    assert len(marker_cells) == 2, "expected exactly two singleton marker pixels"

    dest = [sp for sp in sprites if _marker_ports_for_sprite(sp, g, marker_cells)]
    assert len(dest) == 2, "expected exactly two marked destination sprites"
    dest_ids = {id(sp) for sp in dest}

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


# --------------- generator ---------------

GRID_H, GRID_W = 10, 10
BG, GREY, MARKER = 0, 5, 2
SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
DEST_MIN_CENTER_DIST = 4
MARKER_MIN_DIST = 3
PATH_TRIES = 50
GEN_RETRIES = 400
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


def _marker_port_options(grid, cr, cc, bc):
    """All free in-bounds marker cells available for this sprite."""
    opts = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == BG:
            opts.append((er, ec))
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
    """Generate a winding 8-connected grey path with no thick clusters."""
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


def _assign_colors(border_colors, num_sprites):
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
        forbidden = {BG, GREY, MARKER, border_colors[si]}
        if num_sources == 1:
            forbidden.update([border_colors[0], border_colors[1]])
        else:
            forbidden.add(border_colors[j])
        candidates = [c for c in SPRITE_PALETTE if c not in forbidden]
        if not candidates:
            return None
        source_center_colors.append(random.choice(candidates))

    distractor_center_colors = []
    for di in distractor_indices:
        candidates = [c for c in SPRITE_PALETTE if c != border_colors[di]]
        if not candidates:
            return None
        distractor_center_colors.append(random.choice(candidates))

    output_dest_centers = (source_center_colors * 2)[:2] if num_sources == 1 else source_center_colors

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
        "output_dest_centers": output_dest_centers,
    }


def _place_dests(output_grid, shape_type, border_colors, output_dest_centers):
    """Place 2 destination sprites and choose one marker port for each."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_centers = []
    marker_cells = []
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
            options = _marker_port_options(output_grid, cr, cc, bc)
            if i == 1:
                options = [
                    p
                    for p in options
                    if abs(p[0] - marker_cells[0][0]) + abs(p[1] - marker_cells[0][1]) >= MARKER_MIN_DIST
                ]
            if not options:
                _erase_sprite(output_grid, patch, r0, c0)
                continue

            marker_cell = random.choice(options)
            dest_centers.append((cr, cc))
            marker_cells.append(marker_cell)
            dest_records.append({"center_pos": (cr, cc), "border": int(bc)})
            placed = True
            break

        if not placed:
            return None

    return dest_centers, marker_cells, dest_records


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


def _endpoint_candidates(grid, forbidden_ports):
    """Free cells that are not ports and are not 4-adjacent to a colored object."""
    cands = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            if grid[r, c] != BG or (r, c) in forbidden_ports:
                continue
            if any(
                0 <= nr < GRID_H and 0 <= nc < GRID_W and grid[nr, nc] != BG
                for nr, nc in neighbors4(r, c)
            ):
                continue
            cands.append((r, c))
    return cands


def _place_path(input_grid, sprite_records):
    """Generate a grey path away from every valid sprite-port cell."""
    forbidden_ports = _sprite_port_cells(sprite_records, input_grid)
    blocked = input_grid.copy()
    for r, c in forbidden_ports:
        if blocked[r, c] == BG:
            blocked[r, c] = 9

    endpoints = _endpoint_candidates(blocked, forbidden_ports)
    if len(endpoints) < 2:
        return None

    for _ in range(PATH_TRIES):
        start, end = random.sample(endpoints, 2)
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) < 4:
            continue
        target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
        path_cells = _gen_winding_path(blocked, start, end, target_len)
        if path_cells is None:
            continue
        if not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
            continue
        if not _is_8_connected(path_cells):
            continue
        if any(cell in forbidden_ports for cell in path_cells):
            continue
        return path_cells
    return None


def _gen_attempt():
    shape_type = random.choice(["box", "plus"])
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(SPRITE_PALETTE, num_sprites)

    colors = _assign_colors(border_colors, num_sprites)
    if colors is None:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(output_grid, shape_type, border_colors, colors["output_dest_centers"])
    if result is None:
        return None
    dest_center_positions, marker_cells, sprite_records = result

    input_grid = output_grid.copy()
    for i, (cr, cc) in enumerate(dest_center_positions):
        input_grid[cr, cc] = colors["dest_center_colors"][i]

    for mr, mc in marker_cells:
        if input_grid[mr, mc] != BG:
            return None
        input_grid[mr, mc] = MARKER

    sprites_to_place = (
        [(border_colors[si], colors["source_center_colors"][j]) for j, si in enumerate(colors["source_indices"])]
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

    path_cells = _place_path(input_grid, sprite_records)
    if path_cells is None:
        return None

    for r, c in path_cells:
        if input_grid[r, c] != BG or output_grid[r, c] != BG:
            return None
        input_grid[r, c] = GREY
        output_grid[r, c] = GREY

    trail_ports = _sprite_port_cells(sprite_records, input_grid)
    endpoint_port_count = sum(1 for cell in [path_cells[0], path_cells[-1]] if cell in trail_ports)
    if endpoint_port_count != 0:
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
