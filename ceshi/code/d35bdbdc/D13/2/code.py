# Variant of d35bdbdc
# Dimension: D13 — 属性转移的范围 (transform)
# Change: 仅转移中心色（单像素或小区域的颜色替换）... → 转移精灵的边框形状

"""
Design analysis for the D13 variant.

a) Concrete behavior and trace:
   - Objects are still 3x3 bordered sprites with a unique center pixel and a grey path
     that selects exactly two destination sprites through endpoint ports.
   - The lookup mechanism stays visible: each destination center color equals the border
     color of its source sprite.
   - The changed part is the payload. Original task:
       destination.center <- source.center
     Variant:
       destination.border_mask <- source.border_mask
     while keeping the destination's own border color and center color.
   - Small trace:
       input dest A = red box with teal center, source B = teal plus with yellow center.
       Original output would keep A as a box and recolor its center to yellow.
       Variant output keeps A's teal center, removes A's corner border pixels, and leaves
       a red plus because source B's shape is plus.

b) Complexity calibration:
   - Only two border masks are used: `box` and `plus`. This is enough to make transfer
     visible while keeping inference feasible from 2-4 examples.
   - Generator forces every matched destination to start with the opposite mask from its
     source, so the output cannot silently collapse to the input.
   - Lookup colors still repeat across samples through the destination centers, giving
     consistent support for the source-selection rule.

c) Hidden assumptions from the original solver that must change:
   - Original `_detect_bordered_sprites` was harmless because all sprites in one example
     shared a shape; for this variant, solver must recover each sprite's own mask.
   - Original final write `out[center] = src.center` assumed only one pixel changes.
     That would miss the new requirement because the border footprint must be cleared and
     redrawn around the destination center.
   - Original generator encoded the transformation entirely in destination center colors.
     That no longer differentiates input/output, so destination shapes must now differ
     between input and output.

d) Placement legality:
   - To avoid ambiguous partial masks, generator places all sprites fully on-grid.
   - Sprites still do not overlap or 4-touch other non-grey objects.
   - Port cells remain exactly two Manhattan steps from a destination center with a
     matching border-colored intermediate cell, so endpoint detection remains valid.

e) Visual observability:
   - The transferred attribute is now geometric rather than chromatic. Human solvers can
     see that destination centers stay the same while border footprints switch between
     box and plus according to the matched source.
   - No ordering or transformation depends on invisible color ids; colors are only used
     as direct visible lookup labels.

f) Visual cleanliness:
   - Full 3x3 sprites are clearer than clipped fragments for a shape-transfer task.
   - Only two destinations survive, so the output remains sparse and easy to parse.

g) Color-change inferability:
   - There is no color transfer in the new rule. Destination center colors remain fixed.
   - Colors only support source matching, which is already explicit in the input because
     the destination center visibly equals the source border color.

h) Generator-side guarantees for solver assumptions:
   - Every matched source border color is unique.
   - Every destination has a valid grey endpoint port.
   - Every matched destination/source pair has different input vs output shapes.
   - All sprite masks are exactly one of the two supported templates.
"""

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


CARDINAL_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SHAPE_OFFSETS = {
    "box": set(CARDINAL_OFFSETS + DIAGONAL_OFFSETS),
    "plus": set(CARDINAL_OFFSETS),
}
SHAPES = ["box", "plus"]


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
        for dr, dc in CARDINAL_OFFSETS:
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


def _infer_shape(g: np.ndarray, center_pos: tuple[int, int], border_color: int) -> str:
    """Recover whether a fully visible sprite border is a `box` or a `plus`."""
    cr, cc = center_pos
    offsets = {
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and int(g[cr + dr, cc + dc]) == border_color
    }
    if offsets == SHAPE_OFFSETS["box"]:
        return "box"
    if offsets == SHAPE_OFFSETS["plus"]:
        return "plus"
    raise AssertionError(f"unexpected sprite border mask at {center_pos}: {sorted(offsets)}")


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once (its location is the center)
      - border footprint is either a full 3x3 box ring or a 3x3 plus
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
        cr, cc = center_pos
        if not (0 < cr < g.shape[0] - 1 and 0 < cc < g.shape[1] - 1):
            continue
        shape = _infer_shape(g, center_pos, int(border_color))
        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
                "shape": shape,
            }
        )
    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint is a port for this sprite if it is 2 steps from the center and the
    intermediate cell along that direction is the sprite border color.
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


def _resolve_destinations(g: np.ndarray, bg: int, grey: int) -> tuple[list[dict], list[dict]]:
    """Detect sprites and resolve which two are selected by the grey trail endpoints."""
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
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 dest sprites from trail endpoints"
    return sprites, dest


def _redraw_destination_shape(out: np.ndarray, sprite: dict, new_shape: str, bg: int) -> None:
    """Clear the current 3x3 border footprint around a destination and draw a new one."""
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    center = int(sprite["center"])

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc2 = cr + dr, cc + dc
            if dr == 0 and dc == 0:
                continue
            out[rr, cc2] = bg
    out[cr, cc] = center
    for dr, dc in SHAPE_OFFSETS[new_shape]:
        out[cr + dr, cc + dc] = border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer border shapes between bordered sprites matched by color, keeping only path-connected ones.

    Input structure:
    - The grid contains background, a grey (5) one-stroke path, and several 3x3 bordered sprites.
    - Every sprite has a unique center pixel plus a border footprint that is either a box ring or a plus.
    - The two grey trail endpoints connect to exactly two destination sprites through visible ports.

    Rule:
    - Use the grey path endpoints to identify the two destination sprites.
    - For each destination, find the source sprite whose border color matches the destination's center color.
    - Keep the destination's colors, but replace its border footprint with the source sprite's border shape.
    - Delete all non-destination sprites; keep the grey path and the transformed destinations.

    Key insight:
    - Destination center colors still act as lookup labels into source border colors, but the payload being
      transferred is now the source sprite's geometry rather than its center color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites, dest = _resolve_destinations(g, bg, grey)
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
        _redraw_destination_shape(out, d, str(src["shape"]), bg)

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
    """Return a 3x3 sprite patch whose center is at (1,1)."""
    p = np.zeros((3, 3), dtype=int)
    for dr, dc in SHAPE_OFFSETS[shape_type]:
        p[1 + dr, 1 + dc] = border_color
    p[1, 1] = center_color
    return p


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=False)


def _erase_patch_box(grid: np.ndarray, r0: int, c0: int) -> None:
    grid[r0 : r0 + 3, c0 : c0 + 3] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
    H, W = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != 0:
                rr, cc = r0 + dr, c0 + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    return True
                if grid[rr, cc] != 0:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < H and 0 <= nc < W:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All top-left positions for fully visible 3x3 sprites."""
    return [(r0, c0) for r0 in range(GRID_H - 2) for c0 in range(GRID_W - 2)]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 == 0 or r0 == GRID_H - 3 or c0 == 0 or c0 == GRID_W - 3


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
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int,
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
                    and grid[nr, nc] == 0
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
    dirs = list(CARDINAL_OFFSETS)
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == 0:
            return (er, ec)
    return None


def _other_shape(shape_type: str) -> str:
    return "plus" if shape_type == "box" else "box"


def _assign_variant(border_colors: list[int], num_sprites: int) -> dict | None:
    """Assign sprite roles, lookup colors, and input/output shapes."""
    remaining = list(range(2, num_sprites))
    num_sources = random.choice(SOURCE_COUNTS) if len(remaining) >= 2 else max(1, len(remaining))
    if num_sources == 0:
        return None
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    if num_sources == 1:
        dest_to_source = [source_indices[0], source_indices[0]]
    else:
        dest_to_source = [source_indices[0], source_indices[1]]

    dest_center_colors = [border_colors[si] for si in dest_to_source]

    free_centers = [c for c in range(1, 10) if c != GREY and c not in border_colors]
    if not free_centers:
        return None

    sprite_center_colors: dict[int, int] = {}
    for si in source_indices + distractor_indices:
        sprite_center_colors[si] = random.choice(free_centers)

    source_shapes = {si: random.choice(SHAPES) for si in source_indices}
    dest_output_shapes = [source_shapes[si] for si in dest_to_source]
    dest_input_shapes = [_other_shape(shape) for shape in dest_output_shapes]
    distractor_shapes = {di: random.choice(SHAPES) for di in distractor_indices}

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_to_source": dest_to_source,
        "dest_center_colors": dest_center_colors,
        "sprite_center_colors": sprite_center_colors,
        "source_shapes": source_shapes,
        "dest_input_shapes": dest_input_shapes,
        "dest_output_shapes": dest_output_shapes,
        "distractor_shapes": distractor_shapes,
    }


def _place_dests(output_grid: np.ndarray, dest_specs: list[dict], min_port_dist: int):
    """Place the two output destination sprites and return their centers."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions = []
    first_port = None

    for i, spec in enumerate(dest_specs):
        patch = _make_sprite_patch(spec["shape"], spec["border"], spec["center"])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc, spec["border"])
            if port is None:
                _erase_patch_box(output_grid, r0, c0)
                continue

            if i == 1:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_patch_box(output_grid, r0, c0)
                    continue

            dest_center_positions.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break

        if not placed:
            return None

    return dest_center_positions


def _rewrite_dest_input_shape(
    input_grid: np.ndarray,
    center_pos: tuple[int, int],
    shape_type: str,
    border_color: int,
    center_color: int,
) -> None:
    """Replace an output destination patch with its input counterpart."""
    cr, cc = center_pos
    r0, c0 = cr - 1, cc - 1
    _erase_patch_box(input_grid, r0, c0)
    _place_sprite(input_grid, _make_sprite_patch(shape_type, border_color, center_color), r0, c0)


def _gen_attempt() -> np.ndarray | None:
    palette = [c for c in range(1, 10) if c != GREY]
    num_sprites = random.choice(SPRITE_COUNTS)
    border_colors = random.sample(palette, num_sprites)

    variant = _assign_variant(border_colors, num_sprites)
    if variant is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_specs = [
        {
            "shape": variant["dest_output_shapes"][i],
            "border": border_colors[i],
            "center": variant["dest_center_colors"][i],
        }
        for i in range(2)
    ]
    dest_center_positions = _place_dests(output_grid, dest_specs, min_port_dist)
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
        _rewrite_dest_input_shape(
            input_grid,
            dest_center_positions[i],
            variant["dest_input_shapes"][i],
            border_colors[i],
            variant["dest_center_colors"][i],
        )

    sprites_to_place = []
    for si in variant["source_indices"]:
        sprites_to_place.append(
            (
                variant["source_shapes"][si],
                border_colors[si],
                variant["sprite_center_colors"][si],
            )
        )
    for di in variant["distractor_indices"]:
        sprites_to_place.append(
            (
                variant["distractor_shapes"][di],
                border_colors[di],
                variant["sprite_center_colors"][di],
            )
        )

    all_pos = _sprite_positions()
    for shape_type, border_color, center_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, border_color, center_color)
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
