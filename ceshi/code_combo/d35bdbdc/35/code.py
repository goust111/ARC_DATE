# Combo variant of d35bdbdc (index 35)
# combo_spec: D1/3+D13/2
# Name: 形状匹配+边框形状印刻

"""
Design analysis for combo D1/3 + D13/2.

a) Concrete behavior and trace:
   - Keep the original high-level scene: a grey one-stroke trail selects exactly two
     destination sprites by endpoint ports, while other sprites are detached and will be
     deleted in the output.
   - The old task used color lookup:
       destination.center_color -> source.border_color -> write source.center_color
   - The fused task moves both the match and the payload into geometry:
       destination.cavity_shape -> source.border_shape -> redraw destination border_shape
   - Concretely, every sprite still lives in a fully visible 3x3 box with one center
     pixel. Source sprites use one of four 3-cell corner-L border masks. A destination
     sprite uses the complementary 5-cell border mask, so the background holes inside its
     3x3 ring visibly spell the source shape it wants.
   - Trace:
       source A border = {N, E, NE}
       destination B input border = ring \ {N, E, NE}
       therefore B's visible cavity = {N, E, NE}
       output redraws B's border to {N, E, NE}, keeping B's own colors.
     The same shape is first the matching key and then the transferred result.

b) Complexity calibration:
   - The shape vocabulary is intentionally small and repeated: four rotations of a 3-cell
     corner-L on the 8-neighbor ring. This is enough to make shape identity obvious while
     keeping the puzzle inferable from a few examples.
   - Exactly two detached sources are always present, one for each destination, and an
     optional distractor may use a third unmatched shape. This prevents trivial
     "everything unmatched disappears" simplification and gives actual support for the
     geometry-matching rule.
   - Destination input masks are always the ring complement of their matched source masks,
     so the input/output change is large and never collapses to a no-op.

c) Hidden assumptions from the original solver that must change:
   - Original sprite parsing used 4-connectivity and tolerated clipped masks. That breaks
     here because destination complement masks can contain corner-only pieces. The solver
     must switch to fully visible sprites and 8-connected component parsing.
   - Original lookup table keyed by source border color. That is invalid because colors are
     now irrelevant to matching; the table must be keyed by normalized border masks.
   - Original destination update touched only the center pixel. That no longer reflects the
     rule; the solver must erase the full 8-cell ring around each destination center and
     redraw the matched source mask in the destination's own border color.
   - Because cavity cells are part of the evidence, the generator must stop the grey path
     from entering any sprite's 3x3 bbox, even through background holes.

d) Placement legality:
   - All sprites are fully on-grid. Their full 3x3 boxes are reserved, so sprites never
     overlap and the grey path never occupies a cavity hole.
   - Non-destination sprites are rejected if one of the two grey endpoints would also be a
     valid port for them; this keeps endpoint ownership unambiguous.
   - 8-neighbor separation is enforced between colored sprite pixels so component parsing
     stays stable under the solver's 8-connectivity.

e) Visual observability:
   - A human can see the rule directly: the detached source has a sparse 3-cell border
     shape, while each path-selected destination has a hole with exactly that same shape.
     The output turns that hole silhouette into the surviving destination border.
   - No hidden ordering uses numeric color ids. Colors are decorative labels only; the
     decisive repeated evidence is geometric.

f) Visual cleanliness:
   - The canvas still contains only four or five small sprites plus one thin grey trail.
   - Using full 3x3 reservation avoids accidental overlap between cavities, grey cells, and
     other sprites, so the hole shapes remain easy to read.

g) Change inferability:
   - There is no color mapping to infer anymore. Centers and border colors are chosen to
     avoid accidental support for the old color-chain rule: destination center colors never
     equal any sprite border color.
   - The decisive variable is whether a destination's missing border cells match a detached
     source's occupied border cells.

h) Generator guarantees for the new solver assumptions:
   - Every destination cavity shape matches exactly one detached source border shape.
   - The two matched source shapes are distinct.
   - Every sprite mask is either one of the four source masks or one of their complements.
   - Every destination has at least one valid input-time port, and no detached sprite claims
     either trail endpoint as its own port.
"""

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8


CARDINAL_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
RING_OFFSETS = tuple(CARDINAL_OFFSETS + DIAGONAL_OFFSETS)
RING_SET = frozenset(RING_OFFSETS)

SOURCE_SHAPES = {
    "ne": frozenset({(-1, 0), (0, 1), (-1, 1)}),
    "se": frozenset({(1, 0), (0, 1), (1, 1)}),
    "sw": frozenset({(1, 0), (0, -1), (1, -1)}),
    "nw": frozenset({(-1, 0), (0, -1), (-1, -1)}),
}
SOURCE_MASKS = tuple(SOURCE_SHAPES.values())
COMPLEMENT_MASKS = tuple(frozenset(RING_SET - mask) for mask in SOURCE_MASKS)
ALLOWED_MASKS = set(SOURCE_MASKS) | set(COMPLEMENT_MASKS)

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
        for dr, dc in CARDINAL_OFFSETS:
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


def _infer_border_mask(g: np.ndarray, center_pos: tuple[int, int], border_color: int) -> frozenset[tuple[int, int]]:
    cr, cc = center_pos
    return frozenset(
        (dr, dc)
        for dr, dc in RING_OFFSETS
        if int(g[cr + dr, cc + dc]) == border_color
    )


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Sprite = an 8-connected two-color component with exactly one center pixel and an allowed
    3x3 border mask around that center. Full visibility is required because holes encode data.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=8)
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
        if border_color in (bg, grey) or center_color in (bg, grey):
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

        border_mask = _infer_border_mask(g, center_pos, int(border_color))
        if border_mask not in ALLOWED_MASKS:
            continue

        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
                "border_mask": border_mask,
                "cavity_mask": frozenset(RING_SET - border_mask),
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
    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected bordered sprites in the grid"

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
        assert cands, "expected at least one candidate destination per endpoint"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 dest sprites from trail endpoints"
    return sprites, dest


def _redraw_destination_shape(
    out: np.ndarray,
    sprite: dict,
    new_mask: frozenset[tuple[int, int]],
    bg: int,
) -> None:
    """Clear the 8-cell ring around a destination center, then draw the requested border mask."""
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    center = int(sprite["center"])

    for dr, dc in RING_OFFSETS:
        out[cr + dr, cc + dc] = bg
    out[cr, cc] = center
    for dr, dc in new_mask:
        out[cr + dr, cc + dc] = border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match path-selected destinations to detached sources by cavity/outline shape, then imprint the matched source border shape.

    Input structure:
    - The grid contains background, a grey (5) one-stroke trail, and four or five fully visible 3x3 sprites.
    - Every sprite has one unique center pixel plus a sparse border mask on the surrounding 8-cell ring.
    - The two grey trail endpoints touch exactly two destination sprites through visible ports.
    - Detached source sprites use small corner-L border masks; each destination instead shows the complementary mask, so its missing ring cells form an internal cavity shape.

    Rule:
    - Use the grey trail endpoints to identify the two destination sprites.
    - For each destination, inspect the shape of its missing ring cells and find the detached source whose border mask has exactly that shape.
    - Keep the destination's colors, but erase its current border and redraw the matched source's border mask around the same center.
    - Delete all non-destination sprites; keep the grey trail and the transformed destinations.

    Key insight:
    - Geometry does double duty: the source shape is both the lookup key (as a destination cavity silhouette) and the transferred payload (as the destination's final border shape). Colors no longer participate in matching.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites, dest = _resolve_destinations(g, bg, grey)
    dest_ids = {id(sp) for sp in dest}

    by_mask: dict[frozenset[tuple[int, int]], list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        by_mask.setdefault(sp["border_mask"], []).append(sp)

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
                if out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    for d in dest:
        srcs = by_mask.get(d["cavity_mask"], [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination cavity should match exactly one detached source shape"
        _redraw_destination_shape(out, d, srcs[0]["border_mask"], bg)

    return out


def _make_sprite_patch(
    border_mask: frozenset[tuple[int, int]],
    border_color: int,
    center_color: int,
) -> np.ndarray:
    p = np.zeros((3, 3), dtype=int)
    for dr, dc in border_mask:
        p[1 + dr, 1 + dc] = border_color
    p[1, 1] = center_color
    return p


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != BG:
                grid[r0 + dr, c0 + dc] = int(patch[dr, dc])


def _erase_patch_box(grid: np.ndarray, r0: int, c0: int) -> None:
    grid[r0 : r0 + 3, c0 : c0 + 3] = BG


def _sprite_positions() -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in range(GRID_H - 2) for c0 in range(GRID_W - 2)]


def _box_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)}


def _box_is_clear(grid: np.ndarray, r0: int, c0: int) -> bool:
    return np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == BG)


def _is_edge(r0: int, c0: int) -> bool:
    return r0 == 0 or r0 == GRID_H - 3 or c0 == 0 or c0 == GRID_W - 3


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < GRID_H and 0 <= cc < GRID_W):
                return True
            if grid[rr, cc] != BG:
                return True
            patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = rr + dr, cc + dc
                if (nr, nc) in patch_cells:
                    continue
                if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


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
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    target_len: int,
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
    dirs = list(CARDINAL_OFFSETS)
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == BG:
            return (er, ec)
    return None


def _claims_endpoint_port(
    center_pos: tuple[int, int],
    border_mask: frozenset[tuple[int, int]],
    endpoints: set[tuple[int, int]],
) -> bool:
    cr, cc = center_pos
    for dr, dc in CARDINAL_OFFSETS:
        if (dr, dc) not in border_mask:
            continue
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) in endpoints:
            return True
    return False


def _choose_variant() -> dict:
    num_sprites = random.choice(SPRITE_COUNTS)
    num_distractors = num_sprites - 4
    source_names = random.sample(list(SOURCE_SHAPES), 2)
    dest_to_source = source_names[:]
    random.shuffle(dest_to_source)
    distractor_names = random.sample(
        [name for name in SOURCE_SHAPES if name not in source_names],
        num_distractors,
    )

    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, num_sprites)
    center_pool = [c for c in palette if c not in border_colors]
    assert center_pool, "need non-border colors for centers"

    dest_center_colors = [random.choice(center_pool) for _ in range(2)]
    source_center_colors = [random.choice(center_pool) for _ in range(2)]
    distractor_center_colors = [random.choice(center_pool) for _ in range(num_distractors)]

    return {
        "border_colors": border_colors,
        "dest_specs": [
            {
                "source_name": dest_to_source[i],
                "input_mask": frozenset(RING_SET - SOURCE_SHAPES[dest_to_source[i]]),
                "output_mask": SOURCE_SHAPES[dest_to_source[i]],
                "center": dest_center_colors[i],
            }
            for i in range(2)
        ],
        "source_specs": [
            {
                "shape_name": source_names[i],
                "mask": SOURCE_SHAPES[source_names[i]],
                "center": source_center_colors[i],
            }
            for i in range(2)
        ],
        "distractor_specs": [
            {
                "shape_name": distractor_names[i],
                "mask": SOURCE_SHAPES[distractor_names[i]],
                "center": distractor_center_colors[i],
            }
            for i in range(num_distractors)
        ],
    }


def _place_destinations(
    input_grid: np.ndarray,
    dest_specs: list[dict],
    dest_border_colors: list[int],
    min_port_dist: int,
) -> tuple[list[dict], set[tuple[int, int]]] | None:
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    placed: list[dict] = []
    reserved_boxes: set[tuple[int, int]] = set()
    first_port = None

    for i, spec in enumerate(dest_specs):
        patch = _make_sprite_patch(spec["input_mask"], dest_border_colors[i], spec["center"])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        ok = False
        for r0, c0 in pool:
            box = _box_cells(r0, c0)
            if box & reserved_boxes:
                continue
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            cr, cc = r0 + 1, c0 + 1
            port = _find_port_dir(input_grid, cr, cc, dest_border_colors[i])
            if port is None:
                _erase_patch_box(input_grid, r0, c0)
                continue
            if i == 1:
                assert first_port is not None
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_patch_box(input_grid, r0, c0)
                    continue
            placed.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "center_pos": (cr, cc),
                    "port": port,
                    "border": dest_border_colors[i],
                    "center": spec["center"],
                    "input_mask": spec["input_mask"],
                    "output_mask": spec["output_mask"],
                }
            )
            reserved_boxes |= box
            if i == 0:
                first_port = port
            ok = True
            break

        if not ok:
            return None

    return placed, reserved_boxes


def _blocked_path_grid(
    input_grid: np.ndarray,
    reserved_boxes: set[tuple[int, int]],
    endpoints: list[tuple[int, int]],
) -> np.ndarray:
    blocked = input_grid.copy()
    for r, c in reserved_boxes:
        if blocked[r, c] == BG:
            blocked[r, c] = -1
    for r, c in endpoints:
        blocked[r, c] = BG
    return blocked


def _build_expected_output(
    input_grid: np.ndarray,
    dest_infos: list[dict],
    non_dest_infos: list[dict],
) -> np.ndarray:
    out = input_grid.copy()
    for info in non_dest_infos:
        _erase_patch_box(out, info["r0"], info["c0"])
    for info in dest_infos:
        _erase_patch_box(out, info["r0"], info["c0"])
        patch = _make_sprite_patch(info["output_mask"], info["border"], info["center"])
        _place_sprite(out, patch, info["r0"], info["c0"])
    return out


def _geom_metric_from_pair(inp: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    """Return (#destinations whose cavity matched the output mask, #destinations)."""
    sprites_in, dest_in = _resolve_destinations(inp, BG, GREY)
    _ = sprites_in
    sprites_out = _detect_bordered_sprites(out, BG, GREY)
    out_by_center = {sp["center_pos"]: sp for sp in sprites_out}
    hits = 0
    for d in dest_in:
        out_sp = out_by_center.get(d["center_pos"])
        if out_sp is None:
            continue
        if d["cavity_mask"] == out_sp["border_mask"]:
            hits += 1
    return hits, len(dest_in)


def _gen_attempt() -> np.ndarray | None:
    variant = _choose_variant()
    border_colors = variant["border_colors"]
    dest_border_colors = border_colors[:2]
    source_border_colors = border_colors[2:4]
    distractor_border_colors = border_colors[4:]

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    input_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    placed_dests = _place_destinations(
        input_grid,
        variant["dest_specs"],
        dest_border_colors,
        min_port_dist,
    )
    if placed_dests is None:
        return None
    dest_infos, reserved_boxes = placed_dests

    port_cells = [info["port"] for info in dest_infos]
    path_grid = _blocked_path_grid(input_grid, reserved_boxes, port_cells)
    path_cells = _gen_winding_path(path_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if (r, c) in reserved_boxes:
            return None
        if input_grid[r, c] != BG:
            return None
        input_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and input_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    non_dest_infos: list[dict] = []
    endpoint_set = set(port_cells)
    sprite_specs = []
    for i, spec in enumerate(variant["source_specs"]):
        sprite_specs.append(
            {
                "mask": spec["mask"],
                "center": spec["center"],
                "border": source_border_colors[i],
            }
        )
    for i, spec in enumerate(variant["distractor_specs"]):
        sprite_specs.append(
            {
                "mask": spec["mask"],
                "center": spec["center"],
                "border": distractor_border_colors[i],
            }
        )

    all_pos = _sprite_positions()
    for spec in sprite_specs:
        patch = _make_sprite_patch(spec["mask"], spec["border"], spec["center"])
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            box = _box_cells(r0, c0)
            if box & reserved_boxes:
                continue
            if not _box_is_clear(input_grid, r0, c0):
                continue
            if _overlaps(input_grid, patch, r0, c0):
                continue
            center_pos = (r0 + 1, c0 + 1)
            if _claims_endpoint_port(center_pos, spec["mask"], endpoint_set):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            reserved_boxes |= box
            non_dest_infos.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "center_pos": center_pos,
                    "mask": spec["mask"],
                    "border": spec["border"],
                    "center": spec["center"],
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _build_expected_output(input_grid, dest_infos, non_dest_infos)

    try:
        out = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(out, expected):
        return None

    hits, total = _geom_metric_from_pair(input_grid, out)
    if hits != total:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
