# Combo variant of d35bdbdc (index 24)
# combo_spec: D5/3+D13/2
# Name: 混色棋盘效果+边框形状印刻

"""
Design analysis for the combo variant.

a) Concrete behavior and trace:
   - The input still contains a grey one-stroke trail plus several fully visible 3x3
     bordered sprites. Each sprite has one border color and one center color.
   - The two grey trail endpoints still select the two destinations through the original
     2-step port rule, and each destination center color still names a source sprite by
     matching that source's border color.
   - Original task payload:
       destination.center <- source.center
   - Combo payload:
       1. destination.border_mask <- source.border_mask
       2. destination.inner_patch <- mix(destination.center, source.center)
     Concretely, the whole 3x3 destination patch is repainted. First a two-color
     checker-like mix is drawn from the old destination center and the matched source
     center; then the source border mask is stamped in the destination's own border
     color on top of it.
   - For the matched `plus` shape used by the generator, the output patch becomes:
         S B D
         B S B
         D B S
     where B = destination border color, S = source center color, D = destination's
     original center color. This makes both the mixed center and the shape imprint
     visible at once.

b) Complexity calibration:
   - The solver supports both `box` and `plus` source masks, but the generator forces
     matched sources to be `plus` and destination inputs to be `box`.
   - This is deliberate: with a transferred `box` ring on a 3x3 sprite, only the single
     center cell would remain available for mixing, so the D5/3 effect would visually
     collapse. Restricting matched sources to `plus` preserves a real five-cell mixed
     region without changing the original lookup logic.
   - Distractors can still be either `box` or `plus`, so the shape vocabulary remains
     present in the examples and the transferred mask is not reduced to an invisible
     constant inside the solver.

c) Hidden assumptions reviewed:
   - Original `_detect_bordered_sprites` tolerated clipped sprites because only one
     center pixel changed. That is unsafe here because the output repaint depends on a
     precise local mask. The combo generator therefore uses only fully visible sprites,
     and the solver explicitly infers each input sprite shape from a full 3x3 footprint.
   - Original final write touched one center cell. That assumption is invalid now:
     the combo output rewrites the entire 3x3 neighborhood around each destination.
   - Original generator encoded all visible change in destination centers. That would
     under-specify the combo, so the generator must also force destination input shapes
     to differ from the transferred source shape.

d) Placement legality:
   - All sprites are fully on-grid 3x3 patches.
   - `_overlaps()` still forbids overlap and 4-neighbor touching with other non-grey
     objects, keeping component detection stable.
   - Destination patches in the output still expose border-colored cardinal cells, so
     the same port finder can place a valid grey path before the input sprites are
     restored.

e) Visual observability:
   - A human sees the same color-lookup cue as the base task: each connected target's
     center color points to another sprite's border color.
   - The output then shows two simultaneous consequences of following that cue:
     the target stops being a box and gains the source's plus border shape, and its
     3x3 interior becomes a two-color pattern using both the source center color and
     the target's original center color.
   - No step depends on invisible color ids or hidden ordering.

f) Visual cleanliness:
   - Only the two path-selected destinations survive.
   - The mixed patch is confined to a single 3x3 area per destination, so the output
     stays sparse and readable on a 10x10 grid.

g) Color-change inferability:
   - Border colors remain unique.
   - Source center colors are drawn from colors that are not used as any border color,
     making the imported color visually attributable to the matched source rather than
     to another border token.
   - The destination's original center color is also preserved inside the mixed patch,
     which makes the "blend rather than replace" behavior recoverable from examples.

h) Fusion-specific anti-degeneration guarantee:
   - The solver is generic, but the generator rejects samples unless each transformed
     destination differs from the original one in both shape and center-region pattern.
   - Because matched sources are `plus`, each output destination necessarily contains
     two surviving destination-color corner cells and three source-color mixed cells.

i) New solver assumptions that the generator enforces:
   - Every input sprite is a full 3x3 `box` or `plus`.
   - Every destination center color matches exactly one source border color.
   - Every matched source uses the `plus` mask, so the combo effect is never reduced to
     the original single-center replacement.
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

    def dfs(cur: tuple[int, int], visited: frozenset) -> list[tuple[int, int]] | None:
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
    Bordered sprite = a full 3x3 4-connected component with exactly 2 colors:
      - one repeated border color,
      - one unique center color,
      - border mask is either `box` or `plus`.
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
        if border_color in (bg, grey) or center_color == grey:
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
        if len(pts) not in (5, 9):
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


def _mix_color(dr: int, dc: int, dest_color: int, src_color: int) -> int:
    """
    Diagonal checker-like mix on a 3x3 patch.

    For a transferred `plus` mask, this leaves:
      src at center + NW + SE
      dest at NE + SW
    """
    return int(src_color if dr == dc else dest_color)


def _redraw_combo_destination(
    out: np.ndarray, sprite: dict, source_shape: str, src_center: int,
) -> None:
    """Repaint a destination with a mixed inner patch plus the source border shape."""
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    dest_center = int(sprite["center"])

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out[cr + dr, cc + dc] = _mix_color(dr, dc, dest_center, int(src_center))
    for dr, dc in SHAPE_OFFSETS[source_shape]:
        out[cr + dr, cc + dc] = border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint path-selected destinations with a mixed center patch and the source border shape.

    Input structure:
    - The grid contains background, a grey (5) one-stroke trail, and several fully visible
      3x3 bordered sprites whose masks are either `box` or `plus`.
    - The trail endpoints identify exactly two destination sprites through 2-step ports.
    - Each destination center color still names a source sprite by matching that source's
      border color.

    Rule:
    - Use the grey path endpoints to identify the two destination sprites.
    - For each destination, find the source sprite whose border color equals the
      destination's original center color.
    - Replace the whole 3x3 destination patch: paint a two-color checker-like mix from the
      destination center color and the source center color, then stamp the source border
      shape on top using the destination's own border color.
    - Delete all non-destination sprites and any stray non-grey foreground; keep the grey
      path and the transformed destinations.

    Key insight:
    - The destination center is still a lookup token into another sprite's border color,
      but the payload is now fused: the matched source contributes both its center color
      and its border geometry, while the destination keeps its own border color as the ink
      for the imprinted shape.
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
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
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
        _redraw_combo_destination(out, d, str(src["shape"]), int(src["center"]))

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
GEN_RETRIES = 400
DEST_INPUT_SHAPE = "box"
MATCHED_SOURCE_SHAPE = "plus"


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 sprite patch whose center is at (1,1)."""
    p = np.zeros((3, 3), dtype=int)
    for dr, dc in SHAPE_OFFSETS[shape_type]:
        p[1 + dr, 1 + dc] = border_color
    p[1, 1] = center_color
    return p


def _make_combo_patch(
    border_color: int, dest_color: int, src_color: int, source_shape: str,
) -> np.ndarray:
    """Return the transformed destination patch used in the output."""
    p = np.zeros((3, 3), dtype=int)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            p[1 + dr, 1 + dc] = _mix_color(dr, dc, int(dest_color), int(src_color))
    for dr, dc in SHAPE_OFFSETS[source_shape]:
        p[1 + dr, 1 + dc] = int(border_color)
    return p


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=False)


def _erase_patch_box(grid: np.ndarray, r0: int, c0: int) -> None:
    grid[r0 : r0 + 3, c0 : c0 + 3] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == 0:
                continue
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return True
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
    """Generate a winding 8-connected grey path."""
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
    """Find a valid port direction for a sprite-like patch centered at (cr,cc)."""
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


def _assign_variant(border_colors: list[int], num_sprites: int) -> dict | None:
    """Assign source/distractor roles, lookup colors, and input/output shapes."""
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
    need_unique = len(source_indices)
    if len(free_centers) < need_unique:
        return None
    source_centers = random.sample(free_centers, k=need_unique)

    sprite_center_colors: dict[int, int] = {}
    for si, center_color in zip(source_indices, source_centers, strict=False):
        sprite_center_colors[si] = int(center_color)
    for di in distractor_indices:
        choices = [c for c in range(1, 10) if c != GREY and c != border_colors[di]]
        sprite_center_colors[di] = int(random.choice(choices))

    source_shapes = {si: MATCHED_SOURCE_SHAPE for si in source_indices}
    dest_input_shapes = [DEST_INPUT_SHAPE, DEST_INPUT_SHAPE]
    distractor_shapes = {di: random.choice(SHAPES) for di in distractor_indices}

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_to_source": dest_to_source,
        "dest_center_colors": dest_center_colors,
        "sprite_center_colors": sprite_center_colors,
        "source_shapes": source_shapes,
        "dest_input_shapes": dest_input_shapes,
        "distractor_shapes": distractor_shapes,
    }


def _place_dests(output_grid: np.ndarray, dest_specs: list[dict], min_port_dist: int):
    """Place the two output destination patches and return their centers."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions = []
    first_port = None

    for i, spec in enumerate(dest_specs):
        patch = _make_combo_patch(spec["border"], spec["dest_color"], spec["src_color"], spec["source_shape"])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_patch(output_grid, patch, r0, c0)
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


def _rewrite_dest_input(
    input_grid: np.ndarray,
    center_pos: tuple[int, int],
    shape_type: str,
    border_color: int,
    center_color: int,
) -> None:
    """Replace an output destination patch with its input bordered sprite."""
    cr, cc = center_pos
    r0, c0 = cr - 1, cc - 1
    _erase_patch_box(input_grid, r0, c0)
    _place_patch(input_grid, _make_sprite_patch(shape_type, border_color, center_color), r0, c0)


def _combo_signature(inp: np.ndarray, out: np.ndarray) -> tuple[int, int]:
    """
    Return:
      - number of destinations whose shape changed from box to plus-like output
      - total count of surviving destination-color corner cells in the output
    """
    bg = _bg_boundary(inp)
    sprites, dest = _resolve_destinations(inp, bg, GREY)
    _ = sprites
    shape_changes = 0
    preserved_corners = 0
    for d in dest:
        cr, cc = d["center_pos"]
        in_cardinals = sum(
            int(inp[cr + dr, cc + dc] == d["border"]) for dr, dc in CARDINAL_OFFSETS
        )
        out_cardinals = sum(
            int(out[cr + dr, cc + dc] == d["border"]) for dr, dc in CARDINAL_OFFSETS
        )
        out_corners = [int(out[cr + dr, cc + dc]) for dr, dc in DIAGONAL_OFFSETS]
        if d["shape"] == "box" and in_cardinals == 4 and out_cardinals == 4 and all(
            color != d["border"] for color in out_corners
        ):
            shape_changes += 1
        preserved_corners += sum(color == int(d["center"]) for color in out_corners)
    return shape_changes, preserved_corners


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
    dest_specs = []
    for i in range(2):
        src_idx = variant["dest_to_source"][i]
        dest_specs.append(
            {
                "border": border_colors[i],
                "dest_color": variant["dest_center_colors"][i],
                "src_color": variant["sprite_center_colors"][src_idx],
                "source_shape": variant["source_shapes"][src_idx],
            }
        )
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
        _rewrite_dest_input(
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
                _place_patch(input_grid, patch, r0, c0)
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(input_grid)
        if not np.array_equal(out, output_grid):
            return None
        shape_changes, preserved_corners = _combo_signature(input_grid, out)
        if shape_changes != 2 or preserved_corners < 4:
            return None
        return input_grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
