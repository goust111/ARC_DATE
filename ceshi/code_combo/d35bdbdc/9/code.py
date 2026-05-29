# Combo variant of d35bdbdc (index 9)
# combo_spec: D6/1+D9/0
# Name: 幽灵轮廓+紧凑裁剪

# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Base task: a grey one-stroke trail picks exactly two destination sprites by 2-step ports.
#      Each destination uses its center color as a lookup key into another sprite's border color,
#      then receives that source sprite's center color.
#    - D6 changes what happens to every non-destination sprite: it is not erased completely.
#      Instead, only its center is cleared to background and its border survives as a hollow
#      "ghost" outline.
#    - D9 changes the canvas: the final output is not the original 10x10 grid, but the tight crop
#      around the grey trail and the two connected destination sprites.
#    - Fusion trace:
#      original: unrelated sprite [border=7, center=3] -> deleted; final canvas stays 10x10.
#      combo: the same sprite -> becomes a 7-colored outline with background center, but that
#      outline is visible only if it lies inside the crop rectangle defined by path + destinations.
#      If the ghost sits outside that core rectangle, it disappears because the crop removes it.
# b) Complexity calibration
#    - The lookup chain, sprite shapes, and path topology stay unchanged, so the reasoning burden
#      remains "find two endpoint sprites, transfer colors by border-key matching".
#    - To make both dimensions observable, each generated instance has at least two non-destination
#      sprites and the generator forces at least one to remain visible inside the crop and at least
#      one to lie fully outside the crop.
#    - This avoids degeneration into pure D6 (all ghosts visible) or pure D9 (all ghosts cropped
#      away), while keeping the rule set small and repeatable.
# c) Hidden assumptions in the original main()
#    - The old deletion loop assumed every non-destination sprite must vanish; D6 invalidates that.
#    - The old crop-by-all-non-bg approach from plain D9 would be wrong here: visible ghost pixels
#      would incorrectly enlarge the crop. The crop must be computed only from grey cells and the
#      two destination sprites.
#    - The old BFS keep-mask around destination-connected pixels would also be wrong after ghosting,
#      because plus-shape ghosts split into disconnected border arms. Keeping sprite pixels by their
#      original metadata is the safe way to preserve intended ghosts while still dropping extras.
#    - Border->source lookup still depends on original input sprite metadata, so the solver must
#      build that mapping before mutating any centers.
# d) Placement legality
#    - The generator remains output-first: build the kept core (two destinations + path), lock its
#      crop bbox, then place non-destination ghosts on the full 10x10 canvas.
#    - Destination sprites and the trail are constrained to an inner frame so the crop is strictly
#      smaller than the input. Ghosts are then placed with the usual no-overlap constraints, with
#      one forced fully inside the crop bbox and one forced fully outside it.
# e) Visual observability
#    - Across samples, the solver-facing structure is stable: the same two endpoint-linked sprites
#      change centers, while unrelated sprites are hollowed out. The additional clue is that some
#      hollow sprites remain visible and some vanish with the outer padding.
#    - A human can see both mechanisms directly from 2-3 examples: the surviving crop is smaller,
#      and any non-kept sprite that still appears is always missing its center.
# f) Visual cleanliness
#    - The crop is anchored by a single core structure rather than by all colored pixels, so the
#      output stays focused. Visible ghosts never overlap the destinations or path cells.
#    - For box shapes, ghosts remain clear 3x3 rings; for plus shapes, they become separated arms,
#      which is still visually distinct because the center hole aligns with the original sprite.
# g) Change inferability
#    - No new arbitrary recoloring is introduced. Only destination centers change by the same
#      border-color lookup as in the base task. Non-destination changes are structural: center to
#      background plus crop-based retention/removal, which is inferable without hidden tie-breakers.

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, neighbors4, neighbors8, paste


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
      - center color appears exactly once
    This accepts clipped sprites near edges.
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
    """Return endpoint cells that satisfy the 2-step port rule for this sprite."""
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


def _crop_to_bbox(g: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _core_bbox(out: np.ndarray, bg: int, grey: int, dest: list[dict]) -> tuple[int, int, int, int]:
    mask = out == grey
    for sp in dest:
        for r, c in sp["pts"]:
            if out[r, c] != bg:
                mask[r, c] = True
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty kept core"
    return bbox_coords(coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer colors into the two path-linked sprites, ghost every other sprite, then crop by the connected core.

    Input structure:
    - The grid contains background, a grey (5) one-stroke trail, and several small bordered
      sprites. Each sprite has one repeated border color and one single-cell center color.
    - The two trail endpoints act as ports: each endpoint sits two steps away from the center of
      one destination sprite with that sprite's border cell in between.

    Rule:
    - Use the grey trail endpoints to identify the two destination sprites.
    - For each destination, treat its original center color as a lookup key: find the unique source
      sprite whose border color matches that key, then copy the source center color into the
      destination center.
    - Every non-destination sprite becomes a ghost by keeping its border pixels and repainting only
      its center to background. Any extra colored non-sprite debris is still cleared.
    - Finally, crop to the minimal rectangle containing only the grey trail and the two destination
      sprites. Ghost outlines inside that rectangle remain visible; ghosts outside it disappear with
      the crop.

    Key insight:
    - The path still determines which two sprites participate in the color transfer, but the rest of
      the sprites are hollowed out rather than erased. The output boundary is controlled only by the
      path-connected core, so visible ghosts are those that happen to fall inside that core crop.
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
    assert all(d is not None for d in dest), "expected exactly 2 destination sprites"
    dest_ids = {id(sp) for sp in dest}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        cr, cc = sp["center_pos"]
        out[cr, cc] = bg

    keep = np.zeros(out.shape, dtype=bool)
    for sp in sprites:
        for r, c in sp["pts"]:
            if out[r, c] != bg and out[r, c] != grey:
                keep[r, c] = True
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

    bbox = _core_bbox(out, bg, grey, dest)
    return _crop_to_bbox(out, bbox)


# --------------- generator ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400
PATH_MARGIN = 1


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus sprite centered at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


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
    """Check overlap and 4-touch against non-bg, non-grey cells."""
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
    """All top-left positions whose center stays on-grid; allows edge clipping."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _dest_positions() -> list[tuple[int, int]]:
    """Destination placements whose full 3x3 sprite and every port stay inside the inner frame."""
    positions = []
    for r0 in range(0, GRID_H - 2):
        for c0 in range(0, GRID_W - 2):
            cr, cc = r0 + 1, c0 + 1
            if PATH_MARGIN + 2 <= cr <= GRID_H - PATH_MARGIN - 3 and PATH_MARGIN + 2 <= cc <= GRID_W - PATH_MARGIN - 3:
                positions.append((r0, c0))
    return positions


def _in_path_frame(r: int, c: int) -> bool:
    return PATH_MARGIN <= r < GRID_H - PATH_MARGIN and PATH_MARGIN <= c < GRID_W - PATH_MARGIN


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
    """Generate an 8-connected grey trail constrained to the inner frame."""
    if not (_in_path_frame(*start) and _in_path_frame(*end)):
        return None

    path = [start]
    visited = {start}
    cr, cc = start

    def score(cell: tuple[int, int]) -> int:
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

    for _ in range(target_len + 30):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cr + dr, cc + dc
                if not _in_path_frame(nr, nc):
                    continue
                if (nr, nc) in visited or grid[nr, nc] != BG or _too_thick(nr, nc, visited):
                    continue
                moves.append((nr, nc, dr != 0 and dc != 0))

        if not moves:
            return None

        dist = abs(cr - end[0]) + abs(cc - end[1])
        budget = target_len - len(path)
        diag = [m for m in moves if m[2]]
        away = [m for m in moves if score((m[0], m[1])) > dist]

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
                chosen = min(pool, key=lambda m: score((m[0], m[1])))
        else:
            best = sorted(moves, key=lambda m: score((m[0], m[1])))
            top = [m for m in best if score((m[0], m[1])) <= score((best[0][0], best[0][1])) + 1]
            dtop = [m for m in top if m[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if not _in_path_frame(*end) or _too_thick(end[0], end[1], visited):
            return None
        path.append(end)

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
        if not _in_path_frame(er, ec):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == BG:
            return (er, ec)
    return None


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
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
        random.choice([c for c in range(1, 10) if c != GREY and c != border_colors[di]])
        for di in distractor_indices
    ]

    output_dest_centers = (source_center_colors * 2)[:2] if num_sources == 1 else source_center_colors
    return dict(
        source_indices=source_indices,
        distractor_indices=distractor_indices,
        dest_center_colors=dest_center_colors,
        source_center_colors=source_center_colors,
        distractor_center_colors=distractor_center_colors,
        output_dest_centers=output_dest_centers,
    )


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    pool_all = _dest_positions()
    dest_center_positions = []
    first_port = None

    for i in range(2):
        bc = border_colors[i]
        patch = _make_sprite_patch(shape_type, bc, output_dest_centers[i])
        pool = list(pool_all)
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


def _bbox_strictly_inside(bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 > 0 and c0 > 0 and r1 < GRID_H - 1 and c1 < GRID_W - 1


def _patch_cells(patch: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W:
                cells.add((rr, cc))
    return cells


def _classify_positions(
    shape_type: str, bbox: tuple[int, int, int, int]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    proto = _make_sprite_patch(shape_type, 1, BG)
    r0, c0, r1, c1 = bbox
    inside: list[tuple[int, int]] = []
    outside: list[tuple[int, int]] = []
    for pos in _sprite_positions():
        cells = _patch_cells(proto, pos[0], pos[1])
        if not cells:
            continue
        if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells):
            inside.append(pos)
        elif all(not (r0 <= r <= r1 and c0 <= c <= c1) for r, c in cells):
            outside.append(pos)
    return inside, outside


def _place_non_dest_ghosts(
    output_grid: np.ndarray,
    shape_type: str,
    sprites_to_place: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
) -> list[dict] | None:
    if len(sprites_to_place) < 2:
        return None

    inside_positions, outside_positions = _classify_positions(shape_type, bbox)
    if not inside_positions or not outside_positions:
        return None

    placed_infos: list[dict] = []
    all_positions = _sprite_positions()
    categories = ["inside", "outside"] + ["any"] * (len(sprites_to_place) - 2)

    for (border_color, center_color), category in zip(sprites_to_place, categories, strict=False):
        ghost_patch = _make_sprite_patch(shape_type, border_color, BG)
        if category == "inside":
            pool = list(inside_positions)
        elif category == "outside":
            pool = list(outside_positions)
        else:
            pool = list(all_positions)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, ghost_patch, r0, c0):
                continue
            _place_sprite(output_grid, ghost_patch, r0, c0)
            placed_infos.append(
                {
                    "border": int(border_color),
                    "center": int(center_color),
                    "center_pos": (r0 + 1, c0 + 1),
                }
            )
            placed = True
            break
        if not placed:
            return None

    return placed_infos


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

    core_output = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(core_output, shape_type, border_colors, colors["output_dest_centers"], min_port_dist)
    if result is None:
        return None
    dest_center_positions = result

    port_cells = []
    for i in range(2):
        cr, cc = dest_center_positions[i]
        port = _find_port_dir(core_output, cr, cc, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(core_output, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if core_output[r, c] != BG:
            return None
        core_output[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and core_output[nr, nc] == GREY
        )
        if deg > 1:
            return None

    core_bbox = bbox_coords(np.argwhere(core_output != BG))
    if not _bbox_strictly_inside(core_bbox):
        return None

    full_output = core_output.copy()
    sprites_to_place = (
        [(border_colors[si], colors["source_center_colors"][j]) for j, si in enumerate(colors["source_indices"])]
        + [(border_colors[di], colors["distractor_center_colors"][j]) for j, di in enumerate(colors["distractor_indices"])]
    )
    placed_non_dest = _place_non_dest_ghosts(full_output, shape_type, sprites_to_place, core_bbox)
    if placed_non_dest is None:
        return None

    expected_output = _crop_to_bbox(full_output, core_bbox)

    input_grid = full_output.copy()
    for i in range(2):
        cr, cc = dest_center_positions[i]
        input_grid[cr, cc] = colors["dest_center_colors"][i]
    for sp in placed_non_dest:
        cr, cc = sp["center_pos"]
        input_grid[cr, cc] = int(sp["center"])

    try:
        out = main(input_grid)
        if np.array_equal(out, expected_output):
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
