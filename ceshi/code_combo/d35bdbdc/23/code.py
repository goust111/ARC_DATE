# Combo variant of d35bdbdc (index 23)
# combo_spec: D3/0+D6/3
# Name: 单色实心精灵+未连接精灵重着灰色
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace
# - Preserve the original destination cue: the two endpoints of the grey
#   one-stroke trail still select exactly two bordered sprites via the same
#   port rule as the seed task.
# - Apply D3/0 only to sources: sources are no longer bordered sprites with an
#   inner transferable color. They are monochrome solid 2x2 blocks.
# - Because a solid source has only one visible color, the matching rule must
#   change. The only robust visible key is `destination.border`, so a source
#   matches a destination when `source.color == destination.border`.
# - After a match, the destination center is repainted with the source color.
#   Since that source color equals the destination border, the destination
#   collapses into a monochrome bordered sprite in the output.
# - Apply D6/3 to every non-destination object: bordered distractors and solid
#   sources keep their footprint but every visible pixel turns grey. The path
#   therefore stays, the two endpoint-linked bordered sprites stay colored, and
#   every unrelated object is "absorbed" into the same grey network color.
# - Small trace:
#   seed task:
#     destination border=3, destination center=8
#     source sprite border=8, source center=2
#     output destination center -> 2
#   fused task:
#     destination border=3, destination center=8
#     solid source block color=3
#     output destination center -> 3
#     the 2x2 source block itself -> all grey
#   difference:
#     the lookup chain changes from center->other border->other center into
#     border->solid color, and the used source is no longer deleted but greyed.
#
# b) Complexity calibration
# - Matching solid sources to destination.center would usually create no visible
#   change, so the lookup must target destination.border instead.
# - The generator uses 1-2 sources for 2 destinations. That yields both visible
#   branches humans can infer from a few examples:
#   - one surviving destination stays two-color, the other collapses to one color
#   - both surviving destinations collapse to one color
# - Destination input centers are chosen outside the bordered-sprite border
#   palette. This blocks accidental fallback to the seed task's center->border
#   lookup and makes the fused logic behaviorally distinct.
# - At least one solid source is placed 4-adjacent to the grey trail when legal,
#   so its greyified output visually merges with the path instead of looking
#   like an isolated recolor.
#
# c) Implicit assumption audit for main()
# - `_detect_bordered_sprites` still works because destinations and bordered
#   distractors keep the original two-color structure in the input.
# - The original border->sprite lookup is invalid for sources now; a new
#   `_detect_solid_sources` pass is required.
# - The original destination update used `d["center"]` as the lookup key. That
#   assumption breaks because solid sources have no inner color. The fused solver
#   must use `d["border"]` and write the source's single color.
# - The original cleanup loop assumed non-destination objects vanish. Under D6/3
#   that becomes wrong for both bordered distractors and solid sources; their
#   visible cells must be rewritten to grey before residual debris cleanup.
# - Greyifying sources in the output would destroy source identity, so the solver
#   must build the source-color lookup from the untouched input before editing.
#
# d) Placement legality
# - Destination and distractor bordered sprites remain 3x3 box/plus shapes and
#   may be clipped at the grid edge exactly like the seed task.
# - Solid sources are fixed 2x2 filled blocks placed fully in bounds. This keeps
#   detection unambiguous and prevents clipped 2x2 fragments from colliding with
#   bordered-sprite logic.
# - No non-grey object overlaps another non-grey object. Non-destination objects
#   may touch grey so the absorption effect is visible, but they do not overlap
#   the path and do not 4-touch other colored objects.
# - Output is built first, then input is derived by restoring destination centers
#   and recoloring grey silhouettes of sources/distractors back to their input
#   colors. This guarantees `main(input) == output`.
#
# e) Visual observability
# - A human still has a stable anchor: only the two path-endpoint bordered
#   sprites stay colored in the output.
# - Solid 2x2 blocks are the only fully filled non-grey objects in the input,
#   so they are recognizably a different source type.
# - When a solid block color matches a surviving sprite border, that destination
#   loses its contrasting center in the output. Unmatched destinations keep the
#   contrasting center. Meanwhile all unrelated objects become grey silhouettes.
# - These two coupled changes are legible from 2-4 examples without hidden
#   ordering or arithmetic on color ids.
#
# f) Visual cleanliness
# - Object count stays small (4-5 total objects including destinations), so the
#   grid does not turn into a dense grey blob after recoloring.
# - Only the non-destination objects are greyified; the two retained bordered
#   sprites stay colored and therefore remain easy to locate.
# - The generator merely prefers, not forces, multiple non-destination objects to
#   stack around the same path segment, which avoids unreadable clutter.
#
# g) Change inferability
# - Color change is highly structured: the only center recolor possible is
#   "center becomes border color if a same-colored solid source exists."
# - Grey recoloring is also structured: every object not selected by the two path
#   endpoints keeps its shape but turns into the path color.
# - The palette freedom is limited enough that repeated examples support both
#   clauses of the rule instead of creating many one-off color cases.
#
# h) New solver assumptions and generator guarantees
# - Every source component is exactly one monochrome 2x2 block.
# - Source colors are unique, so a destination border matches at most one source.
# - Destination border colors are unique, so a source can affect at most one
#   destination.
# - Destination input centers avoid all bordered-sprite border colors, so the
#   seed task's lookup cannot accidentally reproduce the fused output.
# - At least one source exists in every sample, and at least one source is
#   grey-adjacent in output, so the fusion-specific "absorbed source" behavior is
#   present rather than only latent.

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "solvers" / "arc_utils.py").exists():
            return parent
    raise RuntimeError("Could not locate repo root containing solvers/arc_utils.py")


ROOT = _find_repo_root()
for extra in [ROOT, ROOT / "solvers"]:
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.append(extra_str)

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
    cells = {(int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False)}
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        cr, cc = sp["center_pos"]
        border = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == border:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    n = len(cells)

    def dfs(
        cur: tuple[int, int], visited: frozenset[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
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
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly
    2 colors: one repeated border color plus one single-pixel center color.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        pts_list = [(int(r), int(c)) for r, c in pts]
        cols = [int(g[r, c]) for r, c in pts_list]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {color: cols.count(color) for color in uniq}
        center_color = min(counts, key=lambda color: counts[color])
        if counts[center_color] != 1:
            continue
        border_color = [color for color in uniq if color != center_color][0]
        if border_color in (bg, grey) or center_color in (bg, grey):
            continue
        center_pos = next(
            ((r, c) for r, c in pts_list if int(g[r, c]) == center_color),
            None,
        )
        if center_pos is None:
            continue
        sprites.append(
            {
                "pts": pts_list,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _detect_solid_sources(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """Detect monochrome 2x2 filled source blocks."""
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sources: list[dict] = []
    for pts in comps:
        pts_list = [(int(r), int(c)) for r, c in pts]
        colors = {int(g[r, c]) for r, c in pts_list}
        if len(colors) != 1 or len(pts_list) != 4:
            continue
        rows = [r for r, _ in pts_list]
        cols = [c for _, c in pts_list]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        full = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if set(pts_list) != full:
            continue
        sources.append({"pts": pts_list, "color": int(next(iter(colors)))})
    return sources


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, eps: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    A grey endpoint is a port for this sprite if it is exactly two cells away
    from the center in a cardinal direction and the intermediate cell is border.
    """
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in eps:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == border:
            ports.append((er, ec))
    return ports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint the endpoint-linked bordered sprites using matching solid sources, then absorb all other objects into grey.

    Input structure:
    - The grid contains background, one grey (5) winding trail, exactly two path-linked bordered sprites, 0-2 extra bordered distractors, and 1-2 monochrome solid 2x2 source blocks.
    - Only bordered sprites participate in the endpoint port rule; solid sources are separate objects that never define the path endpoints.

    Rule:
    - Use the two grey trail endpoints to identify the two destination bordered sprites.
    - For each destination, look for the unique solid source whose single color equals that destination's border color.
    - If such a source exists, repaint the destination center with that source color; then recolor every non-destination object, including all solid sources and bordered distractors, entirely to grey.

    Key insight:
    - The matching cue moved from a source sprite's border/center pair to a solid source's single color, while the cleanup stage changed from deletion to grey absorption. The only objects that keep non-grey color are the two bordered sprites selected by the path endpoints.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"
    sources = _detect_solid_sources(g, bg, grey)

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    endpoints = [te1, te2]
    options = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in endpoints}

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(options[te1]) <= len(options[te2]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = endpoints[i]
        other = endpoints[1 - i]
        candidates = [sp for sp in options[ep] if id(sp) not in used]
        best = max(
            candidates,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 dest sprites from trail endpoints"
    dest_ids = {id(sp) for sp in dest}

    by_color: dict[int, list[dict]] = {}
    for src in sources:
        by_color.setdefault(int(src["color"]), []).append(src)

    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = grey

    for src in sources:
        for r, c in src["pts"]:
            out[r, c] = grey

    keep = np.zeros(out.shape, dtype=bool)
    queue: list[tuple[int, int]] = []
    for d in dest:
        for r, c in d["pts"]:
            if out[r, c] in (bg, grey) or keep[r, c]:
                continue
            keep[r, c] = True
            queue.append((r, c))
    while queue:
        r, c = queue.pop()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]):
                continue
            if keep[nr, nc] or out[nr, nc] in (bg, grey):
                continue
            keep[nr, nc] = True
            queue.append((nr, nc))
    drop = (out != bg) & (out != grey) & (~keep)
    out[drop] = bg

    for d in dest:
        srcs = by_color.get(int(d["border"]), [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination border should match at most one solid source"
        dr, dc = d["center_pos"]
        out[dr, dc] = int(srcs[0]["color"])

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
TOTAL_OBJECT_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 500


def _make_bordered_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 patch for a bordered box or plus. Center is at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _make_solid_patch(color: int, size: int = 2) -> np.ndarray:
    return np.full((size, size), color, dtype=int)


def _greyify_patch(patch: np.ndarray) -> np.ndarray:
    """Recolor every non-background patch cell to grey, preserving the footprint."""
    return np.where(patch != BG, GREY, BG).astype(int)


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    H, W = grid.shape
    for dr in range(h):
        for dc in range(w):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < H and 0 <= cc < W and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(
    grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, allow_touch_grey: bool = True,
) -> bool:
    """Check overlap or 4-touch with existing colored objects."""
    H, W = grid.shape
    h, w = patch.shape
    patch_cells: set[tuple[int, int]] = set()
    for dr in range(h):
        for dc in range(w):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < H and 0 <= cc < W:
                if grid[rr, cc] != BG:
                    return True
                patch_cells.add((rr, cc))
    allowed_neighbors = {BG}
    if allow_touch_grey:
        allowed_neighbors.add(GREY)
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) in patch_cells or not (0 <= nr < H and 0 <= nc < W):
                continue
            if int(grid[nr, nc]) not in allowed_neighbors:
                return True
    return False


def _touches_grey(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Whether the patch would be 4-adjacent to at least one grey cell."""
    H, W = grid.shape
    h, w = patch.shape
    patch_cells: set[tuple[int, int]] = set()
    for dr in range(h):
        for dc in range(w):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < H and 0 <= cc < W:
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == GREY:
                return True
    return False


def _bordered_positions() -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _solid_positions(size: int = 2) -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in range(0, GRID_H - size + 1) for c0 in range(0, GRID_W - size + 1)]


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
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int,
) -> list[tuple[int, int]] | None:
    path = [start]
    visited = {start}
    cr, cc = start

    def score(cell: tuple[int, int, bool]) -> int:
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

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
        diag = [move for move in moves if move[2]]
        away = [move for move in moves if score(move) > dist]

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
            chosen = random.choice(pool) if random.random() < PATH_MODERATE_RANDOM_P else min(pool, key=score)
        else:
            best = sorted(moves, key=score)
            top = [move for move in best if score(move) <= score(best[0]) + 1]
            dtop = [move for move in top if move[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if _too_thick(end[0], end[1], visited):
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


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, border_color: int) -> tuple[int, int] | None:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == border_color and grid[er, ec] == BG:
            return (er, ec)
    return None


def _pick_layout_counts() -> tuple[int, int]:
    total_objects = random.choice(TOTAL_OBJECT_COUNTS)
    valid_sources = [count for count in SOURCE_COUNTS if total_objects - count >= 2]
    num_sources = random.choice(valid_sources)
    num_bordered = total_objects - num_sources
    return num_bordered, num_sources


def _assign_variant_colors(border_colors: list[int], num_sources: int) -> dict[str, list[int]] | None:
    palette = [c for c in range(1, 10) if c != GREY]
    non_border = [c for c in palette if c not in border_colors]
    if len(non_border) < 2:
        return None

    dest_input_centers = random.sample(non_border, 2)
    matched_dest_indices = sorted(random.sample([0, 1], num_sources))
    output_dest_centers = [
        border_colors[i] if i in matched_dest_indices else dest_input_centers[i]
        for i in range(2)
    ]
    source_colors = [border_colors[i] for i in matched_dest_indices]
    distractor_center_colors = [random.choice(non_border) for _ in border_colors[2:]]
    return {
        "dest_input_centers": dest_input_centers,
        "matched_dest_indices": matched_dest_indices,
        "output_dest_centers": output_dest_centers,
        "source_colors": source_colors,
        "distractor_center_colors": distractor_center_colors,
    }


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    all_pos = _bordered_positions()
    edge_pos = [pos for pos in all_pos if _is_edge(pos[0], pos[1])]
    int_pos = [pos for pos in all_pos if not _is_edge(pos[0], pos[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_centers: list[tuple[int, int]] = []
    first_port = None
    for i in range(2):
        patch = _make_bordered_patch(shape_type, border_colors[i], output_dest_centers[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_patch(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc, border_colors[i])
            if port is None:
                _erase_patch(output_grid, patch, r0, c0)
                continue
            if i == 1:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_patch(output_grid, patch, r0, c0)
                    continue
            dest_centers.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break
        if not placed:
            return None
    return dest_centers


def _try_place_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    patch: np.ndarray,
    grey_patch: np.ndarray,
    positions: list[tuple[int, int]],
    *,
    require_touch_grey: bool = False,
    prefer_touch_grey: bool = False,
) -> bool:
    touch_pool: list[tuple[int, int]] = []
    other_pool: list[tuple[int, int]] = []
    for r0, c0 in positions:
        if _overlaps(input_grid, patch, r0, c0, allow_touch_grey=True):
            continue
        if _touches_grey(input_grid, patch, r0, c0):
            touch_pool.append((r0, c0))
        else:
            other_pool.append((r0, c0))

    pools: list[list[tuple[int, int]]]
    if require_touch_grey:
        pools = [touch_pool]
    elif prefer_touch_grey:
        pools = [touch_pool, other_pool]
    else:
        pools = [touch_pool + other_pool]

    for pool in pools:
        random.shuffle(pool)
        for r0, c0 in pool:
            _place_patch(input_grid, patch, r0, c0)
            _place_patch(output_grid, grey_patch, r0, c0)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    num_bordered, num_sources = _pick_layout_counts()

    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, num_bordered)
    colors = _assign_variant_colors(border_colors, num_sources)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_centers = _place_dests(
        output_grid,
        shape_type,
        border_colors,
        colors["output_dest_centers"],
        min_port_dist,
    )
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
        input_grid[cr, cc] = colors["dest_input_centers"][i]

    all_bordered_positions = _bordered_positions()
    for border_color, center_color in zip(border_colors[2:], colors["distractor_center_colors"], strict=False):
        patch = _make_bordered_patch(shape_type, border_color, center_color)
        grey_patch = _greyify_patch(patch)
        if not _try_place_pair(
            input_grid,
            output_grid,
            patch,
            grey_patch,
            all_bordered_positions,
            prefer_touch_grey=True,
        ):
            return None

    solid_positions = _solid_positions(size=2)
    for idx, color in enumerate(colors["source_colors"]):
        patch = _make_solid_patch(color, size=2)
        grey_patch = _greyify_patch(patch)
        touched = _try_place_pair(
            input_grid,
            output_grid,
            patch,
            grey_patch,
            solid_positions,
            require_touch_grey=(idx == 0),
            prefer_touch_grey=True,
        )
        if not touched:
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
