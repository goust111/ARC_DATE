# Combo variant of d35bdbdc (index 15)
# combo_spec: D1/2+D15/0
# Name: 反转匹配+颜色图例
#
# Design analysis for the fused variant
# ------------------------------------
# concrete_form:
#   - The playfield keeps the original bordered-sprite + grey-trail setup, but the canvas
#     grows to 12x10 so rows 0-1 can host a legend strip.
#   - Each legend column is a visible color pair: row 0 is a source-like center color key,
#     row 1 is the border color that goes with that key. The lower 10x10 playfield contains
#     the grey trail plus 4-5 bordered sprites.
# behavior_trace:
#   - Original task:
#       destination.center == source.border, then write source.center.
#       Example: dest(border=2, center=7), source(border=7, center=4) -> output center 4.
#   - Fused task:
#       source.center == destination.border, and the legend explicitly lists center->border.
#       Example: dest(border=2, center=8), source(center=2, border=4), legend has 2->4
#       -> output center 4.
#   - Difference:
#       The destination center is no longer the lookup key. The destination border is the key,
#       and the top legend externalizes the reversed pair type "center -> border".
# complexity_calibration:
#   - Exactly 2 sources are generated because the 2 destination border colors are distinct,
#     so each destination needs its own unique source center match.
#   - The legend always has 2-3 columns by mirroring all non-destination sprites. This keeps
#     the legend visually present in every sample instead of being a single incidental clue.
#   - Destination input centers are chosen from colors outside all sprite border colors, so the
#     old destination-center lookup cannot accidentally explain the outputs.
# implicit_assumption_audit:
#   - Border-based background detection breaks once the legend occupies the outer edge, so the
#     solver must use the global color mode as background.
#   - Sprite detection must exclude the legend rows; otherwise the top strip would contaminate
#     connected components and break bordered-sprite parsing.
#   - The original lookup table keyed by source.border is invalid. The fused solver instead
#     needs a center-keyed legend map and a center-keyed source consistency check.
#   - Deletion logic must preserve the legend explicitly, otherwise the standard cleanup would
#     erase it along with non-destination sprites.
# placement_legality:
#   - Rows 0-1 are reserved for the legend. Sprites and the grey path are restricted to rows
#     2-11, using the same non-overlap / non-touching constraints as the seed task.
#   - Source centers are assigned to the two destination border colors, distractor centers stay
#     unique, and destination decoy centers are chosen from the remaining non-border colors.
# visual_observability:
#   - The legend pairs repeat the same relation shown inside deleted source sprites:
#     center color on top, border color below. A human can therefore notice that the border of
#     each path-connected sprite should be read as a key into those pairs.
#   - Because destination centers are decoys rather than keys, the change from input to output
#     is visibly not explainable by the original border<-center rule.
# cleanliness_and_inferability:
#   - The legend is confined to one clean strip, while all active objects remain in the lower
#     playfield. This avoids overlapping symbolic clutter.
#   - Only a small number of color-pair examples appear per grid, but each pair is repeated by
#     a concrete sprite, giving enough structure for humans to infer the center->border mapping.

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "solvers" / "arc_utils.py").exists():
            return parent
    raise RuntimeError("could not locate repo root containing solvers/arc_utils.py")


ROOT = _repo_root()
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_mode(g: np.ndarray) -> int:
    """Most common color in the full grid."""
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[cnts.argmax()])


def _legend_info(
    g: np.ndarray, bg: int, grey: int,
) -> tuple[dict[int, int], np.ndarray]:
    """Read the 2-row top legend as a center->border map plus a reservation mask."""
    assert g.shape[0] >= 2, "legend requires at least two rows"
    cols = [
        c
        for c in range(g.shape[1])
        if int(g[0, c]) not in (bg, grey) and int(g[1, c]) not in (bg, grey)
    ]
    assert len(cols) >= 2, "expected a multi-column legend on the top edge"

    center_to_border: dict[int, int] = {}
    mask = np.zeros(g.shape, dtype=bool)
    for c in cols:
        center_color = int(g[0, c])
        border_color = int(g[1, c])
        assert center_color not in center_to_border, "legend keys must be unique"
        center_to_border[center_color] = border_color
        mask[0, c] = True
        mask[1, c] = True
    return center_to_border, mask


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
        border = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ir, ic = cr + dr, cc + dc
            er, ec = cr + 2 * dr, cc + 2 * dc
            if (er, ec) in cells and 0 <= ir < h and 0 <= ic < w and int(g[ir, ic]) == border:
                port_cells.add((er, ec))

    start = deg1[0] if deg1 else next(iter(cells))
    total = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset) -> list[tuple[int, int]] | None:
        if len(visited) == total:
            return [cur] if cur in port_cells else None
        for nb in adj[cur]:
            if nb in visited:
                continue
            result = dfs(nb, visited | {nb})
            if result is not None:
                return [cur] + result
        return None

    path = dfs(start, frozenset({start}))
    assert path, "could not trace grey cells as one-stroke trail"
    return path[0], path[-1]


def _detect_bordered_sprites(
    g: np.ndarray, bg: int, grey: int, reserved: np.ndarray,
) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey/reserved) with:
      - exactly 2 colors,
      - one singleton center color,
      - one repeated border color.
    """
    mask = (g != bg) & (g != grey) & (~reserved)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        cols = [int(g[r, c]) for r, c in pts]
        uniq = sorted(set(cols))
        if len(uniq) != 2:
            continue
        counts = {c: cols.count(c) for c in uniq}
        center_candidates = [c for c in uniq if counts[c] == 1]
        if len(center_candidates) != 1:
            continue
        center_color = center_candidates[0]
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


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, eps: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    A grey endpoint is a port for this sprite if:
      - it is exactly 2 steps from the sprite center in a cardinal direction,
      - the intermediate cell is the sprite's border color.
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


def _select_destinations(
    sprites: list[dict], g: np.ndarray, eps_pair: list[tuple[int, int]],
) -> list[dict]:
    """Resolve the two destination sprites from the trail endpoints."""
    trail_eps = set(eps_pair)
    opts = {
        ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)]
        for ep in eps_pair
    }

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    order = [0, 1] if len(opts[eps_pair[0]]) <= len(opts[eps_pair[1]]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = eps_pair[i]
        other = eps_pair[1 - i]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        assert cands, "each trail endpoint should touch at least one candidate sprite"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))

    assert all(d is not None for d in dest), "expected exactly two destination sprites"
    return dest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the two path-linked sprites by querying a center->border legend with their border colors.

    Input structure:
    - Rows 0-1 contain an edge legend: each occupied column shows a center-color key on top and the
      corresponding border color directly below it.
    - Rows 2-11 contain the original scene pattern: a grey one-stroke trail plus several bordered
      sprites, where the trail endpoints connect to exactly two destination sprites through ports.

    Rule:
    - Keep the legend, the grey trail, and the two destination sprites; delete every other sprite/component.
    - For each destination, use its border color as the lookup key. Find the legend entry whose top color
      equals that border, and write the paired bottom color into the destination center.
    - The non-destination sprites serve as concrete witnesses of the same reversed relation:
      the intended source also has center == destination border and border == written color.

    Key insight:
    - The lookup direction is reversed relative to the seed task, so the destination border is the query.
      The top legend makes that reversal explicit by showing the relevant pair type as center -> border.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_mode(g)
    grey = 5

    center_to_border, legend_mask = _legend_info(g, bg, grey)
    sprites = _detect_bordered_sprites(g, bg, grey, legend_mask)
    assert sprites, "expected at least one bordered sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    dest = _select_destinations(sprites, g, [te1, te2])
    dest_ids = {id(sp) for sp in dest}

    by_center: dict[int, list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        by_center.setdefault(int(sp["center"]), []).append(sp)

    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = legend_mask.copy()
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
    out[(out != bg) & (out != grey) & (~keep)] = bg

    for d in dest:
        want_center = int(d["border"])
        mapped_border = center_to_border.get(want_center)
        srcs = by_center.get(want_center, [])
        if srcs:
            assert len(srcs) == 1, "each destination border should match at most one source center"
            src_border = int(srcs[0]["border"])
            if mapped_border is not None:
                assert src_border == mapped_border, "legend and source sprite disagree on the mapping"
            mapped_border = src_border
        if mapped_border is None:
            continue
        dr, dc = d["center_pos"]
        out[dr, dc] = mapped_border

    return out


# --------------- generator ---------------

GRID_H, GRID_W = 12, 10
PLAY_R0 = 2
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
NUM_SOURCES = 2
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 300


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch with the center at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent background cells."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a placement by clearing every non-background patch cell."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check for overlap or 4-touching with existing non-bg, non-grey cells."""
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
    """All valid sprite anchors whose centers remain in the playfield."""
    return [
        (r0, c0)
        for r0 in range(PLAY_R0, GRID_H)
        for c0 in range(-1, GRID_W)
        if PLAY_R0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= PLAY_R0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check whether adding one grey cell would exceed the local 8-neighbor degree limit."""
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
    """Generate a winding 8-connected grey path inside the lower playfield."""
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
                if not (PLAY_R0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if (nr, nc) in visited or grid[nr, nc] != BG or _too_thick(nr, nc, visited):
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
        if PLAY_R0 <= end[0] < GRID_H and not _too_thick(end[0], end[1], visited):
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


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, border_color: int) -> tuple[int, int] | None:
    """Find a valid grey-port cell for a sprite centered at (cr,cc)."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (PLAY_R0 <= ir < GRID_H and 0 <= ic < GRID_W):
            continue
        if not (PLAY_R0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == border_color and grid[er, ec] == BG:
            return (er, ec)
    return None


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    """
    Assign exactly 2 sources so that:
      - each source center equals one destination border,
      - each destination output center becomes the matched source border,
      - all destination input centers stay outside the border-color set.
    """
    if num_sprites < 2 + NUM_SOURCES:
        return None

    remaining = list(range(2, num_sprites))
    source_indices = random.sample(remaining, NUM_SOURCES)
    distractor_indices = [idx for idx in remaining if idx not in source_indices]

    border_set = set(border_colors)
    non_border_pool = [c for c in range(1, 10) if c != GREY and c not in border_set]
    if len(non_border_pool) < 2 + len(distractor_indices):
        return None
    random.shuffle(non_border_pool)

    distractor_center_by_index: dict[int, int] = {}
    cursor = 0
    for di in distractor_indices:
        distractor_center_by_index[di] = non_border_pool[cursor]
        cursor += 1

    dest_center_colors = non_border_pool[cursor : cursor + 2]
    if len(dest_center_colors) < 2:
        return None

    shuffled_sources = source_indices[:]
    random.shuffle(shuffled_sources)
    source_center_by_index = {
        shuffled_sources[0]: border_colors[0],
        shuffled_sources[1]: border_colors[1],
    }
    output_dest_centers = [border_colors[shuffled_sources[0]], border_colors[shuffled_sources[1]]]

    legend_entries = [
        (source_center_by_index[idx], border_colors[idx]) for idx in source_indices
    ] + [
        (distractor_center_by_index[idx], border_colors[idx]) for idx in distractor_indices
    ]

    legend_keys = [key for key, _value in legend_entries]
    if len(set(legend_keys)) != len(legend_keys):
        return None

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "source_center_by_index": source_center_by_index,
        "distractor_center_by_index": distractor_center_by_index,
        "dest_center_colors": dest_center_colors,
        "output_dest_centers": output_dest_centers,
        "legend_entries": legend_entries,
    }


def _place_legend(grid: np.ndarray, legend_entries: list[tuple[int, int]]) -> None:
    """Place a contiguous 2-row legend strip on the top edge."""
    order = legend_entries[:]
    random.shuffle(order)
    width = len(order)
    c0 = random.randint(0, GRID_W - width)
    for j, (center_color, border_color) in enumerate(order):
        grid[0, c0 + j] = center_color
        grid[1, c0 + j] = border_color


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    """Place the 2 destination sprites and return their center positions."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_centers = []
    first_port = None

    for i in range(2):
        border = border_colors[i]
        patch = _make_sprite_patch(shape_type, border, output_dest_centers[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc, border)
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue
            if i == 1:
                assert first_port is not None
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue
            dest_centers.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break

        if not placed:
            return None

    return dest_centers


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

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    _place_legend(output_grid, colors["legend_entries"])

    dest_center_positions = _place_dests(
        output_grid,
        shape_type,
        border_colors,
        colors["output_dest_centers"],
        min_port_dist,
    )
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
            if PLAY_R0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc = dest_center_positions[i]
        input_grid[cr, cc] = colors["dest_center_colors"][i]

    sprites_to_place = [
        (border_colors[idx], colors["source_center_by_index"][idx]) for idx in colors["source_indices"]
    ] + [
        (border_colors[idx], colors["distractor_center_by_index"][idx])
        for idx in colors["distractor_indices"]
    ]
    all_pos = _sprite_positions()
    for border, center in sprites_to_place:
        patch = _make_sprite_patch(shape_type, border, center)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
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
