# Combo variant of d35bdbdc (index 11)
# combo_spec: D10/0+D14/0
# Name: 一对一配对+未匹配目标保留

import random
from collections import deque
from itertools import combinations

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste

# Design analysis for the combo variant
# -------------------------------------
# fused_spec:
#   concretization:
#     - Keep the original visual vocabulary: a 10x10 grid, grey one-stroke trail, and
#       tiny 3x3 bordered sprites with one special center pixel.
#     - Collapse the role count to exactly 2 sprites total:
#         1. one target sprite touched by both grey trail endpoints through valid ports
#         2. one non-connected source sprite elsewhere
#     - Output keeps only the target plus the grey trail. The source disappears after
#       supplying a possible color transfer.
#   behavior_trace:
#     - Original task: the two trail endpoints pick two surviving destination sprites;
#       each destination center is used as a lookup key into some other sprite border.
#     - Here both endpoints point to the same target sprite, so there is only one
#       surviving target and one candidate source.
#     - Concrete matched example:
#         target border=2 center=7, source border=7 center=3
#         output keeps the target and changes its center 7->3.
#     - Concrete unmatched example:
#         target border=2 center=8, source border=7 center=3
#         output keeps the target but leaves its center at 8 because 8 != 7.
#     - The D10 change removes the many-to-many search; the D14 change makes the
#       "do nothing" branch visible when the only source does not match.
#   complexity_calibration:
#     - The scene is simplified to 2 sprites, but not to a trivial free choice:
#       the solver still must distinguish the path-selected target from the deleted source,
#       then compare target.center against source.border.
#     - Matched and unmatched cases are both generated, so the no-change branch is
#       observable across a small set of examples instead of hiding behind luck.
#     - Colors stay low-cardinality and fully visible; no ordering depends on numeric
#       color ids or other non-visual metadata.
#   hidden_assumption_review:
#     - Original `main()` assumed there were exactly 2 destination sprites, one per
#       trail endpoint. That breaks here because both endpoints belong to the same target.
#     - Original border lookup across all sprites is also too broad for the fused rule:
#       the only legal source is the single non-target sprite.
#     - New solver assumptions:
#         * exactly 2 bordered sprites exist
#         * exactly 1 sprite has both trail endpoints as valid ports
#         * the other sprite is the only candidate source
#       The generator explicitly constructs and validates all three.
#   placement_legality:
#     - Generator first places the output target, then chooses two of its valid port
#       cells and draws a grey path between them.
#     - The source is placed afterward at a location that does not overlap the path and
#       does not expose any valid port onto any grey cell, so only the target is path-connected.
#     - Oracle validation (`main(input) == output`) rejects any accidental ambiguity.
#   observability_and_neatness:
#     - Human-facing cue: the whole grey trail clearly hooks into one surviving sprite,
#       while the other sprite disappears. The only question is whether its border
#       color matches the survivor's original center.
#     - Visual clutter is low because there are only 2 sprites, one deletion, and at
#       most one center-pixel recolor in the output.


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
    assert cells, "expected a grey trail"

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
      - center color appears exactly once (its location is the 'center')
    This also handles clipped/partial borders near the grid edge.
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
        sprites.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, path_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey cell is a valid port for this sprite if:
      - it is exactly 2 steps away from the sprite center in a cardinal direction
      - the intermediate cell is the sprite border color
    """
    cr, cc = sprite["center_pos"]
    b = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for ir, ic in neighbors4(cr, cc):
        dr, dc = ir - cr, ic - cc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if (er, ec) not in path_cells:
            continue
        if 0 <= ir < g.shape[0] and 0 <= ic < g.shape[1] and int(g[ir, ic]) == b:
            ports.append((er, ec))
    return ports


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the single grey-selected target sprite, optionally recoloring its center from the only source sprite.

    Input structure:
    - The grid contains background, a grey one-stroke trail, and exactly two bordered sprites of the usual small type.
    - Both trail endpoints are valid ports of the same target sprite; the other sprite is not connected to the trail.

    Rule:
    - Identify the unique sprite touched by both trail endpoints and delete every other non-grey object.
    - Treat the remaining non-target sprite as the only source candidate.
    - If source.border equals the target's original center color, replace the target center with source.center; otherwise leave the target unchanged.

    Key insight:
    - The grey path no longer selects two separate survivors. It singles out one target, and the whole task reduces to testing whether the only deleted sprite's border color matches that target's center color before deletion.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert len(sprites) == 2, "expected exactly 2 bordered sprites"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}

    endpoint_ports = {id(sp): _ports_for_sprite(sp, g, trail_eps) for sp in sprites}
    dests = [sp for sp in sprites if len(endpoint_ports[id(sp)]) == 2]
    assert len(dests) == 1, "expected exactly one sprite connected to both trail endpoints"
    dest = dests[0]

    sources = [sp for sp in sprites if id(sp) != id(dest)]
    assert len(sources) == 1, "expected exactly one non-target source sprite"
    src = sources[0]

    out = g.copy()
    keep = {(int(r), int(c)) for r, c in dest["pts"]}
    drop = (out != bg) & (out != grey)
    for r, c in keep:
        drop[r, c] = False
    out[drop] = bg

    if int(src["border"]) == int(dest["center"]):
        dr, dc = dest["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
TARGET_EDGE_WEIGHT = 4
TARGET_INTERIOR_WEIGHT = 1
MATCH_WEIGHTS = [1, 1]
GEN_RETRIES = 600


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch with center at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent zeros and clipping at borders."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a placement by clearing every non-transparent patch cell back to background."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check overlap or 4-touch with existing non-bg, non-grey content."""
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
    """All valid top-left offsets for a 3x3 patch whose center stays on-grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Reject cells that would create a fat 8-connected cluster."""
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
    """Generate a winding 8-connected grey path from start to end."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(cell: tuple[int, int]) -> int:
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
        if not _too_thick(end[0], end[1], visited):
            path.append(end)
        else:
            return None

    return path


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
    """Check 8-connectivity of a cell set."""
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


def _available_ports(grid: np.ndarray, center_pos: tuple[int, int], border_color: int) -> list[tuple[int, int]]:
    """All empty port cells available for a sprite centered at center_pos."""
    cr, cc = center_pos
    ports = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == border_color and int(grid[er, ec]) == BG:
            ports.append((er, ec))
    return ports


def _sprite_touches_path(
    grid: np.ndarray,
    center_pos: tuple[int, int],
    border_color: int,
    path_cells: set[tuple[int, int]],
) -> bool:
    """True iff any valid port of the sprite would lie on a grey path cell."""
    return any(port in path_cells for port in _available_ports(grid, center_pos, border_color))


def _assign_colors() -> dict | None:
    """Choose colors for the 1-target/1-source matched or unmatched case."""
    palette = [c for c in range(1, 10) if c != GREY]
    target_border, source_border = random.sample(palette, 2)
    matched = random.choices([True, False], weights=MATCH_WEIGHTS)[0]

    if matched:
        target_center_in = source_border
        source_center_choices = [
            c for c in palette if c not in {target_border, source_border, target_center_in}
        ]
        if not source_center_choices:
            return None
        source_center = random.choice(source_center_choices)
        target_center_out = source_center
    else:
        target_center_choices = [c for c in palette if c not in {target_border, source_border}]
        if not target_center_choices:
            return None
        target_center_in = random.choice(target_center_choices)
        source_center_choices = [
            c for c in palette if c not in {target_border, source_border, target_center_in}
        ]
        if not source_center_choices:
            return None
        source_center = random.choice(source_center_choices)
        target_center_out = target_center_in

    return {
        "target_border": target_border,
        "source_border": source_border,
        "target_center_in": target_center_in,
        "target_center_out": target_center_out,
        "source_center": source_center,
        "matched": matched,
    }


def _choose_target_positions() -> list[tuple[int, int]]:
    """Edge-biased placement order for the target sprite."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    random.shuffle(edge_pos)
    random.shuffle(int_pos)
    if random.choices([True, False], weights=[TARGET_EDGE_WEIGHT, TARGET_INTERIOR_WEIGHT])[0]:
        return edge_pos + int_pos
    return int_pos + edge_pos


def _place_target_and_path(
    output_grid: np.ndarray,
    shape_type: str,
    target_border: int,
    target_center_out: int,
    target_len: int,
) -> dict | None:
    """Place the output target and connect two of its ports with the grey path."""
    target_patch = _make_sprite_patch(shape_type, target_border, target_center_out)
    for r0, c0 in _choose_target_positions():
        if _overlaps(output_grid, target_patch, r0, c0):
            continue
        _place_sprite(output_grid, target_patch, r0, c0)
        center_pos = (r0 + 1, c0 + 1)
        all_ports = _available_ports(output_grid, center_pos, target_border)
        port_pairs = [pair for pair in combinations(all_ports, 2) if abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]) >= 3]
        random.shuffle(port_pairs)

        for port_a, port_b in port_pairs:
            path_cells = _gen_winding_path(output_grid, port_a, port_b, target_len)
            if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
                continue
            if not _is_8_connected(path_cells):
                continue
            other_ports = set(all_ports) - {port_a, port_b}
            if other_ports & set(path_cells):
                continue

            valid = True
            for rr, cc in path_cells:
                if output_grid[rr, cc] != BG:
                    valid = False
                    break
            if not valid:
                continue

            for rr, cc in path_cells:
                output_grid[rr, cc] = GREY

            try:
                sprite = {
                    "pts": [(rr, cc) for rr in range(max(0, r0), min(GRID_H, r0 + 3))
                            for cc in range(max(0, c0), min(GRID_W, c0 + 3))
                            if target_patch[rr - r0, cc - c0] != BG],
                    "border": target_border,
                    "center": target_center_out,
                    "center_pos": center_pos,
                }
                eps = set(_find_trail_endpoints(output_grid, GREY, [sprite]))
                if eps != {port_a, port_b}:
                    raise AssertionError
            except Exception:
                for rr, cc in path_cells:
                    output_grid[rr, cc] = BG
                continue

            return {
                "target_patch": target_patch,
                "target_r0": r0,
                "target_c0": c0,
                "target_center_pos": center_pos,
                "path_cells": path_cells,
                "path_set": set(path_cells),
                "ports": (port_a, port_b),
            }

        _erase_sprite(output_grid, target_patch, r0, c0)
    return None


def _place_source(
    input_grid: np.ndarray,
    shape_type: str,
    source_border: int,
    source_center: int,
    path_set: set[tuple[int, int]],
) -> bool:
    """Place the lone source away from the path and without valid ports onto it."""
    source_patch = _make_sprite_patch(shape_type, source_border, source_center)
    positions = _sprite_positions()
    random.shuffle(positions)
    for r0, c0 in positions:
        if _overlaps(input_grid, source_patch, r0, c0):
            continue
        _place_sprite(input_grid, source_patch, r0, c0)
        center_pos = (r0 + 1, c0 + 1)
        touches = _sprite_touches_path(input_grid, center_pos, source_border, path_set)
        if not touches:
            return True
        _erase_sprite(input_grid, source_patch, r0, c0)
    return False


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    colors = _assign_colors()
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)

    placed = _place_target_and_path(
        output_grid,
        shape_type,
        colors["target_border"],
        colors["target_center_out"],
        target_len,
    )
    if placed is None:
        return None

    input_grid = output_grid.copy()
    tr, tc = placed["target_center_pos"]
    input_grid[tr, tc] = colors["target_center_in"]

    if not _place_source(
        input_grid,
        shape_type,
        colors["source_border"],
        colors["source_center"],
        placed["path_set"],
    ):
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
