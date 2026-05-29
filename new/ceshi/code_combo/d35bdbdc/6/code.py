# Combo variant of d35bdbdc (index 6)
# combo_spec: D4/0+D6/0
# Name: 边缘精灵为目标+保留未连接精灵

# Design analysis for the combo variant
# a) Concrete behavior trace:
#    - Objects stay the same as the seed task: a 10x10 grid with one grey winding trail
#      and several two-color bordered sprites, each having a repeated border color and a
#      single center pixel of a different color.
#    - Original task: the two sprites touched by the grey trail endpoints are the only
#      survivors, and their centers are recolored via border->center lookup.
#    - D4/0 changes destination selection: edge-touching sprites become the targets even
#      if the grey trail points somewhere else.
#    - D6/0 changes cleanup: non-target sprites are no longer deleted.
#    - In this combo, a typical example has two edge sprites E1/E2 and two interior
#      sprites S1/S2 whose ports are the trail endpoints. Output keeps E1/E2/S1/S2 (and
#      maybe one extra distractor), but only E1/E2 have their center colors replaced.
#      Thus the visible contrast is "path touches interior sprites, yet only edge sprites
#      change, and everything else remains".
# b) Complexity calibration:
#    - The generator fixes exactly two edge targets and exactly two interior trail-linked
#      source sprites. This avoids the weak case where path-linked sprites and edge
#      targets coincide, which would blur D4.
#    - There may be one extra preserved interior distractor so outputs consistently show
#      more than the two changed sprites, which makes D6 observable.
#    - Border colors remain unique and the lookup chain uses only two target/source pairs,
#      keeping the mapping legible without many-to-many ambiguity.
# c) Implicit-assumption audit against the original main():
#    - Original endpoint detection and endpoint->sprite matching assumed destinations were
#      determined by the trail. That assumption is removed from `main()` and replaced by
#      edge-contact detection.
#    - Original deletion loops assumed every non-destination object must disappear. That
#      is invalid here and must be removed entirely.
#    - The border->source lookup still holds because border colors stay unique across
#      sprites and target centers are generated to match source borders.
#    - `_bg_boundary()` still assumes background dominates the frame, so the generator
#      limits edge sprites to exactly two small clipped objects and keeps the grey trail
#      off the border.
# d) Placement legality:
#    - Sprites never overlap or 4-touch.
#    - Targets are forced to the frame; sources and optional distractors are forced to the
#      interior.
#    - The grey path is generated only between chosen interior source ports, and candidate
#      cells that would act as ports for edge targets are blocked during path generation.
# e) Visual observability:
#    - Across examples, the grey trail visibly ends at interior sprites, but those sprites
#      stay unchanged in the output.
#    - The only changed cells are the centers of edge-touching sprites, so a human can
#      infer "edge sprites are targets" and "other sprites are preserved" from a few
#      examples without relying on hidden numeric-color ordering.
# f) Visual cleanliness:
#    - All sprites share one simple 3x3 shape per instance (box or plus), remain spaced
#      apart, and the grey trail is a single thin winding component.
#    - Blocking edge-target port cells prevents the path from appearing to touch targets,
#      reducing misleading overlaps between the two selection mechanisms.
# g) Color-change inferability:
#    - Only two center pixels change per grid.
#    - Each changed target center still points to a unique source border color, so the
#      recoloring rule has repeated structural evidence while preserved bystanders show
#      that unchanged sprites are intentional rather than absent.

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
BLOCKED = -1

TARGET_COUNT = 2
SOURCE_COUNT = 2
EXTRA_DISTRACTOR_WEIGHTS = [2, 1]  # weights for [0, 1]

PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
    Clipped edge sprites are allowed.
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


def _touches_grid_edge(sprite: dict, shape: tuple[int, int]) -> bool:
    """A sprite is a combo-target iff any of its cells touches the outer frame."""
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in sprite["pts"])


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two logical endpoints of the grey one-stroke trail."""
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

    def dfs(
        cur: tuple[int, int], visited: frozenset[tuple[int, int]]
    ) -> list[tuple[int, int]] | None:
        if len(visited) == total:
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


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Grey endpoint cells that are valid 2-step ports for this sprite."""
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


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch with the center at local (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent background and clipping at bounds."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo placement by clearing all non-background patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject placements that overlap or 4-touch non-bg, non-grey material."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if int(grid[rr, cc]) != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) not in patch_cells and 0 <= nr < h and 0 <= nc < w:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    """All anchors whose center cell stays inside the 10x10 grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge_anchor(r0: int, c0: int) -> bool:
    """Whether a clipped 3x3 sprite placed at (r0,c0) touches the frame."""
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check whether adding one grey cell makes the trail visually too thick."""
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
    """Generate a winding 8-connected grey trail between two chosen source ports."""
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
                if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                    continue
                if (nr, nc) in visited:
                    continue
                if int(grid[nr, nc]) != BG:
                    continue
                if _too_thick(nr, nc, visited):
                    continue
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
            if random.random() < PATH_MODERATE_RANDOM_P:
                chosen = random.choice(pool)
            else:
                chosen = min(pool, key=score)
        else:
            best = sorted(moves, key=score)
            top = [move for move in best if score(move) <= score(best[0]) + 1]
            dtop = [move for move in top if move[2]]
            chosen = random.choice(dtop) if dtop else best[0]

        cr, cc = chosen[0], chosen[1]
        path.append((cr, cc))
        visited.add((cr, cc))

    if end not in visited:
        if int(grid[end[0], end[1]]) != BG:
            return None
        if _too_thick(end[0], end[1], visited):
            return None
        path.append(end)

    return path


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
    """Check 8-connectivity of a cell set via BFS."""
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


def _possible_port_cells(
    center_pos: tuple[int, int], border_color: int, grid: np.ndarray
) -> list[tuple[int, int]]:
    """All geometric port cells for a sprite, based only on its border geometry."""
    cr, cc = center_pos
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == border_color:
            ports.append((er, ec))
    return ports


def _available_ports(
    center_pos: tuple[int, int], border_color: int, grid: np.ndarray
) -> list[tuple[int, int]]:
    """Port cells that are currently empty on the working grid."""
    return [p for p in _possible_port_cells(center_pos, border_color, grid) if int(grid[p]) == BG]


def _has_grey_port(grid: np.ndarray, center_pos: tuple[int, int], border_color: int) -> bool:
    """True iff the sprite connects to any grey cell through a valid port."""
    cr, cc = center_pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == border_color and int(grid[er, ec]) == GREY:
            return True
    return False


def _assign_colors(border_colors: list[int]) -> dict | None:
    """Assign two edge targets, two interior sources, and maybe one preserved distractor."""
    target_borders = border_colors[:TARGET_COUNT]
    source_borders = border_colors[TARGET_COUNT:TARGET_COUNT + SOURCE_COUNT]
    distractor_borders = border_colors[TARGET_COUNT + SOURCE_COUNT:]

    mapping = [0, 1]
    random.shuffle(mapping)
    target_input_centers = [source_borders[mapping[i]] for i in range(TARGET_COUNT)]

    source_center_pool = [c for c in range(1, 10) if c not in {BG, GREY, *border_colors}]
    if len(source_center_pool) < SOURCE_COUNT:
        return None
    source_centers = random.sample(source_center_pool, SOURCE_COUNT)
    output_target_centers = [source_centers[mapping[i]] for i in range(TARGET_COUNT)]

    used_centers = set(source_centers) | set(target_input_centers)
    distractor_centers: list[int] = []
    for border in distractor_borders:
        candidates = [c for c in range(1, 10) if c not in {BG, GREY, border} and c not in used_centers]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        distractor_centers.append(chosen)
        used_centers.add(chosen)

    return {
        "target_borders": target_borders,
        "source_borders": source_borders,
        "distractor_borders": distractor_borders,
        "target_input_centers": target_input_centers,
        "source_centers": source_centers,
        "output_target_centers": output_target_centers,
        "distractor_centers": distractor_centers,
    }


def _place_edge_targets(
    grid: np.ndarray,
    shape_type: str,
    target_borders: list[int],
    output_target_centers: list[int],
) -> list[dict] | None:
    """Place exactly two targets on the frame, spaced so the scene stays readable."""
    edge_positions = [pos for pos in _sprite_positions() if _is_edge_anchor(pos[0], pos[1])]
    random.shuffle(edge_positions)
    placed: list[dict] = []
    for i in range(TARGET_COUNT):
        patch = _make_sprite_patch(shape_type, target_borders[i], output_target_centers[i])
        found = False
        for r0, c0 in edge_positions:
            if _overlaps(grid, patch, r0, c0):
                continue
            center_pos = (r0 + 1, c0 + 1)
            if any(
                abs(center_pos[0] - sp["center_pos"][0]) + abs(center_pos[1] - sp["center_pos"][1]) < 5
                for sp in placed
            ):
                continue
            _place_sprite(grid, patch, r0, c0)
            placed.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "center_pos": center_pos,
                    "border": target_borders[i],
                    "center": output_target_centers[i],
                }
            )
            found = True
            break
        if not found:
            return None
    return placed


def _place_interior_sources(
    grid: np.ndarray,
    shape_type: str,
    source_borders: list[int],
    source_centers: list[int],
    min_port_dist: int,
) -> tuple[list[dict], list[tuple[int, int]]] | None:
    """Place two interior sprites whose chosen ports will become the grey endpoints."""
    interior_positions = [pos for pos in _sprite_positions() if not _is_edge_anchor(pos[0], pos[1])]
    random.shuffle(interior_positions)
    placed: list[dict] = []
    chosen_ports: list[tuple[int, int]] = []

    for i in range(SOURCE_COUNT):
        patch = _make_sprite_patch(shape_type, source_borders[i], source_centers[i])
        found = False
        for r0, c0 in interior_positions:
            if _overlaps(grid, patch, r0, c0):
                continue
            center_pos = (r0 + 1, c0 + 1)
            if any(
                abs(center_pos[0] - sp["center_pos"][0]) + abs(center_pos[1] - sp["center_pos"][1]) < 4
                for sp in placed
            ):
                continue
            _place_sprite(grid, patch, r0, c0)
            ports = _available_ports(center_pos, source_borders[i], grid)
            if i == 1:
                ports = [
                    p
                    for p in ports
                    if abs(p[0] - chosen_ports[0][0]) + abs(p[1] - chosen_ports[0][1]) >= min_port_dist
                ]
            if not ports:
                _erase_sprite(grid, patch, r0, c0)
                continue
            chosen_port = random.choice(ports)
            placed.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "center_pos": center_pos,
                    "border": source_borders[i],
                    "center": source_centers[i],
                }
            )
            chosen_ports.append(chosen_port)
            found = True
            break
        if not found:
            return None

    return placed, chosen_ports


def _place_extra_distractor(
    grid: np.ndarray,
    shape_type: str,
    border_color: int,
    center_color: int,
) -> bool:
    """Place one preserved interior distractor that does not connect to the grey trail."""
    patch = _make_sprite_patch(shape_type, border_color, center_color)
    interior_positions = [pos for pos in _sprite_positions() if not _is_edge_anchor(pos[0], pos[1])]
    random.shuffle(interior_positions)
    for r0, c0 in interior_positions:
        if _overlaps(grid, patch, r0, c0):
            continue
        _place_sprite(grid, patch, r0, c0)
        center_pos = (r0 + 1, c0 + 1)
        if _has_grey_port(grid, center_pos, border_color):
            _erase_sprite(grid, patch, r0, c0)
            continue
        return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer center colors into the bordered sprites that touch the outer frame, while keeping every sprite.

    Input structure:
    - The grid contains a grey (5) one-stroke trail plus several small bordered sprites.
    - Each sprite uses exactly two colors: one repeated border color and one single center pixel of another color.
    - Some sprites touch the grid boundary and some sit in the interior; the grey trail may visibly connect to interior sprites, but that connection is not how targets are chosen here.

    Rule:
    - Mark every bordered sprite that touches any grid edge as a destination sprite.
    - For each such destination, read its original center color as a lookup key, find the sprite whose border color matches that key, and copy that source sprite's center color into the destination center.
    - Keep all sprites and the grey trail in place; only the centers of edge-touching sprites may change.

    Key insight:
    - This combo separates "which sprites the path touches" from "which sprites transform": the trail becomes a distractor, while boundary contact selects the targets and all non-target sprites remain as unchanged evidence.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    dest = [sp for sp in sprites if _touches_grid_edge(sp, g.shape)]
    assert dest, "expected at least one edge-touching destination sprite"

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()
    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one source border"
        dr, dc = d["center_pos"]
        out[dr, dc] = int(srcs[0]["center"])

    return out


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    extra_distractor_count = random.choices([0, 1], weights=EXTRA_DISTRACTOR_WEIGHTS)[0]
    num_sprites = TARGET_COUNT + SOURCE_COUNT + extra_distractor_count

    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, num_sprites)
    colors = _assign_colors(border_colors)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    target_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    targets = _place_edge_targets(
        target_grid,
        shape_type,
        colors["target_borders"],
        colors["output_target_centers"],
    )
    if targets is None:
        return None

    forbidden_target_ports: set[tuple[int, int]] = set()
    for target in targets:
        forbidden_target_ports.update(_possible_port_cells(target["center_pos"], target["border"], target_grid))

    planning_grid = target_grid.copy()
    for pr, pc in forbidden_target_ports:
        if int(planning_grid[pr, pc]) == BG:
            planning_grid[pr, pc] = BLOCKED

    source_result = _place_interior_sources(
        planning_grid,
        shape_type,
        colors["source_borders"],
        colors["source_centers"],
        min_port_dist,
    )
    if source_result is None:
        return None
    sources, port_cells = source_result

    path_cells = _gen_winding_path(planning_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    output_grid = target_grid.copy()
    for source in sources:
        patch = _make_sprite_patch(shape_type, source["border"], source["center"])
        _place_sprite(output_grid, patch, source["r0"], source["c0"])

    for r, c in path_cells:
        if int(output_grid[r, c]) != BG:
            return None
        output_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(output_grid[nr, nc]) == GREY
        )
        if deg > 1:
            return None

    for border, center in zip(
        colors["distractor_borders"], colors["distractor_centers"], strict=False
    ):
        if not _place_extra_distractor(output_grid, shape_type, border, center):
            return None

    input_grid = output_grid.copy()
    for i, target in enumerate(targets):
        cr, cc = target["center_pos"]
        input_grid[cr, cc] = colors["target_input_centers"][i]

    bg = _bg_boundary(input_grid)
    sprites = _detect_bordered_sprites(input_grid, bg, GREY)
    if len(sprites) != num_sprites:
        return None

    edge_sprites = [sp for sp in sprites if _touches_grid_edge(sp, input_grid.shape)]
    if len(edge_sprites) != TARGET_COUNT:
        return None

    te1, te2 = _find_trail_endpoints(input_grid, GREY, sprites)
    eps = {te1, te2}
    endpoint_linked = [sp for sp in sprites if any(ep in _ports_for_sprite(sp, input_grid, eps) for ep in eps)]
    if len(endpoint_linked) != SOURCE_COUNT:
        return None
    if any(_touches_grid_edge(sp, input_grid.shape) for sp in endpoint_linked):
        return None
    for ep in eps:
        if sum(ep in _ports_for_sprite(sp, input_grid, eps) for sp in sprites) != 1:
            return None

    combo_out = main(input_grid)
    if not np.array_equal(combo_out, output_grid):
        return None

    changed = [
        sp for sp in edge_sprites
        if int(input_grid[sp["center_pos"]]) != int(combo_out[sp["center_pos"]])
    ]
    if len(changed) != TARGET_COUNT:
        return None
    if np.any((input_grid != combo_out) & (input_grid != GREY) & (combo_out != GREY)):
        changed_cells = {
            (int(r), int(c))
            for r, c in zip(*np.where(input_grid != combo_out), strict=False)
        }
        allowed = {sp["center_pos"] for sp in edge_sprites}
        if changed_cells != allowed:
            return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
