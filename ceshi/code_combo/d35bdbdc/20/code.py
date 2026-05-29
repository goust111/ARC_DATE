# Combo variant of d35bdbdc (index 20)
# combo_spec: D4/0+D12/0
# Name: 精灵沿边缘排列+边缘精灵为目标

# Design analysis for the combo variant
# a) Concrete behavior trace
#    - Base task: the grey trail endpoints choose exactly two bordered sprites, those two
#      stay, and each kept center is recolored by looking up another sprite whose border
#      matches the kept center color.
#    - Fused task: all bordered sprites are moved onto the outer frame. Two of them are
#      still attached to the grey trail endpoints, but destination selection now follows
#      D4: every edge-touching sprite is kept.
#    - Concrete example:
#      * Four sprites sit on top / bottom / left / right edges.
#      * The grey trail endpoints touch the top and bottom sprites only.
#      * The left and right sprites have center colors equal to the top/bottom border
#        colors, so they are the only ones that visibly recolor.
#      * Old rule would keep only the two path-linked sprites; the fused rule keeps all
#        four edge sprites, so the extra survivors are the visible signature of the combo.
# b) Complexity calibration
#    - The layout is fixed to four edge sprites. That is enough repetition to make the
#      "all edge sprites survive" rule obvious, while avoiding crowding on a 10x10 grid.
#    - Only two sprites are path-linked and only two sprites use border-lookup recoloring.
#      This keeps the color chain legible and avoids recursive many-to-many matches.
# c) Implicit-assumption audit
#    - Original `main()` assumed trail endpoints determine destinations. That assumption is
#      removed; endpoint logic is kept only for generator validation against degeneration.
#    - The border->source lookup still requires unique border colors, so the generator uses
#      four distinct sprite borders.
#    - The destination flood-fill still works, but now it seeds from all edge-touching
#      sprites. Because every sprite is placed on an edge, no intended sprite is deleted.
#    - `_bg_boundary()` assumes background dominates the frame. The generator keeps the
#      path off the frame and limits frame occupancy to four small sprites.
# d) Placement legality
#    - Every sprite is placed from side-specific anchor sets that guarantee edge contact and
#      an inward-facing port cell for the two source sprites.
#    - Non-grey objects never overlap or 4-touch. Target-side anchors are rejected if any
#      of their possible endpoint ports would coincide with a source endpoint.
#    - The grey path is generated only after all four sprites are reserved, so it cannot
#      overwrite them.
# e) Visual observability
#    - Across examples, all sprites lie on the boundary but only two are touched by the
#      path. The output nevertheless preserves four sprites, so the selection rule is
#      visibly different from the original path-based one.
#    - The recolored sprites are not the path-linked ones, which reinforces that the path
#      is no longer the deciding cue for survival.
# f) Visual cleanliness
#    - One sprite is placed on each side, away from the corners, so the boundary remains
#      readable and the interior has room for a single thin trail.
#    - Only the center pixels of the two lookup targets change, which keeps the output
#      sparse and easy to compare with the input.
# g) Change inferability
#    - The two changing centers each equal a source border color in the input, and each
#      matched source keeps a distinct center color. That gives a direct structural clue
#      for the center-color transfer without relying on color-id ordering.

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two logical endpoints of the grey one-stroke trail."""
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        for port in _possible_port_cells(sp["center_pos"], int(sp["border"]), g.shape):
            if port in cells:
                port_cells.add(port)

    start = deg1[0] if deg1 else next(iter(cells))
    total = len(cells)

    def dfs(cur: tuple[int, int], visited: frozenset[tuple[int, int]]) -> list[tuple[int, int]] | None:
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


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
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


def _touches_grid_edge(sprite: dict, shape: tuple[int, int]) -> bool:
    """Whether any sprite cell touches the outer frame."""
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in sprite["pts"])


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Endpoint cells among `eps` that geometrically attach to the sprite through its border."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every bordered sprite on the outer frame and discard anything off the frame.

    Input structure:
    - The grid contains a grey interior trail and several small bordered sprites made from
      one repeated border color plus one single center pixel of another color.
    - In this combo, all intended sprites are arranged along the four edges of the grid.
      Some may still touch the grey trail via endpoint ports, but that path relation is
      not the criterion for which sprites survive.

    Rule:
    - Treat every bordered sprite that touches the grid boundary as a destination to keep.
    - For each kept sprite, use its original center color as a lookup key: if another
      sprite has that border color, copy that sprite's center color into the kept center.
      If no such source exists, leave the center unchanged.
    - Delete all non-edge sprites and preserve both the grey trail and the edge sprites.

    Key insight: The important cue is now spatial placement on the frame. The grey trail
    may still highlight only some sprites, but survival and recoloring are driven by
    "touches the edge" plus the same border-to-center color lookup as before.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    dest = [sp for sp in sprites if _touches_grid_edge(sp, g.shape)]
    assert dest, "expected at least one edge-touching destination sprite"
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
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


# --------------- generator (c) ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNT = 4
SOURCE_COUNT = 2
PATH_LEN_MIN, PATH_LEN_MAX = 10, 20
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 500

SIDE_PAIRS = [("top", "bottom"), ("left", "right")]
SIDE_ORDER = ("top", "bottom", "left", "right")


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch with its center at local coord (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent (0) and out-of-bounds."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a placement by clearing all non-transparent patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check overlap or 4-touch against existing non-bg, non-grey cells."""
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


def _edge_anchors(side: str) -> list[tuple[int, int]]:
    """Candidate anchors that touch one side while keeping an inward port available."""
    if side == "top":
        return [(0, c0) for c0 in range(1, 7)]
    if side == "bottom":
        return [(GRID_H - 3, c0) for c0 in range(1, 7)]
    if side == "left":
        return [(r0, 0) for r0 in range(1, 7)]
    if side == "right":
        return [(r0, GRID_W - 3) for r0 in range(1, 7)]
    raise ValueError(side)


def _center_from_anchor(anchor: tuple[int, int]) -> tuple[int, int]:
    return anchor[0] + 1, anchor[1] + 1


def _inward_port(center_pos: tuple[int, int], side: str) -> tuple[int, int]:
    cr, cc = center_pos
    if side == "top":
        return cr + 2, cc
    if side == "bottom":
        return cr - 2, cc
    if side == "left":
        return cr, cc + 2
    if side == "right":
        return cr, cc - 2
    raise ValueError(side)


def _possible_port_cells(
    center_pos: tuple[int, int],
    border_color: int,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """All geometric endpoint port cells implied by a sprite center and border."""
    h, w = shape
    cr, cc = center_pos
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < h and 0 <= ic < w and 0 <= er < h and 0 <= ec < w):
            continue
        ports.append((er, ec))
    return ports


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check whether adding (r,c) would create an overly thick 8-neighbor cluster."""
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
    """Generate a thin 8-connected grey path between two reserved ports."""
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
                if grid[nr, nc] != BG:
                    continue
                if _too_thick(nr, nc, visited):
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
        if grid[end[0], end[1]] != BG:
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


def _place_side_sprite(
    grid: np.ndarray,
    shape_type: str,
    border_color: int,
    center_color: int,
    side: str,
    occupied_centers: list[tuple[int, int]],
    forbidden_ports: set[tuple[int, int]],
) -> dict | None:
    """Place one sprite on the requested side while keeping the scene uncluttered."""
    anchors = _edge_anchors(side)
    random.shuffle(anchors)
    patch = _make_sprite_patch(shape_type, border_color, center_color)

    for anchor in anchors:
        r0, c0 = anchor
        if _overlaps(grid, patch, r0, c0):
            continue
        center_pos = _center_from_anchor(anchor)
        if any(abs(center_pos[0] - cr) + abs(center_pos[1] - cc) < 4 for cr, cc in occupied_centers):
            continue
        ports = set(_possible_port_cells(center_pos, border_color, grid.shape))
        if ports & forbidden_ports:
            continue
        _place_sprite(grid, patch, r0, c0)
        return {
            "r0": r0,
            "c0": c0,
            "side": side,
            "center_pos": center_pos,
            "border": border_color,
            "center": center_color,
        }
    return None


def _assign_colors(border_colors: list[int]) -> dict | None:
    """Use two edge-linked source sprites and two edge targets that look them up."""
    source_borders = border_colors[:SOURCE_COUNT]
    target_borders = border_colors[SOURCE_COUNT:]

    forbidden = {BG, GREY, *border_colors}
    source_center_pool = [c for c in range(1, 10) if c not in forbidden]
    if len(source_center_pool) < SOURCE_COUNT:
        return None
    source_centers = random.sample(source_center_pool, SOURCE_COUNT)

    mapping = [0, 1]
    random.shuffle(mapping)
    target_input_centers = [source_borders[mapping[i]] for i in range(2)]
    output_target_centers = [source_centers[mapping[i]] for i in range(2)]

    return {
        "source_borders": source_borders,
        "source_centers": source_centers,
        "target_borders": target_borders,
        "target_input_centers": target_input_centers,
        "output_target_centers": output_target_centers,
    }


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, SPRITE_COUNT)
    colors = _assign_colors(border_colors)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    source_sides = random.choice(SIDE_PAIRS)
    target_sides = [side for side in SIDE_ORDER if side not in source_sides]

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    sprites: list[dict] = []
    occupied_centers: list[tuple[int, int]] = []

    source_sprites: list[dict] = []
    for side, border, center in zip(source_sides, colors["source_borders"], colors["source_centers"], strict=False):
        sp = _place_side_sprite(output_grid, shape_type, border, center, side, occupied_centers, set())
        if sp is None:
            return None
        source_sprites.append(sp)
        sprites.append(sp)
        occupied_centers.append(sp["center_pos"])

    source_ports = [_inward_port(sp["center_pos"], sp["side"]) for sp in source_sprites]
    if abs(source_ports[0][0] - source_ports[1][0]) + abs(source_ports[0][1] - source_ports[1][1]) < min_port_dist:
        return None

    forbidden_ports = set(source_ports)
    target_sprites: list[dict] = []
    for side, border, center in zip(
        target_sides,
        colors["target_borders"],
        colors["output_target_centers"],
        strict=False,
    ):
        sp = _place_side_sprite(output_grid, shape_type, border, center, side, occupied_centers, forbidden_ports)
        if sp is None:
            return None
        target_sprites.append(sp)
        sprites.append(sp)
        occupied_centers.append(sp["center_pos"])

    planning_grid = output_grid.copy()
    path_cells = _gen_winding_path(planning_grid, source_ports[0], source_ports[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if planning_grid[r, c] != BG:
            return None
        planning_grid[r, c] = GREY
        output_grid[r, c] = GREY

    for pr, pc in source_ports:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    input_grid = output_grid.copy()
    for sp, input_center in zip(target_sprites, colors["target_input_centers"], strict=False):
        output_patch = _make_sprite_patch(shape_type, sp["border"], sp["center"])
        input_patch = _make_sprite_patch(shape_type, sp["border"], input_center)
        _erase_sprite(input_grid, output_patch, sp["r0"], sp["c0"])
        _place_sprite(input_grid, input_patch, sp["r0"], sp["c0"])

    bg = _bg_boundary(input_grid)
    parsed = _detect_bordered_sprites(input_grid, bg, GREY)
    if len(parsed) != SPRITE_COUNT:
        return None
    if not all(_touches_grid_edge(sp, input_grid.shape) for sp in parsed):
        return None

    te1, te2 = _find_trail_endpoints(input_grid, GREY, parsed)
    eps = {te1, te2}
    endpoint_linked = [sp for sp in parsed if any(ep in _ports_for_sprite(sp, input_grid, eps) for ep in eps)]
    if len(endpoint_linked) != SOURCE_COUNT:
        return None

    edge_sprites = [sp for sp in parsed if _touches_grid_edge(sp, input_grid.shape)]
    if len(edge_sprites) != SPRITE_COUNT:
        return None
    if len(edge_sprites) <= len(endpoint_linked):
        return None

    try:
        out = main(input_grid)
        if not np.array_equal(out, output_grid):
            return None
    except Exception:
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
