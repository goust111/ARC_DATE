# Variant of d35bdbdc
# Dimension: D14 — 匹配失败的处理 (meta)
# Change: 每个目标精灵都能找到匹配的源精灵（隐含假设无匹配失败）... → 未匹配的目标精灵被删除
#
# Design analysis for the D14 variant
# -----------------------------------
# a) Concrete behavior and flow trace
# - The visual vocabulary stays the same as the base task: a 10x10 grid, bg=0,
#   a grey one-stroke trail (color 5), and 3x3 bordered sprites whose center is
#   a single pixel of a different color. The two trail endpoints still determine
#   the two destination sprites through the existing port rule.
# - The new observable behavior is that one endpoint sprite may disappear
#   entirely in the output if its center color does not name any source border.
#   The other endpoint sprite can still survive and receive a transferred center
#   color from its matched source.
# - Concrete trace:
#   input has two endpoint sprites D0 and D1, one detached source S, and some
#   distractors. Suppose D0.center = S.border, while D1.center is a color absent
#   from every sprite border. In the original task, D0 gets S.center and D1
#   stays unchanged. In this variant, D0 still gets S.center, but D1 is erased
#   completely, just like a deleted distractor. The grey trail remains.
# - Therefore the changed step is not destination detection or path tracing; it
#   is the post-match retention rule for destination sprites.
#
# b) Complexity calibration
# - To make the new rule visible in 2-4 examples, the generator does not leave
#   matching failure as a rare corner case. Each generated instance contains
#   exactly one matched destination and exactly one unmatched destination.
# - This keeps the original color-lookup logic intact while providing a
#   side-by-side contrast within a single example: one endpoint sprite survives,
#   the other endpoint sprite vanishes.
# - The unmatched destination key is chosen from colors absent from all border
#   colors, preventing accidental hidden matches.
#
# c) Hidden assumptions in the original main()
# - Original code assumes `by_border.get(d["center"], [])` is non-empty often
#   enough that leaving the destination in place is harmless. That assumption no
#   longer holds; when `srcs` is empty, the destination must be deleted.
# - Original keep-mask BFS seeds from both destinations unconditionally. Under
#   the new rule this would wrongly preserve an unmatched destination, so keep
#   seeding must use only matched destinations.
# - Original "delete all non-destination sprites" is still valid, but now one of
#   the destinations may be turned into an additional deletion target.
# - The border lookup still assumes at most one source per border color. The
#   generator preserves unique border colors, so this remains sound.
#
# d) Placement legality
# - Sprite geometry, clipping, overlap checks, and path generation remain those
#   of the base task. The generator still places both destinations first so the
#   grey path can connect their ports legally.
# - The output is derived from a temporary full layout containing both
#   destinations; after the path is built, the unmatched destination patch is
#   erased. This keeps port/path legality in the input while making the output
#   obey the new deletion rule.
# - Source and distractor placement still uses the original non-overlap and
#   non-4-touch constraints, and oracle verification rejects ambiguous cases.
#
# e) Visual observability
# - The variant is directly visible because the output contains a dangling grey
#   endpoint where one endpoint sprite used to be, while the other endpoint
#   sprite remains and changes center color.
# - No ordering or transformation depends on hidden numeric properties such as
#   color id rank. The only color logic visible to the solver is equality:
#   destination center color either matches some border color or it does not.
#
# f) Visual cleanliness
# - The task keeps the base layout discipline: small separated sprites, a thin
#   grey trail, and at most one deleted endpoint sprite. This avoids clutter.
# - Using exactly one unmatched destination per instance prevents outputs from
#   collapsing into nearly-empty boards with two dangling trail endpoints.
#
# g) Color-change inferability
# - There is exactly one successful lookup chain per instance, so the surviving
#   endpoint sprite still demonstrates the original transfer mechanism.
# - The unmatched key is a single color absent from all borders, making the new
#   branch interpretable as "no source with that border exists, so that endpoint
#   sprite disappears."

from __future__ import annotations

import os
import sys
import random
from collections import deque

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

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
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
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


def _detect_bordered_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Bordered sprite = a 4-connected component (excluding bg/grey) with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once (its location is the 'center')
    This also handles clipped/partial borders (e.g., near the grid edge).
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
    """
    A grey endpoint (er,ec) is a port for this sprite if:
      - it's exactly 2 steps away from the sprite center (Manhattan distance 2),
      - the intermediate cell (1 step away from center toward endpoint) is the sprite border color.
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


def _erase_sprite_cells(out: np.ndarray, sprite: dict, bg: int) -> None:
    for r, c in sprite["pts"]:
        out[r, c] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer colors into endpoint sprites only when their lookup succeeds, deleting failed targets.

    Input structure:
    - Background cells, a grey (5) one-stroke trail, and several small bordered sprites with one special center pixel.
    - The two grey trail endpoints connect through valid ports to exactly two destination sprites; other sprites are detached lookup candidates or distractors.

    Rule:
    - Use the two trail endpoints to identify the destination sprites.
    - For each destination, use its original center color as a key and search for a sprite whose border color matches that key.
    - If a match exists, keep that destination and replace its center with the source sprite's center color.
    - If no match exists, delete that destination entirely; all non-destination sprites are also removed, while the grey trail remains.

    Key insight:
    - Endpoint sprites are filtered by the success of a border-color lookup: matching ones survive with transferred centers, and unmatched ones vanish like other discarded sprites.
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
    assert all(d is not None for d in dest), "expected exactly 2 dest sprites from trail endpoints"
    dest_ids = {id(sp) for sp in dest}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    matched: list[tuple[dict, dict]] = []
    unmatched: list[dict] = []
    for d in dest:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            unmatched.append(d)
            continue
        assert len(srcs) == 1, "each destination color should match at most one source border"
        matched.append((d, srcs[0]))

    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        _erase_sprite_cells(out, sp, bg)

    for d in unmatched:
        _erase_sprite_cells(out, d, bg)

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d, _src in matched:
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

    for d, src in matched:
        dr, dc = d["center_pos"]
        out[dr, dc] = int(src["center"])

    return out


# --------------- generator (variant) ---------------

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
GEN_RETRIES = 400


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return 3x3 patch for a box or plus. Center is at (1,1)."""
    if shape_type == "box":
        p = np.full((3, 3), border_color, dtype=int)
    else:
        p = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p[1 + dr, 1 + dc] = border_color
    p[1, 1] = center_color
    return p


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent (0) and out-of-bounds."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a _place_sprite by zeroing non-transparent patch cells."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != 0:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check if placing patch at (r0,c0) would overlap or 4-touch non-bg, non-grey cells."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] != 0:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < h and 0 <= cc < w:
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
    """All valid (r0,c0) for a 3x3 patch where center stays on grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 <= 0 or r0 + 3 >= GRID_H or c0 <= 0 or c0 + 3 >= GRID_W


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
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int
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
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == 0:
            return (er, ec)
    return None


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    """Assign one matched destination, one unmatched destination, and the remaining distractors."""
    remaining = list(range(2, num_sprites))
    if not remaining:
        return None

    source_index = random.choice(remaining)
    distractor_indices = [i for i in remaining if i != source_index]
    matched_dest_index = random.choice([0, 1])
    unmatched_dest_index = 1 - matched_dest_index

    unmatched_candidates = [c for c in range(1, 10) if c != GREY and c not in border_colors]
    if not unmatched_candidates:
        return None
    unmatched_dest_center = random.choice(unmatched_candidates)

    source_center_forbidden = {BG, GREY, unmatched_dest_center, *border_colors}
    source_center_candidates = [c for c in range(1, 10) if c not in source_center_forbidden]
    if not source_center_candidates:
        return None
    source_center_color = random.choice(source_center_candidates)

    distractor_center_colors = []
    for di in distractor_indices:
        candidates = [c for c in range(1, 10) if c != GREY and c != border_colors[di]]
        if not candidates:
            return None
        distractor_center_colors.append(random.choice(candidates))

    dest_center_colors = [0, 0]
    temp_output_dest_centers = [0, 0]
    dest_center_colors[matched_dest_index] = border_colors[source_index]
    dest_center_colors[unmatched_dest_index] = unmatched_dest_center
    temp_output_dest_centers[matched_dest_index] = source_center_color
    temp_output_dest_centers[unmatched_dest_index] = unmatched_dest_center

    return {
        "source_index": source_index,
        "distractor_indices": distractor_indices,
        "matched_dest_index": matched_dest_index,
        "unmatched_dest_index": unmatched_dest_index,
        "dest_center_colors": dest_center_colors,
        "temp_output_dest_centers": temp_output_dest_centers,
        "source_center_color": source_center_color,
        "distractor_center_colors": distractor_center_colors,
    }


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    """Place 2 destination sprites on the grid."""
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_center_positions = []
    first_port = None

    for i in range(2):
        bc = border_colors[i]
        patch = _make_sprite_patch(shape_type, bc, output_dest_centers[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc_pos = r0 + 1, c0 + 1

            _place_sprite(output_grid, patch, r0, c0)
            port = _find_port_dir(output_grid, cr, cc_pos, bc)
            if port is None:
                _erase_sprite(output_grid, patch, r0, c0)
                continue

            if i == 1:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(output_grid, patch, r0, c0)
                    continue

            dest_center_positions.append((cr, cc_pos))
            if i == 0:
                first_port = port
            placed = True
            break

        if not placed:
            return None

    return dest_center_positions


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

    full_output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    result = _place_dests(
        full_output_grid,
        shape_type,
        border_colors,
        colors["temp_output_dest_centers"],
        min_port_dist,
    )
    if result is None:
        return None
    dest_center_positions = result

    port_cells = []
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        port = _find_port_dir(full_output_grid, cr, cc_pos, border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(full_output_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if full_output_grid[r, c] != BG:
            return None
        full_output_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and full_output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    input_grid = full_output_grid.copy()
    for i in range(2):
        cr, cc_pos = dest_center_positions[i]
        input_grid[cr, cc_pos] = colors["dest_center_colors"][i]

    sprites_to_place = [
        (border_colors[colors["source_index"]], colors["source_center_color"])
    ] + [
        (border_colors[di], colors["distractor_center_colors"][j])
        for j, di in enumerate(colors["distractor_indices"])
    ]
    all_pos = _sprite_positions()
    for bc, cc_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, bc, cc_color)
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

    output_grid = full_output_grid.copy()
    doomed = colors["unmatched_dest_index"]
    dcr, dcc = dest_center_positions[doomed]
    doomed_patch = _make_sprite_patch(
        shape_type,
        border_colors[doomed],
        colors["temp_output_dest_centers"][doomed],
    )
    _erase_sprite(output_grid, doomed_patch, dcr - 1, dcc - 1)

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
