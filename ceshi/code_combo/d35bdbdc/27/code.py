# Combo variant of d35bdbdc (index 27)
# combo_spec: D1/0+D3/0+D6/0
# Name: 中心色匹配+单色精灵+保留未连接
#
# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace
# - Seed task: the grey trail selects two bordered destination sprites. Each
#   destination uses its center color as a lookup key into some other bordered
#   sprite's border color, then copies that source sprite's center into the
#   destination center. Every unconnected sprite is deleted.
# - D1 changes the key from source.border to source.center. D3 changes the
#   source object itself into a monochrome solid block whose only meaningful
#   color is its "center color". D6 keeps all unconnected objects in the
#   output instead of deleting them.
# - Those three changes interact badly if applied naively: with a monochrome
#   source and center-to-center matching, copying source.center into
#   destination.center would be a no-op because the two colors already match.
# - The fused rule therefore keeps the new match key but moves the visible
#   write target: when a solid source block color matches a destination center,
#   repaint the entire destination sprite with that color. Small trace:
#   input destination border=7 center=3, detached solid source block color=3.
#   Original seed output would depend on some bordered source with border=3.
#   Fused output instead turns the path-connected destination into a solid
#   3-colored sprite while the detached source block and all bystanders stay.
#
# b) Complexity calibration
# - Sources are 2x2 solid blocks so they are visually distinct from bordered
#   3x3 box/plus sprites.
# - Source colors are drawn from colors unused as bordered-sprite borders.
#   This avoids accidental overlap with the seed task's border-based chain and
#   makes the new center-based key legible.
# - The generator uses 1 or 2 sources. With 1 source, both destinations share
#   one center color, giving repeated evidence for the match. With 2 sources,
#   both destinations can change independently without introducing ambiguity.
# - Bordered distractors, when present, avoid source colors at their centers so
#   the only repeated center colors that matter are the destination/source pair.
#
# c) Implicit assumption audit
# - The original `_detect_bordered_sprites` assumption still holds for
#   destinations and bordered distractors, so that detector is reused.
# - The original `by_border` table is invalid because sources are no longer
#   bordered sprites. The solver needs a second detector for monochrome source
#   blocks and must index those blocks by their single color.
# - The original "delete all non-dest material" logic is explicitly wrong after
#   D6, so the fused solver must skip every cleanup stage and start from a copy
#   of the full input.
# - Because the new visible change is "destination becomes monochrome", the
#   solver assumes matched destinations start as two-color bordered sprites.
#   The generator guarantees that by restoring the input border colors after
#   constructing the final solid-color outputs.
#
# d) Placement legality
# - Destination and bordered distractor sprites still come from the seed 3x3
#   box/plus family, may clip at edges, and never overlap or 4-touch other
#   colored objects.
# - Solid sources are fully in-bounds 2x2 squares. They also avoid touching the
#   grey path so the endpoint cue remains visually clean.
# - Bordered distractors are placed only after the path is drawn, and any
#   placement that would create a valid grey port is rejected. This preserves
#   the guarantee that exactly two bordered sprites are endpoint-connected.
#
# e) Visual observability
# - Across examples, the only objects that change are the two sprites singled
#   out by the grey endpoints.
# - Each changed destination shares its input center color with a detached solid
#   source block, and in the output that destination becomes entirely that same
#   color. Unconnected bordered sprites and source blocks remain unchanged.
# - The rule can be inferred from 2-4 examples without using color-id ordering
#   or any hidden metadata: the relevant cues are object type, trail endpoint
#   connectivity, and repeated visible colors.
#
# f) Visual cleanliness
# - The output is intentionally denser than the seed task because D6 preserves
#   bystanders, but the object families remain separable: thin grey trail,
#   small bordered sprites, and small solid blocks.
# - Rejecting extra grey ports prevents preserved bordered sprites from looking
#   half-selected by the trail, which would otherwise blur the task structure.
#
# g) Color-change inferability
# - The number of active lookup colors is at most 2, and one-source cases repeat
#   the same lookup twice. That keeps the color story small enough for humans to
#   infer: "the destination center color identifies the solid source block, and
#   the whole connected sprite changes to that block's color."
#
# h) Generator guarantees for new solver assumptions
# - Every source is a unique-color 2x2 solid component.
# - Every destination center matches exactly one source color.
# - No preserved bordered sprite has a valid grey port.
# - The seed solver cannot accidentally explain the samples because sources are
#   not bordered sprites and unconnected objects remain visible in the output.

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
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
    This also handles clipped/partial borders (e.g., near the grid edge).
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
        counts = {c: cols.count(c) for c in uniq}
        center_color = min(counts, key=lambda c: counts[c])
        if counts[center_color] != 1:
            continue
        border_color = [c for c in uniq if c != center_color][0]
        if border_color in (bg, grey) or center_color in (bg, grey):
            continue

        center_pos = None
        for r, c in pts_list:
            if int(g[r, c]) == center_color:
                center_pos = (int(r), int(c))
                break
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
        want = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if set(pts_list) != want:
            continue
        sources.append({"pts": pts_list, "color": int(next(iter(colors)))})
    return sources


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint (er,ec) is a port for this sprite if:
      - it's exactly 2 steps away from the sprite center,
      - the intermediate cell is the sprite border color.
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
    Repaint only trail-connected bordered sprites using center-matched solid sources while keeping all bystanders.

    Input structure:
    - The grid contains background, one grey (5) one-stroke trail, several two-color bordered sprites, and 1-2 detached solid 2x2 source blocks.
    - Only bordered sprites can connect to the grey trail: each valid destination has a grey endpoint two steps from its center with a border-colored intermediate cell.

    Rule:
    - Use the two grey trail endpoints to identify the two destination bordered sprites.
    - For each destination, find the solid source block whose single color equals that destination's center color.
    - If a match exists, repaint the entire destination sprite with that matched color.
    - Leave every non-destination object untouched; only the two endpoint-connected sprites may change.

    Key insight:
    - The destination center color is now the lookup key, and the matching source contributes a single monochrome color.
      Because key and payload are the same visible color, the observable effect shifts from "replace one center pixel" to "collapse the whole connected sprite into that matched solid color."
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"
    sources = _detect_solid_sources(g, bg, grey)

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

    by_color: dict[int, list[dict]] = {}
    for src in sources:
        by_color.setdefault(int(src["color"]), []).append(src)

    out = g.copy()
    for d in dest:
        want_color = int(d["center"])
        srcs = by_color.get(want_color, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination center should match at most one solid source"
        for r, c in d["pts"]:
            out[r, c] = want_color

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
GEN_RETRIES = 400


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


def _make_solid_patch(shape_type: str, color: int) -> np.ndarray:
    """Return the output-form destination sprite: same mask, one color everywhere."""
    return _make_bordered_patch(shape_type, color, color)


def _make_source_patch(color: int, size: int = 2) -> np.ndarray:
    return np.full((size, size), color, dtype=int)


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


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, allow_touch_grey: bool = True) -> bool:
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
    """Generate a winding 8-connected grey path."""
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
            chosen = random.choice(pool) if random.random() < PATH_MODERATE_RANDOM_P else min(pool, key=score)
        else:
            best = sorted(moves, key=score)
            top = [m for m in best if score(m) <= score(best[0]) + 1]
            dtop = [m for m in top if m[2]]
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


def _find_open_port_dir(grid: np.ndarray, cr: int, cc: int) -> tuple[int, int] | None:
    """Find a cardinal direction where the sprite has a cell at distance 1 and empty space at distance 2."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] != BG and grid[er, ec] == BG:
            return (er, ec)
    return None


def _has_grey_port(grid: np.ndarray, cr: int, cc: int, border_color: int) -> bool:
    """True iff the bordered sprite centered at (cr,cc) connects to any grey cell via a valid port."""
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == border_color and grid[er, ec] == GREY:
            return True
    return False


def _pick_layout_counts() -> tuple[int, int]:
    total_objects = random.choice(TOTAL_OBJECT_COUNTS)
    valid_sources = [n for n in SOURCE_COUNTS if total_objects - n >= 2]
    num_sources = random.choice(valid_sources)
    num_bordered = total_objects - num_sources
    return num_bordered, num_sources


def _assign_colors(border_colors: list[int], num_sources: int) -> dict | None:
    palette = [c for c in range(1, 10) if c != GREY]
    non_border = [c for c in palette if c not in border_colors]
    if len(non_border) < num_sources:
        return None

    source_colors = random.sample(non_border, num_sources)
    if num_sources == 1:
        dest_center_colors = [source_colors[0], source_colors[0]]
    else:
        dest_center_colors = list(source_colors)

    distractor_centers = []
    for border_color in border_colors[2:]:
        forbidden = {BG, GREY, border_color, *source_colors}
        candidates = [c for c in palette if c not in forbidden]
        if not candidates:
            return None
        distractor_centers.append(random.choice(candidates))

    return {
        "source_colors": source_colors,
        "dest_center_colors": dest_center_colors,
        "distractor_centers": distractor_centers,
    }


def _place_destinations(
    output_grid: np.ndarray,
    shape_type: str,
    dest_colors: list[int],
    min_port_dist: int,
) -> list[dict] | None:
    all_pos = _bordered_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    placed: list[dict] = []
    first_port = None
    for i, color in enumerate(dest_colors):
        patch = _make_solid_patch(shape_type, color)
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)

        success = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_patch(output_grid, patch, r0, c0)
            port = _find_open_port_dir(output_grid, cr, cc)
            if port is None:
                _erase_patch(output_grid, patch, r0, c0)
                continue
            if i == 1:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_patch(output_grid, patch, r0, c0)
                    continue
            placed.append({"r0": r0, "c0": c0, "center_pos": (cr, cc), "port": port})
            if i == 0:
                first_port = port
            success = True
            break
        if not success:
            return None

    return placed


def _restore_input_destinations(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    dest_center_colors: list[int],
    dest_layout: list[dict],
) -> np.ndarray:
    input_grid = output_grid.copy()
    for i, info in enumerate(dest_layout):
        patch = _make_bordered_patch(shape_type, border_colors[i], dest_center_colors[i])
        _erase_patch(input_grid, _make_solid_patch(shape_type, dest_center_colors[i]), info["r0"], info["c0"])
        _place_patch(input_grid, patch, info["r0"], info["c0"])
    return input_grid


def _place_preserved_bordered(
    output_grid: np.ndarray,
    input_grid: np.ndarray,
    shape_type: str,
    sprites: list[tuple[int, int]],
) -> bool:
    positions = _bordered_positions()
    for border_color, center_color in sprites:
        patch = _make_bordered_patch(shape_type, border_color, center_color)
        pool = list(positions)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0, allow_touch_grey=False):
                continue
            _place_patch(output_grid, patch, r0, c0)
            cr, cc = r0 + 1, c0 + 1
            if _has_grey_port(output_grid, cr, cc, border_color):
                _erase_patch(output_grid, patch, r0, c0)
                continue
            _place_patch(input_grid, patch, r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_preserved_sources(output_grid: np.ndarray, input_grid: np.ndarray, colors: list[int]) -> bool:
    positions = _solid_positions(size=2)
    for color in colors:
        patch = _make_source_patch(color, size=2)
        pool = list(positions)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(output_grid, patch, r0, c0, allow_touch_grey=False):
                continue
            _place_patch(output_grid, patch, r0, c0)
            _place_patch(input_grid, patch, r0, c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    num_bordered, num_sources = _pick_layout_counts()

    palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(palette, num_bordered)
    colors = _assign_colors(border_colors, num_sources)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_layout = _place_destinations(output_grid, shape_type, colors["dest_center_colors"], min_port_dist)
    if dest_layout is None:
        return None

    port_cells = [info["port"] for info in dest_layout]
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

    input_grid = _restore_input_destinations(
        output_grid,
        shape_type,
        border_colors,
        colors["dest_center_colors"],
        dest_layout,
    )

    bordered_distractors = list(zip(border_colors[2:], colors["distractor_centers"], strict=False))
    if not _place_preserved_bordered(output_grid, input_grid, shape_type, bordered_distractors):
        return None
    if not _place_preserved_sources(output_grid, input_grid, colors["source_colors"]):
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
