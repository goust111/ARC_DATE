# Variant of d35bdbdc
# Dimension: D3 — 源对象类型 (src)
# Change: 带中心色和边框色的精灵（有两层颜色结构：外层边框色、内层中心色）... → 单色实心精灵

# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete variant and behavior trace
# - Keep the original destination mechanism: two bordered sprites are selected by
#   the endpoints of the grey one-stroke path.
# - Keep bordered sprites as the only path-connectable objects. This preserves
#   the original spatial cue and makes the D3 change isolated to source objects.
# - Replace source sprites with monochrome solid 2x2 blocks. A source no longer
#   has separate border/center colors; its only visible attribute is its single
#   color.
# - Matching rule is therefore redefined: a source block matches a destination
#   when source.color == destination.border.
# - Action after a match: repaint the destination center with the source color.
#   Because the source color equals the destination border, the destination
#   becomes monochrome in the output. If no source matches a destination border,
#   that destination center stays unchanged.
# - Small trace:
#   original task at one destination:
#     dest border=3, dest center=8, some other sprite has border=8 and center=2
#     output center becomes 2
#   new variant at one destination:
#     dest border=3, dest center=8, separate solid source block color=3
#     output center becomes 3
#   difference:
#     original logic follows center_color -> source.border -> source.center
#     new logic follows border_color -> solid source color, with no second color
#     inside the source object.
#
# b) Complexity calibration
# - A monochrome source that matches destination.center would produce no visible
#   change, so the match key must be a different destination attribute.
# - Matching to destination.border keeps the task visually legible: the center
#   either collapses into the border color or stays as-is.
# - To avoid making the source blocks irrelevant, the generator uses 1 or 2
#   sources for 2 destinations. This yields both branches:
#   - one destination changes, one stays unchanged
#   - both destinations change
# - Destination center colors are chosen to avoid every bordered sprite border
#   color. That prevents accidental fallback to the original "center looks up a
#   bordered source" mechanism.
#
# c) Implicit assumption audit against the original solver
# - _detect_bordered_sprites still works for destinations and bordered
#   distractors because those objects keep the old two-color structure.
# - The original by_border lookup over bordered sprites cannot represent the new
#   source type, so the solver must add a separate detector for monochrome
#   source blocks.
# - The original destination update used d["center"] as the lookup key; this is
#   invalid now because a source has no inner color to transfer. The new solver
#   must use d["border"] as the lookup key and source.color as the written value.
# - Cleanup logic that deletes non-destination colored components remains valid,
#   but it must explicitly remove both bordered distractors and solid sources.
#
# d) Placement legality
# - Bordered sprites keep the original 3x3 box/plus family and can still be
#   clipped at edges.
# - Solid sources are fixed 2x2 filled squares placed fully in-bounds. This
#   makes source detection reliable and visually distinct from bordered sprites.
# - Non-destination objects are placed so they neither overlap nor 4-touch other
#   non-background, non-grey objects. On the input grid, they also avoid touching
#   the grey path to keep endpoint-object relations readable.
# - The grey trail is still generated after destinations are placed, so port
#   validity is checked against the real destination border geometry.
#
# e) Visual observability
# - A human can see that only the two path-endpoint bordered sprites survive.
# - The source blocks are the only fully solid non-grey objects in the input.
# - When a source block color matches a surviving sprite border, that sprite's
#   contrasting center disappears in the output; unmatched destinations keep the
#   contrasting center. This is observable in 2-3 examples.
#
# f) Visual cleanliness
# - Source blocks are small and isolated from the path.
# - Bordered distractors keep the original visual language, so the image still
#   resembles the seed task instead of becoming a mixed bag of unrelated shapes.
# - Because only destinations remain in the output, the final grid stays sparse.
#
# g) Color-change inferability
# - The output change is not an arbitrary recoloring. It is always "center turns
#   into the border color if a same-colored solid source block exists elsewhere."
# - The number of changed destinations is only 1 or 2, which is small enough for
#   examples to support the rule without introducing many case distinctions.
#
# h) Post-change solver assumptions that the generator must guarantee
# - Every source block is a monochrome filled 2x2 component.
# - Source block colors are unique, so a border color matches at most one source.
# - Destination border colors are unique, so a source can affect at most one
#   destination.
# - Destination centers avoid all bordered-sprite border colors, ensuring the
#   old solver would not accidentally simulate the new behavior.

from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
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
    This handles clipped borders near the edge, as long as the center is present.
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
        if border_color in (bg, grey) or center_color in (bg, grey):
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
                "pts": [(int(r), int(c)) for r, c in pts],
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
        cols = {int(g[r, c]) for r, c in pts_list}
        if len(cols) != 1 or len(pts_list) != 4:
            continue
        rows = [r for r, _ in pts_list]
        cols_idx = [c for _, c in pts_list]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols_idx), max(cols_idx)
        if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
            continue
        want = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if set(pts_list) != want:
            continue
        sources.append({"pts": pts_list, "color": int(next(iter(cols)))})
    return sources


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint is a port for this sprite if:
      - it is exactly 2 steps from the sprite center in a cardinal direction
      - the intermediate cell is the sprite border color
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint path-connected bordered sprites using matching monochrome source blocks, then remove the rest.

    Input structure:
    - Background, a grey (5) one-stroke trail, several two-color bordered sprites, and 1-2 solid 2x2 source blocks.
    - The grey trail endpoints still identify exactly two destination bordered sprites via the same port rule as the seed task.

    Rule:
    - Use the grey trail endpoints to choose the two destination bordered sprites.
    - For each destination, look for a solid source block whose single color equals that destination's border color.
    - If such a source exists, repaint the destination center with that color; then delete every non-destination object, including bordered distractors and source blocks.

    Key insight:
    - The source object no longer stores two colors. Instead, its single visible color acts as both the lookup key and the transferred value, so a matched destination collapses into a monochrome version of itself.
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
    dest_ids = {id(sp) for sp in dest}

    by_color: dict[int, list[dict]] = {}
    for src in sources:
        by_color.setdefault(int(src["color"]), []).append(src)

    out = g.copy()

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        for r, c in sp["pts"]:
            out[r, c] = bg

    for src in sources:
        for r, c in src["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in dest:
        for r, c in d["pts"]:
            if out[r, c] not in (bg, grey) and not keep[r, c]:
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


def _make_solid_patch(color: int, size: int = 2) -> np.ndarray:
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
    valid_sources = [n for n in SOURCE_COUNTS if total_objects - n >= 2]
    num_sources = random.choice(valid_sources)
    num_bordered = total_objects - num_sources
    return num_bordered, num_sources


def _assign_variant_colors(border_colors: list[int], num_sources: int) -> dict | None:
    palette = [c for c in range(1, 10) if c != GREY]
    non_border = [c for c in palette if c not in border_colors]
    if len(non_border) < 2:
        return None

    dest_input_centers = random.sample(non_border, 2)
    distractor_center_colors = [random.choice(non_border) for _ in border_colors[2:]]
    matched_dest_indices = sorted(random.sample([0, 1], num_sources))
    output_dest_centers = [
        border_colors[i] if i in matched_dest_indices else dest_input_centers[i]
        for i in range(2)
    ]
    source_colors = [border_colors[i] for i in matched_dest_indices]
    return {
        "dest_input_centers": dest_input_centers,
        "distractor_center_colors": distractor_center_colors,
        "matched_dest_indices": matched_dest_indices,
        "output_dest_centers": output_dest_centers,
        "source_colors": source_colors,
    }


def _place_dests(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    all_pos = _bordered_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
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


def _place_bordered_distractors(
    grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    center_colors: list[int],
) -> bool:
    all_pos = _bordered_positions()
    for border_color, center_color in zip(border_colors[2:], center_colors, strict=False):
        patch = _make_bordered_patch(shape_type, border_color, center_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if not _overlaps(grid, patch, r0, c0, allow_touch_grey=False):
                _place_patch(grid, patch, r0, c0)
                placed = True
                break
        if not placed:
            return False
    return True


def _place_sources(grid: np.ndarray, source_colors: list[int]) -> bool:
    positions = _solid_positions(size=2)
    for color in source_colors:
        patch = _make_solid_patch(color, size=2)
        pool = list(positions)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if not _overlaps(grid, patch, r0, c0, allow_touch_grey=False):
                _place_patch(grid, patch, r0, c0)
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
    colors = _assign_variant_colors(border_colors, num_sources)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
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
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and output_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    input_grid = output_grid.copy()
    for i in range(2):
        cr, cc = dest_center_positions[i]
        input_grid[cr, cc] = colors["dest_input_centers"][i]

    if not _place_bordered_distractors(
        input_grid,
        shape_type,
        border_colors,
        colors["distractor_center_colors"],
    ):
        return None
    if not _place_sources(input_grid, colors["source_colors"]):
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
