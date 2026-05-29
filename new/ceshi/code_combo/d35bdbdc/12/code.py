# Combo variant of d35bdbdc (index 12)
# combo_spec: D3/3+D5/0
# Name: 空心框架精灵+边框色转移
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and traced example
# - Keep the original grey-path mechanism: the two trail endpoints still choose
#   the only two surviving destination sprites.
# - Keep destinations as the old small bordered sprites, because the port rule
#   depends on a visible center pixel plus a border cell one step away.
# - Change only the source family: every usable source is now a fully visible
#   5x5 hollow square frame whose ring is one color and whose exact middle cell
#   is a different single pixel. The eight other interior cells stay background.
# - Change the transferred attribute: the destination center still serves as the
#   lookup key, but a match now means "some hollow frame has this ring color",
#   and the written attribute is that frame's ring color onto the destination
#   border, not the source center into the destination center.
# - Small trace:
#   original seed logic:
#     dest border=3, dest center=7; another bordered sprite has border=7, center=2
#     -> output becomes border=3, center=2
#   fused logic here:
#     dest border=3, dest center=7; elsewhere a hollow frame has ring=7, center-dot=2
#     -> output becomes border=7, center=7
#   The center dot inside the hollow source is visible but irrelevant to the
#   transfer; what matters is the frame ring itself.
#
# b) Complexity calibration
# - A 3x3 "hollow frame" would collapse back into the original box appearance,
#   so the source must become larger. A 5x5 ring is the smallest clean version
#   that is visually unmistakable as hollow.
# - Two 5x5 frames plus the original 4-5 sprites do not fit cleanly on a 10x10
#   canvas with a long grey trail, so the generator reduces the scene to the two
#   destinations plus one hollow-frame source. This keeps the D3 change visible
#   without making placement impossible.
# - To keep the source logically necessary, the generator does not always let
#   both destinations match. Most samples have exactly one matching destination,
#   so the output cannot be explained by a purely local "border becomes center"
#   rule.
#
# c) Implicit assumption audit against the seed solver
# - `_detect_bordered_sprites` still works for destinations because those objects
#   keep the old two-color structure.
# - The seed lookup table over bordered sprites is no longer valid for sources.
#   Hollow-frame sources are split into two connected components: the ring and
#   the center dot. The solver therefore needs a new detector that recovers a
#   5x5 ring component and reads the single dot from the frame cavity.
# - The old write step changed one center cell. That is invalid after D5/0. The
#   variant must repaint every non-center cell of a matched destination.
# - Cleanup by "keep only destination-connected colored cells" still works and
#   conveniently removes both the frame ring and its detached center dot.
#
# d) Placement legality
# - Destinations may still be edge-clipped 3x3 boxes or pluses, so the port rule
#   remains identical to the seed task.
# - Hollow-frame sources are always placed fully in-bounds. This avoids clipped
#   frame detection ambiguities and guarantees the exact middle dot exists.
# - The generator reserves source cells before placing destinations and the grey
#   trail, so the later input reconstruction cannot overlap anything.
#
# e) Visual observability
# - A human sees two object families: path-touching bordered sprites and one
#   larger hollow square elsewhere.
# - The output keeps only the path-touching pair. When a surviving sprite's
#   center color is also seen as the ring color of the hollow frame, its border
#   adopts that color; otherwise it stays unchanged.
# - No hidden ordering by color id is used. The evidence is entirely visible in
#   the grid.
#
# f) Visual cleanliness
# - Only one hollow frame is used, which keeps the 10x10 scene readable instead
#   of crowding the path corridor.
# - The frame's empty cavity and isolated center dot make it visually distinct
#   from the filled/clipped destination sprites.
# - The output stays sparse because every non-destination colored component is
#   erased.
#
# g) Color-change inferability
# - The output change is structured: border recoloring occurs iff the
#   destination's center color is witnessed as a hollow-frame ring elsewhere.
# - The generator biases toward exactly one matched destination, giving repeated
#   support that the source frame is a gating cue rather than decoration.
#
# h) Non-degeneracy metric used during validation
# - `matched_dest_count`: how many destinations have a center color equal to a
#   hollow-frame ring color. The generator enforces `matched_dest_count >= 1`
#   and heavily favors `matched_dest_count == 1`.
# - `border_change_count`: how many destinations actually change border color.
#   This equals `matched_dest_count` here and must be positive.
# - `naive_local_match`: whether the output equals the weaker rule "every
#   destination border simply changes to its own center color". Samples with
#   exactly one match make this false and prove the hollow frame is operative.
#
# i) New solver assumptions guaranteed by the generator
# - There is exactly one fully visible 5x5 hollow-frame source.
# - Hollow-frame ring colors are unique, so a destination center matches at most
#   one source.
# - Every destination is a valid 2-color bordered sprite with a unique center.
# - The hollow frame never overlaps or 4-touches destinations or the grey trail.

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


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
BLOCKED = -1
SPRITE_SHAPES = ("box", "plus")
MATCH_COUNT_WEIGHTS = [3, 1]  # weights for [1, 2] matched destinations
INTERIOR_DEST_WEIGHTS = [3, 1]  # weights for [0, 1] interior destinations
PATH_LEN_MIN, PATH_LEN_MAX = 8, 18
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 600


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _component_bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


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
    Destination sprite = one 4-connected component (excluding bg/grey) with exactly 2 colors:
      - one border color appearing multiple times
      - one center color appearing exactly once
    Clipped edge placements are allowed as long as the unique center pixel remains visible.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts_arr in comps:
        pts = [(int(r), int(c)) for r, c in pts_arr]
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
                center_pos = (r, c)
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


def _detect_hollow_frame_sources(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Hollow source = a fully visible 5x5 monochrome ring component plus one colored dot at
    the exact center of its cavity. The eight other interior cells must be background.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sources: list[dict] = []
    for pts_arr in comps:
        pts = [(int(r), int(c)) for r, c in pts_arr]
        cols = {int(g[r, c]) for r, c in pts}
        if len(cols) != 1:
            continue
        border_color = next(iter(cols))
        if border_color in (bg, grey):
            continue
        r0, c0, r1, c1 = _component_bbox(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            continue
        ring = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(pts) != ring:
            continue

        cr, cc = r0 + 2, c0 + 2
        center_color = int(g[cr, cc])
        if center_color in (bg, grey, border_color):
            continue

        ok = True
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                val = int(g[r, c])
                if (r, c) == (cr, cc):
                    if val != center_color:
                        ok = False
                        break
                elif val != bg:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        sources.append(
            {
                "border": border_color,
                "center": center_color,
                "frame_pts": sorted(ring),
                "center_pos": (cr, cc),
            }
        )
    return sources


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint is a port for this bordered sprite if:
      - it is exactly 2 steps away from the unique center in a cardinal direction
      - the intermediate cell is the sprite's border color
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


def _recolor_sprite_border(out: np.ndarray, sprite: dict, new_border: int) -> None:
    """Recolor every non-center cell of a detected bordered sprite."""
    center_pos = tuple(sprite["center_pos"])
    for r, c in sprite["pts"]:
        if (r, c) != center_pos:
            out[r, c] = int(new_border)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor path-selected sprite borders from hollow-frame sources, then erase every unused object.

    Input structure:
    - The grid contains background, one grey (5) one-stroke trail, exactly two path-touching
      bordered destination sprites, and one or more larger hollow square frames elsewhere.
    - Each destination is a clipped 3x3-like box or plus with one repeated border color and a
      single different center pixel.
    - Each hollow source is a fully visible 5x5 square ring in one color, with only one extra
      colored pixel at the exact middle of its cavity.

    Rule:
    - Use the two grey trail endpoints to identify the destination sprites through the original
      port rule.
    - For each destination, use its center color as a key and search the hollow sources for a
      frame whose ring color matches that key.
    - If a match exists, recolor every non-center cell of that destination to the source frame's
      ring color while keeping the destination center unchanged.
    - Delete every hollow frame, every non-destination bordered sprite, and any other non-grey
      colored component.

    Key insight:
    - The lookup token still sits at the destination center, but the transferable attribute now
      lives on a visually different source object family: a hollow frame. A destination changes
      only when its center color is witnessed as a frame-ring color elsewhere in the input.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert len(sprites) >= 2, "expected at least two bordered destination candidates"

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
        assert cands, "trail endpoint has no compatible bordered sprite"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 destinations from trail endpoints"

    by_frame_border: dict[int, list[dict]] = {}
    for src in _detect_hollow_frame_sources(g, bg, grey):
        by_frame_border.setdefault(int(src["border"]), []).append(src)

    out = g.copy()
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
        srcs = by_frame_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one hollow-frame border"
        _recolor_sprite_border(out, d, int(srcs[0]["border"]))

    return out


def _make_dest_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 destination patch for a box or plus."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _make_hollow_source_patch(border_color: int, center_color: int) -> np.ndarray:
    """Return a 5x5 hollow square ring with one center dot."""
    patch = np.zeros((5, 5), dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    patch[2, 2] = center_color
    return patch


def _reserve_patch(patch: np.ndarray) -> np.ndarray:
    """Turn a visible patch into an occupancy-only blocker mask."""
    out = np.zeros_like(patch)
    out[patch != BG] = BLOCKED
    return out


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(patch.shape[0]):
        for dc in range(patch.shape[1]):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and 4-touching with other non-background / non-grey cells."""
    h, w = grid.shape
    patch_cells: set[tuple[int, int]] = set()
    for dr in range(patch.shape[0]):
        for dc in range(patch.shape[1]):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] != BG:
                    return True
                patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in [(rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)]:
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, GREY):
                return True
    return False


def _dest_positions() -> list[tuple[int, int]]:
    """All valid top-lefts for a clipped 3x3 destination whose center stays on-grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _source_positions() -> list[tuple[int, int]]:
    """All valid top-lefts for a fully visible 5x5 hollow source."""
    return [(r0, c0) for r0 in range(GRID_H - 4) for c0 in range(GRID_W - 4)]


def _is_edge_dest(r0: int, c0: int) -> bool:
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
    """Generate a winding 8-connected grey path on currently empty cells."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(cell: tuple[int, int, bool]) -> int:
        return abs(cell[0] - end[0]) + abs(cell[1] - end[1])

    for _ in range(target_len + 20):
        if max(abs(cr - end[0]), abs(cc - end[1])) <= 1:
            break

        moves: list[tuple[int, int, bool]] = []
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
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
    return len(seen) == len(cell_set)


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, border_color: int) -> tuple[int, int] | None:
    """Find a valid grey endpoint cell for a destination centered at (cr,cc)."""
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


def _border_offsets(shape_type: str) -> list[tuple[int, int]]:
    if shape_type == "box":
        return [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    return [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _paint_dest_border(grid: np.ndarray, shape_type: str, center_pos: tuple[int, int], color: int) -> None:
    """Paint the visible border cells of a possibly clipped destination."""
    cr, cc = center_pos
    for dr, dc in _border_offsets(shape_type):
        rr, cc2 = cr + dr, cc + dc
        if 0 <= rr < GRID_H and 0 <= cc2 < GRID_W and grid[rr, cc2] != BG:
            grid[rr, cc2] = color


def _choose_colors() -> dict | None:
    """Choose colors while keeping the hollow-frame lookup visible and unambiguous."""
    palette = [c for c in range(1, 10) if c != GREY]
    source_border = random.choice(palette)
    remaining = [c for c in palette if c != source_border]
    if len(remaining) < 3:
        return None

    dest_borders = random.sample(remaining, 2)
    unmatched_choices = [c for c in remaining if c not in dest_borders]
    if not unmatched_choices:
        return None

    match_count = random.choices([1, 2], weights=MATCH_COUNT_WEIGHTS)[0]
    matched_indices = set(random.sample([0, 1], match_count))
    dest_centers: list[int] = []
    for i in range(2):
        if i in matched_indices:
            dest_centers.append(source_border)
        else:
            choices = [c for c in unmatched_choices if c != dest_borders[i]]
            if not choices:
                return None
            dest_centers.append(random.choice(choices))

    source_center_choices = [c for c in palette if c != source_border]
    if not source_center_choices:
        return None

    return {
        "source_border": source_border,
        "source_center": random.choice(source_center_choices),
        "dest_borders": dest_borders,
        "dest_centers": dest_centers,
        "matched_indices": matched_indices,
    }


def _place_reserved_source(plan_grid: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    reserve = _reserve_patch(patch)
    positions = _source_positions()
    random.shuffle(positions)
    for r0, c0 in positions:
        if _overlaps(plan_grid, reserve, r0, c0):
            continue
        _place_sprite(plan_grid, reserve, r0, c0)
        return (r0, c0)
    return None


def _place_dests(
    plan_grid: np.ndarray,
    shape_type: str,
    output_dest_borders: list[int],
    dest_centers: list[int],
    min_port_dist: int,
) -> list[tuple[int, int]] | None:
    """Place the two destinations on the planning grid."""
    all_pos = _dest_positions()
    edge_pos = [p for p in all_pos if _is_edge_dest(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge_dest(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_centers_out: list[tuple[int, int]] = []
    first_port = None

    for i in range(2):
        patch = _make_dest_patch(shape_type, output_dest_borders[i], dest_centers[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(plan_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(plan_grid, patch, r0, c0)
            port = _find_port_dir(plan_grid, cr, cc, output_dest_borders[i])
            if port is None:
                _erase_sprite(plan_grid, patch, r0, c0)
                continue
            if i == 1:
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(plan_grid, patch, r0, c0)
                    continue
            dest_centers_out.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break
        if not placed:
            return None

    return dest_centers_out


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(SPRITE_SHAPES)
    colors = _choose_colors()
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    source_patch = _make_hollow_source_patch(colors["source_border"], colors["source_center"])
    plan_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    source_pos = _place_reserved_source(plan_grid, source_patch)
    if source_pos is None:
        return None

    output_dest_borders = [
        colors["source_border"] if i in colors["matched_indices"] else colors["dest_borders"][i]
        for i in range(2)
    ]
    dest_positions = _place_dests(
        plan_grid,
        shape_type,
        output_dest_borders,
        colors["dest_centers"],
        min_port_dist,
    )
    if dest_positions is None:
        return None

    port_cells = []
    for i in range(2):
        cr, cc = dest_positions[i]
        port = _find_port_dir(plan_grid, cr, cc, output_dest_borders[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(plan_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None

    for r, c in path_cells:
        if plan_grid[r, c] != BG:
            return None
        plan_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and plan_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    output_grid = plan_grid.copy()
    output_grid[output_grid == BLOCKED] = BG

    input_grid = output_grid.copy()
    for i in range(2):
        _paint_dest_border(input_grid, shape_type, dest_positions[i], colors["dest_borders"][i])
    _place_sprite(input_grid, source_patch, source_pos[0], source_pos[1])

    if not colors["matched_indices"]:
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
