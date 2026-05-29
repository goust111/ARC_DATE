# Combo variant of d35bdbdc (index 3)
# combo_spec: D1/1+D13/0
# Name: 边框色匹配+整体精灵复制

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste

# Design analysis for the combo variant
# a) Concreteization + behavior trace:
#    Keep the original scene grammar: a thin grey trail chooses exactly two destination
#    sprites through endpoint ports, while the other sprites sit elsewhere. The fused
#    change alters both "who matches whom" and "what gets transferred". Example:
#      - destination = full box(border=2, center=7), connected to a grey endpoint
#      - source      = clipped box(border=2, center=8), sitting unconnected at an edge
#    Original task:
#      - destination looks up source by destination.center == source.border, so the
#        example above would not even match unless 7 happened to be a border elsewhere
#      - even when matched, only one center pixel would change
#    Fused task:
#      - destination now matches the unconnected sprite with the same border color 2
#      - destination is erased and redrawn as the whole observed source footprint,
#        centered at the destination center, so a full 3x3 destination can become a
#        clipped L-shape or 2x2 corner copy. The difference is object-level, not just
#        center-color-level.
# b) Complexity calibration:
#    The generator uses exactly two meaningful border duplications: each of the two
#    destination border colors appears once more on its paired source. Any optional fifth
#    sprite gets a unique border. This keeps same-border pairing unambiguous and visually
#    supported without filling the grid with many repeated outlines. Destination centers
#    are chosen outside the border palette so the old border<-center lookup is visibly
#    wrong on every sample.
# c) Hidden assumptions from the original main():
#    The original solver assumed border colors were unique, so `by_border` could include
#    destinations without causing self-matches. That breaks under D1/1, because every
#    destination would match itself first. The combo solver therefore builds the lookup
#    only from non-destination sprites. A second assumption was that preserving the
#    destination border was correct; D13/0 breaks that too, so the final write can no
#    longer be `out[center] = src.center`. We must clear destination pixels and redraw
#    the translated source footprint.
# d) Placement legality:
#    Destination sprites are placed first with full 3x3 support so path ports are easy to
#    realize and the copied-shape effect remains visible. The grey path is then drawn
#    between valid ports. Only after the path exists are sources and distractors placed.
#    Sources are forced onto clipped edge/corner placements and rejected if they overlap,
#    4-touch another sprite, or share a grey endpoint as one of their own ports.
# e) Visual observability:
#    A human now sees two cues at once: repeated border colors indicate the pairing, and
#    the surviving path-connected sprite changes its entire silhouette to match the
#    disconnected same-border sprite. Because destination centers are not border colors,
#    the old cross-attribute rule is not supported by the examples.
# f) Visual cleanliness:
#    The grid still contains only 4 or 5 small sprites and one thin grey trail. Repeated
#    borders are limited to the two source/destination pairs. Sources are separated from
#    the path endpoints to avoid endpoint-ownership ambiguity, and the copied partial
#    shapes remain easy to read because destinations start as full sprites.
# g) Inferability of the change:
#    The generator guarantees at least one non-center pixel difference between the fused
#    output and a center-only same-border transfer: both meaningful sources are clipped,
#    while both destinations are full sprites. Thus D13/0 is not latent; object copying is
#    necessary to explain the outputs, and D1/1 is necessary because destination centers
#    never name the source border colors.


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

    def dfs(
        cur: tuple[int, int], visited: frozenset[tuple[int, int]]
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
                "pts": [(int(r), int(c)) for r, c in pts],
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


def _select_destinations(g: np.ndarray, grey: int, sprites: list[dict]) -> list[dict]:
    """Resolve the two destination sprites chosen by the grey trail endpoints."""
    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    opts = {
        ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)]
        for ep in eps_pair
    }

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
    return dest


def _translated_sprite_cells(sprite: dict, dst_center: tuple[int, int]) -> list[tuple[int, int, int]]:
    """Translate an observed sprite so that its center lands at dst_center."""
    src_r, src_c = sprite["center_pos"]
    dst_r, dst_c = dst_center
    cells: list[tuple[int, int, int]] = []
    for r, c in sprite["pts"]:
        rr = int(dst_r + (int(r) - src_r))
        cc = int(dst_c + (int(c) - src_c))
        color = int(sprite["center"]) if (int(r), int(c)) == (src_r, src_c) else int(sprite["border"])
        cells.append((rr, cc, color))
    return cells


def _redraw_destination_as_source(out: np.ndarray, dest: dict, src: dict, bg: int) -> None:
    """Replace the destination component with the translated source component."""
    for r, c in dest["pts"]:
        out[int(r), int(c)] = bg
    for rr, cc, color in _translated_sprite_cells(src, dest["center_pos"]):
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = color


def _center_only_same_border_baseline(grid: np.ndarray) -> np.ndarray:
    """Diagnostic baseline: same-border match but only transfer the source center color."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5
    sprites = _detect_bordered_sprites(g, bg, grey)
    dest = _select_destinations(g, grey, sprites)
    dest_ids = {id(sp) for sp in dest}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
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
                if keep[nr, nc] or out[nr, nc] in (bg, grey):
                    continue
                keep[nr, nc] = True
                q.append((nr, nc))
    out[(out != bg) & (out != grey) & (~keep)] = bg

    for d in dest:
        srcs = by_border.get(int(d["border"]), [])
        if not srcs:
            continue
        assert len(srcs) == 1
        dr, dc = d["center_pos"]
        out[dr, dc] = int(srcs[0]["center"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the two trail-selected destination sprites by whole same-border source sprites, keeping only path-connected survivors.

    Input structure:
    - The grid contains background, a thin grey (5) one-stroke trail, and several small two-color bordered sprites.
    - Exactly two sprites are attached to the two grey trail endpoints through the original port rule; these are the destinations.
    - Other sprites are disconnected from the trail, and some of them reuse a destination's border color.

    Rule:
    - Identify the two destinations from the grey trail endpoints.
    - For each destination, find the non-destination sprite whose border color equals that destination's border color.
    - Delete every non-destination sprite and any remaining stray non-grey component, then erase each destination and redraw the entire matched source sprite centered at the destination center.

    Key insight:
    - The grey trail only tells you which two sprites survive; the actual pairing cue is "same border color", and the payload is the whole observed source shape rather than just its center pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"

    dest = _select_destinations(g, grey, sprites)
    dest_ids = {id(sp) for sp in dest}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        if id(sp) in dest_ids:
            continue
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
        srcs = by_border.get(int(d["border"]), [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination border should match exactly one non-destination source border"
        _redraw_destination_as_source(out, d, srcs[0], bg)

    return out


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a centered 3x3 box or plus patch."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Place patch cells on grid, skipping transparent background and clipping to bounds."""
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    """Undo a patch placement by clearing every non-background patch cell."""
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and int(patch[dr, dc]) != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Check if placing patch at (r0,c0) would overlap or 4-touch another sprite."""
    h, w = grid.shape
    patch_cells = set()
    for dr in range(3):
        for dc in range(3):
            if int(patch[dr, dc]) == BG:
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
    """All valid (r0,c0) where the sprite center remains on-grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _full_sprite_positions() -> list[tuple[int, int]]:
    """Placements where the full 3x3 sprite is in-bounds."""
    return [(r0, c0) for r0 in range(0, GRID_H - 2) for c0 in range(0, GRID_W - 2)]


def _clipped_sprite_positions() -> list[tuple[int, int]]:
    """Placements where the sprite is partially clipped by the grid boundary."""
    full = set(_full_sprite_positions())
    return [pos for pos in _sprite_positions() if pos not in full]


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
                    and int(grid[nr, nc]) == BG
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
    """Find one usable port cell for the sprite centered at (cr,cc)."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == bc and int(grid[er, ec]) == BG:
            return (er, ec)
    return None


def _ports_from_center(grid: np.ndarray, cr: int, cc: int, bc: int) -> set[tuple[int, int]]:
    """All potential port cells for a placed sprite centered at (cr,cc)."""
    ports: set[tuple[int, int]] = set()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == bc:
            ports.add((er, ec))
    return ports


def _observed_sprite_offsets(
    shape_type: str, r0: int, c0: int
) -> set[tuple[int, int]]:
    """Relative offsets of all visible sprite cells from the center for a clipped/full placement."""
    patch = _make_sprite_patch(shape_type, 1, 2)
    cr, cc = r0 + 1, c0 + 1
    offsets: set[tuple[int, int]] = set()
    for dr in range(3):
        for dc in range(3):
            if int(patch[dr, dc]) == BG:
                continue
            rr, cc2 = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc2 < GRID_W:
                offsets.add((rr - cr, cc2 - cc))
    return offsets


def _assign_colors(dest_borders: list[int], distractor_count: int) -> dict | None:
    """Choose centers so old lookup fails and the two source copies remain visible."""
    used_borders = set(dest_borders)
    available = [c for c in range(1, 10) if c != GREY and c not in used_borders]
    needed = 4 + distractor_count
    if len(available) < needed:
        return None
    random.shuffle(available)
    dest_center_colors = available[:2]
    source_center_colors = available[2:4]
    distractor_center_colors = available[4 : 4 + distractor_count]
    return {
        "dest_center_colors": dest_center_colors,
        "source_center_colors": source_center_colors,
        "distractor_center_colors": distractor_center_colors,
    }


def _place_destinations(
    input_grid: np.ndarray,
    shape_type: str,
    dest_borders: list[int],
    dest_centers: list[int],
    min_port_dist: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place two full destination sprites and return top-lefts, centers, and ports."""
    pools = list(_full_sprite_positions())
    random.shuffle(pools)

    placed_r0s: list[tuple[int, int]] = []
    centers: list[tuple[int, int]] = []
    ports: list[tuple[int, int]] = []

    for i in range(2):
        bc = dest_borders[i]
        patch = _make_sprite_patch(shape_type, bc, dest_centers[i])
        candidates = list(_full_sprite_positions())
        random.shuffle(candidates)
        ok = False
        for r0, c0 in candidates:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(input_grid, patch, r0, c0)
            port = _find_port_dir(input_grid, cr, cc, bc)
            if port is None:
                _erase_sprite(input_grid, patch, r0, c0)
                continue
            if i == 1:
                dist = abs(port[0] - ports[0][0]) + abs(port[1] - ports[0][1])
                if dist < min_port_dist:
                    _erase_sprite(input_grid, patch, r0, c0)
                    continue
            placed_r0s.append((r0, c0))
            centers.append((cr, cc))
            ports.append(port)
            ok = True
            break
        if not ok:
            return None
    return placed_r0s, centers, ports


def _place_other_sprite(
    input_grid: np.ndarray,
    patch: np.ndarray,
    candidate_positions: list[tuple[int, int]],
    forbidden_ports: set[tuple[int, int]],
    shape_type: str,
    dest_r0: tuple[int, int] | None = None,
    require_shape_change: bool = False,
) -> tuple[int, int] | None:
    """Place a non-destination sprite while avoiding endpoint ambiguity."""
    random.shuffle(candidate_positions)
    bc = int(patch[patch != BG][0])
    for r0, c0 in candidate_positions:
        if _overlaps(input_grid, patch, r0, c0):
            continue
        if require_shape_change and dest_r0 is not None:
            if _observed_sprite_offsets(shape_type, r0, c0) == _observed_sprite_offsets(shape_type, *dest_r0):
                continue
        _place_sprite(input_grid, patch, r0, c0)
        cr, cc = r0 + 1, c0 + 1
        if forbidden_ports & _ports_from_center(input_grid, cr, cc, bc):
            _erase_sprite(input_grid, patch, r0, c0)
            continue
        return r0, c0
    return None


def _gen_attempt() -> np.ndarray | None:
    shape_type = random.choice(["box", "plus"])
    num_sprites = random.choice(SPRITE_COUNTS)
    distractor_count = num_sprites - 4

    palette = [c for c in range(1, 10) if c != GREY]
    dest_borders = random.sample(palette, 2)
    distractor_borders = random.sample([c for c in palette if c not in dest_borders], distractor_count)
    colors = _assign_colors(dest_borders, distractor_count)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    input_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_data = _place_destinations(
        input_grid,
        shape_type,
        dest_borders,
        colors["dest_center_colors"],
        min_port_dist,
    )
    if dest_data is None:
        return None
    dest_r0s, dest_centers, port_cells = dest_data

    path_cells = _gen_winding_path(input_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None
    for r, c in path_cells:
        if int(input_grid[r, c]) != BG:
            return None
        input_grid[r, c] = GREY
    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and int(input_grid[nr, nc]) == GREY
        )
        if deg > 1:
            return None

    trail_eps = set(port_cells)

    clipped_positions = _clipped_sprite_positions()
    for i in range(2):
        patch = _make_sprite_patch(shape_type, dest_borders[i], colors["source_center_colors"][i])
        placed = _place_other_sprite(
            input_grid,
            patch,
            list(clipped_positions),
            trail_eps,
            shape_type,
            dest_r0=dest_r0s[i],
            require_shape_change=True,
        )
        if placed is None:
            return None

    all_positions = _sprite_positions()
    for i in range(distractor_count):
        patch = _make_sprite_patch(
            shape_type,
            distractor_borders[i],
            colors["distractor_center_colors"][i],
        )
        placed = _place_other_sprite(
            input_grid,
            patch,
            list(all_positions),
            trail_eps,
            shape_type,
        )
        if placed is None:
            return None

    try:
        out = main(input_grid)
        baseline = _center_only_same_border_baseline(input_grid)
        if np.array_equal(out, baseline):
            return None
        non_center_diff = np.count_nonzero((out != baseline) & (out != GREY) & (baseline != GREY))
        if non_center_diff < 2:
            return None
        return input_grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
