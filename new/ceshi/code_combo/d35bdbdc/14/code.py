# Combo variant of d35bdbdc (index 14)
# combo_spec: D11/2+D14/1
# Name: 按面积排序处理+未匹配目标删除
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and flow trace
# - The scene still contains a grey one-stroke trail and several bordered sprites.
#   The two trail endpoints still identify the two destination sprites through the
#   original port rule.
# - The fused change happens after destination detection. Example: one full 3x3
#   box endpoint sprite has visible area 9, one corner-clipped box endpoint
#   sprite has visible area 4, and both centers equal the same detached source
#   border color 7. If that source's center is 2, then:
#   * base task: both endpoint sprites survive and both centers become 2;
#   * D14-only: same result here, because both do have a matching source;
#   * fused rule: destinations are processed by visible area, so the area-9
#     sprite claims the only source first, becomes 2, and the area-4 sprite no
#     longer has an available match and is deleted.
# - Therefore the deletion branch is no longer driven only by "border absent";
#   it can also be caused by competition for a shared source.
#
# b) Complexity calibration
# - The generator forces both endpoint sprites to request the same source and
#   forces their visible areas to differ clearly: a full interior sprite versus a
#   corner-clipped sprite. This keeps the area ordering visually obvious and
#   prevents the combo from collapsing back to the original task.
# - Only one competitive source is used. That is enough to expose the priority
#   rule without creating several simultaneous lookup conflicts.
#
# c) Hidden assumptions audit
# - Original code implicitly assumed every destination could use its matching
#   source independently. That is false here; source usage must be tracked.
# - Original keep-mask seeding from both destinations would wrongly preserve a
#   smaller target that loses the competition, so the keep phase must seed only
#   matched survivors.
# - Source lookup is restricted to non-destination sprites. The generator
#   guarantees the lookup source is detached, so endpoint sprites never serve as
#   sources for each other.
# - Area ties would make the new order hard to infer. The generator rejects such
#   cases and always constructs strictly different destination areas.
#
# d) Placement legality
# - Sprite geometry, clipping, non-overlap, and the thin 8-connected trail are
#   kept from the base task.
# - The smaller destination is placed in a corner so it is visibly clipped but
#   still has a legal port. The larger destination is placed fully inside the
#   grid so it keeps full area and a legal port. The path is drawn only after
#   both ports are fixed.
#
# e) Visual observability
# - The critical control variable is visible pixel area, not a hidden numeric
#   attribute. Across examples, the bigger endpoint sprite survives while the
#   smaller endpoint sprite disappears, even though both show the same lookup key
#   in their centers.
# - Because both destinations ask for the same source border color, a solver that
#   ignores competition would recolor both and be wrong in a directly observable
#   way.
#
# f) Visual cleanliness
# - The layout keeps only one shared source and at most two distractors. The
#   output therefore shows one surviving endpoint sprite, one dangling grey
#   endpoint, and no additional clutter.
#
# g) Change inferability
# - Color logic stays simple: both endpoint centers repeat the same lookup key,
#   and exactly one detached source carries that border color. The extra rule the
#   solver must infer is that area priority decides which endpoint gets the
#   transfer.

from __future__ import annotations

import random
import sys
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
      - center color appears exactly once
    Clipped edge sprites are allowed as long as the visible component still fits the pattern.
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
    """Return trail endpoints that form a legal port for this sprite."""
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


def _resolve_destinations(g: np.ndarray, grey: int, sprites: list[dict]) -> list[dict]:
    """Resolve the two endpoint destination sprites using the original endpoint logic."""
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
        assert cands, "each endpoint should have at least one sprite candidate"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 destination sprites"
    return dest  # type: ignore[return-value]


def _apply_transfer_logic(
    g: np.ndarray,
    bg: int,
    grey: int,
    sprites: list[dict],
    dest: list[dict],
    *,
    competitive: bool,
    delete_unmatched: bool,
) -> np.ndarray:
    """Shared renderer for the combo rule and the non-competitive baseline check."""
    dest_ids = {id(sp) for sp in dest}
    source_sprites = [sp for sp in sprites if id(sp) not in dest_ids]

    by_border: dict[int, list[dict]] = {}
    for sp in source_sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    if competitive:
        process_order = sorted(
            dest,
            key=lambda sp: (-len(sp["pts"]), sp["center_pos"][0], sp["center_pos"][1]),
        )
    else:
        process_order = list(dest)

    used_sources: set[int] = set()
    matched: list[tuple[dict, dict]] = []
    doomed: list[dict] = []
    for d in process_order:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if competitive:
            srcs = [sp for sp in srcs if id(sp) not in used_sources]
        if not srcs:
            doomed.append(d)
            continue
        assert len(srcs) == 1, "generator should keep border colors unique among sources"
        src = srcs[0]
        matched.append((d, src))
        if competitive:
            used_sources.add(id(src))

    out = g.copy()
    for sp in source_sprites:
        _erase_sprite_cells(out, sp, bg)

    if delete_unmatched:
        for d in doomed:
            _erase_sprite_cells(out, d, bg)

    seed_dests = [d for d, _src in matched] if delete_unmatched else [*dest]
    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in seed_dests:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer one shared source into endpoint sprites with area-priority competition and delete losers.

    Input structure:
    - The grid contains background, a thin grey trail, and several small bordered sprites.
    - The two trail endpoints still identify two destination sprites through the same port rule as the original task.
    - Detached sprites can act as lookup sources: each has a border color and a single center color.

    Rule:
    - Resolve the two endpoint destinations first.
    - Process those destinations from larger visible area to smaller visible area.
    - For each destination, use its original center color as a lookup key for a detached source whose border matches that key.
    - A source can be used only once. If no unused matching source remains, delete that destination entirely.
    - Delete all non-destination sprites as well; the grey trail stays.

    Key insight:
    - Matching is no longer independent across the two endpoint sprites. Their visible sizes determine priority, so a larger endpoint can consume the only valid source and cause the smaller endpoint to disappear.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = 5

    sprites = _detect_bordered_sprites(g, bg, grey)
    assert sprites, "expected at least one bordered sprite in the grid"
    dest = _resolve_destinations(g, grey, sprites)
    return _apply_transfer_logic(g, bg, grey, sprites, dest, competitive=True, delete_unmatched=True)


# --------------- generator (combo variant) ---------------

GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 500


def _make_sprite_patch(shape_type: str, border_color: int, center_color: int) -> np.ndarray:
    """Return a 3x3 box or plus patch centered at (1,1)."""
    if shape_type == "box":
        patch = np.full((3, 3), border_color, dtype=int)
    else:
        patch = np.zeros((3, 3), dtype=int)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            patch[1 + dr, 1 + dc] = border_color
    patch[1, 1] = center_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != 0:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and side-touch with existing non-background, non-grey cells."""
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
    """All patch anchors that keep the sprite center on-grid."""
    return [
        (r0, c0)
        for r0 in range(-1, GRID_H)
        for c0 in range(-1, GRID_W)
        if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
    ]


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Prevent local 8-neighborhood clusters thicker than the original path style."""
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


def _port_options(grid: np.ndarray, cr: int, cc: int, bc: int) -> list[tuple[int, int]]:
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if grid[ir, ic] == bc and grid[er, ec] == BG:
            ports.append((er, ec))
    return ports


def _visible_area(patch: np.ndarray, r0: int, c0: int) -> int:
    area = 0
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < GRID_H and 0 <= cc < GRID_W and patch[dr, dc] != BG:
                area += 1
    return area


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _assign_colors(border_colors: list[int], num_sprites: int) -> dict | None:
    """Assign two competing destinations, one shared source, and the remaining distractors."""
    remaining = list(range(2, num_sprites))
    if not remaining:
        return None

    source_index = random.choice(remaining)
    distractor_indices = [i for i in remaining if i != source_index]
    shared_lookup_color = border_colors[source_index]

    source_center_candidates = [c for c in range(1, 10) if c not in {BG, GREY, *border_colors}]
    if not source_center_candidates:
        return None
    source_center_color = random.choice(source_center_candidates)

    distractor_center_colors = []
    for di in distractor_indices:
        candidates = [c for c in range(1, 10) if c not in {GREY, border_colors[di]}]
        if not candidates:
            return None
        distractor_center_colors.append(random.choice(candidates))

    return {
        "source_index": source_index,
        "distractor_indices": distractor_indices,
        "dest_center_colors": [shared_lookup_color, shared_lookup_color],
        "temp_output_dest_centers": [source_center_color, source_center_color],
        "source_center_color": source_center_color,
        "distractor_center_colors": distractor_center_colors,
    }


def _large_positions() -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in range(1, GRID_H - 3) for c0 in range(1, GRID_W - 3)]


def _small_positions() -> list[tuple[int, int]]:
    return [(-1, -1), (-1, GRID_W - 2), (GRID_H - 2, -1), (GRID_H - 2, GRID_W - 2)]


def _place_dest_role(
    grid: np.ndarray,
    patch: np.ndarray,
    border_color: int,
    positions: list[tuple[int, int]],
    *,
    other_port: tuple[int, int] | None,
    min_port_dist: int,
) -> dict | None:
    pool = list(positions)
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(grid, patch, r0, c0):
            continue
        cr, cc = r0 + 1, c0 + 1
        _place_sprite(grid, patch, r0, c0)
        ports = _port_options(grid, cr, cc, border_color)
        random.shuffle(ports)
        chosen_port = None
        for port in ports:
            if other_port is not None and _manhattan(port, other_port) < min_port_dist:
                continue
            chosen_port = port
            break
        if chosen_port is None:
            _erase_sprite(grid, patch, r0, c0)
            continue
        return {
            "top_left": (r0, c0),
            "center_pos": (cr, cc),
            "port": chosen_port,
            "area": _visible_area(patch, r0, c0),
            "patch": patch,
        }
    return None


def _place_competing_dests(
    output_grid: np.ndarray,
    shape_type: str,
    border_colors: list[int],
    output_dest_centers: list[int],
    min_port_dist: int,
) -> dict | None:
    """Place one full-area winner destination and one clipped loser destination."""
    big_patch = _make_sprite_patch(shape_type, border_colors[0], output_dest_centers[0])
    small_patch = _make_sprite_patch(shape_type, border_colors[1], output_dest_centers[1])

    small = _place_dest_role(
        output_grid,
        small_patch,
        border_colors[1],
        _small_positions(),
        other_port=None,
        min_port_dist=0,
    )
    if small is None:
        return None

    big = _place_dest_role(
        output_grid,
        big_patch,
        border_colors[0],
        _large_positions(),
        other_port=small["port"],
        min_port_dist=min_port_dist,
    )
    if big is None:
        _erase_sprite(output_grid, small_patch, *small["top_left"])
        return None

    if big["area"] <= small["area"]:
        _erase_sprite(output_grid, big_patch, *big["top_left"])
        _erase_sprite(output_grid, small_patch, *small["top_left"])
        return None

    return {
        "center_positions": [big["center_pos"], small["center_pos"]],
        "port_cells": [big["port"], small["port"]],
        "areas": [big["area"], small["area"]],
        "patches": [big_patch, small_patch],
        "top_lefts": [big["top_left"], small["top_left"]],
    }


def _baseline_without_competition(grid: np.ndarray) -> np.ndarray:
    """D14-only style baseline: unmatched destinations delete, but shared sources can serve both."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY
    sprites = _detect_bordered_sprites(g, bg, grey)
    dest = _resolve_destinations(g, grey, sprites)
    return _apply_transfer_logic(g, bg, grey, sprites, dest, competitive=False, delete_unmatched=True)


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
    placed = _place_competing_dests(
        full_output_grid,
        shape_type,
        border_colors,
        colors["temp_output_dest_centers"],
        min_port_dist,
    )
    if placed is None:
        return None

    port_cells = placed["port_cells"]
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
    for i, (cr, cc) in enumerate(placed["center_positions"]):
        input_grid[cr, cc] = colors["dest_center_colors"][i]

    sprites_to_place = [
        (border_colors[colors["source_index"]], colors["source_center_color"])
    ] + [
        (border_colors[di], colors["distractor_center_colors"][j])
        for j, di in enumerate(colors["distractor_indices"])
    ]
    all_pos = _sprite_positions()
    for bc, center_color in sprites_to_place:
        patch = _make_sprite_patch(shape_type, bc, center_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed_sprite = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_sprite(input_grid, patch, r0, c0)
            placed_sprite = True
            break
        if not placed_sprite:
            return None

    output_grid = full_output_grid.copy()
    _erase_sprite(output_grid, placed["patches"][1], *placed["top_lefts"][1])

    try:
        combo_out = main(input_grid)
        baseline_out = _baseline_without_competition(input_grid)
        if np.array_equal(combo_out, baseline_out):
            return None

        if not np.array_equal(combo_out, output_grid):
            return None

        sprites = _detect_bordered_sprites(input_grid, BG, GREY)
        dest = _resolve_destinations(input_grid, GREY, sprites)
        if {tuple(d["center_pos"]) for d in dest} != {tuple(p) for p in placed["center_positions"]}:
            return None
        if len({len(d["pts"]) for d in dest}) != 2:
            return None
        if len({int(d["center"]) for d in dest}) != 1:
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
