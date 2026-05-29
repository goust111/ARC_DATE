# Variant of d35bdbdc
# Dimension: D3 — 源对象类型 (src)
# Change: 带中心色和边框色的精灵（有两层颜色结构：外层边框色、内层中心色）... → 带对角标记的精灵

# Design analysis for this D3 variant
# -----------------------------------
# a) Concretization and behavior trace
# - The original task used 2-color bordered sprites everywhere. This variant keeps the
#   two grey-path-connected destination sprites as ordinary bordered 3x3 boxes, but all
#   non-destination sprites become marked source sprites: a full 3x3 box with
#   border color, center color, and exactly one recolored corner pixel (the diagonal marker).
# - The border color still acts as the lookup key. For each destination, its original
#   center color names the source by border color, exactly as in the seed task.
# - The new behavior happens at transfer time. Once a source is found, compare the
#   destination's location to the source center:
#   TL/TR/BL/BR quadrant relative to the source center.
#   If that quadrant matches the marked corner, transfer the source marker color.
#   Otherwise transfer the source center color.
# - Small trace:
#   original: dest.center == blue-border source -> write source.center to dest.center.
#   variant: dest.center == blue-border source -> if dest lies in source's marked
#   corner direction, write source.marker; else write source.center.
#   The difference is visible because the source contains two candidate payload colors.
#
# b) Complexity calibration
# - The marker must matter often enough to be inferable. The generator therefore forces
#   each example to contain both transfer modes:
#   at least one destination receives a marker color and at least one receives a center color.
# - Only one extra symbolic degree is introduced: the marked corner among {TL,TR,BL,BR}.
#   The rest of the task (path endpoints, deletion, border-color lookup) stays stable.
#
# c) Implicit assumptions audit
# - The original component detector assumed exactly two colors per sprite and identified
#   the center by "the unique minority color". That breaks immediately once sources carry
#   a third color; we therefore split detection into:
#   ordinary 2-color destination boxes, and
#   fully visible 3x3 marked source boxes.
# - The original mixed box and plus shapes. A plus cannot safely absorb a diagonal corner
#   marker under 4-connectivity because the corner would detach into a separate component.
#   To keep the new marker visible and solver-detectable, this variant restricts sprites
#   to box-shaped 3x3 objects.
# - Directional transfer needs the destination not to lie in the same row or same column
#   as its matched source, otherwise the quadrant would be ambiguous. The generator
#   explicitly enforces nonzero row and column offsets for all matched source-destination pairs.
#
# d) Placement legality
# - Destinations may still be edge-clipped, as in the original task, because the port rule
#   only needs center + one border cell + a grey endpoint.
# - Marked sources are always placed fully in-bounds so their diagonal marker is never clipped away.
# - Placement still forbids overlap and 4-touching between colored components.
#
# e) Visual observability
# - The new rule is visible in the input itself: some source boxes have a recolored corner.
# - The output only keeps the two path-connected destinations, so the solver must infer that
#   the kept center colors come from one of two payload colors on the matched source.
# - No hidden ordering by color id is used anywhere.
#
# f) Visual cleanliness
# - The marked corner stays inside the 3x3 source box, so no extra disconnected noise is added.
# - Sources are separated from the grey trail and from each other by the existing overlap guard.
#
# g) Color-change inferability
# - Each matched source contributes two visible candidate colors: center and marker.
#   The output shows which one is chosen. Because each example is forced to exhibit both
#   outcomes, the directional marker is not a dormant feature.
#
# h) Variant metric used after implementation
# - marker_transfer_count = number of destinations whose final color came from a source marker.
# - center_transfer_count = number of destinations whose final color came from a source center.
# - Non-degenerate examples require both counts to be positive.
#
# i) New solver assumptions checked against generator
# - Every destination center color matches at most one source border color: guaranteed by unique borders.
# - Every marked source is a full 3x3 box with counts {border:7, center:1, marker:1}: guaranteed.
# - Every matched destination lies off both the source row and source column: guaranteed.

import random
from collections import Counter, deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


GRID_H, GRID_W = 10, 10
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
INTERIOR_DEST_WEIGHTS = [3, 1]
PATH_LEN_MIN, PATH_LEN_MAX = 10, 24
MIN_PORT_DIST_FRAC = 3
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 400

CORNER_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, 2),
    "BL": (2, 0),
    "BR": (2, 2),
}


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _component_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _make_dest_patch(border_color: int, center_color: int) -> np.ndarray:
    patch = np.full((3, 3), border_color, dtype=int)
    patch[1, 1] = center_color
    return patch


def _make_source_patch(border_color: int, center_color: int, marker_color: int, marker_corner: str) -> np.ndarray:
    patch = _make_dest_patch(border_color, center_color)
    mr, mc = CORNER_OFFSETS[marker_corner]
    patch[mr, mc] = marker_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=True)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = grid.shape
    for dr in range(3):
        for dc in range(3):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w and patch[dr, dc] != BG:
                grid[rr, cc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and also reject 4-touching another colored component."""
    h, w = grid.shape
    patch_cells: set[tuple[int, int]] = set()
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
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) not in (BG, GREY):
                return True
    return False


def _sprite_positions(clipped_ok: bool) -> list[tuple[int, int]]:
    if clipped_ok:
        return [
            (r0, c0)
            for r0 in range(-1, GRID_H)
            for c0 in range(-1, GRID_W)
            if 0 <= r0 + 1 < GRID_H and 0 <= c0 + 1 < GRID_W
        ]
    return [(r0, c0) for r0 in range(0, GRID_H - 2) for c0 in range(0, GRID_W - 2)]


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


def _gen_winding_path(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int) -> list[tuple[int, int]] | None:
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
        if _too_thick(end[0], end[1], visited):
            return None
        path.append(end)
    return path


def _is_8_connected(cells: list[tuple[int, int]]) -> bool:
    if len(cells) <= 1:
        return True
    cell_set = set(cells)
    start = next(iter(cell_set))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in cell_set and nb not in seen:
                    seen.add(nb)
                    q.append(nb)
    return len(seen) == len(cell_set)


def _find_port_dir(grid: np.ndarray, cr: int, cc: int, border_color: int) -> tuple[int, int] | None:
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(dirs)
    for dr, dc in dirs:
        ir, ic = cr + dr, cc + dc
        er, ec = cr + 2 * dr, cc + 2 * dc
        if not (0 <= ir < GRID_H and 0 <= ic < GRID_W and 0 <= er < GRID_H and 0 <= ec < GRID_W):
            continue
        if int(grid[ir, ic]) == border_color and int(grid[er, ec]) == BG:
            return er, ec
    return None


def _find_trail_endpoints(g: np.ndarray, grey: int, sprites: list[dict]) -> tuple[tuple[int, int], tuple[int, int]]:
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


def _detect_dest_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Destination sprite = a 4-connected component with exactly 2 colors:
      - border color appears multiple times
      - center color appears exactly once
    Edge clipping is allowed, as in the seed task.
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
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "center_pos": center_pos,
            }
        )
    return sprites


def _detect_marked_sources(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Marked source = a full 3x3 box with:
      - border color in 7 cells
      - center color in the middle cell
      - marker color in exactly one corner cell
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sources: list[dict] = []
    for pts in comps:
        if len(pts) != 9:
            continue
        r0, c0, r1, c1 = _component_bbox(pts)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        cells = {(int(r), int(c)) for r, c in pts}
        if len(cells) != 9:
            continue
        if any((r, c) not in cells for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            continue

        center_pos = (r0 + 1, c0 + 1)
        center_color = int(g[center_pos])
        ring_pos = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) != center_pos]
        ring_vals = [int(g[r, c]) for r, c in ring_pos]
        counts = Counter(ring_vals)
        if sorted(counts.values()) != [1, 7]:
            continue
        border_color = max(counts, key=counts.get)
        marker_color = [col for col, cnt in counts.items() if cnt == 1][0]
        if border_color in (bg, grey) or center_color in (bg, grey) or marker_color in (bg, grey):
            continue
        marker_pos = next((pos for pos in ring_pos if int(g[pos]) == marker_color), None)
        if marker_pos is None:
            continue
        marker_corner = None
        for name, (dr, dc) in CORNER_OFFSETS.items():
            if marker_pos == (r0 + dr, c0 + dc):
                marker_corner = name
                break
        if marker_corner is None:
            continue
        sources.append(
            {
                "pts": pts,
                "border": int(border_color),
                "center": int(center_color),
                "marker": int(marker_color),
                "center_pos": center_pos,
                "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
                "marker_corner": marker_corner,
            }
        )
    return sources


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
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


def _dest_quadrant_from_source(src_center: tuple[int, int], dest_center: tuple[int, int]) -> str | None:
    sr, sc = src_center
    dr, dc = dest_center
    if sr == dr or sc == dc:
        return None
    return ("T" if dr < sr else "B") + ("L" if dc < sc else "R")


def _source_transfer_color(src: dict, dest: dict) -> int:
    quadrant = _dest_quadrant_from_source(src["center_pos"], dest["center_pos"])
    assert quadrant is not None, "matched source/destination must differ in both row and column"
    return int(src["marker"] if quadrant == src["marker_corner"] else src["center"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer colors from marked source boxes into path-selected destination boxes, then remove everything else.

    Input structure:
    - Background, a grey one-stroke trail, two ordinary bordered destination boxes, and several fully visible marked source boxes.
    - Each marked source has three visible roles inside one 3x3 box: border color, center color, and a single recolored corner marker.
    - The grey trail endpoints attach only to the two destinations via the same port rule as the seed task.

    Rule:
    - Use the grey trail endpoints to identify the two destination boxes.
    - For each destination, find the marked source whose border color equals the destination's original center color.
    - Compare destination position to that source: if the destination lies in the marker corner's quadrant, copy the marker color; otherwise copy the source center color.
    - Delete all non-destination colored components; keep the grey trail and the two destinations.

    Key insight: The destination center still names the source by border color, but the source's diagonal marker decides which of its two payload colors is actually transferred.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    dest_sprites = _detect_dest_sprites(g, bg, grey)
    marked_sources = _detect_marked_sources(g, bg, grey)
    assert dest_sprites, "expected destination sprites"
    assert marked_sources, "expected marked source sprites"

    te1, te2 = _find_trail_endpoints(g, grey, dest_sprites)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in dest_sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

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
    assert all(d is not None for d in dest), "expected exactly 2 destinations from trail endpoints"
    dest_ids = {id(sp) for sp in dest}

    by_border: dict[int, list[dict]] = {}
    for sp in marked_sources:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    for sp in marked_sources:
        for r, c in sp["pts"]:
            out[r, c] = bg

    keep = np.zeros(out.shape, dtype=bool)
    q: list[tuple[int, int]] = []
    for d in dest:
        for r, c in d["pts"]:
            if out[r, c] != bg and out[r, c] != grey and not keep[r, c]:
                keep[r, c] = True
                q.append((int(r), int(c)))
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
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one marked source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = _source_transfer_color(src, d)

    return out


def _choose_role_layout(num_sprites: int) -> dict | None:
    remaining = list(range(2, num_sprites))
    num_sources = random.choice(SOURCE_COUNTS) if len(remaining) >= 2 else max(1, len(remaining))
    if num_sources == 0:
        return None
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]
    if num_sources == 1:
        dest_to_source = [source_indices[0], source_indices[0]]
    else:
        dest_to_source = [source_indices[0], source_indices[1]]
    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "dest_to_source": dest_to_source,
    }


def _assign_colors(border_colors: list[int], layout: dict) -> dict | None:
    palette = [c for c in range(1, 10) if c != GREY]
    source_specs: dict[int, dict[str, int]] = {}
    for idx in layout["source_indices"] + layout["distractor_indices"]:
        forbidden = {BG, GREY, border_colors[idx]}
        candidates = [c for c in palette if c not in forbidden]
        if len(candidates) < 2:
            return None
        center_color, marker_color = random.sample(candidates, 2)
        source_specs[idx] = {"center": center_color, "marker": marker_color}

    dest_lookup_colors = [border_colors[src_idx] for src_idx in layout["dest_to_source"]]
    return {
        "source_specs": source_specs,
        "dest_lookup_colors": dest_lookup_colors,
    }


def _place_dests(base_grid: np.ndarray, border_colors: list[int], dest_lookup_colors: list[int], min_port_dist: int) -> list[tuple[int, int]] | None:
    all_pos = _sprite_positions(clipped_ok=True)
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_centers: list[tuple[int, int]] = []
    first_port: tuple[int, int] | None = None

    for i in range(2):
        patch = _make_dest_patch(border_colors[i], dest_lookup_colors[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(base_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(base_grid, patch, r0, c0)
            port = _find_port_dir(base_grid, cr, cc, border_colors[i])
            if port is None:
                _erase_sprite(base_grid, patch, r0, c0)
                continue
            if i == 1:
                assert first_port is not None
                dist = abs(port[0] - first_port[0]) + abs(port[1] - first_port[1])
                if dist < min_port_dist:
                    _erase_sprite(base_grid, patch, r0, c0)
                    continue
            dest_centers.append((cr, cc))
            if i == 0:
                first_port = port
            placed = True
            break
        if not placed:
            return None
    return dest_centers


def _relation_matches(marker_corner: str, src_center: tuple[int, int], dest_center: tuple[int, int]) -> bool:
    quadrant = _dest_quadrant_from_source(src_center, dest_center)
    return quadrant is not None and quadrant == marker_corner


def _place_marked_source(
    grid: np.ndarray,
    border_color: int,
    center_color: int,
    marker_color: int,
    matched_dest_centers: list[tuple[int, int]],
    required_behavior: bool | None,
    require_mixed: bool,
) -> dict | None:
    positions = _sprite_positions(clipped_ok=False)
    random.shuffle(positions)
    corners = list(CORNER_OFFSETS)
    random.shuffle(corners)

    for r0, c0 in positions:
        src_center = (r0 + 1, c0 + 1)
        if any(src_center[0] == dr or src_center[1] == dc for dr, dc in matched_dest_centers):
            continue
        for marker_corner in corners:
            if matched_dest_centers:
                rels = [_relation_matches(marker_corner, src_center, dpos) for dpos in matched_dest_centers]
                if require_mixed and (all(rels) or not any(rels)):
                    continue
                if required_behavior is not None and rels != [required_behavior]:
                    continue
            patch = _make_source_patch(border_color, center_color, marker_color, marker_corner)
            if _overlaps(grid, patch, r0, c0):
                continue
            _place_sprite(grid, patch, r0, c0)
            return {
                "patch": patch,
                "r0": r0,
                "c0": c0,
                "center_pos": src_center,
                "marker_corner": marker_corner,
                "border": border_color,
                "center": center_color,
                "marker": marker_color,
                "matched_dest_centers": matched_dest_centers,
                "transfer_modes": [
                    "marker" if _relation_matches(marker_corner, src_center, dpos) else "center"
                    for dpos in matched_dest_centers
                ],
            }
    return None


def _build_output_grid(base_grid: np.ndarray, dest_centers: list[tuple[int, int]], dest_to_source: list[int], placed_sources: dict[int, dict]) -> tuple[np.ndarray, list[str]]:
    out = base_grid.copy()
    transfer_modes: list[str] = []
    for dest_idx, src_idx in enumerate(dest_to_source):
        src = placed_sources[src_idx]
        dest_center = dest_centers[dest_idx]
        mode = "marker" if _relation_matches(src["marker_corner"], src["center_pos"], dest_center) else "center"
        transfer_modes.append(mode)
        color = int(src["marker"] if mode == "marker" else src["center"])
        out[dest_center] = color
    return out, transfer_modes


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    num_sprites = random.choice(SPRITE_COUNTS)
    border_palette = [c for c in range(1, 10) if c != GREY]
    border_colors = random.sample(border_palette, num_sprites)

    layout = _choose_role_layout(num_sprites)
    if layout is None:
        return None
    colors = _assign_colors(border_colors, layout)
    if colors is None:
        return None

    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    base_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_centers = _place_dests(base_grid, border_colors, colors["dest_lookup_colors"], min_port_dist)
    if dest_centers is None:
        return None

    port_cells: list[tuple[int, int]] = []
    for i in range(2):
        port = _find_port_dir(base_grid, dest_centers[i][0], dest_centers[i][1], border_colors[i])
        if port is None:
            return None
        port_cells.append(port)

    path_cells = _gen_winding_path(base_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None
    for r, c in path_cells:
        if base_grid[r, c] != BG:
            return None
        base_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and base_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    input_grid = base_grid.copy()
    placed_sources: dict[int, dict] = {}

    if len(layout["source_indices"]) == 1:
        src_idx = layout["source_indices"][0]
        spec = colors["source_specs"][src_idx]
        info = _place_marked_source(
            input_grid,
            border_colors[src_idx],
            spec["center"],
            spec["marker"],
            matched_dest_centers=dest_centers,
            required_behavior=None,
            require_mixed=True,
        )
        if info is None:
            return None
        placed_sources[src_idx] = info
    else:
        required_modes = [True, False]
        random.shuffle(required_modes)
        for req, src_idx, dest_idx in zip(required_modes, layout["source_indices"], [0, 1], strict=False):
            spec = colors["source_specs"][src_idx]
            info = _place_marked_source(
                input_grid,
                border_colors[src_idx],
                spec["center"],
                spec["marker"],
                matched_dest_centers=[dest_centers[dest_idx]],
                required_behavior=req,
                require_mixed=False,
            )
            if info is None:
                return None
            placed_sources[src_idx] = info

    for src_idx in layout["distractor_indices"]:
        spec = colors["source_specs"][src_idx]
        info = _place_marked_source(
            input_grid,
            border_colors[src_idx],
            spec["center"],
            spec["marker"],
            matched_dest_centers=[],
            required_behavior=None,
            require_mixed=False,
        )
        if info is None:
            return None
        placed_sources[src_idx] = info

    output_grid, transfer_modes = _build_output_grid(base_grid, dest_centers, layout["dest_to_source"], placed_sources)
    if "marker" not in transfer_modes or "center" not in transfer_modes:
        return None

    try:
        out = main(input_grid)
        if np.array_equal(out, output_grid):
            return input_grid, {
                "transfer_modes": transfer_modes,
                "source_count": len(layout["source_indices"]),
                "distractor_count": len(layout["distractor_indices"]),
            }
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
