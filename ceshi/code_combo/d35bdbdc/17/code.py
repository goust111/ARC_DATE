# Combo variant of d35bdbdc (index 17)
# combo_spec: D3/2+D15/2
# Name: 带对角标记的精灵+方向箭头标记

# Design analysis for this combo variant
# --------------------------------------
# a) Concretization and behavior trace
# - Keep the original scaffold: a 10x10 grid, a grey one-stroke path, and exactly two
#   path-connected destination sprites that survive in the output.
# - Destinations stay ordinary 2-color bordered 3x3 boxes. Their input center color is no
#   longer enough to identify a unique source: both destinations deliberately use the same
#   lookup color, and exactly two sources share that same border color.
# - Each source is a full 3x3 box with three simultaneous layers:
#   border color in 5 cells,
#   a 2-pixel arrow stub made from the transferable payload color,
#   one extra diagonal marker pixel in a different corner.
# - Matching now has two geometric stages on top of the original color chain:
#   destination.center -> candidate source border class,
#   source.arrow -> dominant-axis direction to the target destination,
#   source.diagonal marker corner -> which quadrant inside that direction is the real target.
# - Small trace:
#   both destinations have center = blue, and there are two blue-border sources.
#   Both sources point right, so both destinations lie to the right of each source.
#   One source marks TR, the other marks BR.
#   The upper-right destination receives the TR source payload; the lower-right destination
#   receives the BR source payload.
#
# b) Complexity calibration
# - The generator forces border ambiguity: both surviving destinations share one lookup color,
#   and both real sources share that border color. So color matching alone is insufficient.
# - The generator also forces direction ambiguity: for each real source, both destinations lie
#   in the encoded arrow direction. So arrow direction alone is insufficient.
# - The only remaining discriminator is the diagonal marker corner, limited to one of two
#   corners compatible with the chosen arrow direction. This keeps the symbolic alphabet small.
#
# c) Implicit assumptions audit
# - The seed solver assumed every sprite had exactly two colors. That breaks here because real
#   sources have counts {border:5, payload:3, marker:1}; detection must split destination boxes
#   from full 3-color source boxes.
# - The seed solver assumed destination.center selects at most one source border. That also breaks:
#   two sources intentionally share the same border color. The solver must gather all sources in
#   that border class and then resolve the target by arrow direction plus marker quadrant.
# - The D15-style directional cue is compressed to a 2-pixel center+neighbor arrow stub. That
#   keeps all four corners free so the diagonal-marker system can coexist without overlap.
# - Plus-shaped sprites are disallowed. A plus cannot host both a diagonal marker and a clean
#   3-pixel arrow while remaining a single readable object.
#
# d) Placement legality
# - Destinations may still be edge-clipped because the port rule only depends on the center and
#   one cardinal border cell.
# - Sources are always fully visible 3x3 boxes so both the arrow and the diagonal marker are
#   never clipped away.
# - Source placement is filtered so the two destinations are both in the source's arrow direction
#   but occupy different compatible quadrants. Overlap and 4-touch constraints are preserved.
#
# e) Visual observability
# - The path still clearly marks the two surviving destinations.
# - The input visibly contains two deleted source boxes, each with an arrow and an extra corner
#   marker. Across examples, both sources can point in the same direction while targeting different
#   kept destinations, making it apparent that the corner marker is doing real work.
# - No hidden ordering by color id is used; all routing information is geometric and visible.
#
# f) Visual cleanliness
# - Every symbolic cue stays inside a 3x3 box. No detached noise pixels are introduced.
# - The arrow is a compact 2-pixel center+edge pointer, while the marker remains a single corner
#   pixel. The two symbolic systems are therefore visually separable inside one 3x3 box.
#
# g) Non-degeneracy targets used after implementation
# - border_candidate_count per destination should be 2, not 1.
# - direction_candidate_count per source should be 2, not 1.
# - Only the diagonal marker should reduce that ambiguous pair to one actual target.
#
# h) New solver assumptions checked against the generator
# - Real sources are full 3x3 boxes with color multiplicities exactly {6,2,1}: guaranteed.
# - For each real source, both destinations share the arrow direction but occupy the two marker
#   quadrants allowed for that direction: guaranteed by placement search.
# - For each destination, among same-border sources exactly one source matches both the direction
#   and the quadrant tests: guaranteed by using one source per compatible corner.

import random
from collections import Counter, deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


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
GEN_RETRIES = 800

CORNER_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, 2),
    "BL": (2, 0),
    "BR": (2, 2),
}

ARROW_PATTERNS = {
    "up": {(0, 0), (-1, 0)},
    "down": {(0, 0), (1, 0)},
    "left": {(0, 0), (0, -1)},
    "right": {(0, 0), (0, 1)},
}

DIRECTION_MARKERS = {
    "up": ("TL", "TR"),
    "down": ("BL", "BR"),
    "left": ("TL", "BL"),
    "right": ("TR", "BR"),
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


def _make_source_patch(
    border_color: int,
    payload_color: int,
    marker_color: int,
    direction: str,
    marker_corner: str,
) -> np.ndarray:
    patch = np.full((3, 3), border_color, dtype=int)
    for dr, dc in ARROW_PATTERNS[direction]:
        patch[1 + dr, 1 + dc] = payload_color
    mr, mc = CORNER_OFFSETS[marker_corner]
    assert patch[mr, mc] == border_color, "marker corner must not overlap the arrow"
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


def _gen_winding_path(
    grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int
) -> list[tuple[int, int]] | None:
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


def _detect_arrow_marker_sources(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Real source = a full 3x3 box with color counts {border:6, payload:2, marker:1}.
    The payload forms one of four 2-pixel arrow stubs and includes the center cell.
    The marker is a singleton corner not occupied by the arrow.
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
        if any((r, c) not in cells for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            continue

        vals = [int(g[r, c]) for r, c in pts]
        counts = Counter(vals)
        if sorted(counts.values()) != [1, 2, 6]:
            continue

        center_pos = (r0 + 1, c0 + 1)
        payload_color = int(g[center_pos])
        if counts[payload_color] != 2:
            continue
        border_color = next(col for col, cnt in counts.items() if cnt == 6)
        marker_color = next(col for col, cnt in counts.items() if cnt == 1)
        if border_color in (bg, grey) or payload_color in (bg, grey) or marker_color in (bg, grey):
            continue

        payload_pts = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == payload_color]
        rel = {(r - center_pos[0], c - center_pos[1]) for r, c in payload_pts}
        direction = None
        for name, pattern in ARROW_PATTERNS.items():
            if rel == pattern:
                direction = name
                break
        if direction is None:
            continue

        marker_pos = next((int(r), int(c)) for r, c in pts if int(g[r, c]) == marker_color)
        marker_corner = None
        for name, (dr, dc) in CORNER_OFFSETS.items():
            if marker_pos == (r0 + dr, c0 + dc):
                marker_corner = name
                break
        if marker_corner is None or marker_corner not in DIRECTION_MARKERS[direction]:
            continue

        sources.append(
            {
                "pts": pts,
                "border": int(border_color),
                "payload": int(payload_color),
                "marker": int(marker_color),
                "center_pos": center_pos,
                "marker_pos": marker_pos,
                "marker_corner": marker_corner,
                "direction": direction,
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


def _direction_to(src: tuple[int, int], dst: tuple[int, int]) -> str | None:
    """Dominant-axis direction from src to dst. Ties are ambiguous."""
    dr = int(dst[0] - src[0])
    dc = int(dst[1] - src[1])
    if dr == 0 and dc == 0:
        return None
    if abs(dr) == abs(dc):
        return None
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _source_matches_dest(src: dict, dest: dict) -> bool:
    if _direction_to(src["center_pos"], dest["center_pos"]) != src["direction"]:
        return False
    return _dest_quadrant_from_source(src["center_pos"], dest["center_pos"]) == src["marker_corner"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer payload colors from ambiguous source classes into the two path-selected destination boxes, then remove everything else.

    Input structure:
    - The grid contains background, a grey (5) one-stroke trail, two ordinary bordered destination boxes attached to the trail endpoints, and additional full 3x3 source boxes.
    - Each real source box has three simultaneous parts: a shared border color, a 2-pixel payload-colored arrow stub, and one extra diagonal marker pixel in a corner.
    - The two surviving destinations deliberately share one lookup center color, so that color alone only identifies a source class rather than a single source.

    Rule:
    - Use the grey trail endpoints to identify the two destination boxes, exactly as in the seed task.
    - For each destination, gather all source boxes whose border color equals that destination's original center color.
    - Among those same-border candidates, keep the unique source whose arrow points toward the destination and whose diagonal marker corner matches the destination's quadrant relative to that source.
    - Replace the destination center with that source's payload color, and delete all non-destination colored components.

    Key insight:
    - Matching is now layered: the destination center color selects a source border class, the arrow gives a coarse global direction, and the diagonal marker resolves which destination inside that direction cone is the true target.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    dest_sprites = _detect_dest_sprites(g, bg, grey)
    sources = _detect_arrow_marker_sources(g, bg, grey)
    assert dest_sprites, "expected destination sprites"
    assert sources, "expected marked arrow sources"

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
    for sp in sources:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = g.copy()

    for sp in sources:
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
        cands = by_border.get(int(d["center"]), [])
        matched = [src for src in cands if _source_matches_dest(src, d)]
        assert len(matched) == 1, "each destination should have exactly one source after geometric decoding"
        dr, dc = d["center_pos"]
        out[dr, dc] = int(matched[0]["payload"])

    return out


def _place_dests(base_grid: np.ndarray, dest_border_colors: list[int], lookup_color: int, min_port_dist: int) -> list[tuple[int, int]] | None:
    all_pos = _sprite_positions(clipped_ok=True)
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_centers: list[tuple[int, int]] = []
    first_port: tuple[int, int] | None = None

    for i in range(2):
        patch = _make_dest_patch(dest_border_colors[i], lookup_color)
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(base_grid, patch, r0, c0):
                continue
            cr, cc = r0 + 1, c0 + 1
            _place_sprite(base_grid, patch, r0, c0)
            port = _find_port_dir(base_grid, cr, cc, dest_border_colors[i])
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
    if dest_centers[0][0] == dest_centers[1][0] or dest_centers[0][1] == dest_centers[1][1]:
        return None
    return dest_centers


def _source_candidate_infos(
    grid: np.ndarray,
    dest_centers: list[tuple[int, int]],
    direction: str,
    marker_corner: str,
    border_color: int,
    payload_color: int,
    marker_color: int,
) -> list[dict]:
    infos: list[dict] = []
    allowed_quads = set(DIRECTION_MARKERS[direction])
    patch = _make_source_patch(border_color, payload_color, marker_color, direction, marker_corner)
    for r0, c0 in _sprite_positions(clipped_ok=False):
        src_center = (r0 + 1, c0 + 1)
        dirs = [_direction_to(src_center, dpos) for dpos in dest_centers]
        quads = [_dest_quadrant_from_source(src_center, dpos) for dpos in dest_centers]
        if any(d is None for d in dirs) or any(q is None for q in quads):
            continue
        if dirs != [direction, direction]:
            continue
        if set(quads) != allowed_quads:
            continue
        target_indices = [i for i, quad in enumerate(quads) if quad == marker_corner]
        if len(target_indices) != 1:
            continue
        if _overlaps(grid, patch, r0, c0):
            continue
        infos.append(
            {
                "patch": patch,
                "r0": r0,
                "c0": c0,
                "center_pos": src_center,
                "border": border_color,
                "payload": payload_color,
                "marker": marker_color,
                "direction": direction,
                "marker_corner": marker_corner,
                "target_idx": target_indices[0],
                "direction_candidate_count": sum(1 for d in dirs if d == direction),
            }
        )
    random.shuffle(infos)
    return infos


def _place_real_sources(
    input_grid: np.ndarray,
    dest_centers: list[tuple[int, int]],
    lookup_color: int,
    payload_colors: list[int],
    marker_colors: list[int],
) -> list[dict] | None:
    directions = list(DIRECTION_MARKERS)
    random.shuffle(directions)
    for direction in directions:
        corners = list(DIRECTION_MARKERS[direction])
        random.shuffle(corners)
        cands0 = _source_candidate_infos(
            input_grid, dest_centers, direction, corners[0], lookup_color, payload_colors[0], marker_colors[0]
        )
        cands1 = _source_candidate_infos(
            input_grid, dest_centers, direction, corners[1], lookup_color, payload_colors[1], marker_colors[1]
        )
        if not cands0 or not cands1:
            continue
        for info0 in cands0:
            _place_sprite(input_grid, info0["patch"], info0["r0"], info0["c0"])
            for info1 in cands1:
                if _overlaps(input_grid, info1["patch"], info1["r0"], info1["c0"]):
                    continue
                if info0["target_idx"] == info1["target_idx"]:
                    continue
                _place_sprite(input_grid, info1["patch"], info1["r0"], info1["c0"])
                return [info0, info1]
            _erase_sprite(input_grid, info0["patch"], info0["r0"], info0["c0"])
    return None


def _place_distractor(grid: np.ndarray, border_color: int, center_color: int) -> bool:
    patch = _make_dest_patch(border_color, center_color)
    pool = _sprite_positions(clipped_ok=True)
    random.shuffle(pool)
    for r0, c0 in pool:
        if _overlaps(grid, patch, r0, c0):
            continue
        _place_sprite(grid, patch, r0, c0)
        return True
    return False


def _choose_colors() -> dict | None:
    palette = [c for c in range(1, 10) if c != GREY]
    dest_borders = random.sample(palette, 2)
    remaining = [c for c in palette if c not in dest_borders]
    if len(remaining) < 3:
        return None
    lookup_color = random.choice(remaining)
    remaining = [c for c in remaining if c != lookup_color]
    payload_colors = random.sample(remaining, 2)
    marker_choices = [c for c in palette if c not in {lookup_color, *payload_colors}]
    if len(marker_choices) < 2:
        return None
    marker_colors = random.sample(marker_choices, 2)
    distractor_border_choices = [c for c in palette if c != lookup_color]
    return {
        "dest_borders": dest_borders,
        "lookup_color": lookup_color,
        "payload_colors": payload_colors,
        "marker_colors": marker_colors,
        "distractor_border_choices": distractor_border_choices,
    }


def _build_output_grid(base_grid: np.ndarray, dest_centers: list[tuple[int, int]], placed_sources: list[dict]) -> np.ndarray:
    out = base_grid.copy()
    for src in placed_sources:
        dest_center = dest_centers[int(src["target_idx"])]
        out[dest_center] = int(src["payload"])
    return out


def _compute_behavior_metrics(input_grid: np.ndarray) -> dict:
    g = np.asarray(input_grid, dtype=int)
    bg = _bg_boundary(g)
    dests = _detect_dest_sprites(g, bg, GREY)
    sources = _detect_arrow_marker_sources(g, bg, GREY)
    te1, te2 = _find_trail_endpoints(g, GREY, dests)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in dests if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    picked: list[dict] = [None, None]  # type: ignore[list-item]
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
        picked[i] = best
        used.add(id(best))

    border_candidate_counts = []
    for d in picked:
        border_candidate_counts.append(sum(1 for s in sources if int(s["border"]) == int(d["center"])))

    direction_candidate_counts = []
    resolved_counts = []
    for s in sources:
        in_direction = [d for d in picked if _direction_to(s["center_pos"], d["center_pos"]) == s["direction"]]
        resolved = [d for d in picked if _source_matches_dest(s, d)]
        direction_candidate_counts.append(len(in_direction))
        resolved_counts.append(len(resolved))

    return {
        "border_candidate_counts": border_candidate_counts,
        "direction_candidate_counts": direction_candidate_counts,
        "resolved_counts": resolved_counts,
    }


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    colors = _choose_colors()
    if colors is None:
        return None

    num_sprites = random.choice(SPRITE_COUNTS)
    num_distractors = num_sprites - 4
    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)
    min_port_dist = max(3, target_len // MIN_PORT_DIST_FRAC)

    base_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    dest_centers = _place_dests(base_grid, colors["dest_borders"], colors["lookup_color"], min_port_dist)
    if dest_centers is None:
        return None

    port_cells: list[tuple[int, int]] = []
    for i in range(2):
        port = _find_port_dir(base_grid, dest_centers[i][0], dest_centers[i][1], colors["dest_borders"][i])
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
    placed_sources = _place_real_sources(
        input_grid,
        dest_centers,
        colors["lookup_color"],
        colors["payload_colors"],
        colors["marker_colors"],
    )
    if placed_sources is None:
        return None

    for _ in range(num_distractors):
        bc = random.choice(colors["distractor_border_choices"])
        choices = [c for c in range(1, 10) if c not in (GREY, bc)]
        if not choices:
            return None
        if not _place_distractor(input_grid, bc, random.choice(choices)):
            return None

    output_grid = _build_output_grid(base_grid, dest_centers, placed_sources)
    try:
        metrics = _compute_behavior_metrics(input_grid)
    except Exception:
        return None
    if metrics["border_candidate_counts"] != [2, 2]:
        return None
    if sorted(metrics["direction_candidate_counts"]) != [2, 2]:
        return None
    if sorted(metrics["resolved_counts"]) != [1, 1]:
        return None

    try:
        out = main(input_grid)
        if np.array_equal(out, output_grid):
            return input_grid, metrics
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"Failed to generate input after {GEN_RETRIES} attempts")
