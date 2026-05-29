# Combo variant of d35bdbdc (index 34)
# combo_spec: D2/1+D3/2
# Name: 路径连接角落+带对角标记精灵

# Design analysis for this combo variant
# --------------------------------------
# a) Concretization and behavior trace
# - Seed task: grey trail endpoints selected two destination sprites via center-based
#   ports, then each destination copied the center color from a source whose border
#   matched the destination's original center.
# - Combo task: the trail no longer points to the center-axis ports. Each endpoint now
#   sits diagonally outside one corner of a destination 3x3 box, so the endpoint names
#   one of {TL, TR, BL, BR}. Separately, every non-destination sprite is a 3x3 box with
#   border color, center color, and a single recolored diagonal corner marker.
# - Fused transfer:
#   1. Use the two grey endpoints to find the two destination boxes and record which
#      corner each endpoint touches.
#   2. Use the destination's original center color as the lookup key into source border
#      colors, as before.
#   3. If the matched source's marked corner equals the touched destination corner,
#      transfer the source marker color; otherwise transfer the source center color.
# - Small trace:
#   destination A touched at TL, original center = blue. Matched blue-border source is
#   marked at TL, so A receives source.marker.
#   destination B touched at BR, original center = blue. Same source is still matched,
#   but BR != TL, so B receives source.center.
#   Compared with the seed task, the extra diagonal marker is not decorative: it changes
#   which payload color is copied, and the path corner determines when that happens.
#
# b) Complexity calibration
# - The corner vocabulary is deliberately small: four named corners only.
# - Each generated example forces both outcomes to appear:
#   at least one destination receives a marker transfer and at least one receives a
#   center transfer. This keeps the corner-comparison rule visible in 2-4 examples.
# - The task keeps the original border-color lookup chain, so there is only one new
#   decision after source matching: compare destination path-corner vs source marker-corner.
#
# c) Implicit assumptions audit
# - The seed detector assumed every sprite had exactly two colors. That fails once
#   sources gain diagonal markers, so detection is split into:
#   ordinary 2-color destination boxes and
#   3-color marked source boxes.
# - The seed generator allowed box/plus shapes and clipping. Both become awkward here:
#   plus shapes do not occupy bbox corners, and clipped corners would hide either the
#   path-attachment site or the diagonal marker. This combo therefore restricts all
#   sprites to fully visible 3x3 boxes.
# - The seed port rule used "center -> border -> endpoint" on cardinal axes. That is
#   replaced everywhere with "bbox corner -> diagonal outside endpoint".
# - The seed transfer always wrote source.center. The combo transfer now assumes every
#   matched source has exactly one marked corner and two candidate payload colors.
#
# d) Placement legality
# - All sprites are full 3x3 boxes placed fully in-bounds.
# - Placement rejects overlap and 4-touching between colored components, so sprites stay
#   visually separated.
# - Destination corner ports are chosen from in-bounds diagonal-outside cells only.
# - The grey path is drawn only through background cells, so it never overwrites sprites.
#
# e) Visual observability
# - The input shows two independent but similarly shaped diagonal cues:
#   grey endpoints attached at destination corners, and marked source corners.
# - The output keeps only the two path-connected destinations, so the solver must infer
#   that corner-to-corner agreement, not raw color id ordering, decides whether the
#   marker color or center color survives.
#
# f) Visual cleanliness
# - No detached marker pixels are used; every marker stays inside its source box.
# - The path only touches destinations at one outside-diagonal endpoint cell, so the
#   corner relation is legible instead of being buried inside thicker overlap.
#
# g) Change inferability
# - Each matched source visibly exposes two payload colors: center and marker.
# - Because every generated instance exhibits both transfer modes, the examples show
#   that border-color lookup alone is insufficient; the corner match must also be read.
#
# h) Variant metric used after implementation
# - corner_match_count = number of destinations where touched corner == source marker corner.
# - marker_transfer_count = number of destinations receiving the source marker color.
# - center_transfer_count = number of destinations receiving the source center color.
# - Non-degenerate examples require corner_match_count == 1 and both transfer counts > 0.
#
# i) New solver assumptions checked against generator
# - Destination boxes are the only 2-color full 3x3 boxes: guaranteed.
# - Source/distractor boxes are full 3x3 marked boxes with ring counts {border:7, marker:1}
#   plus a unique center: guaranteed.
# - Each grey endpoint is a valid corner port for exactly one destination in generated
#   examples: enforced by placement and validated by oracle checks.
# - Every destination center color matches at most one source border color because
#   border colors are unique within each generated grid.

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
GEN_RETRIES = 600

CORNER_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, 2),
    "BL": (2, 0),
    "BR": (2, 2),
}

PORT_DELTAS = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
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
    border_color: int, center_color: int, marker_color: int, marker_corner: str
) -> np.ndarray:
    patch = _make_dest_patch(border_color, center_color)
    mr, mc = CORNER_OFFSETS[marker_corner]
    patch[mr, mc] = marker_color
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0, transparent=BG, clip=False)


def _erase_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            grid[r0 + dr, c0 + dc] = BG


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and 4-touching other colored components."""
    h, w = grid.shape
    patch_cells: set[tuple[int, int]] = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < h and 0 <= cc < w):
                return True
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


def _sprite_positions() -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in range(0, GRID_H - 2) for c0 in range(0, GRID_W - 2)]


def _is_edge(r0: int, c0: int) -> bool:
    return r0 == 0 or r0 == GRID_H - 3 or c0 == 0 or c0 == GRID_W - 3


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


def _corner_ports_for_bbox(
    bbox: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int], str]]:
    r0, c0, r1, c1 = bbox
    corners = {
        "TL": (r0, c0),
        "TR": (r0, c1),
        "BL": (r1, c0),
        "BR": (r1, c1),
    }
    ports: list[tuple[tuple[int, int], str]] = []
    for name, (cr, cc) in corners.items():
        dr, dc = PORT_DELTAS[name]
        pr, pc = cr + dr, cc + dc
        ports.append(((pr, pc), name))
    return ports


def _valid_corner_ports(grid: np.ndarray, r0: int, c0: int) -> list[tuple[tuple[int, int], str]]:
    bbox = (r0, c0, r0 + 2, c0 + 2)
    valid: list[tuple[tuple[int, int], str]] = []
    for (pr, pc), name in _corner_ports_for_bbox(bbox):
        if 0 <= pr < GRID_H and 0 <= pc < GRID_W and int(grid[pr, pc]) == BG:
            valid.append(((pr, pc), name))
    return valid


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two endpoints of the grey one-stroke trail under 8-connectivity."""
    cells = set((int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False))
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        for (pr, pc), _name in _corner_ports_for_bbox(sp["bbox"]):
            if (pr, pc) in cells:
                port_cells.add((pr, pc))

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


def _detect_dest_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """Destination sprite = a full 3x3 box with 8 border cells and 1 center cell."""
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        if len(pts) != 9:
            continue
        r0, c0, r1, c1 = _component_bbox(pts)
        if (r1 - r0, c1 - c0) != (2, 2):
            continue
        cells = {(int(r), int(c)) for r, c in pts}
        if any((r, c) not in cells for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            continue
        center_pos = (r0 + 1, c0 + 1)
        center_color = int(g[center_pos])
        ring_pos = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) != center_pos]
        ring_vals = [int(g[r, c]) for r, c in ring_pos]
        uniq = sorted(set(ring_vals + [center_color]))
        if len(uniq) != 2:
            continue
        if any(int(g[r, c]) != ring_vals[0] for r, c in ring_pos):
            continue
        border_color = int(ring_vals[0])
        if border_color in (bg, grey) or center_color in (bg, grey) or center_color == border_color:
            continue
        sprites.append(
            {
                "pts": pts,
                "bbox": (r0, c0, r1, c1),
                "border": border_color,
                "center": center_color,
                "center_pos": center_pos,
            }
        )
    return sprites


def _detect_marked_sources(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Marked source = a full 3x3 box with:
      - border color in 7 ring cells,
      - center color in the middle,
      - marker color in exactly one corner.
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

        center_pos = (r0 + 1, c0 + 1)
        center_color = int(g[center_pos])
        if center_color in (bg, grey):
            continue

        ring_pos = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) != center_pos]
        ring_vals = [int(g[r, c]) for r, c in ring_pos]
        counts = Counter(ring_vals)
        if sorted(counts.values()) != [1, 7]:
            continue

        border_color = max(counts, key=counts.get)
        marker_color = next(col for col, cnt in counts.items() if cnt == 1)
        if border_color in (bg, grey) or marker_color in (bg, grey):
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
                "bbox": (r0, c0, r1, c1),
                "border": int(border_color),
                "center": center_color,
                "marker": int(marker_color),
                "center_pos": center_pos,
                "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
                "marker_corner": marker_corner,
            }
        )
    return sources


def _corner_port_for_sprite(
    sprite: dict, endpoint: tuple[int, int]
) -> str | None:
    for port_cell, name in _corner_ports_for_bbox(sprite["bbox"]):
        if port_cell == endpoint:
            return name
    return None


def _resolve_destinations(
    sprites: list[dict], endpoint_a: tuple[int, int], endpoint_b: tuple[int, int]
) -> tuple[list[dict], list[str]]:
    endpoints = [endpoint_a, endpoint_b]
    opts: dict[tuple[int, int], list[tuple[dict, str]]] = {}
    for ep in endpoints:
        opts[ep] = []
        for sp in sprites:
            corner = _corner_port_for_sprite(sp, ep)
            if corner is not None:
                opts[ep].append((sp, corner))

    dest: list[dict] = [None, None]  # type: ignore[list-item]
    corners: list[str] = ["", ""]
    order = [0, 1] if len(opts[endpoint_a]) <= len(opts[endpoint_b]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = endpoints[i]
        other = endpoints[1 - i]
        cands = [(sp, corner) for sp, corner in opts[ep] if id(sp) not in used]
        assert cands, "expected a destination sprite for each trail endpoint"
        best_sp, best_corner = max(
            cands,
            key=lambda item: abs(item[0]["center_pos"][0] - other[0]) + abs(item[0]["center_pos"][1] - other[1]),
        )
        dest[i] = best_sp
        corners[i] = best_corner
        used.add(id(best_sp))

    assert all(d is not None for d in dest), "expected exactly 2 destination sprites"
    return dest, corners


def _transfer_color(src: dict, touched_corner: str) -> int:
    return int(src["marker"] if src["marker_corner"] == touched_corner else src["center"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer colors from marked source boxes into corner-selected destination boxes, then remove everything else.

    Input structure:
    - The grid contains a grey one-stroke trail, two ordinary bordered 3x3 destination boxes, and several marked 3x3 source boxes.
    - Each destination box has one border color and one center color.
    - Each marked source box has a border color, a center color, and one recolored diagonal corner marker.
    - The two grey trail endpoints sit diagonally outside one corner of the two destination boxes, so each endpoint identifies both a destination and a corner name.

    Rule:
    - Find the two destinations from the grey trail endpoints.
    - For each destination, use its original center color as a lookup key to find the marked source whose border color matches.
    - Compare the destination's touched corner with the source's marked corner:
      same corner -> write the source marker color;
      different corner -> write the source center color.
    - Delete every non-destination colored component while keeping the grey trail and the two destinations.

    Key insight: The destination center still names the source by border color, but the final transferred color depends on a second cue: whether the path-selected destination corner matches the source's diagonal marker corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    dest_sprites = _detect_dest_sprites(g, bg, grey)
    marked_sources = _detect_marked_sources(g, bg, grey)
    assert dest_sprites, "expected destination boxes"
    assert marked_sources, "expected marked source boxes"

    te1, te2 = _find_trail_endpoints(g, grey, dest_sprites)
    dest, touched_corners = _resolve_destinations(dest_sprites, te1, te2)
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
            rr, cc = int(r), int(c)
            if out[rr, cc] not in (bg, grey) and not keep[rr, cc]:
                keep[rr, cc] = True
                q.append((rr, cc))
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

    for d, touched_corner in zip(dest, touched_corners, strict=False):
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination color should match at most one marked source border"
        src = srcs[0]
        dr, dc = d["center_pos"]
        out[dr, dc] = _transfer_color(src, touched_corner)

    assert len(dest_ids) == 2
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


def _place_dests(
    base_grid: np.ndarray,
    border_colors: list[int],
    dest_lookup_colors: list[int],
    min_port_dist: int,
    require_distinct_corners: bool,
) -> list[dict] | None:
    all_pos = _sprite_positions()
    edge_pos = [p for p in all_pos if _is_edge(p[0], p[1])]
    int_pos = [p for p in all_pos if not _is_edge(p[0], p[1])]
    num_interior_dest = random.choices([0, 1], weights=INTERIOR_DEST_WEIGHTS)[0]

    dest_infos: list[dict] = []
    first_port: tuple[int, int] | None = None
    first_corner: str | None = None

    for i in range(2):
        patch = _make_dest_patch(border_colors[i], dest_lookup_colors[i])
        pool = list(int_pos if i < num_interior_dest else edge_pos)
        random.shuffle(pool)
        placed = False
        for r0, c0 in pool:
            if _overlaps(base_grid, patch, r0, c0):
                continue
            _place_sprite(base_grid, patch, r0, c0)
            valid_ports = _valid_corner_ports(base_grid, r0, c0)
            random.shuffle(valid_ports)
            chosen: tuple[tuple[int, int], str] | None = None
            for port_cell, corner_name in valid_ports:
                if i == 1:
                    assert first_port is not None and first_corner is not None
                    dist = abs(port_cell[0] - first_port[0]) + abs(port_cell[1] - first_port[1])
                    if dist < min_port_dist:
                        continue
                    if require_distinct_corners and corner_name == first_corner:
                        continue
                chosen = (port_cell, corner_name)
                break
            if chosen is None:
                _erase_sprite(base_grid, patch, r0, c0)
                continue
            port_cell, corner_name = chosen
            dest_infos.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "center_pos": (r0 + 1, c0 + 1),
                    "port": port_cell,
                    "corner": corner_name,
                    "border": border_colors[i],
                }
            )
            if i == 0:
                first_port = port_cell
                first_corner = corner_name
            placed = True
            break
        if not placed:
            return None
    return dest_infos


def _place_marked_source(
    grid: np.ndarray,
    border_color: int,
    center_color: int,
    marker_color: int,
    allowed_marker_corners: set[str] | None,
) -> dict | None:
    positions = _sprite_positions()
    random.shuffle(positions)

    corner_order = list(CORNER_OFFSETS)
    random.shuffle(corner_order)
    if allowed_marker_corners is not None:
        corner_order = [corner for corner in corner_order if corner in allowed_marker_corners]
        if not corner_order:
            return None

    for r0, c0 in positions:
        for marker_corner in corner_order:
            patch = _make_source_patch(border_color, center_color, marker_color, marker_corner)
            if _overlaps(grid, patch, r0, c0):
                continue
            _place_sprite(grid, patch, r0, c0)
            return {
                "patch": patch,
                "r0": r0,
                "c0": c0,
                "center_pos": (r0 + 1, c0 + 1),
                "marker_corner": marker_corner,
                "border": border_color,
                "center": center_color,
                "marker": marker_color,
            }
    return None


def _build_output_grid(
    base_grid: np.ndarray, dest_infos: list[dict], dest_to_source: list[int], placed_sources: dict[int, dict]
) -> tuple[np.ndarray, dict]:
    out = base_grid.copy()
    transfer_modes: list[str] = []
    corner_matches = 0
    for dest_idx, src_idx in enumerate(dest_to_source):
        src = placed_sources[src_idx]
        dest_info = dest_infos[dest_idx]
        match = src["marker_corner"] == dest_info["corner"]
        if match:
            corner_matches += 1
        transfer_modes.append("marker" if match else "center")
        out[dest_info["center_pos"]] = int(src["marker"] if match else src["center"])
    metrics = {
        "corner_match_count": corner_matches,
        "marker_transfer_count": transfer_modes.count("marker"),
        "center_transfer_count": transfer_modes.count("center"),
        "transfer_modes": transfer_modes,
        "dest_corners": [info["corner"] for info in dest_infos],
    }
    return out, metrics


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
    require_distinct_corners = len(layout["source_indices"]) == 1
    dest_infos = _place_dests(
        base_grid,
        border_colors,
        colors["dest_lookup_colors"],
        min_port_dist,
        require_distinct_corners=require_distinct_corners,
    )
    if dest_infos is None:
        return None

    path_cells = _gen_winding_path(base_grid, dest_infos[0]["port"], dest_infos[1]["port"], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None
    for r, c in path_cells:
        if base_grid[r, c] != BG:
            return None
        base_grid[r, c] = GREY

    for info in dest_infos:
        pr, pc = info["port"]
        deg = sum(
            1
            for nr, nc in neighbors8(pr, pc)
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and base_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None
        if sum(
            1
            for d in dest_infos
            if _corner_port_for_sprite({"bbox": d["bbox"]}, info["port"]) is not None
        ) != 1:
            return None

    input_grid = base_grid.copy()
    placed_sources: dict[int, dict] = {}

    if len(layout["source_indices"]) == 1:
        src_idx = layout["source_indices"][0]
        spec = colors["source_specs"][src_idx]
        chosen_corner = random.choice([dest_infos[0]["corner"], dest_infos[1]["corner"]])
        info = _place_marked_source(
            input_grid,
            border_colors[src_idx],
            spec["center"],
            spec["marker"],
            allowed_marker_corners={chosen_corner},
        )
        if info is None:
            return None
        placed_sources[src_idx] = info
    else:
        required_matches = [True, False]
        random.shuffle(required_matches)
        for src_idx, dest_idx, need_match in zip(
            layout["source_indices"], [0, 1], required_matches, strict=False
        ):
            spec = colors["source_specs"][src_idx]
            dest_corner = dest_infos[dest_idx]["corner"]
            if need_match:
                allowed = {dest_corner}
            else:
                allowed = set(CORNER_OFFSETS) - {dest_corner}
            info = _place_marked_source(
                input_grid,
                border_colors[src_idx],
                spec["center"],
                spec["marker"],
                allowed_marker_corners=allowed,
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
            allowed_marker_corners=None,
        )
        if info is None:
            return None
        placed_sources[src_idx] = info

    output_grid, metrics = _build_output_grid(base_grid, dest_infos, layout["dest_to_source"], placed_sources)
    if metrics["marker_transfer_count"] == 0 or metrics["center_transfer_count"] == 0:
        return None
    if metrics["corner_match_count"] != 1:
        return None

    try:
        out = main(input_grid)
        if np.array_equal(out, output_grid):
            metrics.update(
                {
                    "source_count": len(layout["source_indices"]),
                    "distractor_count": len(layout["distractor_indices"]),
                }
            )
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
