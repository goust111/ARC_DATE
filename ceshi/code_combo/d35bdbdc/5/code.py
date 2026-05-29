# Combo variant of d35bdbdc (index 5)
# combo_spec: D3/1+D13/1
# Name: 三层嵌套精灵+图案转移

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors4, neighbors8, paste


# Design analysis
# ---------------
# a) Concrete behavior trace:
#    The original task has 3x3 sprites with one border color and one center key, and the output
#    changes only the destination center pixel. This combo keeps the same lookup chain
#    "destination center color -> matching source border color", but upgrades each source payload
#    to a nested 3x3 interior: 8 cells of one middle color around a distinct core color.
#    Example trace:
#      input source: border=7, interior=
#        4 4 4
#        4 8 4
#        4 4 4
#      input destination: border=2, interior=
#        6 6 6
#        6 7 6
#        6 6 6
#      because destination center=7, it points to the source above.
#      original rule would only write the source core to the destination center;
#      combo rule writes the whole nested interior, so output destination becomes
#        4 4 4
#        4 8 4
#        4 4 4
#    The visible difference is that the transferred content is now a spatial 3x3 pattern, and
#    that pattern comes from a richer three-layer source object rather than a one-pixel source.
#
# b) Complexity calibration:
#    A full 5x5 framed sprite is the smallest clean geometry that can simultaneously show
#    a persistent border and a transferred 3x3 interior. To keep placement unambiguous, the grid
#    is enlarged to 15x15 and sprites sit on a small set of non-overlapping anchors. Source
#    interiors are not arbitrary; they are restricted to "uniform middle ring + distinct core",
#    which makes the D3 change visible while keeping the hypothesis space small enough for ARC.
#
# c) Implicit assumption audit:
#    The original solver assumed: two-color components, a port two cells from the center, and a
#    one-pixel writeback. None of those survive this combo. Detection must instead validate 5x5
#    framed sprites with non-grey 3x3 interiors; the port now lies three cells from the center
#    with the border sitting two cells away; the writeback must copy an entire 3x3 array. The
#    endpoint->destination matching rule is still based only on geometry, so the generator must
#    ensure non-destination sprites cannot also claim the chosen endpoints as valid ports.
#
# d) Placement legality:
#    The generator uses five safe 5x5 anchors (four corners plus center), reserves every chosen
#    footprint before path generation, and only chooses grey endpoints from free cells one step
#    beyond a destination frame. This prevents overlap, path/sprite collisions, and illegal ports.
#
# e) Visual observability:
#    Humans can see the lookup key because each destination input has a simple interior whose
#    center pixel equals some source border color. They can also see what is copied because sources
#    have a visibly different nested 3x3 payload and the output preserves that payload's layout.
#    No ordering depends on numeric color ids; all decisions are based on path endpoints and shape.
#
# f) Visual cleanliness:
#    Every sprite occupies its own 5x5 island, the grey trail stays in empty space, and the output
#    removes all non-destination sprites. This leaves two framed targets plus one path, so the
#    transformed region is easy to isolate by eye.
#
# g) Change inferability:
#    The number of colors is controlled: each sprite has one border and two interior colors at most,
#    and destination input interiors are intentionally simple. Across examples, the repeated pattern
#    "destination center equals source border; destination output interior equals that source's full
#    nested interior" provides enough structure to infer both the color lookup and the wider
#    transfer scope from just a few demonstrations.
#
# New solver assumptions the generator must satisfy:
# - Every sprite is a full 5x5 frame with a uniform outer ring and a non-bg/non-grey 3x3 interior.
# - Each destination center key matches at most one source border color.
# - Source interiors must be genuine nested patterns, and destination input interiors must differ
#   from the transferred source interiors so the whole-pattern transfer is observable.
# - Non-destination sprites must not also be valid endpoint matches for the chosen grey ports.


SPRITE_SIZE = 5
INNER_SIZE = 3
CENTER_OFFSET = 2
PORT_DIST = 3
PORT_BORDER_DIST = 2

GRID_H, GRID_W = 15, 15
BG, GREY = 0, 5
SPRITE_COUNTS = [4, 5]
SOURCE_COUNTS = [1, 2]
PATH_LEN_MIN, PATH_LEN_MAX = 12, 30
MIN_PORT_DIST = 6
MAX_THICKNESS = 3
PATH_WANDER_AWAY_P = 0.5
PATH_WANDER_DIAG_P = 0.6
PATH_MODERATE_RANDOM_P = 0.3
GEN_RETRIES = 500

ANCHORS = [(0, 0), (0, 10), (10, 0), (10, 10), (5, 5)]


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _nested_interior(middle: int, core: int) -> np.ndarray:
    interior = np.full((INNER_SIZE, INNER_SIZE), middle, dtype=int)
    interior[1, 1] = core
    return interior


def _simple_dest_interior(key_color: int, fill_color: int) -> np.ndarray:
    interior = np.full((INNER_SIZE, INNER_SIZE), fill_color, dtype=int)
    interior[1, 1] = key_color
    return interior


def _make_sprite_patch(border_color: int, interior: np.ndarray) -> np.ndarray:
    patch = np.full((SPRITE_SIZE, SPRITE_SIZE), border_color, dtype=int)
    patch[1:-1, 1:-1] = np.asarray(interior, dtype=int)
    return patch


def _place_sprite(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0)


def _sprite_footprint(r0: int, c0: int) -> tuple[slice, slice]:
    return slice(r0, r0 + SPRITE_SIZE), slice(c0, c0 + SPRITE_SIZE)


def _paint_reserved(mask: np.ndarray, r0: int, c0: int) -> None:
    rs, cs = _sprite_footprint(r0, c0)
    mask[rs, cs] = True


def _center_of(top_left: tuple[int, int]) -> tuple[int, int]:
    return top_left[0] + CENTER_OFFSET, top_left[1] + CENTER_OFFSET


def _detect_framed_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Framed sprite = a full 5x5 component whose outer ring is one border color and whose
    3x3 interior contains only non-bg, non-grey colors. Sources and destinations share this
    geometry; the transferred content is the full 3x3 interior.
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    ring_mask = np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
    ring_mask[1:-1, 1:-1] = False

    for pts in comps:
        if len(pts) != SPRITE_SIZE * SPRITE_SIZE:
            continue
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        if (r1 - r0 + 1) != SPRITE_SIZE or (c1 - c0 + 1) != SPRITE_SIZE:
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border_vals = np.unique(patch[ring_mask])
        if len(border_vals) != 1:
            continue
        border_color = int(border_vals[0])
        if border_color in (bg, grey):
            continue

        interior = patch[1:-1, 1:-1].copy()
        if np.any(interior == bg) or np.any(interior == grey):
            continue

        center_pos = (r0 + CENTER_OFFSET, c0 + CENTER_OFFSET)
        sprites.append(
            {
                "pts": [(int(r), int(c)) for r, c in pts],
                "border": border_color,
                "center": int(interior[1, 1]),
                "center_pos": center_pos,
                "top_left": (r0, c0),
                "interior": interior.copy(),
            }
        )

    return sprites


def _ports_for_sprite(sprite: dict, g: np.ndarray, eps: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    A grey endpoint is a valid port if it lies 3 cells from the sprite center in a cardinal
    direction and the intervening cell at distance 2 is part of the border ring.
    """
    cr, cc = sprite["center_pos"]
    border = int(sprite["border"])
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        br, bc = cr + PORT_BORDER_DIST * dr, cc + PORT_BORDER_DIST * dc
        er, ec = cr + PORT_DIST * dr, cc + PORT_DIST * dc
        if (er, ec) not in eps:
            continue
        if 0 <= br < g.shape[0] and 0 <= bc < g.shape[1] and int(g[br, bc]) == border:
            ports.append((er, ec))
    return ports


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two grey endpoints of the one-stroke trail under 8-connectivity."""
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
        border = int(sp["border"])
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            br, bc = cr + PORT_BORDER_DIST * dr, cc + PORT_BORDER_DIST * dc
            er, ec = cr + PORT_DIST * dr, cc + PORT_DIST * dc
            if (er, ec) in cells and 0 <= br < h and 0 <= bc < w and int(g[br, bc]) == border:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer full nested interior patterns between framed sprites selected by the grey trail.

    Input structure:
    - The grid contains background, a grey (5) one-stroke path, and several 5x5 framed sprites.
    - Each sprite has a uniform border ring and a non-grey 3x3 interior; source sprites use a
      nested interior made of one middle color surrounding a distinct core color.
    - The two grey trail endpoints sit one cell beyond the border of the two destination sprites.

    Rule:
    - Use the grey trail endpoints to identify the two destination sprites.
    - For each destination, treat its original center pixel as a lookup key and find the source
      sprite whose border color matches that key.
    - Copy the matched source sprite's entire 3x3 interior into the destination interior without
      rotation or reflection, then delete every non-destination sprite.
    - Keep the grey path and the destination borders in place.

    Key insight:
    - The path tells you which two sprites survive, while each surviving sprite's center color is
      only an address pointing to some other sprite's border.
    - What gets transferred is not a single color but the source's whole nested interior pattern,
      so the destination must inherit both the middle ring and the core in their original layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites = _detect_framed_sprites(g, bg, grey)
    assert sprites, "expected at least one framed sprite in the grid"

    te1, te2 = _find_trail_endpoints(g, grey, sprites)
    trail_eps = {te1, te2}
    eps_pair = [te1, te2]
    opts = {ep: [sp for sp in sprites if ep in _ports_for_sprite(sp, g, trail_eps)] for ep in eps_pair}

    dest: list[dict | None] = [None, None]
    order = [0, 1] if len(opts[te1]) <= len(opts[te2]) else [1, 0]
    used: set[int] = set()
    for i in order:
        ep = eps_pair[i]
        other = eps_pair[1 - i]
        cands = [sp for sp in opts[ep] if id(sp) not in used]
        assert cands, "expected an endpoint-matched destination candidate"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly 2 destination sprites from trail endpoints"
    dest_sprites = [d for d in dest if d is not None]
    dest_ids = {id(sp) for sp in dest_sprites}

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
    for d in dest_sprites:
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

    for d in dest_sprites:
        want_border = int(d["center"])
        srcs = by_border.get(want_border, [])
        if not srcs:
            continue
        assert len(srcs) == 1, "each destination key should match at most one source border"
        src = srcs[0]
        r0, c0 = d["top_left"]
        out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = src["interior"]

    return out


def _too_thick(r: int, c: int, occupied: set[tuple[int, int]]) -> bool:
    """Check whether adding (r,c) would make the 8-neighborhood path too thick."""
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
    """Generate a winding 8-connected grey trail that stays in empty cells."""
    path = [start]
    visited = {start}
    cr, cc = start

    def score(m: tuple[int, int, bool]) -> int:
        return abs(m[0] - end[0]) + abs(m[1] - end[1])

    for _ in range(target_len + 40):
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


def _endpoint_deg8(path_cells: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum(1 for nr, nc in neighbors8(r, c) if (nr, nc) in path_cells)


def _port_options(top_left: tuple[int, int], reserved: np.ndarray) -> list[tuple[int, int]]:
    cr, cc = _center_of(top_left)
    ports: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        er, ec = cr + PORT_DIST * dr, cc + PORT_DIST * dc
        if 0 <= er < GRID_H and 0 <= ec < GRID_W and not reserved[er, ec]:
            ports.append((er, ec))
    return ports


def _choose_two_distinct_ports(
    dest_positions: list[tuple[int, int]], reserved: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    opts0 = _port_options(dest_positions[0], reserved)
    opts1 = _port_options(dest_positions[1], reserved)
    if not opts0 or not opts1:
        return None
    pairs = [(p0, p1) for p0 in opts0 for p1 in opts1 if p0 != p1]
    pairs = [
        pair
        for pair in pairs
        if abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1]) >= MIN_PORT_DIST
    ]
    if not pairs:
        return None
    return random.choice(pairs)


def _pick_color(exclude: set[int]) -> int:
    candidates = [c for c in range(1, 10) if c not in exclude]
    if not candidates:
        raise ValueError("no colors available")
    return random.choice(candidates)


def _assign_payloads(border_colors: list[int], num_sprites: int) -> dict | None:
    remaining = list(range(2, num_sprites))
    if not remaining:
        return None

    max_sources = min(2, len(remaining))
    source_count_choices = [n for n in SOURCE_COUNTS if n <= max_sources]
    if not source_count_choices:
        return None
    num_sources = random.choice(source_count_choices)
    random.shuffle(remaining)
    source_indices = remaining[:num_sources]
    distractor_indices = remaining[num_sources:]

    if num_sources == 1:
        dest_keys = [border_colors[source_indices[0]], border_colors[source_indices[0]]]
        source_for_dest = [source_indices[0], source_indices[0]]
    else:
        picked = source_indices[:2]
        dest_keys = [border_colors[picked[0]], border_colors[picked[1]]]
        source_for_dest = picked

    source_interiors: dict[int, np.ndarray] = {}
    source_specs: dict[int, tuple[int, int]] = {}
    used_nested: set[tuple[int, int]] = set()
    for si in source_indices:
        options: list[tuple[int, int]] = []
        for middle in range(1, 10):
            for core in range(1, 10):
                if middle in {BG, GREY, border_colors[si]}:
                    continue
                if core in {BG, GREY, border_colors[si], middle}:
                    continue
                if (middle, core) in used_nested:
                    continue
                options.append((middle, core))
        if not options:
            return None
        middle, core = random.choice(options)
        used_nested.add((middle, core))
        source_specs[si] = (middle, core)
        source_interiors[si] = _nested_interior(middle, core)

    dest_input_interiors: list[np.ndarray] = []
    dest_output_interiors: list[np.ndarray] = []
    for i, source_idx in enumerate(source_for_dest):
        key = dest_keys[i]
        src_middle, src_core = source_specs[source_idx]
        fill = _pick_color({BG, GREY, border_colors[i], key, src_middle, src_core})
        dest_input = _simple_dest_interior(key, fill)
        dest_output = source_interiors[source_idx].copy()
        if np.array_equal(dest_input, dest_output):
            return None
        dest_input_interiors.append(dest_input)
        dest_output_interiors.append(dest_output)

    distractor_interiors: dict[int, np.ndarray] = {}
    for di in distractor_indices:
        if random.random() < 0.6:
            middle = _pick_color({BG, GREY, border_colors[di]})
            core = _pick_color({BG, GREY, border_colors[di], middle})
            distractor_interiors[di] = _nested_interior(middle, core)
        else:
            key = _pick_color({BG, GREY, border_colors[di]})
            fill = _pick_color({BG, GREY, border_colors[di], key})
            distractor_interiors[di] = _simple_dest_interior(key, fill)

    return {
        "source_indices": source_indices,
        "distractor_indices": distractor_indices,
        "source_interiors": source_interiors,
        "distractor_interiors": distractor_interiors,
        "dest_input_interiors": dest_input_interiors,
        "dest_output_interiors": dest_output_interiors,
    }


def _gen_attempt() -> np.ndarray | None:
    num_sprites = random.choice(SPRITE_COUNTS)
    chosen_positions = random.sample(ANCHORS, num_sprites)
    border_colors = random.sample([c for c in range(1, 10) if c != GREY], num_sprites)

    payloads = _assign_payloads(border_colors, num_sprites)
    if payloads is None:
        return None

    reserved = np.zeros((GRID_H, GRID_W), dtype=bool)
    for r0, c0 in chosen_positions:
        _paint_reserved(reserved, r0, c0)

    dest_positions = chosen_positions[:2]
    ports = _choose_two_distinct_ports(dest_positions, reserved)
    if ports is None:
        return None

    path_blockers = np.zeros((GRID_H, GRID_W), dtype=int)
    for i, (r0, c0) in enumerate(chosen_positions):
        if i < 2:
            patch = _make_sprite_patch(border_colors[i], payloads["dest_output_interiors"][i])
        else:
            patch = np.full((SPRITE_SIZE, SPRITE_SIZE), 9, dtype=int)
        _place_sprite(path_blockers, patch, r0, c0)

    p0, p1 = ports
    cheb = max(abs(p0[0] - p1[0]), abs(p0[1] - p1[1]))
    lo = max(PATH_LEN_MIN, cheb + 1)
    if lo > PATH_LEN_MAX:
        return None
    target_len = random.randint(lo, PATH_LEN_MAX)
    path_cells = _gen_winding_path(path_blockers, p0, p1, target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None
    path_set = set(path_cells)
    if _endpoint_deg8(path_set, p0) != 1 or _endpoint_deg8(path_set, p1) != 1:
        return None

    output_grid = np.zeros((GRID_H, GRID_W), dtype=int)
    for i in range(2):
        r0, c0 = dest_positions[i]
        patch = _make_sprite_patch(border_colors[i], payloads["dest_output_interiors"][i])
        _place_sprite(output_grid, patch, r0, c0)

    for r, c in path_cells:
        if output_grid[r, c] != BG:
            return None
        output_grid[r, c] = GREY

    input_grid = output_grid.copy()
    for i in range(2):
        r0, c0 = dest_positions[i]
        input_grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = payloads["dest_input_interiors"][i]

    for j, source_idx in enumerate(payloads["source_indices"]):
        idx = 2 + j
        r0, c0 = chosen_positions[idx]
        patch = _make_sprite_patch(border_colors[idx], payloads["source_interiors"][source_idx])
        _place_sprite(input_grid, patch, r0, c0)

    offset = 2 + len(payloads["source_indices"])
    for j, distractor_idx in enumerate(payloads["distractor_indices"]):
        idx = offset + j
        r0, c0 = chosen_positions[idx]
        patch = _make_sprite_patch(border_colors[idx], payloads["distractor_interiors"][distractor_idx])
        _place_sprite(input_grid, patch, r0, c0)

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
