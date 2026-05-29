# Combo variant of d35bdbdc (index 16)
# combo_spec: D6/2+D13/3
# Name: 未连接精灵缩为单像素+计数式转移

"""
Design analysis for the fused D6/2 + D13/3 variant.

concretization_and_trace:
  - We keep the original high-level cast: a grey one-stroke trail chooses exactly two
    destination sprites, while all other sprites are disconnected from that trail.
  - To make counting visible, every sprite becomes a 5x5 frame with a 3x3 uniform
    interior. A destination's interior color is still a lookup key, but now that key can
    match multiple source frames that all share the same border color and interior color.
  - Output behavior on one concrete group:
      old base task:
        destination lookup color 7 -> exactly one source with border 7 -> destination
        center becomes that source's interior color
      fused task:
        destination lookup color 7 -> suppose there are 3 sources with border 7 and
        shared interior color 2 -> destination interior is cleared and rewritten as
        three color-2 dots in fixed tally cells; those 3 source frames themselves shrink
        to three isolated color-2 center pixels elsewhere on the board
  - This creates the intended contrast: destination dots are an in-frame count code,
    while disconnected sprites survive only as single-pixel remnants.

complexity_calibration:
  - Counts are restricted to the small set {(2,1), (2,2), (3,1)} across the two
    destinations. That guarantees at least one destination shows a genuine multi-dot tally,
    so the task does not collapse back to one-pixel transfer.
  - Source groups are repeated by border color and by interior color. This gives the
    solver and the human viewer visible evidence that multiplicity matters, not the order
    of objects or any hidden numeric property of colors.
  - We use exactly two source groups, one per destination lookup color. This keeps the
    mapping legible without introducing too many distinct cases in one sample.

implicit_assumption_review:
  - The original bordered-sprite parser expected exactly one center pixel. That assumption
    breaks as soon as destinations must hold 2-3 dots, so detection is replaced with a
    rectangular-frame parser whose interior is a 3x3 block.
  - The original border->source lookup assumed at most one match. That is now false by
    design, so the solver counts the full matching group and asserts only that the group
    shares one interior color.
  - The original cleanup logic deleted every non-destination component, which would erase
    the new D6 remnants. The fused solver therefore rebuilds the output from scratch:
    path stays, destinations are redrawn as tally frames, and every non-destination frame
    contributes exactly one center pixel.
  - Because destination tally dots can be disconnected from the frame border, drawing from
    scratch is safer than using connectivity-based cleanup.

placement_legality:
  - All frames are placed fully in-bounds on a 20x20 board, with no overlap and no
    4-neighbor touching against other colored objects.
  - Destination ports are one cell outside a frame-side midpoint and are forced to remain
    in-bounds, so the grey trail endpoints always have valid geometric meaning.
  - Source placement happens only after the path is fixed, which prevents a later source
    frame from blocking or corrupting the trail.

observability_and_cleanliness:
  - From a few examples, a human can see two simultaneous regularities:
      1. every disconnected frame collapses to one interior-colored dot at its center
      2. each destination frame shows a number of interior dots equal to how many detached
         frames share the lookup border color named by that destination's input interior
  - No step uses color-id ordering, numeric ranking, or hidden metadata. Only visibly
    repeated frame-border groups and visible path connectivity matter.
  - The output stays readable because all count dots are confined to destination interiors,
    while all remnants are isolated single pixels away from those frames.

new_solver_assumptions_checked_against_generator:
  - Every matching source group has a shared interior color: guaranteed by generator.
  - Each destination lookup color corresponds to exactly one source group: guaranteed by
    using distinct lookup border colors for the two groups.
  - At least one destination has count > 1 in every sample: guaranteed by the restricted
    count patterns used by the generator.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, neighbors8, paste


FRAME_SIZE = 5
GRID_H = 20
GRID_W = 20
BG = 0
GREY = 5
PATH_LEN_MIN = 14
PATH_LEN_MAX = 42
MIN_PORT_DIST = 10
COUNT_PATTERNS = [(2, 1), (2, 2), (3, 1)]
COUNT_PATTERN_WEIGHTS = [4, 3, 2]
GEN_RETRIES = 500


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _frame_patch(border_color: int, inner_color: int) -> np.ndarray:
    """5x5 frame with a uniform 3x3 interior."""
    patch = np.full((FRAME_SIZE, FRAME_SIZE), border_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _dot_cells_for_bbox(bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    """Fixed tally layout inside a 3x3 interior."""
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    center = (mid_r, mid_c)
    left = (mid_r, mid_c - 1)
    right = (mid_r, mid_c + 1)
    ordered = [
        center,
        left,
        right,
        (mid_r - 1, mid_c),
        (mid_r + 1, mid_c),
        (mid_r - 1, mid_c - 1),
        (mid_r - 1, mid_c + 1),
        (mid_r + 1, mid_c - 1),
        (mid_r + 1, mid_c + 1),
    ]
    if count <= 0:
        return []
    if count == 1:
        return [center]
    if count == 2:
        return [left, right]
    return ordered[:count]


def _output_dest_patch(border_color: int, dot_color: int, count: int, bg: int) -> np.ndarray:
    """5x5 destination frame with a cleared interior and tally dots."""
    patch = _frame_patch(border_color, bg)
    for rr, cc in _dot_cells_for_bbox((0, 0, FRAME_SIZE - 1, FRAME_SIZE - 1), count):
        patch[rr, cc] = dot_color
    return patch


def _detect_framed_sprites(g: np.ndarray, bg: int, grey: int) -> list[dict]:
    """
    Framed sprite = a full 5x5 rectangle using exactly 2 colors:
      - one border color on the outer ring
      - one uniform interior color on the inner 3x3 block
    """
    mask = (g != bg) & (g != grey)
    comps = components_mask(mask, connectivity=4)
    sprites: list[dict] = []
    for pts in comps:
        rs = [int(r) for r, _ in pts]
        cs = [int(c) for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0 + 1, c1 - c0 + 1) != (FRAME_SIZE, FRAME_SIZE):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg) or np.any(patch == grey):
            continue
        border_mask = np.zeros_like(patch, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        border_vals = {int(x) for x in patch[border_mask]}
        inner_vals = {int(x) for x in patch[1:-1, 1:-1].flatten()}
        if len(border_vals) != 1 or len(inner_vals) != 1:
            continue
        border_color = next(iter(border_vals))
        center_color = next(iter(inner_vals))
        if border_color in (bg, grey) or center_color in (bg, grey) or border_color == center_color:
            continue
        sprites.append(
            {
                "pts": [(int(r), int(c)) for r, c in pts],
                "bbox": (r0, c0, r1, c1),
                "border": border_color,
                "center": center_color,
                "center_pos": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    return sprites


def _frame_port_candidates(sprite: dict) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return (endpoint, border-midpoint) candidates around a 5x5 frame."""
    r0, c0, r1, c1 = sprite["bbox"]
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    return [
        ((r0 - 1, mid_c), (r0, mid_c)),
        ((r1 + 1, mid_c), (r1, mid_c)),
        ((mid_r, c0 - 1), (mid_r, c0)),
        ((mid_r, c1 + 1), (mid_r, c1)),
    ]


def _ports_for_sprite(
    sprite: dict, g: np.ndarray, eps: set[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    """Grey endpoints that sit exactly one step outside the midpoint of a frame side."""
    ports: list[tuple[int, int]] = []
    for endpoint, border_mid in _frame_port_candidates(sprite):
        er, ec = endpoint
        br, bc = border_mid
        if not (0 <= er < g.shape[0] and 0 <= ec < g.shape[1]):
            continue
        if eps is not None and endpoint not in eps:
            continue
        if int(g[br, bc]) == int(sprite["border"]):
            ports.append(endpoint)
    return ports


def _find_trail_endpoints(
    g: np.ndarray, grey: int, sprites: list[dict],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Find the two endpoints of the grey one-stroke trail under 8-connectivity."""
    cells = {(int(r), int(c)) for r, c in zip(*np.where(g == grey), strict=False)}
    assert cells, "expected a grey trail"
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        adj[(r, c)] = [(nr, nc) for nr, nc in neighbors8(r, c) if (nr, nc) in cells]

    deg1 = [cell for cell, nbs in adj.items() if len(nbs) == 1]
    if len(deg1) >= 2:
        return deg1[0], deg1[1]

    port_cells: set[tuple[int, int]] = set()
    for sp in sprites:
        for endpoint, _border_mid in _frame_port_candidates(sp):
            if endpoint in cells:
                port_cells.add(endpoint)

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count matching source frames inside the destinations and collapse every detached frame to one pixel.

    Input structure:
    - The board contains background, a grey (5) one-stroke trail, and several 5x5 framed sprites.
    - Each frame has a single border color and a uniform 3x3 interior color.
    - Exactly two frames touch the grey trail through endpoint ports just outside a side midpoint; all other frames are disconnected.

    Rule:
    - Use the two trail endpoints to identify the destination frames.
    - For each destination, treat its original interior color as a lookup key and count how many non-destination frames have that border color.
    - Redraw each destination as the same border frame with a cleared interior, then place N tally dots in fixed interior cells using the shared interior color of the matching source group.
    - Every non-destination frame is erased except for one surviving center pixel in its own interior color; the grey trail is preserved.

    Key insight:
    - One visual channel shows multiplicity inside the connected destinations, while a second channel keeps the disconnected frames only as singleton remnants. The destination interior color names a border-color group, and the size of that visible group becomes the tally drawn inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    grey = GREY

    sprites = _detect_framed_sprites(g, bg, grey)
    assert len(sprites) >= 2, "expected at least two framed sprites"

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
        assert cands, "expected a destination frame for each trail endpoint"
        best = max(
            cands,
            key=lambda sp: abs(sp["center_pos"][0] - other[0]) + abs(sp["center_pos"][1] - other[1]),
        )
        dest[i] = best
        used.add(id(best))
    assert all(d is not None for d in dest), "expected exactly two destinations"
    dest_ids = {id(sp) for sp in dest}

    by_border: dict[int, list[dict]] = {}
    for sp in sprites:
        by_border.setdefault(int(sp["border"]), []).append(sp)

    out = np.full(g.shape, bg, dtype=int)
    out[g == grey] = grey

    for sp in sprites:
        if id(sp) in dest_ids:
            continue
        cr, cc = sp["center_pos"]
        out[cr, cc] = int(sp["center"])

    for d in dest:
        want_border = int(d["center"])
        srcs = [sp for sp in by_border.get(want_border, []) if id(sp) not in dest_ids]
        count = len(srcs)
        dot_color = bg
        if srcs:
            fill_colors = {int(sp["center"]) for sp in srcs}
            assert len(fill_colors) == 1, "matching source group must share one interior color"
            dot_color = next(iter(fill_colors))
        r0, c0, _r1, _c1 = d["bbox"]
        paste(out, _output_dest_patch(int(d["border"]), dot_color, count, bg), r0, c0)

    return out


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    paste(grid, patch, r0, c0)


def _overlaps(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    """Reject overlap and 4-neighbor touching with existing non-bg, non-grey pixels."""
    h, w = patch.shape
    patch_cells = set()
    for dr in range(h):
        for dc in range(w):
            if patch[dr, dc] == BG:
                continue
            rr, cc = r0 + dr, c0 + dc
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                return True
            if int(grid[rr, cc]) != BG:
                return True
            patch_cells.add((rr, cc))
    for rr, cc in patch_cells:
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if (nr, nc) in patch_cells:
                continue
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if int(grid[nr, nc]) not in (BG, GREY):
                    return True
    return False


def _sprite_positions() -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in range(0, GRID_H - FRAME_SIZE + 1) for c0 in range(0, GRID_W - FRAME_SIZE + 1)]


def _edgeish_positions() -> list[tuple[int, int]]:
    near = {1, GRID_H - FRAME_SIZE - 1}
    return [(r0, c0) for r0, c0 in _sprite_positions() if r0 in near or c0 in near]


def _port_candidates_from_bbox(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    candidates = [(r0 - 1, mid_c), (r1 + 1, mid_c), (mid_r, c0 - 1), (mid_r, c1 + 1)]
    return [(r, c) for r, c in candidates if 0 <= r < GRID_H and 0 <= c < GRID_W]


def _shortest_path8(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Shortest 8-neighbor path on background cells, with randomized tie-order."""
    blocked = set() if blocked is None else set(blocked)
    q = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        nbs = list(neighbors8(*cur))
        random.shuffle(nbs)
        for nr, nc in nbs:
            nb = (nr, nc)
            if not (0 <= nr < GRID_H and 0 <= nc < GRID_W):
                continue
            if nb in parent or nb in blocked:
                continue
            if nb != end and int(grid[nr, nc]) != BG:
                continue
            parent[nb] = cur
            q.append(nb)
    if end not in parent:
        return None
    path = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _gen_winding_path(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int], target_len: int):
    """Generate a thin 8-connected path, preferring a random detour when feasible."""
    candidates = [
        (r, c)
        for r in range(GRID_H)
        for c in range(GRID_W)
        if int(grid[r, c]) == BG and (r, c) not in (start, end)
    ]
    random.shuffle(candidates)
    for waypoint in candidates[:40]:
        p1 = _shortest_path8(grid, start, waypoint)
        if p1 is None:
            continue
        blocked = set(p1[:-1])
        p2 = _shortest_path8(grid, waypoint, end, blocked=blocked)
        if p2 is None:
            continue
        path = p1 + p2[1:]
        if PATH_LEN_MIN <= len(path) <= PATH_LEN_MAX:
            return path

    direct = _shortest_path8(grid, start, end)
    if direct is None:
        return None
    return direct if len(direct) <= PATH_LEN_MAX else None


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


def _build_role_colors():
    counts = list(random.choices(COUNT_PATTERNS, weights=COUNT_PATTERN_WEIGHTS, k=1)[0])
    random.shuffle(counts)
    palette = [c for c in range(1, 10) if c != GREY]
    random.shuffle(palette)

    dest_border_colors = palette[:2]
    lookup_colors = palette[2:4]
    dot_colors = palette[4:6]

    dest_specs = []
    for i in range(2):
        dest_specs.append(
            {
                "border": dest_border_colors[i],
                "lookup": lookup_colors[i],
                "dot_color": dot_colors[i],
                "count": counts[i],
            }
        )

    source_specs = []
    for i, count in enumerate(counts):
        for _ in range(count):
            source_specs.append((lookup_colors[i], dot_colors[i]))
    random.shuffle(source_specs)
    return dest_specs, source_specs


def _place_dests(layout_grid: np.ndarray, dest_specs: list[dict]):
    positions: list[tuple[int, int]] = []
    port_cells: list[tuple[int, int]] = []
    pool = _edgeish_positions()
    random.shuffle(pool)

    for i, spec in enumerate(dest_specs):
        patch = _frame_patch(spec["border"], spec["lookup"])
        placed = False
        for r0, c0 in pool:
            if _overlaps(layout_grid, patch, r0, c0):
                continue
            bbox = (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1)
            ports = _port_candidates_from_bbox(bbox)
            random.shuffle(ports)
            if i == 1:
                ports = [
                    p
                    for p in ports
                    if abs(p[0] - port_cells[0][0]) + abs(p[1] - port_cells[0][1]) >= MIN_PORT_DIST
                ]
            if not ports:
                continue
            chosen_port = ports[0]
            _place_patch(layout_grid, patch, r0, c0)
            positions.append((r0, c0))
            port_cells.append(chosen_port)
            placed = True
            break
        if not placed:
            return None
    return positions, port_cells


def _build_base_output(
    dest_specs: list[dict],
    dest_positions: list[tuple[int, int]],
    path_cells: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((GRID_H, GRID_W), BG, dtype=int)
    for spec, (r0, c0) in zip(dest_specs, dest_positions, strict=False):
        _place_patch(out, _output_dest_patch(spec["border"], spec["dot_color"], spec["count"], BG), r0, c0)
    for r, c in path_cells:
        out[r, c] = GREY
    return out


def _gen_attempt() -> np.ndarray | None:
    dest_specs, source_specs = _build_role_colors()
    target_len = random.randint(PATH_LEN_MIN, PATH_LEN_MAX)

    layout_grid = np.full((GRID_H, GRID_W), BG, dtype=int)
    placed = _place_dests(layout_grid, dest_specs)
    if placed is None:
        return None
    dest_positions, port_cells = placed

    path_cells = _gen_winding_path(layout_grid, port_cells[0], port_cells[1], target_len)
    if path_cells is None or not (PATH_LEN_MIN <= len(path_cells) <= PATH_LEN_MAX):
        return None
    if not _is_8_connected(path_cells):
        return None
    for r, c in path_cells:
        if layout_grid[r, c] != BG:
            return None
        layout_grid[r, c] = GREY

    for pr, pc in port_cells:
        deg = sum(
            1
            for nr, nc in ((pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1))
            if 0 <= nr < GRID_H and 0 <= nc < GRID_W and layout_grid[nr, nc] == GREY
        )
        if deg > 1:
            return None

    input_grid = layout_grid.copy()
    output_grid = _build_base_output(dest_specs, dest_positions, path_cells)

    all_pos = _sprite_positions()
    for border_color, inner_color in source_specs:
        patch = _frame_patch(border_color, inner_color)
        pool = list(all_pos)
        random.shuffle(pool)
        placed_source = False
        for r0, c0 in pool:
            if _overlaps(input_grid, patch, r0, c0):
                continue
            _place_patch(input_grid, patch, r0, c0)
            cr, cc = r0 + FRAME_SIZE // 2, c0 + FRAME_SIZE // 2
            if output_grid[cr, cc] != BG:
                return None
            output_grid[cr, cc] = inner_color
            placed_source = True
            break
        if not placed_source:
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
