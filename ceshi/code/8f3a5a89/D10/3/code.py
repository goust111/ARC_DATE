# Variant of 8f3a5a89
# Dimension: D10 — 多步处理/迭代逻辑 (multi_step)
# Change: 两步洪填——第一次洪填（仅穿过背景和标记）确定reach0，用于识别结构墙；第二次洪填（还可穿过装饰墙）确定最终可达区域... → 链式传播：边界涂色触发邻近区域

from __future__ import annotations

from collections import Counter, deque
from itertools import combinations
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


# -----------------------------------------------------------------------------
# Design analysis for the D10 variant
# -----------------------------------------------------------------------------
# concretization_and_trace:
#   input:
#     - Keep the original palette base: bg=8, wall=1, marker=6.
#     - Blue border-touching components are structural chamber walls.
#     - Blue interior fragments are decorations; they are traversable and stay blue
#       only inside activated chambers.
#     - The generator arranges the structural walls as a horizontal or vertical
#       chain of chambers. The marker stays at bottom-left, so the first chamber is
#       the leftmost chamber (vertical layout) or the bottom chamber (horizontal).
#   output:
#     - Chamber 0 (containing the marker) gets the original style of 1-cell inner
#       border, colored orange.
#     - Any other chamber that shares a structural wall with an already activated
#       chamber is triggered next and gets its own inner border in the next color.
#     - Depth-1 chambers are red, depth-2 chambers are green, depth-3 chambers are
#       yellow. The generator keeps the chain short enough that these colors remain
#       visually learnable from a few examples.
#   trace_example:
#     - Original task on three strip chambers separated by two full-height walls:
#       the first flood sees only chamber 0, so only the first separator is
#       considered structural. The second separator is misread as decoration, so
#       chambers 1 and 2 merge into one large reachable area.
#     - Variant task on the same input:
#       1. identify all border-touching blue components as structural walls;
#       2. split passable space into chamber components;
#       3. start from the marker chamber and paint its inner border orange;
#       4. across the shared wall, activate the next chamber and paint it red;
#       5. continue to the next chamber and paint it green.
#     - Difference: the output now depends on iterative region-to-region
#       propagation, not just a single reachable side of the first wall.
# complexity_calibration:
#   - The chamber graph is intentionally small (2-4 chambers) so the propagation
#     depth is visible, but the solver still must recover a multi-step BFS over
#     regions rather than a single flood-fill.
#   - Colors are limited to a fixed ordered sequence [7,2,3,4], avoiding an
#     underconstrained arbitrary mapping.
# implicit_assumptions_audit:
#   - Original assumption that "structural walls are adjacent to reach0" no longer
#     holds: deeper separators in the chain are not adjacent to the marker flood.
#     The solver therefore redefines structural walls as all wall-color components
#     touching the outer border, while the generator guarantees decorations never
#     touch the border.
#   - Original second flood through non-structural wall pixels remains valid, but
#     it must now be done for every chamber component, not only the marker side.
#   - Painting a single fill color would hide propagation order, so the solver uses
#     depth-indexed colors and the generator ensures chain depth >= 1.
# placement_legality:
#   - Structural separators are full-span lines, so chambers are disjoint, inside
#     bounds, and easy to re-identify.
#   - Decorations are placed strictly away from the outer border and away from
#     structural walls, preventing accidental reinterpretation as structural walls.
# visual_observability:
#   - The chamber strips make the chain relation obvious: each newly colored ring
#     appears immediately across a shared blue wall.
#   - Orientation varies (horizontal or vertical), chamber sizes vary, and chain
#     length varies, so the rule is learnable but not tied to one rigid template.
# visual_cleanliness:
#   - Only three object types overlap visually: blue walls, a single magenta
#     marker, and sparse blue interior decorations. Fills stay on background cells.
# color_inference:
#   - The ordered progression orange -> red -> green -> yellow is tied to graph
#     distance from the marker chamber, not to any hidden numeric property.
# new_main_assumptions_and_generator_guarantees:
#   - There is exactly one unique non-background corner pixel (the marker) and it
#     is at bottom-left.
#   - Every true structural wall touches the outer border; every decoration does
#     not. The generator enforces this directly.
#   - The structural-wall adjacency graph is connected as a chain, guaranteeing at
#     least two activated chambers and therefore visible D10 behavior.


BG = 8
WALL = 1
MARKER = 6
CHAIN_COLORS = (7, 2, 3, 4)

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)
MIN_REGION_SPAN = 4
MAX_REGIONS = 4
NUM_DECOR_RANGE = (0, 3)
DECOR_AREA_RANGE = (1, 3)
GEN_ATTEMPTS = 400


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _identify_marker_and_wall(g: np.ndarray) -> tuple[int, tuple[int, int, int], int]:
    h, w = g.shape
    bg = get_bg(g)
    counts = Counter(int(v) for v in g.ravel())
    marker = None
    for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    other_cols = [col for col in counts if col not in (bg, marker[2])]
    assert other_cols, "no wall color found"
    wall_col = max(other_cols, key=lambda col: counts[col])
    return int(bg), marker, int(wall_col)


def _solve_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, marker, wall_col = _identify_marker_and_wall(g)
    mr, mc, marker_col = marker

    structural = np.zeros((h, w), dtype=bool)
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = [(int(r), int(c)) for r, c in comp_arr]
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            for r, c in comp:
                structural[r, c] = True

    passable = ~structural
    region_id = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []
    for rid, comp_arr in enumerate(components_mask(passable, connectivity=4)):
        coords = [(int(r), int(c)) for r, c in comp_arr]
        for r, c in coords:
            region_id[r, c] = int(rid)
        regions.append(coords)

    start_rid = int(region_id[mr, mc])
    assert start_rid >= 0, "marker cannot lie on a structural wall"

    adj = [set() for _ in regions]
    wall_touch_regions: list[set[int]] = []
    for comp_arr in components_mask(structural, connectivity=4):
        touched: set[int] = set()
        for r, c in comp_arr:
            rr = int(r)
            cc = int(c)
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and region_id[nr, nc] >= 0:
                    touched.add(int(region_id[nr, nc]))
        wall_touch_regions.append(touched)
        for a, b in combinations(sorted(touched), 2):
            adj[a].add(b)
            adj[b].add(a)

    depth_by_region = {start_rid: 0}
    q = deque([start_rid])
    while q:
        rid = q.popleft()
        for nb in sorted(adj[rid]):
            if nb not in depth_by_region:
                depth_by_region[nb] = depth_by_region[rid] + 1
                q.append(nb)

    out = g.copy()
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == wall_col and not structural[r, c]:
                rid = int(region_id[r, c])
                if rid < 0 or rid not in depth_by_region:
                    out[r, c] = bg

    for rid, depth in depth_by_region.items():
        fill_col = CHAIN_COLORS[min(depth, len(CHAIN_COLORS) - 1)]
        for r, c in regions[rid]:
            if int(out[r, c]) != bg:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                out[r, c] = fill_col
                continue
            if any(0 <= nr < h and 0 <= nc < w and structural[nr, nc] for nr, nc in neighbors8(r, c)):
                out[r, c] = fill_col
                continue
            if any(not (0 <= nr < h and 0 <= nc < w) or int(region_id[nr, nc]) != rid for nr, nc in neighbors4(r, c)):
                out[r, c] = fill_col

    meta = {
        "bg": int(bg),
        "wall_col": int(wall_col),
        "marker_col": int(marker_col),
        "start_region": int(start_rid),
        "depth_by_region": {int(k): int(v) for k, v in depth_by_region.items()},
        "region_count": int(len(regions)),
        "activated_regions": int(len(depth_by_region)),
        "max_depth": int(max(depth_by_region.values())),
        "structural_components": int(len(components_mask(structural, connectivity=4))),
        "wall_touch_regions": [sorted(int(x) for x in touched) for touched in wall_touch_regions],
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint chamber borders in a chain starting from the corner marker region.

    Input structure:
    - The background is the dominant color; blue pixels form border-touching walls
      plus optional interior blue decoration fragments.
    - A unique magenta marker sits on one corner and marks the first chamber.
    - Structural walls split the background into multiple enclosed chambers that
      may share a wall with another chamber.

    Rule:
    - Treat every border-touching blue component as a real wall; interior blue
      fragments are traversable decoration.
    - Split the non-structural space into chamber components, then start from the
      chamber containing the corner marker.
    - Paint a 1-cell inner border for that chamber, then propagate across shared
      structural walls: depth-1 chambers get the next color, depth-2 chambers the
      next one, and so on.

    Key insight: The task is no longer a single reachable-side fill; the painted
    boundary of one chamber activates neighboring chambers through the wall graph.
    """
    out, _meta = _solve_details(grid)
    return out


def _random_partition(total: int, parts: int, min_size: int, rng: random.Random) -> list[int]:
    sizes = [min_size] * parts
    extra = total - parts * min_size
    for _ in range(extra):
        sizes[rng.randrange(parts)] += 1
    return sizes


def _place_chain_walls(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, str, int]:
    feasible: list[tuple[str, list[int]]] = []
    for orientation, span in (("vertical", w), ("horizontal", h)):
        counts = [n for n in range(2, MAX_REGIONS + 1) if n * MIN_REGION_SPAN + (n - 1) <= span]
        if counts:
            feasible.append((orientation, counts))
    orientation, counts = rng.choice(feasible)
    n_regions = rng.choice(counts)

    structural = np.zeros((h, w), dtype=bool)
    if orientation == "vertical":
        widths = _random_partition(w - (n_regions - 1), n_regions, MIN_REGION_SPAN, rng)
        cursor = 0
        for idx, width in enumerate(widths):
            cursor += width
            if idx < n_regions - 1:
                structural[:, cursor] = True
                cursor += 1
    else:
        heights = _random_partition(h - (n_regions - 1), n_regions, MIN_REGION_SPAN, rng)
        cursor = h
        for idx, height in enumerate(heights):
            cursor -= height
            if idx < n_regions - 1:
                cursor -= 1
                structural[cursor, :] = True

    return structural, orientation, int(n_regions)


def _place_decor(allowed_mask: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed

    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 3:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(canvas, patch, transparent=0, border=0, padding=1, padding_connectivity=8)
            except ValueError:
                continue
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    structural, _orientation, n_regions = _place_chain_walls(h, w, rng)
    if structural[marker_rc]:
        return None

    inp = np.full((h, w), BG, dtype=int)
    inp[structural] = WALL
    inp[marker_rc] = MARKER

    safe_for_decor = ~structural
    safe_for_decor[marker_rc] = False
    safe_for_decor[0, :] = False
    safe_for_decor[-1, :] = False
    safe_for_decor[:, 0] = False
    safe_for_decor[:, -1] = False
    safe_for_decor[_dilate8(structural)] = False

    decor = _place_decor(safe_for_decor, rng.randint(*NUM_DECOR_RANGE), rng)
    if decor is None:
        return None
    inp[decor] = WALL

    out, meta = _solve_details(inp)
    if meta["activated_regions"] != n_regions:
        return None
    if meta["max_depth"] < 1:
        return None

    used_fill_colors = {int(v) for v in out.ravel() if int(v) in CHAIN_COLORS}
    if len(used_fill_colors) < 2:
        return None

    # Ensure the marker chamber does not absorb the whole board visually.
    start_depth = meta["depth_by_region"][meta["start_region"]]
    if start_depth != 0:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 D10 variant")
