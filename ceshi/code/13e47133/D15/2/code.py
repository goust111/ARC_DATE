# Variant of 13e47133
# Dimension: D15 — 符号指令：调色板构建的特殊规则 (symbol_instructions)
# Change: 无显式符号编码；标记像素的距离隐式决定调色板顺序，距离 > 0 时背景色自动前置... → 墙壁上的标记色点指示对应区域

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for the D15 variant
# a) Concrete behavior:
#    The old task stored the palette start implicitly: the shallowest internal marker
#    was always the first visible band, with the base color auto-prepended when the
#    first marker began at distance 1. The variant keeps those internal markers, but
#    also recolors one wall cell near a region with one of that region's marker colors.
#    Example: if a region has markers [red at d=0, green at d=1], the old palette is
#    [red, green]; if a neighboring wall cell is green, the new palette becomes
#    [green, red]. If the markers are [yellow at d=1] then the old palette is
#    [base, yellow], and a yellow wall dot rotates it to [yellow, base].
# b) Complexity calibration:
#    The colored wall dot is only a rotation signal, not a second independent palette.
#    This keeps the main structure recognizable while producing visibly different
#    outer rings. Generator-side constraints require at least one region whose wall
#    symbol changes the starting band, so the variant does not collapse back to the
#    original behavior.
# c) Hidden-assumption audit against the original main():
#    - "wall is a single color" no longer holds, because some wall cells are recolored.
#    - "free = g != wall" no longer holds, because clue-colored wall cells would be
#      mistaken for region pixels. The new solver reconstructs the wall structurally:
#      any non-base cell with an orthogonal non-base neighbor belongs to the wall.
#    - "region-adjacent colored cells are always internal markers" no longer holds.
#      The solver now separates internal markers from wall clues using that wall mask.
#    - "palette order is fully determined after sorting markers by depth" no longer
#      holds. The depth order still gives the cyclic list, but the wall clue rotates
#      that list before filling.
# d) Placement legality:
#    Regions are still rectangles carved by one-cell-thick wall lines, so component
#    detection and Chebyshev distances remain well-defined. Internal markers stay
#    isolated inside their region. Wall clues are written only onto existing wall
#    cells, one per marked region, and only on straight wall segments to keep the
#    symbol placement legible.
# e) Visual observability:
#    Humans can match a wall dot's color to a same-colored internal marker in one of
#    the adjacent regions. Across examples, the outermost ring of that region starts
#    with the wall-dot color, even when the shallowest internal marker would have
#    suggested another start under the old rule.
# f) Visual cleanliness:
#    The grid uses only three element types: base-colored region cells, one-pixel
#    internal markers, and one-pixel wall symbols on otherwise uniform walls. No
#    object overlaps another object's footprint.
# g) Color inferability:
#    The wall symbol never invents a new color. It repeats a color already present as
#    an internal region marker, so the clue is local and structurally grounded rather
#    than an arbitrary recoloring table.
# Assumption audit for the new main():
#    - Every true wall/clue cell has at least one orthogonal non-base neighbor.
#      Generator guarantee: walls are line segments; clue dots overwrite wall cells.
#    - Every internal marker is orthogonally isolated from other non-base cells.
#      Generator guarantee: markers lie on a corner diagonal inside a rectangle.
#    - A clue wall cell is adjacent to the intended region, and no other region on the
#      same wall side uses that clue color in its own marker set.
#      Generator guarantee: candidate clue cells are filtered against neighboring
#      regions before stamping.


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance to the component boundary; -1 outside the component."""
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _wall_mask_from_structure(g: np.ndarray, base: int) -> np.ndarray:
    """
    Recover the separator walls after some wall pixels have been recolored as clues.
    Internal markers may touch a wall on one side, so simple non-base adjacency is
    not enough. A true wall cell lies on a horizontal/vertical non-base chain, or is
    an endpoint of such a chain.
    """
    h, w = g.shape
    non_base = g != base
    wall_mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not non_base[r, c]:
                continue
            dirs = []
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and non_base[rr, cc]:
                    dirs.append((dr, dc))

            if len(dirs) >= 3:
                wall_mask[r, c] = True
                continue

            if len(dirs) == 2:
                (dr1, dc1), (dr2, dc2) = dirs
                if dr1 == -dr2 and dc1 == -dc2:
                    wall_mask[r, c] = True
                continue

            if len(dirs) == 1:
                dr, dc = dirs[0]
                rr, cc = r + dr, c + dc
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and non_base[nr, nc]:
                    wall_mask[r, c] = True
    return wall_mask


def _build_region_infos(g: np.ndarray):
    h, w = g.shape
    bg = _mode(g)
    wall_mask = _wall_mask_from_structure(g, bg)
    wall_color = bg if not wall_mask.any() else _mode(g[wall_mask])

    comp_index = np.full((h, w), -1, dtype=int)
    infos = []
    for idx, pts in enumerate(_cc4(~wall_mask)):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True
            comp_index[int(r), int(c)] = idx

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        marker_colors: set[int] = set()
        for r, c in pts:
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == base:
                continue
            marker_colors.add(v)
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        palette: list[int]
        if dist_to_color:
            palette = []
            ds = sorted(dist_to_color)
            if ds[0] > 0:
                palette.append(base)
            for d in ds:
                palette.append(dist_to_color[d])
        else:
            palette = [base]

        infos.append(
            {
                "pts": pts,
                "dist": dist,
                "palette": palette,
                "marker_colors": marker_colors,
            }
        )

    return bg, wall_mask, wall_color, comp_index, infos


def _clue_start_colors(
    g: np.ndarray,
    wall_mask: np.ndarray,
    wall_color: int,
    comp_index: np.ndarray,
    infos: list[dict],
) -> dict[int, int]:
    starts: dict[int, int] = {}
    for r, c in np.argwhere(wall_mask):
        r = int(r)
        c = int(c)
        clue = int(g[r, c])
        if clue == wall_color:
            continue
        neighbors: list[int] = []
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                idx = int(comp_index[rr, cc])
                if idx >= 0 and idx not in neighbors:
                    neighbors.append(idx)
        matches = [idx for idx in neighbors if clue in infos[idx]["marker_colors"]]
        if len(matches) == 1 and matches[0] not in starts:
            starts[matches[0]] = clue
    return starts


def _rotate_palette(palette: list[int], start_color: int | None) -> list[int]:
    if start_color is None or start_color not in palette:
        return palette
    k = palette.index(start_color)
    return palette[k:] + palette[:k]


def _solve(grid: np.ndarray, use_clues: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _bg, wall_mask, wall_color, comp_index, infos = _build_region_infos(g)
    starts = _clue_start_colors(g, wall_mask, wall_color, comp_index, infos) if use_clues else {}

    for idx, info in enumerate(infos):
        palette = _rotate_palette(info["palette"], starts.get(idx))
        m = len(palette)
        for r, c in info["pts"]:
            r = int(r)
            c = int(c)
            d = int(info["dist"][r, c])
            out[r, c] = palette[d % m]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill walled regions with concentric color bands, using wall symbols to choose the palette start.

    Input structure:
    - Thin axis-aligned walls divide the board into separate regions of one dominant base color.
    - Inside each region, isolated colored pixels still provide the palette entries by their Chebyshev depth from the region boundary.
    - Some wall pixels are recolored as tiny symbols; each such wall dot repeats one marker color from a neighboring region.

    Rule:
    - For every region, sort its internal marker colors by boundary distance, and prepend the base color if the first marker begins deeper than distance 0.
    - If an adjacent wall symbol matches one of that region's marker colors, rotate that palette so the matching color becomes the first band color.
    - Paint the region with palette[distance mod palette_length], while leaving all wall pixels, including the colored symbols, unchanged.

    Key insight: The region markers tell you which colors belong in the cycle, but the colored wall dot tells you where that cycle should start.
    """
    return _solve(grid, use_clues=True)


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 4, 4]
START_DIST_MAX = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Variant visibility ----
CLUE_REGIONS_MIN = 2
VISIBLE_ROTATIONS_MIN = 1

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _place_walls(H: int, W: int, rng: random.Random):
    wall_mask = np.zeros((H, W), dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)

    for _ in range(n_splits):
        candidates = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            can_h = (r1 - r0 + 1) >= SPLITTABLE_LEN
            can_v = (c1 - c0 + 1) >= SPLITTABLE_LEN
            if can_h or can_v:
                candidates.append((i, can_h, can_v))
        if not candidates:
            break

        i, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(i)
        axes = ("h",) * can_h + ("v",) * can_v
        axis = rng.choice(axes)

        if axis == "h":
            rr = rng.randint(r0 + MIN_REGION_DIM, r1 - MIN_REGION_DIM)
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _rect_contains(rect, r: int, c: int) -> bool:
    return rect[0] <= r <= rect[2] and rect[1] <= c <= rect[3]


def _wall_neighbors(wall_mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    h, w = wall_mask.shape
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and wall_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _is_straight_wall_cell(wall_mask: np.ndarray, r: int, c: int) -> bool:
    nbrs = _wall_neighbors(wall_mask, r, c)
    if len(nbrs) == 1:
        return True
    if len(nbrs) != 2:
        return False
    (r1, c1), (r2, c2) = nbrs
    return r1 == r2 or c1 == c2


def _region_wall_candidates(rect, wall_mask: np.ndarray) -> list[tuple[int, int]]:
    H, W = wall_mask.shape
    r0, c0, r1, c1 = rect
    candidates: list[tuple[int, int]] = []

    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0 and wall_mask[r, c0 - 1]:
            candidates.append((r, c0 - 1))
        if c1 + 1 < W and wall_mask[r, c1 + 1]:
            candidates.append((r, c1 + 1))
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0 and wall_mask[r0 - 1, c]:
            candidates.append((r0 - 1, c))
        if r1 + 1 < H and wall_mask[r1 + 1, c]:
            candidates.append((r1 + 1, c))

    seen = set()
    filtered = []
    for cell in candidates:
        if cell in seen:
            continue
        seen.add(cell)
        if _is_straight_wall_cell(wall_mask, cell[0], cell[1]):
            filtered.append(cell)
    return filtered


def _adjacent_rect_ids(rects: list[tuple[int, int, int, int]], r: int, c: int) -> list[int]:
    ids: list[int] = []
    for idx, rect in enumerate(rects):
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if _rect_contains(rect, rr, cc):
                ids.append(idx)
                break
    return ids


def _plan_region(rect, marker_pool: list[int], start_color: int | None, rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d)
    if upper <= 0:
        return {"markers": [], "start_color": None, "visible_rotation": False, "marker_colors": set()}

    options = list(range(upper + 1))
    weights = MARKER_COUNT_WEIGHTS[: upper + 1]
    n = rng.choices(options, weights=weights)[0]
    if n == 0:
        return {"markers": [], "start_color": None, "visible_rotation": False, "marker_colors": set()}

    if start_color is None:
        return None

    start_d = rng.randint(0, min(START_DIST_MAX, max_d - (n - 1)))
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1

    colors = [None] * n
    if n == 1:
        clue_index = 0
    elif start_d > 0:
        clue_index = rng.randrange(n)
    else:
        clue_index = rng.randrange(1, n)
    colors[clue_index] = start_color

    others = rng.sample([col for col in marker_pool if col != start_color], n - 1)
    oi = 0
    for i in range(n):
        if colors[i] is None:
            colors[i] = others[oi]
            oi += 1

    markers = []
    for k, col in enumerate(colors):
        d = start_d + k
        rr = br + dr * d
        cc = bc + dc * d
        markers.append((d, int(col), rr, cc))

    first_palette_color = colors[0] if start_d == 0 else None
    visible_rotation = start_d > 0 or start_color != first_palette_color
    return {
        "markers": markers,
        "start_color": int(start_color),
        "visible_rotation": visible_rotation,
        "marker_colors": set(int(col) for col in colors),
    }


def _stamp_region_markers(grid: np.ndarray, rects, plans, bg: int):
    used = set()
    for rect, plan in zip(rects, plans):
        for _d, col, rr, cc in plan["markers"]:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(int(col))
    return used


def _assign_clue_cells(rects, wall_mask: np.ndarray, plans, rng: random.Random):
    marker_cells = {
        (int(rr), int(cc))
        for plan in plans
        for _d, _col, rr, cc in plan["markers"]
    }
    candidate_info = []
    for idx, (rect, plan) in enumerate(zip(rects, plans)):
        if not plan["markers"]:
            continue
        start_color = plan["start_color"]
        valid = []
        for cell in _region_wall_candidates(rect, wall_mask):
            if any((cell[0] + dr, cell[1] + dc) in marker_cells for dr, dc in DIR4):
                continue
            adj = _adjacent_rect_ids(rects, cell[0], cell[1])
            if idx not in adj:
                continue
            conflict = False
            for other in adj:
                if other != idx and start_color in plans[other]["marker_colors"]:
                    conflict = True
                    break
            if not conflict:
                valid.append(cell)
        rng.shuffle(valid)
        if not valid:
            return None
        candidate_info.append((len(valid), idx, valid))

    used = set()
    assigned = {}
    for _count, idx, valid in sorted(candidate_info):
        pick = next((cell for cell in valid if cell not in used), None)
        if pick is None:
            return None
        used.add(pick)
        assigned[idx] = pick
    return assigned


def _stamp_wall_clues(grid: np.ndarray, clue_cells: dict[int, tuple[int, int]], plans, wall: int):
    for idx, cell in clue_cells.items():
        r, c = cell
        if grid[r, c] != wall:
            return False
        grid[r, c] = plans[idx]["start_color"]
    return True


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)
    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    available_start_colors = marker_pool[:]
    rng.shuffle(available_start_colors)
    plans = []
    for rect in rects:
        start_color = available_start_colors.pop() if available_start_colors else None
        plan = _plan_region(rect, marker_pool, start_color, rng)
        if plan is None:
            return None
        plans.append(plan)

    clue_regions = sum(1 for plan in plans if plan["markers"])
    visible_rotations = sum(1 for plan in plans if plan["visible_rotation"])
    if clue_regions < CLUE_REGIONS_MIN or visible_rotations < VISIBLE_ROTATIONS_MIN:
        return None

    used = _stamp_region_markers(grid, rects, plans, bg)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None

    clue_cells = _assign_clue_cells(rects, wall_mask.astype(bool), plans, rng)
    if clue_cells is None:
        return None
    if not _stamp_wall_clues(grid, clue_cells, plans, wall):
        return None

    if not _validate_color_order(grid, bg, wall):
        return None

    _bg2, wall_mask2, wall_color2, comp_index2, infos2 = _build_region_infos(grid)
    starts2 = _clue_start_colors(grid, wall_mask2, wall_color2, comp_index2, infos2)
    if len(starts2) != clue_regions:
        return None
    if int(np.sum(wall_mask2 & (grid != wall_color2))) != clue_regions:
        return None
    actual_rotations = 0
    for idx, start in starts2.items():
        palette = infos2[idx]["palette"]
        if start in palette and palette[0] != start:
            actual_rotations += 1
    if actual_rotations < VISIBLE_ROTATIONS_MIN:
        return None

    try:
        baseline = _solve(grid, use_clues=False)
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    if np.array_equal(baseline, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
