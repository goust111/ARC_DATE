# Combo variant of 13e47133 (index 17)
# combo_spec: D2/1+D8/1
# Name: 渐变非循环+调色板反转

from __future__ import annotations

import os
import random
import sys

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../../"))
_SOLVER_DIR = os.path.join(_ROOT, "solvers")
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import components_mask


# design_analysis:
#   a) Concrete behavior trace:
#      - The visible objects stay the same as in the source task: one wall color
#        splits the board into 3-6 rectangular free regions; each region is mostly
#        base-colored with 0-3 isolated marker pixels on consecutive diagonal
#        depths from one corner.
#      - Original rule for a region with boundary-depth markers [(0,A),(1,B),(2,C)]
#        builds palette [A,B,C] and paints depths as A,B,C,A,B,C,...
#      - The fused rule first builds the same local palette, then reverses it to
#        [C,B,A], then stops cycling: depths become C,B,A,A,A,...
#      - If the shallowest marker starts at depth 1, original palette [base,A,B]
#        becomes reversed [B,A,base], so the deepest marker moves to the outer
#        ring and the region center eventually freezes to base instead of looping.
#   b) Complexity calibration:
#      - The rule still uses only local geometry and short palettes, so the task
#        does not gain extra object types or hidden global bookkeeping.
#      - To make both dimensions visible, the generator forces one focus region
#        with max depth >= 3 and markers at depths 0,1,2. That single region
#        simultaneously exposes outer-ring reversal and inner non-cycling.
#      - A second anchored region is also forced to start at depth 1 when
#        possible, so the "prepend base then reverse" branch is visible in the
#        training distribution instead of being left to chance.
#   c) Hidden-assumption review for main():
#      - `dist_to_color` still assumes at most one meaningful marker per distance;
#        generator keeps one diagonal marker per depth, so this remains valid.
#      - `base = mode(region)` still assumes markers are sparse minority pixels;
#        generator caps markers at 3 per region and keeps regions much larger.
#      - The old `palette[d % m]` periodicity assumption is invalid here. It is
#        replaced with two coordinated steps: reverse the palette, then clamp the
#        index with `min(d, m - 1)`.
#      - Reversing after optional base-prepending is intentional: the whole usable
#        palette order flips, including the inserted base if it exists.
#   d) Placement legality:
#      - Recursive full-row/full-column splits keep free components rectangular and
#        disjoint, so diagonal depth placements stay in bounds.
#      - The focus region is chosen only from rectangles with Chebyshev depth >= 3,
#        making depths 0,1,2 legal and guaranteeing at least one deeper cell.
#   e) Visual observability:
#      - The transformation depends only on visible marker depth and visible ring
#        position, never on numeric color ids.
#      - Samples are rejected unless the fused output differs from the original
#        task, the D2-only variant, and the D8-only variant. This prevents silent
#        collapse where one control dimension has no visible effect.
#   f) Visual cleanliness:
#      - Inputs contain only walls, large base-colored regions, and isolated single
#        pixel markers. Outputs remain clean concentric frames per region.
#   g) Color-rule inferability:
#      - Each palette is short and structurally anchored by the marker depths.
#        Humans can compare boundary ring colors against deeper markers and see
#        that order is reversed, then compare inner depths and see that colors stop
#        advancing after the last palette entry instead of wrapping.
#   h) New solver assumptions and generator guarantees:
#      - Solver assumes a wall color can still be recovered as the second-most
#        common color; generator validates that frequency ordering before return.
#      - Solver assumes every non-base cell inside a region is a marker; generator
#        never places extra noise objects, so that interpretation is safe.


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neigh (Chebyshev) distance to boundary for True-cells in component_mask.
    Boundary cells are those that touch False or the grid edge in their 8-neighborhood.
    Returns dist array with -1 outside component.
    """
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
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    is_boundary = True
                    break
                if not component_mask[rr, cc]:
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


def _build_palette(g: np.ndarray, pts, base: int, dist: np.ndarray, *, reverse_palette: bool) -> list[int]:
    dist_to_color: dict[int, int] = {}
    for r, c in pts:
        v = int(g[r, c])
        if v == base:
            continue
        d = int(dist[r, c])
        if d < 0:
            continue
        if d not in dist_to_color:
            dist_to_color[d] = v

    if not dist_to_color:
        palette = [int(base)]
    else:
        ds = sorted(dist_to_color)
        palette = []
        if ds[0] > 0:
            palette.append(int(base))
        for d in ds:
            palette.append(int(dist_to_color[d]))

    if reverse_palette:
        palette = list(reversed(palette))
    return palette


def _solve(grid: np.ndarray, *, reverse_palette: bool, cyclic: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)
        palette = _build_palette(g, pts, base, dist, reverse_palette=reverse_palette)

        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            idx = d % m if cyclic else min(d, m - 1)
            out[r, c] = palette[idx]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each wall-separated region using a reversed depth palette that progresses inward only once.

    Input structure:
    - A single wall color partitions the board into several non-wall regions.
    - Inside each region, one dominant base color is interrupted by a few isolated marker pixels placed at increasing Chebyshev depths from a corner.

    Rule:
    - For each region, collect marker colors in ascending boundary distance order; if the first marker is not on the boundary, prepend the region base just as in the source task.
    - Reverse that palette so the deepest marker color becomes the outermost band.
    - Fill the region by boundary distance without cycling: layer 0 uses the first reversed color, layer 1 the next, and every deeper layer after the final palette entry stays on that final color.

    Key insight: Marker depth still defines a local palette, but the mapping is flipped and the trajectory is capped, so the deepest marker color moves outward while the innermost available color absorbs the whole remaining core.
    """
    return _solve(grid, reverse_palette=True, cyclic=False)


def _main_original(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, reverse_palette=False, cyclic=True)


def _main_d2_only(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, reverse_palette=False, cyclic=False)


def _main_d8_only(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, reverse_palette=True, cyclic=True)


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
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Variant visibility ----
FOCUS_MIN_DEPTH = 3
BRANCH_MIN_DEPTH = 2

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 500

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


def _marker_coords(rect, start_d: int, n: int, rng: random.Random):
    r0, c0, r1, c1 = rect
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    return [(start_d + k, br + dr * (start_d + k), bc + dc * (start_d + k)) for k in range(n)]


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper <= 0:
        return []

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return []

    max_start = min(START_DIST_MAX, max_d - (n - 1))
    if max_start < 0:
        return []

    start_d = rng.randint(0, max_start)
    coords = _marker_coords(rect, start_d, n, rng)
    colors = rng.sample(marker_pool, n)
    return [(d, col, rr, cc) for (d, rr, cc), col in zip(coords, colors)]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _place_focus_markers(rect, marker_pool: list[int], rng: random.Random):
    coords = _marker_coords(rect, start_d=0, n=3, rng=rng)
    colors = rng.sample(marker_pool, 3)
    return [(d, col, rr, cc) for (d, rr, cc), col in zip(coords, colors)]


def _place_branch_markers(rect, marker_pool: list[int], rng: random.Random):
    coords = _marker_coords(rect, start_d=1, n=2, rng=rng)
    colors = rng.sample(marker_pool, 2)
    return [(d, col, rr, cc) for (d, rr, cc), col in zip(coords, colors)]


def _stamp_markers(
    grid,
    rects,
    bg,
    marker_pool,
    focus_index: int,
    branch_index: int,
    rng: random.Random,
):
    used = set()
    focus_rect = rects[focus_index]

    for _d, col, rr, cc in _place_focus_markers(focus_rect, marker_pool, rng):
        if not (focus_rect[0] <= rr <= focus_rect[2] and focus_rect[1] <= cc <= focus_rect[3]):
            return None
        if grid[rr, cc] != bg:
            return None
        grid[rr, cc] = col
        used.add(col)

    branch_rect = rects[branch_index]
    for _d, col, rr, cc in _place_branch_markers(branch_rect, marker_pool, rng):
        if not (branch_rect[0] <= rr <= branch_rect[2] and branch_rect[1] <= cc <= branch_rect[3]):
            return None
        if grid[rr, cc] != bg:
            return None
        grid[rr, cc] = col
        used.add(col)

    for i, rect in enumerate(rects):
        if i in (focus_index, branch_index):
            continue
        for _d, col, rr, cc in _place_markers(rect, marker_pool, rng):
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(col)

    return used


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _choose_focus_region(rects, rng: random.Random):
    candidates = []
    for i, rect in enumerate(rects):
        depth = _max_cheb_depth(rect[2] - rect[0] + 1, rect[3] - rect[1] + 1)
        if depth >= FOCUS_MIN_DEPTH:
            candidates.append(i)
    if not candidates:
        return None
    return rng.choice(candidates)


def _choose_branch_region(rects, focus_index: int, rng: random.Random):
    candidates = []
    for i, rect in enumerate(rects):
        if i == focus_index:
            continue
        depth = _max_cheb_depth(rect[2] - rect[0] + 1, rect[3] - rect[1] + 1)
        if depth >= BRANCH_MIN_DEPTH:
            candidates.append(i)
    if not candidates:
        return None
    return rng.choice(candidates)


def _fusion_visible(inp: np.ndarray) -> bool:
    combo = main(inp)
    return (
        not np.array_equal(combo, _main_original(inp))
        and not np.array_equal(combo, _main_d2_only(inp))
        and not np.array_equal(combo, _main_d8_only(inp))
    )


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    focus_index = _choose_focus_region(rects, rng)
    if focus_index is None:
        return None
    branch_index = _choose_branch_region(rects, focus_index, rng)
    if branch_index is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used = _stamp_markers(grid, rects, bg, marker_pool, focus_index, branch_index, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None
    if not _fusion_visible(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
