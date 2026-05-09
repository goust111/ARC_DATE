# Combo variant of 13e47133 (index 15)
# combo_spec: D8/3+D14/0
# Name: 面积决定框架宽度+严格矩形区域

import os
import sys
import random

import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../"))
_SOLVER_DIR = os.path.join(_ROOT, "solvers")
if _SOLVER_DIR not in sys.path:
    sys.path.append(_SOLVER_DIR)

from arc_utils import components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    - Original task: a wall color splits the board into free regions, marker
#      depths define a palette, and depth `d` is colored by `palette[d % m]`.
#      For one depth-1 marker this yields an endless alternation base/marker/
#      base/marker as the region goes inward.
#    - This combo keeps the marker-order reading step, but only allows strict
#      rectangles and removes periodic repetition. Example:
#      5x5 rectangle, base=B, one marker R at depth 1 -> seed=[B,R], area=25,
#      width=1, max_depth=2, target len=3, effective palette=[B,R,B]. Depths
#      0|1|2 become B|R|B.
#      12x12 rectangle with the same seed -> area=144, width=2, max_depth=5,
#      target len=3, effective palette=[B,R,B]. Depths 0,1|2,3|4,5 become
#      B|R|B. The same visible marker therefore yields broader frames in the
#      larger box.
# b) Complexity calibration:
#    - Regions use one marker each at the same depth. That removes extra degrees
#      of freedom, so the main thing to infer is the area-to-width relation.
#    - Width is a small stepwise-linear function of area, then clipped by the
#      rectangle depth so every chosen width is still visible in the output.
#      This is easier to infer than arbitrary per-size behavior but still
#      visibly different from the original cyclic rule.
# c) Broken old assumptions:
#    - `palette[d % m]` is no longer valid; every palette entry now owns one
#      contiguous depth segment.
#    - Region shape can no longer vary. The solver may exploit the fact that a
#      valid region is a filled rectangle, so Chebyshev depth is just the
#      minimum distance to the four sides.
#    - Marker count no longer fixes the final number of rings: area expands or
#      shrinks the effective palette length through band width.
# d) Placement legality:
#    - The generator only performs full-row or full-column recursive splits, so
#      every non-wall component stays rectangular and disjoint.
#    - Minimum region side length is 5, so a depth-1 marker is always strictly
#      inside its rectangle and every region has max depth at least 2.
# e) Visual observability:
#    - Each sample contains at least two observable band-count groups, with at
#      least two rectangles in both the narrow-band and wide-band groups.
#    - At least one marker color is reused across the smallest and largest
#      groups, so a human can compare same-color regions directly and see that
#      larger rectangles produce broader frames.
# f) Visual cleanliness:
#    - One marker per region keeps the inputs sparse. Because regions are perfect
#      rectangles, the outputs are clean nested boxes instead of mixed L/T cases.
# g) Color inference:
#    - Colors are never remapped numerically. The marker color order is read
#      directly from geometry, then repeated only enough to populate the number
#      of contiguous bands chosen from region area.
# h) New solver assumptions and generator guarantees:
#    - Solver assumes the wall color is the second most common color and that
#      generated regions are rectangles; the generator validates the color order
#      and constructs only rectangular regions.
#    - Solver also has a fallback boundary-distance routine for off-spec shapes,
#      so it remains safe if a component is unexpectedly non-rectangular.


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

AREA_WIDTH_STEP = 80


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
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


def _sorted_pts(pts) -> np.ndarray:
    return np.asarray(sorted((int(r), int(c)) for r, c in pts), dtype=int)


def _component_mask(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[pts[:, 0], pts[:, 1]] = True
    return mask


def _rectangle_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_distance(shape: tuple[int, int], pts: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    dist = np.full(shape, -1, dtype=int)
    rr = pts[:, 0]
    cc = pts[:, 1]
    dist[rr, cc] = np.minimum.reduce([rr - r0, r1 - rr, cc - c0, c1 - cc]).astype(int)
    return dist


def _component_distance(shape: tuple[int, int], pts: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    bbox = _rectangle_bbox(pts)
    area = pts.shape[0]
    rect_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    if area == rect_area:
        return _rectangle_distance(shape, pts, bbox), bbox, area
    mask = _component_mask(shape, pts)
    return _boundary_distance(mask), bbox, area


def _build_seed_palette(g: np.ndarray, pts: np.ndarray, base: int, dist: np.ndarray) -> list[int]:
    dist_to_color: dict[int, int] = {}
    for r, c in pts:
        val = int(g[r, c])
        if val == base:
            continue
        d = int(dist[r, c])
        if d < 0:
            continue
        if d not in dist_to_color:
            dist_to_color[d] = val

    if not dist_to_color:
        return [int(base)]

    palette: list[int] = []
    ordered_depths = sorted(dist_to_color)
    if ordered_depths[0] > 0:
        palette.append(int(base))
    for d in ordered_depths:
        palette.append(int(dist_to_color[d]))
    return palette


def _frame_width_from_area(area: int, max_depth: int) -> int:
    if max_depth <= 0:
        return 1
    raw = 1 + max(0, int(area) - 1) // AREA_WIDTH_STEP
    return max(1, min(raw, int(max_depth)))


def _target_palette_len(area: int, max_depth: int, seed_len: int) -> int:
    width = _frame_width_from_area(area, max_depth)
    target = 1 + (max_depth // width)
    target = max(int(seed_len), int(target))
    return min(target, max_depth + 1)


def _expand_palette(seed: list[int], target_len: int) -> list[int]:
    return [int(seed[i % len(seed)]) for i in range(target_len)]


def _segment_index(depth: int, max_depth: int, n_segments: int) -> int:
    if max_depth <= 0 or n_segments <= 1:
        return 0
    return min(n_segments - 1, (((depth + 1) * n_segments) - 1) // (max_depth + 1))


def _solve(grid: np.ndarray, *, cyclic: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall

    for comp in _cc4(free):
        pts = _sorted_pts(comp)
        dist, _bbox, area = _component_distance((h, w), pts)
        mask = _component_mask((h, w), pts)
        base = _mode(g[mask])
        seed_palette = _build_seed_palette(g, pts, base, dist)
        max_depth = int(dist[mask].max())

        if cyclic:
            palette = seed_palette
        else:
            target_len = _target_palette_len(area, max_depth, len(seed_palette))
            palette = _expand_palette(seed_palette, target_len)

        for r, c in pts:
            d = int(dist[r, c])
            if cyclic:
                out[r, c] = palette[d % len(palette)]
            else:
                band = _segment_index(d, max_depth, len(palette))
                out[r, c] = palette[band]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-separated rectangles with area-scaled concentric color bands.

    Input structure:
    - One wall color partitions the grid into disjoint free regions, and every
      valid free region is a filled axis-aligned rectangle.
    - Each rectangle has one dominant base color and sparse marker pixels whose
      Chebyshev depths from the rectangle boundary encode the seed color order.

    Rule:
    - For each rectangle, sort marker colors by boundary depth; if the shallowest
      marker is not on the boundary, prepend the base color to the seed palette.
    - Convert rectangle area into a frame width, use that width to determine how
      many contiguous depth bands the rectangle should have, repeat the seed
      palette only enough to fill those bands, then paint one band per color.

    Key insight: Marker depth still fixes color order, but strict rectangles make
    depth layers perfectly regular, and region area now controls band thickness
    instead of letting the palette repeat forever.
    """
    return _solve(grid, cyclic=False)


# ---- Canvas bounds ----
H_MIN, H_MAX = 18, 30
W_MIN, W_MAX = 18, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 4
MIN_REGION_DIM = 5
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 4, 6

# ---- Marker setup ----
MARKER_DEPTH = 1
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 5

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 600

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(height: int, width: int) -> int:
    return (min(height, width) - 1) // 2


def _rect_area(rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _rect_dims(rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _place_walls(height: int, width: int, rng: random.Random):
    wall_mask = np.zeros((height, width), dtype=int)
    rects = [(0, 0, height - 1, width - 1)]
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

        idx, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(idx)
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


def _band_groups(rects) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, rect in enumerate(rects):
        max_depth = _max_cheb_depth(*_rect_dims(rect))
        bands = _target_palette_len(_rect_area(rect), max_depth, seed_len=2)
        groups.setdefault(bands, []).append(idx)
    return groups


def _validate_rects(rects) -> bool:
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return False
    if not all(_max_cheb_depth(*_rect_dims(rect)) >= 2 for rect in rects):
        return False
    groups = _band_groups(rects)
    if len(groups) < 2:
        return False
    min_group = min(groups)
    max_group = max(groups)
    if min_group >= max_group:
        return False
    return len(groups[min_group]) >= 2 and len(groups[max_group]) >= 2


def _assign_marker_colors(rects, marker_pool: list[int], rng: random.Random):
    groups = _band_groups(rects)
    band_counts = sorted(groups)
    if len(band_counts) < 2:
        return None

    active_n = min(len(marker_pool), rng.randint(DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX))
    if active_n < DISTINCT_MARKER_COLORS_MIN:
        return None
    active = rng.sample(marker_pool, active_n)

    colors = [None] * len(rects)
    bridge = active[0]
    colors[rng.choice(groups[band_counts[0]])] = bridge
    colors[rng.choice(groups[band_counts[-1]])] = bridge

    remaining = [i for i, val in enumerate(colors) if val is None]
    rng.shuffle(remaining)
    cycle = active[1:] if len(active) > 1 else active
    for j, idx in enumerate(remaining):
        colors[idx] = cycle[j % len(cycle)]

    if len(set(colors)) < DISTINCT_MARKER_COLORS_MIN:
        return None
    return colors


def _place_marker(rect, color: int, rng: random.Random):
    r0, c0, r1, c1 = rect
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    rr = br + dr * MARKER_DEPTH
    cc = bc + dc * MARKER_DEPTH
    return rr, cc, color


def _stamp_markers(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    colors = _assign_marker_colors(rects, marker_pool, rng)
    if colors is None:
        return None

    used = set()
    for rect, color in zip(rects, colors):
        rr, cc, col = _place_marker(rect, color, rng)
        if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
            return None
        if grid[rr, cc] != bg:
            return None
        grid[rr, cc] = col
        used.add(col)

    groups = _band_groups(rects)
    min_bands = min(groups)
    max_bands = max(groups)
    min_colors = {colors[i] for i in groups[min_bands]}
    max_colors = {colors[i] for i in groups[max_bands]}
    if not (min_colors & max_colors):
        return None
    return used


def _validate_color_order(grid, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _gen_attempt(rng: random.Random):
    height = rng.randint(H_MIN, H_MAX)
    width = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)
    wall_mask, rects = _place_walls(height, width, rng)
    if not _validate_rects(rects):
        return None

    grid = np.full((height, width), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, _solve(grid, cyclic=True)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
