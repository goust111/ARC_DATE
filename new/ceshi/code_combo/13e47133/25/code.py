# Combo variant of 13e47133 (index 25)
# combo_spec: D1/0+D8/1
# Name: 螺旋填充+反转调色板
#
# Design analysis
# ---------------
# concretization_and_trace:
#   original:
#     - Non-wall cells form rectangular regions created by full-row/full-column walls.
#     - Each region has 0..3 marker pixels on one inward diagonal from a corner.
#     - Marker colors are read by increasing Chebyshev depth; if depth 0 is absent,
#       base is prepended. The region is then colored by boundary-depth layers.
#   fused_variant:
#     - Keep the same wall-partitioned rectangular regions.
#     - Replace the fill index with a true rectangular spiral path, so color depends on
#       path position, not boundary layer.
#     - Keep a depth-based palette cue, but reverse it before filling:
#       shallow->deep markers [A, B, C] become spiral palette [C, B, A, base].
#     - Add a 2-cell same-color entrance seed at one corner edge. That seed uses the
#       deepest marker color C, so the solver can infer both spiral start/direction and
#       that the outer spiral begins with the reversed end of the depth list.
#   small example:
#     - Suppose a top-left clockwise region has diagonal markers:
#       depth 1 = red, depth 2 = green, depth 3 = yellow.
#     - Input also places yellow on the corner cell and on its right neighbor.
#     - Original task would have produced concentric frames using [base, red, green, yellow].
#     - The fused rule instead runs a clockwise spiral from that corner and repeats
#       [yellow, green, red, base] along the spiral order. The entrance seed matches the
#       first spiral color while the diagonal still advertises the reversed depth relation.
#
# complexity_calibration:
#   - Corner choice has 4 values and turn direction has 2 values, but both are repeated
#     across regions via the visible corner seed geometry instead of hidden metadata.
#   - Diagonal marker count is capped at 3 and always starts at depth 1. Fixing the start
#     depth avoids mixing the direction seed with the distance-coded markers and keeps the
#     reverse-palette effect visible in every active region.
#   - Some regions stay inactive (all base), so examples contain contrast between "do
#     nothing" regions and transformed regions without overcrowding the grid.
#
# implicit_assumptions_review:
#   - Original main() assumed non-base cells can be grouped solely by boundary depth. That
#     is no longer sufficient because two extra seed cells are now present and path order
#     controls the output.
#   - Original dist % len(palette) coloring assumed distance layers are the observable
#     mechanism. On rectangles that would make a literal loop-colored spiral collapse back
#     to concentric frames, so the fused solver must use spiral index modulo palette length.
#   - Original code did not need to infer a start corner or turn direction. The fused
#     solver now assumes every active region exposes exactly one same-color 2-cell corner
#     seed and exactly one consecutive inward diagonal of markers from that same corner.
#   - Region detection via connected non-wall components still holds, but decoding now
#     additionally assumes those components are full rectangles. The generator preserves it.
#
# placement_legality_and_generator_guarantees:
#   - Walls are still built by recursive full-row/full-column splits, so every free region
#     is a rectangle with min side length 3.
#   - For an active region, the generator first reserves the first two cells of one valid
#     spiral path as the seed, then places diagonal markers strictly inside the rectangle
#     at depths 1..k from the same corner. These placements never overlap and never cross
#     a wall.
#   - The generator rejects any sample whose region cues are ambiguous under the solver's
#     candidate search.
#
# visual_observability_and_cleanliness:
#   - Each active region has two simple cues only: a corner domino that points to the
#     spiral entrance, and a short inward diagonal that lists colors by depth.
#   - The deepest diagonal color is duplicated at the entrance, creating a clean visual
#     bridge between "reverse the depth order" and "start the spiral here".
#   - No color-id ordering is used. Everything is recoverable from visible geometry:
#     corner, edge direction, diagonal depth, and repeated color.
#
# inferability_of_change:
#   - Compared with the original task, the output is no longer made of closed frames. The
#     stripe continues around bends, so multiple regions in one example reinforce that the
#     path itself matters.
#   - Compared with a non-reversed spiral variant, the outermost spiral color matches the
#     deepest diagonal marker rather than the shallowest one. This creates an explicit,
#     local, human-visible reversal cue near every active corner.
#
# new_main_assumptions_audit:
#   - If a region is active, exactly one spiral candidate has its first two cells colored
#     with the same non-base value.
#   - The same corner also has a consecutive diagonal of 1..3 non-base cells at depths
#     1..k, with distinct colors, and the deepest diagonal color equals the seed color.
#   - There are no extra non-base cells elsewhere in the region. generate_input() enforces
#     all three assumptions and rejects any sample where decoding is not unique.

import random

import numpy as np

from arc_utils import components_mask

DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CORNER_SPECS = [
    ("tl", True, (0, 0), 1),
    ("tl", False, (0, 0), 2),
    ("tr", True, (0, 1), 2),
    ("tr", False, (0, 1), 3),
    ("br", True, (1, 1), 3),
    ("br", False, (1, 1), 0),
    ("bl", True, (1, 0), 0),
    ("bl", False, (1, 0), 1),
]
CORNER_DIAG = {
    "tl": (1, 1),
    "tr": (1, -1),
    "br": (-1, -1),
    "bl": (-1, 1),
}


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance to the component boundary, kept for regression checks."""
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


def _spiral_path_rect(rect, corner: str, clockwise: bool) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    for name, cw, (row_from, col_from), init_dir in CORNER_SPECS:
        if name == corner and cw == clockwise:
            sr = 0 if row_from == 0 else h - 1
            sc = 0 if col_from == 0 else w - 1
            break
    else:
        raise ValueError("unknown spiral spec")

    turn = 1 if clockwise else -1
    visited = np.zeros((h, w), dtype=bool)
    path: list[tuple[int, int]] = []
    r, c, d = sr, sc, init_dir

    for _ in range(h * w):
        visited[r, c] = True
        path.append((r0 + r, c0 + c))
        nr, nc = r + DIR4[d][0], c + DIR4[d][1]
        if not (0 <= nr < h and 0 <= nc < w and not visited[nr, nc]):
            d = (d + turn) % 4
            nr, nc = r + DIR4[d][0], c + DIR4[d][1]
        r, c = nr, nc
    return path


def _diagonal_cells(rect, corner: str, count: int, start_depth: int = 1) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    br = r0 if corner in ("tl", "tr") else r1
    bc = c0 if corner in ("tl", "bl") else c1
    dr, dc = CORNER_DIAG[corner]
    return [(br + dr * d, bc + dc * d) for d in range(start_depth, start_depth + count)]


def _max_diag_depth(rect) -> int:
    r0, c0, r1, c1 = rect
    return min(r1 - r0 + 1, c1 - c0 + 1) - 1


def _infer_region_layout(g: np.ndarray, pts: list[tuple[int, int]], base: int):
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    rect = (min(rows), min(cols), max(rows), max(cols))
    area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
    if area != len(pts):
        raise ValueError("variant solver expects rectangular regions")

    marker_cells = {(r, c): int(g[r, c]) for r, c in pts if int(g[r, c]) != base}
    if not marker_cells:
        return rect, None, None, []

    candidates = []
    for corner, clockwise, *_ in CORNER_SPECS:
        path = _spiral_path_rect(rect, corner, clockwise)
        seed0, seed1 = path[0], path[1]
        seed_color = int(g[seed0])
        if seed_color == base or seed_color != int(g[seed1]):
            continue

        diag_colors: list[int] = []
        expected_cells = {seed0, seed1}
        for cell in _diagonal_cells(rect, corner, _max_diag_depth(rect)):
            v = int(g[cell])
            if v == base:
                break
            diag_colors.append(v)
            expected_cells.add(cell)

        if not diag_colors:
            continue
        if len(diag_colors) > 3:
            continue
        if len(set(diag_colors)) != len(diag_colors):
            continue
        if seed_color != diag_colors[-1]:
            continue
        if set(marker_cells) != expected_cells:
            continue

        candidates.append((corner, clockwise, path, diag_colors))

    if not candidates:
        raise ValueError("could not infer spiral seed and diagonal palette")
    if len(candidates) != 1:
        raise ValueError("ambiguous region cues")

    corner, clockwise, path, diag_colors = candidates[0]
    return rect, (corner, clockwise), path, diag_colors


def _frame_fill_output(grid: np.ndarray) -> np.ndarray:
    """Original concentric-frame behavior for anti-regression checks."""
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
        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        if dist_to_color:
            ds = sorted(dist_to_color)
            palette = ([] if ds[0] == 0 else [int(base)]) + [int(dist_to_color[d]) for d in ds]
        else:
            palette = [int(base)]

        for r, c in pts:
            out[r, c] = palette[int(dist[r, c]) % len(palette)]
    return out


def _forward_spiral_output(grid: np.ndarray) -> np.ndarray:
    """Same spiral cue decoding, but without the D8 reversal."""
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    for pts in _cc4(g != wall):
        vals = np.array([g[r, c] for r, c in pts], dtype=int)
        base = _mode(vals)
        _rect, _spec, path, diag_colors = _infer_region_layout(g, pts, base)
        if path is None:
            for r, c in pts:
                out[r, c] = base
            continue
        palette = diag_colors + [base]
        for i, (r, c) in enumerate(path):
            out[r, c] = palette[i % len(palette)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each wall-partitioned region into a spiral stripe pattern whose color cycle is the reverse of the diagonal depth cue.

    Input structure:
    - The wall color partitions the board into rectangular free regions.
    - In an active region, one corner contains a 2-cell same-color seed on the first step of a hidden spiral.
    - The same corner also has a short inward diagonal of distinct marker colors, ordered from shallow to deeper Chebyshev depths.

    Rule:
    - For each region, find the unique corner and turn direction whose spiral starts with the visible 2-cell seed.
    - Read the diagonal marker colors by increasing depth, reverse that list, then append the region base color to complete the repeating palette.
    - Repaint the entire region by walking the inferred spiral path and assigning palette[path_index % palette_length].

    Key insight: The corner domino tells you where the spiral begins, while the diagonal tells you the palette in the opposite depth order. The outer spiral therefore starts with the deepest diagonal color, not the shallowest one.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    for pts in _cc4(g != wall):
        vals = np.array([g[r, c] for r, c in pts], dtype=int)
        base = _mode(vals)
        _rect, _spec, path, diag_colors = _infer_region_layout(g, pts, base)
        if path is None:
            for r, c in pts:
                out[r, c] = base
            continue

        palette = list(reversed(diag_colors)) + [base]
        for i, (r, c) in enumerate(path):
            out[r, c] = palette[i % len(palette)]

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

ACTIVE_REGION_MIN = 2
MARKER_STYLE_WEIGHTS = [
    ("inactive", 2),
    ("active1", 1),
    ("active2", 4),
    ("active3", 3),
]

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9
GEN_MAX_ATTEMPTS = 500


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
            wall_mask[rr, c0:c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0:r1 + 1, cc] = 1
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


def _cue_spec(rng: random.Random):
    labels, weights = zip(*MARKER_STYLE_WEIGHTS)
    choice = rng.choices(labels, weights=weights)[0]
    if choice == "inactive":
        return 0
    return int(choice[-1])


def _stamp_region_cues(grid, rect, bg: int, marker_pool: list[int], rng: random.Random):
    marker_count = _cue_spec(rng)
    if marker_count == 0:
        return {
            "active": False,
            "used_colors": set(),
            "corner": None,
            "clockwise": None,
            "diag_colors": [],
        }

    if marker_count > min(3, _max_diag_depth(rect)):
        return None

    corner, clockwise, *_ = rng.choice(CORNER_SPECS)
    path = _spiral_path_rect(rect, corner, clockwise)
    diag_colors = rng.sample(marker_pool, marker_count)
    seed_color = diag_colors[-1]

    placements = [(path[0], seed_color), (path[1], seed_color)]
    placements.extend(zip(_diagonal_cells(rect, corner, marker_count), diag_colors))

    seen = set()
    for (r, c), _color in placements:
        if (r, c) in seen:
            return None
        seen.add((r, c))
        if grid[r, c] != bg:
            return None

    for (r, c), color in placements:
        grid[r, c] = color

    return {
        "active": True,
        "used_colors": set(diag_colors),
        "corner": corner,
        "clockwise": clockwise,
        "diag_colors": diag_colors,
    }


def _stamp_all_cues(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    infos = []
    used_colors = set()
    for rect in rects:
        info = _stamp_region_cues(grid, rect, bg, marker_pool, rng)
        if info is None:
            return None, None
        infos.append(info)
        used_colors |= info["used_colors"]
    return infos, used_colors


def _validate_color_order(grid, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _candidate_signature(g: np.ndarray, rect, base: int):
    pts = [(r, c) for r in range(rect[0], rect[2] + 1) for c in range(rect[1], rect[3] + 1)]
    _rect, spec, _path, diag_colors = _infer_region_layout(g, pts, base)
    return spec, diag_colors


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)
    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    infos, used_colors = _stamp_all_cues(grid, rects, bg, marker_pool, rng)
    if infos is None:
        return None

    active_regions = sum(info["active"] for info in infos)
    if active_regions < ACTIVE_REGION_MIN:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used_colors) <= min(DISTINCT_MARKER_COLORS_MAX, len(marker_pool))):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    for rect, info in zip(rects, infos):
        if not info["active"]:
            continue
        try:
            spec, diag_colors = _candidate_signature(grid, rect, bg)
        except Exception:
            return None
        if spec != (info["corner"], info["clockwise"]):
            return None
        if diag_colors != info["diag_colors"]:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, _frame_fill_output(grid)):
        return None
    if np.array_equal(out, _forward_spiral_output(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
