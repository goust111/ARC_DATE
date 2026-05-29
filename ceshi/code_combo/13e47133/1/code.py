# Combo variant of 13e47133 (index 1)
# combo_spec: D1/0+D2/0
# Name: 螺旋路径+周期性颜色重复
#
# Design analysis
# ---------------
# concretization_and_trace:
#   original:
#     - Walls split the canvas into 4-connected free regions.
#     - Marker colors are decoded by Chebyshev depth from the region boundary.
#     - The output uses palette[depth % m], so rectangles become concentric frames.
#   combo:
#     - Keep the same wall-partitioned rectangular regions and dominant base color.
#     - Replace "distance layers" with a corner-started rectangular spiral path.
#     - Input markers are no longer isolated by depth. They occupy one short
#       consecutive interval on the spiral entrance and appear as repeated pairs
#       like [A, A, B, B] or [A, A, B, B, C, C].
#     - The output follows the whole spiral path, but palette advancement happens
#       once every 2 visited cells, so the winding stripe reads AABBAABB... .
#     small_trace:
#       Suppose a 6x6 region uses the top-left clockwise spiral.
#       Original with palette [A, B] would color by frame depth:
#         outer ring A, next ring B, next ring A, ...
#       Combo input may show the first 4 spiral cells as [A, A, B, B].
#       The combo output continues along the spiral path itself:
#         path cells 0-1 -> A, 2-3 -> B, 4-5 -> A, 6-7 -> B, ...
#       This is visually different from both the original frames and from a plain
#       ABAB spiral because each color occupies a 2-cell run.
#
# complexity_calibration:
#   - The branch space is still modest: 4 start corners x 2 turn directions, with
#     a fixed run length of 2 and short palettes of length 2-4.
#   - Each sample contains multiple active regions, so the same 2-cell periodicity
#     repeats enough times for humans to infer it.
#   - Cue lengths are capped at 4 or 6 colored cells. That is long enough to reveal
#     both spiral orientation and the AABB beat, but short enough to avoid clutter.
#
# implicit_assumptions_review:
#   - The original `dist = _boundary_distance(...)` and `palette[dist % m]` logic
#     assume boundary depth is the controlling index. That assumption is invalid in
#     this combo; the controlling index is spiral visitation order.
#   - The original "one marker per depth" assumption also breaks. Here markers form
#     contiguous path cues, often touching orthogonally, so cue decoding must use
#     path order instead of distance buckets.
#   - `bg = mode(grid)` and `wall = second mode` still rely on markers being sparse;
#     the generator keeps regions large enough that this remains true.
#   - The solver now assumes each free component is a filled rectangle. The wall
#     generator uses full-row/full-column splits, so every component satisfies it.
#
# placement_legality_and_generator_guarantees:
#   - Every cue cell comes from a valid prefix of a rectangular spiral path, so it
#     always lies inside the chosen region and never touches a wall cell.
#   - Regions are at least 5x5, which leaves enough room for a visible spiral
#     entrance and for a 4- or 6-cell cue without running off the path.
#   - The generator validates that each active region is uniquely decodable by the
#     same candidate search used by the solver, rejecting ambiguous cue layouts.
#
# visual_observability:
#   - The start corner and turning direction are visible because the colored cue is
#     a literal piece of the future spiral, not an unordered set of depths.
#   - The periodicity is visible because every cue already shows repeated pairs, and
#     the output continues that exact 2-cell rhythm over a much longer path.
#   - Degenerate samples are rejected when the combo output is too close either to
#     the plain spiral-without-pairs output or to the original concentric-frame
#     output on the same input.
#
# cleanliness_and_color_inference:
#   - Free regions contain only three visual ingredients: base fill, wall lines, and
#     one short colored cue near a corner. There are no overlapping annotations.
#   - Distinct marker colors within a region stay few and ordered by visible path
#     position, so humans infer color progression from geometry rather than from
#     hidden properties such as numeric color ids.
#
# new_main_assumptions_audit:
#   - If a region is active, all non-base cells in that region lie on one
#     consecutive spiral interval whose first index is either 0 or 2.
#   - Cue colors arrive in exact 2-cell blocks, and each block corresponds to one
#     palette entry. A delayed cue with start index 2 means the palette begins with
#     one 2-cell base block before the first non-base block.
#   - `generate_input()` explicitly constructs those conditions and checks the
#     recovered `(corner, direction, start_idx, palette)` against the planted one.

import random

import numpy as np

from arc_utils import components_mask

DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BAND_RUN_LEN = 2
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
SPIRAL_CHOICES = [(corner, clockwise) for corner, clockwise, *_ in CORNER_SPECS]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Original-task helper kept for generator-side regression checks."""
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


def _enumerate_region_candidates(g: np.ndarray, pts: list[tuple[int, int]], base: int):
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    rect = (min(rows), min(cols), max(rows), max(cols))
    area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
    if area != len(pts):
        return rect, []

    marker_cells = {(r, c): int(g[r, c]) for r, c in pts if int(g[r, c]) != base}
    if not marker_cells:
        return rect, []

    candidates = []
    for corner, clockwise in SPIRAL_CHOICES:
        path = _spiral_path_rect(rect, corner, clockwise)
        idx = {cell: i for i, cell in enumerate(path)}
        try:
            positions = sorted(idx[cell] for cell in marker_cells)
        except KeyError:
            continue

        if positions[0] not in (0, BAND_RUN_LEN):
            continue
        if positions != list(range(positions[0], positions[0] + len(positions))):
            continue
        if len(positions) < 2 * BAND_RUN_LEN or len(positions) % BAND_RUN_LEN != 0:
            continue

        colors = [int(g[path[i]]) for i in positions]
        palette_nonbase: list[int] = []
        ok = True
        for i in range(0, len(colors), BAND_RUN_LEN):
            block = colors[i:i + BAND_RUN_LEN]
            if len(block) != BAND_RUN_LEN or len(set(block)) != 1:
                ok = False
                break
            palette_nonbase.append(int(block[0]))
        if not ok or len(set(palette_nonbase)) != len(palette_nonbase):
            continue

        start_idx = positions[0]
        palette = ([int(base)] if start_idx == BAND_RUN_LEN else []) + palette_nonbase
        for pos, color in zip(positions, colors):
            expected = palette[(pos // BAND_RUN_LEN) % len(palette)]
            if color != expected:
                ok = False
                break
        if not ok:
            continue

        candidates.append(
            {
                "corner": corner,
                "clockwise": clockwise,
                "start_idx": start_idx,
                "path": path,
                "palette": palette,
                "marker_len": len(positions),
            }
        )

    return rect, candidates


def _infer_region_pattern(g: np.ndarray, pts: list[tuple[int, int]], base: int):
    rect, candidates = _enumerate_region_candidates(g, pts, base)
    marker_count = sum(int(g[r, c]) != base for r, c in pts)
    if marker_count == 0:
        return rect, None, None, [int(base)]
    if not candidates:
        raise ValueError("could not infer spiral cue")

    candidates.sort(key=lambda cand: (cand["marker_len"], -cand["start_idx"]), reverse=True)
    best = candidates[0]
    best_sig = (best["marker_len"], -best["start_idx"])
    if len(candidates) > 1:
        second = candidates[1]
        if (second["marker_len"], -second["start_idx"]) == best_sig:
            raise ValueError("ambiguous spiral cue")

    return rect, (best["corner"], best["clockwise"], best["start_idx"]), best["path"], best["palette"]


def _solve_spiral_with_run(grid: np.ndarray, band_run_len: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        vals = np.array([g[r, c] for r, c in pts], dtype=int)
        base = _mode(vals)
        try:
            _rect, _spec, path, palette = _infer_region_pattern(g, pts, base)
        except ValueError:
            path = None
            palette = [int(base)]

        if path is None:
            for r, c in pts:
                out[r, c] = int(base)
            continue

        for i, (r, c) in enumerate(path):
            out[r, c] = int(palette[(i // band_run_len) % len(palette)])

    return out


def _frame_fill_output(grid: np.ndarray) -> np.ndarray:
    """Original concentric-frame behavior for generator-side anti-regression checks."""
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
            out[r, c] = int(palette[int(dist[r, c]) % len(palette)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions by extending a short spiral cue into a paired-run spiral pattern.

    Input structure:
    - The wall color splits the grid into rectangular free regions.
    - Each region has one dominant base color and may contain a short colored cue near one corner.
    - In active regions, the cue sits on consecutive cells of a corner-started spiral and appears in repeated pairs such as A,A,B,B or A,A,B,B,C,C.

    Rule:
    - For each region, test the eight rectangular spirals (4 corners x 2 turn directions) and find the one whose marked cells form one consecutive cue interval, beginning either immediately or after one 2-cell base block.
    - Collapse each same-color pair in that cue into one palette entry; if the cue starts after two base cells, prepend the base color to the palette.
    - Repaint the whole region along the inferred spiral path, advancing to the next palette entry every two visited cells, so the path color sequence repeats as AABBAABB... .

    Key insight: The cue is both geometric and temporal. Its position tells you which spiral to continue, and its repeated pairs tell you that palette colors persist for two spiral steps before the pattern advances.
    """
    return _solve_spiral_with_run(grid, BAND_RUN_LEN)


# --------------- generator ---------------

H_MIN, H_MAX = 14, 30
W_MIN, W_MAX = 16, 30

N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 5
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

ACTIVE_REGION_MIN = 2
MARKER_STYLE_WEIGHTS = [
    ("inactive", 2),
    ("offset0_len2", 4),
    ("offset0_len3", 3),
    ("offset2_len2", 2),
    ("offset2_len3", 1),
]

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

MIN_DIFF_SPIRAL_CELLS = 20
MIN_DIFF_FRAME_CELLS = 20
MIN_OFFSET_REGIONS = 1
MIN_PALETTE3_REGIONS = 1

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
        return None
    start_idx = 0 if choice.startswith("offset0") else BAND_RUN_LEN
    nonbase_len = int(choice[-1])
    return start_idx, nonbase_len


def _stamp_region_cue(grid, rect, bg: int, marker_pool: list[int], rng: random.Random):
    spec = _cue_spec(rng)
    if spec is None:
        return {
            "active": False,
            "used_colors": set(),
            "corner": None,
            "clockwise": None,
            "start_idx": None,
            "palette": [int(bg)],
        }

    start_idx, nonbase_len = spec
    corner, clockwise = rng.choice(SPIRAL_CHOICES)
    path = _spiral_path_rect(rect, corner, clockwise)
    cue_len = BAND_RUN_LEN * nonbase_len
    cells = path[start_idx:start_idx + cue_len]
    if len(cells) != cue_len:
        return None

    cue_colors = rng.sample(marker_pool, nonbase_len)
    cue_values = [color for color in cue_colors for _ in range(BAND_RUN_LEN)]
    for (r, c), color in zip(cells, cue_values):
        if grid[r, c] != bg:
            return None
        grid[r, c] = int(color)

    palette = ([int(bg)] if start_idx == BAND_RUN_LEN else []) + [int(color) for color in cue_colors]
    return {
        "active": True,
        "used_colors": set(int(color) for color in cue_colors),
        "corner": corner,
        "clockwise": clockwise,
        "start_idx": start_idx,
        "palette": palette,
    }


def _stamp_all_cues(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    infos = []
    used_colors = set()
    for rect in rects:
        info = _stamp_region_cue(grid, rect, bg, marker_pool, rng)
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
    r0, c0, r1, c1 = rect
    pts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    _rect, spec, _path, palette = _infer_region_pattern(g, pts, base)
    return spec, palette


def _variant_metrics(grid: np.ndarray, infos) -> dict[str, int]:
    combo_out = _solve_spiral_with_run(grid, BAND_RUN_LEN)
    plain_spiral_out = _solve_spiral_with_run(grid, 1)
    frame_out = _frame_fill_output(grid)

    return {
        "diff_spiral": int(np.count_nonzero(combo_out != plain_spiral_out)),
        "diff_frame": int(np.count_nonzero(combo_out != frame_out)),
        "active_regions": sum(int(info["active"]) for info in infos),
        "offset_regions": sum(int(info["active"] and info["start_idx"] == BAND_RUN_LEN) for info in infos),
        "palette3_regions": sum(int(info["active"] and len(info["palette"]) >= 3) for info in infos),
    }


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
    if infos is None or used_colors is None:
        return None

    if sum(int(info["active"]) for info in infos) < ACTIVE_REGION_MIN:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used_colors) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    for rect, info in zip(rects, infos):
        if not info["active"]:
            continue
        try:
            spec, palette = _candidate_signature(grid, rect, bg)
        except Exception:
            return None
        if spec != (info["corner"], info["clockwise"], info["start_idx"]):
            return None
        if palette != info["palette"]:
            return None

    metrics = _variant_metrics(grid, infos)
    if metrics["diff_spiral"] < MIN_DIFF_SPIRAL_CELLS:
        return None
    if metrics["diff_frame"] < MIN_DIFF_FRAME_CELLS:
        return None
    if metrics["offset_regions"] < MIN_OFFSET_REGIONS:
        return None
    if metrics["palette3_regions"] < MIN_PALETTE3_REGIONS:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
