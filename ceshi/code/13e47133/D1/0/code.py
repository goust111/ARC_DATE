# Variant of 13e47133
# Dimension: D1 — 路径机制：同心框架填充方式 (path)
# Change: 从区域边界向内计算切比雪夫距离，以 palette[dist % len] 产生同心矩形框架
#   → 改为沿区域内部的螺旋路径顺序循环填色
#
# Design analysis
# ---------------
# reference_task:
#   source_task da515329 was requested as the spiral-path reference. In this workspace,
#   only /home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/da515329.json is
#   present; no solver code file for da515329 exists here, so the implementation below
#   uses the reference samples as the baseline for "continuous spiral path" behavior.
#
# concretization_and_trace:
#   original:
#     - Each non-wall region is a rectangle from recursive wall splits.
#     - Marker colors are read by Chebyshev depth from the boundary.
#     - Output colors every cell by palette[boundary_depth % len(palette)].
#     - On rectangles this yields closed concentric frames.
#   variant:
#     - Keep the same partition-into-regions setup.
#     - Replace the controlling index: not boundary depth, but position along a
#       corner-anchored spiral traversal of the region.
#     - Input markers now sit on the first few cells of that spiral path, so they
#       encode both start corner and clockwise/counterclockwise direction.
#     - The first two cue cells intentionally share one color, creating a visible
#       2-cell "seed" on the spiral entrance; later cue cells extend the repeating
#       sequence. If the first marked cell is at path index 1, the base color is
#       prepended to the sequence.
#     small example:
#       For a top-left clockwise region, the spiral begins along the top edge.
#       Input cue cells could be [A, A, B] on path positions [0, 1, 2].
#       Original logic would have ignored that left-to-right order and instead grouped
#       by boundary depth; the variant repeats [A, B] along the full spiral path,
#       producing a winding stripe rather than nested frames.
#
# why_the_adaptation_is_necessary:
#   A literal "one color per full rectangular loop" spiral would be visually identical
#   to the original concentric-frame output on the generator's rectangular regions.
#   That would make D1 unobservable. To preserve the intended spiral-path mechanism and
#   make the change visible, the solver uses the spiral path order itself as the cyclic
#   index. This is the minimal compatible adjustment needed for ARC observability.
#
# complexity_calibration:
#   - Start corner has 4 choices; direction has 2 choices. Those branches repeat across
#     regions, but the cue is local and geometric rather than tied to color ids.
#   - Cue length is short (2-4 cells), preventing clutter.
#   - Regions without cues remain base-filled, so examples contain both active and
#     inactive regions, making the transformation easier to isolate.
#
# implicit_assumptions_review:
#   - The original _boundary_distance and dist % m coloring rely on "distance layers are
#     the controlling structure". That assumption is invalid here, so the variant must
#     replace both marker decoding and fill indexing.
#   - The original "one marker per depth" assumption also breaks: spiral-path cues need
#     ordered cells near the entrance, not one cell per depth ring.
#   - The original region detection via non-wall connected components still holds.
#   - This main() now assumes each region is a filled rectangle, because the generator's
#     wall construction uses full-row/full-column cuts. The generator re-enforces that.
#
# placement_legality_and_generator_guarantees:
#   - Every cue cell is chosen from the first few cells of a valid rectangular spiral,
#     so it is inside the region and never overlaps a wall.
#   - The generator rejects layouts whose cues would be ambiguous under multiple spiral
#     candidates.
#   - Regions are at least 3x3, so each spiral has enough room for a visible entrance.
#
# visual_observability_and_cleanliness:
#   - The repeated first cue color forms a short directional seed at a corner/edge.
#   - Later cue colors extend from that seed in traversal order, so the input visibly
#     hints at how the output should continue.
#   - No sorting depends on color id; only visible cue geometry and local order matter.
#
# new_main_assumptions_audit:
#   - If a region contains markers, they occupy one consecutive short interval of a
#     single spiral candidate, starting at path index 0 or 1.
#   - The first two marked cells on that interval share a color, uniquely anchoring the
#     entrance direction.
#   - generate_input() constructs exactly that pattern and rejects ambiguous cases.

import random

import numpy as np

from arc_utils import components_mask

DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
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


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Original task helper kept for generator-side regression checks."""
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = (
                r == 0 or r == h - 1 or c == 0 or c == w - 1
                or not component_mask[r - 1, c]
                or not component_mask[r + 1, c]
                or not component_mask[r, c - 1]
                or not component_mask[r, c + 1]
            )
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR4:
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


def _infer_region_pattern(g: np.ndarray, pts: list[tuple[int, int]], base: int):
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    rect = (min(rows), min(cols), max(rows), max(cols))
    area = (rect[2] - rect[0] + 1) * (rect[3] - rect[1] + 1)
    if area != len(pts):
        raise ValueError("variant solver expects rectangular regions")

    marker_cells = {(r, c): int(g[r, c]) for r, c in pts if int(g[r, c]) != base}
    if not marker_cells:
        return rect, None, None, [base]

    candidates = []
    for corner, clockwise, *_ in CORNER_SPECS:
        path = _spiral_path_rect(rect, corner, clockwise)
        idx = {cell: i for i, cell in enumerate(path)}
        try:
            positions = sorted(idx[cell] for cell in marker_cells)
        except KeyError:
            continue
        if positions[0] not in (0, 1):
            continue
        if positions != list(range(positions[0], positions[0] + len(positions))):
            continue

        colors = [int(g[path[i]]) for i in positions]
        if len(colors) < 2:
            continue
        if colors[0] != colors[1]:
            continue
        tail = colors[2:]
        if len(set(tail)) != len(tail):
            continue
        if colors[0] in tail:
            continue

        pattern = ([base] if positions[0] == 1 else []) + [colors[0]] + tail
        if len(pattern) < 2:
            continue

        candidates.append((len(colors), -positions[0], corner, clockwise, path, pattern))

    if not candidates:
        raise ValueError("could not infer spiral seed")

    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][:2] == best[:2]:
        raise ValueError("ambiguous spiral seed")

    _, _, corner, clockwise, path, pattern = best
    return rect, (corner, clockwise), path, pattern


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions by extending a short spiral cue into a full spiral stripe pattern.

    Input structure:
    - The wall color splits the grid into rectangular free regions.
    - Each region has one dominant base color, plus a few cue pixels placed on the first cells of a corner-started spiral.
    - Active regions begin with a 2-cell same-color seed that reveals the spiral entrance and turn direction.

    Rule:
    - For each region, test the eight rectangular spirals (4 corners x 2 turn directions) and find the one whose first marked cells match the cue.
    - Read the cue colors in spiral order to obtain the repeating sequence; if the cue starts at path index 1, prepend the base color.
    - Repaint the whole region by following that spiral path and assigning sequence[path_index % sequence_length].

    Key insight: The markers no longer describe distance layers. They are a literal prefix of the path to continue, so the solver must recover the correct spiral start and orientation before tiling the color sequence through the region.
    """
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
        _rect, _spec, path, pattern = _infer_region_pattern(g, pts, base)
        if path is None:
            for r, c in pts:
                out[r, c] = base
            continue
        m = len(pattern)
        for i, (r, c) in enumerate(path):
            out[r, c] = pattern[i % m]

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

ACTIVE_REGION_MIN = 2
NONBASE_COLORS_MAX = 3
MARKER_STYLE_WEIGHTS = [
    ("inactive", 2),
    ("offset1_len1", 2),
    ("offset1_len2", 2),
    ("offset1_len3", 1),
    ("offset0_len2", 3),
    ("offset0_len3", 2),
]

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9
GEN_MAX_ATTEMPTS = 400


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
    if choice.startswith("offset1"):
        return 1, int(choice[-1])
    return 0, int(choice[-1])


def _stamp_region_cues(grid, rect, bg: int, marker_pool: list[int], rng: random.Random):
    spec = _cue_spec(rng)
    if spec is None:
        return {
            "active": False,
            "used_colors": set(),
            "corner": None,
            "clockwise": None,
            "start_idx": None,
            "pattern_len": 1,
        }

    start_idx, nonbase_len = spec
    corner, clockwise, *_ = rng.choice(CORNER_SPECS)
    path = _spiral_path_rect(rect, corner, clockwise)

    cue_colors = rng.sample(marker_pool, nonbase_len)
    cue_cells = [cue_colors[0], cue_colors[0], *cue_colors[1:]]
    cells = path[start_idx:start_idx + len(cue_cells)]
    if len(cells) != len(cue_cells):
        return None
    for (r, c), color in zip(cells, cue_cells):
        if grid[r, c] != bg:
            return None
        grid[r, c] = color

    return {
        "active": True,
        "used_colors": set(cue_colors),
        "corner": corner,
        "clockwise": clockwise,
        "start_idx": start_idx,
        "pattern_len": nonbase_len + start_idx,
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
    r0, c0, r1, c1 = rect
    pts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    _rect, spec, _path, pattern = _infer_region_pattern(g, pts, base)
    return spec, pattern


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
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used_colors) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    # Ensure each active region is uniquely decodable by the solver's cue parser.
    for rect, info in zip(rects, infos):
        if not info["active"]:
            continue
        try:
            spec, pattern = _candidate_signature(grid, rect, bg)
        except Exception:
            return None
        if spec != (info["corner"], info["clockwise"]):
            return None
        expected_len = info["pattern_len"]
        if len(pattern) != expected_len:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, _frame_fill_output(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
