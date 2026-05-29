# Variant of 35ab12c3
# Dimension: D1 — 匹配条件——同色配对规则 (match_cond)
# Change: color_match——同色像素配对（s1.color == s2.color），同一颜色的所有像素按极角排序后形成多... → 互补色配对
#
# Design analysis
# - Concrete behavior:
#   The original task treated every non-bg color as an independent vertex set.
#   This variant keeps the same geometry and optional shadow mechanism, but each
#   shape now uses a fixed complementary color pair: (1,8), (2,4), or (3,7).
#   A solver must merge both colors of one pair into a single vertex cycle,
#   rather than connecting equal-color pixels separately. Input vertices of one
#   shape alternate between the two paired colors; output fills the connecting
#   edges with the start vertex color, while opposite-color vertices remain
#   visible at corners/endpoints because drawing only overwrites background.
# - Small trace:
#   Original logic on vertices [1,1,1,1] built one cycle for color 1.
#   New logic on vertices [1,8,1,8] must first merge {1,8}, then sort all four
#   coordinates together, then connect them as one polygon. Running the old
#   same-color logic would instead create two unrelated 1-colored and 8-colored
#   fragments, so the output diverges materially.
# - Complexity calibration:
#   Pairing is limited to three globally fixed visible families. This keeps the
#   hypothesis space small enough for ARC while still requiring a genuine
#   “infer the color relation” step. We forbid two shapes from sharing the same
#   pair in one grid, otherwise pairwise grouping would become ambiguous.
# - Hidden assumptions from the original solver:
#   1. `for col in colors:` assumed one color == one shape. This breaks, so the
#      main loop now iterates over complementary families instead.
#   2. `_clear_in_original(... allow={col})` assumed only the active color could
#      appear on a valid segment interior. This is no longer true because a
#      partner-color vertex may lie on the same shape, so `allow` must include
#      both colors of the pair.
#   3. Shadow seed detection used “adjacent non-bg different color”. That still
#      works only if partner-color vertices are never 8-neighbors of the anchor.
#      The generator preserves the original minimum gap constraints so paired
#      vertices are separated by at least one background cell.
#   4. Carrying adjacent non-shape pixels must exclude both colors of the pair,
#      otherwise the solver could mistake a partner color for a shadow marker.
# - Placement legality:
#   Patches are still placed by their solved output footprints with >=1 cell
#   padding. Shapes use unique pair families per grid, shadows use only neutral
#   color 6, and optional distractors also use 6 but are never adjacent to a
#   shape because placement is done on the full solved patch.
# - Visual observability:
#   The difference from the original task is visible immediately: one geometric
#   object contains two alternating vertex colors, yet the output connects them
#   into a single outline. Across examples, the same three pair families recur,
#   so a human can infer the mapping from repeated support rather than from
#   hidden numeric ids.
# - Color-rule inferability:
#   Only three pairings are ever used, and every active shape uses both colors of
#   its pair. The neutral shadow/distractor color never participates in pairing,
#   which keeps the mapping crisp.
# - New solver assumptions audited:
#   Each active pair family appears in at most one shape per grid; both colors of
#   that family are present; any adjacent non-pair pixel at the chosen start
#   vertex is a shadow anchor, not another shape. The generator explicitly
#   guarantees all three assumptions.

import math
import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BG = 0
PAIR_FAMILIES = [(1, 8), (2, 4), (3, 7)]
PAIR_OF = {a: b for a, b in PAIR_FAMILIES}
PAIR_OF.update({b: a for a, b in PAIR_FAMILIES})
NEUTRAL_COLOR = 6

# ---- CFG: canvas and placement ----
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: structure ----
SHAPE_COUNT_RANGE = (2, 4)          # 2-3 complementary-pair shapes
MAX_SHADOWS_CAP = 2
DISTRACTOR_PROB = 0.45
SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300

# ---- CFG: shape geometry ----
LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)
SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

# ---- CFG: distractor geometry ----
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _draw_line(arr: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    for rr, cc in _segment_cells(a[0], a[1], b[0], b[1]):
        arr[rr, cc] = color


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _rotate_cycle(cycle: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    idx = cycle.index(start)
    return cycle[idx:] + cycle[:idx]


def _seed_index(cycle: list[tuple[int, int]], g: np.ndarray, bg: int, shape_colors: set[int]) -> int:
    h, w = g.shape
    seed_idxs: list[int] = []
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v not in shape_colors:
                    seed_idxs.append(i)
                    break
    if seed_idxs:
        return min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    return min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    out = g.copy()

    for col_a, col_b in PAIR_FAMILIES:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere((g == col_a) | (g == col_b))]
        if len(pts_rc) < 2:
            continue
        present = {int(g[r, c]) for r, c in pts_rc}
        if present != {col_a, col_b}:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        shape_colors = {col_a, col_b}
        start = _seed_index(cycle, g, bg, shape_colors)
        cycle = cycle[start:] + cycle[:start]
        base_color = int(g[cycle[0]])

        carried: list[tuple[int, int, int]] = []
        seen_carried: set[tuple[int, int, int]] = set()
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                item = (drn, dcn, v)
                if v != bg and v not in shape_colors and item not in seen_carried:
                    seen_carried.add(item)
                    carried.append(item)

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow=shape_colors):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, base_color, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    anchor: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch() -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), NEUTRAL_COLOR, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = NEUTRAL_COLOR
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), NEUTRAL_COLOR, dtype=int)
    return patch.copy(), patch.copy()


def _pair_coloring(cycle: list[tuple[int, int]], pair: tuple[int, int], start: tuple[int, int]) -> tuple[list[tuple[int, int]], int, int]:
    rotated = _rotate_cycle(cycle, start)
    if np.random.rand() < 0.5:
        main_color, partner_color = pair
    else:
        partner_color, main_color = pair
    return rotated, int(main_color), int(partner_color)


def _make_shape_patch(pair: tuple[int, int], has_shadow: bool) -> tuple[np.ndarray, np.ndarray] | None:
    cycle = _angle_cycle(_make_vertices())
    if len(cycle) < 2:
        return None
    start = min(cycle)
    shape_cells = _shape_cells(cycle)

    shadow = None
    if has_shadow:
        shadow = _choose_shadow(cycle, shape_cells, start)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(shape_cells, cycle, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    rotated, main_color, partner_color = _pair_coloring(cycle, pair, start)
    shifted_vertices = [(r - r0, c - c0) for r, c in rotated]
    for i, (r, c) in enumerate(shifted_vertices):
        inp[r, c] = main_color if i % 2 == 0 else partner_color

    if has_shadow and shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = NEUTRAL_COLOR
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == NEUTRAL_COLOR
        )
        if nb_hits != 1:
            return None

    out = main(inp)
    if np.array_equal(inp, out):
        return None

    fill_added = np.argwhere((out != inp) & (out == main_color))
    if len(fill_added) == 0:
        return None
    if partner_color not in inp:
        return None
    if has_shadow and np.count_nonzero(out == NEUTRAL_COLOR) <= np.count_nonzero(inp == NEUTRAL_COLOR):
        return None
    return inp, out


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    chosen_pairs = [PAIR_FAMILIES[i] for i in np.random.permutation(len(PAIR_FAMILIES))[:shape_count]]
    shadow_count = int(np.random.randint(min(MAX_SHADOWS_CAP, shape_count) + 1))
    shadow_targets = set(np.random.choice(shape_count, size=shadow_count, replace=False).tolist()) if shadow_count else set()
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)

    for idx, pair in enumerate(chosen_pairs):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(pair, idx in shadow_targets)
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor:
        if not _try_place(inp, out, *_make_distractor_patch()):
            return None

    expected_colors = {c for pair in chosen_pairs for c in pair}
    if shadow_targets or has_distractor:
        expected_colors.add(NEUTRAL_COLOR)

    in_colors = set(colors_present(inp, exclude=[BG]))
    if in_colors != expected_colors:
        return None
    if int(get_bg(inp)) != BG or int(get_bg(out)) != BG:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
