# Variant of 35ab12c3
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: continuous_trail——沿路径的每个背景格子都涂上形状颜色，形成连续实线... → 周期性颜色交替轨迹
#
# Design analysis for the variant
# a) Concrete behavior and trace:
#    - Original task: a shape color `A` has 2+ vertices. Solver connects drawable H/V/45deg
#      edges and paints every background trail cell with `A`. If a single adjacent shadow
#      anchor `B` sits near the chosen start vertex, the parallel copied trail is painted
#      solid `B`.
#    - Variant: keep the same vertex detection, edge filtering, start-vertex choice, and
#      shadow-offset logic, but replace "solid trail fill" with a periodic trail. For the
#      main edge, cells follow `A,B,A,B,...` with the source vertex already being `A`.
#      For the parallel shadow edge, cells follow the opposite phase `B,A,B,A,...` because
#      the shadow anchor already provides the `B` seed. Example on a horizontal segment:
#        input:   A . . A      with shadow anchor B just above the left A
#                 . B . .
#        output:  A B A A      main trail alternates from the left endpoint
#                 B A B B      shifted copy alternates in opposite phase
#      The endpoints stay as the original input markers; only background trail cells are
#      newly painted, so the task remains "connect the markers", not "recolor the markers".
# b) Complexity calibration:
#    - To keep the striping legible, the generator makes every shape carry exactly one
#      shadow color. That yields a repeated 2-color cycle per shape instead of mixing
#      several unrelated policies in one example.
#    - To avoid ambiguous endings where two same-color vertices would force an odd-length
#      segment to break the visible alternation at the far endpoint, generator segment
#      lengths are sampled as even numbers. This is an adaptation for observability, not a
#      new inference burden: the solver itself still tolerates odd lengths.
# c) Hidden assumptions in original `main()` that matter here:
#    - `_draw_line_bg_only` assumed every new trail cell should take one constant color.
#      That is no longer valid; we need a step-indexed palette lookup.
#    - The original phase order of drawing (main first, then shadows) was "harmless"
#      because every line was monochrome. It still matters now, because shadow-colored
#      cells appear on the main trail; drawing main first preserves the priority that the
#      actual shape geometry is established before offset copies try to use remaining bg.
#    - Start-vertex selection must remain stable. The periodic phase is anchored at the
#      chosen start, so generator guarantees the unique shadow anchor is attached to the
#      lexicographically smallest seed vertex, matching the original tie-break rule.
# d) Placement legality:
#    - The geometry footprint is unchanged: periodic coloring uses the same cells as the
#      old solid trails. Therefore the existing output-first patch placement strategy still
#      works once the local output patch is generated with the new solver.
#    - Generator keeps a >=1-cell buffer between placed patches, and shadow offsets are
#      still restricted to non-collinear 8-neighbor directions so copied trails do not
#      collapse onto their source trail.
# e) Visual observability:
#    - The visible clue is structural: each shape/shadow pair forms two parallel striped
#      polylines using the two colors already present in the input. No color-id ordering is
#      used. A human can infer from 2-3 examples that the neighbor color participates in
#      the trail instead of staying a solid shadow.
# f) Visual cleanliness:
#    - Only one shadow per shape is used, which avoids multi-layer overlaps and keeps each
#      pair readable as "main striped trail + offset striped trail".
# g) Color-change inferability:
#    - The color cycle is always derived from visibly adjacent colors of one local group:
#      shape color and its single shadow color. This keeps the mapping local and repeated,
#      rather than asking the solver to infer a global palette permutation.

import math
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


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


def _periodic_fill_bg_only(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    palette: list[int],
    bg: int,
) -> None:
    if not palette:
        return
    for k, (rr, cc) in enumerate(_segment_cells(r0, c0, r1, c1)):
        if out[rr, cc] == bg:
            out[rr, cc] = int(palette[k % len(palette)])


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _shape_plan(
    g: np.ndarray,
    col: int,
    bg: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == int(col))]
    if len(pts_rc) < 2:
        return None

    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return None

    seed_idxs = []
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_idxs.append(i)
                    break

    if seed_idxs:
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    else:
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

    cycle = cycle[start:] + cycle[:start]

    carried: list[tuple[int, int, int]] = []
    sr0, sc0 = cycle[0]
    for drn, dcn in DIR8:
        rr = sr0 + drn
        cc = sc0 + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != col:
                carried.append((drn, dcn, v))

    if len(cycle) == 2:
        edges = [(cycle[0], cycle[1])]
    else:
        edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), (r1, c1) in edges:
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={int(col)}):
            continue
        drawable_edges.append(((r0, c0), (r1, c1)))

    return carried, drawable_edges


def _main_palette(col: int, carried: list[tuple[int, int, int]]) -> list[int]:
    palette = [int(col)]
    seen = {int(col)}
    for _drn, _dcn, ncol in carried:
        ncol = int(ncol)
        if ncol not in seen:
            palette.append(ncol)
            seen.add(ncol)
    return palette


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        plan = _shape_plan(g, int(col), int(bg))
        if plan is None:
            continue
        carried, drawable_edges = plan

        main_palette = _main_palette(int(col), carried)
        for (r0, c0), (r1, c1) in drawable_edges:
            _periodic_fill_bg_only(out, r0, c0, r1, c1, main_palette, int(bg))

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < g.shape[0] and 0 <= cc0 < g.shape[1] and 0 <= rr1 < g.shape[0] and 0 <= cc1 < g.shape[1]):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, int(bg), allow=set()):
                    continue
                _periodic_fill_bg_only(out, rr0, cc0, rr1, cc1, [int(ncol), int(col)], int(bg))

    return out


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = main(g)
    striped_groups = 0
    trail_bg_cells = 0
    off_color_main_cells = 0

    for col in sorted(set(int(v) for v in np.unique(g)) - {int(bg)}):
        plan = _shape_plan(g, int(col), int(bg))
        if plan is None:
            continue
        carried, drawable_edges = plan
        if carried:
            striped_groups += 1
        for (r0, c0), (r1, c1) in drawable_edges:
            for rr, cc in _segment_cells(r0, c0, r1, c1):
                if int(g[rr, cc]) != int(bg):
                    continue
                trail_bg_cells += 1
                if int(out[rr, cc]) != int(col):
                    off_color_main_cells += 1

    return {
        "striped_groups": int(striped_groups),
        "trail_bg_cells": int(trail_bg_cells),
        "off_color_main_cells": int(off_color_main_cells),
    }


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)         # 2-3 shapes; each one has exactly one shadow
DISTRACTOR_PROB = 0.35
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 12
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

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


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


def _even_choices(lo: int, hi: int) -> list[int]:
    return [int(v) for v in range(int(lo), int(hi)) if v % 2 == 0]


def _even_rand(lo: int, hi: int) -> int:
    choices = _even_choices(lo, hi)
    if not choices:
        raise ValueError(f"no even values in range [{lo}, {hi})")
    return int(np.random.choice(choices))


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        choices = [g for g in range(LINE_MIN_GAP, gap_hi + 1) if g % 2 == 0]
        if not choices:
            return [(0, 0), (0, LINE_TOTAL_LENGTH)]
        gap = int(np.random.choice(choices))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = _even_rand(*TRIANGLE_SIDE_RANGE)
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = _even_rand(*RECT_H_RANGE)
    w = _even_rand(*RECT_W_RANGE)
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = _even_rand(*DIAMOND_ARM_RANGE)
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = _even_rand(*PENTAGON_ROOF_RANGE)
    wall = _even_rand(*PENTAGON_WALL_RANGE)
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = _even_rand(*HEXAGON_SLANT_RANGE)
    wall = _even_rand(*HEXAGON_WALL_RANGE)
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
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
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


def _make_shape_patch(shape_color: int, shadow_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)
    shadow = _choose_shadow(vertices, cells)
    if shadow is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted_vertices:
        inp[r, c] = int(shape_color)

    (ar, ac), offset = shadow
    anchor = (ar - r0, ac - c0)
    sa = (anchor[0] + offset[0], anchor[1] + offset[1])
    if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
        return None
    inp[sa] = int(shadow_color)

    nb_hits = sum(
        1
        for rr, cc in neighbors8(anchor[0], anchor[1])
        if 0 <= rr < h and 0 <= cc < w and int(inp[rr, cc]) == int(shadow_color)
    )
    if nb_hits != 1:
        return None

    out = main(inp)
    if np.count_nonzero((out == int(shadow_color)) & (inp != int(shadow_color))) == 0:
        return None
    if np.count_nonzero((out == int(shape_color)) & (inp == BG)) == 0:
        return None
    return inp, out


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


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    total = 2 * int(shape_count) + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count:2 * shape_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


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
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    can_add_distractor = 2 * shape_count + 1 <= len(COLORS)
    has_distractor = bool(can_add_distractor and np.random.rand() < DISTRACTOR_PROB)

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette

    for shape_color, shadow_color in zip(shape_colors, shadow_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), int(shadow_color))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(int(distractor_color))):
            return None

    total_colors = 2 * shape_count + int(has_distractor)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _variant_metrics(inp)
    if metrics["striped_groups"] != shape_count:
        return None
    if metrics["off_color_main_cells"] < shape_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
