# Combo variant of 35ab12c3 (index 26)
# combo_spec: D4/1+D20/2
# Name: 颜色交替轨迹+阴影颜色编码偏移方向
#
# Design analysis for the combo
# a) Concrete behavior / trace:
#    - Base task: same-color vertices are connected by H/V/45deg segments; an adjacent
#      foreign-color pixel indicates a parallel shadow copy using that pixel's color.
#    - D20/2 changes the shadow semantics first: the adjacent pixel becomes a direction
#      marker. Its color, not its geometric side, determines the actual shadow offset.
#      We fix 4 reusable marker colors: 2=up, 1=right, 7=down, 8=left. The cue pixel may
#      sit on any other 8-neighbor of the anchor vertex, and it disappears in the output.
#    - D4/1 then changes how both trails are painted: instead of solid lines, the main
#      trail alternates [shape, marker, shape, marker, ...], while the offset shadow trail
#      alternates in opposite phase [marker, shape, marker, shape, ...].
#    - Small concrete example:
#        input:   A . . A        output:  A B A A
#                 . . . .                 B A B B
#                 . B . .                 . . . .
#      Here B is placed below-left of the start vertex, but B encodes "up". Original logic
#      would copy the shadow below-left and keep the cue pixel. This combo instead deletes
#      the cue, draws the shadow one row above, and uses A/B striping on both trajectories.
# b) Complexity calibration:
#    - To keep the rule inferable, every shape has exactly one marker and every grid has
#      2-3 shapes. Shape colors come from {3,4,6}; marker colors come from {1,2,7,8}.
#      This prevents shape colors from colliding with reusable direction colors.
#    - Segment lengths are sampled as even values so the alternating main trail ends on a
#      shape-colored endpoint, making the periodic rule visually clean instead of "almost"
#      alternating at the far vertex.
#    - Marker colors are unique within one grid, so a single example usually exhibits
#      multiple encoded directions without adding layered shadow complexity.
# c) Hidden assumptions audit for main():
#    - The original solver assumed every non-bg color could be treated as a shape/shadow
#      object. That breaks once marker colors are globally reusable, so marker colors must
#      be excluded from the shape-color loop.
#    - The original `carried.append((dr,dc,color))` assumed cue position equals shadow
#      offset. That no longer holds; we must keep both the cue geometry and the cue color,
#      then choose between them explicitly depending on the reasoning mode.
#    - The original solid drawing helper assumed one constant line color. The combo needs
#      step-indexed palette lookup because stripe phase matters.
#    - `out = g.copy()` originally preserved the cue pixel in the output. That is wrong for
#      encoded-direction cues, so the render pass first erases marker colors from the base.
#    - The start vertex must be stable because stripe phase is anchored there. Generator
#      therefore attaches the cue to the lexicographically smallest seed vertex.
# d) Placement legality:
#    - Local patches are generated input-first, but their output is always computed by the
#      combo solver. Placement uses the union footprint of input and output occupancy, not
#      output alone, so a cue pixel cannot overlap another object's invisible-output cell.
#    - Patch bboxes include shape vertices, the off-path cue pixel, and the actual shifted
#      shadow footprint, preventing clipping and ensuring `random_free_location` sees the
#      full required canvas area.
# e) Visual observability:
#    - Humans can read two repeated signals at once: the same marker color always produces
#      the same shadow displacement, and that same color also appears as the alternating
#      stripe partner on both the main and shadow trails.
#    - Because the cue is deliberately not placed on the real displacement side, geometric
#      proximity alone is insufficient; color must be interpreted as information.
# f) Visual cleanliness:
#    - Distractors are disabled. One cue per shape and at most three shapes keeps the grid
#      sparse enough that stripe pairs remain legible.
#    - Only cardinal encoded offsets are used. This reduces visual clutter relative to
#      eight possible directions while still being clearly different from the cue side.
# g) Color-change inferability:
#    - Marker colors are few, fixed, and reused across samples. Their role is reinforced by
#      both the displaced shadow direction and the stripe palette, so the color mapping is
#      structural rather than arbitrary.

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, get_bg as _bg_color, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

OFFSET_TO_COLOR = {
    (-1, 0): 2,
    (0, 1): 1,
    (1, 0): 7,
    (0, -1): 8,
}
COLOR_TO_OFFSET = {color: offset for offset, color in OFFSET_TO_COLOR.items()}
SHADOW_COLORS = sorted(COLOR_TO_OFFSET)


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


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


def _fill_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if int(out[rr, cc]) == int(bg):
            out[rr, cc] = int(color)


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
        if int(out[rr, cc]) == int(bg):
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


def _is_shadow_marker_color(col: int) -> bool:
    return int(col) in COLOR_TO_OFFSET


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
            if 0 <= rr < h and 0 <= cc < w and _is_shadow_marker_color(int(g[rr, cc])):
                seed_idxs.append(i)
                break

    if seed_idxs:
        start = min(seed_idxs, key=lambda idx: (cycle[idx][0], cycle[idx][1]))
    else:
        start = min(range(len(cycle)), key=lambda idx: (cycle[idx][0], cycle[idx][1]))

    cycle = cycle[start:] + cycle[:start]

    cues: list[tuple[int, int, int]] = []
    sr0, sc0 = cycle[0]
    seen = set()
    for drn, dcn in DIR8:
        rr = sr0 + drn
        cc = sc0 + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            key = (drn, dcn, v)
            if _is_shadow_marker_color(v) and key not in seen:
                cues.append(key)
                seen.add(key)

    edges = [(cycle[0], cycle[1])] if len(cycle) == 2 else [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), (r1, c1) in edges:
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={int(col)}):
            continue
        drawable_edges.append(((r0, c0), (r1, c1)))

    return cues, drawable_edges


def _render_output(grid: np.ndarray, *, periodic: bool, encoded_direction: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(_bg_color(g))

    out = g.copy()
    for shadow_color in SHADOW_COLORS:
        out[out == int(shadow_color)] = int(bg)

    shape_colors = sorted(set(int(v) for v in np.unique(g)) - {int(bg)} - set(SHADOW_COLORS))
    for col in shape_colors:
        plan = _shape_plan(g, int(col), int(bg))
        if plan is None:
            continue
        cues, drawable_edges = plan

        marker_palette = []
        seen_marker_colors = set()
        for _cue_dr, _cue_dc, marker_color in cues:
            if int(marker_color) not in seen_marker_colors:
                marker_palette.append(int(marker_color))
                seen_marker_colors.add(int(marker_color))
        main_palette = [int(col)] + marker_palette if marker_palette else [int(col)]

        for (r0, c0), (r1, c1) in drawable_edges:
            if periodic:
                _periodic_fill_bg_only(out, r0, c0, r1, c1, main_palette, int(bg))
            else:
                _fill_line_bg_only(out, r0, c0, r1, c1, int(col), int(bg))

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for cue_dr, cue_dc, marker_color in cues:
                off_dr, off_dc = COLOR_TO_OFFSET[int(marker_color)] if encoded_direction else (int(cue_dr), int(cue_dc))
                if drs * off_dc - dcs * off_dr == 0:
                    continue
                rr0 = r0 + off_dr
                cc0 = c0 + off_dc
                rr1 = r1 + off_dr
                cc1 = c1 + off_dc
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, int(bg), allow=set()):
                    continue
                if periodic:
                    _periodic_fill_bg_only(out, rr0, cc0, rr1, cc1, [int(marker_color), int(col)], int(bg))
                else:
                    _fill_line_bg_only(out, rr0, cc0, rr1, cc1, int(marker_color), int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color vertices as striped trails, where a neighboring marker color encodes the true shadow direction.

    Input structure:
    - Background is the most common color.
    - Shape vertices use non-marker colors and appear as 2+ points that can be ordered into
      H/V/45deg edges.
    - A single adjacent marker pixel in one of {1,2,7,8} sits near the start vertex of each
      shape. Its color encodes a cardinal shadow offset, but its physical position may differ.

    Rule:
    - Remove all marker pixels from the output seed.
    - For each shape, choose the seeded start vertex, connect drawable edges, and stripe the
      main trail with the periodic palette [shape_color, marker_color, ...].
    - Copy each edge in the marker color's encoded direction, not the cue's geometric side,
      and stripe that shadow trail with opposite phase [marker_color, shape_color, ...].

    Key insight:
    The cue color carries two linked pieces of information at once: it names the shadow
    direction globally and it becomes the alternating partner color on both rendered trails.
    """
    return _render_output(grid, periodic=True, encoded_direction=True)


# --------------- generator ---------------

BG = 0
SHAPE_COLORS = [3, 4, 6]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)   # 2-3 shapes
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

SHAPE_PLACEMENT_RETRIES = 40
LOCAL_GEOMETRY_RETRIES = 80
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
        choices = [gap for gap in range(LINE_MIN_GAP, gap_hi + 1) if gap % 2 == 0]
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
    return [(vertices[0], vertices[1])] if len(vertices) == 2 else [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    cue: tuple[int, int],
    actual_offset: tuple[int, int],
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices + [cue]
    coords.extend((r + actual_offset[0], c + actual_offset[1]) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
    combo_out = _render_output(g, periodic=True, encoded_direction=True)
    solid_out = _render_output(g, periodic=False, encoded_direction=True)
    geom_out = _render_output(g, periodic=True, encoded_direction=False)

    striped_groups = 0
    misdirected_markers = 0
    off_color_main_cells = 0
    unique_marker_colors: set[int] = set()

    for col in sorted(set(int(v) for v in np.unique(g)) - {int(bg)} - set(SHADOW_COLORS)):
        plan = _shape_plan(g, int(col), int(bg))
        if plan is None:
            continue
        cues, drawable_edges = plan
        if cues:
            striped_groups += 1
        for cue_dr, cue_dc, marker_color in cues:
            unique_marker_colors.add(int(marker_color))
            if (int(cue_dr), int(cue_dc)) != COLOR_TO_OFFSET[int(marker_color)]:
                misdirected_markers += 1
        for (r0, c0), (r1, c1) in drawable_edges:
            for rr, cc in _segment_cells(r0, c0, r1, c1):
                if int(g[rr, cc]) != int(bg):
                    continue
                if int(combo_out[rr, cc]) != int(col):
                    off_color_main_cells += 1

    return {
        "shape_groups": int(len(sorted(set(int(v) for v in np.unique(g)) - {int(bg)} - set(SHADOW_COLORS)))),
        "striped_groups": int(striped_groups),
        "misdirected_markers": int(misdirected_markers),
        "off_color_main_cells": int(off_color_main_cells),
        "unique_marker_colors": int(len(unique_marker_colors)),
        "stripe_delta_cells": int(np.count_nonzero(combo_out != solid_out)),
        "direction_delta_cells": int(np.count_nonzero(combo_out != geom_out)),
    }


def _make_shape_patch(shape_color: int, marker_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    actual_offset = COLOR_TO_OFFSET[int(marker_color)]
    for _ in range(LOCAL_GEOMETRY_RETRIES):
        vertices = _make_vertices()
        shape_cells = _shape_cells(vertices)
        anchor = min(vertices)
        shifted_shape = {(r + actual_offset[0], c + actual_offset[1]) for r, c in shape_cells}

        marker_dirs = list(DIR8)
        np.random.shuffle(marker_dirs)
        for cue_dir in marker_dirs:
            if tuple(cue_dir) == tuple(actual_offset):
                continue
            cue = (anchor[0] + cue_dir[0], anchor[1] + cue_dir[1])
            if cue in shape_cells or cue in shifted_shape:
                continue
            if sum(1 for vr, vc in vertices if max(abs(cue[0] - vr), abs(cue[1] - vc)) == 1) != 1:
                continue

            r0, c0, r1, c1 = _patch_bbox(shape_cells, vertices, cue, actual_offset)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            inp = np.full((h, w), BG, dtype=int)

            for vr, vc in vertices:
                inp[vr - r0, vc - c0] = int(shape_color)
            cue_rel = (cue[0] - r0, cue[1] - c0)
            if int(inp[cue_rel]) != int(BG):
                continue
            inp[cue_rel] = int(marker_color)

            out = main(inp)
            metrics = _combo_metrics(inp)
            if int(out[cue_rel]) != int(BG):
                continue
            if metrics["striped_groups"] != 1:
                continue
            if metrics["misdirected_markers"] != 1:
                continue
            if metrics["off_color_main_cells"] < 1:
                continue
            if metrics["stripe_delta_cells"] < 2:
                continue
            if metrics["direction_delta_cells"] < 2:
                continue
            return inp, out
    return None


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _occupancy_patch(patch_in: np.ndarray, patch_out: np.ndarray) -> np.ndarray:
    return np.where((patch_in != int(BG)) | (patch_out != int(BG)), 9, int(BG)).astype(int)


def _occupancy_canvas(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    return np.where((inp != int(BG)) | (out != int(BG)), 9, int(BG)).astype(int)


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    placement_canvas = _occupancy_canvas(inp, out)
    placement_patch = _occupancy_patch(patch_in, patch_out)
    try:
        loc = random_free_location(
            placement_canvas,
            placement_patch,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    shape_colors = list(np.random.permutation(SHAPE_COLORS)[:shape_count])
    marker_colors = list(np.random.permutation(SHADOW_COLORS)[:shape_count])

    for shape_color, marker_color in zip(shape_colors, marker_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), int(marker_color))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if int(_bg_color(inp)) != int(BG) or int(_bg_color(out)) != int(BG):
        return None
    if len(colors_present(inp, exclude=[BG])) != shape_count + shape_count:
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _combo_metrics(inp)
    if metrics["shape_groups"] != shape_count:
        return None
    if metrics["striped_groups"] != shape_count:
        return None
    if metrics["misdirected_markers"] != shape_count:
        return None
    if metrics["unique_marker_colors"] < min(2, shape_count):
        return None
    if metrics["off_color_main_cells"] < shape_count:
        return None
    if metrics["stripe_delta_cells"] < 2 * shape_count:
        return None
    if metrics["direction_delta_cells"] < 2 * shape_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
