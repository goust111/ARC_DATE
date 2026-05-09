# Combo variant of 136b0064 (index 29)
# combo_spec: D3/1+D5/1+D13/0
# Name: 形状解码独立于颜色系统

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve()
_SOLVER_DIR = None
for _parent in _ROOT.parents:
    candidate = _parent / "solvers" / "arc_utils.py"
    if candidate.exists():
        _SOLVER_DIR = candidate.parent
        break
if _SOLVER_DIR is None:
    raise RuntimeError("Could not locate solvers/arc_utils.py from combo variant file")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg


# Design analysis for combo D3/1+D5/1+D13/0
# a) Concretization and behavior trace:
#    - Layout stays the same as the base task: a 7-column left canvas with two tile columns of
#      monochrome 3x3 glyphs, a full-height divider at column 7, and one gray start pixel on the
#      right. The four glyph masks are unchanged, so area 4/5/6/7 still means down-2 / right-3 /
#      left-4 / left-2 respectively.
#    - Fused change 1 (D3): glyph color is now distractor data. A cross can be orange, teal, or
#      maroon and still mean "go left 4"; therefore the solver must decode motion from shape only.
#    - Fused change 2 (D5): segment body colors ignore glyph colors and instead follow a fixed
#      cycle by read order. With palette (2,8,6), the 1st/2nd/3rd/4th decoded segments use
#      red/teal/pink/red regardless of which glyph colors were seen in the input.
#    - Fused change 3 (D13): after stepping down one row for the next instruction, that first
#      contact cell is always gray. The remaining cells of the segment body use the cycle color.
#    - Tiny trace: suppose three glyphs decode to areas [5,4,6]. Base task would paint
#      [glyph-color, glyph-color, glyph-color]. The fused task instead paints:
#      step 1 -> gray joint then red body to the right,
#      step 2 -> gray joint then teal body downward,
#      step 3 -> gray joint then pink body to the left.
#      Input glyph colors can vary arbitrarily and do not affect any output pixel.
# b) Complexity calibration:
#    - Keep the original four masks, fixed tile lattice, and 7-column walk area so the puzzle
#      remains recognizably the same task family.
#    - Use a short cycle of three body colors. This is long enough to show repetition inside
#      4-10 segments, but small enough for a human to infer from 2-4 examples.
#    - Glyph colors are sampled from a restrained palette {1,3,7,9} that is disjoint from the
#      output cycle {2,8,6}; the generator forces repeated-shape/different-color and
#      repeated-color/different-shape evidence so "color is decoy" is visible rather than
#      theoretical.
# c) Hidden-assumption review of the original main():
#    - Original assumption "glyph color == segment color" is invalid and must be removed.
#      Segment paint now depends only on index in the decoded sequence.
#    - Original color-based ignoring of left-side components is no longer safe because glyph colors
#      are independent. The solver must treat every 3x3 monochrome component on the left as a
#      glyph candidate, regardless of its color.
#    - Original assumption that the first pixel of each new segment has the segment color is also
#      invalid. That cell is now a fixed gray joint, while the body keeps the cycle color.
#    - The geometric assumptions still hold: column-major order, unique divider, unique right-side
#      gray start, and the area-to-motion mapping are unchanged.
# d) Placement legality:
#    - Tile slots remain at c0 in {0,4} and r0 in {0,4,8,...}; 3x3 glyphs cannot overlap.
#    - Path sampling only accepts moves whose final endpoint stays within columns 0..6 and below
#      row H-1. Since rows strictly increase, later segments do not collide with earlier ones.
#    - Glyph colors exclude 0/4/5 so the background, divider, and right-side start marker remain
#      visually distinct and the input never contains extra gray cells.
# e) Visual observability:
#    - The generator forces two independent clues in the input: at least one repeated shape uses
#      two different glyph colors, and at least one glyph color is shared by different shapes.
#    - It also forces two independent clues in the output logic: the same shape must appear in
#      different cycle slots, and the same cycle slot must cover different shapes. This makes the
#      cycle-vs-shape independence visible on the canvas, not just in code.
# f) Visual cleanliness:
#    - The output remains a single monotone-downward polyline with one gray joint per segment, so
#      the gray connectors and colored bodies are easy to separate by eye.
#    - Avoiding glyph gray/yellow colors prevents the input from visually blending with the gray
#      start/joints or yellow divider, and it also cleanly separates input colors from the output
#      cycle colors.
# g) Change inferability:
#    - The body-color rule is a fixed 3-cycle, not a large arbitrary mapping, and every generated
#      sample includes at least one repeated cycle color landing on different motions.
#    - The input-side decoy colors repeat in a structured way, so a human can notice that shape
#      repetition predicts movement while color repetition does not predict paint.
# Solver/generator contract after the fusion:
#    - main() assumes the left side contains exactly the valid 3x3 glyph components on background.
#    - generate_input() guarantees the path is in-bounds, the cycle rule is observable, glyph
#      colors visibly break the old binding, and the fused output differs from simpler baselines
#      such as "paint by glyph color" or "cycle colors without gray joints".


N_MIN, N_MAX = 2, 5
GRID_W = 15
DIVIDER_COL = 7
DIVIDER_COLOR = 4
LEFT_W = 7
TILE_COLS = (0, 4)
TILE_ROW_STRIDE = 4
GRAY = 5
GRAY_START_ROW = 0
MAX_ATTEMPTS = 600
PATH_ATTEMPTS = 120
COLOR_ATTEMPTS = 240
MIN_DIRECTION_CHANGES = 2

_TEMPLATES = {
    4: np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    5: np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
    6: np.array([[1, 1, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    7: np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=int),
}
_AREA_TO_LEN = {4: 2, 5: 3, 6: 4, 7: 2}
_AREA_TO_DIR = {5: 1, 6: -1, 7: -1}
_AREA_DELTA = {4: (2, 0), 5: (1, 2), 6: (1, -3), 7: (1, -1)}
_AREAS = (4, 5, 6, 7)
_OLD_AREA_TO_COLOR = {4: 6, 5: 1, 6: 3, 7: 2}
_GLYPH_COLORS = (1, 3, 7, 9)
_SEGMENT_PALETTE = (2, 8, 6)


def find_full_height_constant_col(grid: np.ndarray, *, bg: int | None = None) -> tuple[int | None, int | None]:
    """Find a full-height constant non-background column."""
    _H, W = grid.shape
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    for c in range(int(W)):
        col = grid[:, int(c)]
        if (col == col[0]).all() and int(col[0]) != bg:
            return int(c), int(col[0])
    return None, None


def _collect_glyphs(left: np.ndarray, *, bg: int) -> list[dict[str, int]]:
    glyphs: list[dict[str, int]] = []
    for comp in find_components(left, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != 3 or bbox_w(b) != 3:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        glyphs.append(
            {
                "color": int(comp["color"]),
                "r0": int(r0),
                "c0": int(c0),
                "area": int(comp["area"]),
            }
        )
    glyphs.sort(key=lambda d: (int(d["c0"]), int(d["r0"])))
    return glyphs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode tiled glyph shapes into a polyline whose paint system is independent from the glyph colors.

    Input structure:
    - A full-height divider splits the board into a left glyph panel and a right start-marker panel.
    - The left panel contains monochromatic 3x3 glyphs on background; the right panel contains one gray (5) start cell.

    Rule:
    - Remove the divider, shift the gray start into the left canvas, and read the glyphs in column-major order.
    - Each glyph's 3x3 shape still determines the motion: area 4 → down 2, area 5 → right 3, area 6 → left 4, area 7 → left 2.
    - For every decoded step, first place a gray joint one row below the current endpoint, then draw the rest of that segment with the next color from a fixed repeating palette.

    Key insight:
    - Glyph color is now decoy information. Shape alone chooses the path, segment index chooses the body color, and the one-cell joints are always gray.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape
    bg = int(get_bg(g))

    div_c, _div_color = find_full_height_constant_col(g, bg=bg)
    assert div_c is not None

    out = np.zeros((H, int(div_c)), dtype=int)

    gray = np.argwhere(g == GRAY)
    if gray.size == 0:
        return out
    r_cur, c_cur = map(int, gray[0])
    c_cur = int(c_cur - (int(div_c) + 1))
    if not (0 <= c_cur < int(div_c)):
        return out
    out[int(r_cur), int(c_cur)] = GRAY

    left = g[:, : int(div_c)]
    glyphs = _collect_glyphs(left, bg=bg)

    for idx, glyph in enumerate(glyphs):
        area = int(glyph["area"])
        assert area in _AREA_TO_LEN, f"unexpected glyph area: {area}"
        seg_len = int(_AREA_TO_LEN[area])
        body_color = int(_SEGMENT_PALETTE[idx % len(_SEGMENT_PALETTE)])

        r_cur = int(r_cur + 1)
        assert r_cur < H, f"segment exits bottom: r_cur={r_cur}, H={H}"
        out[r_cur, int(c_cur)] = GRAY

        if area == 4:
            end_r = r_cur + (seg_len - 1)
            assert end_r < H, f"vertical segment exits bottom: end_r={end_r}, H={H}"
            for _ in range(seg_len - 1):
                r_cur = int(r_cur + 1)
                out[r_cur, int(c_cur)] = body_color
        else:
            dir_h = int(_AREA_TO_DIR[area])
            end_c = int(c_cur) + dir_h * (seg_len - 1)
            assert 0 <= end_c < int(div_c), (
                f"horizontal segment exits canvas: c_cur={c_cur}, dir={dir_h}, seg_len={seg_len}, div_c={div_c}"
            )
            for _ in range(seg_len - 1):
                c_cur = int(c_cur + dir_h)
                out[r_cur, c_cur] = body_color

    return out


def _direction_change_count(areas: list[int]) -> int:
    kinds = [a == 4 for a in areas]
    return sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])


def _cycle_is_observable(areas: list[int]) -> bool:
    palette_len = len(_SEGMENT_PALETTE)
    same_shape_diff_cycle = any(
        len({idx % palette_len for idx, area2 in enumerate(areas) if area2 == area}) >= 2
        for area in set(areas)
    )
    same_cycle_diff_shape = any(
        len({areas[idx] for idx in range(len(areas)) if idx % palette_len == slot}) >= 2
        for slot in range(palette_len)
    )
    return same_shape_diff_cycle and same_cycle_diff_shape


def _path_branch_coverage_ok(areas: list[int]) -> bool:
    unique_shapes = len(set(areas))
    if unique_shapes < min(3, len(areas)):
        return False
    if len(areas) >= 6 and 6 not in areas:
        return False
    return True


def _choose_area(opts: list[int], seen_counts: dict[int, int]) -> int:
    weights: list[float] = []
    for area in opts:
        count = seen_counts.get(area, 0)
        weight = 1.0 / (1 + count)
        if count == 0:
            weight += 1.5
        if area == 6 and count == 0:
            weight += 1.0
        weights.append(weight)
    return int(random.choices(opts, weights=weights, k=1)[0])


def _sample_path(N: int, start_col: int) -> list[int] | None:
    """Sample a valid area sequence whose geometry and output-cycle evidence are both clear."""
    H = 4 * N - 1
    for _ in range(PATH_ATTEMPTS):
        r, c = GRAY_START_ROW, start_col
        areas: list[int] = []
        seen_counts: dict[int, int] = {}
        for _step in range(2 * N):
            opts = [a for a in _AREAS if r + _AREA_DELTA[a][0] < H and 0 <= c + _AREA_DELTA[a][1] < LEFT_W]
            if not opts:
                break
            area = _choose_area(opts, seen_counts)
            areas.append(area)
            seen_counts[area] = seen_counts.get(area, 0) + 1
            dr, dc = _AREA_DELTA[area]
            r += dr
            c += dc

        if len(areas) != 2 * N:
            continue
        if not _path_branch_coverage_ok(areas):
            continue
        if _direction_change_count(areas) < MIN_DIRECTION_CHANGES:
            continue
        if len(set(areas)) < 2:
            continue
        if not any(areas.count(area) >= 2 for area in set(areas)):
            continue
        if not _cycle_is_observable(areas):
            continue
        return areas
    return None


def _glyph_colors_informative(areas: list[int], glyph_colors: list[int]) -> bool:
    if len(set(glyph_colors)) < 2:
        return False

    same_shape_diff_glyph_color = any(
        len({glyph_colors[idx] for idx, area2 in enumerate(areas) if area2 == area}) >= 2 for area in set(areas)
    )
    if not same_shape_diff_glyph_color:
        return False

    same_glyph_color_diff_shape = any(
        len({areas[idx] for idx, color2 in enumerate(glyph_colors) if color2 == color}) >= 2
        for color in set(glyph_colors)
    )
    if not same_glyph_color_diff_shape:
        return False

    canonical_breaks = sum(color != _OLD_AREA_TO_COLOR[area] for area, color in zip(areas, glyph_colors))
    if canonical_breaks < max(2, len(areas) // 2):
        return False

    cycle_mismatches = sum(
        color != _SEGMENT_PALETTE[idx % len(_SEGMENT_PALETTE)] for idx, color in enumerate(glyph_colors)
    )
    return cycle_mismatches >= max(2, len(areas) // 2)


def _sample_glyph_colors(areas: list[int]) -> list[int] | None:
    groups: dict[int, list[int]] = {}
    for idx, area in enumerate(areas):
        groups.setdefault(area, []).append(idx)

    repeated_areas = [area for area, idxs in groups.items() if len(idxs) >= 2]
    if not repeated_areas or len(groups) < 2:
        return None

    for _ in range(COLOR_ATTEMPTS):
        repeated_area = random.choice(repeated_areas)
        repeat_i0, repeat_i1 = random.sample(groups[repeated_area], 2)
        other_area = random.choice([area for area in groups if area != repeated_area])
        other_idx = random.choice(groups[other_area])

        shared_color = random.choice(_GLYPH_COLORS)
        alternate_color = random.choice([c for c in _GLYPH_COLORS if c != shared_color])

        palette_size = random.randint(2, min(4, len(_GLYPH_COLORS)))
        palette = set(random.sample(_GLYPH_COLORS, k=palette_size))
        palette.update({shared_color, alternate_color})
        palette_list = sorted(palette)

        colors: list[int | None] = [None] * len(areas)
        colors[repeat_i0] = shared_color
        colors[repeat_i1] = alternate_color
        colors[other_idx] = shared_color

        for idx in range(len(areas)):
            if colors[idx] is None:
                colors[idx] = random.choice(palette_list)

        final_colors = [int(c) for c in colors]
        if _glyph_colors_informative(areas, final_colors):
            return final_colors
    return None


def _simulate_output(
    H: int,
    start_col: int,
    areas: list[int],
    glyph_colors: list[int],
    *,
    body_mode: str,
    gray_joints: bool,
) -> np.ndarray:
    out = np.zeros((H, LEFT_W), dtype=int)
    out[GRAY_START_ROW, start_col] = GRAY

    r_cur, c_cur = GRAY_START_ROW, start_col
    for idx, (area, glyph_color) in enumerate(zip(areas, glyph_colors)):
        if body_mode == "glyph":
            body_color = int(glyph_color)
        elif body_mode == "cycle":
            body_color = int(_SEGMENT_PALETTE[idx % len(_SEGMENT_PALETTE)])
        elif body_mode == "canonical":
            body_color = int(_OLD_AREA_TO_COLOR[area])
        else:
            raise ValueError(f"unknown body_mode: {body_mode}")

        r_cur += 1
        out[r_cur, c_cur] = GRAY if gray_joints else body_color

        if area == 4:
            for _ in range(_AREA_TO_LEN[area] - 1):
                r_cur += 1
                out[r_cur, c_cur] = body_color
        else:
            dir_h = _AREA_TO_DIR[area]
            for _ in range(_AREA_TO_LEN[area] - 1):
                c_cur += dir_h
                out[r_cur, c_cur] = body_color
    return out


def _behavior_metrics(H: int, start_col: int, areas: list[int], glyph_colors: list[int]) -> dict[str, int]:
    expected = _simulate_output(H, start_col, areas, glyph_colors, body_mode="cycle", gray_joints=True)
    glyph_gray = _simulate_output(H, start_col, areas, glyph_colors, body_mode="glyph", gray_joints=True)
    glyph_plain = _simulate_output(H, start_col, areas, glyph_colors, body_mode="glyph", gray_joints=False)
    cycle_plain = _simulate_output(H, start_col, areas, glyph_colors, body_mode="cycle", gray_joints=False)
    canonical_plain = _simulate_output(H, start_col, areas, glyph_colors, body_mode="canonical", gray_joints=False)

    return {
        "same_shape_diff_glyph_color": int(
            any(len({glyph_colors[idx] for idx, area2 in enumerate(areas) if area2 == area}) >= 2 for area in set(areas))
        ),
        "same_glyph_color_diff_shape": int(
            any(len({areas[idx] for idx, color2 in enumerate(glyph_colors) if color2 == color}) >= 2 for color in set(glyph_colors))
        ),
        "same_shape_diff_cycle_slot": int(
            any(
                len({idx % len(_SEGMENT_PALETTE) for idx, area2 in enumerate(areas) if area2 == area}) >= 2
                for area in set(areas)
            )
        ),
        "same_cycle_slot_diff_shape": int(
            any(
                len({areas[idx] for idx in range(len(areas)) if idx % len(_SEGMENT_PALETTE) == slot}) >= 2
                for slot in range(len(_SEGMENT_PALETTE))
            )
        ),
        "glyph_cycle_mismatches": int(
            sum(color != _SEGMENT_PALETTE[idx % len(_SEGMENT_PALETTE)] for idx, color in enumerate(glyph_colors))
        ),
        "canonical_breaks": int(sum(color != _OLD_AREA_TO_COLOR[area] for area, color in zip(areas, glyph_colors))),
        "diff_from_glyph_gray": int(not np.array_equal(expected, glyph_gray)),
        "diff_from_glyph_plain": int(not np.array_equal(expected, glyph_plain)),
        "diff_from_cycle_plain": int(not np.array_equal(expected, cycle_plain)),
        "diff_from_canonical_plain": int(not np.array_equal(expected, canonical_plain)),
    }


def _render_grid(N: int, start_col: int, areas: list[int], glyph_colors: list[int]) -> np.ndarray:
    H = 4 * N - 1
    grid = np.zeros((H, GRID_W), dtype=int)
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    grid[GRAY_START_ROW, DIVIDER_COL + 1 + start_col] = GRAY

    for i, (area, color) in enumerate(zip(areas, glyph_colors)):
        tile_col = TILE_COLS[i // N]
        r0 = TILE_ROW_STRIDE * (i % N)
        grid[r0 : r0 + 3, tile_col : tile_col + 3] = _TEMPLATES[area] * int(color)
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    N = random.randint(N_MIN, N_MAX)
    H = 4 * N - 1
    start_col = random.randint(0, LEFT_W - 1)

    areas = _sample_path(N, start_col)
    if areas is None:
        return None

    glyph_colors = _sample_glyph_colors(areas)
    if glyph_colors is None:
        return None

    metrics = _behavior_metrics(H, start_col, areas, glyph_colors)
    if metrics["same_shape_diff_glyph_color"] != 1:
        return None
    if metrics["same_glyph_color_diff_shape"] != 1:
        return None
    if metrics["same_shape_diff_cycle_slot"] != 1:
        return None
    if metrics["same_cycle_slot_diff_shape"] != 1:
        return None
    if metrics["glyph_cycle_mismatches"] < max(2, len(areas) // 2):
        return None
    if metrics["canonical_breaks"] < max(2, len(areas) // 2):
        return None
    if metrics["diff_from_glyph_gray"] != 1:
        return None
    if metrics["diff_from_glyph_plain"] != 1:
        return None
    if metrics["diff_from_cycle_plain"] != 1:
        return None
    if metrics["diff_from_canonical_plain"] != 1:
        return None

    inp = _render_grid(N, start_col, areas, glyph_colors)
    expected = _simulate_output(H, start_col, areas, glyph_colors, body_mode="cycle", gray_joints=True)
    return inp, expected, metrics


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            attempt = _gen_attempt()
            if attempt is None:
                continue
            inp, expected, _metrics = attempt
            out = main(inp)
            if out.shape != expected.shape:
                continue
            if not np.array_equal(out, expected):
                continue
            return inp
        except Exception:
            continue
    raise RuntimeError(f"generate_input: failed after {MAX_ATTEMPTS} attempts")
