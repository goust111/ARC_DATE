# Combo variant of 31f7f899 (index 9)
# combo_spec: D9/3+D11/0
# Name: 干扰色块+图例指示方向

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Base task: the scene has a full-width baseline plus several vertical solid
#      bars centered on that baseline. The solver sorts the bar heights and reassigns
#      the sorted silhouettes into the original left-to-right colored slots.
#    - Combo change 1 (D9/3): add small monochrome distractor blocks that are not part
#      of the sorting. They keep their positions from input to output.
#    - Combo change 2 (D11/0): add a tiny 3x3 corner legend whose staircase direction
#      tells whether the bars should be sorted ascending or descending.
#    - Small fused trace:
#        input bars at slot colors [red, green, yellow] have heights [5, 1, 3];
#        distractors are a 2x2 block near the top-right and a 1x2 block at the
#        bottom-left; the legend is the descending staircase.
#        Original solver would incorrectly treat every non-base component as sortable,
#        so the legend and distractors would contaminate the slot list and shape list.
#        Fused solver instead does three steps:
#          1. detect the top-left staircase and decode direction = descending,
#          2. keep only baseline-crossing solid centered rectangles as true bars,
#          3. sort bar heights to [5, 3, 1] and place them into the original slots,
#             while preserving the legend and distractors unchanged.
#      The difference from the source task is therefore not one isolated tweak: the
#      legend and distractors both change how components must be parsed before sorting.
# b) Complexity calibration
#    - The legend has exactly two states (ascending / descending), so the control
#      variable is visually clear rather than combinatorially large.
#    - Distractors use only tiny rectangles (1x2, 2x1, 2x2), so the extra reasoning is
#      "filter them out" rather than "solve a second object puzzle".
#    - Real bars still have distinct odd heights and widths in {1, 3}, which keeps the
#      bar ranking stable and visually obvious.
#    - Every generated sample contains both a legend and at least two distractors, so
#      the fusion is visible in each example instead of being a rare corner case.
# c) Implicit-assumption audit of the original main()
#    - Base-row detection via the most frequent non-background color on a row still
#      works because the baseline spans the full width while the legend/distractors are
#      small. The generator guarantees that.
#    - "Every non-bg non-baseline component is a sortable bar" is broken twice:
#      the legend is also such a component, and the distractors are too.
#    - "Non-touching component => ignore or instruction" is also insufficient in the
#      fused setting because both the legend and distractors do not touch the baseline.
#      They must be separated by a dedicated legend-shape predicate.
#    - The hard-coded ascending sort direction is broken by the legend.
#    - Reconstructing only bg + baseline + sorted bars would erase both the legend and
#      distractors, so the output pass must copy them through explicitly.
# d) Placement legality
#    - The generator reserves the top-left 3x3 legend area plus one spacer column. Bars
#      are placed only to the right of that reserve.
#    - Bar heights are capped to leave at least two-row top and bottom margins for
#      distractors. Distractors are placed only inside those margins, so they never
#      overlap either the input bars or the redistributed output bars.
#    - Distractor colors exclude the background, baseline, legend, and bar colors, so
#      color identity cannot accidentally merge object families.
# e) Visual observability
#    - Across 2-4 examples, a human can see three stable facts:
#      1. the corner staircase persists and flips orientation across some examples,
#      2. the small floating blocks never move,
#      3. only the baseline-crossing bars get permuted, and their permutation follows
#         the staircase direction.
#      That is enough support to infer the fused rule from pixels alone.
# f) New solver assumptions after the modification
#    - `_is_sortable_bar()` assumes true bars are solid rectangles, touch the baseline,
#      and are vertically centered on it. The generator constructs bars exactly this
#      way, and keeps distractors away from the baseline so they cannot satisfy it.
#    - `_legend_direction()` assumes the legend is the unique 3x3 staircase at the
#      fixed top-left corner. The generator reserves that area and excludes the legend
#      color from every other object family.
#    - Sorting assumes the filtered bar list is non-empty and uses distinct heights.
#      The generator always creates 3-5 bars with distinct odd heights.
#    - Reconstruction assumes preserved extras do not overlap bar slots. The generator
#      enforces vertical separation via the top/bottom margin placement.

BAR_COLORS = [1, 2, 3, 4, 5, 7, 9]
DISTRACTOR_SHAPES = [(1, 2), (2, 1), (2, 2)]

LEGEND_COLOR = 0
LEGEND_ASC = np.array(
    [
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=int,
)
LEGEND_DESC = np.fliplr(LEGEND_ASC)
LEGEND_TOP = 0
LEGEND_LEFT = 0
LEGEND_H, LEGEND_W = map(int, LEGEND_ASC.shape)
LEGEND_RESERVED_COLS = LEGEND_W + 1


def _base_info(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    for r in range(g.shape[0]):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    return int(best[1]), int(best[2])


def _component_patch(comp: dict[str, object]) -> np.ndarray:
    r0, c0, r1, c1 = [int(x) for x in comp["bbox"]]
    patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - int(r0), int(c) - int(c0)] = 1
    return patch


def _legend_direction(comp: dict[str, object]) -> int | None:
    r0, c0, _r1, _c1 = [int(x) for x in comp["bbox"]]
    if (int(r0), int(c0)) != (int(LEGEND_TOP), int(LEGEND_LEFT)):
        return None
    patch = _component_patch(comp)
    if np.array_equal(patch, LEGEND_ASC):
        return 1
    if np.array_equal(patch, LEGEND_DESC):
        return -1
    return None


def _is_sortable_bar(comp: dict[str, object], base_r: int) -> bool:
    pts = np.array(comp["coords"], dtype=int)
    r0, c0, r1, c1 = [int(x) for x in comp["bbox"]]
    height = int(bbox_h((r0, c0, r1, c1)))
    width = int(bbox_w((r0, c0, r1, c1)))
    area = int(comp["area"])

    if not np.any(pts[:, 0] == int(base_r)):
        return False
    if r0 + r1 != 2 * int(base_r):
        return False
    if area != height * width:
        return False

    base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
    return base_cols == list(range(int(c0), int(c1) + 1))


def _object_from_component(comp: dict[str, object], base_r: int) -> dict[str, object]:
    r0, c0, r1, c1 = [int(x) for x in comp["bbox"]]
    pts = [(int(r), int(c)) for r, c in comp["coords"]]
    base_cols = sorted(c for r, c in pts if int(r) == int(base_r))
    if base_cols:
        anchor = int(base_cols[len(base_cols) // 2])
    else:
        anchor = int((int(c0) + int(c1)) // 2)
    rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
    return {
        "color": int(comp["color"]),
        "anchor": int(anchor),
        "height": int(bbox_h((r0, c0, r1, c1))),
        "size": int(len(rel)),
        "rel": rel,
    }


def _extract_scene(
    g: np.ndarray, bg: int, base_r: int, base_col: int
) -> tuple[int, dict[str, object] | None, list[dict[str, object]], list[dict[str, object]]]:
    direction = 1
    legend: dict[str, object] | None = None
    bars: list[dict[str, object]] = []
    extras: list[dict[str, object]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        legend_dir = _legend_direction(comp)
        if legend_dir is not None:
            if legend is None:
                legend = {"color": int(col), "coords": np.array(comp["coords"], dtype=int)}
                direction = int(legend_dir)
            continue

        if _is_sortable_bar(comp, base_r):
            bars.append(_object_from_component(comp, base_r))
        else:
            extras.append({"color": int(col), "coords": np.array(comp["coords"], dtype=int)})

    return int(direction), legend, bars, extras


def _render_output(
    shape: tuple[int, int],
    bg: int,
    base_r: int,
    base_col: int,
    legend: dict[str, object] | None,
    extras: list[dict[str, object]],
    bars: list[dict[str, object]],
    direction: int,
) -> np.ndarray:
    H, W = map(int, shape)
    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = int(base_col)

    if legend is not None:
        out[tuple(np.array(legend["coords"], dtype=int).T)] = int(legend["color"])
    for comp in extras:
        out[tuple(np.array(comp["coords"], dtype=int).T)] = int(comp["color"])

    slots = sorted(bars, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(bars, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))
    if int(direction) < 0:
        shapes = list(reversed(shapes))

    for slot, shape_obj in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape_obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - The top-left staircase legend says whether bar heights should be assigned to
      slots ascending or descending.
    - Only baseline-crossing solid centered rectangles are sortable bars.
    - Floating non-bar components are distractors and must be preserved in place.
    - Slots stay fixed left-to-right and keep their original colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _base_info(g, bg)
    direction, legend, bars, extras = _extract_scene(g, bg, base_r, base_col)
    assert len(bars) >= 1
    return _render_output(g.shape, bg, base_r, base_col, legend, extras, bars, direction)


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(span: int, k: int, width: int, offset: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(span - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(offset + gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _legend_patch(direction: int, bg: int) -> np.ndarray:
    shape = LEGEND_ASC if int(direction) > 0 else LEGEND_DESC
    return np.where(shape == 1, int(LEGEND_COLOR), int(bg)).astype(int)


def _draw_bars(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[int(base_r), :] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _place_distractors(n: int, base_r: int, max_height: int, colors: list[int]) -> list[dict[str, int]]:
    top_h = int(base_r - max_height // 2)
    bottom_start = int(base_r + max_height // 2 + 1)
    bottom_h = int(n - bottom_start)
    occupancy = np.zeros((n, n), dtype=bool)
    occupancy[int(LEGEND_TOP) : int(LEGEND_TOP + LEGEND_H), int(LEGEND_LEFT) : int(LEGEND_RESERVED_COLS)] = True
    distractors: list[dict[str, int]] = []

    for color in colors:
        placed = False
        for _ in range(100):
            h, w = DISTRACTOR_SHAPES[int(np.random.randint(len(DISTRACTOR_SHAPES)))]
            bands: list[tuple[int, int]] = []
            if h <= top_h:
                bands.append((0, top_h))
            if h <= bottom_h:
                bands.append((bottom_start, n))
            if not bands:
                break
            band_r0, band_r1 = bands[int(np.random.randint(len(bands)))]
            r = int(np.random.randint(band_r0, band_r1 - h + 1))
            c = int(np.random.randint(0, n - w + 1))
            if occupancy[r : r + h, c : c + w].any():
                continue
            occupancy[r : r + h, c : c + w] = True
            distractors.append({"color": int(color), "r": int(r), "c": int(c), "h": int(h), "w": int(w)})
            placed = True
            break
        if not placed:
            raise ValueError("failed to place distractor")

    return distractors


def _add_legend_and_distractors(
    grid: np.ndarray, bg: int, direction: int, distractors: list[dict[str, int]]
) -> np.ndarray:
    out = np.array(grid, copy=True)
    paste(out, _legend_patch(direction, bg), int(LEGEND_TOP), int(LEGEND_LEFT), transparent=int(bg))
    for d in distractors:
        patch = np.full((int(d["h"]), int(d["w"])), int(d["color"]), dtype=int)
        paste(out, patch, int(d["r"]), int(d["c"]))
    return out


def _ascending_filtered_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _base_info(g, bg)
    _direction, legend, bars, extras = _extract_scene(g, bg, base_r, base_col)
    return _render_output(g.shape, bg, base_r, base_col, legend, extras, bars, 1)


def _naive_no_filter_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _base_info(g, bg)

    direction = 1
    legend: dict[str, object] | None = None
    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue
        legend_dir = _legend_direction(comp)
        if legend_dir is not None:
            if legend is None:
                legend = {"color": int(col), "coords": np.array(comp["coords"], dtype=int)}
                direction = int(legend_dir)
            continue
        objects.append(_object_from_component(comp, base_r))

    return _render_output(g.shape, bg, base_r, base_col, legend, [], objects, direction)


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _base_info(g, bg)
    direction, _legend, bars, extras = _extract_scene(g, bg, base_r, base_col)
    fused = main(g)
    ascending_only = _ascending_filtered_output(g)
    no_filter = _naive_no_filter_output(g)
    return {
        "direction": int(direction),
        "bar_count": int(len(bars)),
        "distractor_count": int(len(extras)),
        "direction_changes_output": int(direction < 0 and not np.array_equal(fused, ascending_only)),
        "distractors_change_output": int(not np.array_equal(fused, no_filter)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6
    direction = int(np.random.choice([1, -1]))

    usable_w = int(n - LEGEND_RESERVED_COLS)
    height_choices = list(range(1, min(n - 4, 9) + 1, 2))
    max_count = min(5, len(height_choices), (usable_w + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if usable_w >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(usable_w, k, width, LEGEND_RESERVED_COLS)
    colors = [int(c) for c in np.random.choice(BAR_COLORS, size=k, replace=False)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    target_heights = sorted_heights if direction > 0 else list(reversed(sorted_heights))

    distractor_pool = [int(c) for c in range(10) if c not in {bg, base_color, LEGEND_COLOR, *colors}]
    max_distractors = min(3, len(distractor_pool))
    if max_distractors < 2:
        return None
    dcount = int(np.random.randint(2, max_distractors + 1))
    distractor_colors = [int(c) for c in np.random.choice(distractor_pool, size=dcount, replace=False)]
    distractors = _place_distractors(n, base_r, max(sorted_heights), distractor_colors)

    out = _add_legend_and_distractors(
        _draw_bars(n, base_r, base_color, bg, lefts, width, colors, target_heights),
        bg,
        direction,
        distractors,
    )

    for _ in range(30):
        perm = np.random.permutation(k)
        in_heights = [int(sorted_heights[i]) for i in perm]
        if in_heights != target_heights:
            break
    else:
        return None

    inp = _add_legend_and_distractors(
        _draw_bars(n, base_r, base_color, bg, lefts, width, colors, in_heights),
        bg,
        direction,
        distractors,
    )

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _behavior_metrics(inp)
    if metrics["distractor_count"] < 2:
        return None
    if metrics["distractors_change_output"] == 0:
        return None
    if direction < 0 and metrics["direction_changes_output"] == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
