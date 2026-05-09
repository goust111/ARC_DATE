# Combo variant of 8f215267 (index 27)
# combo_spec: D4/0+D9/0+D16/0
# Name: 面积计数+多行排列+更多框架

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for combo D4/0 + D9/0 + D16/0
# ------------------------------------------------
# a) Concrete behavior and trace
#    - The global scene keeps the original left/right split, but is enlarged:
#      five or six 5x11 hollow frames are stacked on the left, and a wider right-side region holds
#      many small monochrome sprites.
#    - D4 changes the counted quantity. The output no longer depends on "how many same-color
#      sprites exist", but on "how much same-color area exists". To keep the tally readable,
#      countable sprites are restricted to even areas 2, 4, and 6, and each 2 cells contribute
#      one tally unit. So a domino adds 1 dot, a 2x2 block adds 2 dots, and a 2x3 block adds 3.
#    - D16 changes how large tallies are rendered. The first four dots fill the center row from
#      right to left, then the row above, then the row below, for a fixed capacity of 12 cells.
#    - D9 increases the number of frames to 5-6, so the solver must repeat the same color-match
#      reasoning more times and the generator needs a taller grid.
#    - Trace example:
#      * blue frame gets sprites with areas [6, 4, 6] -> total area 16 -> 8 dots.
#      * green frame gets sprites with areas [2, 2, 2] -> total area 6 -> 3 dots.
#      * Under the original task, both frames would be counted by object number (3 and 3).
#      * In this combo, the blue frame fills 4 center-row dots plus 4 upper-row dots, while the
#        green frame shows only 3 center-row dots. This exposes both the area rule and multi-row
#        rendering at once.
#
# b) Complexity calibration
#    - The area-to-dot mapping is deliberately low-arity: only three sprite sizes are used, with
#      contributions 1/2/3. That is enough to break "count objects" while staying visually legible.
#    - Every generated sample includes:
#      * one frame with three sprites but 7-9 dots,
#      * one frame with three sprites but only 3-4 dots,
#      * one frame with four sprites but fewer dots than the heavy frame.
#      This gives repeated evidence that area, not sprite count, controls the tally.
#    - Every sample also has at least one frame above the single-row limit (>4 dots), so D16 is
#      always visible instead of appearing only in rare edge cases.
#
# c) Implicit-assumption audit of main()
#    - `find_components(..., monochromatic=True)` still works because frames and sprites remain
#      monochrome connected components.
#    - The frame heuristic `(h >= 5) and (w >= 5) and area < h*w*0.7` still separates input-time
#      empty frames from sprites because sprites stay at most 3x2 / 2x3 and frames stay hollow.
#    - The original line `sum(1 for p in patterns if p["color"] == col)` is no longer valid:
#      color match alone is insufficient because the counted value is total area, not component
#      count. The solver must instead sum component areas and divide by 2.
#    - The old center-row loop relied on counts fitting four slots. That premise is broken by the
#      larger area totals, so the draw logic must enumerate 12 legal tally positions across three
#      rows.
#    - Dots on the upper/lower interior rows touch the frame border by 4-neighborhood, but frame
#      detection runs only on the input, where those interior dots are absent.
#
# d) Placement legality
#    - Generator remains output-first: build frames with the intended dots, erase those dots to
#      form the input, then place right-side sprites, then validate with `main(inp) == out`.
#    - Right-side placement uses a separate canvas plus `random_free_location(..., padding=1,
#      padding_connectivity=8)`, so sprites never overlap or merge.
#    - Frame counts are capped at 12 by construction, matching the three-row tally capacity.
#
# e) Visual observability
#    - A human sees repeated comparisons where the same number of same-color sprites leads to
#      different dot totals because the sprites have visibly different areas.
#    - A human also sees tallies extending off the center row in every sample, so the extra rows
#      read as a systematic continuation, not a one-off exception.
#    - The increased frame count is visible at a glance: the left column is denser and taller than
#      the original 3-4 frame task.
#
# f) Visual cleanliness
#    - Frames stay aligned in one clean left column. Sprites stay in a separate right band, so no
#      region mixes frame borders, dots, and loose sprites.
#    - Countable sprite shapes are simple filled rectangles/dominoes, making area easy to inspect.
#    - Interior dot positions are regular and right-aligned across all rows.
#
# g) Color inference
#    - Colors still mean only "which scattered sprites belong to which frame".
#    - No hidden ordering uses numeric color ids.
#    - The only new inference is geometric: larger same-color sprites contribute more tally units.
#
# h) New solver assumptions and how the generator guarantees them
#    - Assumption: same-color tally units are integer because total counted area is divisible by 2.
#      Guarantee: every countable sprite has area 2, 4, or 6.
#    - Assumption: frame tallies fit the 12 legal cells.
#      Guarantee: sampled area-units per frame are capped at 12.
#    - Assumption: the combo is not visually degenerate back to the original rule.
#      Guarantee: generator rejects any sample where the original one-row object-count solver would
#      produce the same output.


FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 33
RIGHT_C0 = 14
RIGHT_W = GRID_W - RIGHT_C0

AREA_UNIT = 2
MAX_DOTS_PER_ROW = 4
DOT_ROW_OFFSETS = (0, -1, 1)
MAX_DOTS_PER_FRAME = MAX_DOTS_PER_ROW * len(DOT_ROW_OFFSETS)

N_FRAMES_RANGE = (5, 7)
N_DISTRACTORS_RANGE = (1, 4)
BOTTOM_PAD_RANGE = (0, 5)

MAX_ATTEMPTS = 400

HEAVY_TEMPLATES = (
    (3, 3, 2),
    (3, 2, 2),
    (3, 3, 3),
)
PEER_TEMPLATES = (
    (1, 1, 1),
    (2, 1, 1),
)
CONTRAST_TEMPLATES = (
    (1, 1, 1, 1),
    (2, 1, 1, 1),
)


def _is_frame_component(comp: dict) -> bool:
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    area = int(comp["area"])
    return (h >= 5) and (w >= 5) and (area < (h * w * 0.7))


def _split_components(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    regions = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_frame_component(comp):
            frames.append(comp)
        else:
            patterns.append(comp)
    return frames, patterns


def _tally_positions_from_bbox(bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    r_mid = (r0 + r1) // 2
    positions: list[tuple[int, int]] = []
    remaining = int(count)
    for row_offset in DOT_ROW_OFFSETS:
        rr = r_mid + row_offset
        if not (r0 < rr < r1):
            continue
        for i in range(MAX_DOTS_PER_ROW):
            if remaining <= 0:
                return positions
            cc = int(c1 - 2 - 2 * i)
            if cc > c0:
                positions.append((rr, cc))
                remaining -= 1
    return positions


def _erase_patterns(base: np.ndarray, patterns: list[dict], bg: int) -> np.ndarray:
    out = base.copy()
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg
    return out


def _area_tally(patterns: list[dict], color: int) -> int:
    total_area = sum(int(p["area"]) for p in patterns if int(p["color"]) == int(color))
    return total_area // AREA_UNIT


def _object_tally(patterns: list[dict], color: int) -> int:
    return sum(1 for p in patterns if int(p["color"]) == int(color))


def _draw_area_output(base: np.ndarray, frames: list[dict], patterns: list[dict]) -> np.ndarray:
    out = base.copy()
    for f in frames:
        color = int(f["color"])
        tally = _area_tally(patterns, color)
        for rr, cc in _tally_positions_from_bbox(tuple(map(int, f["bbox"])), tally):
            out[int(rr), int(cc)] = color
    return out


def _draw_legacy_output(base: np.ndarray, frames: list[dict], patterns: list[dict]) -> np.ndarray:
    out = base.copy()
    for f in frames:
        color = int(f["color"])
        tally = _object_tally(patterns, color)
        if tally <= 0:
            continue
        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(tally):
            cc = int(c1 - 2 - 2 * i)
            if cc > c0:
                out[int(r_mid), int(cc)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert same-color sprite area into dots inside the matching frames.

    Input structure:
    - The background is the most frequent color.
    - Five or six distinct-color hollow 5x11 frames are stacked on the left.
    - Small solid sprites are scattered in a wider right-side region.
    - Countable sprites use only areas 2, 4, or 6, so every 2 cells represent one tally unit.

    Rule:
    - Remove every scattered sprite from the grid.
    - For each frame color C, sum the total area of all same-color sprites outside the frames.
    - Convert that area to N = total_area // 2.
    - Draw N dots in the matching frame: center row first from the right every 2 columns, then
      continue on the row above, then on the row below.

    Key insight:
    - The frame is tallying area-units rather than object count. A larger same-color sprite can
      contribute multiple dots, so fewer big sprites can outscore more small ones.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, patterns = _split_components(g, bg)
    cleaned = _erase_patterns(g, patterns, bg)
    return _draw_area_output(cleaned, frames, patterns)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """
    Original-task behavior kept only for generator-side anti-degeneration checks.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, patterns = _split_components(g, bg)
    cleaned = _erase_patterns(g, patterns, bg)
    return _draw_legacy_output(cleaned, frames, patterns)


def _frame_patch(color: int, bg: int, tally: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for rr, cc in _tally_positions_from_bbox((0, 0, FRAME_H - 1, FRAME_W - 1), tally):
        patch[int(rr), int(cc)] = color
    return patch


def _random_partition(rng: np.random.Generator, total_units: int) -> list[int]:
    parts: list[int] = []
    remaining = int(total_units)
    while remaining > 0:
        hi = min(3, remaining)
        choices = [v for v in range(1, hi + 1)]
        if hi >= 2 and float(rng.random()) < 0.65:
            choices = [v for v in choices if v >= min(2, hi)]
        part = int(choices[int(rng.integers(0, len(choices)))])
        parts.append(part)
        remaining -= part
    rng.shuffle(parts)
    return parts


def _sample_unit_lists(rng: np.random.Generator, n_frames: int) -> list[list[int]]:
    unit_lists: list[list[int] | None] = [None] * int(n_frames)
    order = [int(x) for x in rng.permutation(n_frames)]

    heavy_idx, peer_idx, contrast_idx = order[:3]
    unit_lists[heavy_idx] = list(HEAVY_TEMPLATES[int(rng.integers(0, len(HEAVY_TEMPLATES)))])
    unit_lists[peer_idx] = list(PEER_TEMPLATES[int(rng.integers(0, len(PEER_TEMPLATES)))])
    unit_lists[contrast_idx] = list(CONTRAST_TEMPLATES[int(rng.integers(0, len(CONTRAST_TEMPLATES)))])

    remaining_indices = order[3:]
    if remaining_indices:
        empty_idx = remaining_indices[0]
        unit_lists[empty_idx] = []
        remaining_indices = remaining_indices[1:]

    extra_overflow_idx = None
    if remaining_indices and float(rng.random()) < 0.55:
        extra_overflow_idx = remaining_indices[0]

    for idx in remaining_indices:
        if idx == extra_overflow_idx:
            total_units = int(rng.integers(5, 9))
        else:
            if float(rng.random()) < 0.25:
                total_units = 0
            else:
                total_units = int(rng.integers(1, 7))
        unit_lists[idx] = _random_partition(rng, total_units) if total_units > 0 else []

    return [list(units) for units in unit_lists]


def _sprite_patch(color: int, bg: int, units: int, vertical: bool) -> np.ndarray:
    if units == 1:
        shape = (2, 1) if vertical else (1, 2)
    elif units == 2:
        shape = (2, 2)
    elif units == 3:
        shape = (3, 2) if vertical else (2, 3)
    else:
        raise ValueError(f"unsupported sprite units: {units}")
    patch = np.full(shape, bg, dtype=int)
    patch[:, :] = color
    return patch


def _make_sprite(rng: np.random.Generator, color: int, bg: int, units: int) -> np.ndarray:
    vertical = bool(rng.integers(0, 2))
    return _sprite_patch(color, bg, units, vertical)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    unit_lists = _sample_unit_lists(rng, n_frames)
    tally_counts = [sum(units) for units in unit_lists]
    if max(tally_counts) <= MAX_DOTS_PER_ROW:
        return None
    if max(tally_counts) > MAX_DOTS_PER_FRAME:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE))
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, tally) in enumerate(zip(frame_colors, tally_counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, tally), r0, FRAME_C0)

    inp = out.copy()
    for r0, tally in zip(frame_rows, tally_counts):
        bbox = (r0, FRAME_C0, r0 + FRAME_H - 1, FRAME_C0 + FRAME_W - 1)
        for rr, cc in _tally_positions_from_bbox(bbox, tally):
            inp[int(rr), int(cc)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    sprites: list[tuple[int, int]] = []
    for color, units_list in zip(frame_colors, unit_lists):
        sprites.extend((color, units) for units in units_list)
    for color in distractor_colors:
        units = int(rng.integers(1, 4))
        sprites.append((int(color), units))
    rng.shuffle(sprites)

    try:
        for color, units in sprites:
            sprite = _make_sprite(rng, color, bg, units)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=1,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)

    if np.array_equal(_legacy_main(inp), out):
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
