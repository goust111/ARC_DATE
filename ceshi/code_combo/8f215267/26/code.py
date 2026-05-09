# Combo variant of 8f215267 (index 26)
# combo_spec: D3/0+D8/0+D16/1
# Name: 不同尺寸框架+自适应间距+紧凑裁剪

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
#
# combo_summary:
#   - D3 contributes variable-width hollow frames: height stays 5, width is chosen
#     from {7, 9, 11}, and different frames in one puzzle need not share a width.
#   - D16 changes the tally layout itself: N matching sprites no longer mean "place
#     dots from the right every 2 columns". Instead, the N dots are spread across
#     the frame interior as evenly as possible. One dot sits at the interior center;
#     multiple dots span from the inner-left edge to the inner-right edge with near-
#     uniform gaps. This makes the actual frame width part of the reasoning chain.
#   - D8 then crops the solved result to the union bbox of the whole frame stack.
#     Because frames are left-aligned but widths differ, the crop width becomes the
#     widest frame width rather than a hard-coded constant.
#
# concrete_behavior_trace:
#   - Original task:
#       * every frame is 5x11
#       * 3 matching sprites -> draw on center row at relative cols [8, 6, 4]
#       * return the full 23-column canvas
#   - Combo task example:
#       * frame A is 5x7 and has 3 matching sprites -> interior cols are 1..5, so
#         the evenly spread tally lands at [1, 3, 5]
#       * frame B is 5x9 and has 2 matching sprites -> tally lands at [1, 7]
#       * frame C is 5x11 and has 1 matching sprite -> tally lands at [5], the
#         interior center, not at the right side
#       * after solving, discard all non-frame margins and the right sprite field,
#         keeping only the bbox spanning the frame stack
#   - Difference:
#       * width now changes both the capacity and the dot spacing
#       * dot placement is count-adaptive rather than fixed-step
#       * output size depends on the visible frame layout
#
# complexity_calibration:
#   - Widths are restricted to three odd choices {7, 9, 11}; this keeps the rule
#     learnable while still making width differences visually obvious.
#   - Each sample uses at least two widths, and some samples omit width 11 so the
#     crop width is sometimes 9 instead of always 11. That makes the D3+D8 link
#     observable rather than latent.
#   - Matching counts are moderate (6-11 total), with at least one frame receiving
#     2+ sprites, and generator-side caps leave at least one interior gap per frame.
#     This keeps the adaptive spacing visible instead of collapsing to only empty,
#     single-dot, or fully saturated center rows.
#
# implicit_assumption_audit:
#   - Component split:
#       The old solver relied on a loose "large and sparse" test. Here that is too
#       weak because width matters, so the solver assumes true frames are exact 5 x
#       {7,9,11} perimeter rectangles. The generator guarantees input frames are
#       undotted perimeter-only rectangles, while sprites remain much smaller.
#   - Pattern removal loop:
#       Treating every non-frame monochrome component as a removable sprite still
#       works because generator never places any other foreground objects.
#   - Dot drawing loop:
#       The solver now assumes the number of matching sprites fits inside the frame
#       interior width. Generator enforces count <= width-2 for each frame.
#   - Crop step:
#       The solver assumes the desired output is exactly the union bbox of the input
#       frames. Generator uses fixed left alignment, fixed vertical spacing, and
#       validates every sample with main(input), so the crop boundary is stable.
#
# placement_legality_and_cleanliness:
#   - Frames are vertically stacked with a 2-row gap, so they never overlap even
#     when widths differ.
#   - Sprites are still confined to a separate right-side canvas and pasted with
#     8-neighborhood padding, so they neither touch frames nor each other.
#   - The cropped output contains only the frame stack; no right-side clutter or
#     extra top/bottom background survives.
#
# observability:
#   - Human solvers can compare narrow vs wide frames and see that one dot centers
#     itself, two dots stretch to both inner sides, and larger counts subdivide the
#     available width more densely.
#   - Crop width changes with the widest frame present, so frame geometry affects
#     both interior dot layout and final canvas shape.
#   - No color-id ordering or any invisible numeric rule is used; only visible color
#     matching, visible frame size, and visible cropping matter.


FRAME_H = 5
FRAME_WIDTH_CHOICES = (7, 9, 11)
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

N_FRAMES_RANGE = (3, 5)        # 3-4 frames
N_DISTRACTORS_RANGE = (1, 3)   # 1-2 distractors
MATCH_TOTAL_RANGE = (6, 12)    # 6-11 matching sprites total
TOP_MARGIN_RANGE = (1, 4)      # 1-3 blank rows above the stack
BOTTOM_PAD_RANGE = (0, 7)

SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MAX_ATTEMPTS = 400

WIDTH_TEMPLATES = {
    3: (
        (7, 7, 9),
        (7, 9, 9),
        (7, 9, 11),
        (7, 11, 11),
        (9, 9, 11),
    ),
    4: (
        (7, 7, 9, 9),
        (7, 7, 9, 11),
        (7, 9, 9, 11),
        (7, 9, 11, 11),
        (9, 9, 11, 11),
    ),
}


def _frame_capacity(width: int) -> int:
    """Maximum number of dots that can occupy distinct interior columns."""
    return int(width) - 2


def _generator_count_cap(width: int) -> int:
    """
    Keep one interior background cell in generated examples for readability.

    The solver still supports the full geometric capacity, but the generator avoids
    fully saturating the center row so the adaptive spacing remains visually legible.
    """
    return max(1, _frame_capacity(width) - 1)


def _adaptive_dot_cols(width: int, n_dots: int) -> list[int]:
    """
    Return in-patch columns for N dots spread across the frame interior.

    One dot sits at the interior center.
    Two or more dots span the full interior from inner-left to inner-right.
    """
    width = int(width)
    n = max(0, min(int(n_dots), _frame_capacity(width)))
    if n == 0:
        return []
    if n == 1:
        return [width // 2]

    span = width - 3
    cols = [1 + int(round(i * span / (n - 1))) for i in range(n)]
    return cols


def _is_input_frame(comp: dict) -> bool:
    """
    Recognize an undotted hollow frame in the input.

    Generator guarantees:
    - frames are monochrome
    - height is always 5
    - width is one of 7/9/11
    - the input contains only the border, never the solved dots
    """
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    if h != FRAME_H or w not in FRAME_WIDTH_CHOICES:
        return False

    expected_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected_area:
        return False

    r0, c0, r1, c1 = map(int, b)
    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    border = set()
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    return coords == border


def _split_input_components(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Partition monochrome components into frames vs removable sprites."""
    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_input_frame(comp):
            frames.append(comp)
        else:
            patterns.append(comp)
    return frames, patterns


def _crop_to_frame_bbox(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    """Tight-crop to the union bbox of all detected frames."""
    if not frames:
        return np.array(grid, dtype=int).copy()
    coords = np.concatenate([np.array(f["coords"], dtype=int) for f in frames], axis=0)
    r0, c0, r1, c1 = bbox_coords(coords)
    return np.array(grid, dtype=int)[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert scattered same-color sprites into evenly spread tallies inside variable-width frames.

    Input structure:
    - The background is the most common color.
    - The left side contains 3-4 hollow monochrome frames stacked vertically; every frame is 5 rows tall,
      but widths vary across 7, 9, and 11.
    - The right side contains small monochrome sprites whose colors either match a frame color or act as distractors.

    Rule:
    - Remove every small sprite from the scene.
    - For each frame of color C, count how many removed sprites also had color C.
    - Draw that many dots on the frame's middle row, but distribute them evenly across the available interior width:
      one dot is centered, while multiple dots span from the inner-left edge to the inner-right edge.
    - Return only the union bbox of the whole frame stack, cropping away the sprite region and extra margins.

    Key insight:
    Frame width is no longer cosmetic: it determines both the final crop width and the spacing of the tally dots,
    so the solver must infer the layout from each detected frame rather than from a single fixed template.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _split_input_components(g, bg)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        color = int(f["color"])
        count = sum(1 for p in patterns if int(p["color"]) == color)
        if count <= 0:
            continue

        r0, c0, _r1, _c1 = map(int, f["bbox"])
        width = bbox_w(f["bbox"])
        r_mid = r0 + FRAME_H // 2
        for dc in _adaptive_dot_cols(width, count):
            out[r_mid, c0 + int(dc)] = color

    return _crop_to_frame_bbox(out, frames)


def _frame_patch(color: int, bg: int, width: int, n_dots: int) -> np.ndarray:
    """Build a 5 x width hollow rectangle with an adaptive tally on the center row."""
    patch = np.full((FRAME_H, int(width)), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[FRAME_H // 2, _adaptive_dot_cols(width, n_dots)] = color
    return patch


def _sample_frame_widths(rng: np.random.Generator, n_frames: int) -> list[int]:
    """
    Pick a width template with guaranteed variety and variable crop widths.

    At least two widths appear in every sample, and some templates top out at width 9,
    so the cropped output width is not always 11.
    """
    templates = WIDTH_TEMPLATES[int(n_frames)]
    widths = list(templates[int(rng.integers(0, len(templates)))])
    rng.shuffle(widths)
    return [int(w) for w in widths]


def _sample_counts(
    rng: np.random.Generator,
    capacities: list[int],
    total_match: int,
) -> list[int]:
    """
    Distribute matching sprites across frames with visibility constraints.

    Guarantees:
    - at least two frames are active
    - at least one frame gets 2+ sprites so adaptive spacing is visible
    - no frame exceeds its generator cap, which stays below geometric saturation
    """
    n_frames = len(capacities)
    for _ in range(200):
        n_active = int(rng.integers(max(2, n_frames - 1), n_frames + 1))
        active = set(int(x) for x in rng.choice(n_frames, size=n_active, replace=False))

        counts = [1 if i in active else 0 for i in range(n_frames)]
        remaining = int(total_match - sum(counts))
        if remaining < 0:
            continue

        boostable = [i for i in active if capacities[i] >= 2]
        if remaining > 0 and boostable:
            idx = int(boostable[int(rng.integers(0, len(boostable)))])
            counts[idx] += 1
            remaining -= 1

        residual = [cap - count for cap, count in zip(capacities, counts)]
        order = [int(x) for x in rng.permutation(n_frames)]
        for pos, idx in enumerate(order):
            caps_left = sum(residual[j] for j in order[pos + 1 :])
            lo = max(0, remaining - caps_left)
            hi = min(residual[idx], remaining)
            add = int(rng.integers(lo, hi + 1))
            counts[idx] += add
            residual[idx] -= add
            remaining -= add

        if remaining != 0:
            continue
        if max(counts) < 2:
            continue
        return counts

    raise RuntimeError("failed to sample visible adaptive counts")


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino of the given area inside an h x w bounding box."""
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.append(nb)
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    """Generate a small random connected sprite, tight-cropped to its bounding box."""
    area = int(rng.integers(*SPRITE_AREA_RANGE))
    dims = [
        (h, w)
        for h in range(1, SPRITE_MAX_H + 1)
        for w in range(1, SPRITE_MAX_W + 1)
        if h * w >= area
    ]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _crop_from_layout(grid: np.ndarray, frame_rows: list[int], frame_widths: list[int]) -> np.ndarray:
    """Crop using the known frame placements from generation time."""
    r0 = min(frame_rows)
    r1 = max(r + FRAME_H - 1 for r in frame_rows)
    c0 = FRAME_C0
    c1 = FRAME_C0 + max(frame_widths) - 1
    return np.array(grid, dtype=int)[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    frame_widths = _sample_frame_widths(rng, n_frames)
    capacities = [_generator_count_cap(w) for w in frame_widths]

    max_match = min(MATCH_TOTAL_RANGE[1] - 1, sum(capacities))
    min_match = min(max(MATCH_TOTAL_RANGE[0], n_frames + 1), max_match)
    total_match = int(rng.integers(min_match, max_match + 1))

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, capacities, total_match)

    top_margin = int(rng.integers(*TOP_MARGIN_RANGE))
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE))
    height = top_margin + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    full_out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, width, count) in enumerate(zip(frame_colors, frame_widths, counts)):
        r0 = top_margin + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(full_out, _frame_patch(color, bg, width, count), r0, FRAME_C0)
    out = _crop_from_layout(full_out, frame_rows, frame_widths)

    inp = full_out.copy()
    for r0, width, count in zip(frame_rows, frame_widths, counts):
        mid_r = r0 + FRAME_H // 2
        for dc in _adaptive_dot_cols(width, count):
            inp[mid_r, FRAME_C0 + int(dc)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
