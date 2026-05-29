# Combo variant of 8f215267 (index 6)
# combo_spec: D3/0+D8/0
# Name: 不同尺寸框架+紧凑裁剪

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# combo_behavior:
#   summary:
#     - Keep the original color-matching tally rule.
#     - Fuse two changes so they depend on the same visible signal: frame width.
#       Each frame is still a 5-row hollow rectangle, but widths vary in {7, 9, 11}.
#       After tallying, the result is cropped to the union bbox of the frame stack.
#   concrete_trace:
#     - Original task:
#       a 5x11 frame can host up to 4 dots at relative columns [8, 6, 4, 2], and the
#       solver returns the full 23-column canvas even after sprites are erased.
#     - Combo task:
#       a 5x7 frame only offers [4, 2], a 5x9 frame offers [6, 4, 2], and a 5x11
#       frame offers [8, 6, 4, 2]. After drawing those dots, the output is cropped
#       to the frame union bbox, so a sample whose widest frame is 9 columns wide
#       produces a 9-column output, while a sample with an 11-wide frame produces
#       an 11-column output.
#     - Difference:
#       the same detected frame geometry now controls both the interior dot capacity
#       and the output width. This is the intended coupling between D3 and D8.
#   complexity_calibration:
#     - Widths are restricted to the small odd set {7, 9, 11}; that keeps slot
#       inference learnable while still making frame sizes visibly different.
#     - Every generated puzzle contains at least two distinct widths and at least one
#       frame narrower than the widest one, so the D3 change is always observable.
#     - The widest frame is sampled from {9, 11}; this makes the crop width vary
#       across samples instead of collapsing to the original width every time.
#   implicit_assumption_audit:
#     - Original assumption "all large sparse components are frames" is tightened to
#       an exact perimeter check, because variable widths make a precise recognizer
#       safer and the generator guarantees perimeter-only input frames.
#     - Original assumption "the solver can hard-code width 11" is no longer valid.
#       Dot slots are recomputed from the detected bbox width for each frame.
#     - New crop step must not rediscover frames after dots are added, because the
#       solved frames are no longer pure perimeters. The solver therefore crops using
#       the frame components found in the input, and the generator crops using the
#       known frame layout it just created.
#   placement_legality:
#     - Frames stay left-aligned at one column, stacked vertically with a 2-row gap,
#       so they never overlap even when widths differ.
#     - Sprites are still generated inside a separate right-side canvas and pasted at
#       columns 13..21, which remains strictly to the right of even an 11-wide frame.
#     - Matching counts are sampled against width-derived capacities, so no frame is
#       asked to display more dots than it can physically hold.
#   observability_and_cleanliness:
#     - Humans can see two linked effects: narrow frames have fewer possible dot
#       positions, and the output's right boundary follows the widest frame instead of
#       the fixed 23-column canvas.
#     - No ordering depends on raw color ids; only visible color equality matters.
#     - The cropped output contains only the frame stack, so the right sprite field
#       disappears cleanly rather than competing with the final tally display.
#
# main_assumptions_after_change:
#   - Input frames are monochrome 5x{7,9,11} hollow rectangles with no pre-drawn dots.
#   - Every non-frame monochrome component is a sprite and can be erased to background.
#   - Sprites never overlap or touch frames, so removing them cannot damage frame bboxes.
#   - The desired crop is exactly the union bbox of the input frame components.
#   - generate_input() enforces those assumptions and validates each sample with main().


FRAME_H = 5
FRAME_WIDTH_CHOICES = (7, 9, 11)
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9
TOP_MARGIN_RANGE = (1, 4)

N_FRAMES_RANGE = (3, 5)        # 3-4 frames
N_DISTRACTORS_RANGE = (1, 3)   # 1-2 distractors
MATCH_TOTAL_RANGE = (4, 9)     # 4-8 matching sprites total
BOTTOM_PAD_RANGE = (0, 7)

SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MAX_ATTEMPTS = 300


def _frame_dot_cols(width: int) -> list[int]:
    """Return legal in-frame dot columns for a frame of the given width."""
    width = int(width)
    return [width - 3 - 2 * i for i in range((width - 3) // 2)]


def _is_hollow_frame(comp: dict) -> bool:
    """Recognize the exact 5x{7,9,11} perimeter-only frames used in the input."""
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


def _split_components(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Partition components into input frames vs scattered sprites."""
    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_hollow_frame(comp):
            frames.append(comp)
        else:
            patterns.append(comp)
    frames.sort(key=lambda comp: tuple(map(int, comp["bbox"][:2])))
    return frames, patterns


def _crop_to_frame_bbox(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    """Crop to the union bbox of the already-detected input frames."""
    if not frames:
        return np.array(grid, dtype=int).copy()
    coords = np.concatenate([np.array(f["coords"], dtype=int) for f in frames], axis=0)
    r0, c0, r1, c1 = bbox_coords(coords)
    return np.array(grid, dtype=int)[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _crop_to_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop grid to an explicit inclusive bbox."""
    r0, c0, r1, c1 = map(int, bbox)
    return np.array(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1]


def _frame_union_bbox(frame_rows: list[int], frame_widths: list[int]) -> tuple[int, int, int, int]:
    """Union bbox of the generated frame stack."""
    r0 = min(int(r) for r in frame_rows)
    r1 = max(int(r) + FRAME_H - 1 for r in frame_rows)
    c0 = FRAME_C0
    c1 = max(FRAME_C0 + int(w) - 1 for w in frame_widths)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove scattered sprites, tally each frame's same-color matches with width-aware dots,
    and return only the frame region.

    Input structure:
    - The background is the dominant color.
    - The left side contains vertically stacked hollow frames, all 5 rows tall but with
      widths chosen from 7, 9, or 11 columns.
    - Small monochrome sprites are scattered outside the frames in the right-hand region,
      and some sprite colors match frame colors while others are distractors.

    Rule:
    - Detect the perimeter-only frames and treat every other monochrome component as a sprite.
    - Erase all sprites to background.
    - For each frame color, count matching-color sprites and draw that many dots on the
      frame's center row, starting from the right interior cell and stepping left by 2.
      The legal dot slots are inferred from that frame's actual width.
    - After the tally is drawn, crop the solved canvas to the union bbox of all frames.

    Key insight:
    - Frame geometry carries two pieces of information at once: width determines how many
      tally positions fit inside a frame, and the union of frame widths determines the
      final cropped output boundary.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _split_components(g, bg)

    for pattern in patterns:
        for r, c in pattern["coords"]:
            out[int(r), int(c)] = bg

    for frame in frames:
        color = int(frame["color"])
        num_patterns = sum(1 for pattern in patterns if int(pattern["color"]) == color)
        if num_patterns <= 0:
            continue

        r0, c0, _r1, _c1 = map(int, frame["bbox"])
        r_mid = r0 + FRAME_H // 2
        width = bbox_w(frame["bbox"])
        for dc in _frame_dot_cols(width)[:num_patterns]:
            out[r_mid, c0 + int(dc)] = color

    return _crop_to_frame_bbox(out, frames)


def _frame_patch(color: int, bg: int, width: int, n_dots: int) -> np.ndarray:
    """Build a 5 x width hollow rectangle with n_dots on the center row."""
    patch = np.full((FRAME_H, int(width)), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[FRAME_H // 2, _frame_dot_cols(width)[: int(n_dots)]] = color
    return patch


def _sample_frame_widths(rng: np.random.Generator, n_frames: int) -> list[int]:
    """
    Sample visible width variation with variable crop width.

    The widest frame is 9 or 11, never always 11 by construction, and every puzzle
    also contains at least one narrower frame.
    """
    max_width = int(rng.choice((9, 11)))
    narrower_choices = [w for w in FRAME_WIDTH_CHOICES if w < max_width]
    widths = [max_width, int(rng.choice(narrower_choices))]
    all_choices = [w for w in FRAME_WIDTH_CHOICES if w <= max_width]
    while len(widths) < n_frames:
        widths.append(int(rng.choice(all_choices)))
    rng.shuffle(widths)
    return widths


def _sample_counts(
    rng: np.random.Generator,
    frame_widths: list[int],
    capacities: list[int],
    total_match: int,
) -> list[int]:
    """
    Distribute matches across frames while respecting width-derived capacities.

    At least one widest frame and one narrower frame receive dots so the variable-width
    tally rule is visible in the solved examples, not only in the empty frame geometry.
    """
    n_frames = len(capacities)
    active_count = int(rng.integers(2, min(n_frames, total_match) + 1))
    max_width = max(int(width) for width in frame_widths)
    widest = [i for i, width in enumerate(frame_widths) if int(width) == max_width]
    narrower = [i for i, width in enumerate(frame_widths) if int(width) < max_width]

    active = {
        int(rng.choice(widest)),
        int(rng.choice(narrower)),
    }
    remaining_candidates = [i for i in range(n_frames) if i not in active]
    extra = active_count - len(active)
    if extra > 0:
        active.update(int(x) for x in rng.choice(remaining_candidates, size=extra, replace=False))

    counts = [1 if i in active else 0 for i in range(n_frames)]
    residual_caps = [cap - counts[i] for i, cap in enumerate(capacities)]
    remaining = total_match - sum(counts)
    order = [int(x) for x in rng.permutation(n_frames)]

    for pos, idx in enumerate(order):
        caps_left = sum(residual_caps[j] for j in order[pos + 1 :])
        lo = max(0, remaining - caps_left)
        hi = min(residual_caps[idx], remaining)
        add = int(rng.integers(lo, hi + 1))
        counts[idx] += add
        remaining -= add

    return counts


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
    """Generate a small random connected sprite, tight-cropped to its bbox."""
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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    frame_widths = _sample_frame_widths(rng, n_frames)
    capacities = [len(_frame_dot_cols(width)) for width in frame_widths]

    max_match = min(MATCH_TOTAL_RANGE[1] - 1, sum(capacities))
    min_match = min(MATCH_TOTAL_RANGE[0], max_match)
    total_match = int(rng.integers(min_match, max_match + 1))

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]
    counts = _sample_counts(rng, frame_widths, capacities, total_match)

    top_margin = int(rng.integers(*TOP_MARGIN_RANGE))
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = top_margin + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    full_out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, width, count) in enumerate(zip(frame_colors, frame_widths, counts)):
        r0 = top_margin + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(full_out, _frame_patch(color, bg, width, count), r0, FRAME_C0)

    out_bbox = _frame_union_bbox(frame_rows, frame_widths)
    out = _crop_to_bbox(full_out, out_bbox)

    inp = full_out.copy()
    for r0, width, count in zip(frame_rows, frame_widths, counts):
        mid_r = r0 + FRAME_H // 2
        for dc in _frame_dot_cols(width)[:count]:
            inp[mid_r, FRAME_C0 + int(dc)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [color for color, count in zip(frame_colors, counts) for _ in range(count)]
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
