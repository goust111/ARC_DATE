# Combo variant of 8f215267 (index 8)
# combo_spec: D3/0+D16/1
# Name: 不同尺寸框架+自适应间距

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for combo D3/0 + D16/1
# ---------------------------------------
# a) Concrete behavior trace
#    - Original task: every frame is 5x11, so the solver can hardcode dot columns [8, 6, 4, 2]
#      relative to the frame's left edge. Counting and placement are decoupled because spacing never changes.
#    - D3 alone changes only width: a 5x7 frame would expose only two legal fixed-gap slots [4, 2],
#      a 5x9 frame three slots [6, 4, 2], and a 5x11 frame four slots [8, 6, 4, 2].
#    - This combo changes both width and spacing. Example:
#      width=9, count=3 -> original-style fixed-gap placement would be [6, 4, 2];
#      combo placement uses the full inner span 1..7 and distributes the dots evenly -> [1, 4, 7].
#      width=7, count=3 would even overflow the old D3 capacity of two slots, but now fits as [1, 3, 5].
#    - Therefore the solver must infer two visible facts at once:
#      (1) same-color sprites are counted per frame, and
#      (2) the frame interior is filled by an even-spacing formula that depends on both width and count.
# b) Complexity calibration
#    - Widths are limited to {7, 9, 11}. This keeps the size vocabulary small enough for ARC while still making
#      right edges visibly misaligned.
#    - Counts remain moderate, but the generator always includes at least one frame whose dot count exceeds the old
#      fixed-gap capacity for that width. This prevents regression to the D3-only interpretation.
#    - Each puzzle also includes another nontrivial positive count, so samples show both "counting" and
#      "recomputed spacing" instead of only a single saturated frame.
# c) Implicit-assumption audit against the original main()
#    - Original frame detection used "large and sparse". That is too loose once frame widths vary and dots may use
#      the full interior, so the combo solver switches to an exact hollow-rectangle test on the input.
#    - Original placement assumed a fixed right-start and step-2 recurrence. That assumption is invalid now.
#      The new helper computes legal columns from (width, n_dots) directly.
#    - Original capacity was implicit in the loop guard `cc > c0`. That no longer defines capacity; capacity is now
#      the whole interior width, i.e. at most width-2 visible dot cells.
# d) Placement legality
#    - Frames are still height 5, left-aligned, and separated by two blank rows, so they never overlap.
#    - Max width stays 11, preserving a clean split between the left frame area and the right sprite area.
#    - The generator uses output-first construction, removes dots to make the input, and scatters sprites only in
#      the right canvas with padding, so frames and sprites stay disjoint and in-bounds.
# e) Visual observability
#    - Width differences are directly visible because frame right edges differ.
#    - Adaptive spacing is directly visible because single dots sit at the center, two dots stretch to both sides,
#      and higher counts spread across the whole interior rather than hugging the right edge.
#    - No color ordering or hidden metadata is used: the only matching cue is visible color equality.
# f) Visual cleanliness
#    - Only one object family occupies the left region (frames) and only one occupies the right region (sprites).
#    - Dots remain on the center row only, which keeps the output legible even when counts rise.
# g) Post-change solver assumptions that the generator must satisfy
#    - Frames in the input must be perimeter-only rectangles of height 5 and width in {7, 9, 11}.
#    - Non-frame components are treated as sprites, so generator sprites must not accidentally form hollow frames.
#    - For every frame, matching-sprite count must not exceed interior width (width-2); generator enforces that.


FRAME_H = 5
FRAME_WIDTH_CHOICES = (7, 9, 11)
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

N_FRAMES_RANGE = (3, 5)        # 3-4 frames
N_DISTRACTORS_RANGE = (1, 3)   # 1-2 distractors
MATCH_TOTAL_RANGE = (6, 12)    # 6-11 matching sprites total
BOTTOM_PAD_RANGE = (0, 7)

SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MAX_ATTEMPTS = 300


def _legacy_capacity(width: int) -> int:
    """Old fixed-gap capacity: right-start, then every 2 columns."""
    return (int(width) - 3) // 2


def _adaptive_capacity(width: int) -> int:
    """New capacity: every interior cell could host a dot if needed."""
    return int(width) - 2


def _round_half_up(num: int, den: int) -> int:
    """Integer nearest-rounding with .5 rounded upward."""
    return (2 * int(num) + int(den)) // (2 * int(den))


def _adaptive_dot_cols(width: int, n_dots: int) -> list[int]:
    """
    Return in-patch columns for the combo rule.

    Dots always stay on the center row.
    - 0 dots: empty
    - 1 dot: center of the interior
    - N>=2 dots: evenly distributed from the left inner wall to the right inner wall
    """
    width = int(width)
    n_dots = int(n_dots)
    if n_dots <= 0:
        return []

    inner_w = width - 2
    if n_dots > inner_w:
        raise ValueError(f"cannot place {n_dots} dots inside width {width}")
    if n_dots == 1:
        return [width // 2]

    span = inner_w - 1
    cols = [1 + _round_half_up(i * span, n_dots - 1) for i in range(n_dots)]
    if len(set(cols)) != n_dots:
        raise ValueError(f"dot placement collapsed for width={width}, n={n_dots}")
    return cols


def _legacy_dot_cols(width: int, n_dots: int) -> list[int]:
    """Old fixed-gap columns, truncated by that regime's capacity."""
    width = int(width)
    n_dots = min(int(n_dots), _legacy_capacity(width))
    return [width - 3 - 2 * i for i in range(n_dots)]


def _is_hollow_frame(comp: dict) -> bool:
    """
    Exact input-frame recognizer.

    generate_input() guarantees that input frames contain only the perimeter pixels;
    dots appear only in the output.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the scattered sprites and redraw each frame as a width-aware tally display.

    Input structure:
    - The background is the most frequent color.
    - On the left are 3-4 hollow monochrome frames, all height 5 but with widths chosen from 7/9/11.
    - On the right are small monochrome sprites; some share a frame color and some are distractors.

    Rule:
    - Erase every sprite from the grid.
    - For each frame color C, count how many sprites have color C.
    - Place that many dots on the frame's center row, but do not use a fixed gap:
      1 dot goes in the interior center, and 2+ dots are spread evenly across the frame's inner width.

    Key insight:
    - Frames are color-matched counters, and their actual width determines how the counted dots are laid out.
      The solver must infer both the counting relation and the adaptive spacing formula from the visible geometry.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_hollow_frame(comp):
            frames.append(comp)
        else:
            patterns.append(comp)

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

    return out


def _frame_patch(color: int, bg: int, width: int, n_dots: int) -> np.ndarray:
    """Build a 5 x width hollow frame with adaptive dots on the center row."""
    patch = np.full((FRAME_H, int(width)), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    patch[mid_r, _adaptive_dot_cols(width, n_dots)] = color
    return patch


def _sample_frame_widths(rng: np.random.Generator, n_frames: int) -> list[int]:
    """
    Guarantee visible width variation.

    For 3 frames we use all three widths exactly once.
    For 4 frames we add one duplicate after including all three widths.
    """
    widths = list(FRAME_WIDTH_CHOICES)
    if n_frames == 4:
        widths.append(int(rng.choice(FRAME_WIDTH_CHOICES)))
    rng.shuffle(widths)
    return [int(w) for w in widths[:n_frames]]


def _sample_counts(rng: np.random.Generator, widths: list[int]) -> list[int]:
    """
    Sample counts that visibly exercise the combo rule.

    Guarantees:
    - at least one frame exceeds the old fixed-gap capacity for its width;
    - at least one additional frame has a nontrivial positive count;
    - total matching sprites stays moderate for clean right-side placement.
    """
    n_frames = len(widths)
    caps = [_adaptive_capacity(w) for w in widths]
    legacy_caps = [_legacy_capacity(w) for w in widths]

    counts = [0] * n_frames

    overflow_candidates = [i for i, w in enumerate(widths) if w in (7, 9)]
    if not overflow_candidates:
        overflow_candidates = list(range(n_frames))
    special = int(rng.choice(overflow_candidates))
    counts[special] = legacy_caps[special] + 1

    other_indices = [i for i in range(n_frames) if i != special]
    second = int(rng.choice(other_indices))
    second_hi = min(caps[second], 3)
    counts[second] = int(rng.integers(2, second_hi + 1))

    n_active = int(rng.integers(2, n_frames + 1))
    if n_active > 2:
        extra_active = rng.choice(other_indices, size=n_active - 2, replace=False)
        for idx in extra_active:
            idx = int(idx)
            if counts[idx] == 0:
                counts[idx] = 1

    base_total = sum(counts)
    max_total = min(MATCH_TOTAL_RANGE[1] - 1, sum(caps))
    min_total = max(MATCH_TOTAL_RANGE[0], base_total)
    target_total = int(rng.integers(min_total, max_total + 1))

    remaining = target_total - base_total
    while remaining > 0:
        candidates = [i for i in range(n_frames) if counts[i] < caps[i]]
        if not candidates:
            break
        idx = int(rng.choice(candidates))
        add_hi = min(remaining, caps[idx] - counts[idx])
        add = int(rng.integers(1, add_hi + 1))
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
    """Generate a small connected monochrome sprite and crop it tightly."""
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
    counts = _sample_counts(rng, frame_widths)

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, width, count) in enumerate(zip(frame_colors, frame_widths, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, width, count), r0, FRAME_C0)

    inp = out.copy()
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

    # Anti-degeneration check: at least one frame must differ from the old fixed-gap regime in a meaningful way.
    changed = any(
        counts[i] > _legacy_capacity(frame_widths[i])
        or _adaptive_dot_cols(frame_widths[i], counts[i]) != _legacy_dot_cols(frame_widths[i], counts[i])
        for i in range(n_frames)
        if counts[i] > 0
    )
    if not changed:
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
