# Combo variant of 8f215267 (index 4)
# combo_spec: D2/0+D6/2
# Name: 多色精灵+面积超标干扰

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The left side keeps the original 5x11 hollow frames stacked vertically.
#    - The right side now contains two kinds of multicolor 4-connected sprites:
#      counted small sprites (area 3-5) and oversized distractors (area 6-8).
#    - Every sprite uses exactly two colors. One is a frame color and occupies more
#      cells; the other is a repeated accent color and occupies fewer cells.
#    - Matching is now a two-step decision:
#        1. group the whole multicolor component and infer its dominant color;
#        2. keep it only if its total area is at most 5.
#    - Concrete trace:
#      old original logic saw a sprite shaped like `main-accent-main` as two separate
#      monochrome main-color pieces and would overcount by splitting one object into two.
#      A D2-only solver fixes that by counting the whole component via majority color,
#      but it would still count a large 7-cell component of that same dominant color.
#      The combo solver must do both: treat each two-color component as one sprite and
#      then discard the ones whose area exceeds 5.
# b) Complexity calibration
#    - Small and large sprites come from finite template libraries instead of arbitrary
#      growth. This keeps the visual grammar compact and repeated across examples.
#    - All templates visibly use both colors, and the main-color cells are split into
#      multiple monochrome islands bridged by accent cells. This prevents the multicolor
#      change from collapsing into "one solid sprite with a decorative pixel".
#    - Large distractors differ from counted sprites by an obvious size band: 6-8 cells
#      versus 3-5 cells. The threshold is visually learnable from a few examples.
# c) Implicit assumption audit against the original main()
#    - Original `find_components(... monochromatic=True)` assumed each sprite had one
#      color; that breaks because one logical sprite now spans two colors.
#    - Original `comp["color"]` for sprites assumed the component color was the sprite
#      identity; now the identity must be derived by counting colors inside the full
#      multicolor component.
#    - Original `sum(... if color matches)` assumed every same-color sprite counted.
#      That breaks because same-dominant-color sprites with area 6-8 are distractors.
#    - Frame detection still relies on large hollow geometry, but now it must also
#      require exactly one color so a multicolor sprite can never be mistaken for a frame.
# d) Placement legality
#    - Frames remain in the left block and are never edited after construction except
#      for removing the provisional dots when deriving the input.
#    - All sprites are placed on a separate right canvas with padding=1 and 8-neighbor
#      exclusion, so distinct sprites cannot merge into one component.
#    - Large templates stay within 3x3 after crop, so they fit the right strip and
#      remain far smaller than any frame bbox.
# e) Visual observability
#    - A solver can see two independent clues in the same scene:
#      multi-colority inside each sprite and a size contrast among same-main-color sprites.
#    - The generator forces at least one frame color to appear on both a counted small
#      sprite and an oversized distractor, so "same dominant color but different fate"
#      is directly visible.
#    - The generator also rejects samples where the original monochrome solver or a
#      multicolor-no-area solver would coincidentally produce the same output.
# f) Visual neatness
#    - The scene still has a clean left/right partition with no overlapping marks.
#    - Only the frame interiors change in the output, so the result remains legible.
# g) Change inferability
#    - Accent colors are drawn from only two non-frame colors reused across many sprites.
#      This makes the dominant-color rule easier to infer than if every sprite used a
#      new color pair.
#    - Because the distractors reuse frame colors as their dominant color, the remaining
#      distinguishing cue is size, which is observable without relying on color IDs.
# 9) New assumptions introduced by the combo main()
#    - Each sprite must have a unique dominant color; the generator guarantees main >
#      accent in every template.
#    - Counted sprites must have area <= 5 and oversized distractors area >= 6; the
#      template libraries enforce that separation exactly.
#    - At least one counted sprite must require multicolor grouping, and at least one
#      oversized distractor must share its dominant color with a frame that also has a
#      counted small sprite; the generator enforces both before accepting a sample.


COUNTED_MAX_AREA = 5

# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_SPRITES_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)
N_ACCENT_COLORS = 2

# --- placement / retry ---
SPRITE_PADDING = 1
MAX_ATTEMPTS = 500

# Template legend: 0 = bg, 1 = dominant/main color, 2 = accent color.
# All templates are single 4-connected multicolor components. The main-color cells are
# split into multiple monochrome components so grouping the full multicolor object is
# necessary.
SMALL_SPRITE_TEMPLATES = [
    np.array([[1, 2, 1]], dtype=int),                       # area 3
    np.array([[1], [2], [1]], dtype=int),                   # area 3
    np.array([[1, 2], [0, 1]], dtype=int),                  # area 3
    np.array([[1, 0], [2, 1]], dtype=int),                  # area 3
    np.array([[1, 2, 1], [0, 1, 0]], dtype=int),            # area 4
    np.array([[1, 0], [2, 1], [1, 0]], dtype=int),          # area 4
    np.array([[1, 2, 1], [0, 2, 1]], dtype=int),            # area 5
    np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=int), # area 5
]

LARGE_SPRITE_TEMPLATES = [
    np.array([[1, 2, 1], [1, 2, 1]], dtype=int),            # area 6
    np.array([[1, 2, 1], [0, 2, 1], [1, 0, 1]], dtype=int), # area 7
    np.array([[1, 2, 1], [1, 2, 1], [1, 0, 1]], dtype=int), # area 8
    np.array([[1, 2, 1], [0, 2, 1], [1, 2, 1]], dtype=int), # area 8
]


def _component_color_counts(grid: np.ndarray, comp: dict) -> Counter[int]:
    counts: Counter[int] = Counter()
    for r, c in comp["coords"]:
        counts[int(grid[int(r), int(c)])] += 1
    return counts


def _dominant_color(grid: np.ndarray, comp: dict) -> int | None:
    counts = _component_color_counts(grid, comp)
    if not counts:
        return None
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    if len(winners) != 1:
        return None
    return int(winners[0])


def _is_frame(grid: np.ndarray, comp: dict) -> bool:
    counts = _component_color_counts(grid, comp)
    if len(counts) != 1:
        return False
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    area = int(comp["area"])
    return (h >= 5) and (w >= 5) and (area < (h * w * 0.7))


def _frame_color(grid: np.ndarray, comp: dict) -> int | None:
    counts = _component_color_counts(grid, comp)
    if len(counts) != 1:
        return None
    return next(iter(counts))


def _render_tallies(
    grid: np.ndarray,
    *,
    use_multicolor_components: bool,
    apply_area_limit: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=not use_multicolor_components,
        return_format="dict",
    )

    frames: list[dict] = []
    sprites: list[dict] = []
    for comp in regions:
        if _is_frame(g, comp):
            frames.append(comp)
        else:
            sprites.append(comp)

    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = bg

    for frame in frames:
        frame_col = _frame_color(g, frame)
        if frame_col is None:
            continue

        num_patterns = 0
        for sprite in sprites:
            sprite_col = _dominant_color(g, sprite) if use_multicolor_components else int(sprite["color"])
            if sprite_col != frame_col:
                continue
            if apply_area_limit and int(sprite["area"]) > COUNTED_MAX_AREA:
                continue
            num_patterns += 1

        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, frame["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(num_patterns):
            cc = int(c1 - 2 - 2 * i)
            if cc > c0:
                out[r_mid, cc] = frame_col

    return out


def _solve_monochrome_color_only(grid: np.ndarray) -> np.ndarray:
    return _render_tallies(grid, use_multicolor_components=False, apply_area_limit=False)


def _solve_multicolor_no_area_limit(grid: np.ndarray) -> np.ndarray:
    return _render_tallies(grid, use_multicolor_components=True, apply_area_limit=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count only the small multicolor sprites whose dominant color matches a frame.

    Input structure:
    - The background is the most frequent color.
    - The left side contains hollow monochrome rectangular frames in distinct colors.
    - The right side contains isolated two-color sprites. Some are small enough to count,
      while others are larger distractors even though their dominant color still matches a frame.

    Rule:
    - Treat each two-color connected component as one sprite and infer its dominant color.
    - Remove every sprite from the output, but count only sprites whose dominant color
      matches a frame and whose total area is at most 5.
    - Draw one dot per counted sprite on the frame's middle row, spaced every 2 cells
      from the right edge toward the left.

    Key insight:
    - Color matching alone is no longer sufficient: a sprite must first be grouped across
      both colors to recover its main color, and then filtered by size before it can
      contribute to the tally inside its frame.
    """
    return _render_tallies(grid, use_multicolor_components=True, apply_area_limit=True)


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    dot_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, dot_cols] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    counts = [0] * n_frames
    order = rng.permutation(n_frames)
    remaining = total_match
    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        lo = max(0, remaining - MAX_DOTS_PER_FRAME * frames_left)
        hi = min(MAX_DOTS_PER_FRAME, remaining)
        val = int(rng.integers(lo, hi + 1))
        counts[int(idx)] = val
        remaining -= val
    return counts


def _crop_nonzero(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(patch[int(r0): int(r1) + 1, int(c0): int(c1) + 1], dtype=int)


def _main_cells_are_split(sprite: np.ndarray, main_color: int, bg: int) -> bool:
    comps = find_components(
        sprite,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[main_color],
        return_format="dict",
    )
    return len(comps) >= 2


def _make_sprite(
    rng: np.random.Generator,
    main_color: int,
    accent_color: int,
    bg: int,
    templates: list[np.ndarray],
) -> np.ndarray:
    template = np.array(templates[int(rng.integers(0, len(templates)))], dtype=int)
    patch = np.rot90(template, int(rng.integers(0, 4)))
    if int(rng.integers(0, 2)) == 1:
        patch = np.fliplr(patch)
    patch = _crop_nonzero(patch)

    sprite = np.full(patch.shape, bg, dtype=int)
    sprite[patch == 1] = main_color
    sprite[patch == 2] = accent_color
    return sprite


def _pick_distractor_indices(
    rng: np.random.Generator,
    counts: list[int],
    n_distractors: int,
) -> list[int]:
    preferred = [i for i, count in enumerate(counts) if 0 < count < MAX_DOTS_PER_FRAME]
    if not preferred:
        return []

    anchor = int(rng.choice(preferred))
    indices = [anchor]

    others = [i for i, count in enumerate(counts) if i != anchor and count < MAX_DOTS_PER_FRAME]
    rng.shuffle(others)
    for idx in others:
        if len(indices) >= n_distractors:
            break
        indices.append(int(idx))

    while len(indices) < n_distractors:
        indices.append(anchor)
    return indices


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_sprites = int(rng.integers(*TOTAL_SPRITES_RANGE))
    total_match = total_sprites - n_distractors

    if total_match <= 0 or total_match > n_frames * MAX_DOTS_PER_FRAME:
        return None

    palette_size = 1 + n_frames + N_ACCENT_COLORS
    palette = rng.choice(np.arange(10), size=palette_size, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1: 1 + n_frames]]
    accent_colors = [int(x) for x in palette[1 + n_frames:]]

    counts = _sample_counts(rng, n_frames, total_match)
    distractor_indices = _pick_distractor_indices(rng, counts, n_distractors)
    if len(distractor_indices) != n_distractors:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    small_main_colors = [color for color, count in zip(frame_colors, counts) for _ in range(count)]
    large_main_colors = [frame_colors[idx] for idx in distractor_indices]
    rng.shuffle(small_main_colors)
    rng.shuffle(large_main_colors)

    accent_stream = [accent_colors[i % len(accent_colors)] for i in range(total_sprites)]
    rng.shuffle(accent_stream)

    split_small = 0
    large_same_color_with_small = 0
    try:
        for main_color, accent_color in zip(large_main_colors, accent_stream[:len(large_main_colors)]):
            sprite = _make_sprite(rng, main_color, accent_color, bg, LARGE_SPRITE_TEMPLATES)
            if int(np.count_nonzero(sprite != bg)) <= COUNTED_MAX_AREA:
                return None
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
            if main_color in small_main_colors:
                large_same_color_with_small += 1

        offset = len(large_main_colors)
        for main_color, accent_color in zip(small_main_colors, accent_stream[offset: offset + len(small_main_colors)]):
            sprite = _make_sprite(rng, main_color, accent_color, bg, SMALL_SPRITE_TEMPLATES)
            if int(np.count_nonzero(sprite != bg)) > COUNTED_MAX_AREA:
                return None
            if _main_cells_are_split(sprite, main_color, bg):
                split_small += 1
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

    if split_small != len(small_main_colors):
        return None
    if large_same_color_with_small <= 0:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_solve_monochrome_color_only(inp), out):
        return None
    if np.array_equal(_solve_multicolor_no_area_limit(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
