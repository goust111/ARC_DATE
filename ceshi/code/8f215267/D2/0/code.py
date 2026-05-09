# Variant of 8f215267
# Dimension: D2 — 源对象：精灵类型 (src)
# Change: 小型单色连通精灵，面积2-8像素，外接矩形最大4x3，散布在框架右侧区域... → 多色精灵

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - The left side still contains 5x11 hollow monochrome frames.
#    - The right side no longer contains monochrome sprites. Each sprite is a single
#      4-connected object made of exactly two colors: a dominant "main color" and a
#      minority "accent color".
#    - Matching uses the sprite's main color, not the accent color.
#    - Input -> output behavior is unchanged at the frame side: all right-side sprites
#      disappear, and each frame receives as many dots as there are sprites whose main
#      color matches the frame color.
#    - Key local trace:
#      old logic on sprite [main accent main] saw two monochrome main-color pieces and
#      would count them as two patterns; new logic must first group the full multicolor
#      component, then count color frequencies inside it, then treat it as one sprite.
#      This is the precise process-level change introduced by D2.
# b) Complexity calibration
#    - The variant must be visually stronger than "a mostly monochrome shape with one
#      harmless accent pixel". To enforce that, every generated sprite uses templates
#      where the main-color cells are split into multiple monochrome components that
#      are connected only through accent cells.
#    - The sprite family is still small and repeated: a finite template library with
#      3-5 foreground cells and bbox at most 3x3. This keeps the task learnable from
#      2-4 examples instead of exploding into too many arbitrary shapes.
# c) Implicit assumption audit against the original main()
#    - Original `find_components(... monochromatic=True)` assumed one connected object
#      equals one color. That fails here because one sprite spans two colors.
#    - Original `patterns.append(comp)` and `comp["color"]` assumed the component color
#      was the sprite identity. That also fails: the solver must derive a dominant color
#      by counting colors inside a multicolor component.
#    - Original frame detection used shape only. In this variant frames remain
#      monochrome, so the new solver additionally requires `len(colors)==1` for frames,
#      ensuring multicolor sprites cannot be mistaken for frames.
# d) Placement legality
#    - Frames remain vertically stacked on the left and never touch sprites.
#    - Sprites are placed in a dedicated right canvas with padding 1 and 8-neighbor
#      exclusion, so two multicolor sprites cannot merge into one connected component.
#    - Accent colors are chosen from colors disjoint from background, frame colors, and
#      distractor main colors, so accent pixels never create accidental color matches.
# e) Visual observability
#    - A human can now see that right-side objects are more complex: they contain two
#      colors, and the repeated frame dots depend on the majority color among those two.
#    - Because many sprites look like "broken" main-color pieces bridged by accent
#      pixels, examples visibly support the idea that the whole two-color object counts
#      as one unit.
# f) Visual neatness
#    - The scene remains partitioned into a clean left frame region and a clean right
#      sprite region. No marks are added inside sprites, and the output still only adds
#      dots inside frames, so the grid stays legible.
# g) Color inferability
#    - Only two accent colors are used globally per sample, repeated across many sprites.
#      This gives a visible structure: accents vary, but the counted color is the color
#      that occupies more cells inside each sprite.
# 9) New assumptions introduced by the new main()
#    - Each sprite must have a unique dominant color; generator guarantees `main > accent`.
#    - Sprites must remain isolated as 4-connected components; generator enforces padded
#      placement in the right region.
#    - Frames must stay monochrome hollow rectangles; generator preserves the original
#      frame construction exactly.


def _component_color_counts(grid: np.ndarray, comp: dict) -> Counter[int]:
    counts: Counter[int] = Counter()
    for r, c in comp["coords"]:
        counts[int(grid[int(r), int(c)])] += 1
    return counts


def _frame_color(grid: np.ndarray, comp: dict) -> int | None:
    counts = _component_color_counts(grid, comp)
    if len(counts) != 1:
        return None
    return next(iter(counts))


def _dominant_color(grid: np.ndarray, comp: dict) -> int | None:
    counts = _component_color_counts(grid, comp)
    if not counts:
        return None
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    if len(winners) != 1:
        return None
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count multicolor sprites by their dominant color and display the tally in frames.

    Input structure:
    - bg = most common color.
    - Hollow monochrome rectangular frames in various colors.
    - Small multicolor sprites scattered outside frames, each using exactly two colors:
      a dominant main color and a minority accent color.

    Rule:
    - Remove all right-side sprites from the grid.
    - For each frame of color C, count how many sprites have dominant color C.
    - Draw that many dots along the frame's center row, spaced by 2, starting from the right.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        colors = _component_color_counts(g, comp)
        is_frame = (len(colors) == 1) and (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        col = _frame_color(g, f)
        if col is None:
            continue
        num_patterns = sum(1 for p in patterns if _dominant_color(g, p) == col)
        if num_patterns <= 0:
            continue

        r0, c0, _r1, c1 = map(int, f["bbox"])
        r_mid = r0 + FRAME_H // 2
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col

    return out


# --------------- generator (codex D2 variant) ---------------

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
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)
N_ACCENT_COLORS = 2

# --- sprite placement ---
SPRITE_PADDING = 1
MAX_ATTEMPTS = 300

# Template legend: 0 = bg, 1 = main color, 2 = accent color.
# Every template is a single multicolor component but has multiple disconnected
# main-color monochrome subcomponents, forcing the solver to group across colors.
SPRITE_TEMPLATES = [
    np.array([[1, 2, 1]], dtype=int),
    np.array([[1], [2], [1]], dtype=int),
    np.array([[1, 2], [0, 1]], dtype=int),
    np.array([[1, 0], [2, 1]], dtype=int),
    np.array([[1, 2, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [2, 1], [1, 0]], dtype=int),
    np.array([[1, 2, 1], [0, 2, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 2, 0], [0, 1, 2], [0, 0, 1]], dtype=int),
]


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


def _make_sprite(rng: np.random.Generator, main_color: int, accent_color: int, bg: int) -> np.ndarray:
    template = np.array(SPRITE_TEMPLATES[int(rng.integers(0, len(SPRITE_TEMPLATES)))], dtype=int)
    patch = np.rot90(template, int(rng.integers(0, 4)))
    if int(rng.integers(0, 2)) == 1:
        patch = np.fliplr(patch)
    patch = _crop_nonzero(patch)

    sprite = np.full(patch.shape, bg, dtype=int)
    sprite[patch == 1] = main_color
    sprite[patch == 2] = accent_color
    return sprite


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    palette_size = 1 + n_frames + n_distractors + N_ACCENT_COLORS
    palette = rng.choice(np.arange(10), size=palette_size, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1: 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames: 1 + n_frames + n_distractors]]
    accent_colors = [int(x) for x in palette[1 + n_frames + n_distractors:]]

    counts = _sample_counts(rng, n_frames, total_match)
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
    pattern_main_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_main_colors.extend(distractor_colors)
    rng.shuffle(pattern_main_colors)

    accent_assignments = [accent_colors[i % len(accent_colors)] for i in range(len(pattern_main_colors))]
    rng.shuffle(accent_assignments)

    split_sprite_count = 0
    try:
        for main_color, accent_color in zip(pattern_main_colors, accent_assignments):
            sprite = _make_sprite(rng, main_color, accent_color, bg)
            if _main_cells_are_split(sprite, main_color, bg):
                split_sprite_count += 1
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

    if split_sprite_count != len(pattern_main_colors):
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
