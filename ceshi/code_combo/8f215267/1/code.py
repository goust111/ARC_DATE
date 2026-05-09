# Combo variant of 8f215267 (index 1)
# combo_spec: D1/0+D2/0
# Name: 形状匹配+多色精灵

from __future__ import annotations

from collections import Counter

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste, random_free_location
except ImportError:  # pragma: no cover - robustness for direct file execution
    import sys

    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste, random_free_location


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
   - Backbone kept from the original/spec:
     left side = vertically stacked 5x11 hollow frames; right side = small scattered sprites;
     output still removes right-side sprites and writes right-aligned dots inside each frame.
   - D1 contribution:
     each frame now contains a same-color marker shape attached to the upper inner wall.
     A sprite belongs to a frame iff the sprite's full occupied shape exactly matches that
     frame marker shape. Frame color is no longer a matching cue.
   - D2 contribution:
     every right-side sprite is one 4-connected object using exactly two colors. The main
     color occupies more cells than the accent color, but matching ignores both colors and
     uses the sprite's whole silhouette. This forces the solver to group a multicolor object
     first, then compare its geometry.
   - Local trace on one frame:
     old task: blue frame + three blue monochrome sprites -> draw three blue dots.
     combo task: blue frame with an embedded "long-L" marker + three multicolor sprites whose
     occupied cells form that same long-L shape, in colors unrelated to blue -> draw three
     blue dots. A color-based count fails, and a monochromatic component extractor also fails
     because each sprite's main-color cells are split by the accent bridge.

b) Complexity calibration
   - The marker catalog is small and reused: seven exact shapes, from which 3-4 distinct
     active frame markers are sampled per grid. This keeps the task learnable from a few
     examples instead of creating arbitrary one-off shapes.
   - All active marker shapes are asymmetric and orientation-sensitive. External sprites are
     copies in the same orientation, so the signal is visible but not overloaded by extra
     rotation rules.
   - Multicolor sprites are not "almost monochrome with a harmless speck": the accent cell is
     chosen as an articulation point, so removing color information incorrectly splits the
     object into multiple monochrome pieces.

c) Implicit assumption audit against the original main()
   - Original `find_components(... monochromatic=True)` breaks because one logical sprite now
     spans two colors. The combo solver must use `monochromatic=False`.
   - Original `comp["color"]` on patterns breaks because pattern identity is not its color
     anymore. The combo solver instead computes a normalized occupied-cell shape signature.
   - Original frame detection relied only on size / sparsity. Here frames remain monochrome,
     while sprites are multicolor, so the solver explicitly requires one color for frames.
   - Adding markers creates interior foreground cells inside a frame component. The solver
     assumes those interior cells are the marker and extracts them from the frame interior.
   - Drawing dots on the center row must not overwrite markers, so all markers are anchored
     near the upper-left interior and avoid the right-aligned dot columns.

d) Placement legality
   - Frame markers are strictly inside the 5x11 frame bbox, but each touches the top border
     by 4-neighborhood so the frame and marker remain a single monochrome component.
   - Right-side sprites are placed on a separate canvas with padding 1 and 8-neighbor
     exclusion, preventing overlap and accidental merging into one component.
   - Distractor shapes are sampled from the inactive portion of the shape catalog, so no
     distractor accidentally matches an active frame marker.

e) Visual observability
   - A human can compare repeated frame markers to repeated right-side silhouettes. Because
     sprite colors are disjoint from frame colors, color grouping immediately looks wrong.
   - The multicolor construction is visible: the outside objects plainly contain two colors,
     yet the output depends only on the outline matching the embedded marker.

f) Visual neatness
   - The scene stays partitioned: frames occupy the left strip, sprites occupy the right strip.
     Nothing overlaps frames, and dots only appear on the center row in the output.
   - Marker shapes live in a small upper-left pocket of each frame, so the tally dots remain
     visually separated from the matching cue.

g) Change inferability
   - The color system has controlled entropy: frame colors are unique, while sprite main colors
     come from a small shared palette and accent colors come from two global repeated colors.
     This makes it obvious that colors vary independently of which frame receives dots.
   - Shapes, not numeric color ids, provide the stable structure. The repeated marker-sprite
     correspondences are enough to infer the rule from 2-4 examples.

9) New assumptions introduced by the new main()
   - Every frame contains exactly one attached interior marker and no input dots.
   - Every right-side sprite is a single multicolor 4-connected component whose occupied
     silhouette is one of the catalog shapes.
   - Each sprite has a unique dominant main color and at least one accent bridge cell; this
     is not used for matching, but it guarantees the multicolor-object interpretation is real.
   - The generator enforces all of these assumptions explicitly before accepting a sample.
"""


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
N_FRAMES_RANGE = (3, 5)        # 3-4 frames
N_DISTRACTORS_RANGE = (1, 3)   # 1-2 distractors
TOTAL_PATTERNS_RANGE = (5, 11) # 5-10 total sprites
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)
N_ACCENT_COLORS = 2
N_MAIN_COLORS_RANGE = (2, 4)   # 2-3 shared main colors

# --- placement / retries ---
SPRITE_PADDING = 1
MAX_ATTEMPTS = 500


# Catalog entries:
# - cells: normalized occupied cells used for matching
# - accent: one occupied cell recolored as the accent in outside sprites
# - marker_anchor: top-left insertion point inside the 5x11 frame patch
SHAPE_LIBRARY: dict[str, dict[str, object]] = {
    "L4a": {
        "cells": [(0, 0), (0, 1), (0, 2), (1, 0)],
        "accent": (0, 0),
        "marker_anchor": (1, 1),
    },
    "L4b": {
        "cells": [(0, 0), (0, 1), (1, 1), (2, 1)],
        "accent": (1, 1),
        "marker_anchor": (1, 2),
    },
    "T4": {
        "cells": [(0, 0), (0, 1), (0, 2), (1, 1)],
        "accent": (0, 1),
        "marker_anchor": (1, 2),
    },
    "J4": {
        "cells": [(0, 0), (0, 1), (1, 0), (2, 0)],
        "accent": (1, 0),
        "marker_anchor": (1, 1),
    },
    "T5a": {
        "cells": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        "accent": (0, 1),
        "marker_anchor": (1, 2),
    },
    "T5b": {
        "cells": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
        "accent": (0, 2),
        "marker_anchor": (1, 1),
    },
    "P5": {
        "cells": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
        "accent": (1, 0),
        "marker_anchor": (1, 1),
    },
}


def _shape_signature(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    arr = np.array(coords, dtype=int)
    if arr.size == 0:
        return (0, 0, ())
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    rel = [(int(r - r0), int(c - c0)) for r, c in arr]
    h = int(arr[:, 0].max() - r0 + 1)
    w = int(arr[:, 1].max() - c0 + 1)
    return (h, w, tuple(sorted(rel)))


def _component_color_counts(grid: np.ndarray, comp: dict) -> Counter[int]:
    counts: Counter[int] = Counter()
    for r, c in comp["coords"]:
        counts[int(grid[int(r), int(c)])] += 1
    return counts


def _frame_color(grid: np.ndarray, comp: dict) -> int | None:
    counts = _component_color_counts(grid, comp)
    if len(counts) != 1:
        return None
    return int(next(iter(counts)))


def _component_signature(comp: dict) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, _r1, _c1 = map(int, comp["bbox"])
    rel = [(int(r - r0), int(c - c0)) for r, c in comp["coords"]]
    return _shape_signature(rel)


def _frame_marker_signature(comp: dict) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    interior = [
        (int(r - r0), int(c - c0))
        for r, c in comp["coords"]
        if int(r0) < int(r) < int(r1) and int(c0) < int(c) < int(c1)
    ]
    return _shape_signature(interior)


def _is_frame(grid: np.ndarray, comp: dict) -> bool:
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    area = int(comp["area"])
    colors = _component_color_counts(grid, comp)
    return (len(colors) == 1) and (h >= 5) and (w >= 5) and (area < (h * w * 0.7))


def _sprite_patch(shape_name: str, main_color: int, accent_color: int, bg: int) -> np.ndarray:
    spec = SHAPE_LIBRARY[shape_name]
    cells = [(int(r), int(c)) for r, c in spec["cells"]]
    accent = tuple(int(v) for v in spec["accent"])
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h = max(rows) + 1
    w = max(cols) + 1
    patch = np.full((h, w), bg, dtype=int)
    for r, c in cells:
        patch[int(r), int(c)] = accent_color if (r, c) == accent else main_color
    return patch


def _frame_patch(color: int, bg: int, n_dots: int, marker_name: str) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    spec = SHAPE_LIBRARY[marker_name]
    anchor_r, anchor_c = map(int, spec["marker_anchor"])
    for dr, dc in spec["cells"]:
        patch[anchor_r + int(dr), anchor_c + int(dc)] = color

    mid_r = FRAME_H // 2
    for i in range(int(n_dots)):
        dot_c = int(FRAME_W - 3 - 2 * i)
        patch[int(mid_r), dot_c] = color
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count multicolor sprites by shape and represent the tally as dots inside frames.

    Input structure:
    - bg = most common color.
    - Hollow monochrome 5x11 rectangular frames stand on the left.
    - Each frame contains a same-color marker shape attached inside its upper-left region.
    - Small right-side sprites are 4-connected multicolor objects, each using a dominant main
      color plus an accent color.

    Rule:
    - Remove all outside sprites from the grid.
    - For each frame, read the occupied shape of its interior marker.
    - Group each outside sprite as one multicolor component, ignore its colors, and compare its
      full occupied shape against the frame marker.
    - Draw one frame-colored dot per matching sprite on the frame's center row, spaced by 2
      from the right edge.

    Key insight: Colors on the outside objects are deliberately misleading twice over: they do
    not match frame colors, and they also split a single sprite into multiple monochrome pieces.
    The reliable invariant is the full silhouette shared between a frame's embedded marker and
    the multicolor sprites that belong to it.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_frame(g, comp):
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    pattern_sigs = [_component_signature(p) for p in patterns]

    for f in frames:
        frame_color = _frame_color(g, f)
        if frame_color is None:
            continue
        marker_sig = _frame_marker_signature(f)
        num_patterns = sum(1 for sig in pattern_sigs if sig == marker_sig)
        if num_patterns <= 0:
            continue

        r0, c0, _r1, c1 = map(int, f["bbox"])
        r_mid = int(r0 + FRAME_H // 2)
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = int(frame_color)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    counts = _sample_counts(rng, n_frames, total_match)
    if sum(v > 0 for v in counts) < 2:
        return None

    shape_names = list(rng.choice(list(SHAPE_LIBRARY.keys()), size=n_frames, replace=False))

    n_main_colors = int(rng.integers(*N_MAIN_COLORS_RANGE))
    palette_size = 1 + n_frames + n_main_colors + N_ACCENT_COLORS
    palette = list(rng.choice(np.arange(10), size=palette_size, replace=False))
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    main_colors = [int(x) for x in palette[1 + n_frames : 1 + n_frames + n_main_colors]]
    accent_colors = [int(x) for x in palette[1 + n_frames + n_main_colors :]]

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count, shape_name) in enumerate(zip(frame_colors, counts, shape_names)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count, shape_name), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    pattern_specs: list[tuple[str, int, int]] = []
    for shape_name, count in zip(shape_names, counts):
        for _ in range(count):
            main_color = int(main_colors[int(rng.integers(0, len(main_colors)))])
            accent_color = int(accent_colors[int(rng.integers(0, len(accent_colors)))])
            pattern_specs.append((shape_name, main_color, accent_color))

    distractor_names = [name for name in SHAPE_LIBRARY if name not in shape_names]
    for _ in range(n_distractors):
        shape_name = str(distractor_names[int(rng.integers(0, len(distractor_names)))])
        main_color = int(main_colors[int(rng.integers(0, len(main_colors)))])
        accent_color = int(accent_colors[int(rng.integers(0, len(accent_colors)))])
        pattern_specs.append((shape_name, main_color, accent_color))
    rng.shuffle(pattern_specs)

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    split_sprite_count = 0
    try:
        for shape_name, main_color, accent_color in pattern_specs:
            sprite = _sprite_patch(shape_name, main_color, accent_color, bg)
            if not _main_cells_are_split(sprite, main_color, bg):
                return None
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

    if split_sprite_count != len(pattern_specs):
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
