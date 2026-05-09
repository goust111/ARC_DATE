# Variant of 8f215267
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架颜色 (s.color == d.color)，同色精灵被归属于对应框架... → 形状匹配

from __future__ import annotations

"""
Design analysis for the D1 variant
==================================

spec_variant:
  unchanged_backbone:
    - grid width stays 23; frames remain vertically stacked on the left
    - each frame is still a 5x11 hollow rectangle
    - external sprites still live in the right-side region
    - output still removes all external sprites and writes right-aligned dots on the frame center row
  D1 change:
    - each frame now carries a small same-color marker shape attached to its upper-left inner wall
    - external sprites are matched to frames by exact shape equality with that marker, not by color
    - sprite colors are sampled from a palette disjoint from frame colors, so old color matching is visibly wrong

  concrete pixel form:
    - frame component = rectangle perimeter + 2-3 interior cells near the left/top interior edge
    - because the marker touches the border by 4-neighborhood, frame+marker remains one monochrome component
    - external sprites are tight-cropped copies of those marker shapes, scattered on the right
    - distractors are extra sprite shapes that do not match any frame marker

  behavior trace:
    - original rule at one frame:
      frame color = red, outside there are 3 red sprites -> draw 3 red dots
    - new rule at one frame:
      frame color = red, marker = vertical domino, outside there are 3 sprites shaped as vertical domino
      in arbitrary non-red colors -> draw 3 red dots
    - difference:
      counting key moved from `p.color == frame.color` to `shape(p) == marker_shape(frame)`

  complexity calibration:
    - marker shapes come from a small catalog of 5 exact shapes, enough for 3-4 unique frames
    - shapes are asymmetric enough that the matching cue is visible in 2-4 examples
    - colors of outside sprites are intentionally unhelpful but not chaotic: only a small shared sprite palette is used

  old-main assumptions that break under D1:
    - `num_patterns = sum(... if p["color"] == col)` relied on frame/pattern color equivalence; no longer valid
    - treating every non-frame component as a pattern is still fine, but matching must use normalized geometry
    - adding a marker must not split the frame into two components; therefore the marker is attached to the frame border
    - dots must not overwrite the marker; therefore the marker is kept in the upper-left interior area while dots stay right-aligned

  placement legality:
    - marker cells stay strictly inside the frame bbox and touch the frame border, so bbox remains 5x11
    - marker cells never occupy any dot column on the center row
    - right-side sprites are placed with random_free_location and padding, so they do not overlap or touch

  visual observability:
    - a human can compare each embedded frame marker with repeated copies on the right
    - since sprite colors never match frame colors in generated data, color grouping fails immediately and shape grouping is the only consistent rule

  new solver assumptions introduced here:
    - every frame contains exactly one attached interior marker shape in the input
    - external sprites use the same orientation as the frame marker; no rotation/flip equivalence is used
    - generator guarantees distractor shapes are outside the marker catalog used by active frames
"""

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste, random_free_location
except ImportError:  # pragma: no cover - robustness for direct file execution
    import sys

    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste, random_free_location


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
TOTAL_PATTERNS_RANGE = (5, 11) # 5-10 total right-side sprites
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- placement / retries ---
SPRITE_PADDING = 1
MAX_ATTEMPTS = 400


# Marker cells are specified in frame-local coordinates inside the 5x11 patch.
# They all touch the frame border and stay away from the right-aligned dot positions.
FRAME_MARKERS: dict[str, list[tuple[int, int]]] = {
    "h2": [(1, 1), (1, 2)],
    "v2": [(1, 1), (2, 1)],
    "L3": [(1, 1), (1, 2), (2, 1)],
    "h3": [(1, 1), (1, 2), (1, 3)],
    "v3": [(1, 1), (2, 1), (3, 1)],
}


DISTRACTOR_SHAPES: dict[str, list[tuple[int, int]]] = {
    "sq2": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "zig3": [(0, 0), (0, 1), (1, 1)],
    "T4": [(0, 1), (1, 0), (1, 1), (1, 2)],
    "S4": [(0, 1), (0, 2), (1, 0), (1, 1)],
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


def _crop_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    sig = _shape_signature(cells)
    return list(sig[2])


MARKER_SPRITES: dict[str, list[tuple[int, int]]] = {
    name: _crop_cells(cells) for name, cells in FRAME_MARKERS.items()
}


def _sprite_patch(shape_cells: list[tuple[int, int]], color: int, bg: int) -> np.ndarray:
    rows = [r for r, _ in shape_cells]
    cols = [c for _, c in shape_cells]
    h = max(rows) + 1
    w = max(cols) + 1
    patch = np.full((h, w), bg, dtype=int)
    for r, c in shape_cells:
        patch[int(r), int(c)] = color
    return patch


def _frame_patch(color: int, bg: int, n_dots: int, marker_name: str) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for r, c in FRAME_MARKERS[marker_name]:
        patch[int(r), int(c)] = color
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


def _frame_marker_signature(comp: dict) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    interior = [
        (int(r - r0), int(c - c0))
        for r, c in comp["coords"]
        if int(r0) < int(r) < int(r1) and int(c0) < int(c) < int(c1)
    ]
    return _shape_signature(interior)


def _component_signature(comp: dict) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, _r1, _c1 = map(int, comp["bbox"])
    rel = [(int(r - r0), int(c - c0)) for r, c in comp["coords"]]
    return _shape_signature(rel)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count shape-matching patterns per frame and represent the tally as dots inside.

    Input structure:
    - bg = most common color.
    - Hollow 5x11 rectangular frames on the left.
    - Each frame contains a small attached marker shape inside its upper-left interior.
    - Small monochrome sprites are scattered outside the frames.

    Rule:
    - Remove all outside sprites from the grid.
    - For each frame, read its interior marker shape.
    - Count how many outside sprites have the exact same shape, regardless of color.
    - Draw that many dots along the frame's center row, spaced by 2, starting from the right.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    pattern_sigs = [_component_signature(p) for p in patterns]

    for f in frames:
        marker_sig = _frame_marker_signature(f)
        num_patterns = sum(1 for sig in pattern_sigs if sig == marker_sig)
        if num_patterns <= 0:
            continue
        r0, c0, _r1, c1 = map(int, f["bbox"])
        r_mid = int(r0 + FRAME_H // 2)
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = int(f["color"])

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

    marker_names = list(rng.choice(list(FRAME_MARKERS.keys()), size=n_frames, replace=False))

    n_sprite_colors = int(rng.integers(2, 4))
    palette = list(rng.choice(np.arange(10), size=1 + n_frames + n_sprite_colors, replace=False))
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    sprite_colors = [int(x) for x in palette[1 + n_frames :]]

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count, marker_name) in enumerate(zip(frame_colors, counts, marker_names)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count, marker_name), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    pattern_specs: list[tuple[str, int]] = []
    for marker_name, count in zip(marker_names, counts):
        for _ in range(count):
            color = int(sprite_colors[int(rng.integers(0, len(sprite_colors)))])
            pattern_specs.append((marker_name, color))

    distractor_names = [name for name in DISTRACTOR_SHAPES if name not in marker_names]
    for _ in range(n_distractors):
        shape_name = distractor_names[int(rng.integers(0, len(distractor_names)))]
        color = int(sprite_colors[int(rng.integers(0, len(sprite_colors)))])
        pattern_specs.append((shape_name, color))
    rng.shuffle(pattern_specs)

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    try:
        for shape_name, color in pattern_specs:
            if shape_name in MARKER_SPRITES:
                sprite = _sprite_patch(MARKER_SPRITES[shape_name], color, bg)
            else:
                sprite = _sprite_patch(DISTRACTOR_SHAPES[shape_name], color, bg)
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
