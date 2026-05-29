# Variant of 8f215267
# Dimension: D11 — 点的颜色 (transform)
# Change: 绘制的点颜色与对应框架颜色相同... → 点使用背景色
#
# Design analysis
# variant:
#   source_task: none
#   concrete_visualization:
#     input:
#       - Left side contains stacked 5x11 hollow frames in distinct colors.
#       - Right side contains scattered small monochrome sprites.
#       - Frame interiors are background-colored and contain no visible dots.
#     output:
#       - Sprites disappear.
#       - Matching-count information is no longer shown as colored interior dots.
#       - Because "draw the dot in background color" would be invisible inside the already-bg interior,
#         the tally is externalized as background holes punched into the top border of each frame.
#       - Hole columns reuse the original right-to-left spacing pattern: c1-2, c1-4, c1-6, c1-8.
#   behavior_trace:
#     - Example: a red frame has 3 red sprites.
#     - Original rule: erase sprites, then color (r_mid, c1-2), (r_mid, c1-4), (r_mid, c1-6) red.
#     - New rule: erase sprites, keep the center row empty, and instead set
#       (r0, c1-2), (r0, c1-4), (r0, c1-6) to background, creating 3 visible holes.
#     - Difference: counting-by-color is unchanged; only the display channel moves from interior fill
#       to border subtraction so the D11 color inversion becomes observable.
#   complexity_calibration:
#     - The count vocabulary stays small (0-4 per frame), so the mapping remains learnable.
#     - Using one fixed border side avoids too many degrees of freedom.
#     - Holes are repeated across multiple frames because total matching sprites is always >= 3.
#   implicit_assumptions_review:
#     - find_components(... monochromatic=True) assumes each frame is still a single connected component.
#       Top-border holes do not break connectivity because the side borders and bottom border still connect the ring.
#     - is_frame = (h>=5 and w>=5 and area < 0.7*h*w) still holds:
#       intact frames have area 28; holed frames have area 24-28; both are well below 38.5.
#     - Removing patterns by writing bg remains safe because sprites are generated strictly outside frames.
#     - Counting by exact color still assumes frame colors are distinct and distractor colors do not match any frame.
#     - The old write step relied on "writing color onto bg interior is visible"; that premise is false here,
#       so the write target changes to border cells and the written value changes to bg.
#   placement_legality:
#     - Hole positions avoid corners, so the frame bbox and identity stay stable.
#     - Max 4 holes fit on the top border without touching frame corners.
#     - Sprites remain in the right region with padding, so they never merge with frames or each other.
#   visual_observability:
#     - Across examples, the solver-visible change is consistent: more matching sprites means more top-edge holes.
#     - The change is human-visible because the border itself changes shape; it does not rely on hidden color ids.
#   cleanliness:
#     - Only one mark type occupies the frame region in the output: missing border cells.
#     - No extra overlay is added elsewhere, so frames remain easy to parse.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


def _marker_cols_from_right(c0: int, c1: int, n_marks: int) -> list[int]:
    """Return the original right-to-left tally columns, excluding frame corners."""
    cols: list[int] = []
    for i in range(int(n_marks)):
        cc = int(c1 - 2 - 2 * i)
        if cc > int(c0):
            cols.append(cc)
    return cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and encode the tally as border holes.

    Input structure:
    - bg = most common color.
    - Hollow rectangular frames in various colors.
    - Small monochrome pattern sprites scattered outside frames.

    Rule:
    - Remove all small patterns from the grid.
    - For each frame of color C, count how many patterns share color C.
    - Since a "dot in background color" would be invisible inside the frame, represent the tally by
      carving background holes in the frame's top border at the original right-to-left dot positions.
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

    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, _r1, c1 = map(int, f["bbox"])
        for cc in _marker_cols_from_right(c0, c1, num_patterns):
            out[int(r0), int(cc)] = bg

    return out


# --------------- generator (D11 variant) ---------------

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

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_patch(color: int, bg: int, n_holes: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with bg holes on the top border."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for cc in _marker_cols_from_right(0, FRAME_W - 1, n_holes):
        patch[0, int(cc)] = bg
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """Distribute total_match sprites across n_frames, each capped at MAX_DOTS_PER_FRAME."""
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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

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
    for r0, count, color in zip(frame_rows, counts, frame_colors):
        for cc in _marker_cols_from_right(FRAME_C0, FRAME_C0 + FRAME_W - 1, count):
            inp[int(r0), int(cc)] = int(color)

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
