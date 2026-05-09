# Variant of 8f215267
# Dimension: D3 — 目标对象：框架结构 (dst)
# Change: 5x11的空心矩形框架，仅由周边一圈像素组成，垂直排列在网格左半部分，间隔2行... → 水平排列框架

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

"""
Design analysis for the D3 variant
==================================

a) Concrete behavior and trace
   - Original task: several 5x11 hollow frames are stacked vertically on the left.
     Same-color sprites are scattered to the right. The output removes sprites and
     writes the count back as horizontal dots on each frame's middle row, from right
     to left with a step of 2.
   - Variant: keep the same frame shape and color-matching rule, but move the target
     frames into a horizontal row near the top. Sprites are now scattered in the area
     below the frames. To make the changed destination structure visibly matter, the
     tally also rotates: dots are drawn on the frame's middle column, bottom-up.
   - Trace on one frame:
     original input  -> hollow red frame at left, three red sprites in right region
     original output -> same frame gains three red dots on center row
     variant input   -> hollow red frame in top row of frames, three red sprites below
     variant output  -> same frame gains three red dots on center column, rising upward
   - The difference is therefore visible in both space usage and mark direction.

b) Complexity calibration
   - If we kept 3-4 frames of width 11 and only changed stacking direction, the grid
     would exceed common ARC size limits. The minimal adaptation is to use 2 frames.
   - Two frames are still enough because each example repeats the same count-by-color
     mechanism twice, and distractor sprites remain present to prevent a trivial
     "remove everything below" interpretation.
   - Counts are capped at 3 because a 5x11 frame only has 3 interior rows on a single
     center column. We require total matching sprites in [3, 6], so the tally is
     usually nontrivial and often shows a 2- or 3-cell vertical run.

c) Hidden assumptions audited against the old main()
   - Component split by size/sparsity: still valid, because frames remain the only
     large hollow rectangles and sprites stay at most 4x3.
   - "Patterns are outside frames": still valid, but their region moves from right to
     below. The solver must not depend on right-side placement, so counting remains
     global by color.
   - "Draw on center row from the right": no longer valid. This is the main D3-linked
     assumption and must be replaced.
   - "Frame layout does not matter": no longer safe. The new solver explicitly infers
     whether the frame centers spread more horizontally or vertically; horizontal frame
     layout triggers vertical tallies, while vertical layout keeps the legacy rule.

d) Placement legality
   - Frames occupy row 1 and columns [1,11] plus [14,24], leaving margins and staying
     inside a 27-column grid.
   - Sprite region starts two blank rows below the frames, so sprites cannot merge with
     frames under 4-connectivity.
   - Sprites are placed with 8-neighbor padding inside their own canvas to avoid
     overlap, accidental touching, or ambiguous components.

e) Visual observability
   - The changed rule is visible from just a few examples because:
     1. frames are no longer a vertical stack;
     2. sprites are no longer to the right;
     3. the tally direction rotates from horizontal to vertical.
   - This avoids a degenerate case where only the coarse layout changed but the output
     mark geometry still looked essentially the same.

f) Visual cleanliness
   - Frames stay in a clean top band; sprites stay in a clean lower band.
   - No object types overlap the same area, so the solver cue is easy to parse.

g) Color inferability
   - Colors are not ordered by value. Matching is purely by visible equality between a
     frame color and scattered sprite colors. Distractors have colors unused by frames.

h) Variant metric for non-degeneration
   - A useful quantitative check is: on generated samples, how often does the new
     output differ from the old horizontal-dot readout on the exact same input?
   - For this generator it should be 100%, because whenever at least one tally is
     positive, a vertical center-column readout cannot equal the old center-row
     right-to-left readout.

i) New solver assumptions and generator guarantees
   - main() assumes frames are hollow 5x11 rectangles: generator guarantees exact size.
   - main() assumes the frame layout is consistent across the sample: generator places
     both frames on the same row, so the layout inference is stable.
   - main() assumes sprites are separate monochrome components: generator uses connected
     monochrome polyominoes with spacing.
"""

# --- frame geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_AREA = 2 * (FRAME_H + FRAME_W) - 4
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
N_FRAMES = 2

# --- grid geometry ---
GRID_W = 27
SPRITE_R0 = FRAME_R0 + FRAME_H + FRAME_GAP
SPRITE_C0 = 1
SPRITE_W = GRID_W - 2
SPRITE_REGION_H_RANGE = (10, 15)
BOTTOM_PAD_RANGE = (0, 4)

# --- structural counts ---
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_MATCH_RANGE = (3, 7)
MAX_DOTS_PER_FRAME = 3

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 8)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _is_frame(comp: dict) -> bool:
    b = comp["bbox"]
    return (
        bbox_h(b) == FRAME_H
        and bbox_w(b) == FRAME_W
        and int(comp["area"]) == FRAME_AREA
    )


def _frame_layout_is_horizontal(frames: list[dict]) -> bool:
    if len(frames) <= 1:
        return True
    row_centers = [((int(f["bbox"][0]) + int(f["bbox"][2])) // 2) for f in frames]
    col_centers = [((int(f["bbox"][1]) + int(f["bbox"][3])) // 2) for f in frames]
    row_span = max(row_centers) - min(row_centers)
    col_span = max(col_centers) - min(col_centers)
    return col_span >= row_span


def _draw_tally(out: np.ndarray, frame: dict, count: int, horizontal_layout: bool) -> None:
    if count <= 0:
        return
    color = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])
    if horizontal_layout:
        c_mid = (c0 + c1) // 2
        for i in range(int(count)):
            rr = int(r1 - 1 - i)
            if rr > int(r0):
                out[rr, c_mid] = color
    else:
        r_mid = (r0 + r1) // 2
        for i in range(int(count)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[r_mid, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites per frame and write the tally back inside each frame.

    Variant rule:
    - Frames are still matched to sprites by color.
    - When the frames themselves are horizontally arranged, the tally is drawn on the
      frame's center column from the bottom upward.
    - Any non-frame monochrome component is treated as a sprite and removed.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_frame(comp):
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    horizontal_layout = _frame_layout_is_horizontal(frames)
    for frame in frames:
        count = sum(1 for p in patterns if int(p["color"]) == int(frame["color"]))
        _draw_tally(out, frame, count, horizontal_layout)

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    c_mid = FRAME_W // 2
    for i in range(int(n_dots)):
        patch[FRAME_H - 2 - i, c_mid] = color
    return patch


def _sample_counts(rng: np.random.Generator, total_match: int) -> list[int]:
    options = [
        [a, b]
        for a in range(MAX_DOTS_PER_FRAME + 1)
        for b in range(MAX_DOTS_PER_FRAME + 1)
        if a + b == total_match and max(a, b) >= 2
    ]
    if not options:
        raise ValueError(f"no feasible count split for total_match={total_match}")
    return [int(x) for x in options[int(rng.integers(0, len(options)))]]


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
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


def _build_old_style_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames = [comp for comp in regions if _is_frame(comp)]
    patterns = [comp for comp in regions if not _is_frame(comp)]
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg
    for frame in frames:
        count = sum(1 for p in patterns if int(p["color"]) == int(frame["color"]))
        _draw_tally(out, frame, count, horizontal_layout=False)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_match = int(rng.integers(*TOTAL_MATCH_RANGE))
    total_patterns = total_match + n_distractors

    palette = rng.choice(np.arange(10), size=1 + N_FRAMES + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + N_FRAMES]]
    distractor_colors = [int(x) for x in palette[1 + N_FRAMES :]]

    counts = _sample_counts(rng, total_match)
    sprite_region_h = int(rng.integers(*SPRITE_REGION_H_RANGE))
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE))
    height = SPRITE_R0 + sprite_region_h + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_cols = [FRAME_C0 + i * (FRAME_W + FRAME_GAP) for i in range(N_FRAMES)]
    for col, color, count in zip(frame_cols, frame_colors, counts):
        paste(out, _frame_patch(color, bg, count), FRAME_R0, col)

    inp = out.copy()
    for col, count in zip(frame_cols, counts):
        c_mid = col + FRAME_W // 2
        for i in range(int(count)):
            inp[FRAME_R0 + FRAME_H - 2 - i, c_mid] = bg

    sprite_canvas = np.full((height - SPRITE_R0, SPRITE_W), bg, dtype=int)
    pattern_colors = [color for color, count in zip(frame_colors, counts) for _ in range(count)]
    pattern_colors.extend(distractor_colors)
    if len(pattern_colors) != total_patterns:
        raise AssertionError("pattern count mismatch")
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                sprite_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(sprite_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, sprite_canvas, SPRITE_R0, SPRITE_C0, transparent=bg)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_build_old_style_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
