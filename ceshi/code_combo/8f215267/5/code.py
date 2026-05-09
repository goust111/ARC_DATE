# Combo variant of 8f215267 (index 5)
# combo_spec: D3/2+D5/1
# Name: 水平框架+垂直点

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
   - Original task: 5x11 hollow frames are stacked vertically on the left, sprites are
     scattered on the right, and the output replaces those sprites with horizontal dots
     on each frame's center row.
   - Combo task: rotate the scene organization by 90 degrees. The frames are now laid
     out left-to-right in a top band, while sprites occupy a lower band underneath.
     The counting relation stays color-based, but the readout also rotates: dots are
     drawn on the center column from bottom to top.
   - Example trace:
     original input  -> red frame in left stack, 3 red sprites in right region
     original output -> 3 red dots on that frame's middle row
     combo input     -> red frame in top row of frames, 3 red sprites in lower region
     combo output    -> 3 red dots on that frame's middle column, rising upward
   - The visible change is therefore joint, not isolated: destination layout changes,
     sprite region changes, and in-frame tally direction changes with it.

b) Complexity calibration
   - Keeping the original 3-4 frames would not fit once 11-column frames are placed
     horizontally, so the frame count is reduced to exactly 2.
   - We keep the original 5x11 frame shape; that leaves exactly 3 usable interior rows
     on one center column, so the per-frame tally cap becomes 3.
   - Total matching sprites are restricted to 3-6. This keeps examples nontrivial while
     staying inside the 2 x 3 visible dot capacity. Distractors are still present so the
     task is not reducible to "count every sprite below each frame."

c) Hidden assumptions audited against the old main()
   - Component classification by size/sparsity still works because frames remain the only
     large hollow rectangles and sprites remain small polyominoes up to 4x3.
   - The old code implicitly assumed the tally lives on the center row with 2-column
     spacing from the right. That assumption is invalid and is replaced by bottom-up
     filling on the center column.
   - The old generator assumed sprites belong in a right-side strip. In the combo task
     that becomes a lower strip, but main() should not depend on absolute region; it
     only needs to remove non-frame components and count them by color.
   - New assumption: frames are always the exact 5x11 hollow rectangles used by the
     generator. The generator enforces this, including clean separation from sprites.

d) Placement legality
   - Two frames of width 11 with a 2-column gap fit inside a 27-column grid at columns
     [1,11] and [14,24], leaving a margin on both sides.
   - Sprites are placed only in rows below the frame band, starting after a 2-row blank
     gap, so they cannot merge with frames under 4-connectivity.
   - random_free_location with 8-neighbor padding keeps sprites disjoint from one another.

e) Visual observability
   - From 2-3 examples, a human can see that the upper row contains the target frames,
     the lower region contains the counted sprites, and the output replaces those sprites
     with vertical columns of dots.
   - This differs visibly from the source task even when counts are small, because a
     single vertical dot occupies a different position than the original horizontal one.

f) Visual cleanliness
   - Frames and sprites live in separate horizontal bands, avoiding object overlap and
     avoiding the mixed "left frame / right sprite" look of the source task.
   - The only marks added inside frames are the tally dots themselves, so the output
     stays easy to parse.

g) Change inferability
   - Color semantics remain equality-only: a frame counts sprites of the same visible
     color, while distractor colors appear only among sprites and never among frames.
   - No rule depends on numeric color ids, ordering, or any invisible metadata.

h) Non-degeneration metric
   - For each generated input, compare the combo output against a legacy-style output
     that would still draw horizontal dots inside the same horizontal frames.
   - Because every sample has at least one matching sprite, this distance should be
     positive for 100% of samples; otherwise the generator would be collapsing back to
     the old in-frame readout.

i) New solver assumptions and generator guarantees
   - main() assumes every non-frame monochrome component is a sprite: generator ensures
     there are no other foreground object types.
   - main() assumes a frame's bbox is 5x11 and can host at most 3 vertical dots:
     generator caps counts accordingly.
   - main() assumes frame colors are distinct from distractor colors: generator samples
     all colors without replacement.
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
    bbox = comp["bbox"]
    area = int(comp["area"])
    return (
        bbox_h(bbox) == FRAME_H
        and bbox_w(bbox) == FRAME_W
        and FRAME_AREA <= area <= FRAME_AREA + MAX_DOTS_PER_FRAME
    )


def _vertical_dot_rows(bbox: tuple[int, int, int, int], count: int) -> list[int]:
    r0, _c0, r1, _c1 = map(int, bbox)
    return [int(r1 - 1 - i) for i in range(int(count)) if int(r1 - 1 - i) > int(r0)]


def _draw_vertical_tally(out: np.ndarray, frame: dict, count: int) -> None:
    if count <= 0:
        return
    r0, c0, r1, c1 = map(int, frame["bbox"])
    c_mid = (c0 + c1) // 2
    color = int(frame["color"])
    for rr in _vertical_dot_rows((r0, c0, r1, c1), count):
        out[int(rr), int(c_mid)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites for each frame and display the tally as vertical dots.

    Input structure:
    - The background is the most common color.
    - Two hollow 5x11 frames of different colors sit side by side in the upper band.
    - Small monochrome sprites are scattered in a lower band beneath the frames.

    Rule:
    - Treat each non-frame monochrome component as a sprite and erase all sprites.
    - For each frame color C, count how many erased sprites have color C.
    - Draw that many dots on the frame's center column, stacking them from the bottom upward.

    Key insight:
    - The whole counting display has rotated: frames are arranged horizontally, sprites live
      below them, and the recovered tally is shown as a vertical in-frame column instead of
      the source task's horizontal bar.
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

    for pattern in patterns:
        for r, c in pattern["coords"]:
            out[int(r), int(c)] = bg

    for frame in frames:
        count = sum(1 for pattern in patterns if int(pattern["color"]) == int(frame["color"]))
        _draw_vertical_tally(out, frame, count)

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
        [left, right]
        for left in range(MAX_DOTS_PER_FRAME + 1)
        for right in range(MAX_DOTS_PER_FRAME + 1)
        if left + right == total_match and max(left, right) >= 2
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


def _legacy_horizontal_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames = [comp for comp in regions if _is_frame(comp)]
    patterns = [comp for comp in regions if not _is_frame(comp)]
    for pattern in patterns:
        for r, c in pattern["coords"]:
            out[int(r), int(c)] = bg
    for frame in frames:
        count = sum(1 for pattern in patterns if int(pattern["color"]) == int(frame["color"]))
        if count <= 0:
            continue
        r0, c0, r1, c1 = map(int, frame["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(int(count)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = int(frame["color"])
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
    if np.array_equal(_legacy_horizontal_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
