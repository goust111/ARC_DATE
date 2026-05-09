# Combo variant of 8f215267 (index 23)
# combo_spec: D3/2+D5/1+D7/0
# Name: 水平框架+垂直点+保留精灵

from __future__ import annotations

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
   - Original task/spec:
     several 5x11 hollow frames are stacked vertically on the left; small monochrome
     sprites are scattered on the right; output removes those sprites and writes a
     horizontal same-color tally on each frame's center row.
   - Combo task:
     keep the same color-matching idea, but move the two frames into a horizontal top
     band, move sprites into a lower band, keep every sprite in place in the output,
     and draw the tally as a vertical column on each frame's center column.
   - Trace on one color:
     old input  -> red frame at left + three red sprites at right
     old output -> right sprites disappear; red frame gets three horizontal dots
     combo input  -> red frame in the top row of frames + three red sprites below
     combo output -> those three sprites remain; the red frame gains three vertical dots
   - The visible change is therefore threefold at once: frame layout rotates, sprite
     region moves below, and the only output edit is a vertical in-frame addition.

b) Complexity calibration
   - A horizontal row of 3-4 frames would exceed ordinary ARC widths, so the fused
     family uses exactly 2 frames. This is the smallest non-degenerate adaptation.
   - Keeping the original 5x11 frame size means a center column can show only 3 inner
     cells, so the tally cap is reduced from 4 to 3. Total matching sprites are sampled
     in [3, 6], which still produces varied counts while respecting physical capacity.
   - Each sample also includes 1-2 distractor sprites whose colors match no frame, so
     the rule cannot collapse into "count all sprites below".

c) Hidden assumptions audited against the original main()
   - Original frame/pattern separation by size still works, and here it is tightened to
     exact 5x11 hollow rectangles because the generator always uses that shape.
   - The old "remove patterns from output" loop is invalid under D7/0; keeping it would
     destroy required source evidence, so it must be removed.
   - The old tally code assumes a center row and right-to-left spacing; D5/1 breaks that
     assumption and requires bottom-up placement on the center column instead.
   - The old generator implicitly assumed sprites lived to the right of the frames. The
     new solver must not depend on that; it counts all non-frame components globally by
     color, while the generator guarantees they are placed in the lower band for clarity.

d) Placement legality
   - Frames occupy row 1 and columns [1,11] and [14,24] inside a 27-column grid.
   - The sprite region starts at row 8, leaving two full background rows between the top
     frames and the lower sprites, so no sprite can touch or merge with a frame.
   - Sprites are placed on a dedicated canvas with `random_free_location(..., padding=1,
     padding_connectivity=8)`, which prevents overlap, crowding, and out-of-bounds pastes.

e) Visual observability
   - From 2-4 examples, a human can see that the lower sprites do not disappear, yet the
     upper frames gain new same-color dots whose counts match lower same-color sprites.
   - The generator keeps both frame colors represented among the matching sprites, and at
     least one frame count is >= 2, so both color-specific counting and vertical stacking
     stay visible instead of collapsing into a one-frame-only demonstration.
   - Distractors repeat the "same shape but wrong color does nothing" counterexample.

f) Visual neatness
   - The grid is split into a clean top frame band and a clean lower sprite band.
   - Output adds only one kind of mark, inside the frames, and never overlays the lower
     sprites. No region mixes multiple unrelated object types.

g) Change inferability
   - No rule depends on numeric color ids. Colors are used only by visible equality:
     a frame counts exactly the sprites with the same color.
   - The preserved sprites provide stable before/after anchors, so the learner can infer
     that the output operation is "add a vertical tally" rather than "move or absorb".

h) Non-degeneration metric for generated samples
   - For each generated input, the fused output must differ from:
     1. a keep-source + horizontal-dot ablation, proving the vertical tally matters;
     2. a remove-source + vertical-dot ablation, proving sprite preservation matters.
   - The input itself must also satisfy a layout metric: frame-center column span is
     larger than frame-center row span, and every sprite lies strictly below the frames.

9) New assumptions introduced by the new main()
   - Every frame is an exact 5x11 hollow monochrome rectangle with no pre-existing tally.
   - Every non-frame component is a small monochrome sprite separated from the frames.
   - Matching remains pure color equality.
   - The generator enforces all of these assumptions before accepting a sample.
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
TOTAL_MATCH_RANGE = (3, 7)  # 3-6 matching sprites total
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


def _split_components(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    regions = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_frame(comp):
            frames.append(comp)
        else:
            patterns.append(comp)
    return frames, patterns


def _draw_vertical_tally(out: np.ndarray, frame: dict, count: int) -> None:
    if count <= 0:
        return
    color = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])
    c_mid = (c0 + c1) // 2
    for i in range(int(count)):
        rr = int(r1 - 1 - i)
        if rr > int(r0):
            out[rr, c_mid] = color


def _draw_horizontal_tally(out: np.ndarray, frame: dict, count: int) -> None:
    if count <= 0:
        return
    color = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])
    r_mid = (r0 + r1) // 2
    for i in range(int(count)):
        cc = int(c1 - 2 - 2 * i)
        if cc > int(c0):
            out[r_mid, cc] = color


def _frame_layout_is_horizontal(frames: list[dict]) -> bool:
    if len(frames) <= 1:
        return True
    row_centers = [((int(f["bbox"][0]) + int(f["bbox"][2])) // 2) for f in frames]
    col_centers = [((int(f["bbox"][1]) + int(f["bbox"][3])) // 2) for f in frames]
    row_span = max(row_centers) - min(row_centers)
    col_span = max(col_centers) - min(col_centers)
    return col_span > row_span


def _patterns_are_below_frames(frames: list[dict], patterns: list[dict]) -> bool:
    if not frames or not patterns:
        return False
    frame_bottom = max(int(f["bbox"][2]) for f in frames)
    return all(int(p["bbox"][0]) > frame_bottom + 1 for p in patterns)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites for each frame and display the tally as a vertical in-frame column.

    Input structure:
    - The background is the most frequent color.
    - Two hollow 5x11 rectangular frames sit side by side in the upper band.
    - Small monochrome sprites are scattered in the lower band, outside the frames.

    Rule:
    - Treat each hollow rectangle as a frame and every other monochrome component as a sprite.
    - For each frame color C, count how many sprites share color C.
    - Keep the input unchanged and add that many dots of color C on the frame's center column,
      stacked from the bottom interior cell upward.

    Key insight:
    - The lower sprites are evidence that stays visible; the output is produced by adding a
      vertical tally inside each horizontally arranged frame rather than by deleting sources.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _split_components(g, bg)
    for frame in frames:
        count = sum(1 for p in patterns if int(p["color"]) == int(frame["color"]))
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
        [a, b]
        for a in range(MAX_DOTS_PER_FRAME + 1)
        for b in range(MAX_DOTS_PER_FRAME + 1)
        if a + b == total_match and min(a, b) >= 1 and max(a, b) >= 2
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


def _build_keep_horizontal_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    frames, patterns = _split_components(g, bg)
    for frame in frames:
        count = sum(1 for p in patterns if int(p["color"]) == int(frame["color"]))
        _draw_horizontal_tally(out, frame, count)
    return out


def _build_remove_vertical_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    frames, patterns = _split_components(g, bg)
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg
    for frame in frames:
        count = sum(1 for p in patterns if int(p["color"]) == int(frame["color"]))
        _draw_vertical_tally(out, frame, count)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_match = int(rng.integers(*TOTAL_MATCH_RANGE))
    counts = _sample_counts(rng, total_match)

    palette = rng.choice(np.arange(10), size=1 + N_FRAMES + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + N_FRAMES]]
    distractor_colors = [int(x) for x in palette[1 + N_FRAMES :]]

    sprite_region_h = int(rng.integers(*SPRITE_REGION_H_RANGE))
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE))
    height = SPRITE_R0 + sprite_region_h + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_cols = [FRAME_C0 + i * (FRAME_W + FRAME_GAP) for i in range(N_FRAMES)]
    for col, color, count in zip(frame_cols, frame_colors, counts):
        paste(out, _frame_patch(color, bg, count), FRAME_R0, col)

    sprite_canvas = np.full((height - SPRITE_R0, SPRITE_W), bg, dtype=int)
    pattern_colors = [color for color, count in zip(frame_colors, counts) for _ in range(count)]
    pattern_colors.extend(distractor_colors)
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

    paste(out, sprite_canvas, SPRITE_R0, SPRITE_C0, transparent=bg)

    inp = out.copy()
    for col, count in zip(frame_cols, counts):
        c_mid = col + FRAME_W // 2
        for i in range(int(count)):
            inp[FRAME_R0 + FRAME_H - 2 - i, c_mid] = bg

    frames, patterns = _split_components(inp, bg)
    if len(frames) != N_FRAMES:
        return None
    if len(patterns) != len(pattern_colors):
        return None
    if not _frame_layout_is_horizontal(frames):
        return None
    if not _patterns_are_below_frames(frames, patterns):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_build_keep_horizontal_output(inp), out):
        return None
    if np.array_equal(_build_remove_vertical_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
