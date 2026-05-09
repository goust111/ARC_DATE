# Variant of 8f215267
# Dimension: D8 — 输出网格形状 (grid_shape)
# Change: 输出网格与输入网格尺寸完全相同（23列宽，行数不变）... → 紧凑裁剪到框架区域

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D8 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   core_change:
#     before: solve on the full input canvas and return the same Hx23 canvas.
#     after: solve the same tallying task, then crop the output to the union bbox
#       of the colored frames. This removes the right sprite field and any extra
#       top/bottom background margins.
#   visible_effect:
#     input still shows two zones:
#       - left: vertically stacked hollow frames
#       - right: scattered monochrome sprites
#     output now shows only the frame stack itself, with the dots filled in.
#     The width collapses from 23 to FRAME_W (=11); the height collapses to the
#     top of the first frame through the bottom of the last frame.
#   concrete_trace:
#     original:
#       - a red frame at cols 1..11 and a red sprite at cols 15..17
#       - solver erases the sprite, draws one red dot in the frame center row,
#         but still returns all 23 columns.
#     variant:
#       - the same erase+count+draw steps happen first
#       - then the solver crops to the bbox spanning all frames, so cols 12..22
#         disappear and any spare background row above/below the frame stack also
#         disappears.
#       - difference: logic on objects is unchanged; only the final canvas shape
#         changes.
#   complexity_calibration:
#     - Keep frame/pattern recognition identical to avoid adding a second new rule.
#     - Keep multiple frames and distractor sprites so the counting rule remains
#       the dominant inference challenge.
#     - Vary the top margin and the number of frames so the crop is visibly bbox-
#       based rather than "always remove a fixed strip".
#   implicit_assumption_review:
#     - Component classification assumes frames are the only monochrome objects
#       with both height and width >= 5 and with sparse area relative to bbox.
#       This still holds because sprites remain at most 4x3.
#     - Pattern removal assumes every non-frame component is a sprite and can be
#       painted back to bg. Still valid under the variant because only the output
#       shape changes.
#     - Dot drawing assumes the frame bbox remains intact after erasing sprites.
#       Still valid because sprites are always outside frames.
#     - New crop step assumes the desired output bbox is exactly the union of the
#       frame components. Generator guarantees no other foreground survives after
#       solving, so cropping by frame bbox is correct and stable.
#   placement_legality:
#     - Frames remain vertically stacked with fixed size and no overlap.
#     - Sprites are still placed in a dedicated right-side canvas with padding,
#       then pasted into the input strictly to the right of the frames.
#     - Because the expected output is produced from the full solved canvas and
#       then cropped, generator legality and solver legality stay aligned.
#   visual_observability:
#     - Humans can still infer the color-counting rule from the dots.
#     - The D8 change is obvious because the right sprite field vanishes entirely
#       and the output width is much smaller than the input width.
#     - Variable top margin plus optional bottom padding make the crop boundary
#       look data-driven instead of hard-coded.
#   cleanliness:
#     - Output contains only the frame stack; no mixed regions remain.
#     - No extra overlays are introduced.
#   color_reasoning:
#     - No new color mapping is introduced; colors keep their original meaning.
#
# main_assumptions_after_change:
#   - There is at least one frame component.
#   - All surviving non-bg output pixels belong inside the union bbox of frames.
#   - Generator enforces those assumptions by using the same frame geometry and
#     by validating every sample with main(input).


def _split_components(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Partition monochrome components into large hollow frames vs small sprites."""
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
    return frames, patterns


def _crop_to_frame_bbox(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    """Tight-crop to the union bbox of all frame cells."""
    if not frames:
        return np.array(grid, dtype=int).copy()
    coords = np.concatenate([np.array(f["coords"], dtype=int) for f in frames], axis=0)
    r0, c0, r1, c1 = bbox_coords(coords)
    return np.array(grid, dtype=int)[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame, render the tally as interior dots,
    then tight-crop the solved scene to the frame region.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _split_components(g, bg)

    # Remove sprites from the solved canvas.
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    # Count same-color sprites per frame and draw the tally from right to left.
    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, _r1, c1 = map(int, f["bbox"])
        r_mid = int(r0 + FRAME_H // 2)
        for i in range(num_patterns):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[r_mid, cc] = col

    return _crop_to_frame_bbox(out, frames)


# --------------- generator (output-first, then crop) ---------------

# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9
TOP_MARGIN_RANGE = (1, 4)      # [lo, hi) empty rows above the first frame

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


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with n_dots on the center row."""
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
    dims = [(h, w) for h in range(1, SPRITE_MAX_H + 1)
            for w in range(1, SPRITE_MAX_W + 1) if h * w >= area]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _expected_output_from_full(full_grid: np.ndarray) -> np.ndarray:
    """Apply only the D8 crop to a fully solved full-size canvas."""
    g = np.array(full_grid, dtype=int)
    bg = int(get_bg(g))
    frames, _patterns = _split_components(g, bg)
    return _crop_to_frame_bbox(g, frames)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    # --- choose structure ---
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, n_frames, total_match)
    top_margin = int(rng.integers(*TOP_MARGIN_RANGE))
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = top_margin + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    # --- build the full solved canvas first ---
    full_out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = top_margin + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(full_out, _frame_patch(color, bg, count), r0, FRAME_C0)
    out = _expected_output_from_full(full_out)

    # --- derive input by erasing dots and adding sprites on the right ---
    inp = full_out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_canvas, sprite, transparent=bg,
                padding=SPRITE_PADDING, padding_connectivity=8,
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
