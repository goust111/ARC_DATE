# Variant of 8f215267
# Dimension: D6 — 干扰项处理 (meta)
# Change: 存在1-2个干扰精灵，颜色不匹配任何框架，在输出中被移除但不影响计数... → 干扰项保留在输出中

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# concrete_behavior:
#   input:
#     - Left side still contains 3-4 hollow 5x11 frames, each with its own color.
#     - Right side still contains small monochrome sprites.
#     - Some sprites use frame colors and are "absorbed" into same-color frames.
#     - 1-2 distractor sprites use colors that match no frame color.
#   output:
#     - Matching sprites disappear and become dots inside their same-color frames.
#     - Distractor sprites remain exactly where they were in the input.
#     - Therefore the right region is no longer fully cleared; it only loses the
#       color-matched sprites.
#
# behavior_trace:
#   old_logic:
#     - classify every non-frame component as a pattern
#     - erase every pattern
#     - count only same-color patterns when placing dots
#   new_logic:
#     - classify every non-frame component as a pattern
#     - split patterns into absorbable vs distractor using frame colors
#     - erase only absorbable patterns
#     - count only absorbable same-color patterns when placing dots
#   example:
#     - If frames are blue/red/yellow and sprites are blue, blue, red, green,
#       old output erases all four sprites and draws 2 blue dots + 1 red dot.
#       new output erases only blue/blue/red, keeps the green sprite in place,
#       and still draws 2 blue dots + 1 red dot.
#
# complexity_calibration:
#   - The only changed degree of freedom is whether unmatched-color sprites persist.
#   - Distractors still use 1-2 colors not used by frames, so the solver can infer
#     the distinction from visible color mismatch rather than hidden metadata.
#   - Matching counts stay capped at 4 so dots fit and examples stay legible.
#
# old_assumption_audit:
#   - regions = find_components(...): still valid because frames and sprites remain
#     monochrome 4-connected components.
#   - frame detection via large sparse bbox: still valid because only right-side
#     sprite persistence changes, not frame geometry.
#   - old "remove all patterns" loop depended on the old invariant that every
#     non-frame component should vanish. That is false in this variant and must be
#     narrowed to same-color-with-frame components only.
#   - dot counting by color remains valid because distractor colors are guaranteed
#     not to equal any frame color.
#
# placement_legality:
#   - Generator builds the output first: frames with dots plus fixed distractor
#     sprites in the right region.
#   - Input is derived by removing dots and adding matching sprites into remaining
#     free space on the same right-side canvas, with 1-cell padding and no overlap.
#   - This guarantees non-overlap, in-bounds placement, and that distractors stay
#     at identical coordinates between input and output.
#
# observability_and_cleanliness:
#   - In the original task the whole right side empties out; here one or two sprites
#     visibly survive. That difference is easy to notice from 2-3 examples.
#   - Because distractors keep distinct colors and stay in the same uncluttered
#     right-side region, the output remains visually tidy rather than layered.
#
# new_solver_assumptions_and_generator_guarantees:
#   - Assumption: every sprite whose color matches a frame should be absorbed.
#     Guarantee: generator assigns every non-distractor sprite a frame color.
#   - Assumption: distractor colors match no frame color.
#     Guarantee: generator samples distractor colors from palette entries outside
#     the chosen frame colors.
#   - Assumption: sprites remain small enough not to be mistaken for frames.
#     Guarantee: sprite bbox stays within 4x3 and area within 2-8 cells.
# -----------------------------------------------------------------------------


def _split_components(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    """Separate large hollow frames from small sprites using the original heuristic."""
    regions = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites per frame and show the tally as dots inside.

    Variant rule:
    - Sprites whose color matches a frame are absorbed: they disappear and become dots.
    - Distractor sprites whose color matches no frame are preserved in place.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _split_components(g, bg)
    frame_colors = {int(f["color"]) for f in frames}

    # Remove only the sprites that belong to some frame color.
    for p in patterns:
        if int(p["color"]) not in frame_colors:
            continue
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    # Count same-color absorbable sprites and draw dots inside each frame.
    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(num_patterns):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col

    return out


# --------------- generator (codex-att2, D6 variant) ---------------

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
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    # --- build output (frames with dots + preserved distractors) ---
    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    right_out = np.full((height, RIGHT_W), bg, dtype=int)
    try:
        for color in distractor_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_out, sprite, transparent=bg,
                padding=SPRITE_PADDING, padding_connectivity=8,
            )
            paste(right_out, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None
    paste(out, right_out, 0, RIGHT_C0, transparent=bg)

    # --- derive input (dots removed, matching sprites added, distractors unchanged) ---
    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    right_in = right_out.copy()
    match_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
    rng.shuffle(match_colors)

    try:
        for color in match_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_in, sprite, transparent=bg,
                padding=SPRITE_PADDING, padding_connectivity=8,
            )
            paste(right_in, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_in, 0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
