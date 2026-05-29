# Variant of 8f215267
# Dimension: D6 — 干扰项处理 (meta)
# Change: 存在1-2个干扰精灵，颜色不匹配任何框架，在输出中被移除但不影响计数... → 干扰项使用框架颜色但面积超标
#
# Design analysis for this variant
# --------------------------------
# concretization_and_trace:
#   - Keep the original scene grammar: 3-5 hollow 5x11 frames stacked on the left,
#     random monochrome sprites scattered in the right strip, output shows dots inside
#     each frame.
#   - The only semantic change is how right-side sprites are counted:
#       * all non-frame sprites are still removed in the output;
#       * only "small" sprites are counted, where small means area <= 5;
#       * oversized sprites (area >= 6) are distractors even when their color matches
#         a frame color.
#   - Example trace:
#       input has a red frame, two red area-4 sprites, and one red area-7 sprite.
#       Original logic: all three red sprites match by color, so it draws 3 red dots.
#       Variant logic: all three sprites disappear, but only the two area-4 sprites
#         are counted, so it draws 2 red dots.
#       Difference: color match is no longer sufficient; size filtering is required.
# complexity_calibration:
#   - Counted sprites are restricted to areas 2-5.
#   - Oversized distractors are restricted to areas 6-8, so the visual difference is
#     noticeable without changing the overall layout grammar.
#   - Each generated instance is forced to contain at least one oversized same-color
#     distractor whose presence would change the result under the old solver.
# implicit_assumption_review:
#   - Original `patterns.append(comp)` for every non-frame component assumed every
#     non-frame sprite was countable. That is false now.
#   - Original `sum(1 for p in patterns if p["color"] == col)` assumed color alone
#     determined membership. The variant requires `(color match) AND (area <= 5)`.
#   - Frame detection can stay unchanged because frames remain large hollow 5x11
#     rectangles, while sprites keep bbox height/width < 5.
# placement_legality:
#   - Right-side placement still uses `random_free_location(..., padding=1)`, so
#     sprites remain disjoint and never touch frames or each other.
#   - Oversized distractors stay within 4x4 bboxes, so they cannot be misdetected as
#     frames and still fit in the 9-column right region.
# visual_observability_and_cleanliness:
#   - The change is visible because a frame-colored sprite can disappear without adding
#     a corresponding dot.
#   - The generator preserves the original clean left/right separation, so the only
#     new inference burden is the size threshold, not a new spatial rule.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


COUNTED_MAX_AREA = 5


def _solve_with_area_limit(grid: np.ndarray, counted_max_area: int | None) -> np.ndarray:
    """Shared renderer: remove all sprites, but count only those within the area limit."""
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    sprites: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            sprites.append(comp)

    # Every non-frame sprite disappears in the output, including oversized distractors.
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = bg

    for frame in frames:
        col = int(frame["color"])
        num_patterns = 0
        for sprite in sprites:
            if int(sprite["color"]) != col:
                continue
            if counted_max_area is not None and int(sprite["area"]) > int(counted_max_area):
                continue
            num_patterns += 1

        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, frame["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(num_patterns):
            cc = int(c1 - 2 - 2 * i)
            if cc > c0:
                out[r_mid, cc] = col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color small sprites per frame and render the tally as dots.

    Variant rule:
    - Remove all scattered sprites from the output.
    - For each frame color C, count only sprites of color C with area <= 5.
    - Oversized same-color sprites are distractors and do not contribute to the tally.
    """
    return _solve_with_area_limit(grid, COUNTED_MAX_AREA)


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
TOTAL_SPRITES_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- counted sprite shape ---
SMALL_AREA_RANGE = (2, COUNTED_MAX_AREA + 1)  # 2..5
SMALL_MAX_H = 3
SMALL_MAX_W = 3

# --- oversized distractor shape ---
LARGE_AREA_RANGE = (COUNTED_MAX_AREA + 1, 9)  # 6..8
LARGE_MAX_H = 4
LARGE_MAX_W = 4
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 400


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
    """Distribute total counted sprites across frames, each capped at MAX_DOTS_PER_FRAME."""
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


def _make_sprite(
    rng: np.random.Generator,
    color: int,
    bg: int,
    area_range: tuple[int, int],
    max_h: int,
    max_w: int,
) -> np.ndarray:
    """Generate a connected monochrome sprite, tight-cropped to its bbox."""
    area = int(rng.integers(*area_range))
    dims = [(h, w) for h in range(1, max_h + 1) for w in range(1, max_w + 1) if h * w >= area]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _pick_distractor_indices(rng: np.random.Generator, counts: list[int], n_distractors: int) -> list[int]:
    """
    Choose frame indices whose colors will be reused for oversized distractors.

    At least one chosen index must force a different result from the old solver,
    so prefer a frame with 0-3 counted sprites. If possible, prefer 1-3 so the
    same color appears both as countable small sprites and as an oversized distractor.
    """
    preferred = [i for i, count in enumerate(counts) if 0 < count < MAX_DOTS_PER_FRAME]
    fallback = [i for i, count in enumerate(counts) if count < MAX_DOTS_PER_FRAME]
    anchor_pool = preferred if preferred else fallback
    if not anchor_pool:
        return []

    anchor = int(rng.choice(anchor_pool))
    indices = [anchor]

    others = [i for i in range(len(counts)) if i != anchor]
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

    palette = rng.choice(np.arange(10), size=1 + n_frames, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1:]]

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
    large_specs = [frame_colors[idx] for idx in distractor_indices]
    small_specs = [color for color, count in zip(frame_colors, counts) for _ in range(count)]
    rng.shuffle(large_specs)
    rng.shuffle(small_specs)

    try:
        for color in large_specs:
            sprite = _make_sprite(rng, color, bg, LARGE_AREA_RANGE, LARGE_MAX_H, LARGE_MAX_W)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)

        for color in small_specs:
            sprite = _make_sprite(rng, color, bg, SMALL_AREA_RANGE, SMALL_MAX_H, SMALL_MAX_W)
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

    # Variant guard: the new solver must succeed, and the old color-only solver
    # must disagree so the size rule is actually exercised.
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_solve_with_area_limit(inp, counted_max_area=None), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
