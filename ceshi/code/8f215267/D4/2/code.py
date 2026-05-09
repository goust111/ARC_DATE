# Variant of 8f215267
# Dimension: D4 — 变换规则：计数方式 (transform)
# Change: 直接计数与框架同色的精灵数量 N... → 排除干扰项后计数

from __future__ import annotations

"""
Design analysis for the D4 variant
==================================

source_task:
  referenced_task: none
  action: skip

variant_spec:
  preserved_layout:
    - grid width stays 23; left side still holds vertically stacked 5x11 hollow frames
    - right side still holds scattered monochrome components outside the frames
    - output still removes all scattered components and writes tally dots on each frame's center row
  changed_rule:
    - original counting unit: every non-frame component whose color matches a frame color
    - new counting unit: only "true sprites", not "fake sprites"
  true_vs_fake_definition:
    - true sprite: monochrome 4-connected component of area >= 2
    - fake sprite: monochrome 4-connected component of area == 1
    - fake sprites deliberately reuse frame colors, so color alone is no longer sufficient

a) concretization_and_trace:
  input_picture:
    - each frame color still appears on the left as a hollow rectangle
    - the right region now mixes:
      * true sprites in frame colors (small 2-8 cell polyominoes)
      * fake sprites in frame colors (isolated single pixels)
      * ordinary color distractors whose colors match no frame
    - visually, a frame color may appear both as chunky blobs and as isolated specks
  output_picture:
    - all right-side components disappear
    - dots inside a frame count only the chunky blobs of that color, not the isolated specks
  trace_example:
    - suppose the red frame has 2 red blobs, 1 red single pixel, and 0 other relevant objects
    - old logic counts 3 red components and would draw 3 dots
    - new logic removes all 3 objects but counts only the 2 blobs, so it draws 2 dots
    - the difference occurs exactly at the counting step; removal and dot placement stay the same

b) complexity_calibration:
  - fake sprites are restricted to a single visible shape class: isolated one-pixel points
  - true sprites stay in the original 2-8 cell size band, so the distinction is repeated and stable
  - each generated grid includes fake sprites for at least two frame colors, avoiding one-off noise
  - fake-bearing frames are capped so the legacy count would visibly differ from the variant count

c) implicit_assumption_audit_of_original_main:
  - find_components(..., monochromatic=True, connectivity=4):
      assumes each sprite is a separate connected component
      still valid only if generator keeps 1-cell padding so fake points do not merge into true sprites
  - is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7):
      relies on frames being the only large sparse components
      still valid because all right-side sprites remain much smaller than 5x5
  - num_patterns = sum(1 for p in patterns if p["color"] == col):
      this is the line broken by the variant
      same-color single pixels now exist, so color match alone overcounts
  - removing every pattern before drawing:
      remains correct because both true and fake sprites disappear in the output
  - dot placement from the right every 2 columns:
      unchanged; generator keeps the true count within the 4-dot interior capacity

d) placement_legality:
  - all sprites are placed only in the fixed right-side canvas
  - random_free_location with padding=1 and 8-neighbor dilation keeps components isolated
  - isolation guarantees that a 1x1 fake point stays a separate 4-connected component
  - frames never overlap because they remain on the left at fixed rows/cols

e) visual_observability:
  - across examples, the user can see same-color single pixels vanish without contributing dots
  - the chunky same-color blobs still correlate with frame dots exactly as in the original task
  - because fake pixels share the frame color, the new rule is observable and necessary

f) visual_cleanliness:
  - the scene stays split into a left "display" zone and a right "sprite" zone
  - fake points are small but sparse, so they read as deliberate specks rather than clutter
  - no object layers overlap; everything remains monochrome and cleanly separated

g) color_inference:
  - no new color mapping is introduced
  - color meaning remains structural: frame color selects candidate sprites
  - the only added inference is shape-based filtering (single-pixel vs multi-pixel)

new_main_assumptions_checked_against_generator:
  - same-color fake sprites must remain isolated single-cell components
  - true sprites must all have area >= 2
  - frame colors remain distinct
  - total countable sprites per frame must stay <= 4 so dots fit
  - generator explicitly enforces all of the above and rejects samples where the legacy rule
    would accidentally produce the same output as the variant
"""

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste, random_free_location


def _is_frame_component(comp: dict) -> bool:
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    area = int(comp["area"])
    return (h >= 5) and (w >= 5) and (area < (h * w * 0.7))


def _is_true_sprite(comp: dict) -> bool:
    return int(comp["area"]) >= 2


def _split_components(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    regions = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_frame_component(comp):
            frames.append(comp)
        else:
            patterns.append(comp)
    return frames, patterns


def _draw_counts(base: np.ndarray, frames: list[dict], patterns: list[dict], *, count_fakes: bool) -> np.ndarray:
    out = base.copy()
    for f in frames:
        col = int(f["color"])
        if count_fakes:
            num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        else:
            num_patterns = sum(1 for p in patterns if int(p["color"]) == col and _is_true_sprite(p))
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Remove all scattered components outside the frames.
    - For each frame color, count only same-color multi-pixel sprites (area >= 2).
    - Ignore same-color single-pixel fake sprites.
    - Render the tally as right-to-left dots on the frame's center row.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _split_components(g, bg)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    return _draw_counts(out, frames, patterns, count_fakes=False)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """
    Baseline from the original task: count every same-color component, including fake single pixels.
    Used only to reject degenerate generated inputs where the variant would be invisible.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    frames, patterns = _split_components(g, bg)
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg
    return _draw_counts(out, frames, patterns, count_fakes=True)


# --------------- generator ---------------

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
N_COLOR_DISTRACTORS_RANGE = (1, 3)
TOTAL_TRUE_MATCH_RANGE = (4, 9)
MIN_FAKE_FRAMES = 2
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
TRUE_SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 400


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


def _sample_true_counts(rng: np.random.Generator, fake_counts: list[int]) -> list[int]:
    caps = [MAX_DOTS_PER_FRAME - int(fake_n) for fake_n in fake_counts]
    for _ in range(200):
        counts = [int(rng.integers(0, cap + 1)) for cap in caps]
        total = sum(counts)
        if TOTAL_TRUE_MATCH_RANGE[0] <= total < TOTAL_TRUE_MATCH_RANGE[1]:
            return counts
    raise ValueError("failed to sample countable sprite counts")


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
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


def _make_true_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    area = int(rng.integers(*TRUE_SPRITE_AREA_RANGE))
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


def _fake_sprite(color: int) -> np.ndarray:
    return np.array([[color]], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_color_distractors = int(rng.integers(*N_COLOR_DISTRACTORS_RANGE))

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_color_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    n_fake_frames = int(rng.integers(MIN_FAKE_FRAMES, n_frames + 1))
    fake_frame_ids = set(int(x) for x in rng.choice(n_frames, size=n_fake_frames, replace=False))
    fake_counts = [1 if i in fake_frame_ids else 0 for i in range(n_frames)]

    try:
        true_counts = _sample_true_counts(rng, fake_counts)
    except ValueError:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, true_counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, true_counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)

    patches: list[np.ndarray] = []
    for color, count in zip(frame_colors, true_counts):
        for _ in range(count):
            patches.append(_make_true_sprite(rng, color, bg))
    for color in distractor_colors:
        patches.append(_make_true_sprite(rng, color, bg))
    for color, fake_n in zip(frame_colors, fake_counts):
        for _ in range(fake_n):
            patches.append(_fake_sprite(color))

    for idx in rng.permutation(len(patches)):
        patch = patches[int(idx)]
        try:
            r0, c0 = random_free_location(
                right_canvas,
                patch,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(right_canvas, patch, r0, c0, transparent=bg)

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
