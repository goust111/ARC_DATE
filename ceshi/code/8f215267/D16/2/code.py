# Variant of 8f215267
# Dimension: D16 — 点的最大容量约束 (meta)
# Change: 每个框架最多放4个点（受框架内部9列宽度和间距2限制），超出则无法放置... → 溢出标记机制

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

"""
Design analysis for the D16 variant
==================================

variant_spec:
  preserved_structure:
    - Input still contains hollow 5x11 monochrome frames on the left.
    - Input still contains scattered monochrome sprites on the right.
    - Output still removes all sprites and uses each frame as the display for the
      count of same-color sprites.
  core_change:
    - The original task could only show 0..4 because the center row only has four
      usable dot slots at spacing 2.
    - The variant keeps those four inner slots, but adds an overflow channel on the
      border when the true count is 5..8.
  pixel_realization:
    - Counts 0..4: unchanged. Draw dots on the center row from the right at local
      columns 8, 6, 4, 2.
    - Overflow units 1..2 (true counts 5..6): remove cells from the TOP border at
      local columns 2, 4, left-to-right.
    - Overflow units 3..4 (true counts 7..8): after the two top notches are used,
      remove cells from the BOTTOM border at local columns 8, 6, right-to-left.
    - So:
        5 -> 4 inner dots + one top-left notch
        6 -> 4 inner dots + two top-left notches
        7 -> 4 inner dots + two top-left notches + one bottom-right notch
        8 -> 4 inner dots + two top-left notches + two bottom-right notches
  concrete_trace:
    - Original logic for seven red sprites:
      remove sprites -> attempt to draw 7 center-row dots -> only the first 4 fit,
      so the displayed result collapses to the same image as count 4.
    - Variant logic for seven red sprites:
      remove sprites -> draw 4 center-row dots -> overflow is 3 -> cut two top
      border cells and one bottom border cell. The frame now visibly differs from
      both count 4 and count 6.
  complexity_check:
    - There are only two overflow loci, each with at most two slots, so the rule is
      visible and finite rather than open-ended.
    - The top and bottom borders use opposite directions, making the two encoding
      rules visually distinct instead of feeling like a single longer row of marks.
    - Generator forces overflow to appear regularly so the task does not regress to
      the original non-overflow-only behavior.
  implicit_assumptions_review:
    - Frame detection in main() assumes input frames are intact hollow rectangles.
      That remains true because overflow notches exist only in the output; the
      generator restores plain frames when deriving the input.
    - Pattern detection assumes sprites are smaller than frames. Generator keeps
      sprites compact and separated by 8-neighborhood padding so same-color sprites
      never merge into one component.
    - The tally renderer assumes counts are in 0..8 because the border has four
      overflow slots total. Generator guarantees that bound. main() clips gracefully
      if given a larger count, but such inputs are not generated.
    - The original generator removed only center dots to recover the input. That is
      no longer sufficient: variant outputs may also remove border cells, so the
      generator must repaint each frame as a clean empty rectangle before placing
      the sprites.
  placement_legality:
    - Frames keep the original fixed geometry and left-side stacking.
    - Sprites remain confined to the right-side region and are placed with padding=1
      under 8-connectivity so components stay disjoint and never touch frames.
  observability:
    - A solver can see two distinct behaviors from a few examples:
      top-border notches appear when a frame already contains four dots, and
      bottom-border notches appear only after the top-notch capacity is exhausted.
    - No invisible property such as numeric color order is used.
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

# --- tally geometry ---
MAX_INNER_DOTS = 4
TOP_NOTCH_COLS = (2, 4)
BOTTOM_NOTCH_COLS = (8, 6)
MAX_MATCH_PER_FRAME = 8

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
MATCH_TOTAL_RANGE = (7, 13)
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 8)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 500


def _is_frame(comp: dict) -> bool:
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    area = int(comp["area"])
    return (h >= 5) and (w >= 5) and (area < (h * w * 0.7))


def _draw_tally(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, bg: int, count: int) -> None:
    """Render the count into a frame using dots plus overflow notches."""
    r0, c0, r1, c1 = map(int, bbox)
    r_mid = (r0 + r1) // 2

    base = min(int(count), MAX_INNER_DOTS)
    for i in range(base):
        out[r_mid, c1 - 2 - 2 * i] = int(color)

    overflow = min(max(int(count) - MAX_INNER_DOTS, 0), len(TOP_NOTCH_COLS) + len(BOTTOM_NOTCH_COLS))
    top_overflow = min(overflow, len(TOP_NOTCH_COLS))
    bottom_overflow = max(overflow - len(TOP_NOTCH_COLS), 0)

    for i in range(top_overflow):
        out[r0, c0 + TOP_NOTCH_COLS[i]] = int(bg)
    for i in range(bottom_overflow):
        out[r1, c0 + BOTTOM_NOTCH_COLS[i]] = int(bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and show the tally inside the frame.

    Counts 0..4 use the original center-row dots. Counts 5..8 still show four dots,
    and the extra 1..4 units overflow onto frame-border notches.
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

    for f in frames:
        color = int(f["color"])
        count = sum(1 for p in patterns if int(p["color"]) == color)
        if count <= 0:
            continue
        _draw_tally(out, tuple(map(int, f["bbox"])), color, bg, count)

    return out


def _frame_patch(color: int, bg: int, count: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    _draw_tally(patch, (0, 0, FRAME_H - 1, FRAME_W - 1), color, bg, count)
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int) -> list[int]:
    """
    Sample per-frame counts with guaranteed overflow in at least one frame.

    We keep totals moderate so the right-hand sprite region remains placeable while
    still exercising both overflow branches over many generated examples.
    """
    max_overflow_frames = min(2, n_frames)
    for _ in range(300):
        counts = [0] * n_frames
        n_overflow = int(rng.integers(1, max_overflow_frames + 1))
        overflow_idxs = rng.choice(n_frames, size=n_overflow, replace=False)

        for idx in overflow_idxs:
            counts[int(idx)] = int(rng.integers(5, MAX_MATCH_PER_FRAME + 1))

        for idx in range(n_frames):
            if counts[idx] == 0:
                counts[idx] = int(rng.integers(0, MAX_INNER_DOTS + 1))

        total = sum(counts)
        has_overflow = any(v > MAX_INNER_DOTS for v in counts)
        has_nonoverflow = any(v <= MAX_INNER_DOTS for v in counts)
        if MATCH_TOTAL_RANGE[0] <= total < MATCH_TOTAL_RANGE[1] and has_overflow and has_nonoverflow:
            return counts

    raise RuntimeError("failed to sample a valid overflow distribution")


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


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    counts = _sample_counts(rng, n_frames)

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for color, r0 in zip(frame_colors, frame_rows):
        paste(inp, _frame_patch(color, bg, 0), r0, FRAME_C0)

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
