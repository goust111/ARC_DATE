# Variant of 8f215267
# Dimension: D16 — 点的最大容量约束 (meta)
# Change: 每个框架最多放4个点（受框架内部9列宽度和间距2限制），超出则无法放置... → 点可多行排列

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the D16 variant
# -----------------------------------
# a) Concrete behavior and execution trace
#    - The scene keeps the original left/right split:
#      left = vertically stacked 5x11 hollow frames, right = scattered monochrome sprites.
#    - The input still shows empty frames only; the output removes all sprites and writes the
#      count for each frame color back inside that frame.
#    - Original behavior: only the center row can host tally cells, so counts 0..4 map to
#      columns (right to left) [c1-2, c1-4, c1-6, c1-8].
#    - Variant behavior: once those 4 center-row cells are filled, counting continues on the
#      row above the center, then on the row below the center. The order is fixed:
#      center row first, then upper row, then lower row. With the unchanged 5x11 frame, the
#      practical capacity becomes 12 (= 3 rows x 4 columns).
#    - Example with 7 same-color sprites for a frame:
#      old solver: loops i=0..6, but only i=0..3 satisfy cc>c0, so the display silently clips
#      to 4 dots.
#      new solver: writes 4 dots on the center row and 3 more on the row above.
#
# b) Complexity calibration
#    - The change is visible but still learnable: there are only three possible tally rows, and
#      each row always uses the same right-aligned spacing pattern.
#    - The generator guarantees at least one frame count > 4 in every sample, so the dataset
#      does not collapse back to the original single-row regime.
#    - Some samples also reach counts > 8 so the third row is exercised; this prevents the
#      second row from being mistaken for a one-off exception.
#
# c) Implicit-assumption audit of main()
#    - `find_components(..., monochromatic=True)`: still valid because frames remain monochrome
#      hollow rectangles and sprites remain separate monochrome connected components.
#    - `is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7)`: still valid because the input
#      frames stay unchanged; only the output gets extra interior cells. Sprite sizes stay below
#      5x5 so they do not get misclassified as frames.
#    - Removing patterns before drawing counts remains correct.
#    - The old single-row loop depended on an old invariant: all meaningful counts fit before
#      `cc <= c0`. In this variant that invariant is false, so the draw logic must switch from
#      "iterate columns on one row" to "enumerate legal tally cells across three rows".
#    - New subtlety: dots placed on the upper/lower interior rows touch the top/bottom border by
#      4-neighborhood. This is harmless because frame detection only happens on the input, where
#      those dots are absent.
#
# d) Placement legality
#    - Generator is still output-first: build the intended output, erase tally cells to form the
#      input, scatter right-side sprites, then verify `main(input) == output`.
#    - Counts are capped at 12 so the output never asks for cells outside the 3x9 frame interior.
#    - Sprites keep 8-neighborhood padding during placement so equal-color sprites do not merge
#      into larger components and break counting.
#
# e) Visual observability
#    - Every generated grid includes at least one overflowing frame (>4 matches), so a solver can
#      see that counting extends beyond the center row.
#    - A separate sampling mode sometimes forces counts >8, making the third tally row visible.
#
# f) Visual cleanliness
#    - Frame geometry, alignment, and left/right partition remain unchanged.
#    - Sprites are kept compact so the right region can host more objects without becoming muddy.
#    - The tally pattern is regular: each extra row reuses the same right-aligned columns.
#
# g) Color inference
#    - Colors still mean only "which sprites belong to which frame"; there is no sorting or
#      transformation based on color ids. The variant is purely about how larger counts are drawn.


MAX_DOTS_PER_ROW = 4
DOT_ROW_OFFSETS = (0, -1, 1)


def _tally_positions_from_bbox(bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    """Enumerate legal tally cells inside a frame in fixed reading order."""
    r0, c0, r1, c1 = map(int, bbox)
    r_mid = (r0 + r1) // 2
    positions: list[tuple[int, int]] = []
    remaining = int(count)
    for row_offset in DOT_ROW_OFFSETS:
        rr = r_mid + row_offset
        if not (r0 < rr < r1):
            continue
        for i in range(MAX_DOTS_PER_ROW):
            if remaining <= 0:
                return positions
            cc = int(c1 - 2 - 2 * i)
            if cc > c0:
                positions.append((rr, cc))
                remaining -= 1
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and represent the tally as dots inside.

    Variant D16:
    - The first 4 dots still occupy the frame's center row, spaced by 2 from the right.
    - Extra dots continue on the row above the center, then on the row below.
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

        for rr, cc in _tally_positions_from_bbox(tuple(map(int, f["bbox"])), num_patterns):
            out[int(rr), int(cc)] = col

    return out


# --------------- generator (D16 variant) ---------------

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
TOTAL_PATTERNS_RANGE = (8, 14)
MAX_DOTS_PER_FRAME = MAX_DOTS_PER_ROW * len(DOT_ROW_OFFSETS)
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 7)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with a multi-row tally."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for rr, cc in _tally_positions_from_bbox((0, 0, FRAME_H - 1, FRAME_W - 1), n_dots):
        patch[int(rr), int(cc)] = color
    return patch


def _distribute_with_cap(
    rng: np.random.Generator, n_slots: int, total: int, cap: int
) -> list[int]:
    """Randomly distribute `total` across `n_slots`, with each slot <= cap."""
    counts = [0] * int(n_slots)
    order = rng.permutation(n_slots)
    remaining = int(total)
    for pos, idx in enumerate(order):
        slots_left = n_slots - pos - 1
        lo = max(0, remaining - cap * slots_left)
        hi = min(cap, remaining)
        val = int(rng.integers(lo, hi + 1))
        counts[int(idx)] = val
        remaining -= val
    return counts


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """
    Distribute matching sprites while forcing the multi-row behavior to appear.

    Every sample has at least one frame with count > 4.
    Some samples also have one frame with count > 8 so the third tally row appears.
    """
    counts = [0] * int(n_frames)
    focus_idx = int(rng.integers(0, n_frames))
    use_third_row = (total_match >= 9) and (float(rng.random()) < 0.45)
    focus_lo = 9 if use_third_row else 5
    focus_hi = min(MAX_DOTS_PER_FRAME, int(total_match))
    focus_count = int(rng.integers(focus_lo, focus_hi + 1))
    counts[focus_idx] = focus_count

    remainder = int(total_match - focus_count)
    other_indices = [i for i in range(n_frames) if i != focus_idx]
    if other_indices and remainder > 0:
        extras = _distribute_with_cap(rng, len(other_indices), remainder, MAX_DOTS_PER_FRAME)
        for idx, val in zip(other_indices, extras):
            counts[int(idx)] = int(val)
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
        patch[int(r - r_min), int(c - c_min)] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = int(total_patterns - n_distractors)

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
    for r0, count in zip(frame_rows, counts):
        bbox = (r0, FRAME_C0, r0 + FRAME_H - 1, FRAME_C0 + FRAME_W - 1)
        for rr, cc in _tally_positions_from_bbox(bbox, count):
            inp[int(rr), int(cc)] = bg

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
