# Variant of 8f215267
# Dimension: D3 — 目标对象：框架结构 (dst)
# Change: 5x11的空心矩形框架，仅由周边一圈像素组成，垂直排列在网格左半部分，间隔2行
#   → 不同尺寸框架：仍为 5 行高的空心矩形，但宽度在 7/9/11 之间变化；点位仍在中心行从右向左每隔 2 列放置，
#   只是可用容量随框宽自适应变化。

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the D3 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     old: every frame is exactly 5x11, so the solver/generator can silently rely on a fixed 4-dot capacity.
#     new: frames keep the same vertical stacking pattern and hollow-rectangle appearance, but each frame width is
#       chosen independently from {7, 9, 11}. A 5x7 frame can show 2 dots, 5x9 can show 3, 5x11 can show 4.
#   concrete_visuals:
#     - Left half: 3-4 hollow monochrome frames, same top-left column, 2 blank rows between frames, all height=5.
#     - Right half: scattered monochrome sprites, exactly as before.
#     - Output: sprites disappear; each frame displays the number of same-color sprites on its center row.
#       Dot columns are derived from the actual frame width: right inner cell, then every 2 columns leftward.
#   behavior_trace:
#     - Original 5x11 logic: a frame with 3 matching sprites gets dots at relative columns [8,6,4].
#     - Variant 5x7 logic: the same "start from right, step by 2" rule yields only [4,2]; a third dot would not fit.
#     - Therefore the generator must respect per-frame capacity, and the solver must infer dot slots from bbox width.
#   complexity_calibration:
#     - Width choices are limited to three odd sizes {7,9,11}; this keeps the rule learnable from a few examples.
#     - Each generated puzzle contains at least two distinct widths, so the variant is visually observable.
#     - Colors and sprite logic stay unchanged to avoid introducing unrelated reasoning dimensions.
#   implicit_assumption_audit:
#     - The old solver's frame test was "large + sparse". That remains broadly valid, but here we tighten it to
#       an exact hollow-rectangle test because the generator guarantees perimeter-only frames in the input.
#     - The old dot-placement loop already stepped by 2 from the right; what was implicit is the capacity bound.
#       The new solver computes legal dot columns from the detected bbox width instead of relying on a fixed width.
#   placement_legality:
#     - Max frame width stays 11, so the original 23-column canvas and right sprite region remain feasible.
#     - Frames are left-aligned and never overlap because height is fixed and row gaps remain 2.
#     - Sprites are still placed only in the right canvas using random_free_location with padding.
#   observability_and_cleanliness:
#     - Width differences are immediately visible because right edges no longer align.
#     - Dot spacing stays uniform, so human solvers can compare wide vs narrow frames and infer capacity from shape.
#     - No invisible ordering by color value is used; all reasoning is grounded in visible color matching and size.


# --------------- shared geometry ---------------

FRAME_H = 5
FRAME_WIDTH_CHOICES = (7, 9, 11)
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

N_FRAMES_RANGE = (3, 5)        # 3-4 frames
N_DISTRACTORS_RANGE = (1, 3)   # 1-2 distractors
MATCH_TOTAL_RANGE = (5, 9)     # 5-8 matching sprites total
BOTTOM_PAD_RANGE = (0, 7)

SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MAX_ATTEMPTS = 300


def _frame_dot_cols(width: int) -> list[int]:
    """Return in-patch dot columns for a frame of the given width."""
    return [int(width - 3 - 2 * i) for i in range((int(width) - 3) // 2)]


def _is_hollow_frame(comp: dict) -> bool:
    """
    Exact input-frame recognizer for this variant.

    Assumptions enforced by generate_input():
    - frames are monochrome
    - height is always 5
    - width is one of 7/9/11
    - input frames contain only the perimeter pixels (dots are absent in the input)
    """
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    if h != FRAME_H or w not in FRAME_WIDTH_CHOICES:
        return False

    expected_area = 2 * h + 2 * w - 4
    if int(comp["area"]) != expected_area:
        return False

    r0, c0, r1, c1 = map(int, b)
    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    border = set()
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    return coords == border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove all scattered sprites and tally same-color sprite counts inside each frame.

    Variant-specific change:
    - the tally layout is inferred from each frame's actual width rather than a fixed 5x11 template.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_hollow_frame(comp):
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

        r0, c0, _r1, _c1 = map(int, f["bbox"])
        width = bbox_w(f["bbox"])
        r_mid = r0 + FRAME_H // 2
        for dc in _frame_dot_cols(width)[:num_patterns]:
            out[r_mid, c0 + int(dc)] = col

    return out


# --------------- generator ---------------

def _frame_patch(color: int, bg: int, width: int, n_dots: int) -> np.ndarray:
    """Build a 5 x width hollow rectangle with n_dots on the center row."""
    patch = np.full((FRAME_H, width), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    patch[mid_r, _frame_dot_cols(width)[:n_dots]] = color
    return patch


def _sample_frame_widths(rng: np.random.Generator, n_frames: int) -> list[int]:
    """
    Sample widths with guaranteed variability.

    Every puzzle contains at least two distinct widths and always includes a non-11 frame,
    so the D3 change is visible instead of degenerating back to the original all-11 layout.
    """
    widths = list(FRAME_WIDTH_CHOICES)
    if n_frames == 4:
        widths.append(int(rng.choice(FRAME_WIDTH_CHOICES)))
    rng.shuffle(widths)
    return [int(w) for w in widths[:n_frames]]


def _sample_counts(
    rng: np.random.Generator,
    capacities: list[int],
    total_match: int,
) -> list[int]:
    """Distribute total_match across frames, respecting each frame's width-derived capacity."""
    n_frames = len(capacities)
    n_active = int(rng.integers(max(2, n_frames - 1), n_frames + 1))
    active = set(int(x) for x in rng.choice(n_frames, size=n_active, replace=False))

    counts = [1 if i in active else 0 for i in range(n_frames)]
    residual_caps = [cap - counts[i] for i, cap in enumerate(capacities)]
    remaining = total_match - sum(counts)
    order = [int(x) for x in rng.permutation(n_frames)]

    for pos, idx in enumerate(order):
        caps_left = sum(residual_caps[j] for j in order[pos + 1 :])
        lo = max(0, remaining - caps_left)
        hi = min(residual_caps[idx], remaining)
        add = int(rng.integers(lo, hi + 1))
        counts[idx] += add
        remaining -= add

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
    frame_widths = _sample_frame_widths(rng, n_frames)
    capacities = [len(_frame_dot_cols(w)) for w in frame_widths]

    max_match = min(MATCH_TOTAL_RANGE[1] - 1, sum(capacities))
    min_match = min(MATCH_TOTAL_RANGE[0], max_match)
    total_match = int(rng.integers(min_match, max_match + 1))

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, capacities, total_match)

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, width, count) in enumerate(zip(frame_colors, frame_widths, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, width, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, width, count in zip(frame_rows, frame_widths, counts):
        mid_r = r0 + FRAME_H // 2
        for dc in _frame_dot_cols(width)[:count]:
            inp[mid_r, FRAME_C0 + int(dc)] = bg

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
