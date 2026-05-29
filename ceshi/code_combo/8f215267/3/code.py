# Combo variant of 8f215267 (index 3)
# combo_spec: D1/0+D6/2
# Name: 形状匹配+面积超标干扰

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

combo_summary:
  - Keep the original left/right scene grammar: 5x11 hollow frames on the left,
    scattered monochrome sprites on the right, output removes the sprites and draws
    right-aligned dots inside each frame.
  - D1 changes the grouping key from color to shape.
  - D6 adds an area threshold: sprites with area > 5 are removed but never counted.
  - To make both dimensions act on the same objects, a frame marker defines a shape
    family and right-side sprites may be either:
      * scale-1 copies of that marker, area 2-3, therefore countable; or
      * scale-2 copies of that same marker, area 8-12, therefore oversized distractors.

concretization_and_behavior_trace:
  - Frame appearance:
      * each frame is still one monochrome component because its marker is attached
        to the upper-left inner wall by 4-neighborhood contact;
      * marker choices come from a small catalog of 2-3-cell asymmetric shapes.
  - Right-side sprite appearance:
      * counted sprites are tight-cropped copies of those marker shapes;
      * oversized distractors are uniform 2x enlargements of the same shape families.
  - Example:
      * frame A has marker = vertical domino;
      * right side contains two 2-cell vertical dominoes and one 8-cell 2x-enlarged
        vertical domino, all in arbitrary non-frame colors.
      * original solver: color grouping would fail completely because sprite colors
        do not match frame colors.
      * D1-only logic: all three vertical-domino family sprites would be assigned to
        frame A and produce 3 dots.
      * combo logic: the family match still assigns all three to frame A first, but
        the enlarged one has area 8 > 5, so only 2 dots are drawn.
      * difference: the decisive chain is "same shape family" then "small enough".

complexity_calibration:
  - Marker catalog size is 5, enough for 3-4 frames without reuse, so shape identity
    is visible and unambiguous within a sample.
  - Countable sprites are restricted to areas 2-3; oversized distractors are forced
    to 8-12 by scale-2 enlargement. The threshold split at 5 is visually obvious.
  - Sprite colors are drawn from a palette disjoint from frame colors, so color is a
    visible decoy and cannot accidentally explain the mapping.
  - Every generated instance is forced to contain at least one oversized sprite whose
    shape family matches an active frame and whose extra count would change the dot
    display under a shape-only solver.

implicit_assumption_review:
  - Original `num_patterns = sum(... if p["color"] == col)` is invalid because color
    no longer determines ownership.
  - D1's exact-shape equality would make D6 impossible on the same family because a
    larger sprite cannot be exactly equal to a 2-3-cell marker. This combo therefore
    lifts "shape match" to scale-normalized silhouette matching.
  - Frame detection remains valid because only frames have both bbox dimensions >= 5;
    the largest enlarged sprites are at most 6x2, 2x6, or 4x4.
  - The solver now assumes that any countable sprite is a uniform integer-scaled copy
    of some marker silhouette. The generator guarantees that by construction.
  - Dots still fit because counted sprites per frame are capped at 4, and oversized
    distractors are placed only on frames with visible spare capacity under the
    shape-only baseline.

placement_legality:
  - Marker cells stay in the upper-left interior and never touch the right-aligned
    dot columns.
  - Right-side placement still uses `random_free_location(..., padding=1)`, so no
    sprite overlaps or touches another sprite or a frame.
  - Enlarged sprites fit in the 9-column right strip because their bboxes are at most
    6 columns wide and 6 rows tall.

visual_observability_and_cleanliness:
  - The human can compare the tiny marker inside a frame with repeated right-side
    silhouettes in both small and enlarged form.
  - Because every sprite disappears in the output, examples reveal that some same-
    family sprites count while larger same-family sprites do not.
  - Left frames remain clean, and the only extra visual layer is the small marker in
    a consistent corner, so the scene stays readable from 2-4 examples.

generator_guards_and_metrics:
  - Guard 1: combo solver output must equal the constructed output.
  - Guard 2: a shape-only baseline (ignore area) must disagree, proving oversized
    same-family distractors are active.
  - Guard 3: a color+area baseline must disagree, proving shape matching is active.
  - Quantitative metric used after testing: per sample, count
    `oversized_shape_matches = number of sprites whose normalized shape family matches
    some frame marker and whose area > 5`. This must be >= 1 in generated instances.
"""

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


COUNTED_MAX_AREA = 5
LARGE_SCALE = 2

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
N_FRAMES_RANGE = (3, 5)         # 3-4 frames
N_DISTRACTORS_RANGE = (1, 3)    # 1-2 oversized distractors
TOTAL_PATTERNS_RANGE = (5, 11)  # 5-10 total right-side sprites
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- placement / retries ---
SPRITE_PADDING = 1
MAX_ATTEMPTS = 600


# Attached frame markers in frame-local coordinates.
FRAME_MARKERS: dict[str, list[tuple[int, int]]] = {
    "h2": [(1, 1), (1, 2)],
    "v2": [(1, 1), (2, 1)],
    "L3": [(1, 1), (1, 2), (2, 1)],
    "h3": [(1, 1), (1, 2), (1, 3)],
    "v3": [(1, 1), (2, 1), (3, 1)],
}


def _shape_signature(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    arr = np.array(coords, dtype=int)
    if arr.size == 0:
        return (0, 0, ())
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    rel = [(int(r - r0), int(c - c0)) for r, c in arr]
    h = int(arr[:, 0].max() - r0 + 1)
    w = int(arr[:, 1].max() - c0 + 1)
    return (h, w, tuple(sorted(rel)))


def _crop_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(_shape_signature(cells)[2])


MARKER_SPRITES: dict[str, list[tuple[int, int]]] = {
    name: _crop_cells(cells) for name, cells in FRAME_MARKERS.items()
}


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    coords = np.argwhere(mask)
    return _shape_signature([(int(r), int(c)) for r, c in coords])


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _uniform_block_compress(mask: np.ndarray, scale: int) -> np.ndarray | None:
    h, w = map(int, mask.shape)
    scale = int(scale)
    if scale <= 1 or h % scale != 0 or w % scale != 0:
        return None
    blocks = mask.reshape(h // scale, scale, w // scale, scale).transpose(0, 2, 1, 3)
    sums = blocks.sum(axis=(2, 3))
    full = scale * scale
    if not np.all((sums == 0) | (sums == full)):
        return None
    return sums == full


def _component_base_signature(comp: dict) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], int]:
    """
    Normalize a sprite to its unscaled silhouette.

    Assumption: generated sprites are either exact marker copies (scale 1) or uniform
    integer enlargements of those copies. The generator enforces this.
    """
    mask = _component_mask(comp)
    best_sig = _mask_signature(mask)
    best_scale = 1
    for scale in range(min(mask.shape), 1, -1):
        shrunk = _uniform_block_compress(mask, scale)
        if shrunk is None:
            continue
        best_sig = _mask_signature(shrunk)
        best_scale = int(scale)
        break
    return best_sig, best_scale


def _frame_marker_signature(comp: dict) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    interior = [
        (int(r - r0), int(c - c0))
        for r, c in comp["coords"]
        if r0 < int(r) < r1 and c0 < int(c) < c1
    ]
    return _shape_signature(interior)


def _frame_patch(color: int, bg: int, n_dots: int, marker_name: str) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for r, c in FRAME_MARKERS[marker_name]:
        patch[int(r), int(c)] = color
    mid_r = FRAME_H // 2
    for i in range(int(n_dots)):
        dot_c = int(FRAME_W - 3 - 2 * i)
        patch[int(mid_r), dot_c] = color
    return patch


def _sprite_patch(shape_cells: list[tuple[int, int]], color: int, bg: int, scale: int = 1) -> np.ndarray:
    rows = [r for r, _ in shape_cells]
    cols = [c for _, c in shape_cells]
    mask = np.zeros((max(rows) + 1, max(cols) + 1), dtype=bool)
    for r, c in shape_cells:
        mask[int(r), int(c)] = True
    if int(scale) > 1:
        mask = np.kron(mask.astype(int), np.ones((int(scale), int(scale)), dtype=int)).astype(bool)
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
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


def _pick_oversized_indices(
    rng: np.random.Generator, counts: list[int], n_distractors: int
) -> list[int]:
    """
    Choose frame indices whose shape families will get enlarged distractors.

    At least one selected frame must have visible spare capacity so a shape-only
    baseline would draw extra dots.
    """
    preferred = [i for i, count in enumerate(counts) if 0 < count < MAX_DOTS_PER_FRAME]
    fallback = [i for i, count in enumerate(counts) if count < MAX_DOTS_PER_FRAME]
    pool = preferred if preferred else fallback
    if not pool:
        return []

    anchor = int(rng.choice(pool))
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


def _solve(
    grid: np.ndarray,
    *,
    match_mode: str,
    counted_max_area: int | None,
) -> np.ndarray:
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

    # New assumption audit: every non-frame component is a right-side sprite, and the
    # generator guarantees those sprites are isolated monochrome components.
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = bg

    sprite_infos = []
    for sprite in sprites:
        base_sig, scale = _component_base_signature(sprite)
        sprite_infos.append(
            {
                "color": int(sprite["color"]),
                "area": int(sprite["area"]),
                "base_sig": base_sig,
                "scale": int(scale),
            }
        )

    for frame in frames:
        marker_sig = _frame_marker_signature(frame)
        num_patterns = 0
        for sprite in sprite_infos:
            if match_mode == "shape":
                if sprite["base_sig"] != marker_sig:
                    continue
            elif match_mode == "color":
                if sprite["color"] != int(frame["color"]):
                    continue
            else:  # pragma: no cover - internal misuse guard
                raise ValueError(f"unknown match_mode: {match_mode}")

            if counted_max_area is not None and sprite["area"] > int(counted_max_area):
                continue
            num_patterns += 1

        if num_patterns <= 0:
            continue

        r0, c0, _r1, c1 = map(int, frame["bbox"])
        r_mid = int(r0 + FRAME_H // 2)
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = int(frame["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count small shape-matching sprites per frame and show the tallies as dots.

    Input structure:
    - bg is the most frequent color.
    - The left side contains hollow 5x11 frames, each with a tiny attached marker in
      its upper-left interior.
    - The right side contains scattered monochrome sprites that may be either small or
      enlarged versions of those marker silhouettes, in colors unrelated to the frames.

    Rule:
    - Remove every right-side sprite from the output.
    - For each frame, read its marker silhouette and group right-side sprites by the
      same shape family, ignoring color and allowing uniform enlargement.
    - Count only the grouped sprites whose area is at most 5.
    - Draw that many dots on the frame's center row, spaced every 2 columns from the
      right edge inward.

    Key insight:
    - Shape decides which frame a sprite belongs to, but size decides whether that
      sprite is counted at all; enlarged same-shape copies are distractors.
    """
    return _solve(grid, match_mode="shape", counted_max_area=COUNTED_MAX_AREA)


def analyze_fusion_instance(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the generated instance truly exercises the combo behavior.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    sprites: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        if (h >= 5) and (w >= 5) and (area < (h * w * 0.7)):
            frames.append(comp)
        else:
            sprites.append(comp)

    marker_sigs = {_frame_marker_signature(frame) for frame in frames}
    oversized_shape_matches = 0
    small_shape_matches = 0
    for sprite in sprites:
        base_sig, _scale = _component_base_signature(sprite)
        if base_sig not in marker_sigs:
            continue
        if int(sprite["area"]) > COUNTED_MAX_AREA:
            oversized_shape_matches += 1
        else:
            small_shape_matches += 1

    combo = main(g)
    shape_only = _solve(g, match_mode="shape", counted_max_area=None)
    color_area = _solve(g, match_mode="color", counted_max_area=COUNTED_MAX_AREA)
    return {
        "oversized_shape_matches": int(oversized_shape_matches),
        "small_shape_matches": int(small_shape_matches),
        "diff_vs_shape_only": int(not np.array_equal(combo, shape_only)),
        "diff_vs_color_area": int(not np.array_equal(combo, color_area)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    if total_match <= 0 or total_match > n_frames * MAX_DOTS_PER_FRAME:
        return None

    counts = _sample_counts(rng, n_frames, total_match)
    if sum(v > 0 for v in counts) < 2:
        return None

    oversized_indices = _pick_oversized_indices(rng, counts, n_distractors)
    if len(oversized_indices) != n_distractors:
        return None

    marker_names = list(rng.choice(list(FRAME_MARKERS.keys()), size=n_frames, replace=False))

    n_sprite_colors = int(rng.integers(2, 4))
    palette = list(rng.choice(np.arange(10), size=1 + n_frames + n_sprite_colors, replace=False))
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    sprite_colors = [int(x) for x in palette[1 + n_frames :]]

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count, marker_name) in enumerate(zip(frame_colors, counts, marker_names)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count, marker_name), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    sprite_specs: list[tuple[str, int, int]] = []
    for marker_name, count in zip(marker_names, counts):
        for _ in range(count):
            color = int(sprite_colors[int(rng.integers(0, len(sprite_colors)))])
            sprite_specs.append((marker_name, color, 1))

    for idx in oversized_indices:
        marker_name = marker_names[int(idx)]
        color = int(sprite_colors[int(rng.integers(0, len(sprite_colors)))])
        sprite_specs.append((marker_name, color, LARGE_SCALE))

    rng.shuffle(sprite_specs)

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    try:
        for marker_name, color, scale in sprite_specs:
            sprite = _sprite_patch(MARKER_SPRITES[marker_name], color, bg, scale=scale)
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

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_solve(inp, match_mode="shape", counted_max_area=None), out):
        return None
    if np.array_equal(_solve(inp, match_mode="color", counted_max_area=COUNTED_MAX_AREA), out):
        return None

    metrics = analyze_fusion_instance(inp)
    if metrics["oversized_shape_matches"] < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
