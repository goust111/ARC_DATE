# Combo variant of 8698868d (index 29)
# combo_spec: D2/0+D3/1+D18/0
# Name: 矩形精灵+矩形框架+厚边框

from __future__ import annotations

# Design analysis
# =================
# a) Concrete behavior
# The original task pairs square sprites with square frames and uses a 1-pixel frame
# border, so the size relation is `sprite = frame - 2` in both axes. This combo changes
# three geometry facts at once:
# - frames are rectangles with `frame_h != frame_w`,
# - sprites are rectangles with `sprite_h != sprite_w`,
# - the frame border thickness is `t = 2`, so the nesting relation becomes
#   `(sprite_h, sprite_w) = (frame_h - 2t, frame_w - 2t) = (frame_h - 4, frame_w - 4)`.
# A concrete trace:
# - Original style: a 7x7 frame contains a 5x5 sprite, inserted at offset `(1, 1)`.
# - Combo style: a 9x10 frame contains a 5x6 sprite, inserted at offset `(2, 2)`.
# - Frame holes are counted only inside the 2-pixel-protected cavity `region[2:-2, 2:-2]`.
# - Sprite holes are still counted as 4-connected non-sprite-color components, but now in
#   a rectangular sprite whose own solid rim remains 1 pixel thick.
#
# b) Complexity calibration
# The fused change is visually obvious because every example now shows non-square output
# tiles plus a noticeably thicker frame border. To keep the rule learnable, all frames in
# one sample share one rectangle size and all sprites share the matching `-4` rectangle
# size. Hole counts remain a small unique range, capped here at 0..3 by limiting the
# number of frame/sprite pairs to 2-4; this avoids over-constraining the reduced sprite
# interiors created by the thicker border.
#
# c) Hidden assumptions from the original solver that break here
# - `h == w` is no longer valid for either frames or sprites.
# - Grouping by a single `size` loses the independent height/width degrees of freedom.
# - The old relation `sprite_size = frame_size - 2` must become `(h-4, w-4)`.
# - The old frame interior `region[1:-1, 1:-1]` would miscount holes, because a 2-pixel
#   border means rows/cols at offset 1 are still guaranteed frame-colored.
# - Output placement at `+1` is wrong; the sprite must be inset by `+2`.
# - Grid stride must be `(frame_h, frame_w)`, not one square side length.
# The new solver therefore groups by `(h, w)`, detects the `(frame_h-4, frame_w-4)`
# sprite group, counts frame holes inside the cavity past the thick border, and rebuilds
# the output with a 2-pixel inset.
#
# d) Placement legality
# The generator places a tight rectangle grid of frames first, then scatters sprites into
# remaining background with padding so they cannot touch each other or the frame grid.
# Frame holes are sampled only inside the `2..h-3` / `2..w-3` cavity, guaranteeing a
# perfectly solid 2-pixel border on all four sides. Sprites keep a solid 1-pixel rim and
# holes only inside their own inner area, so every colored component stays connected.
#
# e) Visual observability
# A human can infer the combo from 2-4 examples because three cues co-occur in every
# sample: rectangular output tiles, thicker outer borders, and free sprites whose size is
# exactly the frame cavity size rather than the old `-2` square relation. The matching key
# remains the number of 4-connected hole components, which is observable from local shape.
#
# f) Visual cleanliness
# The output is still a clean packed grid with two clear layers: thick frame border outside
# and nested sprite inside. No annotation marks, overlaps, or color-based ordering tricks
# are introduced.
#
# g) Change inferability
# The only color change is still structural: sprite hole pixels are recolored to the color
# of the frame they are inserted into. Because the recoloring happens exactly where the
# sprite meets the frame cavity, the mapping is spatially evident and does not depend on
# hidden numeric properties of the colors.
#
# h) Variant metric for post-test validation
# A sample is combo-valid when the detected frame bbox is non-square and every detected
# sprite bbox equals `(frame_h - 4, frame_w - 4)`. Distribution over `(frame_h, frame_w)`
# and over the observed gaps `(frame_h - sprite_h, frame_w - sprite_w)` is checked after
# testing; a degenerate original-like sample would instead show squares and gap `(2, 2)`.
#
# i) New solver assumptions and generator guarantees
# - Solver assumes there is exactly one visible frame-size group whose paired sprite-size
#   group is `(frame_h - 4, frame_w - 4)` with the same multiplicity. Generator enforces
#   exactly that.
# - Solver assumes the frame grid is full and gapless. Generator always fills every slot.
# - Solver assumes sprite hole counts are unique and match frame hole counts bijectively.
#   Generator uses a consecutive range with no repeats.
# - Solver assumes the 2-pixel frame border is solid. Generator places frame holes only
#   inside the protected cavity so this always holds.

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


BG = 8
BORDER_THICKNESS = 2
SPRITE_BORDER = 1
FRAME_HEIGHT_RANGE = range(8, 11)
FRAME_WIDTH_RANGE = range(8, 11)
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 4)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 60
GEN_RETRIES = 400
MIN_FILL_RATIO = 0.55


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        _n = _r * _c
        if N_RANGE[0] <= _n <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_n, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _candidate_groups(grid: np.ndarray) -> dict[tuple[int, int], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    groups: dict[tuple[int, int], list[dict]] = {}
    for comp in comps:
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        area = int(comp["area"])
        if 4 <= h <= 10 and 4 <= w <= 10 and area >= int(h * w * MIN_FILL_RATIO):
            groups.setdefault((h, w), []).append(comp)
    return groups


def _select_frame_and_sprite_dims(groups: dict[tuple[int, int], list[dict]]) -> tuple[tuple[int, int], tuple[int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for frame_h, frame_w in groups:
        sprite_dims = (frame_h - 2 * BORDER_THICKNESS, frame_w - 2 * BORDER_THICKNESS)
        if frame_h == frame_w:
            continue
        if sprite_dims[0] < 4 or sprite_dims[1] < 4:
            continue
        if sprite_dims not in groups:
            continue
        frame_count = len(groups[(frame_h, frame_w)])
        sprite_count = len(groups[sprite_dims])
        if frame_count != sprite_count:
            continue
        candidates.append((frame_count, frame_h * frame_w, frame_h, frame_w))
    assert candidates
    candidates.sort(reverse=True)
    _count, _area, frame_h, frame_w = candidates[0]
    return (frame_h, frame_w), (frame_h - 2 * BORDER_THICKNESS, frame_w - 2 * BORDER_THICKNESS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match free rectangular sprites to rectangular thick-border frames by connected-hole count,
    then pack the matched sprites into the frame grid.

    Input structure:
    - Background is azure (8). One tight grid of same-sized rectangular frames appears
      somewhere on the canvas, and the same number of smaller rectangular sprites are
      scattered elsewhere on background cells.
    - Each frame has a solid 2-pixel border in its own color; only the inner cavity may
      contain bg holes. Each sprite has its own color, a solid 1-pixel rim, and may
      contain several bg hole components inside.

    Rule:
    - Infer the common frame size `(frame_h, frame_w)` and the sprite size
      `(frame_h-4, frame_w-4)`.
    - For every frame, count the number of 4-connected hole components inside the cavity
      past the 2-pixel border. For every sprite, count its 4-connected hole components.
    - Pair frames and sprites by equal hole-component count, place each sprite inside its
      matched frame at offset `(2, 2)`, and recolor the sprite holes to the frame color.

    Key insight: the correspondence is determined by the number of connected hole regions,
    while the fused geometry changes both axes and the insertion offset from `+1` to `+2`.
    """
    g = np.array(grid, dtype=int)
    groups = _candidate_groups(g)
    assert groups

    frame_dims, sprite_dims = _select_frame_and_sprite_dims(groups)
    frame_h, frame_w = frame_dims
    sprite_h, sprite_w = sprite_dims
    frames = groups[frame_dims]
    sprites_raw = groups[sprite_dims]
    assert frames and sprites_raw

    row_positions = sorted({int(comp["bbox"][0]) for comp in frames})
    col_positions = sorted({int(comp["bbox"][1]) for comp in frames})
    n_rows = len(row_positions)
    n_cols = len(col_positions)
    assert n_rows * n_cols == len(frames)
    assert all(row_positions[i] - row_positions[0] == i * frame_h for i in range(n_rows))
    assert all(col_positions[i] - col_positions[0] == i * frame_w for i in range(n_cols))

    frame_lookup = {(int(comp["bbox"][0]), int(comp["bbox"][1])): comp for comp in frames}
    frame_grid: list[dict[str, int]] = []
    for ri, src_r in enumerate(row_positions):
        for ci, src_c in enumerate(col_positions):
            comp = frame_lookup[(src_r, src_c)]
            region = crop_bbox(g, comp["bbox"])
            frame_color = int(comp["color"])
            inner = region[BORDER_THICKNESS:-BORDER_THICKNESS, BORDER_THICKNESS:-BORDER_THICKNESS]
            _lab, num_holes = label_mask(inner != frame_color, connectivity=4)
            frame_grid.append(
                {
                    "color": frame_color,
                    "holes": int(num_holes),
                    "out_r": int(ri * frame_h),
                    "out_c": int(ci * frame_w),
                }
            )

    sprite_by_holes: dict[int, np.ndarray] = {}
    sprite_color_by_holes: dict[int, int] = {}
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        _lab, num_holes = label_mask(sprite_region != sprite_color, connectivity=4)
        hole_count = int(num_holes)
        assert hole_count not in sprite_by_holes
        sprite_by_holes[hole_count] = sprite_region.copy()
        sprite_color_by_holes[hole_count] = sprite_color

    out = np.full((n_rows * frame_h, n_cols * frame_w), BG, dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        hole_count = int(frame["holes"])
        assert hole_count in sprite_by_holes

        out[fr:fr + frame_h, fc:fc + frame_w] = frame_color
        content = np.array(sprite_by_holes[hole_count], dtype=int).copy()
        sprite_color = int(sprite_color_by_holes[hole_count])
        content[content != sprite_color] = frame_color
        out[
            fr + BORDER_THICKNESS:fr + BORDER_THICKNESS + sprite_h,
            fc + BORDER_THICKNESS:fc + BORDER_THICKNESS + sprite_w,
        ] = content

    return out


def _max_isolated_count(height: int, width: int) -> int:
    if height <= 0 or width <= 0:
        return 0
    return int((height * width + 1) // 2)


def _place_isolated_seeds(height: int, width: int, count: int, margin: int) -> list[tuple[int, int]] | None:
    inner_positions = [
        (r, c)
        for r in range(margin, height - margin)
        for c in range(margin, width - margin)
    ]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite(height: int, width: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((height, width), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite

    hole_h = height - 2 * SPRITE_BORDER
    hole_w = width - 2 * SPRITE_BORDER
    if _max_isolated_count(hole_h, hole_w) < hole_count:
        return None

    seeds = _place_isolated_seeds(height, width, hole_count, SPRITE_BORDER)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, (hole_h * hole_w) // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, hole_h * hole_w - len(all_holes)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (SPRITE_BORDER <= nr < height - SPRITE_BORDER and SPRITE_BORDER <= nc < width - SPRITE_BORDER):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(comps)
                        if j != ci
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_holes.add(cell)
            sprite[cell[0], cell[1]] = BG

    _lab, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if int(num_holes) != hole_count:
        return None
    return sprite


def _make_frame(height: int, width: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((height, width), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    cavity_h = height - 2 * BORDER_THICKNESS
    cavity_w = width - 2 * BORDER_THICKNESS
    if _max_isolated_count(cavity_h, cavity_w) < hole_count:
        return None

    seeds = _place_isolated_seeds(height, width, hole_count, BORDER_THICKNESS)
    if seeds is None:
        return None
    for r, c in seeds:
        frame[r, c] = BG

    inner = frame[BORDER_THICKNESS:-BORDER_THICKNESS, BORDER_THICKNESS:-BORDER_THICKNESS]
    _lab, num_holes = label_mask(inner != frame_color, connectivity=4)
    if int(num_holes) != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_h: int,
    frame_w: int,
    sprite_h: int,
    sprite_w: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_h, cols_grid * frame_w), BG, dtype=int)
    sprite_by_k = {hole_count: patch for hole_count, patch in sprites}
    for idx, (hole_count, _patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = ri * frame_h
        cc = ci * frame_w
        frame_color = frame_colors[idx]
        expected[rr:rr + frame_h, cc:cc + frame_w] = frame_color
        content = sprite_by_k[hole_count].copy()
        content[content == BG] = frame_color
        expected[
            rr + BORDER_THICKNESS:rr + BORDER_THICKNESS + sprite_h,
            cc + BORDER_THICKNESS:cc + BORDER_THICKNESS + sprite_w,
        ] = content
    return expected


def _valid_frame_dims(rows_grid: int, cols_grid: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for frame_h in FRAME_HEIGHT_RANGE:
        for frame_w in FRAME_WIDTH_RANGE:
            if frame_h == frame_w:
                continue
            sprite_h = frame_h - 2 * BORDER_THICKNESS
            sprite_w = frame_w - 2 * BORDER_THICKNESS
            if sprite_h < 4 or sprite_w < 4:
                continue
            hole_h = sprite_h - 2 * SPRITE_BORDER
            hole_w = sprite_w - 2 * SPRITE_BORDER
            if _max_isolated_count(hole_h, hole_w) < max(_VALID_NS) - 1:
                continue
            min_h = max(CANVAS_MIN, rows_grid * frame_h + sprite_h + SPRITE_PADDING)
            min_w = max(CANVAS_MIN, cols_grid * frame_w + sprite_w + SPRITE_PADDING)
            if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
                dims.append((frame_h, frame_w))
    return dims


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])

    valid_dims = _valid_frame_dims(rows_grid, cols_grid)
    if not valid_dims:
        return None
    frame_h, frame_w = random.choice(valid_dims)
    sprite_h = frame_h - 2 * BORDER_THICKNESS
    sprite_w = frame_w - 2 * BORDER_THICKNESS

    grid_pixel_h = rows_grid * frame_h
    grid_pixel_w = cols_grid * frame_w
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_h + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_w + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    height = random.randint(min_h, CANVAS_MAX)
    width = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    # Keep colors disjoint across object roles so frames and free sprites remain
    # visually separable even before the hole-count matching rule is inferred.
    chosen_colors = random.sample(available, 2 * n_frames)
    frame_colors = chosen_colors[:n_frames]
    sprite_colors = chosen_colors[n_frames:]

    hole_h = sprite_h - 2 * SPRITE_BORDER
    hole_w = sprite_w - 2 * SPRITE_BORDER
    max_hole = _max_isolated_count(hole_h, hole_w)
    if max_hole < n_frames - 1:
        return None
    max_start = max(0, max_hole - (n_frames - 1))
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite(sprite_h, sprite_w, hole_count, sprite_colors[idx])
        if patch is None:
            return None
        sprites.append((hole_count, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_h, frame_w, hole_count, frame_colors[idx])
        if patch is None:
            return None
        frames.append((hole_count, patch))

    canvas = np.full((height, width), BG, dtype=int)
    r0 = random.randint(0, height - grid_pixel_h)
    c0 = random.randint(0, width - grid_pixel_w)
    for idx, (_hole_count, patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_h
        cc = c0 + ci * frame_w
        canvas[rr:rr + frame_h, cc:cc + frame_w] = patch

    for _hole_count, patch in sprites:
        try:
            r, c = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_h,
        frame_w,
        sprite_h,
        sprite_w,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
