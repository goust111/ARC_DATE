# Combo variant of 8698868d (index 9)
# combo_spec: D2/0+D3/1
# Name: 矩形精灵嵌入矩形框架

from __future__ import annotations

# Design analysis
# =================
# a) Concrete behavior
# The original task has a tight grid of equal-size square frames and a matching set of
# free square sprites elsewhere on the canvas. A frame with k isolated interior holes is
# paired with the sprite that has k 4-connected hole components, then the sprite is
# nested into the frame and its hole pixels are recolored to the frame color.
#
# This combo changes both sides of that relation at once:
# - Source objects: the free sprites are no longer square; they are rectangles with a
#   solid 1-pixel border and an interior that can contain k connected bg-hole components.
# - Destination objects: the frame grid is also rectangular rather than square.
# A concrete trace is:
# - Original: frame 7x7, sprite 5x5, output step is 7 rows by 7 cols.
# - Combo: frame 6x8, sprite 4x6, output step is 6 rows by 8 cols.
# - Original nesting only needs one scalar side length; combo nesting must preserve both
#   height and width, so the solver must infer `(frame_h, frame_w)` and `(sprite_h,
#   sprite_w)` together.
#
# b) Complexity calibration
# The variation is visible without becoming noisy. Each sample still uses one shared
# frame size and one shared sprite size, so the concept remains learnable from 2-4 ARC
# examples. The added difficulty is exactly the intended one: the solver must reason in
# two dimensions instead of assuming a square cavity.
#
# c) Hidden assumptions in the original solver that break here
# The original code depends on several square-only assumptions:
# - `h == w` when collecting candidate objects.
# - Grouping by one integer size rather than by `(h, w)`.
# - Computing grid strides and output shape from one `frame_size`.
# - Assuming the nested sprite size is `frame_size - 2` in both axes.
# Those assumptions all fail under this combo. The repaired solver therefore:
# - keeps exact component dimensions,
# - selects a non-square frame group whose inner dimensions also exist as sprite dims,
# - reconstructs the grid with separate row and column strides,
# - inserts sprites using explicit `sprite_h` and `sprite_w`.
#
# d) Placement legality
# The generator uses output-first construction. It first chooses one rectangular frame
# size and derives the matching rectangular sprite size. The frame grid is placed as a
# gapless block. Free sprites are then scattered only in remaining bg cells with padding,
# so they cannot overlap each other or the frame grid.
#
# e) Visual observability
# A human can see the difference because both the scattered sprites and the final frame
# grid are obviously rectangular in the same sample. The transformation is still uniform:
# every output cell block is "frame border + same-aspect-ratio sprite inside", matched by
# hole-component count. No invisible numeric convention is required.
#
# f) Visual cleanliness
# The output remains tidy: one rectangular frame per slot, one nested rectangular sprite
# per slot, no extra overlays or annotation colors. The frame holes are sparse single bg
# cells in the input, while sprite holes can be multi-cell components; this keeps the two
# object types visually distinguishable.
#
# g) Color inference
# Color behavior is unchanged from the base task and remains easy to infer. The sprite's
# own color stays on solid cells, and only the sprite's hole pixels are recolored to the
# surrounding frame color after nesting. This recoloring is strongly anchored by the
# enclosing rectangle, so no arbitrary color mapping is introduced.
#
# h) Variant metric for post-test validation
# A generated sample is counted as combo-valid when both `frame_h != frame_w` and
# `sprite_h != sprite_w`. Because sprite dims are derived as `(frame_h - 2, frame_w - 2)`,
# this metric directly checks that the sample does not collapse back to the square task.
#
# i) New solver assumptions and generator guarantees
# - Solver assumes exactly one visible frame-dimension group `(frame_h, frame_w)` with
#   `frame_h != frame_w` and a corresponding sprite group `(frame_h - 2, frame_w - 2)`.
#   Generator constructs exactly one such pair.
# - Solver assumes the frame grid is fully populated and tightly tiled. Generator always
#   places a complete rectangular grid with no missing slots.
# - Solver assumes hole counts are unique across frames and across sprites. Generator uses
#   one consecutive range with no repeats.
# - To keep the nested sprite visually observable, generator assigns colors so each
#   matched frame/sprite pair differs in color.
# - Solver assumes all free sprites have the same rectangular dimensions. Generator enforces
#   that by deriving every sprite from the chosen frame size.

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


BG = 8
FRAME_HEIGHT_RANGE = range(6, 9)
FRAME_WIDTH_RANGE = range(6, 10)
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300
MIN_FILL_RATIO = 0.5


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _rows * _cols <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_rows * _cols, []).append((_rows, _cols))
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
        sprite_dims = (frame_h - 2, frame_w - 2)
        if frame_h < 6 or frame_w < 6 or frame_h == frame_w:
            continue
        if sprite_dims not in groups or sprite_dims[0] == sprite_dims[1]:
            continue
        candidates.append((frame_h * frame_w, len(groups[(frame_h, frame_w)]), frame_h, frame_w))
    assert candidates
    candidates.sort(reverse=True)
    _area, _count, frame_h, frame_w = candidates[0]
    return (frame_h, frame_w), (frame_h - 2, frame_w - 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match rectangular sprites to rectangular frames by connected-hole count, then nest
    each sprite inside its frame.

    Input structure:
    - bg = azure (8). A tight grid of same-size non-square rectangular frames appears in
      one region, and same-size smaller rectangular sprites are scattered elsewhere.
    - Every frame has a solid 1-pixel border of its own color and some isolated bg holes
      in the interior. Every sprite has its own solid-colored border and may contain
      several 4-connected bg-hole components inside.

    Rule:
    - Detect the shared frame dimensions `(frame_h, frame_w)` and the shared sprite
      dimensions `(frame_h-2, frame_w-2)`.
    - Count connected hole components inside each frame interior and inside each sprite.
    - Pair each frame with the sprite that has the same hole-component count.
    - Output only the tight frame grid, inserting the matched sprite into each frame and
      recoloring the sprite's hole pixels to the frame color.

    Key insight:
    The match key is still the number of connected holes, but this combo also requires
    preserving two-dimensional shape: nesting only works if height and width are inferred
    separately instead of assuming every object is square.
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
    frame_grid: list[dict] = []
    for ri, src_r in enumerate(row_positions):
        for ci, src_c in enumerate(col_positions):
            comp = frame_lookup[(src_r, src_c)]
            region = crop_bbox(g, comp["bbox"])
            frame_color = int(comp["color"])
            inner = region[1:-1, 1:-1]
            _labels, num_holes = label_mask(inner != frame_color, connectivity=4)
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
        _labels, num_holes = label_mask(sprite_region != sprite_color, connectivity=4)
        hole_count = int(num_holes)
        assert hole_count not in sprite_by_holes
        sprite_by_holes[hole_count] = sprite_region.copy()
        sprite_color_by_holes[hole_count] = sprite_color

    out = np.full((n_rows * frame_h, n_cols * frame_w), BG, dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_h, fc : fc + frame_w] = frame_color

        hole_count = int(frame["holes"])
        assert hole_count in sprite_by_holes
        sprite = np.array(sprite_by_holes[hole_count], dtype=int).copy()
        sprite_color = int(sprite_color_by_holes[hole_count])
        sprite[sprite != sprite_color] = frame_color
        out[fr + 1 : fr + 1 + sprite_h, fc + 1 : fc + 1 + sprite_w] = sprite

    return out


def _place_isolated_seeds(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    inner_positions = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
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

    inner_height = height - 2
    inner_width = width - 2
    inner_cells = inner_height * inner_width
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(height, width, hole_count)
    if seeds is None:
        return None

    components: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(components):
        grow_steps = random.randint(0, min(max_grow, inner_cells - len(all_holes)))
        for _ in range(grow_steps):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= height - 2 and 1 <= nc <= width - 2):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(components)
                        if j != comp_idx
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

    _labels, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if int(num_holes) != hole_count:
        return None
    return sprite


def _make_frame(height: int, width: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((height, width), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    inner_cells = (height - 2) * (width - 2)
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(height, width, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _labels, num_holes = label_mask(frame != frame_color, connectivity=4)
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
    sprite_by_holes = {hole_count: patch for hole_count, patch in sprites}
    for idx, (hole_count, _patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = ri * frame_h
        cc = ci * frame_w
        frame_color = frame_colors[idx]
        expected[rr : rr + frame_h, cc : cc + frame_w] = frame_color
        content = sprite_by_holes[hole_count].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_h, cc + 1 : cc + 1 + sprite_w] = content
    return expected


def _valid_frame_dims(rows_grid: int, cols_grid: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for frame_h in FRAME_HEIGHT_RANGE:
        for frame_w in FRAME_WIDTH_RANGE:
            if frame_h == frame_w:
                continue
            sprite_h = frame_h - 2
            sprite_w = frame_w - 2
            min_h = max(CANVAS_MIN, rows_grid * frame_h + sprite_h + SPRITE_PADDING)
            min_w = max(CANVAS_MIN, cols_grid * frame_w + sprite_w + SPRITE_PADDING)
            if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
                dims.append((frame_h, frame_w))
    return dims


def _assign_distinct_sprite_colors(
    hole_counts: list[int],
    frame_color_by_hole: dict[int, int],
    sprite_colors: list[int],
) -> dict[int, int] | None:
    remaining = sprite_colors[:]
    assignment: dict[int, int] = {}

    def backtrack(idx: int) -> bool:
        if idx == len(hole_counts):
            return True
        hole_count = hole_counts[idx]
        candidates = [color for color in remaining if color != frame_color_by_hole[hole_count]]
        random.shuffle(candidates)
        for color in candidates:
            remaining.remove(color)
            assignment[hole_count] = color
            if backtrack(idx + 1):
                return True
            remaining.append(color)
            del assignment[hole_count]
        return False

    if not backtrack(0):
        return None
    return assignment


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])

    valid_dims = _valid_frame_dims(rows_grid, cols_grid)
    if not valid_dims:
        return None
    frame_h, frame_w = random.choice(valid_dims)
    sprite_h, sprite_w = frame_h - 2, frame_w - 2

    grid_pixel_h = rows_grid * frame_h
    grid_pixel_w = cols_grid * frame_w
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_h + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_w + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    height = random.randint(min_h, CANVAS_MAX)
    width = random.randint(min_w, CANVAS_MAX)

    available_colors = [color for color in range(10) if color != BG]
    inner_cells = sprite_h * sprite_w
    max_holes = (inner_cells + 1) // 2
    if n_frames > max_holes + 1:
        return None
    max_start = max(0, max_holes - (n_frames - 1))
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    frame_colors = random.sample(available_colors, n_frames)
    frame_color_by_hole = {hole_count: frame_colors[idx] for idx, hole_count in enumerate(hole_counts)}
    sprite_colors = random.sample(available_colors, n_frames)
    sprite_color_by_hole = _assign_distinct_sprite_colors(hole_counts, frame_color_by_hole, sprite_colors)
    if sprite_color_by_hole is None:
        return None

    sprites: list[tuple[int, np.ndarray]] = []
    sprite_hole_counts = hole_counts[:]
    random.shuffle(sprite_hole_counts)
    for hole_count in sprite_hole_counts:
        patch = _make_sprite(sprite_h, sprite_w, hole_count, sprite_color_by_hole[hole_count])
        if patch is None:
            return None
        sprites.append((hole_count, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for hole_count in hole_counts:
        patch = _make_frame(frame_h, frame_w, hole_count, frame_color_by_hole[hole_count])
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
        canvas[rr : rr + frame_h, cc : cc + frame_w] = patch

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
