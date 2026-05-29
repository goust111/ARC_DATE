# Variant of 8698868d
# Dimension: D3 — 目标对象：框架网格的结构 (dst)
# Change: jigsaw_assembly — 多个正方形框架紧密排列成网格（无间隙），每个框架为 1 像素实心边框，内部散布若干单... → 框架为非正方形矩形

from __future__ import annotations

# Design analysis
# =================
# a) Concrete behavior
# The original task uses a tight grid of equal-size square frames. Each frame contains
# k isolated bg holes; each free sprite elsewhere contains exactly k 4-connected hole
# components. Output keeps only the frame grid, inserts the matched sprite into each
# frame, and recolors the sprite holes to the frame color.
#
# This variant changes only the destination structure: every frame is now a non-square
# rectangle of one shared size (frame_h x frame_w, frame_h != frame_w). The nested
# sprites are therefore also rectangular with size (frame_h-2) x (frame_w-2). The
# matching key stays unchanged: connected-hole count. A concrete trace:
# - Original: frame size 7x7, sprite size 5x5, grid step is 7 rows and 7 cols.
# - Variant: frame size 6x8, sprite size 4x6, grid step is 6 rows and 8 cols.
# - Solver difference: instead of grouping candidate components by one scalar side
#   length and assuming square spacing, it must group by exact (h,w) dimensions and
#   reconstruct the output grid with separate row/col strides.
#
# b) Complexity calibration
# The change is visually obvious because output frames become noticeably wider or taller,
# but the rule is still learnable: only one extra degree of freedom (aspect ratio) is
# introduced, and all frames in a sample share the same rectangle size. Hole counts still
# form a small consecutive range so the bijection remains unambiguous.
#
# c) Hidden assumptions in the original solver that break here
# - `if h == w ...`: rejects all rectangular frames/sprites.
# - Grouping by one integer `size`: loses width/height information.
# - `frame_size`, `sprite_size`: assumes equal row/col stride.
# - `n_rows`, `n_cols` computed from square spacing: wrong for rectangular tiling.
# - Output shape `n_rows * frame_size` by `n_cols * frame_size`: wrong when h != w.
# The new solver replaces these with exact `(h,w)` groups and per-axis strides.
#
# d) Placement legality
# The generator keeps frames in a gapless rectangle grid and scatters sprites only in the
# remaining bg area with padding, so no overlap occurs. Every frame is visible
# (non-background color), which lets the solver infer row and column anchors directly from
# component top-left positions.
#
# e) Visual observability
# Across 2-4 examples, a human can see that the output grid is assembled from matched
# rectangular frames rather than square ones. Because the sprites inherit the same aspect
# ratio, the transformation remains coherent and the new behavior is not a hidden numeric
# tweak.
#
# f) Visual cleanliness
# The output is still a clean tight grid with one border layer plus one nested sprite
# layer. No extra marks or overlapping annotations are introduced.
#
# g) Color inference
# Color usage is unchanged: sprite holes are recolored to the surrounding frame color.
# The recoloring is still supported by strong spatial structure (inside the matched frame),
# so no new ambiguous color mapping is introduced.
#
# h) Variant metric for post-test validation
# A generated sample is considered variant-valid if the detected frame dimensions satisfy
# `frame_h != frame_w`. Distribution over `(frame_h, frame_w)` and over orientation
# (`frame_h < frame_w` vs `frame_h > frame_w`) is checked after testing.
#
# i) New solver assumptions and generator guarantees
# - Solver assumes exactly one frame-dimension group `(h,w)` whose inner size `(h-2,w-2)`
#   also appears among components. Generator constructs exactly that pair.
# - Solver assumes every frame slot is occupied and visible. Generator always uses a full
#   grid of non-background-colored frames.
# - Solver assumes unique hole counts among sprites/frames. Generator uses a consecutive
#   range with no repeats.

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
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
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
    candidates: list[tuple[int, int, int, int, int]] = []
    for frame_h, frame_w in groups:
        sprite_dims = (frame_h - 2, frame_w - 2)
        if frame_h >= 6 and frame_w >= 6 and frame_h != frame_w and sprite_dims in groups:
            candidates.append((frame_h * frame_w, len(groups[(frame_h, frame_w)]), frame_h, frame_w, len(groups[sprite_dims])))
    assert candidates
    candidates.sort(reverse=True)
    _area, _count, frame_h, frame_w, _sprite_count = candidates[0]
    return (frame_h, frame_w), (frame_h - 2, frame_w - 2)


def main(grid: np.ndarray) -> np.ndarray:
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
            _lab, num_holes = label_mask(inner != frame_color, connectivity=4)
            frame_grid.append(
                {
                    "color": frame_color,
                    "holes": int(num_holes),
                    "out_r": int(ri * frame_h),
                    "out_c": int(ci * frame_w),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        _lab, num_holes = label_mask(sprite_region != sprite_color, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
            }
        )

    sprite_by_holes: dict[int, dict] = {}
    for sprite in sprites:
        assert sprite["holes"] not in sprite_by_holes
        sprite_by_holes[sprite["holes"]] = sprite

    out = np.full((n_rows * frame_h, n_cols * frame_w), BG, dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_h, fc : fc + frame_w] = frame_color

        assert frame["holes"] in sprite_by_holes
        sprite = sprite_by_holes[frame["holes"]]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        out[fr + 1 : fr + 1 + sprite_h, fc + 1 : fc + 1 + sprite_w] = content

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

    inner_cells = (height - 2) * (width - 2)
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(height, width, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_holes)))
        for _ in range(grow):
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

    _lab, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
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

    _lab, num = label_mask(frame != frame_color, connectivity=4)
    if num != hole_count:
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
    sprite_by_k = {k: patch for k, patch in sprites}
    for idx, (hole_count, _patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * frame_h, ci * frame_w
        frame_color = frame_colors[idx]
        expected[rr : rr + frame_h, cc : cc + frame_w] = frame_color
        content = sprite_by_k[hole_count].copy()
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

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    inner_cells = sprite_h * sprite_w
    max_hole = (inner_cells + 1) // 2
    if n_frames > max_hole + 1:
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
