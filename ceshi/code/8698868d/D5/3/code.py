# Variant of 8698868d
# Dimension: D5 — 输出网格的构成 (grid_shape)
# Change: 输出为紧密的框架网格（rows_grid * frame_size x cols_grid * frame_size），... → 仅输出精灵内容（去除框架边框）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for the D5 crop variant
=======================================

a) Concrete behavior and trace
------------------------------
- Input stays the same as the base task: a tight grid of large square frames plus the same
  number of scattered smaller square sprites. Matching is still by the number of 4-connected
  hole components.
- Original output: each grid slot is a full `frame_size x frame_size` square. The frame border
  remains visible, and the matched sprite is pasted into the inner `(frame_size-2) x (frame_size-2)`
  region. Any sprite hole pixel is recolored to the frame color.
- Variant output: remove the frame border entirely. Each grid slot now contributes only its
  interior sprite patch after recoloring. Therefore the output size changes from
  `(rows_grid * frame_size, cols_grid * frame_size)` to
  `(rows_grid * sprite_size, cols_grid * sprite_size)`, where `sprite_size = frame_size - 2`.

Small trace:
- Suppose `frame_size = 6`, so `sprite_size = 4`.
- Original slot output:
  1. paint a 6x6 block with frame color;
  2. paste the 4x4 matched sprite at offset `(1, 1)`;
  3. result keeps the 1-pixel border.
- Variant slot output:
  1. take the same matched 4x4 sprite;
  2. recolor its hole pixels to the frame color;
  3. place that 4x4 content directly into the output grid with no surrounding border.
- Difference: matching evidence and recoloring stay identical, but the final grid coordinate
  system is compressed by stripping each frame's outer ring.

b) Complexity calibration
-------------------------
- The variant changes only output composition. Matching remains constrained by unique hole counts,
  so there is still a single consistent pairing.
- The cropped output is visually distinct because every slot shrinks by 2 pixels in each dimension,
  and border-only information disappears globally.
- No extra transformation freedom is introduced: there is still one sprite per frame, one key
  feature (hole-count), and one recoloring rule.

c) Hidden-assumption audit of the original `main()`
---------------------------------------------------
- Component detection and square-size grouping rely on input geometry, not on output geometry.
  These assumptions still hold.
- Frame-grid inference from frame top-left positions still holds because input frames are unchanged.
- Hole counting inside frame interiors and inside sprites still holds unchanged.
- The original output builder assumes:
  1. output slot size is `frame_size`;
  2. every slot is first filled with frame color;
  3. sprite content is offset by +1,+1 inside the frame.
  All three are tied to the old D5 behavior and must be replaced.
- In the variant, using the old unconditional frame-color fill would be wrong because it would
  reintroduce the removed border as solid bands. The correct behavior is to emit only the recolored
  sprite content at the compressed slot coordinates.

d) Placement legality
---------------------
- Generator keeps the original legality constraints: frames occupy a tight, non-overlapping grid;
  sprites are placed in remaining background with padding; canvas bounds stay within 15..30.
- Because input geometry is unchanged, the solver can still recover frame size, sprite size,
  grid dimensions, and hole counts exactly as before.

e) Visual observability
-----------------------
- From 2-4 examples, a human can notice that output dimensions are smaller than the frame grid's
  visible footprint in the input.
- The content still aligns by frame-grid order and recoloring, so the new rule is inferable as
  "take only what would have been inside each frame".
- This is analogous in spirit to source task `a251c730`, where the output preserves only a target
  region rather than the full source arrangement.

f) Visual cleanliness
---------------------
- Removing borders reduces clutter rather than increasing it. Each output cell block is a single
  recolored sprite patch with no extra overdraw.
- There is no overlap in the output because each slot receives exactly one `sprite_size x sprite_size`
  patch in a regular tiling.

g) Color-change inferability
----------------------------
- The only color change remains the original one: sprite hole pixels become the matched frame color.
- The number of participating colors is unchanged from the base task, and the structural cue is
  explicit because the frame carrying that color is still visible in the input.

New assumptions introduced by the variant solver
------------------------------------------------
- Every frame must still find exactly one sprite with the same hole count.
- Every matched sprite must have the expected square size `sprite_size = frame_size - 2`.
- The generator guarantees both by constructing unique consecutive hole counts and generating all
  sprites at exactly `frame_size - 2`.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then output only the recolored
    frame interiors (the frame borders are stripped from the final grid).
    """
    g = np.array(grid, dtype=int)
    BG = 8  # Azure background for this puzzle family

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Find square regions (lower fill threshold to catch sprites with holes).
    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((int(h), comp))

    assert square_regions

    # Group by size: larger squares are frames, smaller squares are sprites.
    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(int(frame_size), [])
    sprites_raw = sizes.get(int(sprite_size), [])
    assert frames

    # Determine frame grid structure from frame positions.
    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    # Build frame grid in row-major slot order.
    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != int(frame_color)
            _lab, num_holes = label_mask(hole_mask, connectivity=4)

            frame_grid.append(
                {
                    "color": int(frame_color),
                    "slot_r": int(ri),
                    "slot_c": int(ci),
                    "holes": int(num_holes),
                }
            )

    # Process sprites and count their holes.
    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": int(sprite_color),
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    # Match frames to sprites by hole count.
    matches: dict[tuple[int, int], tuple[dict, dict]] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for idx, sprite in enumerate(sprites):
            if idx not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["slot_r"]), int(frame["slot_c"]))] = (frame, sprite)
                used_sprites.add(int(idx))
                break

    assert len(matches) == len(frame_grid)

    # Build cropped output grid: only the frame interiors remain.
    out_h = int(n_rows * sprite_size)
    out_w = int(n_cols * sprite_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        key = (int(frame["slot_r"]), int(frame["slot_c"]))
        _frame, sprite = matches[key]
        frame_color = int(frame["color"])

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        assert sh == sprite_size and sw == sprite_size

        rr = int(frame["slot_r"] * sprite_size)
        cc = int(frame["slot_c"] * sprite_size)
        out[rr : rr + sh, cc : cc + sw] = content

    return out


# --------------- generator (D5 crop variant) ---------------

BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` cells inside a square, mutually 4-isolated."""
    inner_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    """Build a sprite with exactly `hole_count` 4-connected hole components."""
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite
    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_hole:
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
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    _lab, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build an input frame with exactly `hole_count` isolated interior bg cells."""
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame
    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
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
    sprite_size: int,
) -> np.ndarray:
    """Build the cropped variant output: only recolored sprite interiors."""
    expected = np.full((rows_grid * sprite_size, cols_grid * sprite_size), BG, dtype=int)
    sprite_by_holes = {k: spatch for k, spatch in sprites}
    for idx, (hole_count, _frame_patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * sprite_size, ci * sprite_size
        content = sprite_by_holes[hole_count].copy()
        content[content == BG] = int(frame_colors[idx])
        expected[rr : rr + sprite_size, cc : cc + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    # 1. Pick N and grid layout.
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    # 2. Canvas size (must fit frame grid + room for sprites).
    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    # 3. Colors: distinct within each group, groups may overlap.
    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    # 4. Hole counts: consecutive range with unique values.
    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    # 5. Build sprites and frames.
    sprites: list[tuple[int, np.ndarray]] = []
    for i, holes in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, holes, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((holes, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, holes in enumerate(hole_counts):
        patch = _make_frame(frame_size, holes, frame_colors[i])
        if patch is None:
            return None
        frames.append((holes, patch))

    # 6. Place the frame grid on the canvas.
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_holes, frame_patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    # 7. Place sprites in the remaining background area.
    for _holes, sprite_patch in sprites:
        try:
            r, c = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, r, c, transparent=BG)

    # 8. Oracle check against the cropped-output variant.
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
        sprite_size,
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
