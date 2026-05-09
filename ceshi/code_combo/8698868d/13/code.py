# Combo variant of 8698868d (index 13)
# combo_spec: D4/4+D5/3
# Name: 全图案填充后仅输出内容

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for combo D4/4 + D5/3
=====================================

a) Concrete behavior and trace
------------------------------
- Base task:
  1. detect large square frames and smaller square sprites;
  2. count 4-connected hole components in each frame interior and each sprite;
  3. pair by equal hole-count;
  4. paste the matched sprite inside the frame;
  5. recolor every sprite hole pixel to the frame color;
  6. output the full frame grid including borders.
- D4/4 changes step 5. Instead of recoloring the holes, paste the entire sprite verbatim,
  so its background-colored hole pixels remain visible inside the frame interior.
- D5/3 changes step 6. After the interior has been filled, discard the outer frame border and
  output only the content region of every slot.
- Combined effect:
  because the frame interior size equals the sprite size (`frame_size - 2`), the final output
  for each slot is simply the matched sprite patch itself, unchanged, arranged in frame-grid order.

Small trace:
- Suppose `frame_size = 6`, hence `sprite_size = 4`.
- A frame with 2 isolated interior holes matches a 4x4 sprite with 2 connected hole components.
- Base output slot:
  a 6x6 square in frame color, with the 4x4 sprite inserted at offset `(1,1)` and all sprite
  hole pixels recolored to the frame color.
- Combo output slot:
  take that same 4x4 sprite and emit it directly as the slot result; the hole pixels stay as
  background color and the surrounding 1-pixel border is removed.
- Observable difference from either base or D5-only:
  output now contains background-colored holes, while the visible border disappears globally.

b) Complexity calibration
-------------------------
- Matching is still controlled by unique hole counts, so the pairing stays bijective and stable.
- The fusion is not over-simplified: the solver must still infer frame size, sprite size, grid
  layout, and hole-count matching. Only the final rendering rule changes.
- The visual difference is strong:
  output dimensions shrink by 2 pixels per slot in each axis, and background color now reappears
  inside the output as part of the copied sprites.
- To prevent degeneration toward the base rule, the generator always creates at least one sprite
  with one or more hole pixels, so the combo output differs from any recolor-based variant.

c) Hidden-assumption audit of the original `main()`
---------------------------------------------------
- Component detection and square-size grouping remain valid because the input geometry is unchanged.
- Frame-grid inference from the large-square positions also remains valid.
- Hole counting inside frame interiors and sprite patches remains the correct matching signal.
- The original output builder relied on three assumptions that no longer hold:
  1. each output slot has size `frame_size`;
  2. each slot is first painted with frame color;
  3. non-sprite pixels inside the pasted sprite are recolored to the frame color.
- In the combo variant all three must be removed. The output slot size is now `sprite_size`, there
  is no frame-color underlay, and sprite pixels are copied verbatim.
- New solver assumptions:
  1. every matched sprite really has size `frame_size - 2`;
  2. every frame hole count is unique among frames;
  3. every sprite hole count is unique among sprites.
  The generator enforces all three.

d) Placement legality
---------------------
- Frames are still placed as a tight, non-overlapping grid fully inside the canvas.
- Sprites are placed only in remaining background cells using `random_free_location(..., padding=1)`,
  so they do not overlap frames or each other.
- Canvas dimensions are chosen so the frame grid fits and there is still room for all sprites.
- Every constructed sprite and frame is validated by recounting its hole components before use.

e) Visual observability
-----------------------
- A human seeing a few examples can notice two simultaneous cues:
  1. the output footprint matches the combined frame interiors rather than the full frame grid;
  2. the output contains background-colored holes identical to those visible inside the scattered
     input sprites, proving the holes were preserved rather than recolored.
- The arrangement signal is still structural, not color-order based: row-major frame-grid position
  plus hole-count matching explains which sprite lands in which slot.

f) Visual cleanliness
---------------------
- The output is cleaner than the base task because the border ring is stripped away.
- Each slot contains only one square sprite patch, so there is no overdraw or ambiguous layering.
- Distinct sprite colors and preserved hole shapes keep slots easy to parse.

g) Change inferability
----------------------
- No hidden use of numeric color identity is introduced. Colors matter only as visible object fills.
- The preserved background holes are a strong structural cue for the changed transform rule.
- Because several samples will contain different hole counts and different frame colors that do
  not appear in the output interiors, a solver can infer that frames determine placement only,
  not recoloring.
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered sprites to the frame grid by connected-hole count, then output only the
    matched sprite contents without the surrounding frame borders.

    Input structure:
    - Background is azure (`8`). The input contains a tight grid of larger square frames and the
      same number of smaller square sprites scattered elsewhere on the background.
    - Every frame is a solid-colored square with a 1-pixel border and some background-colored
      holes in its interior. Every sprite is a smaller solid-colored square that may also contain
      background-colored hole regions.

    Rule:
    - Count the number of 4-connected hole components inside each frame interior and inside each
      sprite.
    - Pair each frame with the unique sprite having the same hole-count.
    - Conceptually fill the entire frame interior with the matched sprite's complete pattern,
      keeping the sprite's hole pixels unchanged.
    - Remove the frame borders from the result and output only the tiled content regions.

    Key insight:
    - The frame colors no longer recolor the sprite holes. They only define the grid slots whose
      ordering must be recovered; the final output is the matched sprite patch itself.
    """
    g = np.array(grid, dtype=int)
    bg = 8

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        bbox = comp["bbox"]
        h = int(bbox_h(bbox))
        w = int(bbox_w(bbox))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames
    assert sprites_raw
    assert sprite_size == frame_size - 2

    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)
    assert n_rows * n_cols == len(frames)

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            assert region.shape == (frame_size, frame_size)

            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            inner = region[1:-1, 1:-1]
            hole_mask = inner != frame_color
            _labels, num_holes = label_mask(hole_mask, connectivity=4)

            frame_grid.append(
                {
                    "slot_r": int(ri),
                    "slot_c": int(ci),
                    "holes": int(num_holes),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _labels, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for idx, sprite in enumerate(sprites):
            if idx not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["slot_r"]), int(frame["slot_c"]))] = sprite
                used_sprites.add(int(idx))
                break

    assert len(matches) == len(frame_grid)

    out_h = int(n_rows * sprite_size)
    out_w = int(n_cols * sprite_size)
    out = np.full((out_h, out_w), bg, dtype=int)

    for frame in frame_grid:
        rr = int(frame["slot_r"] * sprite_size)
        cc = int(frame["slot_c"] * sprite_size)
        sprite = matches[(int(frame["slot_r"]), int(frame["slot_c"]))]
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        assert sh == sprite_size and sw == sprite_size
        out[rr : rr + sh, cc : cc + sw] = np.array(sprite["content"], dtype=int)

    return out


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
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _rows * _cols <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_rows * _cols, []).append((_rows, _cols))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` cells inside a bordered square so no pair is 4-adjacent."""
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
    """Build a square sprite with exactly `hole_count` 4-connected background hole components."""
    sprite = np.full((size, size), int(sprite_color), dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
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
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for other_idx, other in enumerate(components)
                        if other_idx != comp_idx
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

    _labels, num_holes = label_mask(sprite != int(sprite_color), connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build an input frame with exactly `hole_count` isolated interior background cells."""
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
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

    _labels, num_holes = label_mask(frame != int(frame_color), connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    rows_grid: int,
    cols_grid: int,
    sprite_size: int,
) -> np.ndarray:
    """Build the combo output: tiled matched sprites with holes preserved."""
    expected = np.full((rows_grid * sprite_size, cols_grid * sprite_size), BG, dtype=int)
    sprite_by_holes = {hole_count: sprite_patch for hole_count, sprite_patch in sprites}
    for idx, (hole_count, _frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * sprite_size, ci * sprite_size
        expected[rr : rr + sprite_size, cc : cc + sprite_size] = sprite_by_holes[int(hole_count)]
    return expected


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(sprite_colors[idx]))
        if patch is None:
            return None
        sprites.append((int(hole_count), patch))

    if not any(int((patch == BG).sum()) > 0 for _hole_count, patch in sprites):
        return None

    frames: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[idx]))
        if patch is None:
            return None
        frames.append((int(hole_count), patch))

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, sprite_size)
    if int((expected == BG).sum()) == 0:
        return None

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_hole_count, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _hole_count, sprite_patch in sprites:
        try:
            rr, cc = random_free_location(
                canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, rr, cc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
