# Combo variant of 8698868d (index 31)
# combo_spec: D3/0+D5/2
# Name: 间隙框架+分隔线输出

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for the combo variant
=====================================

source_task:
  id: 8698868d
  core rule:
    large square frames and smaller square sprites are matched by the number of
    4-connected hole components, then each sprite is nested into its matching frame
    and the sprite holes are recolored to the frame color

combo_spec:
  D3/0 + D5/2
  name: gap-separated frames + separator-line output

combined_variant:
  input:
    frames still form a full rows x cols grid, but adjacent frames are separated by
    1-2 pixels of background instead of touching edge-to-edge
    sprites stay outside the grid's outer bounding box so the inter-frame gaps remain clean
  output:
    preserve the same logical frame order, but do not preserve raw input gap widths
    compress every inter-frame gap into a 1-pixel separator row/column of one fixed
    special color
    frame contents are still built by inserting the matched sprite and recoloring its
    holes to the surrounding frame color

a) Concrete behavior trace
--------------------------
Small example with frame_size = 6 and a 1x3 frame grid:
  original task:
    frame columns are [0, 6, 12]
    solver can slice every 6 columns and output width is 18
  D3-only:
    frame columns might be [0, 8, 15] because the gaps are [2, 1]
    solver must read true component bboxes; output would preserve those bg seams and
    have width 21
  combo here:
    input still uses [0, 8, 15] to make the gaps visible
    solver reads true bbox columns only to recover the logical order [slot 0, slot 1, slot 2]
    output columns become [0, 7, 14]: each frame keeps width 6, but every gap becomes
    exactly one separator column, so output width is 20 and the separator color is explicit
Difference:
  D3 changes how the grid is read from input.
  D5/2 changes how that recovered grid is rendered in output.
  The two changes are coupled: input spacing is the clue, output separators are the encoding.

b) Complexity calibration
-------------------------
The gap sizes are restricted to 1 or 2 pixels only:
  wide enough to be seen by eye, narrow enough that the frame grid still reads as one object.
At least one sampled gap is forced to be 2 pixels:
  this prevents the combo from collapsing into a trivial "replace bg gap with separator color"
  case on every sample.
The output separator is always 1 pixel:
  humans only need one extra layout rule, not arbitrary output gap widths.
Hole counts remain unique consecutive integers:
  matching stays bijective and visually supported by repeated examples.

c) Hidden assumptions in the original main()
--------------------------------------------
Original assumptions that break:
  n_rows/n_cols from `(max_pos - min_pos) // frame_size + 1`:
    false once positive gaps exist, because frame stride is no longer `frame_size`
  slicing at `min + k * frame_size`:
    false for the same reason; such slices cut through bg seams
  output size = `n_rows * frame_size` by `n_cols * frame_size`:
    false because the combo inserts separator rows/cols
  output offsets equal raw input offsets relative to the first frame:
    false because 2-pixel input gaps are normalized to 1-pixel separators in output
Assumptions that still hold:
  frames are the largest square monochromatic components
  sprites are the next square size and have size `frame_size - 2`
  hole count is computed from non-object-color cells inside each object

d) Placement legality
---------------------
Generator guarantees:
  frame offsets are built from cumulative `frame_size + gap` steps, so frames neither overlap
  nor leave the canvas
  the full outer bbox of the frame grid is marked as blocked before sprite placement, so
  sprites cannot fall into the new bg seams
  output layout is deterministic from `(rows, cols, frame_size)` with one separator row/col
  between neighboring slots
  every sample is oracle-checked with `main(input) == expected_output`

e) Visual observability
-----------------------
Across 2-4 examples, a human can see:
  input frames are separated by clean bg lanes
  output lanes are no longer bg and no longer variable-width; they become uniform special-color
  separators
  sprites are still paired by hole topology, because only the correct sprite fills each frame
No rule depends on numeric color ordering or hidden metadata.

f) Visual cleanliness
---------------------
The only extra visual layer is the separator color. It occupies complete rows/columns between
frames, so it does not compete with the internal sprite/frame patterns. Sprites are kept away
from the frame-grid bbox, avoiding clutter around the target structure.

g) Change inferability
----------------------
Color change remains structurally constrained:
  sprite holes are recolored exactly to the host frame color, just as in the source task
New color usage is also constrained:
  one fixed separator color is used only for inter-frame divider lines in the output and never
  for frames or sprites, so its role is easy to infer from few examples.

h) Fusion-specific anti-degeneracy check
----------------------------------------
If every sampled input gap were 1 pixel, the geometry change would be weak because the output
would only differ from a preserved-gap version by color. The generator therefore forces at least
one 2-pixel input gap somewhere in the grid, while the output always uses 1-pixel separators.
This creates measurable compression from implicit spacing to explicit divider lines.

i) New assumptions introduced by the updated solver
---------------------------------------------------
`main()` now assumes:
  the detected large square components occupy a complete rectangular grid:
    generator enforces `len(unique_rows) * len(unique_cols) == number_of_frames`
  all frames in the same logical row share the same top-left row, and likewise for columns:
    generator uses row_offsets/col_offsets derived per boundary
  separator color does not appear on frames or sprites:
    generator samples frame/sprite colors from a palette excluding bg and separator colors
  each hole count maps to exactly one sprite:
    generator keeps the original consecutive unique hole-count construction
"""


BG = 8
SEP = 5
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 400
BLOCKER_COLOR = SEP
PALETTE = [color for color in range(10) if color not in (BG, SEP)]


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _find_square_regions(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

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
    sprites = sizes.get(sprite_size, [])
    assert frames and sprites
    return frame_size, sprite_size, frames, sprites


def _count_holes(region: np.ndarray, obj_color: int) -> int:
    hole_mask = np.asarray(region, dtype=int) != int(obj_color)
    _labels, n_holes = label_mask(hole_mask, connectivity=4)
    return int(n_holes)


def _grid_axes(frames_raw: list[dict]) -> tuple[list[int], list[int]]:
    positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames_raw]
    row_positions = sorted({r for r, _c in positions})
    col_positions = sorted({c for _r, c in positions})
    assert len(row_positions) * len(col_positions) == len(frames_raw)
    return row_positions, col_positions


def _sample_gaps(slot_count: int) -> list[int]:
    return [random.randint(1, 2) for _ in range(max(0, slot_count - 1))]


def _ensure_wide_gap(row_gaps: list[int], col_gaps: list[int]) -> None:
    if 2 in row_gaps or 2 in col_gaps:
        return
    if col_gaps:
        col_gaps[random.randrange(len(col_gaps))] = 2
    else:
        row_gaps[random.randrange(len(row_gaps))] = 2


def _cumulative_offsets(frame_size: int, gaps: list[int]) -> list[int]:
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + frame_size + int(gap))
    return offsets


def _input_gap_metrics(grid: np.ndarray) -> tuple[list[int], list[int]]:
    frame_size, _sprite_size, frames, _sprites = _find_square_regions(np.asarray(grid, dtype=int))
    row_positions, col_positions = _grid_axes(frames)
    row_gaps = [row_positions[i + 1] - row_positions[i] - frame_size for i in range(len(row_positions) - 1)]
    col_gaps = [col_positions[i + 1] - col_positions[i] - frame_size for i in range(len(col_positions) - 1)]
    return row_gaps, col_gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then render the recovered frame grid with explicit separators.

    Input structure:
    - bg is fixed azure (8). Large square frames form a complete grid, but neighboring frames are separated by 1-2 pixel bg gaps.
    - Smaller square sprites of size frame_size-2 are scattered elsewhere, outside the frame-grid bounding box.
    - Each frame and sprite has its own color; sprite holes are bg-colored connected components.

    Rule:
    - Detect the true frame component bboxes and recover the logical row/column grid from their aligned top-left coordinates.
    - Count hole components in each frame interior and in each sprite, then match by equal counts.
    - Build a compact output grid in logical order: frames stay the same size, but every inter-frame gap becomes a 1-pixel separator row/column of the special color.
    - Insert the matched sprite into each frame and recolor the sprite's hole pixels to the surrounding frame color.

    Key insight:
    - The input bg gaps are only clues for recovering frame adjacency; the output does not preserve their raw widths.
      Instead, those implicit gaps are normalized into explicit separator lines while hole-count matching stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    frame_size, _sprite_size, frames_raw, sprites_raw = _find_square_regions(g)

    row_positions, col_positions = _grid_axes(frames_raw)
    row_index = {pos: idx for idx, pos in enumerate(row_positions)}
    col_index = {pos: idx for idx, pos in enumerate(col_positions)}

    frame_grid: list[dict] = []
    for comp in frames_raw:
        bbox = comp["bbox"]
        frame_region = crop_bbox(g, bbox)
        frame_grid.append(
            {
                "grid_r": int(row_index[int(bbox[0])]),
                "grid_c": int(col_index[int(bbox[1])]),
                "color": int(comp["color"]),
                "holes": int(_count_holes(frame_region[1:-1, 1:-1], int(comp["color"]))),
            }
        )
    frame_grid.sort(key=lambda item: (int(item["grid_r"]), int(item["grid_c"])))

    sprites_by_holes: dict[int, dict] = {}
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_count = _count_holes(sprite_region, sprite_color)
        sprites_by_holes[int(hole_count)] = {
            "color": sprite_color,
            "content": sprite_region.copy(),
            "h": int(sprite_region.shape[0]),
            "w": int(sprite_region.shape[1]),
        }
    assert len(sprites_by_holes) == len(sprites_raw)

    n_rows = len(row_positions)
    n_cols = len(col_positions)
    stride = frame_size + 1
    out_h = n_rows * frame_size + max(0, n_rows - 1)
    out_w = n_cols * frame_size + max(0, n_cols - 1)
    out = np.full((out_h, out_w), int(SEP), dtype=int)

    for frame in frame_grid:
        out_r = int(frame["grid_r"]) * stride
        out_c = int(frame["grid_c"]) * stride
        frame_color = int(frame["color"])
        out[out_r : out_r + frame_size, out_c : out_c + frame_size] = frame_color

        sprite = sprites_by_holes[int(frame["holes"])]
        content = np.asarray(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[out_r + 1 : out_r + 1 + sh, out_c + 1 : out_c + 1 + sw] = content

    return out


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
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
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
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

    if _count_holes(sprite, sprite_color) != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
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

    if _count_holes(frame[1:-1, 1:-1], frame_color) != hole_count:
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    stride = frame_size + 1
    out_h = rows_grid * frame_size + max(0, rows_grid - 1)
    out_w = cols_grid * frame_size + max(0, cols_grid - 1)
    expected = np.full((out_h, out_w), int(SEP), dtype=int)

    sprite_by_holes = {hole_count: patch for hole_count, patch in sprites}
    for idx, (hole_count, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = ri * stride
        cc = ci * stride
        frame_color = int(frame_patch[0, 0])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color

        content = np.asarray(sprite_by_holes[int(hole_count)], dtype=int).copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    row_gaps = _sample_gaps(rows_grid)
    col_gaps = _sample_gaps(cols_grid)
    _ensure_wide_gap(row_gaps, col_gaps)

    row_offsets = _cumulative_offsets(frame_size, row_gaps)
    col_offsets = _cumulative_offsets(frame_size, col_gaps)
    grid_pixel_h = row_offsets[-1] + frame_size
    grid_pixel_w = col_offsets[-1] + frame_size

    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    frame_colors = random.sample(PALETTE, n_frames)
    sprite_colors = random.sample(PALETTE, n_frames)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n_frames)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites = []
    for idx, hole_count in enumerate(sprite_hole_perm):
        sprite_patch = _make_sprite_interior(sprite_size, hole_count, sprite_colors[idx])
        if sprite_patch is None:
            return None
        sprites.append((hole_count, sprite_patch))

    frames: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(hole_counts):
        frame_patch = _make_frame(frame_size, hole_count, frame_colors[idx])
        if frame_patch is None:
            return None
        frames.append((hole_count, frame_patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    top = random.randint(0, canvas_h - grid_pixel_h)
    left = random.randint(0, canvas_w - grid_pixel_w)

    for idx, (_hole_count, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = top + row_offsets[ri]
        cc = left + col_offsets[ci]
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    placement_canvas = canvas.copy()
    placement_canvas[top : top + grid_pixel_h, left : left + grid_pixel_w] = BLOCKER_COLOR
    for _hole_count, sprite_patch in sprites:
        try:
            rr, cc = random_free_location(
                placement_canvas,
                sprite_patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, sprite_patch, rr, cc, transparent=BG)
        paste(placement_canvas, sprite_patch, rr, cc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    input_row_gaps, input_col_gaps = _input_gap_metrics(canvas)
    if 2 not in input_row_gaps and 2 not in input_col_gaps:
        return None
    if int((out == SEP).sum()) == 0:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
