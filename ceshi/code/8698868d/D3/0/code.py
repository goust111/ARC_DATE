# Variant of 8698868d
# Dimension: D3 — 目标对象：框架网格的结构 (dst)
# Change: jigsaw_assembly — 多个正方形框架紧密排列成网格（无间隙），每个框架为 1 像素实心边框，内部散布若干单像素背景色洞（各洞 4-隔离） → 框架间有间隙分隔

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for the D3 variant
==================================

source_task:
  none

variant_spec:
  background:
    fixed: 8 (kept from the original family so the solver can still treat azure as bg)
  frames:
    count: 2-6
    size: 6-8, square
    arrangement:
      rows x cols still forms a complete rectangular grid with 1-3 rows and 1-3 cols
      adjacent frames are no longer flush
      every row gap and every col gap is 1 or 2 pixels of pure bg
      gaps may vary independently per boundary, so the solver must read real frame coordinates
    appearance:
      each frame is a solid colored square with 1-pixel border
      interior contains exactly k isolated bg holes, where k is unique per frame
  sprites:
    count: same as frames
    size: frame_size - 2
    appearance: same as original; bg-colored hole components inside a solid sprite color
    placement:
      still scattered away from the frame grid
      additionally constrained to stay outside the full bounding rectangle of the frame grid,
      so no sprite can visually occupy the newly introduced inter-frame gaps
  matching:
    unchanged: frame hole-component count == sprite hole-component count
  output:
    crop to the frame grid only, but preserve the internal bg gaps between neighboring frames
    sprite holes are recolored to the matched frame color as before
    unlike the original task, bg is now present in the output because the gaps remain visible

a) Concrete behavior trace
--------------------------
Original logic on a 1x3 grid with frame_size=6:
  frame top-left columns = [0, 6, 12]
  solver infers n_cols from (max_c - min_c) // frame_size + 1 == 3
  it then slices regions at columns 0, 6, 12 and succeeds because frames are flush.

Variant logic on a 1x3 grid with gaps [1, 2]:
  frame top-left columns = [0, 7, 15]
  the old solver would still infer 3 columns, but would slice at 0, 6, 12, which cuts through
  bg gaps and wrong parts of neighboring frames.
  The new solver instead reads the actual component bboxes, sorts their true top-left positions,
  and maps each frame to output offset (actual_position - minimum_position). This preserves the
  1-pixel and 2-pixel bg seams exactly.

b) Complexity calibration
-------------------------
Using only positive gaps of size 1 or 2 keeps the change visually obvious in 2-4 examples:
  the output is no longer a solid packed rectangle, and bg seams are easy to see.
At the same time, the grid remains regular enough that humans can still identify rows/cols:
  all frames in a row share the same top-left row, and all frames in a column share the same
  top-left column. We avoid more complex staggered layouts because that would change more than D3.

c) Hidden assumptions in the original main()
--------------------------------------------
Relevant original blocks and why they fail here:
  frame_size = largest square size:
    still valid, because frames remain the largest square monochromatic components.
  n_rows = (max_r - min_r) // frame_size + 1 and same for cols:
    no longer sufficient. Positive gaps make frame step larger than frame_size.
  inp_r = min_r + ri * frame_size / inp_c = min_c + ci * frame_size:
    invalid in the variant; these slice points miss the true frame bboxes.
  out_h = n_rows * frame_size and out_w = n_cols * frame_size:
    invalid because the output must include internal bg seams.
  border-majority color inference on arbitrary slices:
    harmless in the original because slices always aligned with real frames; dangerous now because
    a mis-sliced region contains bg gaps. The fix is to use each frame component's own bbox directly.

d) Placement legality
---------------------
Generator guarantees:
  frames never overlap and always fit inside the canvas by construction from cumulative offsets
  gaps are explicit bg strips between adjacent frames
  sprites are placed only outside the frame grid's outer bbox, using a temporary occupancy canvas
  that marks the whole grid rectangle as forbidden
  every generated sample is oracle-checked via main(input) == expected_output

e) Visual observability
-----------------------
Humans can see that:
  input frames form a separated grid rather than a packed block
  output keeps the same seams between frames
  sprites are still matched by hole-count structure, so only one major novelty is introduced
The rule does not depend on numeric color ids, hidden ordering, or invisible metadata.

f) Visual cleanliness
---------------------
The new bg seams are the only extra layer. Sprites are kept away from those seams, preventing
crowding around the target object. The output contains only frame squares and the seams between them.

g) Color-change inference
-------------------------
No new color rule is added. Sprite holes still recolor to the containing frame color exactly as in
the original task, so D3 remains the primary change.

New assumptions introduced by the updated solver
------------------------------------------------
  the set of large square components forms a complete rectangular grid:
    generator enforces rows*cols == number of frames and places one frame per slot
  all frames of a row share the same top-left row, and all frames of a column share the same
  top-left column:
    generator uses cumulative row_offsets / col_offsets
  sprites remain the second-largest square size:
    generator keeps sprite_size = frame_size - 2
"""


# ── Configuration ──────────────────────────────────────────────────────
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
GEN_RETRIES = 400
BLOCKER_COLOR = 0


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
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
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
    assert frames
    return frame_size, sprite_size, frames, sprites


def _count_holes(region: np.ndarray, obj_color: int) -> int:
    hole_mask = region != int(obj_color)
    _lab, num_holes = label_mask(hole_mask, connectivity=4)
    return int(num_holes)


def _frame_gap_metrics(grid: np.ndarray) -> tuple[list[int], list[int]]:
    frame_size, _sprite_size, frames, _sprites = _find_square_regions(np.array(grid, dtype=int))
    positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames]
    row_positions = sorted({r for r, _c in positions})
    col_positions = sorted({c for _r, c in positions})
    row_gaps = [row_positions[i + 1] - row_positions[i] - frame_size for i in range(len(row_positions) - 1)]
    col_gaps = [col_positions[i + 1] - col_positions[i] - frame_size for i in range(len(col_positions) - 1)]
    return row_gaps, col_gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

    Variant change:
    - The frame grid is no longer tightly packed.
    - Output preserves the 1-2 pixel bg seams between neighboring frames.
    """
    g = np.array(grid, dtype=int)
    frame_size, _sprite_size, frames_raw, sprites_raw = _find_square_regions(g)

    frame_positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames_raw]
    row_positions = sorted({r for r, _c in frame_positions})
    col_positions = sorted({c for _r, c in frame_positions})
    assert len(row_positions) * len(col_positions) == len(frames_raw)

    min_r = row_positions[0]
    min_c = col_positions[0]

    frame_grid: list[dict] = []
    for comp in sorted(frames_raw, key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1]))):
        b = comp["bbox"]
        frame_region = crop_bbox(g, b)
        frame_color = int(comp["color"])
        num_holes = _count_holes(frame_region[1:-1, 1:-1], frame_color)
        frame_grid.append(
            {
                "color": frame_color,
                "out_r": int(b[0] - min_r),
                "out_c": int(b[1] - min_c),
                "holes": num_holes,
            }
        )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        sprites.append(
            {
                "color": sprite_color,
                "holes": _count_holes(sprite_region, sprite_color),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    sprites_by_holes: dict[int, dict] = {}
    for sprite in sprites:
        sprites_by_holes[int(sprite["holes"])] = sprite

    out_h = max(int(frame["out_r"]) for frame in frame_grid) + frame_size
    out_w = max(int(frame["out_c"]) for frame in frame_grid) + frame_size
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = sprites_by_holes.get(int(frame["holes"]))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

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
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
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


def _sample_gaps(slot_count: int) -> list[int]:
    return [random.randint(1, 2) for _ in range(max(0, slot_count - 1))]


def _cumulative_offsets(frame_size: int, gaps: list[int]) -> list[int]:
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + frame_size + int(gap))
    return offsets


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    cols_grid: int,
    row_offsets: list[int],
    col_offsets: list[int],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    out_h = row_offsets[-1] + frame_size
    out_w = col_offsets[-1] + frame_size
    expected = np.full((out_h, out_w), BG, dtype=int)
    sprite_by_k = {k: patch for k, patch in sprites}
    for idx, (k, _frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = row_offsets[ri]
        cc = col_offsets[ci]
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_k[k].copy()
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
    for i, hole_count in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, hole_count, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((hole_count, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, hole_count, frame_colors[i])
        if patch is None:
            return None
        frames.append((hole_count, patch))

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

    expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        cols_grid,
        row_offsets,
        col_offsets,
        frame_size,
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
