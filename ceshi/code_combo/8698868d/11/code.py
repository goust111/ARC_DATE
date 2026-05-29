# Combo variant of 8698868d (index 11)
# combo_spec: D3/0+D14/0
# Name: 间隙框架中嵌入精灵

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste


"""
Design analysis for the combo variant
=====================================

combo_spec:
  D3/0 + D14/0

variant_summary:
  Keep the original hole-count matching rule, but move every sprite into the frame-grid's own
  internal empty structure. The grid is no longer a packed rectangle of touching frames. Instead,
  the visible frames occupy a regular rows x cols block while one or more whole slot-rows/columns
  are left empty as bg corridors. Those corridors are separated from neighboring slots by 1-2
  pixel seams and are large enough to hold the detached sprites. The output crops to the full slot
  lattice, so both the seams and the empty corridors stay visible.

a) Concrete behavior trace
--------------------------
Original 2x3 case:
  frames are flush, sprites are far outside the frame block, and the solver can classify objects
  mainly by "inside the packed grid" vs "elsewhere".

Combo 2x3 case:
  visible frames still form a 2x3 rectangle, but the full lattice is 3x4 because one slot-row and
  one slot-column are intentionally blank. Sprites sit centered in those blank slots. A sprite is
  now spatially near several frames, so the solver must detect true square components and recover
  slot coordinates instead of relying on coarse outside-vs-inside separation. After matching by
  connected-hole count, each sprite is inserted into its frame and the blank reservoir slots remain
  empty in the output.

b) Complexity calibration
-------------------------
The extra empty corridors are the minimum structure that makes D14 visible without adding too many
degrees of freedom. There are only two new geometric ideas:
  one or more reserved slot-rows
  one or more reserved slot-columns
This yields extra_rows*cols + extra_cols*rows candidate sprite slots, enough for the original
two-dimensional frame counts N in {4, 6}
while keeping the layout easy to read in 2-4 examples.

c) Hidden assumptions in the original main()
--------------------------------------------
Original assumptions that no longer hold:
  sprites live outside the frame-grid bbox:
    false here; sprites are deliberately inside the lattice bbox.
  the frame block alone defines the output crop:
    false here; the preserved empty slot-row/column must stay in the output.
  stepping by frame_size reconstructs all relevant positions:
    false here; the solver must read actual top-left coordinates because gaps remain visible.
  only the largest squares matter for geometry:
    false here; sprite top-left minus one pixel is needed to recover the empty slot coordinates.

d) Placement legality
---------------------
Generator guarantees:
  visible frames occupy all non-corridor rows x cols slot intersections, so they never overlap
  reserved slot-rows and slot-columns are always internal, never the outermost lattice boundary
  sprites occupy distinct corridor slots and are centered with a 1-pixel bg moat inside each slot
  sampled gap widths and frame size always fit the canvas budget
  the lattice is embedded with only a small outer margin, so the internal corridors stay salient
  every sample is oracle-checked with main(input) == expected_output

e) Visual observability
-----------------------
Humans can see one rectangular frame arrangement plus one or more bg corridors threading through
it. The detached sprites visibly live in those internal gaps, not elsewhere on the canvas. The
output keeps the same lattice shape, but the small squares disappear from the corridors and
reappear nested into the large frames.

f) Visual cleanliness
---------------------
There is no overlap between object types:
  frames fill whole slots
  sprites sit inside empty slots with a bg moat
  seams between slots remain pure bg
This keeps the scene legible even when sprites are very close to frames.

g) Change inferability
----------------------
Color behavior stays simple: a sprite keeps its own color when nested, and only its hole pixels are
recolored to the destination frame color. The new inference burden is geometric, not color-based.

New assumptions introduced by the updated solver
------------------------------------------------
  sprites are centered in frame-sized corridor slots with a 1-pixel inset:
    generator pastes every sprite at slot_origin + (1, 1)
  frames are the largest square components and sprites are the second-largest:
    generator keeps frame_size in {6, 7, 8} and sprite_size = frame_size - 2
  the union of frame slots and occupied sprite slots exposes every slot-row and slot-column used by
  the output crop:
    generator always selects at least one sprite in every reserved row and every reserved column
"""


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 18
CANVAS_MAX = 32
HOLE_START_MAX = 1
SEED_RETRIES = 60
GEN_RETRIES = 500
MIN_FILL_RATIO = 0.5


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if _r < 2 or _c < 2:
            continue
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _max_isolated_holes(square_size: int) -> int:
    inner = max(0, int(square_size) - 2)
    return int((inner * inner + 1) // 2)


def _count_holes(region: np.ndarray, obj_color: int) -> int:
    hole_mask = region != int(obj_color)
    _labels, count = label_mask(hole_mask, connectivity=4)
    return int(count)


def _find_square_regions(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
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
    assert sprites
    return frame_size, sprite_size, frames, sprites


def _sprite_slot_origin(comp: dict) -> tuple[int, int]:
    b = comp["bbox"]
    return int(b[0] - 1), int(b[1] - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each detached sprite to the frame with the same number of hole components, then nest it.

    Input structure:
    - Background is fixed to azure (8). Large square frames occupy a regular rows x cols block, but
      one or more whole slot-rows/columns of the surrounding lattice are left empty.
    - Smaller square sprites are centered inside those empty corridor slots, so they lie inside the
      frame lattice rather than outside it.

    Rule:
    - Recover the lattice using both object types: frame top-lefts give occupied slots, and each
      sprite's slot origin is one pixel above/left of its own bbox because sprites are inset by 1.
    - Count 4-connected hole components inside each frame interior and inside each sprite.
    - Pair equal counts, place the matched sprite inside the corresponding frame border, and recolor
      the sprite's hole pixels to the frame color.

    Key insight:
    - The new difficulty is geometric: sprites are no longer separated from the target structure,
      so the solver must reconstruct the internal slot lattice before applying the original
      hole-count matching rule.
    """
    g = np.array(grid, dtype=int)
    frame_size, _sprite_size, frames_raw, sprites_raw = _find_square_regions(g)

    slot_origins = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames_raw]
    slot_origins.extend(_sprite_slot_origin(comp) for comp in sprites_raw)
    row_positions = sorted({int(r) for r, _c in slot_origins})
    col_positions = sorted({int(c) for _r, c in slot_origins})
    min_r = int(row_positions[0])
    min_c = int(col_positions[0])

    frame_grid: list[dict] = []
    for comp in sorted(frames_raw, key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1]))):
        b = comp["bbox"]
        region = crop_bbox(g, b)
        frame_color = int(comp["color"])
        holes = _count_holes(region[1:-1, 1:-1], frame_color)
        frame_grid.append(
            {
                "color": frame_color,
                "out_r": int(b[0] - min_r),
                "out_c": int(b[1] - min_c),
                "holes": holes,
            }
        )

    sprites_by_holes: dict[int, dict] = {}
    for comp in sprites_raw:
        region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        sprites_by_holes[_count_holes(region, sprite_color)] = {
            "color": sprite_color,
            "content": region.copy(),
            "h": int(region.shape[0]),
            "w": int(region.shape[1]),
        }

    out_h = int(max(row_positions) - min_r + frame_size)
    out_w = int(max(col_positions) - min_c + frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = sprites_by_holes.get(int(frame["holes"]))
        assert sprite is not None
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

    max_grow = max(1, inner_cells // max(1, hole_count) - 1)
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

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    if _count_holes(frame[1:-1, 1:-1], frame_color) != hole_count:
        return None
    return frame


def _sample_gap_list(slot_count: int, frame_size: int, max_dim: int) -> list[int] | None:
    if slot_count <= 1:
        return []
    gaps = [1] * (slot_count - 1)
    slack = int(max_dim - (slot_count * frame_size + sum(gaps)))
    if slack < 0:
        return None
    order = list(range(len(gaps)))
    random.shuffle(order)
    for idx in order:
        if slack <= 0:
            break
        if random.random() < 0.6:
            gaps[idx] += 1
            slack -= 1
    return gaps


def _offsets(frame_size: int, gaps: list[int]) -> list[int]:
    out = [0]
    for gap in gaps:
        out.append(out[-1] + frame_size + int(gap))
    return out


def _choose_extra_axes(rows_frame: int, cols_frame: int, n_frames: int) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for extra_rows, extra_cols in [(1, 1), (2, 1), (1, 2), (2, 2)]:
        capacity = extra_rows * cols_frame + extra_cols * rows_frame
        if capacity >= n_frames:
            candidates.append((extra_rows, extra_cols))
    if not candidates:
        return None
    min_total = min(extra_rows + extra_cols for extra_rows, extra_cols in candidates)
    best = [pair for pair in candidates if pair[0] + pair[1] == min_total]
    return random.choice(best)


def _choose_reserved_indices(total_slots: int, extra_count: int) -> list[int]:
    internal = [idx for idx in range(1, total_slots - 1)]
    assert len(internal) >= extra_count
    return sorted(random.sample(internal, extra_count))


def _slot_indices(
    rows_frame: int,
    cols_frame: int,
    extra_rows: int,
    extra_cols: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    slot_rows = rows_frame + extra_rows
    slot_cols = cols_frame + extra_cols
    reserved_rows = _choose_reserved_indices(slot_rows, extra_rows)
    reserved_cols = _choose_reserved_indices(slot_cols, extra_cols)
    frame_rows = [idx for idx in range(slot_rows) if idx not in reserved_rows]
    frame_cols = [idx for idx in range(slot_cols) if idx not in reserved_cols]
    return frame_rows, frame_cols, reserved_rows, reserved_cols


def _reservoir_slots(
    frame_rows: list[int],
    frame_cols: list[int],
    reserved_rows: list[int],
    reserved_cols: list[int],
) -> list[tuple[int, int]]:
    slots = [(r, c) for r in reserved_rows for c in frame_cols]
    slots.extend((r, c) for r in frame_rows for c in reserved_cols)
    return slots


def _adjacent_to_frame(slot: tuple[int, int], frame_slots: set[tuple[int, int]]) -> bool:
    r, c = slot
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        if (r + dr, c + dc) in frame_slots:
            return True
    return False


def _select_sprite_slots(
    candidates: list[tuple[int, int]],
    count: int,
    reserved_rows: list[int],
    reserved_cols: list[int],
    frame_slots: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    valid = [slot for slot in candidates if _adjacent_to_frame(slot, frame_slots)]
    if len(valid) < count:
        return None
    for _ in range(200):
        chosen = random.sample(valid, count)
        if any(not any(r == row_idx for r, _c in chosen) for row_idx in reserved_rows):
            continue
        if any(not any(c == col_idx for _r, c in chosen) for col_idx in reserved_cols):
            continue
        return chosen
    return None


def _choose_frame_size(n_frames: int, slot_rows: int, slot_cols: int) -> int | None:
    candidates = []
    for frame_size in FRAME_SIZE_RANGE:
        if _max_isolated_holes(frame_size - 2) + 1 < n_frames:
            continue
        min_h = slot_rows * frame_size + (slot_rows - 1)
        min_w = slot_cols * frame_size + (slot_cols - 1)
        if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
            candidates.append(frame_size)
    return random.choice(candidates) if candidates else None


def _build_expected_output(
    frame_specs: list[tuple[int, np.ndarray, tuple[int, int]]],
    sprite_specs: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    row_offsets: list[int],
    col_offsets: list[int],
    frame_size: int,
) -> np.ndarray:
    out_h = row_offsets[-1] + frame_size
    out_w = col_offsets[-1] + frame_size
    expected = np.full((out_h, out_w), BG, dtype=int)
    sprite_by_holes = {holes: patch for holes, patch in sprite_specs}
    for idx, (holes, _frame_patch, (slot_r, slot_c)) in enumerate(frame_specs):
        rr = row_offsets[slot_r]
        cc = col_offsets[slot_c]
        frame_color = int(frame_colors[idx])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_holes[holes].copy()
        sprite_color = int(content[0, 0])
        content[content != sprite_color] = frame_color
        expected[rr + 1 : rr + frame_size - 1, cc + 1 : cc + frame_size - 1] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice(_VALID_NS)
    rows_frame, cols_frame = random.choice(_LAYOUTS_BY_N[n_frames])

    extras = _choose_extra_axes(rows_frame, cols_frame, n_frames)
    if extras is None:
        return None
    extra_rows, extra_cols = extras
    slot_rows = rows_frame + extra_rows
    slot_cols = cols_frame + extra_cols
    frame_size = _choose_frame_size(n_frames, slot_rows, slot_cols)
    if frame_size is None:
        return None
    sprite_size = frame_size - 2

    row_gaps = _sample_gap_list(slot_rows, frame_size, CANVAS_MAX)
    col_gaps = _sample_gap_list(slot_cols, frame_size, CANVAS_MAX)
    if row_gaps is None or col_gaps is None:
        return None
    row_offsets = _offsets(frame_size, row_gaps)
    col_offsets = _offsets(frame_size, col_gaps)
    grid_h = row_offsets[-1] + frame_size
    grid_w = col_offsets[-1] + frame_size

    min_h = max(CANVAS_MIN, grid_h)
    min_w = max(CANVAS_MIN, grid_w)
    max_h = min(CANVAS_MAX, grid_h + 4)
    max_w = min(CANVAS_MAX, grid_w + 4)
    if min_h > max_h or min_w > max_w:
        return None
    canvas_h = random.randint(min_h, max_h)
    canvas_w = random.randint(min_w, max_w)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, n_frames)

    max_hole = _max_isolated_holes(sprite_size)
    start = random.randint(0, min(max_hole - n_frames + 1, HOLE_START_MAX))
    hole_counts = list(range(start, start + n_frames))
    random.shuffle(hole_counts)
    sprite_holes = hole_counts[:]
    random.shuffle(sprite_holes)

    sprites: list[tuple[int, np.ndarray]] = []
    for idx, holes in enumerate(sprite_holes):
        patch = _make_sprite_interior(sprite_size, holes, sprite_colors[idx])
        if patch is None:
            return None
        sprites.append((holes, patch))

    frame_rows, frame_cols, reserved_rows, reserved_cols = _slot_indices(
        rows_frame, cols_frame, extra_rows, extra_cols
    )
    frame_slots = [(sr, sc) for sr in frame_rows for sc in frame_cols]
    frame_slot_set = set(frame_slots)
    reservoir = _reservoir_slots(frame_rows, frame_cols, reserved_rows, reserved_cols)
    sprite_slots = _select_sprite_slots(
        reservoir, n_frames, reserved_rows, reserved_cols, frame_slot_set
    )
    if sprite_slots is None:
        return None

    frames: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    for idx, holes in enumerate(hole_counts):
        patch = _make_frame(frame_size, holes, frame_colors[idx])
        if patch is None:
            return None
        frames.append((holes, patch, frame_slots[idx]))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    top = random.randint(0, canvas_h - grid_h)
    left = random.randint(0, canvas_w - grid_w)

    for _holes, patch, (slot_r, slot_c) in frames:
        rr = top + row_offsets[slot_r]
        cc = left + col_offsets[slot_c]
        canvas[rr : rr + frame_size, cc : cc + frame_size] = patch

    for (holes, patch), (slot_r, slot_c) in zip(sprites, sprite_slots):
        rr = top + row_offsets[slot_r] + 1
        cc = left + col_offsets[slot_c] + 1
        paste(canvas, patch, rr, cc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, frame_colors, row_offsets, col_offsets, frame_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
