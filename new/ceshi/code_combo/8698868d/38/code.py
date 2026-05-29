# Combo variant of 8698868d (index 38)
# combo_spec: D3/0+D18/2
# Name: 间隙框架+缺口边框

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


"""
Design analysis for combo D3/0 + D18/2
======================================

source_task:
  original ARC-AGI-2 task 8698868d

combo_summary:
  keep the original matching key (sprite hole-component count == frame enclosed-hole count),
  but change two structural assumptions at once:
    1. frame squares form a complete rectangular grid with 1-2 pixel bg seams between adjacent slots;
    2. every frame border has a short 1-2 pixel opening, plus a thin inward bg corridor so the
       opening is visibly connected to the interior and can be mistaken for a hole by a naive solver.

a) Concrete behavior trace
--------------------------
Small example: frame_size = 7, one row of three frames, horizontal gaps [1, 2].
  Original task:
    frame top-left columns would be [0, 7, 14].
    Original solver assumes each frame starts every `frame_size` cells and counts holes only inside
    `region[1:-1, 1:-1]`, because borders are sealed.
  Combo variant:
    frame top-left columns become [0, 8, 17].
    Suppose the middle frame has a right-edge gap at border cells (3,6),(4,6) and an inward corridor
    at (4,5). That corridor touches the outside through the edge opening.
    If the old solver sliced at columns [0, 7, 14], it would cut through seams.
    Even if it somehow got the correct crop, counting holes only in the interior would wrongly count
    the corridor as one extra hole.
  Correct combo logic:
    read each frame from its actual monochrome component bbox;
    preserve output offsets using real bbox positions, so the 1-2 pixel seams stay visible;
    count bg components on the full frame patch and ignore any component touching the patch boundary,
    so the open corridor does not count as a hole.

b) Complexity calibration
-------------------------
The two changes are both visually explicit but kept low-entropy:
  seams are only 1 or 2 pixels wide, enough to break flush packing without making the layout hard to
  parse;
  every frame has exactly one notch on one side, with gap length 1-2 and corridor depth 1-2;
  hole counts remain a unique consecutive set, so pairing stays bijective and does not introduce
  extra ambiguous attributes.
This avoids a degenerate puzzle where gaps are too subtle or notches too chaotic.

c) Hidden assumptions in the original main()
--------------------------------------------
Original assumptions that break here:
  `n_rows/n_cols` from `(max_pos - min_pos) // frame_size + 1`:
    invalid because positive seams increase slot spacing.
  slicing frames at `min + slot_index * frame_size`:
    invalid for the same reason; slices would drift into seams or neighboring frames.
  counting holes via `inner = region[1:-1, 1:-1]`:
    invalid because an open notch may send a bg corridor into the interior.
  filling output frames as solid colors:
    invalid if we want visible border gaps to survive in the output.
Combined effect:
  the solver must both recover physical frame placement from component bboxes and distinguish
  boundary-connected bg from truly enclosed bg.

d) Placement legality
---------------------
Generator guarantees:
  row/col offsets are cumulative, so frames never overlap and seams remain pure bg;
  sprites are placed outside the outer bbox of the whole frame grid, so they never occupy seams;
  hole seeds inside a frame avoid the notch cells and their 4-neighbors, preventing accidental
  merging with the open corridor;
  every sample is oracle-checked against `main()`.

e) Visual observability
-----------------------
Across a few examples a human can directly see:
  the output is not a packed rectangle anymore because seams remain;
  frames have visible side openings, but those openings do not change sprite matching;
  only fully enclosed interior holes matter.
No rule depends on numeric color ids or hidden ordering.

f) Visual cleanliness
---------------------
The frame grid is the only dense structure. Sprites stay outside its bounding box, keeping the seam
pattern readable. Each frame has only one short notch, so there is no excessive border noise.

g) Change inferability
----------------------
Color behavior stays simple: sprite hole pixels become the matched frame color after insertion.
The new inferential burden is purely structural:
  the solver must notice that open border-connected bg and enclosed bg play different roles.
That distinction is visible from geometry alone, especially because border gaps persist in output.

New assumptions introduced by the updated solver
------------------------------------------------
  the large square components form a complete rectangular grid:
    generator constructs exactly one frame per row/col slot.
  row positions and column positions are shared within rows/cols:
    generator uses cumulative row_offsets / col_offsets.
  every frame contains exactly one open notch whose bg component touches the frame boundary:
    generator always adds a 1-2 cell border gap plus inward corridor.
  sprite size remains `frame_size - 2`:
    generator preserves the original family relation.
"""


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 80
GEN_RETRIES = 500
BLOCKER_COLOR = 0


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _find_square_regions(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    square_regions: list[tuple[int, dict]] = []
    min_fill_ratio = 0.5
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


def _count_sprite_holes(region: np.ndarray, sprite_color: int) -> int:
    _lab, num = label_mask(region != int(sprite_color), connectivity=4)
    return int(num)


def _count_legacy_inner_components(region: np.ndarray, frame_color: int) -> int:
    inner = region[1:-1, 1:-1]
    _lab, num = label_mask(inner != int(frame_color), connectivity=4)
    return int(num)


def _count_enclosed_frame_holes(region: np.ndarray, frame_color: int) -> int:
    mask = region != int(frame_color)
    lab, num = label_mask(mask, connectivity=4)
    holes = 0
    h, w = region.shape
    for k in range(1, int(num) + 1):
        coords = np.argwhere(lab == k)
        if coords.size == 0:
            continue
        touches_boundary = any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords)
        if not touches_boundary:
            holes += 1
    return int(holes)


def _candidate_positions(size: int, forbidden: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    forbidden = forbidden or set()
    blocked = set(forbidden)
    for r, c in forbidden:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr <= size - 2 and 1 <= nc <= size - 2:
                blocked.add((nr, nc))
    return [(r, c) for r in range(1, size - 1) for c in range(1, size - 1) if (r, c) not in blocked]


def _place_isolated_seeds(
    size: int,
    count: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    positions = _candidate_positions(size, forbidden=forbidden)
    if count == 0:
        return []
    for _ in range(SEED_RETRIES):
        random.shuffle(positions)
        seeds: list[tuple[int, int]] = []
        for r, c in positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), int(sprite_color), dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if hole_count > inner_cells:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    components: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = int(BG)

    max_grow = max(1, inner_cells // hole_count - 1)
    for ci, comp in enumerate(components):
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
                        for j, other in enumerate(components)
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
            sprite[cell[0], cell[1]] = int(BG)

    if _count_sprite_holes(sprite, sprite_color) != int(hole_count):
        return None
    return sprite


def _choose_notch_side(rows_grid: int, cols_grid: int, ri: int, ci: int) -> tuple[str, str]:
    internal_sides: list[str] = []
    outer_sides: list[str] = []

    if ri > 0:
        internal_sides.append("top")
    else:
        outer_sides.append("top")
    if ri < rows_grid - 1:
        internal_sides.append("bottom")
    else:
        outer_sides.append("bottom")
    if ci > 0:
        internal_sides.append("left")
    else:
        outer_sides.append("left")
    if ci < cols_grid - 1:
        internal_sides.append("right")
    else:
        outer_sides.append("right")

    if internal_sides and outer_sides:
        if random.random() < 0.7:
            return random.choice(internal_sides), "internal"
        return random.choice(outer_sides), "outer"
    if internal_sides:
        return random.choice(internal_sides), "internal"
    return random.choice(outer_sides), "outer"


def _make_open_notch(frame_size: int, side: str) -> set[tuple[int, int]]:
    gap_len = random.randint(1, 2)
    depth = random.randint(1, 2)
    notch: set[tuple[int, int]] = set()

    if side in {"top", "bottom"}:
        start = random.randint(1, frame_size - 1 - gap_len)
        mouth = start + random.randint(0, gap_len - 1)
        border_r = 0 if side == "top" else frame_size - 1
        step = 1 if side == "top" else -1
        for j in range(gap_len):
            notch.add((border_r, start + j))
        for d in range(1, depth + 1):
            notch.add((border_r + step * d, mouth))
    else:
        start = random.randint(1, frame_size - 1 - gap_len)
        mouth = start + random.randint(0, gap_len - 1)
        border_c = 0 if side == "left" else frame_size - 1
        step = 1 if side == "left" else -1
        for j in range(gap_len):
            notch.add((start + j, border_c))
        for d in range(1, depth + 1):
            notch.add((mouth, border_c + step * d))

    return notch


def _make_frame(frame_size: int, hole_count: int, frame_color: int, side: str) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
    notch = _make_open_notch(frame_size, side)
    for r, c in notch:
        frame[r, c] = int(BG)

    seeds = _place_isolated_seeds(frame_size, hole_count, forbidden=notch)
    if seeds is None:
        return None
    for r, c in seeds:
        frame[r, c] = int(BG)

    enclosed = _count_enclosed_frame_holes(frame, frame_color)
    legacy = _count_legacy_inner_components(frame, frame_color)
    if int(enclosed) != int(hole_count):
        return None
    if int(legacy) <= int(enclosed):
        return None
    return frame


def _sample_gaps(slot_count: int) -> list[int]:
    return [random.randint(1, 2) for _ in range(max(0, slot_count - 1))]


def _cumulative_offsets(frame_size: int, gaps: list[int]) -> list[int]:
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + frame_size + int(gap))
    return offsets


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    frame_size, _sprite_size, frames_raw, _sprites_raw = _find_square_regions(np.array(grid, dtype=int))
    positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames_raw]
    row_positions = sorted({r for r, _c in positions})
    col_positions = sorted({c for _r, c in positions})

    row_gaps = [row_positions[i + 1] - row_positions[i] - frame_size for i in range(len(row_positions) - 1)]
    col_gaps = [col_positions[i + 1] - col_positions[i] - frame_size for i in range(len(col_positions) - 1)]

    legacy_delta_frames = 0
    total_border_gap_pixels = 0
    for comp in frames_raw:
        region = crop_bbox(grid, comp["bbox"])
        frame_color = int(comp["color"])
        legacy = _count_legacy_inner_components(region, frame_color)
        enclosed = _count_enclosed_frame_holes(region, frame_color)
        if legacy > enclosed:
            legacy_delta_frames += 1

        border_mask = np.zeros(region.shape, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        total_border_gap_pixels += int(np.sum((region == int(BG)) & border_mask))

    return {
        "rows": len(row_positions),
        "cols": len(col_positions),
        "row_gap_count": len(row_gaps),
        "col_gap_count": len(col_gaps),
        "min_gap": min(row_gaps + col_gaps) if (row_gaps or col_gaps) else 0,
        "max_gap": max(row_gaps + col_gaps) if (row_gaps or col_gaps) else 0,
        "legacy_delta_frames": int(legacy_delta_frames),
        "total_border_gap_pixels": int(total_border_gap_pixels),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match scattered sprites to separated notched frames by enclosed-hole count, then insert them.

    Input structure:
    - Background is azure (8). Large same-sized square frame components form a complete rectangular grid, but adjacent rows and columns are separated by 1-2 pixel bg seams.
    - Each frame has its own color, a 1-2 pixel border opening on one side, and possibly a thin bg corridor entering from that opening. Smaller square sprites of size `frame_size - 2` sit elsewhere and may contain bg hole components.

    Rule:
    - Recover frame positions from the actual square component bboxes rather than assuming flush packing.
    - For each frame, count only non-frame-color components that are fully enclosed inside the full square patch; the border opening and any boundary-connected corridor do not count as holes.
    - Match each frame with the unique sprite that has the same number of connected hole components, place the sprite into the frame interior, recolor the sprite's bg holes to the frame color, and keep the inter-frame seams plus visible border gaps in the output crop.

    Key insight:
    - The crucial distinction is between enclosed holes and open bg connected to the outside through a border gap. Using full-patch boundary connectivity for hole counting and real bbox offsets for placement is what lets the solver handle both the seams and the notches together.
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
        patch = crop_bbox(g, b)
        frame_color = int(comp["color"])
        holes = _count_enclosed_frame_holes(patch, frame_color)
        frame_grid.append(
            {
                "color": frame_color,
                "patch": patch.copy(),
                "holes": int(holes),
                "out_r": int(b[0] - min_r),
                "out_c": int(b[1] - min_c),
            }
        )

    sprites_by_holes: dict[int, dict] = {}
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_count = _count_sprite_holes(sprite_region, sprite_color)
        sprites_by_holes[int(hole_count)] = {
            "color": sprite_color,
            "content": sprite_region.copy(),
            "h": int(sprite_region.shape[0]),
            "w": int(sprite_region.shape[1]),
        }

    out_h = max(int(frame["out_r"]) for frame in frame_grid) + frame_size
    out_w = max(int(frame["out_c"]) for frame in frame_grid) + frame_size
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = np.array(frame["patch"], dtype=int)

        sprite = sprites_by_holes.get(int(frame["holes"]))
        assert sprite is not None

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _build_expected_output(
    frames: list[dict],
    sprites: list[tuple[int, np.ndarray]],
    row_offsets: list[int],
    col_offsets: list[int],
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    out_h = row_offsets[-1] + frame_size
    out_w = col_offsets[-1] + frame_size
    expected = np.full((out_h, out_w), int(BG), dtype=int)
    sprite_by_k = {hole_count: patch for hole_count, patch in sprites}

    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = row_offsets[ri]
        cc = col_offsets[ci]
        expected[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

        content = sprite_by_k[int(frame["holes"])].copy()
        content[content == int(BG)] = int(frame["color"])
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

    frames: list[dict] = []
    notch_kinds: list[str] = []
    for idx, hole_count in enumerate(hole_counts):
        ri, ci = divmod(idx, cols_grid)
        side, kind = _choose_notch_side(rows_grid, cols_grid, ri, ci)
        patch = _make_frame(frame_size, hole_count, frame_colors[idx], side)
        if patch is None:
            return None
        frames.append(
            {
                "holes": int(hole_count),
                "patch": patch,
                "color": int(frame_colors[idx]),
                "side": side,
                "kind": kind,
            }
        )
        notch_kinds.append(kind)

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    top = random.randint(0, canvas_h - grid_pixel_h)
    left = random.randint(0, canvas_w - grid_pixel_w)

    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = top + row_offsets[ri]
        cc = left + col_offsets[ci]
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    placement_canvas = canvas.copy()
    placement_canvas[top : top + grid_pixel_h, left : left + grid_pixel_w] = int(BLOCKER_COLOR)
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

    expected = _build_expected_output(frames, sprites, row_offsets, col_offsets, cols_grid, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    metrics = _fusion_metrics(canvas)
    if metrics["max_gap"] < 1:
        return None
    if metrics["legacy_delta_frames"] != len(frames):
        return None
    if metrics["total_border_gap_pixels"] < len(frames):
        return None
    if "internal" not in notch_kinds and (rows_grid > 1 or cols_grid > 1):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
