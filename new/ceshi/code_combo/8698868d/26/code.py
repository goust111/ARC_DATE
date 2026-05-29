# Combo variant of 8698868d (index 26)
# combo_spec: D5/0+D6/0+D15/0
# Name: 全画布+保留精灵+过剩精灵

from __future__ import annotations

# Design analysis for combo D5/0 + D6/0 + D15/0
# a) Concrete behavior and trace:
#    The source task has N frames and N sprites. It matches them by the number of
#    4-connected hole components, nests each sprite into its frame, and returns only
#    the tight frame grid. That crop makes every scattered sprite disappear.
#    This combo changes three linked steps at once:
#      - D15: the input now contains N+K sprites, with K>=1 distractors whose hole
#        counts never appear in any frame, so matching needs a reject step.
#      - D5: the output is the full original canvas, not the cropped frame block.
#      - D6: source sprites are kept in place, so matched sprites are copied into
#        frames while both matched and unmatched source sprites remain visible.
#    Concrete trace: suppose two frames have hole counts {0, 2}, and the scattered
#    sprites have counts {2, 0, 3}. The original logic implicitly expects only two
#    sprites and would return a 1x2 frame strip. The combo logic first discards the
#    count-3 sprite as unmatched, copies the count-0 and count-2 sprites into the
#    two frames, and returns the whole canvas with all three scattered source sprites
#    still present at their original positions.
# b) Complexity calibration:
#    Distractors are not encoded by color or by a different shape family. They are
#    same-sized sprite squares, so humans must genuinely compare hole counts. Their
#    counts are chosen from the small leftover set outside the valid consecutive frame
#    range, keeping the task countable by eye. The generator also forces a canvas
#    margin around the frame grid, so the full-canvas output is visibly different from
#    the original cropped output.
# c) Implicit assumptions review against the original solver:
#    - Square-size splitting still works because frames remain the largest squares and
#      all sprites, valid or distractor, remain the smaller size.
#    - The original greedy matcher assumed `len(sprites) == len(frames)`. That breaks
#      here, so the solver must build frame demand first and ignore sprites whose hole
#      counts are absent from the frame grid.
#    - The original output coordinates `ri*frame_size, ci*frame_size` assumed a fresh
#      cropped canvas. That breaks under D5/D6; the solver must store the frames'
#      original canvas coordinates and rewrite those regions in place.
#    - Starting from an all-background output would wrongly erase every preserved
#      sprite. The combo must instead start from `g.copy()` and only repaint frame
#      squares so that frame holes disappear while scattered sprites stay untouched.
# d) Placement legality:
#    The generator chooses the number of frames only when the sprite size can realize
#    all valid hole counts plus at least one extra distractor count. Frames are placed
#    as one tight block with an outer margin. Sprites are then scattered via
#    `random_free_location(..., padding=1)`, which prevents overlap and even diagonal
#    touching with frames or other sprites.
# e) Visual observability:
#    In examples, humans can see that there are more sprite candidates than frame
#    slots, yet only some of those source sprites reappear nested inside frames. Since
#    every source sprite remains visible, the unmatched distractors can be identified
#    as the sprites that stay only in their source location and never gain a framed
#    copy. The frame block staying in place on a larger canvas makes D5 and D6 both
#    directly observable from a few examples.
# f) Visual cleanliness:
#    The output has only two readable regions: the compact frame block and the
#    separated scattered sprites. No annotations or overlays are added. Because the
#    solver rewrites only the frame squares, there is no local ambiguity about whether
#    a pixel belongs to a source sprite or a destination frame.
# g) Change inferability:
#    No numeric color ordering is used. Colors still serve only as object body colors,
#    while the match key is entirely geometric: visible hole-component count. The hole
#    recoloring rule remains local and repeated across all frames, so the preserved
#    source sprites do not introduce a second hidden color mapping problem.
# h) New assumptions introduced by this combo solver and how the generator enforces them:
#    - The solver assumes the frame grid is tight and axis-aligned; the generator keeps
#      the original rows x cols lattice with no gaps.
#    - The solver assumes each valid frame hole count has enough sprite candidates for
#      matching. The generator creates exactly one valid sprite for each frame count.
#    - The solver assumes source sprites sit outside the frame block so preserving them
#      cannot conflict with frame rewriting. The generator places all sprites after the
#      frame block with padding.
#    - The combo relies on all three dimensions being active. The generator rejects
#      samples unless there is at least one distractor sprite, at least one preserved
#      matched source sprite, at least one preserved unmatched source sprite, and a
#      non-zero canvas margin outside the frame block.

from collections import Counter
import math
from pathlib import Path
import random
import sys

import numpy as np


def _find_solvers_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "solvers" / "arc_utils.py"
        if candidate.exists():
            return candidate.parent
    raise RuntimeError("could not locate solvers/arc_utils.py")


_SOLVERS_DIR = _find_solvers_dir()
if str(_SOLVERS_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVERS_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PLACEMENT_PAD = 1
FRAME_MARGIN = 1
MAX_DISTRACTORS = 2
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 400
MIN_FILL_RATIO = 0.5


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _rows in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _cols in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        _n = _rows * _cols
        if N_RANGE[0] <= _n <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_n, []).append((_rows, _cols))


def _count_holes(region: np.ndarray, solid_color: int) -> int:
    hole_mask = np.array(region, dtype=int) != int(solid_color)
    _labels, num_holes = label_mask(hole_mask, connectivity=4)
    return int(num_holes)


def _max_isolated_count(size: int) -> int:
    inner_side = int(size) - 2
    if inner_side <= 0:
        return 0
    return int((inner_side * inner_side + 1) // 2)


def _find_square_regions(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        box = comp["bbox"]
        height = int(bbox_h(box))
        width = int(bbox_w(box))
        area = int(comp["area"])
        if height == width and 4 <= height <= 10 and area >= int(height * height * MIN_FILL_RATIO):
            square_regions.append((height, comp))

    if not square_regions:
        raise AssertionError("no square regions found")

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)
    frames = sizes.get(frame_size, [])
    sprites = sizes.get(sprite_size, [])
    if not frames:
        raise AssertionError("no frames found")
    return frame_size, sprite_size, frames, sprites


def _extract_scene_signature(grid: np.ndarray) -> dict:
    frame_size, sprite_size, frames_raw, sprites_raw = _find_square_regions(grid)

    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames_raw]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            inp_r = int(min_r + row_idx * frame_size)
            inp_c = int(min_c + col_idx * frame_size)
            region = grid[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            if region.shape != (frame_size, frame_size):
                raise AssertionError("incomplete frame region")

            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            holes = _count_holes(region[1:-1, 1:-1], frame_color)
            frame_grid.append(
                {
                    "inp_r": int(inp_r),
                    "inp_c": int(inp_c),
                    "color": int(frame_color),
                    "holes": int(holes),
                }
            )

    sprites: list[dict] = []
    for idx, comp in enumerate(sprites_raw):
        box = comp["bbox"]
        sprite_region = crop_bbox(grid, box)
        sprite_color = int(comp["color"])
        sprites.append(
            {
                "index": int(idx),
                "bbox": tuple(map(int, box)),
                "color": int(sprite_color),
                "holes": int(_count_holes(sprite_region, sprite_color)),
                "content": np.array(sprite_region, dtype=int).copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    frame_need = Counter(int(frame["holes"]) for frame in frame_grid)
    return {
        "frame_size": int(frame_size),
        "sprite_size": int(sprite_size),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "frames": frame_grid,
        "sprites": sprites,
        "frame_need": frame_need,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match frame slots to sprites by connected-hole count, keep every source sprite visible,
    and rewrite the result on the full original canvas.

    Input structure:
    - Background is azure (8). Larger square frames form one tight rows x cols grid somewhere on the canvas.
    - Smaller square sprites are scattered elsewhere. There are more sprites than frames: one valid sprite per frame
      plus extra distractor sprites of the same size.
    - Frames and sprites each use a solid body color with bg-colored holes inside.

    Rule:
    - Count 4-connected hole components inside each frame interior and inside each sprite.
    - Use only sprites whose hole counts appear in the frame grid; ignore every extra sprite whose count never appears.
    - For each frame, copy the matched sprite into the frame interior and recolor that sprite's hole pixels to the
      frame color.
    - Return the full canvas. The frame block is updated in place, while all scattered source sprites, matched or not,
      remain at their original positions.

    Key insight:
    The puzzle now combines selection and copy: hole count still defines the match, but some same-sized sprites are
    deliberate decoys, and the chosen sprites are duplicated into frames rather than removed from their source locations.
    """
    g = np.array(grid, dtype=int)
    scene = _extract_scene_signature(g)
    frame_size = int(scene["frame_size"])
    frame_grid = scene["frames"]
    frame_need: Counter = scene["frame_need"]

    sprite_buckets: dict[int, list[dict]] = {}
    for sprite in scene["sprites"]:
        holes = int(sprite["holes"])
        if frame_need[holes] > 0:
            sprite_buckets.setdefault(holes, []).append(sprite)

    for hole_count, need in frame_need.items():
        if len(sprite_buckets.get(int(hole_count), [])) < int(need):
            raise AssertionError("missing matched sprite for frame hole count")

    matches: dict[tuple[int, int], dict] = {}
    used_per_hole = Counter()
    for frame in frame_grid:
        holes = int(frame["holes"])
        bucket_index = int(used_per_hole[holes])
        matches[(int(frame["inp_r"]), int(frame["inp_c"]))] = sprite_buckets[holes][bucket_index]
        used_per_hole[holes] += 1

    out = g.copy()
    for frame in frame_grid:
        fr = int(frame["inp_r"])
        fc = int(frame["inp_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = int(frame_color)

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

    components: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for comp_idx, comp in enumerate(components):
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
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    if _count_holes(sprite, sprite_color) != int(hole_count):
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

    if _count_holes(frame, frame_color) != int(hole_count):
        return None
    return frame


def _choose_instance_params() -> tuple[int, int, int, int, int, int]:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    max_sprite_holes = _max_isolated_count(sprite_size)

    n_choices = [n for n in sorted(_LAYOUTS_BY_N) if n <= max_sprite_holes]
    if not n_choices:
        raise AssertionError("no legal frame counts for chosen frame size")
    n_frames = random.choice(n_choices)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n_frames])

    start_choices: list[int] = []
    for start in range(max_sprite_holes - n_frames + 2):
        valid_counts = set(range(start, start + n_frames))
        if start <= HOLE_START_MAX and len(valid_counts) < max_sprite_holes + 1:
            start_choices.append(int(start))
    if not start_choices:
        raise AssertionError("no legal hole-count start values")

    start = random.choice(start_choices)
    distractor_pool = [k for k in range(max_sprite_holes + 1) if k not in set(range(start, start + n_frames))]
    k_extra = random.randint(1, min(MAX_DISTRACTORS, len(distractor_pool)))
    return int(frame_size), int(sprite_size), int(rows_grid), int(cols_grid), int(start), int(k_extra)


def _build_expected_output(
    canvas: np.ndarray,
    frames: list[dict],
    valid_sprites: list[tuple[int, np.ndarray]],
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.array(canvas, dtype=int).copy()
    sprite_by_holes = {int(hole_count): np.array(patch, dtype=int).copy() for hole_count, patch in valid_sprites}
    for frame in frames:
        rr = int(frame["r"])
        cc = int(frame["c"])
        holes = int(frame["holes"])
        frame_color = int(frame["color"])

        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        content = sprite_by_holes[holes].copy()
        content[content == BG] = int(frame_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    try:
        frame_size, sprite_size, rows_grid, cols_grid, start, k_extra = _choose_instance_params()
    except AssertionError:
        return None

    n_frames = rows_grid * cols_grid
    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    total_sprites = n_frames + k_extra
    effective_span = sprite_size + 2 * SPRITE_PLACEMENT_PAD
    needed_area = grid_pixel_h * grid_pixel_w + total_sprites * effective_span * effective_span + 12
    side_lb = max(
        CANVAS_MIN,
        grid_pixel_h + 2 * FRAME_MARGIN,
        grid_pixel_w + 2 * FRAME_MARGIN,
        int(math.ceil(math.sqrt(needed_area))),
    )
    if side_lb > CANVAS_MAX:
        return None
    roomy_lb = max(side_lb, CANVAS_MAX - 5)
    canvas_h = random.randint(roomy_lb, CANVAS_MAX)
    canvas_w = random.randint(roomy_lb, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n_frames)
    sprite_colors = random.sample(available, total_sprites)

    valid_counts_base = list(range(start, start + n_frames))
    frame_hole_counts = valid_counts_base[:]
    sprite_valid_counts = valid_counts_base[:]
    random.shuffle(frame_hole_counts)
    random.shuffle(sprite_valid_counts)

    distractor_pool = [k for k in range(_max_isolated_count(sprite_size) + 1) if k not in set(valid_counts_base)]
    distractor_counts = random.sample(distractor_pool, k_extra)

    valid_sprites: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(sprite_valid_counts):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(sprite_colors[idx]))
        if patch is None:
            return None
        valid_sprites.append((int(hole_count), patch))

    distractor_sprites: list[tuple[int, np.ndarray]] = []
    for j, hole_count in enumerate(distractor_counts):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(sprite_colors[n_frames + j]))
        if patch is None:
            return None
        distractor_sprites.append((int(hole_count), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for idx, hole_count in enumerate(frame_hole_counts):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[idx]))
        if patch is None:
            return None
        frames.append((int(hole_count), patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(FRAME_MARGIN, canvas_h - grid_pixel_h - FRAME_MARGIN)
    c0 = random.randint(FRAME_MARGIN, canvas_w - grid_pixel_w - FRAME_MARGIN)

    frame_records: list[dict] = []
    for idx, (hole_count, frame_patch) in enumerate(frames):
        row_idx, col_idx = divmod(idx, cols_grid)
        rr = int(r0 + row_idx * frame_size)
        cc = int(c0 + col_idx * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch
        frame_records.append(
            {
                "holes": int(hole_count),
                "color": int(frame_colors[idx]),
                "r": int(rr),
                "c": int(cc),
            }
        )

    all_sprites: list[tuple[int, np.ndarray, bool]] = []
    for hole_count, patch in valid_sprites:
        all_sprites.append((int(hole_count), patch, True))
    for hole_count, patch in distractor_sprites:
        all_sprites.append((int(hole_count), patch, False))
    random.shuffle(all_sprites)

    placed_sprites: list[dict] = []
    for hole_count, sprite_patch, is_matched in all_sprites:
        try:
            rr, cc = random_free_location(canvas, sprite_patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, sprite_patch, rr, cc, transparent=BG)
        placed_sprites.append(
            {
                "holes": int(hole_count),
                "matched": bool(is_matched),
                "r": int(rr),
                "c": int(cc),
                "patch": np.array(sprite_patch, dtype=int).copy(),
            }
        )

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(canvas, frame_records, valid_sprites, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    scene = _extract_scene_signature(canvas)
    if len(scene["sprites"]) <= len(scene["frames"]):
        return None

    frame_hole_keys = set(int(k) for k in scene["frame_need"])
    if not any(int(sprite["holes"]) not in frame_hole_keys for sprite in scene["sprites"]):
        return None

    frame_block_mask = np.zeros_like(canvas, dtype=bool)
    frame_block_mask[r0 : r0 + grid_pixel_h, c0 : c0 + grid_pixel_w] = True
    outside_mask = ~frame_block_mask
    if not np.any((canvas != BG) & outside_mask):
        return None
    if not np.array_equal(out[outside_mask], canvas[outside_mask]):
        return None

    preserved_matched = 0
    preserved_unmatched = 0
    for record in placed_sprites:
        rr = int(record["r"])
        cc = int(record["c"])
        patch_out = out[rr : rr + sprite_size, cc : cc + sprite_size]
        if np.array_equal(patch_out, record["patch"]):
            if record["matched"]:
                preserved_matched += 1
            else:
                preserved_unmatched += 1
    if preserved_matched < 1 or preserved_unmatched < 1:
        return None

    if canvas_h * canvas_w == grid_pixel_h * grid_pixel_w:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
