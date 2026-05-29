# Combo variant of 8698868d (index 35)
# combo_spec: D9/3+D15/1
# Name: 逆序匹配+部分空框架

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The input still contains a tight grid of large square frames and a smaller set of square
#      sprites scattered elsewhere on azure background 8.
#    - Each frame encodes a hole-count via isolated bg cells in its interior. Each sprite encodes
#      a hole-count via its number of 4-connected bg-hole components.
#    - The original task used equal-count bijection. This combo changes two stages together:
#      1) there are more frames than sprites;
#      2) matching is rank-based and reversed.
#    - Small trace with frame counts [0,1,2,3,4] and sprite counts [2,3,4]:
#      original equal-subset reading would fill frames 2,3,4;
#      combo reading sorts frames low->high and sprites high->low, producing
#      frame0<-sprite4, frame1<-sprite3, frame2<-sprite2, while frames3 and 4 stay empty.
#    - So the reversal does not merely change which sprite goes into a filled frame; because the
#      cardinalities differ, it also decides which frame positions are left hollow.
#
# b) Complexity calibration
#    - Frame hole-counts stay unique and consecutive, which keeps the ranking visible and avoids
#      ambiguous ties.
#    - Sprite hole-counts are chosen as the highest contiguous subset of the frame counts. This
#      makes the combo visibly different from ordinary equal-subset matching: the low-count frames
#      become the filled ones, not the high-count frames.
#    - Sprite count is kept in 2..total_frames-1 so every example shows both at least one filled
#      frame and at least one empty frame.
#
# c) Implicit-assumption review of the original main()
#    - Component finding and square-size grouping remain valid because object geometry is unchanged.
#    - Grid reconstruction from the detected large squares also remains valid because the frame
#      layout is still gapless.
#    - The original equality test `f["holes"] == s["holes"]` is no longer valid; it must be
#      replaced by sorting frames ascending and sprites descending, then zipping the two lists.
#    - The original full-square paint `out[frame_square] = frame_color` is also invalid now:
#      unmatched frames would become solid instead of hollow. The output must draw borders first,
#      then paste content only for matched frames.
#    - The combo introduces a new assumption that both frame hole-counts and sprite hole-counts
#      are unique. The generator guarantees this.
#
# d) Placement legality
#    - Frames are still pasted as one tight rectangle with no gaps.
#    - Sprites are placed only onto background with padding, so they cannot overlap frames or one
#      another.
#    - Counts are limited by the shared realizable maximum for frames and sprites, so every chosen
#      hole-count can be built physically inside both object types.
#
# e) Visual observability
#    - A human can compare input sprites with output placements by visible hole-count complexity.
#      The most-holed sprites appear inside the least-holed frames.
#    - Because sprites are fewer than frames, some frame positions remain hollow in the output.
#      Choosing the sprite counts as the top subset ensures the filled positions differ from what
#      an equal-count subset rule would predict.
#
# f) Visual cleanliness
#    - Empty frames are rendered as clean single-cell borders with untouched bg interiors.
#    - Filled frames use the same nested-sprite presentation as the base task, so there is only
#      one visual convention to read inside the cropped output grid.
#
# g) Change inferability
#    - Colors remain incidental carriers of object identity. The transform depends only on visible
#      hole counts and on the presence or absence of a sprite partner under reversed ranking.
#    - The generator rejects degenerate cases where combo output matches a simpler equal-subset
#      interpretation, so the training signal consistently expresses the fused rule.


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
TOTAL_FRAME_RANGE = (3, 6)
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300
CANVAS_MIN = 15
CANVAS_MAX = 30


_LAYOUTS_BY_FRAME_COUNT: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        _count = _r * _c
        if TOTAL_FRAME_RANGE[0] <= _count <= TOTAL_FRAME_RANGE[1]:
            _LAYOUTS_BY_FRAME_COUNT.setdefault(_count, []).append((_r, _c))


def _draw_frame_border(out: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    out[r0, c0 : c0 + size] = int(color)
    out[r0 + size - 1, c0 : c0 + size] = int(color)
    out[r0 : r0 + size, c0] = int(color)
    out[r0 : r0 + size, c0 + size - 1] = int(color)


def _max_isolated_holes(square_size: int) -> int:
    inner_side = max(0, int(square_size) - 2)
    inner_cells = inner_side * inner_side
    return (inner_cells + 1) // 2


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

    if hole_count > _max_isolated_holes(size):
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    inner_cells = max(0, (size - 2) ** 2)
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
    if int(num) != int(hole_count):
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    if hole_count > _max_isolated_holes(frame_size):
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _lab, num = label_mask(frame != frame_color, connectivity=4)
    if int(num) != int(hole_count):
        return None
    return frame


def _reverse_partial_matches(
    frame_grid: list[dict[str, int]],
    sprites: list[dict[str, np.ndarray | int]],
) -> dict[tuple[int, int], dict[str, np.ndarray | int]]:
    sorted_frames = sorted(frame_grid, key=lambda frame: int(frame["holes"]))
    sorted_sprites = sorted(sprites, key=lambda sprite: int(sprite["holes"]), reverse=True)
    matches: dict[tuple[int, int], dict[str, np.ndarray | int]] = {}
    for frame, sprite in zip(sorted_frames, sorted_sprites):
        matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite
    return matches


def _paint_output(
    frame_grid: list[dict[str, int]],
    matches: dict[tuple[int, int], dict[str, np.ndarray | int]],
    frame_size: int,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    out = np.full((n_rows * frame_size, n_cols * frame_size), BG, dtype=int)
    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        _draw_frame_border(out, fr, fc, frame_size, frame_color)

        sprite = matches.get((fr, fc))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Nest scattered sprites into a cropped frame grid using reversed hole-count ranking.

    Input structure:
    - Background is azure (8). Larger same-sized square frames form a tight rectangular grid, and
      smaller same-sized square sprites sit elsewhere on the canvas.
    - Every frame interior contains isolated bg holes; every sprite contains bg hole components.
      Frame colors and sprite colors distinguish objects but do not encode the rule.

    Rule:
    - Count 4-connected hole components for each sprite and each frame interior.
    - Sort frames from fewest holes to most holes. Sort sprites from most holes to fewest holes.
    - Pair by rank in that reversed order until sprites run out. Paste each matched sprite inside
      its frame and recolor the sprite's hole pixels to the frame color. Unmatched frames remain
      hollow, showing only their borders in the cropped output grid.

    Key insight:
    - The transform depends on visible hole-count order, not on equal counts, colors, or position.
      Because there are fewer sprites than frames, the reversed ranking also determines which frame
      slots stay empty.
    """
    g = np.array(grid, dtype=int)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

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
    assert len(frames) > len(sprites_raw)

    frame_positions = [(int(frame["bbox"][0]), int(frame["bbox"][1])) for frame in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict[str, int]] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != frame_color
            _lab, num_holes = label_mask(hole_mask, connectivity=4)
            frame_grid.append(
                {
                    "color": frame_color,
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(num_holes),
                }
            )

    sprites: list[dict[str, np.ndarray | int]] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    frame_holes = [int(frame["holes"]) for frame in frame_grid]
    sprite_holes = [int(sprite["holes"]) for sprite in sprites]
    assert len(set(frame_holes)) == len(frame_holes)
    assert len(set(sprite_holes)) == len(sprite_holes)

    matches = _reverse_partial_matches(frame_grid, sprites)
    return _paint_output(frame_grid, matches, frame_size, n_rows, n_cols)


def _build_frame_grid(frames: list[dict[str, int | np.ndarray]], cols_grid: int, frame_size: int) -> list[dict[str, int]]:
    frame_grid: list[dict[str, int]] = []
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        frame_grid.append(
            {
                "color": int(frame["color"]),
                "out_r": int(ri * frame_size),
                "out_c": int(ci * frame_size),
                "holes": int(frame["holes"]),
            }
        )
    return frame_grid


def _build_sprite_records(sprites: list[tuple[int, np.ndarray, int]]) -> list[dict[str, np.ndarray | int]]:
    sprite_records: list[dict[str, np.ndarray | int]] = []
    for holes, patch, color in sprites:
        sprite_records.append(
            {
                "color": int(color),
                "holes": int(holes),
                "content": np.array(patch, dtype=int).copy(),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
            }
        )
    return sprite_records


def _build_expected_output(
    frames: list[dict[str, int | np.ndarray]],
    sprites: list[tuple[int, np.ndarray, int]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
) -> np.ndarray:
    frame_grid = _build_frame_grid(frames, cols_grid, frame_size)
    sprite_records = _build_sprite_records(sprites)
    matches = _reverse_partial_matches(frame_grid, sprite_records)
    return _paint_output(frame_grid, matches, frame_size, rows_grid, cols_grid)


def _build_equal_subset_output(
    frames: list[dict[str, int | np.ndarray]],
    sprites: list[tuple[int, np.ndarray, int]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
) -> np.ndarray:
    frame_grid = _build_frame_grid(frames, cols_grid, frame_size)
    sprite_by_holes = {
        int(holes): {
            "color": int(color),
            "holes": int(holes),
            "content": np.array(patch, dtype=int).copy(),
            "h": int(patch.shape[0]),
            "w": int(patch.shape[1]),
        }
        for holes, patch, color in sprites
    }
    matches: dict[tuple[int, int], dict[str, np.ndarray | int]] = {}
    for frame in frame_grid:
        hole_count = int(frame["holes"])
        if hole_count in sprite_by_holes:
            matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite_by_holes[hole_count]
    return _paint_output(frame_grid, matches, frame_size, rows_grid, cols_grid)


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    shared_hole_cap = min(_max_isolated_holes(frame_size), _max_isolated_holes(sprite_size))
    feasible_frame_counts = [
        count
        for count in sorted(_LAYOUTS_BY_FRAME_COUNT)
        if count >= 3 and count <= shared_hole_cap + 1
    ]
    if not feasible_frame_counts:
        return None

    total_frames = random.choice(feasible_frame_counts)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_FRAME_COUNT[total_frames])
    sprite_count = random.randint(2, total_frames - 1)

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, total_frames)
    sprite_colors = random.sample(available, sprite_count)

    max_start = min(HOLE_START_MAX, shared_hole_cap - total_frames + 1)
    if max_start < 0:
        return None
    start = random.randint(0, max_start)
    frame_hole_counts = list(range(start, start + total_frames))
    sprite_hole_counts = frame_hole_counts[-sprite_count:]

    frame_hole_order = frame_hole_counts[:]
    random.shuffle(frame_hole_order)
    sprite_hole_order = sprite_hole_counts[:]
    random.shuffle(sprite_hole_order)

    sprites: list[tuple[int, np.ndarray, int]] = []
    for i, hole_count in enumerate(sprite_hole_order):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(sprite_colors[i]))
        if patch is None:
            return None
        sprites.append((int(hole_count), patch, int(sprite_colors[i])))

    frames: list[dict[str, int | np.ndarray]] = []
    for i, hole_count in enumerate(frame_hole_order):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[i]))
        if patch is None:
            return None
        frames.append({"holes": int(hole_count), "patch": patch, "color": int(frame_colors[i])})

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    for _holes, patch, _color in sprites:
        try:
            r, c = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    equal_subset = _build_equal_subset_output(frames, sprites, rows_grid, cols_grid, frame_size)
    if np.array_equal(expected, equal_subset):
        return None

    reverse_gap = sum(
        abs(int(frame["holes"]) - int(sprite[0]))
        for frame, sprite in zip(
            sorted(frames, key=lambda item: int(item["holes"])),
            sorted(sprites, key=lambda item: int(item[0]), reverse=True),
        )
    )
    if reverse_gap <= 0:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
