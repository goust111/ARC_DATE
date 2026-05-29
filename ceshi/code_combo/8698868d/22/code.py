# Combo variant of 8698868d (index 22)
# combo_spec: D9/3+D19/0
# Name: 逆序匹配排序输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The visible ingredients stay the same as the base task: one tight grid of large square
#      frames on azure background 8, plus the same number of smaller square sprites scattered
#      elsewhere on the canvas.
#    - Frames still encode a count by isolated bg cells in their interiors; sprites still encode
#      a count by the number of 4-connected hole components in their interiors.
#    - Original behavior on a 1x3 grid with frame hole counts [2, 0, 1] and sprite counts
#      {0, 1, 2} was:
#        slot order kept as [2, 0, 1]
#        matching was equal-count, so output became [frame2<-sprite2, frame0<-sprite0, frame1<-sprite1]
#    - In this combo:
#        1. Match by reversed rank: frames sorted asc [0, 1, 2] pair with sprites sorted desc
#           [2, 1, 0], so assignments are frame0<-sprite2, frame1<-sprite1, frame2<-sprite0.
#        2. Independently sort output frames by frame hole count ascending, so output slots are
#           laid out as frame0, frame1, frame2, yielding [frame0<-sprite2, frame1<-sprite1, frame2<-sprite0].
#    - The important difference is that the solver must keep two orderings in mind: one ordering
#      defines which sprite goes into each frame, and a second ordering defines where each frame
#      appears in the final compact grid.
#
# b) Complexity calibration
#    - Hole counts are kept unique and consecutive 0..N-1, with N in 2..6. This keeps both the
#      reverse matching and the sorted output visually legible and unambiguous.
#    - Generator rejection checks prevent accidental collapse into simpler variants:
#      input frame order may not already be ascending, and the combo output must differ from
#      original, D9-only, and D19-only reference outputs.
#    - Sprite colors stay distinct within the sprite set, so reverse matching changes which sprite
#      appears inside each frame in a way that a human can see from a few examples.
#
# c) Implicit-assumption review of the original main()
#    - Component extraction and square-size grouping remain valid because the object vocabulary and
#      geometry are unchanged.
#    - Reconstructing the frame grid from the large components still works because frames are still
#      placed as a gapless rectangle on the input canvas.
#    - The original equality-based match assumption is invalid: `frame holes == sprite holes`
#      must be replaced with an order-based bijection between ascending frame counts and descending
#      sprite counts.
#    - The original output-placement assumption is also invalid: row-major input frame order can no
#      longer be reused for output. Output slots must be filled by frames sorted ascending by their
#      own hole counts.
#    - Because the same frame order key appears in both stages, the implementation keeps separate
#      variables for "matching order" and "output order" to avoid silently merging the two rules.
#
# d) Placement legality
#    - Frames are pasted as one tight rows_grid x cols_grid rectangle, so solver reconstruction by
#      min/max frame coordinates and frame size is well-defined.
#    - Sprites are placed only on background with padding=1, which prevents overlap with frames and
#      with each other.
#    - Counts 0..N-1 remain physically feasible because the smallest legal frame size is 6, whose
#      4x4 interior can host up to 8 mutually 4-isolated single-cell holes.
#
# e) Visual observability
#    - A human can directly count frame holes in the input grid, notice that output frame order is
#      low-to-high, then compare which sprite silhouettes/colors end up inside those sorted frames.
#    - Because the most-holed sprite goes into the least-holed frame, the two rules reinforce each
#      other without using hidden numeric properties such as raw color ids.
#    - Rejecting already-sorted input frame orders ensures the output reordering is observable over
#      2-3 examples instead of accidentally looking like the D9-only variant.
#
# f) Visual cleanliness
#    - Output remains a clean compact grid of solid frames with nested sprites. No extra markers,
#      labels, or overlapping annotations are introduced.
#    - Hole recoloring is unchanged from the base task, so every output cell is still non-bg and
#      easy to parse.
#
# g) Change inferability
#    - Colors still act only as object identities. They help track which sprite moved where, but the
#      controlling logic is fully visible in the count structure of the holes.
#    - Because the generator fixes the count set to 0..N-1, examples consistently demonstrate a
#      complete rank reversal rather than an arbitrary permutation.
#
# New assumptions introduced by the combo solver
# ----------------------------------------------
# - Frame hole counts and sprite hole counts must form unique sortable sets of equal size.
# - The frame grid must be complete with no missing slots inside its bounding rectangle.
# - At least one frame must move under output sorting, otherwise the combo would reduce to D9-only.
# - At least one frame must receive a different sprite than equal matching would assign, otherwise
#   the combo would reduce to D19-only. The generator explicitly enforces all of these conditions.


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 50
GEN_RETRIES = 300


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _reverse_count_map(counts: list[int]) -> dict[int, int]:
    ordered = sorted(int(v) for v in counts)
    return {int(lo): int(hi) for lo, hi in zip(ordered, reversed(ordered))}


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, int, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        box = comp["bbox"]
        h = int(bbox_h(box))
        w = int(bbox_w(box))
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
    frames_raw = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames_raw
    assert sprites_raw

    frame_positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames_raw]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)
    assert n_rows * n_cols == len(frames_raw)

    frame_grid: list[dict] = []
    frame_index = 0
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
                    "index": int(frame_index),
                    "inp_r": int(inp_r),
                    "inp_c": int(inp_c),
                    "color": int(frame_color),
                    "holes": int(num_holes),
                }
            )
            frame_index += 1

    sprites: list[dict] = []
    for sprite_index, comp in enumerate(sprites_raw):
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "index": int(sprite_index),
                "color": int(sprite_color),
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    return frame_size, sprite_size, n_rows, n_cols, frame_grid, sprites


def _ordered_frames_for_output(frames: list[dict]) -> list[dict]:
    return sorted(frames, key=lambda frame: (int(frame["holes"]), int(frame["index"])))


def _sprite_assignment_by_frame_holes(frames: list[dict], sprites: list[dict], *, reverse_match: bool) -> dict[int, dict]:
    frame_holes = [int(frame["holes"]) for frame in frames]
    sprite_holes = [int(sprite["holes"]) for sprite in sprites]
    assert len(frames) == len(sprites)
    assert len(set(frame_holes)) == len(frames)
    assert len(set(sprite_holes)) == len(sprites)

    ordered_frames = sorted(frames, key=lambda frame: (int(frame["holes"]), int(frame["index"])))
    ordered_sprites = sorted(
        sprites,
        key=lambda sprite: (-int(sprite["holes"]), int(sprite["index"])),
    ) if reverse_match else sorted(
        sprites,
        key=lambda sprite: (int(sprite["holes"]), int(sprite["index"])),
    )

    assignment: dict[int, dict] = {}
    for frame, sprite in zip(ordered_frames, ordered_sprites):
        assignment[int(frame["holes"])] = sprite
    return assignment


def _render_output(
    frames: list[dict],
    sprite_for_frame_holes: dict[int, dict],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    *,
    sort_output: bool,
) -> np.ndarray:
    ordered_frames = _ordered_frames_for_output(frames) if sort_output else list(frames)
    out = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    for out_idx, frame in enumerate(ordered_frames):
        ri, ci = divmod(out_idx, cols_grid)
        out_r = int(ri * frame_size)
        out_c = int(ci * frame_size)
        frame_color = int(frame["color"])
        out[out_r : out_r + frame_size, out_c : out_c + frame_size] = frame_color

        sprite = sprite_for_frame_holes[int(frame["holes"])]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[out_r + 1 : out_r + 1 + sh, out_c + 1 : out_c + 1 + sw] = content
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse-rank match sprites to frames, then emit the frames in ascending hole-count order.

    Input structure:
    - Background is azure (8). Large square frames occupy one tight rectangular grid, while smaller
      square sprites with their own colors are scattered elsewhere on the background.
    - Each frame interior contains isolated background holes; each sprite interior contains one or
      more connected background-hole components. Hole counts are unique across each family.

    Rule:
    - Reconstruct the frame grid and count hole components for every frame and every sprite.
    - Rank frames from fewest holes to most holes, and rank sprites from most holes to fewest.
    - Pair items by those opposite rankings, recolor each matched sprite's hole pixels to the
      containing frame color, and place the sprite inside the frame.
    - Finally, arrange the output grid itself by frame hole count from small to large, ignoring the
      original row-major frame positions from the input.

    Key insight:
    - There are two separate orderings to infer: reverse ordering decides the frame-sprite pairing,
      while forward ordering decides the final output slot arrangement.
    """
    frame_size, _sprite_size, n_rows, n_cols, frames, sprites = _extract_scene(grid)

    frames_for_matching = sorted(frames, key=lambda frame: (int(frame["holes"]), int(frame["index"])))
    sprites_for_matching = sorted(
        sprites,
        key=lambda sprite: (-int(sprite["holes"]), int(sprite["index"])),
    )
    assert len(frames_for_matching) == len(sprites_for_matching)

    sprite_for_frame_holes: dict[int, dict] = {}
    for frame, sprite in zip(frames_for_matching, sprites_for_matching):
        sprite_for_frame_holes[int(frame["holes"])] = sprite

    frames_for_output = _ordered_frames_for_output(frames)
    return _render_output(
        frames_for_output,
        sprite_for_frame_holes,
        n_rows,
        n_cols,
        frame_size,
        sort_output=False,
    )


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
    for comp_idx, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_holes)))
        for _ in range(grow):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = r + dr
                    nc = c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for other_idx, other in enumerate(comps)
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

    _lab, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if int(num_holes) != int(hole_count):
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

    _lab, num_holes = label_mask(frame != frame_color, connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _build_expected_output(
    frames: list[dict],
    sprites: list[dict],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    *,
    reverse_match: bool,
    sort_output: bool,
) -> np.ndarray:
    sprite_for_frame_holes = _sprite_assignment_by_frame_holes(
        frames,
        sprites,
        reverse_match=reverse_match,
    )
    return _render_output(
        frames,
        sprite_for_frame_holes,
        rows_grid,
        cols_grid,
        frame_size,
        sort_output=sort_output,
    )


def _frame_hole_sequence(frames: list[dict]) -> list[int]:
    return [int(frame["holes"]) for frame in frames]


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    height = random.randint(min_h, CANVAS_MAX)
    width = random.randint(min_w, CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, n)
    sprite_colors = random.sample(available, n)

    hole_counts = list(range(n))
    input_frame_holes = hole_counts[:]
    random.shuffle(input_frame_holes)
    if input_frame_holes == sorted(input_frame_holes):
        return None

    sprite_hole_order = hole_counts[:]
    random.shuffle(sprite_hole_order)

    sprites: list[dict] = []
    for idx, holes in enumerate(sprite_hole_order):
        patch = _make_sprite_interior(sprite_size, int(holes), int(sprite_colors[idx]))
        if patch is None:
            return None
        sprites.append(
            {
                "index": int(idx),
                "holes": int(holes),
                "color": int(sprite_colors[idx]),
                "content": np.array(patch, dtype=int),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
            }
        )

    frames: list[dict] = []
    for idx, holes in enumerate(input_frame_holes):
        patch = _make_frame(frame_size, int(holes), int(frame_colors[idx]))
        if patch is None:
            return None
        frames.append(
            {
                "index": int(idx),
                "holes": int(holes),
                "color": int(frame_colors[idx]),
                "patch": np.array(patch, dtype=int),
            }
        )

    combo_expected = _build_expected_output(
        frames,
        sprites,
        rows_grid,
        cols_grid,
        frame_size,
        reverse_match=True,
        sort_output=True,
    )
    original_expected = _build_expected_output(
        frames,
        sprites,
        rows_grid,
        cols_grid,
        frame_size,
        reverse_match=False,
        sort_output=False,
    )
    reverse_only_expected = _build_expected_output(
        frames,
        sprites,
        rows_grid,
        cols_grid,
        frame_size,
        reverse_match=True,
        sort_output=False,
    )
    sorted_only_expected = _build_expected_output(
        frames,
        sprites,
        rows_grid,
        cols_grid,
        frame_size,
        reverse_match=False,
        sort_output=True,
    )

    if np.array_equal(combo_expected, original_expected):
        return None
    if np.array_equal(combo_expected, reverse_only_expected):
        return None
    if np.array_equal(combo_expected, sorted_only_expected):
        return None

    canvas = np.full((height, width), BG, dtype=int)
    r0 = random.randint(0, height - grid_pixel_h)
    c0 = random.randint(0, width - grid_pixel_w)
    for idx, frame in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = int(r0 + ri * frame_size)
        cc = int(c0 + ci * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    for sprite in sprites:
        try:
            r, c = random_free_location(
                canvas,
                np.array(sprite["content"], dtype=int),
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(
            canvas,
            np.array(sprite["content"], dtype=int),
            r,
            c,
            transparent=BG,
        )

    try:
        out = main(canvas)
    except Exception:
        return None
    if out.shape != combo_expected.shape or not np.array_equal(out, combo_expected):
        return None

    detected = _extract_scene(canvas)
    detected_frames = detected[4]
    detected_holes = _frame_hole_sequence(detected_frames)
    if detected_holes == sorted(detected_holes):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
