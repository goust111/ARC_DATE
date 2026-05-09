# Combo variant of 89565ca0 (index 36)
# combo_spec: D1/4+D3/2+D4/0
# Name: 嵌套方框+无噪声+降序

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete object model and behavior trace
# - Original 89565ca0 used damaged rectilinear wall frames plus one scattered noise
#   color. The solver had to reconstruct walls, count enclosed background rooms, drop
#   the unique zero-room noise color, then sort ascending.
# - This combo replaces the source object with monochrome nested square borders. A
#   color with k concentric square outlines directly encodes value k. There is no
#   scattered pad/noise color anywhere in the input, so every non-black color is real.
# - Small trace:
#   original: a green partition frame could reconstruct to 4 enclosed rooms, so its
#   output row was four green cells and then the noise color as padding.
#   combo: a green object with outer/middle/inner square borders has layer count 3, so
#   its output row is three green cells and then black padding. Rows are ordered from
#   largest layer count to smallest.
# - Main difference: the counted quantity changes from hidden room components to visible
#   shell depth, and the ordering direction flips from ascending to descending.
#
# b) Complexity calibration
# - The task stays richer than "count plain squares" because each object may have
#   1-6 layers and each example shows several colors with different depths.
# - Noise is removed by spec, so visual difficulty is intentionally lower. To avoid
#   adding back hidden complexity, the generator uses distinct counts and separated
#   placement; the challenge remains the visible depth-to-bar mapping plus descending
#   order.
#
# c) Implicit-assumption audit of the original main()
# - Original _reconstruct_walls() assumed long dense horizontal/vertical evidence lines.
#   Nested squares do not satisfy that model, so this path must be removed entirely.
# - Original "rooms == 0 identifies the noise color" is invalid here because D4/0
#   removes the extra pad color. The new solver treats every non-background color as an
#   object and uses black as output padding.
# - Original ascending sort is explicitly inverted by D1/4. Using the old direction
#   would silently produce wrong outputs on otherwise valid nested-square inputs.
#
# d) Placement legality
# - Generator builds square bboxes only, keeps them fully inside the grid, and enforces
#   at least a 1-cell gap between object bboxes. This guarantees no overlap, no
#   clipping, and fully visible shells for every color.
#
# e) Visual observability
# - From a few examples, a human can directly see: each color forms nested same-color
#   square borders; more layers mean a longer output bar; and the longest bar appears
#   first because ordering is descending. No invisible metadata such as color id or
#   render order is needed.
#
# f) Visual cleanliness
# - Removing both noise and overlaps keeps each object legible. Black background remains
#   empty space only, so the scene is much cleaner than the original task family.
#
# g) Change inferability
# - No color remapping is introduced. The only inferred relation is structural:
#   "count same-color square shells, keep that color, sort by count descending".
#
# h) Quantitative anti-degeneration target
# - The generator samples several distinct layer counts per grid, leaves visible empty
#   gaps between shells, and validates that the descending output differs from both the
#   ascending order and the naive row-major scan order. This keeps the D1 inversion
#   active rather than cosmetic and prevents thick-border ambiguity.
#
# i) New solver assumptions that the generator guarantees
# - Each non-black color occupies exactly one connected nested-square object.
# - Each visible color bbox is square and at least 3x3.
# - Every shell perimeter is fully present, so exact square-border detection is valid.
# - Counts are distinct in generated examples, so tie ordering never becomes the rule a
#   human must infer from scarce examples.


_FRAME_COUNT_RANGE = (3, 4)
_LAYER_COUNT_RANGE = (1, 5)
_MAX_LAYERS_BY_FRAMES = {4: 4, 3: 5}
_GRID_SIZE_BY_FRAMES = {
    4: (24, 32, 24, 32, 18),
    3: (24, 32, 24, 32, 22),
}
_INNER_SIDE_RANGE = (3, 7)
_LAYER_GAP_RANGE = (2, 3)
_PLACEMENT_TRIES = 200
_GEN_RETRIES = 300


def _paint_square_border(mask: np.ndarray, inset: int) -> None:
    n = int(mask.shape[0])
    r0 = int(inset)
    c0 = int(inset)
    r1 = int(n - 1 - inset)
    c1 = int(n - 1 - inset)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _detect_shell_insets(mask: np.ndarray) -> tuple[list[int], tuple[int, int, int, int] | None]:
    b = bbox(mask)
    if b is None:
        return [], None

    cropped = crop_bbox(mask, b).astype(bool)
    h, w = cropped.shape
    if h != w or h < 3:
        return [], b

    n = int(h)
    insets: list[int] = []
    for inset in range(n // 2 + 1):
        side = int(n - 2 * inset)
        if side < 3:
            break
        r0 = int(inset)
        c0 = int(inset)
        r1 = int(n - 1 - inset)
        c1 = int(n - 1 - inset)
        if (
            bool(cropped[r0, c0 : c1 + 1].all())
            and bool(cropped[r1, c0 : c1 + 1].all())
            and bool(cropped[r0 : r1 + 1, c0].all())
            and bool(cropped[r0 : r1 + 1, c1].all())
        ):
            insets.append(int(inset))
    return insets, b


def _layer_count_for_color(g: np.ndarray, color: int) -> tuple[int, tuple[int, int, int, int] | None]:
    insets, b = _detect_shell_insets(g == int(color))
    return int(len(insets)), b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored nested-square object into a horizontal bar whose length is its nesting depth.

    Input structure:
    - Black background with several separated colored drawings, each made from one or more concentric square borders.
    - There is no scattered noise color and no damaged wall pixel; every non-black color belongs to one real object.
    - Each real object's visible bbox is square, and larger depth means more nested same-color borders.

    Rule:
    - For each non-background color, crop its square bbox and count how many insets still form complete square perimeters.
    - That shell count is the object's value.
    - Sort the objects by shell count in descending order.
    - Output one row per color: fill the first `count` cells with that color and use black to pad the row to the maximum count.

    Key insight:
    - The value is now written directly into the picture as the number of concentric square shells, so there is no need to reconstruct rooms or identify a separate pad color. The only nontrivial step is reading shell depth and reversing the old sort direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    objects: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in colors_present(g, exclude={bg}):
        layers, b = _layer_count_for_color(g, int(color))
        assert b is not None
        assert layers > 0
        objects.append((int(color), int(layers), b))

    assert objects
    ordered = sorted(objects, key=lambda item: (-item[1], item[2][0], item[2][1]))

    width = max(layers for _color, layers, _b in ordered)
    out = np.zeros((len(ordered), width), dtype=int)
    for r, (color, layers, _b) in enumerate(ordered):
        out[r, :layers] = int(color)
    return out


def _choose_layer_counts(frame_count: int) -> list[int] | None:
    max_layers = _MAX_LAYERS_BY_FRAMES.get(frame_count, _LAYER_COUNT_RANGE[1])
    available = list(range(_LAYER_COUNT_RANGE[0], max_layers + 1))
    if frame_count > len(available):
        return None
    counts = random.sample(available, frame_count)
    random.shuffle(counts)
    return [int(v) for v in counts]


def _random_gaps_with_sum(n: int, total: int, low: int, high: int) -> list[int] | None:
    if n == 0:
        return [] if total == 0 else None
    if total < n * low or total > n * high:
        return None

    gaps = [int(low)] * n
    remaining = int(total - n * low)
    for i in range(n):
        add_max = min(int(high - low), remaining)
        add = random.randint(0, add_max)
        gaps[i] += int(add)
        remaining -= int(add)
    idxs = list(range(n))
    random.shuffle(idxs)
    for idx in idxs:
        if remaining <= 0:
            break
        extra = min(int(high - gaps[idx]), remaining)
        if extra > 0:
            gaps[idx] += int(extra)
            remaining -= int(extra)
    if remaining != 0:
        return None
    random.shuffle(gaps)
    return gaps


def _build_nested_frame(target_layers: int, max_side: int) -> np.ndarray | None:
    gap_count = max(0, int(target_layers - 1))
    min_gap_total = int(gap_count * _LAYER_GAP_RANGE[0])
    max_gap_total = int(gap_count * _LAYER_GAP_RANGE[1])
    feasible_gap_total_max = int((max_side - _INNER_SIDE_RANGE[0]) // 2)
    if min_gap_total > feasible_gap_total_max:
        return None

    for _ in range(80):
        gap_total = random.randint(min_gap_total, min(max_gap_total, feasible_gap_total_max))
        gaps = _random_gaps_with_sum(gap_count, gap_total, _LAYER_GAP_RANGE[0], _LAYER_GAP_RANGE[1])
        if gaps is None:
            continue

        inner_max = min(_INNER_SIDE_RANGE[1], max_side - 2 * gap_total)
        if inner_max < _INNER_SIDE_RANGE[0]:
            continue

        inner_side = random.randint(_INNER_SIDE_RANGE[0], inner_max)
        side = int(inner_side + 2 * gap_total)
        mask = np.zeros((side, side), dtype=bool)

        inset = 0
        for layer_idx in range(target_layers):
            if layer_idx > 0:
                inset += gaps[layer_idx - 1]
            _paint_square_border(mask, int(inset))
        return mask
    return None


def _boxes_separated(
    pos_a: tuple[int, int],
    side_a: int,
    pos_b: tuple[int, int],
    side_b: int,
    margin: int = 1,
) -> bool:
    ra, ca = pos_a
    rb, cb = pos_b
    return (
        ra + side_a - 1 + margin < rb
        or rb + side_b - 1 + margin < ra
        or ca + side_a - 1 + margin < cb
        or cb + side_b - 1 + margin < ca
    )


def _place_frames(frames: list[tuple[int, int, np.ndarray]], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = [(-1, -1)] * len(frames)
    order = sorted(range(len(frames)), key=lambda i: -frames[i][2].shape[0])
    for idx in order:
        side = int(frames[idx][2].shape[0])
        placed = False
        for _ in range(_PLACEMENT_TRIES):
            cand = (
                random.randint(0, grid_h - side),
                random.randint(0, grid_w - side),
            )
            if all(
                other_pos[0] < 0
                or _boxes_separated(
                    cand,
                    side,
                    other_pos,
                    int(frames[other_idx][2].shape[0]),
                )
                for other_idx, other_pos in enumerate(positions)
                if other_idx != idx
            ):
                positions[idx] = cand
                placed = True
                break
        if not placed:
            return None
    return positions


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for (color, _layers, mask), (r0, c0) in zip(frames, positions):
        rs, cs = np.where(mask)
        grid[rs + r0, cs + c0] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    layer_counts = _choose_layer_counts(frame_count)
    if layer_counts is None:
        return None

    chosen_colors = random.sample(list(range(1, 10)), frame_count)
    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, frame_max_side = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, layers in zip(chosen_colors, layer_counts):
        build_max_side = min(int(frame_max_side), int(grid_h - 2), int(grid_w - 2))
        mask = _build_nested_frame(int(layers), build_max_side)
        if mask is None:
            return None
        frames.append((int(color), int(layers), mask))

    positions = _place_frames(frames, grid_h, grid_w)
    if positions is None:
        return None

    grid = _render_frames(frames, positions, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        ((layer_counts[i], positions[i][0], positions[i][1], chosen_colors[i]) for i in range(frame_count)),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    max_layers = max(count for count, _r, _c, _color in frame_info)
    expected = np.zeros((frame_count, max_layers), dtype=int)
    for r, (count, _r0, _c0, color) in enumerate(frame_info):
        expected[r, :count] = int(color)

    if not np.array_equal(output, expected):
        return None

    ascending = np.zeros_like(expected)
    asc_info = sorted(frame_info, key=lambda item: (item[0], item[1], item[2]))
    for r, (count, _r0, _c0, color) in enumerate(asc_info):
        ascending[r, :count] = int(color)
    if np.array_equal(expected, ascending):
        return None

    row_major = sorted(frame_info, key=lambda item: (item[1], item[2], item[0], item[3]))
    if [color for _count, _r0, _c0, color in row_major] == [color for _count, _r0, _c0, color in frame_info]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
