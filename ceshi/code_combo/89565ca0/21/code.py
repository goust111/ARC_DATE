# Combo variant of 89565ca0 (index 21)
# combo_spec: D3/2+D4/0+D9/0
# Name: 嵌套方框+无噪声+无重叠

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete object model and behavior trace
# - Original 89565ca0 used colored rectilinear wall grids. A color's value was the
#   number of enclosed background rooms after reconstructing damaged walls.
# - In this combo, every real object is instead one color plus k concentric 1px-thick
#   square borders. The value is the nesting depth k itself.
# - D4/0 removes the scattered pad/noise color from the input entirely, so the scene
#   only contains black background and the real colored objects.
# - D9/0 removes overlap, so equal-count objects can no longer be disambiguated by
#   cover order. The visible replacement rule is lexicographic object position:
#   top-to-bottom, then left-to-right.
# - Small trace:
#   original: a damaged green wall grid might be reconstructed into 4 enclosed rooms,
#   therefore the output row is four green cells and one pad color fills the suffix.
#   combo: a clean green object with shell insets [0, 2, 4, 6] has value 4 directly,
#   therefore the output row is four green cells and black fills the suffix.
#   difference: the counted entity changes from hidden room connectivity to directly
#   visible shell depth; the padding source changes from a separate noise color to the
#   ordinary black background; tie handling changes from occlusion order to position.
#
# b) Complexity calibration
# - The scene remains richer than a trivial count-the-squares puzzle because each
#   object varies in depth and overall side length while every shell stays visually
#   separated by a black moat.
# - Ties are still allowed, but only one tie pair is introduced per sample so the
#   positional tiebreak remains easy to infer from 2-4 examples.
# - Distinct colors are preserved per object, so the output keeps the original "same
#   color becomes same bar color" mapping without adding an arbitrary recoloring step.
#
# c) Implicit-assumption audit of the original main()
# - Original _reconstruct_walls() assumed long dense horizontal/vertical evidence lines.
#   That fails because nested squares do not form spanning grid lines or room partitions.
# - Original _count_enclosed_regions() assumed value is derived from interior background
#   components. That assumption is now false; the value is encoded directly by shell
#   count, so solver logic must operate on square-border insets instead.
# - Original logic used "the only color with zero rooms is the noise color". D4/0 breaks
#   that assumption because no extra input color exists anymore. The combo solver must
#   treat every non-black color as a real object and use black for output padding.
# - Original tie handling depended on overlap visibility. D9/0 removes the possibility of
#   overlap, so a new human-visible ordering cue is required; we use bbox anchor order.
#
# d) Placement legality
# - Generator must keep every nested-square bbox strictly inside the grid and ensure that
#   object bboxes neither overlap nor touch. A one-cell black moat is enforced between
#   bboxes so objects stay visually separable.
# - Colors are unique across objects, so one color always corresponds to one bbox.
#
# e) Visual observability
# - A human looking at examples can directly count nested shells and see that output bar
#   length equals that count. No hidden repair step remains.
# - For tied depths, the solver reads objects by visible position. Because bboxes are
#   disjoint and fully visible, the tie rule is recoverable from examples.
#
# f) Visual cleanliness
# - With no noise and no overlap, the grid is intentionally sparse and readable. Thin
#   shells are separated by black space both within each object and between objects.
#
# g) Change inferability
# - No arbitrary color mapping is introduced. The only learned relation is:
#   "same object color -> same output row color, row length = shell depth".
# - Black padding is inferable because black is already the background color visible
#   throughout the input, not a hidden numeric convention.
#
# h) Quantitative anti-degeneration target
# - Samples should exhibit nested-shell depth as the dominant signal: all non-black
#   colors must have layer_count >= 1, every pair of object bboxes must have a positive
#   gap, and no non-black pixel may appear outside the constructed shell masks.
# - Ties should occur often enough to exercise the positional tiebreak, but not so often
#   that every sample degenerates into a pure reading-order problem.
#
# i) New main() assumptions that generator must satisfy
# - Each non-black color occupies exactly one square bbox.
# - Within that bbox, every true shell is a complete 1px square border.
# - Different objects never overlap and always have distinct top-left anchors.
# - Background is black and also serves as the output padding color.


_FRAME_COUNT_RANGE = (3, 5)
_LAYER_COUNT_RANGE = (1, 6)
_MAX_LAYERS_BY_FRAMES = {5: 4, 4: 4, 3: 5}
_GRID_SIZE_BY_FRAMES = {
    5: (28, 30, 28, 30, 15),
    4: (26, 30, 26, 30, 17),
    3: (24, 30, 24, 30, 19),
}
_INNER_SIDE_RANGE = (3, 7)
_LAYER_GAP_RANGE = (2, 2)
_MIN_OBJECT_GAP = 1
_BUILD_RETRIES = 80
_PLACEMENT_RETRIES = 240
_GEN_RETRIES = 400
_tie_branch_counter = 0


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

    for _ in range(_BUILD_RETRIES):
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


def _detect_shell_insets(mask: np.ndarray) -> tuple[list[int], np.ndarray]:
    b = bbox(mask)
    if b is None:
        return [], np.zeros((0, 0), dtype=bool)

    cropped = crop_bbox(mask, b).astype(bool)
    h, w = cropped.shape
    if h != w or h < 3:
        return [], cropped

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

    return insets, cropped


def _layer_count_for_color(g: np.ndarray, color: int) -> int:
    insets, _cropped = _detect_shell_insets(g == int(color))
    return int(len(insets))


def _bbox_anchor_for_color(g: np.ndarray, color: int) -> tuple[int, int]:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, _r1, _c1 = b
    return int(r0), int(c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored nested-square object into a horizontal bar whose length is its nesting depth.

    Input structure:
    - Black background with several non-overlapping monochrome drawings.
    - Each drawing is a stack of one or more concentric 1px square borders of a single color.
    - There is no extra noise color and no inter-object occlusion, so every shell is fully visible.

    Rule:
    - For each non-black color, take its tight bbox and detect which insets form complete square borders.
    - The number of complete borders is that object's count value.
    - Produce one output row per object, using the same color for the first count cells and black for the remaining cells up to the maximum count.
    - Sort rows by count ascending; ties are broken by visible position, top-to-bottom then left-to-right.

    Key insight:
    - The important quantity is encoded directly by concentric square depth rather than by room connectivity. Once noise and overlap are removed, the only secondary reasoning needed is the spatial order of equally deep objects.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    objects: list[tuple[int, int, int, int]] = []
    for color in colors_present(g, exclude={bg}):
        layers = _layer_count_for_color(g, int(color))
        assert layers > 0
        r0, c0 = _bbox_anchor_for_color(g, int(color))
        objects.append((int(layers), int(r0), int(c0), int(color)))

    assert objects

    ordered = sorted(objects)
    width = max(layers for layers, _r0, _c0, _color in ordered)
    out = np.full((len(ordered), width), bg, dtype=int)
    for r, (layers, _r0, _c0, color) in enumerate(ordered):
        out[r, :layers] = int(color)
    return out


def _next_prefer_tie() -> bool:
    global _tie_branch_counter
    prefer_tie = bool(_tie_branch_counter % 2 == 0)
    _tie_branch_counter += 1
    return prefer_tie


def _choose_layer_counts(frame_count: int, prefer_tie: bool) -> list[int] | None:
    max_layers = _MAX_LAYERS_BY_FRAMES.get(frame_count, _LAYER_COUNT_RANGE[1])
    available = list(range(_LAYER_COUNT_RANGE[0], max_layers + 1))
    allow_tie = frame_count >= 3 and (prefer_tie or frame_count > len(available))
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, min(distinct_needed, len(available)))
        if len(chosen) < distinct_needed:
            return None
        counts = chosen + [random.choice(chosen)]
    else:
        if frame_count > len(available):
            return None
        counts = random.sample(available, frame_count)
    random.shuffle(counts)
    return [int(v) for v in counts]


def _bbox_from_position(position: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    r0, c0 = position
    return int(r0), int(c0), int(r0 + side - 1), int(c0 + side - 1)


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        int(a[2]) + int(gap) < int(b[0])
        or int(b[2]) + int(gap) < int(a[0])
        or int(a[3]) + int(gap) < int(b[1])
        or int(b[3]) + int(gap) < int(a[1])
    )


def _place_frames(
    frames: list[tuple[int, int, np.ndarray]],
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = [(-1, -1)] * len(frames)
    placed_bboxes: list[tuple[int, int, int, int]] = []

    for idx in sorted(range(len(frames)), key=lambda i: -frames[i][2].shape[0]):
        side = int(frames[idx][2].shape[0])
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            candidate = (
                random.randint(0, grid_h - side),
                random.randint(0, grid_w - side),
            )
            candidate_bbox = _bbox_from_position(candidate, side)
            if all(_bboxes_separated(candidate_bbox, other_bbox, _MIN_OBJECT_GAP) for other_bbox in placed_bboxes):
                positions[idx] = candidate
                placed_bboxes.append(candidate_bbox)
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
    for idx, (color, _layers, shell_arr) in enumerate(frames):
        r0, c0 = positions[idx]
        rs, cs = np.where(shell_arr)
        grid[rs + r0, cs + c0] = int(color)
    return grid


def _expected_output(
    frame_colors: list[int],
    layer_counts: list[int],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    info = sorted(
        (int(layer_counts[i]), int(positions[i][0]), int(positions[i][1]), int(frame_colors[i]))
        for i in range(len(frame_colors))
    )
    max_layers = max(count for count, _r0, _c0, _color in info)
    expected = np.zeros((len(info), max_layers), dtype=int)
    for r, (count, _r0, _c0, color) in enumerate(info):
        expected[r, :count] = int(color)
    return expected


def _gen_attempt(prefer_tie: bool) -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    layer_counts = _choose_layer_counts(frame_count, prefer_tie)
    if layer_counts is None:
        return None

    chosen_colors = random.sample(list(range(1, 10)), frame_count)
    frame_colors = [int(c) for c in chosen_colors]

    h_min, h_max, w_min, w_max, frame_max_side = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, layers in zip(frame_colors, layer_counts):
        build_max_side = min(int(frame_max_side), int(grid_h - 2), int(grid_w - 2))
        shell_mask = _build_nested_frame(int(layers), build_max_side)
        if shell_mask is None:
            return None
        frames.append((int(color), int(layers), shell_mask))

    positions = _place_frames(frames, grid_h, grid_w)
    if positions is None:
        return None

    grid = _render_frames(frames, positions, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(frame_colors, layer_counts, positions)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    prefer_tie = _next_prefer_tie()
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt(prefer_tie)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
