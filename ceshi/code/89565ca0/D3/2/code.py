# Variant of 89565ca0
# Dimension: D3 — 源对象：彩色线框架 (src)
# Change: 轴对齐的1像素厚彩色线绘制的分区框架，包含水平和垂直线段形成网格状房间结构... → 嵌套彩色方框

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg


# Design analysis for the D3 variant
# ----------------------------------
# a) Concrete object model and behavior trace
# - Original D3 objects were rectilinear room-partition frames. Their count value was
#   the number of enclosed background rooms after reconstructing damaged grid walls.
# - New D3 objects are monochrome sets of concentric square borders. A real object is
#   one color plus k nested 1px-thick square outlines that share the same center and
#   move inward by 1-2 cells per layer. The object's count value is exactly k.
# - Input still contains several colored objects on black background, one scattered pad
#   color, and some inter-object overlaps. Output still converts each real object into
#   one horizontal color bar whose length equals the object's count, sorted ascending.
# - Small trace:
#   original: a color with a 2x3 wall grid could enclose 4 rooms, so solver rebuilt
#   horizontal/vertical walls and counted enclosed regions = 4.
#   variant: a color with four concentric square borders has no internal partitions;
#   solver instead inspects its square bbox and detects shell insets [0, 3, 5, 8],
#   therefore layer count = 4.
#   difference: the counted entity changes from "background rooms produced by walls"
#   to "square-border layers directly visible as nested shells".
#
# b) Complexity calibration
# - The new objects stay visually richer than a single square because each object can
#   have 1-6 layers with varying shell gaps and varying overall size.
# - The task is still not over-complicated: every real object is defined by only one
#   integer, the layer count. Ties are still limited to one pair, matching the original
#   tiebreaker burden and keeping cover-order inference readable from examples.
#
# c) Implicit-assumption audit of the original main()
# - Original _reconstruct_walls() assumed dense full-width/full-height evidence lines.
#   That fails here because nested squares create only four perimeter segments per layer,
#   not a grid of spanning vertical/horizontal lines.
# - Original _count_enclosed_regions() counted background components inside walls.
#   That value could accidentally correlate with nested-shell depth, but the new solver
#   must not rely on the old grid-wall reconstruction path because the source object is
#   no longer a room partition frame.
# - Original "rooms == 0 means noise color" remains valid after replacement, provided
#   generator ensures only the true object colors yield square-shell evidence.
# - Original _cover_order() only assumed reconstructed masks overlap and the top visible
#   color is unique on overlap cells. That remains valid, so the tiebreaking mechanism
#   is preserved unchanged.
#
# d) Placement legality
# - Each object's true bbox is a square and must remain visible as a square after all
#   overlaps/noise. Generator therefore protects shell corners and side midpoints from
#   occlusion, restricts overlap amount, and only scatters pad pixels on non-frame cells.
# - Equal-count objects are forced to overlap on non-protected shell pixels so cover
#   order is observable and main() can order the tie group deterministically.
#
# e) Visual observability
# - From 2-4 examples, a human can see that some colors appear as one square, others as
#   stacks of 2-6 concentric same-color squares, and the output bar length matches that
#   depth. This is directly visible, unlike the original hidden "room count" criterion.
# - Sorting never depends on raw color ids. The only ordering cue is visible shell count,
#   with overlaps breaking ties exactly as in the original task family.
#
# f) Visual cleanliness
# - Noise is kept off frame pixels. This avoids turning thin shells into unreadable
#   clutter while still preserving the original presence of one scattered non-object
#   color. Overlaps are limited to a small fraction of shell pixels to keep objects
#   distinguishable.
#
# g) Color-change inferability
# - No color remapping is introduced. Object colors persist directly into output bars,
#   so the only inferred relation is "same color -> bar of that color, length = depth".
#
# h) Quantitative anti-degeneration target
# - The new hallmark metric is "nested layer count per real color", not enclosed rooms.
#   Generator explicitly constructs each real color from multiple concentric square
#   borders; this makes the D3 change visible even when the final output shape format is
#   kept from the original task.
#
# i) New main() assumptions that generator must satisfy
# - For each real color, the visible bbox remains square.
# - Every true shell keeps all four corners visible.
# - Every true shell keeps enough side pixels for threshold-based perimeter detection.
# - At most one tie pair exists, and tied objects overlap on shell pixels.


_FRAME_COUNT_RANGE = (3, 4)
_LAYER_COUNT_RANGE = (1, 6)
_MAX_LAYERS_BY_FRAMES = {5: 4, 4: 5, 3: 6}
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    4: (26, 30, 26, 30, 24),
    3: (22, 30, 22, 30, 26),
}
_INNER_SIDE_RANGE = (3, 7)
_LAYER_GAP_RANGE = (1, 2)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_EXTRA_OVERLAP_RANGE = (0, 1)
_OVERLAP_RETRY = 160
_GEN_RETRIES = 300

_SHELL_HIT_FRACTION = 0.5
_SMALL_SIDE_PROTECT_LIMIT = 4
_MAX_OVERLAP_SHARE = 0.2


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


def _mark_protected_points(protected: np.ndarray, inset: int) -> None:
    n = int(protected.shape[0])
    r0 = int(inset)
    c0 = int(inset)
    r1 = int(n - 1 - inset)
    c1 = int(n - 1 - inset)
    side = int(r1 - r0 + 1)
    if side <= _SMALL_SIDE_PROTECT_LIMIT:
        _paint_square_border(protected, inset)
        return

    mid = side // 2
    points = {
        (r0, c0),
        (r0, c1),
        (r1, c0),
        (r1, c1),
        (r0, c0 + mid),
        (r1, c0 + mid),
        (r0 + mid, c0),
        (r0 + mid, c1),
    }
    for r, c in points:
        protected[int(r), int(c)] = True


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
        top = int(cropped[r0, c0 : c1 + 1].sum())
        bottom = int(cropped[r1, c0 : c1 + 1].sum())
        left = int(cropped[r0 : r1 + 1, c0].sum())
        right = int(cropped[r0 : r1 + 1, c1].sum())
        threshold = max(3, math.ceil(_SHELL_HIT_FRACTION * side))
        corners_visible = (
            bool(cropped[r0, c0])
            and bool(cropped[r0, c1])
            and bool(cropped[r1, c0])
            and bool(cropped[r1, c1])
        )
        if corners_visible and min(top, bottom, left, right) >= threshold:
            insets.append(int(inset))

    return insets, cropped


def _layer_count_for_color(g: np.ndarray, color: int) -> int:
    insets, _cropped = _detect_shell_insets(g == int(color))
    return int(len(insets))


def _shells_for_color(g: np.ndarray, color: int) -> np.ndarray:
    mask = g == int(color)
    b = bbox(mask)
    out = np.zeros_like(g, dtype=bool)
    if b is None:
        return out

    insets, cropped = _detect_shell_insets(mask)
    if not insets:
        return out

    shell_mask = np.zeros_like(cropped, dtype=bool)
    for inset in insets:
        _paint_square_border(shell_mask, int(inset))

    r0, c0, r1, c1 = b
    out[r0 : r1 + 1, c0 : c1 + 1] = shell_mask
    return out


def _cover_order(g: np.ndarray, group: list[tuple[int, int, np.ndarray]]) -> list[tuple[int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _layers, _shells in group}
    indegree: dict[int, int] = {color: 0 for color, _layers, _shells in group}

    for i, (ca, _la, sa) in enumerate(group):
        for cb, _lb, sb in group[i + 1 :]:
            overlap = sa & sb
            if not bool(overlap.any()):
                continue
            visible = set(int(v) for v in np.unique(g[overlap]))
            top_colors = visible & {int(ca), int(cb)}
            assert len(top_colors) == 1
            top = top_colors.pop()
            under = int(cb) if top == int(ca) else int(ca)
            if top not in above[under]:
                above[under].add(top)
                indegree[top] += 1

    ordered: list[tuple[int, int, np.ndarray]] = []
    remaining = set(by_color)
    while remaining:
        ready = [color for color in remaining if indegree[color] == 0]
        assert len(ready) == 1
        color = ready[0]
        remaining.remove(color)
        ordered.append(by_color[color])
        for nxt in above[color]:
            indegree[nxt] -= 1
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored nested-square object into a horizontal bar whose length is its nesting depth.

    Input structure:
    - Black background with several colored drawings, each made of one or more concentric square borders.
    - The drawings may overlap each other, and one extra non-black color appears only as scattered pad noise.
    - Every real drawing uses exactly one color and its visible bbox remains a square.

    Rule:
    - For each non-background color, inspect its visible square bbox and detect the insets whose four sides still show a square border.
    - The number of detected square shells is that color's count value.
    - The color with zero detected shells is the pad color.
    - Sort real drawing colors by shell count; equal counts are ordered by which reconstructed shell set lies underneath the other.

    Key insight:
    - The meaningful quantity is no longer the number of rooms inside a line grid. Each true object directly encodes its value as the number of nested same-color square borders, while the scattered pad color never forms a coherent square shell.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        layers = _layer_count_for_color(g, int(color))
        shells = _shells_for_color(g, int(color))
        counts.append((int(color), int(layers), shells))

    drawings = [(color, layers, shells) for color, layers, shells in counts if layers > 0]
    pad_colors = [color for color, layers, _shells in counts if layers == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, np.ndarray]] = []
    for layers in sorted({layers for _color, layers, _shells in drawings}):
        group = [item for item in drawings if item[1] == layers]
        ordered.extend(_cover_order(g, group))

    width = max(layers for _color, layers, _shells in ordered)
    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, layers, _shells) in enumerate(ordered):
        out[r, :layers] = int(color)
    return out


def _choose_layer_counts(frame_count: int) -> list[int] | None:
    max_layers = _MAX_LAYERS_BY_FRAMES.get(frame_count, _LAYER_COUNT_RANGE[1])
    available = list(range(_LAYER_COUNT_RANGE[0], max_layers + 1))
    allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
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


def _build_nested_frame(target_layers: int, max_side: int) -> tuple[np.ndarray, np.ndarray] | None:
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
        protected = np.zeros((side, side), dtype=bool)
        inset = 0
        for layer_idx in range(target_layers):
            if layer_idx > 0:
                inset += gaps[layer_idx - 1]
            _paint_square_border(mask, int(inset))
            _mark_protected_points(protected, int(inset))
        return mask, protected
    return None


def _pair_overlap_stats(
    frame_a: tuple[int, int, np.ndarray, np.ndarray],
    pos_a: tuple[int, int],
    frame_b: tuple[int, int, np.ndarray, np.ndarray],
    pos_b: tuple[int, int],
) -> tuple[bool, int, bool]:
    mask_a = frame_a[2]
    prot_a = frame_a[3]
    mask_b = frame_b[2]
    prot_b = frame_b[3]
    ra, ca = pos_a
    rb, cb = pos_b
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape

    r0 = max(int(ra), int(rb))
    c0 = max(int(ca), int(cb))
    r1 = min(int(ra + ha - 1), int(rb + hb - 1))
    c1 = min(int(ca + wa - 1), int(cb + wb - 1))
    if r0 > r1 or c0 > c1:
        return False, 0, False

    a_rs = slice(int(r0 - ra), int(r1 - ra + 1))
    a_cs = slice(int(c0 - ca), int(c1 - ca + 1))
    b_rs = slice(int(r0 - rb), int(r1 - rb + 1))
    b_cs = slice(int(c0 - cb), int(c1 - cb + 1))

    overlap = mask_a[a_rs, a_cs] & mask_b[b_rs, b_cs]
    if not bool(overlap.any()):
        return False, 0, False

    protected_hit = bool((prot_a[a_rs, a_cs] & mask_b[b_rs, b_cs]).any() or (prot_b[b_rs, b_cs] & mask_a[a_rs, a_cs]).any())
    return True, int(overlap.sum()), protected_hit


def _is_valid_overlap(
    frame_a: tuple[int, int, np.ndarray, np.ndarray],
    pos_a: tuple[int, int],
    frame_b: tuple[int, int, np.ndarray, np.ndarray],
    pos_b: tuple[int, int],
) -> bool:
    has_overlap, overlap_pixels, protected_hit = _pair_overlap_stats(frame_a, pos_a, frame_b, pos_b)
    if not has_overlap or protected_hit:
        return False

    shell_a = int(frame_a[2].sum())
    shell_b = int(frame_b[2].sum())
    allowed = max(1, math.floor(_MAX_OVERLAP_SHARE * min(shell_a, shell_b)))
    return overlap_pixels <= allowed


def _position_compatible_with_others(
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    positions: list[tuple[int, int]],
    idx: int,
    candidate: tuple[int, int],
) -> bool:
    for other_idx, other_pos in enumerate(positions):
        if other_idx == idx or other_pos[0] < 0:
            continue
        if not _is_valid_overlap(frames[idx], candidate, frames[other_idx], other_pos):
            has_overlap, _overlap_pixels, _protected_hit = _pair_overlap_stats(frames[idx], candidate, frames[other_idx], other_pos)
            if has_overlap:
                return False
    return True


def _initial_positions(
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = [(-1, -1)] * len(frames)
    order = sorted(range(len(frames)), key=lambda i: -frames[i][2].shape[0])
    for idx in order:
        side = int(frames[idx][2].shape[0])
        placed = False
        for _ in range(120):
            cand = (
                random.randint(0, grid_h - side),
                random.randint(0, grid_w - side),
            )
            if _position_compatible_with_others(frames, positions, idx, cand):
                positions[idx] = cand
                placed = True
                break
        if not placed:
            return None
    return positions


def _force_overlap(
    positions: list[tuple[int, int]],
    i: int,
    j: int,
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    grid_h: int,
    grid_w: int,
) -> bool:
    if _is_valid_overlap(frames[i], positions[i], frames[j], positions[j]) and _position_compatible_with_others(frames, positions, j, positions[j]):
        return True

    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    ri, ci = positions[i]

    r0_min = max(0, int(ri - fj_h + 1))
    r0_max = min(int(grid_h - fj_h), int(ri + fi_h - 1))
    c0_min = max(0, int(ci - fj_w + 1))
    c0_max = min(int(grid_w - fj_w), int(ci + fi_w - 1))
    if r0_min > r0_max or c0_min > c0_max:
        return False

    for _ in range(_OVERLAP_RETRY):
        rj = random.randint(r0_min, r0_max)
        cj = random.randint(c0_min, c0_max)
        if _is_valid_overlap(frames[i], positions[i], frames[j], (rj, cj)) and _position_compatible_with_others(frames, positions, j, (rj, cj)):
            positions[j] = (int(rj), int(cj))
            return True
    return False


def _render_frames(
    frames: list[tuple[int, int, np.ndarray, np.ndarray]],
    positions: list[tuple[int, int]],
    render_order: list[int],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx in render_order:
        r0, c0 = positions[idx]
        shell_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(shell_arr)
        grid[rs + r0, cs + c0] = int(color)

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        r0, c0 = positions[idx]
        shell_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(shell_arr)
        abs_rs = rs + r0
        abs_cs = cs + c0
        visible = grid[abs_rs, abs_cs] == int(color)
        frame_mask[abs_rs[visible], abs_cs[visible]] = True

    return grid, frame_mask


def _scatter_noise(grid: np.ndarray, frame_mask: np.ndarray, noise_color: int) -> None:
    total_cells = int(grid.size)
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)
    nonframe_rs, nonframe_cs = np.where(~frame_mask)
    if len(nonframe_rs) == 0:
        return
    chosen = np.random.permutation(len(nonframe_rs))[: min(noise_count, len(nonframe_rs))]
    grid[nonframe_rs[chosen], nonframe_cs[chosen]] = int(noise_color)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    layer_counts = _choose_layer_counts(frame_count)
    if layer_counts is None:
        return None

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    h_min, h_max, w_min, w_max, frame_max_side = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for color, layers in zip(frame_colors, layer_counts):
        build_max_side = min(int(frame_max_side), int(grid_h - 2), int(grid_w - 2))
        built = _build_nested_frame(int(layers), build_max_side)
        if built is None:
            return None
        shell_mask, protected = built
        frames.append((int(color), int(layers), shell_mask, protected))

    positions = _initial_positions(frames, grid_h, grid_w)
    if positions is None:
        return None

    tied_pairs = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count) if layer_counts[i] == layer_counts[j]]
    for i, j in tied_pairs:
        if not _force_overlap(positions, i, j, frames, grid_h, grid_w):
            return None

    non_tied = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count) if layer_counts[i] != layer_counts[j]]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        placed = 0
        for i, j in non_tied:
            if _force_overlap(positions, i, j, frames, grid_h, grid_w):
                placed += 1
                if placed >= n_extra:
                    break

    render_order = list(range(frame_count))
    random.shuffle(render_order)
    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color)

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted((layer_counts[i], render_order.index(i), frame_colors[i]) for i in range(frame_count))
    max_layers = max(count for count, _ord, _color in frame_info)
    expected = np.full((frame_count, max_layers), noise_color, dtype=int)
    for r, (count, _ord, color) in enumerate(frame_info):
        expected[r, :count] = int(color)

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
