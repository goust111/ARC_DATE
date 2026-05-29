# Combo variant of 89565ca0 (index 34)
# combo_spec: D1/1+D4/1
# Name: 像素总数排序+多色噪声

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# variant:
#   core_change:
#     Two original assumptions are replaced together:
#     1. Rows are no longer ordered by enclosed-room count. Each real frame still
#        becomes a bar whose length equals its room count, but row order is now
#        determined by the number of currently visible pixels of that frame color
#        in the input grid, ascending.
#     2. Noise is no longer a single distractor color. There are now 2-3 noise
#        colors, all reconstructing to zero rooms; the dominant noise color by
#        visible pixel count becomes the output padding color.
#   concrete_behavior:
#     Small trace:
#       - Suppose the real frames are colors A/B/C with reconstructed room counts
#         4/2/3 and visible pixel counts 31/47/38 after overlap and wall damage.
#       - Noise colors D/E are scattered; both reconstruct to 0 rooms, and D is
#         more frequent than E.
#       - Output rows are ordered A, C, B? No: order is A(31), C(38), B(47),
#         while row lengths remain 4, 3, 2 respectively; trailing cells use D.
#     Relative to the source task, the bars still encode room counts, but their
#     vertical order now follows a different observable statistic.
#   complexity_calibration:
#     The generator keeps 3-5 frames and 2-3 noise colors so the scene remains
#     readable, but it rejects samples where visible frame-pixel totals tie or
#     where visible-pixel order fails to reverse at least one pair of frames with
#     different room counts. This makes the D1 change visible in examples rather
#     than cosmetic or confined to same-length output rows.
#   implicit_assumptions_review:
#     Original `main()` assumed exactly one zero-room color and that sorting by
#     rooms plus overlap tiebreak was sufficient. Neither remains true.
#     The revised solver now assumes:
#       1. Every true frame still reconstructs to `rooms > 0`.
#       2. Every noise color reconstructs to `rooms == 0`.
#       3. The dominant noise color is unique by visible count.
#       4. Real frames have pairwise distinct visible pixel totals.
#     The generator explicitly validates all four conditions.
#   placement_validity:
#     Frames remain axis-aligned, in-bounds, and may overlap in the same bounded
#     way as the source puzzle family. Noise is placed on both background cells
#     and a limited subset of visible wall pixels, so reconstruction thresholds
#     still hold while visible-pixel statistics become nontrivial.
#   observability_and_cleanliness:
#     A human can still isolate the rectilinear frames and count rooms, but must
#     now also estimate which frame color occupies fewer or more pixels under
#     multi-color clutter. Noise color count is capped at 3 and wall damage stays
#     moderate so the puzzle does not devolve into unreadable speckle.
#   generator_guarantees:
#     Every accepted sample satisfies all of:
#       - 2-3 zero-room noise colors are present.
#       - one noise color is uniquely dominant and is the padding color.
#       - visible frame-pixel counts are all distinct.
#       - at least one pair of different-room frames is ordered differently by
#         visible pixels than by room count.
#   new_solver_assumptions:
#     The solver's only new non-source assumption is distinct visible totals for
#     the real frames. The generator enforces this post-render, post-noise, so
#     the solver never needs a hidden tie-break such as raw color id.


def _count_enclosed_regions(walls: np.ndarray) -> int:
    labels, n_labels = label_mask(~walls, connectivity=4)
    h, w = walls.shape
    count = 0
    for k in range(1, n_labels + 1):
        coords = np.argwhere(labels == k)
        assert coords.size > 0
        touches_edge = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        if not touches_edge:
            count += 1
    return int(count)


def _reconstruct_walls(mask: np.ndarray) -> np.ndarray:
    h, _w = mask.shape
    col_counts = mask.sum(axis=0)
    v_threshold = max(3, math.ceil(0.45 * h))
    vcols = [int(c) for c, n in enumerate(col_counts) if int(n) >= v_threshold]
    if len(vcols) < 2:
        return mask.copy()

    row_counts = mask.sum(axis=1)
    h_threshold = max(3, len(vcols) + 1)
    hrows = [int(r) for r, n in enumerate(row_counts) if int(n) >= h_threshold]
    if len(hrows) < 2:
        return mask.copy()

    walls = np.zeros_like(mask, dtype=bool)

    for r in hrows:
        for c0, c1 in zip(vcols, vcols[1:]):
            segment = mask[r, c0 : c1 + 1]
            length = int(c1 - c0 + 1)
            hits = int(segment.sum())
            dense = hits >= max(2, math.ceil(0.45 * length))
            interior_evidence = length > 2 and hits >= 2 and bool(segment[1:-1].any())
            if dense or interior_evidence:
                walls[r, c0 : c1 + 1] = True

    for c in vcols:
        for r0, r1 in zip(hrows, hrows[1:]):
            segment = mask[r0 : r1 + 1, c]
            length = int(r1 - r0 + 1)
            hits = int(segment.sum())
            if hits >= max(2, math.ceil(0.45 * length)):
                walls[r0 : r1 + 1, c] = True

    return walls


def _room_count_for_color(g: np.ndarray, color: int) -> int:
    b = bbox(g == int(color))
    assert b is not None
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    return _count_enclosed_regions(walls)


def _visible_pixels_for_color(g: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(g == int(color)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partitioned frame into a horizontal bar whose length is its room count, ordered by visible wall-pixel totals.

    Input structure:
    - Black background with several colored axis-aligned line frames that partition their interiors into one or more rooms.
    - The scene also contains 2-3 scattered noise colors; these speckles appear both on open background cells and on top of some visible frame-wall cells.
    - Frames may overlap, so the number of pixels still showing a frame's color is not just a function of its geometry.

    Rule:
    - For every non-background color, reconstruct its rectilinear wall layout from dense row and column evidence, then count enclosed 4-connected regions.
    - Colors with positive room counts are the real frames; colors with zero rooms are noise colors.
    - Among the noise colors, choose the uniquely most frequent visible one as the padding color.
    - For each real frame, keep its bar length equal to its room count, but sort the rows by that frame color's visible pixel total in the input, from smallest to largest.

    Key insight:
    - Room counting still identifies which colors are genuine frames, but room count no longer determines row order. The new task is to separate frames from several distractor colors and then rank only the true frame colors by how many pixels of each remain visible after overlap and multi-color damage.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    frame_info: list[tuple[int, int, int]] = []
    noise_colors: list[int] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        if rooms > 0:
            visible = _visible_pixels_for_color(g, int(color))
            frame_info.append((int(color), int(rooms), int(visible)))
        else:
            noise_colors.append(int(color))

    assert frame_info
    assert 2 <= len(noise_colors) <= 3

    noise_pixel_counts = {color: _visible_pixels_for_color(g, color) for color in noise_colors}
    pad = max(noise_colors, key=lambda color: noise_pixel_counts[color])
    assert sum(1 for color in noise_colors if noise_pixel_counts[color] == noise_pixel_counts[pad]) == 1

    visible_counts = [visible for _color, _rooms, visible in frame_info]
    assert len(set(visible_counts)) == len(visible_counts)
    ordered = sorted(frame_info, key=lambda item: item[2])

    width = max(rooms for _color, rooms, _visible in ordered)
    out = np.full((len(ordered), width), int(pad), dtype=int)
    for r, (color, rooms, _visible) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (gpt-5) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    5: (27, 30, 27, 30, 9, 11),
    4: (24, 30, 24, 30, 10, 12),
    3: (22, 30, 22, 30, 11, 13),
}
_CELL_GAP_RANGE = (3, 13)
_NOISE_FRACTION_RANGE = (0.04, 0.07)
_NOISE_COLOR_COUNT_RANGE = (2, 3)
_MIN_NOISE_PIXELS_PER_COLOR = 4
_WALL_DAMAGE_FRACTION = 0.12
_DAMAGE_SHARE_RANGE = (0.05, 0.16)
_FRAME_CANDIDATES = 6
_FRAME_CANDIDATE_RETRIES = 24
_PLACEMENT_RETRIES = 120
_GEN_RETRIES = 500
_MIN_VISIBLE_GAP = 2


def _random_gaps(n: int, min_gap: int, max_gap: int, max_total: int) -> list[int] | None:
    if n * min_gap > max_total:
        return None
    gaps = [min_gap] * n
    remaining = max_total - sum(gaps)
    for i in range(n):
        extra = min(max_gap - gaps[i], remaining)
        if extra > 0:
            add = random.randint(0, extra)
            gaps[i] += add
            remaining -= add
    random.shuffle(gaps)
    return gaps


def _build_frame(target_rooms: int, max_h: int = 22, max_w: int = 25) -> np.ndarray | None:
    if target_rooms == 1:
        gap_h = random.randint(_CELL_GAP_RANGE[0], max(_CELL_GAP_RANGE[0], max_h - 2))
        gap_w = random.randint(_CELL_GAP_RANGE[0], max(_CELL_GAP_RANGE[0], max_w - 2))
        h, w = gap_h + 2, gap_w + 2
        walls = np.zeros((h, w), dtype=bool)
        walls[0, :] = True
        walls[h - 1, :] = True
        walls[:, 0] = True
        walls[:, w - 1] = True
        return walls

    possible_configs = []
    for nv in range(2, 6):
        for nh in range(2, 5):
            cells = (nv - 1) * (nh - 1)
            if target_rooms <= cells <= target_rooms + 4:
                possible_configs.append((nv, nh, cells))
    if not possible_configs:
        for nv in range(2, 7):
            for nh in range(2, 6):
                cells = (nv - 1) * (nh - 1)
                if cells >= target_rooms:
                    possible_configs.append((nv, nh, cells))
    if not possible_configs:
        return None

    n_vcols, n_hrows, total_cells = random.choice(possible_configs)
    n_row_gaps = n_hrows - 1
    n_col_gaps = n_vcols - 1
    min_gap, max_gap = _CELL_GAP_RANGE

    if min_gap * n_row_gaps > max_h - 1 or min_gap * n_col_gaps > max_w - 1:
        return None

    row_gaps = _random_gaps(n_row_gaps, min_gap, max_gap, max_h - 1)
    col_gaps = _random_gaps(n_col_gaps, min_gap, max_gap, max_w - 1)
    if row_gaps is None or col_gaps is None:
        return None

    vcol_positions = [0]
    for gap in col_gaps:
        vcol_positions.append(vcol_positions[-1] + gap)
    hrow_positions = [0]
    for gap in row_gaps:
        hrow_positions.append(hrow_positions[-1] + gap)

    frame_h = hrow_positions[-1] + 1
    frame_w = vcol_positions[-1] + 1
    walls = np.zeros((frame_h, frame_w), dtype=bool)
    for r in hrow_positions:
        walls[r, :] = True
    for c in vcol_positions:
        walls[:, c] = True

    cells_to_merge = total_cells - target_rooms
    if cells_to_merge == 0:
        return walls

    parent = list(range(total_cells))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True

    def cell_id(cr, cc):
        return cr * (n_vcols - 1) + cc

    internal_segments = []
    for hi in range(1, n_hrows - 1):
        r = hrow_positions[hi]
        for ci in range(n_vcols - 1):
            c0, c1 = vcol_positions[ci], vcol_positions[ci + 1]
            internal_segments.append(("h", r, c0, c1, cell_id(hi - 1, ci), cell_id(hi, ci)))
    for ci in range(1, n_vcols - 1):
        c = vcol_positions[ci]
        for hi in range(n_hrows - 1):
            r0, r1 = hrow_positions[hi], hrow_positions[hi + 1]
            internal_segments.append(("v", c, r0, r1, cell_id(hi, ci - 1), cell_id(hi, ci)))

    random.shuffle(internal_segments)
    merged = 0
    for seg in internal_segments:
        if merged >= cells_to_merge:
            break
        if seg[0] == "h":
            _, r, c0, c1, ca, cb = seg
            if find(ca) != find(cb):
                union(ca, cb)
                walls[r, c0 + 1 : c1] = False
                merged += 1
        else:
            _, c, r0, r1, ca, cb = seg
            if find(ca) != find(cb):
                union(ca, cb)
                walls[r0 + 1 : r1, c] = False
                merged += 1

    if merged < cells_to_merge:
        return None
    if _count_enclosed_regions(walls) != target_rooms:
        return None
    return walls


def _build_frame_with_candidates(target_rooms: int, max_h: int, max_w: int) -> tuple[np.ndarray, int] | None:
    walls = _build_frame(target_rooms, max_h=max_h, max_w=max_w)
    if walls is None:
        return None
    return walls, int(walls.sum())


def _choose_frame_variants(room_counts: list[int], max_h: int, max_w: int) -> list[tuple[np.ndarray, int]] | None:
    candidate_sets: list[list[tuple[np.ndarray, int]]] = []
    for target_rooms in room_counts:
        candidates: list[tuple[np.ndarray, int]] = []
        seen: set[tuple[tuple[int, int], int]] = set()
        for _ in range(_FRAME_CANDIDATE_RETRIES):
            built = _build_frame_with_candidates(target_rooms, max_h=max_h, max_w=max_w)
            if built is None:
                continue
            walls, raw_pixels = built
            key = (tuple(map(int, walls.shape)), int(raw_pixels))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((walls, int(raw_pixels)))
            if len(candidates) >= _FRAME_CANDIDATES:
                break
        if not candidates:
            return None
        candidate_sets.append(candidates)

    for _ in range(240):
        chosen = [random.choice(candidates) for candidates in candidate_sets]
        raw_info = [(idx, int(room_counts[idx]), int(raw_pixels)) for idx, (_walls, raw_pixels) in enumerate(chosen)]
        ordered = sorted(raw_info, key=lambda item: item[2])
        raw_counts = [raw for _idx, _rooms, raw in ordered]
        if len(set(raw_counts)) != len(raw_counts):
            continue
        if min(b - a for a, b in zip(raw_counts, raw_counts[1:])) < (_MIN_VISIBLE_GAP + 2):
            continue
        inversion_probe = [(idx, rooms, raw) for idx, rooms, raw in ordered]
        if _count_cross_room_inversions(inversion_probe) == 0:
            continue
        return chosen
    return None


def _place_frames_sparsely(
    frames: list[tuple[int, int, np.ndarray, int]],
    grid_h: int,
    grid_w: int,
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(frames)
    order = sorted(range(len(frames)), key=lambda i: -(frames[i][2].shape[0] * frames[i][2].shape[1]))
    for idx in order:
        fh, fw = frames[idx][2].shape
        placed = None
        for _ in range(_PLACEMENT_RETRIES):
            r0 = random.randint(0, grid_h - fh)
            c0 = random.randint(0, grid_w - fw)
            ok = True
            for j, pos in enumerate(positions):
                if pos is None:
                    continue
                r1, c1 = pos
                oh, ow = frames[j][2].shape
                separated_rows = r0 + fh + 1 <= r1 or r1 + oh + 1 <= r0
                separated_cols = c0 + fw + 1 <= c1 or c1 + ow + 1 <= c0
                if not (separated_rows or separated_cols):
                    ok = False
                    break
            if ok:
                placed = (r0, c0)
                break
        if placed is None:
            return None
        positions[idx] = placed
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _force_overlap(positions, i, j, frames, grid_h, grid_w):
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    ri, ci = positions[i]
    r0_min = max(0, ri - fj_h + 2)
    r0_max = min(grid_h - fj_h, ri + fi_h - 2)
    c0_min = max(0, ci - fj_w + 2)
    c0_max = min(grid_w - fj_w, ci + fi_w - 2)
    if r0_min > r0_max or c0_min > c0_max:
        return
    for _ in range(_OVERLAP_RETRY):
        rj = random.randint(r0_min, r0_max)
        cj = random.randint(c0_min, c0_max)
        if (
            max(ri, rj) <= min(ri + fi_h - 1, rj + fj_h - 1)
            and max(ci, cj) <= min(ci + fi_w - 1, cj + fj_w - 1)
        ):
            positions[j] = (rj, cj)
            return


def _absolute_wall_masks(frames, positions, grid_h, grid_w) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for idx in range(len(frames)):
        mask = np.zeros((grid_h, grid_w), dtype=bool)
        r0, c0 = positions[idx]
        wall_arr = frames[idx][2]
        rs, cs = np.where(wall_arr)
        mask[rs + r0, cs + c0] = True
        masks.append(mask)
    return masks


def _bbox_union_mask(frames, positions, grid_h, grid_w) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx, (_color, _rooms, wall_arr, _raw_pixels) in enumerate(frames):
        r0, c0 = positions[idx]
        h, w = wall_arr.shape
        mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _render_frames(frames, positions, render_order, grid_h, grid_w):
    grid = np.zeros((grid_h, grid_w), dtype=int)
    wall_masks = _absolute_wall_masks(frames, positions, grid_h, grid_w)

    for idx in render_order:
        color = frames[idx][0]
        grid[wall_masks[idx]] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        color = frames[idx][0]
        frame_mask |= wall_masks[idx] & (grid == int(color))

    bbox_mask = _bbox_union_mask(frames, positions, grid_h, grid_w)
    return grid, frame_mask, bbox_mask


def _noise_color_counts(total_noise: int, noise_colors: list[int]) -> dict[int, int] | None:
    n = len(noise_colors)
    if total_noise < n * _MIN_NOISE_PIXELS_PER_COLOR:
        return None

    for _ in range(100):
        if n == 2:
            dom_min = max(_MIN_NOISE_PIXELS_PER_COLOR + 2, math.ceil(0.55 * total_noise))
            dom_max = total_noise - _MIN_NOISE_PIXELS_PER_COLOR
        else:
            dom_min = max(_MIN_NOISE_PIXELS_PER_COLOR + 2, math.ceil(0.45 * total_noise))
            dom_max = total_noise - 2 * _MIN_NOISE_PIXELS_PER_COLOR
        if dom_min > dom_max:
            continue
        dominant = random.randint(dom_min, dom_max)
        remaining = total_noise - dominant
        others = [_MIN_NOISE_PIXELS_PER_COLOR] * (n - 1)
        remaining_extra = remaining - (n - 1) * _MIN_NOISE_PIXELS_PER_COLOR
        for _ in range(remaining_extra):
            others[random.randrange(n - 1)] += 1
        random.shuffle(others)
        if dominant <= max(others):
            continue
        counts = [dominant, *others]
        return {int(color): int(count) for color, count in zip(noise_colors, counts)}
    return None


def _sample_coords(coords: np.ndarray, count: int) -> np.ndarray:
    assert count <= len(coords)
    if count == 0:
        return np.empty((0, 2), dtype=int)
    perm = np.random.permutation(len(coords))[:count]
    return coords[perm]


def _scatter_noise(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    bbox_mask: np.ndarray,
    noise_colors: list[int],
):
    grid_h, grid_w = grid.shape
    total_cells = grid_h * grid_w
    total_noise = int(total_cells * random.uniform(*_NOISE_FRACTION_RANGE))
    counts_by_color = _noise_color_counts(total_noise, noise_colors)
    if counts_by_color is None:
        raise AssertionError("Noise budget too small for multiple visible colors")

    damage_pool = np.argwhere(frame_mask)
    nonwall_pool = np.argwhere(~frame_mask)
    inside_nonwall = np.argwhere(bbox_mask & ~frame_mask)
    outside_nonwall = np.argwhere(~bbox_mask & ~frame_mask)

    max_damage = min(
        total_noise - len(noise_colors),
        int(_WALL_DAMAGE_FRACTION * len(damage_pool)),
        int(total_noise * random.uniform(*_DAMAGE_SHARE_RANGE)),
    )
    damage_count = min(max(0, max_damage), len(damage_pool))
    background_count = total_noise - damage_count
    assert background_count >= len(noise_colors)
    assert background_count <= len(nonwall_pool)

    chosen_background: list[np.ndarray] = []
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    if len(inside_nonwall) > 0:
        pick = _sample_coords(inside_nonwall, 1)
        chosen_background.append(pick)
        reserved[pick[:, 0], pick[:, 1]] = True
    if len(outside_nonwall) > 0:
        pick = _sample_coords(outside_nonwall, 1)
        if not reserved[pick[:, 0], pick[:, 1]].all():
            chosen_background.append(pick)
            reserved[pick[:, 0], pick[:, 1]] = True

    remaining_background = background_count - sum(len(chunk) for chunk in chosen_background)
    remaining_nonwall = np.argwhere((~frame_mask) & (~reserved))
    chosen_background.append(_sample_coords(remaining_nonwall, remaining_background))
    background_coords = np.concatenate(chosen_background, axis=0)
    assert len(background_coords) == background_count

    damage_coords = _sample_coords(damage_pool, damage_count)
    all_coords = np.concatenate([background_coords, damage_coords], axis=0)
    all_coords = all_coords[np.random.permutation(len(all_coords))]

    color_sequence: list[int] = []
    for color in noise_colors:
        color_sequence.extend([int(color)] * int(counts_by_color[color]))
    random.shuffle(color_sequence)
    assert len(color_sequence) == len(all_coords)

    assigned = np.zeros((grid_h, grid_w), dtype=bool)
    bg_perm = np.random.permutation(len(background_coords))
    for idx, color in enumerate(noise_colors):
        r, c = background_coords[bg_perm[idx]]
        grid[r, c] = int(color)
        assigned[r, c] = True
        color_sequence.remove(int(color))

    remaining_coords = [tuple(map(int, rc)) for rc in all_coords if not assigned[rc[0], rc[1]]]
    random.shuffle(remaining_coords)
    for (r, c), color in zip(remaining_coords, color_sequence):
        grid[r, c] = int(color)


def _visible_sorted_frame_info(grid: np.ndarray, frame_colors: list[int], room_counts: list[int]) -> list[tuple[int, int, int]]:
    info = []
    for color, rooms in zip(frame_colors, room_counts):
        info.append((int(color), int(rooms), _visible_pixels_for_color(grid, int(color))))
    return sorted(info, key=lambda item: item[2])


def _count_cross_room_inversions(visible_info: list[tuple[int, int, int]]) -> int:
    rank = {color: idx for idx, (color, _rooms, _visible) in enumerate(visible_info)}
    discord = 0
    for i, (color_a, rooms_a, _visible_a) in enumerate(visible_info):
        for color_b, rooms_b, _visible_b in visible_info[i + 1 :]:
            if rooms_a == rooms_b:
                continue
            if (rank[color_a] - rank[color_b]) * (rooms_a - rooms_b) < 0:
                discord += 1
    return int(discord)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)

    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))
    allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, min(distinct_needed, len(available)))
        if len(chosen) < distinct_needed:
            return None
        room_counts = chosen + [random.choice(chosen)]
    else:
        if frame_count > len(available):
            return None
        room_counts = random.sample(available, frame_count)
    random.shuffle(room_counts)

    noise_count = random.randint(*_NOISE_COLOR_COUNT_RANGE)
    chosen_colors = random.sample(list(range(1, 10)), frame_count + noise_count)
    frame_colors = chosen_colors[:frame_count]
    noise_colors = chosen_colors[frame_count:]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    variant_choices = _choose_frame_variants(
        room_counts,
        max_h=min(f_max_h, grid_h - 2),
        max_w=min(f_max_w, grid_w - 2),
    )
    if variant_choices is None:
        return None

    frames = []
    for (color, target_rooms), (frame_data, raw_pixels) in zip(zip(frame_colors, room_counts), variant_choices):
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((int(color), int(target_rooms), frame_data, int(raw_pixels)))

    positions = _place_frames_sparsely(frames, grid_h, grid_w)
    if positions is None or len(positions) != frame_count:
        return None

    render_order = list(range(frame_count))
    random.shuffle(render_order)
    grid, frame_mask, bbox_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)

    try:
        _scatter_noise(grid, frame_mask, bbox_mask, noise_colors)
    except Exception:
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    reconstructed_rooms = {}
    for color, target_rooms in zip(frame_colors, room_counts):
        try:
            reconstructed_rooms[int(color)] = _room_count_for_color(grid, int(color))
        except Exception:
            return None
        if reconstructed_rooms[int(color)] != int(target_rooms):
            return None

    inferred_noise = [
        color for color in colors_present(grid, exclude={0}) if _room_count_for_color(grid, int(color)) == 0
    ]
    if set(inferred_noise) != set(noise_colors):
        return None
    if len(inferred_noise) != noise_count:
        return None

    noise_counts = {color: _visible_pixels_for_color(grid, int(color)) for color in noise_colors}
    pad_color = max(noise_colors, key=lambda color: noise_counts[color])
    if sum(1 for color in noise_colors if noise_counts[color] == noise_counts[pad_color]) != 1:
        return None
    if any(noise_counts[pad_color] <= noise_counts[color] for color in noise_colors if color != pad_color):
        return None

    visible_info = _visible_sorted_frame_info(grid, frame_colors, room_counts)
    visible_counts = [visible for _color, _rooms, visible in visible_info]
    if len(set(visible_counts)) != frame_count:
        return None
    if min(b - a for a, b in zip(visible_counts, visible_counts[1:])) < _MIN_VISIBLE_GAP:
        return None

    if _count_cross_room_inversions(visible_info) == 0:
        return None

    max_rooms = max(room_counts)
    expected = np.full((frame_count, max_rooms), int(pad_color), dtype=int)
    for r, (color, rooms, _visible) in enumerate(visible_info):
        expected[r, :rooms] = int(color)

    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
