# Variant of 89565ca0
# Dimension: D4 — 噪声/干扰机制 (src)
# Change: 一种颜色的散落噪声像素覆盖4-9%的网格，既出现在框架内外，也会损坏部分墙壁像素... → 多种颜色的噪声

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the D4 variant
# ----------------------------------
# variant:
#   core_change:
#     The input now contains 2-3 noise colors instead of exactly one. All noise
#     colors are scattered across the full grid and some also replace wall pixels.
#     The output still uses one padding color, but that color is now defined as
#     the dominant noise color: among the colors that reconstruct to zero rooms,
#     choose the one with the most visible pixels in the input.
#   concrete_behavior:
#     Real objects are unchanged: each frame color still forms an axis-aligned
#     rectangular wall layout with 1-6 enclosed rooms. Noise colors are the only
#     non-background colors whose reconstructed wall map encloses zero rooms.
#     Example trace:
#       - Original: colors {1,2,5} -> room counts {3,2,0}. Padding is 5.
#       - Variant: colors {1,2,5,7,8} -> room counts {3,2,0,0,0}. If color 5
#         appears 24 times while 7 and 8 appear 9 and 7 times, padding is 5.
#         The bars for colors 1 and 2 are identical to the original task, but
#         their trailing cells use the dominant noise color only.
#   complexity_calibration:
#     The change is visible because every generated puzzle contains at least two
#     distinct zero-room colors, usually three. The dominant noise color remains
#     clearly more frequent than the auxiliary noise colors, so the padding rule
#     is inferable from a few examples rather than being an arbitrary tie-break.
#   implicit_assumptions_review:
#     Original `main()` relied on `len(pad_colors) == 1`; that is no longer true.
#     The loop over non-background colors is still valid, and `rooms > 0` still
#     separates frames from noise as long as damage stays moderate. The generator
#     therefore keeps wall damage under the reconstruction threshold and validates
#     every sample with the new solver. The tie-break logic also assumed overlap
#     pixels still reveal one of the two frame colors; the generator now protects
#     geometric overlap pixels from noise damage so equal-room order remains
#     recoverable.
#   placement_validity:
#     Frames are built and placed exactly as before: in-bounds, overlapping only
#     when allowed, and with at most one tied room-count pair. Noise is sampled
#     only from non-wall cells plus a bounded subset of visible wall cells that
#     excludes overlap evidence pixels, so solver assumptions stay satisfiable.
#   observability_and_cleanliness:
#     Human solvers can still see the same room-count task, but now must ignore
#     several scattered colors instead of a single obvious distractor. The grid
#     remains readable because the main geometry is unchanged and the total noise
#     budget stays at 4-9%, split across a small set of colors rather than many.
#   generator_guarantees:
#     Every sample has 2-3 zero-room colors, a unique dominant one, and expected
#     output padding equal to that dominant noise color. The original one-noise
#     solver should usually fail or disagree on these inputs, confirming the D4
#     behavior is genuinely active rather than a degenerate copy of the source.
#   new_solver_assumptions:
#     1. Every true frame still reconstructs to at least one enclosed room.
#     2. Every noise color reconstructs to zero enclosed rooms.
#     3. The dominant noise color is unique by visible pixel count.
#     4. Equal-room frames overlap enough that one frame color survives on their
#        shared reconstructed wall positions.
#     The generator enforces all four conditions and rejects samples that do not.


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
    assert len(hrows) >= 2

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


def _walls_for_color(g: np.ndarray, color: int) -> np.ndarray:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, r1, c1 = b
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    out = np.zeros_like(g, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = walls
    return out


def _cover_order(g: np.ndarray, group: list[tuple[int, int, np.ndarray]]) -> list[tuple[int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _rooms, _walls in group}
    indegree: dict[int, int] = {color: 0 for color, _rooms, _walls in group}

    for i, (ca, _ra, wa) in enumerate(group):
        for cb, _rb, wb in group[i + 1 :]:
            overlap = wa & wb
            if not bool(overlap.any()):
                continue
            overlap_vals = g[overlap]
            hits_a = int(np.count_nonzero(overlap_vals == int(ca)))
            hits_b = int(np.count_nonzero(overlap_vals == int(cb)))
            assert hits_a != hits_b
            assert hits_a > 0 or hits_b > 0
            top = int(ca) if hits_a > hits_b else int(cb)
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
    Convert each colored partitioned frame into a horizontal bar whose length is its number of enclosed regions.

    Input structure:
    - Black background with several axis-aligned colored line drawings.
    - The drawings are mixed with 2-3 scattered noise colors, and those noise pixels can appear both in open space and on top of wall cells.
    - Each genuine drawing is still a rectangular partitioned frame enclosing one or more background regions.

    Rule:
    - For every non-background color, reconstruct its rectilinear wall layout from dense row and column evidence, then count the enclosed 4-connected regions.
    - Colors with a positive room count are the real frames; colors with zero rooms are all noise colors.
    - Among the zero-room colors, use the most frequent one as the output padding color.
    - Sort the frame colors by enclosed-region count, breaking ties by which reconstructed frame lies underneath the other in an overlap.

    Key insight:
    - Multiple distractor colors now play the role that one scatter color used to play before, so the task is no longer "find the lone non-frame color" but "separate all zero-room colors from the true frames, then identify the dominant noise color for padding."
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        counts.append((int(color), int(rooms), walls))

    drawings = [(color, rooms, walls) for color, rooms, walls in counts if rooms > 0]
    noise_colors = [color for color, rooms, _walls in counts if rooms == 0]
    assert drawings
    assert 2 <= len(noise_colors) <= 3

    noise_pixel_counts = {color: int(np.count_nonzero(g == int(color))) for color in noise_colors}
    pad = max(noise_colors, key=lambda color: noise_pixel_counts[color])
    assert sum(1 for color in noise_colors if noise_pixel_counts[color] == noise_pixel_counts[pad]) == 1

    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))

    width = max(rooms for _color, rooms, _walls in ordered)

    out = np.full((len(ordered), width), int(pad), dtype=int)
    for r, (color, rooms, _walls) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (gpt-5) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)             # number of colored frames per puzzle
_ROOM_COUNT_RANGE = (1, 6)              # rooms per frame
_TIE_PROBABILITY = 0.35                 # chance of one tied room-count pair
_GRID_SIZE_BY_FRAMES = {                # (h_min, h_max, w_min, w_max, frame_max_h, frame_max_w)
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)               # min/max gap between grid lines within a frame
_NOISE_FRACTION_RANGE = (0.04, 0.09)    # fraction of total cells as noise
_NOISE_COLOR_COUNT_RANGE = (2, 3)       # number of distinct noise colors
_MIN_NOISE_PIXELS_PER_COLOR = 4         # keeps every noise color visible
_WALL_DAMAGE_FRACTION = 0.25            # max fraction of visible wall pixels replaced by noise
_DAMAGE_SHARE_RANGE = (0.18, 0.38)      # portion of noise budget spent damaging walls
_EXTRA_OVERLAP_RANGE = (0, 3)           # non-tied overlap pairs to force
_OVERLAP_RETRY = 100                    # attempts to place overlapping frame
_GEN_RETRIES = 400                      # outer retry limit


def _random_gaps(n: int, min_gap: int, max_gap: int, max_total: int) -> list[int] | None:
    """Generate n gaps each in [min_gap, max_gap] with sum <= max_total."""
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
    """Build a boolean wall array with exactly target_rooms enclosed regions."""
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


def _force_overlap(positions, i, j, frames, grid_h, grid_w):
    """Try to reposition frame j so it overlaps frame i. Modifies positions in place."""
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
    for idx, (_color, _rooms, wall_arr) in enumerate(frames):
        r0, c0 = positions[idx]
        h, w = wall_arr.shape
        mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _render_frames(frames, positions, render_order, grid_h, grid_w):
    """Render frame walls and record visible walls plus protected overlap evidence."""
    grid = np.zeros((grid_h, grid_w), dtype=int)
    wall_masks = _absolute_wall_masks(frames, positions, grid_h, grid_w)
    order_rank = {idx: pos for pos, idx in enumerate(render_order)}

    for idx in render_order:
        color = frames[idx][0]
        mask = wall_masks[idx]
        grid[mask] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        color = frames[idx][0]
        visible = wall_masks[idx] & (grid == int(color))
        frame_mask |= visible

    protected_overlap_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            overlap = wall_masks[i] & wall_masks[j]
            if not bool(overlap.any()):
                continue
            top_idx = i if order_rank[i] > order_rank[j] else j
            top_color = int(frames[top_idx][0])
            protected_overlap_mask |= overlap & (grid == top_color)

    bbox_mask = _bbox_union_mask(frames, positions, grid_h, grid_w)
    return grid, frame_mask, protected_overlap_mask, bbox_mask


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
    protected_overlap_mask: np.ndarray,
    bbox_mask: np.ndarray,
    noise_colors: list[int],
):
    """Place 2-3 noise colors on open cells and on a limited set of wall cells."""
    grid_h, grid_w = grid.shape
    total_cells = grid_h * grid_w
    total_noise = int(total_cells * random.uniform(*_NOISE_FRACTION_RANGE))
    counts_by_color = _noise_color_counts(total_noise, noise_colors)
    if counts_by_color is None:
        raise AssertionError("Noise budget too small for multiple visible colors")

    damage_pool = np.argwhere(frame_mask & ~protected_overlap_mask)
    nonwall_pool = np.argwhere(~frame_mask)
    inside_nonwall = np.argwhere(bbox_mask & ~frame_mask)
    outside_nonwall = np.argwhere(~bbox_mask & ~frame_mask)

    max_damage = min(
        total_noise - len(noise_colors),
        int(_WALL_DAMAGE_FRACTION * len(damage_pool)),
        int(total_noise * random.uniform(*_DAMAGE_SHARE_RANGE)),
    )
    max_damage = max(0, max_damage)
    damage_count = min(max_damage, len(damage_pool))
    background_count = total_noise - damage_count
    assert background_count >= len(noise_colors)
    assert background_count <= len(nonwall_pool)

    chosen_background: list[np.ndarray] = []
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    if len(inside_nonwall) > 0:
        inside_pick = _sample_coords(inside_nonwall, 1)
        chosen_background.append(inside_pick)
        reserved[inside_pick[:, 0], inside_pick[:, 1]] = True
    if len(outside_nonwall) > 0:
        outside_pick = _sample_coords(outside_nonwall, 1)
        if not reserved[outside_pick[:, 0], outside_pick[:, 1]].all():
            chosen_background.append(outside_pick)
            reserved[outside_pick[:, 0], outside_pick[:, 1]] = True

    remaining_background = background_count - sum(len(chunk) for chunk in chosen_background)
    remaining_nonwall = np.argwhere((~frame_mask) & (~reserved))
    chosen_background.append(_sample_coords(remaining_nonwall, remaining_background))
    background_coords = np.concatenate(chosen_background, axis=0)
    assert len(background_coords) == background_count

    damage_coords = _sample_coords(damage_pool, damage_count)

    all_coords = np.concatenate([background_coords, damage_coords], axis=0)
    perm = np.random.permutation(len(all_coords))
    all_coords = all_coords[perm]

    color_sequence: list[int] = []
    for color in noise_colors:
        color_sequence.extend([int(color)] * int(counts_by_color[color]))
    random.shuffle(color_sequence)
    assert len(color_sequence) == len(all_coords)

    if len(background_coords) >= len(noise_colors):
        assigned = np.zeros((grid_h, grid_w), dtype=bool)
        bg_perm = np.random.permutation(len(background_coords))
        for idx, color in enumerate(noise_colors):
            r, c = background_coords[bg_perm[idx]]
            grid[r, c] = int(color)
            color_sequence.remove(int(color))
            assigned[r, c] = True

        remaining_coords = [tuple(map(int, rc)) for rc in all_coords if not assigned[rc[0], rc[1]]]
        random.shuffle(remaining_coords)
        for (r, c), color in zip(remaining_coords, color_sequence):
            grid[r, c] = int(color)
    else:
        for (r, c), color in zip(all_coords, color_sequence):
            grid[int(r), int(c)] = int(color)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid puzzle grid."""
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
    pad_color = noise_colors[0]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(target_rooms, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, frame_data))

    positions = [None] * frame_count
    for idx in sorted(range(frame_count), key=lambda i: -frames[i][2].size):
        fh, fw = frames[idx][2].shape
        positions[idx] = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))

    tied_pairs = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] == room_counts[j]
    ]
    for i, j in tied_pairs:
        _force_overlap(positions, i, j, frames, grid_h, grid_w)

    non_tied = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] != room_counts[j]
    ]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        for i, j in non_tied[: min(n_extra, len(non_tied))]:
            _force_overlap(positions, i, j, frames, grid_h, grid_w)

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    grid, frame_mask, protected_overlap_mask, bbox_mask = _render_frames(
        frames, positions, render_order, grid_h, grid_w
    )
    try:
        _scatter_noise(grid, frame_mask, protected_overlap_mask, bbox_mask, noise_colors)
    except Exception:
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        [(room_counts[i], render_order.index(i), frame_colors[i]) for i in range(frame_count)]
    )
    max_rooms = max(rc for rc, _, _ in frame_info)
    expected = np.full((frame_count, max_rooms), int(pad_color), dtype=int)
    for r, (rc, _, color) in enumerate(frame_info):
        expected[r, :rc] = int(color)

    if not np.array_equal(output, expected):
        return None

    inferred_noise = [color for color in colors_present(grid, exclude={0}) if _room_count_for_color(grid, color) == 0]
    if set(inferred_noise) != set(noise_colors):
        return None
    if len(inferred_noise) != noise_count:
        return None
    if sum(1 for color in inferred_noise if np.count_nonzero(grid == color) >= np.count_nonzero(grid == pad_color)) != 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
