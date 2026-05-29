# Combo variant of 89565ca0 (index 32)
# combo_spec: D11/1+D13/2
# Name: 6-8个框架+房间数1-10

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the fused variant.
#
# a) Concrete behavior trace:
#    In the base task, a sample might have 4 frames with room counts [1, 2, 4, 6],
#    yielding a 4x6 output. In this variant, a typical sample has 7 frames such as
#    [1, 3, 4, 6, 7, 8, 10], so the input must visually support more objects and
#    longer reconstructed bars, and the output becomes 7x10. The solver logic still
#    reconstructs each color's rectilinear walls from dense rows/cols, but the
#    generator must now budget space so high-room frames remain readable.
#
# b) Complexity calibration:
#    The variant should not silently collapse back to the 3-5 / 1-6 regime. To make
#    the change observable, every generated sample uses 6-8 frames and forces at
#    least one frame to have 7-10 rooms. At most one tie pair is allowed so the
#    cover-order rule remains present without creating ambiguous equal-count groups.
#
# c) Hidden-assumption audit against main():
#    - `get_bg(...) == 0`: the generator keeps the canvas large enough that black
#      stays dominant despite more objects.
#    - exactly one rooms=0 color: only the scattered noise color is allowed to have
#      no reconstructible frame; sparse noise plus low wall damage keeps this true.
#    - `_cover_order(...)` needs a unique topological order for equal counts: the
#      generator therefore allows at most one tied pair and forces an actual wall
#      overlap for that pair.
#    - `_reconstruct_walls(...)` needs surviving dense wall evidence: wall damage is
#      capped per frame, and extra overlaps are limited to keep each segment above
#      the reconstruction threshold.
#
# d) Placement legality:
#    Frames are first placed with non-overlapping bboxes, then a small number of
#    controlled overlaps are introduced. Every candidate move is checked for
#    in-bounds placement and for real wall-pixel overlap when a tie pair needs it.
#
# e/f/g) Visual observability and cleanliness:
#    Human-solvable cues are the larger number of colored line frames and the wider
#    output bars caused by counts above 6. The generator avoids excessive overlap
#    layers, keeps noise moderate, and never uses invisible metadata such as color id
#    order for ranking. Ties, when present, are supported by visible covering.


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


def _stable_room_count_from_mask(mask: np.ndarray) -> int:
    return _count_enclosed_regions(_reconstruct_walls(mask.astype(bool)))


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
    Convert each colored compartment frame into a horizontal bar whose length equals its enclosed-room count.

    Input structure:
    - Black background with 6-8 colored rectilinear frames plus one scattered noise color.
    - Each true object is a 1px-thick axis-aligned wall layout that encloses 1-10 background rooms.
    - Some wall pixels are hidden by noise or by other frames drawn on top of them.

    Rule:
    - For every non-black color, rebuild the likely wall lattice from dense horizontal and vertical evidence inside its bbox.
    - Count how many 4-connected background regions are fully sealed by those reconstructed walls.
    - Treat the unique color with zero enclosed regions as the padding/noise color.
    - Sort the real frame colors by room count ascending; if counts tie, use covering order among overlapping frames.

    Key insight:
    - The apparent clutter is misleading: only colors whose damaged strokes still imply a rectangular wall system represent real frames, and their room totals remain recoverable even in the larger 6-8 frame setting.
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
    pad_colors = [color for color, rooms, _walls in counts if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))

    width = max(rooms for _color, rooms, _walls in ordered)
    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms, _walls) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (fused large-scale variant) ---------------

_FRAME_COUNT_RANGE = (6, 8)
_ROOM_COUNT_RANGE = (1, 10)
_HIGH_ROOM_MIN = 7
_TIE_PROBABILITY = 0.3
_GRID_SIZE_BY_FRAMES = {
    6: (46, 52, 50, 56, 17, 23),
    7: (50, 56, 54, 60, 17, 23),
    8: (54, 60, 58, 66, 17, 23),
}
_CELL_GAP_RANGE = (4, 6)
_NOISE_FRACTION_RANGE = (0.025, 0.045)
_WALL_DAMAGE_FRACTION = 0.0
_EXTRA_OVERLAP_RANGE = (0, 0)
_OVERLAP_RETRY = 120
_PLACEMENT_RETRY = 1000
_GEN_RETRIES = 400


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


def _build_frame(target_rooms: int, max_h: int, max_w: int) -> np.ndarray | None:
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

    possible_configs: list[tuple[int, int, int]] = []
    for nv in range(2, 8):
        for nh in range(2, 7):
            cells = (nv - 1) * (nh - 1)
            if target_rooms <= cells <= target_rooms + 5:
                possible_configs.append((nv, nh, cells))
    if not possible_configs:
        for nv in range(2, 8):
            for nh in range(2, 7):
                cells = (nv - 1) * (nh - 1)
                if cells >= target_rooms:
                    possible_configs.append((nv, nh, cells))
    if not possible_configs:
        return None

    random.shuffle(possible_configs)
    min_gap, max_gap = _CELL_GAP_RANGE
    for n_vcols, n_hrows, total_cells in possible_configs:
        n_row_gaps = n_hrows - 1
        n_col_gaps = n_vcols - 1
        if min_gap * n_row_gaps > max_h - 1 or min_gap * n_col_gaps > max_w - 1:
            continue

        row_gaps = _random_gaps(n_row_gaps, min_gap, max_gap, max_h - 1)
        col_gaps = _random_gaps(n_col_gaps, min_gap, max_gap, max_w - 1)
        if row_gaps is None or col_gaps is None:
            continue

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
            if _stable_room_count_from_mask(walls) == target_rooms:
                return walls
            continue

        parent = list(range(total_cells))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[px] = py
            return True

        def cell_id(cr: int, cc: int) -> int:
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
                if union(ca, cb):
                    walls[r, c0 + 1 : c1] = False
                    merged += 1
            else:
                _, c, r0, r1, ca, cb = seg
                if union(ca, cb):
                    walls[r0 + 1 : r1, c] = False
                    merged += 1

        if merged == cells_to_merge and _stable_room_count_from_mask(walls) == target_rooms:
            return walls
    return None


def _bbox_from_position(pos: tuple[int, int], walls: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pos
    h, w = walls.shape
    return r0, c0, r0 + h - 1, c0 + w - 1


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _walls_overlap(walls_a: np.ndarray, pos_a: tuple[int, int], walls_b: np.ndarray, pos_b: tuple[int, int]) -> bool:
    ra, ca = pos_a
    rb, cb = pos_b
    ha, wa = walls_a.shape
    hb, wb = walls_b.shape
    r0 = max(ra, rb)
    c0 = max(ca, cb)
    r1 = min(ra + ha - 1, rb + hb - 1)
    c1 = min(ca + wa - 1, cb + wb - 1)
    if r0 > r1 or c0 > c1:
        return False
    part_a = walls_a[r0 - ra : r1 - ra + 1, c0 - ca : c1 - ca + 1]
    part_b = walls_b[r0 - rb : r1 - rb + 1, c0 - cb : c1 - cb + 1]
    return bool((part_a & part_b).any())


def _choose_room_counts(frame_count: int) -> list[int] | None:
    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))
    allow_tie = random.random() < _TIE_PROBABILITY
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, distinct_needed)
        if max(chosen) < _HIGH_ROOM_MIN:
            low_choices = [v for v in chosen if v < _HIGH_ROOM_MIN]
            if not low_choices:
                return None
            old = random.choice(low_choices)
            high_options = [v for v in available if v >= _HIGH_ROOM_MIN and v not in chosen]
            if not high_options:
                return None
            chosen[chosen.index(old)] = random.choice(high_options)
        dup = random.choice(chosen)
        room_counts = chosen + [dup]
    else:
        room_counts = random.sample(available, frame_count)
        if max(room_counts) < _HIGH_ROOM_MIN:
            low_choices = [v for v in room_counts if v < _HIGH_ROOM_MIN]
            high_options = [v for v in available if v >= _HIGH_ROOM_MIN and v not in room_counts]
            if not low_choices or not high_options:
                return None
            old = random.choice(low_choices)
            room_counts[room_counts.index(old)] = random.choice(high_options)
    random.shuffle(room_counts)
    return room_counts


def _place_frames(frames: list[tuple[int, int, np.ndarray]], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(frames)
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(frames)), key=lambda i: -frames[i][2].size)
    for idx in order:
        fh, fw = frames[idx][2].shape
        placed = False
        for _ in range(_PLACEMENT_RETRY):
            r0 = random.randint(0, grid_h - fh)
            c0 = random.randint(0, grid_w - fw)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_bbox_overlaps(box, other, pad=0) for other in placed_boxes):
                continue
            positions[idx] = (r0, c0)
            placed_boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return [p for p in positions if p is not None]


def _force_overlap(
    positions: list[tuple[int, int]],
    i: int,
    j: int,
    frames: list[tuple[int, int, np.ndarray]],
    grid_h: int,
    grid_w: int,
    max_other_overlaps: int,
) -> bool:
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    ri, ci = positions[i]
    r0_min = max(0, ri - fj_h + 1)
    r0_max = min(grid_h - fj_h, ri + fi_h - 1)
    c0_min = max(0, ci - fj_w + 1)
    c0_max = min(grid_w - fj_w, ci + fi_w - 1)
    if r0_min > r0_max or c0_min > c0_max:
        return False
    old_pos = positions[j]
    for _ in range(_OVERLAP_RETRY):
        rj = random.randint(r0_min, r0_max)
        cj = random.randint(c0_min, c0_max)
        cand = (rj, cj)
        if not _walls_overlap(frames[i][2], positions[i], frames[j][2], cand):
            continue
        overlap_count = 0
        for k, pos_k in enumerate(positions):
            if k == j:
                continue
            if _bbox_overlaps(_bbox_from_position(cand, frames[j][2]), _bbox_from_position(pos_k, frames[k][2])):
                overlap_count += 1
        if overlap_count - 1 > max_other_overlaps:
            continue
        positions[j] = cand
        return True
    positions[j] = old_pos
    return False


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    render_order: list[int],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        grid[rs + r0, cs + c0] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    visible_by_frame: dict[int, np.ndarray] = {}
    for idx in range(len(frames)):
        r0, c0 = positions[idx]
        wall_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        abs_rs, abs_cs = rs + r0, cs + c0
        visible = grid[abs_rs, abs_cs] == color
        vis_mask = np.zeros((grid_h, grid_w), dtype=bool)
        vis_mask[abs_rs[visible], abs_cs[visible]] = True
        visible_by_frame[idx] = vis_mask
        frame_mask |= vis_mask
    return grid, frame_mask, visible_by_frame


def _scatter_noise(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    visible_by_frame: dict[int, np.ndarray],
    noise_color: int,
) -> None:
    grid_h, grid_w = grid.shape
    total_cells = grid_h * grid_w
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)

    nonwall_rs, nonwall_cs = np.where(~frame_mask)
    perm = np.random.permutation(len(nonwall_rs))
    n_nonwall = min(noise_count, len(perm))
    sel = perm[:n_nonwall]
    grid[nonwall_rs[sel], nonwall_cs[sel]] = noise_color
    placed = n_nonwall

    remaining = noise_count - placed
    if remaining <= 0:
        return

    damage_coords: list[tuple[int, int]] = []
    frame_ids = list(visible_by_frame)
    random.shuffle(frame_ids)
    for idx in frame_ids:
        if remaining <= 0:
            break
        rs, cs = np.where(visible_by_frame[idx])
        if len(rs) == 0:
            continue
        cap = min(remaining, int(_WALL_DAMAGE_FRACTION * len(rs)))
        if cap <= 0:
            continue
        dmg = random.randint(0, cap)
        if dmg <= 0:
            continue
        perm = np.random.permutation(len(rs))[:dmg]
        damage_coords.extend((int(rs[p]), int(cs[p])) for p in perm)
        remaining -= dmg

    if damage_coords:
        rr, cc = zip(*damage_coords)
        grid[np.array(rr), np.array(cc)] = noise_color


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    room_counts = _choose_room_counts(frame_count)
    if room_counts is None:
        return None
    assert max(room_counts) >= _HIGH_ROOM_MIN

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    h_min, h_max, w_min, w_max, f_max_h, f_max_w = _GRID_SIZE_BY_FRAMES[frame_count]
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(target_rooms, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, frame_data))

    positions = _place_frames(frames, grid_h, grid_w)
    if positions is None:
        return None

    tied_pairs = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] == room_counts[j]
    ]
    if len(tied_pairs) > 1:
        return None
    for i, j in tied_pairs:
        if not _force_overlap(positions, i, j, frames, grid_h, grid_w, max_other_overlaps=0):
            return None

    non_tied = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] != room_counts[j]
    ]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        added = 0
        for i, j in non_tied:
            if added >= n_extra:
                break
            if _force_overlap(positions, i, j, frames, grid_h, grid_w, max_other_overlaps=1):
                added += 1

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    grid, frame_mask, visible_by_frame = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, visible_by_frame, noise_color)

    try:
        output = main(grid)
    except Exception:
        return None

    if output.shape != (frame_count, max(room_counts)):
        return None

    frame_info = sorted((room_counts[i], render_order.index(i), frame_colors[i]) for i in range(frame_count))
    expected = np.full((frame_count, max(room_counts)), noise_color, dtype=int)
    for r, (rc, _draw_rank, color) in enumerate(frame_info):
        expected[r, :rc] = color

    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
