# Combo variant of 89565ca0 (index 37)
# combo_spec: D7/0+D8/1
# Name: 双倍长度+单行输出

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
#    - Source task:
#      after reconstructing each colored frame, the solver counts enclosed rooms,
#      sorts frames by room count, then outputs one padded row per frame where
#      bar length == room count.
#    - D7/0 changes the per-frame encoding length from rooms to 2 * rooms.
#    - D8/1 changes the layout from many padded rows to one row formed by
#      concatenating the per-frame bars in sorted order.
#    - Combined behavior on a small ordered example:
#      ordered frames = [(blue, 1), (green, 2), (red, 4)]
#      source output:
#        [blue, pad,  pad,  pad]
#        [green, green, pad, pad]
#        [red, red, red, red]
#      combo output:
#        [blue, blue, green, green, green, green, red, red, red, red, red, red, red, red]
#      The recognition half is unchanged; only the serialization changes, and the
#      two output edits interact directly because the total width is now the sum
#      of doubled segment lengths rather than max(room_count).
# b) Complexity calibration
#    - The rule stays global and monotone: more rooms still means a longer run.
#    - The visual difference from the source task is strong: every output has
#      height 1, no padding cells, and each run is even-length.
#    - Adjacent segments remain distinguishable because every frame color is
#      unique, so concatenation does not collapse boundaries.
# c) Implicit assumption audit for main()
#    - get_bg(grid) == 0 must still hold. The generator keeps black dominant.
#    - Exactly one non-background color must yield 0 rooms. The generator still
#      uses exactly one scattered noise color and no real zero-room frames.
#    - _cover_order still assumes tied frames overlap enough to reveal which one
#      is on top. The generator continues forcing overlap for every tied pair.
#    - Original output code assumed a rectangular matrix padded by the noise
#      color. That assumption is removed entirely; output width now depends on
#      the sum of transformed run lengths and output height is fixed to 1.
#    - The new concatenation code assumes each true frame contributes a positive
#      segment length. This is safe because room counts are restricted to 1-6.
# d) Placement legality
#    - Input geometry is unchanged from the source spec, so frame placement,
#      overlap forcing, wall damage, and bounds checks remain valid as-is.
#    - Because the combo changes only the output encoding, no extra spatial
#      constraints are needed beyond keeping all frames in-bounds.
# e) Visual observability
#    - Across a few examples a human can count rooms, see the sorted order, and
#      notice two simultaneous regularities: all output is on one row and each
#      color run is twice the counted room total.
#    - This does not depend on hidden metadata such as numeric color ids.
# f) Visualization cleanliness
#    - The input remains the same controlled clutter as the source task.
#    - The output is cleaner than the source: a single crisp strip with distinct
#      color blocks and no filler cells.
# g) Change inferability
#    - No new color remapping exists. Frame colors are preserved in the output,
#      so the only inferred transformation is the joint "sort, double, then
#      concatenate" rule, which is visible from segment lengths and order.


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


def _bar_length(rooms: int) -> int:
    return int(rooms) * 2


def _ordered_drawings(g: np.ndarray) -> tuple[list[tuple[int, int, np.ndarray]], int]:
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

    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))
    return ordered, int(pad_colors[0])


def _encode_concatenated_output(color_room_pairs: list[tuple[int, int]]) -> np.ndarray:
    total_width = sum(_bar_length(rooms) for _color, rooms in color_room_pairs)
    assert total_width > 0
    out = np.zeros((1, total_width), dtype=int)
    c0 = 0
    for color, rooms in color_room_pairs:
        seg_len = _bar_length(rooms)
        assert seg_len > 0
        out[0, c0 : c0 + seg_len] = int(color)
        c0 += seg_len
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the colored partition frames into one continuous horizontal strip whose segment lengths are twice the room counts.

    Input structure:
    - A black field contains several colored axis-aligned wall frames plus one extra scattered noise color.
    - Real frames may overlap and may lose some visible wall pixels to the noise color.
    - After rectilinear reconstruction, each true frame still encloses one or more background rooms.

    Rule:
    - For each non-black color, reconstruct its horizontal and vertical wall segments from dense row/column evidence.
    - Count the 4-connected background regions completely enclosed by those reconstructed walls.
    - Treat the unique zero-room color as noise only, then sort the true frame colors by room count, using cover order to break ties.
    - Encode the sorted frames as one single row by concatenating solid color segments, where each segment has length 2 x room_count.

    Key insight:
    - The object recognition part is unchanged: the stable signal is still each frame's enclosed-room count after wall repair.
    - The output now fuses two readout changes at once: the bars are globally doubled in length and serialized end-to-end instead of stacked as padded rows.
    """
    g = np.asarray(grid, dtype=int)
    ordered, _pad = _ordered_drawings(g)
    return _encode_concatenated_output([(color, rooms) for color, rooms, _walls in ordered])


# --------------- generator (combo variant) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    5: (28, 36, 28, 36, 14, 16),
    4: (24, 32, 24, 32, 16, 20),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.25
_EXTRA_OVERLAP_RANGE = (0, 3)
_OVERLAP_RETRY = 100
_GEN_RETRIES = 600
_PROFILE_POOL = [
    (3, False),
    (3, True),
    (4, False),
    (4, True),
    (5, None),
]


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


def _render_frames(frames, positions, render_order, grid_h, grid_w):
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        grid[rs + r0, cs + c0] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][2]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        abs_rs, abs_cs = rs + r0, cs + c0
        visible = grid[abs_rs, abs_cs] == color
        frame_mask[abs_rs[visible], abs_cs[visible]] = True

    return grid, frame_mask


def _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w):
    total_cells = grid_h * grid_w
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)

    nonwall_rs, nonwall_cs = np.where(~frame_mask)
    perm = np.random.permutation(len(nonwall_rs))
    n_nonwall = min(noise_count, len(perm))
    sel = perm[:n_nonwall]
    grid[nonwall_rs[sel], nonwall_cs[sel]] = noise_color
    placed = n_nonwall

    wall_rs, wall_cs = np.where(frame_mask)
    n_wall = len(wall_rs)
    max_damage = min(noise_count - placed, int(_WALL_DAMAGE_FRACTION * n_wall))
    if max_damage > 0:
        damage_perm = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_perm], wall_cs[damage_perm]] = noise_color


def _expected_output(
    frame_count: int, room_counts: list[int], render_order: list[int], frame_colors: list[int]
) -> np.ndarray:
    frame_info = sorted(
        (room_counts[i], render_order.index(i), frame_colors[i]) for i in range(frame_count)
    )
    return _encode_concatenated_output([(color, rooms) for rooms, _depth, color in frame_info])


def _choose_room_counts(frame_count: int, force_tie: bool | None) -> list[int] | None:
    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))
    if frame_count > len(available):
        return None

    allow_tie = (
        random.random() < _TIE_PROBABILITY if force_tie is None else bool(force_tie)
    ) and frame_count >= 3

    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, min(distinct_needed, len(available)))
        if len(chosen) < distinct_needed:
            return None
        return chosen + [random.choice(chosen)]
    else:
        return random.sample(available, frame_count)


def _extra_overlap_limit(frame_count: int) -> int:
    if frame_count >= 5:
        return 1
    if frame_count == 4:
        return 2
    return _EXTRA_OVERLAP_RANGE[1]


def _gen_attempt(frame_count: int | None = None, force_tie: bool | None = None) -> np.ndarray | None:
    if frame_count is None:
        frame_count = random.randint(*_FRAME_COUNT_RANGE)

    room_counts = _choose_room_counts(frame_count, force_tie)
    if room_counts is None:
        return None
    random.shuffle(room_counts)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

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
    n_extra = random.randint(_EXTRA_OVERLAP_RANGE[0], _extra_overlap_limit(frame_count))
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        for i, j in non_tied[: min(n_extra, len(non_tied))]:
            _force_overlap(positions, i, j, frames, grid_h, grid_w)

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(frame_count, room_counts, render_order, frame_colors)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    target_frame_count, target_tie = random.choice(_PROFILE_POOL)
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt(frame_count=target_frame_count, force_tie=target_tie)
        if result is not None:
            return result
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
