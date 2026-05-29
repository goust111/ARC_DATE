# Combo variant of 89565ca0 (index 25)
# combo_spec: D7/0+D11/0
# Name: 双倍长度+仅2个框架

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior trace
#    - Source task: 3-5 colored rectilinear frames are hidden under one scattered noise color; each frame maps to
#      one output row whose bar length equals its enclosed-room count.
#    - Combo task: the input still contains damaged colored frames plus one noise color, but now there are exactly
#      2 real frames. If the two frames contain 2 and 5 rooms, the source output would have rows of lengths 2 and 5,
#      while this combo output has exactly 2 rows with lengths 4 and 10.
#    - The changed stages are therefore: object-count expectation in the solver/generator and the final rendering
#      transform from rooms -> 2 * rooms. Counting rooms and discarding the zero-room noise color stay unchanged.
# b) Complexity calibration
#    - Fixing the scene to 2 frames simplifies the visual search, but the doubled bar-length rule keeps a visible
#      nontrivial transform: humans still must count rooms, then infer the global x2 scaling from several examples.
#    - Generator uses distinct room counts for the two frames, so both rows carry ordering evidence and the doubled
#      lengths do not collapse into a tie-only case that would hide the scaling relation.
# c) Implicit assumption audit for main()
#    - `get_bg(g) == 0` still must hold; generator leaves most cells black.
#    - Exactly one non-background color must reconstruct to 0 rooms; generator still uses a single scattered noise
#      color and validates every sample with the solver.
#    - New combo-specific assumption: there are exactly 2 positive-room frame colors. `main()` now asserts this, so
#      `generate_input()` must never emit extra reconstructible colors or a missing frame.
#    - Original code assumed row width equals `rooms`; both width computation and row fill span are updated to
#      `_bar_length(rooms)`. Other room-based logic remains valid.
# d) Placement legality
#    - "Legal" here means within the variant's task rules, not necessarily disjoint: frame overlap is still allowed.
#    - The generator always keeps both frame bboxes inside the grid. It optionally creates one moderate overlap, or
#      else keeps the pair separated, which avoids uncontrolled clutter while preserving the source-task flavor.
# e) Visual observability
#    - Across 2-4 examples, a human sees exactly two frame colors in the input and exactly two output rows. Comparing
#      room counts to row lengths reveals a consistent factor of 2, not just a larger absolute width.
#    - The variant cannot silently degenerate to the source mapping because every real frame has `rooms > 0`, hence
#      `2 * rooms != rooms`.
# f) Visualization cleanliness
#    - With only 2 frames, the scene is less crowded than the source task. Optional overlap is limited to a single
#      pair and moderate bbox intersection, so walls remain readable after noise damage.
#    - Output is especially clean: 2 rows only, each with one solid color run followed by the pad/noise color.
# g) Change inferability
#    - No new color remapping is introduced. Each row still uses the source frame's own color, so the only new rule
#      to infer is structural: output row count equals the number of real frames (fixed at 2), and each run length
#      is twice the recovered room count.


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
    assert len(drawings) == 2
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, np.ndarray]] = []
    for rooms in sorted({rooms for _color, rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == rooms]
        ordered.extend(_cover_order(g, group))
    return ordered, pad


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the two colored partitioned frames into two horizontal bars whose lengths are twice their enclosed-room counts.

    Input structure:
    - Black background with exactly two real axis-aligned colored wall frames and one extra scattered noise color.
    - Each real frame is a rectangular line grid that encloses one or more background rooms; noise and overlap may hide parts of the walls.
    - The noise color appears as isolated pixels on background and damaged wall cells, but it does not form a real enclosed frame.

    Rule:
    - For each non-background color, reconstruct rectilinear wall segments from dense row and column evidence inside that color's bounding box.
    - Count the 4-connected background regions fully surrounded by the reconstructed walls.
    - Discard the unique zero-room color as the pad/noise color, keep the two positive-room frame colors, and sort them by room count ascending.
    - Output exactly two rows: each row is the frame color repeated 2 x room_count times, padded on the right with the noise color to the common maximum width.

    Key insight:
    - The scattered color is still only padding/noise, while the meaningful invariant is the room count of each reconstructed frame; the new twist is that every output bar is stretched to double that count and there are only two such bars.
    """
    g = np.asarray(grid, dtype=int)
    ordered, pad = _ordered_drawings(g)

    width = max(_bar_length(rooms) for _color, rooms, _walls in ordered)
    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms, _walls) in enumerate(ordered):
        out[r, : _bar_length(rooms)] = int(color)
    return out


# --------------- generator (combo variant) ---------------

_FRAME_COUNT = 2
_ROOM_COUNT_RANGE = (1, 6)
_GRID_SHAPE_RANGE = (20, 28, 20, 28)
_FRAME_MAX_SHAPE = (22, 25)
_CELL_GAP_RANGE = (3, 13)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.25
_FORCE_OVERLAP_PROBABILITY = 0.4
_MAX_BBOX_OVERLAP_RATIO = 0.4
_PLACE_RETRIES = 120
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


def _boxes_overlap(
    pos_a: tuple[int, int], shape_a: tuple[int, int], pos_b: tuple[int, int], shape_b: tuple[int, int]
) -> bool:
    ra, ca = pos_a
    ha, wa = shape_a
    rb, cb = pos_b
    hb, wb = shape_b
    return max(ra, rb) <= min(ra + ha - 1, rb + hb - 1) and max(ca, cb) <= min(ca + wa - 1, cb + wb - 1)


def _bbox_overlap_area(
    pos_a: tuple[int, int], shape_a: tuple[int, int], pos_b: tuple[int, int], shape_b: tuple[int, int]
) -> int:
    if not _boxes_overlap(pos_a, shape_a, pos_b, shape_b):
        return 0
    ra, ca = pos_a
    ha, wa = shape_a
    rb, cb = pos_b
    hb, wb = shape_b
    overlap_h = min(ra + ha - 1, rb + hb - 1) - max(ra, rb) + 1
    overlap_w = min(ca + wa - 1, cb + wb - 1) - max(ca, cb) + 1
    return int(overlap_h * overlap_w)


def _place_frames(frames, grid_h: int, grid_w: int, want_overlap: bool) -> list[tuple[int, int]] | None:
    shape0 = frames[0][2].shape
    shape1 = frames[1][2].shape
    area_limit = int(_MAX_BBOX_OVERLAP_RATIO * min(shape0[0] * shape0[1], shape1[0] * shape1[1]))
    for _ in range(_PLACE_RETRIES):
        pos0 = (random.randint(0, grid_h - shape0[0]), random.randint(0, grid_w - shape0[1]))
        pos1 = (random.randint(0, grid_h - shape1[0]), random.randint(0, grid_w - shape1[1]))
        overlap_area = _bbox_overlap_area(pos0, shape0, pos1, shape1)
        if want_overlap:
            if overlap_area <= 0 or overlap_area > area_limit:
                continue
        elif overlap_area != 0:
            continue
        return [pos0, pos1]
    return None


def _render_frames(frames, positions, render_order, grid_h: int, grid_w: int):
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


def _scatter_noise(grid, frame_mask: np.ndarray, noise_color: int, grid_h: int, grid_w: int) -> None:
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
    room_counts: list[int], frame_colors: list[int], render_order: list[int], noise_color: int
) -> np.ndarray:
    frame_info = sorted((room_counts[i], render_order.index(i), frame_colors[i]) for i in range(_FRAME_COUNT))
    max_len = max(_bar_length(rc) for rc, _order_idx, _color in frame_info)
    expected = np.full((_FRAME_COUNT, max_len), noise_color, dtype=int)
    for r, (rc, _order_idx, color) in enumerate(frame_info):
        expected[r, : _bar_length(rc)] = color
    return expected


def _gen_attempt() -> np.ndarray | None:
    room_counts = random.sample(list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1)), _FRAME_COUNT)
    chosen_colors = random.sample(list(range(1, 10)), _FRAME_COUNT + 1)
    frame_colors = chosen_colors[:_FRAME_COUNT]
    noise_color = chosen_colors[_FRAME_COUNT]

    h_min, h_max, w_min, w_max = _GRID_SHAPE_RANGE
    max_fh, max_fw = _FRAME_MAX_SHAPE
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(target_rooms, max_h=max_fh, max_w=max_fw)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, frame_data))

    want_overlap = random.random() < _FORCE_OVERLAP_PROBABILITY
    positions = _place_frames(frames, grid_h, grid_w, want_overlap=want_overlap)
    if positions is None:
        return None

    render_order = list(range(_FRAME_COUNT))
    random.shuffle(render_order)

    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(room_counts, frame_colors, render_order, noise_color)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
