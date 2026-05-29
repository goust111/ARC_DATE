# Combo variant of 89565ca0 (index 11)
# combo_spec: D2/0+D13/1
# Name: 颜色消歧+多对相同计数

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# -------------------------------------
# a) concrete behavior trace:
#    - Original task:
#      1. Reconstruct each colored rectilinear frame.
#      2. Count its enclosed background rooms.
#      3. Sort by room count.
#      4. If two frames tie, use overlap depth: the lower covered frame comes first.
#    - Combo task:
#      1. Steps 1-3 stay the same.
#      2. Equal-room groups are now common: every sample has either two tied pairs
#         (counts pattern like [2,2,4,4] or [1,1,3,3,5]) or one tied triple
#         ([2,2,2,5,6]), giving 2 or 3 equal-room pairs.
#      3. Ties are broken only by ARC color id, ascending.
#    - Small example:
#      * Suppose blue(1) and orange(7) both have 3 rooms, while red(2) has 5.
#      * Original output could be [orange, blue, red] if orange sat underneath blue.
#      * Combo output is always [blue, orange, red] because 1 < 7.
#      * The generator places the tied 3-room frames left-to-right as [orange, blue]
#        so the new rule changes the visible order rather than accidentally matching
#        a position-based heuristic.
#
# b) complexity calibration:
#    - Using only one tied pair would make many samples look almost identical to the
#      parent task. This combo therefore forces 2 or 3 equal-room pairs per sample.
#    - The generator keeps the total frame count at 4 or 5 so the scene is still
#      readable; more ties are frequent, but not so many that the counting problem
#      disappears into clutter.
#    - Tied groups occupy clean horizontal slot rows. That keeps the visual layout
#      tidy while still making the tie-break branch fire repeatedly.
#
# c) implicit-assumption audit of the parent main():
#    - The parent `_cover_order()` assumes every equal-room group is ordered by a
#      unique visible overlap DAG. That is no longer valid because tied frames are
#      intentionally disjoint here.
#    - The parent expected-output logic used render order inside tied groups. That
#      assumption must be replaced by raw color-id order.
#    - The room reconstruction code itself still relies on:
#         1) each true frame retaining enough of each wall segment,
#         2) the noise color enclosing zero rooms,
#         3) one unique non-frame color.
#      The generator preserves all three conditions and rejects any sample that
#      violates them under the new solver.
#
# d) placement legality:
#    - Frames are placed in disjoint slots inside the canvas, so they never overlap
#      or leave bounds.
#    - Slot sizes are chosen before building frames, and each frame is regenerated
#      until it fits its assigned slot.
#    - The noise color may damage wall pixels, but protected corner pixels keep each
#      frame bbox stable.
#
# e) visual observability:
#    - D2/0 is the least human-friendly part because it uses the ARC palette order.
#      To make that rule inferable at all, every sample contains repeated ties and
#      the tied frames are spatially arranged in the opposite order from color-id
#      order. Across examples, "position order" visibly fails while one consistent
#      color order keeps working.
#    - This still uses an ARC-visible convention rather than a hidden random number:
#      the palette ids are fixed by the task family.
#
# f) visual cleanliness:
#    - Removing frame-frame overlap is deliberate. With 2-3 tied pairs, repeated
#      overlaps would create a noisy stack of wall fragments and make the input much
#      harder to parse than the logical change requires.
#    - Noise density stays close to the parent task, but the structural objects are
#      separated into slots so each frame remains easy to count.
#
# g) generator-level behavior metric:
#    - `equal_pair_count` = number of equal-room frame pairs in the sample.
#      This is forced to be 2 or 3.
#    - `position_color_conflicts` = among equal-room pairs, how many disagree between
#      row-major input order and ascending color-id order.
#      The generator forces this to equal `equal_pair_count`, so the new tie-break is
#      genuinely active rather than matching a simpler spatial heuristic.
#
# h) new solver-generator contract:
#    - Solver assumptions:
#         1) Exactly one non-background color has zero reconstructed rooms.
#         2) Every true frame has at least one reconstructed room.
#         3) Ties are resolved by ascending numeric color id.
#    - Generator guarantees:
#         1) one unique noise color,
#         2) 4 or 5 true frames,
#         3) both the "2 equal-room pairs" and "3 equal-room pairs" branches occur
#            regularly instead of one branch dominating the sample pool,
#         4) every equal-room pair is spatially ordered opposite to color order.
#
# i) post-change audit target:
#    - Every loop in `main()` now depends only on the color set and reconstructed
#      room counts, not on overlaps. The generator therefore no longer has to create
#      overlap evidence, but it does have to create repeated ties. That is the new
#      critical construction guarantee.


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


def _frame_signature(g: np.ndarray, color: int) -> tuple[int, tuple[int, int]]:
    b = bbox(g == int(color))
    assert b is not None
    rooms = _room_count_for_color(g, int(color))
    return int(rooms), (int(b[0]), int(b[1]))


def _equal_pair_count(room_counts: list[int]) -> int:
    total = 0
    for rooms in sorted(set(room_counts)):
        size = sum(1 for value in room_counts if int(value) == int(rooms))
        total += size * (size - 1) // 2
    return int(total)


def _position_color_conflicts(items: list[tuple[int, int, tuple[int, int]]]) -> int:
    total = 0
    for rooms in sorted({int(rooms) for _color, rooms, _anchor in items}):
        group = [item for item in items if int(item[1]) == int(rooms)]
        if len(group) < 2:
            continue
        pos_order = [int(color) for color, _rooms, _anchor in sorted(group, key=lambda item: item[2])]
        color_order = sorted(int(color) for color, _rooms, _anchor in group)
        pos_rank = {int(color): idx for idx, color in enumerate(pos_order)}
        color_rank = {int(color): idx for idx, color in enumerate(color_order)}
        colors = [int(color) for color, _rooms, _anchor in group]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                ca = int(colors[i])
                cb = int(colors[j])
                pos_cmp = pos_rank[ca] < pos_rank[cb]
                color_cmp = color_rank[ca] < color_rank[cb]
                if pos_cmp != color_cmp:
                    total += 1
    return int(total)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored frame into a horizontal bar whose length equals its room count, with repeated ties resolved by color-id order.

    Input structure:
    - Black background with 4 or 5 colored rectilinear frame drawings, usually laid out separately rather than stacked.
    - One extra non-black color appears as scattered noise and may also replace some wall pixels.
    - Every real drawing is still a rectangular 1-pixel-thick wall grid enclosing one or more background rooms.

    Rule:
    - For every non-background color, reconstruct its wall scaffold from dense row and column evidence inside that color's visible bbox.
    - Count the 4-connected background regions fully enclosed by the reconstructed walls.
    - The unique color with zero enclosed regions is the noise/padding color.
    - Sort the true frame colors by enclosed-room count ascending; when counts tie, sort those colors by ascending ARC color id.

    Key insight:
    - The geometry-recovery step is unchanged, but overlap reasoning has been removed from the decision chain: equal-room groups are common, and the decisive signal inside those groups is the fixed ARC palette order of the frame colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        rooms, _anchor = _frame_signature(g, int(color))
        counts.append((int(color), int(rooms)))

    drawings = [(color, rooms) for color, rooms in counts if int(rooms) > 0]
    pad_colors = [color for color, rooms in counts if int(rooms) == 0]
    assert drawings
    assert len(pad_colors) == 1

    ordered = sorted(drawings, key=lambda item: (int(item[1]), int(item[0])))
    pad = int(pad_colors[0])
    width = max(int(rooms) for _color, rooms in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms) in enumerate(ordered):
        out[r, : int(rooms)] = int(color)
    return out


# --------------- generator (gpt-5) ---------------

_ROOM_COUNT_RANGE = (1, 6)
_GRID_CONFIG = {
    4: (2, 2, 22, 30, 22, 30),
    5: (2, 3, 22, 30, 25, 30),
}
_CELL_GAP_RANGE = (3, 6)
_SLOT_GAP = 2
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.22
_FRAME_BUILD_RETRIES = 60
_GEN_RETRIES = 600


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
        if max_h < 5 or max_w < 5:
            return None
        gap_h = random.randint(_CELL_GAP_RANGE[0], max(_CELL_GAP_RANGE[0], max_h - 2))
        gap_w = random.randint(_CELL_GAP_RANGE[0], max(_CELL_GAP_RANGE[0], max_w - 2))
        h, w = gap_h + 2, gap_w + 2
        if h > max_h or w > max_w:
            return None
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

    row_gaps = _random_gaps(n_row_gaps, min_gap, min(max_gap, max_h - 1), max_h - 1)
    col_gaps = _random_gaps(n_col_gaps, min_gap, min(max_gap, max_w - 1), max_w - 1)
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
    if frame_h > max_h or frame_w > max_w:
        return None

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
        return int(x)

    def union(x: int, y: int) -> bool:
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True

    def cell_id(cr: int, cc: int) -> int:
        return int(cr * (n_vcols - 1) + cc)

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
            if union(int(ca), int(cb)):
                walls[int(r), int(c0) + 1 : int(c1)] = False
                merged += 1
        else:
            _, c, r0, r1, ca, cb = seg
            if union(int(ca), int(cb)):
                walls[int(r0) + 1 : int(r1), int(c)] = False
                merged += 1

    if merged < cells_to_merge:
        return None
    if _count_enclosed_regions(walls) != target_rooms:
        return None
    return walls


def _build_frame_to_fit(target_rooms: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(_FRAME_BUILD_RETRIES):
        walls = _build_frame(target_rooms, max_h=max_h, max_w=max_w)
        if walls is not None:
            return walls
        swapped = _build_frame(target_rooms, max_h=max_w, max_w=max_h)
        if swapped is not None:
            walls = swapped.T.copy()
            if walls.shape[0] <= max_h and walls.shape[1] <= max_w:
                return walls
    return None


def _choose_layout_pattern() -> tuple[int, list[int]]:
    if random.choice((2, 3)) == 3:
        return 5, [3, 1, 1]
    if random.choice((0, 1)) == 0:
        return 4, [2, 2]
    return 5, [2, 2, 1]


def _choose_group_specs(group_sizes: list[int]) -> list[dict[str, object]]:
    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))
    sizes = [int(size) for size in group_sizes]
    room_values = random.sample(available, len(sizes))
    random.shuffle(room_values)
    return [{"rooms": int(rooms), "size": int(size)} for rooms, size in zip(room_values, sizes)]


def _slot_templates(frame_count: int, group_specs: list[dict[str, object]]) -> list[list[int]]:
    sizes = sorted(int(spec["size"]) for spec in group_specs)
    if frame_count == 4:
        assert sizes == [2, 2]
        return [[0, 1], [2, 3]]
    if sizes == [1, 1, 3]:
        return [[0, 1, 2], [3], [4]]
    assert sizes == [1, 2, 2]
    return [[0, 1], [3, 4], [2]]


def _grid_and_slots(frame_count: int) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    rows, cols, h_min, h_max, w_min, w_max = _GRID_CONFIG[frame_count]
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    slot_h = (grid_h - _SLOT_GAP * (rows - 1)) // rows
    slot_w = (grid_w - _SLOT_GAP * (cols - 1)) // cols
    used_h = rows * slot_h + _SLOT_GAP * (rows - 1)
    used_w = cols * slot_w + _SLOT_GAP * (cols - 1)
    top_margin = random.randint(0, grid_h - used_h)
    left_margin = random.randint(0, grid_w - used_w)

    slots = []
    for rr in range(rows):
        for cc in range(cols):
            r0 = top_margin + rr * (slot_h + _SLOT_GAP)
            c0 = left_margin + cc * (slot_w + _SLOT_GAP)
            slots.append((int(r0), int(c0), int(slot_h), int(slot_w)))
    return int(grid_h), int(grid_w), slots


def _assign_group_colors(group_specs: list[dict[str, object]], frame_colors: list[int]) -> list[dict[str, object]]:
    colors = frame_colors[:]
    random.shuffle(colors)
    enriched = []
    cursor = 0
    for spec in group_specs:
        size = int(spec["size"])
        chosen = sorted(int(color) for color in colors[cursor : cursor + size])
        cursor += size
        enriched.append({"rooms": int(spec["rooms"]), "colors": chosen})
    return enriched


def _ordered_groups_for_slots(group_specs: list[dict[str, object]], frame_count: int) -> list[dict[str, object]]:
    sizes = sorted(int(len(spec["colors"])) for spec in group_specs)
    if frame_count == 4:
        pairs = group_specs[:]
        random.shuffle(pairs)
        return pairs

    if sizes == [1, 1, 3]:
        triple = next(spec for spec in group_specs if len(spec["colors"]) == 3)
        singles = [spec for spec in group_specs if len(spec["colors"]) == 1]
        random.shuffle(singles)
        return [triple] + singles

    assert sizes == [1, 2, 2]
    pairs = [spec for spec in group_specs if len(spec["colors"]) == 2]
    singleton = next(spec for spec in group_specs if len(spec["colors"]) == 1)
    random.shuffle(pairs)
    return [pairs[0], pairs[1], singleton]


def _place_frames_in_slots(
    slots: list[tuple[int, int, int, int]],
    slot_groups: list[list[int]],
    group_specs: list[dict[str, object]],
) -> list[tuple[int, int, int, np.ndarray, tuple[int, int]]]:
    placed: list[tuple[int, int, int, np.ndarray, tuple[int, int]]] = []
    for slot_ids, spec in zip(slot_groups, group_specs):
        colors_desc = sorted((int(color) for color in spec["colors"]), reverse=True)
        members = []
        for slot_id, color in zip(slot_ids, colors_desc):
            slot_r0, slot_c0, slot_h, slot_w = slots[slot_id]
            walls = _build_frame_to_fit(int(spec["rooms"]), max_h=slot_h, max_w=slot_w)
            if walls is None:
                raise ValueError("frame build failed")
            members.append((slot_id, color, int(spec["rooms"]), walls))

        if len(members) == 1:
            slot_id, color, rooms, walls = members[0]
            slot_r0, slot_c0, slot_h, slot_w = slots[slot_id]
            fh, fw = walls.shape
            r0 = slot_r0 + random.randint(0, slot_h - fh)
            c0 = slot_c0 + random.randint(0, slot_w - fw)
            placed.append((color, rooms, slot_id, walls, (int(r0), int(c0))))
            continue

        max_shared_r = min(slots[slot_id][2] - walls.shape[0] for slot_id, _color, _rooms, walls in members)
        shared_r = random.randint(0, max_shared_r)
        for slot_id, color, rooms, walls in members:
            slot_r0, slot_c0, slot_h, slot_w = slots[slot_id]
            fh, fw = walls.shape
            r0 = slot_r0 + shared_r
            c0 = slot_c0 + random.randint(0, slot_w - fw)
            placed.append((color, rooms, slot_id, walls, (int(r0), int(c0))))

    return placed


def _render_frames(
    frames: list[tuple[int, int, int, np.ndarray, tuple[int, int]]],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    protected = np.zeros((grid_h, grid_w), dtype=bool)

    for color, _rooms, _slot_id, walls, (r0, c0) in frames:
        fh, fw = walls.shape
        rs, cs = np.where(walls)
        abs_rs = rs + int(r0)
        abs_cs = cs + int(c0)
        grid[abs_rs, abs_cs] = int(color)
        frame_mask[abs_rs, abs_cs] = True
        protected[int(r0), int(c0)] = True
        protected[int(r0), int(c0) + fw - 1] = True
        protected[int(r0) + fh - 1, int(c0)] = True
        protected[int(r0) + fh - 1, int(c0) + fw - 1] = True

    return grid, frame_mask, protected


def _scatter_noise(grid: np.ndarray, frame_mask: np.ndarray, protected: np.ndarray, noise_color: int) -> None:
    grid_h, grid_w = grid.shape
    total_cells = grid_h * grid_w
    noise_count = max(1, int(total_cells * random.uniform(*_NOISE_FRACTION_RANGE)))

    nonwall_rs, nonwall_cs = np.where(~frame_mask)
    if len(nonwall_rs) > 0:
        perm = np.random.permutation(len(nonwall_rs))
        n_nonwall = min(noise_count, len(perm))
        sel = perm[:n_nonwall]
        grid[nonwall_rs[sel], nonwall_cs[sel]] = int(noise_color)
    else:
        n_nonwall = 0

    wall_rs, wall_cs = np.where(frame_mask & ~protected)
    remaining = noise_count - n_nonwall
    if remaining <= 0 or len(wall_rs) == 0:
        return
    max_damage = min(remaining, int(_WALL_DAMAGE_FRACTION * len(wall_rs)))
    if max_damage <= 0:
        return
    damage_sel = np.random.permutation(len(wall_rs))[:max_damage]
    grid[wall_rs[damage_sel], wall_cs[damage_sel]] = int(noise_color)


def _expected_output(frames: list[tuple[int, int, int, np.ndarray, tuple[int, int]]], pad: int) -> np.ndarray:
    ordered = sorted(((int(color), int(rooms)) for color, rooms, _slot_id, _walls, _pos in frames), key=lambda item: (item[1], item[0]))
    width = max(int(rooms) for _color, rooms in ordered)
    out = np.full((len(ordered), width), int(pad), dtype=int)
    for r, (color, rooms) in enumerate(ordered):
        out[r, : int(rooms)] = int(color)
    return out


def _behavior_ok(frames: list[tuple[int, int, int, np.ndarray, tuple[int, int]]]) -> bool:
    items = [(int(color), int(rooms), (int(pos[0]), int(pos[1]))) for color, rooms, _slot_id, _walls, pos in frames]
    room_counts = [int(rooms) for _color, rooms, _anchor in items]
    pairs = _equal_pair_count(room_counts)
    conflicts = _position_color_conflicts(items)
    return pairs in {2, 3} and conflicts == pairs


def _gen_attempt() -> np.ndarray | None:
    frame_count, group_sizes = _choose_layout_pattern()
    group_specs = _choose_group_specs(group_sizes)
    slot_groups = _slot_templates(frame_count, group_specs)
    group_specs = _ordered_groups_for_slots(_assign_group_colors(group_specs, random.sample(list(range(1, 10)), frame_count)), frame_count)

    grid_h, grid_w, slots = _grid_and_slots(frame_count)
    frames = _place_frames_in_slots(slots, slot_groups, group_specs)
    if not _behavior_ok(frames):
        return None

    frame_colors = [int(color) for color, _rooms, _slot_id, _walls, _pos in frames]
    noise_candidates = [color for color in range(1, 10) if color not in frame_colors]
    if len(noise_candidates) != 9 - frame_count:
        return None
    noise_color = int(random.choice(noise_candidates))

    grid, frame_mask, protected = _render_frames(frames, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, protected, noise_color)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(frames, noise_color)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate combo variant D2/0+D13/1 after {_GEN_RETRIES} attempts")
