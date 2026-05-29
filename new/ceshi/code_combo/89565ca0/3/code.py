# Combo variant of 89565ca0 (index 3)
# combo_spec: D1/0+D6/0
# Name: 面积排序+黑色填充

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Input objects remain the original ones: several colored rectilinear line-frames on black plus one scattered noise color.
#    - Each true frame still produces a bar whose colored prefix length equals its enclosed-room count.
#    - Fusion change 1: row order no longer follows room count; it follows recovered bbox area.
#    - Fusion change 2: cells to the right of each colored prefix are no longer filled with the noise color; they are black.
#    - Small trace:
#      suppose three frames have (rooms, area) = A:(2, 84), B:(5, 72), C:(3, 110).
#      Original task output order is A,C,B only if sorted by rooms? no, it is A,C,B because 2<3<5, and padding uses noise color N:
#      [A A N N N]
#      [C C C N N]
#      [B B B B B]
#      Combo output must sort by area instead, giving B,A,C, while padding is black 0:
#      [B B B B B]
#      [A A 0 0 0]
#      [C C C 0 0]
#      The bar lengths still come from room counts, but both the row permutation and the visible filler color change.
# b) Complexity calibration:
#    - The task is not simplified to pure geometry: the solver still needs room counting to determine each bar length.
#    - The generator rejects samples where area order matches room-count order, so the new ranking rule is visible.
#    - Distinct frame areas are enforced to avoid hiding the new rule behind tie-breaking logic.
# c) Old implicit assumptions reviewed:
#    - In the original solver, `rooms == 0` identified the non-frame noise color and also the output padding color.
#      The first part still holds; the second part no longer holds because output padding is always black.
#    - The original `for rooms in sorted(...)` loop coupled ordering with bar length. In the combo variant these are split:
#      area controls row order, room count controls prefix length, black controls suffix fill.
#    - `_cover_order` is kept only as a safe fallback if equal recovered areas ever occur; the generator avoids such cases.
# d) Placement legality:
#    - Every frame is built in-bounds with exact target height and width, then placed fully inside the canvas.
#    - Optional overlaps are modest and validated through the solver oracle, so reconstruction remains stable.
# e) Visual observability:
#    - The examples are forced to contain at least one inversion between room-count order and area order, so a human can see
#      that row lengths are not monotone in the output order.
#    - Because padding is black while the input noise color is non-black, the D6 change is directly visible in every output row
#      that is shorter than the maximum width.
# f) Visual cleanliness:
#    - Rendering keeps the original clean black canvas and 1px wall style.
#    - Overlaps are limited to a few pairs; heavy stacking is avoided so bbox extent and frame structure remain legible.
# g) Color inference:
#    - No hidden color mapping is introduced. Frame colors remain identity labels; black suffix cells are structural padding,
#      not a learned per-example code.
# h) Generator validation metric:
#    - After implementation we measure how often the color order produced by area sorting differs from room-count sorting.
#      We also check that output padding is black rather than the noise color. If both statistics hold broadly, the fusion is active.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes each true frame still reconstructs into a non-empty rectilinear wall layout whose recovered bbox is a
#      good proxy for the intended outer rectangle. The generator supports that by exact-size construction and oracle validation.


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


def _frame_signature(g: np.ndarray, color: int) -> tuple[int, int, int, np.ndarray]:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, r1, c1 = b
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    rooms = _count_enclosed_regions(walls)

    full = np.zeros_like(g, dtype=bool)
    full[r0 : r1 + 1, c0 : c1 + 1] = walls
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    return int(color), int(rooms), int(area), full


def _cover_order(
    g: np.ndarray, group: list[tuple[int, int, int, np.ndarray]]
) -> list[tuple[int, int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _rooms, _area, _walls in group}
    indegree: dict[int, int] = {color: 0 for color, _rooms, _area, _walls in group}

    for i, (ca, _ra, _aa, wa) in enumerate(group):
        for cb, _rb, _ab, wb in group[i + 1 :]:
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

    ordered: list[tuple[int, int, int, np.ndarray]] = []
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
    Convert each colored partition frame into a horizontal bar, ordered by recovered frame area and padded with black.

    Input structure:
    - Black background with several axis-aligned colored wall drawings.
    - One extra non-black color appears as scattered damage/noise and may also replace some wall pixels.
    - Each real drawing is a rectangular line frame that encloses one or more background rooms.

    Rule:
    - For each non-background color, reconstruct the rectilinear wall layout from dense row/column evidence.
    - Count the enclosed 4-connected background regions to get that row's colored-prefix length.
    - The lone color with zero enclosed regions is the noise color, but it is not used for output padding.
    - Sort the true frame colors by the area of their recovered bounding boxes; if two areas tie, break the tie by which frame is underneath the other.
    - Emit one row per frame, fill the first `rooms` cells with that frame color, and fill the remaining cells with black.

    Key insight:
    - The solver must combine topology and geometry: room count still sets each bar length, but the row order comes from outer
      spatial extent, and the right-hand filler no longer carries any information from the scattered noise color.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    stats: list[tuple[int, int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        stats.append(_frame_signature(g, int(color)))

    drawings = [(color, rooms, area, walls) for color, rooms, area, walls in stats if rooms > 0]
    noise_colors = [color for color, rooms, _area, _walls in stats if rooms == 0]
    assert drawings
    assert len(noise_colors) == 1

    ordered: list[tuple[int, int, int, np.ndarray]] = []
    for area in sorted({area for _color, _rooms, area, _walls in drawings}):
        group = [item for item in drawings if item[2] == area]
        ordered.extend(_cover_order(g, group))

    width = max(rooms for _color, rooms, _area, _walls in ordered)
    out = np.full((len(ordered), width), bg, dtype=int)
    for r, (color, rooms, _area, _walls) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (gpt-5) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_GRID_SIZE_BY_FRAMES = {
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.25
_EXTRA_OVERLAP_RANGE = (0, 2)
_OVERLAP_RETRY = 100
_PLACEMENT_RETRIES = 20
_SPEC_RETRIES = 250
_GEN_RETRIES = 300
_FRAME_COUNT_CYCLE = list(range(_FRAME_COUNT_RANGE[0], _FRAME_COUNT_RANGE[1] + 1))
_PREFERRED_FRAME_RETRIES = 180
_FALLBACK_FRAME_RETRIES = 60
_generate_call_count = 0


def _random_gaps_exact(n: int, min_gap: int, max_gap: int, total: int) -> list[int] | None:
    if n * min_gap > total or n * max_gap < total:
        return None
    gaps = [min_gap] * n
    remaining = total - n * min_gap
    order = list(range(n))
    random.shuffle(order)
    extra_cap = max_gap - min_gap
    for i, idx in enumerate(order[:-1]):
        slots_left = len(order) - i - 1
        min_add = max(0, remaining - slots_left * extra_cap)
        max_add = min(extra_cap, remaining)
        add = random.randint(min_add, max_add)
        gaps[idx] += add
        remaining -= add
    last = order[-1]
    gaps[last] += remaining
    if gaps[last] > max_gap:
        return None
    return gaps


def _feasible_configs(target_rooms: int, target_h: int, target_w: int) -> list[tuple[int, int, int]]:
    configs = []
    min_gap, max_gap = _CELL_GAP_RANGE
    for nv in range(2, 6):
        for nh in range(2, 5):
            cells = (nv - 1) * (nh - 1)
            if not (target_rooms <= cells <= target_rooms + 4):
                continue
            n_row_gaps = nh - 1
            n_col_gaps = nv - 1
            if not (n_row_gaps * min_gap <= target_h - 1 <= n_row_gaps * max_gap):
                continue
            if not (n_col_gaps * min_gap <= target_w - 1 <= n_col_gaps * max_gap):
                continue
            configs.append((nv, nh, cells))
    if configs:
        return configs

    for nv in range(2, 7):
        for nh in range(2, 6):
            cells = (nv - 1) * (nh - 1)
            if cells < target_rooms:
                continue
            n_row_gaps = nh - 1
            n_col_gaps = nv - 1
            if not (n_row_gaps * min_gap <= target_h - 1 <= n_row_gaps * max_gap):
                continue
            if not (n_col_gaps * min_gap <= target_w - 1 <= n_col_gaps * max_gap):
                continue
            configs.append((nv, nh, cells))
    return configs


def _build_frame(target_rooms: int, target_h: int, target_w: int) -> np.ndarray | None:
    if target_h < 4 or target_w < 4:
        return None
    if target_rooms == 1:
        walls = np.zeros((target_h, target_w), dtype=bool)
        walls[0, :] = True
        walls[target_h - 1, :] = True
        walls[:, 0] = True
        walls[:, target_w - 1] = True
        return walls

    possible_configs = _feasible_configs(target_rooms, target_h, target_w)
    if not possible_configs:
        return None

    n_vcols, n_hrows, total_cells = random.choice(possible_configs)
    n_row_gaps = n_hrows - 1
    n_col_gaps = n_vcols - 1
    min_gap, max_gap = _CELL_GAP_RANGE

    row_gaps = _random_gaps_exact(n_row_gaps, min_gap, max_gap, target_h - 1)
    col_gaps = _random_gaps_exact(n_col_gaps, min_gap, max_gap, target_w - 1)
    if row_gaps is None or col_gaps is None:
        return None

    vcol_positions = [0]
    for gap in col_gaps:
        vcol_positions.append(vcol_positions[-1] + gap)
    hrow_positions = [0]
    for gap in row_gaps:
        hrow_positions.append(hrow_positions[-1] + gap)

    walls = np.zeros((target_h, target_w), dtype=bool)
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


def _count_inversions(values: list[int]) -> int:
    count = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                count += 1
    return count


def _candidate_dims(max_h: int, max_w: int) -> list[tuple[int, int]]:
    dims = []
    for h in range(6, max_h + 1):
        for w in range(8, max_w + 1):
            area = h * w
            if 60 <= area <= max_h * max_w:
                dims.append((h, w))
    random.shuffle(dims)
    return dims


def _choose_frame_specs(
    frame_count: int, room_counts: list[int], max_h: int, max_w: int
) -> list[tuple[int, int, int]] | None:
    dims = _candidate_dims(max_h, max_w)
    compatibility = {room: [] for room in room_counts}
    for h, w in dims:
        area = h * w
        for room in room_counts:
            if _feasible_configs(room, h, w):
                compatibility[room].append((h, w, area))

    if any(not compatibility[room] for room in room_counts):
        return None

    min_inversions = 1 if frame_count == 3 else 2
    for _ in range(_SPEC_RETRIES):
        used_areas: set[int] = set()
        specs: list[tuple[int, int, int]] = []
        for room in random.sample(room_counts, len(room_counts)):
            options = [item for item in compatibility[room] if item[2] not in used_areas]
            if not options:
                specs = []
                break
            h, w, area = random.choice(options)
            used_areas.add(area)
            specs.append((room, h, w))
        if not specs:
            continue

        by_area = sorted(specs, key=lambda item: item[1] * item[2])
        room_sequence = [room for room, _h, _w in by_area]
        if _count_inversions(room_sequence) < min_inversions:
            continue
        return specs
    return None


def _force_overlap(positions, i, j, frames, grid_h, grid_w):
    fi_h, fi_w = frames[i][3].shape
    fj_h, fj_w = frames[j][3].shape
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
        if max(ri, rj) <= min(ri + fi_h - 1, rj + fj_h - 1) and max(ci, cj) <= min(ci + fi_w - 1, cj + fj_w - 1):
            positions[j] = (rj, cj)
            return


def _render_frames(frames, positions, render_order, grid_h, grid_w):
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][3]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        grid[rs + r0, cs + c0] = color

    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx in render_order:
        r0, c0 = positions[idx]
        wall_arr = frames[idx][3]
        color = frames[idx][0]
        rs, cs = np.where(wall_arr)
        abs_rs, abs_cs = rs + r0, cs + c0
        visible = grid[abs_rs, abs_cs] == color
        frame_mask[abs_rs[visible], abs_cs[visible]] = True
    return grid, frame_mask


def _noise_fraction_range(frame_count: int) -> tuple[float, float]:
    if frame_count >= 5:
        return (0.02, 0.05)
    if frame_count == 4:
        return (0.03, 0.07)
    return _NOISE_FRACTION_RANGE


def _wall_damage_fraction(frame_count: int) -> float:
    if frame_count >= 5:
        return 0.12
    if frame_count == 4:
        return 0.18
    return _WALL_DAMAGE_FRACTION


def _extra_overlap_range(frame_count: int) -> tuple[int, int]:
    if frame_count >= 5:
        return (0, 0)
    if frame_count == 4:
        return (0, 1)
    return _EXTRA_OVERLAP_RANGE


def _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w, frame_count: int):
    total_cells = grid_h * grid_w
    noise_fraction = random.uniform(*_noise_fraction_range(frame_count))
    noise_count = int(total_cells * noise_fraction)

    nonwall_rs, nonwall_cs = np.where(~frame_mask)
    perm = np.random.permutation(len(nonwall_rs))
    n_nonwall = min(noise_count, len(perm))
    sel = perm[:n_nonwall]
    grid[nonwall_rs[sel], nonwall_cs[sel]] = noise_color
    placed = n_nonwall

    wall_rs, wall_cs = np.where(frame_mask)
    n_wall = len(wall_rs)
    max_damage = min(noise_count - placed, int(_wall_damage_fraction(frame_count) * n_wall))
    if max_damage > 0:
        damage_perm = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_perm], wall_cs[damage_perm]] = noise_color


def _gen_attempt(frame_count: int | None = None) -> np.ndarray | None:
    if frame_count is None:
        frame_count = random.randint(*_FRAME_COUNT_RANGE)
    room_counts = random.sample(list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1)), frame_count)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    specs = _choose_frame_specs(frame_count, room_counts, f_max_h, f_max_w)
    if specs is None:
        return None

    frames = []
    for color, (target_rooms, target_h, target_w) in zip(frame_colors, specs):
        frame_data = _build_frame(target_rooms, target_h=target_h, target_w=target_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, fh * fw, frame_data))

    frame_info = sorted(
        [(area, rooms, color) for color, rooms, area, _frame in frames],
        key=lambda item: item[0],
    )
    room_order_colors = [color for rooms, color in sorted((rooms, color) for color, rooms, _area, _frame in frames)]
    area_order_colors = [color for _area, _rooms, color in frame_info]
    if room_order_colors == area_order_colors:
        return None

    expected = np.zeros((frame_count, max(room_counts)), dtype=int)
    for r, (_area, rooms, color) in enumerate(frame_info):
        expected[r, :rooms] = color

    overlap_range = _extra_overlap_range(frame_count)
    non_tied = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count)]
    for _ in range(_PLACEMENT_RETRIES):
        positions = [None] * frame_count
        for idx in sorted(range(frame_count), key=lambda i: -frames[i][3].size):
            fh, fw = frames[idx][3].shape
            positions[idx] = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))

        n_extra = random.randint(*overlap_range)
        if n_extra > 0 and non_tied:
            random.shuffle(non_tied)
            for i, j in non_tied[: min(n_extra, len(non_tied))]:
                _force_overlap(positions, i, j, frames, grid_h, grid_w)

        render_order = list(range(frame_count))
        random.shuffle(render_order)
        grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
        _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w, frame_count)

        try:
            output = main(grid)
        except Exception:
            continue

        if np.array_equal(output, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    global _generate_call_count

    preferred = _FRAME_COUNT_CYCLE[_generate_call_count % len(_FRAME_COUNT_CYCLE)]
    _generate_call_count += 1
    ordered_counts = [preferred] + [count for count in _FRAME_COUNT_CYCLE if count != preferred]

    retries_used = 0
    for idx, frame_count in enumerate(ordered_counts):
        budget = _PREFERRED_FRAME_RETRIES if idx == 0 else _FALLBACK_FRAME_RETRIES
        budget = min(budget, _GEN_RETRIES - retries_used)
        for _ in range(budget):
            result = _gen_attempt(frame_count=frame_count)
            retries_used += 1
            if result is not None:
                return result
        if retries_used >= _GEN_RETRIES:
            break

    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
