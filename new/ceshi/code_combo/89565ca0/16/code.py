# Combo variant of 89565ca0 (index 16)
# combo_spec: D1/0+D5/3+D6/0
# Name: 面积排序+点阵编码+黑色填充

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Input-side object semantics stay the same as the base task: each non-noise color may be a damaged,
#      overlapping rectilinear frame whose walls enclose some number of rooms.
#    - Original pipeline for one example could be:
#      frames A/B/C have (rooms, bbox_area) = (2, 120), (5, 84), (3, 143).
#      Original output sorts by rooms and draws solid bars padded by the noise color:
#      A -> [A,A,noise,noise,noise], C -> [C,C,C,noise,noise], B -> [B,B,B,B,B].
#    - Combo pipeline must instead sort by bbox area and then encode room count as spaced dots on black:
#      B first, then A, then C; rows become
#      B -> [B,0,B,0,B,0,B,0,B]
#      A -> [A,0,A,0,0,0,0,0,0]
#      C -> [C,0,C,0,C,0,0,0,0]
#    - So the changed stages are coupled: room count still determines how many marks appear, but bbox area
#      decides row order and black replaces the old non-black padding everywhere in the output.
# b) Complexity calibration:
#    - The variant must remain human-readable from a few examples, so the dot spacing is fixed at exactly one
#      black cell between consecutive dots. This makes the encoding sparse but not ambiguous.
#    - The generator intentionally rejects cases where area order matches room-count order, otherwise the new
#      ranking rule would be visually invisible and the task would collapse toward the original.
#    - Area ties are also avoided in generation, because tie-breaking through overlap is harder to infer than
#      the primary rule and would obscure the intended "look at object extent" signal.
# c) Implicit assumption audit:
#    - `_count_enclosed_regions` and `_reconstruct_walls` only depend on the input frame geometry; they remain
#      valid because D1/D5/D6 do not alter the input object type.
#    - The original `rooms == 0` heuristic for identifying the scattered noise color still holds, since the
#      combo changes only output ordering/encoding/fill, not the fact that the noise color forms no frame.
#    - The old solver conflated sort key, row length, and padding color. That no longer works:
#      area is now the sort key, room count controls dot multiplicity, and output fill is always black.
#    - `_cover_order` is retained only as a defensive equal-area fallback in `main()`. The generator guarantees
#      unique target areas so this branch is normally inactive, which keeps the visible rule simple.
# d) Placement legality:
#    - Every frame is built fully in-bounds with a valid rectangular outer wall.
#    - Placement samples top-left positions that keep whole bboxes on the canvas.
#    - Optional overlaps are modest and oracle-validated; no output objects overlap because rows are separate.
# e) Visual observability:
#    - Across examples, row lengths are no longer monotone with row order, so a human cannot solve by counting
#      rooms alone. They must notice that smaller bboxes come first even when they emit longer dot rows.
#    - The black fill is directly observable because the extra scattered input color never appears in output.
#      Internal gaps between dots reinforce that the background color is structural, not accidental.
# f) Visual cleanliness:
#    - Output rows use exactly one frame color plus black, avoiding multicolor clutter.
#    - Fixed spacing and left anchoring keep the tally rows easy to compare.
#    - Input overlaps remain limited so frames are still individually recoverable by their dominant wall traces.
# g) Change inferability:
#    - No hidden color remapping is introduced. Frame colors remain identity labels.
#    - The only color-behavior change is that output background is always black, which is strongly cued by the
#      already-black canvas in the input and by repeated black separators between same-color dots.
# h) Generator validation metric:
#    - After implementation we measure how often area-sorted color order differs from room-sorted color order,
#      and how often outputs contain internal black gaps between dots while excluding the noise color entirely.
#      Both metrics must stay high; otherwise the combo has visually degenerated.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes reconstructed bboxes are stable enough that recovered area matches the intended frame
#      footprint. The generator enforces this with exact-size frame construction and solver-oracle rejection.
#    - `main()` assumes at least one true frame has rooms > 0 and exactly one non-frame color has rooms == 0.
#      The generator uses one dedicated noise color and 3-5 true frame colors to guarantee that setup.


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


def _dot_output_width(max_rooms: int) -> int:
    return int(2 * max_rooms - 1)


def _render_dot_rows(rows: list[tuple[int, int]]) -> np.ndarray:
    max_rooms = max(rooms for _color, rooms in rows)
    out = np.zeros((len(rows), _dot_output_width(max_rooms)), dtype=int)
    for r, (color, rooms) in enumerate(rows):
        cols = np.arange(0, 2 * int(rooms), 2, dtype=int)
        out[r, cols] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each colored partitioned frame into a horizontal dot tally, ordered by recovered bbox area and drawn on black.

    Input structure:
    - Black background with several axis-aligned colored wall drawings.
    - One extra non-black color appears as scattered noise and may also overwrite some wall pixels.
    - Each real drawing is a damaged rectangular partition frame that encloses one or more background rooms.

    Rule:
    - For each non-background color, reconstruct the rectilinear wall layout from dense row and column evidence.
    - Count the enclosed 4-connected background regions for that color and recover its visible bbox area.
    - The lone color with zero enclosed regions is only the noise color; it is not rendered in the output.
    - Sort the true frame colors by reconstructed bbox area in ascending order; if areas tie, use cover order as a fallback.
    - Output one row per true frame on a black canvas, placing one dot of the frame color at columns 0, 2, 4, ... for each enclosed room.

    Key insight:
    - The output still encodes room counts, but only as the number of evenly spaced dots. Row order is controlled by frame extent rather than room count, and every unused output cell stays black instead of inheriting the scattered noise color.
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

    return _render_dot_rows([(color, rooms) for color, rooms, _area, _walls in ordered])


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
_SPEC_RETRIES = 250
_GEN_RETRIES = 300


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
        ok = True
        for room in random.sample(room_counts, len(room_counts)):
            options = [item for item in compatibility[room] if item[2] not in used_areas]
            if not options:
                ok = False
                break
            h, w, area = random.choice(options)
            used_areas.add(area)
            specs.append((room, h, w))
        if not ok:
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


def _expected_output(frame_info: list[tuple[int, int, int]]) -> np.ndarray:
    rows = [(color, room_count) for _area, room_count, color in frame_info]
    return _render_dot_rows(rows)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    room_counts = random.sample(list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1)), frame_count)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    _h_min, h_max, _w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(cfg[0], h_max)
    grid_w = random.randint(cfg[2], w_max)

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

    positions = [None] * frame_count
    for idx in sorted(range(frame_count), key=lambda i: -frames[i][3].size):
        fh, fw = frames[idx][3].shape
        positions[idx] = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))

    non_tied = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count)]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
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

    frame_info = sorted((area, rooms, color) for color, rooms, area, _frame in frames)
    expected = _expected_output(frame_info)

    room_order_colors = [color for rooms, color in sorted((rooms, color) for color, rooms, _area, _frame in frames)]
    area_order_colors = [color for area, _rooms, color in frame_info]
    if room_order_colors == area_order_colors:
        return None

    if noise_color in set(int(v) for v in np.unique(expected)):
        return None
    if not np.any(expected == 0):
        return None

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
