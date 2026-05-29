# Variant of 89565ca0
# Dimension: D1 — 排序依据：封闭区域数量 (match_cond)
# Change: 按每个框架的封闭4-连通背景区域数量升序排列... → 按框架的内部网格线数量排序

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


"""
Design analysis for the D1 variant
==================================

a) Concrete behavior:
   - The input still contains several noisy, overlapping colored rectilinear frames on black.
   - Each true frame still encloses one or more rooms, so the output rows remain horizontal bars whose lengths equal
     those room counts.
   - The change is only in row order: instead of sorting bars by room count, sort them by the number of recognizable
     grid lines in the reconstructed frame, defined as `#horizontal line positions + #vertical line positions`.
   - Example trace:
     original logic: frame A has 2 rooms / 7 lines, frame B has 4 rooms / 6 lines -> output order A then B.
     variant logic: the same two bars still have lengths 2 and 4, but now B comes before A because 6 < 7.
     The visible difference is therefore in row order, not bar length encoding.

b) Complexity calibration:
   - If line count and room count are always aligned, the variant visually collapses back to the original task.
   - The generator therefore samples `(room_count, line_count)` pairs with at least one inversion:
     some frame must have more rooms but fewer lines than another frame.
   - Line counts are kept in the small, repeated range 4-9 so humans can compare them by eye across 2-4 examples.

c) Implicit assumptions audit:
   - Original `main()` assumed "sorting key == bar length". That no longer holds, so sorting and rendering must split.
   - Original tie handling depended on equal room counts. In the variant, overlaps only matter for equal line counts.
   - Noise identification via `rooms == 0` still remains valid because the output bars still encode room counts and
     the scattered noise color does not reconstruct into a closed frame.
   - `_reconstruct_walls()` is still safe because the frame geometry and damage model stay rectilinear.

d) Placement legality:
   - Frames stay within bounds and are placed exactly as before.
   - A forced overlap is now required for equal line-count pairs, because cover order only breaks ties for the new key.
   - Generation is still oracle-checked by `main()`, so any illegal placement or ambiguous tie ordering is rejected.

e) Visual observability:
   - The new key is visible: a solver can count line positions directly from the scaffold.
   - The generator avoids degenerate examples where sorting by lines and sorting by rooms would match exactly.
   - This makes the D1 change inferable from the sample pairs rather than hidden behind unchanged outputs.

f) Visual cleanliness:
   - No new mark types are introduced; the puzzle still consists only of frames, overlap damage, and one noise color.
   - The line-count range is modest, so objects stay readable and do not become over-fragmented.

g) Color inference:
   - No color-id-based behavior is introduced.
   - Colors still only label which frame a bar belongs to; ordering depends purely on visible geometry.
"""


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


def _detect_line_positions(mask: np.ndarray) -> tuple[list[int], list[int]]:
    h, _w = mask.shape
    col_counts = mask.sum(axis=0)
    v_threshold = max(3, math.ceil(0.45 * h))
    vcols = [int(c) for c, n in enumerate(col_counts) if int(n) >= v_threshold]
    if len(vcols) < 2:
        return [], []

    row_counts = mask.sum(axis=1)
    h_threshold = max(3, len(vcols) + 1)
    hrows = [int(r) for r, n in enumerate(row_counts) if int(n) >= h_threshold]
    if len(hrows) < 2:
        return vcols, []
    return vcols, hrows


def _reconstruct_walls(mask: np.ndarray) -> np.ndarray:
    vcols, hrows = _detect_line_positions(mask)
    if len(vcols) < 2 or len(hrows) < 2:
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


def _count_grid_lines(walls: np.ndarray) -> int:
    vcols, hrows = _detect_line_positions(walls)
    if len(vcols) < 2 or len(hrows) < 2:
        return 0
    return int(len(vcols) + len(hrows))


def _frame_stats_for_color(g: np.ndarray, color: int) -> tuple[int, int, np.ndarray]:
    b = bbox(g == int(color))
    assert b is not None
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    rooms = _count_enclosed_regions(walls)
    line_count = _count_grid_lines(walls)
    return int(rooms), int(line_count), walls


def _walls_for_color(g: np.ndarray, color: int) -> np.ndarray:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, r1, c1 = b
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    out = np.zeros_like(g, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = walls
    return out


def _cover_order(
    g: np.ndarray, group: list[tuple[int, int, int, np.ndarray]]
) -> list[tuple[int, int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _rooms, _lines, _walls in group}
    indegree: dict[int, int] = {color: 0 for color, _rooms, _lines, _walls in group}

    for i, (ca, _ra, _la, wa) in enumerate(group):
        for cb, _rb, _lb, wb in group[i + 1 :]:
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
    Convert each colored frame into a horizontal bar whose length is its enclosed-room count, but order the bars by grid-line count.

    Input structure:
    - Black background containing several colored rectilinear frame drawings plus one scattered noise color.
    - Real drawings may overlap, and both overlaps and noise erase some wall pixels.
    - Each real drawing still forms a partitioned rectangular scaffold with at least one enclosed room.

    Rule:
    - For each non-black color, reconstruct its horizontal and vertical wall segments from dense row and column evidence.
    - Use the reconstructed scaffold to count its recognizable horizontal and vertical line positions; this total is the sorting key.
    - Also count the enclosed 4-connected background regions; this room count still determines the bar length, and the only color with zero rooms is the pad/noise color.
    - Sort the drawing colors by grid-line count; equal keys are resolved by cover order inferred from reconstructed overlaps.

    Key insight:
    - After rectilinear reconstruction, the number of line positions is easier to read directly from the scaffold than the number of rooms, and the two quantities need not rank frames the same way.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    stats: list[tuple[int, int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        rooms, line_count, _walls_local = _frame_stats_for_color(g, int(color))
        walls = _walls_for_color(g, int(color))
        stats.append((int(color), int(rooms), int(line_count), walls))

    drawings = [(color, rooms, line_count, walls) for color, rooms, line_count, walls in stats if rooms > 0]
    pad_colors = [color for color, rooms, _line_count, _walls in stats if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, int, np.ndarray]] = []
    for line_count in sorted({line_count for _color, _rooms, line_count, _walls in drawings}):
        group = [item for item in drawings if item[2] == line_count]
        ordered.extend(_cover_order(g, group))

    width = max(rooms for _color, rooms, _line_count, _walls in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms, _line_count, _walls) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (variant) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_LINE_COUNT_VALUES = (4, 5, 6, 7, 8, 9)
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)
_NOISE_FRACTION_RANGE = (0.04, 0.09)
_WALL_DAMAGE_FRACTION = 0.25
_EXTRA_OVERLAP_RANGE = (0, 3)
_OVERLAP_RETRY = 100
_FRAME_BUILD_ATTEMPTS = 80
_SPEC_ATTEMPTS = 250
_GEN_RETRIES = 400

_FEASIBLE_ROOM_COUNTS_BY_LINES: dict[int, tuple[int, ...]] = {
    4: (1,),
    5: (1, 2),
    6: (1, 2, 3, 4),
    7: (1, 2, 3, 4, 5, 6),
    8: (1, 2, 3, 4, 5, 6),
    9: (1, 2, 3, 4, 5, 6),
}


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


def _build_frame(target_rooms: int, target_lines: int, max_h: int = 22, max_w: int = 25) -> np.ndarray | None:
    configs = []
    for nv in range(2, 6):
        for nh in range(2, 5):
            cells = (nv - 1) * (nh - 1)
            base_lines = nv + nh
            if target_rooms <= cells <= target_rooms + 8 and base_lines >= target_lines:
                configs.append((nv, nh, cells, base_lines))
    if not configs:
        return None

    random.shuffle(configs)
    for n_vcols, n_hrows, total_cells, _base_lines in configs:
        for _ in range(_FRAME_BUILD_ATTEMPTS):
            n_row_gaps = n_hrows - 1
            n_col_gaps = n_vcols - 1
            min_gap, max_gap = _CELL_GAP_RANGE

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
            if cells_to_merge < 0:
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
                continue
            if _count_enclosed_regions(walls) != target_rooms:
                continue
            if _count_grid_lines(walls) != target_lines:
                continue
            return walls

    return None


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
        if max(ri, rj) <= min(ri + fi_h - 1, rj + fj_h - 1) and max(ci, cj) <= min(ci + fi_w - 1, cj + fj_w - 1):
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


def _has_line_room_inversion(room_counts: list[int], line_counts: list[int]) -> bool:
    for i in range(len(room_counts)):
        for j in range(i + 1, len(room_counts)):
            if (line_counts[i] < line_counts[j] and room_counts[i] > room_counts[j]) or (
                line_counts[i] > line_counts[j] and room_counts[i] < room_counts[j]
            ):
                return True
    return False


def _sample_specs(frame_count: int) -> list[tuple[int, int]] | None:
    for _ in range(_SPEC_ATTEMPTS):
        available_lines = list(_LINE_COUNT_VALUES)
        allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
        if allow_tie:
            distinct_needed = frame_count - 1
            chosen = random.sample(available_lines, distinct_needed)
            line_counts = chosen + [random.choice(chosen)]
        else:
            line_counts = random.sample(available_lines, frame_count)
        random.shuffle(line_counts)

        room_counts = [random.choice(_FEASIBLE_ROOM_COUNTS_BY_LINES[line_count]) for line_count in line_counts]
        if len(set(room_counts)) < 2:
            continue
        if max(Counter(room_counts).values()) > 2:
            continue
        if not _has_line_room_inversion(room_counts, line_counts):
            continue
        return list(zip(room_counts, line_counts))
    return None


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    specs = _sample_specs(frame_count)
    if specs is None:
        return None

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames = []
    room_counts = []
    line_counts = []
    for color, (target_rooms, target_lines) in zip(frame_colors, specs):
        frame_data = _build_frame(target_rooms, target_lines, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, frame_data, target_lines))
        room_counts.append(target_rooms)
        line_counts.append(target_lines)

    positions = [None] * frame_count
    for idx in sorted(range(frame_count), key=lambda i: -frames[i][2].size):
        fh, fw = frames[idx][2].shape
        positions[idx] = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))

    tied_pairs = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count) if line_counts[i] == line_counts[j]]
    for i, j in tied_pairs:
        _force_overlap(positions, i, j, frames, grid_h, grid_w)

    non_tied = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count) if line_counts[i] != line_counts[j]]
    n_extra = random.randint(*_EXTRA_OVERLAP_RANGE)
    if n_extra > 0 and non_tied:
        random.shuffle(non_tied)
        for i, j in non_tied[: min(n_extra, len(non_tied))]:
            _force_overlap(positions, i, j, frames, grid_h, grid_w)

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    recovered = {}
    for color in colors_present(grid, exclude={0}):
        rooms, line_count, _walls = _frame_stats_for_color(grid, int(color))
        recovered[int(color)] = (int(rooms), int(line_count))

    if set(recovered) != set(frame_colors) | {noise_color}:
        return None
    pad_colors = [color for color, (rooms, _line_count) in recovered.items() if rooms == 0]
    if len(pad_colors) != 1 or pad_colors[0] != noise_color:
        return None

    recovered_drawings = []
    recovered_room_counts = []
    recovered_line_counts = []
    for idx, color in enumerate(frame_colors):
        rooms, line_count = recovered[int(color)]
        if (
            rooms < _ROOM_COUNT_RANGE[0]
            or rooms > _ROOM_COUNT_RANGE[1]
            or line_count not in _LINE_COUNT_VALUES
        ):
            return None
        recovered_drawings.append((idx, int(color), rooms, line_count))
        recovered_room_counts.append(rooms)
        recovered_line_counts.append(line_count)

    if not _has_line_room_inversion(recovered_room_counts, recovered_line_counts):
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        (line_count, render_order.index(i), rooms, color)
        for i, color, rooms, line_count in recovered_drawings
    )
    max_rooms = max(rooms for _i, _color, rooms, _line_count in recovered_drawings)
    expected = np.full((frame_count, max_rooms), noise_color, dtype=int)
    for r, (_line_count, _render_rank, rooms, color) in enumerate(frame_info):
        expected[r, :rooms] = color

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
