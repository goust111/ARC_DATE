# Combo variant of 89565ca0 (index 9)
# combo_spec: D1/3+D7/3
# Name: 按网格线数排序+按墙壁线段数编码

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace:
   - The input still shows several colored rectilinear frames on black, plus one
     scattered noise color and occasional frame overlaps.
   - The fused rule changes two different stages:
       1. row order: sort by reconstructed grid-line count
          = #horizontal line positions + #vertical line positions.
       2. bar length: encode reconstructed wall-segment count
          = #straight spans between adjacent intersections.
   - Small trace:
       frame A and frame B both reconstruct to 6 total lines, but A has 10 wall
       segments while B has 12 wall segments.
       Original task: equal-room ties were rare and bar length still meant rooms.
       This combo: A and B must be grouped together because their line counts tie,
       yet their bar lengths differ because segment counts differ. Their relative
       row order is decided only by which frame lies underneath at the overlap.

b) Complexity calibration:
   - In this geometry family, segment count usually increases with line count, so a
     clean cross-line inversion is not stable enough to rely on.
   - The generator therefore enforces one equal-line-count pair with different
     segment counts, and forces the overlap tiebreak to place the longer or shorter
     bar first independently of its length.
   - Line-count values are kept in a modest range and only one tie pair is used, so
     examples stay readable while still exposing the fused rule.

c) Implicit-assumption audit:
   - Original main() assumed the sorting key and bar length used the same metric.
     That is no longer true, so the solver must extract both metrics separately.
   - Original noise detection used room_count == 0. Here bar length no longer uses
     rooms, so pad detection is safer when based on "valid frame signature exists"
     and therefore on segment_count > 0 instead.
   - Overlap tiebreaks used to matter for equal room counts; now they matter for
     equal line counts. The generator must therefore force overlap for line ties,
     not room ties.
   - Cover-order inference assumes at least one overlap pixel remains visible. The
     generator protects tied-pair overlap witnesses from later noise damage.

d) Placement legality:
   - Frames are placed in-bounds.
   - Placement starts from non-overlapping bboxes, then a single-pixel corner
     overlap is introduced only for equal-line-count pairs so tiebreaks are legal
     and unambiguous.
   - Additional overlaps are avoided to keep reconstruction stable and the scene
     visually cleaner.

e) Visual observability:
   - A human can count visible grid lines from the frame scaffold and compare that
     to the output row order.
   - A human can also see that bar length is not the tie-breaking rule: two frames
     with the same line count can produce different bar lengths, yet their row
     order follows overlap depth rather than length.

f) Visual cleanliness:
   - No new symbol types are introduced; the puzzle still only uses black
     background, thin colored walls, overlap occlusion, and one noise color.
   - Noise and wall damage are kept moderate, and only tie overlaps are forced, so
     the frames remain easy to parse.

g) Change inferability:
   - No invisible numeric color-id rule is used.
   - Colors only label which frame/bar corresponds to which object.
   - The fused behavior is supported structurally: row grouping follows line counts,
     filled length follows segment counts, and the generator rejects samples where
     the critical equal-line different-length tie disappears after rendering.
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


def _extract_grid_lines(mask: np.ndarray) -> tuple[list[int], list[int]]:
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
        return [], []
    return hrows, vcols


def _reconstruct_walls(mask: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    hrows, vcols = _extract_grid_lines(mask)
    if len(hrows) < 2 or len(vcols) < 2:
        return np.zeros_like(mask, dtype=bool), hrows, vcols

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

    return walls, hrows, vcols


def _touches_all_bbox_sides(walls: np.ndarray) -> bool:
    return bool(
        walls.shape[0] >= 2
        and walls.shape[1] >= 2
        and walls[0, :].any()
        and walls[-1, :].any()
        and walls[:, 0].any()
        and walls[:, -1].any()
    )


def _count_wall_segments(walls: np.ndarray, hrows: list[int], vcols: list[int]) -> int:
    count = 0
    for r in hrows:
        for c0, c1 in zip(vcols, vcols[1:]):
            if bool(walls[r, c0 : c1 + 1].all()):
                count += 1
    for c in vcols:
        for r0, r1 in zip(hrows, hrows[1:]):
            if bool(walls[r0 : r1 + 1, c].all()):
                count += 1
    return int(count)


def _frame_signature_for_color(g: np.ndarray, color: int) -> tuple[int, int, int, np.ndarray]:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, r1, c1 = b
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls, hrows, vcols = _reconstruct_walls(mask)
    if len(hrows) < 2 or len(vcols) < 2 or not _touches_all_bbox_sides(walls):
        return 0, 0, 0, np.zeros_like(g, dtype=bool)

    segments = _count_wall_segments(walls, hrows, vcols)
    rooms = _count_enclosed_regions(walls)
    if segments < 4 or rooms < 1:
        return 0, 0, 0, np.zeros_like(g, dtype=bool)

    line_count = int(len(hrows) + len(vcols))
    full_walls = np.zeros_like(g, dtype=bool)
    full_walls[r0 : r1 + 1, c0 : c1 + 1] = walls
    return int(line_count), int(segments), int(rooms), full_walls


def _cover_order(
    g: np.ndarray, group: list[tuple[int, int, int, int, np.ndarray]]
) -> list[tuple[int, int, int, int, np.ndarray]]:
    if len(group) == 1:
        return group

    by_color = {int(item[0]): item for item in group}
    above: dict[int, set[int]] = {color: set() for color, _lines, _segments, _rooms, _walls in group}
    indegree: dict[int, int] = {color: 0 for color, _lines, _segments, _rooms, _walls in group}

    for i, (ca, _la, _sa, _ra, wa) in enumerate(group):
        for cb, _lb, _sb, _rb, wb in group[i + 1 :]:
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

    ordered: list[tuple[int, int, int, int, np.ndarray]] = []
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
    Convert each reconstructed colored frame into a horizontal bar, ordering rows by grid-line count and sizing bars by wall-segment count.

    Input structure:
    - Black background with several colored axis-aligned line frames.
    - One extra non-black color appears only as scattered noise and wall damage.
    - Real frames may overlap, but each valid frame still reconstructs into a rectilinear outer box with optional internal dividers.

    Rule:
    - For each non-black color, rebuild the frame scaffold from dense row and column evidence inside that color's bbox.
    - From that scaffold, count its recognizable horizontal-plus-vertical line positions; this total determines row order.
    - Also count the independent wall segments between adjacent intersections; this determines the filled length of the row.
    - The unique color that does not reconstruct into a valid frame is the padding/noise color, and equal line counts are broken by cover order from overlaps.

    Key insight:
    - The same reconstructed scaffold exposes two different stable metrics: line count says where a bar goes, while segment count says how long that bar is, and those two rankings are intentionally not the same.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    stats: list[tuple[int, int, int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        line_count, segments, rooms, walls = _frame_signature_for_color(g, int(color))
        stats.append((int(color), int(line_count), int(segments), int(rooms), walls))

    drawings = [(color, line_count, segments, rooms, walls) for color, line_count, segments, rooms, walls in stats if segments > 0]
    pad_colors = [color for color, _line_count, segments, _rooms, _walls in stats if segments == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered: list[tuple[int, int, int, int, np.ndarray]] = []
    for line_count in sorted({line_count for _color, line_count, _segments, _rooms, _walls in drawings}):
        group = [item for item in drawings if item[1] == line_count]
        ordered.extend(_cover_order(g, group))

    width = max(segments for _color, _line_count, segments, _rooms, _walls in ordered)
    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, _line_count, segments, _rooms, _walls) in enumerate(ordered):
        out[r, :segments] = int(color)
    return out


# --------------- generator (combo variant) ---------------

_FRAME_COUNT_RANGE = (3, 4)
_ROOM_COUNT_RANGE = (1, 6)
_LINE_COUNT_VALUES = (4, 5, 6, 7)
_GRID_SIZE_BY_FRAMES = {
    4: (25, 30, 25, 30, 16, 18),
    3: (22, 30, 22, 30, 18, 22),
}
_CELL_GAP_RANGE = (3, 5)
_NOISE_FRACTION_RANGE = (0.03, 0.06)
_WALL_DAMAGE_FRACTION = 0.12
_GEN_RETRIES = 500
_SPEC_ATTEMPTS = 300
_PLACEMENT_ATTEMPTS = 200
_FRAME_BUILD_ATTEMPTS = 60


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
def _build_frame(
    target_line_count: int, target_segments: int, max_h: int = 22, max_w: int = 25
) -> tuple[np.ndarray, int] | None:
    configs: list[tuple[int, int, int]] = []
    for n_vcols in range(2, 6):
        for n_hrows in range(2, 5):
            if n_vcols + n_hrows < target_line_count:
                continue
            total_cells = (n_vcols - 1) * (n_hrows - 1)
            full_segments = n_hrows * (n_vcols - 1) + n_vcols * (n_hrows - 1)
            removed_needed = full_segments - target_segments
            room_count = total_cells - removed_needed
            if not (_ROOM_COUNT_RANGE[0] <= room_count <= min(_ROOM_COUNT_RANGE[1], total_cells)):
                continue
            if not (0 <= removed_needed <= total_cells - 1):
                continue
            min_gap, _max_gap = _CELL_GAP_RANGE
            if min_gap * (n_hrows - 1) > max_h - 1 or min_gap * (n_vcols - 1) > max_w - 1:
                continue
            configs.append((n_vcols, n_hrows, room_count))

    if not configs:
        return None

    random.shuffle(configs)
    for n_vcols, n_hrows, room_count in configs:
        for _ in range(_FRAME_BUILD_ATTEMPTS):
            min_gap, max_gap = _CELL_GAP_RANGE
            row_gaps = _random_gaps(n_hrows - 1, min_gap, max_gap, max_h - 1)
            col_gaps = _random_gaps(n_vcols - 1, min_gap, max_gap, max_w - 1)
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

            total_cells = (n_vcols - 1) * (n_hrows - 1)
            removed_needed = total_cells - room_count
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

            internal_segments: list[tuple[str, int, int, int, int, int]] = []
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
            removed = 0
            for seg in internal_segments:
                if removed >= removed_needed:
                    break
                if seg[0] == "h":
                    _, r, c0, c1, ca, cb = seg
                    if union(ca, cb):
                        walls[r, c0 + 1 : c1] = False
                        removed += 1
                else:
                    _, c, r0, r1, ca, cb = seg
                    if union(ca, cb):
                        walls[r0 + 1 : r1, c] = False
                        removed += 1

            if removed < removed_needed:
                continue
            calc_hrows, calc_vcols = _extract_grid_lines(walls)
            calc_segments = _count_wall_segments(walls, calc_hrows, calc_vcols)
            if calc_segments != target_segments:
                continue
            if _count_enclosed_regions(walls) != room_count:
                continue
            if len(calc_hrows) + len(calc_vcols) != target_line_count:
                continue
            return walls, int(room_count)

    return None


def _feasible_segment_counts_for_lines(line_count: int) -> tuple[int, ...]:
    values: set[int] = set()
    for segment_count in range(4, 18):
        for _ in range(10):
            built = _build_frame(line_count, segment_count, max_h=18, max_w=22)
            if built is not None:
                values.add(segment_count)
                break
    return tuple(sorted(values))


_SEGMENTS_BY_LINES = {line_count: _feasible_segment_counts_for_lines(line_count) for line_count in _LINE_COUNT_VALUES}


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[2], b[2]) and max(a[1], b[1]) <= min(a[3], b[3])


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if not _boxes_intersect(a, b):
        return 0
    h = min(a[2], b[2]) - max(a[0], b[0]) + 1
    w = min(a[3], b[3]) - max(a[1], b[1]) + 1
    return int(h * w)


def _frame_bbox(position: tuple[int, int], wall_arr: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = position
    h, w = wall_arr.shape
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_frames_without_overlap(frames, grid_h: int, grid_w: int):
    positions = [None] * len(frames)
    for idx in sorted(range(len(frames)), key=lambda i: -frames[i][2].size):
        fh, fw = frames[idx][2].shape
        placed = False
        for _ in range(_PLACEMENT_ATTEMPTS):
            candidate = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))
            cb = _frame_bbox(candidate, frames[idx][2])
            if all(
                positions[j] is None
                or not _boxes_intersect(cb, _frame_bbox(positions[j], frames[j][2]))
                for j in range(len(frames))
            ):
                positions[idx] = candidate
                placed = True
                break
        if not placed:
            return None
    return positions


def _force_corner_overlap(positions, i: int, j: int, frames, grid_h: int, grid_w: int) -> bool:
    ri, ci = positions[i]
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    candidates = [
        (ri + fi_h - 1, ci + fi_w - 1),
        (ri + fi_h - 1, ci - fj_w + 1),
        (ri - fj_h + 1, ci + fi_w - 1),
        (ri - fj_h + 1, ci - fj_w + 1),
    ]
    random.shuffle(candidates)
    ib = _frame_bbox(positions[i], frames[i][2])
    for rj, cj in candidates:
        if not (0 <= rj <= grid_h - fj_h and 0 <= cj <= grid_w - fj_w):
            continue
        jb = _frame_bbox((rj, cj), frames[j][2])
        if _overlap_area(ib, jb) != 1:
            continue
        if any(
            k not in {i, j}
            and positions[k] is not None
            and _boxes_intersect(jb, _frame_bbox(positions[k], frames[k][2]))
            for k in range(len(frames))
        ):
            continue
        positions[j] = (rj, cj)
        return True
    return False


def _overlap_mask_for_pairs(frames, positions, pairs, grid_h: int, grid_w: int) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for i, j in pairs:
        wi = frames[i][2]
        wj = frames[j][2]
        ri, ci = positions[i]
        rj, cj = positions[j]
        abs_i = np.zeros((grid_h, grid_w), dtype=bool)
        abs_j = np.zeros((grid_h, grid_w), dtype=bool)
        rs, cs = np.where(wi)
        abs_i[rs + ri, cs + ci] = True
        rs, cs = np.where(wj)
        abs_j[rs + rj, cs + cj] = True
        mask |= abs_i & abs_j
    return mask


def _boundary_mask(frames, positions, grid_h: int, grid_w: int) -> np.ndarray:
    mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx, (_color, _line_count, wall_arr, _segments, _rooms) in enumerate(frames):
        r0, c0 = positions[idx]
        h, w = wall_arr.shape
        cols = np.where(wall_arr[0, :])[0]
        mask[r0, c0 + cols] = True
        cols = np.where(wall_arr[h - 1, :])[0]
        mask[r0 + h - 1, c0 + cols] = True
        rows = np.where(wall_arr[:, 0])[0]
        mask[r0 + rows, c0] = True
        rows = np.where(wall_arr[:, w - 1])[0]
        mask[r0 + rows, c0 + w - 1] = True
    return mask


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


def _scatter_noise(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    protected_mask: np.ndarray,
    noise_color: int,
    grid_h: int,
    grid_w: int,
) -> None:
    total_cells = grid_h * grid_w
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)

    nonwall_rs, nonwall_cs = np.where((~frame_mask) & (~protected_mask))
    perm = np.random.permutation(len(nonwall_rs))
    n_nonwall = min(noise_count, len(perm))
    sel = perm[:n_nonwall]
    grid[nonwall_rs[sel], nonwall_cs[sel]] = noise_color
    placed = n_nonwall

    wall_rs, wall_cs = np.where(frame_mask & (~protected_mask))
    n_wall = len(wall_rs)
    max_damage = min(noise_count - placed, int(_WALL_DAMAGE_FRACTION * n_wall))
    if max_damage > 0:
        damage_perm = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_perm], wall_cs[damage_perm]] = noise_color
def _has_decoupled_tie(line_counts: list[int], segment_counts: list[int]) -> bool:
    for i in range(len(line_counts)):
        for j in range(i + 1, len(line_counts)):
            if line_counts[i] == line_counts[j] and segment_counts[i] != segment_counts[j]:
                return True
    return False


def _has_tie_order_not_by_length(
    recovered_drawings: list[tuple[int, int, int, int]], render_order: list[int]
) -> bool:
    render_rank = {idx: rank for rank, idx in enumerate(render_order)}
    for i, _color_i, line_i, seg_i in recovered_drawings:
        for j, _color_j, line_j, seg_j in recovered_drawings:
            if i >= j:
                continue
            if line_i != line_j or seg_i == seg_j:
                continue
            if seg_i > seg_j and render_rank[i] < render_rank[j]:
                return True
            if seg_j > seg_i and render_rank[j] < render_rank[i]:
                return True
    return False


def _sample_specs(frame_count: int) -> list[tuple[int, int]] | None:
    tie_lines = [line_count for line_count, segments in _SEGMENTS_BY_LINES.items() if len(segments) >= 2]
    for _ in range(_SPEC_ATTEMPTS):
        if not tie_lines:
            continue
        tie_line = random.choice(tie_lines)
        tie_segments = random.sample(list(_SEGMENTS_BY_LINES[tie_line]), 2)
        if abs(tie_segments[0] - tie_segments[1]) < 2 and len(_SEGMENTS_BY_LINES[tie_line]) >= 3:
            continue

        other_lines = [line_count for line_count in _LINE_COUNT_VALUES if line_count != tie_line]
        if frame_count - 2 > len(other_lines):
            continue
        chosen_other_lines = random.sample(other_lines, frame_count - 2)

        specs = [(tie_line, tie_segments[0]), (tie_line, tie_segments[1])]
        for line_count in chosen_other_lines:
            specs.append((line_count, random.choice(_SEGMENTS_BY_LINES[line_count])))
        random.shuffle(specs)

        line_counts = [line_count for line_count, _segments in specs]
        segment_counts = [segments for _line_count, segments in specs]
        if not _has_decoupled_tie(line_counts, segment_counts):
            continue
        return specs
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
    grid_h = random.randint(max(h_min, h_max - 3), h_max)
    grid_w = random.randint(max(w_min, w_max - 3), w_max)

    frames = []
    target_line_counts: list[int] = []
    target_segment_counts: list[int] = []
    for color, (target_lines, target_segments) in zip(frame_colors, specs):
        built = _build_frame(target_lines, target_segments, max_h=f_max_h, max_w=f_max_w)
        if built is None:
            return None
        frame_data, room_count = built
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_lines, frame_data, target_segments, room_count))
        target_line_counts.append(target_lines)
        target_segment_counts.append(target_segments)

    if not _has_decoupled_tie(target_line_counts, target_segment_counts):
        return None

    positions = _place_frames_without_overlap(frames, grid_h, grid_w)
    if positions is None:
        return None

    tied_pairs = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if target_line_counts[i] == target_line_counts[j]
    ]
    for i, j in tied_pairs:
        if not _force_corner_overlap(positions, i, j, frames, grid_h, grid_w):
            return None

    primary_tie = tied_pairs[0] if tied_pairs else None
    if primary_tie is not None:
        i, j = primary_tie
        longer = i if target_segment_counts[i] > target_segment_counts[j] else j
        shorter = j if longer == i else i
        middle = [k for k in range(frame_count) if k not in {longer, shorter}]
        random.shuffle(middle)
        render_order = [longer] + middle + [shorter]
    else:
        render_order = list(range(frame_count))
        random.shuffle(render_order)

    protected_mask = _overlap_mask_for_pairs(frames, positions, tied_pairs, grid_h, grid_w)
    protected_mask |= _boundary_mask(frames, positions, grid_h, grid_w)
    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, protected_mask, noise_color, grid_h, grid_w)

    recovered = {}
    for color in colors_present(grid, exclude={0}):
        recovered[int(color)] = _frame_signature_for_color(grid, int(color))

    pad_colors = [color for color, (_lines, segments, _rooms, _walls) in recovered.items() if segments == 0]
    if pad_colors != [int(noise_color)]:
        return None

    recovered_drawings: list[tuple[int, int, int, int]] = []
    recovered_line_counts: list[int] = []
    recovered_segment_counts: list[int] = []
    for idx, color in enumerate(frame_colors):
        line_count, segments, rooms, _walls = recovered[int(color)]
        if line_count <= 0 or segments <= 0 or rooms <= 0:
            return None
        recovered_drawings.append((idx, int(color), int(line_count), int(segments)))
        recovered_line_counts.append(int(line_count))
        recovered_segment_counts.append(int(segments))

    if not _has_decoupled_tie(recovered_line_counts, recovered_segment_counts):
        return None
    if not _has_tie_order_not_by_length(recovered_drawings, render_order):
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    frame_info = sorted(
        (line_count, render_order.index(idx), segments, color)
        for idx, color, line_count, segments in recovered_drawings
    )
    max_segments = max(segments for _line_count, _render_rank, segments, _color in frame_info)
    expected = np.full((frame_count, max_segments), noise_color, dtype=int)
    for r, (_line_count, _render_rank, segments, color) in enumerate(frame_info):
        expected[r, :segments] = color

    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
