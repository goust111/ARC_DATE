# Variant of 89565ca0
# Dimension: D4 — 噪声/干扰机制 (src)
# Change: 一种颜色的散落噪声像素覆盖4-9%的网格，既出现在框架内外，也会损坏部分墙壁像素... → 干扰框架替代散落噪声

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the D4 variant
# ==================================
# a) Concrete behavior and trace
#    - Original D4 behavior: one extra color appears as scattered single pixels, so the solver
#      can identify it because that color never encloses a room.
#    - New D4 behavior: the extra color becomes a single connected open rectilinear drawing,
#      visually similar to a frame fragment. It is rendered as a comb/open-shelf shape:
#      one full-height vertical backbone plus several horizontal branches. It may cross real
#      frame walls, but it still has no closed cycle, so its room count stays zero.
#    - Small trace:
#      * Original: noise color inside a frame bbox contributes a few isolated hits; after
#        cropping that color, `_reconstruct_walls` finds no usable closed grid, so rooms = 0.
#      * Variant: distractor color forms long line segments inside one bbox; after cropping,
#        it still has only one dense vertical column, so `_reconstruct_walls` returns the raw
#        open mask instead of inventing a rectangular frame, and rooms = 0.
#      * Difference: the zero-room color is now a structured line object that competes for
#        attention with the true frames instead of being visibly random pixel dust.
# b) Complexity calibration
#    - The distractor is not just a single short stroke; it spans roughly 4-9% of the grid,
#      matching the old noise budget, so the change is visually obvious.
#    - It stays connected and uses only one drawing color, which keeps the rule readable from
#      a few examples instead of introducing many small arbitrary fragments.
#    - Real frames remain 3-5 colors with 1-6 rooms, and at most one room-count tie pair is
#      allowed, preserving the original relational complexity.
# c) Implicit-assumption audit of `main()`
#    - `for color in colors_present(...)`: still valid because the distractor is still a single
#      extra non-black color.
#    - `_room_count_for_color(...)`: originally safe because scattered noise never formed rooms.
#      In the variant this stays safe because the distractor is deliberately open and also built
#      to avoid a second dense vertical column, preventing accidental frame reconstruction.
#    - `drawings = ... if rooms > 0`: unchanged meaning; valid frames are exactly those whose
#      own-color wall evidence encloses at least one region.
#    - `assert len(pad_colors) == 1`: now depends on the generator guaranteeing exactly one
#      extra open-line color and no real frame degenerating to zero rooms after overlaps.
#    - `_cover_order(...)`: still assumes tied frames have an overlap witness pixel colored by
#      one of the tied colors. The generator therefore protects tied-frame shared wall pixels
#      from distractor overwrite.
# d) Placement legality
#    - Real frames are generated exactly as before and may overlap each other.
#    - The distractor is placed fully in-bounds. When overlap is requested, it may overwrite
#      some real wall pixels, but only outside protected tied-overlap witnesses.
#    - The generator validates every sample by running `main()` and comparing with the intended
#      output, so illegal placements are rejected automatically.
# e) Visual observability
#    - A human can now see one extra colored line drawing that looks frame-like but never closes.
#      That is much easier to notice across 2-4 examples than invisible bookkeeping like color id.
#    - Output still uses the distractor color as row padding, so the relation between the unused
#      open drawing and the fill color remains learnable.
# f) Visual cleanliness
#    - The distractor is one connected object instead of many independent pixels, reducing visual
#      speckle. Overlap pressure is capped so the scene stays legible and true frames retain their
#      recognizable room structure.
# g) Color inference
#    - No color remapping is introduced. The distractor color is identified only by geometry:
#      it is the single line-set color whose reconstructed walls enclose zero rooms.


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


def _analyze_color(g: np.ndarray, color: int) -> tuple[int, np.ndarray]:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, r1, c1 = b
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    out = np.zeros_like(g, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = walls
    return _count_enclosed_regions(walls), out


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
    Convert each true partitioned frame into a horizontal bar whose length is its number of enclosed regions.

    Input structure:
    - Black background with several axis-aligned colored line drawings.
    - Most colors are closed rectangular wall grids, and some of those grids overlap each other.
    - One extra non-black color forms an open rectilinear distractor made of line segments; it may cross real walls but it does not enclose any region by itself.

    Rule:
    - For each non-background color, reconstruct its horizontal and vertical wall segments from dense row/column evidence inside that color's bbox.
    - Count the 4-connected background regions fully enclosed by those reconstructed walls.
    - Colors with at least one enclosed region are the real frames; the unique zero-room color is the distractor and also the output padding color.
    - Sort real frame colors by enclosed-region count; equal counts are ordered by which reconstructed frame is covered by the other.

    Key insight:
    - The decisive feature is not whether a color draws long straight segments, but whether those segments close off rooms.
    - The distractor deliberately imitates the line style of the frames while staying open, so it contributes only the padding color and no bar row.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        rooms, walls = _analyze_color(g, int(color))
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


# --------------- generator (gpt-5) ---------------

# ---- CFG: Tunable generation parameters ----
_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.35
_GRID_SIZE_BY_FRAMES = {
    5: (25, 30, 25, 30, 16, 18),
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)
_EXTRA_OVERLAP_RANGE = (0, 3)
_OVERLAP_RETRY = 100
_DISTRACTOR_FRACTION_RANGE = (0.04, 0.09)
_DISTRACTOR_HEIGHT_RANGE = (12, 18)
_DISTRACTOR_RUNG_COUNT_RANGE = (3, 5)
_DISTRACTOR_MAX_WIDTH = 18
_DISTRACTOR_PLACEMENT_RETRY = 250
_DISTRACTOR_OVERLAP_PROB = 0.75
_GEN_RETRIES = 300


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


def _build_distractor(grid_h: int, grid_w: int) -> np.ndarray | None:
    max_h = min(_DISTRACTOR_HEIGHT_RANGE[1], grid_h - 1)
    max_w = min(_DISTRACTOR_MAX_WIDTH, grid_w - 1)
    if max_h < _DISTRACTOR_HEIGHT_RANGE[0] or max_w < 8:
        return None

    for _ in range(80):
        h = random.randint(_DISTRACTOR_HEIGHT_RANGE[0], max_h)
        rung_count = random.randint(_DISTRACTOR_RUNG_COUNT_RANGE[0], min(_DISTRACTOR_RUNG_COUNT_RANGE[1], h - 2))
        backbone_col = random.randint(1, min(3, max_w - 5))
        min_w = max(8, backbone_col + 5)
        if min_w > max_w:
            continue
        w = random.randint(min_w, max_w)

        rows = [0, h - 1]
        if rung_count > 2:
            rows.extend(random.sample(list(range(2, h - 2)), rung_count - 2))
        rows = sorted(rows)

        target_pixels = int(grid_h * grid_w * random.uniform(*_DISTRACTOR_FRACTION_RANGE))
        target_pixels = max(target_pixels, h + 3 * rung_count)
        max_branch = w - backbone_col - 1
        if max_branch < 3:
            continue

        branch_lengths = [3] * rung_count
        remaining = min(target_pixels - h - sum(branch_lengths), rung_count * (max_branch - 3))
        remaining = max(0, remaining)
        order = list(range(rung_count))
        random.shuffle(order)
        for idx in order:
            if remaining <= 0:
                break
            cap = max_branch - branch_lengths[idx]
            add = random.randint(0, min(cap, remaining))
            branch_lengths[idx] += add
            remaining -= add
        while remaining > 0:
            changed = False
            for idx in order:
                if branch_lengths[idx] < max_branch:
                    branch_lengths[idx] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

        mask = np.zeros((h, w), dtype=bool)
        mask[:, backbone_col] = True
        lengths = [backbone_col + branch for branch in branch_lengths]
        random.shuffle(lengths)
        for r, end_c in zip(rows, lengths):
            mask[r, backbone_col : end_c + 1] = True

        if _count_enclosed_regions(_reconstruct_walls(mask)) == 0:
            return mask
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


def _absolute_mask(local_mask: np.ndarray, pos: tuple[int, int], grid_h: int, grid_w: int) -> np.ndarray:
    out = np.zeros((grid_h, grid_w), dtype=bool)
    rs, cs = np.where(local_mask)
    out[rs + pos[0], cs + pos[1]] = True
    return out


def _render_shapes(
    shapes: list[tuple[int, np.ndarray, tuple[int, int]]],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for color, mask, pos in shapes:
        rs, cs = np.where(mask)
        grid[rs + pos[0], cs + pos[1]] = color
    return grid


def _place_distractor(
    distractor_mask: np.ndarray,
    frame_union: np.ndarray,
    protected_overlap: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int]:
    dh, dw = distractor_mask.shape
    rs, cs = np.where(distractor_mask)
    target_overlap = random.random() < _DISTRACTOR_OVERLAP_PROB and bool(frame_union.any())
    preferred_overlap = max(1, min(6, rs.size // 10))
    best_pos = (0, 0)
    best_score = None

    for _ in range(_DISTRACTOR_PLACEMENT_RETRY):
        r0 = random.randint(0, grid_h - dh)
        c0 = random.randint(0, grid_w - dw)
        protected_hits = int(protected_overlap[rs + r0, cs + c0].sum())
        if protected_hits > 0:
            continue
        overlap = int(frame_union[rs + r0, cs + c0].sum())
        if target_overlap:
            if 1 <= overlap <= max(8, preferred_overlap + 2):
                return (r0, c0)
            score = abs(overlap - preferred_overlap)
        else:
            if overlap == 0:
                return (r0, c0)
            score = overlap
        if best_score is None or score < best_score:
            best_score = score
            best_pos = (r0, c0)
    return best_pos


def _expected_output(
    room_counts: list[int],
    frame_colors: list[int],
    render_order: list[int],
    pad_color: int,
) -> np.ndarray:
    frame_info = sorted((room_counts[i], render_order.index(i), frame_colors[i]) for i in range(len(frame_colors)))
    max_rooms = max(rc for rc, _, _ in frame_info)
    expected = np.full((len(frame_colors), max_rooms), pad_color, dtype=int)
    for r, (rc, _, color) in enumerate(frame_info):
        expected[r, :rc] = color
    return expected


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

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    distractor_color = chosen_colors[frame_count]

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

    distractor_mask = _build_distractor(grid_h, grid_w)
    if distractor_mask is None:
        return None

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

    real_abs_masks = [_absolute_mask(frames[i][2], positions[i], grid_h, grid_w) for i in range(frame_count)]
    frame_union = np.zeros((grid_h, grid_w), dtype=bool)
    for mask in real_abs_masks:
        frame_union |= mask

    protected_overlap = np.zeros((grid_h, grid_w), dtype=bool)
    for i, j in tied_pairs:
        protected_overlap |= real_abs_masks[i] & real_abs_masks[j]

    distractor_pos = _place_distractor(distractor_mask, frame_union, protected_overlap, grid_h, grid_w)

    render_order = list(range(frame_count))
    random.shuffle(render_order)
    shapes = [(frames[idx][0], frames[idx][2], positions[idx]) for idx in render_order]
    shapes.append((distractor_color, distractor_mask, distractor_pos))
    grid = _render_shapes(shapes, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(room_counts, frame_colors, render_order, distractor_color)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
