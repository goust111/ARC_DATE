# Combo variant of 89565ca0 (index 20)
# combo_spec: D7/2+D8/2
# Name: 颜色重映射+转置输出

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for D7/2 + D8/2
# --------------------------------
# Original behavior:
# - Recover each frame's room count.
# - Sort frames by room count ascending, breaking ties by cover order.
# - Emit one horizontal row per frame, using THAT frame's own color.
#
# Fused behavior:
# - Keep the same room-count sorting to decide which abstract bar comes first.
# - Replace each sorted bar's own color with the color of the frame at the same
#   rank in the GLOBAL bottom->top cover order.
# - Transpose the output: each sorted bar becomes a bottom-aligned column.
#
# Concrete example:
# - Suppose room-sorted frames are [(red, 2), (blue, 4), (green, 5)].
# - Suppose global cover order is [green, red, blue] from bottom to top.
# - Then the output is 5x3, with column heights [2, 4, 5] but column colors
#   [green, red, blue]. This is visibly different from the original even after
#   transposition, because the height order and the color order now come from
#   different input cues.
#
# Complexity / observability calibration:
# - The color remap is not arbitrary: every output color still comes from an
#   actual frame color, but the sequence is driven by visible overlap layering.
# - The transpose is strong but clean: humans can still read heights directly,
#   now bottom-up per column.
# - The generator rejects degenerate samples where the room-sorted order and the
#   cover-order color sequence match, because those would hide D7/2.
#
# Implicit-assumption audit:
# - The old solver only needed cover order inside equal-room groups; the fused
#   rule needs a UNIQUE global cover order. The generator therefore keeps the
#   seed task's overlap-friendly layout, then rejects any sample whose global
#   layer order is not uniquely recoverable by the new solver.
# - Bboxes must remain recoverable after overlaps, so the generator stays close
#   to the original placement/noise regime instead of aggressively stacking
#   frames on top of each other.
# - "Overlap" must mean actual shared wall pixels, not just overlapping bboxes.
#   Some overlap pairs are forced, and the final sample is accepted only if the
#   solver can derive a unique global cover order from the rendered result.
# - Noise must not erase all layer evidence. The generator protects the forced
#   overlap pixels from wall damage.


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


def _cover_order(
    g: np.ndarray, group: list[tuple[int, int, np.ndarray]]
) -> list[tuple[int, int, np.ndarray]]:
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
            count_a = int(np.count_nonzero(g[overlap] == int(ca)))
            count_b = int(np.count_nonzero(g[overlap] == int(cb)))
            assert count_a != count_b
            assert max(count_a, count_b) > 0
            top = int(ca) if count_a > count_b else int(cb)
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
    Convert overlapped partitioned frames into a transposed histogram whose heights come from room counts and whose colors come from layer order.

    Input structure:
    - Black background with several axis-aligned colored wall frames plus one scattered non-black noise color.
    - Each real frame is a rectilinear line drawing with enclosed rooms; overlaps and noise may hide parts of its walls.
    - The noise color has no enclosed region of its own and also serves as the output padding color.

    Rule:
    - Reconstruct each non-black color's wall lattice, count its enclosed 4-connected background regions, and discard the one color with zero rooms as noise.
    - Recover the global bottom-to-top cover order from contested wall pixels where reconstructed frames overlap.
    - Sort frames by room count ascending, breaking ties by that same cover order; then build a transposed output whose columns follow this sorted order, each bottom-filled to its frame's room count.
    - Recolor those columns by position: the k-th sorted column uses the color of the k-th frame in the global cover order, not the color of the frame that supplied the height.

    Key insight:
    - The output combines two independent abstractions of the same frame set: geometry is reduced to sorted room-count heights, while color is transferred from the separate layering sequence revealed by overlaps.
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
    layered = _cover_order(g, drawings)
    layer_rank = {int(color): i for i, (color, _rooms, _walls) in enumerate(layered)}
    ordered = sorted(drawings, key=lambda item: (int(item[1]), layer_rank[int(item[0])]))

    height = max(rooms for _color, rooms, _walls in ordered)
    width = len(ordered)
    out = np.full((height, width), pad, dtype=int)
    for c, (_src_color, rooms, _walls) in enumerate(ordered):
        transfer_color = int(layered[c][0])
        out[height - int(rooms) :, c] = transfer_color
    return out


# --------------- generator (codex) ---------------

_FRAME_COUNT_RANGE = (3, 4)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 0.25
_GRID_SIZE_BY_FRAMES = {
    4: (22, 30, 22, 30, 18, 22),
    3: (20, 30, 20, 30, 22, 25),
}
_CELL_GAP_RANGE = (3, 13)
_NOISE_FRACTION_RANGE = (0.03, 0.07)
_WALL_DAMAGE_FRACTION = 0.22
_EXTRA_OVERLAP_RANGE = (1, 3)
_OVERLAP_RETRY = 150
_GEN_RETRIES = 500


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


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    render_order: list[int],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def _absolute_overlap_mask(
    wa: np.ndarray,
    pa: tuple[int, int],
    wb: np.ndarray,
    pb: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    ra, ca = pa
    rb, cb = pb
    a_h, a_w = wa.shape
    b_h, b_w = wb.shape
    r0 = max(ra, rb)
    c0 = max(ca, cb)
    r1 = min(ra + a_h, rb + b_h)
    c1 = min(ca + a_w, cb + b_w)
    if r0 >= r1 or c0 >= c1:
        return out
    sa = wa[r0 - ra : r1 - ra, c0 - ca : c1 - ca]
    sb = wb[r0 - rb : r1 - rb, c0 - cb : c1 - cb]
    out[r0:r1, c0:c1] = sa & sb
    return out


def _scatter_noise(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    protected_wall_mask: np.ndarray,
    noise_color: int,
    grid_h: int,
    grid_w: int,
) -> None:
    total_cells = grid_h * grid_w
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)

    nonwall_rs, nonwall_cs = np.where(~frame_mask)
    if len(nonwall_rs) > 0:
        perm = np.random.permutation(len(nonwall_rs))
        n_nonwall = min(noise_count, len(perm))
        sel = perm[:n_nonwall]
        grid[nonwall_rs[sel], nonwall_cs[sel]] = noise_color
    else:
        n_nonwall = 0

    damageable = frame_mask & ~protected_wall_mask
    wall_rs, wall_cs = np.where(damageable)
    n_wall = len(wall_rs)
    max_damage = min(noise_count - n_nonwall, int(_WALL_DAMAGE_FRACTION * n_wall))
    if max_damage > 0:
        damage_perm = np.random.permutation(n_wall)[:max_damage]
        grid[wall_rs[damage_perm], wall_cs[damage_perm]] = noise_color


def _force_overlap(
    positions: list[tuple[int, int]],
    i: int,
    j: int,
    frames: list[tuple[int, int, np.ndarray]],
    grid_h: int,
    grid_w: int,
) -> bool:
    fi_h, fi_w = frames[i][2].shape
    fj_h, fj_w = frames[j][2].shape
    ri, ci = positions[i]
    r0_min = max(0, ri - fj_h + 2)
    r0_max = min(grid_h - fj_h, ri + fi_h - 2)
    c0_min = max(0, ci - fj_w + 2)
    c0_max = min(grid_w - fj_w, ci + fi_w - 2)
    if r0_min > r0_max or c0_min > c0_max:
        return False
    for _ in range(_OVERLAP_RETRY):
        rj = random.randint(r0_min, r0_max)
        cj = random.randint(c0_min, c0_max)
        if (
            max(ri, rj) <= min(ri + fi_h - 1, rj + fj_h - 1)
            and max(ci, cj) <= min(ci + fi_w - 1, cj + fj_w - 1)
        ):
            positions[j] = (rj, cj)
            return True
    return False


def _transposed_original_output(
    ordered: list[tuple[int, int, np.ndarray]], pad: int
) -> np.ndarray:
    height = max(rooms for _color, rooms, _walls in ordered)
    out = np.full((height, len(ordered)), pad, dtype=int)
    for c, (color, rooms, _walls) in enumerate(ordered):
        out[height - int(rooms) :, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)

    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))
    allow_tie = random.random() < _TIE_PROBABILITY and frame_count >= 3
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, distinct_needed)
        room_counts = chosen + [random.choice(chosen)]
    else:
        room_counts = random.sample(available, frame_count)
    random.shuffle(room_counts)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    cfg = _GRID_SIZE_BY_FRAMES[frame_count]
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(target_rooms, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h or fw > grid_w:
            return None
        frames.append((color, target_rooms, frame_data))

    positions: list[tuple[int, int]] = [(-1, -1)] * frame_count
    for idx in sorted(range(frame_count), key=lambda i: -frames[i][2].size):
        fh, fw = frames[idx][2].shape
        positions[idx] = (random.randint(0, grid_h - fh), random.randint(0, grid_w - fw))

    render_order = list(range(frame_count))
    random.shuffle(render_order)

    tied_pairs = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] == room_counts[j]
    ]
    for i, j in tied_pairs:
        _force_overlap(positions, i, j, frames, grid_h, grid_w)

    for i, j in zip(render_order, render_order[1:]):
        _force_overlap(positions, i, j, frames, grid_h, grid_w)

    non_tied = [
        (i, j)
        for i in range(frame_count)
        for j in range(i + 1, frame_count)
        if room_counts[i] != room_counts[j]
    ]
    random.shuffle(non_tied)
    for i, j in non_tied[: random.randint(*_EXTRA_OVERLAP_RANGE)]:
        _force_overlap(positions, i, j, frames, grid_h, grid_w)

    protected_overlap = np.zeros((grid_h, grid_w), dtype=bool)
    for a, b in zip(render_order, render_order[1:]):
        overlap = _absolute_overlap_mask(
            frames[a][2], positions[a], frames[b][2], positions[b], (grid_h, grid_w)
        )
        if int(overlap.sum()) > 0:
            protected_overlap |= overlap

    grid, frame_mask = _render_frames(frames, positions, render_order, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, protected_overlap, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(grid, exclude={0}):
        rooms = _room_count_for_color(grid, int(color))
        walls = _walls_for_color(grid, int(color))
        counts.append((int(color), int(rooms), walls))
    drawings = [(color, rooms, walls) for color, rooms, walls in counts if rooms > 0]
    pad_colors = [color for color, rooms, _walls in counts if rooms == 0]
    if len(drawings) != frame_count or len(pad_colors) != 1:
        return None
    try:
        layered = _cover_order(grid, drawings)
    except Exception:
        return None
    layer_rank = {int(color): i for i, (color, _rooms, _walls) in enumerate(layered)}
    ordered = sorted(drawings, key=lambda item: (int(item[1]), layer_rank[int(item[0])]))
    original_transposed = _transposed_original_output(ordered, int(pad_colors[0]))
    remap_positions = sum(int(layered[i][0]) != int(ordered[i][0]) for i in range(frame_count))
    if remap_positions < 2:
        return None
    if np.array_equal(output, original_transposed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
