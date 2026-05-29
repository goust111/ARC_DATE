# Combo variant of 89565ca0 (index 28)
# combo_spec: D9/2+D14/1
# Name: 重叠区标记色+先解析重叠再计数

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    Original: if two frames share a wall segment, one frame color overwrites the other.
#    Fused: the shared wall segment is drawn in one dedicated marker color instead.
#    Example: a red frame and a green frame partially share an outer border for 4 cells.
#    In the original task, one of {red, green} would still occupy those 4 cells; counting
#    rooms per visible color is therefore a single pass. Here those 4 cells become marker.
#    If the solver ignores marker, each affected frame loses part of its outer wall and can
#    drop from 1+ rooms to 0 rooms. So the correct sequence is:
#    1) detect which color is the overlap marker,
#    2) fold marker pixels back into each frame's wall evidence inside that frame's bbox,
#    3) reconstruct the rectilinear walls,
#    4) count enclosed regions.
# b) Complexity calibration:
#    - One global marker color is used, not a different mixed color per pair.
#    - Each grid contains 1-2 overlap pairs, each overlap is a contiguous border segment
#      of length >= 3, so the effect is easy to see.
#    - Ties are intentionally removed in generation. Once the top color is replaced by a
#      neutral marker, the old "cover-order breaks ties" signal is gone; keeping distinct
#      room counts avoids hidden ambiguity and keeps the new challenge focused on overlap
#      restoration rather than on an invisible z-order.
# c) Implicit assumptions audited against the old solver:
#    - Old assumption: every non-black non-noise color can be counted from its own pixels.
#      Broken: marker pixels hide real wall cells, so counting must use (frame color OR marker).
#    - Old assumption: exactly one non-frame color exists. Broken: now there are two
#      special colors (noise and marker), so role identification must be explicit.
#    - Old assumption: bbox(g == color) is still the true frame bbox. Preserved only because
#      the generator guarantees that every frame keeps visible own-color pixels on all four
#      outer sides even after overlap and noise damage.
# d) Placement legality:
#    - Only selected disjoint pairs overlap.
#    - Each overlap is produced by aligning one outer border of one frame with one outer
#      border of another frame; bboxes intersect only on that strip.
#    - No triple overlaps and no unintended bbox overlaps are allowed.
# e/f/g) Observability and visual cleanliness:
#    - Marker color appears as a long contiguous line segment, not as isolated specks.
#    - Noise never uses the marker color, and noise is not allowed to overwrite marker cells.
#    - Because overlaps are thin border strips rather than deep multi-layer stacks, the grid
#      stays readable and the role of the marker is visually recoverable from a few examples.


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
    if len(hrows) < 2:
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


def _plain_room_count(g: np.ndarray, color: int) -> int:
    b = bbox(g == int(color))
    if b is None:
        return 0
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    return _count_enclosed_regions(walls)


def _valid_restored_frame(walls: np.ndarray) -> bool:
    if walls.shape[0] < 4 or walls.shape[1] < 4:
        return False
    return bool(
        walls[0, :].all()
        and walls[-1, :].all()
        and walls[:, 0].all()
        and walls[:, -1].all()
    )


def _restore_frame_with_marker(
    g: np.ndarray, color: int, marker: int
) -> tuple[int, np.ndarray, tuple[int, int, int, int], int, int]:
    b = bbox(g == int(color))
    assert b is not None
    own = crop_bbox(g == int(color), b).astype(bool)
    marker_mask = crop_bbox(g == int(marker), b).astype(bool)
    support = own | marker_mask
    walls = _reconstruct_walls(support)
    rooms = _count_enclosed_regions(walls)
    explained_marker = int(np.sum(marker_mask & walls))
    stray_marker = int(np.sum(marker_mask & ~walls))
    return int(rooms), walls, b, explained_marker, stray_marker


def _recover_roles_and_drawings(
    g: np.ndarray,
) -> tuple[int, int, list[tuple[int, int, np.ndarray, tuple[int, int, int, int]]]]:
    colors = colors_present(g, exclude={0})
    best: tuple[
        tuple[int, int, int, int],
        int,
        int,
        list[tuple[int, int, np.ndarray, tuple[int, int, int, int]]],
    ] | None = None

    plain_counts = {color: _plain_room_count(g, color) for color in colors}

    for marker in colors:
        if plain_counts[marker] != 0:
            continue
        for pad in colors:
            if pad == marker or plain_counts[pad] != 0:
                continue

            drawings: list[tuple[int, int, np.ndarray, tuple[int, int, int, int]]] = []
            total_explained = 0
            total_stray = 0
            ok = True

            for color in colors:
                if color in {marker, pad}:
                    continue
                rooms, walls, b, explained, stray = _restore_frame_with_marker(g, color, marker)
                if rooms <= 0 or rooms > 6 or not _valid_restored_frame(walls):
                    ok = False
                    break
                drawings.append((int(color), int(rooms), walls, b))
                total_explained += explained
                total_stray += stray

            if not ok or total_explained == 0:
                continue

            room_values = [rooms for _color, rooms, _walls, _b in drawings]
            score = (
                len(set(room_values)),
                total_explained,
                -total_stray,
                -int(np.sum(g == int(pad))),
            )
            if best is None or score > best[0]:
                best = (score, int(pad), int(marker), drawings)

    assert best is not None
    _score, pad, marker, drawings = best
    return pad, marker, drawings


def _drawings_needing_marker(
    g: np.ndarray, frame_colors: list[int], marker: int
) -> int:
    changed = 0
    for color in frame_colors:
        plain = _plain_room_count(g, color)
        restored, _walls, _b, _explained, _stray = _restore_frame_with_marker(g, color, marker)
        if plain != restored:
            changed += 1
    return changed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each damaged line frame into a horizontal bar whose length matches its room count, but only after repairing marker-colored overlaps.

    Input structure:
    - Black background with several axis-aligned colored wall frames.
    - One non-frame color is scattered noise / padding, and a second special color marks wall segments where two frames overlap.
    - Real frames still form rectangular partitioned layouts, but some visible wall pixels have been replaced either by noise or by the overlap marker.

    Rule:
    - First identify which special color is the overlap marker and which one is the padding noise.
    - For each frame color, keep its visible bbox, add marker pixels inside that bbox back into the wall evidence, and reconstruct the rectilinear wall grid.
    - Count the enclosed 4-connected background regions of each restored frame.
    - Sort frames by room count and output one horizontal color bar per frame, padded on the right with the noise color.

    Key insight:
    - The marker-colored overlap is not an extra object and not ordinary noise: it is shared wall evidence that must be reassigned to both participating frames before any room counting makes sense.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    pad, _marker, drawings = _recover_roles_and_drawings(g)
    ordered = sorted(drawings, key=lambda item: (item[1], item[3][0], item[3][1], item[0]))
    width = max(rooms for _color, rooms, _walls, _b in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms, _walls, _b) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator ---------------

_FRAME_COUNT_RANGE = (3, 3)
_ROOM_COUNT_RANGE = (1, 5)
_GRID_SIZE_BY_FRAMES = {
    5: (27, 30, 27, 30, 12, 14),
    4: (24, 30, 24, 30, 13, 16),
    3: (26, 30, 26, 30, 15, 18),
}
_CELL_GAP_RANGE = (3, 8)
_NOISE_FRACTION_RANGE = (0.04, 0.08)
_WALL_DAMAGE_FRACTION = 0.18
_MIN_MARKER_SEGMENT = 3
_MAX_OVERLAP_PAIRS = 1
_GEN_RETRIES = 400


def _all_positions(shape: tuple[int, int], grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    h, w = shape
    positions = [(r, c) for r in range(grid_h - h + 1) for c in range(grid_w - w + 1)]
    random.shuffle(positions)
    return positions


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

    if merged < cells_to_merge or _count_enclosed_regions(walls) != target_rooms:
        return None
    return walls


def _bbox_from_pos(pos: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = pos
    h, w = shape
    return (int(r), int(c), int(r + h - 1), int(c + w - 1))


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _random_nonoverlap_position(
    shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    for pos in _all_positions(shape, grid_h, grid_w):
        bb = _bbox_from_pos(pos, shape)
        if all(not _bboxes_overlap(bb, other) for other in existing):
            return pos
    return None


def _candidate_overlap_positions(
    anchor_pos: tuple[int, int],
    anchor_shape: tuple[int, int],
    partner_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    ri, ci = anchor_pos
    ah, aw = anchor_shape
    bh, bw = partner_shape
    candidates: list[tuple[int, int]] = []

    max_v = min(ah, bh) - 1
    if max_v >= _MIN_MARKER_SEGMENT:
        fixed_c = ci + aw - 1
        for ol in [max_v]:
            candidates.append((ri + ah - ol, fixed_c))
            candidates.append((ri - bh + ol, fixed_c))
        fixed_c = ci - bw + 1
        for ol in [max_v]:
            candidates.append((ri + ah - ol, fixed_c))
            candidates.append((ri - bh + ol, fixed_c))

    max_h = min(aw, bw) - 1
    if max_h >= _MIN_MARKER_SEGMENT:
        fixed_r = ri + ah - 1
        for ol in [max_h]:
            candidates.append((fixed_r, ci + aw - ol))
            candidates.append((fixed_r, ci - bw + ol))
        fixed_r = ri - bh + 1
        for ol in [max_h]:
            candidates.append((fixed_r, ci + aw - ol))
            candidates.append((fixed_r, ci - bw + ol))

    random.shuffle(candidates)
    return candidates


def _place_overlap_partner(
    anchor_pos: tuple[int, int],
    anchor_shape: tuple[int, int],
    partner_shape: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    for pos in _candidate_overlap_positions(anchor_pos, anchor_shape, partner_shape):
        r, c = pos
        h, w = partner_shape
        if r < 0 or c < 0 or r + h > grid_h or c + w > grid_w:
            continue
        bb = _bbox_from_pos(pos, partner_shape)
        if any(_bboxes_overlap(bb, other) for other in existing):
            continue
        return pos
    return None


def _place_overlap_pair(
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
    existing: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
    reserve_shapes: list[tuple[int, int]] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    reserve_shapes = reserve_shapes or []
    for pos_a in _all_positions(shape_a, grid_h, grid_w):
        bb_a = _bbox_from_pos(pos_a, shape_a)
        if any(_bboxes_overlap(bb_a, other) for other in existing):
            continue
        pos_b = _place_overlap_partner(pos_a, shape_a, shape_b, existing, grid_h, grid_w)
        if pos_b is not None:
            trial_boxes = existing + [bb_a, _bbox_from_pos(pos_b, shape_b)]
            feasible = True
            for shape in sorted(reserve_shapes, key=lambda s: -(s[0] * s[1])):
                pos = _random_nonoverlap_position(shape, trial_boxes, grid_h, grid_w)
                if pos is None:
                    feasible = False
                    break
                trial_boxes.append(_bbox_from_pos(pos, shape))
            if feasible:
                return pos_a, pos_b
    return None


def _choose_overlap_pairs(frame_count: int) -> list[tuple[int, int]]:
    indices = list(range(frame_count))
    random.shuffle(indices)
    pair_count = min(_MAX_OVERLAP_PAIRS, frame_count // 2)
    return [(indices[2 * i], indices[2 * i + 1]) for i in range(pair_count)]


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    marker_color: int,
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupancy = np.zeros((grid_h, grid_w), dtype=np.int8)

    for idx, (color, _rooms, wall_arr) in enumerate(frames):
        r0, c0 = positions[idx]
        rs, cs = np.where(wall_arr)
        ar, ac = rs + r0, cs + c0
        first = occupancy[ar, ac] == 0
        grid[ar[first], ac[first]] = int(color)
        grid[ar[~first], ac[~first]] = int(marker_color)
        occupancy[ar, ac] += 1

    assert int(occupancy.max()) <= 2
    return grid, occupancy == 1, occupancy == 2


def _protected_visible_edges(grid: np.ndarray, frame_colors: list[int]) -> np.ndarray:
    protected = np.zeros_like(grid, dtype=bool)
    rows, cols = np.indices(grid.shape)
    for color in frame_colors:
        vis = grid == int(color)
        b = bbox(vis)
        assert b is not None
        r0, c0, r1, c1 = b
        protected |= vis & (
            (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
        )
    return protected


def _scatter_noise(
    grid: np.ndarray,
    single_wall_mask: np.ndarray,
    marker_mask: np.ndarray,
    protected_mask: np.ndarray,
    noise_color: int,
) -> None:
    total_cells = grid.size
    noise_fraction = random.uniform(*_NOISE_FRACTION_RANGE)
    noise_count = int(total_cells * noise_fraction)
    placed = 0

    background_rs, background_cs = np.where(grid == 0)
    if len(background_rs) > 0:
        sel = np.random.permutation(len(background_rs))[: min(noise_count, len(background_rs))]
        grid[background_rs[sel], background_cs[sel]] = int(noise_color)
        placed = len(sel)

    wall_damage_budget = max(0, noise_count - placed)
    damageable = single_wall_mask & ~marker_mask & ~protected_mask
    wall_rs, wall_cs = np.where(damageable)
    max_damage = min(wall_damage_budget, int(_WALL_DAMAGE_FRACTION * len(wall_rs)))
    if max_damage > 0:
        sel = np.random.permutation(len(wall_rs))[:max_damage]
        grid[wall_rs[sel], wall_cs[sel]] = int(noise_color)


def _expected_output(frame_info: list[tuple[int, int]], noise_color: int) -> np.ndarray:
    ordered = sorted([(rooms, color) for color, rooms in frame_info])
    width = max(rooms for rooms, _color in ordered)
    out = np.full((len(ordered), width), noise_color, dtype=int)
    for r, (rooms, color) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    room_counts = random.sample(
        list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1)), frame_count
    )
    random.shuffle(room_counts)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 2)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]
    marker_color = chosen_colors[frame_count + 1]

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
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
        frames.append((int(color), int(target_rooms), frame_data))

    by_area = sorted(range(frame_count), key=lambda idx: frames[idx][2].size)
    overlap_pairs = [(by_area[0], by_area[1])]
    paired = {idx for pair in overlap_pairs for idx in pair}
    positions: list[tuple[int, int] | None] = [None] * frame_count
    existing_boxes: list[tuple[int, int, int, int]] = []

    for i, j in sorted(
        overlap_pairs, key=lambda pair: -(frames[pair[0]][2].size + frames[pair[1]][2].size)
    ):
        reserve_shapes = [
            frames[k][2].shape for k in range(frame_count) if k not in paired and positions[k] is None
        ]
        pair_pos = _place_overlap_pair(
            frames[i][2].shape,
            frames[j][2].shape,
            existing_boxes,
            grid_h,
            grid_w,
            reserve_shapes=reserve_shapes,
        )
        if pair_pos is None:
            return None
        pos_i, pos_j = pair_pos
        positions[i] = pos_i
        positions[j] = pos_j
        existing_boxes.append(_bbox_from_pos(pos_i, frames[i][2].shape))
        existing_boxes.append(_bbox_from_pos(pos_j, frames[j][2].shape))

    for idx in sorted(
        [k for k in range(frame_count) if k not in paired], key=lambda k: -frames[k][2].size
    ):
        if idx in paired:
            continue
        pos = _random_nonoverlap_position(frames[idx][2].shape, existing_boxes, grid_h, grid_w)
        if pos is None:
            return None
        positions[idx] = pos
        existing_boxes.append(_bbox_from_pos(pos, frames[idx][2].shape))

    assert all(pos is not None for pos in positions)
    final_positions = [(int(pos[0]), int(pos[1])) for pos in positions if pos is not None]
    if len(final_positions) != frame_count:
        return None

    grid, single_wall_mask, marker_mask = _render_frames(
        frames, final_positions, marker_color, grid_h, grid_w
    )
    if int(marker_mask.sum()) < _MIN_MARKER_SEGMENT:
        return None

    protected_mask = _protected_visible_edges(grid, frame_colors)
    _scatter_noise(grid, single_wall_mask, marker_mask, protected_mask, noise_color)

    if _plain_room_count(grid, noise_color) != 0 or _plain_room_count(grid, marker_color) != 0:
        return None

    if _drawings_needing_marker(grid, frame_colors, marker_color) < 2:
        return None

    try:
        output = main(grid)
    except Exception:
        return None

    actual_frame_info: list[tuple[int, int]] = []
    for color in frame_colors:
        rooms, walls, _b, _explained, _stray = _restore_frame_with_marker(grid, color, marker_color)
        if rooms < _ROOM_COUNT_RANGE[0] or rooms > _ROOM_COUNT_RANGE[1]:
            return None
        if not _valid_restored_frame(walls):
            return None
        actual_frame_info.append((int(color), int(rooms)))

    if len({rooms for _color, rooms in actual_frame_info}) != frame_count:
        return None

    expected = _expected_output(actual_frame_info, noise_color)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
