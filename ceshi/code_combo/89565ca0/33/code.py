# Combo variant of 89565ca0 (index 33)
# combo_spec: D2/0+D9/0
# Name: 颜色消歧+禁止重叠

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Suppose blue(1) and red(2) frames each contain 3 rooms and a green(3) frame contains 5.
#    In the original task the blue/red tie was resolved by overlap order, so the lower covered
#    frame came first. Here frame bboxes are never allowed to overlap, so that cue disappears.
#    The fused rule therefore sorts the tied blue/red pair by color index: blue before red.
# b) Complexity calibration:
#    A tie must still appear often enough to expose the new rule, so generation forces exactly
#    one tied room-count pair in each sample. Non-overlap removes one reasoning axis, but noisy
#    wall damage and room counting remain unchanged, so the task does not collapse into pure color
#    sorting.
# c) Implicit assumption audit:
#    The original `_cover_order` assumed equal-room frames form a unique overlap DAG; with disjoint
#    frames that assumption becomes false and would assert. The new solver removes that dependency
#    and assumes only that true frames have positive enclosed-room counts while the scattered noise
#    color has zero; the generator preserves that invariant.
# d) Placement legality:
#    Frames are placed largest-first, fully in bounds, with at least one blank-cell gap between
#    their bboxes. Any failed placement rejects the attempt.
# e) Visual observability:
#    Equal-length output bars now recur in every sample, and their order is determined by visible
#    frame colors rather than hidden z-order. Black is never used as a sorting key.
# f) Visual cleanliness:
#    Because frames neither overlap nor touch, each colored line structure stays legible even after
#    noise damage.
# g) Inferability of the change:
#    The color tiebreak would be too weak if ties were rare, so tied counts are mandatory. The
#    repeated presence of same-length bars provides the evidence that color identity, not position
#    or overlap, decides their order.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each damaged colored frame into a horizontal color bar whose length equals its room count.

    Input structure:
    - Black background with several axis-aligned colored line frames plus one scattered noise color.
    - Real frames are rectangular wall grids that may be partially damaged by noise pixels.
    - Frame bboxes never overlap or cover one another, so there is no usable depth relation.

    Rule:
    - For each non-background color, rebuild its rectilinear walls from dense row/column evidence.
    - Count the 4-connected background regions enclosed by those reconstructed walls.
    - The lone color with zero enclosed regions is the padding/noise color.
    - Output one row per true frame, sorted first by room count and then, for ties, by ascending ARC
      color index; fill remaining cells in each row with the padding color.

    Key insight:
    - Once overlap cues are removed, equal-room frames can only be disambiguated by a stable intrinsic
      property. Room count still identifies the bar length, and color identity now resolves ties.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int]] = []
    for color in colors_present(g, exclude={bg}):
        rooms = _room_count_for_color(g, int(color))
        counts.append((int(color), int(rooms)))

    drawings = [(color, rooms) for color, rooms in counts if rooms > 0]
    pad_colors = [color for color, rooms in counts if rooms == 0]
    assert drawings
    assert len(pad_colors) == 1

    pad = int(pad_colors[0])
    ordered = sorted(drawings, key=lambda item: (item[1], item[0]))
    width = max(rooms for _color, rooms in ordered)

    out = np.full((len(ordered), width), pad, dtype=int)
    for r, (color, rooms) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (combo variant) ---------------

_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 6)
_TIE_PROBABILITY = 1.0
_GRID_SIZE_BY_FRAMES = {
    5: (28, 30, 28, 30, 12, 14),
    4: (26, 30, 26, 30, 14, 16),
    3: (22, 30, 22, 30, 18, 20),
}
_CELL_GAP_RANGE = (3, 9)
_NOISE_FRACTION_RANGE = (0.04, 0.08)
_WALL_DAMAGE_FRACTION = 0.22
_FRAME_SEPARATION = 1
_PLACEMENT_RETRY = 250
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


def _build_frame(target_rooms: int, max_h: int, max_w: int) -> np.ndarray | None:
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


def _bboxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_frames(frames: list[tuple[int, int, np.ndarray]], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(frames)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for idx in sorted(range(len(frames)), key=lambda i: -frames[i][2].size):
        fh, fw = frames[idx][2].shape
        for _ in range(_PLACEMENT_RETRY):
            r0 = random.randint(0, grid_h - fh)
            c0 = random.randint(0, grid_w - fw)
            cand = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_bboxes_too_close(cand, box, _FRAME_SEPARATION) for box in placed_boxes):
                continue
            positions[idx] = (r0, c0)
            placed_boxes.append(cand)
            break
        else:
            return None

    return [pos for pos in positions if pos is not None]


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    frame_mask = np.zeros((grid_h, grid_w), dtype=bool)
    for idx, (color, _rooms, wall_arr) in enumerate(frames):
        r0, c0 = positions[idx]
        rs, cs = np.where(wall_arr)
        abs_rs, abs_cs = rs + r0, cs + c0
        grid[abs_rs, abs_cs] = color
        frame_mask[abs_rs, abs_cs] = True
    return grid, frame_mask


def _scatter_noise(grid: np.ndarray, frame_mask: np.ndarray, noise_color: int, grid_h: int, grid_w: int) -> None:
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


def _expected_output(frames: list[tuple[int, int, np.ndarray]], noise_color: int) -> np.ndarray:
    ordered = sorted(((rooms, color) for color, rooms, _walls in frames))
    max_rooms = max(rooms for rooms, _color in ordered)
    out = np.full((len(ordered), max_rooms), noise_color, dtype=int)
    for r, (rooms, color) in enumerate(ordered):
        out[r, :rooms] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    available = list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1))

    allow_tie = frame_count >= 3 and random.random() < _TIE_PROBABILITY
    if allow_tie:
        distinct_needed = frame_count - 1
        chosen = random.sample(available, distinct_needed)
        room_counts = chosen + [random.choice(chosen)]
    else:
        if frame_count > len(available):
            return None
        room_counts = random.sample(available, frame_count)
    random.shuffle(room_counts)
    assert len(room_counts) == frame_count
    assert len(room_counts) > len(set(room_counts))

    cfg = _GRID_SIZE_BY_FRAMES.get(frame_count, _GRID_SIZE_BY_FRAMES[3])
    h_min, h_max, w_min, w_max, f_max_h, f_max_w = cfg
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(w_min, w_max)

    chosen_colors = random.sample(list(range(1, 10)), frame_count + 1)
    frame_colors = chosen_colors[:frame_count]
    noise_color = chosen_colors[frame_count]

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(target_rooms, max_h=f_max_h, max_w=f_max_w)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((color, target_rooms, frame_data))

    footprint = sum((wall.shape[0] + _FRAME_SEPARATION) * (wall.shape[1] + _FRAME_SEPARATION) for _, _, wall in frames)
    if footprint > grid_h * grid_w:
        return None

    positions = _place_frames(frames, grid_h, grid_w)
    if positions is None:
        return None

    grid, frame_mask = _render_frames(frames, positions, grid_h, grid_w)
    _scatter_noise(grid, frame_mask, noise_color, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(frames, noise_color)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
