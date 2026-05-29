# Combo variant of 89565ca0 (index 27)
# combo_spec: D5/2+D14/0
# Name: 堆叠在输入边缘+迭代消除

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, get_bg, label_mask


# -----------------------------------------------------------------------------
# Fused variant notes
#
# D5/2 writes the summary bars back onto the original canvas at the left edge.
# D14/0 must remain observable from the final pair, so room counts are
# recomputed after every elimination from the currently surviving walls inside
# each frame's bbox. Nested foreign-color frames therefore act as extra rooms
# for the outer frame until they are removed.
# -----------------------------------------------------------------------------


def _count_enclosed_regions(walls: np.ndarray) -> int:
    labels, n_labels = label_mask(~walls, connectivity=4)
    h, w = walls.shape
    count = 0
    for k in range(1, n_labels + 1):
        coords = np.argwhere(labels == k)
        if coords.size == 0:
            continue
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


def _room_count_in_bbox(g: np.ndarray, box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    mask = g[r0 : r1 + 1, c0 : c1 + 1] != 0
    walls = _reconstruct_walls(mask)
    return _count_enclosed_regions(walls)


def _frame_records(g: np.ndarray) -> list[dict[str, int]]:
    bg = int(get_bg(g))
    assert bg == 0

    frames: list[dict[str, int]] = []
    for color in colors_present(g, exclude={bg}):
        b = bbox(g == int(color))
        if b is None:
            continue
        box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        frames.append(
            {
                "color": int(color),
                "rooms": _room_count_in_bbox(g, box),
                "r0": box[0],
                "c0": box[1],
                "r1": box[2],
                "c1": box[3],
                "pixels": int(np.sum(g == int(color))),
            }
        )
    return frames


def _next_frame(current: np.ndarray) -> dict[str, int]:
    frames = _frame_records(current)
    assert frames
    frames.sort(
        key=lambda rec: (
            int(rec["rooms"]),
            int(rec["r0"]),
            int(rec["c0"]),
            int(rec["pixels"]),
            int(rec["r1"]) - int(rec["r0"]),
            int(rec["c1"]) - int(rec["c0"]),
            int(rec["color"]),
        )
    )
    return frames[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly remove the least-room frame and stack its color bar onto the
    left edge of the original canvas.

    Input structure:
    - Black background with several colored rectilinear frames.
    - A blank strip is reserved on the left; all frames lie to its right.
    - Some examples include nested differently colored frames.

    Rule:
    - For each surviving color, take its current bbox and count rooms from all
      surviving walls inside that bbox.
    - Repeatedly choose the frame with the fewest current rooms; ties use the
      visible top-left position.
    - Draw a bar of that frame's color and current room count on the next left
      stack row, then erase that frame from the canvas and continue.
    """
    original = np.asarray(grid, dtype=int)
    current = original.copy()
    out = original.copy()

    frame_total = len(colors_present(current, exclude={0}))
    for stack_row in range(frame_total):
        frames = _frame_records(current)
        if not frames:
            break
        picked = _next_frame(current)
        color = int(picked["color"])
        rooms = int(picked["rooms"])
        current[current == color] = 0
        out[out == color] = 0
        out[stack_row, :rooms] = color
    return out


# --------------- generator ---------------

_FRAME_COUNT_RANGE = (3, 5)
_ROOM_COUNT_RANGE = (1, 5)
_GRID_SIZE_BY_FRAMES = {
    5: (24, 30, 26, 30, 11, 12),
    4: (22, 28, 24, 30, 11, 12),
    3: (20, 26, 22, 28, 11, 12),
}
_CELL_GAP_RANGE = (3, 5)
_STACK_SEPARATION = 2
_TOP_LEVEL_GAP = 2
_INNER_MARGIN = 2
_LAYOUT_SLACK = 2
_PLACEMENT_RETRIES = 80
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


def _build_frame(target_rooms: int, max_h: int, max_w: int) -> np.ndarray | None:
    if target_rooms == 1:
        h = random.randint(4, max_h)
        w = random.randint(4, max_w)
        return _rectangle_frame(h, w)

    possible_configs = []
    for nv in range(2, 6):
        for nh in range(2, 5):
            cells = (nv - 1) * (nh - 1)
            if target_rooms <= cells <= target_rooms + 4:
                possible_configs.append((nv, nh, cells))
    if not possible_configs:
        return None

    n_vcols, n_hrows, total_cells = random.choice(possible_configs)
    n_row_gaps = n_hrows - 1
    n_col_gaps = n_vcols - 1
    min_gap, max_gap = _CELL_GAP_RANGE

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
    if frame_h > max_h or frame_w > max_w:
        return None

    walls = np.zeros((frame_h, frame_w), dtype=bool)
    for r in hrow_positions:
        walls[r, :] = True
    for c in vcol_positions:
        walls[:, c] = True

    cells_to_merge = total_cells - target_rooms
    if cells_to_merge == 0:
        return walls if _count_enclosed_regions(walls) == target_rooms else None

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
            if union(int(ca), int(cb)):
                walls[int(r), int(c0) + 1 : int(c1)] = False
                merged += 1
        else:
            _, c, r0, r1, ca, cb = seg
            if union(int(ca), int(cb)):
                walls[int(r0) + 1 : int(r1), int(c)] = False
                merged += 1

    if merged != cells_to_merge:
        return None
    return walls if _count_enclosed_regions(walls) == target_rooms else None


def _rectangle_frame(height: int, width: int) -> np.ndarray:
    walls = np.zeros((height, width), dtype=bool)
    walls[0, :] = True
    walls[height - 1, :] = True
    walls[:, 0] = True
    walls[:, width - 1] = True
    return walls


def _bbox_of(pos: tuple[int, int], wall_arr: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pos
    return (r0, c0, r0 + wall_arr.shape[0] - 1, c0 + wall_arr.shape[1] - 1)


def _strictly_contains(outer_box: tuple[int, int, int, int], inner_box: tuple[int, int, int, int]) -> bool:
    return (
        outer_box[0] < inner_box[0]
        and outer_box[1] < inner_box[1]
        and outer_box[2] > inner_box[2]
        and outer_box[3] > inner_box[3]
    )


def _choose_child_frames(child_count: int, outer_h: int, outer_w: int) -> list[tuple[np.ndarray, tuple[int, int]]] | None:
    for _ in range(_PLACEMENT_RETRIES):
        children: list[tuple[np.ndarray, tuple[int, int]]] = []
        used_boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for _child_idx in range(child_count):
            child_h = random.randint(4, 6)
            child_w = random.randint(4, 6)
            if outer_h - child_h < 2 * _INNER_MARGIN or outer_w - child_w < 2 * _INNER_MARGIN:
                ok = False
                break

            child = _rectangle_frame(child_h, child_w)
            placed = False
            for _ in range(_PLACEMENT_RETRIES):
                r0 = random.randint(_INNER_MARGIN, outer_h - child_h - _INNER_MARGIN)
                c0 = random.randint(_INNER_MARGIN, outer_w - child_w - _INNER_MARGIN)
                box = (r0, c0, r0 + child_h - 1, c0 + child_w - 1)
                expanded = (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
                clash = any(
                    not (
                        expanded[2] < prev[0]
                        or expanded[0] > prev[2]
                        or expanded[3] < prev[1]
                        or expanded[1] > prev[3]
                    )
                    for prev in used_boxes
                )
                if clash:
                    continue
                used_boxes.append(box)
                children.append((child, (r0, c0)))
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok and len(children) == child_count:
            return children
    return None


def _render_frames(
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, (color, _rooms, wall_arr) in enumerate(frames):
        r0, c0 = positions[idx]
        rs, cs = np.where(wall_arr)
        grid[rs + r0, cs + c0] = int(color)
    return grid


def _expected_output(
    grid: np.ndarray,
    frames: list[tuple[int, int, np.ndarray]],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    boxes = [_bbox_of(pos, wall_arr) for pos, (_color, _rooms, wall_arr) in zip(positions, frames)]
    survivors = set(range(len(frames)))

    for stack_row in range(len(frames)):
        if not survivors:
            break
        dynamic_rooms: dict[int, int] = {}
        for idx in survivors:
            rooms = int(frames[idx][1])
            for other in survivors:
                if other == idx:
                    continue
                if _strictly_contains(boxes[idx], boxes[other]):
                    rooms += int(frames[other][1])
            dynamic_rooms[idx] = rooms

        idx = min(
            survivors,
            key=lambda i: (
                dynamic_rooms[i],
                positions[i][0],
                positions[i][1],
                frames[i][0],
            ),
        )
        color, _base_rooms, wall_arr = frames[idx]
        r0, c0 = positions[idx]
        rs, cs = np.where(wall_arr)
        out[rs + r0, cs + c0] = 0
        out[stack_row, : dynamic_rooms[idx]] = int(color)
        survivors.remove(idx)
    return out


def _build_layout(frame_count: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int]]] | None:
    child_count = 2 if frame_count >= 4 and random.random() < 0.55 else 1
    extra_count = frame_count - (2 + child_count)
    if extra_count < 0:
        return None

    outer_h = random.randint(12, 15 if child_count == 2 else 14)
    outer_w = random.randint(12, 16)
    outer = _rectangle_frame(outer_h, outer_w)
    children = _choose_child_frames(child_count, outer_h, outer_w)
    if children is None:
        return None

    cfg = _GRID_SIZE_BY_FRAMES[frame_count]
    _h_min, _h_max, _w_min, _w_max, f_max_h, f_max_w = cfg
    competitor_rooms = 1 + child_count
    competitor = _build_frame(competitor_rooms, max_h=f_max_h, max_w=f_max_w)
    if competitor is None:
        return None

    extra_frames: list[np.ndarray] = []
    extra_rooms: list[int] = []
    for extra_idx in range(extra_count):
        target_rooms = min(_ROOM_COUNT_RANGE[1], competitor_rooms + 1 + extra_idx)
        frame = _build_frame(target_rooms, max_h=f_max_h, max_w=f_max_w)
        if frame is None:
            return None
        extra_frames.append(frame)
        extra_rooms.append(target_rooms)

    stack_width = max([1 + child_count, competitor_rooms] + extra_rooms)
    content_left = stack_width + _STACK_SEPARATION

    row0_items = [competitor] + ([extra_frames[0]] if extra_frames else [])
    row1_items = [outer] + ([extra_frames[1]] if len(extra_frames) >= 2 else [])
    row0_w = sum(item.shape[1] for item in row0_items) + _TOP_LEVEL_GAP * (len(row0_items) - 1)
    row1_w = sum(item.shape[1] for item in row1_items) + _TOP_LEVEL_GAP * (len(row1_items) - 1)
    row0_h = max(item.shape[0] for item in row0_items)
    row1_h = max(item.shape[0] for item in row1_items)
    outer_offset = random.randint(2, 4)

    grid_w = content_left + max(row0_w, outer_offset + row1_w) + _LAYOUT_SLACK
    grid_h = row0_h + _TOP_LEVEL_GAP + row1_h + _LAYOUT_SLACK
    if grid_h < cfg[0] or grid_h > cfg[1] or grid_w < cfg[2] or grid_w > cfg[3]:
        return None

    frames: list[tuple[int, int, np.ndarray]] = []
    positions: list[tuple[int, int]] = []
    colors = random.sample(list(range(1, 10)), frame_count)

    competitor_pos = (0, content_left)
    outer_pos = (row0_h + _TOP_LEVEL_GAP, content_left + outer_offset)

    frames.append((colors[0], 1, outer))
    positions.append(outer_pos)
    for idx, (child, rel_pos) in enumerate(children, start=1):
        frames.append((colors[idx], 1, child))
        positions.append((outer_pos[0] + rel_pos[0], outer_pos[1] + rel_pos[1]))

    frames.append((colors[1 + child_count], competitor_rooms, competitor))
    positions.append(competitor_pos)

    next_color = 2 + child_count
    if extra_frames:
        extra0_pos = (0, competitor_pos[1] + competitor.shape[1] + _TOP_LEVEL_GAP)
        frames.append((colors[next_color], extra_rooms[0], extra_frames[0]))
        positions.append(extra0_pos)
        next_color += 1
    if len(extra_frames) >= 2:
        extra1_pos = (outer_pos[0], outer_pos[1] + outer.shape[1] + _TOP_LEVEL_GAP)
        frames.append((colors[next_color], extra_rooms[1], extra_frames[1]))
        positions.append(extra1_pos)

    for pos, (_color, _rooms, wall_arr) in zip(positions, frames):
        box = _bbox_of(pos, wall_arr)
        if box[2] >= grid_h or box[3] >= grid_w:
            return None

    comp_box = _bbox_of(competitor_pos, competitor)
    outer_box = _bbox_of(outer_pos, outer)
    if not (comp_box[0] < outer_box[0] and comp_box[1] < outer_box[1]):
        return None

    return frames, positions


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    built = _build_layout(frame_count)
    if built is None:
        return None
    frames, positions = built

    grid_h = max(pos[0] + frame[2].shape[0] for pos, frame in zip(positions, frames))
    grid_w = max(pos[1] + frame[2].shape[1] for pos, frame in zip(positions, frames))
    cfg = _GRID_SIZE_BY_FRAMES[frame_count]
    if grid_h < cfg[0] or grid_h > cfg[1] or grid_w < cfg[2] or grid_w > cfg[3]:
        return None

    grid = _render_frames(frames, positions, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(grid, frames, positions)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
