from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


TOGGLE = {
    Color.BLUE: Color.TEAL,
    Color.TEAL: Color.BLUE,
    Color.GREEN: Color.YELLOW,
    Color.YELLOW: Color.GREEN,
}

COLORS = tuple(TOGGLE)
SIDES = ("top", "bottom", "left", "right")


def _gray_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = components_mask(grid == Color.GREY, connectivity=4)
    assert len(comps) == 1, "expected one gray attractor"
    box = bbox_coords(comps[0])
    r0, c0, r1, c1 = box
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == Color.GREY), "gray attractor must be a filled rectangle"
    return box


def _barrier_info(grid: np.ndarray) -> dict:
    comps = components_mask(grid == Color.RED, connectivity=4)
    assert len(comps) == 1, "expected one red barrier"
    coords = comps[0]
    r0, c0, r1, c1 = bbox_coords(coords)
    horizontal = r0 == r1
    vertical = c0 == c1
    assert horizontal ^ vertical, "barrier must be a straight 1-cell-wide segment"
    length = int(coords.shape[0])
    assert 3 <= length <= 6
    return {
        "coords": {(int(r), int(c)) for r, c in coords},
        "bbox": (r0, c0, r1, c1),
        "horizontal": horizontal,
    }


def _objects(grid: np.ndarray) -> list[dict]:
    mask = np.isin(grid, list(COLORS))
    coords = np.argwhere(mask)
    objects = []
    for r, c in coords:
        color = int(grid[int(r), int(c)])
        objects.append({"start": (int(r), int(c)), "color": color})
    assert 4 <= len(objects) <= 6
    return objects


def _direction_distance(pos: tuple[int, int], gray_box: tuple[int, int, int, int]) -> tuple[str, int]:
    r, c = pos
    r0, c0, r1, c1 = gray_box
    candidates: list[tuple[int, str]] = []
    if c0 <= c <= c1:
        if r < r0:
            candidates.append((r0 - r - 1, "down"))
        if r > r1:
            candidates.append((r - r1 - 1, "up"))
    if r0 <= r <= r1:
        if c < c0:
            candidates.append((c0 - c - 1, "right"))
        if c > c1:
            candidates.append((c - c1 - 1, "left"))
    assert candidates, "every object must have a cardinal approach lane"
    candidates.sort()
    return candidates[0][1], int(candidates[0][0])


def _delta(direction: str) -> tuple[int, int]:
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]


def _trace_object(
    start: tuple[int, int],
    direction: str,
    occupied: set[tuple[int, int]],
    red_coords: set[tuple[int, int]],
) -> tuple[tuple[int, int], bool]:
    dr, dc = _delta(direction)
    r, c = start
    crossed = False
    while (r + dr, c + dc) not in occupied:
        r += dr
        c += dc
        crossed = crossed or ((r, c) in red_coords)
    assert (r, c) not in red_coords, "objects may cross the barrier but cannot stop on it"
    return (r, c), crossed


def _simulate(
    grid: np.ndarray,
    *,
    use_blockers: bool = True,
    use_toggle: bool = True,
    descending_order: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    gray_box = _gray_bbox(g)
    barrier = _barrier_info(g)
    objects = _objects(g)

    for obj in objects:
        direction, distance = _direction_distance(obj["start"], gray_box)
        obj["direction"] = direction
        obj["distance"] = distance

    objects.sort(
        key=lambda obj: (int(obj["distance"]), int(obj["start"][0]), int(obj["start"][1])),
        reverse=descending_order,
    )

    out = g.copy()
    for obj in objects:
        r, c = obj["start"]
        out[r, c] = Color.BLACK

    gray_cells = {(int(r), int(c)) for r in range(gray_box[0], gray_box[2] + 1) for c in range(gray_box[1], gray_box[3] + 1)}
    occupied = set(gray_cells)
    traces = []

    for obj in objects:
        final_pos, crossed = _trace_object(obj["start"], obj["direction"], occupied, barrier["coords"])
        final_color = TOGGLE[obj["color"]] if use_toggle and crossed else obj["color"]
        out[final_pos] = final_color
        trace = {
            "start": obj["start"],
            "direction": obj["direction"],
            "distance": obj["distance"],
            "crossed": crossed,
            "final_pos": final_pos,
            "input_color": obj["color"],
            "final_color": final_color,
        }
        traces.append(trace)
        if use_blockers:
            occupied.add(final_pos)

    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """Slide colored pixels toward the gray rectangle and toggle colors when their path crosses the red barrier.

    Input structure:
    - A black background contains one filled gray rectangle, one straight red barrier segment, and 4-6 single-pixel objects.
    - Object colors come from two visible toggle pairs: blue/teal and green/yellow.
    - Each object lies on a cardinal approach lane of the gray rectangle, while the red segment sits on one side of that rectangle.

    Rule:
    - For each object, choose the unique shortest cardinal direction that reaches the gray rectangle's surface lane.
    - Process objects from nearest to farthest; each one slides straight until the next step would hit the gray rectangle or an already docked object.
    - If any traversed path cell belongs to the red barrier, toggle the object's color within its fixed pair; otherwise keep its color.
    - Clear the original object cells, keep the gray rectangle and red barrier unchanged, and place each object at its final docked location.

    Key insight:
    - The barrier is not a wall: it only changes color when a sliding path actually passes through it, while earlier docked objects alter later stopping positions.
    """
    out, _traces = _simulate(grid)
    return out


def _gray_shape_for_side(side: str) -> tuple[int, int]:
    return (2, 3) if side in {"top", "bottom"} else (3, 2)


def _lane_coord(box: tuple[int, int, int, int], side: str, lane: int, distance: int) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return r0 - 1 - distance, c0 + lane
    if side == "bottom":
        return r1 + 1 + distance, c0 + lane
    if side == "left":
        return r0 + lane, c0 - 1 - distance
    return r0 + lane, c1 + 1 + distance


def _other_sides(side: str) -> tuple[str, ...]:
    return tuple(s for s in SIDES if s != side)


def _pick_canvas(side: str, barrier_len: int, cover_right_pair: bool) -> tuple[int, int, tuple[int, int, int, int]]:
    h, w = _gray_shape_for_side(side)
    for _ in range(200):
        H = random.randint(17, 20)
        W = random.randint(17, 20)
        if side in {"top", "bottom"}:
            if cover_right_pair:
                c0_min = 2
                c0_max = W - barrier_len - 2
            else:
                c0_min = barrier_len
                c0_max = W - 5
            if c0_min > c0_max:
                continue
            c0 = random.randint(c0_min, c0_max)
            if side == "top":
                r0 = random.randint(10, H - h - 4)
            else:
                r0 = random.randint(4, H - h - 10)
        else:
            if cover_right_pair:
                r0_min = 2
                r0_max = H - barrier_len - 2
            else:
                r0_min = barrier_len
                r0_max = H - 5
            if r0_min > r0_max:
                continue
            r0 = random.randint(r0_min, r0_max)
            if side == "left":
                c0 = random.randint(10, W - w - 4)
            else:
                c0 = random.randint(4, W - w - 10)
        return H, W, (r0, c0, r0 + h - 1, c0 + w - 1)
    raise RuntimeError("failed to place gray rectangle")


def _barrier_segment(
    box: tuple[int, int, int, int],
    side: str,
    gap: int,
    barrier_len: int,
    cover_right_pair: bool,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    if side == "top":
        row = r0 - gap - 1
        start = c0 + 1 if cover_right_pair else c0 + 1 - barrier_len
        for c in range(start, start + barrier_len):
            cells.append((row, c))
    elif side == "bottom":
        row = r1 + gap + 1
        start = c0 + 1 if cover_right_pair else c0 + 1 - barrier_len
        for c in range(start, start + barrier_len):
            cells.append((row, c))
    elif side == "left":
        col = c0 - gap - 1
        start = r0 + 1 if cover_right_pair else r0 + 1 - barrier_len
        for r in range(start, start + barrier_len):
            cells.append((r, col))
    else:
        col = c1 + gap + 1
        start = r0 + 1 if cover_right_pair else r0 + 1 - barrier_len
        for r in range(start, start + barrier_len):
            cells.append((r, col))
    return cells


def _covered_lanes(cover_right_pair: bool) -> tuple[list[int], int]:
    if cover_right_pair:
        return [1, 2], 0
    return [0, 1], 2


def _free_side_lanes(
    side: str,
    gray_box: tuple[int, int, int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> list[tuple[str, int]]:
    h = gray_box[2] - gray_box[0] + 1
    w = gray_box[3] - gray_box[1] + 1
    lanes = []
    for cand in _other_sides(side):
        if _max_distance(gray_box, cand, shape) < 1:
            continue
        count = w if cand in {"top", "bottom"} else h
        for lane in range(count):
            if _lane_coord(gray_box, cand, lane, 1) not in occupied:
                lanes.append((cand, lane))
    return lanes


def _max_distance(box: tuple[int, int, int, int], side: str, shape: tuple[int, int]) -> int:
    r0, c0, r1, c1 = box
    H, W = shape
    if side == "top":
        return r0 - 1
    if side == "bottom":
        return H - r1 - 2
    if side == "left":
        return c0 - 1
    return W - c1 - 2


def _build_candidate() -> np.ndarray:
    side = random.choice(SIDES)
    barrier_len = random.randint(3, 6)
    cover_right_pair = random.choice([False, True])
    gap = random.randint(2, 4)
    H, W, gray_box = _pick_canvas(side, barrier_len, cover_right_pair)
    grid = np.full((H, W), Color.BLACK, dtype=int)

    r0, c0, r1, c1 = gray_box
    grid[r0 : r1 + 1, c0 : c1 + 1] = Color.GREY
    for r, c in _barrier_segment(gray_box, side, gap, barrier_len, cover_right_pair):
        grid[r, c] = Color.RED

    covered_lanes, uncovered_lane = _covered_lanes(cover_right_pair)
    barrier_distances = [gap + 1, gap + 2, gap + 3, gap + 4]
    lane_plan = [
        (covered_lanes[0], barrier_distances[0]),
        (covered_lanes[1], barrier_distances[1]),
        (covered_lanes[0], barrier_distances[2]),
        (covered_lanes[1], barrier_distances[3]),
    ]
    random.shuffle(lane_plan)
    colors = [Color.BLUE, Color.TEAL, Color.GREEN, Color.YELLOW]
    random.shuffle(colors)

    occupied = set()
    for color, (lane, distance) in zip(colors, lane_plan):
        pos = _lane_coord(gray_box, side, lane, distance)
        occupied.add(pos)
        grid[pos] = color

    same_side_dist = gap + 5
    same_side_color = random.choice(COLORS)
    pos = _lane_coord(gray_box, side, uncovered_lane, same_side_dist)
    occupied.add(pos)
    grid[pos] = same_side_color

    if random.random() < 0.55:
        free_lanes = _free_side_lanes(side, gray_box, grid.shape, occupied)
        extra_side, extra_lane = random.choice(free_lanes)
        max_dist = min(5, _max_distance(gray_box, extra_side, grid.shape))
        extra_dist = random.randint(1, max_dist)
        pos = _lane_coord(gray_box, extra_side, extra_lane, extra_dist)
        grid[pos] = random.choice(COLORS)

    return grid


def _validate_case(grid: np.ndarray) -> bool:
    try:
        gray_box = _gray_bbox(grid)
        barrier = _barrier_info(grid)
        out, traces = _simulate(grid)
        no_toggle, _ = _simulate(grid, use_toggle=False)
        no_blockers, _ = _simulate(grid, use_blockers=False)
        reverse_order, _ = _simulate(grid, descending_order=True)
    except AssertionError:
        return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if sorted(trace["distance"] for trace in traces) != list(dict.fromkeys(sorted(trace["distance"] for trace in traces))):
        return False
    if not {Color.BLUE, Color.TEAL, Color.GREEN, Color.YELLOW}.issubset(set(int(v) for v in grid.ravel())):
        return False
    crossed = [trace for trace in traces if trace["crossed"]]
    not_crossed = [trace for trace in traces if not trace["crossed"]]
    if len(crossed) < 4 or len(not_crossed) < 1:
        return False
    side_dirs = {"top": "down", "bottom": "up", "left": "right", "right": "left"}
    barrier_side = None
    r0, c0, r1, c1 = gray_box
    br0, bc0, br1, bc1 = barrier["bbox"]
    if barrier["horizontal"]:
        barrier_side = "top" if br0 < r0 else "bottom"
    else:
        barrier_side = "left" if bc0 < c0 else "right"
    if not any(trace["direction"] == side_dirs[barrier_side] and not trace["crossed"] for trace in traces):
        return False
    if np.array_equal(out, no_toggle):
        return False
    if np.array_equal(out, no_blockers):
        return False
    if np.array_equal(out, reverse_order):
        return False
    if any(int(out[r, c]) != Color.RED for r, c in barrier["coords"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate()
        if _validate_case(grid):
            return grid
    raise RuntimeError("failed to generate a valid barrier-trigger puzzle")
