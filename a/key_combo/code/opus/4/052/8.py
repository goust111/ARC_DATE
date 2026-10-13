import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
MAROON = 9

FRAME_COLOR = GREY
MARKER_COLORS = (BLUE, RED, GREEN)
OBJECT_COLORS = (YELLOW, PINK, ORANGE)
DIRS = ("up", "down", "left", "right")
MARKER_OFFSET = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MOVE_TO_FRAME = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

FRAME_SIDE = 5
GRID_MIN = 16
GRID_MAX = 22
GEN_RETRY_LIMIT = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_patch(direction: str, marker_color: int) -> np.ndarray:
    patch = np.full((FRAME_SIDE, FRAME_SIDE), BLACK, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    center = FRAME_SIDE // 2
    dr, dc = MARKER_OFFSET[direction]
    patch[center + dr, center + dc] = int(marker_color)
    return patch


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components_mask(grid == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox_coords(coords)
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1
        assert side % 2 == 1 and side >= 5
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch[0, :] == FRAME_COLOR)
        assert np.all(patch[-1, :] == FRAME_COLOR)
        assert np.all(patch[:, 0] == FRAME_COLOR)
        assert np.all(patch[:, -1] == FRAME_COLOR)
        interior = patch[1:-1, 1:-1]
        marker_coords = np.argwhere((interior != BLACK) & (interior != FRAME_COLOR))
        assert len(marker_coords) == 1
        marker_value = int(interior[int(marker_coords[0][0]), int(marker_coords[0][1])])
        mask_other = np.ones_like(interior, dtype=bool)
        mask_other[int(marker_coords[0][0]), int(marker_coords[0][1])] = False
        assert np.all(interior[mask_other] == BLACK)
        mr, mc = map(int, marker_coords[0] + 1)
        center = side // 2
        dr = mr - center
        dc = mc - center
        assert (dr == 0) != (dc == 0)
        if dr < 0:
            direction = "up"
        elif dr > 0:
            direction = "down"
        elif dc < 0:
            direction = "left"
        else:
            direction = "right"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "direction": direction,
                "marker": (r0 + mr, c0 + mc),
                "marker_color": marker_value,
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    assert 2 <= len(frames) <= 3
    return frames


def _relation_to_frame(r: int, c: int, frame: dict) -> tuple[str, int] | None:
    r0, c0, r1, c1 = frame["bbox"]
    if c0 <= c <= c1:
        if r < r0:
            return "up", int(r0 - r - 1)
        if r > r1:
            return "down", int(r - r1 - 1)
    if r0 <= r <= r1:
        if c < c0:
            return "left", int(c0 - c - 1)
        if c > c1:
            return "right", int(c - c1 - 1)
    return None


def _find_objects(grid: np.ndarray) -> list[dict]:
    rows, cols = np.where(np.isin(grid, OBJECT_COLORS))
    objects = []
    for r, c in zip(rows, cols):
        objects.append({"pos": (int(r), int(c)), "color": int(grid[int(r), int(c)])})
    objects.sort(key=lambda item: item["pos"])
    return objects


def _associate_objects(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    objects = _find_objects(grid)
    by_frame = defaultdict(int)
    for obj in objects:
        r, c = obj["pos"]
        frame_idx, side, gap = _match_object(r, c, frames)
        obj["frame_idx"] = int(frame_idx)
        obj["side"] = side
        obj["gap"] = int(gap)
        by_frame[int(frame_idx)] += 1
    assert all(by_frame[idx] >= 1 for idx in range(len(frames)))
    return objects


def _solve(grid: np.ndarray, *, trails: bool, accumulate: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _detect_frames(g)
    objects = _associate_objects(g, frames)
    for obj in objects:
        r, c = obj["pos"]
        out[r, c] = BLACK
    order = sorted(objects, key=lambda item: (item["gap"], item["pos"][0], item["pos"][1]))
    h, w = out.shape
    for obj in order:
        r, c = obj["pos"]
        dr, dc = MOVE_TO_FRAME[obj["side"]]
        cr, cc = r, c
        while True:
            nr, nc = cr + dr, cc + dc
            assert 0 <= nr < h and 0 <= nc < w
            cell = int(out[nr, nc])
            if cell == FRAME_COLOR or (accumulate and cell in OBJECT_COLORS):
                break
            cr, cc = nr, nc
        if trails and (cr, cc) != (r, c):
            tr, tc = r + dr, c + dc
            while (tr, tc) != (cr, cc):
                out[tr, tc] = MAROON
                tr += dr
                tc += dc
        out[cr, cc] = obj["color"]
    return out


def _match_object(r: int, c: int, frames: list[dict]) -> tuple[int, str, int]:
    candidates = []
    for idx, frame in enumerate(frames):
        rel = _relation_to_frame(r, c, frame)
        if rel is None:
            continue
        side, gap = rel
        if side != frame["direction"]:
            continue
        candidates.append((gap, idx, side))
    assert candidates
    best_gap = min(gap for gap, _idx, _side in candidates)
    best = [item for item in candidates if item[0] == best_gap]
    assert len(best) == 1
    gap, frame_idx, side = best[0]
    return int(frame_idx), side, int(gap)


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each outer pixel into its marked square frame, stacking at the marked side and leaving maroon sweep trails.

    Input structure:
    - A black background contains 2 or 3 separated gray hollow odd square frames.
    - Each frame has exactly one colored marker pixel inside, offset from the frame center by one cardinal step.
    - Yellow, pink, and orange single-pixel objects lie outside the frames, each aligned to exactly one frame along a row or column.

    Rule:
    - Read each frame's attraction side from the marker offset: above center means up, below means down, left means left, right means right.
    - An outer object is attracted only to the nearest aligned frame on that marked side.
    - Process objects by increasing empty gap to the target frame border, breaking ties from top to bottom and then left to right.
    - Slide each object straight toward its frame until the next step would hit the gray border or a previously docked object.
    - Keep the docked object's original color, clear its start cell, and paint every strictly intermediate traversed cell maroon.
    - Preserve the frames and their markers.

    Key insight:
    - The marker does not move; its offset inside the hollow square tells which side can accept incoming pixels, and already docked pixels change where later pixels stop.
    """
    return _solve(np.asarray(grid, dtype=int), trails=True, accumulate=True)


def _frame_side_space(frame: dict, shape: tuple[int, int]) -> int:
    h, w = shape
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["direction"]
    if side == "up":
        return int(r0)
    if side == "down":
        return int(h - r1 - 1)
    if side == "left":
        return int(c0)
    return int(w - c1 - 1)


def _lane_positions(frame: dict, gap: int) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["direction"]
    cells = []
    if side == "up":
        row = r0 - gap - 1
        for lane_idx, col in enumerate(range(c0, c1 + 1)):
            cells.append((row, col, lane_idx))
    elif side == "down":
        row = r1 + gap + 1
        for lane_idx, col in enumerate(range(c0, c1 + 1)):
            cells.append((row, col, lane_idx))
    elif side == "left":
        col = c0 - gap - 1
        for lane_idx, row in enumerate(range(r0, r1 + 1)):
            cells.append((row, col, lane_idx))
    else:
        col = c1 + gap + 1
        for lane_idx, row in enumerate(range(r0, r1 + 1)):
            cells.append((row, col, lane_idx))
    return cells


def _row_col_clear(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2


def _sample_gaps(max_gap: int, count: int, *, require_long: bool) -> list[int] | None:
    choices = list(range(max_gap + 1))
    if require_long and max_gap >= 2:
        long_choices = [gap for gap in choices if gap >= 2]
        if not long_choices:
            return None
        first = random.choice(long_choices)
        remaining = [gap for gap in choices if abs(gap - first) >= 2]
        if count == 1:
            return [first]
        if len(remaining) < count - 1:
            return None
        picked = random.sample(remaining, count - 1)
        return sorted([first, *picked])
    if len(choices) < count:
        return None
    random.shuffle(choices)
    picked: list[int] = []
    for gap in choices:
        if all(abs(gap - prev) >= 2 for prev in picked):
            picked.append(gap)
        if len(picked) == count:
            return sorted(picked)
    return None


def _build_grid(shape: tuple[int, int], frames: list[dict], objects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for frame in frames:
        r0, c0, _r1, _c1 = frame["bbox"]
        grid[r0 : r0 + FRAME_SIDE, c0 : c0 + FRAME_SIDE] = _frame_patch(frame["direction"], frame["marker_color"])
    for obj in objects:
        r, c = obj["pos"]
        grid[r, c] = int(obj["color"])
    return grid


def _random_frames(shape: tuple[int, int], n_frames: int) -> list[dict] | None:
    h, w = shape
    directions = random.sample(DIRS, n_frames)
    frames = []
    for direction in directions:
        placed = False
        for _ in range(120):
            r0 = random.randint(1, h - FRAME_SIDE - 2)
            c0 = random.randint(1, w - FRAME_SIDE - 2)
            bbox = (r0, c0, r0 + FRAME_SIDE - 1, c0 + FRAME_SIDE - 1)
            probe = {"bbox": bbox, "direction": direction}
            if _frame_side_space(probe, shape) < 4:
                continue
            ok = True
            for other in frames:
                or0, oc0, or1, oc1 = other["bbox"]
                if not (bbox[2] + 1 < or0 or or1 + 1 < bbox[0] or bbox[3] + 1 < oc0 or oc1 + 1 < bbox[1]):
                    ok = False
                    break
            if not ok:
                continue
            frames.append(
                {
                    "bbox": bbox,
                    "direction": direction,
                    "marker_color": random.choice(MARKER_COLORS),
                }
            )
            placed = True
            break
        if not placed:
            return None
    return frames


def _random_objects(shape: tuple[int, int], frames: list[dict]) -> list[dict] | None:
    h, w = shape
    objects: list[dict] = []
    frame_lane_use: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    stack_frame = random.randrange(len(frames))
    long_frame = random.randrange(len(frames))

    for idx, frame in enumerate(frames):
        max_gap = _frame_side_space(frame, shape) - 1
        if max_gap < 0:
            return None
        obj_count = random.randint(2, 4) if idx == stack_frame else random.randint(1, 3)
        lanes = list(range(5))
        random.shuffle(lanes)
        lane_counts: dict[int, int] = {}
        if idx == stack_frame:
            lane_counts[lanes[0]] = 2
            remaining = obj_count - 2
        else:
            remaining = obj_count
        pool = lanes if idx == stack_frame else lanes[:]
        while remaining > 0:
            lane = random.choice(pool)
            lane_counts[lane] = lane_counts.get(lane, 0) + 1
            remaining -= 1
        used_colors = list(OBJECT_COLORS)
        random.shuffle(used_colors)
        need_long = idx == long_frame
        for lane, count in lane_counts.items():
            gaps = _sample_gaps(max_gap, count, require_long=need_long and count >= 1)
            if gaps is None:
                return None
            need_long = False
            for j, gap in enumerate(gaps):
                coords_for_gap = [item for item in _lane_positions(frame, gap) if item[2] == lane]
                assert len(coords_for_gap) == 1
                r, c, _ = coords_for_gap[0]
                if not (0 <= r < h and 0 <= c < w):
                    return None
                if any((r, c) == other["pos"] or not _row_col_clear((r, c), other["pos"]) for other in objects):
                    return None
                try:
                    frame_idx, _side, matched_gap = _match_object(r, c, frames)
                except AssertionError:
                    return None
                if frame_idx != idx or matched_gap != gap:
                    return None
                color = used_colors[j % len(used_colors)]
                objects.append({"pos": (r, c), "color": int(color)})
                frame_lane_use[idx][lane] += 1
    has_stack = any(count >= 2 for lanes in frame_lane_use.values() for count in lanes.values())
    has_long_trail = False
    full_grid = _build_grid(shape, frames, objects)
    for obj in _associate_objects(full_grid, frames):
        if obj["gap"] >= 2:
            has_long_trail = True
            break
    if not has_stack or not has_long_trail:
        return None
    return objects


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        frames = _detect_frames(grid)
        objects = _associate_objects(grid, frames)
        if len({frame["direction"] for frame in frames}) < 2:
            return False
        if not any(obj["gap"] >= 2 for obj in objects):
            return False
        lane_counts: dict[tuple[int, int], int] = defaultdict(int)
        for obj in objects:
            r, c = obj["pos"]
            lane_key = c if obj["side"] in {"up", "down"} else r
            lane_counts[(obj["frame_idx"], lane_key)] += 1
        if not any(count >= 2 for count in lane_counts.values()):
            return False
        solved = _solve(grid, trails=True, accumulate=True)
        if np.array_equal(grid, solved):
            return False
        if np.array_equal(solved, _solve(grid, trails=False, accumulate=True)):
            return False
        if np.array_equal(solved, _solve(grid, trails=True, accumulate=False)):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    n_frames = random.choice((2, 3))
    frames = _random_frames(shape, n_frames)
    if frames is None:
        return None
    objects = _random_objects(shape, frames)
    if objects is None:
        return None
    grid = _build_grid(shape, frames, objects)
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-frame input")
