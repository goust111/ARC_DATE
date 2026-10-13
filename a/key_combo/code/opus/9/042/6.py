from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))
OBSTACLE_COLORS = (BLUE, RED, YELLOW, PINK, ORANGE, TEAL, MAROON)

MIN_SIZE = 12
MAX_SIZE = 15
MIN_OBSTACLES = 2
MAX_OBSTACLES = 4
MAX_GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            idx = len(components)
            labels[r, c] = idx
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, mask.shape):
                    if mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = idx
                        q.append((nr, nc))
            components.append(cells)
    return labels, components


def _paint_enclosed_black(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if out[r, c] == BLACK and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if out[r, c] == BLACK and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, out.shape):
            if out[nr, nc] == BLACK and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = (out == BLACK) & (~outside)
    out[enclosed] = GREY
    return out


def _simulate(
    grid: np.ndarray,
    *,
    recolor_segments: bool = True,
    fill_enclosed: bool = True,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    obstacle_mask = (g != BLACK) & (g != GREEN)
    obstacle_ids, obstacle_components = _label_components(obstacle_mask)

    out = g.copy()
    path_mask = np.zeros((h, w), dtype=bool)
    step_index = np.full((h, w), -1, dtype=int)
    segment_progress = np.full((h, w), -1, dtype=int)

    if obstacle_mask[0, 0]:
        raise AssertionError("start cell must be black")

    r = c = 0
    dir_idx = 0
    out[r, c] = GREEN
    path_mask[r, c] = True
    step_index[r, c] = 0
    segment_progress[r, c] = 0
    path = [(r, c)]
    current_segment = [(r, c)]
    progress_in_segment = 0
    turns: list[dict] = []
    stagnant_turns = 0

    while len(path) <= h * w * 2:
        dr, dc = DIRS[dir_idx]
        nr, nc = r + dr, c + dc

        if not _in_bounds(g.shape, nr, nc):
            turns.append(
                {
                    "reason": "boundary",
                    "segment": tuple(current_segment),
                    "segment_len": len(current_segment),
                    "at": (r, c),
                    "dir_idx": dir_idx,
                    "obstacle_id": None,
                    "obstacle_color": None,
                }
            )
            current_segment = []
            progress_in_segment = 0
            dir_idx = (dir_idx + 1) % 4
            stagnant_turns += 1
            if stagnant_turns >= 4:
                stop_reason = "trapped"
                break
            continue

        if path_mask[nr, nc]:
            stop_reason = "path"
            break

        if obstacle_mask[nr, nc]:
            obstacle_id = int(obstacle_ids[nr, nc])
            obstacle_color = int(g[nr, nc])
            if recolor_segments:
                for sr, sc in current_segment:
                    out[sr, sc] = obstacle_color
            turns.append(
                {
                    "reason": "obstacle",
                    "segment": tuple(current_segment),
                    "segment_len": len(current_segment),
                    "at": (r, c),
                    "dir_idx": dir_idx,
                    "hit_cell": (nr, nc),
                    "obstacle_id": obstacle_id,
                    "obstacle_color": obstacle_color,
                }
            )
            current_segment = []
            progress_in_segment = 0
            dir_idx = (dir_idx + 1) % 4
            stagnant_turns += 1
            if stagnant_turns >= 4:
                stop_reason = "trapped"
                break
            continue

        fr, fc = nr + dr, nc + dc
        if _in_bounds(g.shape, fr, fc) and path_mask[fr, fc]:
            turns.append(
                {
                    "reason": "lookahead",
                    "segment": tuple(current_segment),
                    "segment_len": len(current_segment),
                    "at": (r, c),
                    "dir_idx": dir_idx,
                    "obstacle_id": None,
                    "obstacle_color": None,
                }
            )
            current_segment = []
            progress_in_segment = 0
            dir_idx = (dir_idx + 1) % 4
            stagnant_turns += 1
            if stagnant_turns >= 4:
                stop_reason = "trapped"
                break
            continue

        stagnant_turns = 0
        r, c = nr, nc
        progress_in_segment += 1
        path.append((r, c))
        out[r, c] = GREEN
        path_mask[r, c] = True
        step_index[r, c] = len(path) - 1
        segment_progress[r, c] = progress_in_segment
        current_segment.append((r, c))
    else:
        stop_reason = "guard"

    if fill_enclosed:
        out = _paint_enclosed_black(out)

    hit_obstacle_ids = {event["obstacle_id"] for event in turns if event["reason"] == "obstacle"}
    turn_counts = {
        "boundary": sum(event["reason"] == "boundary" for event in turns),
        "obstacle": sum(event["reason"] == "obstacle" for event in turns),
        "lookahead": sum(event["reason"] == "lookahead" for event in turns),
    }
    return out, {
        "path": tuple(path),
        "path_mask": path_mask,
        "step_index": step_index,
        "segment_progress": segment_progress,
        "turns": turns,
        "turn_counts": turn_counts,
        "stop_reason": stop_reason,
        "obstacle_components": obstacle_components,
        "obstacle_ids": obstacle_ids,
        "hit_obstacle_ids": hit_obstacle_ids,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a clockwise spiral that bends around colored obstacles, recolors obstacle-terminated segments, and fills enclosed black regions grey.

    Input structure:
    - The input is a square black grid with 2-4 small colored obstacle components.
    - Each obstacle is a 1-2 cell 4-connected monochrome block, using colors other than black and green.
    - The top-left cell is empty and serves as the spiral start.

    Rule:
    - Start at (0,0), moving right, and draw a spiral path one cell at a time.
    - If the next cell is outside the grid, turn clockwise; if the next cell is already on the spiral, stop; if the next cell is an obstacle, turn clockwise without entering it; if the cell two steps ahead is already on the spiral, turn clockwise; otherwise move forward and draw green.
    - Each straight segment is recolored only when it ends at an obstacle turn: the whole finished segment takes that obstacle's color; boundary- and self-caused turns leave the segment green.
    - After the spiral stops, fill every black 4-connected region that cannot reach the grid boundary through black cells with grey, while keeping obstacle cells unchanged.

    Key insight:
    - Obstacles are not just blockers: they redirect the spiral, donate their color to the segment they terminate, and thereby change which black regions become enclosed and later flood-filled.
    """
    out, _meta = _simulate(grid, recolor_segments=True, fill_enclosed=True)
    return out


def _obstacle_structure_ok(grid: np.ndarray, *, require_final_count: bool) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        return False
    if not (MIN_SIZE <= g.shape[0] <= MAX_SIZE):
        return False
    if g[0, 0] != BLACK:
        return False

    mask = (g != BLACK) & (g != GREEN)
    _ids, components = _label_components(mask)
    if require_final_count:
        if not (MIN_OBSTACLES <= len(components) <= MAX_OBSTACLES):
            return False
    elif len(components) > MAX_OBSTACLES:
        return False

    for cells in components:
        if len(cells) not in (1, 2):
            return False
        colors = {int(g[r, c]) for r, c in cells}
        if len(colors) != 1 or next(iter(colors)) not in OBSTACLE_COLORS:
            return False
    return True


def _semantic_checks(grid: np.ndarray) -> bool:
    if not _obstacle_structure_ok(grid, require_final_count=True):
        return False

    output, meta = _simulate(grid, recolor_segments=True, fill_enclosed=True)
    if meta["stop_reason"] != "path":
        return False
    if meta["turn_counts"]["boundary"] < 1:
        return False
    if meta["turn_counts"]["obstacle"] < 2:
        return False
    if meta["turn_counts"]["lookahead"] < 1:
        return False
    if len(meta["hit_obstacle_ids"]) != len(meta["obstacle_components"]):
        return False
    if np.count_nonzero(output == GREY) == 0:
        return False

    path_colors = {int(output[r, c]) for r, c in meta["path"]}
    if GREEN not in path_colors:
        return False
    if not any(color != GREEN for color in path_colors):
        return False

    obstacle_turns = [event for event in meta["turns"] if event["reason"] == "obstacle"]
    if not obstacle_turns or any(event["segment_len"] == 0 for event in obstacle_turns):
        return False

    geometry_only, _ = _simulate(grid, recolor_segments=False, fill_enclosed=True)
    if np.array_equal(output, geometry_only):
        return False

    no_fill, _ = _simulate(grid, recolor_segments=True, fill_enclosed=False)
    if np.array_equal(output, no_fill):
        return False

    empty_grid = np.zeros_like(grid)
    empty_spiral, _ = _simulate(empty_grid, recolor_segments=True, fill_enclosed=True)
    if np.array_equal(output, empty_spiral):
        return False

    return True


def _candidate_obstacle_cells(meta: dict, grid: np.ndarray, min_step: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[int, int]] = []
    path_len = len(meta["path"])
    for r, c in meta["path"]:
        if (r, c) == (0, 0):
            continue
        step = int(meta["step_index"][r, c])
        if step <= min_step or step >= path_len - 3:
            continue
        if meta["segment_progress"][r, c] < 2:
            continue
        if g[r, c] != BLACK:
            continue
        adjacent_obstacles = any(
            ((g[nr, nc] != BLACK) and (g[nr, nc] != GREEN))
            for nr, nc in _neighbors4(r, c, g.shape)
        )
        if adjacent_obstacles:
            continue
        candidates.append((r, c))
    return candidates


def _extend_obstacle(
    grid: np.ndarray,
    seed: tuple[int, int],
    color: int,
    current_meta: dict,
    want_size_two: bool,
) -> np.ndarray | None:
    out = np.asarray(grid, dtype=int).copy()
    r, c = seed
    out[r, c] = color
    if not want_size_two:
        return out

    step_index = current_meta["step_index"]
    options = []
    for nr, nc in _neighbors4(r, c, out.shape):
        if out[nr, nc] != BLACK:
            continue
        prior_step = int(step_index[nr, nc])
        if prior_step != -1 and prior_step < int(step_index[r, c]):
            continue
        adjacent_other = False
        for ar, ac in _neighbors4(nr, nc, out.shape):
            if (ar, ac) == (r, c):
                continue
            if out[ar, ac] != BLACK:
                adjacent_other = True
                break
        if not adjacent_other and (nr, nc) != (0, 0):
            options.append((nr, nc))

    if not options:
        return None
    nr, nc = random.choice(options)
    out[nr, nc] = color
    return out


def _place_one_obstacle(
    grid: np.ndarray,
    color: int,
    want_size_two: bool,
    min_step: int,
) -> tuple[np.ndarray, int] | None:
    _output, meta = _simulate(grid, recolor_segments=True, fill_enclosed=True)
    candidates = _candidate_obstacle_cells(meta, grid, min_step)
    if not candidates:
        return None

    random.shuffle(candidates)
    for seed in candidates:
        proposed = _extend_obstacle(grid, seed, color, meta, want_size_two)
        if proposed is None:
            continue
        if not _obstacle_structure_ok(proposed, require_final_count=False):
            continue
        _, new_meta = _simulate(proposed, recolor_segments=True, fill_enclosed=True)
        obstacle_id = int(new_meta["obstacle_ids"][seed])
        if obstacle_id not in new_meta["hit_obstacle_ids"]:
            continue
        hit_events = [
            event
            for event in new_meta["turns"]
            if event["reason"] == "obstacle" and event["obstacle_id"] == obstacle_id
        ]
        if not hit_events or any(event["segment_len"] == 0 for event in hit_events):
            continue
        hit_step = int(new_meta["step_index"][hit_events[0]["at"]])
        return proposed, hit_step
    return None


def _generate_attempt() -> np.ndarray | None:
    size = random.randint(MIN_SIZE, MAX_SIZE)
    target_obstacles = random.choice((2, 2, 3, 3, 4))
    grid = np.zeros((size, size), dtype=int)
    min_step = 1

    color_pool = [random.choice(OBSTACLE_COLORS) for _ in range(target_obstacles)]
    for idx, color in enumerate(color_pool):
        want_size_two = random.random() < 0.25
        if idx == 0:
            want_size_two = want_size_two and random.random() < 0.6
        placed = _place_one_obstacle(grid, color, want_size_two, min_step)
        if placed is None:
            return None
        grid, min_step = placed

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate obstacle spiral input")
