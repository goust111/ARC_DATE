from __future__ import annotations

import random
from itertools import permutations

import numpy as np


BLACK = 0
GREY = 5
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ACTION_LEFT = "L"
ACTION_RIGHT = "R"
ACTION_STOP = "S"
ALL_ACTIONS = (ACTION_LEFT, ACTION_RIGHT, ACTION_STOP)
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
LEGEND_SIZE = 5
GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _turn(direction: int, action: str) -> int:
    if action == ACTION_LEFT:
        return (direction - 1) % 4
    if action == ACTION_RIGHT:
        return (direction + 1) % 4
    raise ValueError(f"unexpected turn action: {action}")


def _legend_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], dict[int, str]]]:
    h, w = grid.shape
    found = []
    for r0 in range(h - LEGEND_SIZE + 1):
        for c0 in range(w - LEGEND_SIZE + 1):
            win = grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE]
            border = np.concatenate((win[0, :], win[-1, :], win[1:-1, 0], win[1:-1, -1]))
            border_color = int(border[0])
            if border_color == BLACK or not np.all(border == border_color):
                continue

            inner = win[1:4, 1:4]
            if int(inner[1, 1]) != GREY:
                continue

            left = int(inner[1, 0])
            right = int(inner[1, 2])
            up = int(inner[0, 1])
            if min(left, right, up) == BLACK:
                continue
            if len({left, right, up, GREY, border_color}) != 5:
                continue

            probe = inner.copy()
            probe[1, 0] = BLACK
            probe[1, 1] = BLACK
            probe[1, 2] = BLACK
            probe[0, 1] = BLACK
            if np.any(probe != BLACK):
                continue

            mapping = {left: ACTION_LEFT, right: ACTION_RIGHT, up: ACTION_STOP}
            found.append(((r0, c0, r0 + 4, c0 + 4), mapping))
    return found


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[int, str], np.ndarray]:
    candidates = _legend_candidates(grid)
    assert len(candidates) == 1, f"expected exactly one legend, got {len(candidates)}"
    bbox, mapping = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return bbox, mapping, mask


def _start_direction(shape: tuple[int, int], r: int, c: int) -> int:
    assert _is_edge(shape, r, c), "start must lie on the border"
    assert not ((r in (0, shape[0] - 1)) and (c in (0, shape[1] - 1))), "start corner is ambiguous"
    if r == 0:
        return 2
    if c == shape[1] - 1:
        return 3
    if r == shape[0] - 1:
        return 0
    return 1


def _find_start(grid: np.ndarray, legend_mask: np.ndarray) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    edge = np.zeros_like(grid, dtype=bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    coords = np.argwhere(edge & (~legend_mask) & (grid != BLACK))
    assert len(coords) == 1, f"expected exactly one border start, got {len(coords)}"
    r, c = map(int, coords[0])
    return (r, c), _start_direction((h, w), r, c)


def _trace_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: int,
    mapping: dict[int, str],
    legend_mask: np.ndarray,
    start_color: int,
) -> tuple[np.ndarray, dict]:
    out = grid.copy()
    h, w = grid.shape
    r, c = start
    hit_colors: list[int] = []
    turn_headings: list[int] = []
    painted = 0
    seen_states = {(r, c, direction)}

    while True:
        dr, dc = DIRS[direction]
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            break

        cell = int(grid[nr, nc])
        if cell == BLACK:
            out[nr, nc] = start_color
            painted += 1
            r, c = nr, nc
            if _is_edge((h, w), r, c):
                break
            continue

        assert not legend_mask[nr, nc], "ray should not enter the legend"
        assert cell in mapping, f"unexpected blocker color {cell}"
        action = mapping[cell]
        hit_colors.append(cell)
        if action == ACTION_STOP:
            break
        turn_headings.append(direction)
        direction = _turn(direction, action)
        state = (r, c, direction)
        assert state not in seen_states, "path should not loop"
        seen_states.add(state)

    return out, {
        "hit_colors": hit_colors,
        "turn_headings": turn_headings,
        "painted": painted,
    }


def _solve(grid: np.ndarray, override_mapping: dict[int, str] | None = None, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    legend_bbox, decoded_mapping, legend_mask = _find_legend(g)
    _ = legend_bbox
    start, direction = _find_start(g, legend_mask)
    start_color = int(g[start])
    action_by_color = decoded_mapping if override_mapping is None else override_mapping
    out, trace = _trace_ray(g, start, direction, action_by_color, legend_mask, start_color)
    if return_trace:
        return out, trace, decoded_mapping, legend_mask, start, direction
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the legend-defined turn semantics, then extend a start-colored ray that turns only when it hits the matching markers.

    Input structure
    - A small framed legend contains a grey anchor at its center, with three marker colors placed exactly at the anchor's left, right, and above positions.
    - The rest of the grid is a black canvas containing one colored start pixel on the outer border and several single-cell marker pixels using the legend's three colors.
    - Different examples may assign different colors to the legend's left, right, and upper positions.

    Rule
    - Interpret the color left of the grey anchor as "turn left relative to the current ray direction", the color on the right as "turn right", and the color above as "stop".
    - Launch a ray from the border start pixel toward the inside of the canvas; whenever the next cell is black, paint it with the start color and keep moving.
    - When the ray would enter a marker pixel, leave that marker unchanged; if it is a left or right marker, turn at the previous painted cell and continue, and if it is the stop marker or the ray reaches the canvas border, terminate.

    Key insight
    - Marker colors do not have fixed meanings: each example redefines them through the legend, and the turns are relative to the ray's current heading rather than absolute compass directions.
    """
    return _solve(grid)


def _legend_origins(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (1, 1),
        (1, w - LEGEND_SIZE - 1),
        (h - LEGEND_SIZE - 1, 1),
        (h - LEGEND_SIZE - 1, w - LEGEND_SIZE - 1),
    ]


def _expanded_legend_mask(h: int, w: int, origin: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r0, c0 = origin
    r1, c1 = r0 + LEGEND_SIZE - 1, c0 + LEGEND_SIZE - 1
    rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
    rr1, cc1 = min(h - 1, r1 + 1), min(w - 1, c1 + 1)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _edge_candidates(h: int, w: int, reserved: np.ndarray) -> list[tuple[tuple[int, int], int]]:
    candidates = []
    for c in range(1, w - 1):
        if not reserved[0, c]:
            candidates.append(((0, c), 2))
        if not reserved[h - 1, c]:
            candidates.append(((h - 1, c), 0))
    for r in range(1, h - 1):
        if not reserved[r, 0]:
            candidates.append(((r, 0), 1))
        if not reserved[r, w - 1]:
            candidates.append(((r, w - 1), 3))
    return candidates


def _segment_clear_run(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: int,
    blocked: set[tuple[int, int]],
    reserved: np.ndarray,
) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    r, c = current
    cells = []
    step = 1
    while True:
        nr, nc = r + dr * step, c + dc * step
        if not _in_bounds(shape, nr, nc):
            break
        if reserved[nr, nc] or (nr, nc) in blocked:
            break
        cells.append((nr, nc))
        step += 1
    return cells


def _search_path(
    shape: tuple[int, int],
    current: tuple[int, int],
    direction: int,
    actions: list[str],
    final_stop: str,
    blocked: set[tuple[int, int]],
    reserved: np.ndarray,
    records: list[dict],
) -> bool:
    clear_run = _segment_clear_run(shape, current, direction, blocked, reserved)
    if not actions:
        if final_stop == ACTION_STOP:
            candidates = [idx for idx, cell in enumerate(clear_run, start=1) if idx >= 2 and not _is_edge(shape, *cell)]
            random.shuffle(candidates)
            for dist in candidates:
                marker = clear_run[dist - 1]
                path_cells = clear_run[: dist - 1]
                blocked.update(path_cells)
                blocked.add(marker)
                records.append(
                    {
                        "from": current,
                        "dir": direction,
                        "path_cells": path_cells,
                        "marker": marker,
                        "action": ACTION_STOP,
                    }
                )
                return True
            return False

        if len(clear_run) < 2 or not _is_edge(shape, *clear_run[-1]):
            return False
        path_cells = clear_run
        blocked.update(path_cells)
        records.append(
            {
                "from": current,
                "dir": direction,
                "path_cells": path_cells,
                "marker": None,
                "action": "E",
            }
        )
        return True

    action = actions[0]
    candidates = [idx for idx, cell in enumerate(clear_run, start=1) if idx >= 2 and not _is_edge(shape, *cell)]
    random.shuffle(candidates)
    for dist in candidates:
        marker = clear_run[dist - 1]
        path_cells = clear_run[: dist - 1]
        if not path_cells:
            continue
        pivot = path_cells[-1]
        new_dir = _turn(direction, action)

        blocked.update(path_cells)
        blocked.add(marker)
        records.append(
            {
                "from": current,
                "dir": direction,
                "path_cells": path_cells,
                "marker": marker,
                "action": action,
            }
        )
        if _search_path(shape, pivot, new_dir, actions[1:], final_stop, blocked, reserved, records):
            return True
        records.pop()
        blocked.remove(marker)
        for cell in path_cells:
            blocked.remove(cell)
    return False


def _choose_turn_sequence() -> list[str]:
    turn_count = random.randint(2, 5)
    seq = [ACTION_LEFT, ACTION_RIGHT]
    while len(seq) < turn_count:
        seq.append(random.choice((ACTION_LEFT, ACTION_RIGHT)))
    random.shuffle(seq)
    return seq


def _build_legend(grid: np.ndarray, origin: tuple[int, int], border_color: int, action_to_color: dict[str, int]) -> None:
    r0, c0 = origin
    grid[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE] = border_color
    grid[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4] = BLACK
    grid[r0 + 2, c0 + 2] = GREY
    grid[r0 + 2, c0 + 1] = action_to_color[ACTION_LEFT]
    grid[r0 + 2, c0 + 3] = action_to_color[ACTION_RIGHT]
    grid[r0 + 1, c0 + 2] = action_to_color[ACTION_STOP]


def _place_distractors(
    grid: np.ndarray,
    reserved: np.ndarray,
    blocked: set[tuple[int, int]],
    legend_colors: list[int],
    must_have_colors: list[int],
) -> None:
    h, w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if reserved[r, c] or (r, c) in blocked or grid[r, c] != BLACK:
                continue
            neighborhood = grid[r - 1 : r + 2, c - 1 : c + 2]
            if np.count_nonzero(neighborhood) == 0:
                candidates.append((r, c))
    random.shuffle(candidates)
    colors = list(must_have_colors)
    extra_count = random.randint(0, 3)
    for _ in range(extra_count):
        colors.append(random.choice(legend_colors))
    if len(candidates) < len(colors):
        return
    for (r, c), color in zip(candidates, colors):
        grid[r, c] = color


def _records_to_color_trace(records: list[dict], action_to_color: dict[str, int]) -> tuple[list[int], list[int]]:
    hit_colors = []
    headings = []
    for rec in records:
        if rec["marker"] is not None:
            hit_colors.append(action_to_color[rec["action"]])
            if rec["action"] != ACTION_STOP:
                headings.append(rec["dir"])
    return hit_colors, headings


def _semantic_checks(grid: np.ndarray, actual_mapping: dict[int, str], expected_trace: tuple[list[int], list[int]]) -> bool:
    expected_out, trace, decoded_mapping, _legend_mask, _start, _direction = _solve(grid, return_trace=True)
    if decoded_mapping != actual_mapping:
        return False

    hit_colors, headings = expected_trace
    if trace["hit_colors"] != hit_colors or trace["turn_headings"] != headings:
        return False
    if trace["painted"] < 4:
        return False
    if not any(h in (0, 2) for h in headings):
        return False
    if not any(h in (1, 3) for h in headings):
        return False

    colors = list(actual_mapping)
    for actions in permutations(ALL_ACTIONS):
        alt_action_to_color = dict(zip(actions, colors))
        alt_mapping = {color: action for action, color in alt_action_to_color.items()}
        if alt_mapping == actual_mapping:
            continue
        try:
            alt_out = _solve(grid, override_mapping=alt_mapping)
        except Exception:
            continue
        if np.array_equal(alt_out, expected_out):
            return False

    return True


def _generate_once() -> np.ndarray | None:
    h = random.randint(14, 24)
    w = random.randint(14, 24)
    grid = np.zeros((h, w), dtype=int)

    color_pool = random.sample(PALETTE, 5)
    legend_colors = color_pool[:3]
    start_color = color_pool[3]
    border_color = color_pool[4]
    action_to_color = dict(zip(ALL_ACTIONS, random.sample(legend_colors, 3)))
    color_to_action = {color: action for action, color in action_to_color.items()}

    legend_origin = random.choice(_legend_origins(h, w))
    reserved = _expanded_legend_mask(h, w, legend_origin)
    _build_legend(grid, legend_origin, border_color, action_to_color)

    starts = _edge_candidates(h, w, reserved)
    if not starts:
        return None
    start, direction = random.choice(starts)
    grid[start] = start_color

    actions = _choose_turn_sequence()
    final_stop = ACTION_STOP if random.random() < 0.65 else "E"

    blocked = {start}
    records: list[dict] = []
    if not _search_path((h, w), start, direction, actions, final_stop, blocked, reserved, records):
        return None

    for rec in records:
        marker = rec["marker"]
        if marker is not None:
            grid[marker] = action_to_color[rec["action"]]

    used_actions = {rec["action"] for rec in records if rec["action"] in (ACTION_LEFT, ACTION_RIGHT)}
    if used_actions != {ACTION_LEFT, ACTION_RIGHT}:
        return None

    must_have = []
    if final_stop == "E":
        must_have.append(action_to_color[ACTION_STOP])
    _place_distractors(grid, reserved, blocked, legend_colors, must_have)

    trace = _records_to_color_trace(records, action_to_color)
    if not _semantic_checks(grid, color_to_action, trace):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-turn-ray input")
