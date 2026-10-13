import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
MAROON = 9
SIGNAL_COLORS = (1, 2, 3, 4, 6, 7, 8)

DOWN = (1, 0)
UP = (-1, 0)
RIGHT = (0, 1)
LEFT = (0, -1)

GRID_H_MIN = 15
GRID_H_MAX = 22
GRID_W_MIN = 15
GRID_W_MAX = 22
RAY_COUNT_MIN = 2
RAY_COUNT_MAX = 4
MAX_GENERATE_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _target_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    cells = set()
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            cells.add((r0 + dr, c0 + dc))
    return cells


def _prism_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0 + 1, c0 + 1),
    }


def _find_targets(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    claimed: set[tuple[int, int]] = set()
    targets: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            border_cells = _target_cells((r0, c0))
            if border_cells & claimed:
                continue
            center = (r0 + 1, c0 + 1)
            if int(grid[center]) != bg:
                continue
            colors = {int(grid[r, c]) for r, c in border_cells}
            if len(colors) != 1:
                continue
            border_color = next(iter(colors))
            if border_color == bg:
                continue
            claimed |= border_cells
            targets.append(
                {
                    "top_left": (r0, c0),
                    "center": center,
                    "border_color": border_color,
                    "cells": border_cells,
                }
            )
    return targets


def _find_prisms(grid: np.ndarray, bg: int, target_mask: set[tuple[int, int]]) -> list[dict]:
    h, w = grid.shape
    claimed: set[tuple[int, int]] = set()
    prisms: list[dict] = []
    for r0 in range(1, h - 2):
        for c0 in range(1, w - 2):
            cells = _prism_cells((r0, c0))
            if cells & target_mask or cells & claimed:
                continue
            window = grid[r0 : r0 + 2, c0 : c0 + 2]
            counts = Counter(int(v) for v in window.ravel())
            if len(counts) != 2:
                continue
            body_color = None
            marker_color = None
            for color, count in counts.items():
                if count == 3:
                    body_color = color
                elif count == 1:
                    marker_color = color
            if body_color is None or marker_color is None:
                continue
            if body_color == bg or marker_color == bg:
                continue
            marker_rel = None
            for dr in range(2):
                for dc in range(2):
                    if int(window[dr, dc]) == marker_color:
                        marker_rel = (dr, dc)
                        break
                if marker_rel is not None:
                    break
            if marker_rel is None:
                continue
            claimed |= cells
            prisms.append(
                {
                    "top_left": (r0, c0),
                    "body_color": body_color,
                    "marker_color": marker_color,
                    "marker_rel": marker_rel,
                    "cells": cells,
                }
            )
    return prisms


def _find_emitters(
    grid: np.ndarray,
    bg: int,
    reserved: set[tuple[int, int]],
) -> list[dict]:
    h, w = grid.shape
    emitters: list[dict] = []
    for c in range(1, w):
        if (0, c) in reserved:
            continue
        color = int(grid[0, c])
        if color != bg:
            emitters.append({"pos": (0, c), "color": color, "dir": DOWN})
    for r in range(1, h):
        if (r, 0) in reserved:
            continue
        color = int(grid[r, 0])
        if color != bg:
            emitters.append({"pos": (r, 0), "color": color, "dir": RIGHT})
    emitters.sort(key=lambda item: (item["pos"][0], item["pos"][1]))
    return emitters


def _prism_transition(prism: dict, direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    r0, c0 = prism["top_left"]
    mr, mc = prism["marker_rel"]
    if direction == DOWN:
        if mr != 1:
            return None
        out_dir = RIGHT if mc == 1 else LEFT
        exit_pos = (r0 + 1, c0 + 2) if mc == 1 else (r0 + 1, c0 - 1)
    elif direction == UP:
        if mr != 0:
            return None
        out_dir = RIGHT if mc == 1 else LEFT
        exit_pos = (r0, c0 + 2) if mc == 1 else (r0, c0 - 1)
    elif direction == RIGHT:
        if mc != 1:
            return None
        out_dir = DOWN if mr == 1 else UP
        exit_pos = (r0 + 2, c0 + 1) if mr == 1 else (r0 - 1, c0 + 1)
    elif direction == LEFT:
        if mc != 0:
            return None
        out_dir = DOWN if mr == 1 else UP
        exit_pos = (r0 + 2, c0) if mr == 1 else (r0 - 1, c0)
    else:
        return None
    return out_dir, exit_pos, int(prism["marker_color"])


def _visit_cell(
    source: np.ndarray,
    out: np.ndarray,
    pos: tuple[int, int],
    color: int,
    bg: int,
    target_centers: set[tuple[int, int]],
) -> None:
    r, c = pos
    if pos in target_centers:
        out[r, c] = color
    elif int(source[r, c]) == bg:
        out[r, c] = color


def _trace_ray(
    source: np.ndarray,
    out: np.ndarray,
    emitter: dict,
    prism_by_cell: dict[tuple[int, int], dict],
    target_centers: set[tuple[int, int]],
    bg: int,
) -> None:
    h, w = source.shape
    r, c = emitter["pos"]
    direction = emitter["dir"]
    color = int(emitter["color"])
    seen: set[tuple[int, int, int, int]] = set()
    for _ in range(h * w * 8):
        state = (r, c, direction[0], direction[1])
        if state in seen:
            break
        seen.add(state)
        nr = r + direction[0]
        nc = c + direction[1]
        if not _in_bounds(source.shape, nr, nc):
            break
        prism = prism_by_cell.get((nr, nc))
        if prism is not None:
            transition = _prism_transition(prism, direction)
            if transition is None:
                break
            direction, exit_pos, color = transition
            if not _in_bounds(source.shape, exit_pos[0], exit_pos[1]):
                break
            r, c = exit_pos
            _visit_cell(source, out, (r, c), color, bg, target_centers)
        else:
            r, c = nr, nc
            _visit_cell(source, out, (r, c), color, bg, target_centers)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    targets = _find_targets(g, bg)
    target_mask = {cell for target in targets for cell in target["cells"]}
    target_centers = {target["center"] for target in targets}

    prisms = _find_prisms(g, bg, target_mask)
    prism_mask = {cell for prism in prisms for cell in prism["cells"]}
    prism_by_cell = {}
    for prism in prisms:
        for cell in prism["cells"]:
            prism_by_cell[cell] = prism

    emitters = _find_emitters(g, bg, target_mask | prism_mask)
    out = g.copy()
    for emitter in emitters:
        _trace_ray(g, out, emitter, prism_by_cell, target_centers, bg)
    return out


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = direction
    return [(r0 + dr * step, c0 + dc * step) for step in range(1, length + 1)]


def _orthogonal_dirs(direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction in (DOWN, UP):
        return [LEFT, RIGHT]
    return [UP, DOWN]


def _marker_for_turn(direction: tuple[int, int], new_direction: tuple[int, int]) -> tuple[int, int]:
    mapping = {
        (DOWN, LEFT): (1, 0),
        (DOWN, RIGHT): (1, 1),
        (UP, LEFT): (0, 0),
        (UP, RIGHT): (0, 1),
        (RIGHT, UP): (0, 1),
        (RIGHT, DOWN): (1, 1),
        (LEFT, UP): (0, 0),
        (LEFT, DOWN): (1, 0),
    }
    return mapping[(direction, new_direction)]


def _invalid_markers(direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == DOWN:
        return [(0, 0), (0, 1)]
    if direction == UP:
        return [(1, 0), (1, 1)]
    if direction == RIGHT:
        return [(0, 0), (1, 0)]
    return [(0, 1), (1, 1)]


def _prism_top_left_options(pre: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pre
    if direction == DOWN:
        return [(r + 1, c - 1), (r + 1, c)]
    if direction == UP:
        return [(r - 2, c - 1), (r - 2, c)]
    if direction == RIGHT:
        return [(r - 1, c + 1), (r, c + 1)]
    return [(r - 1, c - 2), (r, c - 2)]


def _prism_from_spec(
    top_left: tuple[int, int],
    marker_rel: tuple[int, int],
    marker_color: int,
) -> dict:
    return {
        "top_left": top_left,
        "marker_rel": marker_rel,
        "marker_color": marker_color,
        "cells": _prism_cells(top_left),
    }


def _cells_are_free(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
        if (r, c) in blocked:
            return False
    return True


def _free_steps_from(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> int:
    r, c = start
    dr, dc = direction
    steps = 0
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return steps
        if (r, c) in blocked:
            return steps
        steps += 1


def _choose_signal(exclude: set[int]) -> int:
    choices = [color for color in SIGNAL_COLORS if color not in exclude]
    return random.choice(choices)


def _render_prism(grid: np.ndarray, prism: dict) -> None:
    r0, c0 = prism["top_left"]
    grid[r0 : r0 + 2, c0 : c0 + 2] = MAROON
    mr, mc = prism["marker_rel"]
    grid[r0 + mr, c0 + mc] = int(prism["marker_color"])


def _build_ray(
    shape: tuple[int, int],
    edge: str,
    blocked_objects: set[tuple[int, int]],
    blocked_path: set[tuple[int, int]],
    require_turn: bool,
) -> dict | None:
    h, w = shape
    if edge == "top":
        coords = [(0, c) for c in range(2, w - 2)]
        start_dir = DOWN
    else:
        coords = [(r, 0) for r in range(2, h - 2)]
        start_dir = RIGHT
    random.shuffle(coords)

    turn_choices = [1, 2, 3] if require_turn else [0, 1, 1, 2, 2, 3]
    for start in coords:
        if start in blocked_objects or start in blocked_path:
            continue
        emitter_color = _choose_signal(set())
        turn_goal = random.choice(turn_choices)
        local_objects = set(blocked_objects)
        local_objects.add(start)
        local_path = set(blocked_path)
        segments: list[dict] = []
        prisms: list[dict] = []
        current = start
        direction = start_dir
        current_color = emitter_color
        turns_done = 0
        success = True

        while turns_done < turn_goal:
            turned = False
            for _ in range(40):
                length = random.randint(2, 6)
                segment = _segment_cells(current, direction, length)
                if len(segment) != length or not _cells_are_free(segment, shape, local_objects | local_path):
                    continue
                pre = segment[-1]
                next_dirs = _orthogonal_dirs(direction)
                random.shuffle(next_dirs)
                for next_dir in next_dirs:
                    top_lefts = _prism_top_left_options(pre, direction)
                    random.shuffle(top_lefts)
                    for top_left in top_lefts:
                        marker_rel = _marker_for_turn(direction, next_dir)
                        marker_color = _choose_signal({current_color})
                        prism = _prism_from_spec(top_left, marker_rel, marker_color)
                        if not _cells_are_free(prism["cells"], shape, local_objects | local_path):
                            continue
                        transition = _prism_transition(prism, direction)
                        if transition is None:
                            continue
                        out_dir, exit_pos, _ = transition
                        if out_dir != next_dir:
                            continue
                        if not _in_bounds(shape, exit_pos[0], exit_pos[1]):
                            continue
                        if exit_pos in local_objects or exit_pos in local_path:
                            continue
                        min_after = 2 if turns_done + 1 < turn_goal else 1
                        if _free_steps_from(exit_pos, out_dir, shape, local_objects | local_path | prism["cells"]) < min_after:
                            continue
                        segments.append(
                            {
                                "dir": direction,
                                "cells": segment,
                                "color": current_color,
                                "after_turns": turns_done,
                            }
                        )
                        local_path.update(segment)
                        prisms.append(prism)
                        local_objects.update(prism["cells"])
                        local_path.add(exit_pos)
                        current = exit_pos
                        direction = out_dir
                        current_color = marker_color
                        turns_done += 1
                        turned = True
                        break
                    if turned:
                        break
                if turned:
                    break
            if not turned:
                success = False
                break

        if not success:
            continue

        stop_modes = ["boundary", "block"]
        random.shuffle(stop_modes)
        final_segment = None
        final_prism = None
        for stop_mode in stop_modes:
            if stop_mode == "boundary":
                steps = _free_steps_from(current, direction, shape, local_objects | local_path)
                if steps < 2:
                    continue
                final_segment = _segment_cells(current, direction, steps)
                break
            for _ in range(30):
                length = random.randint(2, 6)
                segment = _segment_cells(current, direction, length)
                if len(segment) != length or not _cells_are_free(segment, shape, local_objects | local_path):
                    continue
                pre = segment[-1]
                top_lefts = _prism_top_left_options(pre, direction)
                random.shuffle(top_lefts)
                markers = _invalid_markers(direction)
                random.shuffle(markers)
                for top_left in top_lefts:
                    for marker_rel in markers:
                        marker_color = _choose_signal({current_color})
                        prism = _prism_from_spec(top_left, marker_rel, marker_color)
                        if not _cells_are_free(prism["cells"], shape, local_objects | local_path):
                            continue
                        final_segment = segment
                        final_prism = prism
                        break
                    if final_segment is not None:
                        break
                if final_segment is not None:
                    break
            if final_segment is not None:
                break

        if final_segment is None:
            continue

        segments.append(
            {
                "dir": direction,
                "cells": final_segment,
                "color": current_color,
                "after_turns": turns_done,
            }
        )
        local_path.update(final_segment)
        if final_prism is not None:
            prisms.append(final_prism)
            local_objects.update(final_prism["cells"])

        return {
            "emitter": {"pos": start, "color": emitter_color, "dir": start_dir},
            "segments": segments,
            "prisms": prisms,
            "object_cells": local_objects - blocked_objects,
            "path_cells": local_path - blocked_path,
            "turn_count": turn_goal,
        }
    return None


def _can_place_hit_target(
    center: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    object_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> bool:
    r, c = center
    if not (1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1):
        return False
    allowed = {(r, c)}
    if direction in (UP, DOWN):
        allowed.update({(r - 1, c), (r + 1, c)})
    else:
        allowed.update({(r, c - 1), (r, c + 1)})
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            cell = (rr, cc)
            if cell in object_cells:
                return False
            if cell in path_cells and cell not in allowed:
                return False
    return True


def _make_target(center: tuple[int, int]) -> dict:
    r, c = center
    top_left = (r - 1, c - 1)
    return {"top_left": top_left, "center": center, "cells": _target_cells(top_left)}


def _place_hit_targets(
    shape: tuple[int, int],
    rays: list[dict],
    object_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> list[dict]:
    candidates: list[dict] = []
    for ray in rays:
        for segment in ray["segments"]:
            cells = segment["cells"]
            if len(cells) < 3:
                continue
            for idx in range(1, len(cells) - 1):
                center = cells[idx]
                if _can_place_hit_target(center, segment["dir"], shape, object_cells, path_cells):
                    candidates.append(
                        {
                            "center": center,
                            "dir": segment["dir"],
                            "after_turns": segment["after_turns"],
                        }
                    )

    if not candidates:
        return []

    chosen: list[dict] = []
    used_objects = set(object_cells)
    used_centers: set[tuple[int, int]] = set()

    def try_add(pool: list[dict]) -> bool:
        random.shuffle(pool)
        for candidate in pool:
            center = candidate["center"]
            if center in used_centers:
                continue
            if not _can_place_hit_target(center, candidate["dir"], shape, used_objects, path_cells):
                continue
            target = _make_target(center)
            chosen.append(target)
            used_centers.add(center)
            used_objects.update(target["cells"])
            return True
        return False

    turned = [candidate for candidate in candidates if candidate["after_turns"] > 0]
    if not try_add(turned):
        return []

    before_turn = [candidate for candidate in candidates if candidate["after_turns"] == 0]
    if before_turn and random.random() < 0.7:
        try_add(before_turn)

    target_goal = random.randint(len(chosen), min(len(chosen) + 2, 3))
    while len(chosen) < target_goal:
        if not try_add(candidates):
            break
    return chosen


def _place_unused_targets(
    shape: tuple[int, int],
    object_cells: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> list[dict]:
    targets: list[dict] = []
    blocked = set(object_cells) | set(path_cells)
    wanted = random.randint(0, 2)
    for _ in range(wanted):
        placed = False
        spots = [(r, c) for r in range(1, shape[0] - 1) for c in range(1, shape[1] - 1)]
        random.shuffle(spots)
        for center in spots:
            target = _make_target(center)
            full_cells = target["cells"] | {center}
            if any(cell in blocked for cell in full_cells):
                continue
            targets.append(target)
            blocked.update(full_cells)
            placed = True
            break
        if not placed:
            break
    return targets


def _render_targets(grid: np.ndarray, targets: list[dict]) -> None:
    for target in targets:
        for r, c in target["cells"]:
            grid[r, c] = GREY


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(GRID_H_MIN, GRID_H_MAX)
    w = random.randint(GRID_W_MIN, GRID_W_MAX)
    shape = (h, w)

    ray_count = random.randint(RAY_COUNT_MIN, RAY_COUNT_MAX)
    edges = ["top", "left"]
    while len(edges) < ray_count:
        edges.append(random.choice(["top", "left"]))
    random.shuffle(edges)

    object_cells: set[tuple[int, int]] = set()
    path_cells: set[tuple[int, int]] = set()
    rays: list[dict] = []
    for idx, edge in enumerate(edges):
        ray = _build_ray(shape, edge, object_cells, path_cells, require_turn=(idx == 0))
        if ray is None:
            return None
        rays.append(ray)
        object_cells.update(ray["object_cells"])
        path_cells.update(ray["path_cells"])

    hit_targets = _place_hit_targets(shape, rays, object_cells, path_cells)
    if not hit_targets:
        return None
    for target in hit_targets:
        object_cells.update(target["cells"])

    unused_targets = _place_unused_targets(shape, object_cells, path_cells)
    for target in unused_targets:
        object_cells.update(target["cells"] | {target["center"]})

    grid = np.full(shape, BG, dtype=int)
    for ray in rays:
        er, ec = ray["emitter"]["pos"]
        grid[er, ec] = int(ray["emitter"]["color"])
        for prism in ray["prisms"]:
            _render_prism(grid, prism)
    _render_targets(grid, hit_targets + unused_targets)

    out = main(grid)
    hit_centers = [target["center"] for target in hit_targets]
    if any(int(out[r, c]) == BG for r, c in hit_centers):
        return None
    if np.array_equal(grid, out):
        return None
    if sum(ray["turn_count"] for ray in rays) < 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid prism-ray puzzle")
