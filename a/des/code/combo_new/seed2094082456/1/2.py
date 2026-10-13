from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 8
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}
MAX_GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _bbox(cells: np.ndarray) -> tuple[int, int, int, int]:
    rows = cells[:, 0]
    cols = cells[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    coords = np.array(sorted((r - min_r, c - min_c) for r, c in cells), dtype=int)
    return coords


def _cell_set(cells: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in cells}


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _touches_boundary(cells: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.any(cells[:, 0] == 0)
        or np.any(cells[:, 0] == height - 1)
        or np.any(cells[:, 1] == 0)
        or np.any(cells[:, 1] == width - 1)
    )


def _in_bounds(cells: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.all(cells[:, 0] >= 0)
        and np.all(cells[:, 0] < height)
        and np.all(cells[:, 1] >= 0)
        and np.all(cells[:, 1] < width)
    )


def _overlaps(cells: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    return any(tuple(map(int, rc)) in occupied for rc in cells)


def _adjacent_to_occupied(cells: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if (nr, nc) in occupied:
                return True
    return False


def _adjacent_on_move_side(
    cells: np.ndarray, occupied: set[tuple[int, int]], move_vec: tuple[int, int]
) -> bool:
    dr, dc = move_vec
    for r, c in cells:
        if (int(r) + dr, int(c) + dc) in occupied:
            return True
    return False


def _shift(cells: np.ndarray, vec: tuple[int, int], steps: int = 1) -> np.ndarray:
    dr, dc = vec
    delta = np.array([int(dr) * int(steps), int(dc) * int(steps)], dtype=int)
    return cells + delta


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            q: deque[tuple[int, int]] = deque([(int(r0), int(c0))])
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            cells = np.array(sorted(pts), dtype=int)
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "area": int(len(pts)),
                    "bbox": _bbox(cells),
                }
            )
    return comps


def _arm_candidate(
    leaf: tuple[int, int], comp_set: set[tuple[int, int]], degrees: dict[tuple[int, int], int]
) -> dict | None:
    neighbors = [n for n in _neighbors4(*leaf) if n in comp_set]
    if len(neighbors) != 1:
        return None
    inner = neighbors[0]
    direction = (int(leaf[0] - inner[0]), int(leaf[1] - inner[1]))
    arm = [leaf]
    prev = leaf
    cur = inner

    while len(arm) < 2 and degrees[cur] == 2:
        nxts = [n for n in _neighbors4(*cur) if n in comp_set and n != prev]
        if len(nxts) != 1:
            break
        nxt = nxts[0]
        if (int(cur[0] - nxt[0]), int(cur[1] - nxt[1])) != direction:
            break
        arm.append(cur)
        prev, cur = cur, nxt

    if len(arm) == 2 and degrees[cur] == 2:
        nxts = [n for n in _neighbors4(*cur) if n in comp_set and n != prev]
        if len(nxts) == 1 and (int(cur[0] - nxts[0][0]), int(cur[1] - nxts[0][1])) == direction:
            return None

    remaining = comp_set - set(arm)
    if not remaining or not _is_connected(remaining):
        return None

    rows = [r for r, _ in comp_set]
    cols = [c for _, c in comp_set]
    rem_rows = [r for r, _ in remaining]
    rem_cols = [c for _, c in remaining]
    comp_r0, comp_r1 = min(rows), max(rows)
    comp_c0, comp_c1 = min(cols), max(cols)
    rem_r0, rem_r1 = min(rem_rows), max(rem_rows)
    rem_c0, rem_c1 = min(rem_cols), max(rem_cols)

    if direction == (-1, 0):
        if leaf[0] != comp_r0 or rem_r0 <= comp_r0 or len({c for _, c in arm}) != 1:
            return None
    elif direction == (1, 0):
        if leaf[0] != comp_r1 or rem_r1 >= comp_r1 or len({c for _, c in arm}) != 1:
            return None
    elif direction == (0, -1):
        if leaf[1] != comp_c0 or rem_c0 <= comp_c0 or len({r for r, _ in arm}) != 1:
            return None
    elif direction == (0, 1):
        if leaf[1] != comp_c1 or rem_c1 >= comp_c1 or len({r for r, _ in arm}) != 1:
            return None
    else:
        return None

    attach_neighbors = [n for n in _neighbors4(*cur) if n in comp_set]
    perpendicular_support = 0
    for nr, nc in attach_neighbors:
        if (nr, nc) == prev:
            continue
        if direction[0] != 0:
            perpendicular_support += int(nc != cur[1])
        else:
            perpendicular_support += int(nr != cur[0])

    if degrees[cur] < 2:
        return None

    score = 10 * len(arm) + 3 * degrees[cur] + perpendicular_support
    return {"dir": direction, "score": score}


def _edge_profile_fallback(cells: np.ndarray) -> tuple[int, int] | None:
    r0, c0, r1, c1 = _bbox(cells)
    comp = _cell_set(cells)
    dirs = []

    top = [(int(r), int(c)) for r, c in cells if int(r) == r0]
    if len(top) == 1 and (r0 + 1, top[0][1]) in comp:
        dirs.append((-1, 0))

    bottom = [(int(r), int(c)) for r, c in cells if int(r) == r1]
    if len(bottom) == 1 and (r1 - 1, bottom[0][1]) in comp:
        dirs.append((1, 0))

    left = [(int(r), int(c)) for r, c in cells if int(c) == c0]
    if len(left) == 1 and (left[0][0], c0 + 1) in comp:
        dirs.append((0, -1))

    right = [(int(r), int(c)) for r, c in cells if int(c) == c1]
    if len(right) == 1 and (right[0][0], c1 - 1) in comp:
        dirs.append((0, 1))

    if len(dirs) == 1:
        return dirs[0]
    return None


def _detect_spike_direction(cells: np.ndarray) -> tuple[int, int]:
    comp = _cell_set(cells)
    degrees = {
        p: sum((n in comp) for n in _neighbors4(*p))
        for p in comp
    }
    candidates = []
    for leaf, degree in degrees.items():
        if degree != 1:
            continue
        cand = _arm_candidate(leaf, comp, degrees)
        if cand is not None:
            candidates.append(cand)

    if candidates:
        candidates.sort(key=lambda item: (-int(item["score"]), DIR_NAMES[item["dir"]]))
        if len(candidates) == 1 or int(candidates[0]["score"]) > int(candidates[1]["score"]):
            return candidates[0]["dir"]

    fallback = _edge_profile_fallback(cells)
    if fallback is not None:
        return fallback

    raise AssertionError("failed to identify a unique spike direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome shape away from its unique thin spike until it hits the border or a larger fixed shape.

    Input structure:
    - The background is azure and forms the empty canvas.
    - Several non-overlapping colored shapes are scattered on the grid; each shape is a 4-connected single-color region.
    - Every shape has a compact body plus exactly one 1-pixel-wide protruding tip of length 1-2 pointing up, down, left, or right.

    Rule:
    - For each shape, detect the direction of its unique protruding tip.
    - Translate the whole shape one cell at a time in the opposite direction.
    - Process shapes from largest area to smallest area; once a shape is placed, later shapes treat it as a blocking obstacle.
    - Stop a moving shape as soon as it touches the grid border or becomes 4-neighbor adjacent to an already fixed shape.

    Key insight:
    - The thin tip acts like an arrow: it does not get removed, but instead tells you which direction the whole object must move away from, with area controlling the move order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    bg = _most_common_color(g)

    components = _find_components(g, bg)
    components.sort(key=lambda comp: (-int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["color"])))

    out = np.full((height, width), int(bg), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for comp in components:
        spike = _detect_spike_direction(comp["cells"])
        move_vec = (-int(spike[0]), -int(spike[1]))
        placed = comp["cells"].copy()

        while True:
            if _touches_boundary(placed, height, width) or _adjacent_to_occupied(placed, occupied):
                break
            nxt = _shift(placed, move_vec, 1)
            if not _in_bounds(nxt, height, width) or _overlaps(nxt, occupied):
                break
            placed = nxt

        out[placed[:, 0], placed[:, 1]] = int(comp["color"])
        occupied.update(_cell_set(placed))

    return out


def _make_shape(spike_dir: tuple[int, int]) -> dict | None:
    for _ in range(80):
        if spike_dir in {(-1, 0), (1, 0)}:
            body_h = random.randint(2, 5)
            body_w = random.randint(3, 5)
        else:
            body_h = random.randint(3, 5)
            body_w = random.randint(2, 5)

        cells = {(r, c) for r in range(body_h) for c in range(body_w)}
        if body_h >= 3 and body_w >= 3 and random.random() < 0.7:
            corner = random.choice([(0, 0), (0, body_w - 1), (body_h - 1, 0), (body_h - 1, body_w - 1)])
            cells.remove(corner)

        arm_len = random.randint(1, 2)
        if spike_dir == (-1, 0):
            anchor_c = random.randint(1, body_w - 2)
            for step in range(1, arm_len + 1):
                cells.add((-step, anchor_c))
        elif spike_dir == (1, 0):
            anchor_c = random.randint(1, body_w - 2)
            for step in range(1, arm_len + 1):
                cells.add((body_h - 1 + step, anchor_c))
        elif spike_dir == (0, -1):
            anchor_r = random.randint(1, body_h - 2)
            for step in range(1, arm_len + 1):
                cells.add((anchor_r, -step))
        else:
            anchor_r = random.randint(1, body_h - 2)
            for step in range(1, arm_len + 1):
                cells.add((anchor_r, body_w - 1 + step))

        norm = _normalize_cells(cells)
        if _detect_spike_direction(norm) != spike_dir:
            continue

        degrees = {
            p: sum((n in _cell_set(norm)) for n in _neighbors4(*p))
            for p in _cell_set(norm)
        }
        if sum(deg == 1 for deg in degrees.values()) != 1:
            continue

        h = int(norm[:, 0].max()) + 1
        w = int(norm[:, 1].max()) + 1
        return {
            "cells": norm,
            "cell_set": _cell_set(norm),
            "h": h,
            "w": w,
            "area": int(len(norm)),
            "spike": spike_dir,
            "move": (-int(spike_dir[0]), -int(spike_dir[1])),
        }
    return None


def _frontier_cells(shape: dict, move_vec: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = move_vec
    frontier = []
    for r, c in shape["cell_set"]:
        if (r + dr, c + dc) not in shape["cell_set"]:
            frontier.append((int(r), int(c)))
    return frontier


def _candidate_boundary_positions(shape: dict, height: int, width: int, occupied_final: set[tuple[int, int]]) -> list[np.ndarray]:
    move = shape["move"]
    positions: list[np.ndarray] = []

    if move == (0, -1):
        anchors = [(r0, 0) for r0 in range(height - int(shape["h"]) + 1)]
    elif move == (0, 1):
        anchors = [(r0, width - int(shape["w"])) for r0 in range(height - int(shape["h"]) + 1)]
    elif move == (-1, 0):
        anchors = [(0, c0) for c0 in range(width - int(shape["w"]) + 1)]
    else:
        anchors = [(height - int(shape["h"]), c0) for c0 in range(width - int(shape["w"]) + 1)]

    random.shuffle(anchors)
    for r0, c0 in anchors:
        pos = shape["cells"] + np.array([int(r0), int(c0)], dtype=int)
        if _overlaps(pos, occupied_final):
            continue
        if _adjacent_to_occupied(pos, occupied_final):
            continue
        positions.append(pos)
    return positions


def _candidate_obstacle_positions(shape: dict, height: int, width: int, occupied_final: set[tuple[int, int]]) -> list[np.ndarray]:
    if not occupied_final:
        return []

    move = shape["move"]
    side_cells = _frontier_cells(shape, move)
    occupied_list = list(occupied_final)
    random.shuffle(side_cells)
    random.shuffle(occupied_list)
    seen_offsets: set[tuple[int, int]] = set()
    positions: list[np.ndarray] = []

    for sr, sc in side_cells:
        for orow, ocol in occupied_list:
            r0 = int(orow - sr - move[0])
            c0 = int(ocol - sc - move[1])
            if (r0, c0) in seen_offsets:
                continue
            seen_offsets.add((r0, c0))
            pos = shape["cells"] + np.array([r0, c0], dtype=int)
            if not _in_bounds(pos, height, width):
                continue
            if _touches_boundary(pos, height, width):
                continue
            if _overlaps(pos, occupied_final):
                continue
            if not _adjacent_on_move_side(pos, occupied_final, move):
                continue
            positions.append(pos)
    random.shuffle(positions)
    return positions


def _endpoint_clear(cells: np.ndarray, occupied_input: set[tuple[int, int]]) -> bool:
    return not _overlaps(cells, occupied_input) and not _adjacent_to_occupied(cells, occupied_input)


def _feasible_reverse_steps(
    final_pos: np.ndarray,
    spike_vec: tuple[int, int],
    height: int,
    width: int,
    occupied_final: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
) -> list[int]:
    feasible: list[int] = []
    step = 0
    while True:
        step += 1
        pos = _shift(final_pos, spike_vec, step)
        if not _in_bounds(pos, height, width):
            break
        if _touches_boundary(pos, height, width):
            break
        if _overlaps(pos, occupied_final) or _adjacent_to_occupied(pos, occupied_final):
            break
        if _endpoint_clear(pos, occupied_input):
            feasible.append(step)
    return feasible


def _render(shape_positions: list[tuple[np.ndarray, int]], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for cells, color in shape_positions:
        grid[cells[:, 0], cells[:, 1]] = int(color)
    return grid


def _build_puzzle() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    n_shapes = random.choice([2, 3, 3, 4])
    colors = random.sample([c for c in range(10) if c != BG], n_shapes)

    shapes = []
    used_areas: set[int] = set()
    for _ in range(200):
        if len(shapes) == n_shapes:
            break
        spike = random.choice(DIRS)
        shape = _make_shape(spike)
        if shape is None or int(shape["area"]) in used_areas:
            continue
        used_areas.add(int(shape["area"]))
        shape["color"] = colors[len(shapes)]
        shapes.append(shape)
    if len(shapes) != n_shapes:
        return None

    shapes.sort(key=lambda shape: -int(shape["area"]))

    if n_shapes == 1:
        stop_modes = ["boundary"]
    else:
        obstacle_count = random.randint(1, n_shapes - 1)
        obstacle_idxs = set(random.sample(range(1, n_shapes), obstacle_count))
        stop_modes = ["boundary"]
        for idx in range(1, n_shapes):
            stop_modes.append("obstacle" if idx in obstacle_idxs else "boundary")

    occupied_final: set[tuple[int, int]] = set()
    occupied_input: set[tuple[int, int]] = set()
    final_positions: list[tuple[np.ndarray, int]] = []
    input_positions: list[tuple[np.ndarray, int]] = []

    for idx, shape in enumerate(shapes):
        if stop_modes[idx] == "boundary":
            finals = _candidate_boundary_positions(shape, height, width, occupied_final)
        else:
            finals = _candidate_obstacle_positions(shape, height, width, occupied_final)
        if not finals:
            return None

        placed = False
        for final_pos in finals[:80]:
            if stop_modes[idx] == "obstacle":
                spike_step = _shift(final_pos, shape["spike"], 1)
                if not _in_bounds(spike_step, height, width):
                    continue
                if _touches_boundary(spike_step, height, width):
                    continue
                if _overlaps(spike_step, occupied_final) or _adjacent_to_occupied(spike_step, occupied_final):
                    continue

            steps = _feasible_reverse_steps(
                final_pos,
                shape["spike"],
                height,
                width,
                occupied_final,
                occupied_input,
            )
            if not steps:
                continue
            step = random.choice(steps)
            input_pos = _shift(final_pos, shape["spike"], step)
            if not _endpoint_clear(input_pos, occupied_input):
                continue

            occupied_final.update(_cell_set(final_pos))
            occupied_input.update(_cell_set(input_pos))
            final_positions.append((final_pos, int(shape["color"])))
            input_positions.append((input_pos, int(shape["color"])))
            placed = True
            break
        if not placed:
            return None

    inp = _render(input_positions, (height, width), BG)
    out = _render(final_positions, (height, width), BG)

    if Counter(inp.ravel()).most_common(1)[0][0] != BG:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        built = _build_puzzle()
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
