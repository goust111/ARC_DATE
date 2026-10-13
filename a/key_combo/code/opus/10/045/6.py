from __future__ import annotations

import random

import numpy as np


BG = 6
BORDER = 0
OBSTACLE = 2
GREY = 5
ORANGE = 7
MARKER = 9

RIGHT = (0, 1)
LEFT = (0, -1)
DOWN = (1, 0)
UP = (-1, 0)
DIRECTIONS = [RIGHT, LEFT, DOWN, UP]

GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _find_container_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    inner = np.argwhere((grid == GREY) | (grid == ORANGE))
    assert len(inner) > 0, "expected grey/orange container interior"
    ir0, ic0, ir1, ic1 = _bbox(inner)
    r0, c0, r1, c1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    assert r0 >= 0 and c0 >= 0 and r1 < grid.shape[0] and c1 < grid.shape[1], "container bbox out of bounds"
    border = _rect_border_coords(r0, c0, r1, c1)
    assert all(int(grid[r, c]) == BORDER for r, c in border), "expected a black rectangular border"
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert not np.any((patch != BORDER) & (patch != GREY) & (patch != ORANGE)), "container bbox has extra colors"
    assert np.all(np.isin(grid[r0 + 1 : r1, c0 + 1 : c1], [GREY, ORANGE])), "expected only grey/orange interior cells"
    return r0, c0, r1, c1


def _find_direction_and_target(
    grid: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = box
    maroon = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER)]
    assert len(maroon) == 2, "expected one direction marker and one target marker"

    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in maroon:
        if c == c0 - 1 and r0 <= r <= r1:
            candidates.append(((r, c), RIGHT))
        if c == c1 + 1 and r0 <= r <= r1:
            candidates.append(((r, c), LEFT))
        if r == r0 - 1 and c0 <= c <= c1:
            candidates.append(((r, c), DOWN))
        if r == r1 + 1 and c0 <= c <= c1:
            candidates.append(((r, c), UP))
    assert len(candidates) == 1, "expected a unique adjacent direction marker"

    marker_pos, direction = candidates[0]
    target = next(pos for pos in maroon if pos != marker_pos)
    dr, dc = direction
    if dr != 0:
        assert target[1] == marker_pos[1], "target must be aligned with the marker column"
        assert (target[0] - marker_pos[0]) * dr > 0, "target must lie ahead of the movement direction"
    else:
        assert target[0] == marker_pos[0], "target must be aligned with the marker row"
        assert (target[1] - marker_pos[1]) * dc > 0, "target must lie ahead of the movement direction"
    return direction, marker_pos, target


def _charge_order(inner_h: int, inner_w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    if direction == RIGHT:
        return [(r, c) for c in range(inner_w) for r in range(inner_h)]
    if direction == LEFT:
        return [(r, c) for c in range(inner_w - 1, -1, -1) for r in range(inner_h)]
    if direction == DOWN:
        return [(r, c) for r in range(inner_h) for c in range(inner_w)]
    return [(r, c) for r in range(inner_h - 1, -1, -1) for c in range(inner_w)]


def _sheared_offset(
    local_r: int, local_c: int, patch_shape: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int]:
    h, w = patch_shape
    if direction == RIGHT:
        return local_r + (w - 1 - local_c), local_c
    if direction == LEFT:
        return local_r + local_c, local_c
    if direction == DOWN:
        return local_r, local_c + (h - 1 - local_r)
    return local_r, local_c + local_r


def _occupied_coords(
    patch: np.ndarray, origin: tuple[int, int], direction: tuple[int, int]
) -> set[tuple[int, int]]:
    r0, c0 = origin
    coords: set[tuple[int, int]] = set()
    for local_r in range(patch.shape[0]):
        for local_c in range(patch.shape[1]):
            if int(patch[local_r, local_c]) == BG:
                continue
            rr, cc = _sheared_offset(local_r, local_c, patch.shape, direction)
            coords.add((r0 + rr, c0 + cc))
    return coords


def _unsheared_coords(patch: np.ndarray, origin: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = origin
    coords: set[tuple[int, int]] = set()
    for local_r in range(patch.shape[0]):
        for local_c in range(patch.shape[1]):
            if int(patch[local_r, local_c]) != BG:
                coords.add((r0 + local_r, c0 + local_c))
    return coords


def _stamp_sheared(
    grid: np.ndarray, patch: np.ndarray, origin: tuple[int, int], direction: tuple[int, int]
) -> None:
    r0, c0 = origin
    for local_r in range(patch.shape[0]):
        for local_c in range(patch.shape[1]):
            color = int(patch[local_r, local_c])
            if color == BG:
                continue
            rr, cc = _sheared_offset(local_r, local_c, patch.shape, direction)
            grid[r0 + rr, c0 + cc] = color


def _next_grey(patch: np.ndarray, direction: tuple[int, int]) -> tuple[int, int] | None:
    inner_h = patch.shape[0] - 2
    inner_w = patch.shape[1] - 2
    for rel_r, rel_c in _charge_order(inner_h, inner_w, direction):
        patch_r, patch_c = rel_r + 1, rel_c + 1
        if int(patch[patch_r, patch_c]) == GREY:
            return patch_r, patch_c
    return None


def _simulate_patch(
    patch: np.ndarray, direction: tuple[int, int], obstacles: set[tuple[int, int]]
) -> tuple[np.ndarray, int]:
    current = np.array(patch, dtype=int)
    dr, dc = direction
    successful_steps = 0

    while True:
        next_grey = _next_grey(current, direction)
        if next_grey is None:
            return current, successful_steps

        candidate = current.copy()
        candidate[next_grey] = ORANGE
        step = successful_steps + 1
        coords = _occupied_coords(candidate, (step * dr, step * dc), direction)
        if coords & obstacles:
            return current, successful_steps

        current = candidate
        successful_steps = step


def _first_collision_step(
    patch: np.ndarray, direction: tuple[int, int], obstacles: set[tuple[int, int]], limit: int
) -> int | None:
    dr, dc = direction
    for step in range(1, limit + 1):
        coords = _occupied_coords(patch, (step * dr, step * dc), direction)
        if coords & obstacles:
            return step
    return None


def _first_unsheared_collision_step(
    patch: np.ndarray, direction: tuple[int, int], obstacles: set[tuple[int, int]], limit: int
) -> int | None:
    dr, dc = direction
    for step in range(1, limit + 1):
        coords = _unsheared_coords(patch, (step * dr, step * dc))
        if coords & obstacles:
            return step
    return None


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    box = _find_container_bbox(g)
    direction, _marker, _target = _find_direction_and_target(g, box)
    obstacles = {tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)}

    r0, c0, r1, c1 = box
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    final_patch, steps = _simulate_patch(patch, direction, obstacles)

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    dr, dc = direction
    _stamp_sheared(out, final_patch, (r0 + steps * dr, c0 + steps * dc), direction)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a bordered container toward its target while charging interior cells and stopping at the first shear-induced collision.

    Input structure:
    - The background is magenta, with one black rectangular container whose interior is grey and orange.
    - A single maroon pixel sits flush against one outside face of the container and indicates the rear side; another maroon pixel lies farther ahead on the same axis as the target marker.
    - One or two red pixels act as fixed obstacles beside the travel corridor.

    Rule:
    - The container moves one cell per step away from the adjacent maroon marker and toward the distant maroon target.
    - At each attempted step, the container is viewed through a fixed shear: slices indexed from the front side are offset by their index, downward for horizontal travel and rightward for vertical travel.
    - In the same attempted step, the next grey interior cell on the rear side turns orange; if the sheared container would overlap any red obstacle, the entire step is canceled and the previous state remains, otherwise the move succeeds. If no grey cells remain after a successful step, the container stops there.

    Key insight:
    - The stopping step cannot be read from translation or energy alone: you must trace the front/back charging order and the sheared footprint that hits the side obstacle before energy runs out.
    """

    return _solve(grid)


def _marker_and_target_positions(
    patch_shape: tuple[int, int], direction: tuple[int, int], target_gap: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = patch_shape
    if direction == RIGHT:
        row = h // 2
        return (row, -1), (row, w - 1 + target_gap)
    if direction == LEFT:
        row = h // 2
        return (row, w), (row, -target_gap)
    if direction == DOWN:
        col = w // 2
        return (-1, col), (h - 1 + target_gap, col)
    col = w // 2
    return (h, col), (-target_gap, col)


def _build_patch(
    inner_h: int, inner_w: int, direction: tuple[int, int], grey_count: int
) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BORDER, dtype=int)
    patch[1:-1, 1:-1] = ORANGE
    order = _charge_order(inner_h, inner_w, direction)
    for rel_r, rel_c in order[:grey_count]:
        patch[rel_r + 1, rel_c + 1] = GREY
    return patch


def _collision_candidates(
    patch: np.ndarray, direction: tuple[int, int], collision_step: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dr, dc = direction
    previous: set[tuple[int, int]] = set()
    for step in range(collision_step):
        previous |= _occupied_coords(patch, (step * dr, step * dc), direction)

    collision_coords = _occupied_coords(patch, (collision_step * dr, collision_step * dc), direction)
    unsheared_collision = _unsheared_coords(patch, (collision_step * dr, collision_step * dc))
    exact = sorted(collision_coords - previous)
    shear_only = sorted((collision_coords - previous) - unsheared_collision)
    return exact, shear_only


def _place_case(
    patch: np.ndarray,
    direction: tuple[int, int],
    collision_step: int,
    target_gap: int,
    obstacles_rel: list[tuple[int, int]],
) -> tuple[np.ndarray, dict] | None:
    marker_rel, target_rel = _marker_and_target_positions(patch.shape, direction, target_gap)
    dr, dc = direction

    all_coords = set(obstacles_rel)
    all_coords.add(marker_rel)
    all_coords.add(target_rel)
    for step in range(collision_step + 1):
        all_coords |= _occupied_coords(patch, (step * dr, step * dc), direction)
    for local_r in range(patch.shape[0]):
        for local_c in range(patch.shape[1]):
            all_coords.add((local_r, local_c))

    rel = np.array(sorted(all_coords), dtype=int)
    min_r, min_c, max_r, max_c = _bbox(rel)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    min_h = max(16, span_h + 2)
    min_w = max(18, span_w + 2)
    max_h = min(26, span_h + 8)
    max_w = min(26, span_w + 8)
    if min_h > max_h or min_w > max_w:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    margin_top = random.randint(1, height - span_h - 1)
    margin_left = random.randint(1, width - span_w - 1)
    shift_r = margin_top - min_r
    shift_c = margin_left - min_c

    grid = np.full((height, width), BG, dtype=int)
    box_top = shift_r
    box_left = shift_c
    grid[box_top : box_top + patch.shape[0], box_left : box_left + patch.shape[1]] = patch

    marker = (marker_rel[0] + shift_r, marker_rel[1] + shift_c)
    target = (target_rel[0] + shift_r, target_rel[1] + shift_c)
    grid[marker] = MARKER
    grid[target] = MARKER

    obstacles = []
    for rr, cc in obstacles_rel:
        pos = (rr + shift_r, cc + shift_c)
        obstacles.append(pos)
        grid[pos] = OBSTACLE

    meta = {
        "direction": direction,
        "collision_step": collision_step,
        "target_gap": target_gap,
        "grey_count": int(np.sum(patch[1:-1, 1:-1] == GREY)),
        "obstacles": set(obstacles),
        "box": (box_top, box_left, box_top + patch.shape[0] - 1, box_left + patch.shape[1] - 1),
    }
    return grid, meta


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    try:
        box = _find_container_bbox(grid)
        direction, marker, target = _find_direction_and_target(grid, box)
        if box != meta["box"] or direction != meta["direction"]:
            return False

        r0, c0, r1, c1 = box
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        obstacles = {tuple(map(int, rc)) for rc in np.argwhere(grid == OBSTACLE)}

        if len(obstacles) not in (1, 2):
            return False
        if obstacles != meta["obstacles"]:
            return False
        if int(np.sum(patch[1:-1, 1:-1] == GREY)) != meta["grey_count"]:
            return False
        if not (meta["collision_step"] < meta["grey_count"]):
            return False

        if direction == RIGHT:
            target_gap = target[1] - c1
        elif direction == LEFT:
            target_gap = c0 - target[1]
        elif direction == DOWN:
            target_gap = target[0] - r1
        else:
            target_gap = r0 - target[0]
        if target_gap != meta["target_gap"] or not (meta["collision_step"] < target_gap):
            return False

        rel_obstacles = {(r - r0, c - c0) for r, c in obstacles}
        exact, shear_only = _collision_candidates(patch, direction, meta["collision_step"])
        if not rel_obstacles.issubset(set(shear_only)):
            return False

        first_collision = _first_collision_step(patch, direction, rel_obstacles, meta["grey_count"] + 3)
        if first_collision != meta["collision_step"]:
            return False
        unsheared_collision = _first_unsheared_collision_step(
            patch, direction, rel_obstacles, meta["grey_count"] + 3
        )
        if unsheared_collision is not None and unsheared_collision <= meta["collision_step"]:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if any(int(out[pos]) != MARKER for pos in (marker, target)):
            return False
        if any(int(out[pos]) != OBSTACLE for pos in obstacles):
            return False
        if np.array_equal(out, grid):
            return False
        return True
    except Exception:
        return False


def _attempt_case() -> np.ndarray | None:
    direction = random.choice(DIRECTIONS)
    inner_h = random.randint(3, 5)
    inner_w = random.randint(2, 3)
    inner_area = inner_h * inner_w

    collision_step = random.randint(2, min(4, inner_area - 1))
    grey_count = random.randint(collision_step + 2, inner_area)
    target_gap = random.randint(max(3, collision_step + 1), grey_count - 1)

    patch = _build_patch(inner_h, inner_w, direction, grey_count)
    _exact, shear_only = _collision_candidates(patch, direction, collision_step)
    if not shear_only:
        return None

    obstacle_count = 2 if len(shear_only) >= 2 and random.random() < 0.45 else 1
    obstacles_rel = random.sample(shear_only, obstacle_count)
    placed = _place_case(patch, direction, collision_step, target_gap, obstacles_rel)
    if placed is None:
        return None

    grid, meta = placed
    if not _validate_case(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-shear collision case")
