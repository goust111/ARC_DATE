from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
TEAL = 8

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_SPRITE_MASKS: list[np.ndarray] | None = None


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        start = (int(r), int(c))
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and bool(mask[nr, nc])
                    and (nr, nc) not in seen
                ):
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        out.append(comp)
    return out


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    mask = grid == BLUE
    assert int(mask.sum()) >= 3, "expected one blue sprite"
    comps = _components(mask)
    assert len(comps) == 1, "expected exactly one blue component"
    b = _bbox(mask)
    assert b is not None
    sprite = np.where(_crop_bbox(mask.astype(int), b) != 0, BLUE, BLACK).astype(int)
    return sprite


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _follow_path(grid: np.ndarray) -> list[str]:
    start_cells = np.argwhere(grid == GREEN)
    end_cells = np.argwhere(grid == RED)
    assert len(start_cells) == 1 and len(end_cells) == 1, "expected singleton start/end"
    start = tuple(map(int, start_cells[0]))

    initial_dirs = []
    for dr, dc in DIRS4:
        nr, nc = start[0] + dr, start[1] + dc
        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BLACK:
            initial_dirs.append((dr, dc))
    assert len(initial_dirs) == 1, "start must expose exactly one black exit"

    position = start
    direction = initial_dirs[0]
    seen_states = {(position, direction)}
    turn_types: list[str] = []

    for _ in range(grid.size * 4):
        nr, nc = position[0] + direction[0], position[1] + direction[1]
        assert _in_bounds(grid.shape, nr, nc), "path ran out of bounds"
        cell = int(grid[nr, nc])
        if cell == RED:
            return turn_types
        if cell == BLACK:
            position = (nr, nc)
        elif cell == TEAL:
            options = []
            for nd in _perpendiculars(direction):
                rr, cc = position[0] + nd[0], position[1] + nd[1]
                if _in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) in (BLACK, RED):
                    options.append(nd)
            assert len(options) == 1, "blocked turn must have exactly one 90-degree continuation"
            new_direction = options[0]
            if direction[0] == 0:
                turn_types.append("lr")
            else:
                turn_types.append("ud")
            direction = new_direction
        else:
            raise AssertionError("path hit an unexpected color")

        state = (position, direction)
        assert state not in seen_states, "path looped instead of reaching the goal"
        seen_states.add(state)

    raise AssertionError("path did not terminate")


def _apply_turns(sprite: np.ndarray, turn_types: list[str]) -> np.ndarray:
    out = sprite.copy()
    for turn_type in turn_types:
        if turn_type == "lr":
            out = np.fliplr(out)
        else:
            out = np.flipud(out)
    mask = out == BLUE
    b = _bbox(mask)
    assert b is not None
    return _crop_bbox(out, b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the obstacle-forced polyline from the green start to the red end and mirror the blue sprite at each turn.

    Input structure:
    - A black grid contains one small blue connected sprite, one green start pixel, one red end pixel, and teal obstacle pixels.
    - The start has exactly one adjacent black exit cell that fixes the initial travel direction.
    - Obstacles are arranged so the path is a unique axis-aligned polyline with one or two forced 90-degree turns.

    Rule:
    - Starting from the green pixel, move straight through black cells.
    - If the next cell is teal, do not enter it; instead turn 90 degrees, using the unique perpendicular black/red continuation.
    - Each horizontal-to-vertical turn applies a left-right mirror to the blue sprite, and each vertical-to-horizontal turn applies an up-down mirror.
    - Apply the mirrors in path order and return the cropped final blue sprite.

    Key insight:
    - The teal pixels do not mark the path itself; they only force turns, and the axis change at each turn determines which mirror axis to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sprite = _extract_sprite(g)
    turn_types = _follow_path(g)
    return _apply_turns(sprite, turn_types)


def _mask_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _sprite_masks() -> list[np.ndarray]:
    global _SPRITE_MASKS
    if _SPRITE_MASKS is not None:
        return _SPRITE_MASKS

    masks: list[np.ndarray] = []
    for h, w in ((2, 2), (2, 3), (3, 2), (3, 3)):
        area = h * w
        for n in range(3, min(5, area) + 1):
            for combo in itertools.combinations(range(area), n):
                mask = np.zeros((h, w), dtype=bool)
                mask.ravel()[list(combo)] = True
                if not _mask_connected(mask):
                    continue
                if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                if np.array_equal(mask, np.fliplr(mask)):
                    continue
                if np.array_equal(mask, np.flipud(mask)):
                    continue
                if np.array_equal(mask, np.rot90(mask, 2)):
                    continue
                masks.append(mask.astype(bool))

    assert masks, "expected non-symmetric sprite masks"
    _SPRITE_MASKS = masks
    return masks


def _random_sprite() -> np.ndarray:
    mask = random.choice(_sprite_masks())
    return np.where(mask, BLUE, BLACK).astype(int)


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _normalize_points(points: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    r_shift = -min(rs)
    c_shift = -min(cs)
    shifted = {(r + r_shift, c + c_shift) for r, c in points}
    rs2 = [r for r, _ in shifted]
    cs2 = [c for _, c in shifted]
    return shifted, (min(rs2), min(cs2), max(rs2), max(cs2))


def _build_path_design() -> dict | None:
    turn_count = random.choices([1, 2], weights=[3, 2])[0]
    initial_axis = random.choice(["h", "v"])
    if initial_axis == "h":
        direction = random.choice([(0, -1), (0, 1)])
    else:
        direction = random.choice([(-1, 0), (1, 0)])

    directions = [direction]
    for _ in range(turn_count):
        directions.append(random.choice(_perpendiculars(directions[-1])))

    lengths = [random.randint(2, 5) for _ in directions]
    lengths[-1] = random.randint(1, 5)

    start = (0, 0)
    pos = start
    path_cells = [start]
    turn_points: list[tuple[int, int]] = []
    for i, step_dir in enumerate(directions):
        for _ in range(lengths[i]):
            pos = _add(pos, step_dir)
            path_cells.append(pos)
        if i < turn_count:
            turn_points.append(pos)
    pre_end = pos
    end = _add(pre_end, directions[-1])

    path_set = set(path_cells)
    if len(path_set) != len(path_cells) or end in path_set:
        return None

    teal_cells: set[tuple[int, int]] = set()
    for step_dir in DIRS4:
        neighbor = _add(start, step_dir)
        if step_dir != directions[0]:
            if neighbor in path_set or neighbor == end:
                return None
            teal_cells.add(neighbor)

    for i, turn_point in enumerate(turn_points):
        before = directions[i]
        after = directions[i + 1]
        forward_block = _add(turn_point, before)
        if forward_block in path_set or forward_block == end:
            return None
        teal_cells.add(forward_block)

        other_options = [nd for nd in _perpendiculars(before) if nd != after]
        assert len(other_options) == 1
        side_block = _add(turn_point, other_options[0])
        if side_block in path_set or side_block == end:
            return None
        teal_cells.add(side_block)

    occupied = path_set | teal_cells | {end}
    r_shift = -min(r for r, _ in occupied)
    c_shift = -min(c for _, c in occupied)

    def shift_point(point: tuple[int, int]) -> tuple[int, int]:
        return point[0] + r_shift, point[1] + c_shift

    path_set = {shift_point(p) for p in path_set}
    teal_cells = {shift_point(p) for p in teal_cells}
    start = shift_point(start)
    end = shift_point(end)
    turn_points = [shift_point(tp) for tp in turn_points]

    if len(path_set | teal_cells | {end}) != len(path_set) + len(teal_cells) + 1:
        return None

    return {
        "start": start,
        "end": end,
        "path": path_set,
        "teal": teal_cells,
        "turn_points": turn_points,
        "directions": directions,
        "turn_count": turn_count,
    }


def _expected_from_directions(sprite: np.ndarray, directions: list[tuple[int, int]]) -> np.ndarray:
    turn_types = []
    for i in range(len(directions) - 1):
        turn_types.append("lr" if directions[i][0] == 0 else "ud")
    return _apply_turns(sprite, turn_types)


def _place_noise(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    sprite_box: tuple[int, int, int, int],
) -> None:
    h, w = grid.shape
    candidates = []
    sr0, sc0, sr1, sc1 = sprite_box
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            if sr0 - 1 <= r <= sr1 + 1 and sc0 - 1 <= c <= sc1 + 1:
                continue
            ok = True
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in forbidden:
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    random.shuffle(candidates)
    for r, c in candidates[: random.randint(1, min(6, len(candidates)))]:
        grid[r, c] = TEAL


def _validate_generated(grid: np.ndarray, turn_count: int, expected: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False

    sprite = _extract_sprite(grid)
    if np.array_equal(out, sprite):
        return False
    if turn_count == 1:
        if np.array_equal(out, np.fliplr(sprite)) == np.array_equal(out, np.flipud(sprite)):
            return False
    if turn_count == 2:
        if not np.array_equal(out, np.rot90(sprite, 2)):
            return False
        if np.array_equal(out, np.fliplr(sprite)) or np.array_equal(out, np.flipud(sprite)):
            return False

    turns = _follow_path(grid)
    if len(turns) != turn_count:
        return False
    if turn_count == 2 and set(turns) != {"lr", "ud"}:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    grid_h = random.randint(15, 18)
    grid_w = random.randint(15, 18)
    sprite = _random_sprite()
    design = _build_path_design()
    if design is None:
        return None

    path_rs = [r for r, _ in design["path"] | design["teal"] | {design["end"]}]
    path_cs = [c for _, c in design["path"] | design["teal"] | {design["end"]}]
    path_h = max(path_rs) + 1
    path_w = max(path_cs) + 1

    spr_h, spr_w = sprite.shape
    sprite_r = random.randint(1, 2)
    sprite_c = random.randint(1, 2)
    sprite_box = (sprite_r, sprite_c, sprite_r + spr_h - 1, sprite_c + spr_w - 1)

    path_r_min = sprite_box[2] + 3
    path_c_min = sprite_box[3] + 3
    path_r_max = grid_h - path_h - 1
    path_c_max = grid_w - path_w - 1
    if path_r_min > path_r_max or path_c_min > path_c_max:
        return None

    path_r0 = random.randint(path_r_min, path_r_max)
    path_c0 = random.randint(path_c_min, path_c_max)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[sprite_r : sprite_r + spr_h, sprite_c : sprite_c + spr_w][sprite == BLUE] = BLUE

    path_cells_global = {(path_r0 + r, path_c0 + c) for r, c in design["path"]}
    teal_cells_global = {(path_r0 + r, path_c0 + c) for r, c in design["teal"]}
    end_r, end_c = path_r0 + design["end"][0], path_c0 + design["end"][1]
    start_r, start_c = path_r0 + design["start"][0], path_c0 + design["start"][1]

    forbidden = path_cells_global | teal_cells_global | {(end_r, end_c)}
    if any(not _in_bounds(grid.shape, r, c) for r, c in forbidden | {(start_r, start_c)}):
        return None
    if any(grid[r, c] != BLACK for r, c in forbidden | {(start_r, start_c)}):
        return None

    for r, c in teal_cells_global:
        grid[r, c] = TEAL
    grid[start_r, start_c] = GREEN
    grid[end_r, end_c] = RED

    forbidden |= {(start_r, start_c)}
    _place_noise(grid, forbidden, sprite_box)

    expected = _expected_from_directions(sprite, design["directions"])
    if not _validate_generated(grid, design["turn_count"], expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-mirror ARC sample")
