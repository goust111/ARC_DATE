from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TOP_BAR = 2
BOTTOM_BAR = 3
FRACTURE = 4
OBSTACLE = 5

DIR_UP = -1
DIR_DOWN = 1

GEN_RETRIES = 400


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_array(arr: np.ndarray, fill: int = BG) -> tuple[np.ndarray, tuple[int, int]]:
    mask = arr != fill
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty patch")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def _mask_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_objects(grid: np.ndarray) -> list[dict]:
    interior = np.zeros_like(grid, dtype=bool)
    interior[1:-1, :] = np.isin(grid[1:-1, :], [TOP_BAR, BOTTOM_BAR, FRACTURE])
    comps = _mask_components(interior)
    objects: list[dict] = []
    for idx, coords in enumerate(comps):
        colors = {int(grid[r, c]) for r, c in coords}
        assert FRACTURE in colors, "each moving object must contain one fracture line"
        body_colors = colors - {FRACTURE}
        assert len(body_colors) == 1, "each moving object must have one body color"
        body_color = body_colors.pop()
        assert body_color in {TOP_BAR, BOTTOM_BAR}
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        objects.append(
            {
                "id": idx,
                "color": int(body_color),
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
            }
        )
    return objects


def _validate_bars(grid: np.ndarray) -> None:
    h, w = grid.shape
    assert h >= 5 and w >= 5
    assert np.all(grid[0, :] == TOP_BAR)
    assert np.all(grid[h - 1, :] == BOTTOM_BAR)


def _front_distance(obj: dict, h: int) -> int:
    r0, _, r1, _ = obj["bbox"]
    if obj["color"] == TOP_BAR:
        return int(r0 - 1)
    return int((h - 2) - r1)


def _paint_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= dst.shape[0] and left + w <= dst.shape[1]
    mask = patch != BG
    view = dst[top : top + h, left : left + w]
    assert np.all(view[mask] == BG), "patch placement must not overlap existing foreground"
    view[mask] = patch[mask]


def _erase_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    dst[top : top + h, left : left + w][mask] = BG


def _target_reached(top: int, height: int, direction: int, grid_h: int) -> bool:
    if direction == DIR_UP:
        return top == 1
    return top + height - 1 == grid_h - 2


def _would_collide(state: np.ndarray, top: int, left: int, patch: np.ndarray, direction: int) -> bool:
    h, w = patch.shape
    if direction == DIR_UP:
        assert top > 1
        return bool(np.any(state[top - 1, left : left + w] != BG))
    assert top + h - 1 < state.shape[0] - 2
    return bool(np.any(state[top + h, left : left + w] != BG))


def _slide_until_stop(
    state: np.ndarray, top: int, left: int, patch: np.ndarray, direction: int
) -> tuple[int, int, bool, int]:
    steps = 0
    cur_top = int(top)
    while not _target_reached(cur_top, patch.shape[0], direction, state.shape[0]):
        if _would_collide(state, cur_top, left, patch, direction):
            return cur_top, left, True, steps
        cur_top += direction
        steps += 1
    return cur_top, left, False, steps


def _split_patch(
    patch: np.ndarray, body_color: int, direction: int
) -> tuple[np.ndarray, tuple[int, int], np.ndarray, tuple[int, int], str]:
    h, w = patch.shape
    assert h != w, "object bbox must be non-square"
    marker_coords = np.argwhere(patch == FRACTURE)
    assert marker_coords.size > 0

    half_a = np.zeros_like(patch)
    half_b = np.zeros_like(patch)
    orientation: str

    if w > h:
        orientation = "wide"
        by_row: dict[int, int] = {}
        for r, c in marker_coords:
            rr = int(r)
            cc = int(c)
            assert rr not in by_row, "wide fracture must contribute one marker per row"
            by_row[rr] = cc
        assert set(by_row) == set(range(h)), "wide fracture must cross every row"
        prev = None
        for rr in range(h):
            if prev is not None:
                assert abs(by_row[rr] - prev) <= 1, "fracture line must stay connected"
            prev = by_row[rr]
            for cc in range(w):
                if patch[rr, cc] != body_color:
                    continue
                if cc < by_row[rr]:
                    half_a[rr, cc] = body_color
                elif cc > by_row[rr]:
                    half_b[rr, cc] = body_color
                else:
                    raise AssertionError("body cell cannot share the fracture column")
    else:
        orientation = "tall"
        by_col: dict[int, int] = {}
        for r, c in marker_coords:
            rr = int(r)
            cc = int(c)
            assert cc not in by_col, "tall fracture must contribute one marker per column"
            by_col[cc] = rr
        assert set(by_col) == set(range(w)), "tall fracture must cross every column"
        prev = None
        for cc in range(w):
            if prev is not None:
                assert abs(by_col[cc] - prev) <= 1, "fracture line must stay connected"
            prev = by_col[cc]
            for rr in range(h):
                if patch[rr, cc] != body_color:
                    continue
                if rr < by_col[cc]:
                    half_a[rr, cc] = body_color
                elif rr > by_col[cc]:
                    half_b[rr, cc] = body_color
                else:
                    raise AssertionError("body cell cannot share the fracture row")

    assert np.count_nonzero(half_a) > 0 and np.count_nonzero(half_b) > 0
    assert len(_mask_components(half_a != BG)) == 1
    assert len(_mask_components(half_b != BG)) == 1

    a_bbox = _bbox_from_coords([tuple(map(int, rc)) for rc in np.argwhere(half_a != BG)])
    b_bbox = _bbox_from_coords([tuple(map(int, rc)) for rc in np.argwhere(half_b != BG)])

    if direction == DIR_UP:
        if a_bbox[0] < b_bbox[0]:
            front, back = half_a, half_b
        elif b_bbox[0] < a_bbox[0]:
            front, back = half_b, half_a
        else:
            raise AssertionError("split halves must have a unique front half for upward motion")
    else:
        if a_bbox[2] > b_bbox[2]:
            front, back = half_a, half_b
        elif b_bbox[2] > a_bbox[2]:
            front, back = half_b, half_a
        else:
            raise AssertionError("split halves must have a unique front half for downward motion")

    front_patch, front_off = _crop_array(front)
    back_patch, back_off = _crop_array(back)
    return front_patch, front_off, back_patch, back_off, orientation


def _simulate(grid: np.ndarray, trace: bool = False) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _validate_bars(g)

    h, _ = g.shape
    objects = _find_objects(g)
    assert 3 <= len(objects) <= 5, "expected 3 to 5 moving objects"

    state = g.copy()
    events: list[dict] = []
    ordered = sorted(objects, key=lambda obj: (_front_distance(obj, h), obj["bbox"][1]))

    for obj in ordered:
        color = int(obj["color"])
        direction = DIR_UP if color == TOP_BAR else DIR_DOWN
        r0, c0, r1, c1 = obj["bbox"]
        patch = obj["patch"]
        _erase_patch(state, patch, r0, c0)

        stop_top, stop_left, collided, pre_steps = _slide_until_stop(state, r0, c0, patch, direction)
        event = {
            "id": obj["id"],
            "color": color,
            "initial_bbox": obj["bbox"],
            "collided": bool(collided),
            "pre_steps": int(pre_steps),
            "post_steps": 0,
            "split_orientation": None,
        }

        if not collided:
            healed = np.where(patch == FRACTURE, color, patch).astype(int)
            _paint_patch(state, healed, stop_top, stop_left)
            event["final_bbox"] = (
                stop_top,
                stop_left,
                stop_top + healed.shape[0] - 1,
                stop_left + healed.shape[1] - 1,
            )
            events.append(event)
            continue

        front_patch, front_off, back_patch, back_off, orientation = _split_patch(patch, color, direction)
        event["split_orientation"] = orientation
        back_top = stop_top + back_off[0]
        back_left = stop_left + back_off[1]
        _paint_patch(state, back_patch, back_top, back_left)

        front_top = stop_top + front_off[0]
        front_left = stop_left + front_off[1]
        front_stop_top, front_stop_left, _collided_again, post_steps = _slide_until_stop(
            state, front_top, front_left, front_patch, direction
        )
        _paint_patch(state, front_patch, front_stop_top, front_stop_left)

        event["post_steps"] = int(post_steps)
        event["back_bbox"] = (
            back_top,
            back_left,
            back_top + back_patch.shape[0] - 1,
            back_left + back_patch.shape[1] - 1,
        )
        event["front_bbox"] = (
            front_stop_top,
            front_stop_left,
            front_stop_top + front_patch.shape[0] - 1,
            front_stop_left + front_patch.shape[1] - 1,
        )
        event["final_bbox"] = event["front_bbox"]
        events.append(event)

    return state, events if trace else []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each cracked object toward its same-color border bar, splitting it only when a collision activates the fracture.

    Input structure:
    - A black background grid whose top row is a full-width red bar and whose bottom row is a full-width green bar.
    - The interior contains 3 to 5 four-connected red or green objects, each with exactly one yellow fracture line crossing the object's shorter bbox axis.
    - The interior also contains 1 to 2 gray square obstacles that do not move.

    Rule:
    - Red objects slide upward toward the top bar; green objects slide downward toward the bottom bar.
    - Process objects in order of increasing initial distance to their target bar, breaking ties by smaller column.
    - Collision testing uses the whole bbox front row of the moving shape. If an object reaches its bar without collision, it stops there and its yellow fracture pixels heal into the body color.
    - If a collision happens first, the object stops before the blocking row, the yellow fracture pixels are erased, the object splits into two halves, the half closer to the target bar keeps sliding, and the other half stays behind.

    Key insight:
    - The yellow line is a dormant crack: it means nothing for a clean slide, but a collision activates it and turns one moving object into a stationary rear half plus a smaller front half that can continue on its own narrower bbox.
    """
    out, _ = _simulate(grid, trace=False)
    return out


def _colorize(template: np.ndarray, body_color: int) -> np.ndarray:
    patch = template.copy()
    patch[patch == 1] = body_color
    return patch.astype(int)


COLLISION_BASES = [
    np.array(
        [
            [1, 4, 0, 0],
            [1, 4, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 4, 0, 0, 0],
            [1, 4, 1, 1, 0],
            [1, 4, 1, 1, 1],
        ],
        dtype=int,
    ),
]

CLEAR_WIDE_BASES = [
    np.array([[1, 4, 1, 1], [1, 4, 1, 0]], dtype=int),
    np.array([[1, 1, 4, 1], [0, 1, 4, 1]], dtype=int),
    np.array([[1, 4, 1, 1, 0], [1, 4, 1, 0, 0], [1, 4, 1, 1, 1]], dtype=int),
]


def _random_transform(
    patch: np.ndarray, *, allow_transpose: bool, vertical_flip: bool | None = None
) -> np.ndarray:
    out = patch.copy()
    if allow_transpose and random.random() < 0.5:
        out = out.T
    if random.random() < 0.5:
        out = np.fliplr(out)
    if vertical_flip is None:
        if random.random() < 0.5:
            out = np.flipud(out)
    elif vertical_flip:
        out = np.flipud(out)
    return out.copy()


def _make_clear_object(body_color: int) -> np.ndarray:
    base = random.choice(CLEAR_WIDE_BASES)
    patch = _random_transform(base, allow_transpose=True, vertical_flip=None)
    return _colorize(patch, body_color)


def _make_collision_object(body_color: int) -> np.ndarray:
    base = random.choice(COLLISION_BASES)
    direction = DIR_UP if body_color == TOP_BAR else DIR_DOWN
    patch = _random_transform(base, allow_transpose=False, vertical_flip=(direction == DIR_DOWN))
    return _colorize(patch, body_color)


def _lane_columns(widths: list[int]) -> tuple[int, list[int]] | None:
    gaps = [random.randint(1, 2) for _ in range(len(widths) + 1)]
    total = sum(widths) + sum(gaps)
    grid_w = max(16, total)
    if grid_w > 30:
        return None
    starts: list[int] = []
    cur = gaps[0]
    for i, width in enumerate(widths):
        starts.append(cur)
        cur += width + gaps[i + 1]
    return grid_w, starts


def _free_path_columns(used: list[tuple[int, int]], start: int, width: int) -> bool:
    end = start + width - 1
    for a, b in used:
        if not (end < a or b < start):
            return False
    return True


def _place_clear_object(grid: np.ndarray, patch: np.ndarray, left: int, used_paths: list[tuple[int, int]]) -> None:
    h = patch.shape[0]
    body_color = int(patch[patch != BG][0])
    width = patch.shape[1]
    assert _free_path_columns(used_paths, left, width)
    if body_color == TOP_BAR:
        top = random.randint(2, grid.shape[0] - h - 3)
    else:
        top = random.randint(2, grid.shape[0] - h - 3)
    _paint_patch(grid, patch, top, left)
    used_paths.append((left, left + width - 1))


def _collision_obstacle_columns(
    patch: np.ndarray, body_color: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    direction = DIR_UP if body_color == TOP_BAR else DIR_DOWN
    front_patch, front_off, back_patch, back_off, _ = _split_patch(patch, body_color, direction)
    front_col_span = (front_off[1], front_off[1] + front_patch.shape[1] - 1)
    back_col_span = (back_off[1], back_off[1] + back_patch.shape[1] - 1)
    if front_col_span[0] <= back_col_span[0]:
        blocked_span = back_col_span
    else:
        blocked_span = back_col_span
    return front_col_span, back_col_span, front_off, back_off


def _place_collision_scenario(
    grid: np.ndarray, patch: np.ndarray, left: int, used_paths: list[tuple[int, int]]
) -> bool:
    h, w = patch.shape
    body_color = int(next(v for v in patch.flat if v in {TOP_BAR, BOTTOM_BAR}))
    direction = DIR_UP if body_color == TOP_BAR else DIR_DOWN
    front_patch, front_off, back_patch, back_off, _ = _split_patch(patch, body_color, direction)
    front_span = (front_off[1], front_off[1] + front_patch.shape[1] - 1)
    back_span = (back_off[1], back_off[1] + back_patch.shape[1] - 1)
    blocked_span = back_span
    blocked_abs_left = left + blocked_span[0]
    blocked_abs_right = left + blocked_span[1]
    obstacle_size = min(random.choice([2, 3]), blocked_abs_right - blocked_abs_left + 1)
    obstacle_left_min = blocked_abs_left
    obstacle_left_max = blocked_abs_right - obstacle_size + 1
    obstacle_left = random.randint(obstacle_left_min, obstacle_left_max)

    if direction == DIR_UP:
        obstacle_top_min = 3
        obstacle_top_max = grid.shape[0] - h - obstacle_size - 5
        if obstacle_top_min > obstacle_top_max:
            return False
        obstacle_top = random.randint(obstacle_top_min, obstacle_top_max)
        obstacle_bottom = obstacle_top + obstacle_size - 1
        stop_top = obstacle_bottom + 1
        min_start = stop_top + 2
        max_start = grid.shape[0] - h - 3
        if min_start > max_start:
            return False
        start_top = random.randint(min_start, max_start)
    else:
        obstacle_top_min = h + 4
        obstacle_top_max = grid.shape[0] - h - 6
        if obstacle_top_min > obstacle_top_max:
            return False
        obstacle_top = random.randint(obstacle_top_min, obstacle_top_max)
        stop_top = obstacle_top - h
        min_start = 2
        max_start = stop_top - 2
        if min_start > max_start:
            return False
        start_top = random.randint(min_start, max_start)

    grid[obstacle_top : obstacle_top + obstacle_size, obstacle_left : obstacle_left + obstacle_size] = OBSTACLE
    _paint_patch(grid, patch, start_top, left)
    used_paths.append((left, left + w - 1))
    return True


def _place_idle_obstacle(grid: np.ndarray, forbidden: list[tuple[int, int]]) -> bool:
    size = random.choice([2, 3])
    cols = list(range(0, grid.shape[1] - size + 1))
    random.shuffle(cols)
    rows = list(range(2, grid.shape[0] - size - 2))
    random.shuffle(rows)
    for left in cols:
        if any(not (left + size - 1 < a or b < left) for a, b in forbidden):
            continue
        for top in rows:
            region = grid[top : top + size, left : left + size]
            if np.all(region == BG):
                grid[top : top + size, left : left + size] = OBSTACLE
                return True
    return False


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        out, events = _simulate(inp, trace=True)
    except AssertionError:
        return False

    if out.shape != inp.shape:
        return False
    if not np.all(out[0, :] == TOP_BAR) or not np.all(out[-1, :] == BOTTOM_BAR):
        return False

    collided = [e for e in events if e["collided"]]
    clear = [e for e in events if not e["collided"]]
    if not collided or not clear:
        return False
    if max(e["post_steps"] for e in collided) <= 0:
        return False

    colors = {e["color"] for e in events}
    if colors != {TOP_BAR, BOTTOM_BAR}:
        return False

    if np.any(out == FRACTURE):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(3, 5)
    extra_count = object_count - 2

    collision_color = random.choice([TOP_BAR, BOTTOM_BAR])
    clear_opposite = BOTTOM_BAR if collision_color == TOP_BAR else TOP_BAR

    patches = [
        ("collision", _make_collision_object(collision_color)),
        ("clear", _make_clear_object(clear_opposite)),
    ]
    extra_colors = [TOP_BAR, BOTTOM_BAR]
    for _ in range(extra_count):
        patches.append(("clear", _make_clear_object(random.choice(extra_colors))))
    random.shuffle(patches)

    widths = [patch.shape[1] for _, patch in patches]
    lane_layout = _lane_columns(widths)
    if lane_layout is None:
        return None
    grid_w, starts = lane_layout
    grid_h = random.randint(14, 20)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[0, :] = TOP_BAR
    grid[-1, :] = BOTTOM_BAR

    used_paths: list[tuple[int, int]] = []
    collision_lanes: list[tuple[int, int]] = []
    for (kind, patch), left in zip(patches, starts):
        if kind == "collision":
            if not _place_collision_scenario(grid, patch, left, used_paths):
                return None
            collision_lanes.append((left, left + patch.shape[1] - 1))
        else:
            _place_clear_object(grid, patch, left, used_paths)

    obstacle_count = int(np.count_nonzero(grid == OBSTACLE) > 0)
    if obstacle_count < 1:
        return None
    if obstacle_count == 1 and random.random() < 0.5:
        _place_idle_obstacle(grid, forbidden=used_paths)

    object_mask = np.isin(grid[1:-1, :], [TOP_BAR, BOTTOM_BAR, FRACTURE])
    if len(_mask_components(object_mask)) != object_count:
        return None
    obstacle_mask = grid[1:-1, :] == OBSTACLE
    obstacle_components = _mask_components(obstacle_mask)
    if not (1 <= len(obstacle_components) <= 2):
        return None
    for comp in obstacle_components:
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h not in {2, 3} or len(comp) != side_h * side_w:
            return None

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid collision-fracture ARC input")
