from __future__ import annotations

from collections import deque
from itertools import product
import random

import numpy as np


BLACK = 0
RED = 2
PALETTE = (1, 3, 4, 7, 8)
GEN_RETRY_LIMIT = 400


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    return int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1


def _orientations(shape: list[tuple[int, int]]) -> list[np.ndarray]:
    base = np.array(shape, dtype=int)
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[np.ndarray] = []
    for rot in range(4):
        cur = base.copy()
        for _ in range(rot):
            cur = np.stack([cur[:, 1], -cur[:, 0]], axis=1)
        for mirror in (False, True):
            arr = cur.copy()
            if mirror:
                arr[:, 1] = -arr[:, 1]
            norm = _normalize(arr)
            key = tuple((int(r), int(c)) for r, c in norm.tolist())
            if key not in seen:
                seen.add(key)
                out.append(norm)
    return out


_BASE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2)],
]

_SHAPES_BY_AREA: dict[int, list[np.ndarray]] = {2: [], 3: [], 4: []}
for _shape in _BASE_SHAPES:
    for _ori in _orientations(_shape):
        _area = int(_ori.shape[0])
        key = tuple((int(r), int(c)) for r, c in _ori.tolist())
        if not any(tuple((int(r), int(c)) for r, c in s.tolist()) == key for s in _SHAPES_BY_AREA[_area]):
            _SHAPES_BY_AREA[_area].append(_ori)


_LANE_CONFIGS = [
    sizes
    for lane_count in range(3, 6)
    for sizes in product((1, 2, 3), repeat=lane_count)
    if 5 <= sum(sizes) <= 8 and any(size > 1 for size in sizes) and any(size == 1 for size in sizes) and sum(size > 1 for size in sizes) <= 2
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_moving_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    object_id = 0
    for color in PALETTE:
        for coords in _components_of_color(grid, color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "id": object_id,
                    "color": int(color),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
            object_id += 1
    return comps


def _wall_specs(grid: np.ndarray) -> list[tuple[str, tuple[int, int], np.ndarray]]:
    H, W = grid.shape
    specs = []
    if np.all(grid[0, :] == RED):
        mask = np.zeros((H, W), dtype=bool)
        mask[0, :] = True
        specs.append(("top", (-1, 0), mask))
    if np.all(grid[H - 1, :] == RED):
        mask = np.zeros((H, W), dtype=bool)
        mask[H - 1, :] = True
        specs.append(("bottom", (1, 0), mask))
    if np.all(grid[:, 0] == RED):
        mask = np.zeros((H, W), dtype=bool)
        mask[:, 0] = True
        specs.append(("left", (0, -1), mask))
    if np.all(grid[:, W - 1] == RED):
        mask = np.zeros((H, W), dtype=bool)
        mask[:, W - 1] = True
        specs.append(("right", (0, 1), mask))
    return specs


def _find_wall(grid: np.ndarray) -> tuple[str, int, int, np.ndarray]:
    specs = _wall_specs(grid)
    assert len(specs) == 1, "expected exactly one full red edge wall"
    side, (dr, dc), mask = specs[0]
    return side, int(dr), int(dc), mask


def _settle_key(comp: dict, dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    if dr == -1:
        return (r0, c0, c1, r1)
    if dr == 1:
        return (-r1, c0, c1, r0)
    if dc == -1:
        return (c0, r0, r1, c1)
    return (-c1, r0, r1, c0)


def _settle_components(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    side, dr, dc, wall_mask = _find_wall(grid)
    _ = side
    comps = _find_moving_components(grid)
    assert comps, "expected at least one moving object"

    H, W = grid.shape
    settled = np.full((H, W), BLACK, dtype=int)
    settled[wall_mask] = RED
    object_ids = np.full((H, W), -1, dtype=int)
    occupied = wall_mask.copy()
    step = np.array([dr, dc], dtype=int)

    for comp in sorted(comps, key=lambda item: _settle_key(item, dr, dc)):
        coords = comp["coords"].astype(int).copy()
        while True:
            nxt = coords + step
            in_bounds = (
                np.all(0 <= nxt[:, 0])
                and np.all(nxt[:, 0] < H)
                and np.all(0 <= nxt[:, 1])
                and np.all(nxt[:, 1] < W)
            )
            if not in_bounds or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
                break
            coords = nxt
        settled[coords[:, 0], coords[:, 1]] = int(comp["color"])
        object_ids[coords[:, 0], coords[:, 1]] = int(comp["id"])
        occupied[coords[:, 0], coords[:, 1]] = True
        comp["settled_coords"] = coords

    return settled, object_ids, comps, wall_mask


def _filter_merged_components(settled: np.ndarray, object_ids: np.ndarray, original_components: list[dict], wall_mask: np.ndarray) -> np.ndarray:
    area_by_id = {int(comp["id"]): int(comp["area"]) for comp in original_components}
    out = np.full_like(settled, BLACK)
    out[wall_mask] = RED
    for color in PALETTE:
        for coords in _components_of_color(settled, color):
            ids = {int(object_ids[r, c]) for r, c in coords.tolist()}
            ids.discard(-1)
            if not ids:
                continue
            max_original = max(area_by_id[obj_id] for obj_id in ids)
            if int(coords.shape[0]) > int(max_original):
                out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only same-color bodies that are created by gravity-driven merging against the red wall.

    Input structure:
    - The grid has black background and one full red edge wall on exactly one side.
    - Away from the wall are several small monochrome 4-connected objects in blue/green/yellow/orange/teal.
    - Objects are separated in the input and keep their shape while sliding.

    Rule:
    - Read the red wall side as the gravity direction and clear the non-wall objects from their starting cells.
    - Settle the objects toward that wall from nearest to farthest, stopping each rigid shape when the next step would hit the wall, the border, or an already settled object.
    - On the settled state, find same-color 4-connected components; keep a component only if its area is strictly larger than every original object that contributes pixels to it, otherwise erase it to black.
    - Preserve the red wall throughout.

    Key insight:
    - The decisive structure is not the original scatter but the post-stacking same-color connectivity: merged stacks survive, lone objects vanish.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0
    settled, object_ids, comps, wall_mask = _settle_components(g)
    return _filter_merged_components(settled, object_ids, comps, wall_mask)


def _rotate_from_top(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return grid.copy()
    if side == "bottom":
        return np.rot90(grid, 2).copy()
    if side == "left":
        return np.rot90(grid, 1).copy()
    if side == "right":
        return np.rot90(grid, -1).copy()
    raise ValueError(side)


def _choose_shape() -> np.ndarray:
    area = random.randint(2, 4)
    return random.choice(_SHAPES_BY_AREA[area]).copy()


def _make_lane(lane_size: int, color: int) -> dict | None:
    for _ in range(80):
        shapes = [_choose_shape() for _ in range(lane_size)]
        max_width = max(_shape_dims(shape)[1] for shape in shapes)
        width_choices = [max_width]
        if max_width < 4:
            width_choices.append(max_width + 1)
        random.shuffle(width_choices)
        for lane_width in width_choices:
            offsets: list[int] = []
            occupied_cols: set[int] = set()
            ok = True
            for shape in shapes:
                _h, w = _shape_dims(shape)
                candidates = []
                for offset in range(lane_width - w + 1):
                    cols = {int(offset + c) for c in shape[:, 1].tolist()}
                    if not offsets or cols & occupied_cols:
                        candidates.append((offset, cols))
                if not candidates:
                    ok = False
                    break
                offset, cols = random.choice(candidates)
                offsets.append(int(offset))
                occupied_cols |= cols
            if ok:
                return {
                    "size": int(lane_size),
                    "color": int(color),
                    "width": int(lane_width),
                    "shapes": shapes,
                    "offsets": offsets,
                }
    return None


def _settle_top(coords: np.ndarray, occupied: np.ndarray, H: int) -> np.ndarray | None:
    cur = coords.copy()
    if np.any(occupied[cur[:, 0], cur[:, 1]]):
        return None
    while True:
        nxt = cur + np.array([-1, 0], dtype=int)
        if np.any(nxt[:, 0] < 0) or np.any(occupied[nxt[:, 0], nxt[:, 1]]):
            return cur
        cur = nxt


def _assign_lane_shifts(objects: list[dict], H: int) -> list[int] | None:
    row_bounds = [(int(obj["coords"][:, 0].min()), int(obj["coords"][:, 0].max())) for obj in objects]

    def feasible_from(index: int, prev_max_row: int) -> bool:
        probe_prev = prev_max_row
        for j in range(index, len(objects)):
            min_row, max_row = row_bounds[j]
            shift = max(1, probe_prev + 2 - min_row)
            if max_row + shift >= H:
                return False
            probe_prev = max_row + shift
        return True

    def rec(index: int, prev_max_row: int) -> list[int] | None:
        if index == len(objects):
            return []
        min_row, max_row = row_bounds[index]
        min_shift = max(1, prev_max_row + 2 - min_row)
        max_shift = H - 1 - max_row
        if min_shift > max_shift:
            return None
        candidates = list(range(min_shift, max_shift + 1))
        random.shuffle(candidates)
        candidates.sort(reverse=True)
        for shift in candidates:
            cur_max = max_row + shift
            if not feasible_from(index + 1, cur_max):
                continue
            rest = rec(index + 1, cur_max)
            if rest is not None:
                return [int(shift)] + rest
        return None

    return rec(0, -1)


def _render_expected(H: int, W: int, lane_objects: list[list[dict]], wall_mask: np.ndarray) -> np.ndarray:
    out = np.full((H, W), BLACK, dtype=int)
    out[wall_mask] = RED
    for objs in lane_objects:
        if len(objs) <= 1:
            continue
        color = int(objs[0]["color"])
        for obj in objs:
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = color
    return out


def _build_top_instance() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(14, 18)
    lane_sizes = list(random.choice(_LANE_CONFIGS))
    random.shuffle(lane_sizes)
    colors = random.sample(PALETTE, len(lane_sizes))

    lanes: list[dict] = []
    for lane_size, color in zip(lane_sizes, colors):
        lane = _make_lane(lane_size, color)
        if lane is None:
            return None
        lanes.append(lane)

    W = sum(lane["width"] for lane in lanes) + (len(lanes) - 1)
    if not (12 <= W <= 18):
        return None

    wall_mask = np.zeros((H, W), dtype=bool)
    wall_mask[0, :] = True
    occupied = wall_mask.copy()
    input_grid = np.full((H, W), BLACK, dtype=int)
    input_grid[wall_mask] = RED

    lane_objects: list[list[dict]] = []
    left = 0
    for lane in lanes:
        objs: list[dict] = []
        for shape, offset in zip(lane["shapes"], lane["offsets"]):
            shape_h, _shape_w = _shape_dims(shape)
            start = shape + np.array([H - shape_h, left + int(offset)], dtype=int)
            settled = _settle_top(start, occupied, H)
            if settled is None:
                return None
            occupied[settled[:, 0], settled[:, 1]] = True
            objs.append({"coords": settled, "color": int(lane["color"]), "area": int(shape.shape[0])})
        lane_objects.append(objs)
        left += int(lane["width"]) + 1

    expected = _render_expected(H, W, lane_objects, wall_mask)
    total_objects = sum(len(objs) for objs in lane_objects)
    if not (5 <= total_objects <= 8):
        return None
    if not any(len(objs) > 1 for objs in lane_objects):
        return None
    if not any(len(objs) == 1 for objs in lane_objects):
        return None

    for objs in lane_objects:
        shifts = _assign_lane_shifts(objs, H)
        if shifts is None:
            return None
        for obj, shift in zip(objs, shifts):
            coords = obj["coords"] + np.array([shift, 0], dtype=int)
            input_grid[coords[:, 0], coords[:, 1]] = int(obj["color"])

    if len(_find_moving_components(input_grid)) != total_objects:
        return None

    settled, object_ids, comps, settled_wall = _settle_components(input_grid)
    merged = _filter_merged_components(settled, object_ids, comps, settled_wall)
    if not np.array_equal(merged, expected):
        return None

    return input_grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _build_top_instance()
        if built is None:
            continue
        top_input, top_expected = built
        side = random.choice(("top", "bottom", "left", "right"))
        inp = _rotate_from_top(top_input, side)
        expected = _rotate_from_top(top_expected, side)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 10")
