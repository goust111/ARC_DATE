from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste
import random


class _Object:
    def __init__(self, bbox: tuple[int, int, int, int], patch: np.ndarray, closed: bool):
        self.bbox = bbox
        self.patch = patch
        self.height = int(patch.shape[0])
        self.closed = bool(closed)


def _object_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c]:
                exterior[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any((~mask) & (~exterior)))


def _objects(g: np.ndarray, bg: int) -> list[_Object]:
    objects: list[_Object] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        bbox = tuple(map(int, comp["bbox"]))
        coords = comp["coords"]
        mask = _object_mask(coords, bbox)
        patch = np.full(mask.shape, bg, dtype=int)
        patch[mask] = color
        objects.append(_Object(bbox, patch, _has_enclosed_hole(mask)))

    assert objects, "expected at least one foreground object"
    return objects


def _can_place(out: np.ndarray, obj: _Object, row: int, bg: int) -> bool:
    _r0, c0, _r1, _c1 = obj.bbox
    h, w = obj.patch.shape
    mask = obj.patch != bg
    region = out[row : row + h, c0 : c0 + w]
    return bool(np.all(region[mask] == bg))


def _place(out: np.ndarray, obj: _Object, row: int, bg: int) -> None:
    _r0, c0, _r1, _c1 = obj.bbox
    paste(out, obj.patch, row, c0, transparent=bg)


def _slide_into_place(out: np.ndarray, obj: _Object, bg: int, upward: bool) -> None:
    H = out.shape[0]
    if upward:
        rows = range(0, H - obj.height + 1)
    else:
        rows = range(H - obj.height, -1, -1)

    for row in rows:
        if _can_place(out, obj, row, bg):
            _place(out, obj, row, bg)
            return

    raise AssertionError("expected a valid slide position")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide closed foreground objects upward and open foreground objects downward, keeping their columns and stacking on collision.

    Input structure
    - The grid has a single background color and several monochromatic 4-connected foreground objects.
    - Some objects are closed frames with at least one enclosed background hole inside their bounding box.
    - The remaining objects are open line-like sprites with no enclosed hole.

    Rule
    - Redraw every object on an empty background canvas without changing its color, shape, or horizontal position.
    - Closed frame objects slide vertically upward to the first non-colliding position.
    - Open objects slide vertically downward to the first non-colliding position.
    - Within each direction, preserve the original vertical order so overlapping columns stack instead of crossing.

    Key insight
    - The only role distinction is whether an object encloses a hole: hole objects pack against the top edge, while non-hole objects pack against the bottom edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full_like(g, bg)
    objects = _objects(g, bg)

    closed = [obj for obj in objects if obj.closed]
    open_objects = [obj for obj in objects if not obj.closed]
    assert closed, "expected at least one closed object"
    assert open_objects, "expected at least one open object"

    for obj in sorted(closed, key=lambda o: (o.bbox[0], o.bbox[1], o.bbox[2], o.bbox[3])):
        _slide_into_place(out, obj, bg, upward=True)

    for obj in sorted(open_objects, key=lambda o: (-o.bbox[2], o.bbox[1], -o.bbox[0], o.bbox[3])):
        _slide_into_place(out, obj, bg, upward=False)

    return out

# --------------- generator (o5x) ---------------

# Canvas and object size bounds.
GRID_SIZE_MIN = 13
GRID_SIZE_MAX = 17
SPRITE_SIZE_MIN = 3
SPRITE_SIZE_MAX = 5

# Object count ranges.
CLOSED_COUNT_MIN = 2
CLOSED_COUNT_MAX = 3
OPEN_COUNT_MIN = 2
OPEN_COUNT_MAX = 4

# Palette constraints.
COLOR_MIN = 1
COLOR_MAX = 9
DISTINCT_COLOR_MIN = 3
DISTINCT_COLOR_MAX = 4
MAX_OBJECTS_PER_COLOR = 2

# Layout constraints.
MIN_ZONE_GAP = 3
LANE_GAP = 1
SCATTER_EDGE_MARGIN = 1
SCATTER_MIDDLE_FIRST_PROB = 0.8

# Retry limit.
GENERATION_ATTEMPTS = 300


def _gen_frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _gen_cross_mask() -> np.ndarray:
    h = random.randint(SPRITE_SIZE_MIN, SPRITE_SIZE_MAX)
    w = random.randint(SPRITE_SIZE_MIN, SPRITE_SIZE_MAX)
    r = random.randint(1, h - 2)
    c = random.randint(1, w - 2)
    mask = np.zeros((h, w), dtype=bool)
    mask[r, :] = True
    mask[:, c] = True
    return mask


def _gen_color_labels(n: int) -> list[int]:
    min_k = max(DISTINCT_COLOR_MIN, (n + 1) // MAX_OBJECTS_PER_COLOR)
    k = random.randint(min_k, min(DISTINCT_COLOR_MAX, n))
    colors = random.sample(range(COLOR_MIN, COLOR_MAX + 1), k)
    counts = [1] * k
    for _ in range(n - k):
        choices = [i for i, count in enumerate(counts) if count < MAX_OBJECTS_PER_COLOR]
        counts[random.choice(choices)] += 1

    labels: list[int] = []
    for color, count in zip(colors, counts):
        labels.extend([color] * count)
    random.shuffle(labels)
    return labels


def _gen_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _gen_partition_lanes(heights: list[int], capacity: int, n_lanes: int) -> list[list[int]] | None:
    order = list(range(len(heights)))
    random.shuffle(order)
    order.sort(key=lambda i: heights[i], reverse=True)
    lanes: list[list[int]] = [[] for _ in range(n_lanes)]
    sums = [0] * n_lanes

    def rec(pos: int) -> bool:
        if pos == len(order):
            return all(lane for lane in lanes)

        idx = order[pos]
        lane_order = list(range(n_lanes))
        random.shuffle(lane_order)
        for lane_i in lane_order:
            if sums[lane_i] + heights[idx] > capacity:
                continue
            if not lanes[lane_i]:
                empty_before = sum(1 for lane in lanes if not lane)
                items_left = len(order) - pos
                if empty_before > items_left:
                    continue
            lanes[lane_i].append(idx)
            sums[lane_i] += heights[idx]
            if rec(pos + 1):
                return True
            sums[lane_i] -= heights[idx]
            lanes[lane_i].pop()
        return False

    return lanes if rec(0) else None


def _gen_random_gaps(total: int, parts: int) -> list[int]:
    gaps = [0] * parts
    for _ in range(total):
        gaps[random.randrange(parts)] += 1
    return gaps


def _gen_assign_columns(objs: list[dict], width: int, capacity: int) -> bool:
    heights = [int(obj["patch"].shape[0]) for obj in objs]
    min_lanes = max(1, (sum(heights) + capacity - 1) // capacity)
    lane_counts = list(range(min_lanes, len(objs) + 1))
    random.shuffle(lane_counts)

    for n_lanes in lane_counts:
        lanes = _gen_partition_lanes(heights, capacity, n_lanes)
        if lanes is None:
            continue
        random.shuffle(lanes)
        lane_widths = [max(int(objs[i]["patch"].shape[1]) for i in lane) for lane in lanes]
        required = sum(lane_widths) + LANE_GAP * (n_lanes - 1)
        if required > width:
            continue

        gaps = _gen_random_gaps(width - required, n_lanes + 1)
        col = gaps[0]
        for lane_i, lane in enumerate(lanes):
            lane_w = lane_widths[lane_i]
            for obj_i in lane:
                obj_w = int(objs[obj_i]["patch"].shape[1])
                objs[obj_i]["c"] = col + random.randint(0, lane_w - obj_w)
            col += lane_w + LANE_GAP + gaps[lane_i + 1]
        return True
    return False


def _gen_can_place_pixels(canvas: np.ndarray, obj: dict, row: int) -> bool:
    patch = obj["patch"]
    c = int(obj["c"])
    h, w = patch.shape
    region = canvas[row : row + h, c : c + w]
    return bool(np.all(region[patch != 0] == 0))


def _gen_pack_group(canvas: np.ndarray, objs: list[dict], upward: bool) -> bool:
    height = int(canvas.shape[0])
    for obj in objs:
        h = int(obj["patch"].shape[0])
        if upward:
            rows = range(0, height - h + 1)
        else:
            rows = range(height - h, -1, -1)
        for row in rows:
            if _gen_can_place_pixels(canvas, obj, row):
                paste(canvas, obj["patch"], row, int(obj["c"]), transparent=0)
                obj["out_r"] = row
                break
        else:
            return False
    return True


def _gen_zone_gap(closed: list[dict], open_objs: list[dict]) -> int:
    closed_bottom = max(int(obj["out_r"]) + int(obj["patch"].shape[0]) - 1 for obj in closed)
    open_top = min(int(obj["out_r"]) for obj in open_objs)
    return open_top - closed_bottom - 1


def _gen_same_color_touch(objs: list[dict]) -> bool:
    placed = []
    for obj in objs:
        coords = np.argwhere(obj["patch"] != 0)
        coords = coords + np.array([int(obj["out_r"]), int(obj["c"])])
        placed.append((int(obj["color"]), coords))

    for i in range(len(placed)):
        color_i, coords_i = placed[i]
        for j in range(i + 1, len(placed)):
            color_j, coords_j = placed[j]
            if color_i != color_j:
                continue
            distances = np.abs(coords_i[:, None, :] - coords_j[None, :, :]).sum(axis=2)
            if np.any(distances <= 1):
                return True
    return False


def _gen_can_scatter(canvas: np.ndarray, obj: dict, row: int) -> bool:
    patch = obj["patch"]
    c = int(obj["c"])
    h, w = patch.shape
    if row < 0 or row + h > canvas.shape[0] or c < 0 or c + w > canvas.shape[1]:
        return False
    mask = patch != 0
    if np.any(canvas[row : row + h, c : c + w][mask] != 0):
        return False

    coords = np.argwhere(mask)
    for rr, cc in coords:
        ar = row + int(rr)
        ac = c + int(cc)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = ar + dr
            nc = ac + dc
            if 0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and canvas[nr, nc] != 0:
                return False
    return True


def _gen_row_candidates(height: int, obj_h: int) -> list[int]:
    rows = list(range(0, height - obj_h + 1))
    middle = [r for r in rows if SCATTER_EDGE_MARGIN <= r <= height - obj_h - SCATTER_EDGE_MARGIN]
    edge = [r for r in rows if r not in middle]
    random.shuffle(middle)
    random.shuffle(edge)
    if random.random() < SCATTER_MIDDLE_FIRST_PROB:
        return middle + edge
    return edge + middle


def _gen_scatter_input(height: int, width: int, closed: list[dict], open_objs: list[dict]) -> np.ndarray | None:
    """Place input objects in solver order while preserving same-group vertical order."""
    canvas = np.zeros((height, width), dtype=int)

    def rec(ci: int, oi: int, prev_closed_top: int, prev_open_bottom: int) -> bool:
        if ci == len(closed) and oi == len(open_objs):
            return True

        choices: list[str] = []
        if ci < len(closed):
            choices.append("closed")
        if oi < len(open_objs):
            choices.append("open")
        random.shuffle(choices)

        for choice in choices:
            if choice == "closed":
                obj = closed[ci]
                h = int(obj["patch"].shape[0])
                for row in _gen_row_candidates(height, h):
                    if row <= prev_closed_top or not _gen_can_scatter(canvas, obj, row):
                        continue
                    paste(canvas, obj["patch"], row, int(obj["c"]), transparent=0)
                    if rec(ci + 1, oi, row, prev_open_bottom):
                        return True
                    canvas[row : row + h, int(obj["c"]) : int(obj["c"]) + obj["patch"].shape[1]][obj["patch"] != 0] = 0
            else:
                obj = open_objs[oi]
                h = int(obj["patch"].shape[0])
                for row in _gen_row_candidates(height, h):
                    bottom = row + h - 1
                    if bottom >= prev_open_bottom or not _gen_can_scatter(canvas, obj, row):
                        continue
                    paste(canvas, obj["patch"], row, int(obj["c"]), transparent=0)
                    if rec(ci, oi + 1, prev_closed_top, bottom):
                        return True
                    canvas[row : row + h, int(obj["c"]) : int(obj["c"]) + obj["patch"].shape[1]][obj["patch"] != 0] = 0
        return False

    return canvas if rec(0, 0, -1, height) else None


def _gen_canvas_shape() -> tuple[int, int]:
    return random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX), random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)


def _gen_objects() -> tuple[list[dict], list[dict], list[dict]]:
    """Build colored closed and open sprites, then randomize each solver group order."""
    n_closed = random.randint(CLOSED_COUNT_MIN, CLOSED_COUNT_MAX)
    n_open = random.randint(OPEN_COUNT_MIN, OPEN_COUNT_MAX)
    closed: list[dict] = []
    for _ in range(n_closed):
        h = random.randint(SPRITE_SIZE_MIN, SPRITE_SIZE_MAX)
        w = random.randint(SPRITE_SIZE_MIN, SPRITE_SIZE_MAX)
        closed.append({"kind": "closed", "mask": _gen_frame_mask(h, w)})

    open_objs: list[dict] = []
    for _ in range(n_open):
        open_objs.append({"kind": "open", "mask": _gen_cross_mask()})

    all_objs = closed + open_objs
    for obj, color in zip(all_objs, _gen_color_labels(len(all_objs))):
        obj["color"] = color
        obj["patch"] = _gen_patch(obj["mask"], color)

    random.shuffle(closed)
    random.shuffle(open_objs)

    return closed, open_objs, all_objs


def _gen_capacity_split(height: int, closed: list[dict], open_objs: list[dict]) -> tuple[int, int] | None:
    max_closed_h = max(int(obj["patch"].shape[0]) for obj in closed)
    max_open_h = max(int(obj["patch"].shape[0]) for obj in open_objs)
    available = height - MIN_ZONE_GAP
    if max_closed_h + max_open_h > available:
        return None
    closed_capacity = random.randint(max_closed_h, available - max_open_h)
    open_capacity = available - closed_capacity
    return closed_capacity, open_capacity


def _gen_assign_layout(height: int, width: int, closed: list[dict], open_objs: list[dict]) -> bool:
    capacities = _gen_capacity_split(height, closed, open_objs)
    if capacities is None:
        return False
    closed_capacity, open_capacity = capacities

    if not _gen_assign_columns(closed, width, closed_capacity):
        return False
    if not _gen_assign_columns(open_objs, width, open_capacity):
        return False
    return True


def _gen_pack_output(height: int, width: int, closed: list[dict], open_objs: list[dict]) -> np.ndarray | None:
    """Render the packed target grid and reject invalid zone separation."""
    output = np.zeros((height, width), dtype=int)
    if not _gen_pack_group(output, closed, upward=True):
        return None
    if not _gen_pack_group(output, open_objs, upward=False):
        return None
    if _gen_zone_gap(closed, open_objs) < MIN_ZONE_GAP:
        return None
    return output


def _gen_checked_input(height: int, width: int, closed: list[dict], open_objs: list[dict], output: np.ndarray) -> np.ndarray | None:
    """Scatter objects into an input grid and keep only examples accepted by the solver oracle."""
    inp = _gen_scatter_input(height, width, closed, open_objs)
    if inp is None or np.array_equal(inp, output):
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def _gen_attempt() -> np.ndarray | None:
    height, width = _gen_canvas_shape()
    closed, open_objs, all_objs = _gen_objects()

    if not _gen_assign_layout(height, width, closed, open_objs):
        return None

    output = _gen_pack_output(height, width, closed, open_objs)
    if output is None or _gen_same_color_touch(all_objs):
        return None

    return _gen_checked_input(height, width, closed, open_objs, output)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ac2e8ecf input")
