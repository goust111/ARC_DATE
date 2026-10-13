from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


PATH_COLOR = 2
MARKER_COLOR = 4

BASE_SIZE = 30
GEN_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (3, 5)
AREA_CHOICES = tuple(range(3, 9))
ROW_SPACING = 5
RUN_LEN_RANGE = (9, 16)
BG_CHOICES = (0, 1, 3, 5, 6, 7, 8, 9)
PADDING_MAX = 2


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _majority_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _object_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (bg, PATH_COLOR, MARKER_COLOR):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            objects.append(
                {
                    "color": color,
                    "area": int(coords.shape[0]),
                    "coords": coords,
                }
            )
    return objects


def _path_start_and_order(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    path_mask = grid == PATH_COLOR
    comps = _mask_components(path_mask)
    assert len(comps) == 1, "expected one red path"
    coords = comps[0]
    h, w = grid.shape

    endpoints: list[tuple[int, int]] = []
    for r, c in coords:
        degree = sum(bool(path_mask[nr, nc]) for nr, nc in _neighbors4(int(r), int(c), h, w))
        assert degree in (1, 2), "path must be a single-pixel non-branching chain"
        if degree == 1:
            endpoints.append((int(r), int(c)))
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == MARKER_COLOR)]
    assert len(markers) == 2, "expected exactly two yellow markers"
    adjacent_endpoints = [
        ep
        for ep in endpoints
        if any(abs(ep[0] - mr) + abs(ep[1] - mc) == 1 for mr, mc in markers)
    ]
    assert len(adjacent_endpoints) == 1, "exactly one endpoint must touch a yellow marker"
    start = adjacent_endpoints[0]

    order = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [
            (nr, nc)
            for nr, nc in _neighbors4(cur[0], cur[1], h, w)
            if path_mask[nr, nc] and (nr, nc) != prev
        ]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must stay unambiguous"
        prev, cur = cur, nxt[0]
        order.append(cur)
    assert len(order) == int(coords.shape[0]), "path traversal must visit every red pixel exactly once"
    return order, start


def _encounter_order(path: list[tuple[int, int]], objects: list[dict[str, object]]) -> list[int]:
    encountered: list[int] = []
    seen: set[int] = set()
    prev_obj: int | None = None
    for r, c in path:
        dists = []
        for obj in objects:
            coords = obj["coords"]
            dist = int(np.min(np.abs(coords[:, 0] - r) + np.abs(coords[:, 1] - c)))
            dists.append(dist)
        best = min(dists)
        nearest = [idx for idx, dist in enumerate(dists) if dist == best]
        if len(nearest) != 1:
            continue
        obj_idx = nearest[0]
        if obj_idx != prev_obj and obj_idx not in seen:
            encountered.append(obj_idx)
            seen.add(obj_idx)
        prev_obj = obj_idx
    assert len(encountered) == len(objects), "path must induce a complete object order"
    return encountered


def main(grid: np.ndarray) -> np.ndarray:
    """Read object colors in path order and rewrite them onto the objects sorted by area.

    Input structure:
    - The grid has a dominant background color plus a single red 4-connected path made of one-pixel-wide cells.
    - The red path has exactly two endpoints, and there are exactly two yellow marker pixels near the endpoints.
    - All remaining non-background pixels form separated monochrome 4-connected objects with pairwise different areas.

    Rule:
    - Choose the endpoint that is 4-neighbor adjacent to a yellow marker as the start of the path, then walk to the other endpoint.
    - While walking, read the colors of the closest objects in the order they are first encountered along the path.
    - Sort all objects by area from small to large, and repaint them with the read color sequence in that order.
    - Keep the path, the yellow markers, and every object's shape and position unchanged.

    Key insight:
    - The red path supplies the read order, while object area supplies an independent write order; solving the puzzle means decoupling and recombining those two permutations.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    objects = _object_components(g, bg)
    assert len(objects) >= 2, "expected multiple objects"

    areas = [int(obj["area"]) for obj in objects]
    assert len(set(areas)) == len(areas), "object areas must be unique"

    path, _start = _path_start_and_order(g)
    read_order = _encounter_order(path, objects)
    write_order = sorted(range(len(objects)), key=lambda idx: int(objects[idx]["area"]))

    out = g.copy()
    for dst_idx, src_idx in zip(write_order, read_order):
        coords = objects[dst_idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(objects[src_idx]["color"])
    return out


def _random_connected_mask(area: int) -> np.ndarray | None:
    dims = [
        (h, w)
        for h in range(1, 4)
        for w in range(2, 6)
        if h * w >= area
    ]
    random.shuffle(dims)
    for h, w in dims:
        for _ in range(120):
            cells = {(random.randrange(h), random.randrange(w))}
            while len(cells) < area:
                frontier: set[tuple[int, int]] = set()
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c, h, w):
                        if (nr, nc) not in cells:
                            frontier.add((nr, nc))
                if not frontier:
                    break
                cells.add(random.choice(tuple(frontier)))
            if len(cells) != area:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if random.random() < 0.5:
                mask = np.fliplr(mask)
            if random.random() < 0.5:
                mask = np.flipud(mask)
            return mask
    return None


def _draw_object(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask) + np.array([r0, c0], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = color
    return coords


def _endpoint_outward(path_mask: np.ndarray, endpoint: tuple[int, int]) -> tuple[int, int]:
    h, w = path_mask.shape
    nbrs = [(nr, nc) for nr, nc in _neighbors4(endpoint[0], endpoint[1], h, w) if path_mask[nr, nc]]
    assert len(nbrs) == 1
    nr, nc = nbrs[0]
    return endpoint[0] - nr, endpoint[1] - nc


def _used_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid != bg)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_with_padding(inp: np.ndarray, out: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _used_bbox(inp, bg)
    top = random.randint(0, min(PADDING_MAX, r0))
    left = random.randint(0, min(PADDING_MAX, c0))
    bottom = random.randint(0, min(PADDING_MAX, inp.shape[0] - 1 - r1))
    right = random.randint(0, min(PADDING_MAX, inp.shape[1] - 1 - c1))
    rs = slice(r0 - top, r1 + bottom + 1)
    cs = slice(c0 - left, c1 + right + 1)
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _build_expected(inp: np.ndarray, object_specs: list[dict[str, object]], path_order: list[int]) -> np.ndarray:
    out = inp.copy()
    area_order = sorted(range(len(object_specs)), key=lambda idx: int(object_specs[idx]["area"]))
    for dst_idx, src_idx in zip(area_order, path_order):
        coords = object_specs[dst_idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(object_specs[src_idx]["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    palette = [color for color in range(10) if color not in (bg, PATH_COLOR, MARKER_COLOR)]

    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(palette, n_objects)
    areas = random.sample(AREA_CHOICES, n_objects)

    masks: list[np.ndarray] = []
    for area in areas:
        mask = _random_connected_mask(area)
        if mask is None:
            return None
        masks.append(mask)

    max_w = max(int(mask.shape[1]) for mask in masks)
    run_len = random.randint(max(max_w + 4, RUN_LEN_RANGE[0]), RUN_LEN_RANGE[1])
    x_left = 4
    x_right = x_left + run_len - 1
    side = random.choice(("above", "below"))

    inp = np.full((BASE_SIZE, BASE_SIZE), bg, dtype=int)
    object_specs: list[dict[str, object]] = []
    run_rows = [4 + i * ROW_SPACING for i in range(n_objects)]
    assert run_rows[-1] < BASE_SIZE

    for i, row in enumerate(run_rows):
        inp[row, x_left : x_right + 1] = PATH_COLOR
        if i + 1 < n_objects:
            col = x_right if i % 2 == 0 else x_left
            inp[row : run_rows[i + 1] + 1, col] = PATH_COLOR

    for i, (mask, color, area, row) in enumerate(zip(masks, colors, areas, run_rows)):
        h, w = mask.shape
        c0_min = x_left + 2
        c0_max = x_right - w - 1
        if c0_min > c0_max:
            return None
        c0 = random.randint(c0_min, c0_max)
        r0 = row - h - 1 if side == "above" else row + 2
        if r0 < 0 or r0 + h > BASE_SIZE:
            return None
        coords = _draw_object(inp, mask, r0, c0, color)
        object_specs.append({"coords": coords, "color": color, "area": area})

    path_mask = inp == PATH_COLOR
    endpoints = []
    for r, c in np.argwhere(path_mask):
        degree = sum(bool(path_mask[nr, nc]) for nr, nc in _neighbors4(int(r), int(c), BASE_SIZE, BASE_SIZE))
        if degree == 1:
            endpoints.append((int(r), int(c)))
    if len(endpoints) != 2:
        return None

    endpoint_a, endpoint_b = sorted(endpoints)
    forward = random.random() < 0.5
    start_ep = endpoint_a if forward else endpoint_b
    end_ep = endpoint_b if forward else endpoint_a

    dr, dc = _endpoint_outward(path_mask, start_ep)
    sr, sc = start_ep[0] + dr, start_ep[1] + dc
    if not (0 <= sr < BASE_SIZE and 0 <= sc < BASE_SIZE and int(inp[sr, sc]) == bg):
        return None
    inp[sr, sc] = MARKER_COLOR

    dr, dc = _endpoint_outward(path_mask, end_ep)
    placed = False
    perps = [(-dc, dr), (dc, -dr)]
    random.shuffle(perps)
    for pr, pc in perps:
        er, ec = end_ep[0] + dr + pr, end_ep[1] + dc + pc
        if 0 <= er < BASE_SIZE and 0 <= ec < BASE_SIZE and int(inp[er, ec]) == bg:
            inp[er, ec] = MARKER_COLOR
            placed = True
            break
    if not placed:
        er, ec = end_ep[0] + 2 * dr, end_ep[1] + 2 * dc
        if not (0 <= er < BASE_SIZE and 0 <= ec < BASE_SIZE and int(inp[er, ec]) == bg):
            return None
        inp[er, ec] = MARKER_COLOR

    intended_order = list(range(n_objects)) if forward else list(reversed(range(n_objects)))
    expected = _build_expected(inp, object_specs, intended_order)
    if np.array_equal(inp, expected):
        return None

    cropped_inp, cropped_expected = _crop_with_padding(inp, expected, bg)
    k = random.randrange(4)
    cropped_inp = np.rot90(cropped_inp, k).copy()
    cropped_expected = np.rot90(cropped_expected, k).copy()

    try:
        actual_objects = _object_components(cropped_inp, _majority_color(cropped_inp))
        actual_path, _ = _path_start_and_order(cropped_inp)
        actual_order = _encounter_order(actual_path, actual_objects)
        areas_now = [int(obj["area"]) for obj in actual_objects]
        if len(actual_order) != n_objects or len(set(areas_now)) != len(areas_now):
            return None
        if not np.array_equal(main(cropped_inp), cropped_expected):
            return None
    except AssertionError:
        return None

    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
