from __future__ import annotations

import random
from collections import deque

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue
            color = int(g[r0, c0])
            queue = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                coords.append((r, c))
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc] or int(g[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    queue.append((rr, cc))

            pts = np.array(coords, dtype=int)
            rows = pts[:, 0]
            cols = pts[:, 1]
            bbox = (
                int(rows.min()),
                int(cols.min()),
                int(rows.max()),
                int(cols.max()),
            )

            touches_top = bool(np.any(rows == 0))
            touches_bottom = bool(np.any(rows == h - 1))
            touches_left = bool(np.any(cols == 0))
            touches_right = bool(np.any(cols == w - 1))
            touches_horizontal = touches_top or touches_bottom
            touches_vertical = touches_left or touches_right
            assert not (
                touches_horizontal and touches_vertical
            ), "objects may not touch both horizontal and vertical edges"

            if touches_horizontal:
                group = "horizontal"
                side = "top" if touches_top else "bottom"
            elif touches_vertical:
                group = "vertical"
                side = "left" if touches_left else "right"
            else:
                group = "interior"
                side = None

            objects.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": bbox,
                    "group": group,
                    "side": side,
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group monochrome foreground objects by edge-contact type, then rotate colors within each edge group.

    Horizontal edge group:
    - objects touching top or bottom, sorted left to right
    - colors cycle right by one position

    Vertical edge group:
    - objects touching left or right, sorted top to bottom
    - colors cycle left by one position

    Interior objects keep their colors.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects = _extract_objects(g, bg)

    horizontal = sorted(objects, key=lambda obj: (obj["group"] != "horizontal", obj["bbox"][1], obj["bbox"][0]))
    vertical = sorted(objects, key=lambda obj: (obj["group"] != "vertical", obj["bbox"][0], obj["bbox"][1]))
    horizontal = [obj for obj in horizontal if obj["group"] == "horizontal"]
    vertical = [obj for obj in vertical if obj["group"] == "vertical"]

    out = g.copy()

    if horizontal:
        donor_colors = [horizontal[(idx - 1) % len(horizontal)]["color"] for idx in range(len(horizontal))]
        for obj, new_color in zip(horizontal, donor_colors):
            pts = obj["coords"]
            out[pts[:, 0], pts[:, 1]] = int(new_color)

    if vertical:
        donor_colors = [vertical[(idx + 1) % len(vertical)]["color"] for idx in range(len(vertical))]
        for obj, new_color in zip(vertical, donor_colors):
            pts = obj["coords"]
            out[pts[:, 0], pts[:, 1]] = int(new_color)

    return out


# --------------- generator ---------------

EDGE_DEPTH = 4
INNER_GAP = 1
BAND_GAP = 1
MIN_BAND = 2
MAX_BAND = 5
MIN_EDGE_COUNT = 2
MAX_EDGE_COUNT = 3
MIN_INTERIOR_COUNT = 1
MAX_INTERIOR_COUNT = 2
MAX_GEN_ATTEMPTS = 400
OBJECT_PLACE_ATTEMPTS = 80


def _balanced_sides(count: int, first: str, second: str) -> list[str]:
    sides = [first, second] * (count // 2)
    if count % 2:
        sides.append(random.choice([first, second]))
    random.shuffle(sides)
    return sides


def _random_bands(
    start: int, end: int, count: int, min_size: int, max_size: int, gap: int
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    length = end - start + 1
    if length < count * min_size + (count - 1) * gap:
        return None

    for _ in range(200):
        widths = [random.randint(min_size, max_size) for _ in range(count)]
        total = sum(widths) + (count - 1) * gap
        if total > length:
            continue

        slack = length - total
        spaces = [0] * (count + 1)
        for _ in range(slack):
            spaces[random.randrange(count + 1)] += 1

        pos = start + spaces[0]
        bands: list[tuple[int, int]] = []
        for idx, width in enumerate(widths):
            bands.append((pos, pos + width - 1))
            pos += width + gap + spaces[idx + 1]
        return bands
    return None


def _random_connected_mask(height: int, width: int, required_side: str | None) -> np.ndarray:
    if required_side == "top":
        start = (0, random.randrange(width))
    elif required_side == "bottom":
        start = (height - 1, random.randrange(width))
    elif required_side == "left":
        start = (random.randrange(height), 0)
    elif required_side == "right":
        start = (random.randrange(height), width - 1)
    else:
        start = (random.randrange(height), random.randrange(width))

    min_cells = min(height * width, max(2, (height * width + 1) // 2))
    target = random.randint(min_cells, height * width)
    cells = {start}

    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    pts = np.array(sorted(cells), dtype=int)
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
    return mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                out[rr, cc] = True
    return out


def _can_place(blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r0 + h > blocked.shape[0] or c0 + w > blocked.shape[1]:
        return False
    return not bool(np.any(mask & blocked[r0 : r0 + h, c0 : c0 + w]))


def _stamp(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    placed = np.zeros_like(blocked)
    h, w = mask.shape
    placed[r0 : r0 + h, c0 : c0 + w] = mask
    grid[placed] = int(color)
    blocked |= _dilate8(placed)


def _validate_generated_grid(
    grid: np.ndarray, bg: int, expected_h: int, expected_v: int, expected_i: int
) -> bool:
    try:
        if _background_color(grid) != bg:
            return False
        objects = _extract_objects(grid, bg)
        if len(objects) != expected_h + expected_v + expected_i:
            return False
        colors = [obj["color"] for obj in objects]
        if len(set(colors)) != len(colors):
            return False

        horizontal = [obj for obj in objects if obj["group"] == "horizontal"]
        vertical = [obj for obj in objects if obj["group"] == "vertical"]
        interior = [obj for obj in objects if obj["group"] == "interior"]
        if len(horizontal) != expected_h or len(vertical) != expected_v or len(interior) != expected_i:
            return False

        if {obj["side"] for obj in horizontal} != {"top", "bottom"}:
            return False
        if {obj["side"] for obj in vertical} != {"left", "right"}:
            return False

        h_keys = [obj["bbox"][1] for obj in horizontal]
        v_keys = [obj["bbox"][0] for obj in vertical]
        if len(set(h_keys)) != len(h_keys) or len(set(v_keys)) != len(v_keys):
            return False

        if np.array_equal(main(grid), grid):
            return False
    except AssertionError:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    horizontal_count = random.randint(MIN_EDGE_COUNT, MAX_EDGE_COUNT)
    vertical_count = random.randint(MIN_EDGE_COUNT, MAX_EDGE_COUNT)
    interior_count = random.randint(MIN_INTERIOR_COUNT, MAX_INTERIOR_COUNT)
    total_objects = horizontal_count + vertical_count + interior_count

    min_h = max(18, 2 * (EDGE_DEPTH + INNER_GAP) + vertical_count * MIN_BAND + (vertical_count - 1) * BAND_GAP)
    min_w = max(18, 2 * (EDGE_DEPTH + INNER_GAP) + horizontal_count * MIN_BAND + (horizontal_count - 1) * BAND_GAP)
    grid_h = random.randint(min_h, min_h + 6)
    grid_w = random.randint(min_w, min_w + 6)
    bg = random.randrange(10)

    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    colors = palette[:total_objects]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    core_r0 = EDGE_DEPTH + INNER_GAP
    core_r1 = grid_h - EDGE_DEPTH - INNER_GAP - 1
    core_c0 = EDGE_DEPTH + INNER_GAP
    core_c1 = grid_w - EDGE_DEPTH - INNER_GAP - 1

    horizontal_bands = _random_bands(core_c0, core_c1, horizontal_count, MIN_BAND, MAX_BAND, BAND_GAP)
    vertical_bands = _random_bands(core_r0, core_r1, vertical_count, MIN_BAND, MAX_BAND, BAND_GAP)
    if horizontal_bands is None or vertical_bands is None:
        return None

    horizontal_sides = _balanced_sides(horizontal_count, "top", "bottom")
    vertical_sides = _balanced_sides(vertical_count, "left", "right")

    color_idx = 0

    for band, side in zip(horizontal_bands, horizontal_sides):
        span = band[1] - band[0] + 1
        placed = False
        for _ in range(OBJECT_PLACE_ATTEMPTS):
            raw_height = random.randint(2, EDGE_DEPTH)
            mask = _random_connected_mask(raw_height, span, side)
            mh, mw = mask.shape
            r0 = 0 if side == "top" else grid_h - mh
            c0 = random.randint(band[0], band[1] - mw + 1)
            if not _can_place(blocked, mask, r0, c0):
                continue
            _stamp(grid, blocked, mask, r0, c0, colors[color_idx])
            placed = True
            break
        if not placed:
            return None
        color_idx += 1

    for band, side in zip(vertical_bands, vertical_sides):
        span = band[1] - band[0] + 1
        placed = False
        for _ in range(OBJECT_PLACE_ATTEMPTS):
            raw_width = random.randint(2, EDGE_DEPTH)
            mask = _random_connected_mask(span, raw_width, side)
            mh, mw = mask.shape
            r0 = random.randint(band[0], band[1] - mh + 1)
            c0 = 0 if side == "left" else grid_w - mw
            if not _can_place(blocked, mask, r0, c0):
                continue
            _stamp(grid, blocked, mask, r0, c0, colors[color_idx])
            placed = True
            break
        if not placed:
            return None
        color_idx += 1

    for _ in range(interior_count):
        placed = False
        for _ in range(OBJECT_PLACE_ATTEMPTS):
            raw_height = random.randint(2, 4)
            raw_width = random.randint(2, 4)
            mask = _random_connected_mask(raw_height, raw_width, None)
            mh, mw = mask.shape
            if core_r1 - mh + 1 < core_r0 or core_c1 - mw + 1 < core_c0:
                return None
            r0 = random.randint(core_r0, core_r1 - mh + 1)
            c0 = random.randint(core_c0, core_c1 - mw + 1)
            if not _can_place(blocked, mask, r0, c0):
                continue
            _stamp(grid, blocked, mask, r0, c0, colors[color_idx])
            placed = True
            break
        if not placed:
            return None
        color_idx += 1

    if not _validate_generated_grid(grid, bg, horizontal_count, vertical_count, interior_count):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
