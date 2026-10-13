from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _line_candidate(
    coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> tuple[str, int] | None:
    r0, c0, r1, c1 = bbox
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(cols) == 1 and len(coords) == r1 - r0 + 1 and rows == list(range(r0, r1 + 1)):
        return ("vertical", int(cols[0]))
    if len(rows) == 1 and len(coords) == c1 - c0 + 1 and cols == list(range(c0, c1 + 1)):
        return ("horizontal", int(rows[0]))
    return None


def _transform_object(
    grid: np.ndarray, coords: list[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    bbox = _bbox(coords)
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in coords:
        by_color.setdefault(int(grid[r, c]), []).append((r, c))

    candidates: list[tuple[int, str, int]] = []
    for color, pts in by_color.items():
        info = _line_candidate(pts, bbox)
        if info is not None:
            orient, idx = info
            candidates.append((len(pts), orient, idx))

    assert candidates, "expected a marker line inside each object"
    candidates.sort()
    _marker_len, orient, line_idx = candidates[0]

    body: list[tuple[int, int, int, int]] = []
    side_counts = Counter()
    for r, c in coords:
        color = int(grid[r, c])
        if orient == "vertical":
            if c == line_idx:
                continue
            side = -1 if c < line_idx else 1
        else:
            if r == line_idx:
                continue
            side = -1 if r < line_idx else 1
        body.append((r, c, color, side))
        side_counts[side] += 1

    assert side_counts[-1] > 0 and side_counts[1] > 0, "marker must split object into two non-empty halves"

    if side_counts[-1] == side_counts[1]:
        small_side = 0
        diff = 0
    elif side_counts[-1] < side_counts[1]:
        small_side = -1
        diff = side_counts[1] - side_counts[-1]
    else:
        small_side = 1
        diff = side_counts[-1] - side_counts[1]

    moved: list[tuple[int, int, int]] = []
    for r, c, color, side in body:
        rr, cc = r, c
        if diff and side == small_side:
            if orient == "vertical":
                cc += -diff if side < 0 else diff
            else:
                rr += -diff if side < 0 else diff
        moved.append((rr, cc, color))
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate each fractured object by moving only its smaller half.

    Input structure:
    - The background is the dominant color.
    - Each foreground object is 8-connected and uses one body color plus one
      marker color.
    - The marker color forms a full straight line that splits the object.

    Rule:
    - Erase the marker line.
    - Count body pixels on each side of the line.
    - Keep the larger side fixed and move the smaller side outward, along the
      line normal, by the area difference.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for comp in _find_components(g != bg):
        for r, c, color in _transform_object(g, comp):
            assert 0 <= r < h and 0 <= c < w, "translated half must remain in-bounds"
            out[r, c] = color
    return out


_MAX_GENERATE_ATTEMPTS = 400
_MAX_PLACEMENT_ATTEMPTS = 120
_MAX_OBJECT_ATTEMPTS = 80
_MAX_DIFF = 4


def _dilate(coords: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr, nc in _neighbors8(r, c, h, w):
            blocked.add((nr, nc))
        blocked.add((r, c))
    return blocked


def _grow_mask(cells: set[tuple[int, int]], target_area: int, h: int, w: int) -> set[tuple[int, int]] | None:
    while len(cells) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors8(r, c, h, w):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))
    return cells


def _vertical_half_mask(height: int, width: int, area: int, *, inner_on_right: bool) -> set[tuple[int, int]] | None:
    assert area >= width
    cols = list(range(width - 1, -1, -1)) if inner_on_right else list(range(width))
    row = random.randrange(height)
    cells = {(row, cols[0])}
    for col in cols[1:]:
        row = min(height - 1, max(0, row + random.choice((-1, 0, 1))))
        cells.add((row, col))
    return _grow_mask(cells, area, height, width)


def _horizontal_half_mask(height: int, width: int, area: int, *, inner_on_bottom: bool) -> set[tuple[int, int]] | None:
    assert area >= height
    rows = list(range(height - 1, -1, -1)) if inner_on_bottom else list(range(height))
    col = random.randrange(width)
    cells = {(rows[0], col)}
    for row in rows[1:]:
        col = min(width - 1, max(0, col + random.choice((-1, 0, 1))))
        cells.add((row, col))
    return _grow_mask(cells, area, height, width)


def _choose_areas(
    min_a: int, cap_a: int, min_b: int, cap_b: int, smaller_is_a: bool
) -> tuple[int, int, int] | None:
    by_diff: dict[int, list[tuple[int, int, int]]] = {}
    for diff in range(1, _MAX_DIFF + 1):
        pairs: list[tuple[int, int, int]] = []
        if smaller_is_a:
            for area_a in range(min_a, cap_a + 1):
                area_b = area_a + diff
                if min_b <= area_b <= cap_b:
                    pairs.append((area_a, area_b, diff))
        else:
            for area_b in range(min_b, cap_b + 1):
                area_a = area_b + diff
                if min_a <= area_a <= cap_a:
                    pairs.append((area_a, area_b, diff))
        if pairs:
            by_diff[diff] = pairs
    if not by_diff:
        return None
    diff = random.choice(sorted(by_diff))
    return random.choice(by_diff[diff])


def _build_vertical_object(body_color: int, marker_color: int) -> dict[str, object] | None:
    height = random.randint(3, 6)
    total_width = random.randint(height + 1, min(height + 6, 11))
    left_width = random.randint(1, total_width - 2)
    right_width = total_width - 1 - left_width
    smaller_side = random.choice(("left", "right"))

    area_info = _choose_areas(
        left_width,
        left_width * height,
        right_width,
        right_width * height,
        smaller_is_a=(smaller_side == "left"),
    )
    if area_info is None:
        return None
    left_area, right_area, diff = area_info

    left_mask = _vertical_half_mask(height, left_width, left_area, inner_on_right=True)
    right_mask = _vertical_half_mask(height, right_width, right_area, inner_on_right=False)
    if left_mask is None or right_mask is None:
        return None

    input_pixels: list[tuple[int, int, int]] = []
    output_pixels: list[tuple[int, int, int]] = []

    for r, c in left_mask:
        input_pixels.append((r, c, body_color))
        shifted_c = c - diff if smaller_side == "left" else c
        output_pixels.append((r, shifted_c, body_color))
    for r in range(height):
        input_pixels.append((r, left_width, marker_color))
    for r, c in right_mask:
        gc = left_width + 1 + c
        input_pixels.append((r, gc, body_color))
        shifted_c = gc + diff if smaller_side == "right" else gc
        output_pixels.append((r, shifted_c, body_color))

    return {"input": input_pixels, "output": output_pixels}


def _build_horizontal_object(body_color: int, marker_color: int) -> dict[str, object] | None:
    width = random.randint(3, 6)
    total_height = random.randint(width + 1, min(width + 6, 11))
    top_height = random.randint(1, total_height - 2)
    bottom_height = total_height - 1 - top_height
    smaller_side = random.choice(("top", "bottom"))

    area_info = _choose_areas(
        top_height,
        top_height * width,
        bottom_height,
        bottom_height * width,
        smaller_is_a=(smaller_side == "top"),
    )
    if area_info is None:
        return None
    top_area, bottom_area, diff = area_info

    top_mask = _horizontal_half_mask(top_height, width, top_area, inner_on_bottom=True)
    bottom_mask = _horizontal_half_mask(bottom_height, width, bottom_area, inner_on_bottom=False)
    if top_mask is None or bottom_mask is None:
        return None

    input_pixels: list[tuple[int, int, int]] = []
    output_pixels: list[tuple[int, int, int]] = []

    for r, c in top_mask:
        input_pixels.append((r, c, body_color))
        shifted_r = r - diff if smaller_side == "top" else r
        output_pixels.append((shifted_r, c, body_color))
    for c in range(width):
        input_pixels.append((top_height, c, marker_color))
    for r, c in bottom_mask:
        gr = top_height + 1 + r
        input_pixels.append((gr, c, body_color))
        shifted_r = gr + diff if smaller_side == "bottom" else gr
        output_pixels.append((shifted_r, c, body_color))

    return {"input": input_pixels, "output": output_pixels}


def _build_object(body_color: int, marker_color: int, orientation: str) -> dict[str, object] | None:
    for _ in range(_MAX_OBJECT_ATTEMPTS):
        if orientation == "vertical":
            obj = _build_vertical_object(body_color, marker_color)
        else:
            obj = _build_horizontal_object(body_color, marker_color)
        if obj is not None:
            return obj
    return None


def _translate_pixels(
    pixels: list[tuple[int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int]]:
    return [(r + dr, c + dc, color) for r, c, color in pixels]


def _place_object(
    obj: dict[str, object],
    shape: tuple[int, int],
    input_blocked: set[tuple[int, int]],
    output_blocked: set[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]] | None:
    h, w = shape
    all_pixels = list(obj["input"]) + list(obj["output"])
    rows = [r for r, _c, _color in all_pixels]
    cols = [c for _r, c, _color in all_pixels]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    r_lo = -min_r
    r_hi = h - 1 - max_r
    c_lo = -min_c
    c_hi = w - 1 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(_MAX_PLACEMENT_ATTEMPTS):
        dr = random.randint(r_lo, r_hi)
        dc = random.randint(c_lo, c_hi)
        input_pixels = _translate_pixels(obj["input"], dr, dc)
        output_pixels = _translate_pixels(obj["output"], dr, dc)

        input_coords = {(r, c) for r, c, _color in input_pixels}
        output_coords = {(r, c) for r, c, _color in output_pixels}
        if any(coord in input_blocked for coord in input_coords):
            continue
        if any(coord in output_blocked for coord in output_coords):
            continue
        return input_pixels, output_pixels
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(15, 24)
    width = random.randint(15, 24)
    object_count = random.randint(1, 3)

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    random.shuffle(palette)
    need_colors = 2 * object_count
    body_marker = palette[:need_colors]

    orientations = [random.choice(("vertical", "horizontal")) for _ in range(object_count)]
    objects: list[dict[str, object]] = []
    for i in range(object_count):
        body_color = body_marker[2 * i]
        marker_color = body_marker[2 * i + 1]
        obj = _build_object(body_color, marker_color, orientations[i])
        if obj is None:
            return None
        objects.append(obj)

    inp = np.full((height, width), bg, dtype=int)
    expected = np.full((height, width), bg, dtype=int)
    input_used: set[tuple[int, int]] = set()
    output_used: set[tuple[int, int]] = set()
    input_blocked: set[tuple[int, int]] = set()
    output_blocked: set[tuple[int, int]] = set()

    for obj in objects:
        placed = _place_object(obj, (height, width), input_blocked, output_blocked)
        if placed is None:
            return None
        input_pixels, output_pixels = placed

        for r, c, color in input_pixels:
            inp[r, c] = color
        for r, c, color in output_pixels:
            expected[r, c] = color

        input_coords = {(r, c) for r, c, _color in input_pixels}
        output_coords = {(r, c) for r, c, _color in output_pixels}
        input_used |= input_coords
        output_used |= output_coords
        input_blocked = _dilate(input_used, height, width)
        output_blocked = _dilate(output_used, height, width)

    if np.count_nonzero(inp == bg) <= np.count_nonzero(inp != bg):
        return None
    if len(_find_components(inp != bg)) != object_count:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after many attempts")
