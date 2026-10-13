from __future__ import annotations

import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BACKGROUND_COLOR = 8
GRID_SIZE_RANGE = (12, 20)
OBJECT_COUNT_RANGE = (2, 4)
PALETTE = tuple(color for color in range(10) if color != BACKGROUND_COLOR)
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400
MIN_GAP = 1


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, object]] = []

    for r0 in range(height):
        for c0 in range(width):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue
            color = int(g[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        stack.append((rr, cc))

            rows = [r for r, _c in coords]
            cols = [c for _r, c in coords]
            objects.append(
                {
                    "color": color,
                    "coords": np.array(sorted(coords), dtype=int),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return objects


def _line_coords(bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if width >= height:
        row = (r0 + r1) // 2
        return np.array([(row, c) for c in range(c0, c1 + 1)], dtype=int)
    col = (c0 + c1) // 2
    return np.array([(r, col) for r in range(r0, r1 + 1)], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each foreground object onto a 1-pixel-wide segment along its longest bbox axis.

    Objects are 4-connected monochrome components on an azure background. Each object
    is replaced by a centered horizontal or vertical line whose length matches the
    object's span along its longer bbox dimension.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    objects = _find_objects(g, bg)
    assert objects, "expected at least one foreground object"

    out = np.full(g.shape, bg, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        line = _line_coords(obj["bbox"])
        for r, c in line:
            rr = int(r)
            cc = int(c)
            assert out[rr, cc] == bg, "compressed objects must not overlap"
            out[rr, cc] = color

    return out


def _odd_choices(limit: int) -> list[int]:
    return [value for value in (3, 5, 7) if value <= limit and value % 2 == 1]


def _horizontal_shape(height: int, width: int) -> np.ndarray:
    mid = height // 2
    cells = {(mid, c) for c in range(width)}

    branch_cols = list(range(width))
    random.shuffle(branch_cols)
    branch_total = random.randint(2, min(width, max(2, width // 2 + 1)))
    top_cols = branch_cols[: max(1, branch_total // 2)]
    bottom_cols = branch_cols[max(1, branch_total // 2) : branch_total]
    if not bottom_cols:
        bottom_cols = [random.choice(branch_cols)]

    top_anchor = random.choice(top_cols)
    bottom_anchor = random.choice(bottom_cols)

    for c in top_cols:
        length = mid if c == top_anchor else random.randint(1, mid)
        for offset in range(1, length + 1):
            cells.add((mid - offset, c))
        tip_r = mid - length
        if 0 < c < width - 1 and random.random() < 0.5:
            cells.add((tip_r, c + random.choice((-1, 1))))

    for c in bottom_cols:
        length = height - 1 - mid if c == bottom_anchor else random.randint(1, height - 1 - mid)
        for offset in range(1, length + 1):
            cells.add((mid + offset, c))
        tip_r = mid + length
        if 0 < c < width - 1 and random.random() < 0.5:
            cells.add((tip_r, c + random.choice((-1, 1))))

    return np.array(sorted(cells), dtype=int)


def _vertical_shape(height: int, width: int) -> np.ndarray:
    mid = width // 2
    cells = {(r, mid) for r in range(height)}

    branch_rows = list(range(height))
    random.shuffle(branch_rows)
    branch_total = random.randint(2, min(height, max(2, height // 2 + 1)))
    left_rows = branch_rows[: max(1, branch_total // 2)]
    right_rows = branch_rows[max(1, branch_total // 2) : branch_total]
    if not right_rows:
        right_rows = [random.choice(branch_rows)]

    left_anchor = random.choice(left_rows)
    right_anchor = random.choice(right_rows)

    for r in left_rows:
        length = mid if r == left_anchor else random.randint(1, mid)
        for offset in range(1, length + 1):
            cells.add((r, mid - offset))
        tip_c = mid - length
        if 0 < r < height - 1 and random.random() < 0.5:
            cells.add((r + random.choice((-1, 1)), tip_c))

    for r in right_rows:
        length = width - 1 - mid if r == right_anchor else random.randint(1, width - 1 - mid)
        for offset in range(1, length + 1):
            cells.add((r, mid + offset))
        tip_c = mid + length
        if 0 < r < height - 1 and random.random() < 0.5:
            cells.add((r + random.choice((-1, 1)), tip_c))

    return np.array(sorted(cells), dtype=int)


def _sample_descriptor(orientation: str, grid_h: int, grid_w: int, color: int) -> dict[str, object] | None:
    if orientation == "horizontal":
        heights = _odd_choices(grid_h - 2)
        if not heights:
            return None
        height = random.choice(heights)
        width_min = max(height + 1, 4)
        width_max = min(grid_w - 2, 10)
        widths = [value for value in range(width_min, width_max + 1) if value > height]
        if not widths:
            return None
        width = random.choice(widths)
        local = _horizontal_shape(height, width)
    else:
        widths = _odd_choices(grid_w - 2)
        if not widths:
            return None
        width = random.choice(widths)
        height_min = max(width + 1, 4)
        height_max = min(grid_h - 2, 10)
        heights = [value for value in range(height_min, height_max + 1) if value > width]
        if not heights:
            return None
        height = random.choice(heights)
        local = _vertical_shape(height, width)

    return {
        "orientation": orientation,
        "color": int(color),
        "height": int(height),
        "width": int(width),
        "local": local,
    }


def _position_range(total: int, span: int) -> tuple[int, int]:
    if total - span >= 2:
        return 1, total - span - 1
    return 0, total - span


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _place_descriptors(grid_h: int, grid_w: int, descriptors: list[dict[str, object]]) -> bool:
    occupied_rects: list[tuple[int, int, int, int]] = []
    descriptors.sort(key=lambda desc: int(desc["height"]) * int(desc["width"]), reverse=True)

    for desc in descriptors:
        height = int(desc["height"])
        width = int(desc["width"])
        r_lo, r_hi = _position_range(grid_h, height)
        c_lo, c_hi = _position_range(grid_w, width)
        if r_lo > r_hi or c_lo > c_hi:
            return False

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(r_lo, r_hi)
            left = random.randint(c_lo, c_hi)
            rect = (top - MIN_GAP, left - MIN_GAP, top + height - 1 + MIN_GAP, left + width - 1 + MIN_GAP)
            if all(_rects_separated(rect, other) for other in occupied_rects):
                desc["top"] = int(top)
                desc["left"] = int(left)
                occupied_rects.append(rect)
                placed = True
                break
        if not placed:
            return False

    return True


def _render_input(grid_h: int, grid_w: int, descriptors: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BACKGROUND_COLOR, dtype=int)
    for desc in descriptors:
        top = int(desc["top"])
        left = int(desc["left"])
        color = int(desc["color"])
        local = np.asarray(desc["local"], dtype=int)
        for r, c in local:
            grid[top + int(r), left + int(c)] = color
    return grid


def _render_expected(grid_h: int, grid_w: int, descriptors: list[dict[str, object]]) -> np.ndarray:
    out = np.full((grid_h, grid_w), BACKGROUND_COLOR, dtype=int)
    for desc in descriptors:
        top = int(desc["top"])
        left = int(desc["left"])
        height = int(desc["height"])
        width = int(desc["width"])
        color = int(desc["color"])
        bbox = (top, left, top + height - 1, left + width - 1)
        for r, c in _line_coords(bbox):
            out[int(r), int(c)] = color
    return out


def _orientations(n_objects: int) -> list[str]:
    horizontal = n_objects // 2
    vertical = n_objects // 2
    if n_objects % 2 == 1:
        if random.random() < 0.5:
            horizontal += 1
        else:
            vertical += 1
    values = ["horizontal"] * horizontal + ["vertical"] * vertical
    random.shuffle(values)
    return values


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_SIZE_RANGE)
    grid_w = random.randint(*GRID_SIZE_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, n_objects)
    orientations = _orientations(n_objects)

    descriptors: list[dict[str, object]] = []
    for orientation, color in zip(orientations, colors):
        desc = _sample_descriptor(orientation, grid_h, grid_w, color)
        if desc is None:
            return None
        descriptors.append(desc)

    if not _place_descriptors(grid_h, grid_w, descriptors):
        return None

    grid = _render_input(grid_h, grid_w, descriptors)
    objects = _find_objects(grid, BACKGROUND_COLOR)
    if len(objects) != n_objects:
        return None

    expected = _render_expected(grid_h, grid_w, descriptors)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
