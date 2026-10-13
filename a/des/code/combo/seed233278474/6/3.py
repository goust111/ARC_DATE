import random
from collections import Counter, deque

import numpy as np


SEPARATOR_COLOR = 8
MARKER_COLOR = 7
BACKGROUND_CHOICES = [0, 1, 2, 3, 4, 5, 6, 9]
MAX_GENERATION_ATTEMPTS = 400
QUADRANTS = ("ru", "lu", "ld", "rd")


def _dominant_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1]
    return int(ranked[0][0])


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for sep in separators:
        assert start < sep
        spans.append((start, sep))
        start = sep + 1
    assert start < length
    spans.append((start, length))
    return spans


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    seen = set()
    start = tuple(map(int, coords[0]))
    queue = deque([start])
    seen.add(start)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                queue.append((rr, cc))
    return len(seen) == int(mask.sum())


def _transform_mask(mask: np.ndarray, quadrant: str) -> np.ndarray:
    if quadrant == "ru":
        return np.fliplr(mask).copy()
    if quadrant == "lu":
        return np.flipud(mask).copy()
    if quadrant == "ld":
        return np.rot90(mask, 2).copy()
    assert quadrant == "rd"
    return mask.copy()


def _quadrant_from_marker(marker: tuple[int, int], center_r: float, center_c: float) -> str:
    mr, mc = marker
    if mr < center_r and mc > center_c:
        return "ru"
    if mr < center_r and mc < center_c:
        return "lu"
    if mr > center_r and mc < center_c:
        return "ld"
    if mr > center_r and mc > center_c:
        return "rd"
    raise AssertionError("marker must lie in a strict quadrant of the object bbox center")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grid by azure separator lines and transform each region's object
    according to the orange marker's quadrant relative to that object's bbox center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    h, w = g.shape
    row_seps = [r for r in range(h) if np.all(g[r, :] == SEPARATOR_COLOR)]
    col_seps = [c for c in range(w) if np.all(g[:, c] == SEPARATOR_COLOR)]
    assert row_seps and col_seps

    row_spans = _spans(h, row_seps)
    col_spans = _spans(w, col_seps)

    bg = _dominant_color(g[g != SEPARATOR_COLOR])
    assert bg not in {SEPARATOR_COLOR, MARKER_COLOR}

    out = g.copy()
    out[g == MARKER_COLOR] = bg

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0:r1, c0:c1]
            marker_coords = np.argwhere(patch == MARKER_COLOR)
            assert marker_coords.shape == (1, 2)
            marker = tuple(map(int, marker_coords[0]))

            colors = set(map(int, np.unique(patch))) - {bg, MARKER_COLOR}
            assert len(colors) == 1
            obj_color = colors.pop()

            obj_mask = patch == obj_color
            assert _is_connected(obj_mask)

            coords = np.argwhere(obj_mask)
            rr0, cc0 = coords.min(axis=0)
            rr1, cc1 = coords.max(axis=0)
            rr0 = int(rr0)
            cc0 = int(cc0)
            rr1 = int(rr1)
            cc1 = int(cc1)

            center_r = (rr0 + rr1) / 2
            center_c = (cc0 + cc1) / 2
            quadrant = _quadrant_from_marker(marker, center_r, center_c)

            bbox_mask = obj_mask[rr0 : rr1 + 1, cc0 : cc1 + 1]
            transformed = _transform_mask(bbox_mask, quadrant)

            patch_out = out[r0:r1, c0:c1]
            patch_out[obj_mask] = bg
            bbox_out = patch_out[rr0 : rr1 + 1, cc0 : cc1 + 1]
            bbox_out[transformed] = obj_color

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(120):
        raw_h = random.randint(2, min(5, max_h))
        raw_w = random.randint(2, min(5, max_w))
        max_area = raw_h * raw_w - 1
        if max_area < 3:
            continue
        area = random.randint(3, max_area)

        mask = np.zeros((raw_h, raw_w), dtype=bool)
        seed = (random.randrange(raw_h), random.randrange(raw_w))
        cells = {seed}
        mask[seed] = True

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < raw_h and 0 <= cc < raw_w and not mask[rr, cc]:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cell = random.choice(frontier)
            mask[cell] = True
            cells.add(cell)

        if int(mask.sum()) != area:
            continue

        cropped = _crop_mask(mask)
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if np.array_equal(cropped, np.fliplr(cropped)):
            continue
        if np.array_equal(cropped, np.flipud(cropped)):
            continue
        if np.array_equal(cropped, np.rot90(cropped, 2)):
            continue
        return cropped

    return None


def _marker_ranges(region_h: int, region_w: int, top: int, left: int, mask: np.ndarray, quadrant: str) -> tuple[range, range]:
    h, w = mask.shape
    if quadrant == "ru":
        return range(0, top - 1), range(left + w + 1, region_w)
    if quadrant == "lu":
        return range(0, top - 1), range(0, left - 1)
    if quadrant == "ld":
        return range(top + h + 1, region_h), range(0, left - 1)
    return range(top + h + 1, region_h), range(left + w + 1, region_w)


def _render_region(region_h: int, region_w: int, bg: int, obj_color: int, quadrant: str) -> np.ndarray | None:
    shape = _random_connected_mask(region_h - 4, region_w - 4)
    if shape is None:
        return None

    h, w = shape.shape
    top = random.randint(2, region_h - h - 2)
    left = random.randint(2, region_w - w - 2)

    row_range, col_range = _marker_ranges(region_h, region_w, top, left, shape, quadrant)
    rows = list(row_range)
    cols = list(col_range)
    if not rows or not cols:
        return None

    region = np.full((region_h, region_w), bg, dtype=int)
    region[top : top + h, left : left + w][shape] = obj_color
    marker = (random.choice(rows), random.choice(cols))
    region[marker] = MARKER_COLOR
    return region


def _separator_positions(sizes: list[int]) -> list[int]:
    positions = []
    cursor = sizes[0]
    for size in sizes[1:]:
        positions.append(cursor)
        cursor += 1 + size
    return positions


def _bounds_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    bounds = []
    cursor = 0
    for size in sizes:
        bounds.append((cursor, cursor + size))
        cursor += size + 1
    return bounds


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BACKGROUND_CHOICES)
    obj_colors = [color for color in BACKGROUND_CHOICES if color != bg]

    row_count = random.randint(2, 4)
    col_count = random.randint(2, 4)
    row_sizes = [random.randint(7, 10) for _ in range(row_count)]
    col_sizes = [random.randint(7, 10) for _ in range(col_count)]

    height = sum(row_sizes) + (row_count - 1)
    width = sum(col_sizes) + (col_count - 1)
    grid = np.full((height, width), bg, dtype=int)

    for r in _separator_positions(row_sizes):
        grid[r, :] = SEPARATOR_COLOR
    for c in _separator_positions(col_sizes):
        grid[:, c] = SEPARATOR_COLOR

    quadrants = list(QUADRANTS)
    total_regions = row_count * col_count
    while len(quadrants) < total_regions:
        quadrants.append(random.choice(QUADRANTS))
    random.shuffle(quadrants)

    row_bounds = _bounds_from_sizes(row_sizes)
    col_bounds = _bounds_from_sizes(col_sizes)
    q_index = 0
    for r0, r1 in row_bounds:
        for c0, c1 in col_bounds:
            obj_color = random.choice(obj_colors)
            region = _render_region(r1 - r0, c1 - c0, bg, obj_color, quadrants[q_index])
            if region is None:
                return None
            grid[r0:r1, c0:c1] = region
            q_index += 1

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
