import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COUNT_PATTERNS = (
    (3, 2, 1),
    (3, 1, 2),
    (2, 3, 1),
    (1, 3, 2),
    (2, 1, 3),
    (1, 2, 3),
    (2, 2, 1),
    (2, 1, 2),
    (1, 2, 2),
    (3, 3, 1),
    (3, 1, 3),
    (1, 3, 3),
    (2, 2, 2),
)
GRID_MIN = 20
GRID_MAX = 26
MAX_SHAPE_TRIES = 120
MAX_GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            result.append(np.asarray(coords, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()),
        int(cols.max()),
    )


def _touch_count(coords: np.ndarray, height: int, width: int) -> int:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return sum(
        (
            int(np.any(rows == 0)),
            int(np.any(rows == height - 1)),
            int(np.any(cols == 0)),
            int(np.any(cols == width - 1)),
        )
    )


def _strip_from_shapes(shapes: list[np.ndarray], bg: int) -> np.ndarray:
    height = max(int(shape.shape[0]) for shape in shapes)
    width = sum(int(shape.shape[1]) for shape in shapes) + len(shapes) - 1
    strip = np.full((height, width), bg, dtype=int)
    cursor = 0
    for shape in shapes:
        h, w = map(int, shape.shape)
        strip[:h, cursor : cursor + w] = shape
        cursor += w + 1
    return strip


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group irregular shapes by how many outer grid edges they touch, then pack each group into a horizontal strip.

    Input structure:
    - The background is a single dominant color.
    - Several monochrome 4-connected irregular shapes are scattered on the grid and do not touch each other.
    - Each shape fits inside a tight bounding box no larger than 5x5.
    - Some shapes lie fully inside the grid, some touch exactly one outer edge, and some touch two outer edges at a corner.

    Rule:
    - Extract every non-background 4-connected shape and count how many distinct grid borders its pixels touch among top, bottom, left, and right.
    - Split the shapes into groups by touch count 0, 1, or 2.
    - Inside each group, sort shapes by bounding-box left column and then bounding-box top row; crop each shape to its tight bounding box and concatenate those crops left-to-right with one background column between neighbors.
    - Stack the non-empty group strips from top to bottom with one background row between strips, ordering the strips by group size from larger to smaller and breaking ties by putting the higher touch count first.

    Key insight:
    - The decisive property is not color or silhouette but the shape's spatial relationship to the outer frame: how many different borders it actually reaches.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    bg = _background_color(g)
    comps = _components(g != bg)
    assert comps, "expected at least one foreground shape"

    grouped: dict[int, list[tuple[int, int, np.ndarray]]] = {0: [], 1: [], 2: []}
    for coords in comps:
        touch = _touch_count(coords, height, width)
        assert 0 <= touch <= 2, "shapes must touch at most two borders"
        r0, c0, r1, c1 = _bbox(coords)
        grouped[touch].append((int(c0), int(r0), g[r0 : r1 + 1, c0 : c1 + 1].copy()))

    strips: list[tuple[int, int, np.ndarray]] = []
    for touch, items in grouped.items():
        if not items:
            continue
        items.sort(key=lambda item: (item[0], item[1]))
        strip = _strip_from_shapes([item[2] for item in items], bg)
        strips.append((len(items), int(touch), strip))

    assert strips, "expected at least one non-empty group"
    strips.sort(key=lambda item: (-item[0], -item[1]))

    out_h = sum(int(strip.shape[0]) for _, _, strip in strips) + len(strips) - 1
    out_w = max(int(strip.shape[1]) for _, _, strip in strips)
    out = np.full((out_h, out_w), bg, dtype=int)

    row = 0
    for index, (_, _, strip) in enumerate(strips):
        h, w = map(int, strip.shape)
        out[row : row + h, :w] = strip
        row += h
        if index != len(strips) - 1:
            row += 1

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_shape() -> np.ndarray:
    for _ in range(MAX_SHAPE_TRIES):
        base_h = random.randint(2, 5)
        base_w = random.randint(2, 5)
        max_area = base_h * base_w - 1
        if max_area < 3:
            continue
        target_area = random.randint(3, min(max_area, 8))
        seed = (random.randrange(base_h), random.randrange(base_w))
        cells = {seed}

        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < base_h and 0 <= nc < base_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        mask = np.zeros((base_h, base_w), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True

        mask = _crop_mask(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        mask = _crop_mask(mask)
        h, w = map(int, mask.shape)
        if h < 2 or w < 2:
            continue
        if int(mask.sum()) == h * w:
            continue
        return mask
    raise RuntimeError("failed to sample an irregular shape")


def _can_place(blocked: np.ndarray, top: int, left: int, shape: np.ndarray) -> bool:
    height, width = map(int, blocked.shape)
    h, w = map(int, shape.shape)
    if top < 0 or left < 0 or top + h > height or left + w > width:
        return False
    r0 = max(0, int(top - 1))
    r1 = min(height, int(top + h + 1))
    c0 = max(0, int(left - 1))
    c1 = min(width, int(left + w + 1))
    return not bool(blocked[r0:r1, c0:c1].any())


def _mark_blocked(blocked: np.ndarray, top: int, left: int, shape: np.ndarray) -> None:
    height, width = map(int, blocked.shape)
    h, w = map(int, shape.shape)
    r0 = max(0, int(top - 1))
    r1 = min(height, int(top + h + 1))
    c0 = max(0, int(left - 1))
    c1 = min(width, int(left + w + 1))
    blocked[r0:r1, c0:c1] = True


def _candidate_positions(group: int, tag: str, shape: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = map(int, shape.shape)
    if group == 2:
        fixed = {
            "tl": (0, 0),
            "tr": (0, width - w),
            "bl": (height - h, 0),
            "br": (height - h, width - w),
        }
        return [fixed[tag]]

    if group == 1:
        if tag == "top":
            cols = list(range(1, width - w))
            random.shuffle(cols)
            return [(0, int(col)) for col in cols]
        if tag == "bottom":
            cols = list(range(1, width - w))
            random.shuffle(cols)
            return [(height - h, int(col)) for col in cols]
        if tag == "left":
            rows = list(range(1, height - h))
            random.shuffle(rows)
            return [(int(row), 0) for row in rows]
        rows = list(range(1, height - h))
        random.shuffle(rows)
        return [(int(row), width - w) for row in rows]

    rows = list(range(1, height - h))
    cols = list(range(1, width - w))
    positions = [(int(r), int(c)) for r in rows for c in cols]
    random.shuffle(positions)
    return positions


def _place_shape(
    grid: np.ndarray,
    blocked: np.ndarray,
    shape: np.ndarray,
    color: int,
    group: int,
    tag: str,
) -> bool:
    height, width = map(int, grid.shape)
    for top, left in _candidate_positions(group, tag, shape, height, width):
        if not _can_place(blocked, top, left, shape):
            continue
        window = grid[top : top + shape.shape[0], left : left + shape.shape[1]]
        window[shape] = int(color)
        _mark_blocked(blocked, top, left, shape)
        return True
    return False


def _group_specs() -> list[tuple[int, str]]:
    counts = random.choice(COUNT_PATTERNS)
    specs: list[tuple[int, str]] = []

    corners = ["tl", "tr", "bl", "br"]
    random.shuffle(corners)
    edges = ["top", "bottom", "left", "right"]
    random.shuffle(edges)

    specs.extend((2, corners[i]) for i in range(counts[2]))
    specs.extend((1, edges[i]) for i in range(counts[1]))
    specs.extend((0, "inner") for _ in range(counts[0]))
    random.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    specs = _group_specs()
    colors = [random.choice(PALETTE) for _ in specs]

    for (group, tag), color in zip(specs, colors):
        shape = _random_shape()
        if not _place_shape(grid, blocked, shape, color, group, tag):
            return None

    comps = _components(grid != BG_COLOR)
    if len(comps) != len(specs):
        return None

    touch_hist = Counter(_touch_count(coords, height, width) for coords in comps)
    if any(touch_hist.get(group, 0) == 0 for group in (0, 1, 2)):
        return None

    out = main(grid)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
