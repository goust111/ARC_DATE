import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_OBJECTS = 2
MAX_OBJECTS = 4
MIN_BBOX = 5
MAX_BBOX = 8
GRID_MARGIN = 4
GRID_SLACK = 8
PLACEMENT_GAP = 1
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 80
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(cols.min()),
        int(rows.max()) + 1,
        int(cols.max()) + 1,
    )


def _compose_local(objects: list[dict], bbox_h: int, bbox_w: int) -> np.ndarray:
    out = np.zeros((bbox_h, bbox_w), dtype=int)
    for obj in sorted(objects, key=lambda item: int(item["layer"])):
        mask = obj["mask"]
        out[mask] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    body_mask = (g != BLACK) & (g != GREY)
    comps = _components(body_mask)
    assert comps, "expected at least one shape"

    objects = []
    bbox_shape = None
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each shape must be monochrome"
        color = colors.pop()

        r0, c0, r1, c1 = _bbox(comp)
        bbox_h = r1 - r0
        bbox_w = c1 - c0
        if bbox_shape is None:
            bbox_shape = (bbox_h, bbox_w)
        else:
            assert bbox_shape == (bbox_h, bbox_w), "all shape bboxes must match"

        layer = int(np.count_nonzero(g[r0:r1, c0:c1] == GREY))
        assert 1 <= layer <= 4, "grey marker count must be in [1, 4]"

        local = comp - np.array([[r0, c0]], dtype=int)
        objects.append({"layer": layer, "color": color, "coords": local})

    assert bbox_shape is not None
    out = np.zeros(bbox_shape, dtype=int)
    for obj in sorted(objects, key=lambda item: int(item["layer"])):
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return out


def _random_shortest_path(
    start: tuple[int, int], end: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]]:
    r, c = start
    path = [(r, c)]
    while (r, c) != end:
        options = []
        if r < end[0]:
            options.append((1, 0))
        elif r > end[0]:
            options.append((-1, 0))
        if c < end[1]:
            options.append((0, 1))
        elif c > end[1]:
            options.append((0, -1))
        dr, dc = rng.choice(options)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _touches_all_sides(shape: set[tuple[int, int]], bbox_h: int, bbox_w: int) -> bool:
    touches_top = any(r == 0 for r, _ in shape)
    touches_bottom = any(r == bbox_h - 1 for r, _ in shape)
    touches_left = any(c == 0 for _, c in shape)
    touches_right = any(c == bbox_w - 1 for _, c in shape)
    return touches_top and touches_bottom and touches_left and touches_right


def _grow_shape(
    bbox_h: int, bbox_w: int, min_open_cells: int, rng: random.Random
) -> np.ndarray | None:
    max_area = bbox_h * bbox_w - min_open_cells
    for _ in range(SHAPE_RETRIES):
        center = (rng.randrange(bbox_h), rng.randrange(bbox_w))
        anchors = [
            (0, rng.randrange(bbox_w)),
            (bbox_h - 1, rng.randrange(bbox_w)),
            (rng.randrange(bbox_h), 0),
            (rng.randrange(bbox_h), bbox_w - 1),
        ]

        shape: set[tuple[int, int]] = set()
        for anchor in anchors:
            shape.update(_random_shortest_path(anchor, center, rng))

        if not _touches_all_sides(shape, bbox_h, bbox_w):
            continue
        if len(shape) > max_area:
            continue

        min_target = max(len(shape), (bbox_h * bbox_w) // 3)
        if min_target > max_area:
            continue
        target_area = rng.randint(min_target, max_area)

        while len(shape) < target_area:
            frontier = set()
            for r, c in shape:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < bbox_h and 0 <= nc < bbox_w and (nr, nc) not in shape:
                        frontier.add((nr, nc))
            if not frontier:
                break
            shape.add(rng.choice(tuple(frontier)))

        if len(shape) != target_area:
            continue
        if not _touches_all_sides(shape, bbox_h, bbox_w):
            continue
        return np.array(sorted(shape), dtype=int)
    return None


def _place_markers(mask: np.ndarray, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    open_cells = [(r, c) for r, c in np.argwhere(~mask)]
    if len(open_cells) < count:
        return None
    rng.shuffle(open_cells)
    return open_cells[:count]


def _rectangles_conflict(
    rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return not (
        ar1 + gap <= br0
        or br1 + gap <= ar0
        or ac1 + gap <= bc0
        or bc1 + gap <= ac0
    )


def _place_rectangles(
    grid_h: int, grid_w: int, bbox_h: int, bbox_w: int, count: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    tops_lefts: list[tuple[int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(0, grid_h - bbox_h)
            left = rng.randint(0, grid_w - bbox_w)
            rect = (top, left, top + bbox_h, left + bbox_w)
            if any(_rectangles_conflict(rect, other, PLACEMENT_GAP) for other in rects):
                continue
            rects.append(rect)
            tops_lefts.append((top, left))
            placed = True
            break
        if not placed:
            return None
    return tops_lefts


def _every_layer_visible(objects: list[dict], bbox_h: int, bbox_w: int) -> bool:
    higher = np.zeros((bbox_h, bbox_w), dtype=bool)
    for obj in sorted(objects, key=lambda item: int(item["layer"]), reverse=True):
        visible = obj["mask"] & ~higher
        if not np.any(visible):
            return False
        higher |= obj["mask"]
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
    bbox_h = rng.randint(MIN_BBOX, MAX_BBOX)
    bbox_w = rng.randint(MIN_BBOX, MAX_BBOX)

    max_dim_h = max(2 * bbox_h + GRID_MARGIN, bbox_h + GRID_MARGIN + 1)
    max_dim_w = max(2 * bbox_w + GRID_MARGIN, bbox_w + GRID_MARGIN + 1)
    grid_h = rng.randint(max_dim_h, max_dim_h + GRID_SLACK)
    grid_w = rng.randint(max_dim_w, max_dim_w + GRID_SLACK)

    positions = _place_rectangles(grid_h, grid_w, bbox_h, bbox_w, object_count, rng)
    if positions is None:
        return None

    colors = rng.sample(PALETTE, object_count)
    layers = rng.sample([1, 2, 3, 4], object_count)

    objects = []
    for idx in range(object_count):
        coords = _grow_shape(bbox_h, bbox_w, min_open_cells=4, rng=rng)
        if coords is None:
            return None
        mask = np.zeros((bbox_h, bbox_w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        markers = _place_markers(mask, layers[idx], rng)
        if markers is None:
            return None
        objects.append(
            {
                "layer": layers[idx],
                "color": colors[idx],
                "mask": mask,
                "markers": markers,
            }
        )

    coverage = np.zeros((bbox_h, bbox_w), dtype=int)
    for obj in objects:
        coverage += obj["mask"].astype(int)
    if int(coverage.max()) < 2:
        return None
    if not _every_layer_visible(objects, bbox_h, bbox_w):
        return None

    expected = _compose_local(objects, bbox_h, bbox_w)
    visible_colors = {int(v) for v in expected.reshape(-1) if int(v) != BLACK}
    if len(visible_colors) < 2:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for (top, left), obj in zip(positions, objects):
        rows, cols = np.where(obj["mask"])
        grid[top + rows, left + cols] = int(obj["color"])
        for mr, mc in obj["markers"]:
            grid[top + mr, left + mc] = GREY

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
