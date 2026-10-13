from collections import defaultdict

import numpy as np


def _components(grid):
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            color = grid[r, c]
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not seen[nr, nc]
                        and grid[nr, nc] == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append((cells, int(color)))
    return comps


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each translated-shape group to the group's unique minority color.

    Input structure:
    The grid has a black background and several separated, 4-connected,
    single-color foreground objects. Objects may use different foreground
    colors. After translating every object so its bounding-box upper-left
    corner is at the origin, objects with identical occupied-cell masks form
    shape groups; every valid group contains at least three objects.

    Rule:
    Find all same-color 4-connected foreground components. Normalize each
    component by translation and group components with identical normalized
    masks. Within each group, count object colors. The unique least frequent
    color in that group is the target color, and every pixel of every object in
    the group is repainted to that target color. Background pixels stay black.

    Key insight:
    The decisive comparison is translation-normalized shape, not position or
    color; once equal shapes are grouped, the single color outlier overrides the
    majority color for the whole group.
    """
    out = np.array(grid, copy=True)
    groups = defaultdict(list)

    for cells, color in _components(grid):
        groups[_normalize(cells)].append((cells, color))

    for objects in groups.values():
        counts = {}
        for _, color in objects:
            counts[color] = counts.get(color, 0) + 1
        fewest = min(counts.values())
        candidates = [color for color, count in counts.items() if count == fewest]
        target = candidates[0]
        for cells, _ in objects:
            for r, c in cells:
                out[r, c] = target
    return out


def _random_polyomino(rng):
    size = int(rng.integers(3, 7))
    cells = {(0, 0)}
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while len(cells) < size:
        base = list(cells)[int(rng.integers(len(cells)))]
        dr, dc = dirs[int(rng.integers(4))]
        cells.add((base[0] + dr, base[1] + dc))
    return _orient(_normalize(cells), rng)


def _orient(shape, rng):
    pts = np.array(shape, dtype=int)
    for _ in range(int(rng.integers(4))):
        pts = np.column_stack((pts[:, 1], -pts[:, 0]))
    if bool(rng.integers(2)):
        pts[:, 1] *= -1
    return _normalize([tuple(p) for p in pts])


def _make_shapes(rng, count):
    shapes = []
    seen = set()
    for _ in range(200):
        shape = _random_polyomino(rng)
        h, w = _dims(shape)
        if h > 6 or w > 6 or shape in seen:
            continue
        shapes.append(shape)
        seen.add(shape)
        if len(shapes) == count:
            return shapes
    fallback = [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ]
    for shape in fallback:
        if shape not in seen:
            shapes.append(shape)
            seen.add(shape)
            if len(shapes) == count:
                break
    return shapes


def _can_place(grid, shape, top, left):
    rows, cols = grid.shape
    for dr, dc in shape:
        r, c = top + dr, left + dc
        if not (0 <= r < rows and 0 <= c < cols) or grid[r, c] != 0:
            return False
    for dr, dc in shape:
        r, c = top + dr, left + dc
        for nr in range(max(0, r - 1), min(rows, r + 2)):
            for nc in range(max(0, c - 1), min(cols, c + 2)):
                if grid[nr, nc] != 0:
                    return False
    return True


def _place_all(rows, cols, objects, rng):
    grid = np.zeros((rows, cols), dtype=int)
    for shape, color in objects:
        h, w = _dims(shape)
        if h > rows or w > cols:
            return None
        placed = False
        for _ in range(500):
            top = int(rng.integers(0, rows - h + 1))
            left = int(rng.integers(0, cols - w + 1))
            if _can_place(grid, shape, top, left):
                for dr, dc in shape:
                    grid[top + dr, left + dc] = color
                placed = True
                break
        if not placed:
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = tuple(range(1, 10))

    for _ in range(120):
        rows = int(rng.integers(18, 31))
        cols = int(rng.integers(18, 31))
        area = rows * cols
        group_count = int(rng.integers(2, 4))
        if area > 650 and bool(rng.integers(2)):
            group_count += 1

        shapes = _make_shapes(rng, group_count)
        objects = []
        for shape in shapes:
            group_size = int(rng.integers(3, 5))
            if area > 520 and rng.random() < 0.35:
                group_size += 1
            majority = int(rng.choice(palette))
            minority_choices = [color for color in palette if color != majority]
            minority = int(rng.choice(minority_choices))
            odd_index = int(rng.integers(group_size))
            for i in range(group_size):
                color = minority if i == odd_index else majority
                objects.append((shape, color))

        rng.shuffle(objects)
        grid = _place_all(rows, cols, objects, rng)
        if grid is not None:
            return grid

    shapes = [
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
    ]
    objects = []
    for shape, majority, minority in ((shapes[0], 2, 4), (shapes[1], 6, 3)):
        for i in range(3):
            objects.append((shape, minority if i == 0 else majority))
    grid = _place_all(30, 30, objects, rng)
    if grid is not None:
        return grid
    grid = np.zeros((30, 30), dtype=int)
    fixed = [
        (shapes[0], 4, 1, 1),
        (shapes[0], 2, 1, 8),
        (shapes[0], 2, 1, 15),
        (shapes[1], 3, 10, 2),
        (shapes[1], 6, 10, 10),
        (shapes[1], 6, 10, 18),
    ]
    for shape, color, top, left in fixed:
        for dr, dc in shape:
            grid[top + dr, left + dc] = color
    return grid
