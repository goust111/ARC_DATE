from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
OBJECT_COLORS = (1, 2, 3, 4, 7)
MAX_ATTEMPTS = 2000


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2, "expected a 2D grid"
    assert int(_get_bg(g)) == BLACK, "background must be black"
    assert np.all(g[:, 0] == GREY), "left wall must be a full grey column"

    movable = (g != BLACK) & (g != GREY)
    comps = _components(movable)
    assert comps, "expected at least one movable object"

    objects: list[dict] = []
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each object must be monochromatic"
        color = int(next(iter(colors)))
        r0, c0, r1, c1 = _bbox(coords)
        objects.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "top": r0,
                "left": c0,
                "area": int(coords.shape[0]),
            }
        )
    return objects


def _slide_objects(shape: tuple[int, int], objects: list[dict]) -> list[dict]:
    h, w = shape
    settled = np.zeros((h, w), dtype=bool)
    settled[:, 0] = True

    moved: list[dict] = []
    order = sorted(objects, key=lambda obj: (obj["left"], obj["top"], obj["color"]))
    for obj in order:
        coords = obj["coords"]
        shift = 0
        while True:
            candidate = coords + np.array([0, -(shift + 1)], dtype=int)
            if int(candidate[:, 1].min()) < 1:
                break
            if np.any(settled[candidate[:, 0], candidate[:, 1]]):
                break
            shift += 1

        final_coords = coords + np.array([0, -shift], dtype=int)
        settled[final_coords[:, 0], final_coords[:, 1]] = True

        moved.append(
            {
                **obj,
                "distance": int(shift),
                "final_coords": final_coords,
                "final_left": int(final_coords[:, 1].min()),
            }
        )
    return moved


def _render_histogram(objects: list[dict]) -> np.ndarray:
    ordered = sorted(objects, key=lambda obj: (obj["top"], obj["left"], obj["color"]))
    heights = [int(obj["distance"]) for obj in ordered]
    assert heights and max(heights) > 0

    out = np.zeros((max(heights), len(ordered)), dtype=int)
    for col, obj in enumerate(ordered):
        d = int(obj["distance"])
        if d > 0:
            out[-d:, col] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored object left into the grey wall, then render each object's travel distance as a histogram bar.

    Input structure:
    - The background is black, and the full leftmost column is a solid grey wall.
    - Several small single-color 4-connected objects appear to the right of the wall.
    - Objects are separated in the input so they can be identified individually.

    Rule:
    - Process the colored objects from nearest to farthest from the grey wall, measuring distance by each object's leftmost occupied column.
    - Rigidly slide each object left until its left edge reaches column 1 or the object would collide with an already settled object.
    - Record each object's horizontal travel distance D.
    - Clear the canvas and output a bottom-aligned histogram whose width is the number of objects and whose height is max(D).
    - Sort histogram columns by the objects' original top-row positions, and fill each column upward with D cells of that object's color.

    Key insight:
    - The output does not show the settled objects themselves; it converts each object's slide distance into a colored bar, and the bar order comes from original vertical position rather than color or distance.
    """
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    moved = _slide_objects(g.shape, objects)
    return _render_histogram(moved)


def _shape_catalog() -> dict[int, list[np.ndarray]]:
    def arr(rows: list[list[int]]) -> np.ndarray:
        return np.array(rows, dtype=bool)

    return {
        2: [
            arr([[1, 1]]),
            arr([[1], [1]]),
        ],
        3: [
            arr([[1, 1, 1]]),
            arr([[1], [1], [1]]),
            arr([[1, 0], [1, 1]]),
            arr([[0, 1], [1, 1]]),
            arr([[1, 1], [1, 0]]),
            arr([[1, 1], [0, 1]]),
        ],
        4: [
            arr([[1, 1, 1, 1]]),
            arr([[1], [1], [1], [1]]),
            arr([[1, 1], [1, 1]]),
            arr([[1, 1, 1], [0, 1, 0]]),
            arr([[0, 1, 0], [1, 1, 1]]),
            arr([[1, 0], [1, 1], [1, 0]]),
            arr([[0, 1], [1, 1], [0, 1]]),
            arr([[1, 1, 0], [0, 1, 1]]),
            arr([[0, 1, 1], [1, 1, 0]]),
        ],
    }


SHAPES = _shape_catalog()


def _choose_masks(n: int) -> list[np.ndarray] | None:
    areas = [random.choice((2, 3, 4)) for _ in range(n)]
    if len(set(areas)) == len(areas):
        areas[random.randrange(n)] = random.choice(areas)

    max_height = 2 if n == 5 else 3 if n == 4 else 4
    masks: list[np.ndarray] = []
    for area in areas:
        candidates = [mask for mask in SHAPES[area] if mask.shape[0] <= max_height]
        if not candidates:
            return None
        masks.append(random.choice(candidates).copy())
    return masks


def _pairwise_disjoint(sets: list[set[int]]) -> bool:
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                return False
    return True


def _semantic_output(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    objects = _extract_objects(grid)
    moved = _slide_objects(grid.shape, objects)
    return moved, _render_histogram(moved)


def _is_valid_instance(grid: np.ndarray) -> bool:
    try:
        moved, expected = _semantic_output(grid)
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False

    n = len(moved)
    if not (3 <= n <= 5):
        return False

    colors = [int(obj["color"]) for obj in moved]
    if len(set(colors)) != n:
        return False

    row_sets = [{int(r) for r in obj["coords"][:, 0]} for obj in moved]
    col_sets = [{int(c) for c in obj["coords"][:, 1]} for obj in moved]
    if not _pairwise_disjoint(row_sets):
        return False
    if not _pairwise_disjoint(col_sets):
        return False

    for i in range(n):
        r0a, c0a, r1a, c1a = moved[i]["bbox"]
        for j in range(i + 1, n):
            r0b, c0b, r1b, c1b = moved[j]["bbox"]
            if not (r1a < r0b or r1b < r0a or c1a < c0b or c1b < c0a):
                return False

    distances = [int(obj["distance"]) for obj in moved]
    if min(distances) <= 0 or len(set(distances)) != n or max(distances) <= 4:
        return False

    final_pixels = set()
    for obj in moved:
        if int(obj["final_left"]) != 1:
            return False
        for r, c in obj["final_coords"]:
            cell = (int(r), int(c))
            if cell in final_pixels:
                return False
            final_pixels.add(cell)

    row_order = sorted(range(n), key=lambda i: (moved[i]["top"], moved[i]["left"], moved[i]["color"]))
    dist_order = sorted(range(n), key=lambda i: (moved[i]["left"], moved[i]["top"], moved[i]["color"]))
    if row_order == dist_order:
        return False

    row_colors = [colors[i] for i in row_order]
    if row_colors == sorted(row_colors):
        return False

    row_areas = [int(moved[i]["area"]) for i in row_order]
    row_distances = [int(moved[i]["distance"]) for i in row_order]
    if not any(
        row_areas[i] == row_areas[j] and row_distances[i] != row_distances[j]
        for i in range(n)
        for j in range(i + 1, n)
    ):
        return False

    if expected.shape != (max(row_distances), n):
        return False
    for col, obj_idx in enumerate(row_order):
        color = int(moved[obj_idx]["color"])
        dist = int(moved[obj_idx]["distance"])
        column = expected[:, col]
        if not np.all(np.isin(column, [BLACK, color])):
            return False
        if int(np.count_nonzero(column == color)) != dist:
            return False
        if not np.all(column[-dist:] == color):
            return False
        if dist < expected.shape[0] and not np.all(column[: expected.shape[0] - dist] == BLACK):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    n = random.randint(3, 5)
    colors = random.sample(OBJECT_COLORS, n)
    masks = _choose_masks(n)
    if masks is None:
        return None

    heights = [int(mask.shape[0]) for mask in masks]
    widths = [int(mask.shape[1]) for mask in masks]

    top_margin = random.randint(0, 1)
    bottom_margin = random.randint(0, 1)
    between_gaps = [random.randint(1, 2) for _ in range(n - 1)]
    h = top_margin + bottom_margin + sum(heights) + sum(between_gaps)
    if not (10 <= h <= 18):
        return None

    left_to_right = list(range(n))
    random.shuffle(left_to_right)
    if left_to_right == list(range(n)):
        return None

    lead_gap = random.randint(1, 3)
    inner_gaps = [random.randint(1, 2) for _ in range(n - 1)]
    right_margin = random.randint(0, 2)
    w = 1 + lead_gap + right_margin + sum(widths) + sum(inner_gaps)
    if not (10 <= w <= 20):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = GREY

    top_rows: list[int] = []
    cur_r = top_margin
    for i in range(n):
        top_rows.append(cur_r)
        cur_r += heights[i]
        if i < n - 1:
            cur_r += between_gaps[i]

    starts = [0] * n
    cur_c = 1 + lead_gap
    for pos, obj_idx in enumerate(left_to_right):
        starts[obj_idx] = cur_c
        cur_c += widths[obj_idx]
        if pos < n - 1:
            cur_c += inner_gaps[pos]

    for i in range(n):
        mask = masks[i]
        rr, cc = np.where(mask)
        grid[top_rows[i] + rr, starts[i] + cc] = colors[i]

    return grid if _is_valid_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-distance histogram input")
