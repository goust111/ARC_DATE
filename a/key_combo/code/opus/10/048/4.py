from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7
TEAL = 8

OUTPUT_SIZE = 5
GRID_MIN = 19
GRID_MAX = 26
OBJECT_COUNT_RANGE = (5, 7)
RED_COUNT_RANGE = (1, 2)
PAIR_CHOICES = [((3, 3), (2, 4)), ((4, 4), (3, 5)), ((5, 5), (4, 6))]
SQUARE_SIZES = [3, 4, 5, 6]
RECT_SIZES = [(2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
MAX_GENERATION_ATTEMPTS = 400
MAX_PATTERN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangle_border_cells(height: int, width: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(height):
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                cells.add((r, c))
    return cells


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _covers_every_row_col(cells: set[tuple[int, int]], height: int, width: int) -> bool:
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return len(rows) == height and len(cols) == width


def _sample_removed_cells(
    height: int, width: int, missing_count: int, require_corner_gap: bool
) -> set[tuple[int, int]] | None:
    border = list(_rectangle_border_cells(height, width))
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    for _ in range(MAX_PATTERN_ATTEMPTS):
        if require_corner_gap:
            corner = random.choice(corners)
            others = [cell for cell in border if cell != corner]
            removed = {corner, *random.sample(others, missing_count - 1)}
        else:
            removed = set(random.sample(border, missing_count))
        kept = set(border) - removed
        if not _covers_every_row_col(kept, height, width):
            continue
        if not _is_connected(kept):
            continue
        return removed
    return None


def _make_incomplete_rectangle(
    height: int, width: int, missing_count: int, require_corner_gap: bool
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    removed = _sample_removed_cells(height, width, missing_count, require_corner_gap)
    if removed is None:
        return None
    sprite = np.zeros((height, width), dtype=int)
    border = _rectangle_border_cells(height, width)
    for r, c in border - removed:
        sprite[r, c] = TEAL
    return sprite, removed


def _boxes_clear(candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _place_boxes(
    height: int, width: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    placed_boxes: list[tuple[int, int, int, int]] = []

    def search(i: int) -> bool:
        if i == len(order):
            return True
        idx = order[i]
        obj_h, obj_w = sizes[idx]
        positions = [(r, c) for r in range(height - obj_h + 1) for c in range(width - obj_w + 1)]
        random.shuffle(positions)
        for r, c in positions:
            box = (r, c, r + obj_h - 1, c + obj_w - 1)
            if not _boxes_clear(box, placed_boxes):
                continue
            placed_boxes.append(box)
            placements[idx] = (r, c)
            if search(i + 1):
                return True
            placed_boxes.pop()
            placements[idx] = None
        return False

    if not search(0):
        return None
    return [(r, c) for r, c in placements if r is not None and c is not None]


def _analyze_teal_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for comp in _components_of_color(grid, TEAL):
        r0, c0, r1, c1 = _bbox(comp)
        local = {(r - r0, c - c0) for r, c in comp}
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        border = _rectangle_border_cells(height, width)
        removed = border - local
        objects.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
                "visible_area": len(comp),
                "removed": removed,
                "missing_count": len(removed),
                "is_square": height == width,
                "has_corner_gap": bool(removed & {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}),
                "is_border_subset": local <= border,
                "rows_covered": _covers_every_row_col(local, height, width),
                "connected": _is_connected(local),
            }
        )
    return objects


def _expected_output(square_count: int) -> np.ndarray:
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for i in range(square_count):
        out[i, i] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each incomplete teal rectangle, count the completed squares, and encode that count on a diagonal.

    Input structure:
    - The grid has a black background with several 4-connected teal objects and a few isolated red distractor pixels.
    - Each teal object is an incomplete rectangle border with 1-3 missing boundary pixels.
    - Every row and column of each object's true bounding box still contains teal, so the box is uniquely recoverable.

    Rule:
    - Ignore all red pixels.
    - For every teal 4-connected object, take its bounding box and fill every black cell inside that box with orange while keeping teal cells unchanged.
    - Treat the completed box as a rectangle, count how many completed rectangles are squares, and output a 5x5 black grid whose first N main-diagonal cells are green.

    Key insight:
    - The square/non-square property is not reliably visible from the damaged teal outline alone; it is determined only after recovering the full bounding box and completing the rectangle.
    """
    g = np.asarray(grid, dtype=int)
    square_count = 0
    for comp in _components_of_color(g, TEAL):
        r0, c0, r1, c1 = _bbox(comp)
        rect = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        rect[rect == BLACK] = ORANGE
        if rect.shape[0] == rect.shape[1]:
            square_count += 1
    return _expected_output(square_count)


def _choose_object_specs() -> list[dict]:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    square_count = random.randint(1, min(4, object_count - 1))

    pair_square, pair_rect = random.choice(PAIR_CHOICES)
    pair_missing = random.randint(1, 3)
    specs = [
        {"height": pair_square[0], "width": pair_square[1], "missing": pair_missing, "corner": True, "kind": "teal"},
        {"height": pair_rect[0], "width": pair_rect[1], "missing": pair_missing, "corner": False, "kind": "teal"},
    ]

    extra_squares = square_count - 1
    extra_rects = object_count - 2 - extra_squares
    for _ in range(extra_squares):
        side = random.choice(SQUARE_SIZES)
        specs.append(
            {"height": side, "width": side, "missing": random.randint(1, 3), "corner": False, "kind": "teal"}
        )
    for _ in range(extra_rects):
        h, w = random.choice(RECT_SIZES)
        if random.choice([False, True]):
            h, w = w, h
        specs.append({"height": h, "width": w, "missing": random.randint(1, 3), "corner": False, "kind": "teal"})

    if not any(spec["missing"] == 3 for spec in specs):
        random.choice(specs)["missing"] = 3
    return specs


def _render_grid(
    grid_shape: tuple[int, int], teal_specs: list[dict], red_count: int
) -> tuple[np.ndarray, list[dict]] | None:
    all_sizes = [(spec["height"], spec["width"]) for spec in teal_specs] + [(1, 1)] * red_count
    placements = _place_boxes(grid_shape[0], grid_shape[1], all_sizes)
    if placements is None:
        return None

    grid = np.zeros(grid_shape, dtype=int)
    placed_specs: list[dict] = []
    for spec, (r0, c0) in zip(teal_specs, placements[: len(teal_specs)]):
        built = _make_incomplete_rectangle(spec["height"], spec["width"], spec["missing"], spec["corner"])
        if built is None:
            return None
        sprite, removed = built
        h, w = sprite.shape
        grid[r0 : r0 + h, c0 : c0 + w] = np.maximum(grid[r0 : r0 + h, c0 : c0 + w], sprite)
        placed_specs.append(
            {
                **spec,
                "top": r0,
                "left": c0,
                "removed": removed,
                "visible_area": int(np.sum(sprite == TEAL)),
                "is_square": spec["height"] == spec["width"],
            }
        )
    for r0, c0 in placements[len(teal_specs) :]:
        grid[r0, c0] = RED
    return grid, placed_specs


def _validate_generated_grid(grid: np.ndarray) -> int:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert set(map(int, np.unique(grid))) <= {BLACK, RED, TEAL}

    red_coords = np.argwhere(grid == RED)
    assert 1 <= len(red_coords) <= 2
    for r, c in red_coords:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                assert int(grid[nr, nc]) == BLACK

    objects = _analyze_teal_objects(grid)
    assert OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]

    for obj in objects:
        assert obj["is_border_subset"]
        assert obj["connected"]
        assert obj["rows_covered"]
        assert 1 <= obj["missing_count"] <= 3

    boxes = [obj["bbox"] for obj in objects]
    for i, box_a in enumerate(boxes):
        for box_b in boxes[i + 1 :]:
            assert _boxes_clear(box_a, [box_b])

    square_count = sum(1 for obj in objects if obj["is_square"])
    nonsquare_count = len(objects) - square_count
    assert 1 <= square_count <= 4
    assert nonsquare_count >= 1
    assert any(obj["has_corner_gap"] for obj in objects)
    assert any(obj["missing_count"] == 3 for obj in objects)

    same_visible_counterexample = False
    for i, obj_a in enumerate(objects):
        for obj_b in objects[i + 1 :]:
            if obj_a["visible_area"] == obj_b["visible_area"] and obj_a["is_square"] != obj_b["is_square"]:
                same_visible_counterexample = True
                break
        if same_visible_counterexample:
            break
    assert same_visible_counterexample

    expected = _expected_output(square_count)
    actual = main(grid)
    assert np.array_equal(actual, expected)
    return square_count


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid_shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
        teal_specs = _choose_object_specs()
        red_count = random.randint(*RED_COUNT_RANGE)
        rendered = _render_grid(grid_shape, teal_specs, red_count)
        if rendered is None:
            continue
        grid, _ = rendered
        try:
            _validate_generated_grid(grid)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
