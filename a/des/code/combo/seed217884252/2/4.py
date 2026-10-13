from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    rectangles: list[dict[str, int]] = []
    for coords in _components(grid == GREY):
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(sub == GREY)), "grey objects must be solid rectangles"
        rectangles.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
    rectangles.sort(key=lambda rect: (rect["r0"], rect["c0"]))
    assert rectangles, "expected at least one grey rectangle"
    return rectangles


def _nearest_rectangle_side(
    rectangles: list[dict[str, int]], row: int, col: int
) -> tuple[int, str] | None:
    candidates: list[tuple[int, int, str]] = []
    for idx, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        if c0 <= col <= c1:
            if row < r0:
                candidates.append((int(r0 - row), int(idx), "top"))
            elif row > r1:
                candidates.append((int(row - r1), int(idx), "bottom"))
        if r0 <= row <= r1:
            if col < c0:
                candidates.append((int(c0 - col), int(idx), "left"))
            elif col > c1:
                candidates.append((int(col - c1), int(idx), "right"))

    if not candidates:
        return None

    best_dist = min(dist for dist, _idx, _side in candidates)
    best = [(idx, side) for dist, idx, side in candidates if dist == best_dist]
    if len(best) != 1:
        return None
    return best[0]


def _paint_side_bar(out: np.ndarray, rect: dict[str, int], side: str, color: int) -> None:
    r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
    if side == "top":
        out[r0 - 1, c0 : c1 + 1] = int(color)
    elif side == "bottom":
        out[r1 + 1, c0 : c1 + 1] = int(color)
    elif side == "left":
        out[r0 : r1 + 1, c0 - 1] = int(color)
    else:
        assert side == "right"
        out[r0 : r1 + 1, c1 + 1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project singleton markers onto the nearest aligned grey rectangle side.

    Each non-grey, non-black pixel is a single-cell marker. It selects the nearest
    grey rectangle that shares its row or column, then paints a one-cell-thick bar
    along the corresponding outside side of that rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rectangles = _extract_rectangles(g)
    markers = [tuple(map(int, cell)) for cell in np.argwhere((g != BLACK) & (g != GREY))]
    out = g.copy()

    for r, c in markers:
        out[r, c] = BLACK

    for r, c in markers:
        target = _nearest_rectangle_side(rectangles, int(r), int(c))
        assert target is not None, "each marker must align to one nearest rectangle"
        rect_idx, side = target
        _paint_side_bar(out, rectangles[rect_idx], side, int(g[r, c]))

    return out


GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
RECT_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (2, 5)
RECT_W_RANGE = (2, 6)
EDGE_MARGIN = 4
MIN_MARKER_DISTANCE = 2
MAX_MARKER_DISTANCE = 4
MAX_GENERATION_ATTEMPTS = 400


def _side_bar_cells(rect: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if side == "top":
        return {(int(r0 - 1), int(c)) for c in range(c0, c1 + 1)}
    if side == "bottom":
        return {(int(r1 + 1), int(c)) for c in range(c0, c1 + 1)}
    if side == "left":
        return {(int(r), int(c0 - 1)) for r in range(r0, r1 + 1)}
    assert side == "right"
    return {(int(r), int(c1 + 1)) for r in range(r0, r1 + 1)}


def _footprint_cells(rect: tuple[int, int, int, int], sides: tuple[str, ...]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    cells = {(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    for side in sides:
        cells |= _side_bar_cells(rect, side)
    return cells


def _expanded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _sample_specs(count: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for _ in range(count):
        height = random.randint(*RECT_H_RANGE)
        width = random.randint(*RECT_W_RANGE)
        side_count = random.randint(1, 4)
        sides = tuple(sorted(random.sample(SIDES, side_count), key=SIDES.index))
        colors = random.sample(MARKER_COLORS, side_count)
        side_colors = {side: int(color) for side, color in zip(sides, colors)}
        footprint_area = int(height * width)
        footprint_area += width * int("top" in sides)
        footprint_area += width * int("bottom" in sides)
        footprint_area += height * int("left" in sides)
        footprint_area += height * int("right" in sides)
        specs.append(
            {
                "height": int(height),
                "width": int(width),
                "sides": sides,
                "side_colors": side_colors,
                "score": int(footprint_area),
            }
        )
    specs.sort(key=lambda spec: (-int(spec["score"]), -int(spec["height"]), -int(spec["width"])))
    return specs


def _place_rectangles(height: int, width: int, specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    blocked: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    for spec in specs:
        rect_h = int(spec["height"])
        rect_w = int(spec["width"])
        if height - EDGE_MARGIN - rect_h < EDGE_MARGIN or width - EDGE_MARGIN - rect_w < EDGE_MARGIN:
            return None

        positions = [
            (r0, c0)
            for r0 in range(EDGE_MARGIN, height - EDGE_MARGIN - rect_h + 1)
            for c0 in range(EDGE_MARGIN, width - EDGE_MARGIN - rect_w + 1)
        ]
        random.shuffle(positions)

        for r0, c0 in positions:
            rect = (int(r0), int(c0), int(r0 + rect_h - 1), int(c0 + rect_w - 1))
            footprint = _footprint_cells(rect, spec["sides"])
            if any(cell in blocked for cell in footprint):
                continue
            placed.append(
                {
                    "bbox": rect,
                    "sides": spec["sides"],
                    "side_colors": dict(spec["side_colors"]),
                }
            )
            blocked |= _expanded(footprint, height, width)
            break
        else:
            return None

    placed.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return placed


def _build_output(height: int, width: int, objects: list[dict[str, object]]) -> np.ndarray:
    out = np.full((height, width), BLACK, dtype=int)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = GREY
        for side in obj["sides"]:
            color = int(obj["side_colors"][side])
            rect = {"r0": r0, "c0": c0, "r1": r1, "c1": c1}
            _paint_side_bar(out, rect, side, color)
    return out


def _touches_marker(marker_cells: set[tuple[int, int]], row: int, col: int) -> bool:
    return any(max(abs(row - rr), abs(col - cc)) <= 1 for rr, cc in marker_cells)


def _marker_candidates(
    obj_idx: int,
    side: str,
    objects: list[dict[str, object]],
    output: np.ndarray,
    marker_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = objects[obj_idx]["bbox"]
    rectangles = [
        {"r0": int(item["bbox"][0]), "c0": int(item["bbox"][1]), "r1": int(item["bbox"][2]), "c1": int(item["bbox"][3])}
        for item in objects
    ]
    candidates: list[tuple[int, int]] = []

    distances = list(range(MIN_MARKER_DISTANCE, MAX_MARKER_DISTANCE + 1))
    random.shuffle(distances)
    for dist in distances:
        if side == "top":
            row = int(r0 - dist)
            line = list(range(c0, c1 + 1))
            random.shuffle(line)
            points = [(row, int(col)) for col in line]
        elif side == "bottom":
            row = int(r1 + dist)
            line = list(range(c0, c1 + 1))
            random.shuffle(line)
            points = [(row, int(col)) for col in line]
        elif side == "left":
            col = int(c0 - dist)
            line = list(range(r0, r1 + 1))
            random.shuffle(line)
            points = [(int(row), col) for row in line]
        else:
            assert side == "right"
            col = int(c1 + dist)
            line = list(range(r0, r1 + 1))
            random.shuffle(line)
            points = [(int(row), col) for row in line]

        for row, col in points:
            if output[row, col] != BLACK:
                continue
            if (row, col) in marker_cells or _touches_marker(marker_cells, row, col):
                continue
            if _nearest_rectangle_side(rectangles, int(row), int(col)) == (int(obj_idx), side):
                candidates.append((int(row), int(col)))

    return candidates


def _build_input(objects: list[dict[str, object]], output: np.ndarray) -> np.ndarray | None:
    inp = np.full_like(output, BLACK)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = GREY

    marker_cells: set[tuple[int, int]] = set()
    for obj_idx, obj in enumerate(objects):
        for side in obj["sides"]:
            choices = _marker_candidates(obj_idx, side, objects, output, marker_cells)
            if not choices:
                return None
            row, col = random.choice(choices)
            inp[row, col] = int(obj["side_colors"][side])
            marker_cells.add((int(row), int(col)))

    return inp


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    rect_count = random.randint(*RECT_COUNT_RANGE)

    specs = _sample_specs(rect_count)
    objects = _place_rectangles(height, width, specs)
    if objects is None:
        return None

    output = _build_output(height, width, objects)
    inp = _build_input(objects, output)
    if inp is None:
        return None
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
