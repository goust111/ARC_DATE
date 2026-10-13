from __future__ import annotations

import random

import numpy as np


DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
GEN_ATTEMPTS = 400
GRID_HEIGHT_RANGE = (12, 24)
GRID_WIDTH_RANGE = (12, 24)
OBJECT_COUNT_RANGE = (1, 4)
LINE_LENGTH_CAP = 6
MARGIN = 1


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r0 in range(height):
        for c0 in range(width):
            if not mask[r0, c0] or seen[r0, c0]:
                continue

            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(cells, dtype=int))

    return components


def _anchor_info(coords: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    if int(coords.shape[0]) != 3:
        return None

    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    if (r1 - r0, c1 - c0) != (1, 1):
        return None

    cells = {tuple(map(int, cell)) for cell in coords}
    block = {
        (int(r0), int(c0)),
        (int(r0), int(c1)),
        (int(r1), int(c0)),
        (int(r1), int(c1)),
    }
    missing = list(block - cells)
    if len(missing) != 1:
        return None

    mr, mc = map(int, missing[0])
    dr = -1 if mr == r0 else 1
    dc = -1 if mc == c0 else 1
    return cells, (int(mr), int(mc)), (int(dr), int(dc))


def _analyze_color(grid: np.ndarray, color: int) -> tuple[set[tuple[int, int]], tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    components = _find_components(grid == int(color))
    assert components, "each non-background color must appear at least once"

    anchor_cells = None
    missing = None
    step = None
    singles: list[tuple[int, int]] = []

    for comp in components:
        anchor = _anchor_info(comp)
        if anchor is not None:
            assert anchor_cells is None, "expected exactly one anchor per color"
            anchor_cells, missing, step = anchor
            continue

        assert int(comp.shape[0]) == 1, "non-anchor pixels must be isolated singletons"
        singles.append(tuple(map(int, comp[0])))

    assert anchor_cells is not None and missing is not None and step is not None, "missing color anchor"
    assert singles, "expected at least one scattered singleton per color"
    return anchor_cells, missing, step, singles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect scattered same-color pixels onto incomplete 2x2 anchors as diagonal rays.

    Input structure:
    - The background is the dominant color.
    - Each non-background color has one 2x2 anchor block missing exactly one corner, so the color appears as a 3-cell L shape.
    - The remaining cells of that color are isolated single pixels scattered elsewhere in the grid.

    Rule:
    - For each color, locate the missing anchor corner and use that corner's direction as a diagonal step.
    - Count that color's scattered single pixels, erase them to background, and place the same number of pixels starting at the missing corner.
    - The first placed pixel fills the missing corner, and each additional pixel extends one more step along the same diagonal direction.

    Key insight:
    - The missing corner of the 2x2 L anchor simultaneously identifies both the starting cell and the diagonal direction of the reconstructed ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    bg = _dominant_color(g)

    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert colors, "expected at least one anchor color"

    out = g.copy()
    for color in colors:
        _anchor_cells, missing, step, singles = _analyze_color(g, int(color))
        for r, c in singles:
            out[int(r), int(c)] = int(bg)

        for k in range(len(singles)):
            rr = int(missing[0] + step[0] * k)
            cc = int(missing[1] + step[1] * k)
            assert 0 <= rr < height and 0 <= cc < width, "ray must stay in bounds"
            assert int(out[rr, cc]) in (int(bg), int(color)), "rays must not overlap other colors"
            out[rr, cc] = int(color)

    return out


def _build_object(
    top: int, left: int, direction: tuple[int, int], line_length: int
) -> dict[str, object] | None:
    dr, dc = map(int, direction)
    missing = (
        int(top + (0 if dr < 0 else 1)),
        int(left + (0 if dc < 0 else 1)),
    )

    square = {
        (int(top), int(left)),
        (int(top), int(left + 1)),
        (int(top + 1), int(left)),
        (int(top + 1), int(left + 1)),
    }
    anchor_cells = square - {missing}
    line_cells = [
        (int(missing[0] + dr * k), int(missing[1] + dc * k))
        for k in range(int(line_length))
    ]
    occupied = set(anchor_cells) | set(line_cells)
    return {
        "anchor_cells": anchor_cells,
        "missing": missing,
        "step": (int(dr), int(dc)),
        "line_cells": line_cells,
        "occupied": occupied,
    }


def _all_inside(cells: set[tuple[int, int]] | list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    return all(
        MARGIN <= int(r) < height - MARGIN and MARGIN <= int(c) < width - MARGIN
        for r, c in cells
    )


def _mark_reserved(mask: np.ndarray, cells: set[tuple[int, int]] | list[tuple[int, int]]) -> None:
    height, width = mask.shape
    for r, c in cells:
        for rr in range(max(0, int(r) - 1), min(height, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(width, int(c) + 2)):
                mask[int(rr), int(cc)] = True


def _place_objects(
    shape: tuple[int, int], specs: list[dict[str, int | tuple[int, int]]]
) -> list[dict[str, object]] | None:
    height, width = map(int, shape)
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, object]] = []

    for spec in sorted(specs, key=lambda item: int(item["line_length"]), reverse=True):
        line_length = int(spec["line_length"])
        direction = tuple(spec["direction"])  # type: ignore[arg-type]
        candidates: list[dict[str, object]] = []
        for top in range(MARGIN, height - MARGIN - 1):
            for left in range(MARGIN, width - MARGIN - 1):
                obj = _build_object(int(top), int(left), direction, int(line_length))
                assert obj is not None
                occupied = obj["occupied"]
                assert isinstance(occupied, set)
                if not _all_inside(occupied, shape):
                    continue
                if any(reserved[int(r), int(c)] for r, c in occupied):
                    continue
                candidates.append(obj)

        if not candidates:
            return None

        choice = random.choice(candidates)
        placed_obj = {
            "color": int(spec["color"]),
            "line_length": int(line_length),
            **choice,
        }
        occupied = placed_obj["occupied"]
        assert isinstance(occupied, set)
        _mark_reserved(reserved, occupied)
        placed.append(placed_obj)

    return placed


def _free_single_positions(
    shape: tuple[int, int],
    reserved: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    free: list[tuple[int, int]] = []
    for r in range(MARGIN, height - MARGIN):
        for c in range(MARGIN, width - MARGIN):
            if reserved[r, c] or (int(r), int(c)) in forbidden:
                continue
            free.append((int(r), int(c)))
    return free


def _place_singles(
    shape: tuple[int, int], objects: list[dict[str, object]]
) -> dict[int, list[tuple[int, int]]] | None:
    reserved = np.zeros(shape, dtype=bool)
    forbidden = set()
    singles_by_color: dict[int, list[tuple[int, int]]] = {}

    for obj in objects:
        anchor_cells = obj["anchor_cells"]
        occupied = obj["occupied"]
        assert isinstance(anchor_cells, set) and isinstance(occupied, set)
        _mark_reserved(reserved, anchor_cells)
        forbidden |= occupied
        singles_by_color[int(obj["color"])] = []

    requests = [int(obj["color"]) for obj in objects for _ in range(int(obj["line_length"]))]
    random.shuffle(requests)

    for color in requests:
        candidates = _free_single_positions(shape, reserved, forbidden)
        if not candidates:
            return None
        cell = random.choice(candidates)
        singles_by_color[int(color)].append(cell)
        _mark_reserved(reserved, [cell])

    return singles_by_color


def _render_pair(
    shape: tuple[int, int],
    bg: int,
    objects: list[dict[str, object]],
    singles_by_color: dict[int, list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, int(bg), dtype=int)
    out = np.full(shape, int(bg), dtype=int)

    for obj in objects:
        color = int(obj["color"])
        anchor_cells = obj["anchor_cells"]
        line_cells = obj["line_cells"]
        assert isinstance(anchor_cells, set) and isinstance(line_cells, list)
        for r, c in anchor_cells:
            inp[int(r), int(c)] = color
            out[int(r), int(c)] = color
        for r, c in line_cells:
            out[int(r), int(c)] = color

    for color, cells in singles_by_color.items():
        for r, c in cells:
            inp[int(r), int(c)] = int(color)

    return inp, out


def _generate_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    height_lo = GRID_HEIGHT_RANGE[0] + max(0, object_count - 2)
    width_lo = GRID_WIDTH_RANGE[0] + max(0, object_count - 2)
    height = random.randint(height_lo, GRID_HEIGHT_RANGE[1])
    width = random.randint(width_lo, GRID_WIDTH_RANGE[1])
    shape = (int(height), int(width))

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    colors = random.sample(palette, object_count)
    directions = random.sample(DIRS, object_count)
    max_line = min(LINE_LENGTH_CAP, max(1, min(shape) // 2 - 1))
    specs = [
        {
            "color": int(color),
            "direction": tuple(direction),
            "line_length": random.randint(1, max_line),
        }
        for color, direction in zip(colors, directions)
    ]

    objects = _place_objects(shape, specs)
    if objects is None:
        return None

    singles_by_color = _place_singles(shape, objects)
    if singles_by_color is None:
        return None

    inp, out = _render_pair(shape, int(bg), objects, singles_by_color)
    if _dominant_color(inp) != int(bg) or _dominant_color(out) != int(bg):
        return None

    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
