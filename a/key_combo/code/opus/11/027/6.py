from __future__ import annotations

import random
from collections import deque

import numpy as np


PALETTE = (1, 2, 3, 4, 5, 6, 7)
BACKGROUND = 0
GRID_MIN = 18
GRID_MAX = 24
MAX_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_objects(grid: np.ndarray) -> list[dict]:
    """Return non-background 4-connected monochrome components."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": bbox,
                    "area": len(coords),
                }
            )
    return objects


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _is_complete_filled_square(grid: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 2:
        return False
    return bool(np.all(_crop(grid, obj["bbox"]) == int(obj["color"])))


def _selected_objects(grid: np.ndarray) -> list[dict]:
    selected = [obj for obj in _find_objects(grid) if _is_complete_filled_square(grid, obj)]
    return sorted(selected, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))


def _render_output(selected: list[dict]) -> np.ndarray:
    width = len(selected) * 3 - 1
    out = np.zeros((2, width), dtype=int)
    for idx, obj in enumerate(selected):
        c0 = idx * 3
        out[:, c0 : c0 + 2] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter complete filled squares, normalize them to 2x2 blocks, and pack them into one ordered row.

    Input structure:
    - A black background contains 5-7 separated foreground objects.
    - Each object is monochrome and occupies a distinct 4-connected component.
    - Some objects are complete filled squares with square bounding boxes; others are distractors such
      as lines, L-shapes, non-square rectangles, or square-bbox fragments with missing cells.

    Rule:
    - Detect every foreground object whose bounding box is a square of side at least 2 and whose entire
      bounding box is filled by that object's color.
    - Discard every object that fails this complete-square test.
    - Sort the kept objects by the top-left corner of their original bounding boxes, first by row and
      then by column.
    - Replace each kept object with a solid 2x2 block of the same color.
    - Pack those 2x2 blocks into a single row with exactly one black column between adjacent blocks.

    Key insight:
    - The deciding feature is structural completeness of the square bbox, not color, area, or merely
      having a square outline; the spatial order of the surviving objects determines the packed output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    selected = _selected_objects(g)
    assert selected, "expected at least one complete filled square"
    return _render_output(selected)


def _full_square_mask(size: int) -> np.ndarray:
    return np.ones((size, size), dtype=bool)


def _notched_square_mask(size: int, rng: random.Random) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    if size == 2:
        mask[rng.randrange(2), rng.randrange(2)] = False
        return mask

    style = rng.choice(("center", "corner", "edge_pair"))
    if style == "center":
        mask[size // 2, size // 2] = False
    elif style == "corner":
        mask[rng.choice((0, size - 1)), rng.choice((0, size - 1))] = False
    else:
        row = rng.choice((0, size - 1))
        col = rng.randrange(1, size - 1)
        mask[row, col] = False
        if size >= 4 and rng.random() < 0.5:
            mask[size - 1 - row, col] = False
    return mask


def _line_mask(length: int, horizontal: bool) -> np.ndarray:
    return np.ones((1, length), dtype=bool) if horizontal else np.ones((length, 1), dtype=bool)


def _rectangle_mask(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def _l_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    return mask


def _mask_to_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _make_target_specs(rng: random.Random, count: int) -> list[dict]:
    while True:
        sizes = [rng.randint(2, 4) for _ in range(count)]
        if len(set(sizes)) >= 2:
            break
    return [
        {
            "kind": "target_square",
            "is_target": True,
            "size": size,
            "mask": _full_square_mask(size),
            "cells": _mask_to_cells(_full_square_mask(size)),
        }
        for size in sizes
    ]


def _make_distractor_specs(rng: random.Random, count: int) -> list[dict]:
    notch_size = rng.randint(2, 4)
    notch_mask = _notched_square_mask(notch_size, rng)
    specs = [
        {
            "kind": "notched_square",
            "is_target": False,
            "size": notch_size,
            "mask": notch_mask,
            "cells": _mask_to_cells(notch_mask),
        }
    ]

    while len(specs) < count:
        kind = rng.choice(("line", "rect", "l_shape"))
        if kind == "line":
            length = rng.randint(2, 5)
            horizontal = rng.random() < 0.5
            mask = _line_mask(length, horizontal)
        elif kind == "rect":
            height, width = rng.choice(((2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)))
            mask = _rectangle_mask(height, width)
        else:
            height = rng.randint(2, 4)
            width = rng.randint(2, 4)
            if height == 2 and width == 2:
                width = 3
            mask = _l_mask(height, width)
        specs.append({"kind": kind, "is_target": False, "mask": mask, "cells": _mask_to_cells(mask)})
    return specs


def _can_place(
    occupied: np.ndarray,
    shape: tuple[int, int],
    cells: list[tuple[int, int]],
    top: int,
    left: int,
) -> bool:
    h, w = shape
    absolute = [(top + r, left + c) for r, c in cells]
    for r, c in absolute:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _place_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in _mask_to_cells(mask):
        grid[top + r, left + c] = color
        occupied[top + r, left + c] = True


def _object_order_signature(objs: list[dict]) -> tuple[int, ...]:
    return tuple(int(obj["color"]) for obj in sorted(objs, key=lambda obj: (obj["bbox"][0], obj["bbox"][1])))


def _size_order_signature(objs: list[dict]) -> tuple[int, ...]:
    return tuple(obj["bbox"][2] - obj["bbox"][0] + 1 for obj in sorted(objs, key=lambda obj: (obj["bbox"][0], obj["bbox"][1])))


def _validate_generated(grid: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    objects = _find_objects(grid)
    selected = [obj for obj in objects if _is_complete_filled_square(grid, obj)]
    rejected = [obj for obj in objects if not _is_complete_filled_square(grid, obj)]

    assert 5 <= len(objects) <= 7
    assert 3 <= len(selected) <= 4
    assert len(rejected) >= 2
    assert len({obj["color"] for obj in objects}) == len(objects)

    square_sides = [obj["bbox"][2] - obj["bbox"][0] + 1 for obj in selected]
    assert len(set(square_sides)) >= 2
    assert any(
        (obj["bbox"][2] - obj["bbox"][0]) == (obj["bbox"][3] - obj["bbox"][1]) and
        (obj["bbox"][2] - obj["bbox"][0] + 1) >= 2
        for obj in rejected
    ), "need at least one square-bbox distractor to make completeness necessary"
    assert any(
        (obj["bbox"][2] - obj["bbox"][0]) != (obj["bbox"][3] - obj["bbox"][1])
        for obj in rejected
    ), "need at least one clearly non-square distractor"

    ordered_selected = sorted(selected, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    color_order = tuple(int(obj["color"]) for obj in ordered_selected)
    assert color_order != tuple(sorted(color_order))
    assert color_order != tuple(sorted(color_order, reverse=True))

    size_order = _size_order_signature(ordered_selected)
    assert size_order != tuple(sorted(size_order))
    assert size_order != tuple(sorted(size_order, reverse=True))

    naive_square_bbox = sorted(
        [
            obj for obj in objects
            if (obj["bbox"][2] - obj["bbox"][0]) == (obj["bbox"][3] - obj["bbox"][1]) and
            (obj["bbox"][2] - obj["bbox"][0] + 1) >= 2
        ],
        key=lambda obj: (obj["bbox"][0], obj["bbox"][1]),
    )
    assert _object_order_signature(naive_square_bbox) != _object_order_signature(ordered_selected)

    out = main(grid)
    assert np.array_equal(out, _render_output(ordered_selected))
    assert out.shape == (2, len(ordered_selected) * 3 - 1)
    return objects, ordered_selected, rejected


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_MIN, GRID_MAX)
    grid_w = rng.randint(GRID_MIN, GRID_MAX)

    square_count = rng.choice((3, 4))
    distractor_count = 2 if square_count == 4 else rng.choice((2, 3))
    total_count = square_count + distractor_count
    colors = list(rng.sample(PALETTE, total_count))

    specs = _make_target_specs(rng, square_count) + _make_distractor_specs(rng, distractor_count)
    for spec, color in zip(specs, colors):
        spec["color"] = color

    # Place larger masks first for easier packing, but keep generation varied by shuffling ties.
    rng.shuffle(specs)
    specs.sort(key=lambda spec: (max(spec["mask"].shape), int(np.sum(spec["mask"]))), reverse=True)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in specs:
        mask = spec["mask"]
        mh, mw = mask.shape
        positions = [(r, c) for r in range(grid_h - mh + 1) for c in range(grid_w - mw + 1)]
        rng.shuffle(positions)
        cells = spec["cells"]

        placed = False
        for top, left in positions:
            if _can_place(occupied, grid.shape, cells, top, left):
                _place_mask(grid, occupied, mask, top, left, int(spec["color"]))
                placed = True
                break
        if not placed:
            return None

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
