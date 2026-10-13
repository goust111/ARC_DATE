from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_SIZE = 30
GENERATION_ATTEMPTS = 600
PLACE_ATTEMPTS = 250


@dataclass(frozen=True)
class Arrow:
    direction: str
    vector: tuple[int, int]
    length: int
    tail: tuple[int, int]
    front: tuple[int, int]
    cells: frozenset[tuple[int, int]]


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 8:
        neighbors = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        )
    else:
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            seen[sr, sc] = True
            stack = [(sr, sc)]
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(coords, dtype=int)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def _match_arrow(coords: np.ndarray) -> Arrow | None:
    cells = frozenset((int(r), int(c)) for r, c in coords)
    r0, c0, r1, c1 = _bbox(coords)

    if r1 - r0 == 2:
        row = r0 + 1
        length = c1 - c0
        if length >= 2:
            right = frozenset(
                [(row, c) for c in range(c0, c1)] + [(r0, c1), (r1, c1)]
            )
            if cells == right:
                return Arrow("R", (0, 1), length, (row, c0), (row, c1), cells)

            left = frozenset(
                [(row, c) for c in range(c0 + 1, c1 + 1)] + [(r0, c0), (r1, c0)]
            )
            if cells == left:
                return Arrow("L", (0, -1), length, (row, c1), (row, c0), cells)

    if c1 - c0 == 2:
        col = c0 + 1
        length = r1 - r0
        if length >= 2:
            down = frozenset(
                [(r, col) for r in range(r0, r1)] + [(r1, c0), (r1, c1)]
            )
            if cells == down:
                return Arrow("D", (1, 0), length, (r0, col), (r1, col), cells)

            up = frozenset(
                [(r, col) for r in range(r0 + 1, r1 + 1)] + [(r0, c0), (r0, c1)]
            )
            if cells == up:
                return Arrow("U", (-1, 0), length, (r1, col), (r0, col), cells)

    return None


def _find_arrows(g: np.ndarray) -> list[Arrow]:
    arrows: list[Arrow] = []
    for comp in _components(g == GREY, connectivity=8):
        arrow = _match_arrow(comp)
        assert arrow is not None, "every grey component must be a valid arrow"
        arrows.append(arrow)
    arrows.sort(key=lambda a: (a.tail[0], a.tail[1], a.direction))
    assert arrows, "expected at least one grey arrow"
    return arrows


def _flood_same_color(g: np.ndarray, start: tuple[int, int], color: int) -> list[tuple[int, int]]:
    h, w = g.shape
    sr, sc = start
    seen = {(sr, sc)}
    stack = [(sr, sc)]
    cells: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return cells


def _source_block(g: np.ndarray, arrow: Arrow) -> tuple[tuple[int, int, int, int], int]:
    h, w = g.shape
    dr, dc = arrow.vector
    tail_r, tail_c = arrow.tail
    sr, sc = tail_r - dr, tail_c - dc
    assert 0 <= sr < h and 0 <= sc < w, "arrow tail must touch a source block"
    color = int(g[sr, sc])
    assert color not in (BLACK, GREY), "source block must be colored"

    cells = _flood_same_color(g, (sr, sc), color)
    r0, c0, r1, c1 = _bbox(cells)
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    assert 1 <= bh <= 3 and 1 <= bw <= 3, "source block must be a 1x1 to 3x3 rectangle"
    assert len(cells) == bh * bw, "source block must be solid"
    assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color)), "source block must be one color"

    if arrow.direction == "R":
        assert c1 == tail_c - 1 and r0 <= tail_r <= r1
    elif arrow.direction == "L":
        assert c0 == tail_c + 1 and r0 <= tail_r <= r1
    elif arrow.direction == "D":
        assert r1 == tail_r - 1 and c0 <= tail_c <= c1
    else:
        assert r0 == tail_r + 1 and c0 <= tail_c <= c1
    return (r0, c0, r1, c1), color


def _copy_bboxes(arrow: Arrow, block_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = block_bbox
    bh = r1 - r0 + 1
    bw = c1 - c0 + 1
    fr, fc = arrow.front
    boxes: list[tuple[int, int, int, int]] = []

    for i in range(arrow.length):
        if arrow.direction == "R":
            left = fc + 1 + i * (bw + 1)
            boxes.append((r0, left, r1, left + bw - 1))
        elif arrow.direction == "L":
            right = fc - 1 - i * (bw + 1)
            boxes.append((r0, right - bw + 1, r1, right))
        elif arrow.direction == "D":
            top = fr + 1 + i * (bh + 1)
            boxes.append((top, c0, top + bh - 1, c1))
        else:
            bottom = fr - 1 - i * (bh + 1)
            boxes.append((bottom - bh + 1, c0, bottom, c1))
    return boxes


def _paint_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use grey arrows as instructions for copying the colored blocks attached to their tails.

    Input structure:
    - The background is black.
    - Grey pixels form one or more separated arrows, each with a horizontal or vertical shaft of length at least two and a two-pixel V tip.
    - The tail end of each arrow is directly adjacent to a solid 1x1 to 3x3 rectangle of one non-black, non-grey color.

    Rule:
    - For each arrow, count only the grey shaft pixels to get the number of copies.
    - Remove the grey shaft and V-tip pixels from the output.
    - Starting just beyond the arrow tip, place that many copies of the tail rectangle in the arrow direction, with one black cell of spacing between consecutive copies.
    - Keep every original colored rectangle in its input position.

    Key insight:
    - The arrow is both a direction marker and a counter; the object being copied is the adjacent colored rectangle at the arrow tail.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    arrows = _find_arrows(g)
    out = g.copy()
    plans: list[tuple[int, list[tuple[int, int, int, int]]]] = []

    for arrow in arrows:
        for r, c in arrow.cells:
            out[r, c] = BLACK
        block_bbox, color = _source_block(g, arrow)
        boxes = _copy_bboxes(arrow, block_bbox)
        for r0, c0, r1, c1 in boxes:
            assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w, "copies must fit within the grid"
        plans.append((color, boxes))

    for color, boxes in plans:
        for box in boxes:
            _paint_box(out, box, color)

    return out


def _arrow_cells(direction: str, tail: tuple[int, int], length: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    r, c = tail
    if direction == "R":
        front = (r, c + length)
        cells = {(r, c + i) for i in range(length)}
        cells.update({(r - 1, c + length), (r + 1, c + length)})
    elif direction == "L":
        front = (r, c - length)
        cells = {(r, c - i) for i in range(length)}
        cells.update({(r - 1, c - length), (r + 1, c - length)})
    elif direction == "D":
        front = (r + length, c)
        cells = {(r + i, c) for i in range(length)}
        cells.update({(r + length, c - 1), (r + length, c + 1)})
    else:
        front = (r - length, c)
        cells = {(r - i, c) for i in range(length)}
        cells.update({(r - length, c - 1), (r - length, c + 1)})
    return cells, front


def _object_layout(
    direction: str,
    tail: tuple[int, int],
    length: int,
    block_h: int,
    block_w: int,
    align: int,
) -> dict | None:
    arrow_cells, front = _arrow_cells(direction, tail, length)
    tr, tc = tail

    if direction == "R":
        block = (tr - align, tc - block_w, tr - align + block_h - 1, tc - 1)
    elif direction == "L":
        block = (tr - align, tc + 1, tr - align + block_h - 1, tc + block_w)
    elif direction == "D":
        block = (tr - block_h, tc - align, tr - 1, tc - align + block_w - 1)
    else:
        block = (tr + 1, tc - align, tr + block_h, tc - align + block_w - 1)

    arrow = Arrow(direction, {"R": (0, 1), "L": (0, -1), "D": (1, 0), "U": (-1, 0)}[direction],
                  length, tail, front, frozenset(arrow_cells))
    copies = _copy_bboxes(arrow, block)
    return {"arrow": arrow_cells, "block": block, "copies": copies}


def _all_cells_for_boxes(boxes: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, r1, c1 in boxes:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((r, c))
    return cells


def _in_bounds(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _reserve(cells: set[tuple[int, int]], reserved: np.ndarray, pad: int = 1) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                reserved[rr, cc] = True


def _free(cells: set[tuple[int, int]], reserved: np.ndarray) -> bool:
    return all(not bool(reserved[r, c]) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    object_count = random.choices([1, 2, 3, 4], weights=[3, 4, 2, 1], k=1)[0]
    min_side = {1: 10, 2: 16, 3: 22, 4: 26}[object_count]
    h = random.randint(min_side, MAX_SIZE)
    w = random.randint(min_side, MAX_SIZE)
    inp = np.full((h, w), BLACK, dtype=int)
    expected = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    colors = random.sample(COLOR_POOL, object_count)
    directions = ["R", "L", "D", "U"]
    random.shuffle(directions)

    for idx in range(object_count):
        placed = False
        direction = directions[idx % len(directions)]
        for _ in range(PLACE_ATTEMPTS):
            length = random.randint(2, 5)
            block_h = random.randint(1, 3)
            block_w = random.randint(1, 3)
            align = random.randint(0, block_h - 1) if direction in ("R", "L") else random.randint(0, block_w - 1)
            tail = (random.randrange(h), random.randrange(w))
            layout = _object_layout(direction, tail, length, block_h, block_w, align)
            if layout is None:
                continue

            block_cells = _all_cells_for_boxes([layout["block"]])
            copy_cells = _all_cells_for_boxes(layout["copies"])
            input_cells = set(layout["arrow"]) | block_cells
            footprint = input_cells | copy_cells

            if not _in_bounds(footprint, h, w):
                continue
            if not _free(footprint, reserved):
                continue

            color = colors[idx]
            for r, c in layout["arrow"]:
                inp[r, c] = GREY
            for r, c in block_cells:
                inp[r, c] = color
                expected[r, c] = color
            for box in layout["copies"]:
                _paint_box(expected, box, color)

            _reserve(footprint, reserved, pad=1)
            placed = True
            break

        if not placed:
            return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
