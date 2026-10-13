import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7

BODY_COLORS = (BLUE, 3, 4, 5, 6, 8, 9)
GENERATE_ATTEMPTS = 500
PLACE_ATTEMPTS = 200
GRID_SIZE_RANGE = (14, 30)
OBJECT_COUNT_RANGE = (2, 5)
RECT_SIZE_RANGE = (3, 8)
PADDING = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not bool(mask[sr, sc]):
                continue
            seen[sr, sc] = True
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _dist2_to_center(cells: list[tuple[int, int]], center_r: float, center_c: float) -> float:
    arr = np.asarray(cells, dtype=float)
    dr = float(arr[:, 0].mean() - center_r)
    dc = float(arr[:, 1].mean() - center_c)
    return dr * dr + dc * dc


def _split_object(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch != BLACK), "each foreground component must be a solid rectangle"

    body_values = set(int(v) for v in np.unique(patch) if int(v) != RED)
    assert len(body_values) == 1, "each marked rectangle must have one body color"
    body_color = body_values.pop()

    red = patch == RED
    assert np.any(red), "each rectangle must contain one red cut line"

    full_rows = [i for i in range(patch.shape[0]) if bool(np.all(red[i, :]))]
    full_cols = [j for j in range(patch.shape[1]) if bool(np.all(red[:, j]))]
    horizontal = (
        len(full_rows) == 1
        and int(np.count_nonzero(red)) == patch.shape[1]
        and 0 < full_rows[0] < patch.shape[0] - 1
    )
    vertical = (
        len(full_cols) == 1
        and int(np.count_nonzero(red)) == patch.shape[0]
        and 0 < full_cols[0] < patch.shape[1] - 1
    )
    assert horizontal ^ vertical, "red cut line must be one complete internal row or column"

    if horizontal:
        line = full_rows[0]
        first = [(r0 + rr - 1, c0 + cc) for rr in range(line) for cc in range(patch.shape[1])]
        second = [
            (r0 + rr + 1, c0 + cc)
            for rr in range(line + 1, patch.shape[0])
            for cc in range(patch.shape[1])
        ]
    else:
        line = full_cols[0]
        first = [(r0 + rr, c0 + cc - 1) for rr in range(patch.shape[0]) for cc in range(line)]
        second = [
            (r0 + rr, c0 + cc + 1)
            for rr in range(patch.shape[0])
            for cc in range(line + 1, patch.shape[1])
        ]

    h, w = grid.shape
    for r, c in first + second:
        assert 0 <= r < h and 0 <= c < w, "translated halves must remain in bounds"
    return first, second, body_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each red-marked rectangle apart and recolor the half farther from the grid center orange.

    Input structure:
    - The background is black.
    - Each foreground object is a solid 4-connected rectangle with one non-red body color.
    - Inside every rectangle, a complete red row or complete red column marks the cut line and leaves two nonempty halves.

    Rule:
    - For each rectangle, erase the whole object including the red cut line.
    - If the cut line is horizontal, move the upper half one cell up and the lower half one cell down.
    - If the cut line is vertical, move the left half one cell left and the right half one cell right.
    - After the move, the half whose centroid is closer to the grid center keeps the original body color, and the farther half becomes orange.

    Key insight:
    - The red line is a structural crack: its orientation chooses the split direction, while center proximity after separation chooses which half keeps the source color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape

    comps = _components(g != BLACK)
    assert comps, "expected at least one marked rectangle"

    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    claims: dict[tuple[int, int], int] = {}

    for coords in comps:
        first, second, body_color = _split_object(g, coords)
        d_first = _dist2_to_center(first, center_r, center_c)
        d_second = _dist2_to_center(second, center_r, center_c)
        assert abs(d_first - d_second) > 1e-9, "center-near half must be unambiguous"

        first_color, second_color = (body_color, ORANGE) if d_first < d_second else (ORANGE, body_color)
        for cells, color in ((first, first_color), (second, second_color)):
            for cell in cells:
                assert cell not in claims, "translated halves must not overlap"
                claims[cell] = color

    out = np.full(g.shape, BLACK, dtype=int)
    for (r, c), color in claims.items():
        out[r, c] = color
    return out


def _candidate_cells(
    top: int,
    left: int,
    height: int,
    width: int,
    direction: str,
    line: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    input_cells = [(top + rr, left + cc) for rr in range(height) for cc in range(width)]
    output_cells: list[tuple[int, int]] = []
    if direction == "horizontal":
        output_cells.extend((top + rr - 1, left + cc) for rr in range(line) for cc in range(width))
        output_cells.extend((top + rr + 1, left + cc) for rr in range(line + 1, height) for cc in range(width))
    else:
        output_cells.extend((top + rr, left + cc - 1) for rr in range(height) for cc in range(line))
        output_cells.extend((top + rr, left + cc + 1) for rr in range(height) for cc in range(line + 1, width))
    return input_cells, output_cells


def _free_with_padding(reserved: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        r0 = max(0, r - PADDING)
        r1 = min(h, r + PADDING + 1)
        c0 = max(0, c - PADDING)
        c1 = min(w, c + PADDING + 1)
        if bool(np.any(reserved[r0:r1, c0:c1])):
            return False
    return True


def _mark_with_padding(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        r0 = max(0, r - PADDING)
        r1 = min(h, r + PADDING + 1)
        c0 = max(0, c - PADDING)
        c1 = min(w, c + PADDING + 1)
        reserved[r0:r1, c0:c1] = True


def _paint_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    direction: str,
) -> bool:
    h, w = grid.shape
    max_h = min(RECT_SIZE_RANGE[1], h - 3)
    max_w = min(RECT_SIZE_RANGE[1], w - 3)
    if max_h < RECT_SIZE_RANGE[0] or max_w < RECT_SIZE_RANGE[0]:
        return False

    for _ in range(PLACE_ATTEMPTS):
        height = random.randint(RECT_SIZE_RANGE[0], max_h)
        width = random.randint(RECT_SIZE_RANGE[0], max_w)
        if direction == "horizontal":
            line = random.randint(1, height - 2)
        else:
            line = random.randint(1, width - 2)

        top = random.randint(1, h - height - 1)
        left = random.randint(1, w - width - 1)
        input_cells, output_cells = _candidate_cells(top, left, height, width, direction, line)
        footprint = input_cells + output_cells
        if not _free_with_padding(reserved, footprint):
            continue

        grid[top : top + height, left : left + width] = color
        if direction == "horizontal":
            grid[top + line, left : left + width] = RED
        else:
            grid[top : top + height, left + line] = RED
        _mark_with_padding(reserved, footprint)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    count = random.randint(OBJECT_COUNT_RANGE[0], min(OBJECT_COUNT_RANGE[1], len(BODY_COLORS)))

    colors = random.sample(BODY_COLORS, count)
    directions = ["horizontal", "vertical"]
    directions.extend(random.choice(("horizontal", "vertical")) for _ in range(count - 2))
    random.shuffle(directions)

    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for color, direction in zip(colors, directions):
        if not _paint_object(grid, reserved, color, direction):
            return None

    try:
        out = main(grid)
    except (AssertionError, Exception):
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_ATTEMPTS} attempts")
