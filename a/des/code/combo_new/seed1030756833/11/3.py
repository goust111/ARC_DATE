import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue

        perimeter = 2 * height + 2 * width - 4
        if len(comp) != perimeter:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == GREY):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == GREY):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == GREY):
            continue
        boxes.append((r0, c0, r1, c1))
    boxes.sort()
    return boxes


def _grow_inside_frame(out: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    seed = out[r0 + 1, c0 + 1 : c1].copy()
    colors = {int(v) for v in seed if int(v) not in (BLACK, GREY)}
    growth_rows = len(colors)
    assert r0 + 1 + growth_rows < r1, "container interior is too short for the seed diversity"

    for i in range(growth_rows):
        row = seed if i % 2 == 0 else seed[::-1]
        out[r0 + 2 + i, c0 + 1 : c1] = row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow mirror-alternating stripe rows below each framed container's seed row.

    Input structure:
    - The background is black.
    - One or more separated rectangular containers have one-pixel grey borders.
    - The top interior row of each container is a seed row containing colored pixels
      and possible black gaps; all lower interior rows are black.

    Rule:
    - For each container, read the full seed row sequence, including black cells.
    - Count the distinct non-black colors in that seed row.
    - Starting immediately below the seed row, draw that many new rows, alternating
      between the original seed sequence and its horizontal mirror.
    - Leave the grey borders and all outside background cells unchanged.

    Key insight: The stripe height is encoded by the number of different colors
    in the seed row, while the row pattern alternates between the seed and its mirror.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    boxes = _frame_bboxes(g)
    assert boxes, "expected at least one grey rectangular container"
    for box in boxes:
        _grow_inside_frame(out, box)
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int) -> tuple[int, int, int, int]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    return r0, c0, r1, c1


def _sample_seed(inner_width: int) -> tuple[np.ndarray, int]:
    max_unique = min(4, inner_width, len(COLOR_POOL))
    unique_count = random.randint(1, max_unique)
    colors = random.sample(COLOR_POOL, unique_count)

    for _ in range(80):
        seed = np.zeros(inner_width, dtype=int)
        positions = random.sample(range(inner_width), unique_count)
        shuffled = colors[:]
        random.shuffle(shuffled)
        for pos, color in zip(positions, shuffled):
            seed[pos] = int(color)
        for c in range(inner_width):
            if seed[c] == BLACK and random.random() < 0.45:
                seed[c] = int(random.choice(colors))

        visible_colors = {int(v) for v in seed if int(v) != BLACK}
        if len(visible_colors) != unique_count:
            continue
        if unique_count > 1 and np.array_equal(seed, seed[::-1]):
            continue
        return seed, unique_count

    seed = np.zeros(inner_width, dtype=int)
    for i, color in enumerate(colors):
        seed[i] = int(color)
    if unique_count > 1:
        seed[-1] = int(colors[0])
    return seed, unique_count


def _expected_from_boxes(inp: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = inp.copy()
    for box in boxes:
        _grow_inside_frame(out, box)
    return out


def _gen_attempt() -> np.ndarray | None:
    container_count = random.randint(1, 4)
    specs = []
    gaps = [random.randint(1, 3) for _ in range(max(0, container_count - 1))]

    for _ in range(container_count):
        inner_width = random.randint(3, 8)
        seed, unique_count = _sample_seed(inner_width)
        inner_height = 1 + unique_count + random.randint(0, 4)
        specs.append(
            {
                "inner_width": inner_width,
                "inner_height": inner_height,
                "seed": seed,
                "height": inner_height + 2,
                "width": inner_width + 2,
            }
        )

    structure_width = sum(spec["width"] for spec in specs) + sum(gaps)
    structure_height = max(spec["height"] for spec in specs)
    left = random.randint(0, 3)
    right = random.randint(0, 3)
    top = random.randint(0, 3)
    bottom = random.randint(0, 3)
    rows = top + structure_height + bottom
    cols = left + structure_width + right
    if rows > 30 or cols > 30:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    c = left
    for i, spec in enumerate(specs):
        r = top
        box = _draw_frame(grid, r, c, spec["height"], spec["width"])
        boxes.append(box)
        r0, c0, _r1, c1 = box
        grid[r0 + 1, c0 + 1 : c1] = spec["seed"]
        c += spec["width"]
        if i < len(gaps):
            c += gaps[i]

    expected = _expected_from_boxes(grid, boxes)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
