from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
GEN_RETRY_LIMIT = 500


def _components_of_color(g: np.ndarray, color: int) -> list[np.ndarray]:
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in zip(*np.where(g == color)):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        todo = deque([(sr, sc)])
        cells = []
        while todo:
            r, c = todo.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and not seen[rr, cc]:
                    if int(g[rr, cc]) == int(color):
                        seen[rr, cc] = True
                        todo.append((rr, cc))
        comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_bbox(g: np.ndarray, coords: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = 2 * height + 2 * width - 4
    if int(coords.shape[0]) != expected:
        return None

    rows = coords[:, 0]
    cols = coords[:, 1]
    on_perimeter = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
    if not bool(np.all(on_perimeter)):
        return None

    if not (
        np.all(g[r0, c0 : c1 + 1] == color)
        and np.all(g[r1, c0 : c1 + 1] == color)
        and np.all(g[r0 : r1 + 1, c0] == color)
        and np.all(g[r0 : r1 + 1, c1] == color)
    ):
        return None

    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == color):
        return None
    return r0, c0, r1, c1


def _find_frames(g: np.ndarray) -> list[dict]:
    frames = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        for coords in _components_of_color(g, color):
            bbox = _frame_bbox(g, coords, color)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            frames.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                }
            )
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """Read each colored container's contents, ordered by container size.

    Input structure:
    - The grid background is black.
    - Several one-pixel-thick colored rectangular frames are scattered as
      containers, with unique bounding-box areas.
    - Inside each frame are one or more pixels of exactly one content color,
      different from every container border color.

    Rule:
    - Detect every complete rectangular frame and its bounding box.
    - For each frame, read the single non-black non-border color in its
      interior as the container's content color.
    - Sort containers by bounding-box area from smallest to largest and return a
      1-row color bar of their content colors.

    Key insight:
    - Border color is only a container identity distractor; the frame size gives
      the order, while the enclosed content color gives the output value.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    records: list[tuple[int, int]] = []
    areas = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        border_color = int(frame["color"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        content = inner[(inner != BG) & (inner != border_color)]
        colors = np.unique(content)
        assert colors.size == 1, "each container must hold exactly one content color"
        area = int(frame["area"])
        records.append((area, int(colors[0])))
        areas.append(area)

    assert len(areas) == len(set(areas)), "container bounding-box areas must be unique"
    return np.array([[color for _area, color in sorted(records)]], dtype=int)


def _paint_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, color: int) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _blob_cells(height: int, width: int, count: int) -> set[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    for _ in range(count - 1):
        frontier = []
        for r, c in cells:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    while len(cells) < count:
        cells.add((random.randrange(height), random.randrange(width)))
    return cells


def _content_cells(height: int, width: int) -> set[tuple[int, int]]:
    total = height * width
    count = random.randint(2, min(7, total))
    mode = random.choice(("blob", "scatter", "line"))

    if mode == "line":
        horizontal_options = []
        if width >= 2:
            horizontal_options.append(True)
        if height >= 2:
            horizontal_options.append(False)
        horizontal = random.choice(horizontal_options)
        if horizontal:
            length = random.randint(2, min(width, count))
            r = random.randrange(height)
            c0 = random.randrange(width - length + 1)
            return {(r, c0 + i) for i in range(length)}
        length = random.randint(2, min(height, count))
        r0 = random.randrange(height - length + 1)
        c = random.randrange(width)
        return {(r0 + i, c) for i in range(length)}

    if mode == "blob":
        return _blob_cells(height, width, count)

    return set(random.sample([(r, c) for r in range(height) for c in range(width)], count))


def _sample_dimensions(n_items: int, grid_h: int, grid_w: int) -> list[tuple[int, int, int]] | None:
    max_h = min(9, grid_h - 2)
    max_w = min(9, grid_w - 2)
    if max_h < 4 or max_w < 4:
        return None

    specs = []
    used_areas = set()
    for _ in range(n_items):
        for _attempt in range(100):
            height = random.randint(4, max_h)
            width = random.randint(4, max_w)
            area = height * width
            if area in used_areas:
                continue
            used_areas.add(area)
            specs.append((height, width, area))
            break
        else:
            return None
    random.shuffle(specs)
    return specs


def _free_positions(occupied: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    grid_h, grid_w = occupied.shape
    positions = []
    for r0 in range(1, grid_h - height):
        for c0 in range(1, grid_w - width):
            window = occupied[r0 - 1 : r0 + height + 1, c0 - 1 : c0 + width + 1]
            if not bool(window.any()):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(16, 30)
    grid_w = random.randint(16, 30)
    n_items = random.randint(3, 5)

    border_colors = random.sample(COLORS, n_items)
    content_palette = [c for c in COLORS if c not in border_colors]
    content_colors = [random.choice(content_palette) for _ in range(n_items)]
    if len(set(content_colors)) == 1:
        replacement = random.choice([c for c in content_palette if c != content_colors[0]])
        content_colors[random.randrange(n_items)] = replacement

    dims = _sample_dimensions(n_items, grid_h, grid_w)
    if dims is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    expected_records = []

    specs = list(zip(dims, border_colors, content_colors))
    random.shuffle(specs)
    for (height, width, area), border, content in specs:
        positions = _free_positions(occupied, height, width)
        if not positions:
            return None
        r0, c0 = random.choice(positions)

        _paint_frame(grid, r0, c0, height, width, border)
        for rr, cc in _content_cells(height - 2, width - 2):
            grid[r0 + 1 + rr, c0 + 1 + cc] = content

        occupied[r0 - 1 : r0 + height + 1, c0 - 1 : c0 + width + 1] = True
        expected_records.append((area, content))

    expected = np.array([[color for _area, color in sorted(expected_records)]], dtype=int)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed2047415201/10/5 input")
