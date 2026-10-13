import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 500


def _prefix_len(values: np.ndarray, color: int) -> int:
    n = 0
    for value in values:
        if int(value) != color:
            break
        n += 1
    return n


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """Project each colored rectangle in its grey marker direction using the matching edge scale length.

    Input structure:
    - Black is the background.
    - The top row contains a maroon run starting at column 0; its length is the
      horizontal projection scale.
    - The left column contains a maroon run starting at row 0; its length is the
      vertical projection scale.
    - Inside the grid are separated solid monochrome rectangles. Each rectangle
      has exactly one adjacent grey marker either on its right side or below it.

    Rule:
    - A right-side grey marker projects that rectangle to the right for the top
      row maroon-run length, beginning one column after the rectangle and
      replacing the marker.
    - A below grey marker projects that rectangle downward for the left column
      maroon-run length, beginning one row after the rectangle and replacing the
      marker.
    - Projection cells outside the grid are truncated, and the edge scale bars
      themselves stay unchanged.

    Key insight:
    - The grey marker gives only the direction; projection distance is a global
      scale read from the matching edge bar, horizontal for rightward movement
      and vertical for downward movement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h > 1 and w > 1, "expected a grid with scale bars and interior cells"

    horizontal_len = _prefix_len(g[0, :], MAROON)
    vertical_len = _prefix_len(g[:, 0], MAROON)
    assert horizontal_len > 0 and vertical_len > 0, "missing maroon scale bars"
    assert np.all(g[0, horizontal_len:] == BLACK), "top scale row must be black after the maroon run"
    assert np.all(g[vertical_len:, 0] == BLACK), "left scale column must be black after the maroon run"

    out = g.copy()
    ignored = {BLACK, GREY, MAROON}
    for color in np.unique(g):
        color = int(color)
        if color in ignored:
            continue
        for coords in _components(g == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

            rect = g[r0 : r1 + 1, c0 : c1 + 1]
            assert rect.size == len(coords), "object must fill its bounding rectangle"
            assert np.all(rect == color), "object must be monochrome and solid"

            has_right_marker = c1 + 1 < w and bool(np.any(g[r0 : r1 + 1, c1 + 1] == GREY))
            has_down_marker = r1 + 1 < h and bool(np.any(g[r1 + 1, c0 : c1 + 1] == GREY))
            assert has_right_marker != has_down_marker, "each object needs one unambiguous grey direction marker"

            if has_right_marker:
                end = min(w, c1 + 1 + horizontal_len)
                out[r0 : r1 + 1, c1 + 1 : end] = color
            else:
                end = min(h, r1 + 1 + vertical_len)
                out[r1 + 1 : end, c0 : c1 + 1] = color

    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    return out


def _rect_cells(r0: int, c0: int, height: int, width: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r0 + height)
        for c in range(c0, c0 + width)
    }


def _projection_cells(
    rows: int,
    cols: int,
    r0: int,
    c0: int,
    height: int,
    width: int,
    direction: str,
    horizontal_len: int,
    vertical_len: int,
) -> tuple[set[tuple[int, int]], bool]:
    if direction == "right":
        start = c0 + width
        end = min(cols, start + horizontal_len)
        cells = {(r, c) for r in range(r0, r0 + height) for c in range(start, end)}
        return cells, start + horizontal_len > cols

    start = r0 + height
    end = min(rows, start + vertical_len)
    cells = {(r, c) for r in range(start, end) for c in range(c0, c0 + width)}
    return cells, start + vertical_len > rows


def _expanded(cells: set[tuple[int, int]], rows: int, cols: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    expanded.add((nr, nc))
    return expanded


def _try_place_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved: np.ndarray,
    color: int,
    direction: str,
    horizontal_len: int,
    vertical_len: int,
    force_truncated: bool,
) -> tuple[int, int, int, int, str] | None:
    rows, cols = grid.shape
    for _ in range(400):
        height = random.randint(1, 3)
        width = random.randint(1, 4)

        min_r, min_c = 2, 2
        max_r = rows - height - 1
        max_c = cols - width - 1
        if direction == "right":
            max_c = cols - width - 1
            if force_truncated:
                min_c = max(min_c, cols - width - horizontal_len + 1)
        else:
            max_r = rows - height - 1
            if force_truncated:
                min_r = max(min_r, rows - height - vertical_len + 1)

        if min_r > max_r or min_c > max_c:
            continue

        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)
        object_cells = _rect_cells(r0, c0, height, width)
        if direction == "right":
            marker = (random.randint(r0, r0 + height - 1), c0 + width)
        else:
            marker = (r0 + height, random.randint(c0, c0 + width - 1))
        actual_cells = object_cells | {marker}
        projection_cells, truncated = _projection_cells(
            rows, cols, r0, c0, height, width, direction, horizontal_len, vertical_len
        )
        if force_truncated and not truncated:
            continue

        if any(reserved[r, c] for r, c in actual_cells):
            continue
        if any(reserved[r, c] for r, c in projection_cells):
            continue
        if any(occupied[r, c] for r, c in _expanded(actual_cells, rows, cols)):
            continue

        for r, c in object_cells:
            grid[r, c] = color
            occupied[r, c] = True
        mr, mc = marker
        grid[mr, mc] = GREY
        occupied[mr, mc] = True
        for r, c in actual_cells | projection_cells:
            reserved[r, c] = True
        return r0, c0, height, width, direction

    return None


def _expected_from_specs(
    grid: np.ndarray,
    specs: list[tuple[int, int, int, int, int, str]],
    horizontal_len: int,
    vertical_len: int,
) -> np.ndarray:
    out = grid.copy()
    rows, cols = out.shape
    for color, r0, c0, height, width, direction in specs:
        if direction == "right":
            start = c0 + width
            out[r0 : r0 + height, start : min(cols, start + horizontal_len)] = color
        else:
            start = r0 + height
            out[start : min(rows, start + vertical_len), c0 : c0 + width] = color
    out[0, :] = grid[0, :]
    out[:, 0] = grid[:, 0]
    return out


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(13, 26)
    cols = random.randint(13, 26)
    horizontal_len = random.randint(2, min(7, cols - 4))
    vertical_len = random.randint(2, min(7, rows - 4))

    grid = np.full((rows, cols), BLACK, dtype=int)
    grid[0, :horizontal_len] = MAROON
    grid[:vertical_len, 0] = MAROON

    occupied = np.zeros((rows, cols), dtype=bool)
    reserved = np.zeros((rows, cols), dtype=bool)
    occupied[0, :horizontal_len] = True
    occupied[:vertical_len, 0] = True
    reserved[0, :] = True
    reserved[:, 0] = True

    object_count = random.randint(2, 5)
    directions = ["right", "down"]
    directions.extend(random.choice(["right", "down"]) for _ in range(object_count - 2))
    random.shuffle(directions)
    colors = random.sample(OBJECT_COLORS, object_count)
    force_trunc_idx = random.randrange(object_count) if random.random() < 0.45 else None

    specs: list[tuple[int, int, int, int, int, str]] = []
    for idx, (color, direction) in enumerate(zip(colors, directions)):
        placement = _try_place_object(
            grid,
            occupied,
            reserved,
            color,
            direction,
            horizontal_len,
            vertical_len,
            force_truncated=(idx == force_trunc_idx),
        )
        if placement is None:
            return None
        r0, c0, height, width, direction = placement
        specs.append((color, r0, c0, height, width, direction))

    expected = _expected_from_specs(grid, specs, horizontal_len, vertical_len)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid projection-scale input")
