import random
from collections import Counter

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400
REGION_SIZE_RANGE = range(6, 8)
REGION_COUNT_RANGE = range(2, 5)
COLOR_POOL = tuple(range(10))


ARROW_PATTERNS = {
    frozenset({(0, 0), (1, 1), (2, 0)}): "right",
    frozenset({(0, 1), (1, 0), (2, 1)}): "left",
    frozenset({(0, 1), (1, 0), (1, 2)}): "up",
    frozenset({(0, 0), (0, 2), (1, 1)}): "down",
}

ARROW_BBOX = {
    "right": (3, 2),
    "left": (3, 2),
    "up": (2, 3),
    "down": (2, 3),
}


def _mode_color(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    assert flat, "expected at least one value"
    return int(Counter(flat).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


def _is_4_connected(coords: np.ndarray) -> bool:
    points = {tuple(map(int, rc)) for rc in np.asarray(coords)}
    if not points:
        return False
    seen = set()
    stack = [next(iter(points))]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in _neighbors4(*cur):
            if nxt in points and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(points)


def _uniform_spans(lines: list[int], limit: int) -> list[tuple[int, int]] | None:
    uniq = sorted(set(int(v) for v in lines))
    if len(uniq) != len(lines) or not uniq:
        return None
    bounds = [-1] + uniq + [int(limit)]
    spans: list[tuple[int, int]] = []
    lengths: list[int] = []
    for a, b in zip(bounds, bounds[1:]):
        start = int(a) + 1
        end = int(b) - 1
        length = end - start + 1
        if length <= 0:
            return None
        spans.append((start, end))
        lengths.append(length)
    if len(set(lengths)) != 1:
        return None
    return spans


def _find_partition(g: np.ndarray) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, g.shape)
    candidates = []
    for color in np.unique(g):
        color = int(color)
        row_lines = [r for r in range(h) if np.all(g[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(g[:, c] == color)]
        if not row_lines or not col_lines:
            continue
        row_spans = _uniform_spans(row_lines, h)
        col_spans = _uniform_spans(col_lines, w)
        if row_spans is None or col_spans is None:
            continue
        candidates.append((len(row_lines) + len(col_lines), color, row_lines, col_lines, row_spans, col_spans))
    assert candidates, "expected full separator rows and columns"
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, sep_color, row_lines, col_lines, row_spans, col_spans = candidates[0]
    return int(sep_color), row_lines, col_lines, row_spans, col_spans


def _arrow_direction(coords: np.ndarray) -> str | None:
    if int(coords.shape[0]) != 3:
        return None
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    return ARROW_PATTERNS.get(norm)


def _project_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    colors = [int(v) for v in np.unique(patch) if int(v) != int(bg)]
    assert len(colors) == 2, "each region must contain one arrow color and one shape color"

    arrow_matches: list[tuple[int, str]] = []
    for color in colors:
        coords = np.argwhere(patch == color)
        direction = _arrow_direction(coords)
        if direction is not None:
            arrow_matches.append((int(color), direction))

    assert len(arrow_matches) == 1, "expected exactly one 3-cell arrow marker per region"
    arrow_color, direction = arrow_matches[0]
    shape_color = next(color for color in colors if color != arrow_color)
    shape_coords = np.argwhere(patch == shape_color)
    assert _is_4_connected(shape_coords), "shape must be 4-connected"

    out = np.full_like(patch, bg)
    if direction in ("left", "right"):
        rows = np.unique(shape_coords[:, 0])
        target_col = 0 if direction == "left" else int(patch.shape[1] - 1)
        out[rows, target_col] = shape_color
    else:
        cols = np.unique(shape_coords[:, 1])
        target_row = 0 if direction == "up" else int(patch.shape[0] - 1)
        out[target_row, cols] = shape_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each intra-cell shape by its one-dimensional projection on the border selected by the cell's arrow marker.

    Input structure:
    - A single separator color fills complete rows and complete columns, splitting the canvas into equal-sized rectangular regions.
    - Inside every region, the non-separator area contains background cells, one 4-connected monochrome shape, and one 3-pixel arrow marker in another color.
    - The arrow marker is an 8-connected "^", "v", "<", or ">" pattern; different regions may point in different directions.

    Rule:
    - Keep all separator rows and columns unchanged.
    - For each region, detect the arrow direction and erase both the original arrow and the original shape.
    - If the arrow points left or right, mark the rows touched by the shape and paint those rows on the corresponding inner border column.
    - If the arrow points up or down, mark the columns touched by the shape and paint those columns on the corresponding inner border row.

    Key insight:
    - The arrow does not move the shape; it chooses the viewing direction for a projection, so the 2D object is flattened into a 1-cell-thick band on one edge of its own region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_color, row_lines, col_lines, row_spans, col_spans = _find_partition(g)

    region_mask = np.ones_like(g, dtype=bool)
    region_mask[row_lines, :] = False
    region_mask[:, col_lines] = False
    bg = _mode_color(g[region_mask])

    out = np.full_like(g, bg)
    out[row_lines, :] = sep_color
    out[:, col_lines] = sep_color

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            out[r0 : r1 + 1, c0 : c1 + 1] = _project_patch(patch, bg)

    return out


LAYOUTS = [
    (n_rows, n_cols, region_h, region_w)
    for n_rows in REGION_COUNT_RANGE
    for n_cols in REGION_COUNT_RANGE
    for region_h in REGION_SIZE_RANGE
    for region_w in REGION_SIZE_RANGE
    if n_rows * region_h + (n_rows - 1) <= 30 and n_cols * region_w + (n_cols - 1) <= 30
]


def _direction_multiset(count: int) -> list[str]:
    dirs = list(DIRECTIONS) * (count // len(DIRECTIONS))
    if count % len(DIRECTIONS):
        dirs.extend(random.sample(DIRECTIONS, count % len(DIRECTIONS)))
    random.shuffle(dirs)
    return dirs


def _arrow_cells(direction: str, top: int, left: int) -> list[tuple[int, int]]:
    if direction == "right":
        return [(top, left), (top + 1, left + 1), (top + 2, left)]
    if direction == "left":
        return [(top, left + 1), (top + 1, left), (top + 2, left + 1)]
    if direction == "up":
        return [(top, left + 1), (top + 1, left), (top + 1, left + 2)]
    if direction == "down":
        return [(top, left), (top, left + 2), (top + 1, left + 1)]
    raise ValueError(direction)


def _grow_shape(height: int, width: int, blocked: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    allowed = {
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (r, c) not in blocked
    }
    if len(allowed) < 4:
        return None

    max_area = min(8, len(allowed))
    for _ in range(80):
        target_area = random.randint(4, max_area)
        start = random.choice(tuple(allowed))
        cells = {start}
        frontier = {nbr for nbr in _neighbors4(*start) if nbr in allowed}
        while len(cells) < target_area and frontier:
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            if nxt in cells:
                continue
            cells.add(nxt)
            for nbr in _neighbors4(*nxt):
                if nbr in allowed and nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != target_area:
            continue
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue
        arr = np.array(sorted(cells), dtype=int)
        if not _is_4_connected(arr):
            continue
        return cells
    return None


def _region_patch(height: int, width: int, direction: str, bg: int, arrow_color: int, shape_color: int) -> np.ndarray | None:
    arrow_h, arrow_w = ARROW_BBOX[direction]
    top_choices = range(1, height - arrow_h)
    left_choices = range(1, width - arrow_w)
    if not top_choices or not left_choices:
        return None

    for _ in range(120):
        top = random.choice(tuple(top_choices))
        left = random.choice(tuple(left_choices))
        arrow = _arrow_cells(direction, top, left)

        blocked = set(arrow)
        for r, c in arrow:
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < height and 0 <= nc < width:
                    blocked.add((nr, nc))

        shape = _grow_shape(height, width, blocked)
        if shape is None:
            continue

        patch = np.full((height, width), bg, dtype=int)
        for r, c in shape:
            patch[r, c] = shape_color
        for r, c in arrow:
            patch[r, c] = arrow_color
        return patch
    return None


def _separator_lines(n_regions: int, region_size: int) -> list[int]:
    return [i * (region_size + 1) - 1 for i in range(1, n_regions)]


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols, region_h, region_w = random.choice(LAYOUTS)
    height = n_rows * region_h + (n_rows - 1)
    width = n_cols * region_w + (n_cols - 1)

    colors = list(COLOR_POOL)
    random.shuffle(colors)
    bg, sep_color, arrow_color = map(int, colors[:3])
    shape_palette = [int(c) for c in colors[3:]]

    grid = np.full((height, width), bg, dtype=int)
    row_lines = _separator_lines(n_rows, region_h)
    col_lines = _separator_lines(n_cols, region_w)
    grid[row_lines, :] = sep_color
    grid[:, col_lines] = sep_color

    directions = _direction_multiset(n_rows * n_cols)
    idx = 0
    used_shape_colors = set()
    for region_r in range(n_rows):
        r0 = region_r * (region_h + 1)
        for region_c in range(n_cols):
            c0 = region_c * (region_w + 1)
            direction = directions[idx]
            idx += 1
            shape_color = random.choice(shape_palette)
            used_shape_colors.add(shape_color)
            patch = _region_patch(region_h, region_w, direction, bg, arrow_color, shape_color)
            if patch is None:
                return None
            grid[r0 : r0 + region_h, c0 : c0 + region_w] = patch

    if len(used_shape_colors) < 2:
        return None
    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
