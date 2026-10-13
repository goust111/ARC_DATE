import random
from collections import defaultdict, deque

import numpy as np


PRIORITY_COLORS = (2, 1, 3, 4, 7, 6)  # red, blue, green, yellow, orange, magenta
PRIORITY_RANK = {color: rank for rank, color in enumerate(PRIORITY_COLORS)}

BG_COLOR = 0
WALL_COLOR = 5

ROOM_LAYOUTS = [
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (1, 4),
    (4, 1),
    (1, 5),
    (5, 1),
    (1, 6),
    (6, 1),
]
MIN_ROOM_SIDE = 2
MAX_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _most_common_non_bg(grid: np.ndarray, bg: int) -> int:
    values = [int(v) for v in grid.ravel() if int(v) != bg]
    assert values, "expected at least one non-background color"
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    top = r0 - 1
    left = c0 - 1
    bottom = r1 + 1
    right = c1 + 1
    points: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        points.append((top, c))
        points.append((bottom, c))
    for r in range(top + 1, bottom):
        points.append((r, left))
        points.append((r, right))
    return points


def _priority_color(colors: set[int]) -> int:
    assert colors, "expected at least one owner color"
    for color in colors:
        assert color in PRIORITY_RANK, "seed colors must use the visible priority palette"
    return min(colors, key=lambda color: PRIORITY_RANK[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Claim each enclosed rectangle by its seed color, then repaint borders.

    Input structure:
    - bg = black.
    - grey one-cell lines form closed rectangular regions, possibly with shared borders.
    - each enclosed region contains exactly one single-pixel seed color.

    Rule:
    - each enclosed region inherits its seed color;
    - border pixels used by one region take that region's color;
    - border pixels shared by multiple regions take the earliest color from
      [red, blue, green, yellow, orange, magenta];
    - seeds disappear in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _get_bg(g)
    wall = _most_common_non_bg(g, bg)
    passable = g != wall

    regions: list[tuple[tuple[int, int, int, int], int]] = []
    seen_owners: set[int] = set()
    for coords in _components(passable):
        if _touches_edge(coords, h, w):
            continue

        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        assert window.size == len(coords), "each enclosed region must have a solid rectangular interior"
        assert np.all(window != wall), "room interior cannot contain wall pixels"

        owner_colors = {int(v) for v in window.ravel() if int(v) != bg}
        assert len(owner_colors) == 1, "each region must contain exactly one seed color"
        owner = next(iter(owner_colors))
        assert owner not in seen_owners, "seed colors must be unique across regions"
        assert int(np.count_nonzero(window == owner)) == 1, "seed must be a single pixel"

        for br, bc in _frame_points(bbox):
            assert 0 <= br < h and 0 <= bc < w, "room frame must stay inside grid"
            assert int(g[br, bc]) == wall, "room frame must be drawn in wall color"

        seen_owners.add(owner)
        regions.append((bbox, owner))

    assert regions, "expected at least one enclosed region"

    owners_by_border: dict[tuple[int, int], set[int]] = defaultdict(set)
    for bbox, owner in regions:
        for point in _frame_points(bbox):
            owners_by_border[point].add(owner)

    out = np.full_like(g, bg)
    for (r, c), owners in owners_by_border.items():
        out[r, c] = _priority_color(owners)
    return out


def _sample_axis_sizes(rng: random.Random, count: int) -> list[int]:
    if count <= 2:
        max_side = 6
    elif count <= 4:
        max_side = 5
    else:
        max_side = 4
    return [rng.randint(MIN_ROOM_SIDE, max_side) for _ in range(count)]


def _build_room_bboxes(
    row_heights: list[int],
    col_widths: list[int],
    top_margin: int,
    left_margin: int,
) -> list[tuple[int, int, int, int]]:
    row_starts: list[int] = []
    current_row = top_margin + 1
    for height in row_heights:
        row_starts.append(current_row)
        current_row += height + 1

    col_starts: list[int] = []
    current_col = left_margin + 1
    for width in col_widths:
        col_starts.append(current_col)
        current_col += width + 1

    rooms: list[tuple[int, int, int, int]] = []
    for row_start, height in zip(row_starts, row_heights):
        for col_start, width in zip(col_starts, col_widths):
            rooms.append((row_start, col_start, row_start + height - 1, col_start + width - 1))
    return rooms


def _make_grid(
    row_heights: list[int],
    col_widths: list[int],
    margins: tuple[int, int, int, int],
    seed_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    top_margin, left_margin, bottom_margin, right_margin = margins
    room_h = sum(row_heights) + (len(row_heights) - 1)
    room_w = sum(col_widths) + (len(col_widths) - 1)
    structure_h = room_h + 2
    structure_w = room_w + 2
    grid_h = top_margin + structure_h + bottom_margin
    grid_w = left_margin + structure_w + right_margin
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    rooms = _build_room_bboxes(row_heights, col_widths, top_margin, left_margin)
    for room in rooms:
        r0, c0, r1, c1 = room
        for br, bc in _frame_points(room):
            grid[br, bc] = WALL_COLOR

    for color, room in zip(seed_colors, rooms):
        r0, c0, r1, c1 = room
        sr = rng.randint(r0, r1)
        sc = rng.randint(c0, c1)
        grid[sr, sc] = color

    return grid


def _has_shared_and_exclusive_borders(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    wall_pixels = np.argwhere(g == WALL_COLOR)
    if wall_pixels.size == 0:
        return False

    passable = g != WALL_COLOR
    h, w = g.shape
    regions: list[tuple[tuple[int, int, int, int], int]] = []
    for coords in _components(passable):
        if _touches_edge(coords, h, w):
            continue
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        owners = {int(v) for v in window.ravel() if int(v) != BG_COLOR}
        if len(owners) != 1:
            return False
        regions.append((bbox, next(iter(owners))))

    owners_by_border: dict[tuple[int, int], set[int]] = defaultdict(set)
    for bbox, owner in regions:
        for point in _frame_points(bbox):
            owners_by_border[point].add(owner)

    shared = any(len(owners) >= 2 for owners in owners_by_border.values())
    exclusive = any(len(owners) == 1 for owners in owners_by_border.values())
    return shared and exclusive


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows, n_cols = rng.choice(ROOM_LAYOUTS)
    n_rooms = n_rows * n_cols
    row_heights = _sample_axis_sizes(rng, n_rows)
    col_widths = _sample_axis_sizes(rng, n_cols)
    margins = tuple(rng.randint(1, 4) for _ in range(4))
    seed_colors = rng.sample(list(PRIORITY_COLORS), n_rooms)

    grid = _make_grid(row_heights, col_widths, margins, seed_colors, rng)
    if not (10 <= grid.shape[0] <= 30 and 10 <= grid.shape[1] <= 30):
        return None
    if not _has_shared_and_exclusive_borders(grid):
        return None

    solved = main(grid)
    if solved.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
