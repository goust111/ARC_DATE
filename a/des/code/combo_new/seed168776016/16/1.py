from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORS = np.arange(1, 10, dtype=int)
RNG = np.random.default_rng()

MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 28
MIN_ROOM_SPAN = 3
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _interior_regions(grid: np.ndarray, wall_color: int) -> list[list[tuple[int, int]]]:
    sub = grid[1:, :]
    h, w = sub.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(sub != wall_color):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _draw_frame(grid: np.ndarray, top: int, left: int, bottom: int, right: int, wall_color: int) -> tuple[int, int, int, int]:
    grid[top, left : right + 1] = wall_color
    grid[bottom, left : right + 1] = wall_color
    grid[top : bottom + 1, left] = wall_color
    grid[top : bottom + 1, right] = wall_color
    return (top + 1, left + 1, bottom - 1, right - 1)


def _room_area(room: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = room
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _split_options(room: tuple[int, int, int, int]) -> list[str]:
    r0, c0, r1, c1 = room
    options: list[str] = []
    if c1 - c0 + 1 >= 2 * MIN_ROOM_SPAN + 1:
        options.append("v")
    if r1 - r0 + 1 >= 2 * MIN_ROOM_SPAN + 1:
        options.append("h")
    return options


def _split_room(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    wall_color: int,
) -> list[tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = room
    options = _split_options(room)
    if not options:
        return None

    orient = options[int(RNG.integers(len(options)))]
    if orient == "v":
        split_col = int(RNG.integers(c0 + MIN_ROOM_SPAN, c1 - MIN_ROOM_SPAN + 1))
        grid[r0 - 1 : r1 + 2, split_col] = wall_color
        return [(r0, c0, r1, split_col - 1), (r0, split_col + 1, r1, c1)]

    split_row = int(RNG.integers(r0 + MIN_ROOM_SPAN, r1 - MIN_ROOM_SPAN + 1))
    grid[split_row, c0 - 1 : c1 + 2] = wall_color
    return [(r0, c0, split_row - 1, c1), (split_row + 1, c0, r1, c1)]


def _add_stub(grid: np.ndarray, room: tuple[int, int, int, int], wall_color: int) -> None:
    if float(RNG.random()) < 0.35:
        return

    r0, c0, r1, c1 = room
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options: list[str] = []
    if width >= 3:
        options.extend(["left", "right"])
    if height >= 3:
        options.extend(["top", "bottom"])
    if not options:
        return

    side = options[int(RNG.integers(len(options)))]
    if side == "left":
        row = int(RNG.integers(r0, r1 + 1))
        length = int(RNG.integers(1, width))
        grid[row, c0 : c0 + length] = wall_color
    elif side == "right":
        row = int(RNG.integers(r0, r1 + 1))
        length = int(RNG.integers(1, width))
        grid[row, c1 - length + 1 : c1 + 1] = wall_color
    elif side == "top":
        col = int(RNG.integers(c0, c1 + 1))
        length = int(RNG.integers(1, height))
        grid[r0 : r0 + length, col] = wall_color
    else:
        col = int(RNG.integers(c0, c1 + 1))
        length = int(RNG.integers(1, height))
        grid[r1 - length + 1 : r1 + 1, col] = wall_color


def _marker_cells(region: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    order = RNG.permutation(len(region))
    candidates = [region[int(i)] for i in order]
    chosen: list[tuple[int, int]] = []

    for cell in candidates:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen

    for cell in candidates:
        if cell not in chosen:
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each enclosed region with the legend color indexed by its marker count.

    Input structure:
    - The background is black.
    - The top row contains a left-to-right legend of distinct one-cell colors; position 1 maps to count 1, position 2 to count 2, and so on.
    - Below the legend, one wall color forms connected barriers that partition the black area into enclosed regions.
    - Each enclosed region contains one or more single-pixel marker dots of one shared marker color.

    Rule:
    - Read the non-black cells on the top row in left-to-right order to obtain the count-to-color legend.
    - Identify the wall color and the marker color among the remaining non-black pixels.
    - For every non-wall component below the legend that does not touch the grid border, count how many marker dots it contains.
    - Replace every black cell and marker dot in that component with the legend color at the matching count index, while leaving the walls and the legend unchanged.

    Key insight:
    - The marker dots are not copied; their quantity is the lookup key that selects a legend color for the whole enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0

    legend_cols = [int(c) for c in np.where(g[0] != bg)[0]]
    legend = [int(g[0, c]) for c in legend_cols]
    assert legend, "expected a non-empty top-row legend"

    legend_set = set(legend)
    other_counts = Counter(
        int(v)
        for v in g[1:, :].ravel()
        if int(v) != bg and int(v) not in legend_set
    )
    assert len(other_counts) == 2, "expected exactly one wall color and one marker color"

    wall_color = max(other_counts, key=lambda color: other_counts[color])
    marker_color = min(other_counts, key=lambda color: other_counts[color])

    out = g.copy()
    sub = g[1:, :]
    h, w = sub.shape
    for region in _components(sub != wall_color):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in region):
            continue
        marker_count = sum(1 for r, c in region if int(sub[r, c]) == marker_color)
        assert 1 <= marker_count <= len(legend), "marker count must map into the legend"
        fill_color = legend[marker_count - 1]
        for r, c in region:
            out[r + 1, c] = fill_color

    return out


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    w = int(RNG.integers(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))

    legend_len = int(RNG.integers(2, 5))
    chosen = RNG.choice(COLORS, size=legend_len + 2, replace=False)
    legend = [int(v) for v in chosen[:legend_len]]
    wall_color = int(chosen[-2])
    marker_color = int(chosen[-1])

    gap_bits = RNG.integers(0, 2, size=max(0, legend_len - 1))
    legend_cols = [0]
    for gap in gap_bits:
        legend_cols.append(legend_cols[-1] + 1 + int(gap))
    legend_span = legend_cols[-1] + 1
    if legend_span > w - 2:
        return None

    legend_start = int(RNG.integers(1, w - legend_span))
    legend_cols = [legend_start + c for c in legend_cols]

    grid = np.zeros((h, w), dtype=int)
    for idx, col in enumerate(legend_cols):
        grid[0, col] = legend[idx]

    frame_top = int(RNG.integers(2, 5))
    frame_left = int(RNG.integers(1, 4))
    if frame_top + 9 >= h - 1 or frame_left + 9 >= w - 1:
        return None
    frame_bottom = int(RNG.integers(frame_top + 9, h - 1))
    frame_right = int(RNG.integers(frame_left + 9, w - 1))

    base_room = _draw_frame(grid, frame_top, frame_left, frame_bottom, frame_right, wall_color)
    rooms = [base_room]

    target_rooms = int(RNG.integers(legend_len, min(legend_len + 3, 7) + 1))
    while len(rooms) < target_rooms:
        splittable = [i for i, room in enumerate(rooms) if _split_options(room)]
        if not splittable:
            break
        weights = np.array([_room_area(rooms[i]) for i in splittable], dtype=float)
        pick = splittable[int(RNG.choice(len(splittable), p=weights / weights.sum()))]
        new_rooms = _split_room(grid, rooms[pick], wall_color)
        if new_rooms is None:
            break
        rooms[pick : pick + 1] = new_rooms

    if len(rooms) < legend_len:
        return None

    for room in rooms:
        _add_stub(grid, room, wall_color)

    regions = _interior_regions(grid, wall_color)
    if len(regions) < legend_len:
        return None
    if any(len(region) < legend_len for region in regions):
        return None

    region_counts = np.array(
        list(range(1, legend_len + 1))
        + [int(RNG.integers(1, legend_len + 1)) for _ in range(len(regions) - legend_len)],
        dtype=int,
    )
    RNG.shuffle(region_counts)

    expected = grid.copy()
    for region, count in zip(regions, region_counts.tolist(), strict=False):
        markers = _marker_cells(region, int(count))
        if len(markers) != int(count):
            return None
        for r, c in markers:
            grid[r + 1, c] = marker_color
        fill_color = legend[int(count) - 1]
        for r, c in region:
            expected[r + 1, c] = fill_color

    if np.count_nonzero(grid == marker_color) < legend_len:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
