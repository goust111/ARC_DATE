from collections import deque

import numpy as np


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
LAYOUT_OPTIONS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2)]
ROOM_HEIGHT_RANGE = (4, 7)
ROOM_WIDTH_RANGE = (4, 8)
SIDE_MARGIN_RANGE = (1, 4)
TOP_MARGIN_RANGE = (1, 3)

_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(grid) if int(v) != int(bg)]
    if WALL in colors:
        return WALL
    assert colors
    counts = {color: int(np.sum(grid == color)) for color in colors}
    return max(colors, key=lambda color: counts[color])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)

    return components


def _find_closed_regions(grid: np.ndarray, wall: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    regions: list[list[tuple[int, int]]] = []
    for comp in _connected_components(grid != int(wall)):
        touches_border = any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp)
        if not touches_border:
            regions.append(comp)
    regions.sort(key=lambda comp: min(comp))
    return regions


def _row_segments(row: np.ndarray, bg: int, wall: int) -> list[tuple[int, int]]:
    mask = (row != int(bg)) & (row != int(wall))
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, filled in enumerate(mask):
        if bool(filled) and start is None:
            start = int(idx)
        if not bool(filled) and start is not None:
            segments.append((int(start), int(idx)))
            start = None
    if start is not None:
        segments.append((int(start), int(len(row))))
    return segments


def _extract_reference_bands(grid: np.ndarray, bg: int, wall: int) -> dict[int, list[int]]:
    h = grid.shape[0]
    wall_rows = np.where(np.any(grid == int(wall), axis=1))[0]
    assert len(wall_rows) > 0
    first_wall = int(wall_rows.min())
    last_wall = int(wall_rows.max())

    candidates = list(range(0, first_wall)) + list(range(last_wall + 1, h))
    bands: dict[int, list[int]] = {}
    for r in candidates:
        row = grid[r]
        if np.any(row == int(wall)):
            continue
        segments = _row_segments(row, bg=bg, wall=wall)
        if len(segments) != 1:
            continue
        c0, c1 = segments[0]
        colors = [int(v) for v in row[c0:c1]]
        bands[len(colors)] = colors
    assert bands
    return bands


def _ordered_boundary_cells(region_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = region_mask.shape
    edge_map: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]] = {}

    for r, c in np.argwhere(region_mask):
        r = int(r)
        c = int(c)
        if r == 0 or not region_mask[r - 1, c]:
            edge_map[(r, c)] = ((r, c + 1), (r, c))
        if c == w - 1 or not region_mask[r, c + 1]:
            edge_map[(r, c + 1)] = ((r + 1, c + 1), (r, c))
        if r == h - 1 or not region_mask[r + 1, c]:
            edge_map[(r + 1, c + 1)] = ((r + 1, c), (r, c))
        if c == 0 or not region_mask[r, c - 1]:
            edge_map[(r + 1, c)] = ((r, c), (r, c))

    assert edge_map
    start_vertex = min(edge_map)
    ordered: list[tuple[int, int]] = []
    cur = start_vertex

    for _ in range(len(edge_map) + 5):
        nxt, cell = edge_map[cur]
        if not ordered or ordered[-1] != cell:
            ordered.append(cell)
        cur = nxt
        if cur == start_vertex:
            break
    else:
        raise AssertionError("failed to trace a closed boundary")

    if ordered and ordered[0] == ordered[-1]:
        ordered.pop()
    assert ordered
    start_idx = min(range(len(ordered)), key=lambda idx: ordered[idx])
    ordered = ordered[start_idx:] + ordered[:start_idx]
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall = _infer_wall_color(g, bg=bg)
    bands = _extract_reference_bands(g, bg=bg, wall=wall)
    regions = _find_closed_regions(g, wall=wall)

    out = g.copy()
    for region in regions:
        region_mask = np.zeros_like(g, dtype=bool)
        for r, c in region:
            region_mask[r, c] = True

        colors = sorted(
            {
                int(g[r, c])
                for r, c in region
                if int(g[r, c]) != int(bg) and int(g[r, c]) != int(wall)
            }
        )
        kind_count = len(colors)
        assert kind_count in bands

        boundary = _ordered_boundary_cells(
            region_mask
            & (
                np.roll(~region_mask, 1, axis=0)
                | np.roll(~region_mask, -1, axis=0)
                | np.roll(~region_mask, 1, axis=1)
                | np.roll(~region_mask, -1, axis=1)
            )
        )

        for r, c in region:
            out[r, c] = int(bg)

        band = bands[kind_count]
        for idx, (r, c) in enumerate(boundary):
            out[r, c] = int(band[idx % len(band)])

    return out


def _sample_sizes(count: int, low: int, high: int) -> list[int]:
    return [int(_RNG.integers(low, high)) for _ in range(count)]


def _pick_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    idxs = np.atleast_1d(_RNG.choice(len(cells), size=count, replace=False))
    return [cells[int(idx)] for idx in idxs]


def _paint_reference_bands(
    grid: np.ndarray,
    side: str,
    band_rows: list[int],
    lengths: list[int],
) -> dict[int, list[int]]:
    h, w = grid.shape
    del h
    order = list(_RNG.permutation(lengths))
    bands: dict[int, list[int]] = {}
    for row, length in zip(band_rows, order):
        colors = [int(v) for v in _RNG.choice(PALETTE, size=length, replace=False)]
        start = int(_RNG.integers(1, w - length)) if w - length > 1 else 0
        grid[row, start : start + length] = colors
        bands[length] = colors
    assert side in {"top", "bottom"}
    return bands


def _scatter_room_dots(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    kind_count: int,
) -> None:
    r0, c0, r1, c1 = room
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    interior = [(r, c) for r, c in cells if r0 < r < r1 and c0 < c < c1]
    room_colors = [int(v) for v in _RNG.choice(PALETTE, size=kind_count, replace=False)]

    max_extra = min(4, len(cells) - kind_count - 1)
    dot_total = kind_count + int(_RNG.integers(0, max_extra + 1))
    positions = _pick_cells(cells, dot_total)
    if interior and not any(pos in set(interior) for pos in positions):
        outside = [pos for pos in positions if pos not in set(interior)]
        if outside:
            replace_idx = positions.index(outside[0])
            spare = [pos for pos in interior if pos not in positions]
            if spare:
                positions[replace_idx] = spare[int(_RNG.integers(0, len(spare)))]

    color_stream = room_colors + [
        int(_RNG.choice(room_colors)) for _ in range(dot_total - kind_count)
    ]
    _RNG.shuffle(color_stream)
    for (r, c), color in zip(positions, color_stream):
        grid[r, c] = int(color)


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = LAYOUT_OPTIONS[int(_RNG.integers(0, len(LAYOUT_OPTIONS)))]
    room_count = int(room_rows * room_cols)

    room_heights = _sample_sizes(room_rows, *ROOM_HEIGHT_RANGE)
    room_widths = _sample_sizes(room_cols, *ROOM_WIDTH_RANGE)
    main_h = sum(room_heights) + room_rows + 1
    main_w = sum(room_widths) + room_cols + 1

    first_length = int(_RNG.integers(1, 3))
    band_count = int(_RNG.integers(3, 5))
    if first_length + band_count - 1 > 5:
        band_count = 5 - first_length + 1
    lengths = list(range(first_length, first_length + band_count))
    band_height = 2 * len(lengths)

    side = "top" if bool(_RNG.integers(0, 2)) else "bottom"
    left_margin = int(_RNG.integers(*SIDE_MARGIN_RANGE))
    right_margin = int(_RNG.integers(*SIDE_MARGIN_RANGE))

    if side == "top":
        frame_top = band_height
        bottom_margin = int(_RNG.integers(*TOP_MARGIN_RANGE))
        height = frame_top + main_h + bottom_margin
    else:
        frame_top = int(_RNG.integers(*TOP_MARGIN_RANGE))
        band_start = frame_top + main_h + 1
        height = band_start + band_height

    width = left_margin + main_w + right_margin
    if height > 30 or width > 30:
        return None

    frame_left = left_margin
    frame_right = frame_left + main_w - 1
    frame_bottom = frame_top + main_h - 1
    grid = np.full((height, width), BG, dtype=int)

    horiz = [frame_top]
    for size in room_heights:
        horiz.append(horiz[-1] + size + 1)
    vert = [frame_left]
    for size in room_widths:
        vert.append(vert[-1] + size + 1)

    for r in horiz:
        grid[r, frame_left : frame_right + 1] = WALL
    for c in vert:
        grid[frame_top : frame_bottom + 1, c] = WALL

    rooms: list[tuple[int, int, int, int]] = []
    for i in range(room_rows):
        for j in range(room_cols):
            rooms.append((horiz[i] + 1, vert[j] + 1, horiz[i + 1] - 1, vert[j + 1] - 1))

    if len(rooms) < 2:
        return None

    used_counts = [int(_RNG.choice(lengths)) for _ in range(room_count)]
    if len(set(used_counts)) < 2:
        return None

    if side == "top":
        band_rows = [2 * idx for idx in range(len(lengths))]
    else:
        band_rows = [band_start + 2 * idx for idx in range(len(lengths))]
    bands = _paint_reference_bands(grid, side=side, band_rows=band_rows, lengths=lengths)

    for room, kind_count in zip(rooms, used_counts):
        _scatter_room_dots(grid, room, kind_count)

    try:
        detected_bands = _extract_reference_bands(grid, bg=BG, wall=WALL)
        if set(detected_bands) != set(bands):
            return None
        regions = _find_closed_regions(grid, wall=WALL)
        if len(regions) != len(rooms):
            return None
        color_counts = []
        for region in regions:
            colors = {
                int(grid[r, c])
                for r, c in region
                if int(grid[r, c]) not in {BG, WALL}
            }
            if not colors:
                return None
            color_counts.append(len(colors))
        if sorted(color_counts) != sorted(used_counts):
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
