from __future__ import annotations

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
RNG = np.random.default_rng()

MIN_SEGMENTS = 3
MAX_SEGMENTS = 6
MIN_LENGTH = 3
MAX_LENGTH = 5
MIN_GRID = 16
MAX_GRID = 30
MAX_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _segment_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = direction
    return [(r0 + dr * i, c0 + dc * i) for i in range(length)]


def _paint_segment(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    main_color: int,
    alien_color: int,
) -> list[tuple[int, int]]:
    cells = _segment_cells(start, direction, length)
    for r, c in cells[:-1]:
        grid[r, c] = main_color
    grid[cells[-1]] = alien_color
    return cells


def _ordered_component(component: np.ndarray) -> list[tuple[int, int]]:
    rows = np.unique(component[:, 0])
    cols = np.unique(component[:, 1])
    if len(rows) == 1:
        row = int(rows[0])
        return [(row, int(c)) for c in np.sort(component[:, 1])]
    if len(cols) == 1:
        col = int(cols[0])
        return [(int(r), col) for r in np.sort(component[:, 0])]
    raise AssertionError("component must be a straight line")


def _parse_segment(grid: np.ndarray, component: np.ndarray) -> dict:
    ordered = _ordered_component(component)
    length = len(ordered)
    assert MIN_LENGTH <= length <= MAX_LENGTH
    values = [int(grid[r, c]) for r, c in ordered]

    forward = all(v == values[0] for v in values[:-1]) and values[-1] != values[0]
    backward = all(v == values[-1] for v in values[1:]) and values[0] != values[-1]
    assert forward ^ backward, "segment must have exactly one odd-colored endpoint"

    if forward:
        cells = ordered
        main_color = values[0]
        alien_color = values[-1]
    else:
        cells = ordered[::-1]
        main_color = values[-1]
        alien_color = values[0]

    start = cells[0]
    end = cells[-1]
    direction = (_sign(end[0] - start[0]), _sign(end[1] - start[1]))
    return {
        "start": (int(start[0]), int(start[1])),
        "end": (int(end[0]), int(end[1])),
        "direction": (int(direction[0]), int(direction[1])),
        "length": int(length),
        "main_color": int(main_color),
        "alien_color": int(alien_color),
    }


def _extract_segments(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _find_components(grid != bg)
    assert comps, "expected at least one segment"
    segments = [_parse_segment(grid, comp) for comp in comps]
    main_colors = [seg["main_color"] for seg in segments]
    assert len(main_colors) == len(set(main_colors)), "main colors must identify segments uniquely"
    return segments


def _order_segments(segments: list[dict]) -> list[dict]:
    alien_colors = {seg["alien_color"] for seg in segments}
    starts = [seg for seg in segments if seg["main_color"] not in alien_colors]
    assert len(starts) == 1, "expected exactly one chain start"

    main_map = {seg["main_color"]: seg for seg in segments}
    ordered: list[dict] = []
    seen: set[int] = set()
    current = starts[0]
    while True:
        key = id(current)
        assert key not in seen, "chain must not cycle"
        seen.add(key)
        ordered.append(current)
        nxt = main_map.get(current["alien_color"])
        if nxt is None:
            break
        current = nxt

    assert len(ordered) == len(segments), "color chain must consume all segments"
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """Reconnect scattered two-color line segments into the unique full polyline chain.

    Input structure:
    - The grid is mostly black background with several disconnected straight line segments.
    - Each segment is horizontal or vertical, has length 3-5, and contains exactly two colors.
    - Along one segment, all cells except one endpoint share the same main color; the opposite endpoint is a different color.
    - The segments' endpoint colors form a unique chain: one segment's odd endpoint color matches the next segment's main endpoint color.

    Rule:
    - For each segment, infer its direction from the endpoint whose color matches the interior cells toward the differently colored endpoint.
    - Find the unique starting segment whose main color does not appear as any other segment's odd endpoint color.
    - Order the remaining segments by repeatedly matching the current odd endpoint color to the next segment's main color.
    - Draw the ordered chain on a background canvas of the same size: keep the first segment fixed, then place each later segment with its main endpoint one step beyond the previous segment in the previous segment's direction, preserving the later segment's own original orientation.

    Key insight:
    - The scattered geometry is a distraction; the real ordering signal is the visible two-endpoint color code that defines both segment direction and the unique domino-like chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    ordered = _order_segments(_extract_segments(g, bg))

    out = np.full_like(g, bg)
    occupied: set[tuple[int, int]] = set()

    first = ordered[0]
    cells = _paint_segment(
        out,
        first["start"],
        first["direction"],
        first["length"],
        first["main_color"],
        first["alien_color"],
    )
    occupied.update(cells)
    prev_end = cells[-1]
    prev_dir = first["direction"]

    for seg in ordered[1:]:
        start = (prev_end[0] + prev_dir[0], prev_end[1] + prev_dir[1])
        cells = _segment_cells(start, seg["direction"], seg["length"])
        for r, c in cells:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "reconstructed chain must fit the canvas"
            assert (r, c) not in occupied, "reconstructed chain must not overlap itself"
        _paint_segment(out, start, seg["direction"], seg["length"], seg["main_color"], seg["alien_color"])
        occupied.update(cells)
        prev_end = cells[-1]
        prev_dir = seg["direction"]

    return out


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    idx = DIRS.index(direction)
    return DIRS[(idx - 1) % 4]


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    idx = DIRS.index(direction)
    return DIRS[(idx + 1) % 4]


def _sample_directions(n: int) -> list[tuple[int, int]]:
    directions = [DIRS[int(RNG.integers(0, len(DIRS)))]]
    for _ in range(1, n):
        prev = directions[-1]
        choices = [prev, _turn_left(prev), _turn_right(prev)]
        directions.append(choices[int(RNG.integers(0, len(choices)))])
    return directions


def _relative_layout(lengths: list[int], directions: list[tuple[int, int]]) -> dict | None:
    starts: list[tuple[int, int]] = []
    path: list[tuple[int, int]] = []
    current_start = (0, 0)

    for i, (length, direction) in enumerate(zip(lengths, directions)):
        if i > 0:
            prev_cell = path[-1]
            prev_dir = directions[i - 1]
            current_start = (prev_cell[0] + prev_dir[0], prev_cell[1] + prev_dir[1])
        starts.append(current_start)
        path.extend(_segment_cells(current_start, direction, length))

    if len(path) != len(set(path)):
        return None

    index = {cell: i for i, cell in enumerate(path)}
    for i, (r, c) in enumerate(path):
        for dr, dc in DIRS:
            j = index.get((r + dr, c + dc))
            if j is not None and abs(j - i) != 1:
                return None

    rows = [r for r, _ in path]
    cols = [c for _, c in path]
    return {
        "starts": starts,
        "path": path,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
    }


def _shift_layout(layout: dict, offset: tuple[int, int]) -> dict:
    dr, dc = offset
    return {
        "starts": [(r + dr, c + dc) for r, c in layout["starts"]],
        "path": [(r + dr, c + dc) for r, c in layout["path"]],
        "bbox": (
            layout["bbox"][0] + dr,
            layout["bbox"][1] + dc,
            layout["bbox"][2] + dr,
            layout["bbox"][3] + dc,
        ),
    }


def _segment_bbox_shape(direction: tuple[int, int], length: int) -> tuple[int, int]:
    dr, _dc = direction
    return (length, 1) if dr != 0 else (1, length)


def _main_from_top_left(top: int, left: int, direction: tuple[int, int], length: int) -> tuple[int, int]:
    dr, dc = direction
    if (dr, dc) == (0, 1):
        return top, left
    if (dr, dc) == (0, -1):
        return top, left + length - 1
    if (dr, dc) == (1, 0):
        return top, left
    if (dr, dc) == (-1, 0):
        return top + length - 1, left
    raise AssertionError("invalid direction")


def _can_place(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _mark(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _build_output(
    shape: tuple[int, int],
    starts: list[tuple[int, int]],
    directions: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for start, direction, length, main_color, alien_color in zip(
        starts, directions, lengths, colors[:-1], colors[1:]
    ):
        _paint_segment(out, start, direction, length, main_color, alien_color)
    return out


def _scatter_input(
    shape: tuple[int, int],
    output_starts: list[tuple[int, int]],
    directions: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)

    first_cells = _paint_segment(
        grid,
        output_starts[0],
        directions[0],
        lengths[0],
        colors[0],
        colors[1],
    )
    _mark(occupied, first_cells)

    for i in range(1, len(lengths)):
        direction = directions[i]
        length = lengths[i]
        seg_h, seg_w = _segment_bbox_shape(direction, length)
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            top = int(RNG.integers(0, h - seg_h + 1))
            left = int(RNG.integers(0, w - seg_w + 1))
            start = _main_from_top_left(top, left, direction, length)
            cells = _segment_cells(start, direction, length)
            if not _can_place(occupied, cells):
                continue
            _paint_segment(grid, start, direction, length, colors[i], colors[i + 1])
            _mark(occupied, cells)
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(RNG.integers(MIN_SEGMENTS, MAX_SEGMENTS + 1))
    lengths = [int(RNG.integers(MIN_LENGTH, MAX_LENGTH + 1)) for _ in range(n)]
    directions = _sample_directions(n)
    colors = list(map(int, RNG.choice(COLOR_POOL, size=n + 1, replace=False)))

    layout = _relative_layout(lengths, directions)
    if layout is None:
        return None

    min_r, min_c, max_r, max_c = layout["bbox"]
    box_h = max_r - min_r + 1
    box_w = max_c - min_c + 1
    min_h = max(MIN_GRID, box_h + 2)
    min_w = max(MIN_GRID, box_w + 2)
    if min_h > MAX_GRID or min_w > MAX_GRID:
        return None

    h = int(RNG.integers(min_h, MAX_GRID + 1))
    w = int(RNG.integers(min_w, MAX_GRID + 1))
    offset_r = int(RNG.integers(-min_r, h - max_r))
    offset_c = int(RNG.integers(-min_c, w - max_c))
    shifted = _shift_layout(layout, (offset_r, offset_c))

    out = _build_output((h, w), shifted["starts"], directions, lengths, colors)
    inp = _scatter_input((h, w), shifted["starts"], directions, lengths, colors)
    if inp is None:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
