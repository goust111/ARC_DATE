import numpy as np


RED = 2
DIRS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
BG_CHOICES = (0, 7, 8)
BODY_COLOR_POOL = tuple(range(10))
LENGTH_CHOICES = (3, 5, 7)
GRID_HEIGHT_RANGE = (15, 23)
GRID_WIDTH_RANGE = (15, 23)
SEGMENT_COUNT_RANGE = (4, 7)
MAX_PLACEMENT_ATTEMPTS = 500
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _segment_cells(mid_r: int, mid_c: int, length: int, step: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = step
    half = length // 2
    start_r = mid_r - half * dr
    start_c = mid_c - half * dc
    return [(start_r + i * dr, start_c + i * dc) for i in range(length)]


def _horizontal_cells(mid_r: int, mid_c: int, length: int) -> list[tuple[int, int]]:
    half = length // 2
    return [(mid_r, mid_c - half + i) for i in range(length)]


def _expanded_cells(cells: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(h, w, rr, cc):
                    expanded.add((rr, cc))
    return list(expanded)


def _cells_fit(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(_in_bounds(h, w, r, c) for r, c in cells)


def _placeable(mask: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not mask[r, c] for r, c in cells)


def _reserve(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in _expanded_cells(cells, h, w):
        mask[r, c] = True


def _parse_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    red_positions = [tuple(map(int, pos)) for pos in np.argwhere(g == RED)]
    assert red_positions, "expected at least one red start marker"

    segments: list[dict] = []
    used: set[tuple[int, int]] = set()
    non_bg = {tuple(map(int, pos)) for pos in np.argwhere(g != bg)}

    for start_r, start_c in red_positions:
        neighbors: list[tuple[int, int, int]] = []
        for dr, dc in DIRS_8:
            nr = start_r + dr
            nc = start_c + dc
            if not _in_bounds(h, w, nr, nc):
                continue
            color = int(g[nr, nc])
            if color != bg and color != RED:
                neighbors.append((dr, dc, color))
        assert len(neighbors) == 1, "each red marker must touch exactly one line body"
        dr, dc, body_color = neighbors[0]
        assert body_color != RED and body_color != bg

        before_r = start_r - dr
        before_c = start_c - dc
        assert not _in_bounds(h, w, before_r, before_c) or int(g[before_r, before_c]) == bg

        coords = [(start_r, start_c)]
        cur_r = start_r + dr
        cur_c = start_c + dc
        while _in_bounds(h, w, cur_r, cur_c) and int(g[cur_r, cur_c]) == body_color:
            coords.append((cur_r, cur_c))
            cur_r += dr
            cur_c += dc
        assert len(coords) >= 2, "segment must have at least two cells"
        assert not _in_bounds(h, w, cur_r, cur_c) or int(g[cur_r, cur_c]) == bg

        coord_set = set(coords)
        for r, c in coords:
            for ndr in (-1, 0, 1):
                for ndc in (-1, 0, 1):
                    if ndr == 0 and ndc == 0:
                        continue
                    rr = r + ndr
                    cc = c + ndc
                    if not _in_bounds(h, w, rr, cc):
                        continue
                    color = int(g[rr, cc])
                    if color == bg:
                        continue
                    assert (rr, cc) in coord_set, "segments must be isolated from one another"

        for cell in coords:
            assert cell not in used, "segments may not overlap"
            used.add(cell)

        segments.append(
            {
                "start": (start_r, start_c),
                "end": coords[-1],
                "length": len(coords),
                "body_color": body_color,
                "coords": coords,
            }
        )

    assert used == non_bg, "all non-background cells must belong to a red-marked segment"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each red-marked straight line so its direction becomes horizontal to the right.

    Input structure
    - The grid has a dominant background.
    - Each object is a one-pixel-wide straight line segment whose body is one color.
    - Exactly one endpoint of each segment is replaced by a red marker.

    Rule
    - Read the direction from the red endpoint toward the opposite endpoint.
    - Rotate the whole segment so the direction becomes left-to-right.
    - Keep the segment centered on its original midpoint.
    - Erase the original placement.

    Key insight
    - The red endpoint encodes which end is the start, so the required rotation is determined by geometry alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)
    segments = _parse_segments(g)

    out = np.full_like(g, bg)
    for seg in segments:
        start_r, start_c = seg["start"]
        end_r, end_c = seg["end"]
        length = int(seg["length"])
        body_color = int(seg["body_color"])

        mid_r2 = start_r + end_r
        mid_c2 = start_c + end_c
        assert mid_r2 % 2 == 0, "vertical/diagonal segments require an integer midpoint row"
        assert (mid_c2 - (length - 1)) % 2 == 0, "segment midpoint must support centered horizontal placement"

        new_r = mid_r2 // 2
        new_c0 = (mid_c2 - (length - 1)) // 2
        assert 0 <= new_r < h
        assert 0 <= new_c0 and new_c0 + length <= w
        assert np.all(out[new_r, new_c0 : new_c0 + length] == bg), "rotated segments must not overlap"

        out[new_r, new_c0] = RED
        if length > 1:
            out[new_r, new_c0 + 1 : new_c0 + length] = body_color

    return out


def _body_palette(bg: int) -> list[int]:
    return [color for color in BODY_COLOR_POOL if color not in (bg, RED)]


def _expected_output(specs: list[dict], h: int, w: int, bg: int) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for spec in specs:
        cells = spec["output_cells"]
        assert all(out[r, c] == bg for r, c in cells)
        out[cells[0]] = RED
        for r, c in cells[1:]:
            out[r, c] = spec["body_color"]
    return out


def _render_input(specs: list[dict], h: int, w: int, bg: int) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        cells = spec["input_cells"]
        grid[cells[0]] = RED
        for r, c in cells[1:]:
            grid[r, c] = spec["body_color"]
    return grid


def _pick_midpoint(h: int, w: int, length: int, step: tuple[int, int]) -> tuple[int, int] | None:
    dr, dc = step
    half = length // 2
    row_min = half * abs(dr)
    row_max = h - 1 - half * abs(dr)
    col_min = max(half, half * abs(dc))
    col_max = min(w - 1 - half, w - 1 - half * abs(dc))
    if row_min > row_max or col_min > col_max:
        return None
    mid_r = int(_RNG.integers(row_min, row_max + 1))
    mid_c = int(_RNG.integers(col_min, col_max + 1))
    return mid_r, mid_c


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(*GRID_HEIGHT_RANGE))
    w = int(_RNG.integers(*GRID_WIDTH_RANGE))
    bg = int(_RNG.choice(BG_CHOICES))
    n_segments = int(_RNG.integers(*SEGMENT_COUNT_RANGE))

    directions = list(DIRS_8)
    _RNG.shuffle(directions)
    chosen_dirs = directions[:n_segments]
    colors = _RNG.choice(_body_palette(bg), size=n_segments, replace=False).tolist()

    in_reserved = np.zeros((h, w), dtype=bool)
    out_reserved = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []

    for step, body_color in zip(chosen_dirs, colors):
        placed = False
        sampled_lengths = _RNG.permutation(LENGTH_CHOICES)
        for length in sampled_lengths:
            for _ in range(MAX_PLACEMENT_ATTEMPTS):
                midpoint = _pick_midpoint(h, w, int(length), step)
                if midpoint is None:
                    break
                mid_r, mid_c = midpoint
                input_cells = _segment_cells(mid_r, mid_c, int(length), step)
                output_cells = _horizontal_cells(mid_r, mid_c, int(length))
                if not _cells_fit(input_cells, h, w):
                    continue
                if not _cells_fit(output_cells, h, w):
                    continue
                if not _placeable(in_reserved, input_cells):
                    continue
                if not _placeable(out_reserved, output_cells):
                    continue

                _reserve(in_reserved, input_cells)
                _reserve(out_reserved, output_cells)
                specs.append(
                    {
                        "body_color": int(body_color),
                        "input_cells": input_cells,
                        "output_cells": output_cells,
                    }
                )
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    inp = _render_input(specs, h, w, bg)
    out = _expected_output(specs, h, w, bg)

    if np.count_nonzero(inp == RED) != n_segments:
        return None
    if np.unique(inp).size < min(4, n_segments + 1):
        return None
    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
