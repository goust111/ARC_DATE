import numpy as np
from collections import deque


BG = 0
FILL_COLORS = [9, 2, 7, 4, 3]  # maroon, red, orange, yellow, green
NON_BG_COLORS = [1, 5, 6, 8]
GEN_RETRIES = 300
RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _background_components(grid: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[list[tuple[int, int]], bool]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BG:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((cells, touches_edge))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed black region using a color determined only by its area rank.

    Input structure
    - Black background.
    - Multiple 1-pixel-thick horizontal/vertical colored line segments intersect to
      partition the background into open and closed rectangles.

    Rule
    - Find all 4-connected black regions not connected to the grid edge.
    - Compute each closed region's area.
    - Sort distinct areas from small to large and map them to:
      maroon, red, orange, yellow, green.
    - Equal areas receive the same fill color.
    - Leave the line pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    closed_regions: list[list[tuple[int, int]]] = []
    for cells, touches_edge in _background_components(g):
        if not touches_edge:
            closed_regions.append(cells)

    if not closed_regions:
        return out

    distinct_areas = sorted({len(cells) for cells in closed_regions})
    area_to_color = {
        area: FILL_COLORS[min(i, len(FILL_COLORS) - 1)]
        for i, area in enumerate(distinct_areas)
    }

    for cells in closed_regions:
        fill = area_to_color[len(cells)]
        for r, c in cells:
            out[r, c] = fill

    return out


def _sample_gaps(count: int, mode: str, low: int, high: int) -> list[int]:
    assert count >= 1
    if mode == "ties":
        pool_size = int(RNG.integers(1, min(3, high - low + 1) + 1))
        pool = RNG.choice(np.arange(low, high + 1), size=pool_size, replace=False)
        return [int(pool[int(RNG.integers(0, len(pool)))]) for _ in range(count)]

    # Distinct mode tries to spread areas out by using unique gaps when possible.
    values = list(RNG.choice(np.arange(low, high + 1), size=count, replace=False))
    RNG.shuffle(values)
    return [int(v) for v in values]


def _build_partition_positions(size: int, line_count: int, mode: str) -> list[int] | None:
    gap_count = line_count - 1
    if gap_count < 1:
        return None

    for _ in range(60):
        start_margin = int(RNG.integers(1, 5))
        end_margin = int(RNG.integers(1, 5))
        min_gap = 3
        max_gap = 7
        if mode == "distinct" and gap_count > (max_gap - min_gap + 1):
            continue
        gaps = _sample_gaps(gap_count, mode, min_gap, max_gap)
        total = start_margin + end_margin + sum(gaps)
        if total >= size:
            continue
        slack = size - total
        start_margin += int(RNG.integers(0, slack + 1))
        slack = size - (start_margin + end_margin + sum(gaps))
        end_margin += int(RNG.integers(0, slack + 1))

        pos = [start_margin]
        for gap in gaps:
            pos.append(pos[-1] + gap)
        if pos[-1] < size - end_margin:
            return pos
    return None


def _draw_lines(
    height: int,
    width: int,
    h_lines: list[int],
    v_lines: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    segments: list[tuple[str, int, int]] = []
    for r in h_lines:
        segments.append(("h", r, int(RNG.choice(NON_BG_COLORS))))
    for c in v_lines:
        segments.append(("v", c, int(RNG.choice(NON_BG_COLORS))))
    RNG.shuffle(segments)

    for kind, coord, color in segments:
        if kind == "h":
            grid[coord, :] = color
        else:
            grid[:, coord] = color
    return grid


def _enclosed_areas_from_lines(h_lines: list[int], v_lines: list[int]) -> list[int]:
    row_gaps = [h_lines[i + 1] - h_lines[i] - 1 for i in range(len(h_lines) - 1)]
    col_gaps = [v_lines[i + 1] - v_lines[i] - 1 for i in range(len(v_lines) - 1)]
    return [rh * cw for rh in row_gaps for cw in col_gaps]


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(14, 25))
    width = int(RNG.integers(14, 25))
    mode = "ties" if RNG.random() < 0.5 else "distinct"

    line_options = [(2, 3), (3, 2), (3, 3), (4, 3), (3, 4), (4, 4)]
    h_count, v_count = line_options[int(RNG.integers(0, len(line_options)))]

    h_lines = _build_partition_positions(height, h_count, mode)
    v_lines = _build_partition_positions(width, v_count, mode)
    if h_lines is None or v_lines is None:
        return None

    areas = _enclosed_areas_from_lines(h_lines, v_lines)
    unique_areas = sorted(set(areas))
    if len(areas) < 2:
        return None
    if len(unique_areas) > len(FILL_COLORS):
        return None
    if mode == "ties" and len(unique_areas) == len(areas):
        return None
    if mode == "distinct" and len(unique_areas) != len(areas):
        return None

    grid = _draw_lines(height, width, h_lines, v_lines)
    out = main(grid)

    input_colors = set(map(int, np.unique(grid)))
    if input_colors == {BG}:
        return None

    closed_input = [cells for cells, touches_edge in _background_components(grid) if not touches_edge]
    if len(closed_input) != len(areas):
        return None
    if sorted(len(cells) for cells in closed_input) != sorted(areas):
        return None

    output_fill_colors = set(map(int, np.unique(out))) - input_colors
    expected_fill_colors = {FILL_COLORS[i] for i in range(len(unique_areas))}
    if output_fill_colors != expected_fill_colors:
        return None

    # Keep at least one clearly open region connected to the border.
    if not any(touches_edge for _cells, touches_edge in _background_components(grid)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
