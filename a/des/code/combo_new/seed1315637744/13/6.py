import random
from collections import Counter

import numpy as np


GREY = 5
MAROON = 9
MAX_GRID = 30
PAIR_COUNT_RANGE = (2, 5)
INNER_LEGEND_WIDTHS = (4, 5)
GRID_MARGIN_CHOICES = (0, 1)
MIN_ENDPOINT_GAP = 4
PLACEMENT_TRIES = 200
GENERATION_TRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _frame_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_bbox_if_any(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if int(r1 - r0 + 1) < 3 or int(c1 - c0 + 1) < 4:
        return None
    expected = _frame_cells(int(r0), int(c0), int(r1), int(c1))
    actual = {tuple(map(int, rc)) for rc in coords}
    if actual != expected:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend_rows(
    inner: np.ndarray, bg: int, frame_color: int
) -> tuple[dict[int, int], int] | None:
    mapping: dict[int, int] = {}
    bar_color: int | None = None

    for row in inner:
        nz = np.flatnonzero(row != bg)
        if len(nz) < 2:
            return None
        if not np.array_equal(nz, np.arange(nz[0], nz[-1] + 1)):
            return None

        sample_idx = int(nz[0])
        sample_color = int(row[sample_idx])
        run = row[sample_idx + 1 : int(nz[-1]) + 1]
        if run.size == 0 or not (1 <= run.size <= 3):
            return None

        run_color = int(run[0])
        if sample_color in (bg, frame_color) or run_color in (bg, frame_color):
            return None
        if sample_color == run_color or not np.all(run == run_color):
            return None
        if np.any(row[:sample_idx] != bg) or np.any(row[int(nz[-1]) + 1 :] != bg):
            return None
        if sample_color in mapping:
            return None

        if bar_color is None:
            bar_color = run_color
        elif run_color != bar_color:
            return None

        mapping[sample_color] = int(run.size)

    if len(mapping) < 2:
        return None
    assert bar_color is not None
    return mapping, bar_color


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    bg = _most_common_color(grid)
    h, w = grid.shape
    candidates = []

    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        bbox = _frame_bbox_if_any(grid, color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        parsed = _parse_legend_rows(inner, bg, color)
        if parsed is None:
            continue
        mapping, _bar_color = parsed
        if inner.shape[0] != len(mapping):
            continue
        score = (int(r1 + c1), int(r1), int(c1), -int(r0 + c0))
        candidates.append((score, mapping, bbox))

    assert candidates, "expected a framed legend in the lower-right area"
    _score, mapping, (r0, c0, r1, c1) = max(candidates, key=lambda item: item[0])
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mapping, mask


def _draw_connection(
    out: np.ndarray,
    legend_mask: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: int,
    thickness_value: int,
) -> None:
    h, w = out.shape
    radius = thickness_value - 1
    r0, c0 = p0
    r1, c1 = p1

    if r0 == r1:
        row = r0
        left, right = sorted((c0, c1))
        for c in range(left + 1, right):
            for rr in range(row - radius, row + radius + 1):
                if 0 <= rr < h and not legend_mask[rr, c]:
                    out[rr, c] = color
        return

    assert c0 == c1, "pair endpoints must share a row or a column"
    col = c0
    top, bottom = sorted((r0, r1))
    for r in range(top + 1, bottom):
        for cc in range(col - radius, col + radius + 1):
            if 0 <= cc < w and not legend_mask[r, cc]:
                out[r, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color marker pair with a straight line whose thickness is read from the framed legend.

    Input structure:
    - The grid has a dominant background color plus several same-color marker pairs in the main field.
    - Each pair consists of exactly two single pixels of one color, and the two markers share either a row or a column.
    - A small rectangular legend sits in the lower-right area inside a one-pixel frame; each legend row contains one sample color pixel followed immediately by a short same-row bar of another color.

    Rule:
    - Read each legend row as a mapping from the sample color to the bar length.
    - For every marker pair of that color, draw a straight connector between the two markers, excluding the marker pixels themselves.
    - Thickness value 1, 2, 3 maps to connector widths 1, 3, 5 respectively, expanding perpendicular to the connector direction and clipping at the grid boundary; the legend region itself stays unchanged.

    Key insight:
    - The legend does not name thickness directly; the horizontal bar length is an intermediate visual code that must be converted to connector width before drawing each pair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    mapping, legend_mask = _find_legend(g)

    outside = g.copy()
    outside[legend_mask] = bg
    pair_colors = [int(v) for v in np.unique(outside) if int(v) != bg]
    assert set(pair_colors) == set(mapping), "legend colors must match the marker-pair colors"

    out = g.copy()
    for color in pair_colors:
        coords = np.argwhere(outside == color)
        assert coords.shape == (2, 2), "each non-legend color must appear in exactly one pair"
        p0 = tuple(map(int, coords[0]))
        p1 = tuple(map(int, coords[1]))
        assert p0 != p1
        assert p0[0] == p1[0] or p0[1] == p1[1], "pair endpoints must align horizontally or vertically"
        _draw_connection(out, legend_mask, p0, p1, color, int(mapping[color]))

    out[legend_mask] = g[legend_mask]
    return out


def _reserve_bbox(
    occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int
) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
        return False
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _sample_thicknesses(rng: random.Random, count: int) -> list[int]:
    cycle = [1, 2, 3]
    rng.shuffle(cycle)
    values = [cycle[i % 3] for i in range(count)]
    rng.shuffle(values)
    return values


def _sample_orientations(rng: random.Random, count: int) -> list[str]:
    if count == 1:
        return [rng.choice(["h", "v"])]
    n_h = count // 2
    if count % 2 == 1:
        n_h += rng.randint(0, 1)
    n_h = max(1, min(count - 1, n_h))
    orientations = ["h"] * n_h + ["v"] * (count - n_h)
    rng.shuffle(orientations)
    return orientations


def _sample_center_index(rng: random.Random, size: int, radius: int) -> int:
    if radius > 0 and rng.random() < 0.35:
        edge_choices = list(range(radius)) + list(range(size - radius, size))
        if edge_choices:
            return int(rng.choice(edge_choices))
    return int(rng.randrange(size))


def _place_pair(
    rng: random.Random, occupied: np.ndarray, orientation: str, thickness_value: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = occupied.shape
    radius = thickness_value - 1

    for _ in range(PLACEMENT_TRIES):
        if orientation == "h":
            if w - 1 < MIN_ENDPOINT_GAP:
                return None
            row = _sample_center_index(rng, h, radius)
            left = rng.randint(0, w - 1 - MIN_ENDPOINT_GAP)
            right = rng.randint(left + MIN_ENDPOINT_GAP, w - 1)
            r0 = max(0, row - radius)
            r1 = min(h - 1, row + radius)
            c0 = left
            c1 = right
            p0 = (row, left)
            p1 = (row, right)
        else:
            if h - 1 < MIN_ENDPOINT_GAP:
                return None
            col = _sample_center_index(rng, w, radius)
            top = rng.randint(0, h - 1 - MIN_ENDPOINT_GAP)
            bottom = rng.randint(top + MIN_ENDPOINT_GAP, h - 1)
            r0 = top
            r1 = bottom
            c0 = max(0, col - radius)
            c1 = min(w - 1, col + radius)
            p0 = (top, col)
            p1 = (bottom, col)

        if _reserve_bbox(occupied, r0, c0, r1, c1, pad=1):
            return p0, p1

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    inner_w = rng.choice(INNER_LEGEND_WIDTHS)
    sample_col = rng.randint(0, inner_w - 4)
    legend_h = pair_count + 2
    legend_w = inner_w + 2

    min_h = max(14, legend_h + 5, 2 * pair_count + 6)
    min_w = max(14, legend_w + 7, 2 * pair_count + 6)
    if min_h > MAX_GRID or min_w > MAX_GRID:
        return None

    h = rng.randint(min_h, MAX_GRID)
    w = rng.randint(min_w, MAX_GRID)
    bg_choices = [color for color in range(10) if color not in (GREY, MAROON)]
    bg = int(rng.choice(bg_choices))
    pair_palette = [color for color in range(10) if color not in (bg, GREY, MAROON)]
    pair_colors = rng.sample(pair_palette, pair_count)
    thicknesses = _sample_thicknesses(rng, pair_count)
    orientations = _sample_orientations(rng, pair_count)

    grid = np.full((h, w), bg, dtype=int)

    bottom_margin = rng.choice(GRID_MARGIN_CHOICES)
    right_margin = rng.choice(GRID_MARGIN_CHOICES)
    legend_r0 = h - legend_h - bottom_margin
    legend_c0 = w - legend_w - right_margin
    legend_r1 = legend_r0 + legend_h - 1
    legend_c1 = legend_c0 + legend_w - 1
    if legend_r0 < 1 or legend_c0 < 1:
        return None

    grid[legend_r0, legend_c0 : legend_c1 + 1] = GREY
    grid[legend_r1, legend_c0 : legend_c1 + 1] = GREY
    grid[legend_r0 : legend_r1 + 1, legend_c0] = GREY
    grid[legend_r0 : legend_r1 + 1, legend_c1] = GREY

    order = list(range(pair_count))
    rng.shuffle(order)
    for row_offset, idx in enumerate(order):
        rr = legend_r0 + 1 + row_offset
        color = int(pair_colors[idx])
        thickness_value = int(thicknesses[idx])
        grid[rr, legend_c0 + 1 + sample_col] = color
        bar_start = legend_c0 + 2 + sample_col
        grid[rr, bar_start : bar_start + thickness_value] = MAROON

    occupied = np.zeros((h, w), dtype=bool)
    if not _reserve_bbox(occupied, legend_r0, legend_c0, legend_r1, legend_c1, pad=1):
        return None

    for color, thickness_value, orientation in zip(pair_colors, thicknesses, orientations):
        placed = _place_pair(rng, occupied, orientation, thickness_value)
        if placed is None:
            return None
        (r0, c0), (r1, c1) = placed
        grid[r0, c0] = int(color)
        grid[r1, c1] = int(color)

    out = main(grid)
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to generate a valid sample")
