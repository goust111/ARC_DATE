from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (2, 3, 4, 8, 9)  # red, green, yellow, azure, maroon
LAYOUT_OPTIONS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
REGION_SIZE_RANGE = (5, 9)
GRID_HEIGHT_RANGE = (12, 30)
GRID_WIDTH_RANGE = (12, 30)
GEN_RETRY_LIMIT = 400


def _all_grey_rows(grid: np.ndarray) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == GREY))]


def _all_grey_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]


def _spans_from_separators(seps: list[int], limit: int) -> list[tuple[int, int]]:
    assert seps and seps[0] == 0 and seps[-1] == limit - 1, "missing outer grey border"
    spans: list[tuple[int, int]] = []
    for a, b in zip(seps, seps[1:]):
        assert b > a, "separator indices must be increasing"
        assert b - a >= 2, "separator rows/cols must leave nonempty regions"
        spans.append((int(a + 1), int(b - 1)))
    return spans


def _region_depth(r: int, c: int, r0: int, r1: int, c0: int, c1: int) -> int:
    return int(min(r - r0, r1 - r, c - c0, c1 - c))


def _extract_control_colors(grid: np.ndarray, expected_len: int) -> list[int]:
    control = np.asarray(grid[-1], dtype=int)
    nz = np.flatnonzero(control != BLACK)
    assert nz.size == expected_len, "control strip length must equal the region count"
    assert np.all(np.diff(nz) == 1), "control strip must be one contiguous run"
    colors = [int(control[c]) for c in nz]
    assert all(color in MARKER_COLORS for color in colors), "control strip must use marker colors"
    return colors


def _marker_cells(region: np.ndarray) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    h, w = region.shape
    for r in range(h):
        for c in range(w):
            color = int(region[r, c])
            if color == BLACK:
                continue
            assert color in MARKER_COLORS, "unexpected non-marker color inside region"
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) == (r, c):
                        continue
                    assert int(region[rr, cc]) == BLACK, "markers must be isolated single pixels"
            cells.append((int(r), int(c), color))
    assert 2 <= len(cells) <= 4, "each region must contain 2 to 4 markers"
    return cells


def _region_palette_and_source(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> tuple[list[int], int]:
    region = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    markers = []
    for rr, cc, color in _marker_cells(region):
        depth = _region_depth(int(rr), int(cc), 0, region.shape[0] - 1, 0, region.shape[1] - 1)
        markers.append((int(depth), int(color)))
    markers.sort()
    depths = [depth for depth, _color in markers]
    assert len(set(depths)) == len(depths), "marker depths must be distinct inside a region"
    palette: list[int] = []
    if markers[0][0] != 0:
        palette.append(BLACK)
    palette.extend(color for _depth, color in markers)
    return palette, int(markers[0][1])


def _paint_region(out: np.ndarray, r0: int, r1: int, c0: int, c1: int, palette: list[int]) -> None:
    plen = len(palette)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = _region_depth(int(r), int(c), int(r0), int(r1), int(c0), int(c1))
            out[r, c] = int(palette[depth % plen])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a concentric palette in each grey-framed region, then grey out rings in the next region
    using the shallowest marker color of the current region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 3 and g.shape[1] >= 5, "grid is too small"
    assert np.all(g[-1] != GREY), "control row must sit outside the grey frame block"

    body = g[:-1, :]
    sep_rows = _all_grey_rows(body)
    sep_cols = _all_grey_cols(body)
    row_spans = _spans_from_separators(sep_rows, int(body.shape[0]))
    col_spans = _spans_from_separators(sep_cols, int(body.shape[1]))
    regions = [(r0, r1, c0, c1) for r0, r1 in row_spans for c0, c1 in col_spans]
    assert len(regions) >= 2, "expected multiple framed regions"

    _extract_control_colors(g, len(regions))

    out = g.copy()
    palettes: list[list[int]] = []
    source_colors: list[int] = []
    for r0, r1, c0, c1 in regions:
        palette, source_color = _region_palette_and_source(g, r0, r1, c0, c1)
        palettes.append(palette)
        source_colors.append(source_color)
        _paint_region(out, int(r0), int(r1), int(c0), int(c1), palette)

    for idx in range(len(regions) - 1):
        source_color = int(source_colors[idx])
        r0, r1, c0, c1 = regions[idx + 1]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if int(out[r, c]) == source_color:
                    out[r, c] = GREY

    return out


def _ring_cells(height: int, width: int, depth: int) -> list[tuple[int, int]]:
    top = int(depth)
    left = int(depth)
    bottom = int(height - 1 - depth)
    right = int(width - 1 - depth)
    cells: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        cells.append((top, c))
        if bottom != top:
            cells.append((bottom, c))
    for r in range(top + 1, bottom):
        cells.append((r, left))
        if right != left:
            cells.append((r, right))
    return cells


def _pick_marker_positions(height: int, width: int, depths: list[int]) -> dict[int, tuple[int, int]] | None:
    order = depths[:]
    random.shuffle(order)
    candidates = {depth: _ring_cells(height, width, depth)[:] for depth in depths}
    for depth in candidates:
        random.shuffle(candidates[depth])
    chosen: dict[int, tuple[int, int]] = {}

    def ok(cell: tuple[int, int]) -> bool:
        return all(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) > 1 for other in chosen.values())

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        depth = int(order[idx])
        for cell in candidates[depth]:
            if not ok(cell):
                continue
            chosen[depth] = cell
            if dfs(idx + 1):
                return True
            del chosen[depth]
        return False

    return chosen if dfs(0) else None


def _max_depth(height: int, width: int) -> int:
    return int(min((height - 1) // 2, (width - 1) // 2))


def _sample_region_spec(height: int, width: int, required_color: int | None) -> dict[str, object] | None:
    max_depth = _max_depth(height, width)
    if max_depth < 2:
        return None

    include_depth0 = bool(random.getrandbits(1))
    if include_depth0:
        available_depths = list(range(0, max_depth + 1))
    else:
        available_depths = list(range(1, max_depth + 1))
    if len(available_depths) < 2:
        return None

    marker_count = random.randint(2, min(4, len(available_depths)))
    depths = sorted(random.sample(available_depths, marker_count))

    colors = list(MARKER_COLORS)
    if required_color is None:
        marker_colors = random.sample(colors, marker_count)
    else:
        assert required_color in colors
        others = [color for color in colors if color != required_color]
        marker_colors = random.sample(others, marker_count - 1) + [required_color]
        random.shuffle(marker_colors)

    positions = _pick_marker_positions(height, width, depths)
    if positions is None:
        return None

    markers = []
    for depth, color in zip(depths, marker_colors):
        r, c = positions[int(depth)]
        markers.append((int(r), int(c), int(color), int(depth)))
    markers.sort(key=lambda item: item[3])
    source_color = int(markers[0][2])
    return {
        "markers": [(r, c, color) for r, c, color, _depth in markers],
        "source_color": source_color,
    }


def _sample_layout() -> tuple[list[int], list[int]]:
    for _ in range(200):
        rows, cols = random.choice(LAYOUT_OPTIONS)
        row_heights = [random.randint(*REGION_SIZE_RANGE) for _ in range(rows)]
        col_widths = [random.randint(*REGION_SIZE_RANGE) for _ in range(cols)]
        body_h = sum(row_heights) + rows + 1
        body_w = sum(col_widths) + cols + 1
        total_h = body_h + 1
        if GRID_HEIGHT_RANGE[0] <= total_h <= GRID_HEIGHT_RANGE[1] and GRID_WIDTH_RANGE[0] <= body_w <= GRID_WIDTH_RANGE[1]:
            return row_heights, col_widths
    raise RuntimeError("failed to sample a valid layout")


def _region_origins(row_heights: list[int], col_widths: list[int]) -> list[tuple[int, int, int, int]]:
    regions: list[tuple[int, int, int, int]] = []
    r = 1
    for height in row_heights:
        c = 1
        for width in col_widths:
            regions.append((int(r), int(r + height - 1), int(c), int(c + width - 1)))
            c += width + 1
        r += height + 1
    return regions


def _build_input() -> np.ndarray | None:
    row_heights, col_widths = _sample_layout()
    body_h = sum(row_heights) + len(row_heights) + 1
    body_w = sum(col_widths) + len(col_widths) + 1

    regions = _region_origins(row_heights, col_widths)
    body = np.full((body_h, body_w), GREY, dtype=int)

    specs: list[dict[str, object]] = []
    required_color: int | None = None
    for r0, r1, c0, c1 in regions:
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        spec = _sample_region_spec(height, width, required_color)
        if spec is None:
            return None
        specs.append(spec)
        required_color = int(spec["source_color"])

    for (r0, r1, c0, c1), spec in zip(regions, specs):
        body[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        for rr, cc, color in spec["markers"]:
            body[r0 + int(rr), c0 + int(cc)] = int(color)

    grid = np.vstack([body, np.full((1, body_w), BLACK, dtype=int)])
    control_colors = [int(spec["source_color"]) for spec in specs]
    start = random.randint(0, body_w - len(control_colors))
    grid[-1, start : start + len(control_colors)] = np.array(control_colors, dtype=int)

    expected = main(grid)
    if np.array_equal(expected, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate grey-framed region grids with unique marker depths and a bottom control strip.
    """
    for _ in range(GEN_RETRY_LIMIT):
        try:
            grid = _build_input()
        except Exception:
            continue
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if out.shape != grid.shape:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
