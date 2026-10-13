from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

PATTERN_LIBRARY: tuple[np.ndarray, ...] = (
    np.ones((3, 3), dtype=bool),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
)

LEGEND_ROW_COUNT = 5
LEGEND_ROW_H = 3
LEGEND_ROW_GAP = 1
LEGEND_SAMPLE_W = 3
LEGEND_SAMPLE_H = 3

GEN_ROW_CELL_RANGE = (2, 3)
GEN_COL_CELL_RANGE = (2, 3)
GEN_INTERIOR_SIZE_RANGE = (3, 5)
GEN_MARKER_COLORS = (1, 2, 3, 4, 6)
GEN_OUTER_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == MAROON)
    assert len(coords) > 0, "expected a maroon legend frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "legend frame must enclose an interior"
    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros_like(box, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.all(box[border] == MAROON), "legend border must be a full maroon rectangle"
    assert np.count_nonzero(grid == MAROON) == int(border.sum()), "maroon pixels must belong only to the legend border"
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    active_rows = np.any((inner == ORANGE) | (inner == AZURE), axis=1)

    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(active_rows):
        while start < len(active_rows) and not active_rows[start]:
            start += 1
        if start >= len(active_rows):
            break
        end = start
        while end + 1 < len(active_rows) and active_rows[end + 1]:
            end += 1
        spans.append((start, end))
        start = end + 1

    mapping: dict[int, np.ndarray] = {}
    for sr0, sr1 in spans:
        band = inner[sr0 : sr1 + 1]
        count = int(np.count_nonzero(band == ORANGE))
        assert 1 <= count <= 5, "legend digits must encode counts 1..5"
        assert count not in mapping, "legend counts must be unique"

        sample_cells = np.argwhere(band == AZURE)
        assert len(sample_cells) > 0, "each legend row must include an azure sample"
        pr0, pc0 = sample_cells.min(axis=0)
        pr1, pc1 = sample_cells.max(axis=0)
        pattern = (band[pr0 : pr1 + 1, pc0 : pc1 + 1] == AZURE).copy()
        mapping[count] = pattern

    assert mapping, "legend mapping must be non-empty"
    return mapping


def _enclosed_regions(grid: np.ndarray, legend_bbox: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = legend_bbox
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True

    open_mask = (grid != GREY) & ~blocked
    regions: list[list[tuple[int, int]]] = []
    for comp in _components(open_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        regions.append(comp)
    return regions


def _paint_region(
    out: np.ndarray,
    cells: list[tuple[int, int]],
    pattern: np.ndarray,
    color: int,
) -> None:
    r0, c0, _, _ = _bbox(cells)
    ph, pw = pattern.shape
    for r, c in cells:
        out[r, c] = color if pattern[(r - r0) % ph, (c - c0) % pw] else BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-enclosed rectangle with the legend texture keyed by marker count and recolored by the marker color.

    Input structure:
    - The background is black. Grey one-pixel horizontal and vertical segments intersect to form closed rectangular regions.
    - A separate maroon rectangular legend box sits in a corner and contains one row per count: orange pixels encode a number from 1 to 5, and an adjacent azure texture sample shows that count's fill pattern.
    - Each enclosed rectangle contains several same-colored single-pixel markers on black background.

    Rule:
    - Find the maroon legend box and read each row as count -> texture-mask mapping, where the orange run length gives the count and the azure sample gives the repeated pattern.
    - Find every connected non-grey region that is enclosed by the grey walls and ignore the legend area.
    - For each enclosed region, count its non-black marker pixels, select the matching legend texture, and repaint the whole region with that texture using the marker color.
    - The original markers are overwritten by the texture; grey walls and the legend stay unchanged.

    Key insight:
    - The markers encode two independent pieces of information at once: how many markers there are chooses the texture, while their shared color determines the texture's final paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    legend_bbox = _find_legend_bbox(g)
    mapping = _parse_legend(g, legend_bbox)

    out = g.copy()
    for region in _enclosed_regions(g, legend_bbox):
        marker_values = [int(g[r, c]) for r, c in region if int(g[r, c]) != BG]
        assert marker_values, "every enclosed region must contain markers"
        marker_colors = set(marker_values)
        assert len(marker_colors) == 1, "markers inside a region must share one color"

        count = len(marker_values)
        assert count in mapping, "marker count must appear in the legend"
        _paint_region(out, region, mapping[count], marker_values[0])

    return out


def _legend_dims(sample_c0: int) -> tuple[int, int]:
    inner_h = LEGEND_ROW_COUNT * LEGEND_ROW_H + (LEGEND_ROW_COUNT - 1) * LEGEND_ROW_GAP
    inner_w = sample_c0 + LEGEND_SAMPLE_W
    return inner_h + 2, inner_w + 2


def _draw_legend(
    grid: np.ndarray,
    corner: str,
    sample_c0: int,
    count_to_pattern: dict[int, np.ndarray],
    row_order: list[int],
) -> tuple[int, int, int, int]:
    legend_h, legend_w = _legend_dims(sample_c0)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, grid.shape[1] - legend_w
    elif corner == "bl":
        r0, c0 = grid.shape[0] - legend_h, 0
    else:
        r0, c0 = grid.shape[0] - legend_h, grid.shape[1] - legend_w

    grid[r0 : r0 + legend_h, c0 : c0 + legend_w] = BG
    grid[r0, c0 : c0 + legend_w] = MAROON
    grid[r0 + legend_h - 1, c0 : c0 + legend_w] = MAROON
    grid[r0 : r0 + legend_h, c0] = MAROON
    grid[r0 : r0 + legend_h, c0 + legend_w - 1] = MAROON

    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    for row_idx, count in enumerate(row_order):
        band_r0 = inner_r0 + row_idx * (LEGEND_ROW_H + LEGEND_ROW_GAP)
        grid[band_r0 + 1, inner_c0 : inner_c0 + count] = ORANGE

        pattern = count_to_pattern[count]
        sample = grid[band_r0 : band_r0 + LEGEND_SAMPLE_H, inner_c0 + sample_c0 : inner_c0 + sample_c0 + LEGEND_SAMPLE_W]
        sample[pattern] = AZURE

    return r0, c0, r0 + legend_h - 1, c0 + legend_w - 1


def _line_positions(start: int, spans: list[int]) -> list[int]:
    positions = [start]
    cur = start
    for interior in spans:
        cur += interior + 1
        positions.append(cur)
    return positions


def _network_dims(row_spans: list[int], col_spans: list[int]) -> tuple[int, int]:
    height = sum(row_spans) + len(row_spans) + 1
    width = sum(col_spans) + len(col_spans) + 1
    return height, width


def _place_network(
    rng: np.random.Generator,
    h: int,
    w: int,
    legend_bbox: tuple[int, int, int, int],
    net_h: int,
    net_w: int,
) -> tuple[int, int] | None:
    lr0, lc0, lr1, lc1 = legend_bbox
    if lc0 == 0:
        c_min = lc1 + 2
        c_max = w - net_w - 1
    else:
        c_min = 1
        c_max = lc0 - net_w - 2
    if c_min > c_max:
        return None

    r_min = 1
    r_max = h - net_h - 1
    if r_min > r_max:
        return None

    return int(rng.integers(r_min, r_max + 1)), int(rng.integers(c_min, c_max + 1))


def _draw_network(
    grid: np.ndarray,
    r0: int,
    c0: int,
    row_spans: list[int],
    col_spans: list[int],
) -> tuple[list[int], list[int]]:
    r_lines = _line_positions(r0, row_spans)
    c_lines = _line_positions(c0, col_spans)
    for rr in r_lines:
        grid[rr, c0 : c_lines[-1] + 1] = GREY
    for cc in c_lines:
        grid[r0 : r_lines[-1] + 1, cc] = GREY
    return r_lines, c_lines


def _region_cells(r_lines: list[int], c_lines: list[int]) -> list[list[tuple[int, int]]]:
    regions: list[list[tuple[int, int]]] = []
    for i in range(len(r_lines) - 1):
        for j in range(len(c_lines) - 1):
            rows = range(r_lines[i] + 1, r_lines[i + 1])
            cols = range(c_lines[j] + 1, c_lines[j + 1])
            regions.append([(r, c) for r in rows for c in cols])
    return regions


def _build_expected(
    inp: np.ndarray,
    regions: list[list[tuple[int, int]]],
    count_to_pattern: dict[int, np.ndarray],
    region_specs: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    for cells, (count, color) in zip(regions, region_specs):
        _paint_region(out, cells, count_to_pattern[count], color)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    row_cells = int(rng.integers(GEN_ROW_CELL_RANGE[0], GEN_ROW_CELL_RANGE[1] + 1))
    col_cells = int(rng.integers(GEN_COL_CELL_RANGE[0], GEN_COL_CELL_RANGE[1] + 1))
    row_spans = [int(rng.integers(*GEN_INTERIOR_SIZE_RANGE)) for _ in range(row_cells)]
    col_spans = [int(rng.integers(*GEN_INTERIOR_SIZE_RANGE)) for _ in range(col_cells)]
    net_h, net_w = _network_dims(row_spans, col_spans)

    sample_c0 = int(rng.choice([5, 6]))
    legend_h, legend_w = _legend_dims(sample_c0)
    min_h = max(legend_h, net_h + 2)
    min_w = legend_w + net_w + 2
    if min_h > 30 or min_w > 30:
        return None

    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))

    grid = np.zeros((h, w), dtype=int)

    pattern_order = list(rng.permutation(len(PATTERN_LIBRARY)))
    count_to_pattern = {
        count: PATTERN_LIBRARY[pattern_order[count - 1]].copy() for count in range(1, 6)
    }
    row_order = [int(v) for v in rng.permutation(np.arange(1, 6))]
    corner = str(rng.choice(["tl", "tr", "bl", "br"]))
    legend_bbox = _draw_legend(grid, corner, sample_c0, count_to_pattern, row_order)

    placed = _place_network(rng, h, w, legend_bbox, net_h, net_w)
    if placed is None:
        return None
    net_r0, net_c0 = placed
    r_lines, c_lines = _draw_network(grid, net_r0, net_c0, row_spans, col_spans)
    regions = _region_cells(r_lines, c_lines)

    counts = [int(rng.integers(1, 6)) for _ in regions]
    if len(regions) > 1 and len(set(counts)) == 1:
        counts[0] = 1 + (counts[0] % 5)
    colors = [int(rng.choice(GEN_MARKER_COLORS)) for _ in regions]

    region_specs = list(zip(counts, colors))
    for cells, (count, color) in zip(regions, region_specs):
        idx = rng.choice(len(cells), size=count, replace=False)
        for i in idx:
            r, c = cells[int(i)]
            grid[r, c] = color

    expected = _build_expected(grid, regions, count_to_pattern, region_specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_OUTER_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_OUTER_RETRIES} attempts")
