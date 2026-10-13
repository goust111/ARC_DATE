from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIVIDER = 9
PATTERN_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
QUADRANTS = ("tl", "tr", "bl", "br")
MAX_GEN_ATTEMPTS = 400


def _on_boundary(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _find_divider_spec(grid: np.ndarray) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for color in map(int, np.unique(g)):
        coords = np.argwhere(g == color)
        if len(coords) != 4:
            continue
        if not all(_on_boundary(g.shape, int(r), int(c)) for r, c in coords):
            continue

        edge_markers = {
            "top": [],
            "bottom": [],
            "left": [],
            "right": [],
        }
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            if r == 0:
                edge_markers["top"].append((r, c))
            if r == h - 1:
                edge_markers["bottom"].append((r, c))
            if c == 0:
                edge_markers["left"].append((r, c))
            if c == w - 1:
                edge_markers["right"].append((r, c))

        if len(edge_markers["top"]) != 1 or len(edge_markers["bottom"]) != 1:
            continue
        if len(edge_markers["left"]) != 1 or len(edge_markers["right"]) != 1:
            continue

        col = edge_markers["top"][0][1]
        row = edge_markers["left"][0][0]
        if edge_markers["bottom"][0][1] != col or edge_markers["right"][0][0] != row:
            continue
        if row in (0, h - 1) or col in (0, w - 1):
            continue
        return color, row, col
    raise AssertionError("expected one boundary-only divider color with one row pair and one column pair")


def _region_slices(row: int, col: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(0, row), slice(0, col)),
        "tr": (slice(0, row), slice(col + 1, None)),
        "bl": (slice(row + 1, None), slice(0, col)),
        "br": (slice(row + 1, None), slice(col + 1, None)),
    }


def _content_count(region: np.ndarray, divider: int) -> int:
    return int(np.count_nonzero((region != BG) & (region != divider)))


def _content_coords(region: np.ndarray, divider: int) -> set[tuple[int, int]]:
    return {
        (int(r), int(c))
        for r, c in np.argwhere((region != BG) & (region != divider))
    }


def _analyze_regions(grid: np.ndarray) -> tuple[int, int, int, dict[str, np.ndarray], dict[str, int]]:
    divider, row, col = _find_divider_spec(grid)
    g = np.asarray(grid, dtype=int)
    regions: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name, (rs, cs) in _region_slices(row, col).items():
        region = g[rs, cs]
        regions[name] = region
        counts[name] = _content_count(region, divider)
    return divider, row, col, regions, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the boundary-indicated cross, pick the densest quadrant as the template, and mark missing template pixels with the divider color.

    Input structure:
    - The grid has a black background plus exactly four divider-color boundary markers.
    - One marker pair sits on the top/bottom edges and fixes a vertical full-height divider line.
    - One marker pair sits on the left/right edges and fixes a horizontal full-width divider line.
    - The four resulting rectangular regions contain colored pattern pixels other than black and the divider color.

    Rule:
    - Find the boundary-only marker color and extend it into the full vertical and horizontal divider lines, overwriting any cells on those lines.
    - Treat the divider row and divider column as separators and split the remaining cells into four regions.
    - Count non-background, non-divider pixels in each region and use the unique maximum-count region as the template.
    - For each other region, compare cells by region-relative coordinates; whenever the template has a colored pixel and the current region has black at that coordinate, fill that cell with the divider color.
    - Keep pre-existing non-background pixels unchanged and keep the divider lines unchanged.

    Key insight:
    - The same color plays two roles at once: boundary markers become partition lines, and that divider color is also the symbol used to mark template-covered cells missing from the sparser regions.
    """
    g = np.asarray(grid, dtype=int)
    divider, row, col, _regions, counts = _analyze_regions(g)

    out = g.copy()
    out[row, :] = divider
    out[:, col] = divider

    line_free_regions = {
        name: out[rs, cs]
        for name, (rs, cs) in _region_slices(row, col).items()
    }
    max_count = max(counts.values())
    templates = [name for name, count in counts.items() if count == max_count]
    assert len(templates) == 1, "template region must be uniquely densest"
    template_name = templates[0]
    template = line_free_regions[template_name]

    for name, region in line_free_regions.items():
        if name == template_name:
            continue
        overlap_h = min(template.shape[0], region.shape[0])
        overlap_w = min(template.shape[1], region.shape[1])
        for r in range(overlap_h):
            for c in range(overlap_w):
                if template[r, c] not in (BG, divider) and region[r, c] == BG:
                    region[r, c] = divider
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _grow_connected_cells(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
    count: int,
) -> list[tuple[int, int]]:
    inner_rows = list(range(1, region_h - 1))
    inner_cols = list(range(1, region_w - 1))
    assert inner_rows and inner_cols
    start = (int(rng.choice(inner_rows)), int(rng.choice(inner_cols)))
    cells = [start]
    cell_set = {start}
    while len(cells) < count:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, region_h, region_w):
                if nr in (0, region_h - 1) or nc in (0, region_w - 1):
                    continue
                if (nr, nc) not in cell_set:
                    frontier.append((nr, nc))
        if not frontier:
            return _grow_connected_cells(rng, region_h, region_w, count)
        nxt = frontier[int(rng.integers(0, len(frontier)))]
        cell_set.add(nxt)
        cells.append(nxt)

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1 or len(cols) == 1:
        return _grow_connected_cells(rng, region_h, region_w, count)
    return cells


def _make_template_region(
    rng: np.random.Generator,
    region_h: int,
    region_w: int,
) -> np.ndarray:
    region = np.zeros((region_h, region_w), dtype=int)
    max_cells = min(8, (region_h - 2) * (region_w - 2))
    count = int(rng.integers(3, max_cells + 1))
    cells = _grow_connected_cells(rng, region_h, region_w, count)

    n_colors = int(rng.integers(1, min(3, count) + 1))
    palette = list(rng.choice(PATTERN_COLORS, size=n_colors, replace=False))
    colors = [int(palette[int(rng.integers(0, len(palette)))]) for _ in cells]
    if n_colors > 1 and len(set(colors)) == 1:
        colors[-1] = palette[-1]

    for (r, c), color in zip(cells, colors):
        region[r, c] = color
    return region


def _subset_region(
    rng: np.random.Generator,
    template: np.ndarray,
    keep_count: int,
) -> np.ndarray:
    region = np.zeros_like(template)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template != BG)]
    rng.shuffle(coords)
    keep = set(coords[:keep_count])
    for r, c in keep:
        region[r, c] = int(template[r, c])
    return region


def _place_region(grid: np.ndarray, name: str, row: int, col: int, region: np.ndarray) -> None:
    rs, cs = _region_slices(row, col)[name]
    grid[rs, cs] = region


def _expected_output_from_regions(
    inp: np.ndarray,
    row: int,
    col: int,
    template_name: str,
) -> np.ndarray:
    out = inp.copy()
    out[row, :] = DIVIDER
    out[:, col] = DIVIDER
    regions = {
        name: out[rs, cs]
        for name, (rs, cs) in _region_slices(row, col).items()
    }
    template = regions[template_name]
    for name, region in regions.items():
        if name == template_name:
            continue
        for r, c in np.argwhere(template != BG):
            rr = int(r)
            cc = int(c)
            if template[rr, cc] != DIVIDER and region[rr, cc] == BG:
                region[rr, cc] = DIVIDER
    return out


def _validate_semantics(
    inp: np.ndarray,
    template_name: str,
    template_region: np.ndarray,
    non_template_regions: dict[str, np.ndarray],
) -> None:
    divider, row, col, regions, counts = _analyze_regions(inp)
    assert divider == DIVIDER
    assert np.count_nonzero(inp == DIVIDER) == 4
    assert np.all(inp[1:-1, 1:-1] != DIVIDER)
    assert len({region.shape for region in regions.values()}) == 1
    assert template_name == max(counts, key=counts.get)
    assert list(counts.values()).count(counts[template_name]) == 1

    template_coords = _content_coords(template_region, DIVIDER)
    assert counts[template_name] == len(template_coords)
    for name, region in non_template_regions.items():
        region_coords = _content_coords(region, DIVIDER)
        assert region_coords < template_coords, "each non-template region must be a strict subset"
        assert counts[name] == len(region_coords)
        for r, c in region_coords:
            assert region[r, c] == template_region[r, c]

    expected = _expected_output_from_regions(inp, row, col, template_name)
    solved = main(inp)
    assert np.array_equal(solved, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = np.random.default_rng()
        region_h = int(rng.integers(5, 8))
        region_w = int(rng.integers(5, 8))
        h = 2 * region_h + 1
        w = 2 * region_w + 1
        row = region_h
        col = region_w

        template_name = random.choice(QUADRANTS)
        template_region = _make_template_region(rng, region_h, region_w)
        template_count = int(np.count_nonzero(template_region))
        if template_count < 3:
            continue

        other_names = [name for name in QUADRANTS if name != template_name]
        non_template_regions: dict[str, np.ndarray] = {}
        for name in other_names:
            keep_count = int(rng.integers(1, template_count))
            non_template_regions[name] = _subset_region(rng, template_region, keep_count)

        counts = [int(np.count_nonzero(non_template_regions[name])) for name in other_names]
        if not all(count < template_count for count in counts):
            continue

        inp = np.zeros((h, w), dtype=int)
        regions = {template_name: template_region, **non_template_regions}
        for name in QUADRANTS:
            _place_region(inp, name, row, col, regions[name])

        inp[0, col] = DIVIDER
        inp[h - 1, col] = DIVIDER
        inp[row, 0] = DIVIDER
        inp[row, w - 1] = DIVIDER

        try:
            _validate_semantics(inp, template_name, template_region, non_template_regions)
        except AssertionError:
            continue
        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
