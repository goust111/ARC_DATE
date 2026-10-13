from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
TEXTURE_MASKS = (
    np.array([[1, 0], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 0]], dtype=int),
    np.array([[1, 0], [0, 0]], dtype=int),
    np.array([[0, 1], [0, 0]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
)

GEN_RETRIES = 400
PLACEMENT_RETRIES = 160


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _find_regions(grid: np.ndarray) -> list[dict[str, int]]:
    mask = np.zeros_like(grid, dtype=bool)
    if grid.shape[0] > 2:
        mask[2:, :] = grid[2:, :] == GREY

    regions: list[dict[str, int]] = []
    for pts in _components(mask):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        expected_area = 2 * height + 2 * width - 4
        if len(pts) != expected_area:
            continue

        pset = set(pts)
        valid = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_perimeter = r in (r0, r1) or c in (c0, c1)
                if ((r, c) in pset) != on_perimeter:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere(interior != BLACK)
        assert markers.shape[0] == 1, "each region must contain exactly one marker pixel"
        mr, mc = map(int, markers[0])
        marker = int(interior[mr, mc])
        assert marker not in (BLACK, GREY), "marker color must be distinct from background and outline"
        regions.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "marker": marker})

    regions.sort(key=lambda region: (region["r0"], region["c0"]))
    assert regions, "expected at least one grey rectangular region"
    markers = [region["marker"] for region in regions]
    assert len(markers) == len(set(markers)), "region markers must be unique"
    return regions


def _parse_legend(grid: np.ndarray, marker_colors: set[int]) -> dict[int, np.ndarray]:
    assert grid.shape[0] >= 2, "legend requires the top two rows"
    top = grid[:2, :]
    h, w = top.shape
    assert h == 2

    legend: dict[int, np.ndarray] = {}
    for c in range(w - 2):
        marker = int(top[0, c])
        if marker != int(top[1, c]) or marker == BLACK:
            continue
        if marker not in marker_colors or marker in legend:
            continue
        if c > 0 and not np.all(top[:, c - 1] == BLACK):
            continue
        if c + 3 < w and not np.all(top[:, c + 3] == BLACK):
            continue
        legend[marker] = top[:, c + 1 : c + 3].copy()

    assert set(legend) == marker_colors, "legend entries must match the region marker colors"
    return legend


def _tile_region(out: np.ndarray, region: dict[str, int], texture: np.ndarray) -> None:
    r0, c0, r1, c1 = region["r0"], region["c0"], region["r1"], region["c1"]
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            out[r, c] = int(texture[(r - (r0 + 1)) % 2, (c - (c0 + 1)) % 2])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed rectangular region by tiling the 2x2 texture named by its interior marker color.

    Input structure:
    - The background is black.
    - The top two rows contain a legend: each entry has a 2x1 vertical color tag followed by an adjacent 2x2 texture sample, with black separator columns between entries.
    - Below the legend are one-pixel-thick grey rectangular outlines, and each outlined interior contains exactly one marker pixel of a unique color.

    Rule:
    - Keep the legend and all grey outlines unchanged.
    - For each grey rectangle, read its unique interior marker color and find the legend entry whose 2x1 tag has that same color.
    - Tile the entry's 2x2 texture across the rectangle interior, anchored at the interior's top-left cell, covering the original marker and every black interior cell.

    Key insight:
    - The single colored pixel inside each closed region is a lookup key that selects one texture sample from the top legend before the spatial tiling step happens.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a non-empty 2D grid"

    regions = _find_regions(g)
    legend = _parse_legend(g, {region["marker"] for region in regions})

    out = g.copy()
    for region in regions:
        _tile_region(out, region, legend[region["marker"]])
    return out


def _build_texture(a: int, b: int) -> np.ndarray:
    mask = random.choice(TEXTURE_MASKS)
    tex = np.where(mask == 1, a, b).astype(int)
    assert len(set(map(int, tex.reshape(-1)))) == 2
    return tex


def _draw_legend(grid: np.ndarray, order: list[int], textures: dict[int, np.ndarray], left_margin: int) -> None:
    c = left_margin
    for i, marker in enumerate(order):
        grid[:2, c] = marker
        grid[:2, c + 1 : c + 3] = textures[marker]
        c += 3
        if i != len(order) - 1:
            c += 1


def _try_place_regions(grid: np.ndarray, markers: list[int]) -> list[dict[str, int]] | None:
    h, w = map(int, grid.shape)
    reserved = np.zeros((h, w), dtype=bool)
    reserved[:3, :] = True  # legend rows plus a black spacer row
    regions: list[dict[str, int]] = []

    for marker in markers:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            inner_h = random.randint(2, 7)
            inner_w = random.randint(2, 8)
            box_h = inner_h + 2
            box_w = inner_w + 2
            if box_h + 4 > h or box_w + 2 > w:
                continue

            r0 = random.randint(3, h - box_h - 1)
            c0 = random.randint(1, w - box_w - 1)
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            pr0 = r0 - 1
            pc0 = c0 - 1
            pr1 = r1 + 1
            pc1 = c1 + 1
            if reserved[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue

            grid[r0, c0 : c1 + 1] = GREY
            grid[r1, c0 : c1 + 1] = GREY
            grid[r0 : r1 + 1, c0] = GREY
            grid[r0 : r1 + 1, c1] = GREY

            mr = random.randint(r0 + 1, r1 - 1)
            mc = random.randint(c0 + 1, c1 - 1)
            grid[mr, mc] = marker

            reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            regions.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1, "marker": marker})
            placed = True
            break
        if not placed:
            return None

    return regions


def _expected_output(inp: np.ndarray, regions: list[dict[str, int]], textures: dict[int, np.ndarray]) -> np.ndarray:
    out = inp.copy()
    for region in regions:
        _tile_region(out, region, textures[region["marker"]])
    return out


def _generate_attempt() -> np.ndarray | None:
    region_count = random.randint(2, 4)
    markers = random.sample(MARKER_POOL, region_count)
    texture_palette = [color for color in MARKER_POOL if color not in markers]
    assert len(texture_palette) >= 2

    textures: dict[int, np.ndarray] = {}
    seen_textures: set[bytes] = set()
    for marker in markers:
        for _ in range(40):
            a, b = random.sample(texture_palette, 2)
            texture = _build_texture(a, b)
            key = texture.tobytes()
            if key not in seen_textures:
                seen_textures.add(key)
                textures[marker] = texture
                break
        else:
            return None

    order = markers[:]
    random.shuffle(order)
    left_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    legend_width = left_margin + region_count * 3 + (region_count - 1) + right_margin

    min_h = max(12, 6 + region_count * 3)
    min_w = max(legend_width, 11 + region_count * 2)
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    grid = np.full((h, w), BLACK, dtype=int)
    _draw_legend(grid, order, textures, left_margin)

    region_order = markers[:]
    random.shuffle(region_order)
    regions = _try_place_regions(grid, region_order)
    if regions is None:
        return None

    expected = _expected_output(grid, regions, textures)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid texture-tiling ARC input")
