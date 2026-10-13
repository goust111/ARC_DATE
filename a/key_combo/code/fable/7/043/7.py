import random
from collections import Counter, deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair gray blocks with right-edge ports by shared rank, then recolor and connect them with L paths.

    Input structure:
    - The background is black.
    - The left side contains several gray 4-connected solid blocks whose areas are all different.
    - The last column contains the same number of single-pixel colored ports on distinct rows.

    Rule:
    - Rank the gray blocks by increasing area and rank the ports by increasing row index.
    - Pair equal ranks, recolor each whole gray block to its port color, then connect from the block's right edge:
      move vertically on the block's rightmost occupied column to the port row, then move horizontally to the port.
    - Only background cells along the path are painted; existing non-background pixels, including the port, stay unchanged.

    Key insight:
    - The matching is not geometric or distance-based: both sides are sorted independently, and equal ranks determine
      both the recoloring target and the non-crossing L-shaped connection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    bg = get_bg(g)
    assert bg == Color.BLACK

    ports = [(r, int(g[r, w - 1])) for r in range(h) if int(g[r, w - 1]) not in (bg, Color.GREY)]
    assert ports
    assert len({r for r, _ in ports}) == len(ports)
    assert len({color for _, color in ports}) == len(ports)

    allowed_colors = {bg, Color.GREY} | {color for _, color in ports}
    assert set(map(int, np.unique(g))) <= allowed_colors
    assert np.all((g[:, :-1] == bg) | (g[:, :-1] == Color.GREY))

    blocks = components_mask(g == Color.GREY)
    assert len(blocks) == len(ports)
    areas = [int(len(coords)) for coords in blocks]
    assert len(set(areas)) == len(areas)

    ranked_blocks = [coords for _, coords in sorted(zip(areas, blocks), key=lambda item: item[0])]
    ranked_ports = sorted(ports, key=lambda item: item[0])

    out = g.copy()
    for coords, (port_row, port_color) in zip(ranked_blocks, ranked_ports):
        out[coords[:, 0], coords[:, 1]] = int(port_color)

        edge_c = int(coords[:, 1].max())
        edge_rows = coords[coords[:, 1] == edge_c][:, 0]
        start_row = int(edge_rows[np.argmin(np.abs(edge_rows - int(port_row)))])

        r0, r1 = sorted((start_row, int(port_row)))
        for rr in range(r0, r1 + 1):
            if int(out[rr, edge_c]) == bg:
                out[rr, edge_c] = int(port_color)

        for cc in range(edge_c, w):
            if int(out[int(port_row), cc]) == bg:
                out[int(port_row), cc] = int(port_color)

    return out


PORT_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

GEN_RETRIES = 500


def _choose_band_heights(count: int) -> list[int] | None:
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(1, 2)
    gaps = [random.randint(1, 2) for _ in range(count - 1)]
    band_heights = [random.randint(4, 6) for _ in range(count)]
    total = top_margin + bottom_margin + sum(gaps) + sum(band_heights)
    if total > 30:
        return None
    return [top_margin, *band_heights, *gaps, bottom_margin]


def _build_bands(count: int) -> tuple[int, list[tuple[int, int]]] | None:
    spec = _choose_band_heights(count)
    if spec is None:
        return None

    top_margin = spec[0]
    band_heights = spec[1 : 1 + count]
    gaps = spec[1 + count : 1 + count + count - 1]
    bottom_margin = spec[-1]

    bands: list[tuple[int, int]] = []
    row = top_margin
    for i, band_h in enumerate(band_heights):
        bands.append((row, row + band_h - 1))
        row += band_h
        if i < count - 1:
            row += gaps[i]
    height = top_margin + sum(band_heights) + sum(gaps) + bottom_margin
    return height, bands


def _rectangle_choices(band_h: int, max_width: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for rect_h in range(2, min(4, band_h - 1) + 1):
        for rect_w in range(2, max_width + 1):
            choices.append((rect_h, rect_w))
    random.shuffle(choices)
    return choices


def _assign_rectangles(bands: list[tuple[int, int]], width: int) -> list[tuple[int, int]] | None:
    max_width = width // 2 - 2
    if max_width < 2:
        return None

    candidates = [_rectangle_choices(b1 - b0 + 1, max_width) for b0, b1 in bands]
    for opts in candidates:
        if not opts:
            return None

    def search(i: int, prev_area: int, used: set[int], chosen: list[tuple[int, int]]):
        if i == len(candidates):
            return chosen.copy()
        options = candidates[i][:]
        random.shuffle(options)
        for rect_h, rect_w in options:
            area = rect_h * rect_w
            if area <= prev_area or area in used:
                continue
            chosen.append((rect_h, rect_w))
            used.add(area)
            result = search(i + 1, area, used, chosen)
            if result is not None:
                return result
            used.remove(area)
            chosen.pop()
        return None

    return search(0, 0, set(), [])


def _choose_port_row(band: tuple[int, int], block_box: tuple[int, int, int, int], force_vertical: bool) -> int:
    band_r0, band_r1 = band
    block_r0, _, block_r1, _ = block_box
    modes: list[str] = ["inside"]
    if band_r0 <= block_r0 - 1:
        modes.append("above")
    if block_r1 + 1 <= band_r1:
        modes.append("below")

    if force_vertical:
        vertical_modes = [mode for mode in modes if mode != "inside"]
        if vertical_modes:
            mode = random.choice(vertical_modes)
        else:
            mode = "inside"
    else:
        mode = random.choice(modes)

    if mode == "inside":
        return random.randint(block_r0, block_r1)
    if mode == "above":
        return random.randint(band_r0, block_r0 - 1)
    return random.randint(block_r1 + 1, band_r1)


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 5)
    band_data = _build_bands(count)
    if band_data is None:
        return None
    height, bands = band_data

    width = random.randint(16, 28)
    rects = _assign_rectangles(bands, width)
    if rects is None:
        return None

    grid = np.full((height, width), Color.BLACK, dtype=int)
    colors = random.sample(PORT_COLORS, count)

    left_limit = width // 2 - 1
    if left_limit < 4:
        return None

    vertical_index = random.randrange(count) if random.random() < 0.75 else None

    port_rows: list[int] = []
    for i, ((band_r0, band_r1), (rect_h, rect_w), color) in enumerate(zip(bands, rects, colors)):
        max_c0 = left_limit - rect_w
        if max_c0 < 1:
            return None
        c0 = random.randint(1, max_c0)

        max_r0 = band_r1 - rect_h + 1
        if max_r0 < band_r0:
            return None
        r0 = random.randint(band_r0, max_r0)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1

        grid[r0 : r1 + 1, c0 : c1 + 1] = Color.GREY

        port_row = _choose_port_row(
            (band_r0, band_r1),
            (r0, c0, r1, c1),
            force_vertical=(vertical_index == i),
        )
        port_rows.append(port_row)
        grid[port_row, width - 1] = int(color)

    if len(set(port_rows)) != count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
