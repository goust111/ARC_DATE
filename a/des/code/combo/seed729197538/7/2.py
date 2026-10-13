from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_POOL = (1, 2, 3, 4, 6, 7, 8)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_palette(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    palette_mask = (g != BLACK) & (g != GREY) & (g != MAROON)
    pts = [(int(r), int(c)) for r, c in np.argwhere(palette_mask)]
    assert pts, "palette strip not found"

    rows = {r for r, _ in pts}
    cols = {c for _, c in pts}
    if len(rows) == 1:
        row = next(iter(rows))
        assert row in (0, h - 1), "horizontal palette must lie on the border"
        pts.sort(key=lambda p: p[1])
    elif len(cols) == 1:
        col = next(iter(cols))
        assert col in (0, w - 1), "vertical palette must lie on the border"
        pts.sort(key=lambda p: p[0])
    else:
        raise AssertionError("palette pixels must lie on one border row or one border column")

    return [int(g[r, c]) for r, c in pts]


def _find_regions(grid: np.ndarray) -> list[tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int, int, int]]] = []
    for comp in _components(g != GREY):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        markers = [(r, c) for r, c in comp if int(g[r, c]) == MAROON]
        if not markers:
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        regions.append((comp, markers, bbox))
    assert regions, "no enclosed marked regions found"
    return regions


def _marker_orientation(markers: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> str:
    rows = {r for r, _ in markers}
    cols = {c for _, c in markers}
    if len(rows) == 1 and len(cols) > 1:
        return "horizontal"
    if len(cols) == 1 and len(rows) > 1:
        return "vertical"
    if len(rows) == 1 and len(cols) == 1:
        r0, c0, r1, c1 = bbox
        return "horizontal" if (c1 - c0) >= (r1 - r0) else "vertical"
    raise AssertionError("markers must align on one row or one column")


def _paint_region(
    out: np.ndarray,
    comp: list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    fill_color: int,
    orientation: str,
) -> None:
    r0, c0, _r1, _c1 = bbox
    for r, c in comp:
        if orientation == "horizontal":
            out[r, c] = fill_color if (r - r0) % 2 == 0 else BLACK
        else:
            out[r, c] = fill_color if (c - c0) % 2 == 0 else BLACK


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    palette = _find_palette(g)
    out = g.copy()

    for comp, markers, bbox in _find_regions(g):
        count = len(markers)
        assert 1 <= count <= len(palette), "marker count must index the palette"
        fill_color = int(palette[count - 1])
        orientation = _marker_orientation(markers, bbox)
        _paint_region(out, comp, bbox, fill_color, orientation)

    return out


def _line_positions(start: int, spans: list[int]) -> list[int]:
    pos = start
    lines = [pos]
    for span in spans:
        pos += span + 1
        lines.append(pos)
    return lines


def _expected_output(
    grid: np.ndarray,
    regions: list[tuple[tuple[int, int, int, int], int, str]],
    palette: list[int],
) -> np.ndarray:
    out = grid.copy()
    for (r0, c0, r1, c1), count, orientation in regions:
        fill_color = int(palette[count - 1])
        if orientation == "horizontal":
            for r in range(r0, r1 + 1):
                out[r, c0 : c1 + 1] = fill_color if (r - r0) % 2 == 0 else BLACK
        else:
            for c in range(c0, c1 + 1):
                out[r0 : r1 + 1, c] = fill_color if (c - c0) % 2 == 0 else BLACK
    return out


def _place_palette(grid: np.ndarray, edge: str, colors: list[int], rng: random.Random) -> None:
    h, w = grid.shape
    n = len(colors)
    if edge == "top":
        start = rng.randint(0, w - n)
        grid[0, start : start + n] = colors
    elif edge == "bottom":
        start = rng.randint(0, w - n)
        grid[h - 1, start : start + n] = colors
    elif edge == "left":
        start = rng.randint(0, h - n)
        grid[start : start + n, 0] = np.array(colors, dtype=int)
    else:
        start = rng.randint(0, h - n)
        grid[start : start + n, w - 1] = np.array(colors, dtype=int)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_len = rng.randint(3, 5)
    palette = rng.sample(list(PALETTE_POOL), palette_len)
    edge = rng.choice(["top", "bottom", "left", "right"])

    region_rows = rng.randint(1, 3)
    region_cols = rng.randint(2, 3)
    cell_min = palette_len
    row_spans = [rng.randint(cell_min, cell_min + 2) for _ in range(region_rows)]
    col_spans = [rng.randint(cell_min, cell_min + 2) for _ in range(region_cols)]

    frame_h = sum(row_spans) + region_rows + 1
    frame_w = sum(col_spans) + region_cols + 1

    top_pad = rng.randint(2, 3) if edge == "top" else rng.randint(1, 2)
    bottom_pad = rng.randint(2, 3) if edge == "bottom" else rng.randint(1, 2)
    left_pad = rng.randint(2, 3) if edge == "left" else rng.randint(1, 2)
    right_pad = rng.randint(2, 3) if edge == "right" else rng.randint(1, 2)

    h = top_pad + frame_h + bottom_pad
    w = left_pad + frame_w + right_pad
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    frame_top = top_pad
    frame_left = left_pad
    row_lines = _line_positions(frame_top, row_spans)
    col_lines = _line_positions(frame_left, col_spans)

    grid[row_lines, frame_left : frame_left + frame_w] = GREY
    grid[frame_top : frame_top + frame_h, col_lines] = GREY

    regions: list[tuple[tuple[int, int, int, int], int, str]] = []
    for rr in range(region_rows):
        for cc in range(region_cols):
            r0 = row_lines[rr] + 1
            r1 = row_lines[rr + 1] - 1
            c0 = col_lines[cc] + 1
            c1 = col_lines[cc + 1] - 1
            count = rng.randint(2, palette_len)
            orientation = rng.choice(["horizontal", "vertical"])
            regions.append(((r0, c0, r1, c1), count, orientation))

            if orientation == "horizontal":
                r = rng.randint(r0, r1)
                c = rng.randint(c0, c1 - count + 1)
                grid[r, c : c + count] = MAROON
            else:
                r = rng.randint(r0, r1 - count + 1)
                c = rng.randint(c0, c1)
                grid[r : r + count, c] = MAROON

    if len(regions) > 1:
        if len({count for _, count, _ in regions}) == 1:
            bbox, count, orientation = regions[-1]
            alt = 2 if count != 2 else palette_len
            regions[-1] = (bbox, alt, orientation)
            r0, c0, r1, c1 = bbox
            grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            if orientation == "horizontal":
                r = rng.randint(r0, r1)
                c = rng.randint(c0, c1 - alt + 1)
                grid[r, c : c + alt] = MAROON
            else:
                r = rng.randint(r0, r1 - alt + 1)
                c = rng.randint(c0, c1)
                grid[r : r + alt, c] = MAROON
        if len({orientation for _, _, orientation in regions}) == 1:
            bbox, count, orientation = regions[0]
            flipped = "vertical" if orientation == "horizontal" else "horizontal"
            regions[0] = (bbox, count, flipped)
            r0, c0, r1, c1 = bbox
            grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            if flipped == "horizontal":
                r = rng.randint(r0, r1)
                c = rng.randint(c0, c1 - count + 1)
                grid[r, c : c + count] = MAROON
            else:
                r = rng.randint(r0, r1 - count + 1)
                c = rng.randint(c0, c1)
                grid[r : r + count, c] = MAROON

    _place_palette(grid, edge, palette, rng)

    expected = _expected_output(grid, regions, palette)
    if not np.array_equal(main(grid), expected):
        return None
    if len(_find_regions(grid)) != len(regions):
        return None
    if _find_palette(grid) != palette:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
