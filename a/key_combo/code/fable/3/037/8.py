from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append((color, np.array(cells, dtype=int)))
    return components


def _band_ranges(marker_rows: list[int], height: int) -> list[tuple[int, int]]:
    bands = []
    for i, row in enumerate(marker_rows):
        if i == 0:
            lo = 0
        else:
            lo = (marker_rows[i - 1] + row) // 2 + 1
        if i == len(marker_rows) - 1:
            hi = height - 1
        else:
            hi = (row + marker_rows[i + 1]) // 2
        bands.append((int(lo), int(hi)))
    return bands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each edge rectangle into the horizontal jurisdiction band of its same-colored marker and dock it at that band's far boundary.

    Input structure:
    - The background is black.
    - Interior singleton pixels act as colored markers, and their rows are all distinct.
    - Solid monochrome rectangles touch the top or bottom border, and each rectangle color matches one marker color.

    Rule:
    - Sort markers by row and split the grid into horizontal bands at the midpoints between consecutive marker rows.
    - Each rectangle moves vertically inward along its own columns into the band owned by the marker of the same color.
    - A top-edge rectangle stops with its bottom edge on that band's bottom row; a bottom-edge rectangle stops with its top edge on that band's top row. Original rectangle cells clear to black, and markers stay unchanged.

    Key insight:
    - The stopping point is neither the matching marker nor the opposite outer border; it is the unseen far edge of the marker's midpoint-defined jurisdiction band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    markers: dict[int, tuple[int, int]] = {}
    rectangles = []
    for color, coords in _find_components(g):
        r0, c0, r1, c1 = _bbox(coords)
        area = int(coords.shape[0])
        touches_top = r0 == 0
        touches_bottom = r1 == h - 1
        touches_left = c0 == 0
        touches_right = c1 == w - 1

        if area == 1 and not (touches_top or touches_bottom or touches_left or touches_right):
            rr, cc = map(int, coords[0])
            assert color not in markers, "marker colors must be unique"
            markers[color] = (rr, cc)
            continue

        assert touches_top ^ touches_bottom, "rectangles must touch exactly one of top/bottom"
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        assert area == rect_h * rect_w, "moving objects must be solid rectangles"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be filled"
        rectangles.append(
            {
                "color": int(color),
                "c0": int(c0),
                "c1": int(c1),
                "height": int(rect_h),
                "from_top": bool(touches_top),
            }
        )

    assert len(markers) >= 2, "expected at least two markers"
    assert rectangles, "expected at least one edge rectangle"

    ordered_markers = sorted((row, color) for color, (row, _col) in markers.items())
    marker_rows = [row for row, _color in ordered_markers]
    assert len(set(marker_rows)) == len(marker_rows), "marker rows must be distinct"
    bands = _band_ranges(marker_rows, h)
    band_by_color = {color: bands[i] for i, (_row, color) in enumerate(ordered_markers)}

    out = np.full_like(g, BG)
    for color, (r, c) in markers.items():
        out[r, c] = color

    for rect in rectangles:
        color = rect["color"]
        assert color in band_by_color, "each rectangle must match a marker color"
        band_lo, band_hi = band_by_color[color]
        rect_h = rect["height"]
        assert rect_h <= band_hi - band_lo + 1, "rectangle must fit inside its jurisdiction band"

        if rect["from_top"]:
            assert rect_h <= band_lo, "top rectangle must start outside its own band"
            dest_r1 = band_hi
            dest_r0 = dest_r1 - rect_h + 1
        else:
            assert rect_h <= h - 1 - band_hi, "bottom rectangle must start outside its own band"
            dest_r0 = band_lo
            dest_r1 = dest_r0 + rect_h - 1

        out[dest_r0 : dest_r1 + 1, rect["c0"] : rect["c1"] + 1] = color

    for color, (r, c) in markers.items():
        out[r, c] = color
    return out


def _sample_marker_rows(height: int, count: int) -> list[int]:
    for _ in range(200):
        rows = []
        first_hi = height - 3 - 3 * (count - 1)
        if first_hi < 2:
            break
        rows.append(random.randint(2, first_hi))
        ok = True
        for i in range(1, count):
            remaining = count - 1 - i
            lo = rows[-1] + 3
            hi = height - 3 - 3 * remaining
            candidates = [r for r in range(lo, hi + 1) if (r - rows[-1]) % 2 == 1]
            if not candidates:
                ok = False
                break
            rows.append(random.choice(candidates))
        if ok:
            return rows
    raise RuntimeError("failed to sample marker rows")


def _choose_rectangles(
    bands: list[tuple[int, int]], height: int, rect_count: int
) -> list[tuple[int, str, int]]:
    options: list[dict[str, int]] = []
    for band_lo, band_hi in bands:
        band_height = band_hi - band_lo + 1
        marker_options: dict[str, int] = {}
        max_top_height = min(band_height, band_lo)
        if max_top_height >= 1:
            marker_options["top"] = int(max_top_height)
        max_bottom_height = min(band_height, height - 1 - band_hi)
        if max_bottom_height >= 1:
            marker_options["bottom"] = int(max_bottom_height)
        options.append(marker_options)

    top_candidates = [i for i, opt in enumerate(options) if "top" in opt]
    bottom_candidates = [i for i, opt in enumerate(options) if "bottom" in opt]
    assert top_candidates and bottom_candidates

    top_idx = random.choice(top_candidates)
    bottom_choices = [i for i in bottom_candidates if i != top_idx]
    bottom_idx = random.choice(bottom_choices)

    assignments: dict[int, str] = {top_idx: "top", bottom_idx: "bottom"}
    remaining = [i for i in range(len(options)) if i not in assignments]
    random.shuffle(remaining)
    for idx in remaining[: max(0, rect_count - 2)]:
        assignments[idx] = random.choice(tuple(options[idx]))

    rectangles = []
    for idx, side in assignments.items():
        max_height = options[idx][side]
        rect_h = random.randint(1, max_height)
        rectangles.append((idx, side, rect_h))
    return rectangles


def _random_gaps(total_slack: int, gap_count: int) -> list[int]:
    gaps = [0] * gap_count
    for _ in range(total_slack):
        gaps[random.randrange(gap_count)] += 1
    return gaps


def _place_rectangles(width: int, rect_widths: list[int]) -> list[tuple[int, int]]:
    interior = width - 2
    base_between = max(0, len(rect_widths) - 1)
    total_rect = sum(rect_widths)
    slack = interior - total_rect - base_between
    assert slack >= 0

    extras = _random_gaps(slack, len(rect_widths) + 1)
    gaps = extras[:]
    for i in range(1, len(gaps) - 1):
        gaps[i] += 1

    spans = []
    col = 1 + gaps[0]
    for i, rect_w in enumerate(rect_widths):
        spans.append((col, col + rect_w - 1))
        col += rect_w + gaps[i + 1]
    return spans


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 29)
    max_markers = min(len(COLOR_POOL), 1 + (height - 5) // 3)
    if max_markers < 2:
        return None

    marker_count = random.randint(2, min(5, max_markers))
    marker_rows = _sample_marker_rows(height, marker_count)
    bands = _band_ranges(marker_rows, height)

    rect_count = random.randint(2, min(4, marker_count))
    rectangles = _choose_rectangles(bands, height, rect_count)
    rect_widths = [random.randint(2, 5) for _ in rectangles]

    min_width = 2 + sum(rect_widths) + max(0, rect_count - 1) + marker_count
    if min_width > 30:
        return None
    width = random.randint(max(12, min_width), 30)

    spans = _place_rectangles(width, rect_widths)
    occupied_cols = {c for c0, c1 in spans for c in range(c0, c1 + 1)}
    free_cols = [c for c in range(1, width - 1) if c not in occupied_cols]
    if len(free_cols) < marker_count:
        return None

    marker_cols = random.sample(free_cols, marker_count)
    marker_colors = random.sample(COLOR_POOL, marker_count)

    grid = np.zeros((height, width), dtype=int)
    for color, row, col in zip(marker_colors, marker_rows, marker_cols):
        grid[row, col] = color

    for (marker_idx, side, rect_h), rect_w, (c0, c1) in zip(rectangles, rect_widths, spans):
        color = marker_colors[marker_idx]
        if side == "top":
            grid[:rect_h, c0 : c1 + 1] = color
        else:
            grid[height - rect_h :, c0 : c1 + 1] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
