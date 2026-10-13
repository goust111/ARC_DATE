import random
from collections import deque

import numpy as np


BG = 0
DIRECTIONS = ("up", "down", "left", "right")
FRAME_SIZE_CHOICES = (7, 9, 11)
MAX_OBJECTS = 3
MAX_SPIKE_LEN = 6
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80
MARKER_ATTEMPTS = 80


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    mask = g == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
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


def _is_rect_frame(coords: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if not coords:
        return None
    r0, c0, r1, c1 = _component_bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    expected_area = 2 * (height + width) - 4
    if len(coords) != expected_area:
        return None
    coord_set = set(coords)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in coord_set) != on_border:
                return None
    return r0, c0, r1, c1


def _find_frames(g: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for comp in _monochrome_components(g, color):
            bbox = _is_rect_frame(comp)
            if bbox is not None:
                frames.append((color, *bbox))
    frames.sort(key=lambda item: item[1:])
    return frames


def _direction_from_markers(
    marker_rows: np.ndarray,
    marker_cols: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> str:
    frame_center_r = (r0 + r1) / 2.0
    frame_center_c = (c0 + c1) / 2.0
    marker_center_r = float(marker_rows.mean())
    marker_center_c = float(marker_cols.mean())
    dr = marker_center_r - frame_center_r
    dc = marker_center_c - frame_center_c
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    return "up" if dr < 0 else "down"


def _spike_cells(direction: str, r0: int, c0: int, r1: int, c1: int, length: int) -> list[tuple[int, int]]:
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    if direction == "left":
        return [(center_r, c0 - step) for step in range(1, length + 1)]
    if direction == "right":
        return [(center_r, c1 + step) for step in range(1, length + 1)]
    if direction == "up":
        return [(r0 - step, center_c) for step in range(1, length + 1)]
    return [(r1 + step, center_c) for step in range(1, length + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow an outward spike from each rectangular frame using the count and side bias of its interior markers.

    Input structure:
    - Background is black.
    - One or more one-cell-thick monochromatic rectangular outlines appear on the grid.
    - Each frame encloses several non-black single-pixel markers of one shared color, with black space between the markers and the outline.

    Rule:
    - For each frame, count its interior marker pixels to get a spike length N and read their shared color as the spike color.
    - Compare the marker centroid with the geometric center of the frame; the dominant signed offset chooses one of four directions: up, down, left, or right.
    - From the middle cell of that side of the frame, extend a one-cell-wide straight spike of length N outward in the chosen direction.
    - Keep the original outlines and interior markers unchanged.

    Key insight:
    - The marker cloud encodes two independent attributes at once: its size gives spike length, while its centroid bias relative to the frame center gives spike direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame_color, r0, c0, r1, c1 in _find_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = interior != BG
        assert np.any(marker_mask), "each frame must contain at least one marker"

        marker_colors = {int(v) for v in interior[marker_mask]}
        assert len(marker_colors) == 1, "each frame must contain a single marker color"
        marker_color = marker_colors.pop()
        assert marker_color != frame_color, "marker color must differ from the frame color"

        marker_rows, marker_cols = np.where(marker_mask)
        marker_rows = marker_rows + (r0 + 1)
        marker_cols = marker_cols + (c0 + 1)
        direction = _direction_from_markers(marker_rows, marker_cols, r0, c0, r1, c1)
        for sr, sc in _spike_cells(direction, r0, c0, r1, c1, int(marker_rows.size)):
            assert 0 <= sr < g.shape[0] and 0 <= sc < g.shape[1], "spike must stay inside the grid"
            out[sr, sc] = marker_color

    return out


def _output_bbox(r0: int, c0: int, height: int, width: int, direction: str, length: int) -> tuple[int, int, int, int]:
    if direction == "left":
        return r0, c0 - length, r0 + height - 1, c0 + width - 1
    if direction == "right":
        return r0, c0, r0 + height - 1, c0 + width - 1 + length
    if direction == "up":
        return r0 - length, c0, r0 + height - 1, c0 + width - 1
    return r0, c0, r0 + height - 1 + length, c0 + width - 1


def _candidate_origins(
    rows: int,
    cols: int,
    height: int,
    width: int,
    direction: str,
    length: int,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    if direction == "left":
        row_range = range(1, rows - height)
        col_range = range(length + 1, cols - width)
    elif direction == "right":
        row_range = range(1, rows - height)
        col_range = range(1, cols - width - length)
    elif direction == "up":
        row_range = range(length + 1, rows - height)
        col_range = range(1, cols - width)
    else:
        row_range = range(1, rows - height - length)
        col_range = range(1, cols - width)

    origins: list[tuple[int, int]] = []
    for r0 in row_range:
        for c0 in col_range:
            br0, bc0, br1, bc1 = _output_bbox(r0, c0, height, width, direction, length)
            pr0 = max(0, br0 - 1)
            pc0 = max(0, bc0 - 1)
            pr1 = min(rows - 1, br1 + 1)
            pc1 = min(cols - 1, bc1 + 1)
            if not occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                origins.append((r0, c0))
    return origins


def _marker_candidates(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    center_r = height // 2
    center_c = width // 2
    if direction in ("left", "right"):
        row_band = max(1, height // 4)
        rows = range(center_r - row_band, center_r + row_band + 1)
        if direction == "left":
            cols = range(2, center_c)
        else:
            cols = range(center_c + 1, width - 2)
    else:
        col_band = max(1, width // 4)
        cols = range(center_c - col_band, center_c + col_band + 1)
        if direction == "up":
            rows = range(2, center_r)
        else:
            rows = range(center_r + 1, height - 2)

    return [(r, c) for r in rows for c in cols]


def _sample_marker_cells(height: int, width: int, direction: str) -> list[tuple[int, int]] | None:
    candidates = _marker_candidates(height, width, direction)
    if not candidates:
        return None
    max_count = min(MAX_SPIKE_LEN, len(candidates))
    if max_count < 1:
        return None
    for _ in range(MARKER_ATTEMPTS):
        count = random.randint(1, max_count)
        cells = random.sample(candidates, count)
        rows = np.array([r for r, _c in cells], dtype=float)
        cols = np.array([c for _r, c in cells], dtype=float)
        if _direction_from_markers(rows, cols, 0, 0, height - 1, width - 1) == direction:
            return cells
    return None


def _paint_frame(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _build_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        spike = _spike_cells(spec["direction"], spec["r0"], spec["c0"], spec["r1"], spec["c1"], spec["count"])
        for r, c in spike:
            out[r, c] = spec["marker_color"]
    return out


def _sample_object(rows: int, cols: int, occupied: np.ndarray, marker_color: int) -> dict | None:
    for _ in range(PLACE_ATTEMPTS):
        direction = random.choice(DIRECTIONS)
        height = random.choice(FRAME_SIZE_CHOICES)
        width = random.choice(FRAME_SIZE_CHOICES)
        if height >= rows - 2 or width >= cols - 2:
            continue

        markers = _sample_marker_cells(height, width, direction)
        if markers is None:
            continue
        count = len(markers)

        origins = _candidate_origins(rows, cols, height, width, direction, count, occupied)
        if not origins:
            continue
        r0, c0 = random.choice(origins)
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        br0, bc0, br1, bc1 = _output_bbox(r0, c0, height, width, direction, count)
        pr0 = max(0, br0 - 1)
        pc0 = max(0, bc0 - 1)
        pr1 = min(rows - 1, br1 + 1)
        pc1 = min(cols - 1, bc1 + 1)
        occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
        return {
            "direction": direction,
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "count": count,
            "marker_color": marker_color,
            "markers": [(r0 + mr, c0 + mc) for mr, mc in markers],
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(16, 30)
    cols = random.randint(16, 30)
    object_count = random.randint(1, MAX_OBJECTS)
    border_color = random.randint(1, 9)
    marker_palette = [c for c in range(1, 10) if c != border_color]
    if len(marker_palette) < object_count:
        return None
    marker_colors = random.sample(marker_palette, object_count)

    occupied = np.zeros((rows, cols), dtype=bool)
    specs: list[dict] = []
    for marker_color in marker_colors:
        spec = _sample_object(rows, cols, occupied, marker_color)
        if spec is None:
            return None
        specs.append(spec)

    grid = np.zeros((rows, cols), dtype=int)
    for spec in specs:
        _paint_frame(grid, spec["r0"], spec["c0"], spec["r1"], spec["c1"], border_color)
        for r, c in spec["markers"]:
            grid[r, c] = spec["marker_color"]

    expected = _build_expected(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
