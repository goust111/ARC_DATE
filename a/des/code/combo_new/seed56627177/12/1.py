from __future__ import annotations

from collections import deque
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            components.append({"color": color, "coords": np.array(coords, dtype=int)})

    return components


def _rect_frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected_area = 2 * height + 2 * width - 4
    if int(coords.shape[0]) != expected_area:
        return None

    coord_set = {(int(r), int(c)) for r, c in coords}
    for r in range(r0, r1 + 1):
        if (r, c0) not in coord_set or (r, c1) not in coord_set:
            return None
    for c in range(c0, c1 + 1):
        if (r0, c) not in coord_set or (r1, c) not in coord_set:
            return None

    return r0, c0, r1, c1


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for component in _monochrome_components(grid, bg):
        bbox = _rect_frame_bbox(component["coords"])
        if bbox is None:
            continue
        frames.append({"color": int(component["color"]), "bbox": bbox})

    frames.sort(key=lambda item: item["bbox"])
    return frames


def _swap_by_frames(grid: np.ndarray, frames: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    accounted = np.zeros(grid.shape, dtype=bool)

    for frame in frames:
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        border_mask = _frame_mask(grid.shape, (r0, c0, r1, c1))
        assert not np.any(accounted & border_mask), "frames must be disjoint"
        accounted |= border_mask

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = interior != bg
        assert marker_mask.any(), "each frame must contain at least one marker"
        marker_colors = np.unique(interior[marker_mask])
        assert marker_colors.size == 1, "markers inside a frame must share one color"
        marker_color = int(marker_colors[0])
        assert marker_color != frame_color, "marker color must differ from frame color"

        marker_coords = np.argwhere(marker_mask)
        for rr, cc in marker_coords:
            gr = int(r0 + 1 + rr)
            gc = int(c0 + 1 + cc)
            assert not accounted[gr, gc], "markers must not overlap frames or other markers"
            accounted[gr, gc] = True

        out[r0 : r1 + 1, c0] = marker_color
        out[r0 : r1 + 1, c1] = marker_color
        out[r0, c0 : c1 + 1] = marker_color
        out[r1, c0 : c1 + 1] = marker_color

        interior_out = out[r0 + 1 : r1, c0 + 1 : c1]
        interior_out[marker_mask] = frame_color

    assert np.all((grid == bg) | accounted), "all foreground pixels must belong to some frame or its markers"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each rectangular frame color with the unique marker color scattered inside that frame.

    Input structure:
    - The background is the dominant color.
    - Several disjoint one-pixel-thick rectangular frames appear in different colors.
    - Inside each frame, one or more single-cell marker dots are scattered on the background.
      All markers inside one frame share one color, and that color differs from the frame color.

    Rule:
    - Detect every closed rectangular frame and inspect only its interior cells.
    - Read the unique non-background marker color inside that frame.
    - Repaint the entire frame border to the marker color, and repaint every marker dot inside
      the frame to the frame's original border color.
    - Leave all background pixels unchanged.

    Key insight:
    - The transformation is a local color exchange between each frame and the dots enclosed by
      that frame, so the crucial step is assigning each interior marker set to its containing frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"
    return _swap_by_frames(g, frames, bg)


# --------------- generator ---------------

COLOR_COUNT = 10
RECT_COUNT_RANGE = (1, 4)
GRID_SIZE_MIN = 14
GRID_SIZE_MAX = 30
RECT_SIDE_MIN = 4
RECT_SIDE_CAPS = {1: 12, 2: 10, 3: 9, 4: 8}
OUTER_MARGIN = 1
MAX_MARKERS_PER_FRAME = 6
PLACEMENT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 400


def _choose_grid_shape(n_rects: int) -> tuple[int, int]:
    min_side = max(GRID_SIZE_MIN, 10 + 2 * n_rects)
    rows = random.randint(min_side, GRID_SIZE_MAX)
    cols = random.randint(min_side, GRID_SIZE_MAX)
    if random.random() < 0.5:
        delta = random.randint(-4, 4)
        cols = min(GRID_SIZE_MAX, max(min_side, rows + delta))
    return rows, cols


def _sample_rect_sizes(n_rects: int, rows: int, cols: int) -> list[tuple[int, int]]:
    side_cap = min(RECT_SIDE_CAPS[n_rects], rows - 2 * OUTER_MARGIN, cols - 2 * OUTER_MARGIN)
    assert side_cap >= RECT_SIDE_MIN

    sizes = [
        (
            random.randint(RECT_SIDE_MIN, side_cap),
            random.randint(RECT_SIDE_MIN, side_cap),
        )
        for _ in range(n_rects)
    ]
    sizes.sort(key=lambda item: item[0] * item[1], reverse=True)
    return sizes


def _place_rectangles(rows: int, cols: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    reserved = np.zeros((rows, cols), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []

    for height, width in sizes:
        candidates: list[tuple[int, int]] = []
        for r0 in range(OUTER_MARGIN, rows - height - OUTER_MARGIN + 1):
            for c0 in range(OUTER_MARGIN, cols - width - OUTER_MARGIN + 1):
                r1 = r0 + height - 1
                c1 = c0 + width - 1
                if reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = True
        placed.append((r0, c0, r1, c1))

    return placed


def _nonadjacent(points: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) == 1:
                return False
    return True


def _sample_marker_positions(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    max_count = min(MAX_MARKERS_PER_FRAME, (len(cells) + 1) // 2)
    count = random.randint(1, max_count)

    for _ in range(80):
        picked = random.sample(cells, count)
        if _nonadjacent(picked):
            return picked

    for parity in random.sample([0, 1], 2):
        pool = [cell for cell in cells if (cell[0] + cell[1]) % 2 == parity]
        if len(pool) >= count:
            return random.sample(pool, count)

    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _expected_output(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        frame_color = int(spec["frame_color"])
        marker_color = int(spec["marker_color"])
        r0, c0, r1, c1 = spec["bbox"]

        out[r0 : r1 + 1, c0] = marker_color
        out[r0 : r1 + 1, c1] = marker_color
        out[r0, c0 : c1 + 1] = marker_color
        out[r1, c0 : c1 + 1] = marker_color

        for r, c in spec["markers"]:
            out[r, c] = frame_color

    assert _background_color(grid) == bg
    return out


def _attempt_case() -> np.ndarray | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    rows, cols = _choose_grid_shape(n_rects)
    sizes = _sample_rect_sizes(n_rects, rows, cols)
    placements = _place_rectangles(rows, cols, sizes)
    if placements is None:
        return None

    bg = random.randrange(COLOR_COUNT)
    palette = [color for color in range(COLOR_COUNT) if color != bg]
    random.shuffle(palette)
    chosen = palette[: 2 * n_rects]
    frame_colors = chosen[:n_rects]
    marker_colors = chosen[n_rects:]

    grid = np.full((rows, cols), bg, dtype=int)
    specs: list[dict] = []

    for bbox, frame_color, marker_color in zip(placements, frame_colors, marker_colors):
        markers = _sample_marker_positions(*bbox)
        if markers is None:
            return None

        _draw_frame(grid, bbox, frame_color)
        for r, c in markers:
            grid[r, c] = marker_color

        specs.append(
            {
                "bbox": bbox,
                "frame_color": int(frame_color),
                "marker_color": int(marker_color),
                "markers": markers,
            }
        )

    if _background_color(grid) != bg:
        return None

    expected = _expected_output(grid, specs, bg)
    if np.array_equal(grid, expected):
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-marker swap input")
