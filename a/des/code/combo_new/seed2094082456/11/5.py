from __future__ import annotations

import itertools
import random

import numpy as np


BBox = tuple[int, int, int, int]

# --------------- solver helpers ---------------


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    height, width = map(int, grid.shape)
    components: list[dict] = []

    for start_r, start_c in np.argwhere(mask):
        r = int(start_r)
        c = int(start_c)
        if seen[r, c]:
            continue

        stack = [(r, c)]
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        coords = np.array(cells, dtype=int)
        components.append(
            {
                "color": int(color),
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": _bbox_from_coords(coords),
            }
        )

    return components


def _frame_cells(bbox: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    return cells


def _is_rectangular_frame(component: dict) -> bool:
    bbox = component["bbox"]
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    expected = _frame_cells(bbox)
    if component["area"] != len(expected):
        return False

    actual = {tuple(map(int, rc)) for rc in component["coords"]}
    return actual == expected


def _strictly_contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested rectangular frames from outermost to innermost using the top-to-bottom marker sequence trapped inside the innermost frame.

    Input structure:
    - One color is the dominant background.
    - Several one-cell-thick hollow rectangular frames are nested concentrically,
      each frame forming its own monochrome border component.
    - Inside the innermost frame, isolated single-pixel markers with distinct
      colors are arranged from top to bottom.

    Rule:
    - Detect the nested frames and sort them from outermost to innermost.
    - Detect the single-pixel markers inside the innermost frame and sort them
      from top to bottom to obtain a color instruction list.
    - Recolor frame i with marker color i, counting frames from the outside.
    - If there are fewer markers than frames, leave the remaining deeper frames
      in their original colors.
    - Remove all marker pixels in the output.

    Key insight:
    - The innermost marker column is a reversed color program: the first marker
      does not recolor the nearest frame, but the outermost one.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[int(np.argmax(counts))])

    components: list[dict] = []
    for color in colors:
        color_int = int(color)
        if color_int == bg:
            continue
        components.extend(_find_color_components(g, color_int))

    assert components, "expected non-background components"

    frames = [component for component in components if _is_rectangular_frame(component)]
    assert frames, "expected at least one rectangular frame"
    frames.sort(key=lambda component: _bbox_area(component["bbox"]), reverse=True)
    for outer, inner in zip(frames, frames[1:]):
        assert _strictly_contains(outer["bbox"], inner["bbox"]), "frames must form one nested chain"

    innermost = frames[-1]["bbox"]
    frame_ids = {id(component) for component in frames}
    markers: list[dict] = []
    for component in components:
        if id(component) in frame_ids:
            continue
        assert component["area"] == 1, "non-frame components must be isolated markers"
        row, col = map(int, component["coords"][0])
        assert innermost[0] < row < innermost[2] and innermost[1] < col < innermost[3], (
            "markers must be strictly inside the innermost frame"
        )
        markers.append(component)

    assert markers, "expected at least one marker"
    markers.sort(key=lambda component: (int(component["coords"][0, 0]), int(component["coords"][0, 1])))

    out = g.copy()
    for component in markers:
        row, col = map(int, component["coords"][0])
        out[row, col] = bg

    marker_colors = [int(component["color"]) for component in markers]
    for layer_index, frame in enumerate(frames):
        recolor = marker_colors[layer_index] if layer_index < len(marker_colors) else int(frame["color"])
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = recolor

    return out


# --------------- generator ---------------

FRAME_COUNT_RANGE = (2, 4)
DELTA_CHOICES = (2, 3)
SAME_COLUMN_PROB = 0.5
FULL_MAPPING_PROB = 0.5
MAX_ATTEMPTS = 300


def _sample_spaced_rows(height: int, count: int) -> list[int] | None:
    options = [
        list(rows)
        for rows in itertools.combinations(range(height), count)
        if all(b - a >= 2 for a, b in zip(rows, rows[1:]))
    ]
    if not options:
        return None
    return random.choice(options)


def _expected_output(inp: np.ndarray, bboxes: list[BBox], frame_colors: list[int], markers: list[tuple[int, int, int]], bg: int) -> np.ndarray:
    out = inp.copy()
    for row, col, _color in markers:
        out[row, col] = bg

    marker_colors = [color for _row, _col, color in sorted(markers)]
    for index, bbox in enumerate(bboxes):
        recolor = marker_colors[index] if index < len(marker_colors) else frame_colors[index]
        _draw_frame(out, bbox, recolor)
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    full_mapping = random.random() < FULL_MAPPING_PROB
    marker_count = frame_count if full_mapping else random.randint(1, frame_count - 1)

    deltas = [random.choice(DELTA_CHOICES) for _ in range(frame_count - 1)]
    total_inset = sum(deltas)

    min_inner_h = max(3, 2 * marker_count - 1)
    max_inner_h = 26 - 2 * total_inset
    if max_inner_h < min_inner_h:
        return None

    min_inner_w = 3
    max_inner_w = 26 - 2 * total_inset
    if max_inner_w < min_inner_w:
        return None

    inner_h = random.randint(min_inner_h, min(max_inner_h, min_inner_h + 6))
    inner_w = random.randint(min_inner_w, min(max_inner_w, min_inner_w + 6))

    outer_h = inner_h + 2 + 2 * total_inset
    outer_w = inner_w + 2 + 2 * total_inset
    if outer_h > 28 or outer_w > 28:
        return None

    extra_rows = random.randint(2, 30 - outer_h)
    extra_cols = random.randint(2, 30 - outer_w)
    top_margin = random.randint(1, extra_rows - 1)
    left_margin = random.randint(1, extra_cols - 1)
    bottom_margin = extra_rows - top_margin
    right_margin = extra_cols - left_margin

    height = outer_h + top_margin + bottom_margin
    width = outer_w + left_margin + right_margin

    bg = random.randrange(10)
    needed_colors = frame_count + marker_count
    palette = random.sample([color for color in range(10) if color != bg], needed_colors)
    frame_colors = palette[:frame_count]
    marker_colors = palette[frame_count:]

    inp = np.full((height, width), bg, dtype=int)
    outer_bbox = (
        top_margin,
        left_margin,
        top_margin + outer_h - 1,
        left_margin + outer_w - 1,
    )
    bboxes = [outer_bbox]
    for delta in deltas:
        r0, c0, r1, c1 = bboxes[-1]
        bboxes.append((r0 + delta, c0 + delta, r1 - delta, c1 - delta))

    for bbox, color in zip(bboxes, frame_colors):
        _draw_frame(inp, bbox, color)

    innermost = bboxes[-1]
    inner_rows = innermost[2] - innermost[0] - 1
    inner_cols = innermost[3] - innermost[1] - 1
    rows_local = _sample_spaced_rows(inner_rows, marker_count)
    if rows_local is None:
        return None

    if random.random() < SAME_COLUMN_PROB:
        cols_local = [random.randrange(inner_cols)] * marker_count
    else:
        cols_local = [random.randrange(inner_cols) for _ in range(marker_count)]

    markers: list[tuple[int, int, int]] = []
    for local_r, local_c, color in zip(rows_local, cols_local, marker_colors):
        row = innermost[0] + 1 + local_r
        col = innermost[1] + 1 + local_c
        inp[row, col] = color
        markers.append((row, col, color))

    expected = _expected_output(inp, bboxes, frame_colors, markers, bg)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame recoloring input")
