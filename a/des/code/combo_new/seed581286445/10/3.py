from __future__ import annotations

import random

import numpy as np


MAX_GRID_SIZE = 30
MIN_FRAME_SIZE = 4
MAX_FRAME_SIZE = 8
MIN_FRAMES = 2
MAX_FRAMES = 4
MIN_GAP = 2
MAX_GENERATION_ATTEMPTS = 400
ALL_COLORS = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            coords = np.asarray(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * (h + w) - 4:
        return False

    patch = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(patch, expected))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _apply_rule(
    inp: np.ndarray,
    frames: list[dict],
    row_markers: dict[int, tuple[int, int, int]],
    col_markers: dict[int, tuple[int, int, int]],
    bg: int,
) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()

    for r, c, _color in row_markers.values():
        out[r, c] = int(bg)
    for r, c, _color in col_markers.values():
        out[r, c] = int(bg)

    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        _row_r, col_c, col_color = col_markers[idx]
        row_r, _row_c, row_color = row_markers[idx]

        if c0 + 1 <= col_c <= c1 - 1:
            out[r0 + 1 : r1, col_c] = int(col_color)
        if r0 + 1 <= row_r <= r1 - 1:
            out[row_r, c0 + 1 : c1] = int(row_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use isolated row and column markers to draw crossing lines inside each hollow frame.

    Input structure:
    - The background is the dominant color.
    - The foreground consists of several one-pixel-thick hollow rectangular
      frames plus isolated singleton marker pixels outside those frames.
    - For every frame, exactly one singleton lies on a row that cuts through the
      frame's row span, and exactly one singleton lies on a column that cuts
      through the frame's column span.

    Rule:
    - Identify every hollow rectangular frame and every isolated singleton.
    - For each frame, pair the row-overlapping singleton outside the frame
      horizontally with the column-overlapping singleton outside the frame
      vertically.
    - Clear all marker pixels to background, then draw a horizontal line across
      the frame interior on the row marker's row using the row marker's color,
      and a vertical line across the frame interior on the column marker's
      column using the column marker's color.
    - If the two interior strokes cross, the horizontal stroke overwrites the
      crossing cell.

    Key insight:
    The singleton pixels are not noise; they are row and column address markers
    that select where to place two colored strokes inside each matched frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)

    frames: list[dict] = []
    markers: list[tuple[int, int, int]] = []
    for comp in components:
        if int(comp["area"]) == 1:
            r, c = map(int, comp["coords"][0])
            markers.append((r, c, int(comp["color"])))
        elif _is_rect_frame(comp):
            frames.append(comp)
        else:
            raise AssertionError("expected only rectangular frames and singleton markers")

    assert frames, "expected at least one frame"
    assert markers, "expected singleton markers"

    row_markers: dict[int, tuple[int, int, int]] = {}
    col_markers: dict[int, tuple[int, int, int]] = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        row_candidates = [m for m in markers if r0 <= m[0] <= r1 and not (c0 <= m[1] <= c1)]
        col_candidates = [m for m in markers if c0 <= m[1] <= c1 and not (r0 <= m[0] <= r1)]
        assert len(row_candidates) == 1, "each frame must have exactly one row marker"
        assert len(col_candidates) == 1, "each frame must have exactly one column marker"
        row_markers[idx] = row_candidates[0]
        col_markers[idx] = col_candidates[0]

    return _apply_rule(g, frames, row_markers, col_markers, bg)


def _sample_axis_spans(count: int) -> tuple[list[tuple[int, int]], int] | None:
    max_len = min(MAX_FRAME_SIZE, (MAX_GRID_SIZE - MIN_GAP * (count + 1)) // count)
    if max_len < MIN_FRAME_SIZE:
        return None

    lengths = [random.randint(MIN_FRAME_SIZE, max_len) for _ in range(count)]
    gaps = [MIN_GAP] * (count + 1)
    slack = MAX_GRID_SIZE - (sum(lengths) + sum(gaps))
    for _ in range(random.randint(0, slack)):
        gaps[random.randrange(count + 1)] += 1

    spans: list[tuple[int, int]] = []
    pos = gaps[0]
    for i, length in enumerate(lengths):
        start = pos
        end = start + length - 1
        spans.append((int(start), int(end)))
        pos = end + 1 + gaps[i + 1]
    return spans, int(pos)


def _occupied_axis_cells(size: int, spans: list[tuple[int, int]]) -> list[int]:
    return [i for i in range(size) if not any(a <= i <= b for a, b in spans)]


def _far_enough(cell: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - pr), abs(c - pc)) > 1 for pr, pc in placed)


def _pick_row_marker(
    bbox: tuple[int, int, int, int], free_cols: list[int], placed: list[tuple[int, int]]
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    rows = list(range(r0 + 1, r1))
    left_cols = [c for c in free_cols if c <= c0 - 2]
    right_cols = [c for c in free_cols if c >= c1 + 2]
    side_cols = []
    if left_cols:
        side_cols.append(left_cols)
    if right_cols:
        side_cols.append(right_cols)
    random.shuffle(side_cols)

    for cols in side_cols:
        cells = [(r, c) for r in rows for c in cols]
        random.shuffle(cells)
        for cell in cells:
            if _far_enough(cell, placed):
                return cell
    return None


def _pick_col_marker(
    bbox: tuple[int, int, int, int], free_rows: list[int], placed: list[tuple[int, int]]
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    cols = list(range(c0 + 1, c1))
    top_rows = [r for r in free_rows if r <= r0 - 2]
    bottom_rows = [r for r in free_rows if r >= r1 + 2]
    side_rows = []
    if top_rows:
        side_rows.append(top_rows)
    if bottom_rows:
        side_rows.append(bottom_rows)
    random.shuffle(side_rows)

    for rows in side_rows:
        cells = [(r, c) for r in rows for c in cols]
        random.shuffle(cells)
        for cell in cells:
            if _far_enough(cell, placed):
                return cell
    return None


def _sample_marker_colors(bg: int, frame_color: int) -> tuple[int, int]:
    usable = [c for c in ALL_COLORS if c not in {bg, frame_color}]
    row_color = random.choice(usable)
    col_choices = [c for c in usable if c != row_color]
    col_color = random.choice(col_choices)
    return int(row_color), int(col_color)


def _expected_from_metadata(
    inp: np.ndarray,
    frames: list[dict],
    row_markers: dict[int, tuple[int, int, int]],
    col_markers: dict[int, tuple[int, int, int]],
    bg: int,
) -> np.ndarray:
    return _apply_rule(inp, frames, row_markers, col_markers, bg)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(MIN_FRAMES, MAX_FRAMES)
    row_sample = _sample_axis_spans(frame_count)
    col_sample = _sample_axis_spans(frame_count)
    if row_sample is None or col_sample is None:
        return None

    row_spans, height = row_sample
    col_spans, width = col_sample
    permutation = list(range(frame_count))
    random.shuffle(permutation)

    bg = random.choice(ALL_COLORS)
    frame_palette = [c for c in ALL_COLORS if c != bg]
    if len(frame_palette) < frame_count:
        return None
    frame_colors = random.sample(frame_palette, frame_count)

    frames: list[dict] = []
    for i, (r0, r1) in enumerate(row_spans):
        c0, c1 = col_spans[permutation[i]]
        frames.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "color": int(frame_colors[i])})

    free_rows = _occupied_axis_cells(height, row_spans)
    free_cols = _occupied_axis_cells(width, col_spans)
    if not free_rows or not free_cols:
        return None

    row_markers: dict[int, tuple[int, int, int]] = {}
    col_markers: dict[int, tuple[int, int, int]] = {}
    placed_markers: list[tuple[int, int]] = []
    frame_order = list(range(frame_count))
    random.shuffle(frame_order)

    for idx in frame_order:
        bbox = frames[idx]["bbox"]
        row_cell = _pick_row_marker(bbox, free_cols, placed_markers)
        if row_cell is None:
            return None
        placed_markers.append(row_cell)

        col_cell = _pick_col_marker(bbox, free_rows, placed_markers)
        if col_cell is None:
            return None
        placed_markers.append(col_cell)

        row_color, col_color = _sample_marker_colors(bg, int(frames[idx]["color"]))
        row_markers[idx] = (int(row_cell[0]), int(row_cell[1]), int(row_color))
        col_markers[idx] = (int(col_cell[0]), int(col_cell[1]), int(col_color))

    inp = np.full((height, width), int(bg), dtype=int)
    for frame in frames:
        _draw_frame(inp, frame["bbox"], int(frame["color"]))
    for r, c, color in row_markers.values():
        inp[r, c] = int(color)
    for r, c, color in col_markers.values():
        inp[r, c] = int(color)

    expected = _expected_from_metadata(inp, frames, row_markers, col_markers, bg)

    if int(np.unique(inp, return_counts=True)[1].max()) == 0:
        return None
    if _get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
