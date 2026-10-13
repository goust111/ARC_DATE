from __future__ import annotations

import numpy as np


RNG = np.random.default_rng()

GRID_MIN = 12
GRID_MAX = 30
WINDOW_COUNT_MIN = 2
WINDOW_COUNT_MAX = 6
MAX_GENERATION_ATTEMPTS = 400

INNER_SIZE_CHOICES = (1, 3, 5)
ROW_GAP_CHOICES = (2, 3, 4)
COL_GAP_CHOICES = (2, 3, 4)
TOP_BOTTOM_MARGIN_CHOICES = (1, 2, 3, 4)
SIDE_MARGIN_CHOICES = (1, 2, 3, 4)
PALETTE_SIDE_MARGIN_CHOICES = (3, 4, 5)
PALETTE_CORNERS = ("TL", "TR", "BL", "BR")
COUNT_RELATIONS = ("short", "equal", "long")


def _randint(low: int, high: int) -> int:
    return int(RNG.integers(low, high + 1))


def _choice(values):
    return values[int(RNG.integers(0, len(values)))]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": pts, "bbox": _bbox(pts)})
    return comps


def _find_palette(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    height, width = grid.shape
    candidates = []
    for pts in _components_from_mask(grid != bg):
        r0, c0, r1, c1 = _bbox(pts)
        if c0 != c1:
            continue
        if len(pts) != r1 - r0 + 1:
            continue
        if not ((r0 == 0 or r1 == height - 1) and (c0 == 0 or c0 == width - 1)):
            continue
        colors = [int(grid[r, c0]) for r in range(r0, r1 + 1)]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((r0, c0, r1, c1, colors))

    assert len(candidates) == 1, "expected one corner palette"
    r0, c0, r1, _c1, colors = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    return colors, mask


def _is_rect_frame(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if len(comp["coords"]) != 2 * (height + width) - 4:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == color):
        return False
    if np.any(sub[1:-1, 1:-1] == color):
        return False
    return True


def _find_windows(grid: np.ndarray, bg: int) -> list[dict]:
    windows = [comp for comp in _monochrome_components(grid, bg) if _is_rect_frame(grid, comp)]
    assert windows, "expected at least one window frame"
    return windows


def _order_windows(windows: list[dict]) -> list[dict]:
    by_top = sorted(windows, key=lambda window: (window["bbox"][0], window["bbox"][1]))
    rows: list[list[dict]] = []
    current: list[dict] = []
    current_bottom = -1

    for window in by_top:
        r0, _c0, r1, _c1 = window["bbox"]
        if not current or r0 <= current_bottom:
            current.append(window)
            current_bottom = max(current_bottom, r1)
        else:
            rows.append(sorted(current, key=lambda item: (item["bbox"][1], item["bbox"][0])))
            current = [window]
            current_bottom = r1
    if current:
        rows.append(sorted(current, key=lambda item: (item["bbox"][1], item["bbox"][0])))

    ordered: list[dict] = []
    for row in rows:
        ordered.extend(row)
    return ordered


def _draw_cross(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    grid[center_r, c0 + 1 : c1] = int(color)
    grid[r0 + 1 : r1, center_c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint palette colors into window centers as crosses, then erase the palette.

    Input structure:
    - Background fills the grid.
    - Several one-pixel-thick monochrome rectangular frames act as windows.
    - One corner contains a vertical palette made of distinct single pixels.

    Rule:
    - Read windows in visual reading order: top-to-bottom by rows, left-to-right within a row.
    - Read palette colors from top to bottom.
    - Color k paints a full interior cross in window k.
    - Frames stay unchanged and the palette is removed in the output.
    - Extra windows beyond the palette remain empty.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    palette_colors, palette_mask = _find_palette(g, bg)
    windows = _order_windows(_find_windows(g, bg))

    out = g.copy()
    out[palette_mask] = bg
    for window, color in zip(windows, palette_colors):
        _draw_cross(out, window["bbox"], color)
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _partition_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(RNG.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(int(cut - prev))
        prev = int(cut)
    return values


def _choose_palette_length(window_count: int, relation: str, max_colors: int) -> int:
    if relation == "short":
        return _randint(1, window_count - 1)
    if relation == "equal":
        return window_count
    return _randint(window_count + 1, min(max_colors, window_count + 2))


def _render_expected(
    grid: np.ndarray,
    bg: int,
    palette_cells: list[tuple[int, int]],
    palette_colors: list[int],
    ordered_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r, c in palette_cells:
        out[r, c] = int(bg)
    for bbox, color in zip(ordered_boxes, palette_colors):
        _draw_cross(out, bbox, color)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    bg = _randint(0, 9)
    usable_colors = [color for color in range(10) if color != bg]

    window_count = _randint(WINDOW_COUNT_MIN, WINDOW_COUNT_MAX)
    relation = _choice(COUNT_RELATIONS)
    palette_len = _choose_palette_length(window_count, relation, len(usable_colors))
    corner = _choice(PALETTE_CORNERS)

    row_count = _randint(1, min(3, window_count))
    row_sizes = _partition_positive(window_count, row_count)

    frame_colors = RNG.choice(usable_colors, size=window_count, replace=True).astype(int).tolist()
    palette_colors = RNG.choice(usable_colors, size=palette_len, replace=False).astype(int).tolist()

    window_specs = []
    for idx in range(window_count):
        inner_h = _choice(INNER_SIZE_CHOICES)
        inner_w = _choice(INNER_SIZE_CHOICES)
        window_specs.append(
            {
                "height": inner_h + 2,
                "width": inner_w + 2,
                "frame_color": int(frame_colors[idx]),
            }
        )

    row_windows = []
    cursor = 0
    for size in row_sizes:
        row = window_specs[cursor : cursor + size]
        cursor += size
        row_windows.append(row)

    row_heights = [max(spec["height"] for spec in row) for row in row_windows]
    row_widths = []
    row_gaps = []
    for row in row_windows:
        gaps = [_choice(COL_GAP_CHOICES) for _ in range(max(0, len(row) - 1))]
        row_gaps.append(gaps)
        row_widths.append(sum(spec["width"] for spec in row) + sum(gaps))
    vertical_gaps = [_choice(ROW_GAP_CHOICES) for _ in range(max(0, row_count - 1))]

    left_margin = _choice(PALETTE_SIDE_MARGIN_CHOICES) if corner.endswith("L") else _choice(SIDE_MARGIN_CHOICES)
    right_margin = _choice(PALETTE_SIDE_MARGIN_CHOICES) if corner.endswith("R") else _choice(SIDE_MARGIN_CHOICES)
    top_margin = _choice(TOP_BOTTOM_MARGIN_CHOICES)
    bottom_margin = _choice(TOP_BOTTOM_MARGIN_CHOICES)

    rows_min = top_margin + sum(row_heights) + sum(vertical_gaps) + bottom_margin
    cols_min = left_margin + max(row_widths) + right_margin
    rows_min = max(rows_min, palette_len)

    if rows_min > GRID_MAX or cols_min > GRID_MAX:
        return None

    extra_rows_min = max(0, GRID_MIN - rows_min)
    extra_cols_min = max(0, GRID_MIN - cols_min)
    extra_rows_max = GRID_MAX - rows_min
    extra_cols_max = GRID_MAX - cols_min

    extra_rows = _randint(extra_rows_min, min(extra_rows_max, extra_rows_min + 6))
    extra_cols = _randint(extra_cols_min, min(extra_cols_max, extra_cols_min + 8))

    add_top = _randint(0, extra_rows)
    add_left = _randint(0, extra_cols)
    top_margin += add_top
    bottom_margin += extra_rows - add_top
    left_margin += add_left
    right_margin += extra_cols - add_left

    rows = top_margin + sum(row_heights) + sum(vertical_gaps) + bottom_margin
    cols = left_margin + max(row_widths) + right_margin
    if rows < palette_len or rows > GRID_MAX or cols > GRID_MAX:
        return None

    grid = np.full((rows, cols), int(bg), dtype=int)

    ordered_boxes: list[tuple[int, int, int, int]] = []
    y = top_margin
    for row_idx, row in enumerate(row_windows):
        row_height = row_heights[row_idx]
        row_width = row_widths[row_idx]
        x = left_margin + _randint(0, cols - left_margin - right_margin - row_width)
        for spec_idx, spec in enumerate(row):
            r0 = y + (row_height - spec["height"]) // 2
            c0 = x
            r1 = r0 + spec["height"] - 1
            c1 = c0 + spec["width"] - 1
            bbox = (r0, c0, r1, c1)
            ordered_boxes.append(bbox)
            _draw_frame(grid, bbox, spec["frame_color"])
            x = c1 + 1
            if spec_idx < len(row) - 1:
                x += row_gaps[row_idx][spec_idx]
        y += row_height
        if row_idx < row_count - 1:
            y += vertical_gaps[row_idx]

    if corner == "TL":
        palette_cells = [(idx, 0) for idx in range(palette_len)]
    elif corner == "TR":
        palette_cells = [(idx, cols - 1) for idx in range(palette_len)]
    elif corner == "BL":
        palette_cells = [(rows - palette_len + idx, 0) for idx in range(palette_len)]
    else:
        palette_cells = [(rows - palette_len + idx, cols - 1) for idx in range(palette_len)]

    for (r, c), color in zip(palette_cells, palette_colors):
        grid[r, c] = int(color)

    expected = _render_expected(grid, bg, palette_cells, palette_colors, ordered_boxes)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _generate_case()
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate valid input")
