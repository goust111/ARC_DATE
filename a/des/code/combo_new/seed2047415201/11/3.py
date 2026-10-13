import random
from collections import deque

import numpy as np


BLACK = 0
MAGENTA = 6
ORANGE = 7
OBJECT_COLORS = (1, 2, 3, 4, 5, 8, 9)

ARC_GRID_MAX = 30
MAX_GENERATION_ATTEMPTS = 500


def _ranges_between_separators(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(x) for x in separators) + [int(length)]
    ranges = []
    for lo, hi in zip(bounds, bounds[1:]):
        if hi - lo > 1:
            ranges.append((lo + 1, hi - 1))
    return ranges


def _same_color_components(panel: np.ndarray) -> list[dict[str, int]]:
    h, w = panel.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, int]] = []
    ignored = {BLACK, MAGENTA, ORANGE}

    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color in ignored or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if int(panel[nr, nc]) == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            area = len(coords)
            assert area == rect_h * rect_w, "each object must be a filled rectangle"
            assert bool(np.all(panel[r0 : r1 + 1, c0 : c1 + 1] == color))

            objects.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(rect_h),
                    "w": int(rect_w),
                    "area": int(area),
                }
            )

    objects.sort(key=lambda obj: (obj["r0"], obj["c0"]))
    return objects


def _marker_direction(panel: np.ndarray) -> str:
    pts = np.argwhere(panel == MAGENTA)
    assert len(pts) == 2, "each panel must contain exactly one 2-pixel magenta marker"
    (r0, c0), (r1, c1) = sorted((int(r), int(c)) for r, c in pts)
    if r0 == r1 and abs(c1 - c0) == 1:
        return "right"
    if c0 == c1 and abs(r1 - r0) == 1:
        return "down"
    raise AssertionError("magenta marker must be horizontal or vertical")


def _solve_panel(
    source: np.ndarray,
    out: np.ndarray,
    row_range: tuple[int, int],
    col_range: tuple[int, int],
) -> None:
    r0, r1 = row_range
    c0, c1 = col_range
    panel = source[r0 : r1 + 1, c0 : c1 + 1]

    direction = _marker_direction(panel)
    objects = _same_color_components(panel)
    assert 3 <= len(objects) <= 5, "each panel should contain 3 to 5 rectangles"

    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    objects = sorted(objects, key=lambda obj: obj["area"])

    if direction == "right":
        cursor = c0
        for obj in objects:
            h, w, color = obj["h"], obj["w"], obj["color"]
            assert cursor + w - 1 <= c1, "horizontal stack must fit inside the panel"
            out[r1 - h + 1 : r1 + 1, cursor : cursor + w] = color
            cursor += w
    else:
        cursor = r1 + 1
        for obj in objects:
            h, w, color = obj["h"], obj["w"], obj["color"]
            cursor -= h
            assert cursor >= r0, "vertical stack must fit inside the panel"
            out[cursor : cursor + h, c0 : c0 + w] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort each panel's rectangles by area and restack them in the marker direction.

    Input structure:
    - A black grid is divided into rectangular panels by full orange separator
      rows and/or columns.
    - Each panel contains 3 to 5 solid monochrome rectangles in distinct colors.
    - Each panel also contains exactly one 2-pixel magenta direction marker:
      horizontal means stack to the right, vertical means stack downward.

    Rule:
    - For each panel, read the magenta marker orientation and ignore the marker
      afterward.
    - Sort the panel's rectangle objects by visible area from small to large.
    - Clear the panel interior while preserving all orange separator lines.
    - A horizontal marker packs the sorted rectangles from the panel's left edge,
      bottom-aligned and touching with no gaps.
    - A vertical marker packs the sorted rectangles from the panel's bottom edge
      upward, left-aligned and touching with no gaps.

    Key insight:
    - The marker controls the stacking axis, while the rectangle areas control
      the order; these two independent cues determine the final layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    sep_rows = [r for r in range(height) if bool(np.all(g[r, :] == ORANGE))]
    sep_cols = [c for c in range(width) if bool(np.all(g[:, c] == ORANGE))]
    row_ranges = _ranges_between_separators(height, sep_rows)
    col_ranges = _ranges_between_separators(width, sep_cols)
    assert row_ranges and col_ranges, "separator lines must leave at least one panel"

    out = g.copy()
    for row_range in row_ranges:
        for col_range in col_ranges:
            _solve_panel(g, out, row_range, col_range)
    return out


def _mark_padded(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    ph, pw = occupied.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(ph, r + h + pad)
    c1 = min(pw, c + w + pad)
    occupied[r0:r1, c0:c1] = True


def _free_positions(occupied: np.ndarray, h: int, w: int, pad: int = 1) -> list[tuple[int, int]]:
    ph, pw = occupied.shape
    positions = []
    if h > ph or w > pw:
        return positions
    for r in range(ph - h + 1):
        for c in range(pw - w + 1):
            r0 = max(0, r - pad)
            c0 = max(0, c - pad)
            r1 = min(ph, r + h + pad)
            c1 = min(pw, c + w + pad)
            if not bool(occupied[r0:r1, c0:c1].any()):
                positions.append((r, c))
    random.shuffle(positions)
    return positions


def _sample_object_dims(count: int, panel_h: int, panel_w: int, direction: str) -> list[tuple[int, int]] | None:
    max_h = min(4, panel_h)
    max_w = min(4, panel_w)
    candidates = [(h, w) for h in range(1, max_h + 1) for w in range(1, max_w + 1)]

    for _ in range(300):
        dims: list[tuple[int, int]] = []
        areas: set[int] = set()
        for _idx in range(count):
            choices = [(h, w) for h, w in candidates if h * w not in areas]
            if not choices:
                break
            h, w = random.choice(choices)
            dims.append((h, w))
            areas.add(h * w)
        if len(dims) != count:
            continue
        if direction == "right" and sum(w for _h, w in dims) <= panel_w:
            return dims
        if direction == "down" and sum(h for h, _w in dims) <= panel_h:
            return dims
    return None


def _place_shape(
    grid: np.ndarray,
    occupied: np.ndarray,
    panel_r0: int,
    panel_c0: int,
    shape_h: int,
    shape_w: int,
    color: int,
) -> bool:
    positions = _free_positions(occupied, shape_h, shape_w)
    if not positions:
        return False
    r, c = positions[0]
    grid[panel_r0 + r : panel_r0 + r + shape_h, panel_c0 + c : panel_c0 + c + shape_w] = color
    _mark_padded(occupied, r, c, shape_h, shape_w)
    return True


def _place_panel_contents(
    grid: np.ndarray,
    panel_r0: int,
    panel_c0: int,
    panel_h: int,
    panel_w: int,
    direction: str,
) -> bool:
    count = random.choices([3, 4, 5], weights=[4, 4, 2])[0]
    dims = _sample_object_dims(count, panel_h, panel_w, direction)
    if dims is None:
        return False

    occupied = np.zeros((panel_h, panel_w), dtype=bool)
    marker_h, marker_w = (1, 2) if direction == "right" else (2, 1)
    if not _place_shape(grid, occupied, panel_r0, panel_c0, marker_h, marker_w, MAGENTA):
        return False

    colors = random.sample(list(OBJECT_COLORS), count)
    pieces = list(zip(dims, colors))
    random.shuffle(pieces)
    for (rect_h, rect_w), color in pieces:
        if not _place_shape(grid, occupied, panel_r0, panel_c0, rect_h, rect_w, int(color)):
            return False
    return True


def _directions_for_panels(panel_count: int) -> list[str]:
    directions = []
    for i in range(panel_count):
        directions.append("right" if i % 2 == 0 else "down")
    random.shuffle(directions)
    return directions


def _sample_layout() -> tuple[list[int], list[int]]:
    layout = random.choice([(1, 2), (2, 1), (2, 2)])
    row_count, col_count = layout
    if row_count == 1:
        row_heights = [random.randint(11, 18)]
    else:
        row_heights = [random.randint(11, 14) for _ in range(row_count)]

    if col_count == 1:
        col_widths = [random.randint(11, 18)]
    else:
        col_widths = [random.randint(11, 14) for _ in range(col_count)]

    return row_heights, col_widths


def _panel_origins(sizes: list[int]) -> list[int]:
    origins = []
    pos = 0
    for size in sizes:
        origins.append(pos)
        pos += size + 1
    return origins


def _gen_attempt() -> np.ndarray | None:
    row_heights, col_widths = _sample_layout()
    height = sum(row_heights) + len(row_heights) - 1
    width = sum(col_widths) + len(col_widths) - 1
    if height > ARC_GRID_MAX or width > ARC_GRID_MAX:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    r = 0
    for h in row_heights[:-1]:
        r += h
        grid[r, :] = ORANGE
        r += 1
    c = 0
    for w in col_widths[:-1]:
        c += w
        grid[:, c] = ORANGE
        c += 1

    row_origins = _panel_origins(row_heights)
    col_origins = _panel_origins(col_widths)
    panel_specs = []
    for rr, panel_h in zip(row_origins, row_heights):
        for cc, panel_w in zip(col_origins, col_widths):
            panel_specs.append((rr, cc, panel_h, panel_w))

    directions = _directions_for_panels(len(panel_specs))
    for (panel_r0, panel_c0, panel_h, panel_w), direction in zip(panel_specs, directions):
        if not _place_panel_contents(grid, panel_r0, panel_c0, panel_h, panel_w, direction):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.ndim != 2 or max(out.shape) > ARC_GRID_MAX:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
