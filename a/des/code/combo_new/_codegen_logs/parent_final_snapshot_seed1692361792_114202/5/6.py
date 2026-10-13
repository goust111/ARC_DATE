import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

SEGMENT_COUNT_RANGE = (3, 6)
STEP_RANGE = (3, 6)
MINOR_BOUND = 8
STYLE_CHOICES = (1, 2, 3)
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _neighbors4(mask: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    r, c = cell
    result: list[tuple[int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            result.append((nr, nc))
    return result


def _trace_path(path_mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(path_mask)
    assert len(coords) >= 2, "expected a non-trivial path"

    endpoints = []
    for r, c in map(tuple, coords):
        deg = len(_neighbors4(path_mask, (r, c)))
        assert deg in (1, 2), "path must be a simple open polyline"
        if deg == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "expected exactly two path endpoints"

    start = min(endpoints)
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxt = [cell for cell in _neighbors4(path_mask, cur) if cell != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must be unambiguous"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    assert len(ordered) == int(path_mask.sum()), "path traversal missed cells"
    return ordered


def _segment_orientation(
    coords: list[tuple[int, int]],
    prev_cell: tuple[int, int] | None,
    next_cell: tuple[int, int] | None,
) -> str:
    rows = {r for r, _c in coords}
    cols = {c for _r, c in coords}
    if len(rows) == 1:
        return "h"
    if len(cols) == 1:
        return "v"
    if len(coords) == 1:
        r, c = coords[0]
        refs = [cell for cell in (prev_cell, next_cell) if cell is not None]
        for rr, cc in refs:
            if rr == r:
                return "h"
            if cc == c:
                return "v"
    raise AssertionError("each color segment must be straight")


def _extract_segments(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    colored_mask = (grid != BLACK) & (grid != GRAY)
    comps = _components(colored_mask)
    assert comps, "expected a colored path"

    path_comp = max(comps, key=len)
    marker_cells = [comp[0] for comp in comps if len(comp) == 1 and comp is not path_comp]
    assert marker_cells, "expected colored marker dots"
    assert len(marker_cells) + len(path_comp) == sum(len(comp) for comp in comps), "non-path colored components must be singletons"

    path_mask = np.zeros_like(colored_mask)
    for r, c in path_comp:
        path_mask[r, c] = True
    ordered = _trace_path(path_mask)

    segments: list[dict] = []
    i = 0
    while i < len(ordered):
        color = int(grid[ordered[i]])
        j = i
        while j + 1 < len(ordered) and int(grid[ordered[j + 1]]) == color:
            j += 1
        coords = ordered[i : j + 1]
        prev_cell = ordered[i - 1] if i > 0 else None
        next_cell = ordered[j + 1] if j + 1 < len(ordered) else None
        segments.append(
            {
                "color": color,
                "coords": coords,
                "orientation": _segment_orientation(coords, prev_cell, next_cell),
            }
        )
        i = j + 1

    colors = [segment["color"] for segment in segments]
    assert len(colors) == len(set(colors)), "segment colors must be unique"
    assert len(marker_cells) == len(segments), "markers and segments must match one-to-one"
    return segments, marker_cells


def _decode_marker_styles(grid: np.ndarray, marker_cells: list[tuple[int, int]]) -> dict[int, int]:
    marker_set = set(marker_cells)
    styles: dict[int, int] = {}
    used_markers: set[tuple[int, int]] = set()

    for comp in _components(grid == GRAY):
        rows = {r for r, _c in comp}
        cols = sorted(c for _r, c in comp)
        assert len(rows) == 1, "grey pattern must be horizontal"
        assert cols == list(range(cols[0], cols[-1] + 1)), "grey pattern must be contiguous"
        width = len(comp)
        assert width in STYLE_CHOICES, "unsupported grey width"

        below = [(r + 1, c) for r, c in comp if (r + 1, c) in marker_set]
        assert len(below) == 1, "each grey pattern must align to exactly one marker"
        mr, mc = below[0]
        color = int(grid[mr, mc])
        assert color not in (BLACK, GRAY)
        assert color not in styles, "marker colors must be unique"
        styles[color] = width
        used_markers.add((mr, mc))

    assert used_markers == marker_set, "every marker must have a grey pattern"
    return styles


def main(grid: np.ndarray) -> np.ndarray:
    """Redraw each colored segment of a single polyline according to the grey pattern above its matching marker dot.

    Input structure:
    - The background is black.
    - One connected single-pixel polyline is split into straight horizontal or vertical color segments.
    - Outside the path, each segment color appears once more as a single marker dot.
    - Each marker has an immediately adjacent grey horizontal pattern above it whose width is 1, 2, or 3.

    Rule:
    - Pair every path segment with the same-colored marker dot and read the width of the grey pattern above that marker.
    - Width 1 keeps the segment unchanged.
    - Width 2 turns the segment into a dash pattern along the path order, keeping the 1st, 3rd, 5th, ... cells of that segment and erasing the alternating cells to black.
    - Width 3 keeps the segment and adds same-colored parallel lines one cell to each side perpendicular to the segment, painting only onto black cells.
    - Remove all marker dots and all grey patterns in the output.

    Key insight:
    - The task is a three-step decode: isolate the unique colored path, match each segment color to its detached marker, then interpret the grey width as the segment's redraw style.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments, marker_cells = _extract_segments(g)
    styles = _decode_marker_styles(g, marker_cells)
    assert set(styles) == {int(segment["color"]) for segment in segments}

    out = g.copy()
    out[g == GRAY] = BLACK
    for r, c in marker_cells:
        out[r, c] = BLACK

    h, w = out.shape
    for segment in segments:
        color = int(segment["color"])
        coords = segment["coords"]
        style = styles[color]

        if style == 2:
            for idx, (r, c) in enumerate(coords):
                if idx % 2 == 1:
                    out[r, c] = BLACK
            continue

        if style == 3:
            if segment["orientation"] == "h":
                offsets = [(-1, 0), (1, 0)]
            else:
                offsets = [(0, -1), (0, 1)]
            for dr, dc in offsets:
                for r, c in coords:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == BLACK:
                        out[rr, cc] = color

    return out


def _generate_vertices(segment_count: int, major_horizontal: bool) -> list[tuple[int, int]] | None:
    r = 0
    c = 0
    vertices = [(r, c)]
    for i in range(segment_count):
        horizontal = (i % 2 == 0) if major_horizontal else (i % 2 == 1)
        step = random.randint(*STEP_RANGE)
        if horizontal:
            if major_horizontal:
                c += step
            else:
                valid = [sign for sign in (-1, 1) if -MINOR_BOUND <= c + sign * step <= MINOR_BOUND]
                if not valid:
                    return None
                c += random.choice(valid) * step
        else:
            if major_horizontal:
                valid = [sign for sign in (-1, 1) if -MINOR_BOUND <= r + sign * step <= MINOR_BOUND]
                if not valid:
                    return None
                r += random.choice(valid) * step
            else:
                r += step
        vertices.append((r, c))
    return vertices


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("line segment must be axis-aligned")


def _build_segments_from_vertices(vertices: list[tuple[int, int]]) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    run_cells = [_line_cells(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]
    turn_to_prev = [bool(random.getrandbits(1)) for _ in range(len(run_cells) - 1)]

    segments: list[list[tuple[int, int]]] = []
    for i, cells in enumerate(run_cells):
        start = 1 if i > 0 and turn_to_prev[i - 1] else 0
        stop = len(cells) - (1 if i < len(run_cells) - 1 and not turn_to_prev[i] else 0)
        segment = cells[start:stop]
        if len(segment) < 2:
            return [], []
        segments.append(segment)

    path: list[tuple[int, int]] = []
    for segment in segments:
        if path:
            pr, pc = path[-1]
            sr, sc = segment[0]
            if abs(pr - sr) + abs(pc - sc) != 1:
                return [], []
        path.extend(segment)

    if len(path) != len(set(path)):
        return [], []
    return segments, path


def _local_side_targets(
    segment: list[tuple[int, int]],
    orientation: str,
    path_set: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if orientation == "h":
        offsets = [(-1, 0), (1, 0)]
    else:
        offsets = [(0, -1), (0, 1)]
    result: set[tuple[int, int]] = set()
    for r, c in segment:
        for dr, dc in offsets:
            cell = (r + dr, c + dc)
            if cell not in path_set:
                result.add(cell)
    return result


def _choose_styles(segments: list[list[tuple[int, int]]]) -> list[int] | None:
    n = len(segments)
    lengths = [len(segment) for segment in segments]
    for _ in range(200):
        styles = [1, 2, 3] + [random.choice(STYLE_CHOICES) for _ in range(n - 3)]
        random.shuffle(styles)
        if any(style == 2 and length % 2 == 0 for style, length in zip(styles, lengths)):
            continue
        if any(styles[i] == 3 and styles[i + 1] == 3 for i in range(n - 1)):
            continue
        return styles
    return None


def _validate_bold_overlap(segments: list[list[tuple[int, int]]], styles: list[int]) -> bool:
    path_set = {cell for segment in segments for cell in segment}
    bold_targets: dict[tuple[int, int], int] = {}
    for idx, (segment, style) in enumerate(zip(segments, styles)):
        if style != 3:
            continue
        orientation = _segment_orientation(segment, None, None)
        for cell in _local_side_targets(segment, orientation, path_set):
            owner = bold_targets.get(cell)
            if owner is not None and owner != idx:
                return False
            bold_targets[cell] = idx
    return True


def _normalize_segments(segments: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    min_r = min(r for segment in segments for r, _c in segment)
    min_c = min(c for segment in segments for _r, c in segment)
    return [[(r - min_r, c - min_c) for r, c in segment] for segment in segments]


def _layout_marker_groups(widths: list[int], grid_w: int) -> list[tuple[int, int, int]] | None:
    order = list(range(len(widths)))
    random.shuffle(order)
    gaps = [random.randint(1, 3) for _ in range(max(0, len(widths) - 1))]
    total = sum(widths[idx] for idx in order) + sum(gaps)
    if total > grid_w - 2:
        return None
    start = random.randint(1, grid_w - total - 1)
    placements: list[tuple[int, int, int]] = []
    cur = start
    for pos, idx in enumerate(order):
        w = widths[idx]
        placements.append((idx, cur, random.randint(0, w - 1)))
        cur += w
        if pos < len(gaps):
            cur += gaps[pos]
    return placements


def _paint_scene(
    segments: list[list[tuple[int, int]]],
    colors: list[int],
    styles: list[int],
) -> np.ndarray | None:
    bbox_h = max(r for segment in segments for r, _c in segment) + 1
    bbox_w = max(c for segment in segments for _r, c in segment) + 1

    min_h = bbox_h + 6
    min_w = bbox_w + 4
    marker_w = sum(styles) + max(0, len(styles) - 1)
    min_w = max(min_w, marker_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, min(30, min_h + 4))
    grid_w = random.randint(min_w, min(30, min_w + 4))
    row_off = random.randint(4, grid_h - bbox_h - 2)
    col_off = random.randint(2, grid_w - bbox_w - 2)

    marker_layout = _layout_marker_groups(styles, grid_w)
    if marker_layout is None:
        return None

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for segment, color in zip(segments, colors):
        for r, c in segment:
            grid[row_off + r, col_off + c] = color

    for idx, grey_start, anchor in marker_layout:
        width = styles[idx]
        color = colors[idx]
        grid[0, grey_start : grey_start + width] = GRAY
        grid[1, grey_start + anchor] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    segment_count = random.randint(*SEGMENT_COUNT_RANGE)
    major_horizontal = bool(random.getrandbits(1))
    vertices = _generate_vertices(segment_count, major_horizontal)
    if vertices is None:
        return None

    segments, _path = _build_segments_from_vertices(vertices)
    if not segments:
        return None
    styles = _choose_styles(segments)
    if styles is None or not _validate_bold_overlap(segments, styles):
        return None

    segments = _normalize_segments(segments)
    colors = random.sample(COLOR_POOL, len(segments))
    grid = _paint_scene(segments, colors, styles)
    if grid is None:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("failed to generate a valid input after repeated retries")
