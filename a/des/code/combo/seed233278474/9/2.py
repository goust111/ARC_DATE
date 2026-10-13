from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS8]


def _background_color(grid: np.ndarray) -> int:
    values = set(int(v) for v in np.unique(grid))
    if BG in values:
        return BG
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    return int(np.argmax(counts))


def _find_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(comp))

    comps.sort(key=lambda comp: min(comp))
    return comps


def _order_path(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = set(comp)
    if len(cells) <= 1:
        return list(comp)

    degree = {}
    for cell in cells:
        degree[cell] = sum(1 for nbr in _neighbors4(*cell) if nbr in cells)
        assert degree[cell] <= 2, "component must be a non-branching path"

    endpoints = sorted(cell for cell, deg in degree.items() if deg == 1)
    assert len(endpoints) == 2, "component must be an open path"

    path = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nxts = [nbr for nbr in _neighbors4(*cur) if nbr in cells and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt

    assert len(path) == len(cells), "path traversal must cover the full component"
    return path


def _segment_orientation(coords: list[tuple[int, int]]) -> int | None:
    if len(coords) <= 1:
        return None
    same_row = all(r == coords[0][0] for r, _ in coords)
    same_col = all(c == coords[0][1] for _, c in coords)
    assert same_row or same_col, "every color segment must be straight"
    return 0 if same_row else 1


def _extract_segments(grid: np.ndarray, path: list[tuple[int, int]]) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    i = 0
    while i < len(path):
        color = int(grid[path[i]])
        j = i + 1
        while j < len(path) and int(grid[path[j]]) == color:
            j += 1
        coords = path[i:j]
        segments.append(
            {
                "coords": coords,
                "color": color,
                "length": len(coords),
                "orientation": _segment_orientation(coords),
            }
        )
        i = j

    for left, right in zip(segments, segments[1:]):
        assert int(left["color"]) != int(right["color"]), "adjacent segments must have different colors"
        left_axis = left["orientation"]
        right_axis = right["orientation"]
        if left_axis is not None and right_axis is not None:
            assert left_axis != right_axis, "consecutive straight segments must turn"

    return segments


def _sorted_colors(segments: list[dict[str, object]]) -> list[int]:
    order = sorted(range(len(segments)), key=lambda idx: (int(segments[idx]["length"]), idx))
    return [int(segments[idx]["color"]) for idx in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign each polyline's segment colors by stable sorting the segments by length.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for comp in _find_components(g, bg):
        path = _order_path(comp)
        segments = _extract_segments(g, path)
        recolored = _sorted_colors(segments)
        for segment, color in zip(segments, recolored):
            for r, c in segment["coords"]:
                out[r, c] = color

    return out


# --------------- generator ---------------

GRID_H_MIN = 12
GRID_H_MAX = 24
GRID_W_MIN = 12
GRID_W_MAX = 24
MAX_GEN_ATTEMPTS = 300

MIN_SEGMENTS = 2
MAX_SEGMENTS = 6
MIN_SEG_LEN = 2
MAX_SEG_LEN = 7
PATH_ATTEMPTS = 120
PLACE_ATTEMPTS = 120
PLACEMENT_PAD = 1

COLOR_POOL = [0, 1, 2, 3, 4, 5, 6, 7, 9]
AXIS_DIRS = {
    0: [(-1, 0), (1, 0)],
    1: [(0, -1), (0, 1)],
}


def _sample_segment_colors(segment_count: int) -> list[int]:
    distinct = random.randint(2, min(4, segment_count))
    palette = random.sample(COLOR_POOL, distinct)

    for _ in range(100):
        colors = [random.choice(palette)]
        for _ in range(segment_count - 1):
            choices = [color for color in palette if color != colors[-1]]
            colors.append(random.choice(choices))
        if len(set(colors)) == distinct:
            return colors

    offset = random.randrange(distinct)
    return [palette[(offset + i) % distinct] for i in range(segment_count)]


def _sample_segment_lengths(segment_count: int, max_len: int) -> list[int] | None:
    if max_len < MIN_SEG_LEN:
        return None

    for _ in range(100):
        lengths = [random.randint(MIN_SEG_LEN, max_len) for _ in range(segment_count)]
        if segment_count >= 3 and random.random() < 0.45:
            src, dst = random.sample(range(segment_count), 2)
            lengths[dst] = lengths[src]
        if len(set(lengths)) > 1:
            return lengths
    return None


def _bbox_of(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _can_add(next_cell: tuple[int, int], prev_cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if next_cell in occupied:
        return False
    for nbr in _neighbors4(*next_cell):
        if nbr in occupied and nbr != prev_cell:
            return False
    return True


def _make_polyline(max_h: int, max_w: int) -> dict[str, object] | None:
    max_segments = min(MAX_SEGMENTS, 4 if min(max_h, max_w) < 9 else MAX_SEGMENTS)
    if max_segments < MIN_SEGMENTS:
        return None

    for _ in range(PATH_ATTEMPTS):
        segment_count = random.randint(MIN_SEGMENTS, max_segments)
        max_len = min(MAX_SEG_LEN, max(max_h, max_w) // 2 + 1)
        lengths = _sample_segment_lengths(segment_count, max_len)
        if lengths is None:
            continue
        colors = _sample_segment_colors(segment_count)
        start_axis = random.randrange(2)

        def build(
            seg_idx: int,
            axis: int,
            end_cell: tuple[int, int],
            used: set[tuple[int, int]],
            segments: list[list[tuple[int, int]]],
        ) -> list[list[tuple[int, int]]] | None:
            if seg_idx == segment_count:
                return [segment[:] for segment in segments]

            directions = AXIS_DIRS[axis][:]
            random.shuffle(directions)
            for dr, dc in directions:
                seg_cells: list[tuple[int, int]] = [end_cell] if seg_idx == 0 else []
                occupied = set(used)
                cur = end_cell
                steps = lengths[seg_idx] - 1 if seg_idx == 0 else lengths[seg_idx]
                ok = True
                for _ in range(steps):
                    nxt = (cur[0] + dr, cur[1] + dc)
                    if not _can_add(nxt, cur, occupied):
                        ok = False
                        break
                    seg_cells.append(nxt)
                    occupied.add(nxt)
                    cur = nxt
                if not ok:
                    continue

                new_used = used | set(seg_cells)
                r0, c0, r1, c1 = _bbox_of(new_used)
                if r1 - r0 + 1 > max_h or c1 - c0 + 1 > max_w:
                    continue

                segments.append(seg_cells)
                result = build(seg_idx + 1, 1 - axis, seg_cells[-1], new_used, segments)
                segments.pop()
                if result is not None:
                    return result
            return None

        built = build(0, start_axis, (0, 0), {(0, 0)}, [])
        if built is None:
            continue

        path = [cell for segment in built for cell in segment]
        coords = set(path)
        r0, c0, r1, c1 = _bbox_of(coords)
        recolored = _sorted_colors(
            [{"coords": segment, "color": color, "length": len(segment)} for segment, color in zip(built, colors)]
        )
        changed = any(old != new for old, new in zip(colors, recolored))

        return {
            "segments": built,
            "path": path,
            "bbox": (r0, c0, r1, c1),
            "colors": colors,
            "sorted_colors": recolored,
            "changed": changed,
        }

    return None


def _expanded_cells(coords: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    expanded = set()
    for r, c in coords:
        for dr in range(-PLACEMENT_PAD, PLACEMENT_PAD + 1):
            for dc in range(-PLACEMENT_PAD, PLACEMENT_PAD + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return sorted(expanded)


def _place_polyline(polyline: dict[str, object], h: int, w: int, reserved: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = polyline["bbox"]
    row_low = -int(r0)
    row_high = h - 1 - int(r1)
    col_low = -int(c0)
    col_high = w - 1 - int(c1)
    if row_low > row_high or col_low > col_high:
        return None

    translations = [(dr, dc) for dr in range(row_low, row_high + 1) for dc in range(col_low, col_high + 1)]
    random.shuffle(translations)

    for shift_r, shift_c in translations[:PLACE_ATTEMPTS]:
        shifted_path = [(r + shift_r, c + shift_c) for r, c in polyline["path"]]
        expanded = _expanded_cells(shifted_path, h, w)
        if any(reserved[r, c] for r, c in expanded):
            continue

        shifted_segments = [
            [(r + shift_r, c + shift_c) for r, c in segment]
            for segment in polyline["segments"]
        ]
        return {
            "segments": shifted_segments,
            "colors": list(polyline["colors"]),
            "sorted_colors": list(polyline["sorted_colors"]),
            "expanded": expanded,
            "changed": bool(polyline["changed"]),
        }

    return None


def _paint_segments(grid: np.ndarray, segments: list[list[tuple[int, int]]], colors: list[int]) -> None:
    for segment, color in zip(segments, colors):
        for r, c in segment:
            grid[r, c] = int(color)


def _build_expected(shape: tuple[int, int], placed: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for polyline in placed:
        _paint_segments(out, polyline["segments"], polyline["sorted_colors"])
    return out


def _build_input(shape: tuple[int, int], placed: list[dict[str, object]]) -> np.ndarray:
    inp = np.full(shape, BG, dtype=int)
    for polyline in placed:
        _paint_segments(inp, polyline["segments"], polyline["colors"])
    return inp


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_H_MIN, GRID_H_MAX)
    w = random.randint(GRID_W_MIN, GRID_W_MAX)

    if min(h, w) < 15:
        max_objects = 2
    elif min(h, w) < 20:
        max_objects = 3
    else:
        max_objects = 4
    object_count = random.randint(1, max_objects)

    local_h_cap = max(5, min(h - 2, h // 2 + 4))
    local_w_cap = max(5, min(w - 2, w // 2 + 4))

    reserved = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []
    changed_any = False

    for _ in range(object_count):
        placed_one = None
        for _ in range(80):
            polyline = _make_polyline(local_h_cap, local_w_cap)
            if polyline is None:
                continue
            placed_one = _place_polyline(polyline, h, w, reserved)
            if placed_one is not None:
                break
        if placed_one is None:
            return None
        for r, c in placed_one["expanded"]:
            reserved[r, c] = True
        placed.append(placed_one)
        changed_any = changed_any or bool(placed_one["changed"])

    if not changed_any:
        return None

    inp = _build_input((h, w), placed)
    expected = _build_expected((h, w), placed)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
