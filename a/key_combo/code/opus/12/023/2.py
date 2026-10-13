from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
PAIR_COLORS = (BLUE, RED, GREEN)
MAX_GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]


@dataclass(frozen=True)
class Line:
    color: int
    slope: int
    start: Coord
    end: Coord
    path: tuple[Coord, ...]


@dataclass(frozen=True)
class Geometry:
    slope: int
    start: Coord
    end: Coord
    path: tuple[Coord, ...]


@dataclass(frozen=True)
class PlacedLine:
    color: int
    slope: int
    start: Coord
    end: Coord
    path: tuple[Coord, ...]
    upper_rect: tuple[Coord, Coord]
    lower_rect: tuple[Coord, Coord]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw diagonal connectors between same-color rectangles inside a frame, swapping colors after crossings.

    Input structure:
    - The grid has a black background with one gray one-cell-thick rectangular frame whose four corners stay black.
    - Strictly inside the frame there are three colors, and each color forms exactly two solid domino rectangles (1x2 or 2x1).
    - Pixels outside the frame may contain distracting pixels of the same three colors, but they are not part of the task.

    Rule:
    - Ignore everything outside the frame and keep the frame and all rectangles unchanged.
    - For each interior same-color pair, find the unique facing cells that lie on the nearest 45-degree diagonal and draw the open diagonal path between them.
    - If two differently colored paths cross, paint the crossing cell orange and let the two carried colors swap after that crossing, propagating toward each line's endpoint.

    Key insight:
    - The frame isolates the only valid rectangles, while the final colors of the connector pixels can only be determined by tracking every diagonal path and processing crossings in path order.
    """
    g = np.asarray(grid, dtype=int)
    frame = _find_frame(g)
    lines = _extract_lines(g, frame)
    return _paint_output(g, lines)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0, "expected a grey frame"
    r0, c0 = grey_coords.min(axis=0)
    r1, c1 = grey_coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert r1 - r0 >= 3 and c1 - c0 >= 3, "frame must contain an interior"
    assert int(grid[r0, c0]) == BLACK
    assert int(grid[r0, c1]) == BLACK
    assert int(grid[r1, c0]) == BLACK
    assert int(grid[r1, c1]) == BLACK
    assert np.all(grid[r0, c0 + 1 : c1] == GREY)
    assert np.all(grid[r1, c0 + 1 : c1] == GREY)
    assert np.all(grid[r0 + 1 : r1, c0] == GREY)
    assert np.all(grid[r0 + 1 : r1, c1] == GREY)

    expected = np.zeros_like(grid, dtype=bool)
    expected[r0, c0 + 1 : c1] = True
    expected[r1, c0 + 1 : c1] = True
    expected[r0 + 1 : r1, c0] = True
    expected[r0 + 1 : r1, c1] = True
    assert np.array_equal(grid == GREY, expected), "grey pixels must form exactly one frame"
    return r0, c0, r1, c1


def _extract_lines(grid: np.ndarray, frame: tuple[int, int, int, int]) -> list[Line]:
    r0, c0, r1, c1 = frame
    lines: list[Line] = []
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = sorted(int(v) for v in np.unique(interior) if int(v) not in (BLACK, GREY, ORANGE))
    assert colors == sorted(PAIR_COLORS), "frame interior must contain exactly the three pair colors"

    for color in colors:
        mask = interior == color
        comps = _components(mask)
        assert len(comps) == 2, "each interior color must form exactly two components"
        dominos = []
        for comp in comps:
            coords = tuple((int(r + r0 + 1), int(c + c0 + 1)) for r, c in comp)
            assert _is_domino(coords), "each component must be a 1x2 or 2x1 rectangle"
            dominos.append(coords)

        start, end = _facing_cells(dominos[0], dominos[1])
        slope = 1 if end[1] > start[1] else -1
        path = tuple(_diagonal_path(start, end))
        assert path, "connector must have at least one open cell"
        for r, c in path:
            assert r0 < r < r1 and c0 < c < c1
            assert int(grid[r, c]) == BLACK, "connector path must pass through interior background"
        lines.append(Line(color=color, slope=slope, start=start, end=end, path=path))

    assert len(lines) == 3
    return lines


def _paint_output(grid: np.ndarray, lines: list[Line]) -> np.ndarray:
    out = np.array(grid, copy=True)
    crossings = _crossing_events(lines)
    per_line = {idx: [] for idx in range(len(lines))}
    for event_id, event in enumerate(crossings):
        a, b = event["lines"]
        per_line[a].append((event["indices"][a], event_id))
        per_line[b].append((event["indices"][b], event_id))
    for line_events in per_line.values():
        line_events.sort()

    current_color = {idx: lines[idx].color for idx in range(len(lines))}
    next_pos = {idx: 0 for idx in range(len(lines))}
    ordered_events = {event_id: {} for event_id in range(len(crossings))}
    remaining = set(range(len(crossings)))

    while remaining:
        progressed = False
        for event_id in list(remaining):
            a, b = crossings[event_id]["lines"]
            if per_line[a][next_pos[a]][1] != event_id or per_line[b][next_pos[b]][1] != event_id:
                continue
            color_a = current_color[a]
            color_b = current_color[b]
            ordered_events[event_id][a] = color_b
            ordered_events[event_id][b] = color_a
            current_color[a], current_color[b] = color_b, color_a
            next_pos[a] += 1
            next_pos[b] += 1
            remaining.remove(event_id)
            progressed = True
            break
        assert progressed, "crossings must admit a consistent order along every line"

    crossing_cells = {tuple(event["coord"]) for event in crossings}
    for line_id, line in enumerate(lines):
        cursor = 0
        segment_color = line.color
        for cross_index, event_id in per_line[line_id]:
            for cell in line.path[cursor:cross_index]:
                out[cell] = segment_color
            segment_color = ordered_events[event_id][line_id]
            cursor = cross_index + 1
        for cell in line.path[cursor:]:
            out[cell] = segment_color

    for cell in crossing_cells:
        out[cell] = ORANGE
    return out


def _crossing_events(lines: list[Line]) -> list[dict]:
    events = []
    used_cells: set[Coord] = set()
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            shared = set(lines[i].path) & set(lines[j].path)
            assert len(shared) <= 1, "any pair of lines may intersect at most once"
            if not shared:
                continue
            coord = next(iter(shared))
            assert coord not in used_cells, "three lines may not meet at one cell"
            used_cells.add(coord)
            idx_i = lines[i].path.index(coord)
            idx_j = lines[j].path.index(coord)
            events.append({"lines": (i, j), "coord": coord, "indices": {i: idx_i, j: idx_j}})
    return events


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _is_domino(coords: tuple[Coord, ...]) -> bool:
    if len(coords) != 2:
        return False
    (r0, c0), (r1, c1) = sorted(coords)
    return (r0 == r1 and abs(c0 - c1) == 1) or (c0 == c1 and abs(r0 - r1) == 1)


def _facing_cells(first: tuple[Coord, ...], second: tuple[Coord, ...]) -> tuple[Coord, Coord]:
    box_first = _bbox(first)
    box_second = _bbox(second)
    if (box_second[0], box_second[1]) < (box_first[0], box_first[1]):
        first, second = second, first
        box_first, box_second = box_second, box_first

    candidates = []
    for start in first:
        for end in second:
            dr = end[0] - start[0]
            dc = end[1] - start[1]
            if dr > 0 and abs(dc) == dr:
                candidates.append((dr, abs(dc), start, end))
    assert candidates, "same-color rectangles must expose one diagonal facing pair"
    min_dr = min(item[0] for item in candidates)
    best = [(start, end) for dr, _dist, start, end in candidates if dr == min_dr]
    assert len(best) == 1, "facing diagonal pair must be unique"
    return best[0]


def _bbox(coords: tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _diagonal_path(start: Coord, end: Coord) -> list[Coord]:
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    assert dr > 0 and abs(dc) == dr
    step_c = 1 if dc > 0 else -1
    return [(start[0] + k, start[1] + step_c * k) for k in range(1, dr)]


def _domino_cells(endpoint: Coord, slope: int, role: str, orient: str) -> tuple[Coord, Coord]:
    r, c = endpoint
    if role == "upper":
        if orient == "h":
            if slope == 1:
                return tuple(sorted(((r, c - 1), (r, c))))
            return tuple(sorted(((r, c), (r, c + 1))))
        return tuple(sorted(((r - 1, c), (r, c))))
    assert role == "lower"
    if orient == "h":
        if slope == 1:
            return tuple(sorted(((r, c), (r, c + 1))))
        return tuple(sorted(((r, c - 1), (r, c))))
    return tuple(sorted(((r, c), (r + 1, c))))


def _geometry_to_placed(geometry: Geometry, color: int, upper_orient: str, lower_orient: str) -> PlacedLine:
    upper_rect = _domino_cells(geometry.start, geometry.slope, "upper", upper_orient)
    lower_rect = _domino_cells(geometry.end, geometry.slope, "lower", lower_orient)
    return PlacedLine(
        color=color,
        slope=geometry.slope,
        start=geometry.start,
        end=geometry.end,
        path=geometry.path,
        upper_rect=upper_rect,
        lower_rect=lower_rect,
    )


def _valid_placed_lines(lines: list[PlacedLine], ih: int, iw: int, target_crossings: int) -> bool:
    rects = []
    for line in lines:
        rects.append(set(line.upper_rect))
        rects.append(set(line.lower_rect))
    all_rects = [set(rect) for rect in rects]
    if len({cell for rect in all_rects for cell in rect}) != 2 * len(all_rects):
        return False

    for rect in all_rects:
        for r, c in rect:
            if not (0 <= r < ih and 0 <= c < iw):
                return False

    union_rects = set().union(*all_rects)
    for line in lines:
        if set(line.path) & union_rects:
            return False
        for cell in line.path:
            for rect in all_rects:
                if any(_manhattan(cell, other) == 1 for other in rect):
                    return False

    for i, rect_a in enumerate(all_rects):
        for rect_b in all_rects[i + 1 :]:
            if any(_manhattan(a, b) == 1 for a in rect_a for b in rect_b):
                return False

    base_lines = [
        Line(color=line.color, slope=line.slope, start=line.start, end=line.end, path=line.path) for line in lines
    ]
    crossings = _crossing_events(base_lines)
    return len(crossings) == target_crossings


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _sample_generic_geometry(
    rng: np.random.Generator,
    ih: int,
    iw: int,
    slope: int,
    *,
    min_length: int,
    max_length: int,
    row_min: int = 1,
    row_max: int | None = None,
) -> Geometry | None:
    if row_max is None:
        row_max = ih - 2
    row_min = max(1, row_min)
    row_max = min(ih - 2, row_max)
    if row_max - row_min < min_length:
        return None

    for _ in range(120):
        if slope == 1:
            start_r = int(rng.integers(row_min, row_max - min_length + 1))
            start_c = int(rng.integers(1, iw - 1 - min_length))
            max_len = min(row_max - start_r, (iw - 2) - start_c, max_length)
        else:
            start_r = int(rng.integers(row_min, row_max - min_length + 1))
            start_c = int(rng.integers(1 + min_length, iw - 1))
            max_len = min(row_max - start_r, start_c - 1, max_length)
        if max_len < min_length:
            continue
        length = int(rng.integers(min_length, max_len + 1))
        end = (start_r + length, start_c + slope * length)
        path = tuple(_diagonal_path((start_r, start_c), end))
        return Geometry(slope=slope, start=(start_r, start_c), end=end, path=path)
    return None


def _sample_geometry_through_point(
    rng: np.random.Generator,
    ih: int,
    iw: int,
    point: Coord,
    slope: int,
    *,
    min_arm: int,
    max_arm: int,
) -> Geometry | None:
    r, c = point
    if slope == 1:
        max_up = min(r - 1, c - 1, max_arm)
        max_down = min((ih - 2) - r, (iw - 2) - c, max_arm)
    else:
        max_up = min(r - 1, (iw - 2) - c, max_arm)
        max_down = min((ih - 2) - r, c - 1, max_arm)
    if max_up < min_arm or max_down < min_arm:
        return None
    up = int(rng.integers(min_arm, max_up + 1))
    down = int(rng.integers(min_arm, max_down + 1))
    start = (r - up, c - slope * up)
    end = (r + down, c + slope * down)
    return Geometry(slope=slope, start=start, end=end, path=tuple(_diagonal_path(start, end)))


def _build_lines_from_geometries(
    rng: np.random.Generator,
    geometries: list[Geometry],
    target_crossings: int,
    ih: int,
    iw: int,
) -> list[PlacedLine] | None:
    colors = list(PAIR_COLORS)
    rng.shuffle(colors)
    option_lists = [
        [("h", "h"), ("h", "v"), ("v", "h"), ("v", "v")] for _ in geometries
    ]
    combos = list(itertools.product(*option_lists))
    rng.shuffle(combos)
    for combo in combos:
        lines = [
            _geometry_to_placed(geom, color, upper_orient, lower_orient)
            for geom, color, (upper_orient, lower_orient) in zip(geometries, colors, combo)
        ]
        if _valid_placed_lines(lines, ih, iw, target_crossings):
            return lines
    return None


def _sample_two_crossings(rng: np.random.Generator, ih: int, iw: int) -> list[PlacedLine] | None:
    for _ in range(180):
        spine_slope = int(rng.choice((-1, 1)))
        spine = _sample_generic_geometry(
            rng, ih, iw, spine_slope, min_length=6, max_length=min(8, min(ih, iw) - 3)
        )
        if spine is None or len(spine.path) < 5:
            continue
        valid_indices = list(range(1, len(spine.path) - 1))
        pairs = [(i, j) for i in valid_indices for j in valid_indices if j - i >= 2]
        if not pairs:
            continue
        idx_a, idx_b = pairs[int(rng.integers(0, len(pairs)))]
        cross_a = spine.path[idx_a]
        cross_b = spine.path[idx_b]
        flank_slope = -spine_slope
        flank_a = _sample_geometry_through_point(rng, ih, iw, cross_a, flank_slope, min_arm=2, max_arm=5)
        flank_b = _sample_geometry_through_point(rng, ih, iw, cross_b, flank_slope, min_arm=2, max_arm=5)
        if flank_a is None or flank_b is None:
            continue
        geometries = [spine, flank_a, flank_b]
        lines = _build_lines_from_geometries(rng, geometries, 2, ih, iw)
        if lines is not None:
            return lines
    return None


def _sample_one_crossing(rng: np.random.Generator, ih: int, iw: int) -> list[PlacedLine] | None:
    for _ in range(220):
        slope_a = int(rng.choice((-1, 1)))
        line_a = _sample_generic_geometry(
            rng, ih, iw, slope_a, min_length=5, max_length=min(7, min(ih, iw) - 3)
        )
        if line_a is None or len(line_a.path) < 3:
            continue
        cross_index = int(rng.integers(1, len(line_a.path) - 1))
        cross = line_a.path[cross_index]
        line_b = _sample_geometry_through_point(rng, ih, iw, cross, -slope_a, min_arm=2, max_arm=5)
        if line_b is None:
            continue

        top_band_max = line_b.start[0] - 1
        bottom_band_min = line_b.end[0] + 1
        bands = []
        if top_band_max - 1 >= 4:
            bands.append((1, top_band_max))
        if (ih - 2) - bottom_band_min >= 4:
            bands.append((bottom_band_min, ih - 2))
        if not bands:
            continue
        row_min, row_max = bands[int(rng.integers(0, len(bands)))]
        line_c = _sample_generic_geometry(
            rng,
            ih,
            iw,
            slope_a,
            min_length=4,
            max_length=min(6, row_max - row_min),
            row_min=row_min,
            row_max=row_max,
        )
        if line_c is None:
            continue

        geometries = [line_a, line_b, line_c]
        lines = _build_lines_from_geometries(rng, geometries, 1, ih, iw)
        if lines is not None:
            return lines
    return None


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    ih = int(rng.integers(10, 14))
    iw = int(rng.integers(10, 14))
    top = int(rng.integers(1, 3))
    bottom = int(rng.integers(1, 3))
    left = int(rng.integers(1, 3))
    right = int(rng.integers(1, 3))
    h = ih + top + bottom + 2
    w = iw + left + right + 2

    target_crossings = 2 if rng.random() < 0.6 else 1
    if target_crossings == 2:
        lines = _sample_two_crossings(rng, ih, iw)
    else:
        lines = _sample_one_crossing(rng, ih, iw)
    if lines is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    r0, c0 = top, left
    r1, c1 = top + ih + 1, left + iw + 1
    grid[r0, c0 + 1 : c1] = GREY
    grid[r1, c0 + 1 : c1] = GREY
    grid[r0 + 1 : r1, c0] = GREY
    grid[r0 + 1 : r1, c1] = GREY

    for line in lines:
        for r, c in line.upper_rect + line.lower_rect:
            grid[r0 + 1 + r, c0 + 1 + c] = line.color

    outside_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (r0 <= r <= r1 and c0 <= c <= c1)
    ]
    rng.shuffle(outside_cells)
    cursor = 0
    for color in PAIR_COLORS:
        count = int(rng.integers(2, 5))
        for _ in range(count):
            if cursor >= len(outside_cells):
                break
            r, c = outside_cells[cursor]
            cursor += 1
            grid[r, c] = color
    extra = int(rng.integers(0, min(8, len(outside_cells) - cursor + 1)))
    for _ in range(extra):
        if cursor >= len(outside_cells):
            break
        r, c = outside_cells[cursor]
        cursor += 1
        grid[r, c] = int(rng.choice(PAIR_COLORS))

    try:
        frame = _find_frame(grid)
        recovered = _extract_lines(grid, frame)
        crossings = _crossing_events(recovered)
        assert len(crossings) == target_crossings
        out = _paint_output(grid, recovered)
        assert np.any(out != grid)
        assert int(np.sum(out == ORANGE)) == target_crossings
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid framed diagonal-swap input")
