from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
NON_BG_COLORS = list(range(1, 10))

SIDE_TO_INWARD: dict[str, Coord] = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
SIDE_TO_OUTWARD: dict[str, Coord] = {side: (-dr, -dc) for side, (dr, dc) in SIDE_TO_INWARD.items()}
DIR_TO_ENTRY_SIDE: dict[Coord, str] = {direction: side for side, direction in SIDE_TO_INWARD.items()}
DIR_TO_EXIT_SIDE: dict[Coord, str] = {direction: side for side, direction in SIDE_TO_OUTWARD.items()}
PERPENDICULAR_DIRS: dict[Coord, tuple[Coord, Coord]] = {
    (1, 0): ((0, -1), (0, 1)),
    (-1, 0): ((0, -1), (0, 1)),
    (0, 1): ((-1, 0), (1, 0)),
    (0, -1): ((-1, 0), (1, 0)),
}

GRID_SIZE_RANGE = (22, 30)
FRAME_SIZE_RANGE = (4, 8)
START_MARGIN = 2
SEGMENT_LENGTH_RANGE = (3, 7)
RELAY_COUNT_CHOICES = (1, 2, 3)
RELAY_COUNT_WEIGHTS = (2, 4, 3)
EXTRA_FRAME_CHOICES = (0, 1, 2)
EXTRA_FRAME_WEIGHTS = (3, 4, 2)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


@dataclass(frozen=True)
class Frame:
    bbox: BBox
    color: int
    gaps: tuple[Coord, Coord]
    sides: tuple[str, str]

    def gap_side(self, gap: Coord) -> str:
        if gap == self.gaps[0]:
            return self.sides[0]
        if gap == self.gaps[1]:
            return self.sides[1]
        raise AssertionError("gap does not belong to frame")

    def other_gap(self, gap: Coord) -> Coord:
        if gap == self.gaps[0]:
            return self.gaps[1]
        if gap == self.gaps[1]:
            return self.gaps[0]
        raise AssertionError("gap does not belong to frame")


@dataclass(frozen=True)
class FrameCandidate:
    comp_ids: frozenset[int]
    frame: Frame


def _in_bounds(shape: tuple[int, int], coord: Coord) -> bool:
    r, c = coord
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add(coord: Coord, direction: Coord, k: int = 1) -> Coord:
    return coord[0] + k * direction[0], coord[1] + k * direction[1]


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _border_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _interior_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}


def _side_of_gap(gap: Coord, bbox: BBox) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("gap must be a non-corner border cell")


def _component_coords(grid: np.ndarray) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(coords)
    return components


def _parse_frame(grid: np.ndarray, coords: list[Coord]) -> Frame:
    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1, "each foreground component must be monochrome"
    color = colors.pop()

    frame = _try_parse_frame(color, coords)
    assert frame is not None, "component union must encode one frame border"
    return frame


def _try_parse_frame(color: int, coords: list[Coord]) -> Frame | None:
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    present = set(coords)
    border = _border_cells(bbox)
    if not present <= border:
        return None

    gaps = sorted(border - present)
    if len(gaps) != 2 or len(present) != len(border) - 2:
        return None
    try:
        sides = tuple(_side_of_gap(gap, bbox) for gap in gaps)
    except AssertionError:
        return None
    if len(set(sides)) != 2:
        return None
    return Frame(bbox=bbox, color=int(color), gaps=(gaps[0], gaps[1]), sides=sides)


def _direction_between(start: Coord, end: Coord) -> Coord:
    if start[0] == end[0]:
        dc = end[1] - start[1]
        assert dc != 0
        return (0, 1 if dc > 0 else -1)
    if start[1] == end[1]:
        dr = end[0] - start[0]
        assert dr != 0
        return (1 if dr > 0 else -1, 0)
    raise AssertionError("points are not aligned")


def _line_cells(start: Coord, end: Coord) -> list[Coord]:
    direction = _direction_between(start, end)
    cells: list[Coord] = []
    cur = start
    while cur != end:
        cur = _add(cur, direction)
        cells.append(cur)
    return cells


def _route_cells_inside(frame: Frame, entry_gap: Coord) -> list[Coord]:
    entry_side = frame.gap_side(entry_gap)
    exit_gap = frame.other_gap(entry_gap)
    exit_side = frame.gap_side(exit_gap)
    entry_in = SIDE_TO_INWARD[entry_side]
    exit_in = SIDE_TO_INWARD[exit_side]

    if entry_in[0] != 0 and exit_in[1] != 0:
        pivot = (exit_gap[0], entry_gap[1])
    elif entry_in[1] != 0 and exit_in[0] != 0:
        pivot = (entry_gap[0], exit_gap[1])
    elif entry_in == (-exit_in[0], -exit_in[1]):
        if entry_in[0] != 0:
            assert entry_gap[1] == exit_gap[1], "opposite vertical gaps must share a column"
        else:
            assert entry_gap[0] == exit_gap[0], "opposite horizontal gaps must share a row"
        pivot = exit_gap
    else:
        raise AssertionError("invalid gap geometry")

    r0, c0, r1, c1 = frame.bbox
    if pivot != exit_gap:
        assert r0 < pivot[0] < r1 and c0 < pivot[1] < c1, "turn pivot must lie inside the frame"

    cells: list[Coord] = []
    if entry_gap != pivot:
        cells.extend(_line_cells(entry_gap, pivot))
    if pivot != exit_gap:
        cells.extend(_line_cells(pivot, exit_gap))
    return cells


def _trace_ray(grid: np.ndarray, frames: list[Frame], marker: Coord, marker_color: int, start_dir: Coord) -> np.ndarray:
    out = grid.copy()
    gap_lookup: dict[Coord, Frame] = {}
    for frame in frames:
        for gap in frame.gaps:
            assert gap not in gap_lookup, "gaps must be unique"
            gap_lookup[gap] = frame

    current = marker
    direction = start_dir
    seen: set[tuple[Coord, Coord]] = set()
    for _ in range(grid.size * 12):
        state = (current, direction)
        assert state not in seen, "ray path should not loop"
        seen.add(state)

        nxt = _add(current, direction)
        if not _in_bounds(grid.shape, nxt):
            break
        if int(grid[nxt]) != BG:
            break

        current = nxt
        out[current] = marker_color

        frame = gap_lookup.get(current)
        if frame is None:
            continue

        gap_side = frame.gap_side(current)
        inward = SIDE_TO_INWARD[gap_side]
        outward = SIDE_TO_OUTWARD[gap_side]
        if direction == inward:
            for cell in _route_cells_inside(frame, current):
                assert int(grid[cell]) == BG, "ray must cross only background cells"
                out[cell] = marker_color
            current = frame.other_gap(current)
            direction = SIDE_TO_OUTWARD[frame.gap_side(current)]
        else:
            assert direction == outward, "ray must approach a gap from inside or directly from outside"
    else:
        raise AssertionError("ray tracing exceeded the step budget")

    return out


def _find_start_direction(grid: np.ndarray, frames: list[Frame], marker: Coord) -> Coord | None:
    host_frames = []
    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        if r0 < marker[0] < r1 and c0 < marker[1] < c1:
            host_frames.append(frame)
    if len(host_frames) != 1:
        return None

    host = host_frames[0]
    aligned: list[Coord] = []
    for gap in host.gaps:
        if gap[0] != marker[0] and gap[1] != marker[1]:
            continue
        segment = _line_cells(marker, gap)
        if all(int(grid[cell]) == BG for cell in segment):
            aligned.append(_direction_between(marker, gap))
    if len(aligned) != 1:
        return None
    return aligned[0]


def _parse_scene(grid: np.ndarray) -> tuple[list[Frame], Coord, int, Coord]:
    components = _component_coords(grid)
    comp_colors = [int(grid[coords[0]]) for coords in components]

    candidates_by_comp: dict[int, list[FrameCandidate]] = {i: [] for i in range(len(components))}
    seen_candidates: set[tuple[frozenset[int], BBox, tuple[Coord, Coord]]] = set()

    def add_candidate(comp_ids: frozenset[int], frame: Frame | None) -> None:
        if frame is None:
            return
        key = (comp_ids, frame.bbox, tuple(sorted(frame.gaps)))
        if key in seen_candidates:
            return
        seen_candidates.add(key)
        candidate = FrameCandidate(comp_ids=comp_ids, frame=frame)
        for idx in comp_ids:
            candidates_by_comp[idx].append(candidate)

    for i, coords in enumerate(components):
        add_candidate(frozenset({i}), _try_parse_frame(comp_colors[i], coords))

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if comp_colors[i] != comp_colors[j]:
                continue
            union_coords = components[i] + components[j]
            add_candidate(frozenset({i, j}), _try_parse_frame(comp_colors[i], union_coords))

    singleton_ids = [i for i, coords in enumerate(components) if len(coords) == 1]
    assert singleton_ids, "expected at least one singleton marker candidate"

    for marker_idx in singleton_ids:
        marker = components[marker_idx][0]
        marker_color = int(grid[marker])
        remaining = frozenset(idx for idx in range(len(components)) if idx != marker_idx)

        def search(unused: frozenset[int], chosen: list[FrameCandidate]) -> tuple[list[Frame], Coord] | None:
            if not unused:
                frames = [candidate.frame for candidate in chosen]
                if not frames:
                    return None
                start_dir = _find_start_direction(grid, frames, marker)
                if start_dir is None:
                    return None
                return frames, start_dir

            first = min(unused)
            for candidate in candidates_by_comp[first]:
                if candidate.comp_ids <= unused:
                    chosen.append(candidate)
                    result = search(unused - candidate.comp_ids, chosen)
                    if result is not None:
                        return result
                    chosen.pop()
            return None

        result = search(remaining, [])
        if result is not None:
            frames, start_dir = result
            return frames, marker, marker_color, start_dir

    raise AssertionError("failed to reconstruct frames and marker")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the marker-colored light through gap-port rectangle frames and paint the full background path.

    Input structure:
    - The background is black.
    - Several one-cell-thick monochrome hollow rectangles are scattered on the canvas.
    - Every rectangle border has exactly two single-cell non-corner gaps on different sides, and exactly one rectangle contains a lone interior marker pixel.

    Rule:
    - Start from the marker and move straight toward the unique aligned gap of its host rectangle.
    - In open background the ray keeps moving straight; when it reaches a frame gap from outside, it enters that rectangle, follows the inside L-shaped port route to the other gap, and exits along that gap's outward direction.
    - If the next cell is a solid frame wall the ray stops before it, and if the next step would leave the grid the ray exits the picture; paint every traversed background cell with the marker color while leaving walls and the marker unchanged.

    Key insight:
    - The gaps are ports, not damage: each hollow rectangle redirects a straight ray from one side to another, so the whole answer is the chained path induced by the marker start and the ordered gap connections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] <= 30 and g.shape[1] <= 30
    frames, marker, marker_color, start_dir = _parse_scene(g)
    return _trace_ray(g, frames, marker, marker_color, start_dir)


def _frame_from_bbox(
    bbox: BBox,
    color: int,
    first_side: str,
    first_offset: int,
    second_side: str,
    second_offset: int,
) -> Frame:
    r0, c0, r1, c1 = bbox
    assert first_side != second_side

    def gap_for(side: str, offset: int) -> Coord:
        if side == "top":
            assert 1 <= offset <= c1 - c0 - 1
            return (r0, c0 + offset)
        if side == "bottom":
            assert 1 <= offset <= c1 - c0 - 1
            return (r1, c0 + offset)
        if side == "left":
            assert 1 <= offset <= r1 - r0 - 1
            return (r0 + offset, c0)
        assert side == "right"
        assert 1 <= offset <= r1 - r0 - 1
        return (r0 + offset, c1)

    gap1 = gap_for(first_side, first_offset)
    gap2 = gap_for(second_side, second_offset)
    return Frame(bbox=bbox, color=int(color), gaps=(gap1, gap2), sides=(first_side, second_side))


def _frame_from_entry(
    shape: tuple[int, int],
    entry_gap: Coord,
    entry_side: str,
    exit_side: str,
    height: int,
    width: int,
    entry_offset: int,
    exit_offset: int,
    color: int,
) -> Frame | None:
    r, c = entry_gap
    if entry_side == "left":
        r0, c0 = r - entry_offset, c
    elif entry_side == "right":
        r0, c0 = r - entry_offset, c - width + 1
    elif entry_side == "top":
        r0, c0 = r, c - entry_offset
    else:
        assert entry_side == "bottom"
        r0, c0 = r - height + 1, c - entry_offset

    bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= shape[0] or bbox[3] >= shape[1]:
        return None
    try:
        frame = _frame_from_bbox(bbox, color, entry_side, entry_offset, exit_side, exit_offset)
    except AssertionError:
        return None
    if frame.gaps[0] == frame.gaps[1]:
        return None
    return frame


def _render_input(shape: tuple[int, int], frames: list[Frame], marker: Coord, marker_color: int) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        for cell in _border_cells(frame.bbox):
            if cell in frame.gaps:
                continue
            grid[cell] = frame.color
    grid[marker] = marker_color
    return grid


def _paint_expected(inp: np.ndarray, path_cells: list[Coord], marker_color: int) -> np.ndarray:
    out = inp.copy()
    for cell in path_cells:
        assert int(out[cell]) == BG
        out[cell] = marker_color
    return out


def _marker_candidates(start_frame: Frame, exit_gap: Coord) -> list[Coord]:
    other_gap = start_frame.other_gap(exit_gap)
    exit_side = start_frame.gap_side(exit_gap)
    direction = SIDE_TO_INWARD[exit_side]
    candidates: list[Coord] = []
    cur = _add(exit_gap, direction)
    while True:
        r0, c0, r1, c1 = start_frame.bbox
        if not (r0 < cur[0] < r1 and c0 < cur[1] < c1):
            break
        if cur[0] != other_gap[0] and cur[1] != other_gap[1]:
            candidates.append(cur)
        cur = _add(cur, direction)
    return candidates


def _path_segment_is_clear(
    cells: list[Coord],
    occupied_bboxes: set[Coord],
    reserved_path: set[Coord],
) -> bool:
    return all(cell not in occupied_bboxes and cell not in reserved_path for cell in cells)


def _frame_is_clear(
    frame: Frame,
    occupied_bboxes: set[Coord],
    reserved_path: set[Coord],
) -> bool:
    cells = _bbox_cells(frame.bbox)
    return cells.isdisjoint(occupied_bboxes) and cells.isdisjoint(reserved_path)


def _place_start_frame(
    shape: tuple[int, int],
    color: int,
    occupied_bboxes: set[Coord],
    reserved_path: set[Coord],
) -> tuple[Frame, Coord, Coord, list[Coord]] | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        fh = random.randint(*FRAME_SIZE_RANGE)
        fw = random.randint(*FRAME_SIZE_RANGE)
        if fh + 2 * START_MARGIN > h or fw + 2 * START_MARGIN > w:
            continue
        r0 = random.randint(START_MARGIN, h - fh - START_MARGIN)
        c0 = random.randint(START_MARGIN, w - fw - START_MARGIN)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)

        exit_side = random.choice(["top", "bottom", "left", "right"])
        side_choices = [side for side in ("top", "bottom", "left", "right") if side != exit_side]
        side_choices = [side for side in side_choices if side not in {DIR_TO_EXIT_SIDE[SIDE_TO_INWARD[exit_side]]}]
        other_side = random.choice([side for side in side_choices if side not in {exit_side} and side not in {
            {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[exit_side]
        }])

        exit_offset_max = fw - 2 if exit_side in {"top", "bottom"} else fh - 2
        other_offset_max = fw - 2 if other_side in {"top", "bottom"} else fh - 2
        exit_offset = random.randint(1, exit_offset_max)
        other_offset = random.randint(1, other_offset_max)

        frame = _frame_from_bbox(bbox, color, exit_side, exit_offset, other_side, other_offset)
        if not _frame_is_clear(frame, occupied_bboxes, reserved_path):
            continue

        marker_options = _marker_candidates(frame, frame.gaps[0])
        if frame.gap_side(frame.gaps[1]) == exit_side:
            marker_options = _marker_candidates(frame, frame.gaps[1])
            exit_gap = frame.gaps[1]
        else:
            exit_gap = frame.gaps[0]
        if not marker_options:
            continue

        marker = random.choice(marker_options)
        path_cells = _line_cells(marker, exit_gap)
        if not _path_segment_is_clear(path_cells, occupied_bboxes, reserved_path):
            continue
        return frame, marker, exit_gap, path_cells
    return None


def _place_relay_frame(
    shape: tuple[int, int],
    current_pos: Coord,
    current_dir: Coord,
    color: int,
    occupied_bboxes: set[Coord],
    reserved_path: set[Coord],
) -> tuple[Frame, list[Coord], Coord, Coord] | None:
    next_dirs = list(PERPENDICULAR_DIRS[current_dir])
    random.shuffle(next_dirs)
    entry_side = DIR_TO_ENTRY_SIDE[current_dir]

    for next_dir in next_dirs:
        exit_side = DIR_TO_EXIT_SIDE[next_dir]
        for _ in range(PLACEMENT_ATTEMPTS):
            dist = random.randint(*SEGMENT_LENGTH_RANGE)
            entry_gap = _add(current_pos, current_dir, dist)
            if not _in_bounds(shape, entry_gap):
                continue

            fh = random.randint(*FRAME_SIZE_RANGE)
            fw = random.randint(*FRAME_SIZE_RANGE)
            entry_offset_max = fh - 2 if entry_side in {"left", "right"} else fw - 2
            exit_offset_max = fh - 2 if exit_side in {"left", "right"} else fw - 2
            entry_offset = random.randint(1, entry_offset_max)
            exit_offset = random.randint(1, exit_offset_max)
            frame = _frame_from_entry(
                shape,
                entry_gap,
                entry_side,
                exit_side,
                fh,
                fw,
                entry_offset,
                exit_offset,
                color,
            )
            if frame is None or not _frame_is_clear(frame, occupied_bboxes, reserved_path):
                continue

            outside_cells = _line_cells(current_pos, entry_gap)
            inside_cells = _route_cells_inside(frame, entry_gap)
            new_path = outside_cells + inside_cells
            if not _path_segment_is_clear(new_path, occupied_bboxes, reserved_path):
                continue
            return frame, new_path, frame.other_gap(entry_gap), next_dir
    return None


def _place_blocker_frame(
    shape: tuple[int, int],
    current_pos: Coord,
    current_dir: Coord,
    color: int,
    occupied_bboxes: set[Coord],
    reserved_path: set[Coord],
) -> tuple[Frame, list[Coord]] | None:
    hit_side = DIR_TO_ENTRY_SIDE[current_dir]
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    valid_gap_side_pairs = [
        ("top", "right"),
        ("right", "bottom"),
        ("bottom", "left"),
        ("left", "top"),
    ]
    valid_gap_side_pairs = [pair for pair in valid_gap_side_pairs if hit_side not in pair]

    for _ in range(PLACEMENT_ATTEMPTS):
        dist = random.randint(*SEGMENT_LENGTH_RANGE)
        hit_cell = _add(current_pos, current_dir, dist)
        if not _in_bounds(shape, hit_cell):
            continue

        fh = random.randint(*FRAME_SIZE_RANGE)
        fw = random.randint(*FRAME_SIZE_RANGE)
        hit_offset_max = fh - 2 if hit_side in {"left", "right"} else fw - 2
        hit_offset = random.randint(1, hit_offset_max)

        if hit_side == "left":
            bbox = (hit_cell[0] - hit_offset, hit_cell[1], hit_cell[0] - hit_offset + fh - 1, hit_cell[1] + fw - 1)
        elif hit_side == "right":
            bbox = (hit_cell[0] - hit_offset, hit_cell[1] - fw + 1, hit_cell[0] - hit_offset + fh - 1, hit_cell[1])
        elif hit_side == "top":
            bbox = (hit_cell[0], hit_cell[1] - hit_offset, hit_cell[0] + fh - 1, hit_cell[1] - hit_offset + fw - 1)
        else:
            bbox = (hit_cell[0] - fh + 1, hit_cell[1] - hit_offset, hit_cell[0], hit_cell[1] - hit_offset + fw - 1)

        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] >= shape[0] or bbox[3] >= shape[1]:
            continue

        gap_side1, gap_side2 = random.choice(valid_gap_side_pairs)
        offset_max1 = fw - 2 if gap_side1 in {"top", "bottom"} else fh - 2
        offset_max2 = fw - 2 if gap_side2 in {"top", "bottom"} else fh - 2
        frame = _frame_from_bbox(
            bbox,
            color,
            gap_side1,
            random.randint(1, offset_max1),
            gap_side2,
            random.randint(1, offset_max2),
        )
        if hit_cell in frame.gaps:
            continue
        if not _frame_is_clear(frame, occupied_bboxes, reserved_path):
            continue

        path_cells = _line_cells(current_pos, _add(hit_cell, (-current_dir[0], -current_dir[1])))
        if len(path_cells) < 2:
            continue
        if not _path_segment_is_clear(path_cells, occupied_bboxes, reserved_path):
            continue
        return frame, path_cells
    return None


def _place_distractor_frame(
    shape: tuple[int, int],
    color: int,
    occupied_bboxes: set[Coord],
    reserved_path: set[Coord],
) -> Frame | None:
    h, w = shape
    for _ in range(PLACEMENT_ATTEMPTS):
        fh = random.randint(*FRAME_SIZE_RANGE)
        fw = random.randint(*FRAME_SIZE_RANGE)
        if fh >= h or fw >= w:
            continue
        r0 = random.randint(0, h - fh)
        c0 = random.randint(0, w - fw)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)

        side1 = random.choice(["top", "bottom", "left", "right"])
        side2 = random.choice([side for side in ["top", "bottom", "left", "right"] if side != side1 and side != {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[side1]])
        offset_max1 = fw - 2 if side1 in {"top", "bottom"} else fh - 2
        offset_max2 = fw - 2 if side2 in {"top", "bottom"} else fh - 2
        frame = _frame_from_bbox(
            bbox,
            color,
            side1,
            random.randint(1, offset_max1),
            side2,
            random.randint(1, offset_max2),
        )
        if _frame_is_clear(frame, occupied_bboxes, reserved_path):
            return frame
    return None


def _path_to_edge(shape: tuple[int, int], current_pos: Coord, current_dir: Coord) -> list[Coord]:
    cells: list[Coord] = []
    cur = current_pos
    while True:
        nxt = _add(cur, current_dir)
        if not _in_bounds(shape, nxt):
            break
        cells.append(nxt)
        cur = nxt
    return cells


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(*GRID_SIZE_RANGE), random.randint(*GRID_SIZE_RANGE))
    relay_count = random.choices(RELAY_COUNT_CHOICES, weights=RELAY_COUNT_WEIGHTS, k=1)[0]
    extra_count = random.choices(EXTRA_FRAME_CHOICES, weights=EXTRA_FRAME_WEIGHTS, k=1)[0]
    stop_mode = random.choice(["edge", "wall"])

    total_frames = 1 + relay_count + extra_count + int(stop_mode == "wall")
    if total_frames + 1 > len(NON_BG_COLORS):
        return None
    palette = random.sample(NON_BG_COLORS, total_frames + 1)
    marker_color = palette[0]
    frame_colors = palette[1:]
    color_idx = 0

    occupied_bboxes: set[Coord] = set()
    reserved_path: set[Coord] = set()
    frames: list[Frame] = []

    start = _place_start_frame(shape, frame_colors[color_idx], occupied_bboxes, reserved_path)
    if start is None:
        return None
    start_frame, marker, current_pos, path_cells = start
    color_idx += 1
    frames.append(start_frame)
    occupied_bboxes |= _bbox_cells(start_frame.bbox)
    reserved_path |= set(path_cells)
    current_dir = SIDE_TO_OUTWARD[start_frame.gap_side(current_pos)]

    for _ in range(relay_count):
        placed = _place_relay_frame(shape, current_pos, current_dir, frame_colors[color_idx], occupied_bboxes, reserved_path)
        if placed is None:
            return None
        frame, new_path, current_pos, current_dir = placed
        color_idx += 1
        frames.append(frame)
        occupied_bboxes |= _bbox_cells(frame.bbox)
        reserved_path |= set(new_path)
        path_cells.extend(new_path)

    if stop_mode == "edge":
        final_cells = _path_to_edge(shape, current_pos, current_dir)
        if len(final_cells) < 2 or not _path_segment_is_clear(final_cells, occupied_bboxes, reserved_path):
            return None
        path_cells.extend(final_cells)
        reserved_path |= set(final_cells)
    else:
        blocker = _place_blocker_frame(shape, current_pos, current_dir, frame_colors[color_idx], occupied_bboxes, reserved_path)
        if blocker is None:
            return None
        blocker_frame, final_cells = blocker
        color_idx += 1
        frames.append(blocker_frame)
        occupied_bboxes |= _bbox_cells(blocker_frame.bbox)
        reserved_path |= set(final_cells)
        path_cells.extend(final_cells)

    for _ in range(extra_count):
        frame = _place_distractor_frame(shape, frame_colors[color_idx], occupied_bboxes, reserved_path)
        if frame is None:
            return None
        color_idx += 1
        frames.append(frame)
        occupied_bboxes |= _bbox_cells(frame.bbox)

    inp = _render_input(shape, frames, marker, marker_color)
    expected = _paint_expected(inp, path_cells, marker_color)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 011-1")
