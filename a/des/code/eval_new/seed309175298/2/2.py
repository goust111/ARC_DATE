import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
MAGENTA = 6
ORANGE = 7
AZURE = 8

PATH_COLORS = (BLUE, YELLOW, MAGENTA, ORANGE, AZURE)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _ordered_straight_line(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    assert len(rows) == 1 or len(cols) == 1, "segment must be a straight line"
    if len(rows) == 1:
        ordered = sorted(coords, key=lambda p: p[1])
        cols_sorted = [c for _, c in ordered]
        assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1))
        return ordered
    ordered = sorted(coords, key=lambda p: p[0])
    rows_sorted = [r for r, _ in ordered]
    assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1))
    return ordered


def _analyze_segments(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start_pts = np.argwhere(g == GREEN)
    end_pts = np.argwhere(g == RED)
    assert len(start_pts) == 1 and len(end_pts) == 1, "expected unique Green and Red markers"
    start = tuple(map(int, start_pts[0]))
    end = tuple(map(int, end_pts[0]))

    seg_mask = (g != BLACK) & (g != GREEN) & (g != RED)
    comps = _find_components(seg_mask)
    assert comps, "expected at least one path segment"

    segments = {}
    coord_to_seg: dict[tuple[int, int], int] = {}
    body_to_seg_ids: dict[int, list[int]] = {}

    for seg_id, coords in enumerate(comps):
        ordered = _ordered_straight_line(coords)
        assert 3 <= len(ordered) <= 6, "segment length must be between 3 and 6"
        interior = ordered[1:-1]
        interior_colors = {int(g[r, c]) for r, c in interior}
        assert len(interior_colors) == 1, "segment interior must be monochrome"
        body_color = next(iter(interior_colors))
        assert body_color in PATH_COLORS, "segment body must use a path color"

        colors = {pt: int(g[pt]) for pt in ordered}
        assert colors[ordered[0]] != BLACK and colors[ordered[-1]] != BLACK
        assert colors[ordered[0]] not in (GREEN, RED) and colors[ordered[-1]] not in (GREEN, RED)

        segment = {
            "id": seg_id,
            "ordered": ordered,
            "endpoints": (ordered[0], ordered[-1]),
            "colors": colors,
            "body": body_color,
        }
        segments[seg_id] = segment
        body_to_seg_ids.setdefault(body_color, []).append(seg_id)
        for pt in ordered:
            coord_to_seg[pt] = seg_id

    return {
        "grid": g,
        "start": start,
        "end": end,
        "segments": segments,
        "coord_to_seg": coord_to_seg,
        "body_to_seg_ids": body_to_seg_ids,
    }


def _orient_segment(segment: dict, entry_point: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = segment["ordered"]
    if entry_point == ordered[0]:
        return ordered
    if entry_point == ordered[-1]:
        return list(reversed(ordered))
    raise AssertionError("entry point must be one endpoint of the segment")


def _bridge_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    if a[0] == b[0]:
        row = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(row, c) for c in range(c0 + 1, c1)] if c1 - c0 >= 2 else []
    if a[1] == b[1]:
        col = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, col) for r in range(r0 + 1, r1)] if r1 - r0 >= 2 else []
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bridge the black gaps between color-coded path segments to recover one start-to-end relay path.

    Input structure:
    - Black background.
    - One Green single-pixel start marker and one Red single-pixel end marker.
    - Several 4-connected straight path segments, each 3-6 pixels long, whose interior cells are monochrome.
    - Each segment endpoint is a port pixel: internal ports advertise neighboring segment colors, while the terminal ports next to the markers keep their own segment color.

    Rule:
    - Find the segment endpoint adjacent to the Green marker and treat that component as the first segment.
    - Within each segment, the interior color is the segment's body color; after traversing from one endpoint to the other, the far endpoint color names the next segment's body color.
    - The next segment must answer that handshake by having an endpoint colored with the current segment's body color, and the two matching endpoints must be aligned with only Black cells between them.
    - Fill each such black gap with the current segment's body color, continue until the endpoint adjacent to the Red marker is reached, and keep all original segment pixels and both markers unchanged.

    Key insight:
    - The endpoint colors form a bidirectional handshake: one segment points to the next segment's body color, and the next segment replies with the current body color at the matching end, so both the order and the bridge location become unique.
    """
    info = _analyze_segments(np.asarray(grid, dtype=int))
    g = info["grid"]
    start = info["start"]
    end = info["end"]
    segments = info["segments"]
    coord_to_seg = info["coord_to_seg"]
    body_to_seg_ids = info["body_to_seg_ids"]

    start_neighbors = [pt for pt in _neighbors4(*start) if pt in coord_to_seg]
    end_neighbors = [pt for pt in _neighbors4(*end) if pt in coord_to_seg]
    assert len(start_neighbors) == 1, "start marker must touch exactly one segment endpoint"
    assert len(end_neighbors) == 1, "end marker must touch exactly one segment endpoint"

    current_entry = start_neighbors[0]
    current_seg_id = coord_to_seg[current_entry]
    assert current_entry in segments[current_seg_id]["endpoints"]

    out = g.copy()
    visited: set[int] = set()

    while True:
        assert current_seg_id not in visited, "segment order must not loop"
        visited.add(current_seg_id)
        segment = segments[current_seg_id]
        oriented = _orient_segment(segment, current_entry)
        exit_point = oriented[-1]
        exit_color = segment["colors"][exit_point]
        body_color = int(segment["body"])

        if exit_point in end_neighbors:
            break

        candidates = []
        for next_seg_id in body_to_seg_ids.get(exit_color, []):
            if next_seg_id in visited:
                continue
            next_seg = segments[next_seg_id]
            for entry_point in next_seg["endpoints"]:
                if next_seg["colors"][entry_point] != body_color:
                    continue
                bridge = _bridge_cells(exit_point, entry_point)
                if bridge is None or not bridge:
                    continue
                if all(int(g[r, c]) == BLACK for r, c in bridge):
                    candidates.append((next_seg_id, entry_point, bridge))

        assert len(candidates) == 1, "handshake must identify a unique next segment"
        next_seg_id, next_entry, bridge = candidates[0]
        for r, c in bridge:
            out[r, c] = body_color

        current_seg_id = next_seg_id
        current_entry = next_entry

    assert len(visited) == len(segments), "all segments must be used exactly once"
    return out


def _left_turn(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _right_turn(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _add(pt: tuple[int, int], direction: tuple[int, int], scale: int = 1) -> tuple[int, int]:
    return pt[0] + direction[0] * scale, pt[1] + direction[1] * scale


def _path_is_simple(seq: list[tuple[int, int]]) -> bool:
    index = {}
    for i, pt in enumerate(seq):
        if pt in index:
            return False
        index[pt] = i
    for i, (r, c) in enumerate(seq):
        for nr, nc in _neighbors4(r, c):
            j = index.get((nr, nc))
            if j is not None and abs(j - i) != 1:
                return False
    return True


def _build_instance():
    num_segments = random.randint(3, 5)
    body_colors = random.sample(PATH_COLORS, num_segments)
    lengths = [random.randint(3, 6) for _ in range(num_segments)]
    gaps = [random.randint(1, 4) for _ in range(num_segments - 1)]

    directions = [random.choice(DIRS4)]
    for _ in range(num_segments - 1):
        prev = directions[-1]
        if random.random() < 0.5:
            directions.append(prev)
        else:
            directions.append(random.choice([_left_turn(prev), _right_turn(prev)]))

    segments: list[list[tuple[int, int]]] = []
    gap_cells: list[list[tuple[int, int]]] = []
    current_entry = (0, 0)

    for i in range(num_segments):
        direction = directions[i]
        length = lengths[i]
        ordered = [_add(current_entry, direction, step) for step in range(length)]
        segments.append(ordered)
        if i < num_segments - 1:
            gap = [_add(ordered[-1], direction, step) for step in range(1, gaps[i] + 1)]
            gap_cells.append(gap)
            current_entry = _add(ordered[-1], direction, gaps[i] + 1)

    start = _add(segments[0][0], directions[0], -1)
    end = _add(segments[-1][-1], directions[-1], 1)

    full_sequence = [start]
    for i, ordered in enumerate(segments):
        full_sequence.extend(ordered)
        if i < len(gap_cells):
            full_sequence.extend(gap_cells[i])
    full_sequence.append(end)

    if not _path_is_simple(full_sequence):
        return None

    rows = [r for r, _ in full_sequence]
    cols = [c for _, c in full_sequence]
    bbox_h = max(rows) - min(rows) + 1
    bbox_w = max(cols) - min(cols) + 1
    if bbox_h > 24 or bbox_w > 24:
        return None

    height = random.randint(bbox_h + 1, min(30, bbox_h + 6))
    width = random.randint(bbox_w + 1, min(30, bbox_w + 6))
    row_shift = random.randint(0, height - bbox_h) - min(rows)
    col_shift = random.randint(0, width - bbox_w) - min(cols)

    grid = np.full((height, width), BLACK, dtype=int)
    start_shifted = (start[0] + row_shift, start[1] + col_shift)
    end_shifted = (end[0] + row_shift, end[1] + col_shift)
    grid[start_shifted] = GREEN
    grid[end_shifted] = RED

    for i, ordered in enumerate(segments):
        shifted = [(r + row_shift, c + col_shift) for r, c in ordered]
        entry_color = body_colors[i] if i == 0 else body_colors[i - 1]
        exit_color = body_colors[i] if i == num_segments - 1 else body_colors[i + 1]
        grid[shifted[0]] = entry_color
        grid[shifted[-1]] = exit_color
        for r, c in shifted[1:-1]:
            grid[r, c] = body_colors[i]

    if np.count_nonzero(grid == GREEN) != 1 or np.count_nonzero(grid == RED) != 1:
        return None

    out = main(grid)
    expected_added = sum(gaps)
    if int(np.count_nonzero(out != grid)) != expected_added:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-path ARC input")
