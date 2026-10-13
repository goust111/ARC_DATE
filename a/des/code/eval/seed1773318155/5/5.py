from __future__ import annotations

import random
from collections import deque

import numpy as np


START_COLOR = 3
END_COLOR = 2
TURN_COLOR = 7
ALL_COLORS = tuple(range(10))
INPUT_COLOR_POOL = tuple(c for c in ALL_COLORS if c not in (START_COLOR, END_COLOR, TURN_COLOR))

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SEGMENT_COUNT_RANGE = (3, 7)
SEGMENT_LENGTH_RANGE = (2, 5)
MARGIN_RANGE = (1, 4)
MAX_SPAN = 22
GEN_RETRIES = 400
PATH_RETRIES = 200


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] - b[0], a[1] - b[1])


def _scale(v: tuple[int, int], k: int) -> tuple[int, int]:
    return (v[0] * k, v[1] * k)


def _left_turn(v: tuple[int, int]) -> tuple[int, int]:
    dr, dc = v
    return (-dc, dr)


def _right_turn(v: tuple[int, int]) -> tuple[int, int]:
    dr, dc = v
    return (dc, -dr)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _find_unique_singleton(grid: np.ndarray, color: int) -> tuple[int, int]:
    comps = [comp for comp in _components_of_color(grid, color) if len(comp) == 1]
    assert len(comps) == 1, f"expected one singleton component of color {color}"
    return comps[0][0]


def _ordered_segment(cells: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], str]:
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    assert 2 <= len(cells) <= 5, "each segment must be length 2-5"

    if len(rows) == 1:
        row = rows[0]
        ordered = sorted(cells, key=lambda p: p[1])
        expected = [(row, c) for c in range(ordered[0][1], ordered[-1][1] + 1)]
        assert ordered == expected, "horizontal segment must be contiguous"
        return ordered, "H"

    if len(cols) == 1:
        col = cols[0]
        ordered = sorted(cells, key=lambda p: p[0])
        expected = [(r, col) for r in range(ordered[0][0], ordered[-1][0] + 1)]
        assert ordered == expected, "vertical segment must be contiguous"
        return ordered, "V"

    raise AssertionError("segment must be straight")


def _extract_segments(
    grid: np.ndarray, bg: int, start_marker: tuple[int, int], end_marker: tuple[int, int]
) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    blocked = {start_marker, end_marker}
    segments: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            if (r, c) in blocked or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or (nr, nc) in blocked or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))

            ordered, orientation = _ordered_segment(cells)
            inward_dirs = (_sub(ordered[1], ordered[0]), _sub(ordered[-2], ordered[-1]))
            segments.append(
                {
                    "id": len(segments),
                    "color": color,
                    "cells": ordered,
                    "orientation": orientation,
                    "endpoints": (ordered[0], ordered[-1]),
                    "inward_dirs": inward_dirs,
                }
            )

    assert segments, "expected at least one segment"
    return segments


def _gap_midpoint(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    assert (dr == 2 and dc == 0) or (dr == 0 and dc == 2), "connected endpoints must be one gap apart"
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def _endpoint_links(
    grid: np.ndarray,
    bg: int,
    segments: list[dict],
    start_marker: tuple[int, int],
    end_marker: tuple[int, int],
) -> tuple[dict[tuple[int, int], list[tuple[str, tuple[int, int] | None]]], tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    endpoint_links: dict[tuple[int, int], list[tuple[str, tuple[int, int] | None]]] = {}
    endpoint_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for seg in segments:
        for end_idx, endpoint in enumerate(seg["endpoints"]):
            key = (seg["id"], end_idx)
            endpoint_links[key] = []
            endpoint_lookup[endpoint] = key

    def attach_marker(marker_name: str, marker_pos: tuple[int, int]) -> tuple[int, int]:
        adjacent_non_bg: list[tuple[int, int]] = []
        for nr, nc in _neighbors4(*marker_pos):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != bg and (nr, nc) not in (start_marker, end_marker):
                adjacent_non_bg.append((nr, nc))
        assert len(adjacent_non_bg) == 1, f"{marker_name} marker must touch exactly one segment cell"
        endpoint = adjacent_non_bg[0]
        assert endpoint in endpoint_lookup, f"{marker_name} marker must touch an endpoint"
        ref = endpoint_lookup[endpoint]
        endpoint_links[ref].append((marker_name, None))
        return ref

    start_ref = attach_marker("start", start_marker)
    end_ref = attach_marker("end", end_marker)

    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1 :]:
            for end_a, ep_a in enumerate(seg_a["endpoints"]):
                for end_b, ep_b in enumerate(seg_b["endpoints"]):
                    same_row = ep_a[0] == ep_b[0] and abs(ep_a[1] - ep_b[1]) == 2
                    same_col = ep_a[1] == ep_b[1] and abs(ep_a[0] - ep_b[0]) == 2
                    if not (same_row or same_col):
                        continue
                    gap = _gap_midpoint(ep_a, ep_b)
                    if int(grid[gap]) != bg:
                        continue
                    endpoint_links[(seg_a["id"], end_a)].append(("segment", (seg_b["id"], end_b)))
                    endpoint_links[(seg_b["id"], end_b)].append(("segment", (seg_a["id"], end_a)))

    for key, links in endpoint_links.items():
        assert len(links) == 1, f"endpoint {key} must have exactly one external connection"

    return endpoint_links, start_ref, end_ref


def _ordered_path(
    segments: list[dict],
    endpoint_links: dict[tuple[int, int], list[tuple[str, tuple[int, int] | None]]],
    start_ref: tuple[int, int],
    end_ref: tuple[int, int],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    visited: set[int] = set()
    current_seg, entry_end = start_ref

    while True:
        assert current_seg not in visited, "path may not branch or cycle"
        visited.add(current_seg)
        path.append((current_seg, entry_end))

        exit_end = 1 - entry_end
        link_kind, target = endpoint_links[(current_seg, exit_end)][0]
        if link_kind == "end":
            assert (current_seg, exit_end) == end_ref, "path must terminate at the red marker endpoint"
            break

        assert link_kind == "segment" and target is not None
        current_seg, entry_end = target

    assert len(visited) == len(segments), "all segments must lie on the unique path"
    return path


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    start_marker = _find_unique_singleton(g, START_COLOR)
    end_marker = _find_unique_singleton(g, END_COLOR)
    assert start_marker != end_marker
    segments = _extract_segments(g, bg, start_marker, end_marker)
    endpoint_links, start_ref, end_ref = _endpoint_links(g, bg, segments, start_marker, end_marker)
    path = _ordered_path(segments, endpoint_links, start_ref, end_ref)
    return {
        "bg": bg,
        "start_marker": start_marker,
        "end_marker": end_marker,
        "segments": segments,
        "endpoint_links": endpoint_links,
        "path": path,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered line fragments into the unique start-to-end path,
    fill one-cell gaps using the color of the earlier segment, and mark turn
    interiors in orange.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    bg = info["bg"]
    segments = info["segments"]
    path = info["path"]

    out = g.copy()
    for idx in range(len(path) - 1):
        seg_id, entry_end = path[idx]
        next_seg_id, next_entry_end = path[idx + 1]

        seg = segments[seg_id]
        next_seg = segments[next_seg_id]
        exit_point = seg["endpoints"][1 - entry_end]
        next_entry_point = next_seg["endpoints"][next_entry_end]

        gap = _gap_midpoint(exit_point, next_entry_point)
        assert int(out[gap]) == bg, "gap cell must be background in the input"
        out[gap] = int(seg["color"])

        if seg["orientation"] != next_seg["orientation"]:
            dr, dc = next_seg["inward_dirs"][next_entry_end]
            corner = (gap[0] + dr, gap[1] + dc)
            assert 0 <= corner[0] < out.shape[0] and 0 <= corner[1] < out.shape[1]
            assert int(out[corner]) == bg, "turn interior must be background in the input"
            out[corner] = TURN_COLOR

    return out


def _line_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [_add(start, _scale(direction, k)) for k in range(length)]


def _segment_color_sequence(rng: random.Random, count: int, bg: int) -> list[int]:
    palette = [c for c in INPUT_COLOR_POOL if c != bg]
    colors: list[int] = []
    previous = None
    for _ in range(count):
        choices = [c for c in palette if c != previous]
        color = rng.choice(choices)
        colors.append(color)
        previous = color
    return colors


def _candidate_kinds(
    rng: random.Random,
    prev_dir: tuple[int, int],
    new_start: tuple[int, int],
    gap: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict[str, list[tuple[tuple[int, int], int, list[tuple[int, int]], tuple[int, int] | None]]]:
    kinds: dict[str, list[tuple[tuple[int, int], int, list[tuple[int, int]], tuple[int, int] | None]]] = {
        "straight": [],
        "left": [],
        "right": [],
    }

    directions = {
        "straight": prev_dir,
        "left": _left_turn(prev_dir),
        "right": _right_turn(prev_dir),
    }
    lengths = list(range(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1] + 1))
    rng.shuffle(lengths)

    for kind, direction in directions.items():
        for length in lengths:
            cells = _line_cells(new_start, direction, length)
            corner = _add(gap, direction) if kind != "straight" else None
            cell_set = set(cells)
            if cell_set & occupied or cell_set & reserved:
                continue
            if gap in occupied or gap in reserved:
                continue
            if corner is not None and (corner in occupied or corner in reserved or corner in cell_set or corner == gap):
                continue
            if any(any(neigh in occupied for neigh in _neighbors4(r, c)) for r, c in cells):
                continue
            kinds[kind].append((direction, length, cells, corner))

    return kinds


def _choose_kind(
    rng: random.Random,
    kinds: dict[str, list[tuple[tuple[int, int], int, list[tuple[int, int]], tuple[int, int] | None]]],
) -> str | None:
    available = [kind for kind, options in kinds.items() if options]
    if not available:
        return None
    weights = []
    for kind in available:
        weights.append(2 if kind == "straight" else 1)
    return rng.choices(available, weights=weights, k=1)[0]


def _build_path_geometry(
    rng: random.Random,
) -> tuple[list[dict], tuple[int, int], tuple[int, int], set[tuple[int, int]]] | None:
    segment_count = rng.randint(*SEGMENT_COUNT_RANGE)
    first_dir = rng.choice(DIRS)
    first_len = rng.randint(*SEGMENT_LENGTH_RANGE)
    first_start = (0, 0)
    first_cells = _line_cells(first_start, first_dir, first_len)
    start_marker = _sub(first_start, first_dir)

    occupied = set(first_cells)
    reserved = {start_marker}
    segments = [{"dir": first_dir, "cells": first_cells}]

    for _ in range(segment_count - 1):
        prev_dir = segments[-1]["dir"]
        prev_end = segments[-1]["cells"][-1]
        gap = _add(prev_end, prev_dir)
        new_start = _add(prev_end, _scale(prev_dir, 2))
        kinds = _candidate_kinds(rng, prev_dir, new_start, gap, occupied, reserved)
        chosen_kind = _choose_kind(rng, kinds)
        if chosen_kind is None:
            return None
        direction, _length, cells, corner = rng.choice(kinds[chosen_kind])
        occupied.update(cells)
        reserved.add(gap)
        if corner is not None:
            reserved.add(corner)
        segments.append({"dir": direction, "cells": cells})

    end_marker = _add(segments[-1]["cells"][-1], segments[-1]["dir"])
    if end_marker in occupied or end_marker in reserved:
        return None
    reserved.add(end_marker)

    all_points = occupied | reserved
    rows = [r for r, _c in all_points]
    cols = [c for _r, c in all_points]
    if max(rows) - min(rows) + 1 > MAX_SPAN or max(cols) - min(cols) + 1 > MAX_SPAN:
        return None

    return segments, start_marker, end_marker, reserved


def _translate_point(point: tuple[int, int], offset: tuple[int, int]) -> tuple[int, int]:
    return (point[0] + offset[0], point[1] + offset[1])


def _render_grid(
    rng: random.Random,
    segment_shapes: list[dict],
    start_marker: tuple[int, int],
    end_marker: tuple[int, int],
) -> np.ndarray:
    all_points = set()
    for seg in segment_shapes:
        all_points.update(seg["cells"])
    all_points.add(start_marker)
    all_points.add(end_marker)

    rows = [r for r, _c in all_points]
    cols = [c for _r, c in all_points]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    top = rng.randint(*MARGIN_RANGE)
    bottom = rng.randint(*MARGIN_RANGE)
    left = rng.randint(*MARGIN_RANGE)
    right = rng.randint(*MARGIN_RANGE)
    height = (rmax - rmin + 1) + top + bottom
    width = (cmax - cmin + 1) + left + right
    offset = (-rmin + top, -cmin + left)

    bg = rng.choice(INPUT_COLOR_POOL)
    segment_colors = _segment_color_sequence(rng, len(segment_shapes), bg)
    grid = np.full((height, width), bg, dtype=int)

    for seg, color in zip(segment_shapes, segment_colors):
        translated_cells = [_translate_point(cell, offset) for cell in seg["cells"]]
        seg["cells"] = translated_cells
        seg["color"] = color
        for r, c in translated_cells:
            grid[r, c] = color

    start_marker_t = _translate_point(start_marker, offset)
    end_marker_t = _translate_point(end_marker, offset)
    grid[start_marker_t] = START_COLOR
    grid[end_marker_t] = END_COLOR
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    geometry = _build_path_geometry(rng)
    if geometry is None:
        return None

    segments, start_marker, end_marker, _reserved = geometry
    segments = [{"dir": seg["dir"], "cells": list(seg["cells"])} for seg in segments]
    grid = _render_grid(rng, segments, start_marker, end_marker)

    try:
        analyzed = _analyze_grid(grid)
        if len(analyzed["path"]) != len(segments):
            return None
        main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
