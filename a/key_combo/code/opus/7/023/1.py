from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
BORDER_COLORS = (1, 3, 4, 6)
SNAKE_COLORS = (2, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        comp: list[tuple[int, int]] = []
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _trace_snake(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = np.isin(grid, SNAKE_COLORS)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert coords, "expected a snake path"
    coord_set = set(coords)
    comps = _connected_components(mask)
    assert len(comps) == 1, "snake must be a single 4-connected component"

    degree: dict[tuple[int, int], int] = {}
    for cell in coords:
        degree[cell] = sum((nbr in coord_set) for nbr in _neighbors4(*cell))
        assert degree[cell] in (1, 2), "snake path cannot branch"

    endpoints = [cell for cell, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "snake must have exactly two endpoints"
    start = min(endpoints)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        order.append(cur)
        next_cells = [nbr for nbr in _neighbors4(*cur) if nbr in coord_set and nbr != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "snake traversal must remain unique"
        prev, cur = cur, next_cells[0]

    assert len(order) == len(coords), "snake traversal missed cells"
    return order


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    h, w = grid.shape
    for color in BORDER_COLORS:
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == color)]
        if not pts:
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue

        perimeter: list[tuple[int, int]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in (r0, r1) or c in (c0, c1):
                    perimeter.append((r, c))

        perimeter_set = set(perimeter)
        if any((r, c) not in perimeter_set for r, c in pts):
            continue

        missing = [(r, c) for r, c in perimeter if int(grid[r, c]) != color]
        if len(missing) not in (0, 2):
            continue
        if any((r in (r0, r1) and c in (c0, c1)) for r, c in missing):
            continue
        if any(int(grid[r, c]) not in SNAKE_COLORS for r, c in missing):
            continue

        valid_perimeter = all(
            int(grid[r, c]) == color or (r, c) in missing for r, c in perimeter
        )
        if not valid_perimeter:
            continue

        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "missing": tuple(sorted(missing)),
            }
        )

    bboxes = [frame["bbox"] for frame in frames]
    assert len(bboxes) == len(set(bboxes)), "frame detection is ambiguous"
    return sorted(frames, key=lambda frame: frame["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the traversed marker-rich frames using the snake's arrival color, and erase everything else.

    Input structure:
    - The background is black.
    - Blue/green/yellow/pink rectangular frames have one-pixel borders and black-or-grey interiors.
    - A single 4-connected multicolor snake in red/orange/teal/maroon may overwrite two border cells and a straight interior strip while crossing some frames.

    Rule:
    - Trace the snake from the endpoint with smaller row, breaking ties by smaller column.
    - For each frame, record the color of the first snake cell that lies inside that frame's bounding box.
    - Count grey cells only in the strict interior of each frame.
    - Keep a frame iff it was traversed and its grey-count is greater than one; kept frames preserve their border and recolor their whole strict interior to the recorded arrival color.
    - Frames not traversed, or traversed with grey-count at most one, are erased to black, and the snake is erased everywhere.

    Key insight:
    - The snake is only a temporal probe: traversal order decides which border crossing is the entry, while interior grey-count independently decides whether that visited frame survives at all.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    snake_order = _trace_snake(g)
    arrivals: dict[tuple[int, int, int, int], int] = {}

    for r, c in snake_order:
        for frame in frames:
            bbox = frame["bbox"]
            if bbox in arrivals:
                continue
            r0, c0, r1, c1 = bbox
            if r0 <= r <= r1 and c0 <= c <= c1:
                arrivals[bbox] = int(g[r, c])

    out = np.zeros_like(g)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        grey_count = int(np.count_nonzero(interior == GREY))
        if frame["bbox"] not in arrivals or grey_count <= 1:
            continue

        arrival_color = arrivals[frame["bbox"]]
        out[r0, c0 : c1 + 1] = frame["color"]
        out[r1, c0 : c1 + 1] = frame["color"]
        out[r0 : r1 + 1, c0] = frame["color"]
        out[r0 : r1 + 1, c1] = frame["color"]
        out[r0 + 1 : r1, c0 + 1 : c1] = arrival_color

    return out


def _random_partition(total: int, parts: int, min_len: int, max_len: int) -> list[int] | None:
    if parts * min_len > total or parts * max_len < total:
        return None
    values = [min_len] * parts
    remain = total - sum(values)
    while remain:
        idxs = [i for i, value in enumerate(values) if value < max_len]
        if not idxs:
            return None
        i = random.choice(idxs)
        values[i] += 1
        remain -= 1
    random.shuffle(values)
    return values


def _build_staircase_path(h: int, w: int) -> list[tuple[int, int]] | None:
    horizontal_runs = 4 if w <= 21 else random.choice([4, 5])
    h_steps = [random.randint(3, 5) for _ in range(horizontal_runs)]
    if 2 + sum(h_steps) > w - 3:
        return None

    v_steps = [random.randint(3, 5) for _ in range(horizontal_runs - 1)]
    start_row = random.randint(4, h - 5)
    current_row = start_row
    used_up = False
    used_down = False
    vertical_dirs: list[int] = []

    for i, steps in enumerate(v_steps):
        feasible: list[int] = []
        if current_row + steps <= h - 3:
            feasible.append(1)
        if current_row - steps >= 2:
            feasible.append(-1)
        if not feasible:
            return None

        random.shuffle(feasible)
        chosen = feasible[0]
        if not used_up and -1 in feasible and random.random() < 0.7:
            chosen = -1
        elif not used_down and 1 in feasible and random.random() < 0.7:
            chosen = 1

        vertical_dirs.append(chosen)
        current_row += chosen * steps
        used_up |= chosen == -1
        used_down |= chosen == 1

    if not used_up or not used_down:
        return None

    path = [(start_row, 2)]
    r, c = path[0]
    for i in range(horizontal_runs):
        for _ in range(h_steps[i]):
            c += 1
            path.append((r, c))
        if i == horizontal_runs - 1:
            break
        for _ in range(v_steps[i]):
            r += vertical_dirs[i]
            path.append((r, c))

    assert max(rr for rr, _ in path) < h and max(cc for _, cc in path) < w
    if path[-1] < path[0]:
        path.reverse()
    return path


def _path_runs(path: list[tuple[int, int]]) -> list[dict]:
    runs: list[dict] = []
    diffs = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    start = 0
    while start < len(diffs):
        end = start
        while end + 1 < len(diffs) and diffs[end + 1] == diffs[start]:
            end += 1
        runs.append(
            {
                "start_idx": start,
                "end_idx": end + 1,
                "dir": diffs[start],
                "cells": path[start : end + 2],
            }
        )
        start = end + 1
    return runs


def _path_cells_in_bbox(
    path_set: set[tuple[int, int]], bbox: tuple[int, int, int, int]
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r, c in path_set if r0 <= r <= r1 and c0 <= c <= c1}


def _enumerate_frame_candidates(
    path: list[tuple[int, int]], shape: tuple[int, int]
) -> list[dict]:
    h, w = shape
    path_set = set(path)
    candidates: list[dict] = []

    for run_index, run in enumerate(_path_runs(path)):
        cells = run["cells"]
        dr, dc = run["dir"]
        max_len = min(6, len(cells))
        for seg_len in range(4, max_len + 1):
            for start in range(0, len(cells) - seg_len + 1):
                segment = cells[start : start + seg_len]
                if dr == 0:
                    row = segment[0][0]
                    cols = [c for _, c in segment]
                    c0, c1 = min(cols), max(cols)
                    for frame_h in range(4, 7):
                        for interior_row in range(1, frame_h - 1):
                            r0 = row - interior_row
                            r1 = r0 + frame_h - 1
                            if r0 < 0 or r1 >= h:
                                continue
                            bbox = (r0, c0, r1, c1)
                            inside = _path_cells_in_bbox(path_set, bbox)
                            if inside != set(segment):
                                continue
                            candidates.append(
                                {
                                    "bbox": bbox,
                                    "segment": segment,
                                    "run_index": run_index,
                                    "dir": (dr, dc),
                                    "orientation": "h",
                                    "path_indices": tuple(
                                        path.index(cell) for cell in segment
                                    ),
                                }
                            )
                else:
                    col = segment[0][1]
                    rows = [r for r, _ in segment]
                    r0, r1 = min(rows), max(rows)
                    for frame_w in range(4, 7):
                        for interior_col in range(1, frame_w - 1):
                            c0 = col - interior_col
                            c1 = c0 + frame_w - 1
                            if c0 < 0 or c1 >= w:
                                continue
                            bbox = (r0, c0, r1, c1)
                            inside = _path_cells_in_bbox(path_set, bbox)
                            if inside != set(segment):
                                continue
                            candidates.append(
                                {
                                    "bbox": bbox,
                                    "segment": segment,
                                    "run_index": run_index,
                                    "dir": (dr, dc),
                                    "orientation": "v",
                                    "path_indices": tuple(
                                        path.index(cell) for cell in segment
                                    ),
                                }
                            )

    return candidates


def _bboxes_separate(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1
) -> bool:
    return (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _pick_traversed_frames(
    candidates: list[dict], total_frames: int
) -> list[dict] | None:
    traversed_count = 2 if total_frames == 3 else random.choice([2, 3])
    horizontals = [cand for cand in candidates if cand["orientation"] == "h"]
    upward_verticals = [cand for cand in candidates if cand["dir"] == (-1, 0)]
    if not horizontals or not upward_verticals:
        return None

    random.shuffle(horizontals)
    random.shuffle(upward_verticals)
    random.shuffle(candidates)

    for first in horizontals:
        for second in upward_verticals:
            if first["bbox"] == second["bbox"]:
                continue
            if first["run_index"] == second["run_index"]:
                continue
            if not _bboxes_separate(first["bbox"], second["bbox"]):
                continue
            chosen = [first, second]
            used_runs = {first["run_index"], second["run_index"]}
            for cand in candidates:
                if len(chosen) >= traversed_count:
                    break
                if cand["run_index"] in used_runs:
                    continue
                if any(not _bboxes_separate(cand["bbox"], prev["bbox"]) for prev in chosen):
                    continue
                chosen.append(cand)
                used_runs.add(cand["run_index"])
            if len(chosen) == traversed_count:
                return chosen
    return None


def _search_untraversed_bbox(
    shape: tuple[int, int],
    path_set: set[tuple[int, int]],
    existing: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    h, w = shape
    sizes = [(fh, fw) for fh in range(4, 8) for fw in range(4, 8)]
    random.shuffle(sizes)
    for frame_h, frame_w in sizes:
        placements: list[tuple[int, int, int, int]] = []
        for r0 in range(1, h - frame_h):
            for c0 in range(1, w - frame_w):
                bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
                if any(not _bboxes_separate(bbox, other) for other in existing):
                    continue
                nearby_path = False
                for r in range(r0 - 1, r0 + frame_h + 1):
                    for c in range(c0 - 1, c0 + frame_w + 1):
                        if (r, c) in path_set:
                            nearby_path = True
                            break
                    if nearby_path:
                        break
                if nearby_path:
                    continue
                placements.append(bbox)
        if placements:
            return random.choice(placements)
    return None


def _choose_marker_counts(traversed_count: int, untraversed_count: int) -> tuple[list[int], list[int]]:
    traversed = [random.choice([2, 3]), random.choice([0, 1])]
    while len(traversed) < traversed_count:
        traversed.append(random.randint(0, 3))
    random.shuffle(traversed)

    untraversed = [random.choice([2, 3])]
    while len(untraversed) < untraversed_count:
        untraversed.append(random.randint(0, 3))
    random.shuffle(untraversed)
    return traversed, untraversed


def _strict_interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _assign_snake_colors(path: list[tuple[int, int]]) -> list[int] | None:
    palette_size = random.choice([3, 4])
    palette = random.sample(list(SNAKE_COLORS), palette_size)
    min_segments = max(4, palette_size, (len(path) + 6) // 7)
    max_segments = min(10, len(path) // 2)
    if min_segments > max_segments:
        return None
    segment_count = random.randint(min_segments, max_segments)
    lengths = _random_partition(len(path), segment_count, 2, 7)
    if lengths is None:
        return None

    seg_colors = palette[:]
    random.shuffle(seg_colors)
    while len(seg_colors) < segment_count:
        options = [color for color in palette if color != seg_colors[-1]]
        seg_colors.append(random.choice(options))

    colors: list[int] = []
    for length, color in zip(lengths, seg_colors):
        colors.extend([color] * length)
    return colors


def _render_grid(
    shape: tuple[int, int],
    path: list[tuple[int, int]],
    path_colors: list[int],
    frames: list[dict],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = frame["color"]
        grid[r1, c0 : c1 + 1] = frame["color"]
        grid[r0 : r1 + 1, c0] = frame["color"]
        grid[r0 : r1 + 1, c1] = frame["color"]
        for r, c in frame["markers"]:
            grid[r, c] = GREY

    for (r, c), color in zip(path, path_colors):
        grid[r, c] = color
    return grid


def _expected_output_from_meta(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for frame in frames:
        if not frame["keep"]:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        out[r0, c0 : c1 + 1] = frame["color"]
        out[r1, c0 : c1 + 1] = frame["color"]
        out[r0 : r1 + 1, c0] = frame["color"]
        out[r0 : r1 + 1, c1] = frame["color"]
        out[r0 + 1 : r1, c0 + 1 : c1] = frame["arrival_color"]
    return out


def _validate_semantics(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    frames: list[dict],
    expected: np.ndarray,
) -> bool:
    traced = _trace_snake(grid)
    if traced != path:
        return False

    detected = _find_frames(grid)
    detected_key = sorted((item["bbox"], item["color"]) for item in detected)
    meta_key = sorted((item["bbox"], item["color"]) for item in frames)
    if detected_key != meta_key:
        return False

    traversed = [frame for frame in frames if frame["traversed"]]
    untraversed = [frame for frame in frames if not frame["traversed"]]
    if not any(frame["grey_count"] > 1 for frame in traversed):
        return False
    if not any(frame["grey_count"] <= 1 for frame in traversed):
        return False
    if not any(frame["grey_count"] > 1 for frame in untraversed):
        return False
    if not any(frame["dir"] == (-1, 0) for frame in traversed):
        return False
    if not any(frame["orientation"] == "h" for frame in traversed):
        return False
    if len({frame["arrival_color"] for frame in traversed}) < 2:
        return False
    if not any(frame["entry_color"] != frame["exit_color"] for frame in traversed):
        return False

    return np.array_equal(main(grid), expected)


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    path = _build_staircase_path(h, w)
    if path is None:
        return None

    total_frames = random.choice([3, 4])
    candidates = _enumerate_frame_candidates(path, (h, w))
    traversed_candidates = _pick_traversed_frames(candidates, total_frames)
    if traversed_candidates is None:
        return None

    path_set = set(path)
    used_bboxes = [cand["bbox"] for cand in traversed_candidates]
    untraversed_needed = total_frames - len(traversed_candidates)
    untraversed_bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(untraversed_needed):
        bbox = _search_untraversed_bbox((h, w), path_set, used_bboxes + untraversed_bboxes)
        if bbox is None:
            return None
        untraversed_bboxes.append(bbox)

    border_colors = random.sample(list(BORDER_COLORS), total_frames)
    traversed_counts, untraversed_counts = _choose_marker_counts(
        len(traversed_candidates), len(untraversed_bboxes)
    )

    path_colors = _assign_snake_colors(path)
    if path_colors is None:
        return None

    frames: list[dict] = []
    for index, cand in enumerate(traversed_candidates):
        bbox = cand["bbox"]
        segment = list(cand["segment"])
        segment_indices = sorted(cand["path_indices"])
        interior_cells = [cell for cell in _strict_interior_cells(bbox) if cell not in path_set]
        grey_count = traversed_counts[index]
        if grey_count > len(interior_cells):
            return None
        markers = random.sample(interior_cells, grey_count)
        arrival_idx = segment_indices[0]
        exit_idx = segment_indices[-1]
        frames.append(
            {
                "bbox": bbox,
                "color": border_colors[index],
                "markers": markers,
                "grey_count": grey_count,
                "traversed": True,
                "keep": grey_count > 1,
                "segment": segment,
                "orientation": cand["orientation"],
                "dir": cand["dir"],
                "arrival_color": path_colors[arrival_idx],
                "entry_color": path_colors[arrival_idx],
                "exit_color": path_colors[exit_idx],
            }
        )

    for index, bbox in enumerate(untraversed_bboxes):
        interior_cells = _strict_interior_cells(bbox)
        grey_count = untraversed_counts[index]
        if grey_count > len(interior_cells):
            return None
        markers = random.sample(interior_cells, grey_count)
        frames.append(
            {
                "bbox": bbox,
                "color": border_colors[len(traversed_candidates) + index],
                "markers": markers,
                "grey_count": grey_count,
                "traversed": False,
                "keep": False,
                "segment": [],
                "orientation": "none",
                "dir": (0, 0),
                "arrival_color": -1,
                "entry_color": -1,
                "exit_color": -1,
            }
        )

    expected = _expected_output_from_meta((h, w), frames)
    grid = _render_grid((h, w), path, path_colors, frames)
    if not _validate_semantics(grid, path, frames, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-frame ARC instance")
