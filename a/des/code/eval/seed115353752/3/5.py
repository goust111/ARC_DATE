import random
from collections import deque

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_LOOP_COUNT = 2
MAX_LOOP_COUNT = 4
MIN_SCENE_H = 14
MAX_SCENE_H = 24
MIN_SCENE_W = 16
MAX_SCENE_W = 30
MIN_OUTER = 7
MAX_OUTER_H = 11
MAX_OUTER_W = 12
BOX_PAD = 2
LEGEND_ROWS = 2
SPLIT_PROB = 0.5
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    cells: list[tuple[int, int]] = []
    if ar == br:
        step = 1 if ac <= bc else -1
        for c in range(ac, bc + step, step):
            cells.append((ar, c))
        return cells
    if ac == bc:
        step = 1 if ar <= br else -1
        for r in range(ar, br + step, step):
            cells.append((r, ac))
        return cells
    raise ValueError("segment must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned: list[tuple[int, int]] = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return cleaned[:]

    cells: list[tuple[int, int]] = []
    for idx in range(len(cleaned) - 1):
        segment = _line_cells(cleaned[idx], cleaned[idx + 1])
        if idx:
            segment = segment[1:]
        cells.extend(segment)
    return cells


def _parse_legend_rows(rows: np.ndarray) -> list[int] | None:
    if rows.shape[0] != 2 or not np.array_equal(rows[0], rows[1]):
        return None

    row = rows[0]
    w = int(row.shape[0])
    colors: list[int] = []
    c = 0
    while c < w:
        if int(row[c]) == BLACK:
            c += 1
            continue

        if c + 1 >= w:
            return None
        color = int(row[c])
        if color == BLACK or int(row[c + 1]) != color:
            return None
        if not bool(np.all(rows[:, c : c + 2] == color)):
            return None
        if color in colors:
            return None

        colors.append(color)
        c += 2
        gap = 0
        while c < w and int(row[c]) == BLACK:
            gap += 1
            c += 1
        if c < w and gap == 0:
            return None

    return colors or None


def _find_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    h = int(grid.shape[0])
    for top in range(h - LEGEND_ROWS, -1, -1):
        colors = _parse_legend_rows(grid[top : top + LEGEND_ROWS])
        if colors is None:
            continue
        if top + LEGEND_ROWS < h and bool(np.any(grid[top + LEGEND_ROWS :] != BLACK)):
            continue
        if top == 0:
            continue
        return top, colors
    raise ValueError("legend not found")


def _color_order(scene: np.ndarray) -> list[int]:
    seen: dict[int, tuple[int, int]] = {}
    h, w = map(int, scene.shape)
    for r in range(h):
        for c in range(w):
            color = int(scene[r, c])
            if color != BLACK and color not in seen:
                seen[color] = (r, c)
    return [color for color, _pos in sorted(seen.items(), key=lambda item: item[1])]


def _extract_endpoints(mask: np.ndarray) -> list[dict[str, tuple[int, int]]]:
    h, w = map(int, mask.shape)
    endpoints: list[dict[str, tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        neighbors: list[tuple[int, int]] = []
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if _in_bounds(h, w, nr, nc) and bool(mask[nr, nc]):
                neighbors.append((nr, nc))
        if len(neighbors) == 1:
            nr, nc = neighbors[0]
            endpoints.append({"pos": (rr, cc), "dir": (rr - nr, cc - nc)})
    endpoints.sort(key=lambda item: item["pos"])
    return endpoints


def _connection_candidates(
    grid: np.ndarray,
    a: dict[str, tuple[int, int]],
    b: dict[str, tuple[int, int]],
) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    (ar, ac) = a["pos"]
    (adr, adc) = a["dir"]
    (br, bc) = b["pos"]
    (bdr, bdc) = b["dir"]

    start = (ar + adr, ac + adc)
    end = (br + bdr, bc + bdc)
    if not _in_bounds(h, w, start[0], start[1]) or not _in_bounds(h, w, end[0], end[1]):
        return []
    if int(grid[start]) != BLACK or int(grid[end]) != BLACK:
        return []

    point_lists: list[list[tuple[int, int]]] = []
    if start == end:
        point_lists.append([a["pos"], start, b["pos"]])
    elif start[0] == end[0] or start[1] == end[1]:
        point_lists.append([a["pos"], start, end, b["pos"]])
    else:
        point_lists.append([a["pos"], start, (start[0], end[1]), end, b["pos"]])
        point_lists.append([a["pos"], start, (end[0], start[1]), end, b["pos"]])

    candidates: list[dict[str, object]] = []
    seen_paths: set[tuple[tuple[int, int], ...]] = set()
    for points in point_lists:
        ok = True
        for rr, cc in points[1:-1]:
            if not _in_bounds(h, w, rr, cc):
                ok = False
                break
        if not ok:
            continue

        cells = _polyline_cells(points)
        for rr, cc in cells[1:-1]:
            if int(grid[rr, cc]) != BLACK:
                ok = False
                break
        if not ok:
            continue

        key = tuple(cells)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        candidates.append(
            {
                "cells": cells,
                "new_cells": frozenset(cells[1:-1]),
            }
        )

    return candidates


def _is_single_cycle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = map(int, mask.shape)
    total = int(coords.shape[0])
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        degree = 0
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if _in_bounds(h, w, nr, nc) and bool(mask[nr, nc]):
                degree += 1
        if degree != 2:
            return False

    start = (int(coords[0, 0]), int(coords[0, 1]))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(h, w, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))

    return len(seen) == total


def _solve_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    endpoints = _extract_endpoints(mask)
    if not endpoints:
        if not _is_single_cycle(mask):
            raise ValueError("color does not form a single cycle")
        return []

    if len(endpoints) % 2:
        raise ValueError("odd endpoint count")

    count = len(endpoints)
    candidate_map: dict[tuple[int, int], list[dict[str, object]]] = {}
    for i in range(count):
        for j in range(i + 1, count):
            candidates = _connection_candidates(grid, endpoints[i], endpoints[j])
            if candidates:
                candidate_map[(i, j)] = candidates

    base_mask = mask.copy()
    solutions: list[list[list[tuple[int, int]]]] = []

    def available_paths(idx: int, unused: tuple[int, ...], occupied: frozenset[tuple[int, int]]) -> list[tuple[int, list[tuple[int, int]], frozenset[tuple[int, int]]]]:
        paths: list[tuple[int, list[tuple[int, int]], frozenset[tuple[int, int]]]] = []
        for other in unused:
            if other == idx:
                continue
            key = (idx, other) if idx < other else (other, idx)
            for candidate in candidate_map.get(key, []):
                new_cells = candidate["new_cells"]
                if new_cells.isdisjoint(occupied):
                    paths.append((other, candidate["cells"], new_cells))
        return paths

    def backtrack(
        unused: tuple[int, ...],
        occupied: frozenset[tuple[int, int]],
        chosen: list[list[tuple[int, int]]],
    ) -> None:
        if len(solutions) > 1:
            return
        if not unused:
            completed = base_mask.copy()
            for cells in chosen:
                for rr, cc in cells[1:-1]:
                    completed[rr, cc] = True
            if _is_single_cycle(completed):
                solutions.append([path[:] for path in chosen])
            return

        best_idx = None
        best_paths = None
        for idx in unused:
            paths = available_paths(idx, unused, occupied)
            if best_paths is None or len(paths) < len(best_paths):
                best_idx = idx
                best_paths = paths
        if best_idx is None or not best_paths:
            return

        best_paths.sort(key=lambda item: (len(item[1]), item[0], item[1]))
        for other, cells, new_cells in best_paths:
            next_unused = tuple(idx for idx in unused if idx not in (best_idx, other))
            backtrack(next_unused, occupied | new_cells, chosen + [cells])

    backtrack(tuple(range(count)), frozenset(), [])
    if len(solutions) != 1:
        raise ValueError("pairing is not unique")
    return solutions[0]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _interior_black_cells(mask: np.ndarray, scene: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(mask)
    submask = mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, submask.shape)

    padded = np.pad(submask, 1, constant_values=False)
    ph, pw = map(int, padded.shape)
    seen = np.zeros((ph, pw), dtype=bool)
    queue = deque([(0, 0)])
    seen[0, 0] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < ph and 0 <= nc < pw and not seen[nr, nc] and not bool(padded[nr, nc]):
                seen[nr, nc] = True
                queue.append((nr, nc))

    interior: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if bool(submask[r, c]) or bool(seen[r + 1, c + 1]):
                continue
            rr = r0 + r
            cc = c0 + c
            if int(scene[rr, cc]) == BLACK:
                interior.append((rr, cc))
    return interior


def _solve_scene(scene: np.ndarray, legend_colors: list[int]) -> np.ndarray:
    work = np.asarray(scene, dtype=int).copy()
    colors = _color_order(work)
    for color in colors:
        for cells in _solve_color(work, color):
            for r, c in cells[1:-1]:
                work[r, c] = color

    infos: list[dict[str, object]] = []
    for color in colors:
        mask = work == color
        if not _is_single_cycle(mask):
            raise ValueError("completed shape is not a single cycle")
        interior = _interior_black_cells(mask, work)
        infos.append(
            {
                "color": color,
                "bbox": _bbox(mask),
                "area": len(interior),
                "interior": interior,
            }
        )

    if len(infos) != len(legend_colors):
        raise ValueError("legend size does not match loop count")

    ranked = sorted(infos, key=lambda item: (int(item["area"]), item["bbox"][0], item["bbox"][1]))
    out = work.copy()
    for fill_color, info in zip(legend_colors, ranked, strict=True):
        for r, c in info["interior"]:
            out[r, c] = int(fill_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color path fragments into closed loops, rank loops by enclosed area,
    fill them using the bottom legend from left to right, and remove the legend rows.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("expected a non-empty 2D grid")

    legend_top, legend_colors = _find_legend(g)
    scene = g[:legend_top].copy()
    return _solve_scene(scene, legend_colors)


def _draw_rect_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _rect_interior_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _oriented_sides(
    bbox: tuple[int, int, int, int],
    clockwise: bool,
) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    if clockwise:
        return [
            [(r0, c) for c in range(c0 + 1, c1)],
            [(r, c1) for r in range(r0 + 1, r1)],
            [(r1, c) for c in range(c1 - 1, c0, -1)],
            [(r, c0) for r in range(r1 - 1, r0, -1)],
        ]
    return [
        [(r0, c) for c in range(c1 - 1, c0, -1)],
        [(r, c0) for r in range(r0 + 1, r1)],
        [(r1, c) for c in range(c0 + 1, c1)],
        [(r, c1) for r in range(r1 - 1, r0, -1)],
    ]


def _anchored_side_segments(
    side_cells: list[tuple[int, int]],
    split: bool,
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    length = len(side_cells)
    if split:
        options: list[tuple[int, int]] = []
        for left_len in range(2, min(4, length - 3) + 1):
            for right_len in range(2, min(4, length - left_len - 1) + 1):
                if left_len + right_len + 1 <= length:
                    options.append((left_len, right_len))
        if options:
            left_len, right_len = rng.choice(options)
            return [side_cells[:left_len], side_cells[length - right_len :]]

    seg_len = rng.randint(2, min(4, length))
    return [side_cells[:seg_len]]


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_boxes(
    scene_h: int,
    scene_w: int,
    sizes: list[tuple[int, int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int, int, int]] | None:
    placed: list[tuple[int, int, int, int, int]] = []
    for outer_h, outer_w, area in sorted(sizes, key=lambda item: item[2], reverse=True):
        min_r = 1
        max_r = scene_h - outer_h - 1
        min_c = 1
        max_c = scene_w - outer_w - 1
        if max_r < min_r or max_c < min_c:
            return None

        ok = False
        for _ in range(300):
            r0 = rng.randint(min_r, max_r)
            c0 = rng.randint(min_c, max_c)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_too_close(bbox, other[:4], BOX_PAD) for other in placed):
                continue
            placed.append((bbox[0], bbox[1], bbox[2], bbox[3], area))
            ok = True
            break
        if not ok:
            return None
    return placed


def _sample_sizes(scene_h: int, scene_w: int, loop_count: int, rng: random.Random) -> list[tuple[int, int, int]] | None:
    max_h = min(MAX_OUTER_H, scene_h - 2)
    max_w = min(MAX_OUTER_W, scene_w - 2)
    if max_h < MIN_OUTER or max_w < MIN_OUTER:
        return None

    sizes: list[tuple[int, int, int]] = []
    used_areas: set[int] = set()
    for _ in range(loop_count):
        found = False
        for _ in range(200):
            outer_h = rng.randint(MIN_OUTER, max_h)
            outer_w = rng.randint(MIN_OUTER, max_w)
            area = (outer_h - 2) * (outer_w - 2)
            if area in used_areas:
                continue
            sizes.append((outer_h, outer_w, area))
            used_areas.add(area)
            found = True
            break
        if not found:
            return None
    return sizes


def _render_loop_input(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    clockwise: bool,
    split_side: int | None,
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    segments: list[list[tuple[int, int]]] = []
    for side_index, side_cells in enumerate(_oriented_sides(bbox, clockwise)):
        side_split = split_side == side_index
        segments.extend(_anchored_side_segments(side_cells, side_split, rng))
    for cells in segments:
        for r, c in cells:
            grid[r, c] = color
    return segments


def _make_expected_output(
    scene_shape: tuple[int, int],
    loop_specs: list[dict[str, object]],
) -> np.ndarray:
    out = np.zeros(scene_shape, dtype=int)
    for spec in loop_specs:
        _draw_rect_border(out, spec["bbox"], int(spec["boundary_color"]))
    for spec in sorted(loop_specs, key=lambda item: int(item["area"])):
        for r, c in _rect_interior_cells(spec["bbox"]):
            out[r, c] = int(spec["fill_color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    loop_count = rng.choices([2, 3, 4], weights=[3, 3, 1])[0]
    scene_h = rng.randint(max(MIN_SCENE_H, 12 + 2 * loop_count), MAX_SCENE_H)
    scene_w = rng.randint(max(MIN_SCENE_W, 14 + 3 * loop_count), MAX_SCENE_W)

    sizes = _sample_sizes(scene_h, scene_w, loop_count, rng)
    if sizes is None:
        return None
    placed = _place_boxes(scene_h, scene_w, sizes, rng)
    if placed is None:
        return None

    colors = list(ARC_COLORS)
    rng.shuffle(colors)
    boundary_colors = colors[:loop_count]
    fill_colors = colors[loop_count : 2 * loop_count]

    scene = np.zeros((scene_h, scene_w), dtype=int)
    loop_specs: list[dict[str, object]] = []
    for idx, (r0, c0, r1, c1, area) in enumerate(placed):
        clockwise = bool(rng.randrange(2))
        split_side = rng.randrange(4) if rng.random() < SPLIT_PROB else None
        bbox = (r0, c0, r1, c1)
        _render_loop_input(scene, bbox, boundary_colors[idx], clockwise, split_side, rng)
        loop_specs.append(
            {
                "bbox": bbox,
                "boundary_color": boundary_colors[idx],
                "area": area,
            }
        )

    ranked_specs = sorted(loop_specs, key=lambda item: int(item["area"]))
    for fill_color, spec in zip(fill_colors, ranked_specs, strict=True):
        spec["fill_color"] = fill_color

    total_h = scene_h + LEGEND_ROWS
    full = np.zeros((total_h, scene_w), dtype=int)
    full[:scene_h] = scene

    legend_width = 3 * loop_count - 1
    legend_start = rng.randint(0, scene_w - legend_width)
    for idx, spec in enumerate(ranked_specs):
        col = legend_start + 3 * idx
        full[scene_h : scene_h + 2, col : col + 2] = int(spec["fill_color"])

    expected = _make_expected_output((scene_h, scene_w), loop_specs)
    try:
        actual = main(full)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return full


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
