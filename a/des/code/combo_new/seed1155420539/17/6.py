from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 400
PATH_RETRIES = 300
PLACE_RETRIES = 300
PADDING = 1


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []

    for r, c in np.argwhere(grid == int(color)):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[Coord] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(sorted(comp))
    return comps


def _path_degrees(cells: set[Coord]) -> dict[Coord, int]:
    return {
        cell: sum((nr, nc) in cells for nr, nc in _neighbors4(cell[0], cell[1]))
        for cell in cells
    }


def _path_endpoints(cells: set[Coord]) -> list[Coord]:
    degrees = _path_degrees(cells)
    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "each segment must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "segments may not branch"
    return endpoints


def _connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _normalize_coords(coords: list[Coord]) -> tuple[list[Coord], Coord]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return [(r - min_r, c - min_c) for r, c in coords], (min_r, min_c)


def _shift_coords(coords: list[Coord], dr: int, dc: int) -> list[Coord]:
    return [(r + dr, c + dc) for r, c in coords]


def _extract_segments(grid: np.ndarray) -> tuple[int, dict[int, dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) not in (bg, GREY))
    assert colors, "expected at least one colored segment"

    segments: dict[int, dict[str, object]] = {}
    all_path_cells: set[Coord] = set()

    for color in colors:
        comps = _components_of_color(g, color)
        path_comps = [comp for comp in comps if len(comp) >= 2]
        assert len(path_comps) == 1, "each color must have one non-singleton path component"
        assert all(len(comp) == 1 for comp in comps if comp not in path_comps), "extra same-color cells must be singleton hints"

        path = path_comps[0]
        path_set = set(path)
        endpoints = _path_endpoints(path_set)
        all_path_cells.update(path_set)
        segments[color] = {
            "color": color,
            "cells": path,
            "cell_set": path_set,
            "endpoints": endpoints,
        }

    grey_hint_count = 0
    for color, segment in segments.items():
        path_set = segment["cell_set"]
        assert isinstance(path_set, set)
        endpoint_infos = []
        for endpoint in segment["endpoints"]:
            r, c = endpoint
            hints = []
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in path_set or int(g[nr, nc]) == bg:
                    continue
                hints.append(
                    {
                        "endpoint": endpoint,
                        "hint_pos": (nr, nc),
                        "hint_color": int(g[nr, nc]),
                        "dir": (dr, dc),
                    }
                )
            assert len(hints) == 1, "each endpoint must have exactly one adjacent hint pixel"
            if int(hints[0]["hint_color"]) == GREY:
                grey_hint_count += 1
            endpoint_infos.append(hints[0])
        segment["endpoint_infos"] = endpoint_infos

    assert grey_hint_count == 2, "the chain must have exactly two grey terminal hints"
    return bg, segments


def _assemble_segments(segments: dict[int, dict[str, object]]) -> dict[Coord, int]:
    endpoint_map: dict[tuple[int, Coord], dict[str, object]] = {}
    grey_endpoints: list[tuple[int, Coord]] = []

    for color, segment in segments.items():
        infos = segment["endpoint_infos"]
        assert isinstance(infos, list)
        for info in infos:
            endpoint = info["endpoint"]
            key = (color, endpoint)
            endpoint_map[key] = info
            if int(info["hint_color"]) == GREY:
                grey_endpoints.append(key)

    partners: dict[tuple[int, Coord], tuple[int, Coord]] = {}
    for key, info in endpoint_map.items():
        hint_color = int(info["hint_color"])
        if hint_color == GREY:
            continue
        assert hint_color in segments, "non-grey hint colors must name another segment"
        candidates = [
            other_key
            for other_key, other_info in endpoint_map.items()
            if other_key[0] == hint_color and int(other_info["hint_color"]) == key[0]
        ]
        assert len(candidates) == 1, "each non-grey hint must have one reciprocal partner"
        partner_key = candidates[0]
        partner_info = endpoint_map[partner_key]
        dr0, dc0 = info["dir"]
        dr1, dc1 = partner_info["dir"]
        assert dr0 + dr1 == 0 and dc0 + dc1 == 0, "paired endpoint hints must face each other"
        partners[key] = partner_key

    start_color = grey_endpoints[0][0]
    positions: dict[int, Coord] = {start_color: (0, 0)}
    q = deque([start_color])

    while q:
        color = q.popleft()
        base_r, base_c = positions[color]
        infos = segments[color]["endpoint_infos"]
        assert isinstance(infos, list)
        for info in infos:
            hint_color = int(info["hint_color"])
            if hint_color == GREY:
                continue
            partner_key = partners[(color, info["endpoint"])]
            partner_color, partner_endpoint = partner_key
            target_r = base_r + info["endpoint"][0] + info["dir"][0] - partner_endpoint[0]
            target_c = base_c + info["endpoint"][1] + info["dir"][1] - partner_endpoint[1]
            target = (target_r, target_c)
            if partner_color in positions:
                assert positions[partner_color] == target, "segment placement must be consistent"
            else:
                positions[partner_color] = target
                q.append(partner_color)

    assert len(positions) == len(segments), "all segments must belong to one connected chain"

    painted: dict[Coord, int] = {}
    for color, segment in segments.items():
        dr, dc = positions[color]
        cells = segment["cells"]
        assert isinstance(cells, list)
        for r, c in cells:
            pos = (r + dr, c + dc)
            assert pos not in painted, "translated segments may not overlap"
            painted[pos] = color

    cell_set = set(painted)
    assert _connected(cell_set), "assembled path must be connected"
    degrees = _path_degrees(cell_set)
    assert list(degrees.values()).count(1) == 2, "assembled path must have exactly two endpoints"
    assert all(degree in (1, 2) for degree in degrees.values()), "assembled path may not branch"
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble color-coded path fragments into one continuous path by following endpoint hint pixels.

    Input structure:
    - The background is black.
    - Several differently colored one-pixel-wide 4-connected path segments are scattered on the grid.
    - Each segment has exactly two endpoints, and each endpoint has one adjacent singleton hint pixel.
    - A non-grey hint pixel uses another segment's color to identify the segment connected at that endpoint, while grey hints mark the two global chain ends.

    Rule:
    - For every colored segment, separate its true path component from same-colored singleton hint pixels elsewhere.
    - Read each endpoint's adjacent hint pixel to recover both the partner segment color and the docking side of the connection.
    - Translate whole segments without rotating them so that reciprocally referenced endpoints meet on the hinted adjacent cells, producing one open 4-connected chain.
    - Remove all hint pixels and return the translated colored path cropped to its minimal bounding box.

    Key insight:
    - The singleton pixels next to endpoints are not decoration: each one is a pointer to the next segment's identity and also fixes where that endpoint must dock in the reconstructed chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, segments = _extract_segments(g)
    assert bg == BLACK, "this task uses a black background"
    painted = _assemble_segments(segments)

    rows = [r for r, _ in painted]
    cols = [c for _, c in painted]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BLACK, dtype=int)
    for (r, c), color in painted.items():
        out[r - min_r, c - min_c] = int(color)
    return out


def _random_snake(height: int, width: int, length: int) -> list[Coord] | None:
    for _ in range(PATH_RETRIES):
        start = (random.randint(0, height - 1), random.randint(0, width - 1))
        path = [start]
        used = {start}

        while len(path) < length:
            r, c = path[-1]
            options = []
            dirs = DIRS4[:]
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr = r + dr
                nc = c + dc
                nxt = (nr, nc)
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if nxt in used:
                    continue
                if any(nb in used and nb != path[-1] for nb in _neighbors4(nr, nc)):
                    continue
                options.append(nxt)
            if not options:
                break
            path.append(random.choice(options))
            used.add(path[-1])

        if len(path) == length:
            return path
    return None


def _segment_lengths(total: int, parts: int) -> list[int]:
    lengths = [2] * parts
    slack = total - 2 * parts
    for _ in range(slack):
        lengths[random.randrange(parts)] += 1
    random.shuffle(lengths)
    return lengths


def _free_hint_dir(path_set: set[Coord], endpoint: Coord) -> Coord:
    options = [d for d in DIRS4 if (endpoint[0] + d[0], endpoint[1] + d[1]) not in path_set]
    assert options, "each chain endpoint must have at least one free hint direction"
    return random.choice(options)


def _build_chunks(path: list[Coord], colors: list[int], lengths: list[int]) -> list[dict[str, object]]:
    path_set = set(path)
    chunks: list[list[Coord]] = []
    start = 0
    for length in lengths:
        chunks.append(path[start : start + length])
        start += length
    assert start == len(path)

    objects: list[dict[str, object]] = []
    for i, chunk in enumerate(chunks):
        first = chunk[0]
        last = chunk[-1]

        if i == 0:
            first_dir = _free_hint_dir(path_set, first)
            first_hint = GREY
        else:
            prev = chunks[i - 1][-1]
            first_dir = (prev[0] - first[0], prev[1] - first[1])
            first_hint = colors[i - 1]

        if i == len(chunks) - 1:
            last_dir = _free_hint_dir(path_set, last)
            last_hint = GREY
        else:
            nxt = chunks[i + 1][0]
            last_dir = (nxt[0] - last[0], nxt[1] - last[1])
            last_hint = colors[i + 1]

        hints = [
            {"at": (first[0] + first_dir[0], first[1] + first_dir[1]), "color": first_hint},
            {"at": (last[0] + last_dir[0], last[1] + last_dir[1]), "color": last_hint},
        ]

        all_cells = chunk + [hint["at"] for hint in hints]
        normalized_all, offset = _normalize_coords(all_cells)
        off_r, off_c = offset
        path_local = [(r - off_r, c - off_c) for r, c in chunk]
        hints_local = [
            {
                "at": (hint["at"][0] - off_r, hint["at"][1] - off_c),
                "color": int(hint["color"]),
            }
            for hint in hints
        ]
        all_local = normalized_all
        height = max(r for r, _ in all_local) + 1
        width = max(c for _, c in all_local) + 1

        objects.append(
            {
                "color": int(colors[i]),
                "path": path_local,
                "hints": hints_local,
                "all": all_local,
                "height": height,
                "width": width,
            }
        )
    return objects


def _render_output(objects: list[dict[str, object]], path: list[Coord], lengths: list[int]) -> np.ndarray:
    colors_by_cell: dict[Coord, int] = {}
    start = 0
    for obj, length in zip(objects, lengths, strict=False):
        color = int(obj["color"])
        for cell in path[start : start + length]:
            colors_by_cell[cell] = color
        start += length

    rows = [r for r, _ in colors_by_cell]
    cols = [c for _, c in colors_by_cell]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BLACK, dtype=int)
    for (r, c), color in colors_by_cell.items():
        out[r - min_r, c - min_c] = int(color)
    return out


def _can_place(occupied: np.ndarray, coords: list[Coord], r0: int, c0: int) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = r0 + r
        cc = c0 + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r1 = max(0, rr - PADDING)
        r2 = min(h, rr + PADDING + 1)
        c1 = max(0, cc - PADDING)
        c2 = min(w, cc + PADDING + 1)
        if bool(occupied[r1:r2, c1:c2].any()):
            return False
    return True


def _paint_object(grid: np.ndarray, occupied: np.ndarray, obj: dict[str, object], r0: int, c0: int) -> None:
    color = int(obj["color"])
    for r, c in obj["path"]:
        grid[r0 + r, c0 + c] = color
        occupied[r0 + r, c0 + c] = True
    for hint in obj["hints"]:
        hr, hc = hint["at"]
        grid[r0 + hr, c0 + hc] = int(hint["color"])
        occupied[r0 + hr, c0 + hc] = True


def _scatter_objects(objects: list[dict[str, object]]) -> np.ndarray | None:
    max_h = max(int(obj["height"]) for obj in objects)
    max_w = max(int(obj["width"]) for obj in objects)
    min_side = max(18, max_h + 2, max_w + 2)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)
    grid = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = sorted(
        range(len(objects)),
        key=lambda idx: (len(objects[idx]["all"]), objects[idx]["height"] * objects[idx]["width"]),
        reverse=True,
    )

    for idx in order:
        obj = objects[idx]
        obj_h = int(obj["height"])
        obj_w = int(obj["width"])
        positions = [(r, c) for r in range(height - obj_h + 1) for c in range(width - obj_w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions[:PLACE_RETRIES]:
            if _can_place(occupied, obj["all"], r0, c0):
                _paint_object(grid, occupied, obj, r0, c0)
                placed = True
                break
        if not placed:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    total_length = random.randint(10, 22)
    max_parts = min(5, total_length // 2)
    part_count = random.randint(2, max_parts)
    colors = random.sample(PALETTE, part_count)

    min_side = max(5, int(np.ceil(np.sqrt(total_length * 2.2))))
    max_side = min(14, max(min_side + 2, 12))
    out_h = random.randint(min_side, max_side)
    out_w = random.randint(min_side, max_side)
    if total_length > (out_h * out_w + 1) // 2:
        return None

    path = _random_snake(out_h, out_w, total_length)
    if path is None:
        return None

    lengths = _segment_lengths(total_length, part_count)
    objects = _build_chunks(path, colors, lengths)
    expected = _render_output(objects, path, lengths)
    if expected.ndim != 2 or max(expected.shape) > 30:
        return None

    grid = _scatter_objects(objects)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-relay puzzle input")
