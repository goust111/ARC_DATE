from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 1
MARKER_COLORS = (2, 3, 4, 5, 6)

MIN_ROOM_SIZE = 5
MAX_ROOM_SIZE = 8
ROOM_COUNT_CHOICES = (2, 3)
MAX_GRID_SIDE = 30
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _majority_marker_color(markers: list[tuple[int, int, int]]) -> int:
    counts = Counter(color for _r, _c, color in markers)
    best = max(counts.values())
    winners = {color for color, count in counts.items() if count == best}
    for _r, _c, color in sorted(markers):
        if color in winners:
            return int(color)
    raise AssertionError("expected at least one marker color winner")


def _flood_black_regions(grid: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    region_ids = np.full((h, w), -1, dtype=int)
    regions: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != BG or int(region_ids[r, c]) != -1:
                continue

            rid = len(regions)
            queue = deque([(r, c)])
            region_ids[r, c] = rid
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, g.shape):
                    if int(g[nr, nc]) != BG or int(region_ids[nr, nc]) != -1:
                        continue
                    region_ids[nr, nc] = rid
                    queue.append((nr, nc))

            regions.append(coords)

    return region_ids, regions


def _analyze_grid(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert int(g.min()) >= 0 and int(g.max()) <= 9, "ARC colors must be in [0, 9]"

    region_ids, regions = _flood_black_regions(g)
    markers_by_region: list[list[tuple[int, int, int]]] = [[] for _ in regions]

    for r, c in np.argwhere((g != BG) & (g != WALL)):
        rr = int(r)
        cc = int(c)
        color = int(g[rr, cc])
        neighbors = {
            int(region_ids[nr, nc])
            for nr, nc in _neighbors4(rr, cc, g.shape)
            if int(g[nr, nc]) == BG
        }
        assert len(neighbors) == 1, "each marker should border exactly one black region"
        rid = neighbors.pop()
        assert rid >= 0
        markers_by_region[rid].append((rr, cc, color))

    infos: list[dict[str, object]] = []
    for rid, coords in enumerate(regions):
        top, left, bottom, right = _bbox(coords)
        height = bottom - top + 1
        width = right - left + 1
        if width > height:
            kind = "wide"
        elif height > width:
            kind = "tall"
        else:
            kind = "square"
        infos.append(
            {
                "region_id": rid,
                "cells": coords,
                "bbox": (top, left, bottom, right),
                "height": height,
                "width": width,
                "kind": kind,
                "markers": sorted(markers_by_region[rid]),
            }
        )
    return infos


def _solve_from_infos(grid: np.ndarray, infos: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    for info in infos:
        markers = list(info["markers"])
        if not markers:
            continue

        top, left, bottom, right = info["bbox"]
        kind = str(info["kind"])

        if kind == "wide":
            for r, c, color in markers:
                tr = int(top + bottom - r)
                assert 0 <= tr < out.shape[0] and int(out[tr, c]) != WALL
                out[tr, c] = int(color)
        elif kind == "tall":
            for r, c, color in markers:
                tc = int(left + right - c)
                assert 0 <= tc < out.shape[1] and int(out[r, tc]) != WALL
                out[r, tc] = int(color)
        else:
            fill_color = _majority_marker_color(markers)
            for r, c in info["cells"]:
                out[r, c] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each wall-bounded region according to its bbox aspect ratio, either mirroring its markers or filling it with the majority marker color.

    Input structure:
    - Black cells are open space and blue cells are one-cell-thick walls.
    - The blue walls split the canvas into multiple 4-connected black regions.
    - Some regions contain 2-5 isolated marker pixels in colors 2-6, while other regions may contain no markers.

    Rule:
    - Flood-fill only the black cells to identify regions, then assign each marker to the unique black region it borders.
    - Compute each region's black-cell bbox. Wide regions mirror every marker across the bbox's horizontal midline; tall regions mirror every marker across the bbox's vertical midline.
    - Square regions recolor all black cells in that region to the majority marker color, breaking color-count ties by the earliest marker in row-major scan order. Walls and original markers stay unchanged.

    Key insight:
    The same marker pixels trigger different operations solely because the surrounding region's geometry changes: aspect ratio selects between spatial mirroring and color-based filling.
    """
    g = np.asarray(grid, dtype=int)
    infos = _analyze_grid(g)
    return _solve_from_infos(g, infos)


def _room_kind(room: dict[str, int]) -> str:
    if room["width"] > room["height"]:
        return "wide"
    if room["height"] > room["width"]:
        return "tall"
    return "square"


def _sample_partition() -> tuple[list[int], list[int]]:
    for _ in range(200):
        row_count = random.choice(ROOM_COUNT_CHOICES)
        col_count = random.choice(ROOM_COUNT_CHOICES)
        row_heights = [random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE) for _ in range(row_count)]
        col_widths = [random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE) for _ in range(col_count)]
        if sum(row_heights) + row_count + 1 > MAX_GRID_SIDE:
            continue
        if sum(col_widths) + col_count + 1 > MAX_GRID_SIDE:
            continue
        if not any(h == w for h in row_heights for w in col_widths):
            continue
        if not any(w > h for h in row_heights for w in col_widths):
            continue
        if not any(h > w for h in row_heights for w in col_widths):
            continue
        return row_heights, col_widths
    raise RuntimeError("failed to sample a valid room partition")


def _build_rooms(row_heights: list[int], col_widths: list[int]) -> list[dict[str, int]]:
    rooms: list[dict[str, int]] = []
    row_start = 1
    room_id = 0
    for row_index, height in enumerate(row_heights):
        col_start = 1
        for col_index, width in enumerate(col_widths):
            rooms.append(
                {
                    "id": room_id,
                    "row": row_index,
                    "col": col_index,
                    "top": row_start,
                    "left": col_start,
                    "height": height,
                    "width": width,
                }
            )
            room_id += 1
            col_start += width + 1
        row_start += height + 1
    return rooms


def _render_walls(row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    grid = np.zeros((h, w), dtype=int)

    r = 0
    grid[r, :] = WALL
    for height in row_heights:
        r += height + 1
        grid[r, :] = WALL
    assert r == h - 1

    c = 0
    grid[:, c] = WALL
    for width in col_widths:
        c += width + 1
        grid[:, c] = WALL
    assert c == w - 1

    return grid


def _chebyshev_far_enough(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen)


def _max_isolated_cells(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    for cell in shuffled:
        if _chebyshev_far_enough(cell, chosen):
            chosen.append(cell)
    return chosen


def _sample_marker_colors(k: int, min_distinct: int, max_distinct: int) -> list[int]:
    high = min(max_distinct, k, len(MARKER_COLORS))
    low = min(min_distinct, high)
    distinct = random.randint(low, high)
    palette = random.sample(list(MARKER_COLORS), distinct)
    colors = palette[:]
    while len(colors) < k:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _mirror_source_pairs(room: dict[str, int], *, horizontal_axis: bool) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    top = room["top"]
    left = room["left"]
    bottom = top + room["height"] - 1
    right = left + room["width"] - 1
    pairs = []

    if horizontal_axis:
        for r in range(top + 1, bottom):
            tr = top + bottom - r
            if tr <= r:
                continue
            for c in range(left + 1, right):
                pairs.append(((r, c), (tr, c)))
    else:
        for c in range(left + 1, right):
            tc = left + right - c
            if tc <= c:
                continue
            for r in range(top + 1, bottom):
                pairs.append(((r, c), (r, tc)))
    return pairs


def _pick_mirror_markers(room: dict[str, int], kind: str) -> list[tuple[int, int, int]] | None:
    pairs = _mirror_source_pairs(room, horizontal_axis=(kind == "wide"))
    random.shuffle(pairs)
    selected: list[tuple[tuple[int, int], tuple[int, int]]] = []
    sources: list[tuple[int, int]] = []
    targets: list[tuple[int, int]] = []

    for source, target in pairs:
        if not _chebyshev_far_enough(source, sources):
            continue
        if not _chebyshev_far_enough(target, targets):
            continue
        selected.append((source, target))
        sources.append(source)
        targets.append(target)

    if len(selected) < 2:
        return None

    count = random.randint(2, min(5, len(selected)))
    chosen = random.sample(selected, count)
    chosen.sort(key=lambda pair: pair[0])
    min_distinct = 2 if count >= 3 and random.random() < 0.75 else 1
    colors = _sample_marker_colors(count, min_distinct=min_distinct, max_distinct=3)
    return [(r, c, color) for ((r, c), _target), color in zip(chosen, colors)]


def _assign_tie_colors(cells: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    k = len(cells)
    if k == 2:
        palette = random.sample(list(MARKER_COLORS), 2)
        colors = [palette[0], palette[1]]
    elif k == 3:
        palette = random.sample(list(MARKER_COLORS), 3)
        colors = [palette[0], palette[1], palette[2]]
    elif k == 4:
        palette = random.sample(list(MARKER_COLORS), 2)
        tail = [palette[0], palette[1]]
        random.shuffle(tail)
        colors = [palette[0], palette[1], *tail]
    elif k == 5:
        palette = random.sample(list(MARKER_COLORS), 3)
        tail = [palette[0], palette[1], palette[2]]
        random.shuffle(tail)
        colors = [palette[0], palette[1], *tail]
    else:
        raise AssertionError("tie cases should use 2-5 markers")
    return [(r, c, color) for (r, c), color in zip(cells, colors)]


def _assign_majority_colors(cells: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    k = len(cells)
    if k == 3:
        pattern_choices = [(2, 1)]
    elif k == 4:
        pattern_choices = [(3, 1), (2, 1, 1)]
    elif k == 5:
        pattern_choices = [(4, 1), (3, 2), (3, 1, 1)]
    else:
        raise AssertionError("unique-majority cases should use 3-5 markers")

    pattern = random.choice(pattern_choices)
    palette = random.sample(list(MARKER_COLORS), len(pattern))
    colors: list[int] = []
    for color, count in zip(palette, pattern):
        colors.extend([color] * count)
    random.shuffle(colors)
    return [(r, c, color) for (r, c), color in zip(cells, colors)]


def _pick_square_markers(room: dict[str, int]) -> list[tuple[int, int, int]] | None:
    top = room["top"]
    left = room["left"]
    bottom = top + room["height"] - 1
    right = left + room["width"] - 1
    candidates = [(r, c) for r in range(top + 1, bottom) for c in range(left + 1, right)]
    max_cells = _max_isolated_cells(candidates)
    if len(max_cells) < 2:
        return None

    make_tie = random.random() < 0.5
    min_count = 2 if make_tie else 3
    if len(max_cells) < min_count:
        make_tie = True
        min_count = 2
    count = random.randint(min_count, min(5, len(max_cells)))
    cells = sorted(random.sample(max_cells, count))

    if make_tie:
        return _assign_tie_colors(cells)
    return _assign_majority_colors(cells)


def _room_markers(room: dict[str, int], kind: str) -> list[tuple[int, int, int]] | None:
    if kind == "wide":
        return _pick_mirror_markers(room, "wide")
    if kind == "tall":
        return _pick_mirror_markers(room, "tall")
    return _pick_square_markers(room)


def _render_blueprint(
    row_heights: list[int],
    col_widths: list[int],
    rooms: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = _render_walls(row_heights, col_widths)
    for room in rooms:
        for r, c, color in room["markers"]:
            inp[r, c] = int(color)

    out = inp.copy()
    for room in rooms:
        markers = list(room["markers"])
        if not markers:
            continue

        top = int(room["top"])
        left = int(room["left"])
        bottom = top + int(room["height"]) - 1
        right = left + int(room["width"]) - 1
        kind = str(room["kind"])

        if kind == "wide":
            for r, c, color in markers:
                out[top + bottom - r, c] = int(color)
        elif kind == "tall":
            for r, c, color in markers:
                out[r, left + right - c] = int(color)
        else:
            fill_color = _majority_marker_color(sorted(markers))
            for r in range(top, bottom + 1):
                for c in range(left, right + 1):
                    if int(inp[r, c]) == BG:
                        out[r, c] = fill_color

    return inp, out


def _validate_generated_input(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    infos = _analyze_grid(g)

    assert g.ndim == 2 and max(g.shape) <= MAX_GRID_SIDE
    assert set(int(v) for v in np.unique(g)).issubset({BG, WALL, *MARKER_COLORS})
    assert any(not info["markers"] for info in infos), "need at least one unmarked region"
    assert any(info["kind"] == "wide" and info["markers"] for info in infos)
    assert any(info["kind"] == "tall" and info["markers"] for info in infos)
    assert any(info["kind"] == "square" and info["markers"] for info in infos)

    for info in infos:
        markers = list(info["markers"])
        if not markers:
            continue

        assert 2 <= len(markers) <= 5
        positions = [(r, c) for r, c, _color in markers]
        for r, c, color in markers:
            assert color in MARKER_COLORS
            for nr, nc in _neighbors4(r, c, g.shape):
                assert int(g[nr, nc]) != WALL, "markers must not touch walls in 4-neighborhood"
        for i, cell_a in enumerate(positions):
            for cell_b in positions[i + 1 :]:
                assert max(abs(cell_a[0] - cell_b[0]), abs(cell_a[1] - cell_b[1])) > 1

        top, left, bottom, right = info["bbox"]
        if info["kind"] == "wide":
            targets = set()
            occupied = {(r, c) for r, c, _color in markers}
            for r, c, _color in markers:
                tr = top + bottom - r
                assert tr != r and int(g[tr, c]) == BG
                assert (tr, c) not in occupied
                targets.add((tr, c))
            assert len(targets) == len(markers)
        elif info["kind"] == "tall":
            targets = set()
            occupied = {(r, c) for r, c, _color in markers}
            for r, c, _color in markers:
                tc = left + right - c
                assert tc != c and int(g[r, tc]) == BG
                assert (r, tc) not in occupied
                targets.add((r, tc))
            assert len(targets) == len(markers)

    return infos


def _validate_semantics(grid: np.ndarray, expected: np.ndarray) -> None:
    parsed_expected = _solve_from_infos(grid, _analyze_grid(grid))
    assert np.array_equal(parsed_expected, expected), "semantic checker disagrees with blueprint output"


def _generate_candidate() -> np.ndarray:
    row_heights, col_widths = _sample_partition()
    rooms = _build_rooms(row_heights, col_widths)

    wide_ids = [room["id"] for room in rooms if _room_kind(room) == "wide"]
    tall_ids = [room["id"] for room in rooms if _room_kind(room) == "tall"]
    square_ids = [room["id"] for room in rooms if _room_kind(room) == "square"]

    chosen_ids = {
        "wide": random.choice(wide_ids),
        "tall": random.choice(tall_ids),
        "square": random.choice(square_ids),
    }

    marked_ids = set(chosen_ids.values())
    spare_ids = [room["id"] for room in rooms if room["id"] not in marked_ids]
    max_extra = max(0, len(spare_ids) - 1)
    extra_count = random.randint(0, min(1, max_extra)) if max_extra else 0
    for extra_id in random.sample(spare_ids, extra_count):
        marked_ids.add(extra_id)

    blueprint_rooms: list[dict[str, object]] = []
    for room in rooms:
        room_copy: dict[str, object] = dict(room)
        kind = _room_kind(room)
        room_copy["kind"] = kind
        room_copy["markers"] = []
        if room["id"] in marked_ids:
            markers = _room_markers(room, kind)
            if markers is None:
                raise AssertionError("failed to place markers in selected room")
            room_copy["markers"] = markers
        blueprint_rooms.append(room_copy)

    inp, expected = _render_blueprint(row_heights, col_widths, blueprint_rooms)
    infos = _validate_generated_input(inp)
    _validate_semantics(inp, expected)
    assert np.array_equal(main(inp), expected)

    assert any(info["kind"] == "wide" and info["markers"] for info in infos)
    assert any(info["kind"] == "tall" and info["markers"] for info in infos)
    assert any(info["kind"] == "square" and info["markers"] for info in infos)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _generate_candidate()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
