from __future__ import annotations

import itertools

import numpy as np


BBox = tuple[int, int, int, int]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _rectangle_from_triplet(coords: np.ndarray) -> dict | None:
    if coords.shape[0] != 3:
        return None
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if rows.size != 2 or cols.size != 2:
        return None

    r0, r1 = map(int, rows)
    c0, c1 = map(int, cols)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    present = {(int(r), int(c)) for r, c in coords}
    missing = list(corners - present)
    if len(missing) != 1:
        return None

    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1, "missing": missing[0]}


def _rectangles_overlap(a: dict, b: dict) -> bool:
    return not (
        a["r1"] < b["r0"]
        or b["r1"] < a["r0"]
        or a["c1"] < b["c0"]
        or b["c1"] < a["c0"]
    )


def _infer_scene(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = _background_color(grid)
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    assert len(colors) >= 2, "expected one marker color plus at least one corner color"

    candidates: list[tuple[int, list[dict]]] = []
    for marker in colors:
        rects: list[dict] = []
        ok = True
        for color in colors:
            if color == marker:
                continue
            rect = _rectangle_from_triplet(np.argwhere(grid == color))
            if rect is None:
                ok = False
                break
            mr, mc = rect["missing"]
            if int(grid[mr, mc]) != marker:
                ok = False
                break
            rect["color"] = int(color)
            rects.append(rect)
        if ok and rects:
            candidates.append((int(marker), rects))

    assert len(candidates) == 1, "expected a unique marker color consistent with all corner triplets"
    marker, rects = candidates[0]

    for rect_a, rect_b in itertools.combinations(rects, 2):
        assert not _rectangles_overlap(rect_a, rect_b), "rectangle regions must be disjoint"

    rects.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["color"]))
    return int(bg), int(marker), rects


def _strict_owner(rects: list[dict], pos: tuple[int, int]) -> dict | None:
    r, c = pos
    owner = None
    for rect in rects:
        if rect["r0"] < r < rect["r1"] and rect["c0"] < c < rect["c1"]:
            assert owner is None, "strictly interior marker cannot belong to two rectangles"
            owner = rect
    return owner


def _render_output(
    shape: tuple[int, int], bg: int, marker: int, rects: list[dict], interior_markers: list[tuple[int, int]]
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        color = rect["color"]
        out[r0, c0] = color
        out[r0, c1] = color
        out[r1, c0] = color
        out[r1, c1] = color

    for r, c in interior_markers:
        rect = _strict_owner(rects, (int(r), int(c)))
        assert rect is not None
        out[int(r), rect["c0"] + 1 : rect["c1"]] = marker
        out[rect["r0"] + 1 : rect["r1"], int(c)] = marker
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each disguised fourth rectangle corner, then expand every interior marker into a marker-color cross clipped by that rectangle.

    Input structure:
    - A background grid contains several corner-color groups; each non-marker color appears as exactly three pixels occupying three corners of one axis-aligned rectangle.
    - All groups use different corner colors, and the missing fourth corner of every rectangle is occupied by the same marker color.
    - Extra marker-color pixels may also appear strictly inside a recovered rectangle or completely outside every rectangle.

    Rule:
    - Infer the unique marker color as the one that sits at the missing corner position for every valid three-corner rectangle group.
    - Recolor each such missing-corner marker into that rectangle's corner color, preserving the other three corner pixels.
    - For every remaining marker pixel that lies strictly inside one recovered rectangle, paint its full row segment and full column segment in marker color, clipped to that rectangle's interior; delete marker pixels that are neither recovered corners nor strict interior sources.

    Key insight:
    - Marker pixels must be reclassified by first completing the hidden rectangles: the same visible color plays three roles at once as stolen corner, interior cross source, and outside decoy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, marker, rects = _infer_scene(g)
    out = np.full_like(g, bg)
    missing_lookup = {tuple(rect["missing"]): int(rect["color"]) for rect in rects}

    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        color = rect["color"]
        out[r0, c0] = color
        out[r0, c1] = color
        out[r1, c0] = color
        out[r1, c1] = color

    for r, c in np.argwhere(g == marker):
        pos = (int(r), int(c))
        if pos in missing_lookup:
            continue
        owner = _strict_owner(rects, pos)
        if owner is None:
            continue
        out[pos[0], owner["c0"] + 1 : owner["c1"]] = marker
        out[owner["r0"] + 1 : owner["r1"], pos[1]] = marker

    return out


GRID_SIDE_RANGE = (16, 28)
RECT_COUNT_RANGE = (2, 4)
RECT_SIDE_RANGE = (5, 9)
RECT_GAP = 1
INTERIOR_MARKER_CHOICES = (0, 1, 1, 2, 2, 3)
OUTSIDE_BAIT_RANGE = (1, 5)
MAX_GEN_ATTEMPTS = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _place_rectangles(
    rng: np.random.Generator, height: int, width: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[tuple[int, int, int, int]] = []

    for rect_h, rect_w in sorted(sizes, key=lambda item: item[0] * item[1], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - rect_h):
            for c0 in range(1, width - rect_w):
                r1 = r0 + rect_h - 1
                c1 = c0 + rect_w - 1
                rr0 = max(0, r0 - RECT_GAP)
                cc0 = max(0, c0 - RECT_GAP)
                rr1 = min(height - 1, r1 + RECT_GAP)
                cc1 = min(width - 1, c1 + RECT_GAP)
                if not reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = candidates[_randint(rng, 0, len(candidates) - 1)]
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        reserved[max(0, r0 - RECT_GAP) : min(height, r1 + RECT_GAP + 1), max(0, c0 - RECT_GAP) : min(width, c1 + RECT_GAP + 1)] = True
        placed.append((r0, c0, r1, c1))

    placed.sort()
    return placed


def _choose_missing_corners(rng: np.random.Generator, count: int) -> list[int]:
    if count == 1:
        return [_randint(rng, 0, 3)]

    if rng.random() < 0.65:
        first = _randint(rng, 0, 3)
        second = _randint(rng, 0, 2)
        if second >= first:
            second += 1
        choices = [first, second]
        while len(choices) < count:
            choices.append(_randint(rng, 0, 3))
        rng.shuffle(choices)
        return choices

    return [_randint(rng, 0, 3) for _ in range(count)]


def _interior_cells(rect: dict) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(rect["r0"] + 1, rect["r1"])
        for c in range(rect["c0"] + 1, rect["c1"])
    ]


def _sample_interior_markers(rng: np.random.Generator, rects: list[dict]) -> list[tuple[int, int]]:
    counts = [int(rng.choice(INTERIOR_MARKER_CHOICES)) for _ in rects]
    if sum(counts) == 0:
        counts[_randint(rng, 0, len(counts) - 1)] = _randint(rng, 1, 3)
    if len(rects) >= 2 and rng.random() < 0.4 and max(counts) < 2:
        counts[_randint(rng, 0, len(counts) - 1)] = 2

    markers: list[tuple[int, int]] = []
    for rect, wanted in zip(rects, counts):
        cells = _interior_cells(rect)
        limit = min(len(cells), int(wanted))
        if limit == 0:
            continue
        rng.shuffle(cells)
        markers.extend(cells[:limit])
    return markers


def _sample_outside_baits(
    rng: np.random.Generator, shape: tuple[int, int], rects: list[dict]
) -> list[tuple[int, int]]:
    height, width = shape
    rows_in_play = {r for rect in rects for r in range(rect["r0"], rect["r1"] + 1)}
    cols_in_play = {c for rect in rects for c in range(rect["c0"], rect["c1"] + 1)}
    all_cells: list[tuple[int, int]] = []
    aligned_cells: list[tuple[int, int]] = []

    for r in range(height):
        for c in range(width):
            inside_any = False
            for rect in rects:
                if rect["r0"] <= r <= rect["r1"] and rect["c0"] <= c <= rect["c1"]:
                    inside_any = True
                    break
            if inside_any:
                continue
            pos = (r, c)
            all_cells.append(pos)
            if r in rows_in_play or c in cols_in_play:
                aligned_cells.append(pos)

    rng.shuffle(all_cells)
    rng.shuffle(aligned_cells)
    bait_count = min(len(all_cells), _randint(rng, *OUTSIDE_BAIT_RANGE))
    chosen: list[tuple[int, int]] = []

    while aligned_cells and len(chosen) < min(bait_count, 2):
        pos = aligned_cells.pop()
        if pos not in chosen:
            chosen.append(pos)
    while all_cells and len(chosen) < bait_count:
        pos = all_cells.pop()
        if pos not in chosen:
            chosen.append(pos)
    return chosen


def _render_input(
    shape: tuple[int, int],
    bg: int,
    marker: int,
    rects: list[dict],
    interior_markers: list[tuple[int, int]],
    outside_baits: list[tuple[int, int]],
) -> np.ndarray:
    inp = np.full(shape, bg, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        color = rect["color"]
        for corner in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            inp[corner] = color
        inp[rect["missing"]] = marker
    for rect in rects:
        mr, mc = rect["missing"]
        inp[mr, mc] = marker
    for r, c in interior_markers + outside_baits:
        inp[int(r), int(c)] = marker
    return inp


def _wrong_keep_outside_markers(grid: np.ndarray) -> np.ndarray:
    bg, marker, rects = _infer_scene(grid)
    out = main(grid)
    missing = {tuple(rect["missing"]) for rect in rects}
    for r, c in np.argwhere(grid == marker):
        pos = (int(r), int(c))
        if pos in missing:
            continue
        if _strict_owner(rects, pos) is None:
            out[pos] = marker
    return out


def _wrong_unclipped_crosses(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, marker, rects = _infer_scene(g)
    out = np.full_like(g, bg)
    missing = {tuple(rect["missing"]) for rect in rects}

    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        color = rect["color"]
        out[r0, c0] = color
        out[r0, c1] = color
        out[r1, c0] = color
        out[r1, c1] = color

    for r, c in np.argwhere(g == marker):
        pos = (int(r), int(c))
        if pos in missing:
            continue
        if _strict_owner(rects, pos) is None:
            continue
        out[pos[0], :] = marker
        out[:, pos[1]] = marker
    return out


def _wrong_leave_missing_as_marker(grid: np.ndarray) -> np.ndarray:
    out = main(grid).copy()
    _bg, marker, rects = _infer_scene(grid)
    for rect in rects:
        out[rect["missing"]] = marker
    return out


def _build_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    height = _randint(rng, *GRID_SIDE_RANGE)
    width = _randint(rng, *GRID_SIDE_RANGE)
    rect_count = _randint(rng, *RECT_COUNT_RANGE)
    sizes = [(_randint(rng, *RECT_SIDE_RANGE), _randint(rng, *RECT_SIDE_RANGE)) for _ in range(rect_count)]
    placed = _place_rectangles(rng, height, width, sizes)
    if placed is None:
        return None

    bg = _randint(rng, 0, 9)
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    marker = int(palette[0])
    corner_colors = [int(color) for color in palette[1 : rect_count + 1]]

    missing_ids = _choose_missing_corners(rng, rect_count)
    corner_order = ((0, 0), (0, 1), (1, 0), (1, 1))
    rects: list[dict] = []
    for color, bbox, missing_id in zip(corner_colors, placed, missing_ids):
        r0, c0, r1, c1 = bbox
        dr, dc = corner_order[int(missing_id)]
        rects.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "color": int(color),
                "missing": (int((r0, r1)[dr]), int((c0, c1)[dc])),
            }
        )

    interior_markers = _sample_interior_markers(rng, rects)
    outside_baits = _sample_outside_baits(rng, (height, width), rects)
    if not interior_markers or not outside_baits:
        return None

    inp = _render_input((height, width), bg, marker, rects, interior_markers, outside_baits)
    expected = _render_output((height, width), bg, marker, rects, interior_markers)

    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(expected, _wrong_keep_outside_markers(inp)):
        return None
    if np.array_equal(expected, _wrong_unclipped_crosses(inp)):
        return None
    if np.array_equal(expected, _wrong_leave_missing_as_marker(inp)):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_attempt(rng)
        if built is None:
            continue
        inp, _expected = built
        return inp
    raise RuntimeError("failed to generate a valid missing-corner rectangle sample")
