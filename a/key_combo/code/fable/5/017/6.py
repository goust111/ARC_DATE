from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


Box = tuple[int, int, int, int]
Coord = tuple[int, int]


def _background(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    maximum = int(counts.max())
    winners = values[counts == maximum]
    assert len(winners) == 1, "the background must be uniquely dominant"
    return int(winners[0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (row, col) coordinate arrays."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for seed_r, seed_c in np.argwhere(mask):
        sr, sc = int(seed_r), int(seed_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[Coord] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _solid_rectangle(coords: np.ndarray) -> Box:
    assert len(coords), "empty foreground component"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = (int(r0), int(c0), int(r1), int(c1))
    area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    assert len(coords) == area, "every foreground object must be a solid rectangle"
    return box


def _open_diagonal(start: Coord, end: Coord) -> list[Coord]:
    dr_total = int(end[0] - start[0])
    dc_total = int(end[1] - start[1])
    assert dr_total > 1 and dr_total == abs(dc_total)
    dc = 1 if dc_total > 0 else -1
    return [(start[0] + k, start[1] + dc * k) for k in range(1, dr_total)]


def _facing_geometry(a: Box, b: Box) -> tuple[Coord, Coord] | None:
    """Facing corners for vertically and horizontally separated rectangles."""
    if b[2] < a[0]:
        a, b = b, a
    if not a[2] < b[0]:
        return None

    if a[3] < b[1]:
        return (a[2], a[3]), (b[0], b[1])
    if b[3] < a[1]:
        return (a[2], a[1]), (b[0], b[3])
    return None


def _strict_connector(a: Box, b: Box) -> tuple[Coord, Coord, list[Coord]] | None:
    facing = _facing_geometry(a, b)
    if facing is None:
        return None
    start, end = facing
    row_delta = end[0] - start[0]
    col_delta = end[1] - start[1]
    if row_delta <= 1 or row_delta != abs(col_delta):
        return None
    return start, end, _open_diagonal(start, end)


def _near_diagonal_error(a: Box, b: Box) -> int | None:
    facing = _facing_geometry(a, b)
    if facing is None:
        return None
    start, end = facing
    return abs((end[0] - start[0]) - abs(end[1] - start[1]))


def _rectangles_by_color(g: np.ndarray, bg: int) -> dict[int, list[Box]]:
    result: dict[int, list[Box]] = {}
    for value in np.unique(g):
        color = int(value)
        if color == bg:
            continue
        boxes = [_solid_rectangle(component) for component in _components(g == color)]
        result[color] = boxes
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the uniquely diagonal pair in each same-color rectangle triple, with shorter lines winning crossings.

    Input structure:
    - A uniquely dominant background surrounds two or more foreground colors.
    - Each foreground color forms exactly three mutually separated solid rectangles.
    - In each triple exactly one pair has its facing corners on one strict 45-degree diagonal;
      the third rectangle is a non-aligned distractor.

    Rule:
    - For each color, paint the open 45-degree segment between the facing corners of
      its unique strictly aligned pair, leaving all three rectangles unchanged.
    - If different segments request the same background cell, compare their complete
      open-segment lengths and give that cell only to the shorter segment.
    - Crossing segments are guaranteed to have unequal lengths.

    Key insight:
    - First reject the near-diagonal distractor using exact facing-corner geometry;
      then resolve all requested cells together, since crossing priority depends on
      each whole segment's length rather than color or drawing order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"
    bg = _background(g)
    groups = _rectangles_by_color(g, bg)
    assert len(groups) >= 2, "expected at least two rectangle colors"

    requests: dict[Coord, list[tuple[int, int]]] = defaultdict(list)
    for color, boxes in groups.items():
        assert len(boxes) == 3, "each foreground color must form exactly three rectangles"
        candidates: list[tuple[Coord, Coord, list[Coord]]] = []
        for i in range(3):
            for j in range(i + 1, 3):
                connector = _strict_connector(boxes[i], boxes[j])
                if connector is not None:
                    candidates.append(connector)
        assert len(candidates) == 1, "each color must have one unique strictly diagonal pair"
        _start, _end, path = candidates[0]
        length = len(path)
        assert length > 0
        for cell in path:
            assert int(g[cell]) == bg, "connector interiors must cross background only"
            requests[cell].append((length, color))

    out = g.copy()
    for cell, contenders in requests.items():
        if len(contenders) > 1:
            lengths = [length for length, _color in contenders]
            assert len(lengths) == len(set(lengths)), "crossing segments must have unequal lengths"
        _length, winner = min(contenders, key=lambda item: item[0])
        out[cell] = int(winner)
    return out


# ------------------------------ generator ------------------------------

_RECT_SIZES = ((1, 2), (2, 1), (2, 2), (2, 3), (3, 2))
_MAX_ATTEMPTS = 500
_COLOR_COUNT_CYCLE = [2, 3, random.choice((2, 3))]
random.shuffle(_COLOR_COUNT_CYCLE)
_generation_count = 0


def _box_cells(box: Box) -> set[Coord]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _inside(box: Box, shape: tuple[int, int]) -> bool:
    return 0 <= box[0] <= box[2] < shape[0] and 0 <= box[1] <= box[3] < shape[1]


def _separated(a: Box, b: Box) -> bool:
    """True when at least one full background row/column separates two boxes."""
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _boxes_for_line(start: Coord, end: Coord) -> tuple[Box, Box]:
    h1, w1 = random.choice(_RECT_SIZES)
    h2, w2 = random.choice(_RECT_SIZES)
    sr, sc = start
    er, ec = end
    if ec > sc:
        upper = (sr - h1 + 1, sc - w1 + 1, sr, sc)
        lower = (er, ec, er + h2 - 1, ec + w2 - 1)
    else:
        upper = (sr - h1 + 1, sc, sr, sc + w1 - 1)
        lower = (er, ec - w2 + 1, er + h2 - 1, ec)
    return upper, lower


def _make_line(color: int, start: Coord, end: Coord) -> dict:
    boxes = _boxes_for_line(start, end)
    path = _open_diagonal(start, end)
    return {
        "color": int(color),
        "start": start,
        "end": end,
        "boxes": boxes,
        "path": path,
        "length": len(path),
    }


def _all_path_cells(lines: list[dict]) -> set[Coord]:
    return {cell for line in lines for cell in line["path"]}


def _line_layout_valid(lines: list[dict], shape: tuple[int, int]) -> bool:
    boxes = [box for line in lines for box in line["boxes"]]
    if not all(_inside(box, shape) for box in boxes):
        return False
    if any(not _separated(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
        return False
    occupied = set().union(*(_box_cells(box) for box in boxes))
    if any(cell in occupied for line in lines for cell in line["path"]):
        return False
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            shared = set(lines[i]["path"]) & set(lines[j]["path"])
            if len(shared) > 1:
                return False
            if shared and lines[i]["length"] == lines[j]["length"]:
                return False
    return True


def _base_cross(colors: list[int], shape: tuple[int, int]) -> list[dict] | None:
    height, width = shape
    for _ in range(120):
        cr = random.randint(8, height - 9)
        cc = random.randint(8, width - 9)
        slope = random.choice((-1, 1))
        a, b, c, d = (random.randint(3, 6) for _ in range(4))
        start1 = (cr - a, cc - slope * a)
        end1 = (cr + b, cc + slope * b)
        start2 = (cr - c, cc + slope * c)
        end2 = (cr + d, cc - slope * d)
        line1 = _make_line(colors[0], start1, end1)
        line2 = _make_line(colors[1], start2, end2)
        if line1["length"] == line2["length"]:
            continue
        lines = [line1, line2]
        if not _line_layout_valid(lines, shape):
            continue
        if set(line1["path"]) & set(line2["path"]) != {(cr, cc)}:
            continue
        return lines
    return None


def _add_third_line(lines: list[dict], color: int, shape: tuple[int, int]) -> bool:
    height, width = shape
    central_crossings = {
        cell
        for i in range(len(lines))
        for j in range(i + 1, len(lines))
        for cell in set(lines[i]["path"]) & set(lines[j]["path"])
    }
    for _ in range(350):
        delta = random.randint(5, 11)
        if height - delta - 4 < 3:
            continue
        sr = random.randint(3, height - delta - 4)
        slope = random.choice((-1, 1))
        if slope == 1:
            if width - delta - 4 < 3:
                continue
            sc = random.randint(3, width - delta - 4)
        else:
            if width - 4 < delta + 3:
                continue
            sc = random.randint(delta + 3, width - 4)
        candidate = _make_line(color, (sr, sc), (sr + delta, sc + slope * delta))
        if candidate["length"] in {line["length"] for line in lines}:
            # This is stronger than necessary, and makes every possible crossing unambiguous.
            continue
        shared = _all_path_cells(lines) & set(candidate["path"])
        if shared & central_crossings:
            continue
        trial = lines + [candidate]
        if not _line_layout_valid(trial, shape):
            continue
        lines.append(candidate)
        return True
    return False


def _choose_decoy(
    pair: tuple[Box, Box], all_boxes: list[Box], path_cells: set[Coord], shape: tuple[int, int]
) -> Box | None:
    height, width = shape
    candidates: list[Box] = []
    sizes = list(_RECT_SIZES)
    random.shuffle(sizes)
    for h, w in sizes:
        positions = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        random.shuffle(positions)
        for r, c in positions:
            box = (r, c, r + h - 1, c + w - 1)
            if any(not _separated(box, old) for old in all_boxes):
                continue
            if _box_cells(box) & path_cells:
                continue
            if any(_strict_connector(box, true_box) is not None for true_box in pair):
                continue
            errors = [_near_diagonal_error(box, true_box) for true_box in pair]
            if 1 not in errors:
                continue
            candidates.append(box)
            if len(candidates) >= 30:
                return random.choice(candidates)
    return random.choice(candidates) if candidates else None


def _render_expected(inp: np.ndarray, lines: list[dict]) -> np.ndarray:
    """Independent generator oracle using the construction metadata."""
    out = inp.copy()
    requests: dict[Coord, list[dict]] = defaultdict(list)
    for line in lines:
        for cell in line["path"]:
            requests[cell].append(line)
    for cell, contenders in requests.items():
        winner = min(contenders, key=lambda line: line["length"])
        out[cell] = int(winner["color"])
    return out


def _semantic_validation(inp: np.ndarray, expected: np.ndarray, lines: list[dict], decoys: dict[int, Box]) -> bool:
    try:
        bg = _background(inp)
        groups = _rectangles_by_color(inp, bg)
        if set(groups) != {int(line["color"]) for line in lines}:
            return False
        all_boxes = [box for boxes in groups.values() for box in boxes]
        if any(not _separated(all_boxes[i], all_boxes[j]) for i in range(len(all_boxes)) for j in range(i + 1, len(all_boxes))):
            return False

        recovered: dict[int, tuple[list[Coord], int]] = {}
        for color, boxes in groups.items():
            if len(boxes) != 3 or decoys[color] not in boxes:
                return False
            matches = []
            for i in range(3):
                for j in range(i + 1, 3):
                    connector = _strict_connector(boxes[i], boxes[j])
                    if connector is not None:
                        matches.append(connector)
            if len(matches) != 1:
                return False
            path = matches[0][2]
            recovered[color] = (path, len(path))
            true_boxes = [box for box in boxes if box != decoys[color]]
            if len(true_boxes) != 2 or 1 not in [_near_diagonal_error(decoys[color], box) for box in true_boxes]:
                return False
            if any(int(inp[cell]) != bg for cell in path):
                return False

        requests: dict[Coord, list[tuple[int, int]]] = defaultdict(list)
        for color, (path, length) in recovered.items():
            for cell in path:
                requests[cell].append((length, color))
        conflicts = {cell: req for cell, req in requests.items() if len(req) > 1}
        if not conflicts:
            return False
        if any(len({length for length, _color in req}) != len(req) for req in conflicts.values()):
            return False
        if any(not any(len(requests[cell]) == 1 for cell in path) for path, _length in recovered.values()):
            return False

        # Explicitly verify that every longer contender yields at every conflict.
        for cell, req in conflicts.items():
            winner_length, winner_color = min(req)
            if int(expected[cell]) != winner_color:
                return False
            if any(int(expected[cell]) == color for length, color in req if length > winner_length):
                return False

        actual = main(inp)
        if not np.array_equal(actual, expected) or np.array_equal(inp, expected):
            return False
        if not np.array_equal(actual[inp != bg], inp[inp != bg]):
            return False

        # A longer-wins shortcut must give a different result on the guaranteed crossing.
        wrong = inp.copy()
        for cell, req in requests.items():
            wrong[cell] = max(req, key=lambda item: item[0])[1]
        if np.array_equal(wrong, expected):
            return False
        return True
    except (AssertionError, IndexError, ValueError):
        return False


def _gen_attempt(color_count: int | None = None) -> np.ndarray | None:
    height = random.randint(23, 30)
    width = random.randint(23, 30)
    shape = (height, width)
    bg = random.randrange(10)
    if color_count is None:
        color_count = random.choice((2, 3))
    assert color_count in (2, 3)
    colors = random.sample([color for color in range(10) if color != bg], color_count)

    lines = _base_cross(colors, shape)
    if lines is None:
        return None
    if color_count == 3 and not _add_third_line(lines, colors[2], shape):
        return None

    all_boxes = [box for line in lines for box in line["boxes"]]
    path_cells = _all_path_cells(lines)
    decoys: dict[int, Box] = {}
    for line in lines:
        decoy = _choose_decoy(line["boxes"], all_boxes, path_cells, shape)
        if decoy is None:
            return None
        decoys[int(line["color"])] = decoy
        all_boxes.append(decoy)

    inp = np.full(shape, bg, dtype=int)
    for line in lines:
        color = int(line["color"])
        for box in (*line["boxes"], decoys[color]):
            r0, c0, r1, c1 = box
            inp[r0 : r1 + 1, c0 : c1 + 1] = color

    expected = _render_expected(inp, lines)
    if not _semantic_validation(inp, expected, lines, decoys):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _generation_count
    color_count = _COLOR_COUNT_CYCLE[_generation_count % len(_COLOR_COUNT_CYCLE)]
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(color_count=color_count)
        if grid is not None:
            _generation_count += 1
            return grid
    raise RuntimeError("failed to generate a valid diagonal-triple crossing input")
