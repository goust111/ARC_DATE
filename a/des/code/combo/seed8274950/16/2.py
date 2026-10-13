from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Support = tuple[str, int]


_DIRECTION_STEPS: dict[str, Coord] = {
    "h": (0, 1),
    "v": (1, 0),
    "d1": (1, 1),
    "d2": (1, -1),
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _support_ids(r: int, c: int) -> tuple[Support, Support, Support, Support]:
    return (("h", r), ("v", c), ("d1", r - c), ("d2", r + c))


def _ordered_mode(colors: list[int]) -> int:
    counts = Counter(colors)
    best_color = None
    best_count = -1
    for color in colors:
        count = int(counts[color])
        if count > best_count:
            best_color = int(color)
            best_count = count
    assert best_color is not None
    return best_color


def _segment_from_points(kind: str, points: list[Coord]) -> list[Coord]:
    first = points[0]
    last = points[-1]
    if kind == "h":
        r = first[0]
        return [(r, c) for c in range(first[1], last[1] + 1)]
    if kind == "v":
        c = first[1]
        return [(r, c) for r in range(first[0], last[0] + 1)]
    if kind == "d1":
        steps = last[0] - first[0]
        return [(first[0] + i, first[1] + i) for i in range(steps + 1)]
    assert kind == "d2"
    steps = last[0] - first[0]
    return [(first[0] + i, first[1] - i) for i in range(steps + 1)]


def _detect_lines(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _background_color(g)

    groups: dict[str, dict[int, list[tuple[int, Coord, int]]]] = {
        "h": {},
        "v": {},
        "d1": {},
        "d2": {},
    }
    height, width = g.shape
    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == bg:
                continue
            groups["h"].setdefault(r, []).append((c, (r, c), color))
            groups["v"].setdefault(c, []).append((r, (r, c), color))
            groups["d1"].setdefault(r - c, []).append((r, (r, c), color))
            groups["d2"].setdefault(r + c, []).append((r, (r, c), color))

    order = {"h": 0, "v": 1, "d1": 2, "d2": 3}
    lines: list[dict] = []
    for kind, buckets in groups.items():
        for key, items in buckets.items():
            if len(items) < 3:
                continue
            items.sort(key=lambda item: item[0])
            points = [coord for _pos, coord, _color in items]
            colors = [int(color) for _pos, _coord, color in items]
            lines.append(
                {
                    "support": (kind, int(key)),
                    "points": points,
                    "segment": _segment_from_points(kind, points),
                    "fill_color": _ordered_mode(colors),
                }
            )

    lines.sort(key=lambda line: (order[line["support"][0]], line["support"][1]))
    return lines


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered markers by implicit straight lines, unify each line to its
    majority color, and fill the full segment between its outermost markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    out = g.copy()
    for line in _detect_lines(g, bg=bg):
        color = int(line["fill_color"])
        for r, c in line["segment"]:
            out[r, c] = color
    return out


# --------------- generator ---------------

_GRID_MIN = 11
_GRID_MAX = 18
_LINE_COUNT_RANGE = (1, 3)
_LINE_SPAN_MIN = 4
_LINE_SPAN_MAX = 8
_MARKER_COUNT_MAX = 5
_ISOLATE_COUNT_RANGE = (0, 4)
_MAX_ATTEMPTS = 400
_COLORS = tuple(range(10))


def _segment_halo(cells: set[Coord], height: int, width: int) -> set[Coord]:
    halo: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    halo.add((rr, cc))
    return halo


def _segment_cells(kind: str, start_r: int, start_c: int, span: int) -> list[Coord]:
    dr, dc = _DIRECTION_STEPS[kind]
    return [(start_r + i * dr, start_c + i * dc) for i in range(span)]


def _sample_line_colors(marker_count: int, span: int, fg_colors: list[int]) -> tuple[list[int], int]:
    majority_count = random.randint(marker_count // 2 + 1, marker_count)
    if marker_count == span and majority_count == marker_count:
        majority_count -= 1

    majority_color = random.choice(fg_colors)
    colors = [majority_color] * majority_count
    if majority_count < marker_count:
        others = [color for color in fg_colors if color != majority_color]
        colors.extend(random.choice(others) for _ in range(marker_count - majority_count))
    random.shuffle(colors)
    return colors, _ordered_mode(colors)


def _candidate_line(
    height: int,
    width: int,
    counts: Counter,
    intended_supports: set[Support],
    occupied: set[Coord],
    halo: set[Coord],
    fg_colors: list[int],
) -> dict | None:
    kinds = list(_DIRECTION_STEPS)
    for _ in range(400):
        kind = random.choice(kinds)
        max_span = {"h": width, "v": height, "d1": min(height, width), "d2": min(height, width)}[kind]
        if max_span < 3:
            continue
        span_min = _LINE_SPAN_MIN if max_span >= _LINE_SPAN_MIN else 3
        span = random.randint(span_min, min(_LINE_SPAN_MAX, max_span))

        if kind == "h":
            start_r = random.randint(0, height - 1)
            start_c = random.randint(0, width - span)
            support = ("h", start_r)
        elif kind == "v":
            start_r = random.randint(0, height - span)
            start_c = random.randint(0, width - 1)
            support = ("v", start_c)
        elif kind == "d1":
            start_r = random.randint(0, height - span)
            start_c = random.randint(0, width - span)
            support = ("d1", start_r - start_c)
        else:
            start_r = random.randint(0, height - span)
            start_c = random.randint(span - 1, width - 1)
            support = ("d2", start_r + start_c)

        if support in intended_supports:
            continue

        segment = _segment_cells(kind, start_r, start_c, span)
        if any(cell in halo for cell in segment):
            continue

        marker_count = random.randint(3, min(_MARKER_COUNT_MAX, span))
        chosen = [0, span - 1]
        if marker_count > 2:
            chosen.extend(random.sample(list(range(1, span - 1)), marker_count - 2))
        chosen.sort()
        points = [segment[index] for index in chosen]

        valid = True
        for r, c in points:
            if (r, c) in occupied:
                valid = False
                break
            for other_support in _support_ids(r, c):
                if other_support in intended_supports and other_support != support:
                    valid = False
                    break
                if other_support != support and counts[other_support] + 1 > 2:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue

        colors, fill_color = _sample_line_colors(marker_count, span, fg_colors)
        return {
            "support": support,
            "points": points,
            "segment": segment,
            "point_colors": colors,
            "fill_color": fill_color,
        }
    return None


def _place_line(
    grid: np.ndarray,
    counts: Counter,
    intended_supports: set[Support],
    occupied: set[Coord],
    line: dict,
) -> None:
    for (r, c), color in zip(line["points"], line["point_colors"]):
        grid[r, c] = int(color)
        occupied.add((r, c))
        for support in _support_ids(r, c):
            counts[support] += 1
    intended_supports.add(line["support"])


def _place_isolates(
    grid: np.ndarray,
    bg: int,
    fg_colors: list[int],
    counts: Counter,
    intended_supports: set[Support],
    occupied: set[Coord],
    halo: set[Coord],
) -> None:
    height, width = grid.shape
    target = random.randint(*_ISOLATE_COUNT_RANGE)
    for _ in range(target):
        for _attempt in range(200):
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            if (r, c) in occupied or (r, c) in halo or int(grid[r, c]) != bg:
                continue
            supports = _support_ids(r, c)
            if any(support in intended_supports for support in supports):
                continue
            if any(counts[support] + 1 > 2 for support in supports):
                continue
            if any(
                (r + dr, c + dc) in occupied
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if (dr, dc) != (0, 0)
            ):
                continue

            grid[r, c] = random.choice(fg_colors)
            occupied.add((r, c))
            for support in supports:
                counts[support] += 1
            break


def _intended_signature(lines: list[dict]) -> set[tuple[Support, tuple[Coord, ...]]]:
    return {
        (line["support"], tuple(line["points"]))
        for line in lines
    }


def _detected_signature(lines: list[dict]) -> set[tuple[Support, tuple[Coord, ...]]]:
    return {
        (line["support"], tuple(line["points"]))
        for line in lines
    }


def _build_expected(grid: np.ndarray, lines: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for line in lines:
        color = int(line["fill_color"])
        for r, c in line["segment"]:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(_GRID_MIN, _GRID_MAX)
    width = random.randint(_GRID_MIN, _GRID_MAX)
    bg = random.choice(_COLORS)
    fg_colors = [color for color in _COLORS if color != bg]
    grid = np.full((height, width), bg, dtype=int)

    counts: Counter = Counter()
    intended_supports: set[Support] = set()
    occupied: set[Coord] = set()
    lines: list[dict] = []
    segment_cells: set[Coord] = set()
    halo: set[Coord] = set()

    line_total = random.randint(*_LINE_COUNT_RANGE)
    for _ in range(line_total):
        line = _candidate_line(height, width, counts, intended_supports, occupied, halo, fg_colors)
        if line is None:
            return None
        _place_line(grid, counts, intended_supports, occupied, line)
        lines.append(line)
        segment_cells.update(line["segment"])
        halo = _segment_halo(segment_cells, height, width)

    _place_isolates(grid, bg, fg_colors, counts, intended_supports, occupied, halo)

    detected = _detect_lines(grid, bg=bg)
    if _detected_signature(detected) != _intended_signature(lines):
        return None

    expected = _build_expected(grid, lines)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
