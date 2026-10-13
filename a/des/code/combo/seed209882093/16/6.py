import random

import numpy as np


Point = tuple[int, int]

_BG = 0
_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_PAIR_TYPES = ("horizontal", "vertical", "l_left", "l_right")
_STRAIGHT_TYPES = {"horizontal", "vertical"}
_TURN_TYPES = {"l_left", "l_right"}

_HEIGHT_RANGE = (8, 15)
_WIDTH_RANGE = (8, 15)
_PAIR_COUNT_RANGE = (3, 6)
_MIN_STRAIGHT_SPAN = 2
_MIN_TURN_SPAN = 2
_CROSSING_PROB = 0.35
_PAIR_PLACE_TRIES = 250
_GEN_TRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _first_seen_colors(grid: np.ndarray, bg: int) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for value in map(int, grid.ravel()):
        if value == bg or value in seen:
            continue
        seen.add(value)
        order.append(value)
    return order


def _path_kind(p1: Point, p2: Point) -> str:
    r1, c1 = p1
    r2, c2 = p2
    if r1 == r2:
        return "horizontal"
    if c1 == c2:
        return "vertical"
    upper, lower = ((r1, c1), (r2, c2)) if r1 < r2 else ((r2, c2), (r1, c1))
    return "l_right" if lower[1] > upper[1] else "l_left"


def _path_cells(p1: Point, p2: Point) -> set[Point]:
    r1, c1 = p1
    r2, c2 = p2
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        return {(r1, c) for c in range(lo, hi + 1)}
    if c1 == c2:
        lo, hi = sorted((r1, r2))
        return {(r, c1) for r in range(lo, hi + 1)}

    upper, lower = ((r1, c1), (r2, c2)) if r1 < r2 else ((r2, c2), (r1, c1))
    if lower[1] > upper[1]:
        cells = {(r, upper[1]) for r in range(upper[0], lower[0] + 1)}
        cells.update((lower[0], c) for c in range(upper[1], lower[1] + 1))
        return cells

    cells = {(upper[0], c) for c in range(lower[1], upper[1] + 1)}
    cells.update((r, lower[1]) for r in range(upper[0], lower[0] + 1))
    return cells


def _draw_connection(out: np.ndarray, color: int, p1: Point, p2: Point) -> None:
    r1, c1 = p1
    r2, c2 = p2
    if r1 == r2:
        lo, hi = sorted((c1, c2))
        out[r1, lo : hi + 1] = color
        return
    if c1 == c2:
        lo, hi = sorted((r1, r2))
        out[lo : hi + 1, c1] = color
        return

    upper, lower = ((r1, c1), (r2, c2)) if r1 < r2 else ((r2, c2), (r1, c1))
    if lower[1] > upper[1]:
        out[upper[0] : lower[0] + 1, upper[1]] = color
        out[lower[0], upper[1] : lower[1] + 1] = color
        return

    out[upper[0], lower[1] : upper[1] + 1] = color
    out[upper[0] : lower[0] + 1, lower[1]] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-color endpoint pair with a straight line or an L turn."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    order = _first_seen_colors(g, bg)
    assert order, "expected at least one colored pair"

    out = np.full_like(g, bg)
    for color in order:
        points = np.argwhere(g == color)
        assert points.shape == (2, 2), "each non-background color must appear exactly twice"
        p1 = tuple(map(int, points[0]))
        p2 = tuple(map(int, points[1]))
        _draw_connection(out, int(color), p1, p2)

    return out


def _sample_pair_types(n_pairs: int) -> list[str]:
    types: list[str] = []
    if n_pairs >= 4 and random.random() < 0.55:
        types.extend(["horizontal", "vertical", "l_left", "l_right"])
    elif n_pairs >= 3 and random.random() < 0.7:
        types.extend([random.choice(("horizontal", "vertical")), "l_left", "l_right"])

    while len(types) < n_pairs:
        types.append(random.choice(_PAIR_TYPES))
    types = types[:n_pairs]

    if not any(t in _STRAIGHT_TYPES for t in types):
        types[random.randrange(n_pairs)] = random.choice(tuple(_STRAIGHT_TYPES))
    if not any(t in _TURN_TYPES for t in types):
        types[random.randrange(n_pairs)] = random.choice(tuple(_TURN_TYPES))
    if len(set(types)) < 2:
        types[-1] = random.choice([t for t in _PAIR_TYPES if t != types[0]])

    random.shuffle(types)
    return types


def _sample_pair(height: int, width: int, kind: str) -> tuple[Point, Point]:
    if kind == "horizontal":
        r = random.randrange(height)
        c1 = random.randrange(width - _MIN_STRAIGHT_SPAN)
        c2 = random.randrange(c1 + _MIN_STRAIGHT_SPAN, width)
        return (r, c1), (r, c2)

    if kind == "vertical":
        c = random.randrange(width)
        r1 = random.randrange(height - _MIN_STRAIGHT_SPAN)
        r2 = random.randrange(r1 + _MIN_STRAIGHT_SPAN, height)
        return (r1, c), (r2, c)

    if kind == "l_right":
        ru = random.randrange(height - _MIN_TURN_SPAN)
        rd = random.randrange(ru + _MIN_TURN_SPAN, height)
        cu = random.randrange(width - _MIN_TURN_SPAN)
        cd = random.randrange(cu + _MIN_TURN_SPAN, width)
        return (ru, cu), (rd, cd)

    assert kind == "l_left"
    ru = random.randrange(height - _MIN_TURN_SPAN)
    rd = random.randrange(ru + _MIN_TURN_SPAN, height)
    cd = random.randrange(width - _MIN_TURN_SPAN)
    cu = random.randrange(cd + _MIN_TURN_SPAN, width)
    return (ru, cu), (rd, cd)


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*_HEIGHT_RANGE)
    width = random.randint(*_WIDTH_RANGE)
    max_pairs = min(_PAIR_COUNT_RANGE[1], len(_COLORS), max(_PAIR_COUNT_RANGE[0], (height * width) // 18))
    n_pairs = random.randint(_PAIR_COUNT_RANGE[0], max_pairs)
    pair_types = _sample_pair_types(n_pairs)
    required_crossings = 1 if random.random() < _CROSSING_PROB else 0

    pairs: list[dict[str, object]] = []
    crossing_cells: set[Point] = set()

    for kind in pair_types:
        placed = False
        for _ in range(_PAIR_PLACE_TRIES):
            p1, p2 = _sample_pair(height, width, kind)
            candidate_path = _path_cells(p1, p2)
            candidate_endpoints = {p1, p2}

            new_crossings: list[Point] = []
            valid = True
            for pair in pairs:
                other_path = pair["path"]
                other_endpoints = pair["endpoints"]
                intersection = candidate_path & other_path
                if not intersection:
                    continue
                if len(intersection) > 1:
                    valid = False
                    break

                cell = next(iter(intersection))
                if cell in candidate_endpoints or cell in other_endpoints or cell in crossing_cells:
                    valid = False
                    break
                new_crossings.append(cell)

            if not valid:
                continue
            if len(crossing_cells) + len(new_crossings) > required_crossings:
                continue

            pairs.append(
                {
                    "kind": kind,
                    "points": (p1, p2),
                    "path": candidate_path,
                    "endpoints": candidate_endpoints,
                }
            )
            crossing_cells.update(new_crossings)
            placed = True
            break

        if not placed:
            return None

    if len(crossing_cells) != required_crossings:
        return None
    if len({str(pair["kind"]) for pair in pairs}) < 2:
        return None

    random.shuffle(pairs)
    colors = random.sample(_COLORS, len(pairs))
    grid = np.full((height, width), _BG, dtype=int)
    for color, pair in zip(colors, pairs):
        p1, p2 = pair["points"]
        grid[p1] = color
        grid[p2] = color

    if int((grid != _BG).sum()) != 2 * len(pairs):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_TRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_TRIES} attempts")
