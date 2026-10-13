from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

CHEBYSHEV = "chebyshev"
MANHATTAN = "manhattan"


Coord = tuple[int, int]


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _grey_components(g: np.ndarray) -> list[np.ndarray]:
    mask = g == GREY
    seen = np.zeros(g.shape, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = g.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for rr, cc in _neighbors8(r, c):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _is_diagonal_segment(coords: np.ndarray) -> bool:
    if len(coords) < 2:
        return False
    by_r = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
    steps = np.diff(by_r, axis=0)
    if np.all(steps == np.array([1, 1])) or np.all(steps == np.array([1, -1])):
        return True
    by_c = coords[np.lexsort((coords[:, 0], coords[:, 1]))]
    steps = np.diff(by_c, axis=0)
    return bool(np.all(np.abs(steps) == 1))


def _is_right_angle_fragment(coords: np.ndarray) -> bool:
    cells = {tuple(map(int, rc)) for rc in coords}
    if len(cells) < 3:
        return False
    for er, ec in cells:
        row_cells = {(r, c) for r, c in cells if r == er}
        col_cells = {(r, c) for r, c in cells if c == ec}
        if row_cells | col_cells != cells:
            continue
        if len(row_cells) < 2 or len(col_cells) < 2:
            continue
        row_cols = sorted(c for _r, c in row_cells)
        col_rows = sorted(r for r, _c in col_cells)
        if row_cols != list(range(row_cols[0], row_cols[-1] + 1)):
            continue
        if col_rows != list(range(col_rows[0], col_rows[-1] + 1)):
            continue
        return True
    return False


def _fragment_metric(coords: np.ndarray) -> str:
    if _is_diagonal_segment(coords):
        return MANHATTAN
    if _is_right_angle_fragment(coords):
        return CHEBYSHEV
    raise AssertionError("each gray hint must be either a diagonal segment or a right-angle corner")


def _distance(a: Coord, b: Coord, metric: str) -> int:
    dr = abs(int(a[0]) - int(b[0]))
    dc = abs(int(a[1]) - int(b[1]))
    if metric == CHEBYSHEV:
        return max(dr, dc)
    if metric == MANHATTAN:
        return dr + dc
    raise ValueError(metric)


def _ring_radius(center: Coord, coords: np.ndarray, metric: str) -> int | None:
    distances = [_distance(center, tuple(map(int, rc)), metric) for rc in coords]
    if len(set(distances)) != 1 or distances[0] == 0:
        return None
    return int(distances[0])


def _control_points(g: np.ndarray) -> tuple[Coord, Coord]:
    red = np.argwhere(g == RED)
    blue = np.argwhere(g == BLUE)
    assert red.shape[0] == 1, "expected one red control pixel"
    assert blue.shape[0] == 1, "expected one blue control pixel"
    return tuple(map(int, red[0])), tuple(map(int, blue[0]))


def _infer_metrics(g: np.ndarray) -> dict[Coord, str]:
    red_center, blue_center = _control_points(g)
    comps = _grey_components(g)
    assert len(comps) == 2, "expected exactly two gray hint fragments"

    fragments = [(_fragment_metric(comp), comp) for comp in comps]
    assert sorted(metric for metric, _comp in fragments) == [CHEBYSHEV, MANHATTAN]

    centers = [red_center, blue_center]
    choices: list[tuple[int, dict[Coord, str]]] = []
    for first_center in (0, 1):
        assignment = [(0, first_center), (1, 1 - first_center)]
        total_radius = 0
        metric_by_center: dict[Coord, str] = {}
        ok = True
        for frag_idx, center_idx in assignment:
            metric, comp = fragments[frag_idx]
            center = centers[center_idx]
            radius = _ring_radius(center, comp, metric)
            if radius is None:
                ok = False
                break
            total_radius += radius
            metric_by_center[center] = metric
        if ok:
            choices.append((total_radius, metric_by_center))

    assert choices, "gray hints must sit on rings around the two controls"
    best_radius = min(radius for radius, _mapping in choices)
    best = [mapping for radius, mapping in choices if radius == best_radius]
    assert len(best) == 1, "gray hints must assign metrics to controls uniquely"
    assert best[0][red_center] != best[0][blue_center], "the two controls must use different metrics"
    return best[0]


def _paint_overlay(shape: tuple[int, int], red_center: Coord, blue_center: Coord, metrics: dict[Coord, str]) -> np.ndarray:
    height, width = shape
    out = np.zeros(shape, dtype=int)
    red_metric = metrics[red_center]
    blue_metric = metrics[blue_center]
    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if cell == red_center:
                out[r, c] = RED
                continue
            if cell == blue_center:
                out[r, c] = BLUE
                continue
            red_parity = _distance(red_center, cell, red_metric) % 2
            blue_parity = _distance(blue_center, cell, blue_metric) % 2
            if red_parity == blue_parity == 1:
                out[r, c] = YELLOW
            elif red_parity == blue_parity == 0:
                out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color the canvas by overlaying two parity distance fields from the red and blue controls.

    Input structure:
    - The background is black, with exactly one red control pixel and one blue control pixel.
    - Each control has one nearby gray hint fragment lying on an equal-distance ring.
    - A right-angle gray fragment indicates a square-ring Chebyshev field, while a diagonal gray fragment indicates a diamond-ring Manhattan field.

    Rule:
    - Infer each control's metric from its own gray fragment; the two inferred metrics are different.
    - For every non-control cell, compute its distance parity to both controls using their inferred metrics.
    - Paint cells yellow when both distances are odd, red when both are even, and black when the parities differ; keep the two control pixels unchanged.

    Key insight:
    - The visible result is not a single set of concentric rings: it is the interference pattern of two different distance fields whose metrics are encoded only by the gray remnants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"
    allowed = {BLACK, BLUE, RED, GREY}
    assert set(map(int, np.unique(g))).issubset(allowed), "input should contain only black, red, blue, and gray"

    red_center, blue_center = _control_points(g)
    metrics = _infer_metrics(g)
    return _paint_overlay(g.shape, red_center, blue_center, metrics)


# --------------- generator ---------------

GEN_SIZE_RANGE = (13, 23)
GEN_CENTER_MARGIN = 4
GEN_MIN_CONTROL_SEP = 6
GEN_MAX_ATTEMPTS = 500
GEN_MIN_OUTPUT_FRACTION = 0.10

_GEN_CALL_COUNT = 0


def _in_bounds(shape: tuple[int, int], cells: list[Coord]) -> bool:
    height, width = shape
    return all(0 <= r < height and 0 <= c < width for r, c in cells)


def _too_close(cells: list[Coord], used: set[Coord], gap: int = 1) -> bool:
    return any(max(abs(r - ur), abs(c - uc)) <= gap for r, c in cells for ur, uc in used)


def _chebyshev_fragment(center: Coord, radius: int, sr: int, sc: int) -> list[Coord]:
    r, c = center
    return [
        (r + sr * radius, c + sc * radius),
        (r + sr * (radius - 1), c + sc * radius),
        (r + sr * radius, c + sc * (radius - 1)),
    ]


def _manhattan_fragment(center: Coord, radius: int, sr: int, sc: int, t: int) -> list[Coord]:
    r, c = center
    return [(r + sr * (t + i), c + sc * (radius - (t + i))) for i in range(3)]


def _fragment_candidates(
    shape: tuple[int, int],
    center: Coord,
    other_center: Coord,
    metric: str,
    used: set[Coord],
) -> list[list[Coord]]:
    candidates: list[list[Coord]] = []
    if metric == CHEBYSHEV:
        for radius in (2, 3, 4):
            for sr in (-1, 1):
                for sc in (-1, 1):
                    cells = _chebyshev_fragment(center, radius, sr, sc)
                    if not _in_bounds(shape, cells) or _too_close(cells, used):
                        continue
                    own = _ring_radius(center, np.array(cells, dtype=int), CHEBYSHEV)
                    other = _ring_radius(other_center, np.array(cells, dtype=int), CHEBYSHEV)
                    if own is None:
                        continue
                    if other is not None and other <= own + 2:
                        continue
                    candidates.append(cells)
    else:
        for radius in (4, 5, 6):
            for sr in (-1, 1):
                for sc in (-1, 1):
                    for t in range(1, radius - 2):
                        cells = _manhattan_fragment(center, radius, sr, sc, t)
                        if not _in_bounds(shape, cells) or _too_close(cells, used):
                            continue
                        own = _ring_radius(center, np.array(cells, dtype=int), MANHATTAN)
                        other = _ring_radius(other_center, np.array(cells, dtype=int), MANHATTAN)
                        if own is None:
                            continue
                        if other is not None and other <= own + 2:
                            continue
                        candidates.append(cells)
    candidates.sort()
    return candidates


def _choose_fragment(
    shape: tuple[int, int],
    center: Coord,
    other_center: Coord,
    metric: str,
    used: set[Coord],
) -> list[Coord] | None:
    candidates = _fragment_candidates(shape, center, other_center, metric, used)
    if not candidates:
        return None
    seed = (
        shape[0] * 101
        + shape[1] * 53
        + center[0] * 29
        + center[1] * 17
        + other_center[0] * 13
        + other_center[1] * 7
        + (3 if metric == CHEBYSHEV else 11)
    )
    return candidates[seed % len(candidates)]


def _expected_from_spec(
    shape: tuple[int, int],
    red_center: Coord,
    blue_center: Coord,
    red_metric: str,
    blue_metric: str,
) -> np.ndarray:
    return _paint_overlay(shape, red_center, blue_center, {red_center: red_metric, blue_center: blue_metric})


def _gen_semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    red_center: Coord,
    blue_center: Coord,
    red_metric: str,
    blue_metric: str,
) -> bool:
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    inferred = _infer_metrics(inp)
    if inferred[red_center] != red_metric or inferred[blue_center] != blue_metric:
        return False

    area = int(inp.size)
    for color in (BLACK, RED, YELLOW):
        if int(np.count_nonzero(expected == color)) < max(3, int(area * GEN_MIN_OUTPUT_FRACTION)):
            return False
    if int(np.count_nonzero(expected == BLUE)) != 1:
        return False

    grey_out = expected[inp == GREY]
    if GREY in set(map(int, np.unique(expected))):
        return False
    if not np.any(grey_out != BLACK):
        return False

    swapped = _expected_from_spec(inp.shape, red_center, blue_center, blue_metric, red_metric)
    if int(np.count_nonzero(swapped != expected)) < max(6, area // 8):
        return False
    return True


def _sample_centers(height: int, width: int) -> tuple[Coord, Coord] | None:
    rows = range(GEN_CENTER_MARGIN, height - GEN_CENTER_MARGIN)
    cols = range(GEN_CENTER_MARGIN, width - GEN_CENTER_MARGIN)
    all_cells = [(r, c) for r in rows for c in cols]
    random.shuffle(all_cells)
    for red_center in all_cells:
        blue_options = [
            cell
            for cell in all_cells
            if max(abs(cell[0] - red_center[0]), abs(cell[1] - red_center[1])) >= GEN_MIN_CONTROL_SEP
        ]
        if blue_options:
            return red_center, random.choice(blue_options)
    return None


def _gen_attempt(red_metric: str) -> np.ndarray | None:
    height = random.randint(*GEN_SIZE_RANGE)
    width = random.randint(*GEN_SIZE_RANGE)
    centers = _sample_centers(height, width)
    if centers is None:
        return None
    red_center, blue_center = centers
    blue_metric = MANHATTAN if red_metric == CHEBYSHEV else CHEBYSHEV

    inp = np.zeros((height, width), dtype=int)
    inp[red_center] = RED
    inp[blue_center] = BLUE
    used: set[Coord] = {red_center, blue_center}

    red_fragment = _choose_fragment(inp.shape, red_center, blue_center, red_metric, used)
    if red_fragment is None:
        return None
    for cell in red_fragment:
        inp[cell] = GREY
    used.update(red_fragment)

    blue_fragment = _choose_fragment(inp.shape, blue_center, red_center, blue_metric, used)
    if blue_fragment is None:
        return None
    for cell in blue_fragment:
        inp[cell] = GREY

    expected = _expected_from_spec(inp.shape, red_center, blue_center, red_metric, blue_metric)
    if not _gen_semantic_checks(inp, expected, red_center, blue_center, red_metric, blue_metric):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    red_metric = CHEBYSHEV if _GEN_CALL_COUNT % 2 else MANHATTAN
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(red_metric)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-field overlay input")
