from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np


BACKGROUND = 0
B_COLOR = 1
RED = 2
A_COLOR = 3

DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SIZE_MIN = 6
SIZE_MAX = 12
MAX_GENERATION_ATTEMPTS = 300
MODE_SEQUENCE = ("plain", "intersection", "intersection")

_GENERATION_CALL_COUNT = 0


def _find_single_pixel(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return tuple(map(int, coords[0]))


def _trace_bounce_path(
    shape: tuple[int, int],
    a_pos: tuple[int, int],
    b_pos: tuple[int, int],
    a_color: int,
    b_color: int,
) -> dict[str, object]:
    h, w = shape
    ar, ac = a_pos
    br, bc = b_pos
    dr = int(math.copysign(1, br - ar))
    dc = int(math.copysign(1, bc - ac))
    assert abs(br - ar) == 1 and abs(bc - ac) == 1, "emitter pixels must be diagonally adjacent"

    r, c = b_pos
    current_color = b_color
    visits: list[tuple[int, int]] = []
    first_colors: dict[tuple[int, int], int] = {}
    seen_states: set[tuple[int, int, int, int, int]] = set()
    bounce_count = 0

    while True:
        state = (r, c, dr, dc, current_color)
        assert state not in seen_states, "path failed to terminate at a corner"
        seen_states.add(state)

        nr, nc = r + dr, c + dc
        bounced = False
        if not (0 <= nr < h):
            dr *= -1
            nr = r + dr
            bounced = True
        if not (0 <= nc < w):
            dc *= -1
            nc = c + dc
            bounced = True
        if bounced:
            bounce_count += 1
            current_color = a_color if current_color == b_color else b_color

        r, c = nr, nc
        visits.append((r, c))
        first_colors.setdefault((r, c), current_color)
        if r in (0, h - 1) and c in (0, w - 1):
            break

    counts = Counter(visits)
    emitter_cells = {a_pos, b_pos}
    single_color_counts = Counter(
        first_colors[pos]
        for pos, n_visits in counts.items()
        if n_visits == 1 and pos not in emitter_cells
    )
    red_count = sum(1 for pos, n_visits in counts.items() if n_visits > 1 and pos not in emitter_cells)
    unique_non_emitter = sum(1 for pos in counts if pos not in emitter_cells)

    return {
        "counts": counts,
        "first_colors": first_colors,
        "bounce_count": bounce_count,
        "path_length": len(visits),
        "red_count": red_count,
        "unique_non_emitter": unique_non_emitter,
        "single_color_counts": single_color_counts,
        "corner": (r, c),
    }


def _render_from_trace(
    grid: np.ndarray,
    a_pos: tuple[int, int],
    b_pos: tuple[int, int],
    trace: dict[str, object],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    counts: Counter[tuple[int, int]] = trace["counts"]  # type: ignore[assignment]
    first_colors: dict[tuple[int, int], int] = trace["first_colors"]  # type: ignore[assignment]
    emitter_cells = {a_pos, b_pos}

    for pos, n_visits in counts.items():
        if pos in emitter_cells:
            continue
        out[pos] = RED if n_visits > 1 else first_colors[pos]
    return out


def _render_constant_color(
    grid: np.ndarray,
    a_pos: tuple[int, int],
    b_pos: tuple[int, int],
    trace: dict[str, object],
    color: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    counts: Counter[tuple[int, int]] = trace["counts"]  # type: ignore[assignment]
    emitter_cells = {a_pos, b_pos}
    for pos, n_visits in counts.items():
        if pos in emitter_cells:
            continue
        out[pos] = RED if n_visits > 1 else color
    return out


def _solve_with_roles(
    grid: np.ndarray,
    a_pos: tuple[int, int],
    b_pos: tuple[int, int],
    a_color: int,
    b_color: int,
) -> np.ndarray:
    trace = _trace_bounce_path(grid.shape, a_pos, b_pos, a_color, b_color)
    return _render_from_trace(grid, a_pos, b_pos, trace)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bounce a two-color diagonal emitter around the canvas, alternating segment colors and turning repeated path cells red.

    Input structure:
    - The grid is black except for one diagonal two-pixel emitter: one green A pixel and one blue B pixel.
    - The emitter pixels touch at a corner, and the launch direction is from the green pixel toward the blue pixel.
    - No other foreground objects are present.

    Rule:
    - Copy the input grid.
    - Starting just beyond the blue pixel, continue in that same diagonal direction one cell at a time.
    - Whenever the path reflects off a canvas edge, switch the segment color between blue and green for subsequent cells.
    - Stop when the path reaches a canvas corner.
    - Any non-emitter cell visited once keeps its segment color, any non-emitter cell visited more than once becomes red, and the two emitter pixels stay unchanged.

    Key insight:
    - The ordered green-to-blue emitter determines both the initial diagonal heading and the alternating color state, while red marks exactly the cells where the bouncing diagonal revisits itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BACKGROUND, B_COLOR, A_COLOR}, "unexpected input colors"

    a_pos = _find_single_pixel(g, A_COLOR)
    b_pos = _find_single_pixel(g, B_COLOR)
    trace = _trace_bounce_path(g.shape, a_pos, b_pos, A_COLOR, B_COLOR)
    return _render_from_trace(g, a_pos, b_pos, trace)


def _valid_shapes() -> list[tuple[int, int]]:
    return [
        (h, w)
        for h in range(SIZE_MIN, SIZE_MAX + 1)
        for w in range(SIZE_MIN, SIZE_MAX + 1)
        if h != w and math.gcd(h - 1, w - 1) == 1
    ]


def _candidate_emitters(h: int, w: int, dr: int, dc: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for br in range(1, h - 1):
        for bc in range(1, w - 1):
            a_pos = (br - dr, bc - dc)
            next_pos = (br + dr, bc + dc)
            if 0 <= a_pos[0] < h and 0 <= a_pos[1] < w and 0 <= next_pos[0] < h and 0 <= next_pos[1] < w:
                candidates.append((a_pos, (br, bc)))
    return candidates


def _trace_is_clear(trace: dict[str, object], a_pos: tuple[int, int], b_pos: tuple[int, int], mode: str) -> bool:
    counts: Counter[tuple[int, int]] = trace["counts"]  # type: ignore[assignment]
    if any(pos in counts for pos in (a_pos, b_pos)):
        return False

    bounce_count = int(trace["bounce_count"])
    path_length = int(trace["path_length"])
    red_count = int(trace["red_count"])
    unique_non_emitter = int(trace["unique_non_emitter"])
    single_color_counts: Counter[int] = trace["single_color_counts"]  # type: ignore[assignment]

    if not (2 <= bounce_count <= 6):
        return False
    if not (8 <= path_length <= 30):
        return False
    if unique_non_emitter > 20:
        return False
    if single_color_counts[A_COLOR] < 2 or single_color_counts[B_COLOR] < 3:
        return False

    if mode == "plain":
        return red_count == 0
    return 1 <= red_count <= 5


def _validate_semantics(grid: np.ndarray, mode: str) -> bool:
    try:
        a_pos = _find_single_pixel(grid, A_COLOR)
        b_pos = _find_single_pixel(grid, B_COLOR)
        trace = _trace_bounce_path(grid.shape, a_pos, b_pos, A_COLOR, B_COLOR)
        out = main(grid)
    except AssertionError:
        return False

    expected = _render_from_trace(grid, a_pos, b_pos, trace)
    if not np.array_equal(out, expected):
        return False

    if not _trace_is_clear(trace, a_pos, b_pos, mode):
        return False

    no_toggle = _render_constant_color(grid, a_pos, b_pos, trace, B_COLOR)
    if np.array_equal(out, no_toggle):
        return False

    reverse = _solve_with_roles(grid, b_pos, a_pos, B_COLOR, A_COLOR)
    if np.array_equal(out, reverse):
        return False

    if mode == "intersection" and RED not in out:
        return False
    if mode == "plain" and RED in out:
        return False

    return True


def _generate_candidate(mode: str) -> np.ndarray | None:
    h, w = random.choice(_valid_shapes())
    dr, dc = random.choice(DIAGONALS)
    candidates = _candidate_emitters(h, w, dr, dc)
    random.shuffle(candidates)

    for a_pos, b_pos in candidates:
        trace = _trace_bounce_path((h, w), a_pos, b_pos, A_COLOR, B_COLOR)
        if not _trace_is_clear(trace, a_pos, b_pos, mode):
            continue

        grid = np.zeros((h, w), dtype=int)
        grid[a_pos] = A_COLOR
        grid[b_pos] = B_COLOR
        if _validate_semantics(grid, mode):
            return grid
    return None


def generate_input() -> np.ndarray:
    global _GENERATION_CALL_COUNT

    mode = MODE_SEQUENCE[_GENERATION_CALL_COUNT % len(MODE_SEQUENCE)]
    _GENERATION_CALL_COUNT += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bounce-path input")
