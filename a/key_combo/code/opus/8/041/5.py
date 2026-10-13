from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
ORANGE = 7
MAROON = 9

_DIR_SEQUENCE = [
    (-1, -1),
    (0, 1),
    (1, 0),
    (1, 1),
    (0, -1),
    (-1, 0),
    (1, -1),
    (-1, 1),
]
_GEN_CALL_COUNT = 0
_MAX_GENERATION_ATTEMPTS = 400


def _sign(delta: int) -> int:
    return int((delta > 0) - (delta < 0))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.shape == (1, 2), f"expected exactly one pixel of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _orange_component_count(grid: np.ndarray) -> int:
    mask = grid == ORANGE
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0

    height, width = grid.shape
    remaining = {tuple(map(int, rc)) for rc in coords}
    count = 0
    while remaining:
        count += 1
        start = remaining.pop()
        queue: deque[tuple[int, int]] = deque([start])
        while queue:
            r, c = queue.popleft()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    queue.append((nr, nc))
    return count


def _vector_from_markers(green: tuple[int, int], yellow: tuple[int, int]) -> tuple[int, int]:
    dr_raw = int(yellow[0] - green[0])
    dc_raw = int(yellow[1] - green[1])
    assert (dr_raw, dc_raw) != (0, 0), "markers must be distinct"
    assert (
        dr_raw == 0 or dc_raw == 0 or abs(dr_raw) == abs(dc_raw)
    ), "markers must share a row, column, or 45-degree diagonal"
    return _sign(dr_raw), _sign(dc_raw)


def _clean_and_analyze_half(half: np.ndarray) -> tuple[np.ndarray, tuple[int, int], int]:
    colors = set(map(int, np.unique(half)))
    assert colors <= {BLACK, GREEN, YELLOW, ORANGE}, "unexpected colors in input half"

    green = _find_single_coord(half, GREEN)
    yellow = _find_single_coord(half, YELLOW)
    direction = _vector_from_markers(green, yellow)
    distance = max(abs(yellow[0] - green[0]), abs(yellow[1] - green[1]))
    assert distance >= 2, "marker distance must be at least two cells"

    clean = half.copy()
    clean[green] = BLACK
    clean[yellow] = BLACK
    assert np.any(clean == ORANGE), "each half must contain an orange pattern"
    assert _orange_component_count(clean) == 1, "each half must contain one orange component"
    return clean, direction, distance


def _compose_output(top_clean: np.ndarray, bottom_clean: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    half_h, width = top_clean.shape
    assert bottom_clean.shape == (half_h, width), "halves must have the same shape"

    dr, dc = direction
    out_h = half_h * 2 + (1 if dr != 0 else 0)
    out_w = width + (1 if dc != 0 else 0)
    out = np.full((out_h, out_w), MAROON, dtype=int)

    top_c0 = 0 if dc >= 0 else 1
    bot_r0 = half_h + (1 if dr != 0 else 0)
    bot_c0 = 0 if dc <= 0 else 1

    out[0:half_h, top_c0 : top_c0 + width] = top_clean
    out[bot_r0 : bot_r0 + half_h, bot_c0 : bot_c0 + width] = bottom_clean
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stitch the two horizontal halves onto a maroon canvas with a marker-coded offset.

    Input structure:
    - The grid has an even number of rows and black background.
    - The top half and bottom half each contain one orange pattern plus exactly one green marker and one yellow marker.
    - In each half, the green and yellow markers lie on the same row, the same column, or a 45-degree diagonal, and both halves encode the same normalized green-to-yellow direction.

    Rule:
    - Split the grid into equal top and bottom halves.
    - In either half, compute the normalized direction vector (dr, dc) from green to yellow, then remove all green and yellow markers to black.
    - Create a maroon canvas of shape (2 * half_height + [dr != 0], width + [dc != 0]).
    - Place the cleaned top half at row 0 and column 0 if dc >= 0 else 1; place the cleaned bottom half at row half_height + [dr != 0] and column 0 if dc <= 0 else 1.

    Key insight:
    - The markers are not copied; their relative direction is a code that chooses the stitch offset, while the orange half-patterns themselves are preserved unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0, "input height must be even"

    half_h = height // 2
    top = g[:half_h, :]
    bottom = g[half_h:, :]

    top_clean, top_dir, _top_dist = _clean_and_analyze_half(top)
    bottom_clean, bottom_dir, _bottom_dist = _clean_and_analyze_half(bottom)
    assert top_dir == bottom_dir, "both halves must encode the same direction"

    out = _compose_output(top_clean, bottom_clean, top_dir)
    assert not np.any((out == GREEN) | (out == YELLOW)), "markers must be removed in the output"
    return out


def _grow_polyomino(box_h: int, box_w: int, target_area: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    cells = {(int(rng.integers(box_h)), int(rng.integers(box_w)))}
    while len(cells) < target_area:
        frontier = sorted(
            {
                (nr, nc)
                for r, c in cells
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells
            }
        )
        if not frontier:
            return set()
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return cells


def _make_orange_half(half_h: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    grid = np.zeros((half_h, width), dtype=int)

    box_h = int(rng.integers(2, min(half_h, 5) + 1))
    box_w = int(rng.integers(3, min(width, 7) + 1))
    max_area = min(box_h * box_w - 1, 9)
    if max_area < 4:
        return None
    target_area = int(rng.integers(4, max_area + 1))
    cells = _grow_polyomino(box_h, box_w, target_area, rng)
    if not cells:
        return None

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 and len(cols) < 2:
        return None

    r0 = int(rng.integers(0, half_h - box_h + 1))
    c0 = int(rng.integers(0, width - box_w + 1))
    for r, c in cells:
        grid[r0 + r, c0 + c] = ORANGE
    return grid


def _marker_site_is_clear(half: np.ndarray, r: int, c: int) -> bool:
    if half[r, c] != BLACK:
        return False
    height, width = half.shape
    for nr in range(r - 1, r + 2):
        for nc in range(c - 1, c + 2):
            if 0 <= nr < height and 0 <= nc < width and half[nr, nc] == ORANGE:
                return False
    return True


def _candidate_marker_pairs(
    half: np.ndarray, direction: tuple[int, int]
) -> list[tuple[int, int, int, int, int]]:
    dr, dc = direction
    height, width = half.shape
    candidates: list[tuple[int, int, int, int, int]] = []
    for r in range(height):
        for c in range(width):
            if not _marker_site_is_clear(half, r, c):
                continue
            for dist in range(2, max(height, width) + 1):
                rr = r + dr * dist
                cc = c + dc * dist
                if not (0 <= rr < height and 0 <= cc < width):
                    break
                line_clear = True
                for step in range(dist + 1):
                    sr = r + dr * step
                    sc = c + dc * step
                    if half[sr, sc] != BLACK:
                        line_clear = False
                        break
                if line_clear and _marker_site_is_clear(half, rr, cc):
                    candidates.append((dist, r, c, rr, cc))
    return candidates


def _add_markers(
    half: np.ndarray, direction: tuple[int, int], rng: np.random.Generator
) -> tuple[np.ndarray, int] | None:
    candidates = _candidate_marker_pairs(half, direction)
    if not candidates:
        return None
    dist, gr, gc, yr, yc = candidates[int(rng.integers(len(candidates)))]
    out = half.copy()
    out[gr, gc] = GREEN
    out[yr, yc] = YELLOW
    return out, dist


def _make_marked_half(
    half_h: int, width: int, direction: tuple[int, int], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int] | None:
    orange_half = _make_orange_half(half_h, width, rng)
    if orange_half is None:
        return None
    marked = _add_markers(orange_half, direction, rng)
    if marked is None:
        return None
    marked_half, distance = marked
    return marked_half, orange_half, distance


def _direction_for_call() -> tuple[int, int]:
    global _GEN_CALL_COUNT
    direction = _DIR_SEQUENCE[_GEN_CALL_COUNT % len(_DIR_SEQUENCE)]
    _GEN_CALL_COUNT += 1
    return direction


def _validate_generated_input(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    if max(inp.shape) > 30:
        return False

    colors = set(map(int, np.unique(inp)))
    if colors - {BLACK, GREEN, YELLOW, ORANGE}:
        return False

    height, width = inp.shape
    if height % 2 != 0:
        return False
    half_h = height // 2

    top = inp[:half_h, :]
    bottom = inp[half_h:, :]
    try:
        top_clean, top_dir, top_dist = _clean_and_analyze_half(top)
        bottom_clean, bottom_dir, bottom_dist = _clean_and_analyze_half(bottom)
    except AssertionError:
        return False

    if top_dir != bottom_dir:
        return False
    if top_dist < 2 or bottom_dist < 2:
        return False
    if np.array_equal(top_clean, bottom_clean):
        return False

    solved = main(inp)
    expected = _compose_output(top_clean, bottom_clean, top_dir)
    if not np.array_equal(solved, expected):
        return False
    if np.any((solved == GREEN) | (solved == YELLOW)):
        return False

    dr, dc = top_dir
    if solved.shape != (2 * half_h + (1 if dr != 0 else 0), width + (1 if dc != 0 else 0)):
        return False

    if dr != 0 and not np.all(solved[half_h, :] == MAROON):
        return False

    if dc > 0:
        bot_r0 = half_h + (1 if dr != 0 else 0)
        if not np.all(solved[:half_h, width] == MAROON):
            return False
        if not np.all(solved[bot_r0:, 0] == MAROON):
            return False
    elif dc < 0:
        bot_r0 = half_h + (1 if dr != 0 else 0)
        if not np.all(solved[:half_h, 0] == MAROON):
            return False
        if not np.all(solved[bot_r0:, width] == MAROON):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    direction = _direction_for_call()

    half_h = int(rng.integers(4, 8))
    width = int(rng.integers(8, 15))

    top_data = _make_marked_half(half_h, width, direction, rng)
    bottom_data = _make_marked_half(half_h, width, direction, rng)
    if top_data is None or bottom_data is None:
        return None

    top_marked, top_clean, top_dist = top_data
    bottom_marked, bottom_clean, bottom_dist = bottom_data

    if np.array_equal(top_clean, bottom_clean):
        return None

    # Encourage the solver to infer normalization, not a fixed marker distance.
    if top_dist == bottom_dist and max(half_h, width) > 4:
        return None

    inp = np.vstack([top_marked, bottom_marked]).astype(int)
    if not _validate_generated_input(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")
