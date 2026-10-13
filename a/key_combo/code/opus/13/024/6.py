from __future__ import annotations

import numpy as np


BG = 0
INPUT_COLORS = (1, 2, 3, 4)

GEN_ATTEMPTS = 500
GRID_MIN = 10
GRID_MAX = 15
MARGIN = 1


def _pair_records(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    records: list[tuple[int, int]] = []
    for color in sorted(colors):
        pts = np.argwhere(g == int(color))
        assert int(pts.shape[0]) == 2, f"color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = [(int(r), int(c)) for r, c in pts]
        same_row = int(r0) == int(r1)
        same_col = int(c0) == int(c1)
        assert same_row ^ same_col, f"color {color} markers must share one axis"
        length = abs(int(c1) - int(c0)) - 1 if same_row else abs(int(r1) - int(r0)) - 1
        assert 1 <= int(length) <= 30
        records.append((int(length), int(color)))
    return sorted(records)


def _build_nested(records: list[tuple[int, int]]) -> np.ndarray:
    assert records
    inner_length, inner_color = records[0]
    side = int(2 * inner_length)
    out = np.full((side, side), int(inner_color), dtype=int)
    for length, color in records[1:]:
        thickness = int(length)
        next_side = int(out.shape[0] + 2 * thickness)
        wrapped = np.full((next_side, next_side), int(color), dtype=int)
        wrapped[thickness : thickness + out.shape[0], thickness : thickness + out.shape[1]] = out
        out = wrapped
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert aligned same-color marker distances into nested concentric square layers.

    Input structure:
    - A black background contains 3 or 4 colors chosen from blue, red, green, and yellow.
    - Each foreground color appears exactly twice as marker pixels.
    - The two markers of a color share one row or one column, so they define a straight span.

    Rule:
    - For each color pair, measure the net span length between its two markers, excluding the endpoints.
    - Sort all colors by that net span length from shortest to longest.
    - The shortest span becomes a solid square of side length `2 * L`; each later color wraps the current shape with a square frame of thickness equal to its own span length.

    Key insight:
    - The markers are not copied into the output; their only role is to encode measurable span lengths, and those lengths become the thickness parameters of the nested square construction.
    """
    return _build_nested(_pair_records(np.asarray(grid, dtype=int)))


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if int(r0) == int(r1):
        lo, hi = sorted((int(c0), int(c1)))
        return {(int(r0), int(c)) for c in range(lo, hi + 1)}
    lo, hi = sorted((int(r0), int(r1)))
    return {(int(r), int(c0)) for r in range(lo, hi + 1)}


def _chebyshev_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def _candidate_pairs(
    shape: tuple[int, int], length: int, orientation: str
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = map(int, shape)
    gap = int(length) + 1
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if orientation == "H":
        for r in range(MARGIN, h - MARGIN):
            for c0 in range(MARGIN, w - MARGIN - gap):
                pairs.append(((int(r), int(c0)), (int(r), int(c0 + gap))))
    else:
        for c in range(MARGIN, w - MARGIN):
            for r0 in range(MARGIN, h - MARGIN - gap):
                pairs.append(((int(r0), int(c)), (int(r0 + gap), int(c))))
    return pairs


def _valid_placement(
    pair: tuple[tuple[int, int], tuple[int, int]],
    orientation: str,
    used_rows: set[int],
    used_cols: set[int],
    used_markers: set[tuple[int, int]],
    used_segments: set[tuple[int, int]],
) -> bool:
    a, b = pair
    if orientation == "H" and int(a[0]) in used_rows:
        return False
    if orientation == "V" and int(a[1]) in used_cols:
        return False
    if a in used_markers or b in used_markers:
        return False
    if any(_chebyshev_close(a, m) or _chebyshev_close(b, m) for m in used_markers):
        return False
    return not bool(_segment_cells(a, b) & used_segments)


def _sample_orientations(count: int) -> list[str]:
    branch = int(np.random.randint(3))
    if branch == 0:
        return ["H"] * int(count)
    if branch == 1:
        return ["V"] * int(count)
    h_count = int(np.random.randint(1, count))
    arr = np.array(["H"] * h_count + ["V"] * (int(count) - h_count), dtype=object)
    return [str(x) for x in np.random.permutation(arr)]


def _render_input(
    shape: tuple[int, int],
    placements: dict[int, tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, (a, b) in placements.items():
        grid[a] = int(color)
        grid[b] = int(color)
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        records = _pair_records(grid)
    except AssertionError:
        return False
    lengths = [length for length, _color in records]
    if len(records) not in (3, 4):
        return False
    if len(set(lengths)) != len(lengths):
        return False
    if any(length < 1 or length > 4 for length in lengths):
        return False

    all_segments: set[tuple[int, int]] = set()
    for color in np.unique(grid):
        if int(color) == BG:
            continue
        pts = np.argwhere(grid == int(color))
        a = (int(pts[0, 0]), int(pts[0, 1]))
        b = (int(pts[1, 0]), int(pts[1, 1]))
        seg = _segment_cells(a, b)
        if seg & all_segments:
            return False
        all_segments |= seg

    out = main(grid)
    expected_side = 2 * sum(lengths)
    return bool(out.shape == (expected_side, expected_side) and max(out.shape) <= 30)


def _gen_attempt() -> np.ndarray | None:
    count = 3 if int(np.random.randint(2)) == 0 else 4
    colors = [int(x) for x in np.random.choice(INPUT_COLORS, size=int(count), replace=False)]
    lengths = [int(x) for x in np.random.choice([1, 2, 3, 4], size=int(count), replace=False)]
    orientations = _sample_orientations(int(count))
    specs = list(zip(colors, lengths, orientations))

    h = int(np.random.randint(GRID_MIN, GRID_MAX + 1))
    w = int(np.random.randint(GRID_MIN, GRID_MAX + 1))
    order = sorted(specs, key=lambda item: int(item[1]), reverse=True)

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    used_markers: set[tuple[int, int]] = set()
    used_segments: set[tuple[int, int]] = set()
    placements: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}

    for color, length, orientation in order:
        candidates = [
            pair
            for pair in _candidate_pairs((h, w), int(length), str(orientation))
            if _valid_placement(pair, str(orientation), used_rows, used_cols, used_markers, used_segments)
        ]
        if not candidates:
            return None
        pair = candidates[int(np.random.randint(len(candidates)))]
        placements[int(color)] = pair
        used_markers.update(pair)
        used_segments |= _segment_cells(*pair)
        if orientation == "H":
            used_rows.add(int(pair[0][0]))
        else:
            used_cols.add(int(pair[0][1]))

    grid = _render_input((h, w), placements)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated non-crossing placement attempts")
