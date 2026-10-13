from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BACKGROUND = 0
OVERLAP = 6
GEN_SIDES = (13, 15, 17)
GEN_BAR_COUNTS = (3, 4, 5)
GEN_COLORS = (1, 2, 3, 4, 7, 8, 9)
GEN_LENGTHS = (3, 5, 7, 9)
GEN_LENGTH_WEIGHTS = (4, 5, 3, 2)
GEN_MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _line_cells(center: tuple[int, int], length: int, orient: str) -> list[tuple[int, int]]:
    cr, cc = map(int, center)
    half = int(length) // 2
    if orient == "H":
        return [(cr, c) for c in range(cc - half, cc + half + 1)]
    if orient == "V":
        return [(r, cc) for r in range(cr - half, cr + half + 1)]
    raise ValueError(f"unknown orientation: {orient}")


def _rotated_orient(orient: str) -> str:
    return "V" if orient == "H" else "H"


def _touch_or_overlap(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]]) -> bool:
    if cells_a & cells_b:
        return True
    for r, c in cells_a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells_b:
                return True
    return False


def _input_cells(bar: dict) -> set[tuple[int, int]]:
    return set(_line_cells(bar["center"], bar["length"], bar["orient"]))


def _rotated_cells(bar: dict) -> set[tuple[int, int]]:
    return set(_line_cells(bar["center"], bar["length"], _rotated_orient(bar["orient"])))


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(set(map(int, g.flat)) - {BACKGROUND})
    assert 3 <= len(colors) <= 5, "expected 3 to 5 colored bars"
    assert OVERLAP not in colors, "magenta is reserved for output overlaps"

    bars: list[dict] = []
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()
    occupied: list[set[tuple[int, int]]] = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, "every foreground color must be visible"

        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        length = max(height, width)

        assert (height == 1) ^ (width == 1), "each object must be a one-cell-thick bar"
        assert int(coords.shape[0]) == length, "bar must be solid and contiguous"
        assert length >= 3 and length % 2 == 1, "bars must have odd length in [3, 9]"

        if height == 1:
            row = int(r0)
            cols = sorted(map(int, coords[:, 1]))
            assert cols == list(range(int(c0), int(c1) + 1)), "horizontal bar must be contiguous"
            orient = "H"
        else:
            col = int(c0)
            rows = sorted(map(int, coords[:, 0]))
            assert rows == list(range(int(r0), int(r1) + 1)), "vertical bar must be contiguous"
            orient = "V"

        center = (int((r0 + r1) // 2), int((c0 + c1) // 2))
        assert center[0] not in seen_rows and center[1] not in seen_cols, "bar centers must have distinct rows and columns"
        seen_rows.add(center[0])
        seen_cols.add(center[1])

        bar = {"color": int(color), "center": center, "length": int(length), "orient": orient}
        cells = _input_cells(bar)
        for prev in occupied:
            assert not _touch_or_overlap(cells, prev), "input bars must not overlap or touch 4-neighbors"
        occupied.append(cells)
        bars.append(bar)

    return bars


def _render_rotated_bars(bars: list[dict], shape: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    claims: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

    for bar in bars:
        for r, c in _rotated_cells(bar):
            assert 0 <= r < height and 0 <= c < width, "rotated bar must stay in bounds"
            claims[(r, c)].add(int(bar["color"]))

    out = np.zeros((height, width), dtype=int)
    for (r, c), colors in claims.items():
        out[r, c] = next(iter(colors)) if len(colors) == 1 else OVERLAP
    return out


def _bars_to_grid(bars: list[dict], side: int) -> np.ndarray:
    grid = np.zeros((side, side), dtype=int)
    for bar in bars:
        for r, c in _input_cells(bar):
            grid[r, c] = int(bar["color"])
    return grid


def _pair_crosses_after_rotation(bar_a: dict, bar_b: dict) -> bool:
    return bool(_rotated_cells(bar_a) & _rotated_cells(bar_b))


def _semantic_checks(bars: list[dict], side: int) -> bool:
    if len({bar["color"] for bar in bars}) != len(bars):
        return False
    if len({bar["center"][0] for bar in bars}) != len(bars):
        return False
    if len({bar["center"][1] for bar in bars}) != len(bars):
        return False
    if {bar["orient"] for bar in bars} != {"H", "V"}:
        return False

    input_sets = [_input_cells(bar) for bar in bars]
    for i in range(len(input_sets)):
        for j in range(i + 1, len(input_sets)):
            if _touch_or_overlap(input_sets[i], input_sets[j]):
                return False

    out = _render_rotated_bars(bars, (side, side))
    magenta_count = int(np.count_nonzero(out == OVERLAP))
    if not (1 <= magenta_count <= 4):
        return False

    cross_pairs = 0
    noncross_mixed_pairs = 0
    for i in range(len(bars)):
        for j in range(i + 1, len(bars)):
            mixed = bars[i]["orient"] != bars[j]["orient"]
            crossed = _pair_crosses_after_rotation(bars[i], bars[j])
            if crossed:
                cross_pairs += 1
            if mixed and not crossed:
                noncross_mixed_pairs += 1
    if cross_pairs == 0 or noncross_mixed_pairs == 0:
        return False

    if np.array_equal(out, _bars_to_grid(bars, side)):
        return False

    if any(int(np.count_nonzero(out == bar["color"])) < 2 for bar in bars):
        return False

    magenta_cells = set(map(tuple, np.argwhere(out == OVERLAP)))
    if not any(_rotated_cells(bar).isdisjoint(magenta_cells) for bar in bars):
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate every colored thin bar 90 degrees about its own center, then recolor cross-color intersections magenta.

    Input structure:
    - The background is black.
    - There are 3 to 5 disconnected one-cell-thick solid bars, each in a distinct non-black color.
    - Every bar is horizontal or vertical, has odd length, and its center row and center column are unique among all bars.
    - Bars do not overlap or touch 4-neighbors, and magenta does not appear in the input.

    Rule:
    - Clear the canvas to black.
    - Rotate each bar 90 degrees around its own center while preserving its center, length, and color.
    - Draw all rotated bars on the cleared canvas.
    - A cell covered by exactly one rotated bar takes that bar's color; a cell covered by two or more different colors becomes magenta.

    Key insight:
    - The magenta cells are not visible in the input: they only appear after every bar is independently rotated to its center-fixed position and the rotated occupancies are compared globally.
    """
    g = np.asarray(grid, dtype=int)
    bars = _extract_bars(g)
    return _render_rotated_bars(bars, g.shape)


def _sample_length(valid_lengths: list[int]) -> int:
    weights = [GEN_LENGTH_WEIGHTS[GEN_LENGTHS.index(length)] for length in valid_lengths]
    return random.choices(valid_lengths, weights=weights, k=1)[0]


def _sample_extra_centers(side: int, length: int, used_rows: set[int], used_cols: set[int]) -> list[tuple[int, int]]:
    half = length // 2
    rows = [r for r in range(half, side - half) if r not in used_rows]
    cols = [c for c in range(half, side - half) if c not in used_cols]
    centers = [(r, c) for r in rows for c in cols]
    random.shuffle(centers)
    return centers


def _find_cross_pair_centers(side: int, bar_a: dict, bar_b: dict) -> tuple[tuple[int, int], tuple[int, int]] | None:
    half_a = bar_a["length"] // 2
    half_b = bar_b["length"] // 2
    rows_a = range(half_a, side - half_a)
    cols_a = range(half_a, side - half_a)
    rows_b = range(half_b, side - half_b)
    cols_b = range(half_b, side - half_b)

    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r_a in rows_a:
        for c_a in cols_a:
            for r_b in rows_b:
                if r_b == r_a:
                    continue
                for c_b in cols_b:
                    if c_b == c_a:
                        continue
                    placed_a = {**bar_a, "center": (r_a, c_a)}
                    placed_b = {**bar_b, "center": (r_b, c_b)}
                    if not _pair_crosses_after_rotation(placed_a, placed_b):
                        continue
                    if _touch_or_overlap(_input_cells(placed_a), _input_cells(placed_b)):
                        continue
                    candidates.append(((r_a, c_a), (r_b, c_b)))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_candidate_bars(side: int, bar_count: int) -> list[dict] | None:
    colors = random.sample(GEN_COLORS, bar_count)
    valid_lengths = [length for length in GEN_LENGTHS if length <= side]

    cross_lengths = random.sample(valid_lengths, 2)
    cross_first_orient = random.choice(("H", "V"))
    bars = [
        {"color": colors[0], "center": None, "length": cross_lengths[0], "orient": cross_first_orient},
        {"color": colors[1], "center": None, "length": cross_lengths[1], "orient": _rotated_orient(cross_first_orient)},
    ]

    if bar_count >= 3:
        bars.append(
            {
                "color": colors[2],
                "center": None,
                "length": _sample_length(valid_lengths),
                "orient": bars[1]["orient"],
            }
        )
    for color in colors[3:]:
        bars.append(
            {
                "color": color,
                "center": None,
                "length": _sample_length(valid_lengths),
                "orient": random.choice(("H", "V")),
            }
        )

    cross_centers = _find_cross_pair_centers(side, bars[0], bars[1])
    if cross_centers is None:
        return None
    bars[0]["center"], bars[1]["center"] = cross_centers

    used_rows = {bars[0]["center"][0], bars[1]["center"][0]}
    used_cols = {bars[0]["center"][1], bars[1]["center"][1]}
    placed = bars[:2]

    for index in range(2, len(bars)):
        candidates = _sample_extra_centers(side, bars[index]["length"], used_rows, used_cols)
        random.shuffle(candidates)
        chosen = None
        for center in candidates:
            candidate = {**bars[index], "center": center}
            if any(_touch_or_overlap(_input_cells(candidate), _input_cells(prev)) for prev in placed):
                continue
            if index == 2 and _pair_crosses_after_rotation(candidate, bars[0]):
                continue
            chosen = candidate
            break
        if chosen is None:
            return None
        bars[index] = chosen
        placed.append(chosen)
        used_rows.add(chosen["center"][0])
        used_cols.add(chosen["center"][1])

    return bars


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        side = random.choice(GEN_SIDES)
        bar_count = random.choice(GEN_BAR_COUNTS)
        bars = _build_candidate_bars(side, bar_count)
        if bars is None or not _semantic_checks(bars, side):
            continue

        grid = _bars_to_grid(bars, side)
        parsed = _extract_bars(grid)
        if not _semantic_checks(parsed, side):
            continue
        if not np.array_equal(main(grid), _render_rotated_bars(bars, (side, side))):
            continue
        return grid

    raise RuntimeError("failed to generate a valid rotating-bar instance")
