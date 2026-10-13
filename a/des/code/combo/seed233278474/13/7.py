from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BELT = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_MAX_ATTEMPTS = 400
GEN_MIN_BELT_GAP = 4


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a non-empty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _unique_largest_non_bg(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    ranked = [(int(color), int(count)) for color, count in counts.items() if int(color) != int(bg)]
    assert ranked, "expected at least one non-background color"
    ranked.sort(key=lambda item: (-item[1], item[0]))
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique belt color"
    return int(ranked[0][0])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if bool(value) and start is None:
            start = idx
        if start is not None and (not bool(value) or idx == len(mask) - 1):
            end = idx if bool(value) else idx - 1
            spans.append((int(start), int(end)))
            start = None
    return spans


def _find_belts(grid: np.ndarray, bg: int, belt_color: int) -> list[tuple[int, int, int, int, int]]:
    belts: list[tuple[int, int, int, int, int]] = []
    for r in range(grid.shape[0]):
        row = grid[r]
        for c0, c1 in _runs(row != bg):
            if c1 - c0 + 1 < 3:
                continue
            segment = row[c0 : c1 + 1]
            left = int(segment[0])
            right = int(segment[-1])
            if left in (bg, belt_color) or right in (bg, belt_color) or left == right:
                continue
            if np.all(segment[1:-1] == belt_color):
                belts.append((int(r), int(c0), int(c1), left, right))

    assert belts, "expected at least one belt"
    rows = [r for r, _c0, _c1, _l, _r in belts]
    assert len(rows) == len(set(rows)), "expected at most one belt per row"
    belts.sort()
    return belts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only cargo whose color matches one belt endpoint, then compact the kept cargo toward the matching side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _unique_mode(g)
    belt_color = _unique_largest_non_bg(g, bg)
    belts = _find_belts(g, bg, belt_color)

    out = g.copy()
    for r, c0, c1, left_color, right_color in belts:
        assert r > 0, "each belt must have a cargo row directly above it"
        cargo_r = r - 1
        cargo = g[cargo_r, c0 : c1 + 1]
        left_count = int(np.sum(cargo == left_color))
        right_count = int(np.sum(cargo == right_color))
        belt_len = int(c1 - c0 + 1)
        assert left_count + right_count <= belt_len, "compacted cargo must fit above the belt"

        out[cargo_r, c0 : c1 + 1] = bg
        if left_count:
            out[cargo_r, c0 : c0 + left_count] = left_color
        if right_count:
            out[cargo_r, c1 - right_count + 1 : c1 + 1] = right_color

    return out


def _sample_spaced_rows(height: int, count: int, min_gap: int) -> list[int] | None:
    candidates = list(range(1, height))
    for _ in range(200):
        rows = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _fill_cargo(
    grid: np.ndarray,
    cargo_r: int,
    c0: int,
    length: int,
    left_color: int,
    right_color: int,
) -> list[tuple[int, int]]:
    branch = random.choices(
        ["mixed", "left_only", "right_only", "all_kept", "none_kept"],
        weights=[4, 2, 2, 2, 1],
        k=1,
    )[0]
    occupied = random.randint(max(1, length // 3), length)
    positions = sorted(random.sample(range(length), occupied))
    distractors = [color for color in PALETTE if color not in (left_color, right_color)]
    distractor_count = random.randint(1, min(3, len(distractors)))
    distractors = random.sample(distractors, distractor_count)

    allowed: list[int]
    required: list[int]
    if branch == "mixed":
        allowed = [left_color, right_color, *distractors]
        required = [left_color, right_color]
        if distractors:
            required.append(random.choice(distractors))
    elif branch == "left_only":
        allowed = [left_color, *distractors]
        required = [left_color]
        if distractors:
            required.append(random.choice(distractors))
    elif branch == "right_only":
        allowed = [right_color, *distractors]
        required = [right_color]
        if distractors:
            required.append(random.choice(distractors))
    elif branch == "all_kept":
        allowed = [left_color, right_color]
        required = [random.choice(allowed)]
        if occupied >= 2:
            required = [left_color, right_color]
    else:
        allowed = distractors
        required = [random.choice(distractors)]

    if len(required) > occupied:
        occupied = len(required)
        positions = sorted(random.sample(range(length), occupied))

    colors = list(required)
    while len(colors) < occupied:
        colors.append(random.choice(allowed))
    random.shuffle(colors)

    cargo_cells: list[tuple[int, int]] = []
    for offset, color in zip(positions, colors, strict=True):
        grid[cargo_r, c0 + offset] = int(color)
        cargo_cells.append((int(offset), int(color)))
    return cargo_cells


def _expected_from_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        row = int(spec["row"])
        c0 = int(spec["c0"])
        length = int(spec["length"])
        left_color = int(spec["left"])
        right_color = int(spec["right"])
        cargo_cells = list(spec["cargo"])

        cargo_r = row - 1
        out[cargo_r, c0 : c0 + length] = BG
        left_count = sum(int(color == left_color) for _offset, color in cargo_cells)
        right_count = sum(int(color == right_color) for _offset, color in cargo_cells)
        if left_count:
            out[cargo_r, c0 : c0 + left_count] = left_color
        if right_count:
            out[cargo_r, c0 + length - right_count : c0 + length] = right_color
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(12, 24)
    width = random.randint(12, 24)
    max_belts = 1 + (height - 2) // GEN_MIN_BELT_GAP
    if max_belts < 2:
        return None

    belt_count = random.randint(2, min(5, max_belts))
    rows = _sample_spaced_rows(height, belt_count, GEN_MIN_BELT_GAP)
    if rows is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    specs: list[dict[str, object]] = []

    for row in rows:
        min_len = 6
        max_len = width
        if max_len < min_len:
            return None
        length = random.randint(min_len, max_len)
        c0 = random.randint(0, width - length)
        c1 = c0 + length - 1
        left_color, right_color = random.sample(PALETTE, 2)

        grid[row, c0] = left_color
        grid[row, c1] = right_color
        if length > 2:
            grid[row, c0 + 1 : c1] = BELT

        cargo = _fill_cargo(grid, row - 1, c0, length, left_color, right_color)
        specs.append(
            {
                "row": int(row),
                "c0": int(c0),
                "length": int(length),
                "left": int(left_color),
                "right": int(right_color),
                "cargo": cargo,
            }
        )

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[BELT] <= 0:
        return None
    non_bg = [(color, count) for color, count in counts.items() if color != BG]
    non_bg.sort(key=lambda item: (-item[1], item[0]))
    if len(non_bg) > 1 and non_bg[0][1] <= non_bg[1][1]:
        return None

    expected = _expected_from_specs(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid belt-sorting puzzle")
