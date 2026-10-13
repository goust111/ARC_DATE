from __future__ import annotations

from collections import Counter
import random

import numpy as np


PINK = 6
Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _trace_path(grid: np.ndarray, bg: int) -> list[Coord]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != bg)]
    assert len(coords) >= 2, "expected a non-background snake path"

    path_set = set(coords)
    degree: dict[Coord, int] = {}
    for r, c in coords:
        degree[(r, c)] = sum((nr, nc) in path_set for nr, nc in _neighbors4(r, c))

    endpoints = [pt for pt, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "snake path must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degree.values()), "snake path must be unbranched"

    order: list[Coord] = []
    previous: Coord | None = None
    current = min(endpoints)
    while True:
        order.append(current)
        r, c = current
        nxt = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if (nr, nc) in path_set and (nr, nc) != previous
        ]
        if not nxt:
            break
        assert len(nxt) == 1, "snake path must not branch"
        previous, current = current, nxt[0]

    assert len(order) == len(path_set), "snake path must be one connected chain"
    return order


def _infer_min_period(values: list[int]) -> tuple[int, dict[int, int]]:
    n = len(values)
    assert any(v == PINK for v in values), "expected pink occluder pixels"
    assert any(v != PINK for v in values), "expected visible path colors"

    for period in range(1, n + 1):
        residue_color: dict[int, int] = {}
        ok = True
        for i, value in enumerate(values):
            if value == PINK:
                continue
            residue = i % period
            if residue in residue_color and residue_color[residue] != value:
                ok = False
                break
            residue_color.setdefault(residue, value)
        if not ok:
            continue

        # Every residue that occurs along the finite path must have at least one
        # visible witness, otherwise a pink cell in that residue is not uniquely
        # recoverable from the input.
        used_residues = {i % period for i in range(n)}
        if used_residues <= residue_color.keys():
            return period, residue_color

    raise AssertionError("visible cells do not determine a complete period")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair pink occlusions on a colored snake path by inferring its one-dimensional repeating period.

    Input structure:
    - The background is the most common color, and pink is not the background.
    - All non-background cells, including pink cells, form one 4-connected unbranched snake path with exactly two endpoints.
    - Ignoring pink cells, the colors along the path are consistent with one smallest repeating period.

    Rule:
    - Trace the snake from an endpoint and number its cells in traversal order.
    - Ignore pink cells and find the smallest period whose visible equal-residue positions all have the same color.
    - Replace each pink path cell with the visible color from its residue class, leaving every other cell unchanged.

    Key insight:
    - The periodic structure lives along the snake's traversal order, not in row and column coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert bg != PINK, "pink must be an occluder, not the background"

    order = _trace_path(g, bg)
    values = [int(g[r, c]) for r, c in order]
    period, table = _infer_min_period(values)

    reverse_period, reverse_table = _infer_min_period(list(reversed(values)))
    n = len(values)
    for i, value in enumerate(values):
        if value == PINK:
            forward = table[i % period]
            backward = reverse_table[(n - 1 - i) % reverse_period]
            assert forward == backward, "endpoint choice must not change the repair"

    out = g.copy()
    for i, (r, c) in enumerate(order):
        if int(g[r, c]) == PINK:
            out[r, c] = table[i % period]
    return out


# --------------- generator ---------------


_MAX_ATTEMPTS = 500
_PERIODS = (6, 7, 8)
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _expand_coarse_path(coarse_path: list[Coord]) -> list[Coord]:
    path: list[Coord] = [coarse_path[0]]
    for (r0, c0), (r1, c1) in zip(coarse_path, coarse_path[1:]):
        path.append(((r0 + r1) // 2, (c0 + c1) // 2))
        path.append((r1, c1))
    return path


def _is_valid_path(path: list[Coord]) -> bool:
    path_set = set(path)
    if len(path_set) != len(path):
        return False
    degrees = [sum(nb in path_set for nb in _neighbors4(*pt)) for pt in path]
    return degrees.count(1) == 2 and all(deg in (1, 2) for deg in degrees)


def _make_path(height: int, width: int, steps: int) -> list[Coord] | None:
    rows = list(range(1, height - 1, 2))
    cols = list(range(1, width - 1, 2))
    if not rows or not cols or steps + 1 > len(rows) * len(cols):
        return None

    for _ in range(350):
        start = (random.randrange(len(rows)), random.randrange(len(cols)))
        coarse = [start]
        used = {start}
        for _step in range(steps):
            r, c = coarse[-1]
            options: list[Coord] = []
            for dr, dc in random.sample(_DIRECTIONS, len(_DIRECTIONS)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < len(rows) and 0 <= nc < len(cols) and (nr, nc) not in used:
                    options.append((nr, nc))
            if not options:
                break

            def onward_count(pt: Coord) -> int:
                rr, cc = pt
                return sum(
                    0 <= rr + dr < len(rows)
                    and 0 <= cc + dc < len(cols)
                    and (rr + dr, cc + dc) not in used
                    for dr, dc in _DIRECTIONS
                )

            options.sort(key=onward_count, reverse=True)
            limit = min(len(options), 2)
            nxt = random.choice(options[:limit])
            coarse.append(nxt)
            used.add(nxt)
        if len(coarse) != steps + 1:
            continue

        actual = _expand_coarse_path([(rows[r], cols[c]) for r, c in coarse])
        if _is_valid_path(actual):
            return actual
    return None


def _make_period(colors: list[int], period: int) -> list[int]:
    a, b, c, d, *extra = colors
    # Residues 1 and 4 both sit between A and C, but they have different
    # answers B and D. This makes local-neighbor completion ambiguous.
    pattern = [a, b, c, a, d, c]
    pattern.extend(extra[: period - len(pattern)])
    return pattern


def _exact_period(values: list[int]) -> int:
    n = len(values)
    for period in range(1, n + 1):
        if all(values[i] == values[i % period] for i in range(n)):
            return period
    return n


def _choose_required_occluders(n: int, period: int) -> tuple[int, int] | None:
    first = [i for i in range(1, n - 1) if i % period == 1]
    second = [i for i in range(1, n - 1) if i % period == 4]
    random.shuffle(first)
    random.shuffle(second)
    for a in first:
        for b in second:
            if abs(a - b) > 1:
                return a, b
    return None


def _corrupt_values(values: list[int], period: int) -> tuple[list[int], set[int]] | None:
    n = len(values)
    required_pair = _choose_required_occluders(n, period)
    if required_pair is None:
        return None

    required = set(required_pair)
    protected = {i - 1 for i in required} | {i + 1 for i in required}
    protected = {i for i in protected if 0 <= i < n}

    low = max(4, n // 9)
    high = min(max(low, n // 4), n - period)
    if high < low:
        return None

    candidates = [i for i in range(n) if i not in required and i not in protected]
    for _ in range(250):
        count = random.randint(low, high)
        if count < len(required) or count - len(required) > len(candidates):
            continue
        corrupt = set(required)
        corrupt.update(random.sample(candidates, count - len(required)))

        trial = list(values)
        for i in corrupt:
            trial[i] = PINK

        visible_residues = {i % period for i, value in enumerate(trial) if value != PINK}
        if visible_residues != set(range(period)):
            continue
        try:
            inferred, _table = _infer_min_period(trial)
            reverse_inferred, _reverse_table = _infer_min_period(list(reversed(trial)))
        except AssertionError:
            continue
        if inferred != period or reverse_inferred != period:
            continue
        return trial, corrupt
    return None


def _render(path: list[Coord], values: list[int], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for (r, c), color in zip(path, values):
        grid[r, c] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(15, 30)
    width = random.randint(15, 30)
    coarse_capacity = ((height - 1) // 2) * ((width - 1) // 2)
    max_steps = min(36, coarse_capacity - 1)
    min_steps = 14
    if max_steps < min_steps:
        return None

    steps = random.randint(min_steps, max_steps)
    path = _make_path(height, width, steps)
    if path is None:
        return None

    period = random.choice(_PERIODS)
    if len(path) < 3 * period:
        return None

    bg = random.choice([0, 7, 8, 9])
    palette = [color for color in range(10) if color not in (bg, PINK)]
    if len(palette) < 6:
        return None
    colors = random.sample(palette, 6)
    pattern = _make_period(colors, period)
    full_values = [pattern[i % period] for i in range(len(path))]
    if _exact_period(pattern) != period or _exact_period(full_values) != period:
        return None
    if len(set(pattern)) == len(pattern):
        return None

    corrupted = _corrupt_values(full_values, period)
    if corrupted is None:
        return None
    input_values, corrupt = corrupted

    inp = _render(path, input_values, (height, width), bg)
    expected = _render(path, full_values, (height, width), bg)
    if _get_bg(inp) != bg:
        return None

    required_colors = {full_values[i] for i in corrupt if i % period in (1, 4)}
    if len(required_colors) < 2:
        return None

    try:
        order = _trace_path(inp, bg)
        solved = main(inp)
    except Exception:
        return None
    if len(order) != len(path):
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid snake-period repair input")
