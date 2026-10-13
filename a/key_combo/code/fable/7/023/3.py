from __future__ import annotations

import random
from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_7_023_3",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
neighbors4 = _ARC_UTILS.neighbors4


def _primitive_period(seq: list[int]) -> int:
    n = len(seq)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        if all(seq[i] == seq[i % p] for i in range(n)):
            return p
    return n


def _trace_snake(mask: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    assert len(cells) >= 3, "expected a snake of length at least 3"

    degree = {}
    for r, c in cells:
        deg = sum((nr, nc) in cells for nr, nc in neighbors4(r, c))
        assert deg in (1, 2), "snake must be a simple path with no branches or loops"
        degree[(r, c)] = deg

    endpoints = [cell for cell, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "snake must have exactly two endpoints"

    ordered = []
    prev = None
    cur = endpoints[0]
    while True:
        ordered.append(cur)
        nxt = [(nr, nc) for nr, nc in neighbors4(*cur) if (nr, nc) in cells and (nr, nc) != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "snake traversal must be unambiguous"
        prev, cur = cur, nxt[0]

    assert len(ordered) == len(cells), "snake traversal must visit every cell exactly once"
    return ordered


def _periodic_repair_candidates(seq: list[int]) -> list[dict]:
    n = len(seq)
    candidates = []
    for p in range(3, n):
        pattern = []
        valid = True
        mismatch_total = 0
        for residue in range(p):
            values = seq[residue:n:p]
            counts = Counter(values)
            best = counts.most_common()
            if len(best) > 1 and best[0][1] == best[1][1]:
                valid = False
                break
            color = int(best[0][0])
            pattern.append(color)
            mismatch_total += sum(cnt for val, cnt in counts.items() if int(val) != color)
            if mismatch_total > 1:
                valid = False
                break
        if not valid or mismatch_total != 1:
            continue
        if _primitive_period(pattern) != p:
            continue
        repaired = [pattern[i % p] for i in range(n)]
        diff = [i for i, (a, b) in enumerate(zip(seq, repaired)) if int(a) != int(b)]
        if len(diff) != 1:
            continue
        idx = diff[0]
        candidates.append(
            {
                "period": p,
                "index": idx,
                "color": int(repaired[idx]),
            }
        )
    return candidates


def _infer_solution(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) == 1, "expected exactly one foreground snake"

    mask = grid != bg
    path = _trace_snake(mask)
    seq = [int(grid[r, c]) for r, c in path]
    n = len(seq)

    forward = {
        (cand["period"], cand["index"], cand["color"])
        for cand in _periodic_repair_candidates(seq)
    }
    reverse = {
        (cand["period"], n - 1 - cand["index"], cand["color"])
        for cand in _periodic_repair_candidates(seq[::-1])
    }
    merged = forward & reverse
    assert len(merged) == 1, "period and defect must be uniquely recoverable from either endpoint"

    period, idx, color = next(iter(merged))
    r, c = path[idx]
    assert int(grid[r, c]) != int(color), "repaired color must change exactly one cell"
    return {
        "bg": bg,
        "path": path,
        "period": int(period),
        "index": int(idx),
        "coord": (int(r), int(c)),
        "color": int(color),
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Repair the unique off-period cell on a folded one-cell-wide snake path.

    Input structure:
    - A dominant background contains exactly one 4-connected multicolor snake.
    - The snake has no branches or loops and therefore has exactly two degree-1 endpoints.
    - Reading colors along the snake yields a sequence that should be a strict repetition of one
      shortest period of length at least 3, except for one wrongly colored cell.

    Rule:
    - Find the two endpoints and traverse the snake cell by cell.
    - Among all periods p >= 3, identify the unique shortest-period pattern for which changing
      exactly one snake cell makes the whole traversal sequence strictly periodic.
    - Return a grid of the same shape, changing only that one defective cell to the color required
      by the recovered period.

    Key insight:
    - The 2D bends hide the repetition completely; the periodic structure becomes visible only after
      straightening the snake into its 1D traversal order.
    """
    grid = np.array(input_grid, dtype=int, copy=True)
    solution = _infer_solution(grid)
    r, c = solution["coord"]
    grid[r, c] = solution["color"]
    return grid


def _count_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        d1 = (r1 - r0, c1 - c0)
        d2 = (r2 - r1, c2 - c1)
        turns += d1 != d2
    return turns


def _valid_next(
    cell: tuple[int, int],
    h: int,
    w: int,
    used: set[tuple[int, int]],
    prev: tuple[int, int],
) -> bool:
    r, c = cell
    if not (0 <= r < h and 0 <= c < w):
        return False
    if cell in used:
        return False
    touching = [(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in used]
    return len(touching) == 1 and touching[0] == prev


def _future_options(
    cell: tuple[int, int],
    h: int,
    w: int,
    used: set[tuple[int, int]],
    prev: tuple[int, int],
) -> int:
    local_used = set(used)
    local_used.add(cell)
    return sum(
        _valid_next((nr, nc), h, w, local_used, cell)
        for nr, nc in neighbors4(*cell)
        if (nr, nc) != prev
    )


def _build_snake_path(h: int, w: int, length: int, min_turns: int) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(starts)

    def dfs(path: list[tuple[int, int]], used: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(path) == length:
            return path[:] if _count_turns(path) >= min_turns else None

        cur = path[-1]
        candidates = []
        for nr, nc in neighbors4(*cur):
            nxt = (nr, nc)
            if not _valid_next(nxt, h, w, used, cur):
                continue
            score = _future_options(nxt, h, w, used, cur)
            candidates.append((score, random.random(), nxt))
        candidates.sort(reverse=True)

        for _score, _noise, nxt in candidates:
            path.append(nxt)
            used.add(nxt)
            result = dfs(path, used)
            if result is not None:
                return result
            used.remove(nxt)
            path.pop()
        return None

    for start in starts[: min(len(starts), 40)]:
        result = dfs([start], {start})
        if result is not None:
            return result
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    bg = random.choice(Color.ALL_COLORS)
    palette = [c for c in Color.NOT_BLACK if c != bg]
    if len(palette) < 6:
        palette = [c for c in Color.ALL_COLORS if c != bg]

    period = random.randint(3, 6)
    motif = random.sample(palette, period)
    repeats = random.randint(3, 5)
    tail = random.randint(0, period - 1)
    length = period * repeats + tail

    h = random.randint(max(7, min(14, length // 2)), min(24, length + 4))
    w = random.randint(max(7, min(14, length // 2)), min(24, length + 4))
    min_turns = random.choice([1, 2, 3, 4, 5])

    path = _build_snake_path(h, w, length, min_turns=min_turns)
    if path is None:
        raise ValueError("failed to build a valid snake path")

    seq = [motif[i % period] for i in range(length)]
    defect_index = random.randrange(length)
    correct_color = seq[defect_index]
    wrong_choices = [c for c in palette if c != correct_color]
    wrong_color = random.choice(wrong_choices)
    seq[defect_index] = wrong_color

    inp = np.full((h, w), bg, dtype=int)
    expected = inp.copy()
    for (r, c), color_in in zip(path, seq):
        inp[r, c] = int(color_in)
    for i, (r, c) in enumerate(path):
        expected[r, c] = int(motif[i % period])

    solver_meta = _infer_solution(inp)
    assert solver_meta["period"] == period
    assert solver_meta["coord"] == path[defect_index]
    assert solver_meta["color"] == correct_color
    assert np.array_equal(main(inp), expected)
    assert np.count_nonzero(inp != expected) == 1

    meta = {
        "bg": bg,
        "period": period,
        "motif": motif,
        "path": path,
        "defect_index": defect_index,
        "correct_color": correct_color,
        "wrong_color": wrong_color,
        "turns": _count_turns(path),
    }
    return inp, expected, meta


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, _expected, meta = _generate_case()
        except (AssertionError, ValueError):
            continue
        if meta["turns"] == 0:
            continue
        return inp
    raise RuntimeError("failed to generate a valid ARC input")
