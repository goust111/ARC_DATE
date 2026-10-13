from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIAGONALS: tuple[tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
BACKGROUND = 0
PALETTE: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)

HEIGHT_RANGE = (12, 23)
WIDTH_RANGE = (12, 23)
PAIR_COUNT_RANGE = (2, 4)
DIRECT_LENGTH_RANGE = (2, 8)
BOUNCE_MIN_STEPS = 5
BOUNCE_MAX_STEPS = 20
BOUNCE_MAX_COUNT = 4
PADDING_RADIUS = 1
GEN_ATTEMPTS = 300
PAIR_PLACEMENT_ATTEMPTS = 120


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != bg)


def _advance(shape: tuple[int, int], r: int, c: int, dr: int, dc: int) -> tuple[int, int, int, int, bool]:
    h, w = shape
    bounced = False
    if not (0 <= r + dr < h):
        dr = -dr
        bounced = True
    if not (0 <= c + dc < w):
        dc = -dc
        bounced = True
    return r + dr, c + dc, dr, dc, bounced


def _trace_path(
    grid: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
    direction: tuple[int, int],
    bg: int,
) -> tuple[list[tuple[int, int]], int] | None:
    h, w = grid.shape
    r, c = start
    dr, dc = direction
    states = {(r, c, dr, dc)}
    path = [start]
    bounce_count = 0

    for _ in range(4 * h * w + 1):
        r, c, dr, dc, bounced = _advance((h, w), r, c, dr, dc)
        if bounced:
            bounce_count += 1

        cell = (r, c)
        if cell == target:
            path.append(cell)
            return path, bounce_count

        if int(grid[r, c]) != bg:
            return None

        path.append(cell)
        state = (r, c, dr, dc)
        if state in states:
            return None
        states.add(state)

    return None


def _find_unique_path(
    grid: np.ndarray,
    start: tuple[int, int],
    target: tuple[int, int],
    bg: int,
) -> tuple[list[tuple[int, int]], int]:
    candidates: list[tuple[list[tuple[int, int]], int]] = []
    for direction in DIAGONALS:
        traced = _trace_path(grid, start, target, direction, bg)
        if traced is not None:
            candidates.append(traced)
    assert len(candidates) == 1, "each endpoint pair must admit exactly one valid initial diagonal direction"
    return candidates[0]


def _all_pair_paths(grid: np.ndarray) -> dict[int, tuple[list[tuple[int, int]], int]]:
    bg = _get_bg(grid)
    paths: dict[int, tuple[list[tuple[int, int]], int]] = {}
    for color in _colors_present(grid, bg):
        coords = np.argwhere(grid == color)
        assert len(coords) == 2, "each foreground color must appear exactly twice"
        endpoints = sorted((int(r), int(c)) for r, c in coords.tolist())
        paths[color] = _find_unique_path(grid, endpoints[0], endpoints[1], bg)
    return paths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair by painting the unique bouncing diagonal path between them.

    Input structure:
    - A dominant background surrounds several foreground colors.
    - Each foreground color appears exactly twice, defining one endpoint pair.
    - Some pairs lie on a direct 45-degree diagonal, while others can meet only by reflecting off the grid edges.

    Rule:
    - For each color, test the four diagonal launch directions from one endpoint.
    - Move one cell at a time on a 45-degree ray; when the next step would leave the grid, reflect the offending component and continue.
    - Exactly one launch direction reaches the partner endpoint before touching any other foreground pixel; paint every traversed background cell of that route with the pair color.

    Key insight:
    - The missing connection is a reversible diagonal billiards path, so the correct line is the unique reflected trajectory that lands on the matching endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    out = g.copy()
    for color, (path, _bounce_count) in _all_pair_paths(g).items():
        for r, c in path[1:-1]:
            assert int(out[r, c]) in (_get_bg(g), color), "paths must not cross other colored endpoints"
            out[r, c] = color
    return out


def _path_clear(blocked: np.ndarray, path: list[tuple[int, int]]) -> bool:
    return all(not bool(blocked[r, c]) for r, c in path)


def _paint_blocked(blocked: np.ndarray, path: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in path:
        for rr in range(max(0, r - PADDING_RADIUS), min(h, r + PADDING_RADIUS + 1)):
            for cc in range(max(0, c - PADDING_RADIUS), min(w, c + PADDING_RADIUS + 1)):
                blocked[rr, cc] = True


def _render_input(shape: tuple[int, int], pairs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for pair in pairs:
        color = int(pair["color"])
        path = pair["path"]
        grid[path[0]] = color
        grid[path[-1]] = color
    return grid


def _direct_candidates(shape: tuple[int, int], blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = shape
    candidates: list[list[tuple[int, int]]] = []
    max_len = min(DIRECT_LENGTH_RANGE[1], h - 1, w - 1)
    for length in range(DIRECT_LENGTH_RANGE[0], max_len + 1):
        for dc in (-1, 1):
            for r0 in range(h - length):
                if dc == 1:
                    c_values = range(w - length)
                else:
                    c_values = range(length, w)
                for c0 in c_values:
                    path = [(r0 + k, c0 + dc * k) for k in range(length + 1)]
                    if _path_clear(blocked, path):
                        candidates.append(path)
    return candidates


def _bounced_candidates(shape: tuple[int, int], blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = shape
    starts = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if not blocked[r, c]]
    random.shuffle(starts)
    candidates: list[list[tuple[int, int]]] = []

    for start in starts[: min(len(starts), 90)]:
        dirs = list(DIAGONALS)
        random.shuffle(dirs)
        for direction in dirs:
            r, c = start
            dr, dc = direction
            path = [start]
            seen_cells = {start}
            seen_states = {(r, c, dr, dc)}
            bounce_count = 0

            for _ in range(4 * h * w + 1):
                r, c, dr, dc, bounced = _advance(shape, r, c, dr, dc)
                if bounced:
                    bounce_count += 1
                cell = (r, c)
                if cell in seen_cells or blocked[r, c]:
                    break
                path.append(cell)
                seen_cells.add(cell)
                state = (r, c, dr, dc)
                if state in seen_states:
                    break
                seen_states.add(state)

                steps = len(path) - 1
                if (
                    bounce_count >= 1
                    and BOUNCE_MIN_STEPS <= steps <= BOUNCE_MAX_STEPS
                    and bounce_count <= BOUNCE_MAX_COUNT
                ):
                    candidates.append(path.copy())

                if steps >= BOUNCE_MAX_STEPS:
                    break

    return candidates


def _pick_pair_path(shape: tuple[int, int], blocked: np.ndarray, pair_type: str) -> list[tuple[int, int]] | None:
    candidates = _bounced_candidates(shape, blocked) if pair_type == "bounce" else _direct_candidates(shape, blocked)
    if not candidates:
        return None

    random.shuffle(candidates)
    if pair_type == "bounce":
        candidates.sort(key=len, reverse=True)
        shortlist = candidates[: min(30, len(candidates))]
        return random.choice(shortlist)
    candidates.sort(key=len)
    shortlist = candidates[: min(50, len(candidates))]
    return random.choice(shortlist)


def _validate_grid(inp: np.ndarray) -> tuple[np.ndarray, dict[int, tuple[list[tuple[int, int]], int]]] | None:
    try:
        paths = _all_pair_paths(inp)
        out = main(inp)
    except Exception:
        return None

    bounce_counts = [bounce_count for _path, bounce_count in paths.values()]
    if not any(count == 0 for count in bounce_counts):
        return None
    if not any(count >= 1 for count in bounce_counts):
        return None
    return out, paths


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, pair_count)

    pair_types = ["direct", "bounce"]
    while len(pair_types) < pair_count:
        pair_types.append(random.choice(["direct", "bounce"]))
    random.shuffle(pair_types)
    pair_types.sort(key=lambda kind: kind == "direct")

    blocked = np.zeros((h, w), dtype=bool)
    pairs: list[dict[str, object]] = []
    for color, pair_type in zip(colors, pair_types):
        path = None
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            path = _pick_pair_path((h, w), blocked, pair_type)
            if path is not None:
                break
        if path is None:
            return None
        pairs.append({"color": color, "path": path, "kind": pair_type})
        _paint_blocked(blocked, path)

    inp = _render_input((h, w), pairs)
    validated = _validate_grid(inp)
    if validated is None:
        return None

    out, paths = validated
    expected = inp.copy()
    for color, (path, _bounce_count) in paths.items():
        for r, c in path[1:-1]:
            expected[r, c] = color
    if not np.array_equal(out, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bouncing-diagonal ARC input")
