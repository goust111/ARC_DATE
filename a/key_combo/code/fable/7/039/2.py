"""ARC task: decode a color-to-direction key from an executed path prefix."""

from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _instruction_data(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    """Find the unique non-background solid separator and the run above it."""
    h, _ = grid.shape
    separators = []
    for r in range(1, h - 1):
        if np.all(grid[r] == grid[r, 0]) and int(grid[r, 0]) != bg:
            separators.append(r)
    assert len(separators) == 1, "expected one solid non-background separator row"
    sep = separators[0]

    cols = np.flatnonzero(grid[sep - 1] != bg)
    assert len(cols) > 0, "instruction strip must be nonempty"
    assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)), (
        "instruction cells must form one horizontal run"
    )
    assert np.all(grid[: sep - 1] == bg), "only the strip may occur above the separator"
    codes = [int(v) for v in grid[sep - 1, cols[0] : cols[-1] + 1]]
    assert 3 <= len(set(codes)) <= 4
    return sep, codes


def _ordered_prefix(
    grid: np.ndarray, sep: int, bg: int
) -> tuple[int, int, list[tuple[int, int]]]:
    """Return start/trail colors and the uniquely ordered executed coordinates."""
    main_area = grid[sep + 1 :]
    colors = [int(v) for v in np.unique(main_area) if int(v) != bg]
    assert len(colors) == 2, "main area must contain only a start and a trail"
    counts = {color: int(np.count_nonzero(main_area == color)) for color in colors}
    starts = [color for color in colors if counts[color] == 1]
    assert len(starts) == 1, "the start color must be a unique singleton"
    start_color = starts[0]
    trail_color = next(color for color in colors if color != start_color)

    sr0, sc = map(int, np.argwhere(main_area == start_color)[0])
    start = (sr0 + sep + 1, sc)
    trail = {tuple(map(int, rc)) for rc in np.argwhere(grid == trail_color)}
    assert trail, "executed trail must be nonempty"

    ordered = [start]
    remaining = set(trail)
    while remaining:
        r, c = ordered[-1]
        candidates = [
            (r + dr, c + dc)
            for dr, dc in DIRS
            if (r + dr, c + dc) in remaining
        ]
        assert len(candidates) == 1, "executed trail must have a unique next step"
        nxt = candidates[0]
        ordered.append(nxt)
        remaining.remove(nxt)

    # This also rejects branches and nonconsecutive side contacts.
    for i, (r, c) in enumerate(ordered):
        adjacent = {
            (r + dr, c + dc)
            for dr, dc in DIRS
            if (r + dr, c + dc) in set(ordered)
        }
        expected = set(ordered[max(0, i - 1) : i]) | set(ordered[i + 1 : i + 2])
        assert adjacent == expected, "visible path must be an induced simple chain"

    return start_color, trail_color, ordered


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the instruction colors from an executed path prefix and finish the path.

    Input structure:
    - A horizontal multicolor instruction strip sits immediately above a solid separator row.
    - Below it, a uniquely colored start pixel is joined to a one-color executed trail.
    - The trail contains one step for each instruction in a proper prefix of the strip.

    Rule:
    - Align successive trail moves with the same-length instruction prefix to recover
      the color-to-cardinal-direction mapping, then follow all remaining instructions
      from the trail endpoint, painting each reached background cell in the trail color.
    - Preserve the start, executed trail, instruction strip, separator, and all dimensions.

    Key insight:
    - The drawn prefix is both a partial result and the complete decoding key: every
      instruction color appears in it, so the sample-specific mapping is uniquely known.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _background(g)
    sep, codes = _instruction_data(g, bg)
    _start_color, trail_color, path = _ordered_prefix(g, sep, bg)

    prefix_len = len(path) - 1
    assert 0 < prefix_len < len(codes), "trail must execute a proper instruction prefix"

    mapping: dict[int, tuple[int, int]] = {}
    for code, before, after in zip(codes[:prefix_len], path, path[1:]):
        direction = (after[0] - before[0], after[1] - before[1])
        assert direction in DIRS
        if code in mapping:
            assert mapping[code] == direction, "one code cannot mean two directions"
        else:
            assert direction not in mapping.values(), "direction codes must be one-to-one"
            mapping[code] = direction
    assert set(mapping) == set(codes), "executed prefix must expose every code"

    out = g.copy()
    r, c = path[-1]
    for code in codes[prefix_len:]:
        assert code in mapping
        dr, dc = mapping[code]
        r, c = r + dr, c + dc
        assert sep < r < g.shape[0] and 0 <= c < g.shape[1], "path leaves main area"
        assert int(g[r, c]) == bg, "unexecuted path must cross fresh background"
        out[r, c] = trail_color
    return out


def _random_induced_path(
    height: int, width: int, length: int, allowed: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]] | None:
    """Make a self-avoiding path with no nonconsecutive orthogonal contacts."""
    start = (random.randrange(height), random.randrange(width))
    path = [start]
    occupied = {start}

    def extend() -> bool:
        if len(path) == length + 1:
            return True
        r, c = path[-1]
        moves = list(allowed)
        random.shuffle(moves)
        for dr, dc in moves:
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width) or nxt in occupied:
                continue
            touching = sum(
                (nxt[0] + ar, nxt[1] + ac) in occupied for ar, ac in DIRS
            )
            if touching != 1:
                continue
            occupied.add(nxt)
            path.append(nxt)
            if extend():
                return True
            path.pop()
            occupied.remove(nxt)
        return False

    return path if extend() else None


def _generate_attempt() -> np.ndarray | None:
    code_count = random.choice((3, 4))
    total_steps = random.randint(max(12, 2 * code_count + 3), 20)
    width = random.randint(total_steps + 2, 30)
    main_height = random.randint(12, 25)
    height = main_height + 2

    allowed = tuple(random.sample(DIRS, code_count))
    path = _random_induced_path(main_height, width, total_steps, allowed)
    if path is None:
        return None
    directions = [
        (b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])
    ]
    prefix_options = [
        p
        for p in range(code_count, total_steps - code_count + 1)
        if set(directions[:p]) == set(allowed)
        and set(directions[p:]) == set(allowed)
        and p >= 5
        and total_steps - p >= 5
    ]
    if not prefix_options:
        return None
    prefix_len = random.choice(prefix_options)

    bg, separator, start_color, trail_color, *code_colors = random.sample(
        range(10), 4 + code_count
    )
    direction_to_code = dict(zip(random.sample(allowed, code_count), code_colors))
    codes = [direction_to_code[d] for d in directions]

    grid = np.full((height, width), bg, dtype=int)
    strip_left = random.randint(1, width - total_steps - 1)
    grid[0, strip_left : strip_left + total_steps] = codes
    grid[1, :] = separator

    global_path = [(r + 2, c) for r, c in path]
    grid[global_path[0]] = start_color
    for point in global_path[1 : prefix_len + 1]:
        grid[point] = trail_color

    # Independent semantic checks before accepting the sample.
    assert set(codes[:prefix_len]) == set(code_colors)
    assert set(codes[prefix_len:]) == set(code_colors)
    assert len(set(global_path)) == total_steps + 1
    for i, point in enumerate(global_path):
        neighbors = {
            (point[0] + dr, point[1] + dc)
            for dr, dc in DIRS
            if (point[0] + dr, point[1] + dc) in set(global_path)
        }
        expected = set(global_path[max(0, i - 1) : i]) | set(global_path[i + 1 : i + 2])
        assert neighbors == expected

    expected = grid.copy()
    for point in global_path[prefix_len + 1 :]:
        assert expected[point] == bg
        expected[point] = trail_color
    solved = main(grid)
    assert np.array_equal(solved, expected), "decoder disagrees with construction oracle"
    assert np.count_nonzero(solved != grid) == total_steps - prefix_len >= 5
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, semantically checked instances of the decoding task."""
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-decoding instruction strip")
