from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import default_rng


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
MAROON = 9

SOURCE_COLORS = frozenset((RED, BLUE, YELLOW, GREEN))
PRIMARY_BITS = {
    RED: 1,
    BLUE: 2,
    YELLOW: 4,
    GREEN: 2 | 4,
    MAGENTA: 1 | 2,
    ORANGE: 1 | 4,
    MAROON: 1 | 2 | 4,
}
BITS_TO_COLOR = {
    1: RED,
    2: BLUE,
    4: YELLOW,
    1 | 2: MAGENTA,
    1 | 4: ORANGE,
    2 | 4: GREEN,
    1 | 2 | 4: MAROON,
}

TARGET_BRANCHES = ("red_blue", "red_yellow", "blue_yellow", "red_green")
SUPPORTED_EVENTS = set(TARGET_BRANCHES) | {"same"}
GENERATE_ATTEMPTS = 300
EXTRA_ATTEMPTS = 24
MIN_PATH_LEN = 3


@dataclass(frozen=True)
class PathSpec:
    axis: str
    line: int
    start: int
    end: int
    direction: int
    color: int


@dataclass(frozen=True)
class SolveResult:
    output: np.ndarray
    events: tuple[str, ...]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each source color along grey paths and mix only at orthogonal crossings before painting the downstream terminals.

    Input structure:
    - The background is black.
    - Grey pixels form one-cell-wide straight horizontal and vertical paths; some path pixels are shared by one horizontal and one vertical path, creating crossings.
    - Each path has exactly one colored source pixel adjacent to one endpoint and a plain grey terminal at the opposite endpoint.
    - Source colors are red, blue, yellow, or green; no mixed colors appear in the input.

    Rule:
    - Infer every path's direction from the endpoint that touches its source pixel, then propagate that source color along the path.
    - At a crossing, combine the upstream horizontal and vertical colors: red+blue->magenta, red+yellow->orange, blue+yellow->green, and red+green->maroon; identical colors stay unchanged.
    - The crossing's resulting color continues along both downstream directions, so later crossings receive the already-mixed upstream color.
    - Recolor every grey path pixel to the propagated result while leaving the source pixels themselves unchanged.

    Key insight:
    - The network is not an undirected flood fill: each straight path has a source-directed flow, and a crossing only depends on the colors arriving from its two upstream path segments.
    """
    return _solve(np.asarray(grid, dtype=int)).output


def _solve(grid: np.ndarray) -> SolveResult:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    unique_colors = set(map(int, np.unique(g)))
    assert unique_colors.issubset(SOURCE_COLORS | {BLACK, GREY}), "unexpected input colors"
    grey_mask = g == GREY
    source_mask = np.isin(g, tuple(SOURCE_COLORS))
    assert np.any(grey_mask), "expected at least one grey path"
    assert np.any(source_mask), "expected at least one source"

    paths, used_sources = _extract_paths(g)
    source_coords = {tuple(map(int, coord)) for coord in np.argwhere(source_mask)}
    assert source_coords == used_sources, "every source must label exactly one path endpoint"

    coords = np.argwhere(grey_mask)
    count = int(coords.shape[0])
    index = -np.ones(g.shape, dtype=int)
    for idx, (r, c) in enumerate(coords):
        index[r, c] = idx

    h_pred = np.full(count, -1, dtype=int)
    v_pred = np.full(count, -1, dtype=int)
    h_src = np.zeros(count, dtype=np.uint8)
    v_src = np.zeros(count, dtype=np.uint8)
    has_h = np.zeros(count, dtype=bool)
    has_v = np.zeros(count, dtype=bool)

    for spec in paths:
        bits = np.uint8(PRIMARY_BITS[spec.color])
        if spec.axis == "h":
            row = spec.line
            for col in range(spec.start, spec.end + 1):
                has_h[index[row, col]] = True
            if spec.direction == 1:
                start_idx = index[row, spec.start]
                h_src[start_idx] |= bits
                for col in range(spec.start + 1, spec.end + 1):
                    cur = index[row, col]
                    assert h_pred[cur] == -1, "horizontal predecessor collision"
                    h_pred[cur] = index[row, col - 1]
            else:
                start_idx = index[row, spec.end]
                h_src[start_idx] |= bits
                for col in range(spec.end - 1, spec.start - 1, -1):
                    cur = index[row, col]
                    assert h_pred[cur] == -1, "horizontal predecessor collision"
                    h_pred[cur] = index[row, col + 1]
        else:
            col = spec.line
            for row in range(spec.start, spec.end + 1):
                has_v[index[row, col]] = True
            if spec.direction == 1:
                start_idx = index[spec.start, col]
                v_src[start_idx] |= bits
                for row in range(spec.start + 1, spec.end + 1):
                    cur = index[row, col]
                    assert v_pred[cur] == -1, "vertical predecessor collision"
                    v_pred[cur] = index[row - 1, col]
            else:
                start_idx = index[spec.end, col]
                v_src[start_idx] |= bits
                for row in range(spec.end - 1, spec.start - 1, -1):
                    cur = index[row, col]
                    assert v_pred[cur] == -1, "vertical predecessor collision"
                    v_pred[cur] = index[row + 1, col]

    assert np.all(has_h | has_v), "every grey pixel must belong to at least one path"

    # Colors only gain primary components, so monotone iteration reaches the unique fixed point quickly.
    states = (h_src | v_src).copy()
    for _ in range(3 * count + 1):
        changed = False
        for idx in range(count):
            new_bits = int(h_src[idx] | v_src[idx])
            if h_pred[idx] != -1:
                new_bits |= int(states[h_pred[idx]])
            if v_pred[idx] != -1:
                new_bits |= int(states[v_pred[idx]])
            if new_bits != int(states[idx]):
                states[idx] = np.uint8(new_bits)
                changed = True
        if not changed:
            break
    else:
        raise AssertionError("signal propagation did not converge")

    assert np.all(states > 0), "every grey pixel must be reachable from a source"

    out = g.copy()
    events: list[str] = []
    for idx, (r, c) in enumerate(coords):
        bits = int(states[idx])
        assert bits in BITS_TO_COLOR, f"unsupported mixed color bits: {bits}"
        out[r, c] = BITS_TO_COLOR[bits]

        if not (has_h[idx] and has_v[idx]):
            continue
        h_in = int(states[h_pred[idx]]) if h_pred[idx] != -1 else int(h_src[idx])
        v_in = int(states[v_pred[idx]]) if v_pred[idx] != -1 else int(v_src[idx])
        events.append(_classify_mix(h_in, v_in))

    return SolveResult(output=out, events=tuple(events))


def _extract_paths(grid: np.ndarray) -> tuple[list[PathSpec], set[tuple[int, int]]]:
    h, w = grid.shape
    grey = grid == GREY
    paths: list[PathSpec] = []
    used_sources: set[tuple[int, int]] = set()

    for row in range(h):
        for start, end in _true_runs(grey[row, :]):
            left = int(grid[row, start - 1]) if start > 0 else BLACK
            right = int(grid[row, end + 1]) if end + 1 < w else BLACK
            left_src = left in SOURCE_COLORS
            right_src = right in SOURCE_COLORS
            if left_src ^ right_src:
                if left_src:
                    paths.append(PathSpec("h", row, start, end, 1, left))
                    used_sources.add((row, start - 1))
                else:
                    paths.append(PathSpec("h", row, start, end, -1, right))
                    used_sources.add((row, end + 1))

    for col in range(w):
        for start, end in _true_runs(grey[:, col]):
            top = int(grid[start - 1, col]) if start > 0 else BLACK
            bottom = int(grid[end + 1, col]) if end + 1 < h else BLACK
            top_src = top in SOURCE_COLORS
            bottom_src = bottom in SOURCE_COLORS
            if top_src ^ bottom_src:
                if top_src:
                    paths.append(PathSpec("v", col, start, end, 1, top))
                    used_sources.add((start - 1, col))
                else:
                    paths.append(PathSpec("v", col, start, end, -1, bottom))
                    used_sources.add((end + 1, col))

    return paths, used_sources


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _classify_mix(a_bits: int, b_bits: int) -> str:
    if a_bits == b_bits:
        return "same"
    pair = frozenset((a_bits, b_bits))
    if pair == frozenset((PRIMARY_BITS[RED], PRIMARY_BITS[BLUE])):
        return "red_blue"
    if pair == frozenset((PRIMARY_BITS[RED], PRIMARY_BITS[YELLOW])):
        return "red_yellow"
    if pair == frozenset((PRIMARY_BITS[BLUE], PRIMARY_BITS[YELLOW])):
        return "blue_yellow"
    if pair == frozenset((PRIMARY_BITS[RED], PRIMARY_BITS[GREEN])):
        return "red_green"
    return "unsupported"


def generate_input() -> np.ndarray:
    rng = default_rng()
    for _ in range(GENERATE_ATTEMPTS):
        branch = str(rng.choice(TARGET_BRANCHES))
        grid = _gen_attempt(rng, branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_ATTEMPTS} attempts")


def _gen_attempt(rng: np.random.Generator, branch: str) -> np.ndarray | None:
    hdir = int(rng.choice((-1, 1)))
    vdir = int(rng.choice((-1, 1)))
    height = int(rng.integers(10, 25))
    width = int(rng.integers(10, 25))
    shape = (height, width)

    core_specs = _build_core_specs(shape, branch, hdir, vdir, rng)
    if core_specs is None:
        return None

    grid = np.full(shape, BLACK, dtype=int)
    if not _can_place_specs(grid, core_specs):
        return None
    _paint_specs(grid, core_specs)

    target_extras = int(rng.integers(0, 3))
    added = 0
    for _ in range(EXTRA_ATTEMPTS):
        if added >= target_extras:
            break
        if rng.random() < 0.35:
            specs = _sample_same_color_cross(shape, hdir, vdir, rng)
        else:
            specs = _sample_isolated_path(shape, hdir, vdir, rng)
        if specs is None or not _can_place_specs(grid, specs):
            continue
        _paint_specs(grid, specs)
        added += 1

    if _is_valid_generated_case(grid, branch):
        return grid
    return None


def _build_core_specs(
    shape: tuple[int, int],
    branch: str,
    hdir: int,
    vdir: int,
    rng: np.random.Generator,
) -> list[PathSpec] | None:
    height, width = shape
    row_candidates = list(range(3, height - 3, 2))
    col_candidates = list(range(3, width - 3, 2))
    if not row_candidates or not col_candidates:
        return None

    if branch == "red_green":
        if rng.random() < 0.5:
            if len(row_candidates) < 2:
                return None
            rows = sorted(int(x) for x in rng.choice(row_candidates, size=2, replace=False))
            col = int(rng.choice(col_candidates))
            upstream = rows[0] if vdir == 1 else rows[-1]
            other = rows[-1] if upstream == rows[0] else rows[0]
            h_rows = rows
            h_colors = [YELLOW if row == upstream else RED for row in rows]
            v_cols = [col]
            v_colors = [BLUE]
        else:
            if len(col_candidates) < 2:
                return None
            cols = sorted(int(x) for x in rng.choice(col_candidates, size=2, replace=False))
            row = int(rng.choice(row_candidates))
            upstream = cols[0] if hdir == 1 else cols[-1]
            h_rows = [row]
            h_colors = [BLUE]
            v_cols = cols
            v_colors = [YELLOW if col == upstream else RED for col in cols]
    else:
        pairs = {
            "red_blue": (RED, BLUE),
            "red_yellow": (RED, YELLOW),
            "blue_yellow": (BLUE, YELLOW),
        }
        color_a, color_b = pairs[branch]
        row = int(rng.choice(row_candidates))
        col = int(rng.choice(col_candidates))
        if rng.random() < 0.5:
            h_rows, h_colors = [row], [color_a]
            v_cols, v_colors = [col], [color_b]
        else:
            h_rows, h_colors = [row], [color_b]
            v_cols, v_colors = [col], [color_a]

    return _cross_specs(shape, hdir, vdir, h_rows, h_colors, v_cols, v_colors, rng)


def _cross_specs(
    shape: tuple[int, int],
    hdir: int,
    vdir: int,
    h_rows: list[int],
    h_colors: list[int],
    v_cols: list[int],
    v_colors: list[int],
    rng: np.random.Generator,
) -> list[PathSpec] | None:
    height, width = shape
    min_col, max_col = min(v_cols), max(v_cols)
    min_row, max_row = min(h_rows), max(h_rows)

    left_limit = min_col - 1
    right_limit = width - 2 - max_col
    top_limit = min_row - 1
    bottom_limit = height - 2 - max_row
    if min(left_limit, right_limit, top_limit, bottom_limit) < 1:
        return None

    specs: list[PathSpec] = []
    for row, color in zip(h_rows, h_colors):
        left_buffer = int(rng.integers(1, min(4, left_limit) + 1))
        right_buffer = int(rng.integers(1, min(4, right_limit) + 1))
        specs.append(
            PathSpec(
                axis="h",
                line=int(row),
                start=min_col - left_buffer,
                end=max_col + right_buffer,
                direction=hdir,
                color=int(color),
            )
        )
    for col, color in zip(v_cols, v_colors):
        top_buffer = int(rng.integers(1, min(4, top_limit) + 1))
        bottom_buffer = int(rng.integers(1, min(4, bottom_limit) + 1))
        specs.append(
            PathSpec(
                axis="v",
                line=int(col),
                start=min_row - top_buffer,
                end=max_row + bottom_buffer,
                direction=vdir,
                color=int(color),
            )
        )
    return specs


def _sample_isolated_path(
    shape: tuple[int, int],
    hdir: int,
    vdir: int,
    rng: np.random.Generator,
) -> list[PathSpec] | None:
    height, width = shape
    axis = "h" if rng.random() < 0.5 else "v"
    if axis == "h":
        max_len = min(8, width - 2)
        if max_len < MIN_PATH_LEN:
            return None
        length = int(rng.integers(MIN_PATH_LEN, max_len + 1))
        row = int(rng.integers(1, height - 1))
        start = int(rng.integers(1, width - length))
        end = start + length - 1
        color = _random_source_color(rng)
        return [PathSpec("h", row, start, end, hdir, color)]

    max_len = min(8, height - 2)
    if max_len < MIN_PATH_LEN:
        return None
    length = int(rng.integers(MIN_PATH_LEN, max_len + 1))
    col = int(rng.integers(1, width - 1))
    start = int(rng.integers(1, height - length))
    end = start + length - 1
    color = _random_source_color(rng)
    return [PathSpec("v", col, start, end, vdir, color)]


def _sample_same_color_cross(
    shape: tuple[int, int],
    hdir: int,
    vdir: int,
    rng: np.random.Generator,
) -> list[PathSpec] | None:
    height, width = shape
    row_candidates = list(range(2, height - 2))
    col_candidates = list(range(2, width - 2))
    if not row_candidates or not col_candidates:
        return None
    row = int(rng.choice(row_candidates))
    col = int(rng.choice(col_candidates))
    color = _random_source_color(rng)
    return _cross_specs(shape, hdir, vdir, [row], [color], [col], [color], rng)


def _random_source_color(rng: np.random.Generator) -> int:
    pool = (RED, BLUE, YELLOW, GREEN)
    weights = np.array((3, 3, 3, 1), dtype=float)
    return int(rng.choice(pool, p=weights / weights.sum()))


def _can_place_specs(grid: np.ndarray, specs: list[PathSpec]) -> bool:
    grey_cells: dict[tuple[int, int], int] = {}
    for spec in specs:
        for cell in _path_cells(spec):
            grey_cells[cell] = grey_cells.get(cell, 0) + 1
            if grey_cells[cell] > 2:
                return False

    for (r, c), _count in grey_cells.items():
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BLACK:
            return False

    source_coords = [_source_coord(spec) for spec in specs]
    if len(set(source_coords)) != len(source_coords):
        return False
    for r, c in source_coords:
        if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BLACK or (r, c) in grey_cells:
            return False

    occupied = set(grey_cells) | set(source_coords)
    for spec in specs:
        tr, tc = _terminal_outside_coord(spec)
        if _in_bounds(grid.shape, tr, tc) and ((tr, tc) in occupied or int(grid[tr, tc]) != BLACK):
            return False
    return True


def _paint_specs(grid: np.ndarray, specs: list[PathSpec]) -> None:
    for spec in specs:
        for r, c in _path_cells(spec):
            grid[r, c] = GREY
    for spec in specs:
        sr, sc = _source_coord(spec)
        grid[sr, sc] = spec.color


def _path_cells(spec: PathSpec) -> list[tuple[int, int]]:
    if spec.axis == "h":
        return [(spec.line, col) for col in range(spec.start, spec.end + 1)]
    return [(row, spec.line) for row in range(spec.start, spec.end + 1)]


def _source_coord(spec: PathSpec) -> tuple[int, int]:
    if spec.axis == "h":
        if spec.direction == 1:
            return spec.line, spec.start - 1
        return spec.line, spec.end + 1
    if spec.direction == 1:
        return spec.start - 1, spec.line
    return spec.end + 1, spec.line


def _terminal_outside_coord(spec: PathSpec) -> tuple[int, int]:
    if spec.axis == "h":
        if spec.direction == 1:
            return spec.line, spec.end + 1
        return spec.line, spec.start - 1
    if spec.direction == 1:
        return spec.end + 1, spec.line
    return spec.start - 1, spec.line


def _in_bounds(shape: tuple[int, int], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def _is_valid_generated_case(grid: np.ndarray, branch: str) -> bool:
    try:
        result = _solve(grid)
    except AssertionError:
        return False

    if branch not in result.events:
        return False
    if branch == "red_green" and "blue_yellow" not in result.events:
        return False
    if any(event not in SUPPORTED_EVENTS for event in result.events):
        return False

    source_coords = np.argwhere(np.isin(grid, tuple(SOURCE_COLORS)))
    for row, col in source_coords:
        grey_neighbors = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = int(row + dr), int(col + dc)
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
                grey_neighbors += 1
        if grey_neighbors != 1:
            return False
    return True
