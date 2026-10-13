import itertools

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

ROUTE_COLORS = (BLUE, RED, GREEN, YELLOW)
PAIR_OPTIONS = tuple(
    pair
    for pair in itertools.combinations(((r, c) for r in range(3) for c in range(3)), 2)
    if pair[0][0] != pair[1][0] and pair[0][1] != pair[1][1]
)
GENERATION_RETRY_LIMIT = 400


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a separator-delimited region grid, then draw same-color orthogonal routes in the compact space with magenta cross-color conflicts.

    Input structure:
    - The grid contains exactly two full-width horizontal separator lines and two full-height vertical separator lines, all orange.
    - These separators divide the canvas into 3x3 rectangular regions.
    - Every non-separator region is uniformly filled with one color, where black denotes empty compact pixels and each non-black route color appears exactly twice.

    Rule:
    - Remove the separator lines by replacing each rectangular region with one compact pixel of that region's fill color, preserving row and column order to obtain a 3x3 grid.
    - For each non-black color pair in the compact grid, choose the upper endpoint as the one with smaller row index, breaking ties by smaller column index, then route horizontally to the partner's column and vertically to the partner's row.
    - Route claims only affect compact pixels that were black in the compressed grid; single-color claims paint that color, while black pixels claimed by multiple different colors become magenta. Existing non-black endpoints are never overwritten.

    Key insight:
    - The visible large grid is only a separator-expanded encoding of a hidden 3x3 topology; all routing and crossing logic happens after compression in that compact coordinate system.
    """
    compact = _compress_regions(np.asarray(grid, dtype=int))
    return _draw_compact_routes(compact)


def _compress_regions(grid: np.ndarray) -> np.ndarray:
    if grid.ndim != 2:
        raise AssertionError("input must be a 2D grid")

    h, w = map(int, grid.shape)
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == ORANGE)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == ORANGE)]
    if len(sep_rows) != 2 or len(sep_cols) != 2:
        raise AssertionError("expected exactly two horizontal and two vertical orange separators")

    row_edges = [-1, *sep_rows, h]
    col_edges = [-1, *sep_cols, w]
    compact = np.zeros((3, 3), dtype=int)

    for i in range(3):
        r0, r1 = row_edges[i] + 1, row_edges[i + 1]
        if r0 >= r1:
            raise AssertionError("every compressed row band must have positive height")
        for j in range(3):
            c0, c1 = col_edges[j] + 1, col_edges[j + 1]
            if c0 >= c1:
                raise AssertionError("every compressed column band must have positive width")
            block = grid[r0:r1, c0:c1]
            values = np.unique(block)
            if len(values) != 1 or int(values[0]) == ORANGE:
                raise AssertionError("each non-separator region must be uniformly colored and non-orange")
            compact[i, j] = int(values[0])

    return compact


def _route_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    start, end = sorted((tuple(a), tuple(b)))
    r0, c0 = start
    r1, c1 = end
    if r0 == r1 or c0 == c1:
        raise AssertionError("same-color endpoints must differ in both row and column")

    cells: list[tuple[int, int]] = []
    step_c = 1 if c1 > c0 else -1
    for col in range(c0 + step_c, c1 + step_c, step_c):
        cells.append((r0, col))

    step_r = 1 if r1 > r0 else -1
    for row in range(r0 + step_r, r1 + step_r, step_r):
        cells.append((row, c1))
    return cells


def _analyze_compact(compact: np.ndarray) -> dict:
    colors = sorted(int(color) for color in np.unique(compact) if int(color) != BLACK)
    claims: dict[tuple[int, int], set[int]] = {}
    endpoint_block = False

    for color in colors:
        points = [tuple(pos) for pos in np.argwhere(compact == color)]
        if len(points) != 2:
            raise AssertionError("each non-black compact color must appear exactly twice")
        if points[0][0] == points[1][0] or points[0][1] == points[1][1]:
            raise AssertionError("same-color compact endpoints must not share a row or column")

        for cell in _route_cells(points[0], points[1]):
            if compact[cell] == BLACK:
                claims.setdefault(cell, set()).add(color)
            elif compact[cell] != color:
                endpoint_block = True

    out = compact.copy()
    pink_count = 0
    for cell, route_colors in claims.items():
        if len(route_colors) == 1:
            out[cell] = next(iter(route_colors))
        else:
            out[cell] = PINK
            pink_count += 1

    return {
        "output": out,
        "num_pairs": len(colors),
        "pink_count": pink_count,
        "endpoint_block": endpoint_block,
        "changed_count": int(np.count_nonzero(out != compact)),
        "black_count": int(np.count_nonzero(compact == BLACK)),
    }


def _draw_compact_routes(compact: np.ndarray) -> np.ndarray:
    return _analyze_compact(compact)["output"]


def _canonicalize(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid)
    mapping: dict[int, int] = {}
    next_color = 1
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            if value == 0:
                continue
            if value not in mapping:
                mapping[value] = next_color
                next_color += 1
            out[r, c] = mapping[value]
    return out


def _build_patterns() -> list[dict]:
    patterns: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    for num_pairs in (2, 3):
        for pair_combo in itertools.combinations(PAIR_OPTIONS, num_pairs):
            used = [cell for pair in pair_combo for cell in pair]
            if len(set(used)) != 2 * num_pairs:
                continue

            grid = np.zeros((3, 3), dtype=int)
            for color, pair in enumerate(pair_combo, start=1):
                for r, c in pair:
                    grid[r, c] = color

            grid = _canonicalize(grid)
            key = tuple(int(x) for x in grid.flat)
            if key in seen:
                continue

            info = _analyze_compact(grid)
            if info["changed_count"] < 2:
                continue

            seen.add(key)
            patterns.append(
                {
                    "compact": grid,
                    "num_pairs": info["num_pairs"],
                    "pink_count": info["pink_count"],
                    "endpoint_block": info["endpoint_block"],
                    "changed_count": info["changed_count"],
                }
            )

    return patterns


PATTERNS = _build_patterns()
PATTERN_GROUPS = {
    "two_plain": [p for p in PATTERNS if p["num_pairs"] == 2 and p["pink_count"] == 0 and not p["endpoint_block"]],
    "two_overlap": [p for p in PATTERNS if p["num_pairs"] == 2 and p["pink_count"] >= 1],
    "three_plain": [p for p in PATTERNS if p["num_pairs"] == 3 and p["pink_count"] == 0],
    "three_overlap": [p for p in PATTERNS if p["num_pairs"] == 3 and p["pink_count"] >= 1],
}
MODE_NAMES = ("two_plain", "two_overlap", "three_plain", "three_overlap")
MODE_WEIGHTS = np.array((0.15, 0.30, 0.25, 0.30), dtype=float)


def _paint_separator_grid(compact: np.ndarray, row_sizes: np.ndarray, col_sizes: np.ndarray) -> np.ndarray:
    height = int(np.sum(row_sizes) + 2)
    width = int(np.sum(col_sizes) + 2)
    grid = np.full((height, width), ORANGE, dtype=int)

    row = 0
    for i, block_h in enumerate(row_sizes):
        col = 0
        for j, block_w in enumerate(col_sizes):
            grid[row : row + block_h, col : col + block_w] = int(compact[i, j])
            col += int(block_w)
            if j < 2:
                col += 1
        row += int(block_h)
        if i < 2:
            row += 1

    return grid


def _recolor_pattern(abstract: np.ndarray, actual_colors: tuple[int, ...]) -> np.ndarray:
    compact = np.zeros_like(abstract)
    for color_id, actual in enumerate(actual_colors, start=1):
        compact[abstract == color_id] = actual
    return compact


def _mode_ok(mode: str, compact: np.ndarray) -> None:
    info = _analyze_compact(compact)
    if info["changed_count"] < 2:
        raise AssertionError("generated compact sample is too trivial")
    if info["black_count"] not in {3, 5}:
        raise AssertionError("compact parity must leave 3 or 5 black cells")
    if mode.endswith("overlap") and info["pink_count"] == 0:
        raise AssertionError("overlap mode must contain at least one magenta conflict")
    if mode.endswith("plain") and info["pink_count"] != 0:
        raise AssertionError("plain mode must avoid magenta conflicts")
    if mode.startswith("three") and not info["endpoint_block"]:
        raise AssertionError("three-pair mode should expose the endpoint-preservation branch")
    if mode == "two_plain" and info["endpoint_block"]:
        raise AssertionError("two_plain should remain free of endpoint blocking")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    # With 9 compact cells and every active color appearing exactly twice, parity
    # forces 2 or 3 color pairs, hence 5 or 3 black cells in the compressed grid.
    for _ in range(GENERATION_RETRY_LIMIT):
        mode = rng.choice(MODE_NAMES, p=MODE_WEIGHTS)
        pattern = rng.choice(PATTERN_GROUPS[mode])
        actual_colors = tuple(rng.permutation(ROUTE_COLORS)[: pattern["num_pairs"]])
        compact = _recolor_pattern(pattern["compact"], actual_colors)
        row_sizes = rng.integers(3, 8, size=3)
        col_sizes = rng.integers(3, 8, size=3)
        grid = _paint_separator_grid(compact, row_sizes, col_sizes)

        try:
            _mode_ok(mode, compact)
            if not np.array_equal(_compress_regions(grid), compact):
                raise AssertionError("separator expansion must round-trip through compression")
            expected = _draw_compact_routes(compact)
            observed = main(grid)
            if not np.array_equal(observed, expected):
                raise AssertionError("main() must agree with compact-space oracle")
            if np.array_equal(expected, compact):
                raise AssertionError("output must differ from the compressed input")
        except AssertionError:
            continue

        return grid

    raise RuntimeError("failed to generate a valid separator-routing input")
