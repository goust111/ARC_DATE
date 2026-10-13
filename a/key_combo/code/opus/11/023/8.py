import random
from collections import Counter

import numpy as np


BACKGROUND = 0
SYMBOL_COLORS = (1, 2, 3)
CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
EDGE_POSITIONS = ((0, 1), (1, 0), (1, 2), (2, 1))
CORNER_TO_DIRECTION = {
    (0, 0): (-1, -1),
    (0, 2): (-1, 1),
    (2, 0): (1, -1),
    (2, 2): (1, 1),
}
CORNER_TO_ADJACENT_EDGES = {
    (0, 0): {(0, 1), (1, 0)},
    (0, 2): {(0, 1), (1, 2)},
    (2, 0): {(1, 0), (2, 1)},
    (2, 2): {(1, 2), (2, 1)},
}

GEN_HEIGHT_RANGE = (18, 24)
GEN_WIDTH_RANGE = (18, 24)
GEN_SYMBOL_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 300

_SELF_TESTED = False


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return int(counts.most_common(1)[0][0])


def _build_symbol_patch(color: int, missing_corner: tuple[int, int], count: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    extra_missing = 7 - int(count)
    assert extra_missing in (0, 1, 2)

    candidates = list(EDGE_POSITIONS)
    for _ in range(100):
        removed_edges = set(random.sample(candidates, extra_missing))
        if removed_edges >= CORNER_TO_ADJACENT_EDGES[missing_corner]:
            continue
        break
    else:
        raise RuntimeError("failed to sample a readable symbol patch")

    for rr in range(3):
        for cc in range(3):
            if (rr, cc) == (1, 1):
                continue
            if (rr, cc) == missing_corner:
                continue
            if (rr, cc) in removed_edges:
                continue
            patch[rr, cc] = int(color)
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _find_symbols(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    h, w = g.shape
    symbols = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != bg:
                continue

            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            ring_values = []
            for rr in range(3):
                for cc in range(3):
                    if (rr, cc) == (1, 1):
                        continue
                    value = int(patch[rr, cc])
                    if value != bg:
                        ring_values.append((rr, cc, value))

            if not 5 <= len(ring_values) <= 7:
                continue

            colors = {value for _, _, value in ring_values}
            if len(colors) != 1:
                continue

            missing_corners = [corner for corner in CORNERS if int(patch[corner]) == bg]
            if len(missing_corners) != 1:
                continue

            color = int(next(iter(colors)))
            symbols.append(
                {
                    "center": (r, c),
                    "color": color,
                    "count": len(ring_values),
                    "missing_corner": missing_corners[0],
                }
            )

    return symbols


def _apply_rays(
    grid: np.ndarray,
    symbols: list[dict],
    *,
    count_override: int | None = None,
    reverse_direction: bool = False,
    reflect_horizontal: bool = False,
    reflect_vertical: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()
    h, w = out.shape

    for symbol in sorted(symbols, key=lambda item: (item["center"][0], item["center"][1])):
        r0, c0 = symbol["center"]
        dr, dc = CORNER_TO_DIRECTION[symbol["missing_corner"]]
        if reverse_direction:
            dr, dc = -dr, -dc
        if reflect_horizontal:
            dc = -dc
        if reflect_vertical:
            dr = -dr

        ray_len = int(symbol["count"] if count_override is None else count_override)
        for step in range(1, ray_len + 1):
            r = r0 + dr * step
            c = c0 + dc * step
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(out[r, c]) != bg:
                break
            out[r, c] = int(symbol["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count each hollow 3x3 symbol's colored pixels and extend a same-color diagonal ray through its missing corner.

    Input structure:
    - The grid has a black background and several isolated 3x3 hollow symbols in colors 1, 2, or 3.
    - Each symbol has a background center, exactly one missing corner pixel, and may miss extra edge pixels, so its ring contains 5 to 7 foreground pixels.
    - Symbols are spaced apart so each symbol can be read from its own 3x3 neighborhood.

    Rule:
    - Detect every 3x3 hollow symbol and count its non-background ring pixels N.
    - Map the missing corner to a diagonal direction: top-left -> (-1,-1), top-right -> (-1,+1), bottom-left -> (+1,-1), bottom-right -> (+1,+1).
    - Starting one step away from the symbol center, draw up to N same-color pixels along that diagonal; stop early if the next cell is already non-background.
    - Keep the original symbol pixels unchanged.

    Key insight:
    - The missing corner tells you where to fire, while the remaining ring-pixel count tells you how far to fire, so both spatial and numeric clues are necessary.
    """
    g = np.asarray(grid, dtype=int)
    return _apply_rays(g, _find_symbols(g))


def _choose_counts(n_symbols: int) -> list[int]:
    counts = [5, 6, 7]
    while len(counts) < n_symbols:
        counts.append(random.choice((5, 6, 7)))
    random.shuffle(counts)
    return counts


def _choose_missing_corners(n_symbols: int) -> list[tuple[int, int]]:
    corners = list(CORNERS)
    random.shuffle(corners)
    picks = corners[: min(n_symbols, len(corners))]
    while len(picks) < n_symbols:
        picks.append(random.choice(CORNERS))
    random.shuffle(picks)
    return picks


def _choose_colors(n_symbols: int) -> list[int]:
    colors = list(SYMBOL_COLORS)
    random.shuffle(colors)
    picks = colors[: min(n_symbols, len(colors))]
    while len(picks) < n_symbols:
        picks.append(random.choice(SYMBOL_COLORS))
    random.shuffle(picks)
    return picks


def _symbol_cells(center: tuple[int, int], patch: np.ndarray) -> set[tuple[int, int]]:
    r0, c0 = center
    cells = set()
    for rr in range(3):
        for cc in range(3):
            if int(patch[rr, cc]) != BACKGROUND:
                cells.add((r0 - 1 + rr, c0 - 1 + cc))
    return cells


def _ray_cells(center: tuple[int, int], missing_corner: tuple[int, int], count: int) -> list[tuple[int, int]]:
    dr, dc = CORNER_TO_DIRECTION[missing_corner]
    return [(center[0] + dr * step, center[1] + dc * step) for step in range(1, int(count) + 1)]


def _center_clear(candidate: tuple[int, int], placed_centers: list[tuple[int, int]]) -> bool:
    return all(max(abs(candidate[0] - r), abs(candidate[1] - c)) >= 7 for r, c in placed_centers)


def _candidate_valid(
    shape: tuple[int, int],
    center: tuple[int, int],
    patch: np.ndarray,
    missing_corner: tuple[int, int],
    count: int,
    used_symbol_cells: set[tuple[int, int]],
    used_ray_cells: set[tuple[int, int]],
    placed_centers: list[tuple[int, int]],
) -> bool:
    h, w = shape
    if not _center_clear(center, placed_centers):
        return False

    symbol_cells = _symbol_cells(center, patch)
    ray_cells = _ray_cells(center, missing_corner, count)

    for r, c in symbol_cells | set(ray_cells):
        if not (0 <= r < h and 0 <= c < w):
            return False

    if symbol_cells & used_symbol_cells:
        return False
    if symbol_cells & used_ray_cells:
        return False
    if set(ray_cells) & used_symbol_cells:
        return False
    if set(ray_cells) & used_ray_cells:
        return False

    return True


def _wrong_hypothesis_matches(grid: np.ndarray, symbols: list[dict], expected: np.ndarray) -> bool:
    checks = [
        _apply_rays(grid, symbols, count_override=5),
        _apply_rays(grid, symbols, count_override=7),
        _apply_rays(grid, symbols, reverse_direction=True),
        _apply_rays(grid, symbols, reflect_horizontal=True),
        _apply_rays(grid, symbols, reflect_vertical=True),
    ]
    return any(np.array_equal(candidate, expected) for candidate in checks)


def _validate_generated_case(grid: np.ndarray, expected_symbol_count: int) -> bool:
    symbols = _find_symbols(grid)
    if len(symbols) != expected_symbol_count:
        return False
    if len({symbol["count"] for symbol in symbols}) < 2:
        return False
    if len({symbol["missing_corner"] for symbol in symbols}) < 2:
        return False

    expected = _apply_rays(grid, symbols)
    if np.array_equal(expected, grid):
        return False
    if not np.array_equal(main(grid), expected):
        return False
    if _wrong_hypothesis_matches(grid, symbols, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    n_symbols = random.randint(*GEN_SYMBOL_COUNT_RANGE)

    counts = _choose_counts(n_symbols)
    corners = _choose_missing_corners(n_symbols)
    colors = _choose_colors(n_symbols)

    specs = [
        {"count": count, "missing_corner": corner, "color": color}
        for count, corner, color in zip(counts, corners, colors)
    ]
    random.shuffle(specs)
    specs.sort(key=lambda item: -int(item["count"]))

    grid = np.zeros((h, w), dtype=int)
    used_symbol_cells: set[tuple[int, int]] = set()
    used_ray_cells: set[tuple[int, int]] = set()
    placed_centers: list[tuple[int, int]] = []

    for spec in specs:
        patch = _build_symbol_patch(spec["color"], spec["missing_corner"], spec["count"])
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            center = (random.randint(1, h - 2), random.randint(1, w - 2))
            if not _candidate_valid(
                (h, w),
                center,
                patch,
                spec["missing_corner"],
                spec["count"],
                used_symbol_cells,
                used_ray_cells,
                placed_centers,
            ):
                continue

            _place_patch(grid, patch, center[0] - 1, center[1] - 1)
            used_symbol_cells.update(_symbol_cells(center, patch))
            used_ray_cells.update(_ray_cells(center, spec["missing_corner"], spec["count"]))
            placed_centers.append(center)
            placed = True
            break

        if not placed:
            return None

    if len(set(map(int, np.unique(grid))) - {BACKGROUND}) < 2:
        return None

    if not _validate_generated_case(grid, n_symbols):
        return None

    return grid


def _run_solver_self_checks() -> None:
    # Direction + length from a 7-pixel symbol.
    grid = np.zeros((12, 12), dtype=int)
    patch = _build_symbol_patch(1, (0, 0), 7)
    _place_patch(grid, patch, 6, 6)
    out = main(grid)
    expected = grid.copy()
    for step in range(1, 8):
        expected[7 - step, 7 - step] = 1
    assert np.array_equal(out, expected)

    # Extra edge removals must shorten the ray.
    grid = np.zeros((12, 12), dtype=int)
    patch = _build_symbol_patch(2, (2, 2), 5)
    _place_patch(grid, patch, 2, 2)
    out = main(grid)
    symbols = _find_symbols(grid)
    assert len(symbols) == 1 and symbols[0]["count"] == 5
    expected = _apply_rays(grid, symbols)
    assert np.array_equal(out, expected)

    # Stop when a blocker is encountered.
    grid = np.zeros((12, 12), dtype=int)
    patch = _build_symbol_patch(3, (2, 2), 6)
    _place_patch(grid, patch, 1, 1)
    grid[6, 6] = 2
    out = main(grid)
    assert int(out[4, 4]) == 3 and int(out[5, 5]) == 3 and int(out[6, 6]) == 2
    assert int(out[7, 7]) == 0

    # Wrong hypotheses must disagree on a mixed case.
    grid = np.zeros((20, 20), dtype=int)
    patch_a = _build_symbol_patch(1, (0, 2), 7)
    patch_b = _build_symbol_patch(2, (2, 0), 5)
    _place_patch(grid, patch_a, 9, 3)
    _place_patch(grid, patch_b, 3, 12)
    symbols = _find_symbols(grid)
    expected = _apply_rays(grid, symbols)
    assert not _wrong_hypothesis_matches(grid, symbols, expected)


def generate_input() -> np.ndarray:
    global _SELF_TESTED

    if not _SELF_TESTED:
        _run_solver_self_checks()
        _SELF_TESTED = True

    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid puzzle-8 input")
