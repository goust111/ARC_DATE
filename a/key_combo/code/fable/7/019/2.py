from __future__ import annotations

import numpy as np


OPERATOR = 5
ARC_COLORS = tuple(range(10))
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _parse(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    """Return logical cells and horizontal/vertical edges using (row, column) coordinates."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3 and height % 2 == 1 and width % 2 == 1

    # Every odd/odd position is an unambiguous separator intersection.
    background = int(g[1, 1])
    logical = g[::2, ::2].copy()
    assert bool(np.all(logical != background))
    assert bool(np.all(logical != OPERATOR))

    separator_mask = np.ones(g.shape, dtype=bool)
    separator_mask[::2, ::2] = False
    separator_values = g[separator_mask]
    assert background != OPERATOR
    assert bool(np.all((separator_values == background) | (separator_values == OPERATOR)))
    # Intersections of separator rows and columns carry no operator.
    assert bool(np.all(g[1::2, 1::2] == background))

    horizontal = [
        (r, c)
        for r in range(logical.shape[0])
        for c in range(logical.shape[1] - 1)
        if int(g[2 * r, 2 * c + 1]) == OPERATOR
    ]
    vertical = [
        (r, c)
        for r in range(logical.shape[0] - 1)
        for c in range(logical.shape[1])
        if int(g[2 * r + 1, 2 * c]) == OPERATOR
    ]

    horizontal_cells = {(r, c) for r, c in horizontal} | {(r, c + 1) for r, c in horizontal}
    vertical_cells = {(r, c) for r, c in vertical} | {(r + 1, c) for r, c in vertical}
    assert horizontal_cells.isdisjoint(vertical_cells)
    assert horizontal or vertical
    return logical, horizontal, vertical


def _apply_swaps(
    logical: np.ndarray,
    horizontal: list[tuple[int, int]],
    vertical: list[tuple[int, int]],
) -> np.ndarray:
    """Apply the explicitly specified left-to-right, then top-to-bottom composition."""
    out = logical.copy()
    horizontal_set = set(horizontal)
    vertical_set = set(vertical)
    rows, cols = out.shape
    for r in range(rows):
        for c in range(cols - 1):
            if (r, c) in horizontal_set:
                out[r, c], out[r, c + 1] = out[r, c + 1], out[r, c]
    for c in range(cols):
        for r in range(rows - 1):
            if (r, c) in vertical_set:
                out[r, c], out[r + 1, c] = out[r + 1, c], out[r, c]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Treat grey defects on separator lines as ordered swaps, then remove all separator rows and columns.

    Input structure:
    - Colored logical cells occupy every even-row/even-column intersection of an odd-sized grid.
    - All other positions share one background color except for sparse grey operator pixels.
    - A grey pixel lies midway between two horizontal or vertical neighboring logical cells, and no
      logical cell touches operators of both orientations.

    Rule:
    - In each logical row, apply grey horizontal swaps from left to right; in each logical column,
      apply grey vertical swaps from top to bottom. Then output only the even/even logical cells,
      omitting every separator row, separator column, and grey operator.

    Key insight:
    - Adjacent grey operators compose sequentially: a chain moves colors cyclically, so its swaps
      cannot be evaluated independently before the separator grid is downsampled away.
    """
    logical, horizontal, vertical = _parse(grid)
    return _apply_swaps(logical, horizontal, vertical)


def _edge_cells(orientation: str, edge: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = edge
    if orientation == "h":
        return {(r, c), (r, c + 1)}
    return {(r, c), (r + 1, c)}


def _runs(edges: list[tuple[int, int]], orientation: str) -> list[list[tuple[int, int]]]:
    """Return the logical-cell paths induced by contiguous edges of one orientation."""
    remaining = set(edges)
    runs: list[list[tuple[int, int]]] = []
    while remaining:
        r, c = min(remaining)
        if orientation == "h":
            start = c
            while (r, start - 1) in remaining:
                start -= 1
            end = start
            while (r, end) in remaining:
                remaining.remove((r, end))
                end += 1
            runs.append([(r, x) for x in range(start, end + 1)])
        else:
            start = r
            while (start - 1, c) in remaining:
                start -= 1
            end = start
            while (end, c) in remaining:
                remaining.remove((end, c))
                end += 1
            runs.append([(x, c) for x in range(start, end + 1)])
    return runs


def _reverse_apply(
    logical: np.ndarray,
    horizontal: list[tuple[int, int]],
    vertical: list[tuple[int, int]],
) -> np.ndarray:
    """Deliberately wrong oracle: compose each row/column in the opposite order."""
    out = logical.copy()
    for r, c in sorted(horizontal, key=lambda e: (e[0], -e[1])):
        out[r, c], out[r, c + 1] = out[r, c + 1], out[r, c]
    for r, c in sorted(vertical, key=lambda e: (e[1], -e[0])):
        out[r, c], out[r + 1, c] = out[r + 1, c], out[r, c]
    return out


def _semantic_valid(
    grid: np.ndarray,
    logical: np.ndarray,
    horizontal: list[tuple[int, int]],
    vertical: list[tuple[int, int]],
) -> bool:
    """Check structure, order dependence, every operator's necessity, and non-degeneracy."""
    try:
        parsed_logic, parsed_h, parsed_v = _parse(grid)
        actual = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    if not np.array_equal(parsed_logic, logical) or parsed_h != sorted(horizontal) or parsed_v != sorted(vertical):
        return False

    expected = _apply_swaps(logical, sorted(horizontal), sorted(vertical))
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(expected, logical):
        return False  # downsampling alone must not solve the example
    if np.array_equal(expected, _reverse_apply(logical, horizontal, vertical)):
        return False  # the stated composition order must be visibly necessary

    all_edges = [("h", edge) for edge in horizontal] + [("v", edge) for edge in vertical]
    for orientation, edge in all_edges:
        reduced_h = [e for e in horizontal if not (orientation == "h" and e == edge)]
        reduced_v = [e for e in vertical if not (orientation == "v" and e == edge)]
        if np.array_equal(expected, _apply_swaps(logical, reduced_h, reduced_v)):
            return False
    return len(horizontal) >= 1 and len(vertical) >= 1 and max(grid.shape) <= 30


def _generation_attempt(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(4, 9))
    cols = int(rng.integers(4, 9))
    background = int(rng.choice([color for color in ARC_COLORS if color != OPERATOR]))
    logic_colors = tuple(color for color in ARC_COLORS if color not in (background, OPERATOR))
    primary = "h" if int(rng.integers(2)) == 0 else "v"

    if primary == "h":
        r = int(rng.integers(rows))
        c = int(rng.integers(cols - 2))
        horizontal = [(r, c), (r, c + 1)]
        vertical: list[tuple[int, int]] = []
    else:
        r = int(rng.integers(rows - 2))
        c = int(rng.integers(cols))
        vertical = [(r, c), (r + 1, c)]
        horizontal = []

    h_cells = set().union(*(_edge_cells("h", e) for e in horizontal)) if horizontal else set()
    v_cells = set().union(*(_edge_cells("v", e) for e in vertical)) if vertical else set()

    # Force the other orientation to occur on cells disjoint from the primary chain.
    if primary == "h":
        candidates = [
            (rr, cc)
            for rr in range(rows - 1)
            for cc in range(cols)
            if _edge_cells("v", (rr, cc)).isdisjoint(h_cells)
        ]
        if not candidates:
            return None
        edge = candidates[int(rng.integers(len(candidates)))]
        vertical.append(edge)
        v_cells |= _edge_cells("v", edge)
    else:
        candidates = [
            (rr, cc)
            for rr in range(rows)
            for cc in range(cols - 1)
            if _edge_cells("h", (rr, cc)).isdisjoint(v_cells)
        ]
        if not candidates:
            return None
        edge = candidates[int(rng.integers(len(candidates)))]
        horizontal.append(edge)
        h_cells |= _edge_cells("h", edge)

    # Add a small random number of legal operators, including occasional longer chains.
    pool = [("h", (r, c)) for r in range(rows) for c in range(cols - 1)]
    pool += [("v", (r, c)) for r in range(rows - 1) for c in range(cols)]
    rng.shuffle(pool)
    target_extra = int(rng.integers(0, 6))
    for orientation, edge in pool:
        if target_extra == 0:
            break
        if edge in (horizontal if orientation == "h" else vertical):
            continue
        cells = _edge_cells(orientation, edge)
        if orientation == "h" and cells.isdisjoint(v_cells):
            horizontal.append(edge)
            h_cells |= cells
            target_extra -= 1
        elif orientation == "v" and cells.isdisjoint(h_cells):
            vertical.append(edge)
            v_cells |= cells
            target_extra -= 1

    horizontal.sort()
    vertical.sort()
    logical = rng.choice(logic_colors, size=(rows, cols), replace=True).astype(int)
    # Distinct colors on every operator path make each ordered swap observably meaningful.
    for path in _runs(horizontal, "h") + _runs(vertical, "v"):
        if len(path) > len(logic_colors):
            return None
        colors = rng.choice(logic_colors, size=len(path), replace=False)
        for (rr, cc), color in zip(path, colors):
            logical[rr, cc] = int(color)

    grid = np.full((2 * rows - 1, 2 * cols - 1), background, dtype=int)
    grid[::2, ::2] = logical
    for r, c in horizontal:
        grid[2 * r, 2 * c + 1] = OPERATOR
    for r, c in vertical:
        grid[2 * r + 1, 2 * c] = OPERATOR

    if not _semantic_valid(grid, logical, horizontal, vertical):
        return None
    output = _apply_swaps(logical, horizontal, vertical)
    signature = (tuple(map(int, output.shape)), output.tobytes())
    if signature in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(signature)
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, order-sensitive instance satisfying every structural invariant."""
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _generation_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate separator-swap instance")
