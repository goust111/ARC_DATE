from __future__ import annotations

from collections import Counter
import random

import numpy as np


# This file is deliberately self-contained so that SourceFileLoader can load it
# without relying on the caller's sys.path.  It follows arc_utils' canonical
# conventions: grids have shape (rows, columns), and coordinates are (r, c).


def _infer_layout(grid: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    """Return (separator color, N, divider rows, divider columns)."""
    h, w = grid.shape
    candidates: list[tuple[int, int, list[int], list[int]]] = []

    for value in np.unique(grid):
        separator = int(value)
        row_divs = [r for r in range(h) if bool(np.all(grid[r, :] == separator))]
        col_divs = [c for c in range(w) if bool(np.all(grid[:, c] == separator))]
        if len(row_divs) < 3 or len(row_divs) != len(col_divs):
            continue
        if row_divs[0] != 0 or row_divs[-1] != h - 1:
            continue
        if col_divs[0] != 0 or col_divs[-1] != w - 1:
            continue

        row_steps = np.diff(row_divs)
        col_steps = np.diff(col_divs)
        if not (np.all(row_steps == row_steps[0]) and np.all(col_steps == col_steps[0])):
            continue
        step = int(row_steps[0])
        n = len(row_divs) - 1

        # Each cell interior is itself an N x N blueprint.  Thus a divider-to-
        # divider step is N+1, and the cell grid also has N rows and N columns.
        if step != n + 1 or int(col_steps[0]) != step:
            continue
        if h != n * step + 1 or w != n * step + 1:
            continue
        if row_divs != list(range(0, h, step)):
            continue
        if col_divs != list(range(0, w, step)):
            continue
        candidates.append((separator, n, row_divs, col_divs))

    assert len(candidates) == 1, "expected one visible regular separator grid"
    return candidates[0]


def _extract_cells(grid: np.ndarray, n: int) -> list[list[np.ndarray]]:
    """Extract cell interiors in the same N x N (row, column) reference frame."""
    step = n + 1
    return [
        [
            grid[i * step + 1 : (i + 1) * step, j * step + 1 : (j + 1) * step].copy()
            for j in range(n)
        ]
        for i in range(n)
    ]


def _pattern_key(pattern: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in pattern.ravel())


def _stamp_cell(out: np.ndarray, n: int, r: int, c: int, pattern: np.ndarray) -> None:
    """Opaque-paste one N x N pattern into cell (r, c)."""
    step = n + 1
    r0, c0 = r * step + 1, c * step + 1
    out[r0 : r0 + n, c0 : c0 + n] = pattern


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete two minority cell classes whose inner patterns are cross-blueprints for one another.

    Input structure:
    - One-pixel monochrome separators divide the square grid into N x N cells, each with an N x N interior.
    - A unique most frequent interior pattern is the majority pattern; exactly two other interior patterns A and B are visible.
    - Relative to the majority pattern, A's changed pixels mark every B-cell coordinate, while B's changed pixels mark every A-cell coordinate.
    - At least one cell of each minority class is already visible, and some blueprint-marked cells still show the majority pattern.

    Rule:
    - Compare A and B with the majority pattern in their own N x N interiors to obtain their active-pixel masks.
    - In the same row-column reference frame on the N x N cell grid, stamp B at A's active pixels and stamp A at B's active pixels.
    - Replace only blueprint-marked cells that still have the majority pattern; preserve separators, already completed cells, and all unmarked majority cells.

    Key insight:
    - Each minority pattern points to the other class, not itself: the visible A cells lie on B's active pixels and the visible B cells lie on A's active pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "grid must be a non-empty 2D array"
    _separator, n, _row_divs, _col_divs = _infer_layout(g)
    cells = _extract_cells(g, n)

    arrays: dict[tuple[int, ...], np.ndarray] = {}
    keys: list[list[tuple[int, ...]]] = []
    for row in cells:
        key_row: list[tuple[int, ...]] = []
        for pattern in row:
            key = _pattern_key(pattern)
            arrays.setdefault(key, pattern)
            key_row.append(key)
        keys.append(key_row)

    counts = Counter(key for row in keys for key in row)
    assert len(counts) == 3, "expected one majority pattern and exactly two minority patterns"
    max_count = max(counts.values())
    majority_candidates = [key for key, count in counts.items() if count == max_count]
    assert len(majority_candidates) == 1, "majority pattern must be uniquely most frequent"
    majority_key = majority_candidates[0]
    minority_keys = [key for key in counts if key != majority_key]
    assert len(minority_keys) == 2

    p_key, q_key = minority_keys
    majority = arrays[majority_key]
    p_pattern, q_pattern = arrays[p_key], arrays[q_key]
    p_blueprint = p_pattern != majority  # positions of Q cells
    q_blueprint = q_pattern != majority  # positions of P cells
    assert bool(p_blueprint.any()) and bool(q_blueprint.any()), "blueprints must be non-empty"
    assert not bool(np.any(p_blueprint & q_blueprint)), "the two complete cell classes cannot overlap"

    p_cells = np.array([[keys[r][c] == p_key for c in range(n)] for r in range(n)], dtype=bool)
    q_cells = np.array([[keys[r][c] == q_key for c in range(n)] for r in range(n)], dtype=bool)
    majority_cells = np.array(
        [[keys[r][c] == majority_key for c in range(n)] for r in range(n)], dtype=bool
    )

    # Visible minority cells are the calibration code that uniquely establishes
    # the cross-reference and rules out reading either pattern as its own map.
    assert bool(p_cells.any()) and bool(q_cells.any()), "both minority classes must already be visible"
    assert not bool(np.any(p_cells & ~q_blueprint)), "P cells must be marked by Q's blueprint"
    assert not bool(np.any(q_cells & ~p_blueprint)), "Q cells must be marked by P's blueprint"
    assert not bool(np.any(q_blueprint & ~(p_cells | majority_cells)))
    assert not bool(np.any(p_blueprint & ~(q_cells | majority_cells)))

    out = g.copy()
    for r in range(n):
        for c in range(n):
            if q_blueprint[r, c]:
                if majority_cells[r, c]:
                    _stamp_cell(out, n, r, c, p_pattern)
            elif p_blueprint[r, c]:
                if majority_cells[r, c]:
                    _stamp_cell(out, n, r, c, q_pattern)
    return out


def _random_blueprints(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Make disjoint, non-degenerate maps with boundary and interior coverage."""
    if n == 3:
        size_a, size_b = random.choice(((2, 2), (2, 3), (3, 2)))
    else:
        size_a = random.randint(3, 5)
        size_b = random.randint(3, 5)

    all_coords = [(r, c) for r in range(n) for c in range(n)]
    for _ in range(100):
        chosen = random.sample(all_coords, size_a + size_b)
        a_coords = chosen[:size_a]
        b_coords = chosen[size_a:]
        union = a_coords + b_coords
        has_corner = any(r in (0, n - 1) and c in (0, n - 1) for r, c in union)
        has_interior = any(0 < r < n - 1 and 0 < c < n - 1 for r, c in union)
        a_touches_edge = any(r in (0, n - 1) or c in (0, n - 1) for r, c in a_coords)
        b_touches_edge = any(r in (0, n - 1) or c in (0, n - 1) for r, c in b_coords)
        if has_corner and has_interior and a_touches_edge and b_touches_edge:
            a = np.zeros((n, n), dtype=bool)
            b = np.zeros((n, n), dtype=bool)
            for r, c in a_coords:
                a[r, c] = True
            for r, c in b_coords:
                b[r, c] = True
            return a, b
    raise RuntimeError("could not sample non-degenerate blueprints")


def _random_majority_pattern(n: int, base: int, accent: int) -> np.ndarray:
    """Create a visible shared motif, so active pixels require pattern comparison."""
    pattern = np.full((n, n), int(base), dtype=int)
    accent_count = random.randint(1, n)
    for index in random.sample(range(n * n), accent_count):
        pattern[index // n, index % n] = int(accent)
    return pattern


def _render_grid(
    n: int,
    separator: int,
    majority: np.ndarray,
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    a_cells: set[tuple[int, int]],
    b_cells: set[tuple[int, int]],
) -> np.ndarray:
    size = n * (n + 1) + 1
    grid = np.full((size, size), int(separator), dtype=int)
    for r in range(n):
        for c in range(n):
            pattern = pattern_a if (r, c) in a_cells else pattern_b if (r, c) in b_cells else majority
            _stamp_cell(grid, n, r, c, pattern)
    return grid


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    n: int,
    majority: np.ndarray,
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    blueprint_a: np.ndarray,
    blueprint_b: np.ndarray,
    shown_a: set[tuple[int, int]],
    shown_b: set[tuple[int, int]],
) -> bool:
    """Check semantics, uniqueness clues, and anti-shortcut counterexamples."""
    target_a = {tuple(map(int, rc)) for rc in np.argwhere(blueprint_b)}
    target_b = {tuple(map(int, rc)) for rc in np.argwhere(blueprint_a)}

    # Both cross-blueprints contribute missing cells, so neither step is cosmetic.
    if not shown_a or not shown_b:
        return False
    if not (shown_a < target_a and shown_b < target_b):
        return False
    if target_a & target_b or target_a | target_b == {(r, c) for r in range(n) for c in range(n)}:
        return False

    # Existing cells contradict self-reference because the maps are disjoint.
    own_a = {tuple(map(int, rc)) for rc in np.argwhere(blueprint_a)}
    own_b = {tuple(map(int, rc)) for rc in np.argwhere(blueprint_b)}
    if shown_a <= own_a or shown_b <= own_b:
        return False

    # Independently inspect the generated cell states, without calling main.
    input_cells = _extract_cells(inp, n)
    output_cells = _extract_cells(expected, n)
    added_a = added_b = 0
    for r in range(n):
        for c in range(n):
            coord = (r, c)
            wanted = pattern_a if coord in target_a else pattern_b if coord in target_b else majority
            if not np.array_equal(output_cells[r][c], wanted):
                return False
            if coord in target_a - shown_a:
                added_a += int(np.array_equal(input_cells[r][c], majority))
            if coord in target_b - shown_b:
                added_b += int(np.array_equal(input_cells[r][c], majority))
    if added_a == 0 or added_b == 0:
        return False

    # A self-reading shortcut produces a visibly different completion.
    self_read = _render_grid(n, int(inp[0, 0]), majority, pattern_a, pattern_b, own_a, own_b)
    if np.array_equal(self_read, expected):
        return False

    try:
        solved = main(inp)
    except (AssertionError, ValueError, IndexError):
        return False
    return bool(np.array_equal(solved, expected) and not np.array_equal(inp, expected))


_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.choice((3, 4))
    separator, base, accent, mark_a, mark_b = random.sample(range(10), 5)
    majority = _random_majority_pattern(n, base, accent)
    blueprint_a, blueprint_b = _random_blueprints(n)

    pattern_a = majority.copy()
    pattern_b = majority.copy()
    pattern_a[blueprint_a] = int(mark_a)
    pattern_b[blueprint_b] = int(mark_b)

    # A's blueprint names all B cells; B's blueprint names all A cells.
    target_a = [tuple(map(int, rc)) for rc in np.argwhere(blueprint_b)]
    target_b = [tuple(map(int, rc)) for rc in np.argwhere(blueprint_a)]
    shown_a = set(random.sample(target_a, random.randint(1, len(target_a) - 1)))
    shown_b = set(random.sample(target_b, random.randint(1, len(target_b) - 1)))

    inp = _render_grid(n, separator, majority, pattern_a, pattern_b, shown_a, shown_b)
    expected = _render_grid(n, separator, majority, pattern_a, pattern_b, set(target_a), set(target_b))
    signature = (expected.shape, expected.tobytes())
    if signature in _SEEN_OUTPUTS:
        return None
    if not _validate_generated(
        inp,
        expected,
        n,
        majority,
        pattern_a,
        pattern_b,
        blueprint_a,
        blueprint_b,
        shown_a,
        shown_b,
    ):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a diverse, semantically validated cross-blueprint puzzle input."""
    for _ in range(500):
        result = _generate_attempt()
        if result is None:
            continue
        inp, expected = result
        _SEEN_OUTPUTS.add((expected.shape, expected.tobytes()))
        return inp
    raise RuntimeError("failed to generate a unique valid cross-blueprint input")
