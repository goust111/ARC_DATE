from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def _line_compatible(a: np.ndarray, b: np.ndarray, marker: int) -> bool:
    return bool(np.all((a == b) | (a == marker) | (b == marker)))


def _run_starts_ignore_marker(grid: np.ndarray, axis: int, marker: int) -> list[int]:
    length = int(grid.shape[axis])
    starts = [0]
    for i in range(1, length):
        if axis == 0:
            prev = grid[i - 1, :]
            cur = grid[i, :]
        else:
            prev = grid[:, i - 1]
            cur = grid[:, i]
        if not _line_compatible(cur, prev, marker):
            starts.append(i)
    return starts


def _compress_ignore_marker(grid: np.ndarray, marker: int) -> tuple[np.ndarray, list[int], list[int]]:
    row_starts = _run_starts_ignore_marker(grid, axis=0, marker=marker)
    row_compressed = grid[row_starts, :]
    col_starts = _run_starts_ignore_marker(row_compressed, axis=1, marker=marker)
    return row_compressed[:, col_starts].copy(), row_starts, col_starts


def _is_solid_rectangle(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].all())


def _solid_rectangle_candidates(grid: np.ndarray) -> list[int]:
    candidates: list[int] = []
    for color in sorted(set(map(int, grid.ravel()))):
        mask = grid == color
        if mask.sum() <= 1:
            continue
        if _is_solid_rectangle(mask):
            candidates.append(int(color))
    return candidates


def _recover_biaxial_markers(comp: np.ndarray, marker: int) -> np.ndarray:
    out = comp.copy()
    h, w = out.shape
    while True:
        changed = False
        for r, c in np.argwhere(out == marker):
            rr = h - 1 - int(r)
            cc = w - 1 - int(c)
            values = set()
            if int(out[rr, c]) != marker:
                values.add(int(out[rr, c]))
            if int(out[r, cc]) != marker:
                values.add(int(out[r, cc]))
            if len(values) > 1:
                raise AssertionError("horizontal and vertical mirrors disagree")
            if len(values) == 1:
                out[int(r), int(c)] = values.pop()
                changed = True
        if not changed:
            break
    if np.any(out == marker):
        raise AssertionError("some compressed marker cells have no visible horizontal/vertical mirror")
    return out


def _is_biaxially_symmetric(grid: np.ndarray) -> bool:
    return bool(np.array_equal(grid, grid[::-1, :]) and np.array_equal(grid, grid[:, ::-1]))


def _try_solve_with_marker(grid: np.ndarray, marker: int) -> np.ndarray | None:
    compressed, _rows, _cols = _compress_ignore_marker(grid, marker)
    if not np.any(compressed == marker):
        return None
    restored = _recover_biaxial_markers(compressed, marker)
    if not _is_biaxially_symmetric(restored):
        return None
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress away adjacent duplicated bands, then repair the compressed marker region by biaxial reflection.

    Input structure:
    - The input is a run-length-expanded color grid: each row band and column band may be stretched by a different adjacent repeat count.
    - One color forms a unique solid rectangle that occludes part of the stretched pattern.
    - When comparing neighboring rows or columns for duplication, the marker color acts as a wildcard.

    Rule:
    - Find the solid rectangular marker color.
    - Scan rows top-to-bottom and merge each run of adjacent marker-compatible duplicate rows to its first representative; then do the same for columns left-to-right.
    - In the compressed grid, replace every remaining marker cell with the unique non-marker value found at its horizontal or vertical mirror location.

    Key insight:
    - The large input is not itself symmetric; symmetry only appears after removing the unequal row/column stretch factors, so compression must happen before reflection-based repair.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    solutions: list[tuple[int, np.ndarray]] = []
    for marker in _solid_rectangle_candidates(g):
        try:
            restored = _try_solve_with_marker(g, marker)
        except AssertionError:
            continue
        if restored is not None:
            solutions.append((int(marker), restored))

    assert len(solutions) == 1, "expected a unique solid-rectangle marker interpretation"
    return solutions[0][1]


PALETTE = tuple(range(1, 10))
COMPRESSED_SIZE_OPTIONS = tuple(range(5, 10))
EXPAND_MIN = 1
EXPAND_MAX = 3
MARKER_MIN_SIDE = 2
MARKER_MAX_AREA = 18
GEN_ATTEMPTS = 500


def _mirror(i: int, n: int) -> int:
    return n - 1 - i


def _interval_overlaps_mirror(start: int, end: int, n: int) -> bool:
    mirror_start = _mirror(end, n)
    mirror_end = _mirror(start, n)
    return not (end < mirror_start or mirror_end < start)


def _build_biaxial_seed(h: int, w: int, colors: Iterable[int]) -> np.ndarray:
    palette = list(colors)
    top_h = (h + 1) // 2
    left_w = (w + 1) // 2
    basis = np.array(
        [[random.choice(palette) for _ in range(left_w)] for _ in range(top_h)],
        dtype=int,
    )
    seed = np.empty((h, w), dtype=int)
    for r in range(h):
        rr = min(r, h - 1 - r)
        for c in range(w):
            cc = min(c, w - 1 - c)
            seed[r, c] = int(basis[rr, cc])
    return seed


def _adjacent_lines_distinct(grid: np.ndarray, axis: int) -> bool:
    for i in range(1, grid.shape[axis]):
        if axis == 0:
            if np.array_equal(grid[i - 1, :], grid[i, :]):
                return False
        else:
            if np.array_equal(grid[:, i - 1], grid[:, i]):
                return False
    return True


def _good_seed(seed: np.ndarray) -> bool:
    if len(set(map(int, seed.ravel()))) < 3:
        return False
    if not _adjacent_lines_distinct(seed, axis=0):
        return False
    if not _adjacent_lines_distinct(seed, axis=1):
        return False
    for color in set(map(int, seed.ravel())):
        if _is_solid_rectangle(seed == color):
            return False
    return True


def _seed_attempt() -> tuple[np.ndarray, int] | None:
    h = random.choice(COMPRESSED_SIZE_OPTIONS)
    w = random.choice(COMPRESSED_SIZE_OPTIONS)
    k = random.randint(3, min(6, len(PALETTE) - 1))
    pattern_colors = random.sample(PALETTE, k)
    marker_choices = [c for c in PALETTE if c not in pattern_colors]
    if not marker_choices:
        return None
    for _ in range(60):
        seed = _build_biaxial_seed(h, w, pattern_colors)
        if _good_seed(seed):
            return seed, random.choice(marker_choices)
    return None


def _rectangles_by_category(h: int, w: int) -> dict[str, list[tuple[int, int, int, int]]]:
    buckets = {"off_axis": [], "row_straddle": [], "col_straddle": []}
    for r0 in range(h):
        for r1 in range(r0 + MARKER_MIN_SIDE - 1, h):
            for c0 in range(w):
                for c1 in range(c0 + MARKER_MIN_SIDE - 1, w):
                    height = r1 - r0 + 1
                    width = c1 - c0 + 1
                    area = height * width
                    if area > MARKER_MAX_AREA or area >= h * w:
                        continue
                    row_overlap = _interval_overlaps_mirror(r0, r1, h)
                    col_overlap = _interval_overlaps_mirror(c0, c1, w)
                    if row_overlap and col_overlap:
                        continue
                    rect = (r0, r1, c0, c1)
                    if row_overlap:
                        buckets["row_straddle"].append(rect)
                    elif col_overlap:
                        buckets["col_straddle"].append(rect)
                    else:
                        buckets["off_axis"].append(rect)
    return buckets


def _visibility_counts(h: int, w: int, rect: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, r1, c0, c1 = rect
    horizontal_only = 0
    vertical_only = 0
    both_visible = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            h_visible = not (r0 <= _mirror(r, h) <= r1)
            v_visible = not (c0 <= _mirror(c, w) <= c1)
            if h_visible and v_visible:
                both_visible += 1
            elif h_visible:
                horizontal_only += 1
            elif v_visible:
                vertical_only += 1
            else:
                raise AssertionError("invalid rectangle hides both mirrors for a cell")
    return horizontal_only, vertical_only, both_visible


def _sample_runs(length: int) -> list[int]:
    while True:
        runs = [random.randint(EXPAND_MIN, EXPAND_MAX) for _ in range(length)]
        if max(runs) == 1:
            continue
        if len(set(runs)) == 1:
            continue
        if runs == runs[::-1]:
            continue
        return runs


def _expand(seed: np.ndarray, row_runs: list[int], col_runs: list[int]) -> np.ndarray:
    return np.repeat(np.repeat(seed, row_runs, axis=0), col_runs, axis=1)


def _expanded_rect(row_runs: list[int], col_runs: list[int], rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, r1, c0, c1 = rect
    rr0 = sum(row_runs[:r0])
    rr1 = sum(row_runs[: r1 + 1]) - 1
    cc0 = sum(col_runs[:c0])
    cc1 = sum(col_runs[: c1 + 1]) - 1
    return rr0, rr1, cc0, cc1


def _apply_marker(seed: np.ndarray, marker: int, rect: tuple[int, int, int, int]) -> np.ndarray:
    masked = seed.copy()
    r0, r1, c0, c1 = rect
    masked[r0 : r1 + 1, c0 : c1 + 1] = marker
    return masked


def _class_ok(category: str, counts: tuple[int, int, int]) -> bool:
    horizontal_only, vertical_only, both_visible = counts
    if category == "row_straddle":
        return vertical_only > 0 and both_visible > 0
    if category == "col_straddle":
        return horizontal_only > 0 and both_visible > 0
    return both_visible > 0


def _semantic_checks(
    inp: np.ndarray,
    clean_expanded: np.ndarray,
    seed: np.ndarray,
    masked_seed: np.ndarray,
    marker: int,
    category: str,
    rect: tuple[int, int, int, int],
) -> bool:
    if np.array_equal(clean_expanded, clean_expanded[::-1, :]):
        return False
    if np.array_equal(clean_expanded, clean_expanded[:, ::-1]):
        return False
    if _solid_rectangle_candidates(inp) != [marker]:
        return False
    compressed, row_starts, col_starts = _compress_ignore_marker(inp, marker)
    if not np.array_equal(compressed, masked_seed):
        return False
    if len(row_starts) != seed.shape[0] or len(col_starts) != seed.shape[1]:
        return False
    if not np.any(compressed == marker):
        return False
    if not _class_ok(category, _visibility_counts(seed.shape[0], seed.shape[1], rect)):
        return False
    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, seed))


def _gen_attempt() -> np.ndarray | None:
    made = _seed_attempt()
    if made is None:
        return None
    seed, marker = made
    h, w = seed.shape

    buckets = _rectangles_by_category(h, w)
    available = [name for name, rects in buckets.items() if rects]
    if not available:
        return None
    category = random.choice(available)
    rect = random.choice(buckets[category])

    row_runs = _sample_runs(h)
    col_runs = _sample_runs(w)
    clean_expanded = _expand(seed, row_runs, col_runs)
    if max(clean_expanded.shape) > 30:
        return None

    rr0, rr1, cc0, cc1 = _expanded_rect(row_runs, col_runs, rect)
    inp = clean_expanded.copy()
    inp[rr0 : rr1 + 1, cc0 : cc1 + 1] = marker

    masked_seed = _apply_marker(seed, marker, rect)
    if not _semantic_checks(inp, clean_expanded, seed, masked_seed, marker, category, rect):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compressed-symmetry input")
