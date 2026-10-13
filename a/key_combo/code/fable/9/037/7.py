from __future__ import annotations

import random
from itertools import combinations

import numpy as np


SCALE = 3
COLOR_POOL = tuple(range(1, 10))
DEPTH_CHOICES = (2, 3, 3)
GEN_RETRIES = 400


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _black_components(grid: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(grid == 0, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4((rr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_window(grid: np.ndarray) -> tuple[int, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, int, int]] = []
    for comp in _black_components(g):
        r0, c0, r1, c1 = _bbox(comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if comp.shape[0] != bh * bw:
            continue
        if not np.all(g[r0 : r1 + 1, c0 : c1 + 1] == 0):
            continue
        if h % bh != 0 or w % bw != 0:
            continue
        k_r = h // bh
        k_c = w // bw
        if k_r == k_c and k_r > 1:
            candidates.append((r0, c0, r1, c1, k_r))
    if len(candidates) != 1:
        raise AssertionError(f"expected exactly one proportional black window, got {len(candidates)}")
    return candidates[0]


def _compress_uniform(grid: np.ndarray, k: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if h % k != 0 or w % k != 0:
        raise AssertionError("grid must be divisible by the inferred scale")
    out = np.empty((h // k, w // k), dtype=int)
    for r in range(0, h, k):
        for c in range(0, w, k):
            block = g[r : r + k, c : c + k]
            value = int(block[0, 0])
            if not np.all(block == value):
                raise AssertionError("every kxk block must be monochrome before compression")
            out[r // k, c // k] = value
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique proportional black window and recursively replace it with a compressed self-copy.

    Input structure:
    - Non-black content is arranged on an integer kxk block lattice, so each visible kxk block is monochrome.
    - Among the black connected components, exactly one is a solid rectangle whose height and width are both 1/k of the full canvas.
    - Other black regions may exist as distractors, but they do not satisfy that same proportional-window condition.

    Rule:
    - Detect the solid black component whose bounding box is exactly the canvas size divided by the same integer k in both directions.
    - Compress the whole current canvas by turning each monochrome kxk block into one cell, producing a miniature copy of the full picture.
    - Paste that miniature into the black window, then repeat the same reasoning inside the pasted copy.
    - Stop when the newly implied window has shrunk to a single cell; that final cell stays black.

    Key insight:
    - The target black rectangle is not a hole to color once; it is a self-similar window, so solving the puzzle requires detecting the scale from the window-to-canvas ratio and then recursing on the miniature copy.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1, k = _find_window(g)
    if r0 == r1 and c0 == c1:
        return g.copy()

    miniature = _compress_uniform(g, k)
    filled = main(miniature)
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = filled
    return out


def _connected(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    stack = [cells[0]]
    seen = {cells[0]}
    while stack:
        cell = stack.pop()
        for nxt in _neighbors4(cell):
            if nxt in cell_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cell_set)


def _valid_distractor_shapes() -> list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(SCALE) for c in range(SCALE)]
    layouts: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for window in cells:
        others = [cell for cell in cells if cell != window]
        for size in (2, 3):
            for subset in combinations(others, size):
                if not _connected(subset):
                    continue
                if any(cell in _neighbors4(window) for cell in subset):
                    continue
                layouts.append((window, tuple(sorted(subset))))
    return layouts


_BASE_LAYOUTS = _valid_distractor_shapes()


def _build_base_grid(
    window: tuple[int, int],
    distractor: tuple[tuple[int, int], ...],
    colors: list[int],
) -> np.ndarray:
    base = np.zeros((SCALE, SCALE), dtype=int)
    distractor_set = set(distractor)
    free = [(r, c) for r in range(SCALE) for c in range(SCALE) if (r, c) != window and (r, c) not in distractor_set]
    values = colors[:]
    while len(values) < len(free):
        values.append(random.choice(colors))
    random.shuffle(values)
    for (r, c), value in zip(free, values):
        base[r, c] = int(value)
    for r, c in distractor:
        base[r, c] = 0
    base[window] = 0
    return base


def _scale_base(base: np.ndarray, depth: int) -> np.ndarray:
    factor = SCALE ** (depth - 1)
    return np.kron(base, np.ones((factor, factor), dtype=int))


def _explicit_output(base: np.ndarray, depth: int, window: tuple[int, int]) -> np.ndarray:
    grid = _scale_base(base, depth)
    if depth == 1:
        return grid
    factor = SCALE ** (depth - 1)
    wr, wc = window
    smaller = _explicit_output(base, depth - 1, window)
    out = grid.copy()
    out[wr * factor : (wr + 1) * factor, wc * factor : (wc + 1) * factor] = smaller
    return out


def _verify_levels(base: np.ndarray, depth: int) -> None:
    cur = _scale_base(base, depth)
    black_comp_counts = []
    for level in range(depth, 0, -1):
        r0, c0, r1, c1, k = _find_window(cur)
        if k != SCALE:
            raise AssertionError("generator must reveal the fixed scale from the window ratio")
        if (r1 - r0 + 1) != SCALE ** (level - 1) or (c1 - c0 + 1) != SCALE ** (level - 1):
            raise AssertionError("window size must shrink by the same factor at each level")
        black_comp_counts.append(len(_black_components(cur)))
        if level > 1:
            cur = _compress_uniform(cur, SCALE)
    if any(count < 2 for count in black_comp_counts):
        raise AssertionError("each generated input level must include at least one black distractor")


def _gen_attempt() -> np.ndarray | None:
    window, distractor = random.choice(_BASE_LAYOUTS)
    nonblack_count = SCALE * SCALE - 1 - len(distractor)
    n_colors = random.randint(2, min(4, nonblack_count))
    colors = random.sample(COLOR_POOL, n_colors)
    depth = random.choice(DEPTH_CHOICES)

    base = _build_base_grid(window, distractor, colors)
    inp = _scale_base(base, depth)
    expected = _explicit_output(base, depth, window)

    try:
        _verify_levels(base, depth)
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if depth > 2:
        one_step = inp.copy()
        wr, wc = window
        factor = SCALE ** (depth - 1)
        one_step[wr * factor : (wr + 1) * factor, wc * factor : (wc + 1) * factor] = _compress_uniform(inp, SCALE)
        if np.array_equal(one_step, expected):
            return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid recursive-window ARC input")
