import random
from collections import deque

import numpy as np


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Minimal 4-connected component finder, aligned with arc_utils (r, c) coordinates."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _component_count(grid: np.ndarray) -> int:
    return len(_components4(np.asarray(grid) != 0))


def _coarse_mask(grid: np.ndarray, scale: int) -> np.ndarray:
    """
    Downscale the whole input foreground mask to a scale x scale miniature.

    Each miniature cell represents one proportional row-bin and one proportional
    column-bin of the original grid; it is foreground iff that bin contains at
    least one foreground pixel. This keeps the geometry tied to the full-grid
    coordinate system while matching the required NxN block size.
    """
    g = np.asarray(grid)
    h, w = g.shape
    row_edges = np.linspace(0, h, scale + 1, dtype=int)
    col_edges = np.linspace(0, w, scale + 1, dtype=int)
    mini = np.zeros((scale, scale), dtype=bool)
    fg = g != 0
    for rr in range(scale):
        r0, r1 = int(row_edges[rr]), int(row_edges[rr + 1])
        for cc in range(scale):
            c0, c1 = int(col_edges[cc]), int(col_edges[cc + 1])
            mini[rr, cc] = bool(np.any(fg[r0:r1, c0:c1]))
    return mini


def main(grid: np.ndarray) -> np.ndarray:
    """Count foreground components, then expand each foreground pixel into a colored miniature self-copy.

    Input structure:
    - The grid is black-background and contains several separated monochrome foreground components.
    - Foreground components are interpreted with 4-connectivity in full-grid (r, c) coordinates.
    - The number of foreground components is the control value N.

    Rule:
    - Count all non-black 4-connected foreground components and call that count N.
    - Create an output of shape (N * input_rows, N * input_cols), so each input pixel owns one NxN output block.
    - Convert the whole input foreground mask into an NxN miniature by partitioning the input rows and columns into N proportional bins and marking a miniature cell foreground when its source bin contains any foreground pixel.
    - For each background input pixel, leave its output block black.
    - For each foreground input pixel, paste that NxN miniature into its output block and recolor every foreground cell of the miniature to the source pixel's component color.

    Key insight:
    - The same connected-component count controls both the output geometry and the resolution of the stamped self-similar miniature, while each source pixel contributes only its component color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    n = _component_count(g)
    assert n >= 1
    h, w = map(int, g.shape)
    mini = _coarse_mask(g, n)

    out = np.zeros((n * h, n * w), dtype=int)
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0:
                continue
            block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
            block[mini] = color
    return out


def _random_component_shape(area: int) -> np.ndarray:
    """Grow a small connected polyomino and normalize it to its tight bbox."""
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    shape = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        shape[r, c] = True
    return shape


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    fg = np.argwhere(mask)
    for r, c in fg:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _choose_component_areas(n: int) -> list[int] | None:
    total = random.randint(max(6, 2 * n), min(12, 5 * n))
    areas = []
    remain = total
    for i in range(n):
        left = n - i - 1
        lo = max(2, remain - 5 * left)
        hi = min(5, remain - 2 * left)
        if lo > hi:
            return None
        area = random.randint(lo, hi)
        areas.append(area)
        remain -= area
    random.shuffle(areas)
    return areas


def _choose_colors(n: int) -> list[int]:
    palette = [1, 2, 3]
    if n == 2:
        return random.sample(palette, 2)

    distinct = random.choice([2, 3])
    base = random.sample(palette, distinct)
    colors = [random.choice(base) for _ in range(n)]
    if len(set(colors)) == 1:
        colors[0] = base[0]
        colors[1] = base[-1]
    if len(set(colors)) == n and random.random() < 0.75:
        colors[-1] = colors[0]
    random.shuffle(colors)
    return colors


def _place_components(h: int, w: int, shapes: list[np.ndarray], colors: list[int]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)

    order = list(range(len(shapes)))
    order.sort(key=lambda idx: shapes[idx].size, reverse=True)
    for idx in order:
        shape = shapes[idx]
        sh, sw = map(int, shape.shape)
        if sh > h or sw > w:
            return None

        placements = []
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                sl = blocked[r0 : r0 + sh, c0 : c0 + sw]
                if np.any(sl & shape):
                    continue
                placements.append((r0, c0))
        if not placements:
            return None

        r0, c0 = random.choice(placements)
        color = int(colors[idx])
        for rr in range(sh):
            for cc in range(sw):
                if shape[rr, cc]:
                    grid[r0 + rr, c0 + cc] = color
                    occupied[r0 + rr, c0 + cc] = True
        blocked = _dilate8(occupied)
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    comps = _components4(g != 0)
    n = len(comps)
    if not (2 <= n <= 4):
        return False

    areas = sorted(len(comp) for comp in comps)
    if any(area < 2 or area > 5 for area in areas):
        return False
    if not (6 <= int(np.count_nonzero(g)) <= 12):
        return False

    # Input colors stay visible but are not a reliable shortcut for component count.
    used_colors = {int(v) for v in g.flat if int(v) != 0}
    if len(used_colors) < 2 or len(used_colors) > 3:
        return False
    if n >= 3 and len(used_colors) == n:
        return False

    mini = _coarse_mask(g, n)
    if not np.any(mini) or np.all(mini):
        return False
    density = float(np.mean(mini))
    if not (0.25 <= density <= 0.8):
        return False

    out = main(g)
    if out.shape != (n * g.shape[0], n * g.shape[1]) or max(out.shape) > 30:
        return False

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
            if int(g[r, c]) == 0:
                if np.any(block):
                    return False
            else:
                expected = np.zeros((n, n), dtype=int)
                expected[mini] = int(g[r, c])
                if not np.array_equal(block, expected):
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3, 4])
    max_side = min(7, 30 // n)
    h = random.randint(5, max_side)
    w = random.randint(5, max_side)

    areas = _choose_component_areas(n)
    if areas is None:
        return None

    shapes = [_random_component_shape(area) for area in areas]
    colors = _choose_colors(n)
    grid = _place_components(h, w, shapes, colors)
    if grid is None:
        return None

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
