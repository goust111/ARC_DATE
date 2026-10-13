import random

import numpy as np


BLACK = 0
GRAY = 5
MAGENTA = 6
PALETTE_COLORS = (1, 2, 3, 4, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "right", "bottom", "left")

GEN_ATTEMPTS = 400
PALETTE_LAYOUT_ATTEMPTS = 80
PATH_BUILD_ATTEMPTS = 120


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _is_boundary(r: int, c: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return r in (0, h - 1) or c in (0, w - 1)


def _is_corner(r: int, c: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return (r in (0, h - 1)) and (c in (0, w - 1))


def _side_of(r: int, c: int, shape: tuple[int, int]) -> str:
    h, w = shape
    if r == 0:
        return "top"
    if c == w - 1:
        return "right"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    raise AssertionError("cell is not on the boundary")


def _side_cells(side: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    raise ValueError(f"unknown side {side}")


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _path_component(grid: np.ndarray) -> np.ndarray:
    path_like = (grid == GRAY) | (grid == MAGENTA)
    comps = _connected_components(path_like)
    assert comps, "expected a gray/magenta path component"
    comps.sort(key=lambda comp: (-int(comp.shape[0]), int(comp[:, 0].min()), int(comp[:, 1].min())))
    best = comps[0]
    if len(comps) > 1:
        assert int(best.shape[0]) > int(comps[1].shape[0]), "path component must be uniquely largest"
    mask = np.zeros_like(path_like, dtype=bool)
    mask[best[:, 0], best[:, 1]] = True
    return mask


def _path_endpoints(path_mask: np.ndarray) -> list[tuple[int, int]]:
    endpoints: list[tuple[int, int]] = []
    for r, c in np.argwhere(path_mask):
        degree = sum(1 for nr, nc in _neighbors4(int(r), int(c), path_mask.shape) if path_mask[nr, nc])
        if degree == 1:
            endpoints.append((int(r), int(c)))
        else:
            assert degree == 2, "path must be a simple non-branching chain"
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    return endpoints


def _ordered_path(path_mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [
            (nr, nc)
            for nr, nc in _neighbors4(cur[0], cur[1], path_mask.shape)
            if path_mask[nr, nc] and (nr, nc) != prev
        ]
        if not nxts:
            break
        assert len(nxts) == 1, "path traversal must stay unique"
        prev, cur = cur, nxts[0]
        path.append(cur)
    assert len(path) == int(path_mask.sum()), "path traversal must cover every path cell exactly once"
    return path


def _palette_sequence(grid: np.ndarray, path_mask: np.ndarray) -> list[tuple[int, int]]:
    cols_and_colors: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color in (BLACK, GRAY):
            continue
        if path_mask[0, c]:
            continue
        cols_and_colors.append((c, color))
    assert cols_and_colors, "expected a non-empty top-row palette"
    return cols_and_colors


def _split_segments(path: list[tuple[int, int]], grid: np.ndarray) -> list[list[tuple[int, int]]]:
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for cell in path:
        current.append(cell)
        if int(grid[cell]) == MAGENTA:
            segments.append(current)
            current = []
    assert current, "path cannot end on a marker"
    segments.append(current)
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a boundary-to-boundary maze path segment by segment using the top-row palette order.

    Input structure:
    - The background is black.
    - One single-cell-wide gray path runs 4-connected from one boundary endpoint to another without branches; some path cells are magenta separators.
    - The top row contains an ordered sequence of isolated palette pixels in non-black, non-gray colors, separated by black cells and not connected to the path.

    Rule:
    - Extract the unique gray/magenta path component and find its two degree-1 endpoints.
    - Let A be the endpoint with smaller Manhattan distance to the leftmost palette pixel on row 0; orient the path from A to the other endpoint B.
    - Walk from A to B and cut the path whenever a magenta cell is reached, assigning that magenta cell to the segment that ends there.
    - Recolor segment i with the i-th palette color read left to right on the top row, while leaving the palette row unchanged.

    Key insight:
    - The top-row colors form a palette, the magenta cells are ordered segment delimiters, and the correct recoloring depends on traversing the path from the endpoint nearest the palette's left edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    path_mask = _path_component(g)
    palette = _palette_sequence(g, path_mask)
    endpoints = _path_endpoints(path_mask)

    ref = (0, palette[0][0])
    d0 = _manhattan(endpoints[0], ref)
    d1 = _manhattan(endpoints[1], ref)
    assert d0 != d1, "path direction must be uniquely determined"
    start = endpoints[0] if d0 < d1 else endpoints[1]

    ordered = _ordered_path(path_mask, start)
    segments = _split_segments(ordered, g)
    colors = [color for _col, color in palette]
    assert len(segments) == len(colors), "palette length must match the number of path segments"

    out = g.copy()
    for segment, color in zip(segments, colors):
        for r, c in segment:
            out[r, c] = color
    return out


def _touches_used_elsewhere(
    cell: tuple[int, int],
    used: set[tuple[int, int]],
    previous: tuple[int, int],
    shape: tuple[int, int],
) -> bool:
    for nr, nc in _neighbors4(cell[0], cell[1], shape):
        if (nr, nc) in used and (nr, nc) != previous:
            return True
    return False


def _build_path_exact(
    start: tuple[int, int],
    target: tuple[int, int],
    n_cells: int,
    shape: tuple[int, int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    path = [start]
    used = {start}

    def dfs() -> bool:
        cur = path[-1]
        remaining = n_cells - len(path)
        if cur == target:
            return remaining == 0
        if remaining == 0:
            return False

        candidates: list[tuple[int, int]] = []
        for dr, dc in DIRS4:
            nr, nc = cur[0] + dr, cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < shape[0] and 0 <= nc < shape[1]):
                continue
            if nxt in used:
                continue
            if nxt != target and _is_boundary(nr, nc, shape):
                continue
            if _touches_used_elsewhere(nxt, used, cur, shape):
                continue

            rem_after = remaining - 1
            dist = _manhattan(nxt, target)
            if dist > rem_after:
                continue
            if (rem_after - dist) % 2 != 0:
                continue
            if nxt == target and rem_after != 0:
                continue

            candidates.append(nxt)

        if remaining <= 4:
            candidates.sort(key=lambda cell: _manhattan(cell, target))
        else:
            rng.shuffle(candidates)

        for nxt in candidates:
            used.add(nxt)
            path.append(nxt)
            if dfs():
                return True
            path.pop()
            used.remove(nxt)
        return False

    return path if dfs() else None


def _random_composition(total: int, parts: int, minimum: int, rng: random.Random) -> list[int]:
    assert total >= parts * minimum
    extra = total - parts * minimum
    buckets = [minimum] * parts
    for _ in range(extra):
        buckets[rng.randrange(parts)] += 1
    rng.shuffle(buckets)
    return buckets


def _marker_indices(path_len: int, n_segments: int, rng: random.Random) -> list[int]:
    lengths = _random_composition(path_len, n_segments, minimum=2, rng=rng)
    indices: list[int] = []
    acc = 0
    for seg_len in lengths[:-1]:
        acc += seg_len
        indices.append(acc - 1)
    assert indices and all(0 < idx < path_len - 1 for idx in indices)
    return indices


def _choose_palette_positions(
    width: int,
    n_segments: int,
    top_path_cols: set[int],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
    desired_start_idx: int,
    rng: random.Random,
) -> list[int] | None:
    for _ in range(PALETTE_LAYOUT_ATTEMPTS):
        rel = [0]
        for _ in range(n_segments - 1):
            rel.append(rel[-1] + rng.choice((2, 3)))
        used_width = rel[-1] + 1
        if used_width > width:
            continue

        starts = list(range(width - used_width + 1))
        rng.shuffle(starts)
        for start_col in starts:
            positions = [start_col + offset for offset in rel]
            if any(any(abs(col - path_col) <= 1 for path_col in top_path_cols) for col in positions):
                continue

            ref = (0, positions[0])
            d0 = _manhattan(endpoints[0], ref)
            d1 = _manhattan(endpoints[1], ref)
            if d0 == d1:
                continue
            closer = 0 if d0 < d1 else 1
            if closer == desired_start_idx:
                return positions
    return None


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(10, 18)
    w = rng.randint(12, 24)
    shape = (h, w)

    n_segments = rng.randint(2, 5)
    side_order = list(SIDES)
    rng.shuffle(side_order)
    start_side, end_side = side_order[:2]
    start = rng.choice(_side_cells(start_side, shape))
    target = rng.choice(_side_cells(end_side, shape))

    base_steps = _manhattan(start, target)
    min_cells = 2 * n_segments + rng.randint(2, 6)
    possible_lengths = [
        base_steps + 1 + extra
        for extra in range(0, 13, 2)
        if base_steps + 1 + extra >= min_cells and base_steps + 1 + extra <= 24
    ]
    if not possible_lengths:
        return None
    n_cells = rng.choice(possible_lengths)

    path: list[tuple[int, int]] | None = None
    for _ in range(PATH_BUILD_ATTEMPTS):
        built = _build_path_exact(start, target, n_cells, shape, rng)
        if built is not None:
            path = built
            break
    if path is None:
        return None

    desired_start_idx = rng.randrange(2)
    top_path_cols = {c for r, c in path if r == 0}
    palette_positions = _choose_palette_positions(
        w,
        n_segments,
        top_path_cols,
        (start, target),
        desired_start_idx,
        rng,
    )
    if palette_positions is None:
        return None

    grid = np.full(shape, BLACK, dtype=int)
    for r, c in path:
        grid[r, c] = GRAY

    for idx in _marker_indices(len(path), n_segments, rng):
        mr, mc = path[idx]
        grid[mr, mc] = MAGENTA

    palette_colors = rng.sample(PALETTE_COLORS, n_segments)
    for col, color in zip(palette_positions, palette_colors):
        grid[0, col] = color

    # Re-run the solver to reject degenerate layouts such as ambiguous direction.
    out = main(grid)
    assert out.shape == grid.shape
    assert np.any(out != grid), "generated task must change under the rule"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
