from __future__ import annotations

import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MIRROR_TYPES = ("horizontal", "vertical", "diag", "anti")

GRID_ROWS = (12, 24)
GRID_COLS = (12, 24)
PAIR_COUNT_RANGE = (1, 3)
MASK_SIDE_RANGE = (3, 6)
MASK_AREA_RANGE = (4, 10)
SHIFT_LIMIT = 6
MASK_ATTEMPTS = 80
PAIR_ATTEMPTS = 120
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords + np.array([int(r0), int(c0)], dtype=int)


def _coords_in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    return (
        int(coords[:, 0].min()) >= 0
        and int(coords[:, 1].min()) >= 0
        and int(coords[:, 0].max()) < shape[0]
        and int(coords[:, 1].max()) < shape[1]
    )


def _cell_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _expanded_cells(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in _cell_set(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    expanded.add((nr, nc))
    return expanded


def _is_connected_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    cells = {tuple(map(int, rc)) for rc in coords}
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _mirror_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "horizontal":
        return np.flipud(mask)
    if kind == "vertical":
        return np.fliplr(mask)
    if kind == "diag":
        return mask.T.copy()
    assert kind == "anti"
    return np.fliplr(np.flipud(mask).T).copy()


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = {start}

    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        nxt = random.choice(tuple(frontier))
        cells.add(nxt)
        mask[nxt] = True

    return mask


def _sample_fragment(kind: str) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        if kind in {"diag", "anti"}:
            side = random.randint(*MASK_SIDE_RANGE)
            height = side
            width = side
        else:
            height = random.randint(*MASK_SIDE_RANGE)
            width = random.randint(*MASK_SIDE_RANGE)

        max_area = min(MASK_AREA_RANGE[1], height * width - 1)
        min_area = min(MASK_AREA_RANGE[0], max_area)
        if min_area > max_area:
            continue

        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(height, width, area)
        if int(mask.sum()) != area:
            continue
        if not _spans_bbox(mask):
            continue
        mirrored = _mirror_mask(mask, kind)
        if np.array_equal(mask, mirrored):
            continue
        union = mask | mirrored
        if not _is_connected_mask(union):
            continue
        if int(union.sum()) == int(mask.sum()):
            continue
        return mask
    return None


def _shift_candidates(height: int, width: int, grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    max_dr = min(SHIFT_LIMIT, (grid_shape[0] - height) // 2)
    max_dc = min(SHIFT_LIMIT, (grid_shape[1] - width) // 2)
    candidates = []
    for dr in range(-max_dr, max_dr + 1):
        for dc in range(-max_dc, max_dc + 1):
            if dr == 0 and dc == 0:
                continue
            if abs(dr) + abs(dc) < 2:
                continue
            candidates.append((dr, dc))
    random.shuffle(candidates)
    return candidates


def _place_pair(
    inp: np.ndarray,
    out: np.ndarray,
    color: int,
    kind: str,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
) -> bool:
    shape = inp.shape
    mask = _sample_fragment(kind)
    if mask is None:
        return False

    mirrored = _mirror_mask(mask, kind)
    union = mask | mirrored
    height, width = mask.shape

    for dr, dc in _shift_candidates(height, width, shape):
        row_lo = abs(dr)
        row_hi = shape[0] - height - abs(dr)
        col_lo = abs(dc)
        col_hi = shape[1] - width - abs(dc)
        if row_lo > row_hi or col_lo > col_hi:
            continue

        rows = list(range(row_lo, row_hi + 1))
        cols = list(range(col_lo, col_hi + 1))
        random.shuffle(rows)
        random.shuffle(cols)

        for r0 in rows:
            for c0 in cols:
                coords_a = _coords_from_mask(mask, r0 + dr, c0 + dc)
                coords_b = _coords_from_mask(mirrored, r0 - dr, c0 - dc)
                coords_out = _coords_from_mask(union, r0, c0)

                if not (_coords_in_bounds(coords_a, shape) and _coords_in_bounds(coords_b, shape)):
                    continue

                cells_a = _cell_set(coords_a)
                cells_b = _cell_set(coords_b)
                if cells_a & cells_b:
                    continue
                if cells_a & _expanded_cells(coords_b, shape):
                    continue

                if cells_a & input_forbidden or cells_b & input_forbidden:
                    continue
                if _cell_set(coords_out) & output_forbidden:
                    continue

                inp[coords_a[:, 0], coords_a[:, 1]] = color
                inp[coords_b[:, 0], coords_b[:, 1]] = color
                out[coords_out[:, 0], coords_out[:, 1]] = color
                input_forbidden.update(_expanded_cells(coords_a, shape))
                input_forbidden.update(_expanded_cells(coords_b, shape))
                output_forbidden.update(_expanded_cells(coords_out, shape))
                return True

    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each same-colored fragment pair into their shared midpoint so the two halves merge into one symmetric figure.

    Input structure:
    - The grid has one dominant background color.
    - Each non-background color appears in exactly two separated 4-connected components.
    - The two components of a color are matching fragments related by a reflection, but they are placed apart on the canvas.

    Rule:
    - For each foreground color, find its two 4-connected components and compute each component's bounding-box center.
    - Take the midpoint of those two centers, then translate both components rigidly toward that midpoint until their bounding-box centers coincide there.
    - Draw the translated copies on an otherwise background canvas; overlapping cells of the same color remain that color.

    Key insight:
    - The two fragments already encode a complete reflected figure; the task is to halve their separation and overlay them at the midpoint between their bounding-box centers.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = np.full_like(g, bg)

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    for color in colors:
        comps = _find_components_of_color(g, color)
        assert len(comps) == 2, "each foreground color must form exactly two components"

        bbox_a = _bbox(comps[0])
        bbox_b = _bbox(comps[1])
        center2_a = (bbox_a[0] + bbox_a[2], bbox_a[1] + bbox_a[3])
        center2_b = (bbox_b[0] + bbox_b[2], bbox_b[1] + bbox_b[3])

        diff_r2 = center2_b[0] - center2_a[0]
        diff_c2 = center2_b[1] - center2_a[1]
        assert diff_r2 % 4 == 0 and diff_c2 % 4 == 0, "pair midpoint must allow integer translations"

        shift_a = np.array([diff_r2 // 4, diff_c2 // 4], dtype=int)
        shift_b = -shift_a
        moved_a = comps[0] + shift_a
        moved_b = comps[1] + shift_b

        assert _coords_in_bounds(moved_a, g.shape) and _coords_in_bounds(moved_b, g.shape)
        out[moved_a[:, 0], moved_a[:, 1]] = color
        out[moved_b[:, 0], moved_b[:, 1]] = color

    return out


def _oracle_accepts(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_ROWS)
    width = random.randint(*GRID_COLS)
    bg = random.randrange(10)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    palette = [color for color in range(10) if color != bg]
    if len(palette) < pair_count:
        return None

    colors = random.sample(palette, pair_count)
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()

    for color in colors:
        kind = random.choice(MIRROR_TYPES)
        placed = False
        for _ in range(PAIR_ATTEMPTS):
            if _place_pair(inp, out, color, kind, input_forbidden, output_forbidden):
                placed = True
                break
        if not placed:
            return None

    if int(np.count_nonzero(inp != bg)) * 2 >= inp.size:
        return None

    if not _oracle_accepts(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
