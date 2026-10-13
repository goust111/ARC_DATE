import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MAX_GENERATION_ATTEMPTS = 400
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_DIM_RANGE = (3, 5)
OBJECT_AREA_RANGE = (4, 8)
ACCENT_COUNT_CAP = 3


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_shape() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*OBJECT_DIM_RANGE)
        w = random.randint(*OBJECT_DIM_RANGE)
        area = random.randint(OBJECT_AREA_RANGE[0], min(OBJECT_AREA_RANGE[1], h * w))

        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        cells = [seed]

        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            seen_frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in seen_frontier:
                        frontier.append((nr, nc))
                        seen_frontier.add((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            mask[cell] = True
            cells.append(cell)

        if len(cells) != area:
            continue

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        return mask
    raise RuntimeError("failed to sample a valid object mask")


def _arrow_relative_cells(mask: np.ndarray, axis: str, side: str) -> list[tuple[int, int]] | None:
    h, w = mask.shape
    occupied = np.argwhere(mask)
    rows = occupied[:, 0]
    cols = occupied[:, 1]

    if axis == "vertical":
        edge_col = int(cols.max()) if side == "left" else int(cols.min())
        anchors = [r for r in range(1, h - 1) if mask[r, edge_col]]
        if not anchors:
            return None
        anchor = random.choice(anchors)
        if side == "left":
            return [(anchor, w + 1), (anchor - 1, w), (anchor + 1, w)]
        return [(anchor, -2), (anchor - 1, -1), (anchor + 1, -1)]

    edge_row = int(rows.max()) if side == "top" else int(rows.min())
    anchors = [c for c in range(1, w - 1) if mask[edge_row, c]]
    if not anchors:
        return None
    anchor = random.choice(anchors)
    if side == "top":
        return [(h + 1, anchor), (h, anchor - 1), (h, anchor + 1)]
    return [(-2, anchor), (-1, anchor - 1), (-1, anchor + 1)]


def _find_mirror_line(grid: np.ndarray) -> tuple[str, int]:
    grey_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    grey_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if len(grey_rows) == 1 and not grey_cols:
        return "horizontal", grey_rows[0]
    if len(grey_cols) == 1 and not grey_rows:
        return "vertical", grey_cols[0]
    raise AssertionError("expected exactly one full grey mirror line")


def _source_side(grid: np.ndarray, axis: str, line_idx: int) -> str:
    if axis == "vertical":
        left = int(np.count_nonzero(grid[:, :line_idx] == MAROON))
        right = int(np.count_nonzero(grid[:, line_idx + 1 :] == MAROON))
        if (left > 0) ^ (right > 0):
            return "left" if left > 0 else "right"
    else:
        top = int(np.count_nonzero(grid[:line_idx, :] == MAROON))
        bottom = int(np.count_nonzero(grid[line_idx + 1 :, :] == MAROON))
        if (top > 0) ^ (bottom > 0):
            return "top" if top > 0 else "bottom"
    raise AssertionError("maroon arrows must occupy exactly one side of the mirror line")


def _mirror_coord(axis: str, line_idx: int, r: int, c: int) -> tuple[int, int]:
    if axis == "vertical":
        return r, 2 * line_idx - c
    return 2 * line_idx - r, c


def _object_mask_on_source(grid: np.ndarray, axis: str, line_idx: int, side: str) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    if axis == "vertical":
        if side == "left":
            mask[:, :line_idx] = True
        else:
            mask[:, line_idx + 1 :] = True
    else:
        if side == "top":
            mask[:line_idx, :] = True
        else:
            mask[line_idx + 1 :, :] = True
    special = (grid == BLACK) | (grid == GREY) | (grid == MAROON)
    return mask & ~special


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each arrow-marked two-color object across the central grey axis and swap its colors in the reflected copy.

    Input structure:
    - The background is black.
    - A one-pixel-thick grey line spans the entire middle row or middle column.
    - One side of the grey line contains several small connected objects, each made from exactly two non-black, non-grey colors.
    - Each source object has a three-pixel maroon arrow on its mirror-facing side, while the opposite side is otherwise empty.

    Rule:
    - Keep the original objects and the grey mirror line unchanged.
    - Remove every maroon arrow back to black.
    - Reflect each source-side object across the grey line onto the opposite side.
    - In the reflected copy, swap the object's two colors so every pixel of one object color becomes the other.

    Key insight:
    - The spatial transformation is a plain mirror reflection, but the reflected copy also inverts the object's internal two-color assignment.
    """
    g = np.asarray(grid, dtype=int)
    axis, line_idx = _find_mirror_line(g)
    side = _source_side(g, axis, line_idx)
    source_mask = _object_mask_on_source(g, axis, line_idx, side)

    out = g.copy()
    out[out == MAROON] = BLACK

    for comp in _components(source_mask):
        colors = sorted({int(g[r, c]) for r, c in comp})
        assert len(colors) == 2, "each source object must use exactly two colors"
        a, b = colors
        for r, c in comp:
            rr, cc = _mirror_coord(axis, line_idx, r, c)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]
            if axis == "vertical":
                assert (side == "left" and cc > line_idx) or (side == "right" and cc < line_idx)
            else:
                assert (side == "top" and rr > line_idx) or (side == "bottom" and rr < line_idx)
            out[rr, cc] = b if int(g[r, c]) == a else a

    return out


def _all_neighbors8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked.add((rr, cc))
    return blocked


def _absolute_cells(
    top: int,
    left: int,
    mask: np.ndarray,
    colors: tuple[int, int],
    arrow_rel: list[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    coords = np.argwhere(mask)
    accent_hi = min(ACCENT_COUNT_CAP, max(1, (len(coords) - 1) // 2))
    accent_count = random.randint(1, accent_hi)
    accent_indices = {int(idx) for idx in random.sample(range(len(coords)), accent_count)}
    a, b = colors

    object_cells: list[tuple[int, int, int]] = []
    for idx, (r, c) in enumerate(coords):
        color = b if idx in accent_indices else a
        object_cells.append((top + int(r), left + int(c), color))

    arrow_cells = [(top + int(r), left + int(c)) for r, c in arrow_rel]
    return object_cells, arrow_cells


def _build_expected(
    inp: np.ndarray,
    axis: str,
    line_idx: int,
    placed: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    out[out == MAROON] = BLACK
    for item in placed:
        a, b = item["pair"]
        for r, c, color in item["object_cells"]:
            rr, cc = _mirror_coord(axis, line_idx, int(r), int(c))
            out[rr, cc] = b if int(color) == a else a
    return out


def _dims_for_layout(axis: str, count: int) -> tuple[int, int]:
    if axis == "vertical":
        h = random.randint(max(13, 4 * count + 5), 29)
        w = random.randrange(15, 30, 2)
        return h, w
    h = random.randrange(15, 30, 2)
    w = random.randint(max(13, 4 * count + 5), 29)
    return h, w


def _side_choices(axis: str) -> tuple[str, str]:
    return ("left", "right") if axis == "vertical" else ("top", "bottom")


def _feasible_position(
    grid_shape: tuple[int, int],
    axis: str,
    line_idx: int,
    side: str,
    mask: np.ndarray,
) -> tuple[range, range]:
    h, w = grid_shape
    mh, mw = mask.shape

    if axis == "vertical":
        row_range = range(0, h - mh + 1)
        if side == "left":
            col_range = range(0, line_idx - mw - 1)
        else:
            col_range = range(line_idx + 3, w - mw + 1)
        return row_range, col_range

    col_range = range(0, w - mw + 1)
    if side == "top":
        row_range = range(0, line_idx - mh - 1)
    else:
        row_range = range(line_idx + 3, h - mh + 1)
    return row_range, col_range


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(("horizontal", "vertical"))
    count = random.randint(*OBJECT_COUNT_RANGE)
    h, w = _dims_for_layout(axis, count)
    side = random.choice(_side_choices(axis))

    grid = np.full((h, w), BLACK, dtype=int)
    line_idx = w // 2 if axis == "vertical" else h // 2
    if axis == "vertical":
        grid[:, line_idx] = GREY
    else:
        grid[line_idx, :] = GREY

    all_pairs = [(a, b) for i, a in enumerate(OBJECT_COLORS) for b in OBJECT_COLORS[i + 1 :]]
    random.shuffle(all_pairs)
    blocked: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    for pair in all_pairs[:count]:
        placed_one = False
        for _ in range(200):
            ordered_pair = pair if random.random() < 0.5 else (pair[1], pair[0])
            mask = _random_shape()
            arrow_rel = _arrow_relative_cells(mask, axis, side)
            if arrow_rel is None:
                continue

            row_range, col_range = _feasible_position(grid.shape, axis, line_idx, side, mask)
            if not row_range or not col_range:
                return None
            top = random.choice(tuple(row_range))
            left = random.choice(tuple(col_range))

            object_cells, arrow_cells = _absolute_cells(top, left, mask, ordered_pair, arrow_rel)
            semantic_cells = {(r, c) for r, c, _ in object_cells} | set(arrow_cells)
            if any(grid[r, c] != BLACK for r, c in semantic_cells):
                continue
            if semantic_cells & blocked:
                continue

            mirror_cells = {_mirror_coord(axis, line_idx, r, c) for r, c, _ in object_cells}
            if any(grid[rr, cc] != BLACK for rr, cc in mirror_cells):
                continue

            for r, c, color in object_cells:
                grid[r, c] = color
            for r, c in arrow_cells:
                grid[r, c] = MAROON

            blocked |= _all_neighbors8(semantic_cells, grid.shape)
            placed.append({"pair": ordered_pair, "object_cells": object_cells, "arrow_cells": arrow_cells})
            placed_one = True
            break
        if not placed_one:
            return None

    expected = _build_expected(grid, axis, line_idx, placed)
    if np.array_equal(grid, expected):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-swap ARC input")
