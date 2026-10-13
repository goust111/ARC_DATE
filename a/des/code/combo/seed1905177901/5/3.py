from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    assert coords.size > 0, "empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    trimmed = _trim_mask(mask)
    return trimmed.tobytes(), tuple(map(int, trimmed.shape))


def _is_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _spans_from_lines(lines: list[int], size: int) -> list[tuple[int, int]]:
    if not lines:
        return []
    borders = []
    if lines[0] != 0:
        borders.append(-1)
    borders.extend(lines)
    if lines[-1] != size - 1:
        borders.append(size)

    spans = []
    for a, b in zip(borders, borders[1:]):
        if b - a <= 1:
            return []
        spans.append((int(a + 1), int(b - 1)))
    return spans


def _find_structure(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    best = None
    for color in map(int, np.unique(g)):
        row_lines = list(map(int, np.flatnonzero(np.all(g == color, axis=1))))
        col_lines = list(map(int, np.flatnonzero(np.all(g == color, axis=0))))
        if len(row_lines) < 2 or len(col_lines) < 2:
            continue

        row_spans = _spans_from_lines(row_lines, g.shape[0])
        col_spans = _spans_from_lines(col_lines, g.shape[1])
        if len(row_spans) not in (3, 4) or len(row_spans) != len(col_spans):
            continue

        row_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_heights) != 1 or len(col_widths) != 1:
            continue

        score = len(row_lines) + len(col_lines)
        candidate = (score, color, row_spans, col_spans)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None, "failed to detect divider framework"
    return int(best[1]), best[2], best[3]


def _infer_bg(grid: np.ndarray, divider: int, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> int:
    counts: Counter[int] = Counter()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            values, freqs = np.unique(patch, return_counts=True)
            for value, freq in zip(values, freqs):
                value = int(value)
                if value != divider:
                    counts[value] += int(freq)
    assert counts, "no cell interior pixels found"
    return counts.most_common(1)[0][0]


def _extract_object(cell: np.ndarray, bg: int) -> dict:
    cell = np.asarray(cell, dtype=int)
    fg_mask = cell != bg
    assert fg_mask.any(), "cell is empty"
    colors = {int(v) for v in np.unique(cell[fg_mask])}
    assert len(colors) == 1, "cell must contain a single monochrome object"
    assert _is_connected(fg_mask), "object must be 4-connected"

    coords = np.argwhere(fg_mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = fg_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "mask": mask,
        "color": next(iter(colors)),
        "shape_key": _shape_key(mask),
    }


def _transform_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hflip":
        return np.fliplr(mask)
    if kind == "vflip":
        return np.flipud(mask)
    if kind == "rot180":
        return np.rot90(mask, 2)
    raise AssertionError(f"unknown transform {kind}")


def _transform_cell(cell: np.ndarray, obj: dict, bg: int, kind: str) -> np.ndarray:
    out = np.array(cell, copy=True)
    r0, c0, r1, c1 = obj["bbox"]
    new_bbox = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    new_bbox[_transform_mask(obj["mask"], kind)] = int(obj["color"])
    out[r0 : r1 + 1, c0 : c1 + 1] = new_bbox
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each cell object according to whether its shape matches the cell above
    or the cell to the left inside a divider-defined grid.
    """
    g = np.asarray(grid, dtype=int)
    divider, row_spans, col_spans = _find_structure(g)
    bg = _infer_bg(g, divider, row_spans, col_spans)

    objects: list[list[dict]] = []
    for r0, r1 in row_spans:
        row = []
        for c0, c1 in col_spans:
            row.append(_extract_object(g[r0 : r1 + 1, c0 : c1 + 1], bg))
        objects.append(row)

    out = g.copy()
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            if i == 0 or j == 0:
                continue

            current = objects[i][j]
            above = objects[i - 1][j]
            left = objects[i][j - 1]

            if current["shape_key"] == above["shape_key"]:
                kind = "hflip"
            elif current["shape_key"] == left["shape_key"]:
                kind = "vflip"
            else:
                kind = "rot180"

            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            out[r0 : r1 + 1, c0 : c1 + 1] = _transform_cell(cell, current, bg, kind)

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_DIVIDER = 9
GEN_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
GEN_GRID_ORDERS = (3, 4)
GEN_CELL_SIZE_RANGE = (5, 7)
GEN_RETRIES = 400

_BASE_SHAPES = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
)


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    cur = np.asarray(mask, dtype=bool)
    for _ in range(4):
        variants.append(cur)
        variants.append(np.fliplr(cur))
        cur = np.rot90(cur)
    return variants


def _good_shape(mask: np.ndarray, max_h: int, max_w: int) -> bool:
    mask = _trim_mask(mask)
    if mask.shape[0] > max_h or mask.shape[1] > max_w:
        return False
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    if int(mask.sum()) < 3:
        return False
    if not _is_connected(mask):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    return True


def _shape_library(max_h: int, max_w: int) -> list[np.ndarray]:
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    library = []
    for base in _BASE_SHAPES:
        for variant in _all_variants(base):
            trimmed = _trim_mask(variant)
            if not _good_shape(trimmed, max_h, max_w):
                continue
            key = _shape_key(trimmed)
            if key in seen:
                continue
            seen.add(key)
            library.append(trimmed)
    assert len(library) >= 6, "shape library unexpectedly too small"
    return library


def _pick_index(pool_size: int, forbidden: set[int], rng: random.Random) -> int:
    choices = [idx for idx in range(pool_size) if idx not in forbidden]
    assert choices, "no available shape choice"
    return int(rng.choice(choices))


def _build_shape_ids(order: int, pool_size: int, rng: random.Random) -> list[list[int]]:
    ids = [[-1] * order for _ in range(order)]
    ids[0][0] = rng.randrange(pool_size)
    ids[0][1] = _pick_index(pool_size, {ids[0][0]}, rng)
    ids[1][0] = _pick_index(pool_size, {ids[0][0], ids[0][1]}, rng)

    for j in range(2, order):
        forbidden = {ids[0][j - 1]}
        if j == 2:
            forbidden.add(ids[1][0])
        ids[0][j] = _pick_index(pool_size, forbidden, rng)

    for i in range(2, order):
        ids[i][0] = _pick_index(pool_size, {ids[i - 1][0]}, rng)

    ids[1][1] = ids[1][0]  # vertical branch: same as left, different from above
    ids[1][2] = ids[0][2]  # horizontal branch: same as above
    ids[2][1] = _pick_index(pool_size, {ids[1][1], ids[2][0]}, rng)  # rotate branch

    for i in range(1, order):
        for j in range(1, order):
            if ids[i][j] != -1:
                continue
            above = ids[i - 1][j]
            left = ids[i][j - 1]
            branches = ["hflip", "rot180"]
            if left != above:
                branches.append("vflip")
            branch = rng.choice(branches)
            if branch == "hflip":
                ids[i][j] = above
            elif branch == "vflip":
                ids[i][j] = left
            else:
                ids[i][j] = _pick_index(pool_size, {above, left}, rng)

    return ids


def _branch_kinds(shape_ids: list[list[int]]) -> set[str]:
    branches = set()
    for i in range(1, len(shape_ids)):
        for j in range(1, len(shape_ids[0])):
            if shape_ids[i][j] == shape_ids[i - 1][j]:
                branches.add("hflip")
            elif shape_ids[i][j] == shape_ids[i][j - 1]:
                branches.add("vflip")
            else:
                branches.add("rot180")
    return branches


def _cell_origin(i: int, j: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return 1 + i * (cell_h + 1), 1 + j * (cell_w + 1)


def _paint_mask(patch: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    bbox = np.array(patch[top : top + h, left : left + w], copy=True)
    bbox[mask] = color
    patch[top : top + h, left : left + w] = bbox


def _expected_output(grid: np.ndarray, bg: int, shape_ids: list[list[int]], cells: list[list[dict]], cell_h: int, cell_w: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    order = len(shape_ids)
    for i in range(1, order):
        for j in range(1, order):
            if shape_ids[i][j] == shape_ids[i - 1][j]:
                kind = "hflip"
            elif shape_ids[i][j] == shape_ids[i][j - 1]:
                kind = "vflip"
            else:
                kind = "rot180"

            r0, c0 = _cell_origin(i, j, cell_h, cell_w)
            cell = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = _transform_cell(cell, cells[i][j], bg, kind)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    order = rng.choice(GEN_GRID_ORDERS)
    cell_h = rng.randint(*GEN_CELL_SIZE_RANGE)
    cell_w = rng.randint(*GEN_CELL_SIZE_RANGE)

    library = _shape_library(cell_h - 2, cell_w - 2)
    shape_ids = _build_shape_ids(order, len(library), rng)
    if _branch_kinds(shape_ids) != {"hflip", "vflip", "rot180"}:
        return None

    height = order * cell_h + order + 1
    width = order * cell_w + order + 1
    grid = np.full((height, width), GEN_BG, dtype=int)

    for r in range(0, height, cell_h + 1):
        grid[r, :] = GEN_DIVIDER
    for c in range(0, width, cell_w + 1):
        grid[:, c] = GEN_DIVIDER

    cells: list[list[dict]] = []
    for i in range(order):
        row = []
        for j in range(order):
            mask = np.array(library[shape_ids[i][j]], copy=True)
            h, w = mask.shape
            top = rng.randint(1, cell_h - h - 1)
            left = rng.randint(1, cell_w - w - 1)
            color = rng.choice(GEN_COLORS)

            r0, c0 = _cell_origin(i, j, cell_h, cell_w)
            patch = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            _paint_mask(patch, mask, top, left, int(color))

            row.append(
                {
                    "bbox": (top, left, top + h - 1, left + w - 1),
                    "mask": mask,
                    "color": int(color),
                    "shape_key": _shape_key(mask),
                }
            )
        cells.append(row)

    expected = _expected_output(grid, GEN_BG, shape_ids, cells, cell_h, cell_w)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 3")
