from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

MARKER_COLORS = (RED, GREEN, YELLOW, PINK)
GRID_SIDE = 4
MAX_GENERATION_ATTEMPTS = 300

_GEN_CALL_COUNT = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _spans(size: int, separators: np.ndarray) -> list[tuple[int, int]]:
    seps = list(map(int, separators))
    assert len(seps) == 3
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in seps:
        spans.append((start, sep))
        start = sep + 1
    spans.append((start, size))
    assert len(spans) == GRID_SIDE and all(lo < hi for lo, hi in spans)
    return spans


def _separator_positions(cell_size: int) -> list[int]:
    return [cell_size, 2 * cell_size + 1, 3 * cell_size + 2]


def _classify_cell(cell: np.ndarray) -> dict:
    assert np.all(cell != GREY)
    non_black = np.argwhere(cell != BLACK)
    if non_black.size == 0:
        return {"kind": "empty"}

    colors = set(map(int, np.unique(cell))) - {BLACK}
    if colors == {BLUE}:
        comps = _components_mask(cell == BLUE)
        assert len(comps) == 1
        comp = comps[0]
        assert 3 <= comp.shape[0] <= 6
        comp_bbox = bbox(cell == BLUE)
        assert comp_bbox is not None
        return {"kind": "shape", "coords": comp, "bbox": comp_bbox}

    assert non_black.shape[0] == 1
    r, c = map(int, non_black[0])
    color = int(cell[r, c])
    assert color in MARKER_COLORS
    return {"kind": "marker", "pos": (r, c), "color": color}


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows = np.flatnonzero(np.all(g == GREY, axis=1))
    sep_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert len(sep_rows) == 3 and len(sep_cols) == 3
    assert np.all(np.diff(sep_rows) >= 2) and np.all(np.diff(sep_cols) >= 2)

    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)

    non_line = g[np.ix_(~np.isin(np.arange(g.shape[0]), sep_rows), ~np.isin(np.arange(g.shape[1]), sep_cols))]
    assert not np.any(non_line == GREY)

    shapes: dict[tuple[int, int], dict] = {}
    markers: dict[tuple[int, int], dict] = {}
    empties = 0
    for cr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            info = _classify_cell(g[r0:r1, c0:c1])
            info["cell"] = (cr, cc)
            info["top_left"] = (r0, c0)
            if info["kind"] == "shape":
                shapes[(cr, cc)] = info
            elif info["kind"] == "marker":
                markers[(cr, cc)] = info
            else:
                empties += 1

    assert len(shapes) == 4
    assert len(markers) == 4
    assert empties == GRID_SIDE * GRID_SIDE - 8
    return {"grid": g, "row_spans": row_spans, "col_spans": col_spans, "shapes": shapes, "markers": markers}


def _find_unique_marker(shape_cell: tuple[int, int], markers: dict[tuple[int, int], dict]) -> tuple[str, tuple[int, int], dict]:
    sr, sc = shape_cell
    aligned: list[tuple[str, tuple[int, int], dict]] = []
    for marker_cell, marker in markers.items():
        mr, mc = marker_cell
        if mr == sr and mc != sc:
            aligned.append(("row", marker_cell, marker))
        elif mc == sc and mr != sr:
            aligned.append(("col", marker_cell, marker))
    assert len(aligned) == 1
    return aligned[0]


def _direction_from_cells(shape_cell: tuple[int, int], marker_cell: tuple[int, int], axis: str) -> str:
    sr, sc = shape_cell
    mr, mc = marker_cell
    if axis == "row":
        return "left" if mc < sc else "right"
    return "up" if mr < sr else "down"


def _selected_half(coords: np.ndarray, comp_bbox: tuple[int, int, int, int], direction: str) -> np.ndarray:
    r0, c0, r1, c1 = comp_bbox
    if direction in ("up", "down"):
        height = r1 - r0 + 1
        assert height % 2 == 0
        split = r0 + height // 2
        selected = coords[:, 0] < split if direction == "up" else coords[:, 0] >= split
    else:
        width = c1 - c0 + 1
        assert width % 2 == 0
        split = c0 + width // 2
        selected = coords[:, 1] < split if direction == "left" else coords[:, 1] >= split
    assert np.any(selected) and np.any(~selected)
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the indicated half of each blue shape according to its aligned marker's table direction.

    Input structure:
    - The background is black, with three full grey separator rows and three full grey separator columns forming a 4x4 table.
    - Exactly four cells contain one 4-connected blue shape, exactly four other cells contain one non-blue single-pixel marker, and the remaining cells are empty.
    - For each shape cell, exactly one marker cell lies in the same table row or the same table column.

    Rule:
    - Recover the 4x4 cell indices from the full grey separator lines.
    - For each shape cell, find its unique aligned marker cell.
    - If the marker shares the table column, it selects a horizontal split of the shape bbox: marker above means recolor the upper half, marker below means recolor the lower half.
    - If the marker shares the table row, it selects a vertical split of the shape bbox: marker left means recolor the left half, marker right means recolor the right half.
    - Recolor only the blue pixels inside the chosen half to the marker color; keep the other blue pixels, the markers, the grey lines, and the background unchanged.

    Key insight:
    - The marker color is only half of the control signal: its relative table direction to the shape chooses both the split axis and which half of the bbox changes.
    """
    parsed = _parse_grid(grid)
    out = parsed["grid"].copy()

    for shape_cell, shape in parsed["shapes"].items():
        axis, marker_cell, marker = _find_unique_marker(shape_cell, parsed["markers"])
        direction = _direction_from_cells(shape_cell, marker_cell, axis)
        selected = _selected_half(shape["coords"], shape["bbox"], direction)
        top, left = shape["top_left"]
        rr = top + shape["coords"][:, 0]
        cc = left + shape["coords"][:, 1]
        out[rr[selected], cc[selected]] = marker["color"]

    return out


BASE_SHAPE_CELLS = [(0, 1), (1, 1), (3, 2), (3, 3)]
BASE_MARKER_CELLS = [(0, 0), (1, 0), (2, 2), (2, 3)]
LAYOUT_TRANSFORMS = ("identity", "flip_lr", "flip_ud", "rot180")

SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
]


def _transform_cell(cell: tuple[int, int], name: str) -> tuple[int, int]:
    r, c = cell
    if name == "identity":
        return r, c
    if name == "flip_lr":
        return r, GRID_SIDE - 1 - c
    if name == "flip_ud":
        return GRID_SIDE - 1 - r, c
    if name == "rot180":
        return GRID_SIDE - 1 - r, GRID_SIDE - 1 - c
    raise ValueError(name)


def _supports_direction(mask: np.ndarray, direction: str) -> bool:
    if direction in ("left", "right"):
        width = mask.shape[1]
        if width % 2 != 0:
            return False
        split = width // 2
        return bool(np.any(mask[:, :split]) and np.any(mask[:, split:]))
    height = mask.shape[0]
    if height % 2 != 0:
        return False
    split = height // 2
    return bool(np.any(mask[:split, :]) and np.any(mask[split:, :]))


def _random_shape(direction: str, used: set[tuple[bytes, tuple[int, int]]]) -> np.ndarray:
    candidates = [shape for shape in SHAPE_LIBRARY if _supports_direction(shape, direction)]
    random.shuffle(candidates)
    for base in candidates:
        for _ in range(4):
            shape = base.copy()
            if random.random() < 0.5:
                shape = np.flipud(shape)
            if random.random() < 0.5:
                shape = np.fliplr(shape)
            key = (shape.tobytes(), shape.shape)
            if key not in used:
                used.add(key)
                return shape.astype(int, copy=False)
    shape = random.choice(candidates).copy()
    if random.random() < 0.5:
        shape = np.flipud(shape)
    if random.random() < 0.5:
        shape = np.fliplr(shape)
    return shape.astype(int, copy=False)


def _build_grid(cell_h: int, cell_w: int) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    h = GRID_SIDE * cell_h + 3
    w = GRID_SIDE * cell_w + 3
    grid = np.zeros((h, w), dtype=int)
    sep_rows = _separator_positions(cell_h)
    sep_cols = _separator_positions(cell_w)
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY
    return grid, _spans(h, np.array(sep_rows)), _spans(w, np.array(sep_cols))


def _place_shape(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], cell: tuple[int, int], mask: np.ndarray) -> None:
    cr, cc = cell
    r0, r1 = row_spans[cr]
    c0, c1 = col_spans[cc]
    cell_h = r1 - r0
    cell_w = c1 - c0
    mh, mw = mask.shape
    assert mh <= cell_h and mw <= cell_w
    off_r = random.randint(0, cell_h - mh)
    off_c = random.randint(0, cell_w - mw)
    view = grid[r0 + off_r : r0 + off_r + mh, c0 + off_c : c0 + off_c + mw]
    view[mask.astype(bool)] = BLUE


def _place_marker(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], cell: tuple[int, int], color: int) -> None:
    cr, cc = cell
    r0, r1 = row_spans[cr]
    c0, c1 = col_spans[cc]
    rr = random.randint(r0, r1 - 1)
    cc0 = random.randint(c0, c1 - 1)
    grid[rr, cc0] = color


def _validate_generated(grid: np.ndarray) -> None:
    parsed = _parse_grid(grid)
    out = main(grid)
    axes = set()
    directions = set()

    for shape_cell, shape in parsed["shapes"].items():
        axis, marker_cell, marker = _find_unique_marker(shape_cell, parsed["markers"])
        direction = _direction_from_cells(shape_cell, marker_cell, axis)
        axes.add(axis)
        directions.add(direction)
        selected = _selected_half(shape["coords"], shape["bbox"], direction)
        top, left = shape["top_left"]
        rr = top + shape["coords"][:, 0]
        cc = left + shape["coords"][:, 1]
        assert np.all(out[rr[selected], cc[selected]] == marker["color"])
        assert np.all(out[rr[~selected], cc[~selected]] == BLUE)
        changed = out[rr, cc] != grid[rr, cc]
        assert np.array_equal(changed, selected)

    assert axes == {"row", "col"}
    assert len(directions) == 2
    assert np.array_equal(out[grid == GREY], grid[grid == GREY])

    marker_pixels = (grid != out) & (grid != BLUE)
    assert not np.any(marker_pixels)


def _gen_attempt() -> np.ndarray:
    global _GEN_CALL_COUNT
    cell_h = random.choice((3, 4))
    cell_w = random.choice((3, 4))
    grid, row_spans, col_spans = _build_grid(cell_h, cell_w)

    transform = LAYOUT_TRANSFORMS[_GEN_CALL_COUNT % len(LAYOUT_TRANSFORMS)]
    _GEN_CALL_COUNT += 1

    shape_cells = [_transform_cell(cell, transform) for cell in BASE_SHAPE_CELLS]
    marker_cells = [_transform_cell(cell, transform) for cell in BASE_MARKER_CELLS]
    directions = [
        _direction_from_cells(shape_cell, marker_cell, "row" if shape_cell[0] == marker_cell[0] else "col")
        for shape_cell, marker_cell in zip(shape_cells, marker_cells)
    ]

    colors = list(MARKER_COLORS)
    random.shuffle(colors)
    used_shapes: set[tuple[bytes, tuple[int, int]]] = set()

    for shape_cell, direction in zip(shape_cells, directions):
        mask = _random_shape(direction, used_shapes)
        _place_shape(grid, row_spans, col_spans, shape_cell, mask)

    for marker_cell, color in zip(marker_cells, colors):
        _place_marker(grid, row_spans, col_spans, marker_cell, color)

    _validate_generated(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
