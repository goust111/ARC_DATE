from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
CORNER_POOL = (1, 2, 3, 4, 6, 7, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_TO_INDEX = {name: idx for idx, name in enumerate(SIDE_ORDER)}
SIDE_COMBOS = {
    count: [tuple(combo) for combo in combinations(SIDE_ORDER, count)]
    for count in range(4)
}
RNG = np.random.default_rng()
GEN_ATTEMPTS = 300
OBJECT_ATTEMPTS = 80


def _majority_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _corner_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[1:3, 1:3] = True
    mask[1:3, width - 3 : width - 1] = True
    mask[height - 3 : height - 1, 1:3] = True
    mask[height - 3 : height - 1, width - 3 : width - 1] = True
    return mask


def _corner_colors(grid: np.ndarray) -> list[int]:
    height, width = grid.shape
    samples = [
        grid[1:3, 1:3],
        grid[1:3, width - 3 : width - 1],
        grid[height - 3 : height - 1, width - 3 : width - 1],
        grid[height - 3 : height - 1, 1:3],
    ]
    return [_majority_color(sample.reshape(-1)) for sample in samples]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue: deque[tuple[int, int]] = deque([(r, c)])
        cells: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            cells.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.array(cells, dtype=int))
    return components


def _touch_count(coords: np.ndarray, height: int, width: int) -> int:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int((rows == 1).any())
        + int((cols == width - 2).any())
        + int((rows == height - 2).any())
        + int((cols == 1).any())
    )


def _object_components(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    height, width = grid.shape
    search_mask = np.ones((height, width), dtype=bool)
    search_mask[[0, height - 1], :] = False
    search_mask[:, [0, width - 1]] = False
    search_mask &= ~_corner_mask(height, width)
    background = _majority_color(grid[search_mask])
    object_mask = search_mask & (grid != background)
    return background, _find_components(object_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each interior object from the corner sample that matches how many frame sides it touches.

    Input structure:
    - The background fills the interior, and the entire outer ring is a 1-pixel frame.
    - Four 2x2 colored sample blocks sit just inside the frame at the four corners.
    - Several same-colored 4-connected foreground objects lie inside the frame, and some of them touch one or more frame sides by 4-neighbor adjacency.

    Rule:
    - For each foreground object, count how many distinct frame sides it touches: top, right, bottom, and left.
    - Use that count to choose a corner sample color in clockwise order: 0 touches -> top-left, 1 -> top-right, 2 -> bottom-right, 3 -> bottom-left.
    - Repaint every cell of the object with the selected sample color while leaving the frame, the corner samples, and the background unchanged.

    Key insight:
    - The objects are classified only by the number of frame sides they contact, and the four corner blocks encode the recoloring map for counts 0 through 3.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    sample_colors = _corner_colors(g)
    _background, components = _object_components(g)

    out = g.copy()
    for coords in components:
        color = sample_colors[_touch_count(coords, height, width)]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _buffer_mask(occupied: np.ndarray) -> np.ndarray:
    buf = occupied.copy()
    buf[1:, :] |= occupied[:-1, :]
    buf[:-1, :] |= occupied[1:, :]
    buf[:, 1:] |= occupied[:, :-1]
    buf[:, :-1] |= occupied[:, 1:]
    return buf


def _base_allowed(height: int, width: int, sides: tuple[str, ...]) -> np.ndarray:
    allowed = np.zeros((height, width), dtype=bool)
    allowed[1 : height - 1, 1 : width - 1] = True
    allowed &= ~_corner_mask(height, width)

    side_set = set(sides)
    if "top" not in side_set:
        allowed[1, :] = False
    if "right" not in side_set:
        allowed[:, width - 2] = False
    if "bottom" not in side_set:
        allowed[height - 2, :] = False
    if "left" not in side_set:
        allowed[:, 1] = False
    return allowed


def _pick_anchor(available: np.ndarray, side: str) -> tuple[int, int] | None:
    height, width = available.shape
    if side == "top":
        band = available[1, 4 : width - 4]
        cols = np.flatnonzero(band)
        if cols.size == 0:
            return None
        return 1, int(cols[int(RNG.integers(cols.size))] + 4)
    if side == "right":
        band = available[4 : height - 4, width - 2]
        rows = np.flatnonzero(band)
        if rows.size == 0:
            return None
        return int(rows[int(RNG.integers(rows.size))] + 4), width - 2
    if side == "bottom":
        band = available[height - 2, 4 : width - 4]
        cols = np.flatnonzero(band)
        if cols.size == 0:
            return None
        return height - 2, int(cols[int(RNG.integers(cols.size))] + 4)
    band = available[4 : height - 4, 1]
    rows = np.flatnonzero(band)
    if rows.size == 0:
        return None
    return int(rows[int(RNG.integers(rows.size))] + 4), 1


def _pick_seed(available: np.ndarray) -> tuple[int, int] | None:
    candidates = np.argwhere(available)
    if candidates.size == 0:
        return None
    idx = int(RNG.integers(len(candidates)))
    r, c = candidates[idx]
    return int(r), int(c)


def _strict_interior(available: np.ndarray) -> np.ndarray:
    interior = available.copy()
    interior[1, :] = False
    interior[-2, :] = False
    interior[:, 1] = False
    interior[:, -2] = False
    return interior


def _draw_segment(shape_mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        shape_mask[r0, min(c0, c1) : max(c0, c1) + 1] = True
    else:
        shape_mask[min(r0, r1) : max(r0, r1) + 1, c0] = True


def _draw_path(shape_mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    if RNG.random() < 0.5:
        _draw_segment(shape_mask, (r0, c0), (r0, c1))
        _draw_segment(shape_mask, (r0, c1), (r1, c1))
    else:
        _draw_segment(shape_mask, (r0, c0), (r1, c0))
        _draw_segment(shape_mask, (r1, c0), (r1, c1))


def _grow_shape(shape_mask: np.ndarray, available: np.ndarray, target_size: int) -> None:
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(shape_mask):
        r = int(r)
        c = int(c)
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < available.shape[0] and 0 <= nc < available.shape[1]:
                if available[nr, nc] and not shape_mask[nr, nc]:
                    frontier.add((nr, nc))

    while int(shape_mask.sum()) < target_size and frontier:
        picks = tuple(frontier)
        pick = picks[int(RNG.integers(len(picks)))]
        frontier.remove(pick)
        if shape_mask[pick] or not available[pick]:
            continue
        shape_mask[pick] = True
        for dr, dc in DIRS4:
            nr = pick[0] + dr
            nc = pick[1] + dc
            if 0 <= nr < available.shape[0] and 0 <= nc < available.shape[1]:
                if available[nr, nc] and not shape_mask[nr, nc]:
                    frontier.add((nr, nc))


def _shape_fits(shape_mask: np.ndarray, available: np.ndarray) -> bool:
    return bool(np.all(~shape_mask | available))


def _full_column_choices(available: np.ndarray) -> np.ndarray:
    height, width = available.shape
    cols = [c for c in range(3, width - 3) if bool(available[1 : height - 1, c].all())]
    return np.array(cols, dtype=int)


def _full_row_choices(available: np.ndarray) -> np.ndarray:
    height, width = available.shape
    rows = [r for r in range(3, height - 3) if bool(available[r, 1 : width - 1].all())]
    return np.array(rows, dtype=int)


def _pick_from(values: np.ndarray) -> int | None:
    if values.size == 0:
        return None
    return int(values[int(RNG.integers(values.size))])


def _base_shape(height: int, width: int, sides: tuple[str, ...], available: np.ndarray) -> np.ndarray | None:
    shape_mask = np.zeros((height, width), dtype=bool)
    interior = _strict_interior(available)
    side_set = set(sides)

    if len(sides) == 0:
        seed = _pick_seed(interior)
        if seed is None:
            return None
        shape_mask[seed] = True
        return shape_mask

    if len(sides) == 1:
        anchor = _pick_anchor(available, sides[0])
        pivot = _pick_seed(interior)
        if anchor is None or pivot is None:
            return None
        _draw_path(shape_mask, anchor, pivot)
        return shape_mask

    if len(sides) == 2 and side_set == {"top", "bottom"}:
        col = _pick_from(_full_column_choices(available))
        if col is None:
            return None
        shape_mask[1 : height - 1, col] = True
        return shape_mask

    if len(sides) == 2 and side_set == {"left", "right"}:
        row = _pick_from(_full_row_choices(available))
        if row is None:
            return None
        shape_mask[row, 1 : width - 1] = True
        return shape_mask

    if len(sides) == 2:
        pivot = _pick_seed(interior)
        if pivot is None:
            return None
        for side in sides:
            anchor = _pick_anchor(available, side)
            if anchor is None:
                return None
            _draw_path(shape_mask, anchor, pivot)
        return shape_mask

    missing = next(side for side in SIDE_ORDER if side not in side_set)
    if missing in {"left", "right"}:
        col = _pick_from(_full_column_choices(available))
        if col is None:
            return None
        edge_col = 1 if missing == "right" else width - 2
        rows = [
            r
            for r in range(4, height - 4)
            if bool(available[r, min(col, edge_col) : max(col, edge_col) + 1].all())
        ]
        row = _pick_from(np.array(rows, dtype=int))
        if row is None:
            return None
        shape_mask[1 : height - 1, col] = True
        shape_mask[row, min(col, edge_col) : max(col, edge_col) + 1] = True
        return shape_mask

    row = _pick_from(_full_row_choices(available))
    if row is None:
        return None
    edge_row = height - 2 if missing == "top" else 1
    cols = [
        c
        for c in range(4, width - 4)
        if bool(available[min(row, edge_row) : max(row, edge_row) + 1, c].all())
    ]
    col = _pick_from(np.array(cols, dtype=int))
    if col is None:
        return None
    shape_mask[row, 1 : width - 1] = True
    shape_mask[min(row, edge_row) : max(row, edge_row) + 1, col] = True
    return shape_mask


def _build_object(
    height: int, width: int, sides: tuple[str, ...], occupied: np.ndarray
) -> np.ndarray | None:
    base_allowed = _base_allowed(height, width, sides)
    available = base_allowed & ~_buffer_mask(occupied)
    if not available.any():
        return None

    shape_mask = _base_shape(height, width, sides, available)
    if shape_mask is None or not _shape_fits(shape_mask, available):
        return None
    base_size = int(shape_mask.sum())
    extra = int(RNG.integers(1, 5 + len(sides)))
    target_size = base_size + extra
    _grow_shape(shape_mask, available, target_size)
    coords = np.argwhere(shape_mask)
    if coords.size == 0:
        return None
    if _touch_count(coords, height, width) != len(sides):
        return None
    return shape_mask


def _validate_counts(grid: np.ndarray, expected_counts: list[int]) -> bool:
    _background, components = _object_components(grid)
    if len(components) != len(expected_counts):
        return False
    actual = sorted(_touch_count(coords, *grid.shape) for coords in components)
    return actual == sorted(expected_counts)


def _gen_attempt() -> np.ndarray | None:
    height = int(RNG.integers(16, 27))
    width = int(RNG.integers(16, 27))
    grid = np.full((height, width), BLACK, dtype=int)
    grid[[0, height - 1], :] = GREY
    grid[:, [0, width - 1]] = GREY

    sample_colors = list(RNG.choice(CORNER_POOL, size=4, replace=False))
    grid[1:3, 1:3] = sample_colors[0]
    grid[1:3, width - 3 : width - 1] = sample_colors[1]
    grid[height - 3 : height - 1, width - 3 : width - 1] = sample_colors[2]
    grid[height - 3 : height - 1, 1:3] = sample_colors[3]

    extra_count = int(RNG.integers(0, 3))
    counts = [0, 1, 2, 3] + list(RNG.integers(0, 4, size=extra_count))
    specs = [
        (count, SIDE_COMBOS[count][int(RNG.integers(len(SIDE_COMBOS[count])))])
        for count in counts
    ]
    RNG.shuffle(specs)
    specs.sort(key=lambda item: -len(item[1]))

    occupied = np.zeros((height, width), dtype=bool)
    for count, sides in specs:
        shape = None
        for _ in range(OBJECT_ATTEMPTS):
            shape = _build_object(height, width, sides, occupied)
            if shape is not None:
                break
        if shape is None:
            return None
        occupied |= shape
        grid[shape] = AZURE

    if not _validate_counts(grid, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
