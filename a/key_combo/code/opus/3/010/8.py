from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5

SPRITE_DIMS = [(3, 3), (3, 5), (5, 3), (5, 5)]
GRID_MIN = 22
GRID_MAX = 30
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 160


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    r0 = int(r0)
    c0 = int(c0)
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > dst.shape[0] or c0 + w > dst.shape[1]:
        raise ValueError("paste out of bounds")
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    sprite_mask = (grid != BLACK) & (grid != GRAY)
    components = _components_mask(sprite_mask)
    assert len(components) == 1, "expected exactly one non-gray sprite"
    patch = _crop_bbox(grid, _bbox_coords(components[0]))
    h, w = patch.shape
    assert h % 2 == 1 and w % 2 == 1, "sprite bbox must have odd dimensions"
    assert int(patch[h // 2, w // 2]) != BLACK, "sprite center must be occupied"
    colors = set(map(int, np.unique(patch[patch != BLACK])))
    assert colors == {BLUE, RED}, "sprite must use blue and red only"
    return patch


def _extract_marker_array(grid: np.ndarray) -> tuple[list[int], list[int]]:
    coords = np.argwhere(grid == GRAY)
    assert coords.shape[0] >= 9, "marker array must contain at least 3x3 points"
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    assert len(rows) >= 3 and len(cols) >= 3, "marker array must be at least 3x3"
    assert coords.shape[0] == len(rows) * len(cols), "markers must fill a full rectangle"
    row_step = rows[1] - rows[0]
    col_step = cols[1] - cols[0]
    assert row_step > 0 and col_step > 0
    assert all(rows[i + 1] - rows[i] == row_step for i in range(len(rows) - 1))
    assert all(cols[i + 1] - cols[i] == col_step for i in range(len(cols) - 1))
    marker_set = {tuple(map(int, p)) for p in coords}
    assert marker_set == {(r, c) for r in rows for c in cols}, "markers must form a cartesian product"
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a multicolor sprite onto a gray marker array and recolor only the perimeter-array copies.

    Input structure:
    - A black background contains one 4-connected multicolor sprite made of blue and red cells.
    - Separate gray singleton markers form a full rectangular array with uniform row and column spacing.
    - The sprite bbox has odd height and width, so its bbox center is a grid cell.

    Rule:
    - Extract the sprite bbox, treating black cells inside the bbox as transparent.
    - Read the gray markers as a rectangular marker array and index each marker by its row and column within that array.
    - Stamp the sprite centered on every marker position.
    - If a marker lies on the array perimeter (top row, bottom row, left column, or right column), recolor that entire stamped copy green; otherwise keep the sprite's original blue/red colors.
    - Remove the original sprite and cover the gray markers with the stamped copies.

    Key insight:
    - The recoloring rule depends on each marker's position inside the marker array, not on the sprite geometry or on the marker's absolute grid location.
    """
    g = np.asarray(grid, dtype=int)
    sprite = _extract_sprite(g)
    rows, cols = _extract_marker_array(g)

    out = np.zeros(g.shape, dtype=int)
    green_sprite = np.where(sprite != BLACK, GREEN, BLACK)
    half_r = sprite.shape[0] // 2
    half_c = sprite.shape[1] // 2
    last_row = len(rows) - 1
    last_col = len(cols) - 1

    for ri, marker_r in enumerate(rows):
        for ci, marker_c in enumerate(cols):
            patch = green_sprite if (ri in (0, last_row) or ci in (0, last_col)) else sprite
            _paste(out, patch, marker_r - half_r, marker_c - half_c, transparent=BLACK)
    return out


def _is_asymmetric(mask: np.ndarray) -> bool:
    same_shape_variants = [
        np.flipud(mask),
        np.fliplr(mask),
        np.rot90(mask, 2),
    ]
    if mask.shape[0] == mask.shape[1]:
        same_shape_variants.extend([np.rot90(mask, 1), np.rot90(mask, 3), mask.T, np.fliplr(mask.T)])
    return all(not np.array_equal(mask, variant) for variant in same_shape_variants)


def _path_cells(start: tuple[int, int], target: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = start
    tr, tc = target
    path = {(r, c)}
    while (r, c) != (tr, tc):
        moves: list[tuple[int, int]] = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        path.add((r, c))
    return path


def _generate_mask(h: int, w: int) -> np.ndarray:
    center = (h // 2, w // 2)
    min_area = max(5, int(np.ceil(h * w * 0.45)))
    max_area = min(h * w - 1, int(np.floor(h * w * 0.75)))

    for _ in range(SHAPE_ATTEMPTS):
        cells = {center}
        targets = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        for target in targets:
            cells |= _path_cells(center, target)

        lower = max(min_area, len(cells))
        upper = max(lower, max_area)
        target_area = random.randint(lower, upper)

        frontier = {
            (nr, nc)
            for r, c in cells
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
        }
        while len(cells) < target_area and frontier:
            chosen = random.choice(tuple(frontier))
            cells.add(chosen)
            frontier.discard(chosen)
            for nr, nc in _neighbors4(*chosen):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not mask[center]:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(_components_mask(mask)) != 1:
            continue
        if not _is_asymmetric(mask):
            continue
        return mask

    raise RuntimeError("failed to generate a valid sprite mask")


def _generate_sprite() -> np.ndarray:
    h, w = random.choice(SPRITE_DIMS)
    mask = _generate_mask(h, w)
    coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
    random.shuffle(coords)

    n = len(coords)
    low_blue = 2 if n >= 4 else 1
    high_blue = n - low_blue
    n_blue = random.randint(low_blue, high_blue)
    colors = [BLUE] * n_blue + [RED] * (n - n_blue)
    random.shuffle(colors)

    sprite = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(coords, colors):
        sprite[r, c] = color
    return sprite


def _render_output(
    shape: tuple[int, int],
    sprite: np.ndarray,
    array_r: int,
    array_c: int,
    rows: int,
    cols: int,
    row_step: int,
    col_step: int,
    green_pred,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    green_sprite = np.where(sprite != BLACK, GREEN, BLACK)
    for ri in range(rows):
        for ci in range(cols):
            patch = green_sprite if green_pred(ri, ci, rows, cols) else sprite
            _paste(out, patch, array_r + ri * row_step, array_c + ci * col_step, transparent=BLACK)
    return out


def _gen_layout(sprite: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = sprite.shape
    row_counts = [n for n in range(3, 6) if h + (n - 1) * (h + 2) <= 28]
    col_counts = [n for n in range(3, 6) if w + (n - 1) * (w + 2) <= 28]
    if not row_counts or not col_counts:
        return None

    rows = random.choice(row_counts)
    cols = random.choice(col_counts)
    row_step = random.randint(h + 2, min(h + 5, (28 - h) // (rows - 1)))
    col_step = random.randint(w + 2, min(w + 5, (28 - w) // (cols - 1)))

    array_h = h + (rows - 1) * row_step
    array_w = w + (cols - 1) * col_step
    gap = random.randint(2, 4)

    sides: list[str] = []
    if array_h + h + gap + 2 <= GRID_MAX and max(array_w, w) + 2 <= GRID_MAX:
        sides.extend(["above", "below"])
    if array_w + w + gap + 2 <= GRID_MAX and max(array_h, h) + 2 <= GRID_MAX:
        sides.extend(["left", "right"])
    if not sides:
        return None

    side = random.choice(sides)
    if side in {"above", "below"}:
        min_h = array_h + h + gap + 2
        min_w = max(array_w, w) + 2
    else:
        min_h = max(array_h, h) + 2
        min_w = array_w + w + gap + 2

    if max(GRID_MIN, min_h) > GRID_MAX or max(GRID_MIN, min_w) > GRID_MAX:
        return None

    grid_h = random.randint(max(GRID_MIN, min_h), GRID_MAX)
    grid_w = random.randint(max(GRID_MIN, min_w), GRID_MAX)

    if side == "above":
        slack = grid_h - (array_h + h + gap + 2)
        top_extra = random.randint(0, slack)
        sprite_r = 1 + top_extra
        array_r = sprite_r + h + gap
        sprite_c = random.randint(1, grid_w - w - 1)
        array_c = random.randint(1, grid_w - array_w - 1)
    elif side == "below":
        slack = grid_h - (array_h + h + gap + 2)
        top_extra = random.randint(0, slack)
        array_r = 1 + top_extra
        sprite_r = array_r + array_h + gap
        sprite_c = random.randint(1, grid_w - w - 1)
        array_c = random.randint(1, grid_w - array_w - 1)
    elif side == "left":
        slack = grid_w - (array_w + w + gap + 2)
        left_extra = random.randint(0, slack)
        sprite_c = 1 + left_extra
        array_c = sprite_c + w + gap
        sprite_r = random.randint(1, grid_h - h - 1)
        array_r = random.randint(1, grid_h - array_h - 1)
    else:
        slack = grid_w - (array_w + w + gap + 2)
        left_extra = random.randint(0, slack)
        array_c = 1 + left_extra
        sprite_c = array_c + array_w + gap
        sprite_r = random.randint(1, grid_h - h - 1)
        array_r = random.randint(1, grid_h - array_h - 1)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _paste(grid, sprite, sprite_r, sprite_c, transparent=BLACK)

    marker_rows = [array_r + h // 2 + i * row_step for i in range(rows)]
    marker_cols = [array_c + w // 2 + j * col_step for j in range(cols)]
    for marker_r in marker_rows:
        for marker_c in marker_cols:
            grid[marker_r, marker_c] = GRAY

    expected = _render_output(
        grid.shape,
        sprite,
        array_r,
        array_c,
        rows,
        cols,
        row_step,
        col_step,
        lambda ri, ci, rr, cc: ri in (0, rr - 1) or ci in (0, cc - 1),
    )

    if expected[sprite_r : sprite_r + h, sprite_c : sprite_c + w].any():
        return None
    if not ({GREEN, BLUE, RED} <= set(map(int, np.unique(expected)))):
        return None

    wrong_outputs = [
        _render_output(grid.shape, sprite, array_r, array_c, rows, cols, row_step, col_step, lambda *_: True),
        _render_output(grid.shape, sprite, array_r, array_c, rows, cols, row_step, col_step, lambda *_: False),
        _render_output(
            grid.shape,
            sprite,
            array_r,
            array_c,
            rows,
            cols,
            row_step,
            col_step,
            lambda ri, ci, rr, cc: ri in (0, rr - 1),
        ),
        _render_output(
            grid.shape,
            sprite,
            array_r,
            array_c,
            rows,
            cols,
            row_step,
            col_step,
            lambda ri, ci, rr, cc: ci in (0, cc - 1),
        ),
        _render_output(
            grid.shape,
            sprite,
            array_r,
            array_c,
            rows,
            cols,
            row_step,
            col_step,
            lambda ri, ci, rr, cc: ri in (0, rr - 1) and ci in (0, cc - 1),
        ),
    ]
    if any(np.array_equal(expected, wrong) for wrong in wrong_outputs):
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sprite = _generate_sprite()
        built = _gen_layout(sprite)
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid example after {GEN_ATTEMPTS} attempts")
