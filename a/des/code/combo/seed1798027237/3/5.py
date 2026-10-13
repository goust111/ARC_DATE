import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

MARKER_COLORS = (BLUE, RED, GREEN)
LINE_COLORS = (MAROON, BLUE, RED, GREEN)
OBJECT_COLORS = (YELLOW, GRAY, MAGENTA, ORANGE, CYAN)

MAX_GENERATION_ATTEMPTS = 300
SHAPE_ATTEMPTS = 200


def _divider_rows(grid: np.ndarray) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if np.all(np.isin(row, LINE_COLORS)) and np.any(row == MAROON):
            rows.append(r)
    return rows


def _divider_cols(grid: np.ndarray) -> list[int]:
    cols = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if np.all(np.isin(col, LINE_COLORS)) and np.any(col == MAROON):
            cols.append(c)
    return cols


def _border_coords(top: int, bottom: int, left: int, right: int) -> list[tuple[int, int]]:
    coords = [(top, c) for c in range(left, right + 1)]
    coords.extend((bottom, c) for c in range(left, right + 1))
    coords.extend((r, left) for r in range(top + 1, bottom))
    coords.extend((r, right) for r in range(top + 1, bottom))
    return coords


def _crop_nonblack(arr: np.ndarray) -> np.ndarray:
    coords = np.argwhere(arr != BLACK)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform(sprite: np.ndarray, marker_color: int, count: int) -> np.ndarray:
    if marker_color == RED:
        return np.rot90(sprite, -count)
    if marker_color == BLUE:
        return np.fliplr(sprite) if count % 2 == 1 else sprite.copy()
    if marker_color == GREEN:
        return np.flipud(sprite) if count % 2 == 1 else sprite.copy()
    raise AssertionError(f"unexpected marker color: {marker_color}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each maroon-bounded sub-rectangle as a container.

    The border marker color selects a transform:
    - red: rotate clockwise 90 degrees
    - blue: flip left-right
    - green: flip up-down

    The number of border markers (1..3) is the number of times that transform is
    applied. The transformed object is then re-centered inside its container.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    div_rows = _divider_rows(g)
    div_cols = _divider_cols(g)
    assert len(div_rows) >= 2 and len(div_cols) >= 2
    assert div_rows[0] == 0 and div_rows[-1] == g.shape[0] - 1
    assert div_cols[0] == 0 and div_cols[-1] == g.shape[1] - 1

    out = g.copy()
    for top, bottom in zip(div_rows, div_rows[1:]):
        for left, right in zip(div_cols, div_cols[1:]):
            assert bottom - top >= 2 and right - left >= 2

            border = _border_coords(top, bottom, left, right)
            markers = [(r, c, int(g[r, c])) for r, c in border if int(g[r, c]) in MARKER_COLORS]
            marker_colors = {color for _, _, color in markers}
            assert len(marker_colors) == 1
            marker_color = next(iter(marker_colors))
            marker_count = len(markers)
            assert 1 <= marker_count <= 3

            inner = g[top + 1 : bottom, left + 1 : right]
            sprite = _crop_nonblack(inner)
            transformed = _transform(sprite, marker_color, marker_count)

            out[top + 1 : bottom, left + 1 : right] = BLACK
            inner_h, inner_w = inner.shape
            h, w = transformed.shape
            assert h <= inner_h and w <= inner_w
            dest_r = top + 1 + (inner_h - h) // 2
            dest_c = left + 1 + (inner_w - w) // 2
            out[dest_r : dest_r + h, dest_c : dest_c + w] = transformed
    return out


def _neighbors(cell: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = cell
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            result.append((rr, cc))
    return result


def _connected_random_mask(h: int, w: int, target_cells: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (np.random.randint(h), np.random.randint(w))
    mask[start] = True
    frontier = set(_neighbors(start, h, w))
    while int(mask.sum()) < target_cells and frontier:
        frontier_list = list(frontier)
        idx = np.random.randint(len(frontier_list))
        cell = frontier_list[idx]
        frontier.remove(cell)
        mask[cell] = True
        for nxt in _neighbors(cell, h, w):
            if not mask[nxt]:
                frontier.add(nxt)
    return mask


def _is_asymmetric(mask: np.ndarray) -> bool:
    tests = (
        np.rot90(mask, -1),
        np.rot90(mask, -2),
        np.rot90(mask, -3),
        np.fliplr(mask),
        np.flipud(mask),
    )
    return all(not np.array_equal(mask, other) for other in tests)


def _make_shape(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = np.random.randint(3, max_h + 1)
        w = np.random.randint(3, max_w + 1)
        if min(h, w) < 3:
            continue
        area = h * w
        target = np.random.randint(max(4, area // 3), area)
        mask = _connected_random_mask(h, w, target)
        cropped = _crop_nonblack(mask.astype(int)).astype(bool)
        if min(cropped.shape) < 2:
            continue
        if cropped.sum() < 4:
            continue
        if _is_asymmetric(cropped):
            return cropped.astype(int)
    return None


def _cover_choices(size: int, count: int) -> list[int]:
    base = list(range(1, count + 1))
    if size > count:
        extra = list(np.random.choice(base, size=size - count, replace=True))
    else:
        extra = []
    values = base + extra
    np.random.shuffle(values)
    return values


def _color_choices(size: int) -> list[int]:
    base = [RED, BLUE, GREEN]
    if size > 3:
        base.extend(np.random.choice(base, size=size - 3, replace=True).tolist())
    np.random.shuffle(base)
    return base


def _line_positions(lengths: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for length in lengths:
        cursor += length + 1
        positions.append(cursor)
    return positions


def _place_markers(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    color: int,
    count: int,
    horizontal_strip: bool,
) -> None:
    if horizontal_strip:
        candidates = [(top, c) for c in range(left + 1, right)]
        candidates.extend((bottom, c) for c in range(left + 1, right))
    else:
        candidates = [(r, left) for r in range(top + 1, bottom)]
        candidates.extend((r, right) for r in range(top + 1, bottom))
    assert len(candidates) >= count
    order = np.random.permutation(len(candidates))[:count]
    for idx in order:
        r, c = candidates[int(idx)]
        grid[r, c] = color


def _paint_sprite(
    grid: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    color: int,
    sprite_mask: np.ndarray,
) -> None:
    inner_h = bottom - top - 1
    inner_w = right - left - 1
    h, w = sprite_mask.shape
    assert h <= inner_h - 2 and w <= inner_w - 2

    row = np.random.randint(1, inner_h - h)
    col = np.random.randint(1, inner_w - w)
    sprite = sprite_mask * color
    patch = grid[top + 1 + row : top + 1 + row + h, left + 1 + col : left + 1 + col + w]
    assert np.all(patch == BLACK)
    patch[sprite != BLACK] = color


def _gen_attempt() -> np.ndarray | None:
    horizontal_strip = bool(np.random.randint(2))
    if horizontal_strip:
        row_heights = [np.random.randint(6, 10)]
        col_widths = [np.random.randint(6, 11) for _ in range(np.random.randint(3, 6))]
    else:
        row_heights = [np.random.randint(6, 11) for _ in range(np.random.randint(3, 6))]
        col_widths = [np.random.randint(6, 10)]

    div_rows = _line_positions(row_heights)
    div_cols = _line_positions(col_widths)
    h = div_rows[-1] + 1
    w = div_cols[-1] + 1

    grid = np.full((h, w), BLACK, dtype=int)
    grid[div_rows, :] = MAROON
    grid[:, div_cols] = MAROON

    n_containers = len(row_heights) * len(col_widths)
    obj_colors = list(np.random.permutation(OBJECT_COLORS)[:n_containers])
    marker_colors = _color_choices(n_containers)
    marker_counts = _cover_choices(n_containers, 3)

    idx = 0
    for i, (top, bottom) in enumerate(zip(div_rows, div_rows[1:])):
        for j, (left, right) in enumerate(zip(div_cols, div_cols[1:])):
            inner_h = bottom - top - 1
            inner_w = right - left - 1
            max_dim = min(inner_h, inner_w) - 2
            if max_dim < 3:
                return None

            sprite_mask = _make_shape(max_dim, max_dim)
            if sprite_mask is None:
                return None

            _paint_sprite(grid, top, bottom, left, right, obj_colors[idx], sprite_mask)
            _place_markers(
                grid,
                top,
                bottom,
                left,
                right,
                marker_colors[idx],
                marker_counts[idx],
                horizontal_strip,
            )
            idx += 1

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
