from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
TEAL = 8
CONTROL_COLORS = (2, 3, 4, 5, 6, 7, 9)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return -1


def _line_pair_count(marker_mask: np.ndarray) -> int:
    total = 0
    h, w = marker_mask.shape
    for r in range(h):
        total += max(0, int(np.count_nonzero(marker_mask[r])) - 1)
    for c in range(w):
        total += max(0, int(np.count_nonzero(marker_mask[:, c])) - 1)
    return total


def _connection_masks(marker_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    marker_mask = np.asarray(marker_patch == BLUE, dtype=bool)
    h, w = marker_mask.shape
    row_fill = np.zeros((h, w), dtype=bool)
    col_fill = np.zeros((h, w), dtype=bool)

    for r in range(h):
        cols = np.flatnonzero(marker_mask[r])
        for a, b in zip(cols, cols[1:]):
            if int(b) > int(a) + 1:
                row_fill[r, int(a) + 1 : int(b)] = True

    for c in range(w):
        rows = np.flatnonzero(marker_mask[:, c])
        for a, b in zip(rows, rows[1:]):
            if int(b) > int(a) + 1:
                col_fill[int(a) + 1 : int(b), c] = True

    return row_fill, col_fill


def _build_tile(marker_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tile = np.asarray(marker_patch, dtype=int).copy()
    row_fill, col_fill = _connection_masks(marker_patch)
    fill_mask = (row_fill | col_fill) & (tile == BLACK)
    tile[fill_mask] = TEAL
    return tile, row_fill, col_fill


def _extract_parts(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    blue_box = _bbox(g == BLUE)
    assert blue_box is not None, "expected blue marker region"

    other_colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BLACK, BLUE))
    assert len(other_colors) == 1, "expected exactly one control color"
    control_color = int(other_colors[0])

    ctrl_box = _bbox(g == control_color)
    assert ctrl_box is not None, "expected control sprite region"

    marker_patch = _crop(g, blue_box)
    ctrl_patch = _crop(g, ctrl_box)

    assert marker_patch.shape[0] == marker_patch.shape[1], "marker bbox must be square"
    return marker_patch, ctrl_patch, control_color


def _scale_with_tile(tile: np.ndarray, ctrl_patch: np.ndarray) -> np.ndarray:
    s = int(tile.shape[0])
    h, w = map(int, ctrl_patch.shape)
    out = np.full((h * s, w * s), BLACK, dtype=int)
    for r in range(h):
        for c in range(w):
            if int(ctrl_patch[r, c]) != BLACK:
                _paste(out, tile, r * s, c * s)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue marker pixels into a square tile, then use that tile to pixel-scale a separate monochrome sprite.

    Input structure:
    - A black background contains two disjoint regions separated by empty rows and columns.
    - One region is a square bbox of blue marker pixels; some markers share rows and some share columns.
    - The other region is a monochrome rectangular control sprite with black holes inside its bbox.

    Rule:
    - Within the blue-marker region, connect same-row adjacent markers with teal cells across the intervening background cells.
    - Also connect same-column adjacent markers with teal cells across the intervening background cells.
    - Crop the blue markers' square bbox after adding those teal segments; that multicolor square becomes the tile.
    - Crop the control sprite's bbox, and for each non-black control pixel stamp one full copy of the tile into the corresponding output block; leave black blocks for black control pixels.

    Key insight:
    - The tile is not given directly: it must first be constructed from the markers' row/column alignment, and only that constructed tile is then repeated to enlarge the separate control sprite.
    """
    marker_patch, ctrl_patch, _control_color = _extract_parts(grid)
    tile, _row_fill, _col_fill = _build_tile(marker_patch)
    return _scale_with_tile(tile, ctrl_patch)


def _random_connected_mask(h: int, w: int, target: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    steps = ["D"] * (h - 1) + ["R"] * (w - 1)
    rng.shuffle(steps)

    r = c = 0
    mask[r, c] = True
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    if int(mask.sum()) > target:
        return None

    while int(mask.sum()) < target:
        frontier: list[tuple[int, int]] = []
        for rr, cc in np.argwhere(mask):
            rr = int(rr)
            cc = int(cc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        mask[nr, nc] = True

    return mask


def _sample_marker_patch(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(400):
        s = int(rng.integers(3, 6))
        k = int(rng.integers(3, 6))
        cells = rng.choice(s * s, size=k, replace=False)
        patch = np.zeros((s, s), dtype=int)
        patch[np.unravel_index(cells, (s, s))] = BLUE

        box = _bbox(patch == BLUE)
        if box != (0, 0, s - 1, s - 1):
            continue
        if _line_pair_count(patch == BLUE) < 2:
            continue

        tile, row_fill, col_fill = _build_tile(patch)
        if not row_fill.any() or not col_fill.any():
            continue
        if not np.any(row_fill & ~col_fill):
            continue
        if not np.any(col_fill & ~row_fill):
            continue
        if int(np.count_nonzero(tile == TEAL)) < 2:
            continue
        if int(np.count_nonzero(tile == BLACK)) * 2 < s * s:
            continue

        return patch, tile
    raise RuntimeError("failed to sample marker patch")


def _sample_control_patch(s: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(400):
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        if h * s > 30 or w * s > 30:
            continue

        area = h * w
        min_fill = max(h + w - 1, int(np.ceil(area * 0.35)))
        max_fill = min(area - 2, int(np.floor(area * 0.72)))
        if min_fill > max_fill:
            continue
        target = int(rng.integers(min_fill, max_fill + 1))
        mask = _random_connected_mask(h, w, target, rng)
        if mask is None:
            continue

        if int(mask.sum()) < 2 or int((~mask).sum()) < 2:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if h >= 3 and w >= 3 and not np.any(~mask[1:-1, 1:-1]):
            continue

        if bool(rng.integers(2)):
            mask = np.flipud(mask)
        if bool(rng.integers(2)):
            mask = np.fliplr(mask)
        if bool(rng.integers(2)):
            mask = mask.T
            h, w = mask.shape
            if h * s > 30 or w * s > 30:
                continue

        color = int(rng.choice(np.array(CONTROL_COLORS, dtype=int)))
        return np.where(mask, color, BLACK).astype(int)
    raise RuntimeError("failed to sample control patch")


def _place_diagonal(
    marker_shape: tuple[int, int],
    ctrl_shape: tuple[int, int],
    h: int,
    w: int,
    layout: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]]:
    mh, mw = map(int, marker_shape)
    ch, cw = map(int, ctrl_shape)
    border = 1
    row_gap = int(rng.integers(2, 6))
    col_gap = int(rng.integers(2, 6))

    marker_top = layout in (0, 2)
    marker_left = layout in (0, 3)

    top_h = mh if marker_top else ch
    bottom_h = ch if marker_top else mh
    left_w = mw if marker_left else cw
    right_w = cw if marker_left else mw

    top_r_max = h - border - row_gap - bottom_h - top_h
    left_c_max = w - border - col_gap - right_w - left_w
    assert top_r_max >= border and left_c_max >= border

    top_r = int(rng.integers(border, top_r_max + 1))
    bottom_r = int(rng.integers(top_r + top_h + row_gap, h - border - bottom_h + 1))
    left_c = int(rng.integers(border, left_c_max + 1))
    right_c = int(rng.integers(left_c + left_w + col_gap, w - border - right_w + 1))

    marker_r = top_r if marker_top else bottom_r
    ctrl_r = bottom_r if marker_top else top_r
    marker_c = left_c if marker_left else right_c
    ctrl_c = right_c if marker_left else left_c
    return (marker_r, marker_c), (ctrl_r, ctrl_c)


def _semantic_checks(inp: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray)
    assert inp.ndim == 2
    assert max(inp.shape) <= 30
    assert int(np.count_nonzero(inp == TEAL)) == 0, "input must not already contain teal lines"

    marker_patch, ctrl_patch, control_color = _extract_parts(inp)
    tile, row_fill, col_fill = _build_tile(marker_patch)
    out = main(inp)

    s = int(marker_patch.shape[0])
    assert 3 <= s <= 5
    assert marker_patch.shape == (s, s)
    assert 3 <= int(np.count_nonzero(marker_patch == BLUE)) <= 5
    assert _line_pair_count(marker_patch == BLUE) >= 2
    assert row_fill.any() and col_fill.any()
    assert np.any(row_fill & ~col_fill)
    assert np.any(col_fill & ~row_fill)
    assert int(np.count_nonzero(tile == BLACK)) * 2 >= s * s

    ch, cw = map(int, ctrl_patch.shape)
    assert 3 <= ch <= 6 and 3 <= cw <= 6
    assert np.any(ctrl_patch == BLACK)
    assert np.any(ctrl_patch == control_color)
    assert control_color in CONTROL_COLORS

    blue_box = _bbox(inp == BLUE)
    ctrl_box = _bbox(inp == control_color)
    assert blue_box is not None and ctrl_box is not None
    assert _axis_gap(blue_box[0], blue_box[2], ctrl_box[0], ctrl_box[2]) >= 2
    assert _axis_gap(blue_box[1], blue_box[3], ctrl_box[1], ctrl_box[3]) >= 2

    assert out.shape == (ch * s, cw * s)
    assert max(out.shape) <= 30
    assert np.any(out != BLACK)
    assert np.any(out == BLACK)

    row_only_tile = marker_patch.copy()
    row_only_tile[row_fill & (row_only_tile == BLACK)] = TEAL
    col_only_tile = marker_patch.copy()
    col_only_tile[col_fill & (col_only_tile == BLACK)] = TEAL
    marker_only_out = _scale_with_tile(marker_patch, ctrl_patch)
    row_only_out = _scale_with_tile(row_only_tile, ctrl_patch)
    col_only_out = _scale_with_tile(col_only_tile, ctrl_patch)
    full_out = _scale_with_tile(tile, ctrl_patch)

    assert np.array_equal(out, full_out)
    assert not np.array_equal(full_out, marker_only_out), "line-construction step must matter"
    assert not np.array_equal(full_out, row_only_out), "column connections must matter"
    assert not np.array_equal(full_out, col_only_out), "row connections must matter"


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        marker_patch, tile = _sample_marker_patch(rng)
        control_patch = _sample_control_patch(int(tile.shape[0]), rng)

        h = int(rng.integers(18, 26))
        w = int(rng.integers(18, 26))
        layout = int(rng.integers(0, 4))
        marker_pos, ctrl_pos = _place_diagonal(marker_patch.shape, control_patch.shape, h, w, layout, rng)

        grid = np.zeros((h, w), dtype=int)
        _paste(grid, marker_patch, *marker_pos)
        _paste(grid, control_patch, *ctrl_pos)

        try:
            _semantic_checks(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate valid input")
