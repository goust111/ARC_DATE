from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
MARKER = 5

GEN_ATTEMPTS = 400


def _scene_spec(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    markers = np.argwhere(g == MARKER)
    assert markers.shape[0] == 4, "expected exactly four gray markers"

    r0, c0 = markers.min(axis=0)
    r1, c1 = markers.max(axis=0)
    bbox = (int(r0), int(c0), int(r1), int(c1))

    tile_in = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    tile_out = tile_in.copy()
    tile_out[tile_out == MARKER] = GREEN

    target_cols = sorted({int(c) for _r, c in markers})
    assert 2 <= len(target_cols) <= 3, "markers must occupy two or three distinct columns"
    return bbox, tile_in, tile_out, target_cols


def _overlay_tile(
    base: np.ndarray,
    tile: np.ndarray,
    target_cols: list[int],
    *,
    order: str = "ltr",
    opaque_black: bool = False,
) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    height, width = map(int, out.shape)
    tile_h, tile_w = map(int, tile.shape)

    cols = sorted({int(c) for c in target_cols})
    if order == "rtl":
        cols = list(reversed(cols))
    else:
        assert order == "ltr"

    for target_c in cols:
        if target_c >= width:
            continue
        for top in range(0, height, tile_h):
            for dr in range(tile_h):
                rr = top + dr
                if rr >= height:
                    break
                for dc in range(tile_w):
                    cc = target_c + dc
                    if cc >= width:
                        break
                    val = int(tile[dr, dc])
                    if opaque_black or val != BG:
                        out[rr, cc] = val
    return out


def _oracle_output(
    grid: np.ndarray,
    *,
    target_cols: list[int] | None = None,
    order: str = "ltr",
    opaque_black: bool = False,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bbox, _tile_in, tile_out, true_target_cols = _scene_spec(g)
    base = g.copy()
    base[base == MARKER] = BG
    cols = true_target_cols if target_cols is None else target_cols
    return _overlay_tile(base, tile_out, cols, order=order, opaque_black=opaque_black)


def _repeat_strip(tile: np.ndarray, height: int) -> np.ndarray:
    reps = (int(height) + tile.shape[0] - 1) // tile.shape[0]
    return np.tile(tile, (reps, 1))[:height, :]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the gray-marker bbox as a tile and vertically spread it at the markers' columns.

    Input structure:
    - The background is black.
    - Blue pixels form a sparse pattern across the grid.
    - Exactly four gray marker pixels lie inside a non-edge bounding box; their row/column extrema define the tile window and their absolute columns are the spread targets.

    Rule:
    - Crop the marker bounding box and recolor only the gray marker cells inside it to green, leaving blue and black cells unchanged.
    - Remove the original gray markers from the base grid.
    - For each distinct marker column, from left to right, left-align the tile to that column and repeat it downward from the top row, clipping at the grid edges.
    - Only non-black tile cells overwrite the base grid; black tile cells stay transparent.

    Key insight:
    - The same four gray markers encode both what tile to extract and which absolute columns should receive the vertical tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert set(map(int, np.unique(g))).issubset({BG, BLUE, MARKER}), "unexpected input colors"

    _bbox, _tile_in, tile_out, target_cols = _scene_spec(g)
    out = g.copy()
    out[out == MARKER] = BG

    height, width = map(int, g.shape)
    strip = _repeat_strip(tile_out, height)
    tile_w = tile_out.shape[1]

    for target_c in target_cols:
        if target_c >= width:
            continue
        crop_w = min(tile_w, width - target_c)
        patch = strip[:, :crop_w]
        mask = patch != BG
        window = out[:, target_c : target_c + crop_w]
        window[mask] = patch[mask]

    return out


def _sample_height(tile_h: int) -> int:
    vals = [h for h in range(max(10, tile_h + 4), 17)]
    divisible = [h for h in vals if h % tile_h == 0]
    non_divisible = [h for h in vals if h % tile_h != 0]
    if divisible and non_divisible:
        pool = divisible if random.random() < 0.4 else non_divisible
        return random.choice(pool)
    return random.choice(vals)


def _sample_width(tile_w: int) -> int:
    vals = [w for w in range(max(11, 2 * tile_w + 1), 18)]
    return random.choice(vals)


def _sample_markers(tile_h: int, tile_w: int, target_count: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    internal_rows = random.sample(range(1, tile_h - 1), 2)
    left = 0
    right = tile_w - 1

    if target_count == 2:
        if random.choice([False, True]):
            markers = {(0, left), (tile_h - 1, left), (internal_rows[0], right), (internal_rows[1], right)}
            conflict_blue = (0, right)
        else:
            markers = {(0, right), (tile_h - 1, right), (internal_rows[0], left), (internal_rows[1], left)}
            conflict_blue = (0, left)
        return markers, conflict_blue

    mid = random.randint(1, tile_w - 2)
    markers = {(0, mid), (tile_h - 1, mid), (internal_rows[0], left), (internal_rows[1], right)}
    conflict_blue = (0, random.choice([left, right]))
    return markers, conflict_blue


def _make_tile_input(
    tile_h: int,
    tile_w: int,
    target_count: int,
) -> tuple[np.ndarray, list[int]]:
    markers, conflict_blue = _sample_markers(tile_h, tile_w, target_count)
    target_local_cols = sorted({c for _r, c in markers})
    non_target_cols = [c for c in range(tile_w) if c not in target_local_cols]
    assert non_target_cols, "tile must have at least one non-target column"

    free_cells = [(r, c) for r in range(tile_h) for c in range(tile_w) if (r, c) not in markers]
    mandatory_blue = {conflict_blue}
    while True:
        cell = (random.randrange(tile_h), random.choice(non_target_cols))
        if cell not in markers and cell not in mandatory_blue:
            mandatory_blue.add(cell)
            break

    for _ in range(600):
        blue_cells = set(mandatory_blue)
        extras = [cell for cell in free_cells if cell not in blue_cells]
        extra_n = random.randint(0, min(5, len(extras)))
        if extra_n:
            blue_cells.update(random.sample(extras, extra_n))

        tile = np.zeros((tile_h, tile_w), dtype=int)
        for r, c in blue_cells:
            tile[r, c] = BLUE
        for r, c in markers:
            tile[r, c] = MARKER

        tile_out = tile.copy()
        tile_out[tile_out == MARKER] = GREEN

        blue_cols = sorted({int(c) for _r, c in np.argwhere(tile == BLUE)})
        if not blue_cols or set(blue_cols) == set(target_local_cols):
            continue
        if not any(c not in target_local_cols for c in blue_cols):
            continue
        if not np.any(tile == BG):
            continue

        witness = _find_transparent_local(tile_out, target_local_cols)
        if witness is None:
            continue
        if not _order_sensitive(tile_out, target_local_cols):
            continue
        if not _wrong_column_shortcut_fails(tile_out, target_local_cols, blue_cols):
            continue
        return tile, target_local_cols

    raise RuntimeError("failed to sample a valid tile")


def _find_transparent_local(tile_out: np.ndarray, target_local_cols: list[int]) -> tuple[int, int] | None:
    sorted_cols = sorted(target_local_cols)
    unique_width = sorted_cols[1] - sorted_cols[0]
    if unique_width <= 0:
        return None

    row_limit = min(2, tile_out.shape[0])
    for r in range(row_limit):
        for c in range(unique_width):
            if tile_out[r, c] == BG:
                return int(r), int(c)
    return None


def _order_sensitive(tile_out: np.ndarray, target_local_cols: list[int]) -> bool:
    cols = [c - target_local_cols[0] for c in target_local_cols]
    width = max(cols) + tile_out.shape[1]
    blank = np.zeros((tile_out.shape[0] * 2, width), dtype=int)
    left_to_right = _overlay_tile(blank, tile_out, cols, order="ltr")
    right_to_left = _overlay_tile(blank, tile_out, cols, order="rtl")
    return not np.array_equal(left_to_right, right_to_left)


def _wrong_column_shortcut_fails(
    tile_out: np.ndarray,
    target_local_cols: list[int],
    blue_local_cols: list[int],
) -> bool:
    true_cols = [c - target_local_cols[0] for c in target_local_cols]
    blue_cols = [c - target_local_cols[0] for c in blue_local_cols]
    nonblack_cols = sorted({int(c) for _r, c in np.argwhere(tile_out != BG)})
    nonblack_cols = [c - target_local_cols[0] for c in nonblack_cols]

    width = max(nonblack_cols + true_cols) + tile_out.shape[1]
    blank = np.zeros((tile_out.shape[0] * 2, width), dtype=int)
    expected = _overlay_tile(blank, tile_out, true_cols)
    return (
        not np.array_equal(expected, _overlay_tile(blank, tile_out, blue_cols))
        and not np.array_equal(expected, _overlay_tile(blank, tile_out, nonblack_cols))
    )


def _inside_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _scatter_noise(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    reserved: set[tuple[int, int]],
) -> None:
    height, width = map(int, grid.shape)
    target = random.randint(2, min(7, max(2, (height * width) // 20)))
    chosen = set(reserved)
    candidates = [(r, c) for r in range(height) for c in range(width) if not _inside_bbox(r, c, bbox)]
    random.shuffle(candidates)

    for r, c in candidates:
        if len(chosen) - len(reserved) >= target:
            break
        if grid[r, c] != BG or (r, c) in chosen:
            continue
        if any(max(abs(r - rr), abs(c - cc)) < 2 for rr, cc in chosen):
            continue
        grid[r, c] = BLUE
        chosen.add((r, c))


def _validate_scene(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({BG, BLUE, MARKER})

    bbox, tile_in, tile_out, target_abs_cols = _scene_spec(g)
    r0, c0, r1, c1 = bbox
    height, width = map(int, g.shape)
    assert r0 >= 2 and c0 >= 2 and r1 <= height - 3 and c1 <= width - 3

    target_local_cols = [c - c0 for c in target_abs_cols]
    blue_local_cols = sorted({int(c) for _r, c in np.argwhere(tile_in == BLUE)})
    assert np.any(tile_in == BLUE), "tile must contain blue content"
    assert np.any(tile_in == BG), "tile must contain transparent black content"
    assert set(blue_local_cols) != set(target_local_cols)
    assert any(c not in target_local_cols for c in blue_local_cols)

    witness = _find_transparent_local(tile_out, target_local_cols)
    assert witness is not None
    assert witness[0] < r0
    assert _order_sensitive(tile_out, target_local_cols)
    assert _wrong_column_shortcut_fails(tile_out, target_local_cols, blue_local_cols)

    expected = _oracle_output(g)
    blue_target_abs = sorted({c0 + c for c in blue_local_cols})
    nonblack_target_abs = sorted({c0 + int(c) for _r, c in np.argwhere(tile_out != BG)})

    assert not np.array_equal(expected, _oracle_output(g, target_cols=blue_target_abs))
    assert not np.array_equal(expected, _oracle_output(g, target_cols=nonblack_target_abs))
    assert not np.array_equal(expected, _oracle_output(g, opaque_black=True))
    assert not np.array_equal(expected, _oracle_output(g, order="rtl"))


def _generate_attempt() -> np.ndarray | None:
    tile_h = random.choice([4, 5])
    tile_w = random.choice([4, 5])
    target_count = random.choice([2, 3])

    height = _sample_height(tile_h)
    width = _sample_width(tile_w)

    max_box_c0 = width - 2 * tile_w + 1
    if max_box_c0 < 2:
        return None

    try:
        tile_in, target_local_cols = _make_tile_input(tile_h, tile_w, target_count)
    except RuntimeError:
        return None
    tile_out = tile_in.copy()
    tile_out[tile_out == MARKER] = GREEN

    witness_local = _find_transparent_local(tile_out, target_local_cols)
    if witness_local is None:
        return None

    box_r0 = random.randint(2, height - tile_h - 2)
    box_c0 = random.randint(2, max_box_c0)
    bbox = (box_r0, box_c0, box_r0 + tile_h - 1, box_c0 + tile_w - 1)

    grid = np.zeros((height, width), dtype=int)
    grid[box_r0 : box_r0 + tile_h, box_c0 : box_c0 + tile_w] = tile_in

    witness_abs = (int(witness_local[0]), int(box_c0 + witness_local[1]))
    if _inside_bbox(witness_abs[0], witness_abs[1], bbox):
        return None
    grid[witness_abs] = BLUE

    _scatter_noise(grid, bbox, {witness_abs})

    try:
        _validate_scene(grid)
        if not np.array_equal(main(grid), _oracle_output(grid)):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
