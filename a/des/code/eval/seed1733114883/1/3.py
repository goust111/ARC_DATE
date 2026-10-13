import numpy as np
from collections import deque


TILE = 3
BG = 0
SEP = 5
RETRIES = 300
ALL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color) and int(np.count_nonzero(grid == color)) == h:
            candidates.append((c, color))
    assert len(candidates) == 1, "expected one full-height separator"
    return candidates[0]


def _parse_directory(left: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    tiles: dict[int, np.ndarray] = {}
    comps = _components(left != bg)
    assert len(comps) >= 2, "expected multiple directory tiles"
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (TILE, TILE), "directory tiles must be 3x3"
        assert len(comp) == TILE * TILE, "directory tiles must be fully filled"
        marker = int(patch[0, 0])
        assert marker != bg
        assert marker not in tiles, "directory markers must be unique"
        tiles[marker] = patch.copy()
    return tiles


def _parse_canvas(right: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], list[list[np.ndarray]]]:
    pts = np.argwhere(right != bg)
    assert pts.size > 0, "expected a canvas region"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    canvas = right[r0 : r1 + 1, c0 : c1 + 1]
    assert canvas.shape[0] % TILE == 0 and canvas.shape[1] % TILE == 0, "canvas must tile by 3x3 blocks"
    rows = canvas.shape[0] // TILE
    cols = canvas.shape[1] // TILE
    patches = [
        [canvas[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE].copy() for c in range(cols)]
        for r in range(rows)
    ]
    return (int(r0), int(c0), int(r1), int(c1)), patches


def _missing_marker(patch: np.ndarray, bg: int) -> int | None:
    nz = np.argwhere(patch != bg)
    if nz.shape[0] == 1 and tuple(nz[0]) == (1, 1):
        return int(patch[1, 1])
    return None


def _axis_flip(patch: np.ndarray, axis: str) -> np.ndarray:
    return np.flipud(patch) if axis == "horizontal" else np.fliplr(patch)


def _mirror_pos(axis: str, r: int, c: int, rows: int, cols: int) -> tuple[int, int]:
    if axis == "horizontal":
        return rows - 1 - r, c
    return r, cols - 1 - c


def _axis_valid(patches: list[list[np.ndarray]], bg: int, axis: str) -> tuple[bool, int]:
    rows, cols = len(patches), len(patches[0])
    if axis == "horizontal" and rows % 2 != 0:
        return False, 0
    if axis == "vertical" and cols % 2 != 0:
        return False, 0

    support = 0
    for r in range(rows):
        for c in range(cols):
            mr, mc = _mirror_pos(axis, r, c, rows, cols)
            if (r, c) >= (mr, mc):
                continue
            a = patches[r][c]
            b = patches[mr][mc]
            a_missing = _missing_marker(a, bg) is not None
            b_missing = _missing_marker(b, bg) is not None
            if a_missing and b_missing:
                return False, 0
            if not a_missing and not b_missing:
                if not np.array_equal(a, _axis_flip(b, axis)):
                    return False, 0
                support += 1
    return support > 0, support


def main(grid: np.ndarray) -> np.ndarray:
    """Fill missing 3x3 tiles by directory lookup and reflection across the canvas symmetry axis."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    sep_col, _sep_color = _find_separator(g, bg)
    tiles = _parse_directory(g[:, :sep_col], bg)
    canvas_bbox, patches = _parse_canvas(g[:, sep_col + 1 :], bg)

    candidates = []
    for axis in ("horizontal", "vertical"):
        ok, support = _axis_valid(patches, bg, axis)
        if ok:
            candidates.append((axis, support))
    assert candidates, "no valid symmetry axis found"
    best_support = max(support for _, support in candidates)
    best = [axis for axis, support in candidates if support == best_support]
    assert len(best) == 1, "symmetry axis must be unique"
    axis = best[0]

    rows, cols = len(patches), len(patches[0])
    out = g.copy()
    box_r0, box_c0, _box_r1, _box_c1 = canvas_bbox
    abs_c0 = sep_col + 1 + box_c0

    for r in range(rows):
        for c in range(cols):
            marker = _missing_marker(patches[r][c], bg)
            if marker is None:
                continue
            assert marker in tiles, "missing tile marker not found in directory"
            mr, mc = _mirror_pos(axis, r, c, rows, cols)
            partner = patches[mr][mc]
            assert _missing_marker(partner, bg) is None, "missing tile cannot mirror another missing tile"

            base_tile = tiles[marker]
            filled = _axis_flip(partner, axis)
            mirrored_base = _axis_flip(base_tile, axis)
            assert np.array_equal(filled, base_tile) or np.array_equal(filled, mirrored_base)

            rr = box_r0 + r * TILE
            cc = abs_c0 + c * TILE
            out[rr : rr + TILE, cc : cc + TILE] = filled
            patches[r][c] = filled
    return out


def _random_tile(rng: np.random.Generator, marker: int, body_palette: list[int]) -> np.ndarray:
    while True:
        patch = np.empty((TILE, TILE), dtype=int)
        patch[0, 0] = marker
        colors = rng.choice(body_palette, size=8, replace=True)
        if len(set(map(int, colors))) < 2:
            continue
        idx = 0
        for r in range(TILE):
            for c in range(TILE):
                if (r, c) == (0, 0):
                    continue
                patch[r, c] = int(colors[idx])
                idx += 1
        if np.array_equal(patch, np.flipud(patch)) or np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch


def _build_complete_canvas(
    rng: np.random.Generator, tiles: list[np.ndarray], axis: str, base_first_side: bool
) -> tuple[np.ndarray, np.ndarray]:
    if axis == "horizontal":
        rows = int(rng.choice([4, 6]))
        cols = int(rng.choice([3, 5]))
    else:
        rows = int(rng.choice([3, 5]))
        cols = int(rng.choice([4, 6]))

    tile_ids = np.full((rows, cols), -1, dtype=int)
    base_mask = np.zeros((rows, cols), dtype=bool)

    if axis == "horizontal":
        sample = rng.integers(0, len(tiles), size=(rows // 2, cols))
        while len(set(map(int, sample.ravel()))) < min(2, len(tiles)):
            sample = rng.integers(0, len(tiles), size=(rows // 2, cols))
        for r in range(rows // 2):
            for c in range(cols):
                tile_ids[r, c] = int(sample[r, c])
                tile_ids[rows - 1 - r, c] = int(sample[r, c])
                if base_first_side:
                    base_mask[r, c] = True
                else:
                    base_mask[rows - 1 - r, c] = True
    else:
        sample = rng.integers(0, len(tiles), size=(rows, cols // 2))
        while len(set(map(int, sample.ravel()))) < min(2, len(tiles)):
            sample = rng.integers(0, len(tiles), size=(rows, cols // 2))
        for r in range(rows):
            for c in range(cols // 2):
                tile_ids[r, c] = int(sample[r, c])
                tile_ids[r, cols - 1 - c] = int(sample[r, c])
                if base_first_side:
                    base_mask[r, c] = True
                else:
                    base_mask[r, cols - 1 - c] = True

    canvas = np.full((rows * TILE, cols * TILE), BG, dtype=int)
    for r in range(rows):
        for c in range(cols):
            tile = tiles[int(tile_ids[r, c])]
            if base_mask[r, c]:
                patch = tile
            else:
                patch = _axis_flip(tile, axis)
            canvas[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = patch
    return canvas, tile_ids


def _choose_missing_positions(
    rng: np.random.Generator, rows: int, cols: int, axis: str
) -> list[tuple[int, int]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            mr, mc = _mirror_pos(axis, r, c, rows, cols)
            key = tuple(sorted(((r, c), (mr, mc))))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((key[0], key[1]))
    assert pairs, "need at least one interior mirrored pair"

    count = int(rng.integers(1, min(4, len(pairs)) + 1))
    chosen_pairs = rng.choice(len(pairs), size=count, replace=False)
    missing: list[tuple[int, int]] = []
    for idx in np.atleast_1d(chosen_pairs):
        a, b = pairs[int(idx)]
        missing.append(a if rng.random() < 0.5 else b)
    return missing


def _assemble_grid(
    rng: np.random.Generator, directory_tiles: list[np.ndarray], canvas: np.ndarray
) -> tuple[np.ndarray, tuple[int, int]]:
    dir_h = len(directory_tiles) * TILE + (len(directory_tiles) - 1)
    dir_w = TILE
    canvas_h, canvas_w = canvas.shape

    dir_top = int(rng.integers(0, 3))
    canvas_top = int(rng.integers(0, 4))
    bottom_margin = int(rng.integers(0, 3))
    left_margin = int(rng.integers(0, 3))
    gap_before_sep = int(rng.integers(1, 3))
    gap_after_sep = int(rng.integers(1, 3))
    right_margin = int(rng.integers(0, 3))

    h = max(dir_top + dir_h, canvas_top + canvas_h) + bottom_margin
    sep_col = left_margin + dir_w + gap_before_sep
    canvas_left = sep_col + 1 + gap_after_sep
    w = canvas_left + canvas_w + right_margin

    grid = np.full((h, w), BG, dtype=int)
    grid[:, sep_col] = SEP

    row = dir_top
    for tile in directory_tiles:
        grid[row : row + TILE, left_margin : left_margin + TILE] = tile
        row += TILE + 1

    grid[canvas_top : canvas_top + canvas_h, canvas_left : canvas_left + canvas_w] = canvas
    return grid, (canvas_top, canvas_left)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    tile_count = int(rng.integers(2, 5))
    marker_colors = list(rng.choice(ALL_COLORS, size=tile_count, replace=False))
    body_palette = [color for color in ALL_COLORS if color not in marker_colors]
    rng.shuffle(marker_colors)

    directory_tiles: list[np.ndarray] = []
    for marker in marker_colors:
        tile = _random_tile(rng, marker, body_palette)
        directory_tiles.append(tile)

    order = list(rng.permutation(tile_count))
    directory_tiles = [directory_tiles[i] for i in order]
    canvas_tiles = [tile.copy() for tile in directory_tiles]

    axis = "horizontal" if rng.random() < 0.5 else "vertical"
    base_first_side = bool(rng.integers(0, 2))
    full_canvas, tile_ids = _build_complete_canvas(rng, canvas_tiles, axis, base_first_side)
    rows = full_canvas.shape[0] // TILE
    cols = full_canvas.shape[1] // TILE

    missing_positions = _choose_missing_positions(rng, rows, cols, axis)
    inp_canvas = full_canvas.copy()
    for r, c in missing_positions:
        tile_id = int(tile_ids[r, c])
        marker = int(canvas_tiles[tile_id][0, 0])
        inp_canvas[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = BG
        inp_canvas[r * TILE + 1, c * TILE + 1] = marker

    grid, (canvas_top, canvas_left) = _assemble_grid(rng, directory_tiles, inp_canvas)
    if _get_bg(grid) != BG:
        return None
    try:
        solved = main(grid)
    except AssertionError:
        return None

    if np.array_equal(solved, grid):
        return None

    expected_grid = grid.copy()
    expected_grid[
        canvas_top : canvas_top + full_canvas.shape[0],
        canvas_left : canvas_left + full_canvas.shape[1],
    ] = full_canvas

    if not np.array_equal(solved, expected_grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
