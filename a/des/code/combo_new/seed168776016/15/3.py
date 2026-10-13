from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
DICT_ROWS = 2
DICT_COLS = 2
SPLIT_ROW = 2
MAX_GENERATION_ATTEMPTS = 400

SIGNATURE_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 2)),
    ((0, 0), (0, 1), (1, 2)),
    ((0, 2), (1, 0), (1, 1)),
)

TEXTURE_LIBRARY: tuple[np.ndarray, ...] = (
    np.array([[1], [0]], dtype=bool),
    np.array([[1, 0]], dtype=bool),
    np.array([[1, 0], [0, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _collect_component(grid: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _normalize_points(points: np.ndarray | list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    pts = [tuple(map(int, p)) for p in points]
    assert pts, "expected at least one point"
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _tile_mask(tile: np.ndarray, shape: tuple[int, int], row_phase: int = 0, col_phase: int = 0) -> np.ndarray:
    th, tw = map(int, tile.shape)
    rows = (np.arange(shape[0]) + int(row_phase)) % th
    cols = (np.arange(shape[1]) + int(col_phase)) % tw
    return tile[np.ix_(rows, cols)]


def _extract_minimal_tile(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    assert h > 0 and w > 0 and bool(mask.any()), "sample texture must be non-empty"

    ph = 1
    for cand in range(1, h + 1):
        if np.array_equal(mask, np.take(mask, np.arange(h) % cand, axis=0)):
            ph = cand
            break

    pw = 1
    for cand in range(1, w + 1):
        if np.array_equal(mask, np.take(mask, np.arange(w) % cand, axis=1)):
            pw = cand
            break

    tile = mask[:ph, :pw].copy()
    assert np.array_equal(mask, _tile_mask(tile, mask.shape)), "sample zone must be a tiled texture"
    return tile


def _dictionary_corner_and_color(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        color = int(grid[corner])
        if color != bg:
            return corner, color
    raise AssertionError("expected a corner-touching dictionary")


def _parse_dictionary(
    grid: np.ndarray, bg: int
) -> tuple[int, dict[tuple[tuple[int, int], ...], np.ndarray], tuple[int, int, int, int]]:
    start, sep_color = _dictionary_corner_and_color(grid, bg)
    comp = _collect_component(grid, start, sep_color)
    r0, c0, r1, c1 = _bbox(comp)
    view = grid[r0 : r1 + 1, c0 : c1 + 1]

    row_lines = [r for r in range(view.shape[0]) if bool(np.all(view[r, :] == sep_color))]
    col_lines = [c for c in range(view.shape[1]) if bool(np.all(view[:, c] == sep_color))]
    assert len(row_lines) >= 3 and len(col_lines) >= 3, "expected a separator lattice in the corner"
    assert row_lines[0] == 0 and row_lines[-1] == view.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == view.shape[1] - 1

    mapping: dict[tuple[tuple[int, int], ...], np.ndarray] = {}
    for rr in range(len(row_lines) - 1):
        for cc in range(len(col_lines) - 1):
            if row_lines[rr + 1] - row_lines[rr] <= 1 or col_lines[cc + 1] - col_lines[cc] <= 1:
                continue
            cell = view[row_lines[rr] + 1 : row_lines[rr + 1], col_lines[cc] + 1 : col_lines[cc + 1]]
            blank_rows = [r for r in range(cell.shape[0]) if bool(np.all(cell[r, :] == bg))]
            assert blank_rows, "each dictionary cell needs a blank separator row"
            split = int(blank_rows[0])
            signature_points = np.argwhere(cell[:split, :] == YELLOW)
            sample_mask = cell[split + 1 :, :] == YELLOW
            assert signature_points.size > 0, "dictionary cell must contain signature seeds"
            assert sample_mask.size > 0 and bool(sample_mask.any()), "dictionary cell must contain a texture sample"
            signature = _normalize_points(signature_points)
            assert signature not in mapping, "dictionary signatures must be unique"
            mapping[signature] = _extract_minimal_tile(sample_mask)

    assert len(mapping) == DICT_ROWS * DICT_COLS, "expected one signature-texture pair per dictionary cell"
    return sep_color, mapping, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each framed seed pattern to the corner dictionary, then fill the frame interior with the matching texture in the frame's own color.

    Input structure:
    - The background is black.
    - One corner contains a separator-colored dictionary lattice whose cells each show a yellow signature seed pattern above a yellow texture sample.
    - Elsewhere are several one-pixel-thick hollow rectangular frames of distinct colors, and each frame interior contains only black plus a few yellow seed pixels.

    Rule:
    - Read every dictionary cell by splitting it at the blank row: the yellow pixels above that row define the cell's signature, and the yellow sample below it defines a repeating texture tile.
    - For each hollow frame, extract the yellow seed pattern inside the frame, normalize it up to translation, and match it to the identical dictionary signature without rotation or reflection.
    - Replace the frame interior with a tiled copy of the matched texture, recolored from yellow to the frame's border color; the dictionary itself stays unchanged.

    Key insight:
    - The yellow seed arrangement is a position-coded key that selects a texture independently of color, while the frame border color tells you which color that selected texture should use in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    sep_color, signature_to_tile, _dict_bbox = _parse_dictionary(g, bg)

    out = g.copy()
    colors = sorted(set(map(int, np.unique(g))) - {bg, YELLOW, sep_color})
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, "frame color must appear"
        points = [tuple(map(int, pt)) for pt in coords]
        r0, c0, r1, c1 = _bbox(points)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose an interior"

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        frame_mask = np.zeros(sub.shape, dtype=bool)
        frame_mask[0, :] = True
        frame_mask[-1, :] = True
        frame_mask[:, 0] = True
        frame_mask[:, -1] = True
        assert np.array_equal(sub == color, frame_mask), "foreground frame must be a one-pixel-thick hollow rectangle"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_points = np.argwhere(interior == YELLOW)
        assert seed_points.size > 0, "each frame must contain seeds"
        signature = _normalize_points(seed_points)
        assert signature in signature_to_tile, "frame signature must exist in the dictionary"
        tile = signature_to_tile[signature]

        fill = _tile_mask(tile, interior.shape)
        out_view = out[r0 + 1 : r1, c0 + 1 : c1]
        out_view[:] = bg
        out_view[fill] = color

    return out


def _corner_origin(corner: str, grid_h: int, grid_w: int, patch_h: int, patch_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, grid_w - patch_w
    if corner == "BL":
        return grid_h - patch_h, 0
    if corner == "BR":
        return grid_h - patch_h, grid_w - patch_w
    raise AssertionError(corner)


def _choose_canvas(dict_h: int, dict_w: int) -> tuple[int, int]:
    min_h = max(24, dict_h + 6)
    min_w = max(24, dict_w + 6)
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)
    if random.random() < 0.6:
        delta = random.randint(-3, 3)
        w = min(30, max(min_w, h + delta))
    return h, w


def _render_dictionary_cell(cell_h: int, cell_w: int, signature: tuple[tuple[int, int], ...], tile: np.ndarray) -> np.ndarray:
    cell = np.full((cell_h, cell_w), BG, dtype=int)
    sig_h = max(r for r, _ in signature) + 1
    sig_w = max(c for _, c in signature) + 1
    sig_col = random.randint(0, cell_w - sig_w)
    for r, c in signature:
        cell[r, sig_col + c] = YELLOW

    sample_h = cell_h - SPLIT_ROW - 1
    sample_mask = _tile_mask(
        tile,
        (sample_h, cell_w),
        row_phase=random.randrange(tile.shape[0]),
        col_phase=random.randrange(tile.shape[1]),
    )
    cell[SPLIT_ROW + 1 :, :][sample_mask] = YELLOW
    return cell


def _stamp_dictionary(
    grid: np.ndarray,
    corner: str,
    sep_color: int,
    cell_h: int,
    cell_w: int,
    pairs: list[tuple[tuple[tuple[int, int], ...], np.ndarray]],
) -> tuple[int, int, int, int]:
    patch_h = DICT_ROWS * cell_h + DICT_ROWS + 1
    patch_w = DICT_COLS * cell_w + DICT_COLS + 1
    r0, c0 = _corner_origin(corner, grid.shape[0], grid.shape[1], patch_h, patch_w)
    r1 = r0 + patch_h - 1
    c1 = c0 + patch_w - 1

    row_lines = [r0 + i * (cell_h + 1) for i in range(DICT_ROWS + 1)]
    col_lines = [c0 + i * (cell_w + 1) for i in range(DICT_COLS + 1)]
    for rr in row_lines:
        grid[rr, c0 : c1 + 1] = sep_color
    for cc in col_lines:
        grid[r0 : r1 + 1, cc] = sep_color

    idx = 0
    for dr in range(DICT_ROWS):
        for dc in range(DICT_COLS):
            signature, tile = pairs[idx]
            idx += 1
            cell = _render_dictionary_cell(cell_h, cell_w, signature, tile)
            top = row_lines[dr] + 1
            left = col_lines[dc] + 1
            grid[top : top + cell_h, left : left + cell_w] = cell

    return r0, c0, r1, c1


def _place_frame(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    color: int,
    signature: tuple[tuple[int, int], ...],
    tile: np.ndarray,
    interior_h: int,
    interior_w: int,
) -> bool:
    outer_h = interior_h + 2
    outer_w = interior_w + 2
    max_r0 = grid.shape[0] - outer_h - 1
    max_c0 = grid.shape[1] - outer_w - 1
    if max_r0 < 1 or max_c0 < 1:
        return False

    sig_h = max(r for r, _ in signature) + 1
    sig_w = max(c for _, c in signature) + 1
    assert interior_h >= sig_h and interior_w >= sig_w

    for _ in range(250):
        r0 = random.randint(1, max_r0)
        c0 = random.randint(1, max_c0)
        r1 = r0 + outer_h
        c1 = c0 + outer_w
        if bool(occupied[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1].any()):
            continue

        grid[r0, c0 : c0 + outer_w] = color
        grid[r0 + outer_h - 1, c0 : c0 + outer_w] = color
        grid[r0 : r0 + outer_h, c0] = color
        grid[r0 : r0 + outer_h, c0 + outer_w - 1] = color
        expected[r0, c0 : c0 + outer_w] = color
        expected[r0 + outer_h - 1, c0 : c0 + outer_w] = color
        expected[r0 : r0 + outer_h, c0] = color
        expected[r0 : r0 + outer_h, c0 + outer_w - 1] = color

        seed_r = random.randint(0, interior_h - sig_h)
        seed_c = random.randint(0, interior_w - sig_w)
        for sr, sc in signature:
            grid[r0 + 1 + seed_r + sr, c0 + 1 + seed_c + sc] = YELLOW

        fill = _tile_mask(tile, (interior_h, interior_w))
        exp_view = expected[r0 + 1 : r0 + 1 + interior_h, c0 + 1 : c0 + 1 + interior_w]
        exp_view[:] = BG
        exp_view[fill] = color

        occupied[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1] = True
        return True

    return False


def _generate_once() -> np.ndarray | None:
    cell_h = random.randint(7, 8)
    cell_w = random.randint(7, 8)
    dict_h = DICT_ROWS * cell_h + DICT_ROWS + 1
    dict_w = DICT_COLS * cell_w + DICT_COLS + 1
    grid_h, grid_w = _choose_canvas(dict_h, dict_w)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    corner = random.choice(("TL", "TR", "BL", "BR"))
    frame_count = random.randint(2, 4)

    palette = [color for color in range(1, 10) if color != YELLOW]
    random.shuffle(palette)
    sep_color = palette[0]
    frame_colors = palette[1 : 1 + frame_count]

    textures = list(TEXTURE_LIBRARY)
    random.shuffle(textures)
    signatures = list(SIGNATURE_LIBRARY)
    random.shuffle(signatures)
    pairs = list(zip(signatures, textures))
    random.shuffle(pairs)
    signature_to_tile = {signature: tile for signature, tile in pairs}

    dict_bbox = _stamp_dictionary(grid, corner, sep_color, cell_h, cell_w, pairs)

    expected = grid.copy()
    occupied = np.zeros_like(grid, dtype=bool)
    r0, c0, r1, c1 = dict_bbox
    occupied[max(0, r0 - 1) : min(grid_h, r1 + 2), max(0, c0 - 1) : min(grid_w, c1 + 2)] = True

    chosen_signatures = random.sample(list(signature_to_tile), frame_count)
    dims = [(random.randint(4, 7), random.randint(4, 8)) for _ in range(frame_count)]
    order = sorted(range(frame_count), key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)

    for idx in order:
        signature = chosen_signatures[idx]
        tile = signature_to_tile[signature]
        interior_h, interior_w = dims[idx]
        if not _place_frame(
            grid,
            expected,
            occupied,
            frame_colors[idx],
            signature,
            tile,
            interior_h,
            interior_w,
        ):
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
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
