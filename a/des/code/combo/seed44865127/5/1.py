import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 9

HFLIP = "hflip"
VFLIP = "vflip"
CW = "cw"
CCW = "ccw"
OPS = (HFLIP, VFLIP, CW, CCW)

SYMBOL_TO_OP = {
    2: HFLIP,  # red
    1: VFLIP,  # blue
    3: CW,     # green
    4: CCW,    # yellow
}
OP_TO_SYMBOL = {op: color for color, op in SYMBOL_TO_OP.items()}

MARKER_COLORS = (5, 6, 7, 8)
OBJECT_COLORS = (1, 2, 3, 4)

LAYOUTS = ((2, 2), (2, 3), (3, 2))
TILE_SIZES = (7, 9)
SHAPE_DIMS = ((3, 3), (3, 5), (5, 3), (5, 5))
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200


def _mode_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    assert flat.size > 0
    return int(np.bincount(flat, minlength=10).argmax())


def _find_separator(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    best = None
    for color in map(int, np.unique(grid)):
        row_hits = np.flatnonzero(np.all(grid == color, axis=1)).tolist()
        col_hits = np.flatnonzero(np.all(grid == color, axis=0)).tolist()
        if not row_hits or not col_hits:
            continue
        score = (len(row_hits) + len(col_hits), len(row_hits) * len(col_hits), color)
        if best is None or score > best[:3]:
            best = (score[0], score[1], score[2], row_hits, col_hits)
    assert best is not None, "expected a color that forms full separator rows and columns"
    return int(best[2]), list(map(int, best[3])), list(map(int, best[4]))


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start <= sep - 1:
            spans.append((start, sep - 1))
        start = sep + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _parse_layout(grid: np.ndarray):
    separator, sep_rows, sep_cols = _find_separator(grid)
    row_spans = _spans(grid.shape[0], sep_rows)
    assert len(row_spans) in (2, 3), "expected 2 or 3 region rows"
    assert len({r1 - r0 + 1 for r0, r1 in row_spans}) == 1, "region rows must be equal height"

    col_spans = _spans(grid.shape[1], sep_cols)
    assert len(col_spans) >= 3, "expected left legend span plus 2 or 3 region columns"
    legend_span = col_spans[0]
    region_spans = col_spans[1:]
    assert legend_span[1] - legend_span[0] + 1 >= 2, "legend must be at least two columns wide"
    assert len(region_spans) in (2, 3), "expected 2 or 3 region columns"
    assert len({c1 - c0 + 1 for c0, c1 in region_spans}) == 1, "region columns must be equal width"
    assert len(row_spans) * len(region_spans) in (4, 6), "expected 4-6 total regions"

    background = _mode_color(grid[grid != separator])
    assert background != separator
    return separator, background, row_spans, region_spans, legend_span, set(sep_rows)


def _read_legend(
    grid: np.ndarray,
    background: int,
    legend_span: tuple[int, int],
    sep_rows: set[int],
) -> dict[int, str]:
    marker_col = legend_span[0]
    symbol_col = marker_col + 1
    mapping: dict[int, str] = {}
    for r in range(grid.shape[0]):
        if r in sep_rows:
            continue
        marker = int(grid[r, marker_col])
        symbol = int(grid[r, symbol_col])
        if marker == background and symbol == background:
            continue
        assert marker != background and symbol != background, "legend entries must use two visible pixels"
        op = SYMBOL_TO_OP.get(symbol)
        assert op is not None, "legend symbol color must encode one of the four transforms"
        if marker in mapping:
            assert mapping[marker] == op, "marker color must map consistently"
        mapping[marker] = op
    assert mapping, "expected at least one legend mapping"
    return mapping


def _coords_of_color(tile: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(tile == color)]


def _transform_points(
    coords: list[tuple[int, int]],
    op: str,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    rows = np.array([r for r, _ in coords], dtype=int)
    cols = np.array([c for _, c in coords], dtype=int)
    center2_r = int(rows.min() + rows.max())
    center2_c = int(cols.min() + cols.max())

    transformed: set[tuple[int, int]] = set()
    for r, c in coords:
        dr2 = 2 * r - center2_r
        dc2 = 2 * c - center2_c
        if op == HFLIP:
            new_dr2, new_dc2 = dr2, -dc2
        elif op == VFLIP:
            new_dr2, new_dc2 = -dr2, dc2
        elif op == CW:
            new_dr2, new_dc2 = dc2, -dr2
        else:
            assert op == CCW
            new_dr2, new_dc2 = -dc2, dr2

        nr2 = center2_r + new_dr2
        nc2 = center2_c + new_dc2
        assert nr2 % 2 == 0 and nc2 % 2 == 0, "rotation must land on integer cell centers"
        nr = nr2 // 2
        nc = nc2 // 2
        assert 0 <= nr < shape[0] and 0 <= nc < shape[1], "transformed object must stay in its region"
        transformed.add((int(nr), int(nc)))

    assert len(transformed) == len(coords), "geometric transform must be bijective"
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    In separator-defined regions, use each region's marker color to look up a transform
    in the left-edge legend, then apply that geometric transform to the region's main object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, background, row_spans, region_spans, legend_span, sep_rows = _parse_layout(g)
    legend = _read_legend(g, background, legend_span, sep_rows)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in region_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = [int(color) for color in np.unique(tile) if int(color) != background]
            assert separator not in colors, "region interiors must not contain separator pixels"

            counts = {color: int(np.count_nonzero(tile == color)) for color in colors}
            marker_candidates = [color for color, count in counts.items() if count == 1 and color in legend]
            assert len(marker_candidates) == 1, "each region must contain exactly one legend-mapped marker pixel"
            marker_color = marker_candidates[0]

            object_colors = [color for color in colors if color != marker_color]
            assert len(object_colors) == 1, "each region must contain exactly one monochrome object color"
            object_color = object_colors[0]
            object_coords = _coords_of_color(tile, object_color)
            assert len(object_coords) > 1, "object must have area > 1 so it is distinguishable from the marker"

            transformed = _transform_points(object_coords, legend[marker_color], tile.shape)
            new_tile = np.full(tile.shape, background, dtype=int)
            for rr, cc in transformed:
                new_tile[rr, cc] = object_color
            out[r0 : r1 + 1, c0 : c1 + 1] = new_tile

    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(h), random.randrange(w))}
    frontier = set(_neighbors4(*next(iter(cells)), h, w))

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nb in _neighbors4(cell[0], cell[1], h, w):
            if nb not in cells:
                frontier.add(nb)

    if len(cells) != area:
        return None

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _embed_centered(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=bool)
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top : top + h, left : left + w] = mask
    return canvas


def _shape_changes_under(mask: np.ndarray, op: str) -> bool:
    square = _embed_centered(mask)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(square)]
    transformed = _transform_points(coords, op, square.shape)
    return transformed != set(coords)


def _random_object_mask(op: str) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h, w = random.choice(SHAPE_DIMS)
        min_area = max(4, (h * w) // 3)
        max_area = min(h * w - 2, (2 * h * w) // 3 + 1)
        area = random.randint(min_area, max_area)
        mask = _random_connected_mask(h, w, area)
        if mask is None:
            continue
        if _shape_changes_under(mask, op):
            return mask
    return None


def _separator_rows(region_rows: int, tile_size: int) -> list[int]:
    return [i * tile_size + (i - 1) for i in range(1, region_rows)]


def _separator_cols(region_cols: int, tile_size: int) -> list[int]:
    cols = [2]
    cols.extend(3 + i * tile_size + (i - 1) for i in range(1, region_cols))
    return cols


def _region_origin(row_idx: int, col_idx: int, tile_size: int) -> tuple[int, int]:
    return row_idx * (tile_size + 1), 3 + col_idx * (tile_size + 1)


def _place_region_object(
    tile: np.ndarray,
    object_color: int,
    marker_color: int,
    op: str,
) -> bool:
    mask = _random_object_mask(op)
    if mask is None:
        return False

    mh, mw = mask.shape
    safe_size = max(mh, mw)
    safe_half = safe_size // 2
    tile_h, tile_w = tile.shape
    center_r = random.randint(safe_half, tile_h - 1 - safe_half)
    center_c = random.randint(safe_half, tile_w - 1 - safe_half)

    top = center_r - mh // 2
    left = center_c - mw // 2
    object_cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        tile[rr, cc] = object_color
        object_cells.append((rr, cc))

    _transform_points(object_cells, op, tile.shape)

    forbidden = set(object_cells)
    for r, c in object_cells:
        for rr in range(max(0, r - 1), min(tile_h, r + 2)):
            for cc in range(max(0, c - 1), min(tile_w, c + 2)):
                forbidden.add((rr, cc))

    marker_candidates = [
        (r, c)
        for r in range(tile_h)
        for c in range(tile_w)
        if (r, c) not in forbidden
    ]
    if not marker_candidates:
        return False

    mr, mc = random.choice(marker_candidates)
    tile[mr, mc] = marker_color
    return True


def _all_non_separator_rows(h: int, sep_rows: list[int]) -> list[int]:
    sep = set(sep_rows)
    return [r for r in range(h) if r not in sep]


def _gen_attempt() -> np.ndarray | None:
    region_rows, region_cols = random.choice(LAYOUTS)
    tile_size = random.choice(TILE_SIZES)
    h = region_rows * tile_size + (region_rows - 1)
    w = 3 + region_cols * tile_size + (region_cols - 1)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    sep_rows = _separator_rows(region_rows, tile_size)
    sep_cols = _separator_cols(region_cols, tile_size)
    for r in sep_rows:
        grid[r, :] = SEPARATOR
    for c in sep_cols:
        grid[:, c] = SEPARATOR

    marker_colors = list(MARKER_COLORS)
    random.shuffle(marker_colors)
    op_to_marker = {op: marker for op, marker in zip(OPS, marker_colors)}

    legend_rows = random.sample(_all_non_separator_rows(h, sep_rows), k=len(OPS))
    random.shuffle(legend_rows)
    for row, op in zip(legend_rows, OPS):
        grid[row, 0] = op_to_marker[op]
        grid[row, 1] = OP_TO_SYMBOL[op]

    ops = list(OPS)
    while len(ops) < region_rows * region_cols:
        ops.append(random.choice(OPS))
    random.shuffle(ops)

    region_index = 0
    for rr in range(region_rows):
        for cc in range(region_cols):
            r0, c0 = _region_origin(rr, cc, tile_size)
            tile = grid[r0 : r0 + tile_size, c0 : c0 + tile_size]
            object_color = random.choice(OBJECT_COLORS)
            op = ops[region_index]
            marker_color = op_to_marker[op]
            if not _place_region_object(tile, object_color, marker_color, op):
                return None
            region_index += 1

    out = main(grid)
    assert out.shape == grid.shape
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
