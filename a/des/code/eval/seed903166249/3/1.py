import random
from collections import deque

import numpy as np


BG = 0
SEP_COLOR = 5
DRAW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
LAYER_COUNTS = (2, 3, 4)
GEN_RETRIES = 400


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
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        value = int(grid[0, c])
        if value == BG:
            continue
        if not np.all(grid[:, c] == value):
            continue
        if c == 0 or c == w - 1:
            continue
        if not np.any(grid[:, :c] != BG):
            continue
        if not np.any(grid[:, c + 1 :] != BG):
            continue
        candidates.append((c, value))
    assert len(candidates) == 1, "expected one full-height separator column"
    return candidates[0]


def _parse_palette(right: np.ndarray) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for row in right:
        nz = np.flatnonzero(row != BG)
        if len(nz) == 0:
            continue
        start, end = int(nz[0]), int(nz[-1])
        assert len(nz) == end - start + 1, "palette row must be a single contiguous run"
        colors = [int(v) for v in row[start : end + 1]]
        mapping[len(colors)] = colors
    assert mapping, "palette must contain at least one row"
    return mapping


def _component_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_layers(left: np.ndarray, comp: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], list[np.ndarray]]:
    r0, c0, r1, c1 = _component_bbox(comp)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp:
        local[r - r0, c - c0] = True
    assert not local.all(), "frame group must have a black interior"

    remaining = local.copy()
    top, left_col = 0, 0
    bottom, right_col = local.shape[0] - 1, local.shape[1] - 1
    layers: list[np.ndarray] = []

    while remaining.any():
        border = np.zeros_like(local, dtype=bool)
        border[top, left_col : right_col + 1] = True
        border[bottom, left_col : right_col + 1] = True
        border[top : bottom + 1, left_col] = True
        border[top : bottom + 1, right_col] = True
        assert np.all(remaining[border]), "component must be nested rectangular borders"
        layers.append(border.copy())
        remaining[border] = False
        top += 1
        bottom -= 1
        left_col += 1
        right_col -= 1
        if top > bottom or left_col > right_col:
            break

    assert not remaining.any(), "unexpected pixels remain after peeling frame layers"
    return (r0, c0, r1, c1), layers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_col, _sep_color = _find_separator(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    palette = _parse_palette(right)

    comps = _components(left != BG)
    assert comps, "left side must contain frame groups"

    out = g.copy()
    for comp in comps:
        (r0, c0, _r1, _c1), layers = _extract_layers(left, comp)
        count = len(layers)
        assert count in palette, "missing palette row for layer count"
        colors = palette[count]
        for layer_idx, layer_mask in enumerate(layers):
            rr, cc = np.where(layer_mask)
            out[r0 + rr, c0 + cc] = colors[layer_idx]
    return out


def _draw_nested_frame(height: int, width: int, colors: list[int]) -> np.ndarray:
    tile = np.full((height, width), BG, dtype=int)
    for layer, color in enumerate(colors):
        top = layer
        bottom = height - 1 - layer
        left = layer
        right = width - 1 - layer
        tile[top, left : right + 1] = color
        tile[bottom, left : right + 1] = color
        tile[top : bottom + 1, left] = color
        tile[top : bottom + 1, right] = color
    return tile


def _group_spec() -> tuple[int, int, int]:
    layers = random.choice(LAYER_COUNTS)
    min_size = 2 * layers + 1
    height = random.randint(min_size, min(min_size + 4, 11))
    width = random.randint(min_size, min(min_size + 4, 11))
    return layers, height, width


def _random_color_row(length: int) -> list[int]:
    return random.sample(DRAW_COLORS, length)


def _pick_positions(limit: int, count: int) -> list[int] | None:
    choices = list(range(limit))
    for _ in range(100):
        picked = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _place_groups(
    height: int, width: int, tiles: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    occ = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(tiles)), key=lambda i: int(tiles[i].shape[0] * tiles[i].shape[1]), reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(tiles)

    for idx in order:
        tile = tiles[idx]
        th, tw = tile.shape
        for _ in range(250):
            r = random.randint(1, height - th - 1)
            c = random.randint(1, width - tw - 1)
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(height, r + th + 1)
            c1 = min(width, c + tw + 1)
            if occ[r0:r1, c0:c1].any():
                continue
            occ[r0:r1, c0:c1] = True
            placements[idx] = (r, c)
            break
        if placements[idx] is None:
            return None
    return [pos for pos in placements if pos is not None]


def _expected_output(
    grid: np.ndarray,
    placements: list[tuple[int, int]],
    specs: list[tuple[int, int, int]],
    palette: dict[int, list[int]],
) -> np.ndarray:
    out = grid.copy()
    for (r, c), (layers, height, width) in zip(placements, specs):
        out[r : r + height, c : c + width] = _draw_nested_frame(height, width, palette[layers])
    return out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(2, 4)
    specs = [_group_spec() for _ in range(group_count)]

    max_h = max(height for _layers, height, _width in specs)
    max_w = max(width for _layers, _height, width in specs)
    padded_area = sum((height + 2) * (width + 2) for _layers, height, width in specs)

    min_h = max(15, max_h + 3)
    min_left_w = max(13, max_w + 3)
    height = random.randint(min_h, 24)
    min_left_w = max(min_left_w, (padded_area + height - 1) // height)
    if min_left_w > 20:
        return None
    left_width = random.randint(min_left_w, 20)
    right_width = random.randint(6, 9)

    distinct_layers = sorted({layers for layers, _height, _width in specs})
    palette_rows = {layers: _random_color_row(layers) for layers in distinct_layers}

    tiles: list[np.ndarray] = []
    for layers, tile_h, tile_w in specs:
        original = _random_color_row(layers)
        if original == palette_rows[layers]:
            original = original[1:] + original[:1]
        tiles.append(_draw_nested_frame(tile_h, tile_w, original))

    placements = _place_groups(height, left_width, tiles)
    if placements is None:
        return None

    total_width = left_width + 1 + right_width
    grid = np.full((height, total_width), BG, dtype=int)
    grid[:, left_width] = SEP_COLOR

    for tile, (r, c) in zip(tiles, placements):
        th, tw = tile.shape
        grid[r : r + th, c : c + tw] = tile

    palette_row_positions = _pick_positions(height, len(distinct_layers))
    if palette_row_positions is None:
        return None
    random.shuffle(distinct_layers)
    for row, layers in zip(palette_row_positions, distinct_layers):
        colors = palette_rows[layers]
        start = random.randint(0, right_width - layers)
        grid[row, left_width + 1 + start : left_width + 1 + start + layers] = colors

    expected = _expected_output(grid, placements, specs, palette_rows)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
