from __future__ import annotations

import random
from collections import Counter

import numpy as np


CORNER_NAMES = ("TL", "TR", "BL", "BR")
PALETTE = tuple(range(10))
MAX_ATTEMPTS = 300


def _mode(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).reshape(-1)]
    assert flat, "expected at least one value"
    return Counter(flat).most_common(1)[0][0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(set(int(x) for x in separators)):
        if start <= idx - 1:
            spans.append((start, idx - 1))
        start = idx + 1
    if start <= length - 1:
        spans.append((start, length - 1))
    return spans


def _corner_coords(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = map(int, shape)
    return {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }


def _corner_markers(tile: np.ndarray, separator: int) -> list[str]:
    markers = []
    for name, (r, c) in _corner_coords(tile.shape).items():
        if int(tile[r, c]) == int(separator):
            markers.append(name)
    return markers


def _single_corner_separator(tile: np.ndarray, separator: int) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(tile == separator)]
    if len(coords) != 1:
        return False
    return coords[0] in set(_corner_coords(tile.shape).values())


def _detect_separator_grid(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best = None
    for color in map(int, np.unique(g)):
        sep_rows = [r for r in range(h) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not sep_rows or not sep_cols:
            continue
        row_spans = _spans(h, sep_rows)
        col_spans = _spans(w, sep_cols)
        if not row_spans or not col_spans:
            continue

        heights = {r1 - r0 + 1 for r0, r1 in row_spans}
        widths = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(heights) != 1 or len(widths) != 1:
            continue
        tile_h = heights.pop()
        tile_w = widths.pop()
        if tile_h != tile_w:
            continue

        score = (len(row_spans) * len(col_spans), len(sep_rows) + len(sep_cols))
        if best is None or score > best[0]:
            best = (score, color, row_spans, col_spans, tile_h)

    assert best is not None, "expected a square separator grid"
    _, separator, row_spans, col_spans, tile_size = best
    return int(separator), row_spans, col_spans, int(tile_size)


def _rotated_to_corner(source_tile: np.ndarray, separator: int, target_corner: str) -> np.ndarray:
    matches = []
    for k in range(4):
        rotated = np.rot90(source_tile, k)
        if _corner_markers(rotated, separator) == [target_corner] and _single_corner_separator(rotated, separator):
            matches.append(rotated)
    assert len(matches) == 1, "expected exactly one rotation per target corner"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the unique template tile into every marked tile, rotating it so the template anchor corner lands on the marker corner.

    Input structure:
    - Full separator-colored rows and columns partition the grid into equal square tiles.
    - Exactly one tile contains a colored template plus one separator-colored corner anchor.
    - Some background tiles contain exactly one separator-colored corner marker; the rest are plain background.

    Rule:
    - Identify the template tile and keep it unchanged.
    - For each marked background tile, rotate the whole template by 0, 90, 180, or 270 degrees so the template anchor moves to that tile's marked corner.
    - Copy that rotated template into the marked tile while leaving separator lines and unmarked tiles unchanged.

    Key insight:
    - The separator-colored corner pixel is both a placement marker and an orientation code: its corner position tells which rotation the template needs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    separator, row_spans, col_spans, tile_size = _detect_separator_grid(g)
    bg = _mode(g[g != separator])

    source_tile: np.ndarray | None = None
    targets: list[tuple[int, int, str]] = []

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (tile_size, tile_size)

            has_pattern = bool(np.any((tile != bg) & (tile != separator)))
            sep_count = int(np.count_nonzero(tile == separator))
            corner_markers = _corner_markers(tile, separator)

            if has_pattern:
                assert source_tile is None, "expected exactly one template tile"
                assert sep_count == 1 and len(corner_markers) == 1
                source_tile = tile.copy()
            elif sep_count:
                assert sep_count == 1 and len(corner_markers) == 1
                assert np.all((tile == bg) | (tile == separator))
                targets.append((r0, c0, corner_markers[0]))
            else:
                assert np.all(tile == bg)

    assert source_tile is not None, "expected one template tile"
    assert targets, "expected at least one marked destination tile"

    out = g.copy()
    for r0, c0, target_corner in targets:
        out[r0 : r0 + tile_size, c0 : c0 + tile_size] = _rotated_to_corner(
            source_tile,
            separator,
            target_corner,
        )
    return out


def _neighbors4(size: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < size and 0 <= cc < size:
            out.append((rr, cc))
    return out


def _random_pattern_tile(size: int, bg: int, separator: int, anchor_corner: str, colors: list[int]) -> np.ndarray | None:
    anchor = _corner_coords((size, size))[anchor_corner]
    usable = [(r, c) for r in range(size) for c in range(size) if (r, c) != anchor]

    min_cells = max(4, size)
    max_cells = min(len(usable) - 1, size + 4, (size * size) // 2)
    if min_cells > max_cells:
        return None

    tile = np.full((size, size), bg, dtype=int)
    tile[anchor] = separator

    count = random.randint(min_cells, max_cells)
    seed = random.choice(usable)
    occupied = {seed}
    while len(occupied) < count:
        frontier = []
        for r, c in occupied:
            for rr, cc in _neighbors4(size, r, c):
                if (rr, cc) != anchor and (rr, cc) not in occupied:
                    frontier.append((rr, cc))
        remaining = [cell for cell in usable if cell not in occupied]
        if not remaining:
            break
        if frontier and random.random() < 0.8:
            occupied.add(random.choice(frontier))
        else:
            occupied.add(random.choice(remaining))

    occupied_list = list(occupied)
    base_color = colors[0]
    for r, c in occupied_list:
        tile[r, c] = base_color

    accent_cells = random.randint(1, min(4, len(occupied_list)))
    for r, c in random.sample(occupied_list, accent_cells):
        tile[r, c] = random.choice(colors[1:])

    pattern_only = tile.copy()
    pattern_only[anchor] = bg
    color_count = len(set(int(v) for v in pattern_only.flatten() if int(v) != bg))
    rotation_count = len({np.rot90(pattern_only, k).tobytes() for k in range(4)})
    used_rows = len({r for r, _ in occupied_list})
    used_cols = len({c for _, c in occupied_list})
    if color_count < 2 or rotation_count < 4 or used_rows < 2 or used_cols < 2:
        return None
    return tile


def _grid_shape(tile_rows: int, tile_cols: int, tile_size: int) -> tuple[int, int]:
    return (
        tile_rows * tile_size + (tile_rows - 1),
        tile_cols * tile_size + (tile_cols - 1),
    )


def _tile_origins(tile_rows: int, tile_cols: int, tile_size: int) -> list[tuple[int, int]]:
    return [
        (tr * (tile_size + 1), tc * (tile_size + 1))
        for tr in range(tile_rows)
        for tc in range(tile_cols)
    ]


def _assign_marker_corners(count: int) -> list[str]:
    order = list(CORNER_NAMES)
    random.shuffle(order)
    markers = order[: min(count, len(order))]
    while len(markers) < count:
        markers.append(random.choice(CORNER_NAMES))
    random.shuffle(markers)
    return markers


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.randint(2, 4)
    tile_cols = random.randint(3, 4)
    tile_size = random.randint(4, 6)
    height, width = _grid_shape(tile_rows, tile_cols, tile_size)
    if max(height, width) > 30:
        return None

    pattern_color_count = random.randint(2, 3)
    colors = random.sample(PALETTE, 2 + pattern_color_count)
    separator, bg = colors[0], colors[1]
    pattern_colors = colors[2:]

    grid = np.full((height, width), bg, dtype=int)
    for r in range(tile_size, height, tile_size + 1):
        grid[r, :] = separator
    for c in range(tile_size, width, tile_size + 1):
        grid[:, c] = separator

    origins = _tile_origins(tile_rows, tile_cols, tile_size)
    total_tiles = len(origins)
    mark_count = random.randint(3, min(total_tiles - 2, 6))

    source_idx = random.randrange(total_tiles)
    remaining = [idx for idx in range(total_tiles) if idx != source_idx]
    target_indices = random.sample(remaining, mark_count)

    anchor_corner = random.choice(CORNER_NAMES)
    source_tile = None
    for _ in range(100):
        source_tile = _random_pattern_tile(
            tile_size,
            bg,
            separator,
            anchor_corner,
            pattern_colors,
        )
        if source_tile is not None:
            break
    if source_tile is None:
        return None

    src_r0, src_c0 = origins[source_idx]
    grid[src_r0 : src_r0 + tile_size, src_c0 : src_c0 + tile_size] = source_tile

    marker_corners = _assign_marker_corners(mark_count)
    for idx, corner in zip(target_indices, marker_corners):
        r0, c0 = origins[idx]
        rr, cc = _corner_coords((tile_size, tile_size))[corner]
        grid[r0 + rr, c0 + cc] = separator

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-rotation stamp input")
