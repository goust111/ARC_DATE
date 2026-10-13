from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
RED = 2
SEPARATOR_WIDTH = 1
MAX_GENERATION_ATTEMPTS = 400

STOP_TEMPLATES = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
]
VERTICALISH_TEMPLATES = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
]


def _separator_lines(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = map(int, g.shape)
    candidates = []
    for color in np.unique(g):
        color = int(color)
        rows = [r for r in range(h) if np.all(g[r, :] == color)]
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for sep in separators:
        assert start < sep
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length
    spans.append((start, length - 1))
    return spans


def _mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _tile_rows_cols(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], list[list[np.ndarray]]]:
    sep_color, sep_rows, sep_cols = _separator_lines(g)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)
    tile_hs = {r1 - r0 + 1 for r0, r1 in row_spans}
    tile_ws = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(tile_hs) == 1 and len(tile_ws) == 1, "expected equal-sized tiles"
    tiles = [
        [g[r0 : r1 + 1, c0 : c1 + 1].copy() for c0, c1 in col_spans]
        for r0, r1 in row_spans
    ]
    return sep_color, row_spans, col_spans, tiles


def _vertical_bar_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0 or coords.shape[0] < 2:
        return False
    cols = sorted(set(map(int, coords[:, 1])))
    if len(cols) != 1:
        return False
    rows = sorted(map(int, coords[:, 0]))
    return rows == list(range(rows[0], rows[-1] + 1))


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    sep_color, row_spans, col_spans, tiles = _tile_rows_cols(g)
    tile_rows = len(row_spans)
    tile_cols = len(col_spans)

    mixed_tiles = []
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            tile = tiles[tr][tc]
            if len(np.unique(tile)) > 1:
                mixed_tiles.append(tile)
    assert mixed_tiles, "expected at least one occupied non-barrier tile"

    bg = _mode(np.concatenate([tile.reshape(-1) for tile in mixed_tiles]))
    mixed_colors = sorted(set(int(v) for tile in mixed_tiles for v in np.unique(tile)))
    sprite_colors = [color for color in mixed_colors if color != bg]
    assert len(sprite_colors) == 1, "expected one sprite color inside occupied tiles"
    sprite_color = int(sprite_colors[0])

    barrier_candidates = []
    for tr in range(tile_rows):
        row_colors = []
        ok = True
        for tc in range(tile_cols):
            vals = np.unique(tiles[tr][tc])
            if len(vals) != 1:
                ok = False
                break
            color = int(vals[0])
            if color == bg:
                ok = False
                break
            row_colors.append(color)
        if ok and len(set(row_colors)) == 1:
            barrier_candidates.append((tr, row_colors[0]))
    assert len(barrier_candidates) == 1, "expected one solid barrier tile row"
    barrier_r, barrier_color = barrier_candidates[0]
    assert 0 < barrier_r < tile_rows - 1

    columns: list[list[dict[str, object]]] = [[] for _ in range(tile_cols)]
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            tile = tiles[tr][tc]
            vals = set(map(int, np.unique(tile)))
            if tr == barrier_r:
                assert vals == {barrier_color}, "barrier row tiles must be solid"
                continue
            if tr > barrier_r:
                assert vals == {bg}, "below the barrier row every input tile must be empty"
                continue

            assert vals.issubset({bg, sprite_color}), "upper tiles may contain only background and sprite"
            if vals == {bg}:
                continue

            mask = tile == sprite_color
            assert np.any(mask) and np.any(tile == bg), "sprites must sit inside otherwise empty tiles"
            columns[tc].append(
                {
                    "src_row": tr,
                    "tile": tile.copy(),
                    "pass_through": _vertical_bar_mask(mask),
                }
            )

    assert any(columns), "expected at least one occupied tile"
    return {
        "grid": g,
        "separator": int(sep_color),
        "row_spans": row_spans,
        "col_spans": col_spans,
        "tile_rows": tile_rows,
        "tile_cols": tile_cols,
        "bg": int(bg),
        "sprite": int(sprite_color),
        "barrier_row": int(barrier_r),
        "barrier_color": int(barrier_color),
        "columns": columns,
    }


def _paint_tile(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    tile_r: int,
    tile_c: int,
    patch: np.ndarray,
) -> None:
    r0, r1 = row_spans[tile_r]
    c0, c1 = col_spans[tile_c]
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop tile contents downward inside a separator lattice, with only one-column vertical bars passing through the red barrier row.

    Input structure:
    - Uniform separator lines partition the grid into equal-sized tiles.
    - One whole tile row is filled with a solid red color, forming a coarse barrier row.
    - Some tiles above the barrier contain a non-red sprite on background; all tiles below the barrier are empty.

    Rule:
    - Keep the separator grid fixed and move each occupied tile as a whole within its tile column.
    - If a tile's sprite is exactly a one-cell-wide vertical bar, that tile passes through the barrier, lands in the lowest available tile slots of its column, and clears the crossed red barrier tile.
    - All other occupied tiles stop above the barrier; multiple such tiles in one column stack immediately above it while preserving their original top-to-bottom order.

    Key insight:
    - Gravity is lifted from pixels to tiles: the moving unit is the whole tile, but pass-through is decided by the tile's internal sprite shape rather than by the tile itself.
    """
    scene = _extract_scene(grid)
    g = scene["grid"]
    row_spans = scene["row_spans"]
    col_spans = scene["col_spans"]
    tile_rows = int(scene["tile_rows"])
    bg = int(scene["bg"])
    barrier_r = int(scene["barrier_row"])
    barrier_color = int(scene["barrier_color"])

    out = g.copy()
    for tr in range(tile_rows):
        for tc in range(int(scene["tile_cols"])):
            fill = barrier_color if tr == barrier_r else bg
            patch = np.full((row_spans[0][1] - row_spans[0][0] + 1, col_spans[0][1] - col_spans[0][0] + 1), fill, dtype=int)
            _paint_tile(out, row_spans, col_spans, tr, tc, patch)

    for tc, items in enumerate(scene["columns"]):
        if not items:
            continue

        stop_tiles = [item["tile"] for item in items if not item["pass_through"]]
        pass_tiles = [item["tile"] for item in items if item["pass_through"]]
        assert len(stop_tiles) <= barrier_r
        assert len(pass_tiles) <= tile_rows - barrier_r - 1

        if pass_tiles:
            _paint_tile(
                out,
                row_spans,
                col_spans,
                barrier_r,
                tc,
                np.full_like(pass_tiles[0], bg),
            )

        for idx, tile in enumerate(stop_tiles):
            dest_r = barrier_r - len(stop_tiles) + idx
            _paint_tile(out, row_spans, col_spans, dest_r, tc, tile)

        for idx, tile in enumerate(pass_tiles):
            dest_r = tile_rows - len(pass_tiles) + idx
            _paint_tile(out, row_spans, col_spans, dest_r, tc, tile)

    return out


def _base_grid(tile_rows: int, tile_cols: int, tile_size: int, separator: int, bg: int) -> np.ndarray:
    h = tile_rows * tile_size + (tile_rows - 1) * SEPARATOR_WIDTH
    w = tile_cols * tile_size + (tile_cols - 1) * SEPARATOR_WIDTH
    grid = np.full((h, w), bg, dtype=int)
    grid[tile_size :: tile_size + SEPARATOR_WIDTH, :] = separator
    grid[:, tile_size :: tile_size + SEPARATOR_WIDTH] = separator
    return grid


def _spans_from_size(tile_count: int, tile_size: int) -> list[tuple[int, int]]:
    return [
        (idx * (tile_size + SEPARATOR_WIDTH), idx * (tile_size + SEPARATOR_WIDTH) + tile_size - 1)
        for idx in range(tile_count)
    ]


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out


def _stop_patch(tile_size: int, bg: int, sprite: int, *, verticalish: bool) -> np.ndarray:
    templates = VERTICALISH_TEMPLATES if verticalish else STOP_TEMPLATES
    for _ in range(200):
        mask = _transform_mask(random.choice(templates))
        h, w = mask.shape
        if h > tile_size or w > tile_size:
            continue
        if verticalish and (h < 2 or w < 2 or h < w):
            continue
        r0 = random.randint(0, tile_size - h)
        c0 = random.randint(0, tile_size - w)
        patch = np.full((tile_size, tile_size), bg, dtype=int)
        patch[r0 : r0 + h, c0 : c0 + w][mask] = sprite
        if _vertical_bar_mask(patch == sprite):
            continue
        return patch
    raise RuntimeError("failed to create a non-pass stop patch")


def _pass_patch(tile_size: int, bg: int, sprite: int) -> np.ndarray:
    patch = np.full((tile_size, tile_size), bg, dtype=int)
    height = random.randint(2, tile_size)
    top = random.randint(0, tile_size - height)
    col = random.randrange(tile_size)
    patch[top : top + height, col] = sprite
    return patch


def _nonfinal_rows(barrier_r: int, count: int) -> list[int]:
    final_rows = list(range(barrier_r - count, barrier_r))
    choices = list(range(barrier_r))
    for _ in range(200):
        rows = sorted(random.sample(choices, count))
        if rows != final_rows:
            return rows
    raise RuntimeError("failed to pick moving source rows")


def _render_expected(
    tile_rows: int,
    tile_cols: int,
    tile_size: int,
    separator: int,
    bg: int,
    barrier_r: int,
    barrier_color: int,
    stop_specs: dict[int, list[tuple[int, np.ndarray]]],
    pass_specs: dict[int, tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    row_spans = _spans_from_size(tile_rows, tile_size)
    col_spans = _spans_from_size(tile_cols, tile_size)
    inp = _base_grid(tile_rows, tile_cols, tile_size, separator, bg)
    out = _base_grid(tile_rows, tile_cols, tile_size, separator, bg)

    barrier_tile = np.full((tile_size, tile_size), barrier_color, dtype=int)
    for tc in range(tile_cols):
        _paint_tile(inp, row_spans, col_spans, barrier_r, tc, barrier_tile)
        _paint_tile(out, row_spans, col_spans, barrier_r, tc, barrier_tile)

    for tc, specs in stop_specs.items():
        ordered = sorted(specs, key=lambda item: item[0])
        for src_row, patch in ordered:
            _paint_tile(inp, row_spans, col_spans, src_row, tc, patch)
        for idx, (_src_row, patch) in enumerate(ordered):
            dest_r = barrier_r - len(ordered) + idx
            _paint_tile(out, row_spans, col_spans, dest_r, tc, patch)

    for tc, (src_row, patch) in pass_specs.items():
        _paint_tile(inp, row_spans, col_spans, src_row, tc, patch)
        _paint_tile(out, row_spans, col_spans, barrier_r, tc, np.full((tile_size, tile_size), bg, dtype=int))
        _paint_tile(out, row_spans, col_spans, tile_rows - 1, tc, patch)

    return inp, out


def _build_specs(
    tile_rows: int,
    tile_cols: int,
    tile_size: int,
    bg: int,
    sprite: int,
    barrier_r: int,
) -> tuple[dict[int, list[tuple[int, np.ndarray]]], dict[int, tuple[int, np.ndarray]]]:
    cols = list(range(tile_cols))
    random.shuffle(cols)

    stack_col = cols.pop()
    stack_count = random.randint(2, min(3, barrier_r - 1))
    stop_specs: dict[int, list[tuple[int, np.ndarray]]] = {
        stack_col: [
            (src_row, _stop_patch(tile_size, bg, sprite, verticalish=(idx == 0)))
            for idx, src_row in enumerate(_nonfinal_rows(barrier_r, stack_count))
        ]
    }

    max_extra_stop = min(2, len(cols))
    extra_stop_cols = random.sample(cols, random.randint(0, max_extra_stop))
    for tc in extra_stop_cols:
        cols.remove(tc)
        count = 1 if barrier_r == 3 or random.random() < 0.7 else 2
        count = min(count, barrier_r - 1)
        stop_specs[tc] = [
            (src_row, _stop_patch(tile_size, bg, sprite, verticalish=False))
            for src_row in _nonfinal_rows(barrier_r, count)
        ]

    max_pass = 1 if tile_rows - barrier_r - 1 == 1 else 2
    pass_count = random.randint(1, min(max_pass, len(cols)))
    pass_cols = random.sample(cols, pass_count)
    pass_specs = {
        tc: (random.randrange(barrier_r), _pass_patch(tile_size, bg, sprite))
        for tc in pass_cols
    }

    return stop_specs, pass_specs


def _validate_generated_input(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        scene = _extract_scene(inp)
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    barrier_r = int(scene["barrier_row"])
    tile_rows = int(scene["tile_rows"])
    pass_cols = 0
    stacked_stop = False
    wide_stop = False
    for items in scene["columns"]:
        stop_count = sum(1 for item in items if not item["pass_through"])
        pass_count = sum(1 for item in items if item["pass_through"])
        if pass_count:
            pass_cols += 1
        if stop_count >= 2:
            stacked_stop = True
        for item in items:
            if not item["pass_through"]:
                tile = item["tile"]
                cols = np.flatnonzero(np.any(tile == scene["sprite"], axis=0))
                rows = np.flatnonzero(np.any(tile == scene["sprite"], axis=1))
                if len(cols) >= 2 and len(rows) >= 2:
                    wide_stop = True
        if pass_count > 1:
            return False
        if stop_count > barrier_r or pass_count > tile_rows - barrier_r - 1:
            return False

    if pass_cols < 1 or not stacked_stop or not wide_stop:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.choice([5, 6])
    tile_cols = random.randint(4, 6)
    tile_size = random.choice([3, 4])
    barrier_r = random.randint(3, tile_rows - 2)

    separator = random.choice([1, 5, 6, 8])
    sprite = random.choice([3, 4, 7, 8, 9])
    if separator == sprite:
        return None

    stop_specs, pass_specs = _build_specs(tile_rows, tile_cols, tile_size, BLACK, sprite, barrier_r)
    inp, expected = _render_expected(
        tile_rows,
        tile_cols,
        tile_size,
        separator,
        BLACK,
        barrier_r,
        RED,
        stop_specs,
        pass_specs,
    )
    if not _validate_generated_input(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-gravity input")
