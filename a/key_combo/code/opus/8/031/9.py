import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

ENDPOINT_COLORS = (BLUE, RED, GREEN, ORANGE)
TILE_INTERIOR = 3
TILE_STRIDE = 4
GEN_TILE_ROWS = 4
GEN_TILE_COLS = 4
MAX_GENERATION_ATTEMPTS = 400


def _tile_spans(separator_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, is_separator in enumerate(separator_mask.tolist()):
        if is_separator:
            if start is not None:
                spans.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        spans.append((start, len(separator_mask) - 1))

    assert spans, "expected at least one tile span"
    assert spans[0][0] == 1 and spans[-1][1] == len(separator_mask) - 2
    assert all(end - start + 1 == TILE_INTERIOR for start, end in spans)
    for (prev_start, prev_end), (next_start, next_end) in zip(spans, spans[1:]):
        assert next_start - prev_end == 2, "expected one-cell separators between tiles"
    return spans


def _parse_lattice(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_sep = np.all(g == GREY, axis=1)
    col_sep = np.all(g == GREY, axis=0)
    row_spans = _tile_spans(row_sep)
    col_spans = _tile_spans(col_sep)
    assert len(row_spans) in {3, 4}
    assert len(col_spans) in {3, 4}
    return row_spans, col_spans


def _scan_endpoints(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[list[int], dict[int, list[dict[str, int]]]]:
    g = np.asarray(grid, dtype=int)
    structural = {BLACK, GREY}
    ordered_colors: list[int] = []
    by_color: dict[int, list[dict[str, int]]] = {}

    for tile_r, (r0, _r1) in enumerate(row_spans):
        for tile_c, (c0, _c1) in enumerate(col_spans):
            for sub_r in range(TILE_INTERIOR):
                for sub_c in range(TILE_INTERIOR):
                    r = r0 + sub_r
                    c = c0 + sub_c
                    color = int(g[r, c])
                    if color in structural:
                        continue
                    if color not in by_color:
                        ordered_colors.append(color)
                        by_color[color] = []
                    by_color[color].append(
                        {
                            "tile_r": tile_r,
                            "tile_c": tile_c,
                            "sub_r": sub_r,
                            "sub_c": sub_c,
                            "r": r,
                            "c": c,
                        }
                    )

    assert ordered_colors, "expected at least one endpoint pair"
    return ordered_colors, by_color


def _pair_specs(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans = _parse_lattice(g)
    ordered_colors, by_color = _scan_endpoints(g, row_spans, col_spans)

    pairs: list[dict[str, int]] = []
    used_columns: set[int] = set()
    for color in ordered_colors:
        endpoints = by_color[color]
        assert len(endpoints) == 2, "each endpoint color must appear exactly twice"
        top, bottom = sorted(endpoints, key=lambda item: (item["tile_r"], item["tile_c"]))
        assert top["tile_r"] < bottom["tile_r"], "endpoint pair must be vertical"
        assert top["tile_c"] == bottom["tile_c"], "endpoint pair must share a tile column"
        assert top["sub_r"] == bottom["sub_r"] and top["sub_c"] == bottom["sub_c"]
        assert bottom["tile_r"] - top["tile_r"] >= 2, "endpoint pair must leave at least one middle tile"
        assert top["tile_c"] not in used_columns, "different colors must use different tile columns"
        used_columns.add(top["tile_c"])
        pairs.append(
            {
                "color": color,
                "tile_c": top["tile_c"],
                "top_r": top["tile_r"],
                "bottom_r": bottom["tile_r"],
                "sub_r": top["sub_r"],
                "sub_c": top["sub_c"],
            }
        )

    return g, row_spans, col_spans, pairs


def _apply_connections(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    pairs: list[dict[str, int]],
    *,
    anchor: str,
    shear: bool,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for pair in pairs:
        color = int(pair["color"])
        tile_c = int(pair["tile_c"])
        top_r = int(pair["top_r"])
        bottom_r = int(pair["bottom_r"])
        sub_r = int(pair["sub_r"])
        sub_c = int(pair["sub_c"])

        for tile_r in range(top_r + 1, bottom_r):
            if shear:
                if anchor == "bottom":
                    delta = bottom_r - tile_r
                elif anchor == "top":
                    delta = tile_r - top_r
                else:
                    raise ValueError(f"unknown anchor: {anchor}")
                target_sub_c = (sub_c + delta) % TILE_INTERIOR
            else:
                target_sub_c = sub_c

            r = row_spans[tile_r][0] + sub_r
            c = col_spans[tile_c][0] + target_sub_c
            current = int(out[r, c])
            assert current in {BLACK, color}, "path would collide with another meaning"
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color tile endpoints by painting a vertically sheared path through the middle tiles.

    Input structure:
    - Grey separator rows and columns partition the grid into a 3x3 or 4x4 lattice of tiles.
    - Each tile interior is a 3x3 black subgrid, and some tiles contain one colored endpoint pixel.
    - Every endpoint color appears exactly twice, both times in the same tile column and at the same 3x3 subcell.

    Rule:
    - Parse the tile lattice from the grey separators and pair endpoints by color in first-appearance scan order.
    - For each pair, keep the two endpoint pixels unchanged and inspect the tiles strictly between the upper and lower endpoint tiles.
    - In each middle tile, paint the endpoint color at row `sub_r` and column `(sub_c + delta) mod 3`, where `delta` is the current tile row distance from the lower endpoint tile.

    Key insight:
    - The connection is not a straight vertical line in pixel space; the lane is defined in tile coordinates and shears rightward as you move upward away from the lower endpoint anchor.
    """
    g, row_spans, col_spans, pairs = _pair_specs(grid)
    return _apply_connections(g, row_spans, col_spans, pairs, anchor="bottom", shear=True)


def _build_base_grid(tile_rows: int, tile_cols: int) -> np.ndarray:
    height = tile_rows * TILE_STRIDE + 1
    width = tile_cols * TILE_STRIDE + 1
    grid = np.full((height, width), BLACK, dtype=int)
    grid[::TILE_STRIDE, :] = GREY
    grid[:, ::TILE_STRIDE] = GREY
    return grid


def _place_endpoint(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    *,
    tile_r: int,
    tile_c: int,
    sub_r: int,
    sub_c: int,
    color: int,
) -> None:
    r = row_spans[tile_r][0] + sub_r
    c = col_spans[tile_c][0] + sub_c
    assert int(grid[r, c]) == BLACK
    grid[r, c] = color


def _make_pair_specs(tile_cols: int) -> list[dict[str, int]]:
    pair_count = random.randint(2, min(4, tile_cols))
    columns = random.sample(range(tile_cols), pair_count)
    colors = random.sample(ENDPOINT_COLORS, pair_count)
    special_index = random.randrange(pair_count)

    specs: list[dict[str, int]] = []
    for idx, (tile_c, color) in enumerate(zip(columns, colors)):
        sub_r = random.randrange(TILE_INTERIOR)
        if idx == special_index:
            top_r, bottom_r = 0, 3
            sub_c = random.choice((1, 2))
        else:
            if random.random() < 0.65:
                top_r, bottom_r = random.choice(((0, 2), (1, 3)))
            else:
                top_r, bottom_r = 0, 3
            sub_c = random.randrange(TILE_INTERIOR)
        specs.append(
            {
                "color": color,
                "tile_c": tile_c,
                "top_r": top_r,
                "bottom_r": bottom_r,
                "sub_r": sub_r,
                "sub_c": sub_c,
            }
        )

    random.shuffle(specs)
    return specs


def _has_wrap(pair: dict[str, int]) -> bool:
    sub_c = int(pair["sub_c"])
    for tile_r in range(int(pair["top_r"]) + 1, int(pair["bottom_r"])):
        delta = int(pair["bottom_r"]) - tile_r
        if sub_c + delta >= TILE_INTERIOR:
            return True
    return False


def _validate_instance(grid: np.ndarray) -> None:
    g, row_spans, col_spans, pairs = _pair_specs(grid)

    counts = Counter(int(v) for v in g.flatten() if int(v) not in {BLACK, GREY})
    assert counts, "expected colored endpoints"
    assert all(count == 2 for count in counts.values()), "each color must occur exactly twice"
    assert any(pair["bottom_r"] - pair["top_r"] == 3 for pair in pairs), "need a gap-3 pair"
    assert any(_has_wrap(pair) for pair in pairs), "need at least one modulo wrap"

    expected = _apply_connections(g, row_spans, col_spans, pairs, anchor="bottom", shear=True)
    alt_anchor = _apply_connections(g, row_spans, col_spans, pairs, anchor="top", shear=True)
    straight = _apply_connections(g, row_spans, col_spans, pairs, anchor="bottom", shear=False)

    assert np.any(expected != g), "output must differ from input"
    assert np.any(expected != alt_anchor), "lower-endpoint anchor must be semantically necessary"
    assert np.any(expected != straight), "shear must be semantically necessary"
    assert np.array_equal(main(g), expected)

    # Basic structural sanity for the generated layout.
    assert len(row_spans) == GEN_TILE_ROWS
    assert len(col_spans) == GEN_TILE_COLS


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        tile_rows = GEN_TILE_ROWS
        tile_cols = GEN_TILE_COLS
        grid = _build_base_grid(tile_rows, tile_cols)
        row_spans, col_spans = _parse_lattice(grid)
        pair_specs = _make_pair_specs(tile_cols)

        for pair in pair_specs:
            _place_endpoint(
                grid,
                row_spans,
                col_spans,
                tile_r=int(pair["top_r"]),
                tile_c=int(pair["tile_c"]),
                sub_r=int(pair["sub_r"]),
                sub_c=int(pair["sub_c"]),
                color=int(pair["color"]),
            )
            _place_endpoint(
                grid,
                row_spans,
                col_spans,
                tile_r=int(pair["bottom_r"]),
                tile_c=int(pair["tile_c"]),
                sub_r=int(pair["sub_r"]),
                sub_c=int(pair["sub_c"]),
                color=int(pair["color"]),
            )

        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid sheared tile-connection input")
