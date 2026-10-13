from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg


def _palette_from_markers(g: np.ndarray, bg: int) -> list[int]:
    marker_coords = np.argwhere((g != int(bg)) & (g != Color.GREY))
    assert marker_coords.size > 0, "expected a row of colored marker pixels"

    marker_rows = np.unique(marker_coords[:, 0])
    assert marker_rows.size == 1, "marker pixels must all lie in one row"

    ordered = sorted((int(r), int(c)) for r, c in marker_coords)
    cols = [c for _r, c in ordered]
    assert cols == list(range(cols[0], cols[-1] + 1)), "marker pixels must be contiguous"

    palette = [int(g[r, c]) for r, c in ordered]
    assert len(palette) >= 2, "expected at least two marker colors"
    assert len(set(palette)) == len(palette), "marker colors must be unique"
    for color in palette:
        assert np.count_nonzero(g == color) == 1, "marker colors must appear only as markers"
    return palette


def _paint_diagonal_band(out: np.ndarray, coords: np.ndarray, palette: list[int]) -> None:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))

    rows: dict[int, list[int]] = {}
    for r, c in coords:
        rows.setdefault(int(r), []).append(int(c))

    top_cols = rows[r0]
    bottom_cols = rows[r1]
    top_center = (min(top_cols) + max(top_cols)) / 2
    bottom_center = (min(bottom_cols) + max(bottom_cols)) / 2
    assert top_center != bottom_center, "expected a slanted diagonal band"
    slopes_down_right = bottom_center > top_center

    normalized: list[tuple[int, int, int, int]] = []
    top_offsets: list[int] = []
    extent = 0
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        row_offset = rr - r0
        col_offset = cc - c0 if slopes_down_right else c1 - cc
        assert row_offset >= 0 and col_offset >= 0
        normalized.append((rr, cc, row_offset, col_offset))
        extent = max(extent, row_offset, col_offset)
        if rr == r0:
            top_offsets.append(col_offset)

    top_offsets.sort()
    cap_width = len(top_offsets)
    assert cap_width > 0, "expected a nonempty top cap"
    assert top_offsets == list(range(cap_width)), "top cap must start the normalized band"

    last_layer = extent - cap_width + 1
    assert last_layer >= 0, "band must be at least as long as its cap"

    for r, c, row_offset, col_offset in normalized:
        layer = min(row_offset, col_offset, last_layer)
        out[r, c] = palette[layer % len(palette)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey diagonal band by extending the horizontal marker palette through its diagonal layers.

    Input structure
    - The background is dominant, with one contiguous horizontal row of colored
      marker pixels and one or more grey diagonal bands.
    - Each grey band has a slanted rectangular-ribbon shape with a flat top cap
      at the end nearest the marker row.

    Rule
    - Keep the background and marker pixels unchanged.
    - For each grey band, normalize its slant so the top cap begins at local
      column zero, then use the layer min(row_offset, col_offset), clipped to
      the final layer at the far cap.
    - Recolor each grey pixel with the marker palette at layer modulo palette
      length.

    Key insight
    - The visible grey bands are diagonal ribbons; their color pattern is not
      row-wise but layered from the seed end, with the far cap clipped so the
      last layer continues to the end.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert np.any(g == Color.GREY), "expected grey bands"
    palette = _palette_from_markers(g, bg)

    out = g.copy()
    bands = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
    )
    assert bands, "expected at least one grey band"

    for band in bands:
        _paint_diagonal_band(out, band["coords"], palette)

    return out

# --------------- generator (o5x) ---------------

import random

# Palette and marker controls.
GEN_PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_PALETTE_SIZES = [3, 4]
GEN_MARKER_ROW = 1
# Band geometry controls.
GEN_CAP_WIDTH_DELTAS = [0, 1]
GEN_BAND_COUNTS = [1, 2, 3]
GEN_BAND_CYCLES = [1, 2, 3]
GEN_BAND_SLOPES = [-1, 1]
GEN_MIN_PLATEAU_ROWS = 2
# Canvas and placement controls.
GEN_GRID_HEIGHTS = range(14, 28)
GEN_GRID_WIDTHS = range(18, 24)
GEN_BORDER_MARGIN = 1
GEN_BACKGROUND_AREA_FACTOR = 2
# Retry limits.
GEN_MAX_ATTEMPTS = 300


def _gen_band_offsets(size: int, cap_width: int, grow: int) -> list[tuple[int, int, int]]:
    rows = []
    for row_offset in range(size):
        left = max(0, row_offset - grow)
        right = min(cap_width - 1 + row_offset, size - 1)
        rows.append((row_offset, left, right))
    return rows


def _gen_band_area(size: int, cap_width: int, grow: int) -> int:
    return sum(right - left + 1 for _row, left, right in _gen_band_offsets(size, cap_width, grow))


def _gen_band_cells(r0: int, c0: int, size: int, cap_width: int, grow: int, slope: int) -> tuple[np.ndarray, np.ndarray]:
    cells = []
    layers = []
    last_layer = size - cap_width
    for row_offset, left, right in _gen_band_offsets(size, cap_width, grow):
        for col_offset in range(left, right + 1):
            r = r0 + row_offset
            c = c0 + col_offset if slope > 0 else c0 + size - 1 - col_offset
            cells.append((r, c))
            layers.append(min(row_offset, col_offset, last_layer))
    return np.array(cells, dtype=int), np.array(layers, dtype=int)


def _gen_expanded_occupied(occupied: np.ndarray) -> np.ndarray:
    """Return cells blocked by occupied pixels plus their 8-neighborhood."""
    blocked = occupied.copy()
    blocked[1:, :] |= occupied[:-1, :]
    blocked[:-1, :] |= occupied[1:, :]
    blocked[:, 1:] |= occupied[:, :-1]
    blocked[:, :-1] |= occupied[:, 1:]
    blocked[1:, 1:] |= occupied[:-1, :-1]
    blocked[:-1, :-1] |= occupied[1:, 1:]
    blocked[1:, :-1] |= occupied[:-1, 1:]
    blocked[:-1, 1:] |= occupied[1:, :-1]
    return blocked


def _gen_place_bands(
    shape: tuple[int, int],
    specs: list[dict[str, int]],
    occupied: np.ndarray,
) -> list[tuple[dict[str, int], np.ndarray, np.ndarray]] | None:
    """Backtrack over shuffled anchors until all band specs fit with padding."""
    height, width = shape
    order = sorted(range(len(specs)), key=lambda i: specs[i]["size"], reverse=True)

    def search(
        index: int,
        current: np.ndarray,
        placements: list[tuple[dict[str, int], np.ndarray, np.ndarray]],
    ) -> list[tuple[dict[str, int], np.ndarray, np.ndarray]] | None:
        if index == len(order):
            return placements

        spec = specs[order[index]]
        size = spec["size"]
        blocked = _gen_expanded_occupied(current)
        anchors = [
            (r, c)
            for r in range(GEN_BORDER_MARGIN, height - size)
            for c in range(GEN_BORDER_MARGIN, width - size)
        ]
        random.shuffle(anchors)

        for r0, c0 in anchors:
            cells, layers = _gen_band_cells(
                r0,
                c0,
                size,
                spec["cap_width"],
                spec["grow"],
                spec["slope"],
            )
            if np.any(blocked[cells[:, 0], cells[:, 1]]):
                continue
            next_occupied = current.copy()
            next_occupied[cells[:, 0], cells[:, 1]] = True
            found = search(index + 1, next_occupied, placements + [(spec, cells, layers)])
            if found is not None:
                return found
        return None

    return search(0, occupied, [])


def _gen_band_specs(palette_size: int, cap_width: int, num_bands: int) -> tuple[list[dict[str, int]], int]:
    specs = []
    total_band_area = 0
    for _ in range(num_bands):
        grow = cap_width - 1
        valid_cycles = [
            cycles
            for cycles in GEN_BAND_CYCLES
            if cycles * palette_size + cap_width - 1 - 2 * grow >= GEN_MIN_PLATEAU_ROWS
        ]
        cycles = random.choice(valid_cycles)
        size = cycles * palette_size + cap_width - 1
        spec = {
            "size": size,
            "cap_width": cap_width,
            "grow": grow,
            "cycles": cycles,
            "slope": random.choice(GEN_BAND_SLOPES),
        }
        specs.append(spec)
        total_band_area += _gen_band_area(size, cap_width, grow)
    return specs, total_band_area


def _gen_candidate_shapes(
    specs: list[dict[str, int]], total_band_area: int, palette_size: int
) -> list[tuple[int, int]]:
    max_size = max(spec["size"] for spec in specs)
    min_canvas_size = max_size + 2 * GEN_BORDER_MARGIN
    return [
        (height, width)
        for height in GEN_GRID_HEIGHTS
        for width in GEN_GRID_WIDTHS
        if height >= min_canvas_size
        and width >= min_canvas_size
        and height * width > GEN_BACKGROUND_AREA_FACTOR * total_band_area + palette_size
    ]


def _gen_init_canvas(shape: tuple[int, int], palette: list[int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    out = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    marker_starts = list(range(GEN_BORDER_MARGIN, width - len(palette)))
    random.shuffle(marker_starts)
    marker_start = marker_starts[0]
    cols = np.arange(marker_start, marker_start + len(palette))
    out[GEN_MARKER_ROW, cols] = palette
    occupied[GEN_MARKER_ROW, cols] = True
    return out, occupied


def _gen_render_bands(
    out: np.ndarray, placements: list[tuple[dict[str, int], np.ndarray, np.ndarray]], palette: list[int]
) -> np.ndarray:
    """Paint output bands and return the matching input grid."""
    palette_array = np.array(palette, dtype=int)
    band_cells = []
    for _spec, cells, layers in placements:
        band_cells.append(cells)
        out[cells[:, 0], cells[:, 1]] = palette_array[layers % len(palette)]
    inp = out.copy()
    for cells in band_cells:
        inp[cells[:, 0], cells[:, 1]] = Color.GREY
    return inp


def _gen_oracle_accepts(inp: np.ndarray, out: np.ndarray) -> bool:
    if np.count_nonzero(inp == Color.BLACK) <= np.count_nonzero(inp == Color.GREY):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    palette_size = random.choice(GEN_PALETTE_SIZES)
    cap_width = palette_size + random.choice(GEN_CAP_WIDTH_DELTAS)
    num_bands = random.choice(GEN_BAND_COUNTS)
    palette = random.sample(GEN_PALETTE_COLORS, palette_size)
    specs, total_band_area = _gen_band_specs(palette_size, cap_width, num_bands)

    possible_shapes = _gen_candidate_shapes(specs, total_band_area, palette_size)
    if not possible_shapes:
        return None
    height, width = random.choice(possible_shapes)

    out, occupied = _gen_init_canvas((height, width), palette)
    placements = _gen_place_bands((height, width), specs, occupied)
    if placements is None:
        return None

    inp = _gen_render_bands(out, placements, palette)
    return inp if _gen_oracle_accepts(inp, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b457fec5 input")
