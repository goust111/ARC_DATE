from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
BLUE = 1
GREY = 5
TEAL = 8
PALETTE_POOL = [2, 3, 4, 6, 7, 9]

GEN_HEIGHT_RANGE = (20, 28)
GEN_WIDTH_RANGE = (22, 28)
GEN_DIVIDER_RANGE = (8, 11)
GEN_PALETTE_LEN_RANGE = (3, 5)
GEN_BAND_COUNT_CHOICES = [2, 3]
GEN_SLOPES = (-1, 1)
GEN_CAP_WIDTHS = (3, 4)
GEN_MIN_BAND_SIZE = 6
GEN_MAX_BAND_SIZE = 8
GEN_OVERLAP_PROB = 0.35
GEN_BAND_MARGIN = 1
GEN_PLACEMENT_TRIES = 120
GEN_ATTEMPTS = 400


def _extract_palette(g: np.ndarray) -> list[int]:
    rows = [
        r
        for r in range(g.shape[0])
        if np.any(g[r] == BLUE) and np.any(g[r] == TEAL)
    ]
    assert rows, "expected blue-teal bracket rows"

    palette: list[int] = []
    for r in rows:
        blue_cols = np.flatnonzero(g[r] == BLUE)
        teal_cols = np.flatnonzero(g[r] == TEAL)
        assert len(blue_cols) == 1 and len(teal_cols) == 1, "each bracket row needs one blue and one teal marker"
        c0 = int(blue_cols[0])
        c1 = int(teal_cols[0])
        assert c0 < c1, "blue marker must lie left of teal marker"
        segment = g[r, c0 + 1 : c1]
        values = [int(v) for v in segment if int(v) != BG]
        assert values, "each bracket row must hide one palette color"
        palette.append(values[0])

    return palette


def _divider_col(g: np.ndarray) -> int:
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert len(cols) == 1, "expected one full-height grey divider column"
    return int(cols[0])


def _band_offsets(size: int, cap_width: int) -> list[tuple[int, int, int]]:
    grow = cap_width - 1
    rows: list[tuple[int, int, int]] = []
    for row_offset in range(size):
        left = max(0, row_offset - grow)
        right = min(cap_width - 1 + row_offset, size - 1)
        rows.append((row_offset, left, right))
    return rows


def _band_cells(
    r0: int,
    c0: int,
    size: int,
    cap_width: int,
    slope: int,
) -> tuple[np.ndarray, np.ndarray]:
    coords: list[tuple[int, int]] = []
    layers: list[int] = []
    last_layer = size - cap_width
    for row_offset, left, right in _band_offsets(size, cap_width):
        for col_offset in range(left, right + 1):
            r = r0 + row_offset
            c = c0 + col_offset if slope > 0 else c0 + size - 1 - col_offset
            coords.append((r, c))
            layers.append(min(row_offset, col_offset, last_layer))
    return np.array(coords, dtype=int), np.array(layers, dtype=int)


def _augment_overlap_rows(
    g: np.ndarray,
    divider_col: int,
    components: list[dict],
) -> list[np.ndarray]:
    augmented: list[np.ndarray] = []
    for comp in components:
        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        rows: dict[int, list[int]] = {}
        for r, c in coords:
            rows.setdefault(r, []).append(c)

        ordered_rows = list(range(min(rows), max(rows) + 1))
        spans = [(min(rows[r]), max(rows[r])) for r in ordered_rows]
        lengths = [hi - lo + 1 for lo, hi in spans]
        if divider_col + 1 >= g.shape[1] or not any(lo == divider_col + 1 for lo, _hi in spans):
            augmented.append(np.array(sorted(coords), dtype=int))
            continue

        top_center = (spans[0][0] + spans[0][1]) / 2
        bottom_center = (spans[-1][0] + spans[-1][1]) / 2
        slopes_down_right = bottom_center > top_center

        attach_rows: set[int] = set()
        size = len(ordered_rows)
        for cap_width in GEN_CAP_WIDTHS:
            expected_lengths = [right - left + 1 for _row, left, right in _band_offsets(size, cap_width)]
            if slopes_down_right:
                max_prefix = 0
                while max_prefix < size and spans[max_prefix][0] == divider_col + 1:
                    max_prefix += 1
                for prefix in range(max_prefix, -1, -1):
                    candidate = lengths.copy()
                    for i in range(prefix):
                        candidate[i] += 1
                    if candidate == expected_lengths:
                        attach_rows = {ordered_rows[i] for i in range(prefix)}
                        break
            else:
                max_suffix = 0
                while max_suffix < size and spans[size - 1 - max_suffix][0] == divider_col + 1:
                    max_suffix += 1
                for suffix in range(max_suffix, -1, -1):
                    candidate = lengths.copy()
                    for i in range(size - suffix, size):
                        candidate[i] += 1
                    if candidate == expected_lengths:
                        attach_rows = {ordered_rows[i] for i in range(size - suffix, size)}
                        break
            if attach_rows:
                break

        for r in attach_rows:
            coords.add((r, divider_col))
        augmented.append(np.array(sorted(coords), dtype=int))
    return augmented


def _band_components(g: np.ndarray, divider_col: int, bg: int) -> list[np.ndarray]:
    grey_mask = g == GREY
    band_mask = grey_mask.copy()
    band_mask[:, divider_col] = False
    band_grid = np.where(band_mask, GREY, bg)
    components = find_components(
        band_grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )
    assert components, "expected at least one grey band"
    return _augment_overlap_rows(g, divider_col, components)


def _paint_band(out: np.ndarray, coords: np.ndarray, palette: list[int]) -> None:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))

    rows: dict[int, list[int]] = {}
    for r, c in coords:
        rows.setdefault(int(r), []).append(int(c))

    top_cols = rows[r0]
    bottom_cols = rows[r1]
    top_center = (min(top_cols) + max(top_cols)) / 2
    bottom_center = (min(bottom_cols) + max(bottom_cols)) / 2
    assert top_center != bottom_center, "band must be diagonal"
    slopes_down_right = bottom_center > top_center

    normalized: list[tuple[int, int, int, int]] = []
    top_offsets: list[int] = []
    extent = 0
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        row_offset = rr - r0
        col_offset = cc - c0 if slopes_down_right else c1 - cc
        normalized.append((rr, cc, row_offset, col_offset))
        extent = max(extent, row_offset, col_offset)
        if rr == r0:
            top_offsets.append(col_offset)

    top_offsets.sort()
    cap_width = len(top_offsets)
    assert cap_width > 0, "band needs a non-empty top cap"
    assert top_offsets == list(range(cap_width)), "top cap must be contiguous"

    last_layer = extent - cap_width + 1
    assert last_layer >= 0, "band geometry must admit at least one layer"

    for r, c, row_offset, col_offset in normalized:
        layer = min(row_offset, col_offset, last_layer)
        out[r, c] = palette[layer % len(palette)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract a hidden palette from blue-teal bracket rows and use it to recolor grey diagonal bands by layer depth.

    Input structure:
    - The background is black, with 3-5 sparse rows on the left containing one blue marker, one teal marker, and one hidden palette pixel between them.
    - A full-height grey divider column splits the palette-extraction area from the right-side target area.
    - The right side contains 2-3 grey 45-degree ribbon bands with flat top caps; some band cells may occupy the divider column.

    Rule:
    - Read every row containing both blue and teal, ordered from top to bottom.
    - In each such row, take the first non-background color strictly between the markers to build an ordered cyclic palette.
    - Remove all blue and teal markers and clear the grey divider column.
    - Recolor every grey band cell by its normalized diagonal layer, using the palette entry at layer modulo palette length; if a band occupied divider cells, those recolored band cells override the cleared divider.

    Key insight:
    - The palette is not given explicitly as a marker strip: it is hidden inside bracket-defined row segments, and the grey ribbons only become readable after that extraction step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    palette = _extract_palette(g)
    divider_col = _divider_col(g)

    out = g.copy()
    out[g == BLUE] = bg
    out[g == TEAL] = bg
    out[:, divider_col] = bg

    for coords in _band_components(g, divider_col, bg):
        _paint_band(out, coords, palette)

    return out


def _expand_occupied(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    expanded[1:, :] |= mask[:-1, :]
    expanded[:-1, :] |= mask[1:, :]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    expanded[1:, 1:] |= mask[:-1, :-1]
    expanded[1:, :-1] |= mask[:-1, 1:]
    expanded[:-1, 1:] |= mask[1:, :-1]
    expanded[:-1, :-1] |= mask[1:, 1:]
    return expanded


def _sample_palette_rows(height: int, count: int) -> list[int] | None:
    candidates = list(range(1, height - 1))
    for _ in range(GEN_PLACEMENT_TRIES):
        rows = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _paint_palette_rows(
    grid: np.ndarray,
    palette: list[int],
    rows: list[int],
    divider_col: int,
) -> None:
    for r, color in zip(rows, palette):
        blue_lo = 1
        blue_hi = divider_col - 6
        blue_col = random.randint(blue_lo, blue_hi)
        teal_col = random.randint(blue_col + 3, divider_col - 2)
        color_col = random.randint(blue_col + 1, teal_col - 1)
        grid[r, blue_col] = BLUE
        grid[r, teal_col] = TEAL
        grid[r, color_col] = color


def _band_size_choices(palette_len: int, cap_width: int) -> list[int]:
    hi = min(GEN_MAX_BAND_SIZE, palette_len + cap_width - 1)
    return list(range(GEN_MIN_BAND_SIZE, hi + 1))


def _sample_band_specs(palette_len: int, band_count: int) -> list[dict[str, int | bool]]:
    overlap_index = random.randrange(band_count) if random.random() < GEN_OVERLAP_PROB else -1
    specs: list[dict[str, int | bool]] = []
    for i in range(band_count):
        cap_choices = [cap for cap in GEN_CAP_WIDTHS if _band_size_choices(palette_len, cap)]
        cap_width = random.choice(cap_choices)
        size = random.choice(_band_size_choices(palette_len, cap_width))
        specs.append(
            {
                "size": size,
                "cap_width": cap_width,
                "slope": random.choice(GEN_SLOPES),
                "overlap": i == overlap_index,
            }
        )
    return specs


def _place_bands(
    shape: tuple[int, int],
    divider_col: int,
    specs: list[dict[str, int | bool]],
) -> list[tuple[np.ndarray, np.ndarray]] | None:
    height, width = shape
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[np.ndarray, np.ndarray]] = []

    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["size"]), reverse=True)
    for index in order:
        spec = specs[index]
        size = int(spec["size"])
        cap_width = int(spec["cap_width"])
        slope = int(spec["slope"])
        overlap = bool(spec["overlap"])

        r_lo = GEN_BAND_MARGIN
        r_hi = height - size - GEN_BAND_MARGIN
        if r_lo > r_hi:
            return None

        if overlap:
            candidate_cols = [divider_col]
        else:
            c_lo = divider_col + 2
            c_hi = width - size - GEN_BAND_MARGIN
            if c_lo > c_hi:
                return None
            candidate_cols = list(range(c_lo, c_hi + 1))
            random.shuffle(candidate_cols)

        success = False
        for _ in range(GEN_PLACEMENT_TRIES):
            r0 = random.randint(r_lo, r_hi)
            c0 = candidate_cols[0] if overlap else candidate_cols[_ % len(candidate_cols)]
            cells, layers = _band_cells(r0, c0, size, cap_width, slope)
            if np.any(cells[:, 1] < divider_col):
                continue
            if not overlap and np.any(cells[:, 1] <= divider_col + 1):
                continue
            if np.any(cells[:, 0] >= height) or np.any(cells[:, 1] >= width):
                continue
            blocked = _expand_occupied(occupied)
            if np.any(blocked[cells[:, 0], cells[:, 1]]):
                continue
            occupied[cells[:, 0], cells[:, 1]] = True
            placements.append((cells, layers))
            success = True
            break
        if not success:
            return None

    return placements


def _expected_output(
    inp: np.ndarray,
    divider_col: int,
    palette: list[int],
    placements: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    out = inp.copy()
    out[out == BLUE] = BG
    out[out == TEAL] = BG
    out[:, divider_col] = BG
    for cells, layers in placements:
        for (r, c), layer in zip(cells, layers):
            out[int(r), int(c)] = palette[int(layer) % len(palette)]
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    divider_hi = min(GEN_DIVIDER_RANGE[1], width - 11)
    divider_lo = GEN_DIVIDER_RANGE[0]
    if divider_lo > divider_hi:
        return None
    divider_col = random.randint(divider_lo, divider_hi)

    palette_len = random.randint(*GEN_PALETTE_LEN_RANGE)
    palette = random.sample(PALETTE_POOL, palette_len)
    rows = _sample_palette_rows(height, palette_len)
    if rows is None:
        return None

    band_count = random.choice(GEN_BAND_COUNT_CHOICES)
    specs = _sample_band_specs(palette_len, band_count)
    placements = _place_bands((height, width), divider_col, specs)
    if placements is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    _paint_palette_rows(inp, palette, rows, divider_col)

    for cells, _layers in placements:
        inp[cells[:, 0], cells[:, 1]] = GREY
    inp[:, divider_col] = GREY

    out = _expected_output(inp, divider_col, palette, placements)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid puzzle instance")
