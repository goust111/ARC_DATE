from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg, in_bounds, neighbors4
import random


Coord = tuple[int, int]


def _infer_roles(g: np.ndarray) -> tuple[int, int, Coord, list[Coord]]:
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 3, "expected source color, anchor color, and marker color"

    singleton_controls: list[tuple[int, Coord]] = []
    color_coords: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        color_coords[int(color)] = coords
        if len(coords) == 1:
            singleton_controls.append((int(color), (int(coords[0, 0]), int(coords[0, 1]))))
    assert singleton_controls, "expected at least one singleton control pixel"

    candidates: list[tuple[int, int, Coord, list[Coord]]] = []
    for anchor_color, anchor in singleton_controls:
        ar, ac = anchor
        for source_color in colors:
            source_color = int(source_color)
            if source_color == anchor_color:
                continue
            source_coords = color_coords[source_color]
            if len(source_coords) <= 1:
                continue

            adjacent_source = sum(
                1
                for rr, cc in neighbors4(ar, ac)
                if in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == source_color
            )
            if adjacent_source == 0:
                continue

            destinations: list[Coord] = []
            for color in colors:
                color = int(color)
                if color == source_color:
                    continue
                for r_raw, c_raw in color_coords[color]:
                    pos = (int(r_raw), int(c_raw))
                    if color == anchor_color and pos == anchor:
                        continue
                    destinations.append(pos)
            if not destinations:
                continue

            offsets = source_coords - np.array(anchor)
            all_copies_fit = all(
                in_bounds(g.shape, int(r + dr), int(c + dc))
                for r, c in destinations
                for dr, dc in offsets
            )
            if all_copies_fit:
                candidates.append((adjacent_source, source_color, anchor, destinations))

    assert candidates, "expected a source mask anchored next to a singleton control"
    strongest = max(candidate[0] for candidate in candidates)
    best = [candidate for candidate in candidates if candidate[0] == strongest]
    assert len(best) == 1, "expected a unique anchor/source role assignment"
    _, source_color, anchor, destinations = best[0]
    return bg, source_color, anchor, destinations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the foreground source mask from its singleton anchor control pixel to every destination control pixel.

    Input structure
    - A background grid contains a multi-pixel source mask color and control pixels in other colors.
    - One singleton control pixel is the anchor; its 4-neighborhood has the strongest contact with the source mask color.
    - All other non-background, non-source control pixels are destination markers.

    Rule
    - Take the coordinates of the source-color pixels relative to the anchor control pixel.
    - Replace the whole scene with background, then stamp that relative mask in the source color at each destination marker.
    - Erase the original source pixels and all control pixels.

    Key insight
    - The controls mark origins: the singleton anchor gives the source mask's coordinate frame, and the other control pixels are target origins for copied masks.
    """
    g = np.asarray(grid, dtype=int)
    bg, source_color, anchor, destinations = _infer_roles(g)

    out = np.full_like(g, bg)
    offsets = np.argwhere(g == source_color) - np.array(anchor)
    for r, c in destinations:
        for dr, dc in offsets:
            rr = int(r + dr)
            cc = int(c + dc)
            assert in_bounds(g.shape, rr, cc), "copied source mask must stay in bounds"
            out[rr, cc] = source_color
    return out

# --------------- generator (o5x) ---------------

# Fixed task grid and colors.
_GEN_BG = 0
_GEN_SOURCE = 8
_GEN_ANCHOR = 3
_GEN_DEST = 2
_GEN_N = 13

# Source-mask geometry and fill policy.
_GEN_SOURCE_HEIGHT_RANGE = (3, 5)
_GEN_HALF_WIDTH_CHOICES = [1, 2]
_GEN_SOURCE_MIN_PIXELS = 4
_GEN_SOURCE_MAX_PIXELS = 14
_GEN_DENSITY_RANGE = (0.2, 0.7)

# Source unit descriptors: center cell or mirrored LR pair.
_GEN_CENTER_UNIT = 0
_GEN_MIRROR_UNIT = 1
_GEN_UNIT_KIND_CHOICES = [_GEN_CENTER_UNIT, _GEN_MIRROR_UNIT]

# Destination-marker symmetry and retry budget.
_GEN_DEST_COUNT_RANGE = (1, 4)
_GEN_SINGLE_DEST_COUNT = 1
_GEN_MIN_PAIR_OFFSET = 1
_GEN_ATTEMPT_LIMIT = 300

from arc_utils import paste as _arc_paste


def _gen_unit_cells(unit: tuple[int, int, int]) -> list[tuple[int, int]]:
    dr, kind, k = unit
    if kind == _GEN_CENTER_UNIT:
        return [(dr, 0)] if dr != 0 else []
    return [(dr, -k), (dr, k)]


def _gen_add_unit(offsets: set[tuple[int, int]], unit: tuple[int, int, int]) -> None:
    for cell in _gen_unit_cells(unit):
        offsets.add(cell)


def _gen_source_units(rows: list[int], half_width: int) -> list[tuple[int, int, int]]:
    """Return source units in the same order used by the density pass."""
    units: list[tuple[int, int, int]] = []
    for dr in rows:
        if dr != 0:
            units.append((dr, _GEN_CENTER_UNIT, 0))
        for k in range(1, half_width + 1):
            units.append((dr, _GEN_MIRROR_UNIT, k))
    return units


def _gen_anchor_adjacency_patterns(top: int, bottom: int) -> list[list[tuple[int, int]]]:
    """Return mandatory source cells that give the anchor at least two contacts."""
    patterns: list[list[tuple[int, int]]] = [[(0, -1), (0, 1)]]
    if top and bottom:
        patterns.append([(-1, 0), (1, 0)])
    if top:
        patterns.append([(-1, 0), (0, -1), (0, 1)])
    if bottom:
        patterns.append([(1, 0), (0, -1), (0, 1)])
    if top and bottom:
        patterns.append([(-1, 0), (1, 0), (0, -1), (0, 1)])
    return patterns


def _gen_source_offsets() -> list[tuple[int, int]]:
    height = random.randint(*_GEN_SOURCE_HEIGHT_RANGE)
    half_width = random.choice(_GEN_HALF_WIDTH_CHOICES)
    top = random.randint(0, height - 1)
    bottom = height - 1 - top
    rows = list(range(-top, bottom + 1))

    units = _gen_source_units(rows, half_width)
    offsets: set[tuple[int, int]] = set()
    offsets.update(random.choice(_gen_anchor_adjacency_patterns(top, bottom)))

    _gen_add_unit(
        offsets,
        (rows[0], random.choice(_GEN_UNIT_KIND_CHOICES), random.randint(1, half_width)),
    )
    _gen_add_unit(
        offsets,
        (rows[-1], random.choice(_GEN_UNIT_KIND_CHOICES), random.randint(1, half_width)),
    )
    _gen_add_unit(offsets, (random.choice(rows), _GEN_MIRROR_UNIT, half_width))

    optional = [u for u in units if not all(pos in offsets for pos in _gen_unit_cells(u))]
    random.shuffle(optional)
    density = random.uniform(*_GEN_DENSITY_RANGE)
    for unit in optional:
        cells = _gen_unit_cells(unit)
        add_count = sum(pos not in offsets for pos in cells)
        if len(offsets) + add_count <= _GEN_SOURCE_MAX_PIXELS and random.random() < density:
            _gen_add_unit(offsets, unit)

    for unit in optional:
        cells = _gen_unit_cells(unit)
        add_count = sum(pos not in offsets for pos in cells)
        if len(offsets) >= _GEN_SOURCE_MIN_PIXELS:
            break
        if len(offsets) + add_count <= _GEN_SOURCE_MAX_PIXELS:
            _gen_add_unit(offsets, unit)

    offsets.discard((0, 0))
    return sorted(offsets)


def _gen_offset_bounds(offsets: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in offsets]
    cols = [c for _, c in offsets]
    return min(rows), max(rows), min(cols), max(cols)


def _gen_source_patch(offsets: list[tuple[int, int]], color: int) -> tuple[np.ndarray, int, int]:
    min_r, max_r, min_c, max_c = _gen_offset_bounds(offsets)
    patch = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for dr, dc in offsets:
        patch[dr - min_r, dc - min_c] = color
    return patch, min_r, min_c


def _gen_paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    _arc_paste(dst, patch, r0, c0, transparent=_GEN_BG)


def _gen_destination_positions(offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r, max_r, min_c, max_c = _gen_offset_bounds(offsets)
    valid_rows = list(range(-min_r, _GEN_N - max_r))
    valid_cols = set(range(-min_c, _GEN_N - max_c))
    dest_count = random.randint(*_GEN_DEST_COUNT_RANGE)
    pair_count = dest_count // 2
    needs_axis = dest_count % 2 == 1

    axes: list[tuple[int, list[int]]] = []
    for axis in range(_GEN_N):
        if needs_axis and axis not in valid_cols:
            continue
        pair_offsets = [
            d
            for d in range(_GEN_MIN_PAIR_OFFSET, _GEN_N)
            if axis - d in valid_cols and axis + d in valid_cols
        ]
        if pair_count and not pair_offsets:
            continue
        axes.append((axis, pair_offsets))
    if not axes:
        return []

    axis, pair_offsets = random.choice(axes)
    destinations: set[tuple[int, int]] = set()
    if needs_axis:
        destinations.add((random.choice(valid_rows), axis))

    pair_units = [(r, d) for r in valid_rows for d in pair_offsets]
    random.shuffle(pair_units)
    for r, d in pair_units[:pair_count]:
        destinations.add((r, axis - d))
        destinations.add((r, axis + d))

    return sorted(destinations)


def _gen_stamp_output(
    offsets: list[tuple[int, int]],
    destinations: list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((_GEN_N, _GEN_N), _GEN_BG, dtype=int)
    patch, min_r, min_c = _gen_source_patch(offsets, _GEN_SOURCE)
    for r, c in destinations:
        _gen_paste(out, patch, r + min_r, c + min_c)
    return out


def _gen_touches(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _gen_anchor_position(
    offsets: list[tuple[int, int]],
    destinations: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Choose an anchor placement that avoids destination-role ambiguity."""
    min_r, max_r, min_c, max_c = _gen_offset_bounds(offsets)
    dest_set = set(destinations)
    candidates: list[tuple[int, int]] = []
    for ar in range(-min_r, _GEN_N - max_r):
        for ac in range(-min_c, _GEN_N - max_c):
            anchor = (ar, ac)
            if anchor in dest_set or any(_gen_touches(anchor, d) for d in dest_set):
                continue
            source_cells = {(ar + dr, ac + dc) for dr, dc in offsets}
            if source_cells & dest_set:
                continue
            if len(destinations) == _GEN_SINGLE_DEST_COUNT and any(
                _gen_touches(destinations[0], cell) for cell in source_cells
            ):
                continue
            candidates.append(anchor)
    if not candidates:
        return None
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    """Assemble one candidate input and reject it if the solver oracle disagrees."""
    offsets = _gen_source_offsets()
    destinations = _gen_destination_positions(offsets)
    if not destinations:
        return None

    expected = _gen_stamp_output(offsets, destinations)
    anchor = _gen_anchor_position(offsets, destinations)
    if anchor is None:
        return None

    inp = np.full((_GEN_N, _GEN_N), _GEN_BG, dtype=int)
    patch, min_r, min_c = _gen_source_patch(offsets, _GEN_SOURCE)
    _gen_paste(inp, patch, anchor[0] + min_r, anchor[1] + min_c)
    inp[anchor] = _GEN_ANCHOR
    for r, c in destinations:
        inp[r, c] = _GEN_DEST

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPT_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid f5c89df1 input")
