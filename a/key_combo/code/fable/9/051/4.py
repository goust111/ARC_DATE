import random
from collections import Counter

import numpy as np


BACKGROUND = 0
SOURCE = 2
SEPARATOR = 4
MARKER = 8

_GRID_RANGE = (14, 23)
_AXIS_MARGIN = 5
_SHAPE_COUNT_CHOICES = (1, 2, 2, 3, 3)
_SHAPE_HEIGHT_RANGE = (2, 5)
_SHAPE_WIDTH_RANGE = (2, 5)
_MAX_GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    colors = sorted(set(map(int, np.asarray(grid).ravel())))
    excluded = set(map(int, exclude))
    return [color for color in colors if color not in excluded]


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    full_rows = [r for r in range(h) if np.all(grid[r, :] == grid[r, 0]) and int(grid[r, 0]) != bg]
    full_cols = [c for c in range(w) if np.all(grid[:, c] == grid[0, c]) and int(grid[0, c]) != bg]

    if len(full_rows) == 1 and not full_cols:
        axis = int(full_rows[0])
        return "horizontal", axis, int(grid[axis, 0])
    if len(full_cols) == 1 and not full_rows:
        axis = int(full_cols[0])
        return "vertical", axis, int(grid[0, axis])
    raise AssertionError("expected exactly one full-width or full-height separator line")


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    orientation, axis, separator_color = _find_separator(g, bg)

    other_colors = colors_present(g, exclude={bg, separator_color})
    assert len(other_colors) == 2, "expected one source color and one marker color"
    counts = {color: int(np.count_nonzero(g == color)) for color in other_colors}

    marker_candidates = [color for color, count in counts.items() if count == 1]
    assert len(marker_candidates) == 1, "expected a unique singleton marker color"
    marker_color = marker_candidates[0]
    source_candidates = [color for color in other_colors if color != marker_color]
    assert len(source_candidates) == 1
    source_color = source_candidates[0]
    assert counts[source_color] > 1, "source shape must occupy multiple pixels"

    marker_coords = np.argwhere(g == marker_color)
    assert marker_coords.shape == (1, 2)
    marker = tuple(map(int, marker_coords[0]))

    source_coords = np.argwhere(g == source_color)
    assert source_coords.size > 0

    if orientation == "vertical":
        source_side_values = {int(c < axis) - int(c > axis) for _r, c in source_coords}
        assert source_side_values in ({1}, {-1}), "source pixels must all lie on one side of the separator"
        source_side = -1 if next(iter(source_side_values)) == 1 else 1
        assert marker[1] != axis and (marker[1] - axis) * source_side < 0, "marker must lie on the opposite side"
    else:
        source_side_values = {int(r < axis) - int(r > axis) for r, _c in source_coords}
        assert source_side_values in ({1}, {-1}), "source pixels must all lie on one side of the separator"
        source_side = -1 if next(iter(source_side_values)) == 1 else 1
        assert marker[0] != axis and (marker[0] - axis) * source_side < 0, "marker must lie on the opposite side"

    return {
        "grid": g,
        "bg": bg,
        "orientation": orientation,
        "axis": axis,
        "separator_color": separator_color,
        "marker_color": marker_color,
        "source_color": source_color,
        "marker": marker,
        "source_side": source_side,
        "source_coords": source_coords,
    }


def _mirror_coords(source_coords: np.ndarray, orientation: str, axis: int) -> np.ndarray:
    mirrored = np.asarray(source_coords, dtype=int).copy()
    if orientation == "vertical":
        mirrored[:, 1] = 2 * int(axis) - mirrored[:, 1]
    else:
        mirrored[:, 0] = 2 * int(axis) - mirrored[:, 0]
    return mirrored


def _parity_counts(coords: np.ndarray) -> dict[tuple[int, int], int]:
    counts = {(rp, cp): 0 for rp in (0, 1) for cp in (0, 1)}
    for r, c in coords:
        counts[(int(r) % 2, int(c) % 2)] += 1
    return counts


def _render_output(info: dict) -> np.ndarray:
    out = info["grid"].copy()
    mirrored = _mirror_coords(info["source_coords"], info["orientation"], info["axis"])
    mr, mc = info["marker"]
    keep = (mirrored[:, 0] % 2 == mr % 2) & (mirrored[:, 1] % 2 == mc % 2)
    out[mirrored[keep, 0], mirrored[keep, 1]] = info["marker_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the solid source shapes across the separator, but reveal only the parity-matched lattice of the mirror in the marker color.

    Input structure:
    - The grid has a single one-cell-thick full-width or full-height separator line.
    - One side contains one or more solid source shapes of the same color.
    - The opposite side contains only background plus one isolated marker pixel of another color.

    Rule:
    - Reflect every source-shape pixel across the separator line to the opposite side.
    - Among those mirrored pixels, keep only the cells whose row parity and column parity both match the marker pixel.
    - Color the kept mirrored cells with the marker color, and leave the original shapes, separator, marker, and all other cells unchanged.

    Key insight:
    - The marker does not point to a location to copy; it fixes which of the four coordinate-parity lattices the mirror is allowed to appear on.
    """
    return _render_output(_analyze(grid))


def _make_shape_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if height >= 3 and width >= 3 and random.random() < 0.6:
        notch_h = random.randint(1, height - 1)
        notch_w = random.randint(1, width - 1)
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            mask[:notch_h, :notch_w] = False
        elif corner == "tr":
            mask[:notch_h, width - notch_w :] = False
        elif corner == "bl":
            mask[height - notch_h :, :notch_w] = False
        else:
            mask[height - notch_h :, width - notch_w :] = False
    return mask


def _place_shapes(
    grid: np.ndarray, orientation: str, axis: int, source_side: int, shape_count: int
) -> np.ndarray | None:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    all_coords: list[tuple[int, int]] = []

    if orientation == "vertical":
        mirror_depth = min(axis, w - 1 - axis)
        col_min = axis - mirror_depth if source_side < 0 else axis + 1
        col_max = axis - 1 if source_side < 0 else axis + mirror_depth
    else:
        mirror_depth = min(axis, h - 1 - axis)
        row_min = axis - mirror_depth if source_side < 0 else axis + 1
        row_max = axis - 1 if source_side < 0 else axis + mirror_depth

    for _ in range(shape_count):
        placed = False
        for _attempt in range(120):
            shape_h = random.randint(_SHAPE_HEIGHT_RANGE[0], min(_SHAPE_HEIGHT_RANGE[1], h - 2))
            shape_w = random.randint(_SHAPE_WIDTH_RANGE[0], min(_SHAPE_WIDTH_RANGE[1], w - 2))

            if orientation == "vertical":
                if shape_w > col_max - col_min + 1 or shape_h > h - 2:
                    continue
                r0 = random.randint(1, h - shape_h - 1)
                c0 = random.randint(col_min, col_max - shape_w + 1)
            else:
                if shape_h > row_max - row_min + 1 or shape_w > w - 2:
                    continue
                r0 = random.randint(row_min, row_max - shape_h + 1)
                c0 = random.randint(1, w - shape_w - 1)

            rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
            rr1, cc1 = min(h, r0 + shape_h + 1), min(w, c0 + shape_w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            mask = _make_shape_mask(shape_h, shape_w)
            coords = [(r0 + r, c0 + c) for r, c in np.argwhere(mask)]
            if len(coords) < 3:
                continue

            for r, c in coords:
                occupied[r, c] = True
                all_coords.append((r, c))
            placed = True
            break
        if not placed:
            return None

    if not all_coords:
        return None

    for r, c in all_coords:
        grid[r, c] = SOURCE
    return np.asarray(all_coords, dtype=int)


def _marker_candidates(
    shape: tuple[int, int],
    orientation: str,
    axis: int,
    source_side: int,
    mirrored: np.ndarray,
    parity: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    mirror_set = {tuple(map(int, rc)) for rc in mirrored}

    if orientation == "vertical":
        side_cells = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if ((c > axis) if source_side < 0 else (c < axis))
        ]
    else:
        side_cells = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if ((r > axis) if source_side < 0 else (r < axis))
        ]

    filtered = [cell for cell in side_cells if cell not in mirror_set and (cell[0] % 2, cell[1] % 2) == parity]
    if not filtered:
        return []

    rows = mirrored[:, 0]
    cols = mirrored[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    far = [
        cell
        for cell in filtered
        if not (r0 - 1 <= cell[0] <= r1 + 1 and c0 - 1 <= cell[1] <= c1 + 1)
    ]
    return far or filtered


def _build_instance() -> np.ndarray | None:
    h = random.randint(*_GRID_RANGE)
    w = random.randint(*_GRID_RANGE)
    orientation = random.choice(("vertical", "horizontal"))
    source_side = random.choice((-1, 1))
    shape_count = random.choice(_SHAPE_COUNT_CHOICES)

    if orientation == "vertical":
        if w < 2 * _AXIS_MARGIN + 1:
            return None
        axis = random.randint(_AXIS_MARGIN, w - _AXIS_MARGIN - 1)
    else:
        if h < 2 * _AXIS_MARGIN + 1:
            return None
        axis = random.randint(_AXIS_MARGIN, h - _AXIS_MARGIN - 1)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    if orientation == "vertical":
        grid[:, axis] = SEPARATOR
    else:
        grid[axis, :] = SEPARATOR

    source_coords = _place_shapes(grid, orientation, axis, source_side, shape_count)
    if source_coords is None:
        return None

    mirrored = _mirror_coords(source_coords, orientation, axis)
    parity_counts = _parity_counts(mirrored)
    nonzero_parities = [parity for parity, count in parity_counts.items() if count > 0]
    if len(nonzero_parities) < 4:
        return None

    chosen_parity = random.choice(nonzero_parities)
    marker_options = _marker_candidates(grid.shape, orientation, axis, source_side, mirrored, chosen_parity)
    if not marker_options:
        return None
    mr, mc = random.choice(marker_options)
    grid[mr, mc] = MARKER

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    info = _analyze(grid)
    mirrored = _mirror_coords(info["source_coords"], info["orientation"], info["axis"])
    h, w = grid.shape
    assert np.all((0 <= mirrored[:, 0]) & (mirrored[:, 0] < h))
    assert np.all((0 <= mirrored[:, 1]) & (mirrored[:, 1] < w))

    mirror_set = {tuple(map(int, rc)) for rc in mirrored}
    assert info["marker"] not in mirror_set, "marker must not sit on a mirrored pixel"

    parity_counts = _parity_counts(mirrored)
    assert all(count > 0 for count in parity_counts.values()), "mirrored shape should cover all four parity classes"

    marker_parity = (info["marker"][0] % 2, info["marker"][1] % 2)
    chosen_count = parity_counts[marker_parity]
    assert 0 < chosen_count < len(mirrored), "parity sieve must keep some but not all mirrored pixels"

    outputs = []
    for parity in sorted(parity_counts):
        keep = {(int(r), int(c)) for r, c in mirrored if (int(r) % 2, int(c) % 2) == parity}
        outputs.append(keep)
    assert len({frozenset(coords) for coords in outputs}) == 4, "all parity phases should yield distinct mirror residues"

    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out, _render_output(info))
    assert not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _build_instance()
        if grid is None:
            continue
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid parity-sieve mirror instance")
