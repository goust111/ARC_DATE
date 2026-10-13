from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400

ROW_OPTIONS = (2, 3, 4)
COL_OPTIONS = (2, 3, 4)
MIN_BAND = 4
MAX_BAND = 8
MIN_GAP = 2
MAX_GAP = 4
MIN_MARGIN = 1
MAX_MARGIN = 4
MIN_RECT_SIDE = 3

SINGLE_CLUTTER_RANGE = (8, 18)
SMALL_BLOB_COUNT_RANGE = (3, 8)
SMALL_BLOB_AREA_RANGE = (2, 5)
LARGE_BLOB_COUNT_RANGE = (1, 2)
LARGE_BLOB_AREA_RANGE = (9, 12)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_for_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    rectangles: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != bg):
        for coords in _components_for_color(grid, color):
            if len(coords) < 9:
                continue
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 3 or width < 3:
                continue
            if len(coords) != height * width:
                continue
            rectangles.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    assert rectangles, "expected at least one filled rectangle"
    return rectangles


def _overlap_groups(intervals: list[tuple[int, int, int]]) -> tuple[dict[int, int], int]:
    ordered = sorted(intervals, key=lambda item: (item[1], item[2], item[0]))
    mapping: dict[int, int] = {}
    group_id = -1
    current_end = None
    for idx, start, end in ordered:
        if current_end is None or start > current_end:
            group_id += 1
            current_end = end
        else:
            current_end = max(current_end, end)
        mapping[idx] = group_id
    return mapping, group_id + 1


def _compress_rectangles(
    rectangles: list[dict],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]]]:
    row_map, row_count = _overlap_groups(
        [(idx, rect["bbox"][0], rect["bbox"][2]) for idx, rect in enumerate(rectangles)]
    )
    col_map, col_count = _overlap_groups(
        [(idx, rect["bbox"][1], rect["bbox"][3]) for idx, rect in enumerate(rectangles)]
    )
    assert len(rectangles) == row_count * col_count, "expected a complete rectangle lattice"

    table = np.full((row_count, col_count), -1, dtype=int)
    rect_by_cell: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for idx, rect in enumerate(rectangles):
        cell = (row_map[idx], col_map[idx])
        assert table[cell] == -1, "expected one rectangle per lattice cell"
        table[cell] = int(rect["color"])
        rect_by_cell[cell] = rect["bbox"]

    assert np.all(table >= 0), "every lattice cell must be occupied"
    return table, rect_by_cell


def _biaxial_orbits(shape: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    rows, cols = shape
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(rows):
        for c in range(cols):
            orbit = tuple(
                sorted(
                    {
                        (r, c),
                        (r, cols - 1 - c),
                        (rows - 1 - r, c),
                        (rows - 1 - r, cols - 1 - c),
                    }
                )
            )
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _find_unique_orbit_mismatch(table: np.ndarray) -> tuple[tuple[int, int], int]:
    found: list[tuple[tuple[int, int], int]] = []
    for orbit in _biaxial_orbits(table.shape):
        colors = [int(table[r, c]) for r, c in orbit]
        counts = Counter(colors)
        if len(counts) == 1:
            continue
        major_color, major_count = counts.most_common(1)[0]
        if sum(1 for count in counts.values() if count == major_count) != 1:
            raise AssertionError("orbit does not have a unique majority color")
        mismatches = [(r, c) for r, c in orbit if int(table[r, c]) != major_color]
        if len(mismatches) != 1:
            raise AssertionError("orbit mismatch must affect exactly one cell")
        found.append((mismatches[0], int(major_color)))
    assert len(found) == 1, "expected exactly one mismatching orbit cell"
    return found[0]


def _table_is_biaxially_symmetric(table: np.ndarray) -> bool:
    for orbit in _biaxial_orbits(table.shape):
        colors = {int(table[r, c]) for r, c in orbit}
        if len(colors) != 1:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Correct the lone wrong-colored lattice rectangle by voting inside the compact mirror-symmetric color table.

    Input structure:
    - The grid has a black background with a few scattered clutter pixels or blobs.
    - The meaningful objects are monochrome filled rectangles, each at least 3x3, and each exactly fills its bounding box.
    - These rectangles form a complete m x n lattice when grouped by overlapping bbox row bands and overlapping bbox column bands.

    Rule:
    - Ignore every non-rectangle distractor and compress the rectangle lattice into an m x n color table.
    - Group compact-table cells by the four-way orbit induced by horizontal and vertical reflection.
    - Exactly one lattice cell disagrees with the majority color in its orbit; recolor the corresponding full rectangle to that orbit majority, while preserving everything else.

    Key insight:
    - The symmetry lives only after abstracting each rectangle to one table cell, so the task is to repair a rectangle selected by the compressed table's mirror-orbit majority vote, not by pixel-level symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rectangles = _extract_rectangles(g)
    table, rect_by_cell = _compress_rectangles(rectangles)
    bad_cell, fixed_color = _find_unique_orbit_mismatch(table)

    r0, c0, r1, c1 = rect_by_cell[bad_cell]
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = fixed_color
    return out


def _sample_band_extents(
    rng: np.random.Generator, count: int
) -> tuple[list[tuple[int, int]], int] | None:
    for _ in range(200):
        lengths = rng.integers(MIN_BAND, MAX_BAND + 1, size=count)
        gaps = rng.integers(MIN_GAP, MAX_GAP + 1, size=max(0, count - 1))
        margin0 = int(rng.integers(MIN_MARGIN, MAX_MARGIN + 1))
        margin1 = int(rng.integers(MIN_MARGIN, MAX_MARGIN + 1))
        total = margin0 + int(lengths.sum()) + int(gaps.sum()) + margin1
        if total > 30:
            continue
        extents: list[tuple[int, int]] = []
        pos = margin0
        for i, length in enumerate(lengths):
            start = pos
            end = start + int(length) - 1
            extents.append((start, end))
            pos = end + 1
            if i < count - 1:
                pos += int(gaps[i])
        return extents, total
    return None


def _sample_intervals_in_band(
    rng: np.random.Generator, extent: tuple[int, int], count: int
) -> list[tuple[int, int]]:
    start, end = extent
    anchor = int(rng.integers(start, end + 1))
    span = end - start + 1
    intervals: list[tuple[int, int]] = []
    for _ in range(count):
        length = int(rng.integers(MIN_RECT_SIDE, span + 1))
        lo_min = max(start, anchor - length + 1)
        lo_max = min(anchor, end - length + 1)
        lo = int(rng.integers(lo_min, lo_max + 1))
        intervals.append((lo, lo + length - 1))
    return intervals


def _choose_palettes(
    rng: np.random.Generator, cell_count: int
) -> tuple[list[int], list[int]]:
    colors = list(PALETTE)
    rng.shuffle(colors)
    clutter_count = int(rng.integers(1, 3))
    rect_palette_size = int(rng.integers(3, min(6, cell_count, 9 - clutter_count) + 1))
    rect_palette = [int(c) for c in colors[:rect_palette_size]]
    clutter_palette = [int(c) for c in colors[rect_palette_size : rect_palette_size + clutter_count]]
    return rect_palette, clutter_palette


def _build_tables(
    rng: np.random.Generator, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], list[int], list[int]]:
    rect_palette, clutter_palette = _choose_palettes(rng, rows * cols)
    orbits = _biaxial_orbits((rows, cols))
    size4_orbits = [orbit for orbit in orbits if len(orbit) == 4]
    assert size4_orbits, "need at least one four-cell orbit"

    for _ in range(200):
        output_table = np.zeros((rows, cols), dtype=int)
        for orbit in orbits:
            color = int(rect_palette[int(rng.integers(len(rect_palette)))])
            for r, c in orbit:
                output_table[r, c] = color
        if len(set(int(v) for v in output_table.flat)) < 2:
            continue
        bad_orbit = size4_orbits[int(rng.integers(len(size4_orbits)))]
        bad_cell = bad_orbit[int(rng.integers(len(bad_orbit)))]
        majority = int(output_table[bad_cell])
        wrong_choices = [color for color in rect_palette if color != majority]
        wrong_color = int(wrong_choices[int(rng.integers(len(wrong_choices)))])
        input_table = output_table.copy()
        input_table[bad_cell] = wrong_color
        return input_table, output_table, bad_cell, rect_palette, clutter_palette
    raise RuntimeError("failed to build a non-degenerate symmetric table")


def _render_lattice(
    rng: np.random.Generator,
    table: np.ndarray,
    row_extents: list[tuple[int, int]],
    col_extents: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int, int, int]]]:
    rows, cols = table.shape
    h, w = shape
    grid = np.zeros((h, w), dtype=int)

    row_intervals = [_sample_intervals_in_band(rng, row_extents[r], cols) for r in range(rows)]
    col_intervals = [_sample_intervals_in_band(rng, col_extents[c], rows) for c in range(cols)]

    rect_bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            r0, r1 = row_intervals[r][c]
            c0, c1 = col_intervals[c][r]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(table[r, c])
            rect_bboxes[(r, c)] = (r0, c0, r1, c1)
    return grid, rect_bboxes


def _coords_to_patch(coords: set[tuple[int, int]]) -> np.ndarray:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    r1 = max(r for r, _ in coords)
    c1 = max(c for _, c in coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch


def _random_polyomino(rng: np.random.Generator, area: int) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < area:
        frontier: list[tuple[int, int]] = []
        seen = set()
        for r, c in coords:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt not in coords and nxt not in seen:
                    frontier.append(nxt)
                    seen.add(nxt)
        coords.add(frontier[int(rng.integers(len(frontier)))])
    return _coords_to_patch(coords)


def _small_blob_patch(rng: np.random.Generator) -> np.ndarray:
    area = int(rng.integers(SMALL_BLOB_AREA_RANGE[0], SMALL_BLOB_AREA_RANGE[1] + 1))
    return _random_polyomino(rng, area)


def _large_irregular_patch(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        area = int(rng.integers(LARGE_BLOB_AREA_RANGE[0], LARGE_BLOB_AREA_RANGE[1] + 1))
        patch = _random_polyomino(rng, area)
        ph, pw = patch.shape
        if ph >= 3 and pw >= 3 and int(patch.sum()) != ph * pw:
            return patch
    raise RuntimeError("failed to build a large irregular clutter blob")


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + ph + pad)
    c1 = min(w, left + pw + pad)
    if np.any(grid[r0:r1, c0:c1] != BG):
        return False
    return True


def _place_patch(
    rng: np.random.Generator, grid: np.ndarray, patch: np.ndarray, color: int, pad: int = 1
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False
    for _ in range(200):
        top = int(rng.integers(0, h - ph + 1))
        left = int(rng.integers(0, w - pw + 1))
        if not _can_place_patch(grid, patch, top, left, pad=pad):
            continue
        view = grid[top : top + ph, left : left + pw]
        view[patch] = int(color)
        return True
    return False


def _add_clutter(
    rng: np.random.Generator, grid: np.ndarray, clutter_palette: list[int]
) -> tuple[bool, int]:
    clutter_pixels = 0
    single_count = int(rng.integers(SINGLE_CLUTTER_RANGE[0], SINGLE_CLUTTER_RANGE[1] + 1))
    small_count = int(rng.integers(SMALL_BLOB_COUNT_RANGE[0], SMALL_BLOB_COUNT_RANGE[1] + 1))
    large_count = int(rng.integers(LARGE_BLOB_COUNT_RANGE[0], LARGE_BLOB_COUNT_RANGE[1] + 1))
    if rng.random() < 0.35:
        large_count = 0

    for _ in range(single_count):
        color = int(clutter_palette[int(rng.integers(len(clutter_palette)))])
        if not _place_patch(rng, grid, np.array([[True]], dtype=bool), color, pad=1):
            return False, clutter_pixels
        clutter_pixels += 1

    for _ in range(small_count):
        patch = _small_blob_patch(rng)
        color = int(clutter_palette[int(rng.integers(len(clutter_palette)))])
        if not _place_patch(rng, grid, patch, color, pad=1):
            return False, clutter_pixels
        clutter_pixels += int(patch.sum())

    for _ in range(large_count):
        patch = _large_irregular_patch(rng)
        color = int(clutter_palette[int(rng.integers(len(clutter_palette)))])
        if not _place_patch(rng, grid, patch, color, pad=1):
            return False, clutter_pixels
        clutter_pixels += int(patch.sum())

    return True, clutter_pixels


def _validate_case(
    grid: np.ndarray,
    input_table: np.ndarray,
    output_table: np.ndarray,
    bad_cell: tuple[int, int],
) -> bool:
    try:
        rectangles = _extract_rectangles(grid)
        table, rect_by_cell = _compress_rectangles(rectangles)
        if not np.array_equal(table, input_table):
            return False

        detected_bad, fixed_color = _find_unique_orbit_mismatch(table)
        if detected_bad != bad_cell:
            return False
        if fixed_color != int(output_table[bad_cell]):
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False

        bbox = rect_by_cell[bad_cell]
        expected = grid.copy()
        r0, c0, r1, c1 = bbox
        expected[r0 : r1 + 1, c0 : c1 + 1] = int(output_table[bad_cell])
        if not np.array_equal(out, expected):
            return False

        out_rectangles = _extract_rectangles(out)
        out_table, _ = _compress_rectangles(out_rectangles)
        if not np.array_equal(out_table, output_table):
            return False
        if not _table_is_biaxially_symmetric(out_table):
            return False
    except AssertionError:
        return False
    return True


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    rows = int(rng.choice(ROW_OPTIONS))
    cols = int(rng.choice(COL_OPTIONS))

    row_info = _sample_band_extents(rng, rows)
    col_info = _sample_band_extents(rng, cols)
    if row_info is None or col_info is None:
        raise RuntimeError("failed to sample lattice bands")
    row_extents, height = row_info
    col_extents, width = col_info

    input_table, output_table, bad_cell, _rect_palette, clutter_palette = _build_tables(rng, rows, cols)
    grid, rect_bboxes = _render_lattice(rng, input_table, row_extents, col_extents, (height, width))

    ok, clutter_pixels = _add_clutter(rng, grid, clutter_palette)
    if not ok or clutter_pixels == 0:
        raise RuntimeError("failed to place clutter")

    meta = {
        "input_table": input_table,
        "output_table": output_table,
        "bad_cell": bad_cell,
        "rect_bboxes": rect_bboxes,
    }
    return grid, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, meta = _generate_case(rng)
        except RuntimeError:
            continue
        if _validate_case(grid, meta["input_table"], meta["output_table"], meta["bad_cell"]):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
