import random
from collections import Counter

import numpy as np


BG = 0
SOURCE_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4, 7)

GRID_ROWS_RANGE = (13, 17)
GRID_COLS_RANGE = (13, 17)
MARKER_COUNT_RANGE = (5, 8)
MAX_ATTEMPTS = 500

# Every template contains the center cell (0, 0), fits in a 3x3 bbox,
# stays 4-connected, and has four distinct 90-degree rotations.
PATTERN_TEMPLATES = (
    ((-1, 0), (0, 0), (0, 1), (1, 1)),
    ((-1, 0), (0, 0), (0, 1), (1, 0), (1, 1)),
    ((-1, 0), (0, 0), (0, 1), (1, 0), (1, -1)),
    ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 1)),
    ((-1, -1), (-1, 0), (0, 0), (0, 1), (1, 0), (1, 1)),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rotate_point(dr: int, dc: int, turns: int) -> tuple[int, int]:
    turns %= 4
    for _ in range(turns):
        dr, dc = dc, -dr
    return dr, dc


def _rotate_offsets(offsets: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_rotate_point(dr, dc, turns) for dr, dc in offsets))


def _bbox(coords) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_source(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[tuple[int, int], ...], tuple[int, int], tuple[int, int, int, int]]:
    comps = [comp for comp in _color_components(grid, SOURCE_COLOR) if len(comp) >= 2]
    assert len(comps) == 1, "expected one multi-pixel gray source component"
    source_cells = set(comps[0])
    r0, c0, r1, c1 = _bbox(source_cells)
    assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "source bbox must have an integer center"
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    offsets = tuple(sorted((r - center[0], c - center[1]) for r, c in source_cells))
    assert (0, 0) in offsets, "source pattern must contain its center pixel"
    assert 4 <= len(offsets) <= 6, "source pattern area must stay compact"
    assert len({_rotate_offsets(offsets, k) for k in range(4)}) == 4, "source must be orientation-sensitive"
    return source_cells, offsets, center, (r0, c0, r1, c1)


def _marker_list(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val not in (BG, SOURCE_COLOR):
                markers.append((r, c, val))
    return markers


def _expand_bbox(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _inside_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _stamp_cells(r: int, c: int, offsets: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for dr, dc in offsets}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the gray source pattern onto each marker, rotating it by a row-count code.

    Input structure:
    - The background is black, with one compact gray source pattern and several colored
      single-pixel markers elsewhere in the grid.
    - The gray pattern is asymmetric and occupies a small odd-sized bounding box whose
      center pixel belongs to the pattern itself.
    - Markers may share rows; those row groupings are the only clue for the required rotations.

    Rule:
    - Identify the unique gray source component and take its bbox center as the stamping anchor.
    - For each colored marker, count how many other markers lie on the same row; call that count n.
    - Rotate the source pattern clockwise by n * 90 degrees, recolor it with the marker color,
      and stamp it centered on the marker.
    - Keep the original gray source unchanged, and when copies overlap, process markers row by row
      from left to right so later copies overwrite earlier ones.

    Key insight:
    - The rotation angle is not marked directly; it is encoded only by how many peer markers
      share each marker's row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    source_cells, source_offsets, _center, _box = _extract_source(g)
    markers = _marker_list(g)
    assert markers, "expected at least one colored marker"

    row_counts = Counter(r for r, _c, _color in markers)
    out = g.copy()
    h, w = g.shape

    for r, c, color in sorted(markers):
        n = row_counts[r] - 1
        assert 0 <= n <= 3, "row marker count must encode a 0-270 degree rotation"
        rotated = _rotate_offsets(source_offsets, n)
        for rr, cc in _stamp_cells(r, c, rotated):
            assert 0 <= rr < h and 0 <= cc < w, "valid ARC inputs keep every copy in bounds"
            assert (rr, cc) not in source_cells, "copies must not overwrite the gray source"
            out[rr, cc] = color

    return out


def _all_group_patterns() -> list[tuple[int, ...]]:
    patterns: set[tuple[int, ...]] = set()

    def rec(remaining: int, max_part: int, acc: list[int]) -> None:
        if remaining == 0:
            if len(acc) >= 2 and len(set(acc)) >= 2:
                patterns.add(tuple(acc))
            return
        for part in range(min(4, max_part, remaining), 0, -1):
            rec(remaining - part, part, acc + [part])

    for total in range(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1):
        rec(total, 4, [])
    return sorted(patterns)


GROUP_PATTERNS = _all_group_patterns()


def _choose_group_pattern() -> tuple[int, ...]:
    weights = []
    for pattern in GROUP_PATTERNS:
        weight = 1
        if 4 in pattern:
            weight += 3
        if 3 in pattern:
            weight += 2
        if len(set(pattern)) >= 3:
            weight += 1
        weights.append(weight)
    return random.choices(GROUP_PATTERNS, weights=weights, k=1)[0]


def _pick_source_center(h: int, w: int) -> tuple[int, int]:
    corners = {
        "tl": ([1, 2], [1, 2]),
        "tr": ([1, 2], [w - 3, w - 2]),
        "bl": ([h - 3, h - 2], [1, 2]),
        "br": ([h - 3, h - 2], [w - 3, w - 2]),
    }
    row_choices, col_choices = random.choice(list(corners.values()))
    return random.choice(row_choices), random.choice(col_choices)


def _marker_center_ok(
    r: int,
    c: int,
    rotated_offsets: tuple[tuple[int, int], ...],
    shape: tuple[int, int],
    source_cells: set[tuple[int, int]],
    forbidden_box: tuple[int, int, int, int],
    existing_markers: list[tuple[int, int]],
) -> bool:
    if _inside_box(r, c, forbidden_box):
        return False
    if any(max(abs(r - er), abs(c - ec)) <= 1 for er, ec in existing_markers):
        return False
    h, w = shape
    for rr, cc in _stamp_cells(r, c, rotated_offsets):
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if (rr, cc) in source_cells:
            return False
    return True


def _choose_rows(h: int, forbidden_box: tuple[int, int, int, int], row_count: int) -> list[int] | None:
    candidates = []
    for r in range(1, h - 1):
        if forbidden_box[0] <= r <= forbidden_box[2]:
            continue
        candidates.append(r)
    if len(candidates) < row_count:
        return None
    random.shuffle(candidates)
    return candidates[:row_count]


def _pairwise_overlap(markers: list[tuple[int, int, int]], source_offsets: tuple[tuple[int, int], ...]) -> tuple[int, bool]:
    row_counts = Counter(r for r, _c, _color in markers)
    footprints = []
    for r, c, color in markers:
        rotated = _rotate_offsets(source_offsets, row_counts[r] - 1)
        footprints.append((color, _stamp_cells(r, c, rotated)))

    overlap_cells = set()
    visible = False
    for i in range(len(footprints)):
        color_i, cells_i = footprints[i]
        for j in range(i + 1, len(footprints)):
            color_j, cells_j = footprints[j]
            shared = cells_i & cells_j
            if shared:
                overlap_cells |= shared
                if color_i != color_j:
                    visible = True
    return len(overlap_cells), visible


def _place_markers(
    h: int,
    w: int,
    source_offsets: tuple[tuple[int, int], ...],
    source_cells: set[tuple[int, int]],
    forbidden_box: tuple[int, int, int, int],
    group_pattern: tuple[int, ...],
    require_overlap: bool,
) -> list[tuple[int, int, int]] | None:
    for _ in range(200):
        rows = _choose_rows(h, forbidden_box, len(group_pattern))
        if rows is None:
            return None
        random.shuffle(rows)

        markers_rc: list[tuple[int, int]] = []
        occupied: set[tuple[int, int]] = set()
        overlap_used = 0
        success = True

        for row, group_size in zip(rows, group_pattern):
            rotated = _rotate_offsets(source_offsets, group_size - 1)
            cols = list(range(1, w - 1))
            random.shuffle(cols)
            row_markers: list[tuple[int, int]] = []

            for col in cols:
                if not _marker_center_ok(row, col, rotated, (h, w), source_cells, forbidden_box, markers_rc):
                    continue
                footprint = _stamp_cells(row, col, rotated)
                shared = len(footprint & occupied)
                if require_overlap:
                    if overlap_used == 0:
                        if shared > 2:
                            continue
                    elif shared != 0:
                        continue
                elif shared != 0:
                    continue

                markers_rc.append((row, col))
                row_markers.append((row, col))
                occupied |= footprint
                overlap_used += shared
                if len(row_markers) == group_size:
                    break

            if len(row_markers) != group_size:
                success = False
                break

        if not success:
            continue

        if require_overlap and overlap_used == 0:
            continue
        if not require_overlap and overlap_used != 0:
            continue

        colors = [random.choice(MARKER_COLORS) for _ in markers_rc]
        if len(set(colors)) < 2:
            continue
        markers = [(r, c, color) for (r, c), color in zip(markers_rc, colors)]
        overlap_count, visible = _pairwise_overlap(markers, source_offsets)
        if overlap_count > 2:
            continue
        if require_overlap and (overlap_count == 0 or not visible):
            continue
        return markers

    return None


def _render_input(
    h: int,
    w: int,
    source_cells: set[tuple[int, int]],
    markers: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in source_cells:
        grid[r, c] = SOURCE_COLOR
    for r, c, color in markers:
        grid[r, c] = color
    return grid


def _validate_generated_input(grid: np.ndarray, require_overlap: bool) -> bool:
    try:
        source_cells, source_offsets, _center, box = _extract_source(grid)
        markers = _marker_list(grid)
        if not (MARKER_COUNT_RANGE[0] <= len(markers) <= MARKER_COUNT_RANGE[1]):
            return False
        if any(color not in MARKER_COLORS for _r, _c, color in markers):
            return False

        row_counts = Counter(r for r, _c, _color in markers)
        if len({count - 1 for count in row_counts.values()}) < 2:
            return False
        if any(not (1 <= count <= 4) for count in row_counts.values()):
            return False

        forbidden_box = _expand_bbox(box, 1, grid.shape)
        footprints = []
        for r, c, color in markers:
            if _inside_box(r, c, forbidden_box):
                return False
            rotated = _rotate_offsets(source_offsets, row_counts[r] - 1)
            cells = _stamp_cells(r, c, rotated)
            if cells & source_cells:
                return False
            if any(not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]) for rr, cc in cells):
                return False
            footprints.append((color, cells))

        overlap_cells = set()
        visible_overlap = False
        for i in range(len(footprints)):
            color_i, cells_i = footprints[i]
            for j in range(i + 1, len(footprints)):
                color_j, cells_j = footprints[j]
                shared = cells_i & cells_j
                if shared:
                    overlap_cells |= shared
                    if color_i != color_j:
                        visible_overlap = True
        if require_overlap and (not overlap_cells or not visible_overlap):
            return False
        if not require_overlap and overlap_cells:
            return False
        if len(overlap_cells) > 2:
            return False

        out = main(grid)
        if out.shape != grid.shape or np.array_equal(out, grid):
            return False
        return max(grid.shape) <= 30 and max(out.shape) <= 30
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS_RANGE)
    w = random.randint(*GRID_COLS_RANGE)
    base = random.choice(PATTERN_TEMPLATES)
    source_offsets = _rotate_offsets(base, random.randrange(4))
    center = _pick_source_center(h, w)
    source_cells = {(center[0] + dr, center[1] + dc) for dr, dc in source_offsets}
    box = _bbox(source_cells)
    forbidden_box = _expand_bbox(box, 1, (h, w))

    group_pattern = _choose_group_pattern()
    require_overlap = random.random() < 0.3
    markers = _place_markers(h, w, source_offsets, source_cells, forbidden_box, group_pattern, require_overlap)
    if markers is None:
        return None

    grid = _render_input(h, w, source_cells, markers)
    if not _validate_generated_input(grid, require_overlap=require_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
