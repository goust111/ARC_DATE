"""ARC task 3: complementary corners select identical grey copies."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np


BACKGROUND = 0
GREY = 5
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_RNG = np.random.default_rng()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for start_r, start_c in np.argwhere(mask):
        start = (int(start_r), int(start_c))
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _normalized_shape(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    origin = coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords - origin))


def _background_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    maximum = max(counts.values())
    winners = [color for color, count in counts.items() if count == maximum]
    assert len(winners) == 1, "expected one structurally dominant background color"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the identical grey copy named by each marker pair's occupied complementary corner and erase everything else.

    Input structure:
    - A dominant background contains several separated, identically shaped, 4-connected grey copies.
    - Every other foreground color occurs exactly twice as two marker pixels, and markers never overlap grey cells.
    - For each non-aligned pair, exactly one complementary corner of its implicit rectangle lies in a grey copy; different pairs hit different copies.
    - At least one same-row or same-column pair is degenerate, and at least one grey copy is not hit.

    Rule:
    - For markers at (r1,c1) and (r2,c2), inspect (r1,c2) and (r2,c1) in the original grid.
    - A nondegenerate pair recolors the entire grey copy containing its unique grey complementary corner with the marker color.
    - Degenerate pairs select nothing; erase every marker and every unselected grey copy to background.

    Key insight:
    - The markers are opposite corners of an invisible rectangle: the other, unpainted corners select copies, while aligned pairs are deliberate void controls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert np.issubdtype(g.dtype, np.integer), "ARC colors must be integers"

    bg = _background_color(g)
    assert bg != GREY, "grey is reserved for the identical copies"
    grey_components = _components(g == GREY)
    assert len(grey_components) >= 2, "expected multiple grey copies"

    shapes = {_normalized_shape(component) for component in grey_components}
    assert len(shapes) == 1, "all grey components must be identical copies"

    owner = np.full(g.shape, -1, dtype=int)
    for index, component in enumerate(grey_components):
        owner[component[:, 0], component[:, 1]] = index

    marker_colors = [int(color) for color in np.unique(g) if int(color) not in (bg, GREY)]
    assert marker_colors, "expected colored marker pairs"
    selected: dict[int, int] = {}
    degenerate_count = 0

    for color in marker_colors:
        positions = np.argwhere(g == color)
        assert positions.shape == (2, 2), "every marker color must occur exactly twice"
        (r1, c1), (r2, c2) = (tuple(map(int, position)) for position in positions)
        if r1 == r2 or c1 == c2:
            degenerate_count += 1
            continue

        corners = ((r1, c2), (r2, c1))
        hit_corners = [corner for corner in corners if int(owner[corner]) >= 0]
        assert len(hit_corners) == 1, "a nondegenerate pair must have exactly one grey complementary corner"
        copy_index = int(owner[hit_corners[0]])
        assert copy_index not in selected, "different marker colors must select different copies"
        selected[copy_index] = color

    assert selected, "expected at least one selected copy"
    assert degenerate_count >= 1, "expected at least one degenerate marker pair"
    assert len(selected) < len(grey_components), "expected at least one unselected copy"

    out = np.full_like(g, bg)
    for copy_index, color in selected.items():
        component = grey_components[copy_index]
        out[component[:, 0], component[:, 1]] = color
    return out


# Connected polyominoes with multiple rows and columns. A random rigid transform
# is applied once, then the exact resulting orientation is copied everywhere.
_BASE_SHAPES = (
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (0, 2)),
)


def _random_shape(rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    coords = np.asarray(_BASE_SHAPES[int(rng.integers(len(_BASE_SHAPES)))], dtype=int)
    for _ in range(int(rng.integers(4))):
        coords = np.column_stack((coords[:, 1], -coords[:, 0]))
    if bool(rng.integers(2)):
        coords[:, 1] *= -1
    coords -= coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Require two full background rows or columns between sprite bboxes."""
    return a[2] + 2 < b[0] or b[2] + 2 < a[0] or a[3] + 2 < b[1] or b[3] + 2 < a[1]


def _place_copies(
    shape: tuple[tuple[int, int], ...], count: int, height: int, width: int, rng: np.random.Generator
) -> list[tuple[tuple[int, int], ...]] | None:
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    candidates = [
        (r, c)
        for r in range(1, height - shape_h)
        for c in range(1, width - shape_w)
    ]
    rng.shuffle(candidates)
    boxes: list[tuple[int, int, int, int]] = []
    copies: list[tuple[tuple[int, int], ...]] = []
    for r0, c0 in candidates:
        box = (r0, c0, r0 + shape_h - 1, c0 + shape_w - 1)
        if not all(_boxes_separated(box, old) for old in boxes):
            continue
        boxes.append(box)
        copies.append(tuple((r0 + r, c0 + c) for r, c in shape))
        if len(copies) == count:
            return copies
    return None


def _far_from_items(
    cell: tuple[int, int], grey_cells: set[tuple[int, int]], markers: set[tuple[int, int]],
    hidden_corners: set[tuple[int, int]], height: int, width: int,
) -> bool:
    r, c = cell
    if not (0 <= r < height and 0 <= c < width):
        return False
    if cell in grey_cells or cell in markers or cell in hidden_corners:
        return False
    visible = grey_cells | markers
    return all(max(abs(r - vr), abs(c - vc)) >= 2 for vr, vc in visible)


def _choose_selected_pair(
    target_cells: tuple[tuple[int, int], ...], quadrant: tuple[int, int],
    grey_cells: set[tuple[int, int]], markers: set[tuple[int, int]],
    hidden_corners: set[tuple[int, int]], height: int, width: int,
    rng: np.random.Generator, prefer_noncorner: bool,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    rows = [r for r, _ in target_cells]
    cols = [c for _, c in target_cells]
    r_extremes, c_extremes = {min(rows), max(rows)}, {min(cols), max(cols)}
    ordered_targets = list(target_cells)
    rng.shuffle(ordered_targets)
    if prefer_noncorner:
        ordered_targets.sort(key=lambda p: p[0] in r_extremes and p[1] in c_extremes)

    for p in ordered_targets:
        q_candidates = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if (r - p[0]) * quadrant[0] >= 4 and (c - p[1]) * quadrant[1] >= 4
        ]
        rng.shuffle(q_candidates)
        for q in q_candidates:
            if q in grey_cells or q in markers or q in hidden_corners:
                continue
            marker1 = (p[0], q[1])
            marker2 = (q[0], p[1])
            if not _far_from_items(marker1, grey_cells, markers, hidden_corners, height, width):
                continue
            if not _far_from_items(marker2, grey_cells, markers | {marker1}, hidden_corners, height, width):
                continue
            return p, q, marker1, marker2
    return None


def _choose_degenerate_pair(
    horizontal: bool, grey_cells: set[tuple[int, int]], markers: set[tuple[int, int]],
    hidden_corners: set[tuple[int, int]], height: int, width: int, rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if horizontal:
        for r in range(height):
            for c1 in range(width):
                for c2 in range(c1 + 5, width):
                    candidates.append(((r, c1), (r, c2)))
    else:
        for c in range(width):
            for r1 in range(height):
                for r2 in range(r1 + 5, height):
                    candidates.append(((r1, c), (r2, c)))
    rng.shuffle(candidates)
    for first, second in candidates:
        if not _far_from_items(first, grey_cells, markers, hidden_corners, height, width):
            continue
        if not _far_from_items(second, grey_cells, markers | {first}, hidden_corners, height, width):
            continue
        return first, second
    return None


def _validate_generated(
    grid: np.ndarray, copies: list[tuple[tuple[int, int], ...]],
    selected_specs: list[dict], degenerate_specs: list[dict], expected: np.ndarray,
) -> bool:
    """Check puzzle semantics independently from the solver's component traversal."""
    grey_cells = {cell for copy in copies for cell in copy}
    if set(map(tuple, np.argwhere(grid == GREY))) != grey_cells:
        return False
    if len(selected_specs) < 2 or len(copies) <= len(selected_specs):
        return False
    if {spec["copy"] for spec in selected_specs} != set(range(len(selected_specs))):
        return False
    if {spec["horizontal"] for spec in degenerate_specs} != {False, True}:
        return False

    expected_direct = np.full_like(grid, BACKGROUND)
    for spec in selected_specs:
        color, copy_index = spec["color"], spec["copy"]
        marker1, marker2 = spec["markers"]
        p, q = spec["corners"]
        if grid[marker1] != color or grid[marker2] != color or grid[q] != BACKGROUND:
            return False
        if marker1[0] == marker2[0] or marker1[1] == marker2[1]:
            return False
        if {p, q} != {(marker1[0], marker2[1]), (marker2[0], marker1[1])}:
            return False
        if (p in grey_cells) + (q in grey_cells) != 1 or p not in copies[copy_index]:
            return False
        rows, cols = zip(*copies[copy_index])
        # At least the first selected target is not merely a sprite-bbox corner.
        if copy_index == 0 and p[0] in (min(rows), max(rows)) and p[1] in (min(cols), max(cols)):
            return False
        rr, cc = zip(*copies[copy_index])
        expected_direct[rr, cc] = color

    for spec in degenerate_specs:
        first, second = spec["markers"]
        color = spec["color"]
        if grid[first] != color or grid[second] != color:
            return False
        if spec["horizontal"] != (first[0] == second[0]):
            return False
        if not (first[0] == second[0] or first[1] == second[1]):
            return False

    marker_mask = (grid != BACKGROUND) & (grid != GREY)
    if any(np.count_nonzero(grid == color) != 2 for color in np.unique(grid[marker_mask])):
        return False
    if np.any(expected_direct[marker_mask] != BACKGROUND):
        return False
    if not np.array_equal(expected, expected_direct):
        return False
    try:
        actual = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    return np.array_equal(actual, expected_direct)


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    height = int(rng.integers(20, 29))
    width = int(rng.integers(20, 29))
    copy_count = int(rng.integers(4, 7))
    selected_count = int(rng.integers(2, min(4, copy_count - 1) + 1))
    shape = _random_shape(rng)
    copies = _place_copies(shape, copy_count, height, width, rng)
    if copies is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grey_cells = {cell for copy in copies for cell in copy}
    for r, c in grey_cells:
        grid[r, c] = GREY

    total_pairs = selected_count + 2  # two degenerate controls, one per alignment
    colors = [int(c) for c in rng.choice(MARKER_PALETTE, size=total_pairs, replace=False)]
    markers: set[tuple[int, int]] = set()
    hidden_corners: set[tuple[int, int]] = set()
    quadrants = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    rng.shuffle(quadrants)
    selected_specs: list[dict] = []

    # Copies 0..selected_count-1 are selected; random placement makes this index
    # unrelated to any visible ordering of the copies.
    for copy_index in range(selected_count):
        pair = _choose_selected_pair(
            copies[copy_index], quadrants[copy_index], grey_cells, markers,
            hidden_corners, height, width, rng, prefer_noncorner=(copy_index == 0),
        )
        if pair is None:
            return None
        p, q, marker1, marker2 = pair
        color = colors[copy_index]
        grid[marker1] = color
        grid[marker2] = color
        markers.update((marker1, marker2))
        hidden_corners.add(q)
        selected_specs.append(
            {"color": color, "copy": copy_index, "corners": (p, q), "markers": (marker1, marker2)}
        )

    degenerate_specs: list[dict] = []
    orientations = [True, False]
    rng.shuffle(orientations)
    for offset, horizontal in enumerate(orientations):
        pair = _choose_degenerate_pair(horizontal, grey_cells, markers, hidden_corners, height, width, rng)
        if pair is None:
            return None
        color = colors[selected_count + offset]
        first, second = pair
        grid[first] = color
        grid[second] = color
        markers.update(pair)
        degenerate_specs.append({"color": color, "horizontal": horizontal, "markers": pair})

    expected = np.full_like(grid, BACKGROUND)
    for spec in selected_specs:
        rr, cc = zip(*copies[spec["copy"]])
        expected[rr, cc] = spec["color"]

    if not _validate_generated(grid, copies, selected_specs, degenerate_specs, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """Generate a diverse valid instance with selected, unselected, and degenerate cases."""
    for _ in range(500):
        generated = _gen_attempt(_RNG)
        if generated is None:
            continue
        grid, expected = generated
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return grid
    raise RuntimeError("failed to generate a valid complementary-corner instance")
