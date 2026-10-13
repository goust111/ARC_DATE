from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
SPRITE_COLORS = (3, 4, 5, 6, 7, 8, 9)
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _canonical_path(start: tuple[int, int], end: tuple[int, int]):
    """Return (all intermediate cells, diagonal segment, straight segment)."""
    r, c = start
    er, ec = end
    sr, sc = _sign(er - r), _sign(ec - c)
    diagonal = []
    while r != er and c != ec:
        r += sr
        c += sc
        if (r, c) != end:
            diagonal.append((r, c))
    straight = []
    while r != er or c != ec:
        if r != er:
            r += sr
        else:
            c += sc
        if (r, c) != end:
            straight.append((r, c))
    return diagonal + straight, diagonal, straight


def _straight_first_path(start: tuple[int, int], end: tuple[int, int]):
    """The competing shortest path with the cardinal excess before diagonals."""
    r, c = start
    er, ec = end
    sr, sc = _sign(er - r), _sign(ec - c)
    row_steps, col_steps = abs(er - r), abs(ec - c)
    cells = []
    for _ in range(abs(row_steps - col_steps)):
        if row_steps > col_steps:
            r += sr
        else:
            c += sc
        if (r, c) != end:
            cells.append((r, c))
    while r != er and c != ec:
        r += sr
        c += sc
        if (r, c) != end:
            cells.append((r, c))
    return cells


def _sprite_components(grid: np.ndarray, bg: int, markers: set[tuple[int, int]]):
    """Find separated sprites as 8-connected foreground components."""
    mask = grid != bg
    for cell in markers:
        mask[cell] = False
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[set[tuple[int, int]]] = []
    for rr, cc in np.argwhere(mask):
        seed = (int(rr), int(cc))
        if seen[seed]:
            continue
        color = int(grid[seed])
        queue = deque([seed])
        seen[seed] = True
        component = set()
        while queue:
            r, c = queue.popleft()
            component.add((r, c))
            for dr, dc in NEIGHBORS8:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                    and not seen[nr, nc]
                    and mask[nr, nc]
                ):
                    assert int(grid[nr, nc]) == color, "touching sprites must be monochrome"
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(component)
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every whole sprite touched by the invisible canonical blue-to-red king path.

    Input structure:
    - A dominant uniform background contains exactly one blue start pixel and one
      red end pixel, in different rows and columns.
    - The remaining foreground consists of separated monochrome sprites; sprites
      do not touch one another or the endpoint markers, even diagonally.

    Rule:
    - From blue, take diagonal king steps toward red until the current cell shares
      a row or column with red, then take straight steps to red.  Exclude both
      endpoints from this invisible path.
    - If any cell of a sprite intersects the path, erase that entire sprite to the
      background.  Preserve both markers and every untouched sprite, and do not
      draw the path.

    Key insight:
    - Membership in the markers' bounding box is insufficient: the exact
      diagonal-first shortest king path determines which complete sprites vanish.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    blue_cells = np.argwhere(g == BLUE)
    red_cells = np.argwhere(g == RED)
    assert len(blue_cells) == 1 and len(red_cells) == 1, "expected unique blue and red markers"
    start = tuple(map(int, blue_cells[0]))
    end = tuple(map(int, red_cells[0]))
    assert start[0] != end[0] and start[1] != end[1], "markers must differ in row and column"

    path = set(_canonical_path(start, end)[0])
    components = _sprite_components(g, bg, {start, end})
    assert components, "expected at least one sprite"
    out = g.copy()
    for component in components:
        if component & path:
            for r, c in component:
                out[r, c] = bg
    return out


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1):
    expanded = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    expanded.add((nr, nc))
    return expanded


def _place_shape(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    cells: set[tuple[int, int]],
    color: int,
) -> bool:
    if not cells or any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) for r, c in cells):
        return False
    if _expanded(cells, grid.shape) & occupied:
        return False
    for r, c in cells:
        grid[r, c] = color
    occupied.update(cells)
    return True


def _anchored_shapes(anchor: tuple[int, int]):
    r, c = anchor
    shapes = [
        {(r, c), (r + 1, c)},
        {(r, c), (r, c + 1)},
        {(r, c), (r - 1, c)},
        {(r, c), (r, c - 1)},
        {(r, c), (r + 1, c), (r + 1, c + 1)},
        {(r, c), (r, c + 1), (r - 1, c + 1)},
    ]
    random.shuffle(shapes)
    return shapes


def _try_anchored(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    anchors: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    color: int,
    require_off_forbidden: bool = False,
):
    choices = anchors.copy()
    random.shuffle(choices)
    for anchor in choices:
        for cells in _anchored_shapes(anchor):
            if cells & forbidden:
                continue
            if require_off_forbidden and cells <= set(anchors):
                continue
            if _place_shape(grid, occupied, cells, color):
                return cells
    return None


def _semantic_audit(grid: np.ndarray) -> bool:
    """Independently check the intended examples and anti-shortcut witnesses."""
    blue = tuple(map(int, np.argwhere(grid == BLUE)[0]))
    red = tuple(map(int, np.argwhere(grid == RED)[0]))
    path_list, diagonal, straight = _canonical_path(blue, red)
    path, diag_set, straight_set = set(path_list), set(diagonal), set(straight)
    alternative = set(_straight_first_path(blue, red))
    components = _sprite_components(grid, BACKGROUND, {blue, red})
    hit = [obj for obj in components if obj & path]
    survivors = [obj for obj in components if not obj & path]
    if len(hit) < 2 or len(survivors) < 2:
        return False
    if not any(obj & diag_set and not obj & straight_set for obj in hit):
        return False
    if not any(obj & straight_set and not obj & diag_set for obj in hit):
        return False
    if not any(obj & path and obj - path for obj in hit):
        return False
    r0, r1 = sorted((blue[0], red[0]))
    c0, c1 = sorted((blue[1], red[1]))
    if not any(any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in obj) for obj in survivors):
        return False
    if not any(obj & (alternative - path) for obj in survivors):
        return False
    canonical_deleted = {i for i, obj in enumerate(components) if obj & path}
    alternative_deleted = {i for i, obj in enumerate(components) if obj & alternative}
    if canonical_deleted == alternative_deleted:
        return False
    if not any(obj & diag_set and not obj & alternative for obj in hit):
        return False

    expected = grid.copy()
    for obj in hit:
        for cell in obj:
            expected[cell] = BACKGROUND
    output = main(grid)
    if not np.array_equal(output, expected):
        return False
    if int(np.count_nonzero(output != grid)) < 3:
        return False
    return int(np.count_nonzero(output != BACKGROUND)) >= 4


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(15, 24)
    width = random.randint(15, 24)
    # Unequal deltas guarantee nonempty diagonal and straight portions.  Their
    # roles and all four travel quadrants are sampled symmetrically.
    if random.choice((False, True)):
        row_delta = random.randint(7, min(12, height - 5))
        col_delta = random.randint(3, min(row_delta - 3, width - 5))
    else:
        col_delta = random.randint(7, min(12, width - 5))
        row_delta = random.randint(3, min(col_delta - 3, height - 5))
    sr, sc = random.choice((-1, 1)), random.choice((-1, 1))
    r_lo = 2 if sr > 0 else row_delta + 2
    r_hi = height - row_delta - 3 if sr > 0 else height - 3
    c_lo = 2 if sc > 0 else col_delta + 2
    c_hi = width - col_delta - 3 if sc > 0 else width - 3
    if r_lo > r_hi or c_lo > c_hi:
        return None
    start = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
    end = (start[0] + sr * row_delta, start[1] + sc * col_delta)

    grid = np.zeros((height, width), dtype=int)
    grid[start], grid[end] = BLUE, RED
    occupied = {start, end}
    path_list, diagonal, straight = _canonical_path(start, end)
    path = set(path_list)
    alternative = set(_straight_first_path(start, end))
    colors = random.sample(SPRITE_COLORS, 5)

    # Two required erased sprites make both path phases necessary.
    diag_obj = _try_anchored(grid, occupied, diagonal[1:-1] or diagonal, set(), colors[0], True)
    if diag_obj is None:
        return None
    straight_obj = _try_anchored(grid, occupied, straight, set(), colors[1], True)
    if straight_obj is None:
        return None

    # A survivor on the competing straight-first route demonstrates segment order.
    alt_anchors = list(alternative - path)
    alt_obj = _try_anchored(grid, occupied, alt_anchors, path, colors[2])
    if alt_obj is None:
        return None

    # Another bbox-interior survivor defeats erasing everything between markers.
    r0, r1 = sorted((start[0], end[0]))
    c0, c1 = sorted((start[1], end[1]))
    interior = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if (r, c) not in path and (r, c) not in alternative
    ]
    random.shuffle(interior)
    inside_obj = None
    bbox_cells = {
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    }
    for anchor in interior:
        for cells in _anchored_shapes(anchor):
            if cells <= bbox_cells and not cells & (path | alternative):
                if _place_shape(grid, occupied, cells, colors[3]):
                    inside_obj = cells
                    break
        if inside_obj is not None:
            break
    if inside_obj is None:
        return None

    # Add 1--3 varied distractors, including occasional extra erased objects.
    target_extra = random.randint(1, 3)
    free_anchors = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(free_anchors)
    placed = 0
    for anchor in free_anchors:
        if placed >= target_extra:
            break
        color = random.choice(SPRITE_COLORS)
        shapes = _anchored_shapes(anchor)
        if random.random() < 0.35:
            shapes.append({anchor})
        for cells in shapes:
            if _place_shape(grid, occupied, cells, color):
                placed += 1
                break
    if placed != target_extra:
        return None
    if not _semantic_audit(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse, audited instance of the invisible-path filtering task."""
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible king-path instance")
