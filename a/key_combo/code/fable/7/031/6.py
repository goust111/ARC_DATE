"""ARC task: restore a missing sprite from its diagonal counterpart."""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
ShapeKey = tuple[int, int, tuple[int, ...]]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of local (row, column) coordinates."""
    height, width = mask.shape
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    result: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        cells = [start]
        while queue:
            row, col = queue.popleft()
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    queue.append((nr, nc))
                    cells.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _shape_key_and_anchor(coords: np.ndarray) -> tuple[ShapeKey, Coord, np.ndarray]:
    """Normalize a component by its inclusive bbox, preserving the (r,c) convention."""
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    local = coords - np.array([r0, c0])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[local[:, 0], local[:, 1]] = True
    key = (mask.shape[0], mask.shape[1], tuple(map(int, mask.reshape(-1))))
    return key, (r0, c0), mask


def _quadrants(grid: np.ndarray, cross_r: int, cross_c: int):
    return (
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, grid.shape[1])),
        (slice(cross_r + 1, grid.shape[0]), slice(0, cross_c)),
        (slice(cross_r + 1, grid.shape[0]), slice(cross_c + 1, grid.shape[1])),
    )


def _parse_region(region: np.ndarray, background: int):
    colors = set(map(int, region[region != background]))
    assert len(colors) == 1, "each quadrant must use one foreground color"
    color = colors.pop()
    records = []
    for coords in _components(region == color):
        key, anchor, mask = _shape_key_and_anchor(coords)
        records.append((key, anchor, mask))
    assert records, "each quadrant must contain sprites"
    return color, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the one missing sprite at the matching position in the diagonal quadrant.

    Input structure:
    - A background-colored full row and full column split an odd-sized grid into
      four equal quadrants, each containing monochrome, separated sprites in its
      own foreground color.
    - Ignoring color and translation, three quadrants share one sprite-shape
      multiset and exactly one quadrant lacks one occurrence.

    Rule:
    - Normalize every 4-connected sprite by its bounding box and compare the four
      shape multisets to identify the deficient quadrant and missing shape.
    - In that quadrant, draw the missing occurrence at the local bbox position
      absent relative to its diagonal counterpart, using the deficient quadrant's color.

    Key insight:
    - Cross-quadrant multiplicities reveal what is missing, while only the diagonal
      quadrant supplies its position; adjacent-quadrant positions are distractors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 1 and width % 2 == 1, "equal quadrants require odd dimensions"
    cross_r, cross_c = height // 2, width // 2
    background = int(g[cross_r, cross_c])
    assert np.all(g[cross_r, :] == background), "expected a solid separator row"
    assert np.all(g[:, cross_c] == background), "expected a solid separator column"

    regions = _quadrants(g, cross_r, cross_c)
    parsed = [_parse_region(g[rs, cs], background) for rs, cs in regions]
    counters = [Counter(key for key, _anchor, _mask in records) for _color, records in parsed]

    target = Counter()
    for counts in counters:
        for key, count in counts.items():
            target[key] = max(target[key], count)
    deficits = [target - counts for counts in counters]
    deficient = [idx for idx, deficit in enumerate(deficits) if sum(deficit.values()) > 0]
    assert len(deficient) == 1, "expected exactly one deficient quadrant"
    missing_q = deficient[0]
    missing = deficits[missing_q]
    assert sum(missing.values()) == 1 and len(missing) == 1, "expected exactly one missing sprite occurrence"
    missing_key = next(iter(missing))

    diagonal_q = 3 - missing_q
    diagonal_records = parsed[diagonal_q][1]
    local_records = parsed[missing_q][1]
    source_positions = Counter(anchor for key, anchor, _mask in diagonal_records if key == missing_key)
    present_positions = Counter(anchor for key, anchor, _mask in local_records if key == missing_key)
    absent_positions = source_positions - present_positions
    assert sum(absent_positions.values()) == 1, "diagonal layout must identify one unique missing position"
    anchor = next(iter(absent_positions))
    source_masks = [mask for key, pos, mask in diagonal_records if key == missing_key and pos == anchor]
    assert len(source_masks) == 1, "missing diagonal occurrence must be unique at its position"
    sprite_mask = source_masks[0]

    out = g.copy()
    row_slice, col_slice = regions[missing_q]
    r0 = int(row_slice.start) + anchor[0]
    c0 = int(col_slice.start) + anchor[1]
    patch = out[r0 : r0 + sprite_mask.shape[0], c0 : c0 + sprite_mask.shape[1]]
    assert patch.shape == sprite_mask.shape and np.all(patch[sprite_mask] == background), "missing slot must be empty"
    patch[sprite_mask] = parsed[missing_q][0]
    return out


_SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
)


def _dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1)
    result = np.zeros_like(mask)
    for dr in range(3):
        for dc in range(3):
            result |= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return result


def _place_layout(side: int, shapes: list[np.ndarray], forbidden=None):
    """Place sprites with a one-cell orthogonal moat and optional per-shape anchor bans."""
    occupied = np.zeros((side, side), dtype=bool)
    placements: list[tuple[int, int, np.ndarray]] = []
    order = list(range(len(shapes)))
    random.shuffle(order)
    for index in order:
        shape = shapes[index]
        sh, sw = shape.shape
        candidates = []
        for r0 in range(side - sh + 1):
            for c0 in range(side - sw + 1):
                if forbidden is not None and (r0, c0) in forbidden[index]:
                    continue
                patch = np.zeros_like(occupied)
                patch[r0 : r0 + sh, c0 : c0 + sw] = shape
                if not np.any(_dilate(occupied) & patch):
                    candidates.append((r0, c0, patch))
        if not candidates:
            return None
        r0, c0, patch = random.choice(candidates)
        occupied |= patch
        placements.append((index, r0, c0))
    placements.sort()
    return placements


def _render_region(side: int, color: int, shapes: list[np.ndarray], placements, omit=None):
    region = np.full((side, side), -1, dtype=int)
    for index, r0, c0 in placements:
        if index == omit:
            continue
        shape = shapes[index]
        region[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]][shape] = color
    return region


def _generate_attempt() -> np.ndarray | None:
    side = random.randint(9, 14)
    object_count = random.randint(4, min(6, side - 3))

    # One template appears twice. Deleting one copy forces true multiset comparison.
    duplicate = random.choice(_SHAPES)
    distinct_pool = [s for s in _SHAPES if not np.array_equal(s, duplicate)]
    shapes = [duplicate.copy(), duplicate.copy()]
    shapes.extend(s.copy() for s in random.sample(distinct_pool, object_count - 2))

    layout_a = _place_layout(side, shapes)
    if layout_a is None:
        return None
    # Adjacent quadrants must not reuse any anchor occupied by the same shape,
    # including another occurrence of a duplicated template.
    forbidden = []
    for i, shape in enumerate(shapes):
        forbidden.append(
            {
                (layout_a[j][1], layout_a[j][2])
                for j, other in enumerate(shapes)
                if np.array_equal(shape, other)
            }
        )
    layout_b = _place_layout(side, shapes, forbidden=forbidden)
    if layout_b is None:
        return None

    missing_q = random.randrange(4)
    missing_index = random.choice((0, 1))
    missing_layout = layout_a if missing_q in (0, 3) else layout_b
    adjacent_layout = layout_b if missing_q in (0, 3) else layout_a
    missing_anchor = (missing_layout[missing_index][1], missing_layout[missing_index][2])
    # The adjacent occurrence must not accidentally suggest the right location.
    if missing_anchor in {(adjacent_layout[i][1], adjacent_layout[i][2]) for i in (0, 1)}:
        return None

    palette = list(range(10))
    background = random.choice(palette)
    colors = random.sample([color for color in palette if color != background], 4)
    size = 2 * side + 1
    grid = np.full((size, size), background, dtype=int)
    expected = grid.copy()
    slices = _quadrants(grid, side, side)
    for q, (rs, cs) in enumerate(slices):
        placements = layout_a if q in (0, 3) else layout_b
        complete = _render_region(side, colors[q], shapes, placements)
        incomplete = _render_region(side, colors[q], shapes, placements, omit=missing_index if q == missing_q else None)
        complete[complete == -1] = background
        incomplete[incomplete == -1] = background
        expected[rs, cs] = complete
        grid[rs, cs] = incomplete

    # Independent semantic checks: only the intended mask is absent, diagonal
    # layouts coincide, adjacent layouts differ, and the solver restores exactly it.
    changed = expected != grid
    missing_shape = shapes[missing_index]
    if int(changed.sum()) != int(missing_shape.sum()):
        return None
    if not np.all(expected[changed] == colors[missing_q]):
        return None
    try:
        solved = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate instance satisfying the puzzle contract."""
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-diagonal-sprite instance")
