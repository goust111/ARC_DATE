from __future__ import annotations

from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _color_frames(g: np.ndarray, bg: int) -> tuple[list[int], dict[int, np.ndarray]]:
    colors = colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected multiple foreground frame colors"

    frames: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, "foreground colors must have visible pixels"
        r0, c0, r1, c1 = bbox_coords(coords)
        assert r1 > r0 and c1 > c0, "each color must define a rectangular frame"
        frame = _frame_mask(g.shape, (r0, c0, r1, c1))
        assert np.all(frame[g == color]), "visible pixels must lie on the inferred frame perimeter"
        frames[int(color)] = frame
    return colors, frames


def _crossing_edges(g: np.ndarray, colors: list[int], frames: dict[int, np.ndarray]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b in combinations(colors, 2):
        crossing = frames[a] & frames[b]
        for r_raw, c_raw in np.argwhere(crossing):
            r = int(r_raw)
            c = int(c_raw)
            visible = int(g[r, c])
            if visible == a:
                edges.add((b, a))
            elif visible == b:
                edges.add((a, b))
            else:
                assert visible in colors, "frame crossings must be covered by a foreground frame"
    assert edges, "expected visible crossings between frames"
    return edges


def _unique_path(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {color: [other for other in colors if (color, other) in edges] for color in colors}
    paths: list[tuple[int, ...]] = []

    def search(path: tuple[int, ...], remaining: set[int]) -> None:
        if not remaining:
            paths.append(path)
            return
        for nxt in outgoing[path[-1]]:
            if nxt in remaining:
                search((*path, nxt), remaining - {nxt})

    color_set = set(colors)
    for start in colors:
        search((start,), color_set - {start})

    assert len(paths) == 1, "crossings must determine a unique frame order"
    return list(paths[0])


def _render_nested(colors: list[int]) -> np.ndarray:
    size = 2 * len(colors) - 1
    out = np.empty((size, size), dtype=int)
    for layer, color in enumerate(colors):
        end = size - layer - 1
        out[layer : end + 1, layer] = color
        out[layer : end + 1, end] = color
        out[layer, layer : end + 1] = color
        out[end, layer : end + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Untangle overlapping colored rectangular frames into concentric square frames ordered by their visible crossings.

    Input structure
    - The background contains several one-cell-thick axis-aligned rectangular frames, one color per frame.
    - Frames overlap, and the visible color at an overlap records a directed crossing from the hidden frame to the visible frame.
    - Each foreground color still has visible pixels on the perimeter of its full rectangular bounding box.

    Rule
    - Infer each color's full rectangular frame from the bounding box of its visible pixels.
    - Add a directed edge A -> B when B is visible at a crossing of A and B's frame perimeters.
    - Find the unique path that visits every frame color once, then render those colors as one-pixel-thick concentric square frames from outside to inside.

    Key insight
    - The output is not a crop of the input; the crossing graph gives a single frame order, and that order is redrawn as nested rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    colors, frames = _color_frames(g, bg)
    edges = _crossing_edges(g, colors, frames)
    order = _unique_path(colors, edges)
    return _render_nested(order)

# --------------- generator (att1) ---------------

# Color and frame count bounds.
MIN_FRAME_COUNT = 5
MAX_FRAME_COUNT_EXCLUSIVE = 8
AVAILABLE_COLORS = list(range(1, 10))

# Square canvas and rectangular frame size bounds.
MIN_GRID_SIZE = 12
MAX_GRID_SIZE_EXCLUSIVE = 26
GRID_SIZE_PER_FRAME = 2
GRID_SIZE_PADDING = 2
MIN_FRAME_DIM = 3
MAX_FRAME_DIM = 13

# Rejection thresholds and retry budgets.
MIN_OVERLAP_DENSITY = 0.57
LAYOUT_ATTEMPTS = 50
PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 300


def _choose_ordering(rng):
    """Return colors ordered from back/outermost to front/innermost."""
    n = rng.integers(MIN_FRAME_COUNT, MAX_FRAME_COUNT_EXCLUSIVE)
    colors = list(AVAILABLE_COLORS)
    rng.shuffle(colors)
    return list(colors[:n])


def _choose_grid_size(rng, n):
    min_size = max(MIN_GRID_SIZE, n * GRID_SIZE_PER_FRAME + GRID_SIZE_PADDING)
    return int(rng.integers(min_size, MAX_GRID_SIZE_EXCLUSIVE))


def _random_bbox(rng, size, min_dim, max_dim):
    h = rng.integers(min_dim, max_dim + 1)
    w = rng.integers(min_dim, max_dim + 1)
    r0 = rng.integers(0, size - h + 1)
    c0 = rng.integers(0, size - w + 1)
    return (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))


def _place_bboxes(rng, ordering, size):
    """Place frames so every consecutive pair has intersecting perimeters."""
    min_dim = MIN_FRAME_DIM
    max_dim = min(MAX_FRAME_DIM, size - 1)

    for _ in range(LAYOUT_ATTEMPTS):
        bboxes = {}
        ok = True
        for i, color in enumerate(ordering):
            if i == 0:
                bboxes[color] = _random_bbox(rng, size, min_dim, max_dim)
                continue

            placed = False
            for _ in range(PLACEMENT_ATTEMPTS):
                candidate = _random_bbox(rng, size, min_dim, max_dim)
                if _perimeters_overlap(candidate, bboxes[ordering[i - 1]]):
                    bboxes[color] = candidate
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return bboxes
    return None


def _frame_masks_from_bboxes(bboxes, shape):
    masks = {}
    for color, (r0, c0, r1, c1) in bboxes.items():
        mask = np.zeros(shape, dtype=bool)
        mask[r0:r1 + 1, c0] = True
        mask[r0:r1 + 1, c1] = True
        mask[r0, c0:c1 + 1] = True
        mask[r1, c0:c1 + 1] = True
        masks[color] = mask
    return masks


def _front_edges(ordering, frame_masks):
    """Return overlap count and edges implied by back-to-front paint order."""
    edges = set()
    overlapping_pairs = 0
    for a, b in combinations(ordering, 2):
        crossing = frame_masks[a] & frame_masks[b]
        if crossing.any():
            overlapping_pairs += 1
            idx_a = ordering.index(a)
            idx_b = ordering.index(b)
            if idx_a < idx_b:
                edges.add((a, b))
            else:
                edges.add((b, a))
    return overlapping_pairs, edges


def _has_required_overlaps(ordering, overlapping_pairs, edges):
    n = len(ordering)
    total_pairs = n * (n - 1) // 2
    if total_pairs > 0 and overlapping_pairs / total_pairs < MIN_OVERLAP_DENSITY:
        return False
    return all((ordering[i], ordering[i + 1]) in edges for i in range(n - 1))


def _has_unique_order_path(ordering, edges):
    outgoing = {c: [o for o in ordering if (c, o) in edges] for c in ordering}
    paths = []

    def search(path, remaining):
        if len(paths) > 1:
            return
        if not remaining:
            paths.append(tuple(path))
            return
        for nxt in outgoing[path[-1]]:
            if nxt in remaining:
                search(path + [nxt], remaining - {nxt})

    color_set = set(ordering)
    for start in ordering:
        search([start], color_set - {start})
        if len(paths) > 1:
            break
    return len(paths) == 1 and list(paths[0]) == ordering


def _paint_frames(ordering, frame_masks, shape):
    grid = np.zeros(shape, dtype=int)
    for color in ordering:
        grid[frame_masks[color]] = color
    return grid


def _visible_pixels_span_bboxes(grid, ordering, bboxes):
    for color in ordering:
        visible = np.argwhere(grid == color)
        if visible.size == 0:
            return False
        vr0, vc0 = visible.min(axis=0)
        vr1, vc1 = visible.max(axis=0)
        if (vr0, vc0, vr1, vc1) != bboxes[color]:
            return False
    return True


def _passes_oracle(grid, ordering):
    try:
        out = main(grid)
        return np.array_equal(out, _render_nested(ordering))
    except (AssertionError, Exception):
        return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    ordering = _choose_ordering(rng)
    size = _choose_grid_size(rng, len(ordering))

    bboxes = _place_bboxes(rng, ordering, size)
    if bboxes is None:
        return None

    frame_masks = _frame_masks_from_bboxes(bboxes, (size, size))
    overlapping_pairs, edges = _front_edges(ordering, frame_masks)
    if not _has_required_overlaps(ordering, overlapping_pairs, edges):
        return None
    if not _has_unique_order_path(ordering, edges):
        return None

    grid = _paint_frames(ordering, frame_masks, (size, size))
    if not _visible_pixels_span_bboxes(grid, ordering, bboxes):
        return None
    if not _passes_oracle(grid, ordering):
        return None
    return grid


def _perimeters_overlap(bbox1, bbox2):
    """Check if the perimeters of two rectangles share at least one pixel."""
    r0a, c0a, r1a, c1a = bbox1
    r0b, c0b, r1b, c1b = bbox2

    # Quick rejection: no bounding box overlap at all
    if r1a < r0b or r1b < r0a or c1a < c0b or c1b < c0a:
        return False

    # Check horizontal edges of A vs perimeter of B
    for edge_row, edge_c0, edge_c1 in [(r0a, c0a, c1a), (r1a, c0a, c1a)]:
        for b_row in [r0b, r1b]:
            if edge_row == b_row and max(edge_c0, c0b) <= min(edge_c1, c1b):
                return True
        for b_col in [c0b, c1b]:
            if edge_c0 <= b_col <= edge_c1 and r0b <= edge_row <= r1b:
                return True

    # Check vertical edges of A vs perimeter of B
    for edge_col, edge_r0, edge_r1 in [(c0a, r0a, r1a), (c1a, r0a, r1a)]:
        for b_col in [c0b, c1b]:
            if edge_col == b_col and max(edge_r0, r0b) <= min(edge_r1, r1b):
                return True
        for b_row in [r0b, r1b]:
            if edge_r0 <= b_row <= edge_r1 and c0b <= edge_col <= c1b:
                return True

    return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
