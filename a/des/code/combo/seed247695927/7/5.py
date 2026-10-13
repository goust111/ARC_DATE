from __future__ import annotations

from itertools import combinations

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    banned = set() if exclude is None else set(exclude)
    return [int(c) for c in np.unique(np.asarray(grid, dtype=int)) if int(c) not in banned]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _color_frames(grid: np.ndarray, bg: int) -> tuple[list[int], dict[int, np.ndarray]]:
    colors = colors_present(grid, exclude={bg})
    assert len(colors) >= 2, "expected multiple foreground frame colors"

    frames: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 > r0 and c1 > c0, "each color must span a rectangular frame"
        frame = _frame_mask(grid.shape, bbox)
        assert np.all(frame[grid == color]), "visible pixels must lie on the inferred frame perimeter"
        frames[int(color)] = frame
    return colors, frames


def _crossing_edges(grid: np.ndarray, colors: list[int], frames: dict[int, np.ndarray]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b in combinations(colors, 2):
        crossing = frames[a] & frames[b]
        for r_raw, c_raw in np.argwhere(crossing):
            r, c = int(r_raw), int(c_raw)
            visible = int(grid[r, c])
            if visible == a:
                edges.add((b, a))
            elif visible == b:
                edges.add((a, b))
            else:
                assert visible in colors, "crossings must be covered by a foreground color"
    assert edges, "expected at least one visible crossing"
    return edges


def _unique_path(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {color: [other for other in colors if (color, other) in edges] for color in colors}
    paths: list[tuple[int, ...]] = []

    def search(path: tuple[int, ...], remaining: set[int]) -> None:
        if len(paths) > 1:
            return
        if not remaining:
            paths.append(path)
            return
        for nxt in outgoing[path[-1]]:
            if nxt in remaining:
                search((*path, nxt), remaining - {nxt})

    color_set = set(colors)
    for start in colors:
        search((start,), color_set - {start})
        if len(paths) > 1:
            break

    assert len(paths) == 1, "crossings must determine a unique order"
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
    Infer the crossing order of overlapping colored rectangular frames and
    redraw them as concentric square frames from outermost to innermost.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = get_bg(g)
    colors, frames = _color_frames(g, bg)
    edges = _crossing_edges(g, colors, frames)
    order = _unique_path(colors, edges)
    return _render_nested(order)


# --------------- generator ---------------

MIN_FRAME_COUNT = 5
MAX_FRAME_COUNT_EXCLUSIVE = 8
AVAILABLE_COLORS = list(range(1, 10))

MIN_GRID_SIZE = 12
MAX_GRID_SIZE_EXCLUSIVE = 26
GRID_SIZE_PER_FRAME = 2
GRID_SIZE_PADDING = 2
MIN_FRAME_DIM = 3
MAX_FRAME_DIM = 13

MIN_OVERLAP_DENSITY = 0.57
LAYOUT_ATTEMPTS = 50
PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 300


def _choose_ordering(rng: np.random.Generator) -> list[int]:
    n = int(rng.integers(MIN_FRAME_COUNT, MAX_FRAME_COUNT_EXCLUSIVE))
    colors = list(AVAILABLE_COLORS)
    rng.shuffle(colors)
    return list(colors[:n])


def _choose_grid_size(rng: np.random.Generator, n: int) -> int:
    min_size = max(MIN_GRID_SIZE, n * GRID_SIZE_PER_FRAME + GRID_SIZE_PADDING)
    return int(rng.integers(min_size, MAX_GRID_SIZE_EXCLUSIVE))


def _random_bbox(
    rng: np.random.Generator, size: int, min_dim: int, max_dim: int
) -> tuple[int, int, int, int]:
    h = int(rng.integers(min_dim, max_dim + 1))
    w = int(rng.integers(min_dim, max_dim + 1))
    r0 = int(rng.integers(0, size - h + 1))
    c0 = int(rng.integers(0, size - w + 1))
    return r0, c0, r0 + h - 1, c0 + w - 1


def _perimeters_overlap(
    bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]
) -> bool:
    r0a, c0a, r1a, c1a = bbox1
    r0b, c0b, r1b, c1b = bbox2

    if r1a < r0b or r1b < r0a or c1a < c0b or c1b < c0a:
        return False

    for edge_row, edge_c0, edge_c1 in ((r0a, c0a, c1a), (r1a, c0a, c1a)):
        for b_row in (r0b, r1b):
            if edge_row == b_row and max(edge_c0, c0b) <= min(edge_c1, c1b):
                return True
        for b_col in (c0b, c1b):
            if edge_c0 <= b_col <= edge_c1 and r0b <= edge_row <= r1b:
                return True

    for edge_col, edge_r0, edge_r1 in ((c0a, r0a, r1a), (c1a, r0a, r1a)):
        for b_col in (c0b, c1b):
            if edge_col == b_col and max(edge_r0, r0b) <= min(edge_r1, r1b):
                return True
        for b_row in (r0b, r1b):
            if edge_r0 <= b_row <= edge_r1 and c0b <= edge_col <= c1b:
                return True

    return False


def _place_bboxes(
    rng: np.random.Generator, ordering: list[int], size: int
) -> dict[int, tuple[int, int, int, int]] | None:
    min_dim = MIN_FRAME_DIM
    max_dim = min(MAX_FRAME_DIM, size - 1)

    for _ in range(LAYOUT_ATTEMPTS):
        bboxes: dict[int, tuple[int, int, int, int]] = {}
        ok = True
        for i, color in enumerate(ordering):
            if i == 0:
                bboxes[color] = _random_bbox(rng, size, min_dim, max_dim)
                continue
            placed = False
            prev = bboxes[ordering[i - 1]]
            for _ in range(PLACEMENT_ATTEMPTS):
                candidate = _random_bbox(rng, size, min_dim, max_dim)
                if _perimeters_overlap(candidate, prev):
                    bboxes[color] = candidate
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return bboxes
    return None


def _frame_masks_from_bboxes(
    bboxes: dict[int, tuple[int, int, int, int]], shape: tuple[int, int]
) -> dict[int, np.ndarray]:
    return {color: _frame_mask(shape, bbox) for color, bbox in bboxes.items()}


def _front_edges(ordering: list[int], frame_masks: dict[int, np.ndarray]) -> tuple[int, set[tuple[int, int]]]:
    edges: set[tuple[int, int]] = set()
    overlapping_pairs = 0
    positions = {color: i for i, color in enumerate(ordering)}
    for a, b in combinations(ordering, 2):
        if (frame_masks[a] & frame_masks[b]).any():
            overlapping_pairs += 1
            if positions[a] < positions[b]:
                edges.add((a, b))
            else:
                edges.add((b, a))
    return overlapping_pairs, edges


def _has_required_overlaps(ordering: list[int], overlapping_pairs: int, edges: set[tuple[int, int]]) -> bool:
    n = len(ordering)
    total_pairs = n * (n - 1) // 2
    if total_pairs > 0 and overlapping_pairs / total_pairs < MIN_OVERLAP_DENSITY:
        return False
    return all((ordering[i], ordering[i + 1]) in edges for i in range(n - 1))


def _has_unique_order_path(ordering: list[int], edges: set[tuple[int, int]]) -> bool:
    outgoing = {color: [other for other in ordering if (color, other) in edges] for color in ordering}
    paths: list[tuple[int, ...]] = []

    def search(path: tuple[int, ...], remaining: set[int]) -> None:
        if len(paths) > 1:
            return
        if not remaining:
            paths.append(path)
            return
        for nxt in outgoing[path[-1]]:
            if nxt in remaining:
                search((*path, nxt), remaining - {nxt})

    color_set = set(ordering)
    for start in ordering:
        search((start,), color_set - {start})
        if len(paths) > 1:
            break
    return len(paths) == 1 and list(paths[0]) == ordering


def _paint_frames(
    ordering: list[int], frame_masks: dict[int, np.ndarray], shape: tuple[int, int]
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in ordering:
        grid[frame_masks[color]] = color
    return grid


def _visible_pixels_span_bboxes(
    grid: np.ndarray, ordering: list[int], bboxes: dict[int, tuple[int, int, int, int]]
) -> bool:
    for color in ordering:
        visible = np.argwhere(grid == color)
        if visible.size == 0:
            return False
        if bbox_coords(visible) != bboxes[color]:
            return False
    return True


def _passes_oracle(grid: np.ndarray, ordering: list[int]) -> bool:
    try:
        return np.array_equal(main(grid), _render_nested(ordering))
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
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


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
