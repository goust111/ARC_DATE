import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator bounds.
OBJECT_COUNT_RANGE = (3, 6)
SHAPE_SIZE_RANGE = (4, 6)
MAX_SHAPE_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 500
BRANCH_MODES = ("dense", "mixed")


def _normalize(coords: np.ndarray | list[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    if arr.size == 0:
        return arr.reshape(0, 2)
    arr = arr - arr.min(axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _coords_to_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.asarray(coords, dtype=int)}


def _shape_extent(shape: np.ndarray) -> tuple[int, int]:
    shape = np.asarray(shape, dtype=int)
    return int(shape[:, 0].max()) + 1, int(shape[:, 1].max()) + 1


def _is_connected(coords: np.ndarray) -> bool:
    cells = _coords_to_set(coords)
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _render_scene(
    order_high_to_low: list[int],
    positions: dict[int, tuple[int, int]],
    shape: np.ndarray,
    scene_shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(scene_shape, dtype=int)
    for color in reversed(order_high_to_low):
        r0, c0 = positions[color]
        rr = shape[:, 0] + int(r0)
        cc = shape[:, 1] + int(c0)
        out[rr, cc] = int(color)
    return out


def _extract_priority(grid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2 and g.shape[1] >= 3, "grid is too small"

    scene = g[:-1]
    legend = [int(v) for v in g[-1].tolist() if int(v) != BG]
    assert 3 <= len(legend) <= 6, "expected 3-6 legend colors"
    assert len(legend) == len(set(legend)), "legend colors must be unique"

    scene_colors = sorted(set(map(int, np.unique(scene))) - {BG})
    assert scene_colors == sorted(legend), "legend must list exactly the object colors"
    return scene, legend


def _all_top_lefts(scene_shape: tuple[int, int], shape: np.ndarray) -> list[tuple[int, int]]:
    scene_h, scene_w = scene_shape
    obj_h, obj_w = _shape_extent(shape)
    return [
        (r, c)
        for r in range(scene_h - obj_h + 1)
        for c in range(scene_w - obj_w + 1)
    ]


def _visible_signature(coords: np.ndarray, width: int) -> np.ndarray:
    return np.sort(coords[:, 0] * width + coords[:, 1])


def _recover_positions(scene: np.ndarray, order_high_to_low: list[int], shape: np.ndarray) -> dict[int, tuple[int, int]]:
    scene = np.asarray(scene, dtype=int)
    shape = _normalize(shape)
    assert _is_connected(shape), "prototype shape must be connected"

    height, width = scene.shape
    top_lefts = _all_top_lefts(scene.shape, shape)
    occupied = np.zeros_like(scene, dtype=bool)
    positions: dict[int, tuple[int, int]] = {}

    for color in order_high_to_low:
        visible = np.argwhere(scene == int(color))
        assert visible.size > 0, "every object color must stay at least partly visible"
        target = _visible_signature(visible, width)
        candidates = []
        for r0, c0 in top_lefts:
            rr = shape[:, 0] + int(r0)
            cc = shape[:, 1] + int(c0)
            hidden = occupied[rr, cc]
            if hidden.all():
                continue
            candidate_visible = np.column_stack((rr[~hidden], cc[~hidden]))
            if candidate_visible.shape[0] != visible.shape[0]:
                continue
            if np.array_equal(_visible_signature(candidate_visible, width), target):
                candidates.append((int(r0), int(c0)))

        assert len(candidates) == 1, f"expected a unique placement for color {color}"
        r0, c0 = candidates[0]
        positions[int(color)] = (r0, c0)
        rr = shape[:, 0] + r0
        cc = shape[:, 1] + c0
        occupied[rr, cc] = True

    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the bottom-row color priority, recover the full translated copies of one shared shape,
    and re-render the overlaps under the reversed priority chain.
    """
    g = np.asarray(grid, dtype=int)
    scene, order_high_to_low = _extract_priority(g)

    highest = order_high_to_low[0]
    prototype_visible = np.argwhere(scene == highest)
    assert prototype_visible.size > 0, "highest-priority object is missing"
    assert _is_connected(prototype_visible), "highest-priority object must be a single connected shape"
    prototype = _normalize(prototype_visible)

    positions = _recover_positions(scene, order_high_to_low, prototype)
    rebuilt_input = _render_scene(order_high_to_low, positions, prototype, scene.shape)
    assert np.array_equal(rebuilt_input, scene), "scene must match the recovered layout"

    out = np.zeros_like(g)
    reversed_order = list(reversed(order_high_to_low))
    out[:-1] = _render_scene(reversed_order, positions, prototype, scene.shape)
    return out


def _transform_shape(shape: np.ndarray, rotation: int, mirror: bool) -> np.ndarray:
    pts = np.asarray(shape, dtype=int)
    if mirror:
        pts = pts.copy()
        pts[:, 1] *= -1
    for _ in range(rotation % 4):
        pts = np.column_stack((-pts[:, 1], pts[:, 0]))
    return _normalize(pts)


def _grow_polyomino(size: int) -> np.ndarray:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < size:
        nxt = random.choice(tuple(frontier))
        frontier.remove(nxt)
        cells.add(nxt)
        r, c = nxt
        for dr, dc in DIR4:
            nb = (r + dr, c + dc)
            if nb not in cells:
                frontier.add(nb)
    return _normalize(list(cells))


def _interesting_shape(shape: np.ndarray) -> bool:
    shape = _normalize(shape)
    h, w = _shape_extent(shape)
    if h == 1 or w == 1:
        return False
    if shape.shape[0] == h * w:
        return False
    row_counts = np.bincount(shape[:, 0], minlength=h)
    col_counts = np.bincount(shape[:, 1], minlength=w)
    if len(set(map(int, row_counts))) == 1 and len(set(map(int, col_counts))) == 1:
        return False
    return _is_connected(shape)


def _sample_shape() -> np.ndarray:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        size = random.randint(*SHAPE_SIZE_RANGE)
        shape = _grow_polyomino(size)
        shape = _transform_shape(shape, random.randrange(4), bool(random.randrange(2)))
        if _interesting_shape(shape):
            return shape
    raise RuntimeError("failed to sample a non-degenerate shape")


def _sample_scene_shape(shape: np.ndarray, object_count: int) -> tuple[int, int]:
    obj_h, obj_w = _shape_extent(shape)
    for _ in range(80):
        scene_h = random.randint(obj_h + 3, obj_h + 7)
        scene_w = random.randint(max(obj_w + 4, object_count + 1), obj_w + object_count + 7)
        top_left_count = (scene_h - obj_h + 1) * (scene_w - obj_w + 1)
        if top_left_count >= object_count + 4:
            return scene_h, scene_w
    raise RuntimeError("failed to sample a scene size")


def _sample_positions(
    shape: np.ndarray,
    scene_shape: tuple[int, int],
    object_count: int,
    branch_mode: str,
) -> list[tuple[int, int]] | None:
    top_lefts = _all_top_lefts(scene_shape, shape)
    if len(top_lefts) < object_count:
        return None

    if branch_mode == "dense":
        needs_overlap = [False] + [True] * (object_count - 1)
    else:
        overlap_count = random.randint(1, object_count - 2)
        overlap_indices = set(random.sample(range(1, object_count), overlap_count))
        needs_overlap = [i in overlap_indices for i in range(object_count)]

    positions: list[tuple[int, int]] = []
    occupied = np.zeros(scene_shape, dtype=bool)
    min_visible = 2

    for idx in range(object_count):
        random.shuffle(top_lefts)
        overlap_pool: list[tuple[int, int]] = []
        separate_pool: list[tuple[int, int]] = []
        for r0, c0 in top_lefts:
            if (r0, c0) in positions:
                continue
            rr = shape[:, 0] + int(r0)
            cc = shape[:, 1] + int(c0)
            overlap = int(occupied[rr, cc].sum())
            visible = shape.shape[0] - overlap
            if visible < min_visible:
                continue
            if overlap == 0:
                separate_pool.append((int(r0), int(c0)))
            else:
                overlap_pool.append((int(r0), int(c0)))

        if idx == 0:
            pool = separate_pool
        elif needs_overlap[idx]:
            pool = overlap_pool
        else:
            pool = separate_pool

        if not pool:
            return None

        pos = random.choice(pool)
        positions.append(pos)
        rr = shape[:, 0] + pos[0]
        cc = shape[:, 1] + pos[1]
        occupied[rr, cc] = True

    return positions


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    shape = _sample_shape()
    scene_shape = _sample_scene_shape(shape, object_count)
    branch_mode = random.choice(BRANCH_MODES)
    positions = _sample_positions(shape, scene_shape, object_count, branch_mode)
    if positions is None:
        raise RuntimeError("failed to place objects")

    priority = random.sample(PALETTE, object_count)
    position_map = {color: pos for color, pos in zip(priority, positions, strict=False)}
    scene = _render_scene(priority, position_map, shape, scene_shape)
    visible_colors = sorted(set(map(int, np.unique(scene))) - {BG})
    assert visible_colors == sorted(priority), "all object colors must remain visible"

    overlap_pairs = 0
    for i, color_a in enumerate(priority):
        ra, ca = position_map[color_a]
        a = _coords_to_set(shape + np.array([ra, ca], dtype=int))
        for color_b in priority[i + 1 :]:
            rb, cb = position_map[color_b]
            b = _coords_to_set(shape + np.array([rb, cb], dtype=int))
            if a & b:
                overlap_pairs += 1
    assert overlap_pairs >= 1, "need at least one overlap"
    if branch_mode == "mixed":
        assert overlap_pairs < object_count - 1 or object_count == 3

    width = scene_shape[1]
    legend_start = random.randint(0, width - object_count)
    grid = np.zeros((scene_shape[0] + 1, width), dtype=int)
    grid[:-1] = scene
    grid[-1, legend_start : legend_start + object_count] = priority

    expected = np.zeros_like(grid)
    expected[:-1] = _render_scene(list(reversed(priority)), position_map, shape, scene_shape)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, expected = _build_sample()
        except Exception:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
