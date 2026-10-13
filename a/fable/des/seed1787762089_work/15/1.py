import numpy as np


PALETTE = (1, 2, 3, 4, 6, 7, 8)
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    top, left, bottom, right = box
    mask = np.zeros(shape, dtype=bool)
    mask[top, left : right + 1] = True
    mask[bottom, left : right + 1] = True
    mask[top : bottom + 1, left] = True
    mask[top : bottom + 1, right] = True
    return mask


def _reconstruct_frames(grid: np.ndarray) -> tuple[int, list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    bg = _get_bg(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors, "expected at least one frame color"

    boxes: dict[int, tuple[int, int, int, int]] = {}
    masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = grid == color
        box = _bbox(visible)
        assert box is not None, "each frame color must remain visible"
        mask = _frame_mask(grid.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        boxes[color] = box
        masks[color] = mask
    return bg, colors, boxes, masks


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lower, upper in edges:
        if upper not in outgoing[lower]:
            outgoing[lower].add(upper)
            indegree[upper] += 1

    frontier = [color for color in colors if indegree[color] == 0]
    order: list[int] = []
    while frontier:
        assert len(frontier) == 1, "layer order must be uniquely determined"
        current = frontier.pop()
        order.append(current)
        for nxt in outgoing[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(nxt)

    assert len(order) == len(colors), "layer constraints must be acyclic"
    return order


def _infer_layer_order(grid: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg, colors, boxes, masks = _reconstruct_frames(g)

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            assert np.any(masks[color_a] & masks[color_b]), "every pair of frames must overlap"

    edges: set[tuple[int, int]] = set()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            covering = [color for color in colors if masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top != bg, "overlap cells must show one of the covering frames"
            assert top in covering, "overlap cells must show a covering frame color"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    assert edges, "expected overlap constraints"
    order = _unique_layer_order(colors, edges)
    return order, boxes, masks


def _concentric_rings(order: list[int]) -> np.ndarray:
    n = len(order)
    size = 2 * n - 1
    out = np.full((size, size), int(order[-1]), dtype=int)
    for depth, color in enumerate(order[:-1]):
        start = depth
        end = size - depth - 1
        out[start, start : end + 1] = color
        out[end, start : end + 1] = color
        out[start : end + 1, start] = color
        out[start : end + 1, end] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapped rectangular frames, infer the unique bottom-to-top layer order,
    and encode that order as concentric square rings.
    """
    order, _boxes, _masks = _infer_layer_order(np.asarray(grid, dtype=int))
    return _concentric_rings(order)


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    top, left, bottom, right = box
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _side_visibility_ok(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> bool:
    visible = grid == color
    top, left, bottom, right = box
    return bool(
        visible[top, left : right + 1].any()
        and visible[bottom, left : right + 1].any()
        and visible[top : bottom + 1, left].any()
        and visible[top : bottom + 1, right].any()
    )


def _sample_boxes(rng: np.random.Generator, h: int, w: int, n: int) -> list[tuple[int, int, int, int]] | None:
    min_side = max(5, n)
    top_pool_max = h - min_side - n + 1
    left_pool_max = w - min_side - n + 1
    if top_pool_max < n - 1 or left_pool_max < n - 1:
        return None

    tops = sorted(int(v) for v in rng.choice(np.arange(top_pool_max + 1), size=n, replace=False))
    lefts = sorted((int(v) for v in rng.choice(np.arange(left_pool_max + 1), size=n, replace=False)), reverse=True)

    bottom_min = tops[-1] + min_side - 1
    right_min = lefts[0] + min_side - 1
    if h - bottom_min < n or w - right_min < n:
        return None

    bottoms = sorted((int(v) for v in rng.choice(np.arange(bottom_min, h), size=n, replace=False)), reverse=True)
    rights = sorted(int(v) for v in rng.choice(np.arange(right_min, w), size=n, replace=False))
    return [(tops[i], lefts[i], bottoms[i], rights[i]) for i in range(n)]


def _render_candidate(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    layer_order: list[int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for frame_idx in layer_order:
        _paint_frame(grid, boxes[frame_idx], colors[frame_idx])
    return grid


def _candidate_ok(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    colors: list[int],
    layer_order: list[int],
) -> bool:
    expected_order = [int(colors[idx]) for idx in layer_order]
    try:
        inferred_order, inferred_boxes, _masks = _infer_layer_order(grid)
    except AssertionError:
        return False

    if inferred_order != expected_order:
        return False

    for box, color in zip(boxes, colors):
        visible_box = _bbox(grid == color)
        if visible_box != box:
            return False
        if inferred_boxes[color] != box:
            return False
        if not _side_visibility_ok(grid, color, box):
            return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        n = int(rng.integers(3, 6))
        h = int(rng.integers(14, 21))
        w = int(rng.integers(14, 21))
        boxes = _sample_boxes(rng, h, w, n)
        if boxes is None:
            continue

        colors = [int(color) for color in rng.choice(np.array(PALETTE), size=n, replace=False)]
        layer_order = [int(v) for v in rng.permutation(n)]
        grid = _render_candidate((h, w), boxes, colors, layer_order)
        if _candidate_ok(grid, boxes, colors, layer_order):
            return grid

    raise RuntimeError(f"failed to generate a valid sample after {GENERATION_ATTEMPTS} attempts")
