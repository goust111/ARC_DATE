from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("_arc_utils_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _ARC_UTILS = _load_arc_utils()
    Color = _ARC_UTILS.Color
    bbox = _ARC_UTILS.bbox
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


BBox = tuple[int, int, int, int]

_BG = int(Color.BLACK)
_COUNTER = int(Color.GREY)
_FRAME_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)

_GEN_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400
_GEN_FRAME_COUNT_RANGE = (3, 5)
_GEN_SIZE_MAX = 30
_GEN_RESERVED_MARGIN = 3
_GEN_CORNERS = ("tl", "tr", "bl", "br")


def _frame_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _visible_frame_boxes(grid: np.ndarray) -> dict[int, BBox]:
    boxes: dict[int, BBox] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {_BG, _COUNTER}):
        box = bbox(grid == color)
        assert box is not None, f"missing visible pixels for frame color {color}"
        boxes[int(color)] = box
    assert boxes, "expected at least one rectangle frame"
    return boxes


def _unique_topological_order(nodes: list[int], edges: dict[int, set[int]]) -> list[int]:
    indegree = {node: 0 for node in nodes}
    for src in nodes:
        for dst in edges[src]:
            indegree[dst] += 1

    order: list[int] = []
    ready = sorted(node for node in nodes if indegree[node] == 0)
    while ready:
        assert len(ready) == 1, "crossing constraints must determine a unique layer order"
        node = ready.pop()
        order.append(node)
        for dst in sorted(edges[node]):
            indegree[dst] -= 1
            if indegree[dst] == 0:
                ready.append(dst)
        ready.sort()

    assert len(order) == len(nodes), "layer graph must be acyclic"
    return order


def _infer_order(grid: np.ndarray, boxes: dict[int, BBox]) -> list[int]:
    colors = sorted(boxes)
    masks = {color: _frame_mask(grid.shape, boxes[color]) for color in colors}
    edges = {color: set() for color in colors}

    for idx, a in enumerate(colors):
        for b in colors[idx + 1 :]:
            overlap = masks[a] & masks[b]
            if not overlap.any():
                continue
            visible = set(map(int, np.unique(grid[overlap])))
            visible.discard(_BG)
            visible.discard(_COUNTER)
            assert visible and visible <= {a, b}, "crossings must expose exactly one of the two frame colors"
            assert len(visible) == 1, "each pairwise crossing must agree on the same visible winner"
            winner = visible.pop()
            loser = b if winner == a else a
            edges[winner].add(loser)

    return _unique_topological_order(colors, edges)


def _render_layers(shape: tuple[int, int], boxes: dict[int, BBox], order_top_to_bottom: list[int]) -> np.ndarray:
    out = np.full(shape, _BG, dtype=int)
    for color in reversed(order_top_to_bottom):
        out[_frame_mask(shape, boxes[color])] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Peel away the top k overlapping rectangle frames and redraw the surviving lower layers in full.

    Input structure:
    - A black background contains several distinct-color, 1-cell-thick axis-aligned rectangle frames.
    - The frames overlap only through perimeter crossings, and the visible color at each crossing records which frame is on top.
    - A separate gray corner cluster gives the peel depth k by its pixel count.

    Rule:
    - For each non-black, non-gray color, recover that frame's full rectangle from the bbox of its visible pixels.
    - Use every pairwise crossing visibility to infer the unique top-to-bottom layer order of the frames.
    - Delete the top k frames and all gray pixels, then redraw the remaining frames in their original order so pixels formerly hidden by removed layers reappear.

    Key insight:
    - The gray pixels are a counter, not clutter: the task is to reconstruct the full hidden stack of frames, remove the top k layers, and re-render what lies underneath.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    boxes = _visible_frame_boxes(g)
    order_top_to_bottom = _infer_order(g, boxes)

    k = int(np.count_nonzero(g == _COUNTER))
    if k >= len(order_top_to_bottom):
        return np.full(g.shape, _BG, dtype=int)

    remaining_top_to_bottom = order_top_to_bottom[k:]
    return _render_layers(g.shape, boxes, remaining_top_to_bottom)


def _connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    seen = {next(iter(cells))}
    stack = [next(iter(seen))]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == cells


def _corner_cluster(shape: tuple[int, int], corner: str, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    h, w = shape
    if corner == "tl":
        rows = range(0, 3)
        cols = range(0, 3)
        start = (0, 0)
    elif corner == "tr":
        rows = range(0, 3)
        cols = range(w - 3, w)
        start = (0, w - 1)
    elif corner == "bl":
        rows = range(h - 3, h)
        cols = range(0, 3)
        start = (h - 1, 0)
    else:
        rows = range(h - 3, h)
        cols = range(w - 3, w)
        start = (h - 1, w - 1)

    allowed = {(r, c) for r in rows for c in cols}
    cells = {start}
    while len(cells) < count:
        frontier = []
        for r, c in sorted(cells):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if nxt in allowed and nxt not in cells:
                    frontier.append(nxt)
        assert frontier, "corner region must be large enough for the counter cluster"
        choice = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(choice)
    return sorted(cells)


def _sample_sorted(pool: np.ndarray, count: int, rng: np.random.Generator) -> list[int]:
    chosen = rng.choice(pool, size=count, replace=False)
    return sorted(int(v) for v in np.asarray(chosen, dtype=int))


def _sample_boxes(frame_count: int, rng: np.random.Generator) -> tuple[tuple[int, int], list[BBox]] | None:
    min_size = max(18, 2 * frame_count + 10)
    height = int(rng.integers(min_size, _GEN_SIZE_MAX + 1))
    width = int(rng.integers(min_size, _GEN_SIZE_MAX + 1))

    top_pool = np.arange(_GEN_RESERVED_MARGIN, height // 2 - 1, dtype=int)
    bottom_pool = np.arange(height // 2 + 1, height - _GEN_RESERVED_MARGIN, dtype=int)
    left_pool = np.arange(_GEN_RESERVED_MARGIN, width // 2 - 1, dtype=int)
    right_pool = np.arange(width // 2 + 1, width - _GEN_RESERVED_MARGIN, dtype=int)
    if min(len(top_pool), len(bottom_pool), len(left_pool), len(right_pool)) < frame_count:
        return None

    tops = _sample_sorted(top_pool, frame_count, rng)
    bottoms = _sample_sorted(bottom_pool, frame_count, rng)
    lefts = _sample_sorted(left_pool, frame_count, rng)
    rights = _sample_sorted(right_pool, frame_count, rng)

    boxes = [(tops[i], lefts[i], bottoms[i], rights[i]) for i in range(frame_count)]
    return (height, width), boxes


def _generate_case(rng: np.random.Generator):
    frame_count = int(rng.integers(_GEN_FRAME_COUNT_RANGE[0], _GEN_FRAME_COUNT_RANGE[1] + 1))
    sampled = _sample_boxes(frame_count, rng)
    assert sampled is not None
    shape, geometry_boxes = sampled

    colors = list(map(int, rng.choice(np.array(_FRAME_COLORS, dtype=int), size=frame_count, replace=False)))
    rng.shuffle(colors)
    shuffled_boxes = list(geometry_boxes)
    rng.shuffle(shuffled_boxes)
    boxes = {color: box for color, box in zip(colors, shuffled_boxes)}

    order_top_to_bottom = colors.copy()
    rng.shuffle(order_top_to_bottom)
    k = int(rng.integers(1, frame_count))
    corner = _GEN_CORNERS[int(rng.integers(0, len(_GEN_CORNERS)))]
    gray_cells = _corner_cluster(shape, corner, k, rng)

    inp = _render_layers(shape, boxes, order_top_to_bottom)
    for r, c in gray_cells:
        inp[r, c] = _COUNTER
    expected = _render_layers(shape, boxes, order_top_to_bottom[k:])
    return inp, expected, boxes, order_top_to_bottom, gray_cells, k


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    boxes: dict[int, BBox],
    order_top_to_bottom: list[int],
    gray_cells: list[tuple[int, int]],
    k: int,
) -> None:
    recovered_boxes = _visible_frame_boxes(inp)
    assert recovered_boxes == boxes, "visible bboxes must recover the original full frames"

    recovered_order = _infer_order(inp, recovered_boxes)
    assert recovered_order == order_top_to_bottom, "crossings must determine the latent layer order"

    masks = {color: _frame_mask(inp.shape, box) for color, box in boxes.items()}
    coverage = np.zeros(inp.shape, dtype=int)
    for mask in masks.values():
        coverage += mask.astype(int)
    assert int(coverage.max()) <= 2, "generation should avoid triple-overlap crossings"

    for idx, upper in enumerate(order_top_to_bottom):
        for lower in order_top_to_bottom[idx + 1 :]:
            overlap = masks[upper] & masks[lower]
            assert overlap.any(), "every pair of frames should cross"
            seen = set(map(int, np.unique(inp[overlap])))
            seen.discard(_BG)
            seen.discard(_COUNTER)
            assert seen == {upper}, "crossing visibility must match the latent top frame"

    gray_set = set(gray_cells)
    assert len(gray_set) == k and _connected_cells(gray_set), "gray counter pixels must form one connected corner cluster"
    for cell in gray_set:
        assert all(not mask[cell] for mask in masks.values()), "gray counter cluster must stay separate from all frames"

    naive = inp.copy()
    naive[naive == _COUNTER] = _BG
    for color in order_top_to_bottom[:k]:
        naive[naive == color] = _BG
    assert not np.array_equal(naive, expected), "restoration must reveal pixels that naive erasure would miss"

    if k > 0:
        less_peeled = _render_layers(inp.shape, boxes, order_top_to_bottom[k - 1 :])
        assert not np.array_equal(less_peeled, expected), "the gray count must change which layer survives"

    assert np.array_equal(main(inp), expected), "solver must match the latent scene semantics"


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        inp, expected, boxes, order_top_to_bottom, gray_cells, k = _generate_case(_GEN_RNG)
        try:
            _semantic_checks(inp, expected, boxes, order_top_to_bottom, gray_cells, k)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid peel-the-surface ARC input")
