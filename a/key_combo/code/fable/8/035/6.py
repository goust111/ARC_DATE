from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_8_035_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
Color = _ARC_UTILS.Color
bbox = _ARC_UTILS.bbox
colors_present = _ARC_UTILS.colors_present
get_bg = _ARC_UTILS.get_bg


BBox = tuple[int, int, int, int]


def _bbox_of_color(grid: np.ndarray, color: int) -> BBox:
    box = bbox(grid == int(color))
    assert box is not None, f"missing color {color}"
    return tuple(map(int, box))


def _coords_in_bbox(coords: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = map(int, box)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((r0 <= rows) & (rows <= r1) & (c0 <= cols) & (cols <= c1)))


def _infer_rectangles(grid: np.ndarray) -> dict[int, BBox]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    rects: dict[int, BBox] = {}
    for color in colors_present(g, exclude={bg}):
        rect = _bbox_of_color(g, color)
        r0, c0, r1, c1 = rect
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg), "rectangle extent must be fully covered by itself or higher layers"
        rects[int(color)] = rect
    assert rects, "expected at least one foreground rectangle"
    return rects


def _direct_edges(grid: np.ndarray, rects: dict[int, BBox]) -> dict[int, set[int]]:
    g = np.asarray(grid, dtype=int)
    edges = {int(color): set() for color in rects}
    for top in rects:
        coords = np.argwhere(g == int(top))
        assert coords.size > 0
        for bottom in rects:
            if top == bottom:
                continue
            if _coords_in_bbox(coords, rects[bottom]):
                edges[top].add(int(bottom))
    return edges


def _transitive_closure(edges: dict[int, set[int]]) -> dict[int, set[int]]:
    closure: dict[int, set[int]] = {int(color): set() for color in edges}
    for src in edges:
        stack = list(edges[src])
        seen = set()
        while stack:
            node = int(stack.pop())
            if node in seen:
                continue
            seen.add(node)
            stack.extend(edges[node] - seen)
        assert src not in seen, "occlusion graph must be acyclic"
        closure[src] = seen
    return closure


def _unique_total_order(closure: dict[int, set[int]]) -> list[int]:
    colors = sorted(closure)
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            ab = b in closure[a]
            ba = a in closure[b]
            assert ab != ba, "transitive closure must induce a unique total order"
    return sorted(colors, key=lambda color: (-len(closure[color]), int(color)))


def _render_from_rectangles(shape: tuple[int, int], rects: dict[int, BBox], order: list[int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for color in reversed(order):
        r0, c0, r1, c1 = rects[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the rectangles' z-order from occlusion evidence and output the colors from top to bottom.

    Input structure:
    - A black background contains several distinct-colored solid axis-aligned rectangles.
    - Rectangles may partially overlap; in every overlapped cell only the top rectangle's color is visible.
    - Each rectangle leaves enough visible boundary evidence that its full bounding box is uniquely recoverable.

    Rule:
    - Infer each rectangle's full extent from the bounding box of its visible color.
    - Whenever visible pixels of one color lie inside another rectangle's inferred extent, the visible color's rectangle is above the other one.
    - Take the transitive closure of these occlusion relations to obtain the unique total z-order, then output an n by 1 color strip listing colors from topmost to bottommost.

    Key insight:
    - The sorting key is not geometric size or position; it is the hidden stacking order reconstructed from who visibly covers whose inferred rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    rects = _infer_rectangles(g)
    edges = _direct_edges(g, rects)
    closure = _transitive_closure(edges)
    order = _unique_total_order(closure)

    replay = _render_from_rectangles(g.shape, rects, order, bg)
    assert np.array_equal(replay, g), "recovered rectangles and order must exactly replay the scene"
    return np.array(order, dtype=int).reshape(-1, 1)


_GEN_BG = Color.BLACK
_GEN_COLORS = tuple(Color.NOT_BLACK)
_GEN_RECT_COUNTS = (3, 3, 3, 4, 4, 5, 5)
_GEN_STEP = 6
_GEN_LATTICE = 3
_GEN_SIDE_RANGE = (7, 8)
_GEN_PAD = (1, 2)
_GEN_MAX_ATTEMPTS = 2000
_GEN_PRIMARY_SHIFT = (5, 7)
_GEN_ORTHO_SHIFT = (2, 3)


def _adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _path_has_turn(path: list[tuple[int, int]]) -> bool:
    dirs = []
    for i in range(len(path) - 1):
        dr = int(path[i + 1][0]) - int(path[i][0])
        dc = int(path[i + 1][1]) - int(path[i][1])
        dirs.append((dr, dc))
    return len(set(dirs)) > 1


def _generate_path(length: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(_GEN_LATTICE) for c in range(_GEN_LATTICE)]
    for _ in range(400):
        path = [random.choice(cells)]
        used = {path[0]}
        while len(path) < length:
            cur = path[-1]
            candidates = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nxt[0] < _GEN_LATTICE and 0 <= nxt[1] < _GEN_LATTICE):
                    continue
                if nxt in used:
                    continue
                if any(_adjacent(nxt, cell) for cell in path[:-1]):
                    continue
                candidates.append(nxt)
            if not candidates:
                break
            nxt = random.choice(candidates)
            path.append(nxt)
            used.add(nxt)
        if len(path) == length and _path_has_turn(path):
            return path
    return None


def _boxes_from_path(path: list[tuple[int, int]]) -> list[BBox]:
    sizes = [(random.randint(*_GEN_SIDE_RANGE), random.randint(*_GEN_SIDE_RANGE)) for _ in path]
    positions = [(0, 0)]
    for i in range(1, len(path)):
        prev_r0, prev_c0 = positions[-1]
        dr = int(path[i][0]) - int(path[i - 1][0])
        dc = int(path[i][1]) - int(path[i - 1][1])
        primary = random.randint(*_GEN_PRIMARY_SHIFT)
        ortho = random.choice((-1, 1)) * random.randint(*_GEN_ORTHO_SHIFT)
        if dc != 0:
            positions.append((int(prev_r0 + ortho), int(prev_c0 + dc * primary)))
        else:
            positions.append((int(prev_r0 + dr * primary), int(prev_c0 + ortho)))

    boxes: list[BBox] = []
    for (r0, c0), (h, w) in zip(positions, sizes):
        boxes.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))
    return boxes


def _crop_and_shift(boxes: list[BBox]) -> tuple[list[BBox], tuple[int, int]]:
    r0 = min(box[0] for box in boxes)
    c0 = min(box[1] for box in boxes)
    r1 = max(box[2] for box in boxes)
    c1 = max(box[3] for box in boxes)
    pad_top = random.randint(*_GEN_PAD)
    pad_left = random.randint(*_GEN_PAD)
    pad_bottom = random.randint(*_GEN_PAD)
    pad_right = random.randint(*_GEN_PAD)
    shifted = []
    for br0, bc0, br1, bc1 in boxes:
        shifted.append(
            (
                int(br0 - r0 + pad_top),
                int(bc0 - c0 + pad_left),
                int(br1 - r0 + pad_top),
                int(bc1 - c0 + pad_left),
            )
        )
    shape = (int(r1 - r0 + 1 + pad_top + pad_bottom), int(c1 - c0 + 1 + pad_left + pad_right))
    assert max(shape) <= 30
    return shifted, shape


def _flip_boxes_h(boxes: list[BBox], width: int) -> list[BBox]:
    return [(r0, width - 1 - c1, r1, width - 1 - c0) for r0, c0, r1, c1 in boxes]


def _flip_boxes_v(boxes: list[BBox], height: int) -> list[BBox]:
    return [(height - 1 - r1, c0, height - 1 - r0, c1) for r0, c0, r1, c1 in boxes]


def _maybe_transform_boxes(boxes: list[BBox], shape: tuple[int, int]) -> tuple[list[BBox], tuple[int, int]]:
    height, width = map(int, shape)
    if random.random() < 0.5:
        boxes = _flip_boxes_h(boxes, width)
    if random.random() < 0.5:
        boxes = _flip_boxes_v(boxes, height)
    return boxes, shape


def _render_truth(shape: tuple[int, int], order: list[int], rects: dict[int, BBox]) -> np.ndarray:
    return _render_from_rectangles(shape, rects, order, _GEN_BG)


def _visible_bbox_matches(grid: np.ndarray, rects: dict[int, BBox], color: int) -> bool:
    box = bbox(grid == int(color))
    return box is not None and tuple(map(int, box)) == tuple(map(int, rects[int(color)]))


def _truth_edges(order: list[int]) -> set[tuple[int, int]]:
    return {(int(order[i]), int(order[i + 1])) for i in range(len(order) - 1)}


def _heuristics_are_misleading(rects: dict[int, BBox], order: list[int]) -> bool:
    area_seq = sorted(order, key=lambda color: ((rects[color][2] - rects[color][0] + 1) * (rects[color][3] - rects[color][1] + 1), color))
    row_seq = sorted(order, key=lambda color: (rects[color][0], rects[color][1], color))
    col_seq = sorted(order, key=lambda color: (rects[color][1], rects[color][0], color))
    rev = list(reversed(order))
    area_bad = area_seq != order and area_seq != rev
    row_bad = row_seq != order and row_seq != rev
    col_bad = col_seq != order and col_seq != rev
    return area_bad and (row_bad or col_bad)


def _generator_valid(grid: np.ndarray, rects: dict[int, BBox], truth_order: list[int]) -> bool:
    if not all(_visible_bbox_matches(grid, rects, color) for color in truth_order):
        return False

    for color, rect in rects.items():
        r0, c0, r1, c1 = rect
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == _GEN_BG):
            return False
        side_mask = grid == int(color)
        rows = np.any(side_mask, axis=1)
        cols = np.any(side_mask, axis=0)
        if not (rows[r0] and rows[r1] and cols[c0] and cols[c1]):
            return False

    edges = _direct_edges(grid, rects)
    direct = {(src, dst) for src, dsts in edges.items() for dst in dsts}
    if not _truth_edges(truth_order).issubset(direct):
        return False
    if len(direct) >= len(truth_order) * (len(truth_order) - 1) // 2:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    if list(map(int, out[:, 0])) != list(map(int, truth_order)):
        return False

    if not _heuristics_are_misleading(rects, truth_order):
        return False
    return True


def _generate_attempt(count: int) -> np.ndarray | None:
    path = _generate_path(count)
    if path is None:
        return None
    if random.random() < 0.5:
        path = list(reversed(path))

    boxes = _boxes_from_path(path)
    try:
        boxes, shape = _crop_and_shift(boxes)
    except AssertionError:
        return None
    boxes, shape = _maybe_transform_boxes(boxes, shape)

    colors = random.sample(list(_GEN_COLORS), count)
    truth_order = [int(color) for color in colors]
    rects = {int(color): box for color, box in zip(truth_order, boxes)}
    grid = _render_truth(shape, truth_order, rects)

    if not _generator_valid(grid, rects, truth_order):
        return None
    return grid


def generate_input() -> np.ndarray:
    count = int(random.choice(_GEN_RECT_COUNTS))
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _generate_attempt(count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion-order scene")
