from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, colors_present, crop_bbox, get_bg  # noqa: E402


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 500
TEMPLATE_HEIGHT_RANGE = (5, 8)
TEMPLATE_WIDTH_RANGE = (5, 8)
TEMPLATE_FILL_RANGE = (0.42, 0.72)
LAYER_COUNT_OPTIONS = (3, 4)
SHIFT_LIMIT = 3
GRID_MARGIN_RANGE = (1, 5)
PALETTE = tuple(range(10))


def _extract_fragments(grid: np.ndarray, bg: int) -> dict[int, dict[str, object]]:
    fragments: dict[int, dict[str, object]] = {}
    for color in colors_present(grid, exclude={bg}):
        mask = grid == color
        box = bbox(mask)
        assert box is not None, "each foreground color must appear"
        r0, c0, r1, c1 = box
        fragments[int(color)] = {
            "bbox": tuple(map(int, box)),
            "local": mask[r0 : r1 + 1, c0 : c1 + 1].copy(),
        }
    assert fragments, "expected at least one foreground color"
    return fragments


def _recover_template(fragments: dict[int, dict[str, object]]) -> np.ndarray:
    height = max(int(info["local"].shape[0]) for info in fragments.values())
    width = max(int(info["local"].shape[1]) for info in fragments.values())
    template = np.zeros((height, width), dtype=bool)
    for info in fragments.values():
        local = np.asarray(info["local"], dtype=bool)
        h, w = local.shape
        template[:h, :w] |= local
    assert template.any(), "recovered template must be non-empty"
    return template


def _cover_graph(
    grid: np.ndarray,
    bg: int,
    fragments: dict[int, dict[str, object]],
    template: np.ndarray,
) -> dict[int, set[int]]:
    graph = {int(color): set() for color in fragments}
    template = np.asarray(template, dtype=bool)
    th, tw = template.shape
    for color, info in fragments.items():
        local = np.zeros((th, tw), dtype=bool)
        frag = np.asarray(info["local"], dtype=bool)
        fh, fw = frag.shape
        local[:fh, :fw] = frag
        r0, c0, _r1, _c1 = info["bbox"]
        missing = np.argwhere(template & ~local)
        for dr, dc in missing:
            rr = int(r0) + int(dr)
            cc = int(c0) + int(dc)
            occupant = int(grid[rr, cc])
            assert occupant != bg, "a missing template pixel must be occluded by another layer"
            if occupant != color:
                graph[occupant].add(int(color))
    return graph


def _topo_order(colors: list[int], graph: dict[int, set[int]]) -> list[int]:
    indegree = {int(color): 0 for color in colors}
    for src in colors:
        for dst in graph[int(src)]:
            indegree[int(dst)] += 1
    queue = deque(sorted(color for color in colors if indegree[int(color)] == 0))
    order: list[int] = []
    while queue:
        color = int(queue.popleft())
        order.append(color)
        for nxt in sorted(graph[color]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    assert len(order) == len(colors), "occlusion graph must be acyclic"
    return order


def _deepest_color(
    grid: np.ndarray,
    bg: int,
    fragments: dict[int, dict[str, object]],
    template: np.ndarray,
) -> int:
    colors = sorted(int(color) for color in fragments)
    graph = _cover_graph(grid, bg, fragments, template)
    sinks = [color for color in colors if not graph[color]]
    assert len(sinks) == 1, "expected a unique deepest layer"
    order = _topo_order(colors, graph)
    assert order[-1] == sinks[0], "deepest layer must be the unique topological sink"
    return int(sinks[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the shared sprite mask from overlapping copies and return its tight crop in the deepest copy's color.

    Input structure:
    - The dominant color is the background.
    - Each non-background color is the visible fragment of one translated copy of the same unknown sprite mask.
    - The copies overlap by occlusion: higher layers overwrite lower ones, so lower colors may be missing cells inside their own bounding boxes.

    Rule:
    - Crop every color to its bounding box and align those crops by their top-left corners.
    - Take the union of the aligned visible fragments to recover the full sprite mask.
    - For each color, every recovered mask cell that it is missing is occupied by a color above it; use all such cover relations to identify the unique deepest color.
    - Output the recovered mask as a tight crop, painted entirely in that deepest color over the background.

    Key insight:
    - The most complete fragment is the top layer, not the answer; the answer color is the one that never covers anyone below and can only be found from who occupies whose missing cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    fragments = _extract_fragments(g, bg)
    template = _recover_template(fragments)
    deepest = _deepest_color(g, bg, fragments, template)

    box = bbox(template)
    assert box is not None
    tight = crop_bbox(template.astype(int), box).astype(bool)
    out = np.full(tight.shape, bg, dtype=int)
    out[tight] = deepest
    return out


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    vertical_first = random.choice([False, True])
    while (r, c) != (tr, tc):
        mask[r, c] = True
        moves: list[tuple[int, int]] = []
        if r != tr:
            moves.append((r + (1 if tr > r else -1), c))
        if c != tc:
            moves.append((r, c + (1 if tc > c else -1)))
        if len(moves) == 2:
            r, c = moves[0 if vertical_first else 1]
            vertical_first = not vertical_first if random.random() < 0.35 else vertical_first
        else:
            r, c = moves[0]
    mask[tr, tc] = True


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    queue = deque(seen)
    h, w = mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _grow_template(height: int, width: int) -> np.ndarray:
    target_min = max(7, int(np.ceil(height * width * TEMPLATE_FILL_RANGE[0])))
    target_max = max(target_min, int(np.floor(height * width * TEMPLATE_FILL_RANGE[1])))
    target_area = random.randint(target_min, target_max)

    mask = np.zeros((height, width), dtype=bool)
    anchors = {
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    }
    hub = (
        random.randrange(1, height - 1) if height > 2 else 0,
        random.randrange(1, width - 1) if width > 2 else 0,
    )
    for anchor in anchors:
        _draw_path(mask, anchor, hub)

    while int(mask.sum()) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in map(tuple, np.argwhere(mask)):
            for dr, dc in DIR4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        r, c = random.choice(tuple(frontier))
        mask[r, c] = True

    interior = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if mask[r, c]
    ]
    random.shuffle(interior)
    removals = random.randint(0, min(3, len(interior)))
    removed = 0
    for r, c in interior:
        if removed >= removals or int(mask.sum()) <= target_min:
            break
        mask[r, c] = False
        if _is_connected(mask):
            removed += 1
        else:
            mask[r, c] = True

    assert _is_connected(mask)
    assert mask[0].any() and mask[-1].any()
    assert mask[:, 0].any() and mask[:, -1].any()
    assert not np.all(mask)
    return mask


def _template_cells(template: np.ndarray, offset: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = offset
    return {
        (int(r0) + int(r), int(c0) + int(c))
        for r, c in map(tuple, np.argwhere(template))
    }


def _render_scene(
    shape: tuple[int, int],
    bg: int,
    template: np.ndarray,
    colors: list[int],
    offsets: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, (r0, c0) in reversed(list(zip(colors, offsets))):
        for r, c in map(tuple, np.argwhere(template)):
            grid[int(r0) + int(r), int(c0) + int(c)] = int(color)
    return grid


def _bbox_overlap(a: tuple[int, int], b: tuple[int, int], height: int, width: int) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    return max(ar0, br0) < min(ar0 + height, br0 + height) and max(ac0, bc0) < min(ac0 + width, bc0 + width)


def _sample_shifts(layer_count: int) -> list[tuple[int, int]] | None:
    step_pool = [
        (-2, -1), (-2, 0), (-2, 1),
        (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
        (0, -2), (0, -1), (0, 1), (0, 2),
        (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
        (2, -1), (2, 0), (2, 1),
    ]
    for _ in range(120):
        shifts = [(0, 0)]
        used = {(0, 0)}
        ok = True
        for _layer in range(1, layer_count):
            prev_r, prev_c = shifts[-1]
            candidates = []
            for dr, dc in step_pool:
                nr, nc = prev_r + dr, prev_c + dc
                if max(abs(nr), abs(nc)) > SHIFT_LIMIT:
                    continue
                if (nr, nc) in used:
                    continue
                candidates.append((nr, nc))
            if not candidates:
                ok = False
                break
            nxt = random.choice(candidates)
            shifts.append(nxt)
            used.add(nxt)
        if ok and len(set(shifts)) == layer_count:
            return shifts
    return None


def _place_offsets(
    template: np.ndarray,
    layer_count: int,
) -> tuple[tuple[int, int], list[tuple[int, int]], tuple[int, int]] | None:
    height, width = template.shape
    shifts = _sample_shifts(layer_count)
    if shifts is None:
        return None

    min_dr = min(dr for dr, _dc in shifts)
    max_dr = max(dr for dr, _dc in shifts)
    min_dc = min(dc for _dr, dc in shifts)
    max_dc = max(dc for _dr, dc in shifts)

    grid_h_min = height + (max_dr - min_dr) + random.randint(*GRID_MARGIN_RANGE)
    grid_w_min = width + (max_dc - min_dc) + random.randint(*GRID_MARGIN_RANGE)
    grid_h = random.randint(max(grid_h_min, height + 2), min(24, max(grid_h_min, height + 8)))
    grid_w = random.randint(max(grid_w_min, width + 2), min(24, max(grid_w_min, width + 8)))

    row_lo = -min_dr
    row_hi = grid_h - height - max_dr
    col_lo = -min_dc
    col_hi = grid_w - width - max_dc
    if row_lo > row_hi or col_lo > col_hi:
        return None

    base = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
    offsets = [(base[0] + dr, base[1] + dc) for dr, dc in shifts]

    overlaps = []
    for i in range(layer_count):
        touched = False
        for j in range(layer_count):
            if i == j:
                continue
            if _bbox_overlap(offsets[i], offsets[j], height, width):
                touched = True
        overlaps.append(touched)
    if not all(overlaps):
        return None
    return base, offsets, (grid_h, grid_w)


def _expected_output(template: np.ndarray, color: int, bg: int) -> np.ndarray:
    box = bbox(template)
    assert box is not None
    tight = crop_bbox(template.astype(int), box).astype(bool)
    out = np.full(tight.shape, bg, dtype=int)
    out[tight] = int(color)
    return out


def _validate_scene(
    grid: np.ndarray,
    bg: int,
    template: np.ndarray,
    colors: list[int],
    offsets: list[tuple[int, int]],
) -> bool:
    if max(grid.shape) > 30:
        return False

    fragments = _extract_fragments(grid, bg)
    if set(fragments) != set(colors):
        return False

    for color, offset in zip(colors, offsets):
        box = fragments[color]["bbox"]
        if box[:2] != offset:
            return False
        local = np.asarray(fragments[color]["local"], dtype=bool)
        if local.shape != template.shape:
            return False
        if not (local[0].any() and local[-1].any() and local[:, 0].any() and local[:, -1].any()):
            return False

    recovered = _recover_template(fragments)
    if recovered.shape != template.shape or not np.array_equal(recovered, template):
        return False

    graph = _cover_graph(grid, bg, fragments, recovered)
    indegree = {color: 0 for color in colors}
    for src in colors:
        for dst in graph[src]:
            indegree[dst] += 1
    if [color for color in colors if not graph[color]] != [colors[-1]]:
        return False
    for idx, color in enumerate(colors):
        if idx > 0 and indegree[color] == 0:
            return False
        if idx < len(colors) - 1 and not graph[color]:
            return False

    visible_counts = [int(np.sum(grid == color)) for color in colors]
    if len(set(visible_counts)) != len(visible_counts):
        return False
    if int(np.argmin(visible_counts)) == len(colors) - 1:
        return False
    depth_order = list(range(len(colors)))
    count_order = sorted(depth_order, key=lambda i: visible_counts[i], reverse=True)
    if count_order == depth_order:
        return False
    if visible_counts[0] != int(template.sum()):
        return False
    if not all(0 < visible_counts[i] < int(template.sum()) for i in range(1, len(colors))):
        return False

    for r, c in _template_cells(template, offsets[-1]):
        if grid[r, c] == bg:
            return False

    expected = _expected_output(template, colors[-1], bg)
    solved = main(grid)
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    layer_count = random.choice(LAYER_COUNT_OPTIONS)
    colors = random.sample([color for color in PALETTE if color != bg], layer_count)

    height = random.randint(*TEMPLATE_HEIGHT_RANGE)
    width = random.randint(*TEMPLATE_WIDTH_RANGE)
    template = _grow_template(height, width)

    placed = _place_offsets(template, layer_count)
    if placed is None:
        return None
    _base, offsets, grid_shape = placed

    grid = _render_scene(grid_shape, bg, template, colors, offsets)
    if not _validate_scene(grid, bg, template, colors, offsets):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid deep-layer imprint input")
