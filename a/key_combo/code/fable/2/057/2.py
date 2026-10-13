from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
BLACK = int(Color.BLACK)
GREY = int(Color.GREY)
COLOR_POOL = [int(c) for c in Color.NOT_BLACK if int(c) != GREY]
GEN_ATTEMPTS = 400


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_legend_and_canvas(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 3, "expected a 2D grid with legend, separator, and canvas"
    assert np.all(g[:, 1] == GREY), "expected a solid grey separator in column 1"

    legend: list[int] = []
    for value in g[:, 0]:
        color = int(value)
        if color == BLACK:
            break
        legend.append(color)

    assert legend, "expected a non-empty top-aligned legend bar"
    assert np.all(g[len(legend) :, 0] == BLACK), "legend column must be contiguous from the top"
    return legend, g[:, 2:].copy()


def _adjacent_black_components(wall_labels: np.ndarray, black_labels: np.ndarray, wall_id: int) -> set[int]:
    coords = np.argwhere(wall_labels == int(wall_id))
    neighbors: set[int] = set()
    h, w = wall_labels.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                cid = int(black_labels[nr, nc])
                if cid > 0:
                    neighbors.add(cid)
    return neighbors


def _component_depths(canvas: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    black_labels, n_black = label_mask(canvas == BLACK, connectivity=4)
    wall_labels, n_walls = label_mask(canvas != BLACK, connectivity=4)
    assert int(n_black) > 0 and int(n_walls) > 0, "expected black regions and wall regions on the canvas"

    graph = {cid: set() for cid in range(1, int(n_black) + 1)}
    for wall_id in range(1, int(n_walls) + 1):
        neighbors = sorted(_adjacent_black_components(wall_labels, black_labels, wall_id))
        for i, src in enumerate(neighbors):
            for dst in neighbors[i + 1 :]:
                graph[src].add(dst)
                graph[dst].add(src)

    edge_ids = set(map(int, black_labels[0, :]))
    edge_ids |= set(map(int, black_labels[-1, :]))
    edge_ids |= set(map(int, black_labels[:, 0]))
    edge_ids |= set(map(int, black_labels[:, -1]))
    boundary = {cid for cid in edge_ids if cid > 0}
    assert boundary, "expected at least one depth-0 black region on the canvas boundary"

    depths: dict[int, int] = {}
    queue: deque[int] = deque(sorted(boundary))
    for cid in queue:
        depths[int(cid)] = 0

    while queue:
        cid = int(queue.popleft())
        for nxt in sorted(graph[cid]):
            if nxt in depths:
                continue
            depths[nxt] = depths[cid] + 1
            queue.append(nxt)

    assert len(depths) == int(n_black), "every black region must connect to the boundary through nested walls"
    return black_labels, wall_labels, depths


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend, canvas = _parse_legend_and_canvas(g)
    black_labels, _wall_labels, depths = _component_depths(canvas)

    max_depth = max(depths.values())
    assert max_depth <= len(legend), "legend is too short for the observed nesting depth"

    out = g.copy()
    canvas_out = out[:, 2:]
    for cid, depth in depths.items():
        if depth == 0:
            continue
        canvas_out[black_labels == int(cid)] = int(legend[depth - 1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each enclosed black canvas region by the legend color indexed by wall depth.

    Input structure:
    - Column 0 contains a top-aligned legend bar of non-black colors, with black cells below it.
    - Column 1 is a full-height grey separator.
    - Columns 2 onward form a black canvas containing one-cell-thick closed colored wall loops that may be nested or disjoint.

    Rule:
    - In the canvas, every non-black cell is a wall and every black cell is open space.
    - The depth of a black cell is the number of closed wall loops that must be crossed to reach the canvas boundary.
    - Depth-0 cells stay black; depth-d cells are recolored with the d-th legend color from top to bottom.
    - Preserve all wall cells, the legend bar, and the grey separator unchanged.

    Key insight:
    - Wall colors and wall geometry are distractions; only the topological nesting depth of each black region matters, and the legend is an ordered depth lookup table.
    """
    return _solve(grid)


def _crop_true(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _make_wall_loop(base_h: int, base_w: int, irregular: bool) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    pad = 3
    wall = np.zeros((base_h + 2 * pad, base_w + 2 * pad), dtype=bool)
    top = pad
    left = pad
    bottom = top + base_h - 1
    right = left + base_w - 1

    wall[top, left : right + 1] = True
    wall[bottom, left : right + 1] = True
    wall[top : bottom + 1, left] = True
    wall[top : bottom + 1, right] = True

    if irregular:
        side = random.choice(("top", "bottom", "left", "right"))
        if side in {"top", "bottom"}:
            depth = random.randint(1, 2)
            start = random.randint(1, base_w - 4)
            end = random.randint(start + 2, base_w - 2)
            c0 = left + start
            c1 = left + end
            row = top if side == "top" else bottom
            wall[row, c0 + 1 : c1] = False
            if side == "top":
                wall[top - depth : top + 1, c0] = True
                wall[top - depth : top + 1, c1] = True
                wall[top - depth, c0 : c1 + 1] = True
            else:
                wall[bottom : bottom + depth + 1, c0] = True
                wall[bottom : bottom + depth + 1, c1] = True
                wall[bottom + depth, c0 : c1 + 1] = True
        else:
            depth = random.randint(1, 2)
            start = random.randint(1, base_h - 4)
            end = random.randint(start + 2, base_h - 2)
            r0 = top + start
            r1 = top + end
            col = left if side == "left" else right
            wall[r0 + 1 : r1, col] = False
            if side == "left":
                wall[r0, left - depth : left + 1] = True
                wall[r1, left - depth : left + 1] = True
                wall[r0 : r1 + 1, left - depth] = True
            else:
                wall[r0, right : right + depth + 1] = True
                wall[r1, right : right + depth + 1] = True
                wall[r0 : r1 + 1, right + depth] = True

    wall, crop_box = _crop_true(wall)
    crop_r0, crop_c0, _crop_r1, _crop_c1 = crop_box
    core_bbox = (
        top - crop_r0,
        left - crop_c0,
        bottom - crop_r0,
        right - crop_c0,
    )
    _labels, n_parts = label_mask(wall, connectivity=4)
    assert int(n_parts) == 1 and base_h >= 5 and base_w >= 5, "wall shape must stay a single loop"
    return wall, core_bbox


def _single_wall_grid(base_h: int, base_w: int, color: int, irregular: bool) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    wall_mask, core_bbox = _make_wall_loop(base_h, base_w, irregular)
    h, w = wall_mask.shape
    grid = np.zeros((h + 2, w + 2), dtype=int)
    interior = grid[1 : h + 1, 1 : w + 1]
    interior[wall_mask] = int(color)
    r0, c0, r1, c1 = core_bbox
    return grid, (r0 + 1, c0 + 1, r1 + 1, c1 + 1)


def _build_stack(depth: int, colors: list[int], force_irregular_outer: bool) -> tuple[np.ndarray, dict[str, int | bool]]:
    assert depth >= 1 and len(colors) >= depth
    if depth == 1:
        base_h = random.randint(5, 6)
        base_w = random.randint(5, 7)
        irregular = force_irregular_outer or random.random() < 0.45
        grid, _core_bbox = _single_wall_grid(base_h, base_w, colors[0], irregular)
        return grid, {"wall_count": 1, "irregular": irregular}

    inner, meta = _build_stack(depth - 1, colors[1:], False)
    inner_h, inner_w = inner.shape
    base_h = max(6, inner_h + 2)
    base_w = max(6, inner_w + 2)
    irregular = force_irregular_outer or random.random() < 0.35
    outer, core_bbox = _single_wall_grid(base_h, base_w, colors[0], irregular)

    r0, c0, r1, c1 = core_bbox
    row_min = r0 + 1
    col_min = c0 + 1
    row_max = r1 - inner_h
    col_max = c1 - inner_w
    assert row_min <= row_max and col_min <= col_max

    top = random.randint(row_min, row_max)
    left = random.randint(col_min, col_max)
    window = outer[top : top + inner_h, left : left + inner_w]
    assert not np.any((window != BLACK) & (inner != BLACK))
    window[inner != BLACK] = inner[inner != BLACK]

    return outer, {"wall_count": int(meta["wall_count"]) + 1, "irregular": bool(meta["irregular"]) or irregular}


def _split_extra(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _layout_canvas(stacks: list[np.ndarray]) -> np.ndarray:
    canvas_max_h = 30
    canvas_max_w = 28
    items = [np.asarray(stack, dtype=int) for stack in stacks]
    random.shuffle(items)

    options = ["horizontal", "vertical"]
    random.shuffle(options)
    for orientation in options:
        hs = [int(item.shape[0]) for item in items]
        ws = [int(item.shape[1]) for item in items]
        margin_top = random.randint(1, 2)
        margin_bottom = random.randint(1, 2)
        margin_left = random.randint(1, 2)
        margin_right = random.randint(1, 2)
        gap = random.randint(2, 3)

        if orientation == "horizontal":
            min_h = max(hs) + margin_top + margin_bottom
            min_w = sum(ws) + gap * (len(items) - 1) + margin_left + margin_right
            if min_h > canvas_max_h or min_w > canvas_max_w:
                continue
            extra_h = random.randint(0, min(3, canvas_max_h - min_h))
            extra_w = random.randint(0, min(4, canvas_max_w - min_w))
            canvas = np.zeros((min_h + extra_h, min_w + extra_w), dtype=int)
            horizontal_slack = _split_extra(extra_w, len(items) + 1)
            col = margin_left + horizontal_slack[0]
            for idx, item in enumerate(items):
                h, w = item.shape
                row = margin_top + random.randint(0, canvas.shape[0] - margin_top - margin_bottom - h)
                window = canvas[row : row + h, col : col + w]
                assert not np.any((window != BLACK) & (item != BLACK))
                window[item != BLACK] = item[item != BLACK]
                col += w
                if idx + 1 < len(items):
                    col += gap + horizontal_slack[idx + 1]
            return canvas

        min_h = sum(hs) + gap * (len(items) - 1) + margin_top + margin_bottom
        min_w = max(ws) + margin_left + margin_right
        if min_h > canvas_max_h or min_w > canvas_max_w:
            continue
        extra_h = random.randint(0, min(4, canvas_max_h - min_h))
        extra_w = random.randint(0, min(3, canvas_max_w - min_w))
        canvas = np.zeros((min_h + extra_h, min_w + extra_w), dtype=int)
        vertical_slack = _split_extra(extra_h, len(items) + 1)
        row = margin_top + vertical_slack[0]
        for idx, item in enumerate(items):
            h, w = item.shape
            col = margin_left + random.randint(0, canvas.shape[1] - margin_left - margin_right - w)
            window = canvas[row : row + h, col : col + w]
            assert not np.any((window != BLACK) & (item != BLACK))
            window[item != BLACK] = item[item != BLACK]
            row += h
            if idx + 1 < len(items):
                row += gap + vertical_slack[idx + 1]
        return canvas

    raise RuntimeError("failed to fit stacks onto the canvas")


def _bbox_fill_heuristic(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend, canvas = _parse_legend_and_canvas(g)
    out = g.copy()
    canvas_out = out[:, 2:]
    wall_labels, n_walls = label_mask(canvas != BLACK, connectivity=4)
    bboxes = []
    for wall_id in range(1, int(n_walls) + 1):
        coords = np.argwhere(wall_labels == wall_id)
        bboxes.append(_bbox_from_coords(coords))

    for r, c in np.argwhere(canvas == BLACK):
        rr = int(r)
        cc = int(c)
        depth = sum(int(r0 < rr < r1 and c0 < cc < c1) for r0, c0, r1, c1 in bboxes)
        if depth > 0 and depth <= len(legend):
            canvas_out[rr, cc] = int(legend[depth - 1])
    return out


def _flat_enclosure_heuristic(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend, canvas = _parse_legend_and_canvas(g)
    black_labels, _wall_labels, depths = _component_depths(canvas)
    out = g.copy()
    if not legend:
        return out
    canvas_out = out[:, 2:]
    for cid, depth in depths.items():
        if depth > 0:
            canvas_out[black_labels == int(cid)] = int(legend[0])
    return out


def _semantic_accepts(grid: np.ndarray, expected_wall_count: int, require_irregular: bool) -> bool:
    try:
        legend, canvas = _parse_legend_and_canvas(grid)
        black_labels, wall_labels, depths = _component_depths(canvas)
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(grid, out):
        return False

    seen_depths = {int(depth) for depth in depths.values()}
    if 0 not in seen_depths or len(seen_depths) < 3:
        return False
    if max(seen_depths) < 2:
        return False
    if sorted(d for d in seen_depths if d > 0) != list(range(1, max(seen_depths) + 1)):
        return False
    if max(seen_depths) > len(legend):
        return False

    _labels, n_walls = label_mask(canvas != BLACK, connectivity=4)
    if int(n_walls) != int(expected_wall_count):
        return False

    for wall_id in range(1, int(n_walls) + 1):
        coords = np.argwhere(wall_labels == wall_id)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        if r0 == 0 or c0 == 0 or r1 == canvas.shape[0] - 1 or c1 == canvas.shape[1] - 1:
            return False

    if np.array_equal(_bbox_fill_heuristic(grid), out):
        return False
    if np.array_equal(_flat_enclosure_heuristic(grid), out):
        return False

    positive_colors = set()
    for cid, depth in depths.items():
        if depth > 0:
            sample = np.argwhere(black_labels == int(cid))[0]
            positive_colors.add(int(out[int(sample[0]), int(sample[1]) + 2]))
    if len(positive_colors) < 2:
        return False

    if require_irregular:
        if np.array_equal(_bbox_fill_heuristic(grid), out):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    max_depth = random.choice((2, 2, 3))
    legend_len = max_depth + random.choice((0, 1))
    wall_count = max_depth + 1
    if legend_len + wall_count > len(COLOR_POOL):
        return None

    chosen = random.sample(COLOR_POOL, legend_len + wall_count)
    legend_colors = chosen[:legend_len]
    wall_colors = chosen[legend_len:]

    primary, primary_meta = _build_stack(max_depth, wall_colors[:max_depth], True)
    secondary, secondary_meta = _build_stack(1, [wall_colors[max_depth]], random.random() < 0.4)

    primary = np.rot90(primary, random.randrange(4))
    secondary = np.rot90(secondary, random.randrange(4))
    try:
        canvas = _layout_canvas([primary, secondary])
    except RuntimeError:
        return None

    h, w = canvas.shape
    grid = np.zeros((h, w + 2), dtype=int)
    grid[:, 1] = GREY
    grid[:legend_len, 0] = np.array(legend_colors, dtype=int)
    grid[:, 2:] = canvas

    wall_total = int(primary_meta["wall_count"]) + int(secondary_meta["wall_count"])
    require_irregular = bool(primary_meta["irregular"]) or bool(secondary_meta["irregular"])
    if not _semantic_accepts(grid, wall_total, require_irregular):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth recoloring input")
