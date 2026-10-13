from __future__ import annotations

import math
import random
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask


BG = Color.BLACK
FRAME = Color.GREY
INTERIOR_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
LINE_COLORS = [Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
FRAME_SIZE = 5
INTERIOR_SIZE = FRAME_SIZE - 2

# Two non-spatial Hamiltonian cycles over the internal placement order
# [top-left, top-right, bottom-right, bottom-left].
CYCLE_TEMPLATES = [
    ((0, 1, 3, 2), ((0, 1), (1, 3), (2, 3), (0, 2))),
    ((0, 2, 1, 3), ((0, 2), (1, 2), (1, 3), (0, 3))),
]

SHAPE_TEMPLATES = {
    "domino": {(0, 0), (0, 1)},
    "line3": {(0, 0), (0, 1), (0, 2)},
    "ell3": {(0, 0), (1, 0), (1, 1)},
    "square": {(0, 0), (0, 1), (1, 0), (1, 1)},
    "tee": {(0, 1), (1, 0), (1, 1), (1, 2)},
}


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    x0, y0 = c0, r0
    x1, y1 = c1, r1
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    out: list[tuple[int, int]] = []
    while True:
        out.append((int(y0), int(x0)))
        if x0 == x1 and y0 == y1:
            return out
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _neighbors8(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, cell)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _frame_cell_sets(frames: list[dict]) -> list[set[tuple[int, int]]]:
    sets: list[set[tuple[int, int]]] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        cells = {
            (r, c)
            for r in range(int(r0), int(r1) + 1)
            for c in range(int(c0), int(c1) + 1)
        }
        sets.append(cells)
    return sets


def _adjacent_frame_ids(
    cell: tuple[int, int], frame_sets: list[set[tuple[int, int]]]
) -> list[int]:
    nbrs = set(_neighbors8(cell))
    out = []
    for frame_id, frame_cells in enumerate(frame_sets):
        if nbrs & frame_cells:
            out.append(frame_id)
    return out


def _extract_frames(grid: np.ndarray) -> list[dict]:
    comps = components_mask(grid == FRAME, connectivity=4)
    assert len(comps) == 4
    frames: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w == FRAME_SIZE
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == FRAME)
        assert int(border.sum()) == int(coords.shape[0])
        interior = patch[1:-1, 1:-1]
        colors = sorted(set(map(int, interior.flatten())) - {BG})
        assert len(colors) == 1
        color = colors[0]
        mask = interior == color
        assert mask.any()
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "color": int(color),
                "mask": mask,
            }
        )
    return frames


def _extract_lines(grid: np.ndarray, frames: list[dict]) -> dict[int, dict]:
    frame_sets = _frame_cell_sets(frames)
    lines: dict[int, dict] = {}
    for color in LINE_COLORS:
        coords = np.argwhere(grid == color)
        assert coords.shape[0] >= 4
        visible = [tuple(map(int, rc)) for rc in coords]
        endpoints = [
            cell
            for cell in visible
            if len(_adjacent_frame_ids(cell, frame_sets)) == 1
        ]
        assert len(endpoints) == 2
        a_ids = _adjacent_frame_ids(endpoints[0], frame_sets)
        b_ids = _adjacent_frame_ids(endpoints[1], frame_sets)
        assert len(a_ids) == len(b_ids) == 1
        frame_pair = (int(a_ids[0]), int(b_ids[0]))
        assert frame_pair[0] != frame_pair[1]
        full_path = _bresenham(endpoints[0], endpoints[1])
        full_set = set(full_path)
        assert set(visible).issubset(full_set)
        lines[int(color)] = {
            "frames": frame_pair,
            "endpoints": tuple(endpoints),
            "full_path": full_path,
            "full_set": full_set,
            "visible_set": set(visible),
        }
    assert len(lines) == 4
    return lines


def _frame_sequence_for_order(
    order: tuple[int, ...] | list[int],
    lines: dict[int, dict],
    frames: list[dict],
) -> list[int] | None:
    first_a, first_b = lines[int(order[0])]["frames"]
    a_key = tuple(frames[first_a]["center"])
    b_key = tuple(frames[first_b]["center"])
    start = first_a if a_key < b_key else first_b
    current = first_b if start == first_a else first_a
    sequence = [int(start)]
    seen = {int(start)}
    for color in order[1:]:
        a, b = lines[int(color)]["frames"]
        if a == current:
            nxt = b
        elif b == current:
            nxt = a
        else:
            return None
        if nxt == start and len(sequence) != 3:
            return None
        if nxt in seen and nxt != start:
            return None
        sequence.append(int(current))
        seen.add(int(current))
        current = int(nxt)
    if current != start or len(sequence) != 4 or len(set(sequence)) != 4:
        return None
    return sequence


def _infer_solution(grid: np.ndarray) -> tuple[list[dict], dict[int, dict], tuple[int, ...], list[int]]:
    frames = _extract_frames(grid)
    lines = _extract_lines(grid, frames)

    below_edges: set[tuple[int, int]] = set()
    colors = sorted(lines)
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            inter = lines[color_a]["full_set"] & lines[color_b]["full_set"]
            assert len(inter) <= 1
            if not inter:
                continue
            cell = next(iter(inter))
            visible = int(grid[cell])
            assert visible in (color_a, color_b)
            if visible == color_a:
                below_edges.add((color_b, color_a))
            else:
                below_edges.add((color_a, color_b))

    valid: list[tuple[tuple[int, ...], list[int]]] = []
    for order in permutations(colors):
        pos = {color: idx for idx, color in enumerate(order)}
        if any(pos[lo] >= pos[hi] for lo, hi in below_edges):
            continue
        seq = _frame_sequence_for_order(order, lines, frames)
        if seq is not None:
            valid.append((order, seq))
    assert len(valid) == 1
    return frames, lines, valid[0][0], valid[0][1]


def _recolor_by_sequence(grid: np.ndarray, frames: list[dict], sequence: list[int]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    original_colors = {frame_id: int(frame["color"]) for frame_id, frame in enumerate(frames)}
    for idx, frame_id in enumerate(sequence):
        src_color = original_colors[int(sequence[(idx + 1) % len(sequence)])]
        r0, c0, r1, c1 = frames[int(frame_id)]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frames[int(frame_id)]["mask"]] = int(src_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate frame-interior colors along the unique frame cycle encoded by connector crossing layers.

    Input structure:
    - The background is black, with exactly four gray hollow 5x5 square frames spread around the canvas.
    - Each frame interior contains one small connected mask in a single color from blue/red/green/yellow.
    - Four differently colored one-pixel connector segments form one cycle among the frames and may cross near the center.

    Rule:
    - Recover each connector's two endpoint frames from the line-color pixels that touch frame borders.
    - At every line crossing, the visible color is the upper connector, which yields a unique bottom-to-top line order.
    - Among all line orders consistent with those above/below relations, keep the unique order whose consecutive lines can be walked through shared frames starting from the lower line's higher frame (smaller row center).
    - Recolor each frame's interior mask with the original interior color of the next frame in that frame-walk order, wrapping around once; keep mask shapes, frames, lines, and background unchanged.

    Key insight:
    - The color rotation order is not spatial clockwise order: it must be decoded from the connector layer order hidden in the visible crossing colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    frames, _lines, _order, sequence = _infer_solution(g)
    return _recolor_by_sequence(g, frames, sequence)


def _spatial_sequences(centers: list[tuple[float, float]]) -> set[tuple[int, ...]]:
    mean_r = sum(r for r, _ in centers) / len(centers)
    mean_c = sum(c for _, c in centers) / len(centers)
    angles = [math.atan2(r - mean_r, c - mean_c) for r, c in centers]
    cw = [idx for idx, _ in sorted(enumerate(angles), key=lambda item: item[1])]
    ccw = [cw[0]] + list(reversed(cw[1:]))
    out: set[tuple[int, ...]] = set()
    for base in (cw, ccw):
        for shift in range(len(base)):
            out.add(tuple(base[shift:] + base[:shift]))
    return out


def _external_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c0 + 1, c1):
        pts.append((r0 - 1, c))
    for c in range(c0 + 1, c1):
        pts.append((r1 + 1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0 - 1))
    for r in range(r0 + 1, r1):
        pts.append((r, c1 + 1))
    return pts


def _candidate_paths(
    boxes: list[tuple[int, int, int, int]], edge: tuple[int, int]
) -> list[list[tuple[int, int]]]:
    frame_sets = []
    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        frame_sets.append(
            {
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
            }
        )
    all_frames = set().union(*frame_sets)
    u, v = edge
    out: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for p in _external_points(boxes[u]):
        for q in _external_points(boxes[v]):
            path = _bresenham(p, q)
            if len(path) < 6:
                continue
            path_set = set(path)
            if path_set & all_frames:
                continue
            ok = True
            for idx, cell in enumerate(path):
                touching = _adjacent_frame_ids(cell, frame_sets)
                if idx == 0:
                    ok = touching == [u]
                elif idx == len(path) - 1:
                    ok = touching == [v]
                else:
                    ok = not touching
                if not ok:
                    break
            if not ok:
                continue
            key = (path[0], path[-1]) if path[0] <= path[-1] else (path[-1], path[0])
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


def _internal_valid_orders(
    edges: tuple[tuple[int, int], ...], centers: list[tuple[float, float]]
) -> list[tuple[tuple[int, ...], list[int]]]:
    edge_map = {
        idx: {"frames": tuple(map(int, edge))}
        for idx, edge in enumerate(edges)
    }
    dummy_frames = [{"center": centers[idx]} for idx in range(4)]
    out = []
    for order in permutations(range(4)):
        seq = _frame_sequence_for_order(order, edge_map, dummy_frames)
        if seq is not None:
            out.append((order, seq))
    return out


def _path_search(
    candidate_lists: list[list[list[tuple[int, int]]]],
    order: tuple[int, ...],
) -> list[list[tuple[int, int]]] | None:
    ordered = [candidate_lists[idx] for idx in order]
    ordered_sets = [[set(path) for path in bucket] for bucket in ordered]

    first_pairs: list[tuple[int, int, tuple[int, int]]] = []
    for a, path_a in enumerate(ordered[0]):
        for b, path_b in enumerate(ordered[1]):
            inter = ordered_sets[0][a] & ordered_sets[1][b]
            if len(inter) != 1:
                continue
            cell = next(iter(inter))
            if cell in {path_a[0], path_a[-1], path_b[0], path_b[-1]}:
                continue
            first_pairs.append((a, b, cell))
    random.shuffle(first_pairs)

    for a, b, cell01 in first_pairs[:300]:
        set0 = ordered_sets[0][a]
        set1 = ordered_sets[1][b]
        for c, path_c in enumerate(ordered[2]):
            set2 = ordered_sets[2][c]
            if set0 & set2:
                continue
            inter12 = set1 & set2
            if len(inter12) != 1:
                continue
            cell12 = next(iter(inter12))
            if cell12 == cell01:
                continue
            if cell12 in {ordered[1][b][0], ordered[1][b][-1], path_c[0], path_c[-1]}:
                continue
            for d, path_d in enumerate(ordered[3]):
                set3 = ordered_sets[3][d]
                if set0 & set3:
                    continue
                if set1 & set3:
                    continue
                inter23 = set2 & set3
                if len(inter23) != 1:
                    continue
                cell23 = next(iter(inter23))
                if cell23 in {cell01, cell12}:
                    continue
                if cell23 in {path_c[0], path_c[-1], path_d[0], path_d[-1]}:
                    continue
                return [ordered[0][a], ordered[1][b], path_c, path_d]
    return None


def _random_shape(name: str) -> np.ndarray:
    cells = set(SHAPE_TEMPLATES[name])
    shape = np.zeros((INTERIOR_SIZE, INTERIOR_SIZE), dtype=bool)
    max_r = max(r for r, _ in cells) + 1
    max_c = max(c for _, c in cells) + 1
    patch = np.zeros((max_r, max_c), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    for _ in range(random.randint(0, 3)):
        patch = np.rot90(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    rows, cols = np.nonzero(patch)
    r0, c0 = rows.min(), cols.min()
    r1, c1 = rows.max(), cols.max()
    patch = patch[r0 : r1 + 1, c0 : c1 + 1]
    dr = random.randint(0, INTERIOR_SIZE - patch.shape[0])
    dc = random.randint(0, INTERIOR_SIZE - patch.shape[1])
    shape[dr : dr + patch.shape[0], dc : dc + patch.shape[1]] = patch
    return shape


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    interior_color: int,
    mask: np.ndarray,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[mask] = int(interior_color)


def _build_candidate_grid() -> np.ndarray | None:
    height = random.randint(21, 24)
    width = random.randint(21, 24)

    def rand_box(r_lo: int, r_hi: int, c_lo: int, c_hi: int) -> tuple[int, int, int, int]:
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        return (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1)

    boxes = [
        rand_box(3, 5, 3, 5),
        rand_box(2, 4, width - 7, width - 5),
        rand_box(height - 7, height - 5, width - 7, width - 5),
        rand_box(height - 7, height - 5, 1, 3),
    ]
    # Re-sample degenerate center-row ties.
    centers = [((r0 + r1) / 2.0, (c0 + c1) / 2.0) for r0, c0, r1, c1 in boxes]
    if len({center[0] for center in centers}) < 4:
        return None

    _cycle_perm, cycle_edges = random.choice(CYCLE_TEMPLATES)
    candidate_lists = [_candidate_paths(boxes, edge) for edge in cycle_edges]
    if min(len(bucket) for bucket in candidate_lists) == 0:
        return None

    spatial_bad = _spatial_sequences(centers)
    valid_orders = _internal_valid_orders(cycle_edges, centers)
    random.shuffle(valid_orders)

    ordered_paths = None
    for order, frame_seq in valid_orders:
        if tuple(frame_seq) in spatial_bad:
            continue
        ordered_paths = _path_search(candidate_lists, order)
        if ordered_paths is not None:
            chosen_order = order
            break
    if ordered_paths is None:
        return None

    paths_by_edge: dict[int, list[tuple[int, int]]] = {}
    for idx, edge_idx in enumerate(chosen_order):
        paths_by_edge[int(edge_idx)] = ordered_paths[idx]

    interior_color_perm = random.sample(INTERIOR_COLORS, 4)
    shape_names = random.sample(list(SHAPE_TEMPLATES), 4)
    shape_masks = [_random_shape(name) for name in shape_names]
    edge_colors = {
        edge_idx: color for edge_idx, color in zip(range(4), random.sample(LINE_COLORS, 4))
    }

    grid = np.full((height, width), BG, dtype=int)
    for box, color, mask in zip(boxes, interior_color_perm, shape_masks):
        _draw_frame(grid, box, color, mask)
    for edge_idx in chosen_order:
        color = edge_colors[int(edge_idx)]
        for r, c in paths_by_edge[int(edge_idx)]:
            grid[int(r), int(c)] = int(color)

    visible_counts = {color: int(np.sum(grid == color)) for color in LINE_COLORS}
    if min(visible_counts.values()) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate_grid()
        if grid is None:
            continue
        try:
            frames, _lines, _order, sequence = _infer_solution(grid)
        except AssertionError:
            continue

        if tuple(sequence) in _spatial_sequences([frame["center"] for frame in frames]):
            continue
        expected = _recolor_by_sequence(grid, frames, sequence)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid layered-rotation puzzle")
