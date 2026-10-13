from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


RED = 2
BLUE = 1
MASK_COLORS = (3, 4, 5, 6, 7, 8, 9)
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG8 = ORTHO + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHO:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_frames(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    frames: list[dict[str, object]] = []
    red_lookup: dict[tuple[int, int], int] = {}

    for comp in _components(grid == RED):
        r0, c0, r1, c1 = _bbox(comp)
        side_h = int(r1 - r0 + 1)
        side_w = int(c1 - c0 + 1)
        assert side_h == side_w and side_h >= 3

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((side_h, side_w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(comp.shape[0]) == int(border.sum())
        assert bool(np.all(patch[border] == RED))
        assert bool(np.all(patch[~border] != RED))

        interior = patch[1:-1, 1:-1]
        inner_colors = set(map(int, np.unique(interior))) - {int(bg)}
        assert BLUE not in inner_colors
        assert RED not in inner_colors
        assert len(inner_colors) == 1
        color = int(next(iter(inner_colors)))
        mask = interior == color
        assert bool(mask.any())

        border_cells = {
            (r0 + int(rr), c0 + int(cc))
            for rr, cc in np.argwhere(border)
        }
        frame_id = len(frames)
        for cell in border_cells:
            red_lookup[cell] = frame_id

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "color": color,
                "mask": mask,
                "border_cells": border_cells,
            }
        )

    assert len(frames) >= 3
    return frames, red_lookup


def _extract_edges(grid: np.ndarray, frames: list[dict[str, object]], red_lookup: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    blue_components = _components(grid == BLUE)
    edges: set[tuple[int, int]] = set()

    for comp in blue_components:
        touched: set[int] = set()
        for rr, cc in comp:
            r = int(rr)
            c = int(cc)
            for dr, dc in ORTHO:
                frame_id = red_lookup.get((r + dr, c + dc))
                if frame_id is not None:
                    touched.add(int(frame_id))
        assert len(touched) == 2
        a, b = sorted(touched)
        edges.add((int(a), int(b)))

    n = len(frames)
    assert len(blue_components) == n
    assert len(edges) == n

    degree = [0] * n
    graph = {i: set() for i in range(n)}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
        graph[a].add(b)
        graph[b].add(a)
    assert all(d == 2 for d in degree)

    seen = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(graph[node] - seen)
    assert len(seen) == n
    return edges


def _clockwise_order(frames: list[dict[str, object]], edges: set[tuple[int, int]]) -> list[int]:
    mean_r = sum(float(frame["center"][0]) for frame in frames) / len(frames)
    mean_c = sum(float(frame["center"][1]) for frame in frames) / len(frames)

    angles = [
        math.atan2(float(frame["center"][0]) - mean_r, float(frame["center"][1]) - mean_c)
        for frame in frames
    ]
    rounded = {round(angle, 9) for angle in angles}
    assert len(rounded) == len(angles)

    order = sorted(range(len(frames)), key=lambda idx: angles[idx])
    for idx, nxt in zip(order, order[1:] + order[:1]):
        assert tuple(sorted((int(idx), int(nxt)))) in edges
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each frame's interior color to the next frame along the blue cycle."""
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames, red_lookup = _extract_frames(g, bg)
    edges = _extract_edges(g, frames, red_lookup)
    order = _clockwise_order(frames, edges)

    out = g.copy()
    for pos, frame_id in enumerate(order):
        next_id = order[(pos + 1) % len(order)]
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[np.asarray(frames[frame_id]["mask"], dtype=bool)] = int(frames[next_id]["color"])
    return out


GEN_BG = 0
FRAME_SIZES = (5, 6, 7)
FRAME_COUNT_RANGE = (3, 6)
FRAME_CLEARANCE = 2
MAX_ATTEMPTS = 400


def _random_mask(side: int) -> np.ndarray | None:
    total = side * side
    min_fill = max(2, int(math.ceil(total * 0.22)))
    max_fill = min(total - 1, int(math.floor(total * 0.62)))
    if max_fill < min_fill:
        return None
    target = random.randint(min_fill, max_fill)

    start = (random.randrange(side), random.randrange(side))
    cells = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        for dr, dc in DIAG8:
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(cells) < target and frontier:
        options = list(frontier)
        random.shuffle(options)
        options.sort(
            key=lambda rc: sum((rc[0] + dr, rc[1] + dc) in cells for dr, dc in DIAG8),
            reverse=True,
        )
        if random.random() < 0.7:
            pick = options[0]
        else:
            pick = random.choice(options[: min(4, len(options))])
        frontier.discard(pick)
        cells.add(pick)
        add_frontier(pick)

    if len(cells) != target:
        return None

    mask = np.zeros((side, side), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _layout_frames(n_frames: int, frame_size: int) -> tuple[list[tuple[int, int, int, int]], list[int], int, int] | None:
    chord_need = frame_size + FRAME_CLEARANCE + 2
    radius_min = int(math.ceil(chord_need / (2.0 * math.sin(math.pi / n_frames)))) + 1
    side_min = int(math.ceil(2 * (radius_min + frame_size / 2.0 + 4)))

    grid_h = random.randint(max(24, side_min), max(24, side_min) + 8)
    grid_w = random.randint(max(24, side_min), max(24, side_min) + 8)
    center_r = (grid_h - 1) / 2.0
    center_c = (grid_w - 1) / 2.0
    radius_max = min(center_r, center_c) - frame_size / 2.0 - 3.0
    if radius_max < radius_min:
        return None

    rotation = random.uniform(0.0, 2.0 * math.pi)
    angles = [
        (rotation + (2.0 * math.pi * idx / n_frames) + random.uniform(-0.22, 0.22)) % (2.0 * math.pi)
        for idx in range(n_frames)
    ]
    angles.sort()

    boxes: list[tuple[int, int, int, int]] = []
    for angle in angles:
        radius = random.uniform(radius_min, radius_max)
        r0 = int(round(center_r + radius * math.sin(angle) - frame_size / 2.0))
        c0 = int(round(center_c + radius * math.cos(angle) - frame_size / 2.0))
        bbox = (r0, c0, r0 + frame_size - 1, c0 + frame_size - 1)
        if bbox[0] < 1 or bbox[1] < 1 or bbox[2] >= grid_h - 1 or bbox[3] >= grid_w - 1:
            return None
        boxes.append(bbox)

    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            if _boxes_too_close(boxes[i], boxes[j], FRAME_CLEARANCE):
                return None

    order = list(range(n_frames))
    return boxes, order, grid_h, grid_w


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, mask: np.ndarray) -> dict[str, object]:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[mask] = int(color)

    border_cells = {
        (r0, c) for c in range(c0, c1 + 1)
    }
    border_cells.update((r1, c) for c in range(c0, c1 + 1))
    border_cells.update((r, c0) for r in range(r0, r1 + 1))
    border_cells.update((r, c1) for r in range(r0, r1 + 1))

    return {
        "bbox": bbox,
        "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
        "border_cells": border_cells,
    }


def _frame_touch_halo(frame: dict[str, object], h: int, w: int) -> set[tuple[int, int]]:
    border_cells = frame["border_cells"]
    halo: set[tuple[int, int]] = set()
    for r, c in border_cells:
        for dr, dc in ORTHO:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in border_cells:
                halo.add((nr, nc))
    return halo


def _side_candidates(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0 - 1, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1 + 1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0 - 1) for r in range(r0 + 1, r1)]
    return [(r, c1 + 1) for r in range(r0 + 1, r1)]


def _attachment_candidates(frame: dict[str, object], target_center: tuple[float, float], h: int, w: int) -> list[tuple[int, int]]:
    fr, fc = frame["center"]
    dr = target_center[0] - fr
    dc = target_center[1] - fc
    side_scores = {
        "top": -dr,
        "bottom": dr,
        "left": -dc,
        "right": dc,
    }

    ranked: list[tuple[float, float, float, tuple[int, int]]] = []
    for side in ("top", "bottom", "left", "right"):
        starts = _side_candidates(frame["bbox"], side)
        random.shuffle(starts)
        for start in starts:
            r, c = start
            if not (0 <= r < h and 0 <= c < w):
                continue
            distance = abs(float(r) - target_center[0]) + abs(float(c) - target_center[1])
            ranked.append((-float(side_scores[side]), float(distance), random.random(), start))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [start for _score, _dist, _noise, start in ranked]


def _path_between(start: tuple[int, int], end: tuple[int, int], blocked: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        cell = queue.popleft()
        if cell == end:
            break
        neighbors = []
        for dr, dc in ORTHO:
            nr = cell[0] + dr
            nc = cell[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in blocked or nxt in parent:
                continue
            heuristic = abs(nr - end[0]) + abs(nc - end[1]) + random.random() * 0.2
            neighbors.append((heuristic, nxt))
        neighbors.sort(key=lambda item: item[0])
        for _heuristic, nxt in neighbors:
            parent[nxt] = cell
            queue.append(nxt)

    if end not in parent:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _connect_cycle(
    grid: np.ndarray,
    frames: list[dict[str, object]],
    order: list[int],
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    halos = [_frame_touch_halo(frame, h, w) for frame in frames]
    blue_halo: set[tuple[int, int]] = set()

    for pos, frame_id in enumerate(order):
        next_id = order[(pos + 1) % len(order)]
        starts_a = _attachment_candidates(frames[frame_id], frames[next_id]["center"], h, w)
        starts_b = _attachment_candidates(frames[next_id], frames[frame_id]["center"], h, w)

        success = False
        for start_a in starts_a[:12]:
            if start_a in occupied or start_a in blue_halo:
                continue
            for start_b in starts_b[:12]:
                if start_b == start_a or start_b in occupied or start_b in blue_halo:
                    continue

                blocked = set(occupied)
                blocked.update(blue_halo)
                for idx, halo in enumerate(halos):
                    if idx == frame_id:
                        blocked.update(halo - {start_a})
                    elif idx == next_id:
                        blocked.update(halo - {start_b})
                    else:
                        blocked.update(halo)
                blocked.discard(start_a)
                blocked.discard(start_b)

                path = _path_between(start_a, start_b, blocked, h, w)
                if path is None or len(path) < 2:
                    continue

                for cell in path:
                    grid[cell] = BLUE
                    occupied.add(cell)
                for r, c in path:
                    blue_halo.add((r, c))
                    for dr, dc in ORTHO:
                        nr = r + dr
                        nc = c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            blue_halo.add((nr, nc))
                success = True
                break
            if success:
                break
        if not success:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])
    frame_size = random.choice(FRAME_SIZES)
    laid_out = _layout_frames(n_frames, frame_size)
    if laid_out is None:
        return None
    boxes, order, grid_h, grid_w = laid_out

    grid = np.full((grid_h, grid_w), GEN_BG, dtype=int)
    colors = random.sample(list(MASK_COLORS), n_frames)
    frames: list[dict[str, object]] = []

    for bbox, color in zip(boxes, colors):
        mask = _random_mask(frame_size - 2)
        if mask is None:
            return None
        frames.append(_draw_frame(grid, bbox, color, mask))

    occupied = {(int(r), int(c)) for r, c in np.argwhere(grid != GEN_BG)}
    if not _connect_cycle(grid, frames, order, occupied):
        return None

    if int(np.count_nonzero(grid == GEN_BG)) <= grid.size // 2:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if bool(np.array_equal(out, grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
