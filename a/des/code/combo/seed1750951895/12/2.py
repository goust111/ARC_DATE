from __future__ import annotations

import math
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PALETTE = (3, 4, 5, 6, 7, 8, 9)
RNG = np.random.default_rng()

GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_frame(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or len(coords) != 2 * h + 2 * w - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(patch == RED, expected))


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _components(grid == RED):
        assert _is_frame(grid, comp), "every red component must be a one-pixel frame"
        r0, c0, r1, c1 = _bbox(comp)
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        mask = (interior != BLACK) & (interior != RED)
        colors = set(map(int, interior[mask]))
        assert len(colors) == 1 and mask.any(), "each frame interior must contain one non-red color"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "mask": mask.copy(),
                "color": int(next(iter(colors))),
            }
        )
    assert len(frames) >= 3
    return frames


def _frame_graph(grid: np.ndarray, frames: list[dict]) -> list[set[int]]:
    h, w = grid.shape
    frame_map = np.full((h, w), -1, dtype=int)
    for i, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        frame_map[r0, c0 : c1 + 1] = i
        frame_map[r1, c0 : c1 + 1] = i
        frame_map[r0 : r1 + 1, c0] = i
        frame_map[r0 : r1 + 1, c1] = i

    graph = [set() for _ in frames]
    for comp in _components(grid == BLUE):
        touched = set()
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and frame_map[nr, nc] != -1:
                    touched.add(int(frame_map[nr, nc]))
        assert len(touched) == 2, "each blue component must connect exactly two frames"
        a, b = sorted(touched)
        graph[a].add(b)
        graph[b].add(a)

    n = len(frames)
    assert all(len(nei) == 2 for nei in graph), "frames must form a degree-2 ring"
    seen = {0}
    q = deque([0])
    while q:
        cur = q.popleft()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert len(seen) == n, "ring must be connected"
    return graph


def _clockwise_order(frames: list[dict]) -> list[int]:
    centers = np.asarray([frame["center"] for frame in frames], dtype=float)
    cr, cc = centers.mean(axis=0)
    angles = [math.atan2(r - cr, c - cc) for r, c in centers]
    return sorted(range(len(frames)), key=lambda i: angles[i])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    _frame_graph(g, frames)
    order = _clockwise_order(frames)

    out = g.copy()
    rotated_colors = [frames[order[(i + 1) % len(order)]]["color"] for i in range(len(order))]
    for idx, new_color in zip(order, rotated_colors):
        r0, c0, r1, c1 = frames[idx]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frames[idx]["mask"]] = BLACK
        interior[frames[idx]["mask"]] = int(new_color)
    return out


def _boxes_far_apart(boxes: list[tuple[int, int, int, int]], gap: int) -> bool:
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if not (
                a[2] + gap < b[0]
                or b[2] + gap < a[0]
                or a[3] + gap < b[1]
                or b[3] + gap < a[1]
            ):
                return False
    return True


def _shape_mask(h: int, w: int) -> np.ndarray:
    mode = int(RNG.integers(0, 4))
    mask = np.zeros((h, w), dtype=bool)
    if mode == 0:
        rh = int(RNG.integers(1, h + 1))
        rw = int(RNG.integers(1, w + 1))
        r0 = int(RNG.integers(0, h - rh + 1))
        c0 = int(RNG.integers(0, w - rw + 1))
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
    elif mode == 1:
        row = int(RNG.integers(0, h))
        col = int(RNG.integers(0, w))
        mask[row, :] = True
        mask[:, col] = True
    elif mode == 2:
        if RNG.random() < 0.5:
            for k in range(min(h, w)):
                mask[k, k] = True
        else:
            for k in range(min(h, w)):
                mask[k, w - 1 - k] = True
    else:
        cells = h * w
        count = int(RNG.integers(max(1, cells // 4), max(2, cells * 3 // 4 + 1)))
        picks = RNG.choice(cells, size=count, replace=False)
        mask.flat[picks] = True
    if mask.all():
        mask[int(RNG.integers(0, h)), int(RNG.integers(0, w))] = False
    if not mask.any():
        mask[int(RNG.integers(0, h)), int(RNG.integers(0, w))] = True
    return mask


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, mask: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[mask] = int(color)


def _anchor_pair(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ac = (ac0 + ac1) / 2.0
    ar = (ar0 + ar1) / 2.0
    bc = (bc0 + bc1) / 2.0
    br = (br0 + br1) / 2.0
    dc = bc - ac
    dr = br - ar
    if abs(dc) >= abs(dr):
        ra = int(round(ar))
        rb = int(round(br))
        ra = min(max(ra, ar0 + 1), ar1 - 1)
        rb = min(max(rb, br0 + 1), br1 - 1)
        if dc >= 0:
            return (ra, ac1 + 1), (rb, bc0 - 1)
        return (ra, ac0 - 1), (rb, bc1 + 1)
    ca = int(round(ac))
    cb = int(round(bc))
    ca = min(max(ca, ac0 + 1), ac1 - 1)
    cb = min(max(cb, bc0 + 1), bc1 - 1)
    if dr >= 0:
        return (ar1 + 1, ca), (br0 - 1, cb)
    return (ar0 - 1, ca), (br1 + 1, cb)


def _bfs_path(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and 0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if blocked[start] or blocked[goal]:
        return None
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < h and 0 <= nxt[1] < w and not blocked[nxt] and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    return None


def _ring_boxes(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    base = np.linspace(-math.pi / 2, 3 * math.pi / 2, count, endpoint=False)
    base += float(RNG.uniform(-0.25, 0.25))
    base += RNG.uniform(-0.18, 0.18, size=count)
    order = np.argsort(base)
    angles = base[order]
    sizes = []
    boxes = []
    radius_r = h / 2.0 - 5.0
    radius_c = w / 2.0 - 5.0
    for theta in angles:
        fh = int(RNG.integers(5, 9))
        fw = int(RNG.integers(5, 9))
        rr = int(round(cr + radius_r * math.sin(theta)))
        cc2 = int(round(cc + radius_c * math.cos(theta)))
        r0 = rr - fh // 2
        c0 = cc2 - fw // 2
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
            return None
        sizes.append((fh, fw))
        boxes.append((r0, c0, r1, c1))
    if not _boxes_far_apart(boxes, gap=2):
        return None
    centers = [((r0 + r1) / 2.0, (c0 + c1) / 2.0) for r0, c0, r1, c1 in boxes]
    centroid = np.mean(np.asarray(centers, dtype=float), axis=0)
    dists = [abs(center[0] - centroid[0]) + abs(center[1] - centroid[1]) for center in centers]
    if min(dists) < min(h, w) / 5.0:
        return None
    return boxes


def _build_scene() -> np.ndarray | None:
    h = int(RNG.integers(22, 31))
    w = int(RNG.integers(22, 31))
    count = int(RNG.integers(4, 7))
    boxes = _ring_boxes(h, w, count)
    if boxes is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    colors = [int(c) for c in RNG.choice(np.asarray(PALETTE), size=count, replace=False)]
    for bbox, color in zip(boxes, colors):
        r0, c0, r1, c1 = bbox
        mask = _shape_mask(r1 - r0 - 1, c1 - c0 - 1)
        _draw_frame(grid, bbox, color, mask)

    occupied = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in boxes:
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    for i in range(count):
        start, goal = _anchor_pair(boxes[i], boxes[(i + 1) % count])
        blocked = occupied | (grid == BLUE)
        blocked[start] = False
        blocked[goal] = False
        path = _bfs_path(blocked, start, goal)
        if path is None or len(path) < 2:
            return None
        for r, c in path:
            if grid[r, c] != BLACK:
                return None
            grid[r, c] = BLUE
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_scene()
        if grid is None:
            continue
        out = main(grid)
        if np.array_equal(out == RED, grid == RED) and np.array_equal(out == BLUE, grid == BLUE):
            return grid
    raise RuntimeError("failed to generate a valid ring-of-frames puzzle")
