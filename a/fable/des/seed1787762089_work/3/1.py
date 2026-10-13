import math
import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

FRAME_SIZE = 5
INTERIOR_SIZE = 3
FRAME_BORDER_AREA = FRAME_SIZE * 4 - 4
INTERIOR_COLORS = (GREEN, YELLOW, MAGENTA, ORANGE, MAROON)

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GRID_H_RANGE = (22, 29)
GRID_W_RANGE = (22, 29)
FRAME_COUNT_CHOICES = (4, 5)
ANGLE_PERTURB = 0.26
FRAME_GAP = 1
MAX_CONNECTOR_LEN = 28
GENERATION_ATTEMPTS = 400


def _get_bg(g: np.ndarray) -> int:
    counts = np.bincount(g.ravel().astype(int), minlength=10)
    return int(np.argmax(counts))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            pts = []
            while queue:
                cr, cc = queue.popleft()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_border_mask() -> np.ndarray:
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


FRAME_BORDER_MASK = _frame_border_mask()


def _extract_frames(g: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], int]]:
    frames = []
    border_to_frame: dict[tuple[int, int], int] = {}

    for coords in _components_mask(g == GREY, connectivity=4):
        if int(coords.shape[0]) != FRAME_BORDER_AREA:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (FRAME_SIZE, FRAME_SIZE):
            continue

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[FRAME_BORDER_MASK] == GREY):
            continue
        if np.any(patch[~FRAME_BORDER_MASK] == GREY):
            continue

        interior = patch[1:-1, 1:-1]
        colors = set(map(int, np.unique(interior))) - {bg}
        assert len(colors) == 1, "each frame interior must use exactly one foreground color"
        color = colors.pop()
        assert color not in (GREY, AZURE, RED), "interior color must differ from frame, connector, and control colors"

        mask = interior == color
        pixels = int(mask.sum())
        assert 2 <= pixels <= 6, "interior shape must have 2-6 pixels"

        frame_id = len(frames)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "color": int(color),
                "mask": mask.copy(),
            }
        )
        for r, c in coords:
            border_to_frame[(int(r), int(c))] = frame_id

    assert 4 <= len(frames) <= 6, "expected 4-6 grey frames"
    return frames, border_to_frame


def _cycle_edges(g: np.ndarray, frames: list[dict], border_to_frame: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    edges = set()
    for coords in _components_mask(g == AZURE, connectivity=4):
        touched = set()
        for r, c in coords:
            rr, cc = int(r), int(c)
            for dr, dc in DIR4:
                frame_id = border_to_frame.get((rr + dr, cc + dc))
                if frame_id is not None:
                    touched.add(frame_id)
        assert len(touched) == 2, "each azure connector must touch exactly two frames"
        a, b = sorted(touched)
        edges.add((a, b))

    assert len(edges) == len(frames), "cycle must contain one connector per frame pair"
    for frame_id in range(len(frames)):
        degree = sum(frame_id in edge for edge in edges)
        assert degree == 2, "each frame must have degree two in the connector cycle"
    return edges


def _clockwise_order(frames: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    mean_r = sum(frame["center"][0] for frame in frames) / len(frames)
    mean_c = sum(frame["center"][1] for frame in frames) / len(frames)
    angles = [
        math.atan2(frame["center"][0] - mean_r, frame["center"][1] - mean_c)
        for frame in frames
    ]
    rounded = {round(angle, 12) for angle in angles}
    assert len(rounded) == len(frames), "frame centers must have distinct angles"

    order = sorted(range(len(frames)), key=lambda idx: angles[idx])
    for a, b in zip(order, order[1:] + order[:1]):
        assert tuple(sorted((a, b))) in edges, "angle order must follow the azure cycle"
    return order


def _count_control_pixels(g: np.ndarray) -> int:
    comps = _components_mask(g == RED, connectivity=8)
    assert comps, "expected at least one red control pixel"
    for coords in comps:
        assert int(coords.shape[0]) == 1, "red controls must be isolated single pixels"
    count = len(comps)
    assert 1 <= count <= 3, "control count must be between 1 and 3"
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate frame-interior colors clockwise around the azure cycle by the number of isolated red pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frames, border_to_frame = _extract_frames(g, bg)
    edges = _cycle_edges(g, frames, border_to_frame)
    order = _clockwise_order(frames, edges)
    shift = _count_control_pixels(g)

    out = g.copy()
    out[g == RED] = bg
    for idx, frame_id in enumerate(order):
        src_id = order[(idx + shift) % len(order)]
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[frames[frame_id]["mask"]] = frames[src_id]["color"]
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], border_to_frame: dict[tuple[int, int], int], frame_id: int, occupied: set[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for c in range(c0, c1 + 1):
        occupied.add((r0, c))
        occupied.add((r1, c))
        border_to_frame[(r0, c)] = frame_id
        border_to_frame[(r1, c)] = frame_id
    for r in range(r0, r1 + 1):
        occupied.add((r, c0))
        occupied.add((r, c1))
        border_to_frame[(r, c0)] = frame_id
        border_to_frame[(r, c1)] = frame_id


def _grow_shape_mask(rng: random.Random) -> np.ndarray | None:
    target = rng.randint(2, 6)
    mask = np.zeros((INTERIOR_SIZE, INTERIOR_SIZE), dtype=bool)
    start = (rng.randrange(INTERIOR_SIZE), rng.randrange(INTERIOR_SIZE))
    mask[start] = True
    frontier = {start}

    while int(mask.sum()) < target and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.discard(cell)
        for dr, dc in DIR4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < INTERIOR_SIZE and 0 <= nc < INTERIOR_SIZE and not bool(mask[nr, nc]):
                frontier.add((nr, nc))
        if cell != start and bool(mask[cell]):
            continue
        mask[cell] = True
        for dr, dc in DIR4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < INTERIOR_SIZE and 0 <= nc < INTERIOR_SIZE and not bool(mask[nr, nc]):
                frontier.add((nr, nc))

    if int(mask.sum()) != target:
        return None
    return mask


def _place_mask(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, occupied: set[tuple[int, int]], rng: random.Random) -> np.ndarray | None:
    mask = _grow_shape_mask(rng)
    if mask is None:
        return None
    r0, c0, _, _ = bbox
    rows, cols = np.nonzero(mask)
    grid[r0 + 1 + rows, c0 + 1 + cols] = color
    occupied.update((r0 + 1 + int(r), c0 + 1 + int(c)) for r, c in zip(rows, cols))
    return mask


def _frame_bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    r0a, c0a, r1a, c1a = a
    r0b, c0b, r1b, c1b = b
    return not (
        r1a + FRAME_GAP < r0b
        or r1b + FRAME_GAP < r0a
        or c1a + FRAME_GAP < c0b
        or c1b + FRAME_GAP < c0a
    )


def _place_frame_bboxes(n_frames: int, grid_h: int, grid_w: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    center_r = grid_h / 2.0
    center_c = grid_w / 2.0
    base_angles = [2.0 * math.pi * i / n_frames + rng.uniform(-0.18, 0.18) for i in range(n_frames)]
    angles = sorted(base_angles)

    min_radius = FRAME_SIZE + 1.0
    max_radius = min(grid_h, grid_w) / 2.0 - FRAME_SIZE / 2.0 - 2.0
    if max_radius <= min_radius:
        return None

    boxes = []
    for angle in angles:
        radius = rng.uniform(min_radius, max_radius)
        top = int(round(center_r + radius * math.sin(angle) - FRAME_SIZE / 2.0))
        left = int(round(center_c + radius * math.cos(angle) - FRAME_SIZE / 2.0))
        top = max(0, min(grid_h - FRAME_SIZE, top))
        left = max(0, min(grid_w - FRAME_SIZE, left))
        boxes.append((top, left, top + FRAME_SIZE - 1, left + FRAME_SIZE - 1))

    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            if _frame_bboxes_overlap(boxes[i], boxes[j]):
                return None
    return boxes


def _clockwise_order_from_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[list[int], list[tuple[float, float]]] | tuple[None, None]:
    centers = [((r0 + r1) / 2.0, (c0 + c1) / 2.0) for r0, c0, r1, c1 in boxes]
    mean_r = sum(r for r, _ in centers) / len(centers)
    mean_c = sum(c for _, c in centers) / len(centers)
    angles = [math.atan2(r - mean_r, c - mean_c) for r, c in centers]
    if len({round(angle, 12) for angle in angles}) != len(angles):
        return None, None
    order = sorted(range(len(boxes)), key=lambda idx: angles[idx])
    return order, centers


def _best_side(bbox: tuple[int, int, int, int], target_r: float, target_c: float) -> str:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = target_r - center_r
    dc = target_c - center_c
    if abs(dr) > abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _side_positions(bbox: tuple[int, int, int, int], side: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    pairs = []
    if side == "top":
        for c in range(c0 + 1, c1):
            pairs.append(((r0, c), (r0 - 1, c)))
    elif side == "bottom":
        for c in range(c0 + 1, c1):
            pairs.append(((r1, c), (r1 + 1, c)))
    elif side == "left":
        for r in range(r0 + 1, r1):
            pairs.append(((r, c0), (r, c0 - 1)))
    elif side == "right":
        for r in range(r0 + 1, r1):
            pairs.append(((r, c1), (r, c1 + 1)))
    return pairs


def _attachment_candidates(
    bbox: tuple[int, int, int, int],
    frame_id: int,
    target: tuple[float, float],
    grid_h: int,
    grid_w: int,
    occupied_static: set[tuple[int, int]],
    connector_cells: set[tuple[int, int]],
    border_to_frame: dict[tuple[int, int], int],
    rng: random.Random,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    preferred = _best_side(bbox, target[0], target[1])
    sides = ["top", "right", "bottom", "left"]
    sides.remove(preferred)
    rng.shuffle(sides)
    side_order = [preferred] + sides

    candidates = []
    for side in side_order:
        side_cells = _side_positions(bbox, side)
        rng.shuffle(side_cells)
        for border, outside in side_cells:
            r, c = outside
            if not (0 <= r < grid_h and 0 <= c < grid_w):
                continue
            if outside in occupied_static or outside in connector_cells:
                continue
            if any((r + dr, c + dc) in connector_cells for dr, dc in DIR4):
                continue
            touched = {
                border_to_frame[(r + dr, c + dc)]
                for dr, dc in DIR4
                if (r + dr, c + dc) in border_to_frame
            }
            if touched != {frame_id}:
                continue
            candidates.append((border, outside))
    return candidates


def _touching_frames(cell: tuple[int, int], border_to_frame: dict[tuple[int, int], int]) -> set[int]:
    r, c = cell
    touched = set()
    for dr, dc in DIR4:
        frame_id = border_to_frame.get((r + dr, c + dc))
        if frame_id is not None:
            touched.add(frame_id)
    return touched


def _bfs_path(
    start: tuple[int, int],
    end: tuple[int, int],
    start_frame: int,
    end_frame: int,
    grid_h: int,
    grid_w: int,
    occupied_static: set[tuple[int, int]],
    connector_cells: set[tuple[int, int]],
    border_to_frame: dict[tuple[int, int], int],
) -> list[tuple[int, int]] | None:
    endpoint_frames = {start: start_frame, end: end_frame}

    def allowed(cell: tuple[int, int]) -> bool:
        r, c = cell
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return False
        if cell in occupied_static or cell in connector_cells:
            return False
        for dr, dc in DIR4:
            if (r + dr, c + dc) in connector_cells:
                return False
        touched = _touching_frames(cell, border_to_frame)
        if touched:
            allowed_frame = endpoint_frames.get(cell)
            return allowed_frame is not None and touched == {allowed_frame}
        return True

    if not allowed(start) or not allowed(end):
        return None

    queue = deque([start])
    prev = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == end:
            break
        r, c = cell
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in prev or not allowed(nxt):
                continue
            prev[nxt] = cell
            queue.append(nxt)

    if end not in prev:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    if len(path) > MAX_CONNECTOR_LEN:
        return None
    return path


def _draw_path(grid: np.ndarray, path: list[tuple[int, int]], connector_cells: set[tuple[int, int]]) -> None:
    for r, c in path:
        grid[r, c] = AZURE
        connector_cells.add((r, c))


def _connect_cycle(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    centers: list[tuple[float, float]],
    order: list[int],
    occupied_static: set[tuple[int, int]],
    border_to_frame: dict[tuple[int, int], int],
    rng: random.Random,
) -> bool:
    grid_h, grid_w = grid.shape
    connector_cells: set[tuple[int, int]] = set()

    for idx, frame_a in enumerate(order):
        frame_b = order[(idx + 1) % len(order)]
        cand_a = _attachment_candidates(
            boxes[frame_a],
            frame_a,
            centers[frame_b],
            grid_h,
            grid_w,
            occupied_static,
            connector_cells,
            border_to_frame,
            rng,
        )
        cand_b = _attachment_candidates(
            boxes[frame_b],
            frame_b,
            centers[frame_a],
            grid_h,
            grid_w,
            occupied_static,
            connector_cells,
            border_to_frame,
            rng,
        )
        if not cand_a or not cand_b:
            return False

        found = None
        for border_a, start_a in cand_a[:8]:
            for border_b, start_b in cand_b[:8]:
                if start_a == start_b:
                    continue
                path = _bfs_path(
                    start_a,
                    start_b,
                    frame_a,
                    frame_b,
                    grid_h,
                    grid_w,
                    occupied_static,
                    connector_cells,
                    border_to_frame,
                )
                if path is None or len(path) < 2:
                    continue
                found = path
                break
            if found is not None:
                break
        if found is None:
            return False
        _draw_path(grid, found, connector_cells)
    return True


def _corner_cells(corner: str, grid_h: int, grid_w: int, size: int = 5) -> list[tuple[int, int]]:
    if corner == "TL":
        rows = range(size)
        cols = range(size)
    elif corner == "TR":
        rows = range(size)
        cols = range(grid_w - size, grid_w)
    elif corner == "BL":
        rows = range(grid_h - size, grid_h)
        cols = range(size)
    else:
        rows = range(grid_h - size, grid_h)
        cols = range(grid_w - size, grid_w)
    return [(r, c) for r in rows for c in cols]


def _choose_red_positions(grid_h: int, grid_w: int, occupied: set[tuple[int, int]], k: int, rng: random.Random) -> list[tuple[int, int]] | None:
    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    for corner in corners:
        candidates = []
        for cell in _corner_cells(corner, grid_h, grid_w):
            r, c = cell
            if cell in occupied:
                continue
            blocked = False
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w and (nr, nc) in occupied:
                    blocked = True
                    break
            if not blocked:
                candidates.append(cell)

        rng.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for cell in candidates:
            if all(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) > 1 for other in chosen):
                chosen.append(cell)
                if len(chosen) == k:
                    return chosen
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    n_frames = rng.choice(FRAME_COUNT_CHOICES)
    colors = list(INTERIOR_COLORS)
    rng.shuffle(colors)
    colors = colors[:n_frames]

    boxes = _place_frame_bboxes(n_frames, grid_h, grid_w, rng)
    if boxes is None:
        return None

    order, centers = _clockwise_order_from_boxes(boxes)
    if order is None or centers is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied_static: set[tuple[int, int]] = set()
    border_to_frame: dict[tuple[int, int], int] = {}

    for frame_id, bbox in enumerate(boxes):
        _draw_frame(grid, bbox, border_to_frame, frame_id, occupied_static)
        if _place_mask(grid, bbox, colors[frame_id], occupied_static, rng) is None:
            return None

    if not _connect_cycle(grid, boxes, centers, order, occupied_static, border_to_frame, rng):
        return None

    occupied_all = {(int(r), int(c)) for r, c in zip(*np.nonzero(grid != BLACK))}
    k = rng.randint(1, min(3, n_frames - 1))
    red_positions = _choose_red_positions(grid_h, grid_w, occupied_all, k, rng)
    if red_positions is None:
        return None
    for r, c in red_positions:
        grid[r, c] = RED

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert np.all(out[grid == RED] == BLACK)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
