from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7
MASK_COLORS = [3, 4, 5, 6, 8, 9]

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _angle_key(center: tuple[float, float], centroid: tuple[float, float]) -> float:
    dr = float(center[0] - centroid[0])
    dc = float(center[1] - centroid[1])
    return math.atan2(dc, -dr) % (2.0 * math.pi)


def _frame_records(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == RED):
        r0, c0, r1, c1 = _bbox(coords)
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1 and side >= 3, "red component must be a square frame"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((side, side), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(border.sum()) == int(coords.shape[0]), "red frame must occupy the full border"
        assert np.all(patch[border] == RED), "frame border must be red"
        assert np.all(patch[~border] != RED), "frame interior must exclude red"

        inner = patch[1:-1, 1:-1]
        colors = set(map(int, np.unique(inner))) - {BLACK}
        assert len(colors) == 1, "each frame must contain exactly one non-black interior color"
        color = int(next(iter(colors)))
        mask = inner == color
        assert mask.any(), "interior mask cannot be empty"
        assert len(_components(mask)) == 1, "interior mask must be 4-connected"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "color": color,
                "mask": mask,
            }
        )

    assert 4 <= len(frames) <= 5, "expected 4-5 red frames"
    return frames


def _blue_cycle_edges(grid: np.ndarray, frames: list[dict]) -> set[tuple[int, int]]:
    red_to_frame: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]
        for c in range(c0, c1 + 1):
            red_to_frame[(r0, c)] = idx
            red_to_frame[(r1, c)] = idx
        for r in range(r0, r1 + 1):
            red_to_frame[(r, c0)] = idx
            red_to_frame[(r, c1)] = idx

    edges: set[tuple[int, int]] = set()
    for coords in _components(grid == BLUE):
        touched: set[int] = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nbr = red_to_frame.get((int(r + dr), int(c + dc)))
                if nbr is not None:
                    touched.add(int(nbr))
        assert len(touched) == 2, "each blue connector must touch exactly two frames"
        a, b = sorted(touched)
        edges.add((a, b))

    assert len(edges) == len(frames), "cycle must have one connector per frame adjacency"
    degrees = [0] * len(frames)
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    assert all(d == 2 for d in degrees), "each frame must have degree two in the blue cycle"
    return edges


def _frame_order(frames: list[dict]) -> list[int]:
    centroid = (
        sum(frame["center"][0] for frame in frames) / len(frames),
        sum(frame["center"][1] for frame in frames) / len(frames),
    )
    angles = [_angle_key(frame["center"], centroid) for frame in frames]
    assert len({round(a, 9) for a in angles}) == len(angles), "frame angles must be distinct"
    return sorted(range(len(frames)), key=lambda i: angles[i])


def _count_orange_dots(grid: np.ndarray) -> int:
    count = 0
    for coords in _components(grid == ORANGE):
        assert int(coords.shape[0]) == 1, "orange marks must be isolated singletons"
        r, c = map(int, coords[0])
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc):
                assert int(grid[nr, nc]) == BLACK, "orange dots must sit in black background"
        count += 1
    assert count >= 1, "expected at least one orange dot"
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the interior mask colors of the red frame cycle by the number of orange background dots.

    Input structure:
    - The grid has a black background, 4-5 hollow red square frames, and blue connectors linking those frames into one cycle.
    - Each frame contains exactly one 4-connected non-black interior mask whose shape stays inside that frame.
    - A few isolated orange singleton pixels are scattered in the remaining black background.

    Rule:
    - Count the orange background dots to obtain a step count N.
    - Sort the red frames by the angle of their centers around the shared centroid, starting from the top and moving clockwise.
    - Keep every frame border, blue connector, black cell, and orange dot fixed, but recolor each frame's interior mask with the original mask color from the frame N positions clockwise ahead in that circular order.

    Key insight:
    - The orange dots are not decoration; their count parameterizes how far the frame-interior colors rotate around the geometric ring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _frame_records(g)
    _blue_cycle_edges(g, frames)
    order = _frame_order(frames)
    step = _count_orange_dots(g)
    assert step < len(order), "rotation step must be smaller than the number of frames"

    out = g.copy()
    for pos, frame_id in enumerate(order):
        src_id = order[(pos + step) % len(order)]
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[frames[frame_id]["mask"]] = int(frames[src_id]["color"])
    return out


def _box_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _grow_mask(inner_side: int, area: int) -> np.ndarray | None:
    mask = np.zeros((inner_side, inner_side), dtype=bool)
    start = (random.randrange(inner_side), random.randrange(inner_side))
    mask[start] = True
    cells = [start]
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < inner_side and 0 <= nc < inner_side and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        cells.append(cell)
        add_frontier(cell)
    if len(cells) != area:
        return None
    return mask


def _generate_unique_masks(frame_sizes: list[int]) -> list[np.ndarray] | None:
    used: set[tuple[bytes, tuple[int, int]]] = set()
    masks: list[np.ndarray] = []
    for side in frame_sizes:
        inner = side - 2
        max_area = min(6, inner * inner)
        min_area = min(3, max_area)
        choices = list(range(min_area, max_area + 1))
        random.shuffle(choices)
        mask = None
        for area in choices:
            for _ in range(40):
                candidate = _grow_mask(inner, area)
                if candidate is None:
                    continue
                key = (candidate.tobytes(), tuple(candidate.shape))
                if key in used:
                    continue
                used.add(key)
                mask = candidate
                break
            if mask is not None:
                break
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, mask: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = RED
    grid[r0 : r1 + 1, c0] = RED
    grid[r0 : r1 + 1, c1] = RED
    rr, cc = np.argwhere(mask).T
    grid[r0 + 1 + rr, c0 + 1 + cc] = int(color)


def _side_for_delta(dr: float, dc: float) -> str:
    if abs(dr) >= abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _attachment_candidates(
    bbox: tuple[int, int, int, int],
    side: str,
    target_center: tuple[float, float],
    shape: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    target_r, target_c = target_center
    pairs: list[tuple[tuple[int, int], tuple[int, int], float]] = []

    if side in ("top", "bottom"):
        row = r0 if side == "top" else r1
        outside_row = row - 1 if side == "top" else row + 1
        for c in range(c0 + 1, c1):
            outside = (outside_row, c)
            if _in_bounds(shape, outside[0], outside[1]):
                score = abs(c - target_c)
                pairs.append(((row, c), outside, score))
    else:
        col = c0 if side == "left" else c1
        outside_col = col - 1 if side == "left" else col + 1
        for r in range(r0 + 1, r1):
            outside = (r, outside_col)
            if _in_bounds(shape, outside[0], outside[1]):
                score = abs(r - target_r)
                pairs.append(((r, col), outside, score))

    pairs.sort(key=lambda item: item[2])
    return [(border, outside) for border, outside, _ in pairs]


def _halo_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(shape, nr, nc) and (nr, nc) not in cells:
                blocked.add((nr, nc))
    return blocked


def _bfs_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if start == end:
        return [start]
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not _in_bounds(shape, nr, nc):
                continue
            if nxt in prev:
                continue
            if nxt != end and (nxt in occupied or nxt in blocked):
                continue
            prev[nxt] = (r, c)
            if nxt == end:
                path = [end]
                cur = end
                while prev[cur] is not None:
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(nxt)
    return None


def _connect_frames(
    shape: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    order: list[int],
    red_cells: set[tuple[int, int]],
) -> list[list[tuple[int, int]]] | None:
    blue_cells: set[tuple[int, int]] = set()
    connectors: list[list[tuple[int, int]]] = []
    centers = [((r0 + r1) / 2.0, (c0 + c1) / 2.0) for r0, c0, r1, c1 in bboxes]

    for i, a in enumerate(order):
        b = order[(i + 1) % len(order)]
        ar, ac = centers[a]
        br, bc = centers[b]
        side_a = _side_for_delta(br - ar, bc - ac)
        side_b = _side_for_delta(ar - br, ac - bc)

        cand_a = _attachment_candidates(bboxes[a], side_a, centers[b], shape)
        cand_b = _attachment_candidates(bboxes[b], side_b, centers[a], shape)
        if not cand_a or not cand_b:
            return None

        occupied = set(red_cells) | set(blue_cells)
        blocked_base = _halo_cells(occupied, shape)

        found_path = None
        for _border_a, start in cand_a:
            if start in occupied:
                continue
            for _border_b, end in cand_b:
                if end in occupied or start == end:
                    continue
                blocked = set(blocked_base)
                blocked.discard(start)
                blocked.discard(end)
                path = _bfs_path(shape, start, end, blocked, occupied)
                if path is None or len(path) < 2:
                    continue
                found_path = path
                break
            if found_path is not None:
                break

        if found_path is None:
            return None
        connectors.append(found_path)
        blue_cells.update(found_path)

    return connectors


def _place_orange_dots(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    count: int,
) -> bool:
    candidates: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in occupied or int(grid[r, c]) != BLACK:
                continue
            ok = True
            for dr, dc in DIR8:
                nr, nc = r + dr, c + dc
                if int(grid[nr, nc]) != BLACK:
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                break
    if len(chosen) != count:
        return False
    for r, c in chosen:
        grid[r, c] = ORANGE
        occupied.add((r, c))
    return True


def _expected_output(grid: np.ndarray) -> np.ndarray:
    frames = _frame_records(grid)
    order = _frame_order(frames)
    step = _count_orange_dots(grid)
    out = grid.copy()
    for pos, frame_id in enumerate(order):
        src_id = order[(pos + step) % len(order)]
        r0, c0, r1, c1 = frames[frame_id]["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[frames[frame_id]["mask"]] = int(frames[src_id]["color"])
    return out


def _generate_layout() -> tuple[int, int, list[tuple[int, int, int, int]], list[int], list[np.ndarray], int] | None:
    n_frames = random.choice([4, 5])
    dot_count = random.randint(1, n_frames - 1)
    h = random.randint(22, 29)
    w = random.randint(22, 29)
    frame_sizes = [random.choice([4, 5]) for _ in range(n_frames)]
    masks = _generate_unique_masks(frame_sizes)
    if masks is None:
        return None

    colors = random.sample(MASK_COLORS, n_frames)
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    radius_r = random.uniform(h * 0.27, h * 0.35)
    radius_c = random.uniform(w * 0.27, w * 0.35)
    base_shift = random.uniform(0.0, 2.0 * math.pi)
    gap = 2.0 * math.pi / n_frames

    bboxes: list[tuple[int, int, int, int]] = []
    for i, side in enumerate(frame_sizes):
        angle = base_shift + i * gap + random.uniform(-0.18, 0.18)
        rr = center_r - radius_r * math.cos(angle)
        cc = center_c + radius_c * math.sin(angle)
        r0 = int(round(rr - side / 2.0))
        c0 = int(round(cc - side / 2.0))
        r1 = r0 + side - 1
        c1 = c0 + side - 1
        if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
            return None
        bboxes.append((r0, c0, r1, c1))

    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            if _box_overlaps(bboxes[i], bboxes[j], pad=2):
                return None

    return h, w, bboxes, colors, masks, dot_count


def _gen_attempt() -> np.ndarray | None:
    layout = _generate_layout()
    if layout is None:
        return None
    h, w, bboxes, colors, masks, dot_count = layout
    grid = np.zeros((h, w), dtype=int)

    red_cells: set[tuple[int, int]] = set()
    for bbox, color, mask in zip(bboxes, colors, masks):
        _draw_frame(grid, bbox, color, mask)
        r0, c0, r1, c1 = bbox
        for c in range(c0, c1 + 1):
            red_cells.add((r0, c))
            red_cells.add((r1, c))
        for r in range(r0, r1 + 1):
            red_cells.add((r, c0))
            red_cells.add((r, c1))

    frames = _frame_records(grid)
    order = _frame_order(frames)
    connectors = _connect_frames(grid.shape, bboxes, order, red_cells)
    if connectors is None:
        return None

    blue_cells: set[tuple[int, int]] = set()
    for path in connectors:
        for r, c in path:
            if int(grid[r, c]) != BLACK:
                return None
            grid[r, c] = BLUE
            blue_cells.add((r, c))

    occupied = set(red_cells) | blue_cells
    if not _place_orange_dots(grid, occupied, dot_count):
        return None

    try:
        pred = main(grid)
    except Exception:
        return None

    exp = _expected_output(grid)
    if not np.array_equal(pred, exp):
        return None
    if np.array_equal(pred, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
