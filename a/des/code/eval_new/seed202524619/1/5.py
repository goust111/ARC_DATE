import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


_ARROW_PATTERNS: Dict[str, np.ndarray] = {
    "R": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
    "L": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    "U": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    "D": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
}
_ARROW_OFFSETS: Dict[str, List[Tuple[int, int]]] = {
    d: [(int(r), int(c)) for r, c in zip(*np.where(mask))] for d, mask in _ARROW_PATTERNS.items()
}
_DIR_TO_DELTA = {"L": (0, -1), "R": (0, 1), "U": (-1, 0), "D": (1, 0)}


def _max_run_length_1d(line: np.ndarray) -> int:
    best = 0
    run = 0
    for v in line:
        if v:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _max_run_length(mask: np.ndarray, axis: int) -> int:
    best = 0
    if axis == 0:
        # columns
        for col in range(mask.shape[1]):
            best = max(best, _max_run_length_1d(mask[:, col]))
    else:
        for row in range(mask.shape[0]):
            best = max(best, _max_run_length_1d(mask[row, :]))
    return best


def _infer_wall_color(grid: np.ndarray, background: int) -> int:
    values, counts = np.unique(grid, return_counts=True)
    candidates: List[Tuple[float, int]] = []
    for v, c in zip(values, counts):
        v = int(v)
        if v == background:
            continue
        m = grid == v
        score = float(c)
        score += _max_run_length(m, axis=0) + _max_run_length(m, axis=1)
        candidates.append((score, v))
    if not candidates:
        return background
    return max(candidates, key=lambda x: x[0])[1]


def _connected_components(mask: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int, int, int]]]:
    h, w = mask.shape
    room_id = -np.ones((h, w), dtype=np.int32)
    comps: List[np.ndarray] = []
    bboxes: List[Tuple[int, int, int, int]] = []
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or room_id[r, c] != -1:
                continue
            q = deque([(r, c)])
            room_id[r, c] = cid
            cells: List[Tuple[int, int]] = []
            r0 = h
            r1 = -1
            c0 = w
            c1 = -1
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                if cr < r0:
                    r0 = cr
                if cr > r1:
                    r1 = cr
                if cc < c0:
                    c0 = cc
                if cc > c1:
                    c1 = cc
                for dr, dc in dirs:
                    nr = cr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if not mask[nr, nc] or room_id[nr, nc] != -1:
                        continue
                    room_id[nr, nc] = cid
                    q.append((nr, nc))
            comps.append(np.array(cells, dtype=np.int32))
            bboxes.append((r0, r1, c0, c1))
            cid += 1
    return room_id, comps, bboxes


def _find_arrow_in_room(
    grid: np.ndarray,
    room_mask: np.ndarray,
    room_bbox: Tuple[int, int, int, int],
    background: int,
    wall_color: int,
) -> Optional[Tuple[str, int, List[Tuple[int, int]]]]:
    r0, r1, c0, c1 = room_bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    region = grid[room_mask]
    vals, counts = np.unique(region, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(vals, counts)}
    candidates = [v for v in color_counts if v not in (background, wall_color)]
    if not candidates:
        return None
    # Most stable: 5-cell arrow color in room
    ordered_candidates = sorted(candidates, key=lambda c: (color_counts[c] == 5, color_counts[c]), reverse=True)
    for color in ordered_candidates:
        for rr in range(r0, r1 - 1):
            for cc in range(c0, c1 - 1):
                sub = (grid[rr : rr + 3, cc : cc + 3] == color)
                if np.count_nonzero(sub) != 5:
                    continue
                for d, mask in _ARROW_PATTERNS.items():
                    if np.array_equal(sub[mask], np.ones(int(mask.sum()), dtype=bool)):
                        pos = [(rr + int(r), cc + int(c)) for r, c in zip(*np.where(mask))]
                        return d, color, pos
    return None


def _neighbor_room(room_id: np.ndarray, room_bbox: Tuple[int, int, int, int], direction: str, grid: np.ndarray, wall_color: int) -> Optional[int]:
    h, w = grid.shape
    r0, r1, c0, c1 = room_bbox
    candidates = set()
    if direction == "R":
        wall_col = c1 + 1
        if wall_col < w:
            nxt = wall_col + 1
            if nxt < w:
                for r in range(r0, r1 + 1):
                    if grid[r, wall_col] == wall_color and room_id[r, nxt] >= 0:
                        candidates.add(int(room_id[r, nxt]))
    elif direction == "L":
        wall_col = c0 - 1
        nxt = wall_col - 1
        if wall_col >= 0 and nxt >= 0:
            for r in range(r0, r1 + 1):
                if grid[r, wall_col] == wall_color and room_id[r, nxt] >= 0:
                    candidates.add(int(room_id[r, nxt]))
    elif direction == "D":
        wall_row = r1 + 1
        nxt = wall_row + 1
        if wall_row < h and nxt < h:
            for c in range(c0, c1 + 1):
                if grid[wall_row, c] == wall_color and room_id[nxt, c] >= 0:
                    candidates.add(int(room_id[nxt, c]))
    else:  # "U"
        wall_row = r0 - 1
        nxt = wall_row - 1
        if wall_row >= 0 and nxt >= 0:
            for c in range(c0, c1 + 1):
                if grid[wall_row, c] == wall_color and room_id[nxt, c] >= 0:
                    candidates.add(int(room_id[nxt, c]))
    if len(candidates) == 1:
        return candidates.pop()
    return None


def _arrow_push_translation(
    src_bbox: Tuple[int, int, int, int],
    dst_bbox: Tuple[int, int, int, int],
    direction: str,
) -> Tuple[int, int]:
    sr0, sr1, sc0, sc1 = src_bbox
    dr = dc = 0
    tr0, tr1, tc0, tc1 = dst_bbox
    if direction == "R":
        dc = tc0 - sc1
    elif direction == "L":
        dc = tc1 - sc0
    elif direction == "D":
        dr = tr0 - sr1
    else:
        dr = tr1 - sr0
    return dr, dc


def _draw_arrow(grid: np.ndarray, top_left: Tuple[int, int], direction: str, color: int) -> None:
    r0, c0 = top_left
    for dr, dc in _ARROW_OFFSETS[direction]:
        grid[r0 + dr, c0 + dc] = color


def _room_has_collision(
    grid: np.ndarray,
    background: int,
    wall_color: int,
) -> bool:
    room_id, comps, bboxes = _connected_components(grid != wall_color)
    used: set = set()
    for i, room_cells in enumerate(comps):
        room_mask = room_id == i
        found = _find_arrow_in_room(grid, room_mask, bboxes[i], background, wall_color)
        if found is None:
            continue
        direction, arrow_color, _ = found
        cargo_mask = room_mask & (grid != background) & (grid != wall_color) & (grid != arrow_color)
        cargo = np.argwhere(cargo_mask)
        if cargo.size == 0:
            continue
        nbr = _neighbor_room(room_id, bboxes[i], direction, grid, wall_color)
        if nbr is None:
            continue
        dr, dc = _arrow_push_translation(bboxes[i], bboxes[nbr], direction)
        for r, c in cargo:
            nr = int(r + dr)
            nc = int(c + dc)
            if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
                return True
            if (nr, nc) in used:
                return True
            used.add((nr, nc))
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push all room-contained cargos through directional arrow walls into neighboring rooms.

    Input structure:
    - A black background fills most cells.
    - A single wall color (line-width 1) forms full horizontal and vertical separators, partitioning non-wall cells into rectangular rooms.
    - Each room contains one 3×3 orange arrow (up/down/left/right) and zero or more cargo pixels in other non-black/non-wall colors.

    Rule:
    - For each room, detect the arrow orientation and interpret it as the push direction.
    - All cargo pixels in that room (excluding the arrow itself) are translated in that direction into the adjacent room across the separating wall, with a minimal shift that moves them over the wall and places them touching the target room’s entry wall.
    - All rooms push simultaneously; output pixels in each room are the overlay of all incoming cargos.
    - Arrow pixels are cleared to black, and wall pixels stay unchanged.

    Key insight:
    - The wall grid defines independent source/target rooms, so each room’s final contents are determined by neighboring rooms’ arrows, not by its own original cargo.
    """
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("main() expects a 2D grid.")

    h, w = g.shape
    if h == 0 or w == 0:
        return np.array(g, copy=True)

    background = int(np.bincount(g.ravel()).argmax())
    wall_color = _infer_wall_color(g, background)

    room_id, comps, bboxes = _connected_components(g != wall_color)
    out = np.full_like(g, background)
    out[g == wall_color] = wall_color

    for i, room_cells in enumerate(comps):
        room_mask = room_id == i
        found = _find_arrow_in_room(g, room_mask, bboxes[i], background, wall_color)
        if found is None:
            continue
        direction, arrow_color, _ = found
        cargo_mask = room_mask & (g != background) & (g != wall_color) & (g != arrow_color)
        cargo = np.argwhere(cargo_mask)
        if cargo.size == 0:
            continue
        nbr = _neighbor_room(room_id, bboxes[i], direction, g, wall_color)
        if nbr is None:
            continue
        dr, dc = _arrow_push_translation(bboxes[i], bboxes[nbr], direction)
        for r, c in cargo:
            nr = int(r + dr)
            nc = int(c + dc)
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            out[nr, nc] = int(g[r, c])
    return out


def _random_room_dimensions(rng: np.random.Generator, n: int, min_size: int = 3, max_size: int = 8) -> List[int]:
    while True:
        dims = [int(rng.integers(min_size, max_size + 1)) for _ in range(n)]
        if min(dims) >= min_size:
            return dims


def _place_cargo_block(
    rng: np.random.Generator,
    grid: np.ndarray,
    room_bbox: Tuple[int, int, int, int],
    arrow_pixels: set,
    color: int,
) -> bool:
    r0, r1, c0, c1 = room_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    shapes = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]
    for _ in range(50):
        shape = shapes[int(rng.integers(0, len(shapes)))]
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if sh > h or sw > w:
            continue
        pr = int(rng.integers(r0, r1 - sh + 2))
        pc = int(rng.integers(c0, c1 - sw + 2))
        cells = [(pr + dr, pc + dc) for dr, dc in shape]
        if any((r, c) in arrow_pixels for r, c in cells):
            continue
        ok = True
        for r, c in cells:
            if r < r0 or r > r1 or c < c0 or c > c1:
                ok = False
                break
            if (r, c) in arrow_pixels:
                ok = False
                break
        if not ok:
            continue
        for r, c in cells:
            grid[r, c] = color
            arrow_pixels.add((r, c))
        return True
    return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    background = 0
    wall_color = 1
    arrow_color = 7
    cargo_colors = [2, 3, 4, 5, 6, 8, 9, 10, 11]
    for _ in range(1000):
        row_parts = int(rng.integers(2, 5))
        col_parts = int(rng.integers(2, 5))
        heights = _random_room_dimensions(rng, row_parts, 3, 8)
        widths = _random_room_dimensions(rng, col_parts, 3, 8)

        H = sum(heights) + (row_parts - 1)
        W = sum(widths) + (col_parts - 1)
        if H > 30 or W > 30 or H <= 4 or W <= 4:
            continue

        grid = np.full((H, W), background, dtype=int)

        row_ranges = []
        cur = 0
        for i in range(row_parts):
            rs, re = cur, cur + heights[i] - 1
            row_ranges.append((rs, re))
            cur = re + 1
            if i < row_parts - 1:
                grid[cur, :] = wall_color
                cur += 1

        col_ranges = []
        cur = 0
        for j in range(col_parts):
            cs, ce = cur, cur + widths[j] - 1
            col_ranges.append((cs, ce))
            cur = ce + 1
            if j < col_parts - 1:
                grid[:, cur] = wall_color
                cur += 1

        for i, (rs, re) in enumerate(row_ranges):
            for j, (cs, ce) in enumerate(col_ranges):
                dirs = []
                if j + 1 < col_parts:
                    dirs.append("R")
                if j > 0:
                    dirs.append("L")
                if i + 1 < row_parts:
                    dirs.append("D")
                if i > 0:
                    dirs.append("U")
                if not dirs:
                    continue
                direction = str(rng.choice(dirs))

                ar = int(rng.integers(rs, re - 1))
                ac = int(rng.integers(cs, ce - 1))
                _draw_arrow(grid, (ar, ac), direction, arrow_color)

                occupied = {(ar + dr, ac + dc) for dr, dc in _ARROW_OFFSETS[direction]}
                u = float(rng.random())
                if u < 0.4:
                    target_cargo = 0
                elif u < 0.75:
                    target_cargo = 1
                elif u < 0.95:
                    target_cargo = 2
                else:
                    target_cargo = 3

                for _ in range(target_cargo):
                    color = int(cargo_colors[int(rng.integers(0, len(cargo_colors)))])
                    if color == arrow_color or color == wall_color or color == background:
                        color = (color + 1) % 13
                    _place_cargo_block(rng, grid, (rs, re, cs, ce), occupied, color)

        # At least one cargo pixel to keep examples informative.
        cargo_mask = (grid != background) & (grid != wall_color) & (grid != arrow_color)
        if not cargo_mask.any():
            continue
        # Keep generated samples valid for the deterministic parser.
        if _room_has_collision(grid, background, wall_color):
            continue
        out = main(grid)
        if out.shape != grid.shape:
            continue
        return np.array(grid, dtype=int)

    # Fallback: deterministic easy-valid sample.
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [7, 1, 0, 0, 0, 1, 0, 0, 0],
            [7, 1, 0, 0, 0, 1, 0, 0, 0],
            [7, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 1, 7],
            [0, 1, 0, 0, 0, 0, 0, 0, 7],
            [0, 1, 0, 0, 0, 0, 0, 0, 7],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
