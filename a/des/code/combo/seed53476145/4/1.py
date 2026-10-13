import random
from collections import deque

import numpy as np


BG = 0
RED = 2
FRAME_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
FRAME_COUNT_OPTIONS = (2, 3, 4)
FRAME_COUNT_WEIGHTS = (3, 3, 1)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 120


def _collect_component(grid: np.ndarray, start: tuple[int, int], visited: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    color = int(grid[start])
    queue = deque([start])
    visited[start] = True
    coords: list[tuple[int, int]] = []

    while queue:
        r, c = queue.popleft()
        coords.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if visited[nr, nc] or int(grid[nr, nc]) != color:
                continue
            visited[nr, nc] = True
            queue.append((nr, nc))

    return coords


def _frame_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)

    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    comp_set = set(coords)
    if comp_set != border:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if not np.all((interior == BG) | (interior == RED)):
        return None

    return {
        "color": int(grid[r0, c0]),
        "bbox": (r0, c0, r1, c1),
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    frames: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if visited[r, c] or color in (BG, RED):
                continue
            coords = _collect_component(grid, (r, c), visited)
            frame = _frame_from_component(grid, coords)
            if frame is not None:
                frames.append(frame)

    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _draw_rect_ring(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _ring_capacity(inner_height: int, inner_width: int) -> int:
    if inner_height <= 0 or inner_width <= 0:
        return 0
    return (min(inner_height, inner_width) + 1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each frame's red markers and fill its interior with that many spaced concentric rings.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]

        interior_in = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = interior_in == RED
        layer_count = int(np.count_nonzero(marker_mask))

        interior_out = out[r0 + 1 : r1, c0 + 1 : c1]
        interior_out[marker_mask] = BG

        for layer in range(layer_count):
            rr0 = r0 + 1 + 2 * layer
            cc0 = c0 + 1 + 2 * layer
            rr1 = r1 - 1 - 2 * layer
            cc1 = c1 - 1 - 2 * layer
            if rr0 > rr1 or cc0 > cc1:
                break
            _draw_rect_ring(out, rr0, cc0, rr1, cc1, color)

    return out


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _choose_marker_cells(inner_height: int, inner_width: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    parity_groups = [[], []]
    for r in range(inner_height):
        for c in range(inner_width):
            parity_groups[(r + c) & 1].append((r, c))

    valid_groups = [group for group in parity_groups if len(group) >= count]
    if not valid_groups:
        raise ValueError("not enough isolated marker cells")
    return rng.sample(rng.choice(valid_groups), count)


def _sample_frame_spec(color: int, branch: str, rng: random.Random) -> dict:
    if branch == "truncate":
        marker_count = rng.randint(2, 4)
        capacity = rng.randint(1, marker_count - 1)
    else:
        marker_count = rng.randint(1, 4)
        capacity = rng.randint(marker_count, min(4, marker_count + 1))

    tight_dim = rng.choice((2 * capacity - 1, 2 * capacity))
    other_dim_min = max(tight_dim, _ceil_div(2 * marker_count - 1, tight_dim))
    other_dim = rng.randint(other_dim_min, other_dim_min + 4)

    if rng.random() < 0.5:
        inner_height, inner_width = tight_dim, other_dim
    else:
        inner_height, inner_width = other_dim, tight_dim

    return {
        "color": color,
        "marker_count": marker_count,
        "branch": branch,
        "inner_height": inner_height,
        "inner_width": inner_width,
        "outer_height": inner_height + 2,
        "outer_width": inner_width + 2,
    }


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_frame(grid: np.ndarray, occupied: np.ndarray, spec: dict, rng: random.Random) -> bool:
    height, width = grid.shape
    outer_height = int(spec["outer_height"])
    outer_width = int(spec["outer_width"])

    max_top = height - outer_height - 1
    max_left = width - outer_width - 1
    if max_top < 1 or max_left < 1:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(1, max_top)
        left = rng.randint(1, max_left)
        if not _can_place(occupied, top, left, outer_height, outer_width):
            continue

        color = int(spec["color"])
        _draw_rect_ring(grid, top, left, top + outer_height - 1, left + outer_width - 1, color)

        marker_cells = _choose_marker_cells(
            int(spec["inner_height"]),
            int(spec["inner_width"]),
            int(spec["marker_count"]),
            rng,
        )
        for dr, dc in marker_cells:
            grid[top + 1 + dr, left + 1 + dc] = RED

        occupied[top : top + outer_height, left : left + outer_width] = True
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    frame_count = rng.choices(FRAME_COUNT_OPTIONS, weights=FRAME_COUNT_WEIGHTS, k=1)[0]

    colors = rng.sample(FRAME_COLORS, frame_count)
    branches = ["fit", "truncate"] + [rng.choice(("fit", "truncate")) for _ in range(frame_count - 2)]
    rng.shuffle(branches)

    specs = [_sample_frame_spec(color, branch, rng) for color, branch in zip(colors, branches)]
    specs.sort(key=lambda spec: spec["outer_height"] * spec["outer_width"], reverse=True)

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    for spec in specs:
        if not _place_frame(grid, occupied, spec, rng):
            return None

    frames = _find_frames(grid)
    if len(frames) != frame_count:
        return None

    saw_fit = False
    saw_truncate = False
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = int(np.count_nonzero(inner == RED))
        capacity = _ring_capacity(inner.shape[0], inner.shape[1])
        saw_fit |= marker_count <= capacity
        saw_truncate |= marker_count > capacity

    if not (saw_fit and saw_truncate):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
