import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["bbox"]
        frame_color = frame["frame_color"]
        gap_r, gap_c = frame["gap"]
        direction = frame["direction"]

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (interior != bg) & (interior != frame_color)
        marker_count = int(marker_mask.sum())
        if marker_count == 0:
            continue

        colors, counts = np.unique(interior[marker_mask], return_counts=True)
        branch_color = int(colors[np.argmax(counts)])
        dr, dc = DIR_VECTORS[direction]

        for step in range(marker_count):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = branch_color
            else:
                break

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid grid")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(16, 28)
    width = rng.randint(16, 28)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    frame_color = rng.choice(PALETTE)
    marker_palette = [color for color in PALETTE if color != frame_color]

    min_dim = min(height, width)
    if min_dim < 19:
        max_frames = 2
    elif min_dim < 23:
        max_frames = 3
    else:
        max_frames = 4

    frame_count = rng.randint(1, max_frames)
    directions = rng.sample(list(DIR_VECTORS), frame_count)
    marker_colors = rng.sample(marker_palette, frame_count)

    frames = []
    for direction, marker_color in zip(directions, marker_colors):
        placed = _place_frame(
            grid=grid,
            reserved=reserved,
            rng=rng,
            frame_color=frame_color,
            marker_color=marker_color,
            direction=direction,
        )
        if placed is None:
            return None
        frames.append(placed)

    found = _find_frames(grid, bg=0)
    if len(found) != len(frames):
        return None
    if {_frame_signature(frame) for frame in found} != {_frame_signature(frame) for frame in frames}:
        return None

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (interior != 0) & (interior != frame_color)
        if int(marker_mask.sum()) != frame["marker_count"]:
            return None

    return grid


def _place_frame(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    frame_color: int,
    marker_color: int,
    direction: str,
) -> dict | None:
    height, width = grid.shape
    max_box_h = min(8, height - 2)
    max_box_w = min(8, width - 2)

    for _ in range(200):
        box_h = rng.randint(4, max_box_h)
        box_w = rng.randint(4, max_box_w)
        r0 = rng.randint(1, height - box_h - 1)
        c0 = rng.randint(1, width - box_w - 1)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1

        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(height - 1, r1 + 1)
        cc1 = min(width - 1, c1 + 1)
        if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue

        if direction == "up":
            gap = (r0, rng.randint(c0 + 1, c1 - 1))
        elif direction == "down":
            gap = (r1, rng.randint(c0 + 1, c1 - 1))
        elif direction == "left":
            gap = (rng.randint(r0 + 1, r1 - 1), c0)
        else:
            gap = (rng.randint(r0 + 1, r1 - 1), c1)

        ray = _ray_to_edge(height, width, gap, direction)
        if any(reserved[r, c] for r, c in ray):
            continue

        interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        max_markers = min(len(interior_cells), 7)
        if max_markers == 0:
            continue

        marker_count = rng.randint(1, max_markers)
        if max_markers > len(ray) and rng.random() < 0.5:
            marker_count = rng.randint(len(ray) + 1, max_markers)

        marker_cells = rng.sample(interior_cells, marker_count)
        border_cells = [cell for cell in _perimeter_positions(r0, c0, r1, c1) if cell != gap]

        if any(grid[r, c] != 0 for r, c in border_cells):
            continue
        if any(grid[r, c] != 0 for r, c in marker_cells):
            continue

        for r, c in border_cells:
            grid[r, c] = frame_color
        for r, c in marker_cells:
            grid[r, c] = marker_color

        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        for r, c in ray:
            reserved[r, c] = True

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "direction": direction,
            "frame_color": frame_color,
            "marker_count": marker_count,
        }

    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        frames.extend(_find_frames_for_color(grid, bg, color))
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["frame_color"]))
    return frames


def _find_frames_for_color(grid: np.ndarray, bg: int, color: int) -> list[dict]:
    mask = grid == color
    frames = []
    for component in _connected_components(mask):
        if len(component) < 7:
            continue

        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
            continue

        perimeter = _perimeter_positions(r0, c0, r1, c1)
        if len(component) != len(perimeter) - 1:
            continue

        component_set = set(component)
        missing = [cell for cell in perimeter if cell not in component_set]
        if len(missing) != 1:
            continue

        gap = missing[0]
        if int(grid[gap]) != bg:
            continue

        direction = _gap_direction(gap, (r0, c0, r1, c1))
        if direction is None:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "direction": direction,
                "frame_color": color,
            }
        )
    return frames


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []

    for start_r, start_c in np.argwhere(mask):
        start_r = int(start_r)
        start_c = int(start_c)
        if seen[start_r, start_c]:
            continue

        stack = [(start_r, start_c)]
        seen[start_r, start_c] = True
        component = []

        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        components.append(component)

    return components


def _perimeter_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r, c1) for r in range(r0 + 1, r1))
    cells.extend((r1, c) for c in range(c1, c0 - 1, -1))
    cells.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return cells


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "up"
    if r == r1 and c0 < c < c1:
        return "down"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _ray_to_edge(height: int, width: int, start: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    dr, dc = DIR_VECTORS[direction]
    r, c = start
    cells = []
    while 0 <= r < height and 0 <= c < width:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _frame_signature(frame: dict) -> tuple:
    return (frame["bbox"], frame["gap"], frame["direction"], frame["frame_color"])
