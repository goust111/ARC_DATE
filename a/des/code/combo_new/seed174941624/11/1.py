import random
from typing import Iterable

import numpy as np


SIDES = ("top", "bottom", "left", "right")
MODE_CHOICES = (
    (True, False),
    (True, True),
    (False, False),
    (False, True),
)

BG = 0
FRAME_COLOR_CHOICES = (1, 5, 8)
GRID_MIN = 16
GRID_MAX = 30
RECT_COUNT_CHOICES = (1, 2, 2, 3)
INNER_H_RANGE = (2, 5)
INNER_W_RANGE = (2, 5)
PLACEMENT_TRIES = 250
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _gap_side(gap_r: int, gap_c: int, r0: int, c0: int, r1: int, c1: int) -> tuple[str, tuple[int, int]] | None:
    if gap_r == r0 and c0 < gap_c < c1:
        return "top", (-1, 0)
    if gap_r == r1 and c0 < gap_c < c1:
        return "bottom", (1, 0)
    if gap_c == c0 and r0 < gap_r < r1:
        return "left", (0, -1)
    if gap_c == c1 and r0 < gap_r < r1:
        return "right", (0, 1)
    return None


def _detect_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    rects: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components_mask(grid == color, connectivity=4):
            if int(comp.shape[0]) < 7:
                continue
            r0, c0 = map(int, comp.min(axis=0))
            r1, c1 = map(int, comp.max(axis=0))
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            border = _border_coords(r0, c0, r1, c1)
            border_set = set(border)
            comp_set = {(int(r), int(c)) for r, c in comp}
            if not comp_set.issubset(border_set):
                continue
            if len(border_set) - len(comp_set) != 1:
                continue

            missing = list(border_set - comp_set)
            if len(missing) != 1:
                continue
            gap = missing[0]
            gap_color = int(grid[gap])
            if gap_color in (bg, color):
                continue

            side_info = _gap_side(gap[0], gap[1], r0, c0, r1, c1)
            if side_info is None:
                continue
            side, direction = side_info

            rects.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "frame_color": color,
                    "gap": gap,
                    "gap_color": gap_color,
                    "side": side,
                    "direction": direction,
                    "area": (r1 - r0 - 1) * (c1 - c0 - 1),
                }
            )

    rects.sort(key=lambda rect: (rect["bbox"][0], rect["bbox"][1], rect["gap"]))
    return rects


def _ray_targets(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    need: int,
    bg: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    dr, dc = direction
    rr, cc = start
    painted = 0
    cells: list[tuple[int, int]] = []

    while painted < need:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if int(grid[rr, cc]) == bg:
            cells.append((rr, cc))
            painted += 1
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow outward rays from colored border gaps of rectangular frames, using the frame area as the ray length.

    Input structure:
    - The background is black.
    - One or more single-pixel rectangular frame components enclose regions; each frame boundary has exactly one non-frame gap pixel on one side.
    - Inside each rectangle are scattered single-pixel markers, and one interior marker color matches the rectangle's gap color.

    Rule:
    - Detect each rectangular frame by finding a monochrome border component whose bounding-box border is complete except for one side pixel occupied by a different non-black color.
    - Let A be the rectangle's interior area. From the gap position, extend outward along the normal direction of that side and paint exactly A black cells in the gap color, skipping over any non-black cells without changing them.
    - Remove every interior marker whose color matches that rectangle's gap color; keep all other pixels unchanged.

    Key insight:
    - The matching interior color identifies which markers are tied to each gap, but the outward ray length is determined by the rectangle's full interior area rather than by the number of matching markers.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    claims: dict[tuple[int, int], int] = {}
    for rect in _detect_rectangles(g, bg):
        for cell in _ray_targets(g, rect["gap"], rect["direction"], rect["area"], bg):
            claims.setdefault(cell, rect["gap_color"])

        r0, c0, r1, c1 = rect["bbox"]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(g[r, c]) == rect["gap_color"]:
                    out[r, c] = bg

    for (r, c), color in claims.items():
        if int(g[r, c]) == bg:
            out[r, c] = int(color)
    return out


def _corridor_cells(
    gap: tuple[int, int], direction: tuple[int, int], h: int, w: int
) -> list[tuple[int, int]]:
    rr, cc = gap
    dr, dc = direction
    cells = []
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            return cells
        cells.append((rr, cc))


def _gap_for_side(
    r0: int, c0: int, inner_h: int, inner_w: int, side: str, rng: random.Random
) -> tuple[tuple[int, int], tuple[int, int]]:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    if side == "top":
        return (r0, c0 + rng.randint(1, inner_w)), (-1, 0)
    if side == "bottom":
        return (r1, c0 + rng.randint(1, inner_w)), (1, 0)
    if side == "left":
        return (r0 + rng.randint(1, inner_h), c0), (0, -1)
    return (r0 + rng.randint(1, inner_h), c1), (0, 1)


def _sample_interior_markers(
    interior: list[tuple[int, int]],
    gap_color: int,
    palette: list[int],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    area = len(interior)
    n_match = rng.randint(1, min(4, area - 2))
    n_other = rng.randint(1, min(3, area - n_match - 1))
    picks = rng.sample(interior, n_match + n_other)
    match_cells = picks[:n_match]
    other_cells = picks[n_match:]

    other_palette = [color for color in palette if color != gap_color]
    n_other_colors = rng.randint(1, min(2, len(other_palette)))
    other_colors = rng.sample(other_palette, n_other_colors)
    other_markers = [(r, c, rng.choice(other_colors)) for r, c in other_cells]
    return match_cells, other_markers


def _reserve_box(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    mask[rr0:rr1, cc0:cc1] = True


def _try_place_rectangle(
    grid: np.ndarray,
    reserved: np.ndarray,
    frame_color: int,
    gap_color: int,
    free_colors: list[int],
    full_mode: bool,
    obstacle_mode: bool,
    rng: random.Random,
) -> bool:
    h, w = grid.shape

    for _ in range(PLACEMENT_TRIES):
        inner_h = rng.randint(*INNER_H_RANGE)
        inner_w = rng.randint(*INNER_W_RANGE)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        if outer_h + 2 > h or outer_w + 2 > w:
            continue

        side = rng.choice(SIDES)
        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if reserved[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)].any():
            continue

        gap, direction = _gap_for_side(r0, c0, inner_h, inner_w, side, rng)
        corridor = _corridor_cells(gap, direction, h, w)
        if any(reserved[r, c] for r, c in corridor):
            continue

        area = inner_h * inner_w
        corridor_len = len(corridor)
        if corridor_len == 0:
            continue

        if obstacle_mode:
            max_obstacles = min(3, corridor_len)
            obstacle_counts = []
            for n_obstacles in range(1, max_obstacles + 1):
                black_count = corridor_len - n_obstacles
                if full_mode and black_count >= area:
                    obstacle_counts.append(n_obstacles)
                if not full_mode and black_count < area:
                    obstacle_counts.append(n_obstacles)
            if not obstacle_counts:
                continue
            n_obstacles = rng.choice(obstacle_counts)
        else:
            n_obstacles = 0
            black_count = corridor_len
            if full_mode and black_count < area:
                continue
            if not full_mode and black_count >= area:
                continue

        interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        match_cells, other_markers = _sample_interior_markers(interior, gap_color, free_colors, rng)
        obstacle_cells = rng.sample(corridor, n_obstacles) if n_obstacles else []
        obstacle_colors = [color for color in free_colors if color != gap_color]

        for r, c in _border_coords(r0, c0, r1, c1):
            if (r, c) != gap:
                grid[r, c] = frame_color
        grid[gap] = gap_color
        for r, c in match_cells:
            grid[r, c] = gap_color
        for r, c, color in other_markers:
            grid[r, c] = color
        for r, c in obstacle_cells:
            grid[r, c] = rng.choice(obstacle_colors)

        _reserve_box(reserved, r0, c0, r1, c1)
        for r, c in corridor:
            reserved[r, c] = True
        return True

    return False


def _scatter_distractors(
    grid: np.ndarray, reserved: np.ndarray, colors: Iterable[int], rng: random.Random
) -> None:
    h, w = grid.shape
    color_list = list(colors)
    if not color_list:
        return
    for _ in range(rng.randint(0, 2)):
        for _ in range(50):
            r = rng.randint(0, h - 1)
            c = rng.randint(0, w - 1)
            if reserved[r, c] or int(grid[r, c]) != BG:
                continue
            grid[r, c] = rng.choice(color_list)
            reserved[r, c] = True
            break


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice(RECT_COUNT_CHOICES)
    min_side = max(GRID_MIN, 14 + 4 * n_rects)
    h = rng.randint(min_side, GRID_MAX)
    w = rng.randint(min_side, GRID_MAX)

    frame_color = rng.choice(FRAME_COLOR_CHOICES)
    palette = [color for color in range(10) if color not in (BG, frame_color)]
    rng.shuffle(palette)
    gap_colors = palette[:n_rects]
    free_colors = palette[n_rects:]
    if len(free_colors) < 2:
        return None

    modes = list(MODE_CHOICES)
    rng.shuffle(modes)
    while len(modes) < n_rects:
        modes.append(rng.choice(MODE_CHOICES))
    modes = modes[:n_rects]

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for gap_color, (full_mode, obstacle_mode) in zip(gap_colors, modes):
        if not _try_place_rectangle(
            grid,
            reserved,
            frame_color,
            gap_color,
            free_colors,
            full_mode,
            obstacle_mode,
            rng,
        ):
            return None

    _scatter_distractors(grid, reserved, free_colors, rng)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
