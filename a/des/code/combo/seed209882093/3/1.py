from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_ORDER = ("top", "bottom", "left", "right")

# --------------- generator ---------------

GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
OUTER_H_RANGE = (4, 9)
OUTER_W_RANGE = (4, 9)
MAX_SEEDS = 7
RECT_PLACE_TRIES = 200
GEN_RETRIES = 400


def _side_gap_positions(side: str, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _all_bbox_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _pad_cells(cells: set[tuple[int, int]], height: int, width: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    out.add((nr, nc))
    return out


def _ray_cells(
    gaps: dict[str, tuple[int, int]],
    length: int,
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for side, (gr, gc) in gaps.items():
        dr, dc = SIDE_DIRS[side]
        for step in range(1, length + 1):
            cells.add((gr + dr * step, gc + dc * step))
    return cells


def _max_ray_length(side: str, r0: int, c0: int, r1: int, c1: int, height: int, width: int) -> int:
    if side == "top":
        return r0
    if side == "bottom":
        return height - 1 - r1
    if side == "left":
        return c0
    return width - 1 - c1


def _draw_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
    seed_positions: list[tuple[int, int]],
    gaps: dict[str, tuple[int, int]],
) -> None:
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    for gr, gc in gaps.values():
        grid[gr, gc] = BG

    for sr, sc in seed_positions:
        grid[sr, sc] = color

    for side, (gr, gc) in gaps.items():
        dr, dc = SIDE_DIRS[side]
        grid[gr + dr, gc + dc] = color


def _try_place_rectangle(
    grid: np.ndarray,
    forbidden: np.ndarray,
    color: int,
) -> int | None:
    height, width = map(int, grid.shape)
    max_outer_h = min(OUTER_H_RANGE[1], height - 2)
    max_outer_w = min(OUTER_W_RANGE[1], width - 2)
    if max_outer_h < OUTER_H_RANGE[0] or max_outer_w < OUTER_W_RANGE[0]:
        return None

    for _ in range(RECT_PLACE_TRIES):
        outer_h = random.randint(OUTER_H_RANGE[0], max_outer_h)
        outer_w = random.randint(OUTER_W_RANGE[0], max_outer_w)
        max_r0 = height - outer_h - 1
        max_c0 = width - outer_w - 1
        if max_r0 < 1 or max_c0 < 1:
            continue

        r0 = random.randint(1, max_r0)
        c0 = random.randint(1, max_c0)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        gap_count = random.choices([1, 2, 3, 4], weights=[4, 4, 2, 1])[0]
        gap_sides = random.sample(SIDE_ORDER, k=gap_count)

        max_length = min(
            min(_max_ray_length(side, r0, c0, r1, c1, height, width) for side in gap_sides),
            (outer_h - 2) * (outer_w - 2),
            MAX_SEEDS,
        )
        if max_length < 1:
            continue
        seed_count = random.randint(1, max_length)

        gaps: dict[str, tuple[int, int]] = {}
        for side in gap_sides:
            gaps[side] = random.choice(_side_gap_positions(side, r0, c0, r1, c1))

        bbox_cells = _all_bbox_cells(r0, c0, r1, c1)
        ray_cells = _ray_cells(gaps, seed_count)
        if any(not (0 <= r < height and 0 <= c < width) for r, c in ray_cells):
            continue

        reserve = bbox_cells | ray_cells
        if any(bool(forbidden[r, c]) for r, c in reserve):
            continue

        inner_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        seed_positions = random.sample(inner_cells, k=seed_count)
        _draw_rectangle(grid, r0, c0, r1, c1, color, seed_positions, gaps)

        for r, c in _pad_cells(reserve, height, width):
            forbidden[r, c] = True
        return seed_count

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        grid = np.full((height, width), BG, dtype=int)
        forbidden = np.zeros((height, width), dtype=bool)

        area = height * width
        max_rectangles = 1
        if area >= 240:
            max_rectangles = 2
        if area >= 330:
            max_rectangles = 3
        if area >= 430:
            max_rectangles = 4

        target_count = random.randint(1, max_rectangles)
        colors = random.sample(PALETTE, k=target_count)

        seed_counts: list[int] = []
        for color in colors:
            seed_count = _try_place_rectangle(grid, forbidden, color)
            if seed_count is None:
                break
            seed_counts.append(seed_count)

        if len(seed_counts) != target_count:
            continue

        if np.count_nonzero(grid == GREY) == 0:
            continue
        if max(seed_counts) < 2:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")


def _scan_side_for_gaps(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    gaps: list[tuple[int, int]] = []
    for r, c in cells:
        value = int(grid[r, c])
        if value == GREY:
            continue
        if value == BG:
            gaps.append((r, c))
            continue
        return None
    if len(gaps) > 1:
        return None
    return gaps


def _validate_rectangle(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> dict[str, object] | None:
    height, width = map(int, grid.shape)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    if any(int(grid[r, c]) != GREY for r, c in corners):
        return None

    side_cells = {
        "top": _side_gap_positions("top", r0, c0, r1, c1),
        "bottom": _side_gap_positions("bottom", r0, c0, r1, c1),
        "left": _side_gap_positions("left", r0, c0, r1, c1),
        "right": _side_gap_positions("right", r0, c0, r1, c1),
    }

    gaps_by_side: dict[str, list[tuple[int, int]]] = {}
    for side, cells in side_cells.items():
        gaps = _scan_side_for_gaps(grid, cells)
        if gaps is None:
            return None
        gaps_by_side[side] = gaps

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or bool(np.any(interior == GREY)):
        return None

    colored = interior[(interior != BG) & (interior != GREY)]
    colors = np.unique(colored)
    if colors.size != 1:
        return None
    color = int(colors[0])
    seed_count = int(np.count_nonzero(interior == color))
    if seed_count == 0:
        return None

    rays: list[tuple[int, int, int, int]] = []
    for side in SIDE_ORDER:
        for gr, gc in gaps_by_side[side]:
            dr, dc = SIDE_DIRS[side]
            mr = gr + dr
            mc = gc + dc
            if not (0 <= mr < height and 0 <= mc < width):
                return None
            if int(grid[mr, mc]) != color:
                return None
            rays.append((mr, mc, dr, dc))

    return {
        "bbox": (r0, c0, r1, c1),
        "color": color,
        "seed_count": seed_count,
        "rays": rays,
    }


def _find_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    grey_points = [tuple(map(int, coord)) for coord in np.argwhere(grid == GREY)]
    grey_set = set(grey_points)
    found: list[dict[str, object]] = []

    for r0, c0 in grey_points:
        for r1, c1 in grey_points:
            if r1 <= r0 or c1 <= c0:
                continue
            if (r0, c1) not in grey_set or (r1, c0) not in grey_set:
                continue
            rect = _validate_rectangle(grid, r0, c0, r1, c1)
            if rect is not None:
                found.append(rect)

    found.sort(key=lambda item: tuple(item["bbox"]))  # deterministic traversal
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-colored rays outward from frame gaps, with length equal to the number of interior seeds.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for rect in _find_rectangles(g):
        color = int(rect["color"])
        length = int(rect["seed_count"])
        for start_r, start_c, dr, dc in rect["rays"]:
            for step in range(length):
                rr = start_r + dr * step
                cc = start_c + dc * step
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = color

    return out
