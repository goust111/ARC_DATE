from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < int(shape[0]) and 0 <= c < int(shape[1])


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    return {
        UP: RIGHT,
        RIGHT: DOWN,
        DOWN: LEFT,
        LEFT: UP,
    }[direction]


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    return {
        UP: LEFT,
        LEFT: DOWN,
        DOWN: RIGHT,
        RIGHT: UP,
    }[direction]


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == BLUE)
    if len(coords) == 0:
        raise ValueError("expected a blue U-shaped frame")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _headspace_rows(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = frame_box
    interior = grid[r0:r1, c0 + 1 : c1]
    if interior.size == 0:
        raise ValueError("frame interior is empty")
    liquid = interior == RED
    row_has_liquid = liquid.any(axis=1)
    if not row_has_liquid.any():
        raise ValueError("expected red liquid inside the frame")
    first_filled = int(np.argmax(row_has_liquid))
    return first_filled


def _liquid_mask(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = frame_box
    mask = np.zeros_like(grid, dtype=bool)
    interior = grid[r0:r1, c0 + 1 : c1]
    mask[r0:r1, c0 + 1 : c1] = interior == RED
    return mask


def _seed_records(grid: np.ndarray, frame_box: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
    liquid = _liquid_mask(grid, frame_box)
    seeds: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((grid == RED) | (grid == GREEN)):
        rr = int(r)
        cc = int(c)
        if liquid[rr, cc]:
            continue
        seeds.append((rr, cc, int(grid[rr, cc])))
    seeds.sort()
    return seeds


def _ray_direction(color: int) -> tuple[int, int]:
    if color == RED:
        return DOWN
    if color == GREEN:
        return RIGHT
    raise ValueError(f"unexpected seed color: {color}")


def _apply_transform(
    grid: np.ndarray,
    *,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frame_box = _frame_bbox(g)
    steps = _headspace_rows(g, frame_box)
    seeds = _seed_records(g, frame_box)

    ray_info: list[dict[str, object]] = []
    for r, c, color in seeds:
        dr, dc = _ray_direction(color)
        ray_cells: list[tuple[int, int]] = []
        rr, cc = r, c
        for _ in range(steps):
            rr += dr
            cc += dc
            if not _in_bounds(out.shape, rr, cc):
                break
            out[rr, cc] = ORANGE
            ray_cells.append((rr, cc))
        endpoint = ray_cells[-1] if ray_cells else (r, c)
        ray_info.append(
            {
                "seed": (r, c),
                "color": color,
                "heading": (dr, dc),
                "endpoint": endpoint,
                "ray_len": len(ray_cells),
            }
        )

    skip_counts: list[int] = []
    painted_counts: list[int] = []
    for info in ray_info:
        pos = tuple(info["endpoint"])
        heading = tuple(info["heading"])
        skips = 0
        painted = 0
        for step_index in range(steps):
            if step_index == 0:
                candidates = [_turn_right(heading)]
            else:
                candidates = [_turn_right(heading), heading, _turn_left(heading)]

            moved = False
            for direction in candidates:
                nr = int(pos[0]) + int(direction[0])
                nc = int(pos[1]) + int(direction[1])
                if not _in_bounds(out.shape, nr, nc):
                    continue
                if out[nr, nc] != BLACK:
                    continue
                out[nr, nc] = YELLOW
                pos = (nr, nc)
                heading = direction
                painted += 1
                moved = True
                break
            if not moved:
                skips += 1
        skip_counts.append(skips)
        painted_counts.append(painted)

    if not return_stats:
        return out

    stats = {
        "steps": steps,
        "seed_count": len(seeds),
        "skip_counts": skip_counts,
        "painted_counts": painted_counts,
        "seed_colors": [color for _r, _c, color in seeds],
        "ray_lengths": [int(info["ray_len"]) for info in ray_info],
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the liquid headspace inside a blue U-frame to drive remote seed rays and short clockwise spirals.

    Input structure:
    - The background is black.
    - The upper-left area contains one blue open-top U-shaped frame whose interior bottom is filled by a solid red liquid rectangle.
    - The lower-right area contains 2-3 isolated seed pixels: red seeds and green seeds.

    Rule:
    - Count how many empty interior rows lie above the liquid inside the U-frame; call this value N.
    - Each red seed emits an orange downward ray of length N, and each green seed emits an orange rightward ray of length N; the seed pixel itself keeps its original color.
    - Starting from each ray endpoint, grow a yellow clockwise path for N steps: the first attempted step is a 90-degree clockwise turn from the ray direction, then later steps prefer right-turn, then straight, then left; if none is available, that step is skipped.

    Key insight:
    - The U-frame is only a control panel: its liquid headspace encodes a single number N that determines both the ray lengths and the spiral lengths in a distant part of the grid.
    """
    return _apply_transform(grid, return_stats=False)  # type: ignore[return-value]


GEN_GRID_SIDE_MIN = 17
GEN_GRID_SIDE_MAX = 22
GEN_EMPTY_ROWS_MIN = 1
GEN_EMPTY_ROWS_MAX = 5
GEN_INTERIOR_H_MIN = 3
GEN_INTERIOR_H_MAX = 7
GEN_INTERIOR_W_MIN = 3
GEN_INTERIOR_W_MAX = 6
GEN_MARGIN_MIN = 1
GEN_MARGIN_MAX = 3
GEN_MIN_SEED_GAP = 4
GEN_MAX_ATTEMPTS = 400
GEN_SKIP_EXAMPLE_PROB = 0.3


def _render_base_grid(
    height: int,
    width: int,
    top: int,
    left: int,
    interior_h: int,
    interior_w: int,
    empty_rows: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    grid = np.zeros((height, width), dtype=int)
    frame_h = interior_h + 1
    frame_w = interior_w + 2
    r0, c0 = top, left
    r1, c1 = top + frame_h - 1, left + frame_w - 1

    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 + empty_rows : r1, c0 + 1 : c1] = RED
    return grid, (r0, c0, r1, c1)


def _sample_base_layout() -> tuple[np.ndarray, tuple[int, int, int, int], int] | None:
    height = random.randint(GEN_GRID_SIDE_MIN, GEN_GRID_SIDE_MAX)
    width = random.randint(GEN_GRID_SIDE_MIN, GEN_GRID_SIDE_MAX)
    empty_rows = random.randint(GEN_EMPTY_ROWS_MIN, GEN_EMPTY_ROWS_MAX)
    min_interior_h = max(GEN_INTERIOR_H_MIN, empty_rows + 1)
    interior_h = random.randint(min_interior_h, GEN_INTERIOR_H_MAX)
    interior_w = random.randint(GEN_INTERIOR_W_MIN, GEN_INTERIOR_W_MAX)
    frame_h = interior_h + 1
    frame_w = interior_w + 2

    if frame_h + GEN_MARGIN_MAX + 2 > height or frame_w + GEN_MARGIN_MAX + 2 > width:
        return None

    top = random.randint(GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    left = random.randint(GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    if top + frame_h >= height - 2 or left + frame_w >= width - 2:
        return None
    grid, frame_box = _render_base_grid(height, width, top, left, interior_h, interior_w, empty_rows)
    return grid, frame_box, empty_rows


def _ray_cells(seed: tuple[int, int], color: int, steps: int) -> list[tuple[int, int]]:
    dr, dc = _ray_direction(color)
    r, c = seed
    return [(r + dr * step, c + dc * step) for step in range(1, steps + 1)]


def _seed_candidates(
    occupied: np.ndarray,
    frame_box: tuple[int, int, int, int],
    steps: int,
    color: int,
    placed_seeds: list[tuple[int, int, int]],
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    h, w = occupied.shape
    r0, c0, r1, c1 = frame_box
    row_start = max(r1 + 2, h // 2)
    col_start = max(c1 + 2, w // 2)
    candidates: list[tuple[int, int, list[tuple[int, int]]]] = []

    for r in range(row_start, h):
        for c in range(col_start, w):
            if occupied[r, c]:
                continue
            if any(abs(r - pr) + abs(c - pc) < GEN_MIN_SEED_GAP for pr, pc, _color in placed_seeds):
                continue
            ray = _ray_cells((r, c), color, steps)
            if any(not _in_bounds((h, w), rr, cc) for rr, cc in ray):
                continue
            if any(occupied[rr, cc] for rr, cc in ray):
                continue
            candidates.append((r, c, ray))

    random.shuffle(candidates)
    return candidates


def _place_seeds_with_rays(
    base_grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    steps: int,
    seed_colors: list[int],
) -> list[tuple[int, int, int]] | None:
    def search(
        occupied: np.ndarray,
        placed: list[tuple[int, int, int]],
        remaining: list[int],
    ) -> list[tuple[int, int, int]] | None:
        if not remaining:
            return placed

        color = remaining[0]
        for r, c, ray in _seed_candidates(occupied, frame_box, steps, color, placed)[:120]:
            next_occupied = occupied.copy()
            next_occupied[r, c] = True
            for rr, cc in ray:
                next_occupied[rr, cc] = True
            result = search(next_occupied, placed + [(r, c, color)], remaining[1:])
            if result is not None:
                return result
        return None

    occupied = base_grid != BLACK
    return search(occupied, [], seed_colors)


def _planned_seed_colors() -> list[int]:
    seed_count = random.choice([2, 3])
    if seed_count == 2:
        colors = [RED, GREEN]
    else:
        colors = random.choice([[RED, RED, GREEN], [RED, GREEN, GREEN]])
    random.shuffle(colors)
    return colors


def _generator_accepts(inp: np.ndarray, need_skip: bool) -> bool:
    out, stats = _apply_transform(inp, return_stats=True)
    if not np.array_equal(main(inp), out):
        return False

    steps = int(stats["steps"])
    if not (GEN_EMPTY_ROWS_MIN <= steps <= GEN_EMPTY_ROWS_MAX):
        return False
    if stats["seed_count"] not in (2, 3):
        return False
    colors = set(stats["seed_colors"])
    if colors != {RED, GREEN}:
        return False
    if any(ray_len != steps for ray_len in stats["ray_lengths"]):
        return False
    if sum(int(x) for x in stats["painted_counts"]) < int(stats["seed_count"]):
        return False
    if need_skip and sum(int(x) for x in stats["skip_counts"]) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_base_layout()
    if sampled is None:
        return None
    base_grid, frame_box, steps = sampled
    seed_colors = _planned_seed_colors()
    placed_seeds = _place_seeds_with_rays(base_grid, frame_box, steps, seed_colors)
    if placed_seeds is None:
        return None

    inp = base_grid.copy()
    for r, c, color in placed_seeds:
        inp[r, c] = color

    need_skip = random.random() < GEN_SKIP_EXAMPLE_PROB
    if not _generator_accepts(inp, need_skip):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
