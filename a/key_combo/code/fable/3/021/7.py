from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
SCATTER_COLORS = [1, 3, 5, 6, 7, 8, 9]
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GENERATION_ATTEMPTS = 500

_GEN_CALLS = 0


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _spiral_coords(shape: tuple[int, int], center: tuple[int, int]) -> list[tuple[int, int]]:
    """In-bounds cells after the center, in the clipped infinite spiral order."""
    h, w = map(int, shape)
    total = h * w
    r, c = map(int, center)
    seen = {(r, c)}
    order: list[tuple[int, int]] = []
    leg = 1
    direction_index = 0

    while len(seen) < total:
        for _ in range(2):
            dr, dc = DIRECTIONS[direction_index % 4]
            for _step in range(leg):
                r += dr
                c += dc
                if _in_bounds((h, w), r, c) and (r, c) not in seen:
                    seen.add((r, c))
                    order.append((r, c))
                    if len(seen) == total:
                        return order
            direction_index += 1
        leg += 1

    return order


def _stay_on_oob_spiral_coords(
    shape: tuple[int, int], center: tuple[int, int]
) -> list[tuple[int, int]]:
    """A deliberately wrong clipped walk used only to reject ambiguous generated samples."""
    h, w = map(int, shape)
    total = h * w
    r, c = map(int, center)
    seen = {(r, c)}
    order: list[tuple[int, int]] = []
    leg = 1
    direction_index = 0
    budget = max(20, total * 20)

    while len(seen) < total and budget > 0:
        for _ in range(2):
            dr, dc = DIRECTIONS[direction_index % 4]
            for _step in range(leg):
                budget -= 1
                nr, nc = r + dr, c + dc
                if _in_bounds((h, w), nr, nc):
                    r, c = nr, nc
                    if (r, c) not in seen:
                        seen.add((r, c))
                        order.append((r, c))
                        if len(seen) == total:
                            return order
            direction_index += 1
        leg += 1

    return order


def _collect_with_order(grid: np.ndarray, order: list[tuple[int, int]]) -> list[int]:
    bg = _background(grid)
    out: list[int] = []
    for r, c in order:
        value = int(grid[r, c])
        if value == RED:
            break
        if value != bg and value != YELLOW:
            out.append(value)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect scattered colors in the order reached by a clipped square spiral from a yellow center.

    Input structure:
    - The background is the dominant color.
    - Exactly one yellow pixel marks the spiral center.
    - Non-background pixels other than yellow and red are isolated colored scatter pixels.
    - There is at most one red blocker pixel, separate from the yellow center and scatter pixels.

    Rule:
    - Start at the yellow center and walk the infinite square spiral with directions
      up, right, down, left and leg lengths 1, 1, 2, 2, 3, 3, ...
    - Steps outside the grid still move the spiral position and consume the leg length,
      but only in-bounds cells are visited.
    - Append each visited scatter pixel's color to the output row.
    - If the red blocker is visited, stop immediately and ignore every later scatter pixel.
    - Return the collected colors as a 1 by n row; the center and blocker are never output.

    Key insight:
    - The row is ordered by an unseen spiral trajectory, not by raster position, color,
      or distance ring; the red blocker truncates that trajectory.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background(g)
    assert bg not in (RED, YELLOW), "red and yellow are reserved foreground markers"

    centers = np.argwhere(g == YELLOW)
    assert centers.shape[0] == 1, "expected exactly one yellow center"
    center = tuple(map(int, centers[0]))

    blockers = np.argwhere(g == RED)
    assert blockers.shape[0] <= 1, "expected at most one red blocker"

    collected = _collect_with_order(g, _spiral_coords(g.shape, center))
    return np.asarray([collected], dtype=int)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _clear_of_taken(
    coord: tuple[int, int], taken: set[tuple[int, int]], gap: int = 1
) -> bool:
    return all(_chebyshev(coord, other) > gap for other in taken)


def _pick_spaced(
    candidates: list[tuple[int, int]],
    count: int,
    blocked: set[tuple[int, int]],
    gap: int = 1,
) -> list[tuple[int, int]] | None:
    if count <= 0:
        return []

    for _ in range(120):
        taken = set(blocked)
        options = list(candidates)
        random.shuffle(options)
        picked: list[tuple[int, int]] = []
        for coord in options:
            if _clear_of_taken(coord, taken, gap=gap):
                picked.append(coord)
                taken.add(coord)
                if len(picked) == count:
                    return picked
    return None


def _choose_center(h: int, w: int, mode: str) -> tuple[int, int]:
    if mode == "blocker_edge":
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            return 0, random.randint(2, w - 3)
        if side == "bottom":
            return h - 1, random.randint(2, w - 3)
        if side == "left":
            return random.randint(2, h - 3), 0
        return random.randint(2, h - 3), w - 1

    if random.random() < 0.2:
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            return 0, random.randint(1, w - 2)
        if side == "bottom":
            return h - 1, random.randint(1, w - 2)
        if side == "left":
            return random.randint(1, h - 2), 0
        return random.randint(1, h - 2), w - 1

    return random.randint(2, h - 3), random.randint(2, w - 3)


def _scatter_palette(n: int, force_repeat: bool) -> list[int]:
    palette = list(SCATTER_COLORS)
    random.shuffle(palette)
    colors: list[int] = []
    while len(colors) < n:
        if len(colors) < len(palette):
            colors.append(palette[len(colors)])
        else:
            colors.append(random.choice(palette))

    if force_repeat and n >= 5:
        colors[-1] = colors[1]
    elif n >= 5 and random.random() < 0.25:
        colors[random.randrange(2, n)] = colors[0]
    return colors


def _paint_scatter(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    colors: list[int],
    spiral_rank: dict[tuple[int, int], int],
) -> None:
    ordered = sorted(coords, key=lambda rc: int(spiral_rank[rc]))
    for (r, c), color in zip(ordered, colors):
        grid[int(r), int(c)] = int(color)


def _semantic_checks(
    grid: np.ndarray,
    center: tuple[int, int],
    has_blocker: bool,
    require_oob_clip_counterexample: bool,
) -> bool:
    order = _spiral_coords(grid.shape, center)
    rank = {coord: i for i, coord in enumerate(order)}
    solved = list(map(int, main(grid).ravel()))
    explicit = _collect_with_order(grid, order)
    if solved != explicit or len(solved) < 4:
        return False

    scatter: list[tuple[tuple[int, int], int, int]] = []
    blocker_rank = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            coord = (int(r), int(c))
            value = int(grid[r, c])
            if value == RED:
                blocker_rank = rank[coord]
            elif value not in (BG, YELLOW):
                scatter.append((coord, value, rank[coord]))

    if has_blocker:
        if blocker_rank is None:
            return False
        before = [item for item in scatter if int(item[2]) < int(blocker_rank)]
        after = [item for item in scatter if int(item[2]) > int(blocker_rank)]
        if not after or len(before) != len(solved):
            return False
        no_blocker = grid.copy()
        no_blocker[grid == RED] = BG
        if len(_collect_with_order(no_blocker, order)) <= len(solved):
            return False
    else:
        before = scatter

    before = sorted(before, key=lambda item: int(item[2]))
    raster = [color for _coord, color, _idx in sorted(before, key=lambda item: item[0])]
    cr, cc = center
    ring_then_raster = [
        color
        for _coord, color, _idx in sorted(
            before,
            key=lambda item: (
                max(abs(item[0][0] - cr), abs(item[0][1] - cc)),
                item[0][0],
                item[0][1],
            ),
        )
    ]
    if solved == raster or solved == ring_then_raster:
        return False

    if solved == sorted(solved) or solved == sorted(solved, reverse=True):
        return False

    if require_oob_clip_counterexample:
        wrong_order = _stay_on_oob_spiral_coords(grid.shape, center)
        if solved == _collect_with_order(grid, wrong_order):
            return False

    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    h = random.randint(9, 20)
    w = random.randint(9, 20)
    center = _choose_center(h, w, mode)
    order = _spiral_coords((h, w), center)
    if len(order) < 25:
        return None

    rank = {coord: i for i, coord in enumerate(order)}
    grid = np.full((h, w), BG, dtype=int)
    grid[center] = YELLOW

    has_blocker = mode != "no_blocker_repeat"
    blocked = {center}

    if has_blocker:
        lo = max(9, len(order) // 4)
        hi = min(len(order) - 10, (len(order) * 3) // 4)
        if lo >= hi:
            return None
        blocker_candidates = [
            coord
            for coord in order[lo:hi]
            if _clear_of_taken(coord, blocked, gap=1)
        ]
        if not blocker_candidates:
            return None
        blocker = random.choice(blocker_candidates)
        blocker_index = rank[blocker]
        blocked.add(blocker)

        pre_count = random.randint(4, 7)
        post_count = random.randint(1, 3)
        before_candidates = order[:blocker_index]
        after_candidates = order[blocker_index + 1 :]
        pre = _pick_spaced(before_candidates, pre_count, blocked, gap=1)
        if pre is None:
            return None
        post = _pick_spaced(after_candidates, post_count, blocked | set(pre), gap=1)
        if post is None:
            return None

        grid[blocker] = RED
        _paint_scatter(
            grid,
            pre,
            _scatter_palette(pre_count, force_repeat=mode == "blocker_interior"),
            rank,
        )
        _paint_scatter(grid, post, _scatter_palette(post_count, force_repeat=False), rank)
    else:
        count = random.randint(8, 9)
        picked = _pick_spaced(order, count, blocked, gap=1)
        if picked is None:
            return None
        _paint_scatter(grid, picked, _scatter_palette(count, force_repeat=True), rank)

    if not _semantic_checks(
        grid,
        center,
        has_blocker=has_blocker,
        require_oob_clip_counterexample=mode == "blocker_edge",
    ):
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS

    modes = ["blocker_edge", "no_blocker_repeat", "blocker_interior"]
    mode = modes[_GEN_CALLS % len(modes)]
    _GEN_CALLS += 1

    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral roll-call input")
