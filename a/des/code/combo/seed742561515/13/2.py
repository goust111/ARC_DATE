from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
GEN_ATTEMPTS = 600

Coord = tuple[int, int]
Direction = tuple[int, int]
Spec = tuple[int, Coord, Direction]


@dataclass(frozen=True)
class Arrow:
    color: int
    head: Coord
    direction: Direction
    cells: tuple[Coord, Coord, Coord]
    key: tuple[int, int, int, int]


def _arrow_cells_from_head(head: Coord, direction: Direction) -> tuple[Coord, Coord, Coord]:
    r, c = head
    dr, dc = direction
    tail_r = r - dr
    tail_c = c - dc
    if dr != 0:
        return ((r, c), (tail_r, tail_c - 1), (tail_r, tail_c + 1))
    return ((r, c), (tail_r - 1, tail_c), (tail_r + 1, tail_c))


def _infer_head_and_direction(coords: list[Coord]) -> tuple[Coord, Direction]:
    matches: list[tuple[Coord, Direction]] = []
    for i, j in combinations(range(3), 2):
        a = coords[i]
        b = coords[j]
        head = coords[3 - i - j]

        if a[1] == b[1] and abs(a[0] - b[0]) == 2:
            mid_r = (a[0] + b[0]) // 2
            if head[0] == mid_r and abs(head[1] - a[1]) == 1:
                matches.append((head, (0, 1 if head[1] > a[1] else -1)))

        if a[0] == b[0] and abs(a[1] - b[1]) == 2:
            mid_c = (a[1] + b[1]) // 2
            if head[1] == mid_c and abs(head[0] - a[0]) == 1:
                matches.append((head, (1 if head[0] > a[0] else -1, 0)))

    unique_matches = list(dict.fromkeys(matches))
    assert len(unique_matches) == 1, f"cannot infer arrow direction from {coords}"
    return unique_matches[0]


def _detect_arrows(grid: np.ndarray) -> list[Arrow]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    arrows: list[Arrow] = []
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one arrow"

    for color in colors:
        coords = [tuple(map(int, pt)) for pt in np.argwhere(g == color)]
        assert len(coords) == 3, f"color {color} must occupy exactly 3 cells"
        head, direction = _infer_head_and_direction(coords)
        cells = tuple(sorted(coords))
        top_left = min(cells)
        arrows.append(
            Arrow(
                color=color,
                head=head,
                direction=direction,
                cells=cells,
                key=(top_left[0], top_left[1], head[0], head[1]),
            )
        )

    # Different arrows are separated by at least one background cell in every direction.
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG:
                continue
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    other = int(g[nr, nc])
                    if other != BG and other != color:
                        raise AssertionError("arrows may not touch each other")

    arrows.sort(key=lambda arrow: arrow.key)
    return arrows


def _trace_rays(grid: np.ndarray, arrows: list[Arrow] | None = None) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    ordered_arrows = _detect_arrows(g) if arrows is None else sorted(arrows, key=lambda arrow: arrow.key)
    h, w = g.shape

    out = g.copy()
    original_foreground = g != BG
    ray_mask = np.zeros_like(g, dtype=bool)
    stats = {
        "visited": 0,
        "claimed": 0,
        "blocked_by_arrow": 0,
        "blocked_by_ray": 0,
    }

    for arrow in ordered_arrows:
        r, c = arrow.head
        dr, dc = arrow.direction
        r += dr
        c += dc
        while 0 <= r < h and 0 <= c < w:
            stats["visited"] += 1
            if original_foreground[r, c]:
                stats["blocked_by_arrow"] += 1
            elif ray_mask[r, c]:
                stats["blocked_by_ray"] += 1
            else:
                out[r, c] = arrow.color
                ray_mask[r, c] = True
                stats["claimed"] += 1
            r += dr
            c += dc

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Launch a color ray from every 3-pixel arrow, honoring row-major first-hit priority."""
    return _trace_rays(np.asarray(grid, dtype=int))[0]


def _steps_to_edge(head: Coord, direction: Direction, h: int, w: int) -> int:
    r, c = head
    dr, dc = direction
    if dr == -1:
        return r
    if dr == 1:
        return h - 1 - r
    if dc == -1:
        return c
    return w - 1 - c


def _can_place(cells: tuple[Coord, Coord, Coord], occupied: set[Coord], h: int, w: int) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if (nr, nc) in occupied:
                    return False
    return True


def _specs_valid(h: int, w: int, specs: list[Spec]) -> bool:
    occupied: set[Coord] = set()
    for _color, head, direction in specs:
        if _steps_to_edge(head, direction, h, w) < 1:
            return False
        cells = _arrow_cells_from_head(head, direction)
        if not _can_place(cells, occupied, h, w):
            return False
        occupied.update(cells)
    return True


def _random_head(h: int, w: int, direction: Direction, min_steps: int) -> Coord | None:
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if _steps_to_edge((r, c), direction, h, w) >= min_steps
    ]
    return random.choice(candidates) if candidates else None


def _fill_random_specs(h: int, w: int, specs: list[Spec], total: int) -> bool:
    occupied = {
        cell
        for _color, head, direction in specs
        for cell in _arrow_cells_from_head(head, direction)
    }
    used_colors = {color for color, _head, _direction in specs}

    for _ in range(800):
        if len(specs) >= total:
            return True
        remaining = [color for color in PALETTE if color not in used_colors]
        if not remaining:
            break

        color = random.choice(remaining)
        direction = random.choice(DIRS)
        min_steps = 2 if min(h, w) >= 8 and random.random() < 0.7 else 1
        head = _random_head(h, w, direction, min_steps)
        if head is None:
            continue

        cells = _arrow_cells_from_head(head, direction)
        if not _can_place(cells, occupied, h, w):
            continue

        specs.append((color, head, direction))
        used_colors.add(color)
        occupied.update(cells)

    return len(specs) == total


def _build_grid(h: int, w: int, specs: list[Spec]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for color, head, direction in specs:
        for r, c in _arrow_cells_from_head(head, direction):
            assert grid[r, c] == BG
            grid[r, c] = color
    return grid


def _random_orient(grid: np.ndarray) -> np.ndarray:
    oriented = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        oriented = np.fliplr(oriented)
    if random.random() < 0.5:
        oriented = np.flipud(oriented)
    return np.array(oriented, copy=True)


def _choose_total_arrows(h: int, w: int, minimum: int = 2) -> int:
    upper = min(6, len(PALETTE), max(minimum, (h * w) // 20 + 1))
    return random.randint(minimum, upper)


def _accept_candidate(grid: np.ndarray, mode: str) -> np.ndarray | None:
    try:
        arrows = _detect_arrows(grid)
        _out, stats = _trace_rays(grid, arrows)
    except AssertionError:
        return None

    if len(arrows) < 2:
        return None
    if len({("v" if arrow.direction[0] else "h") for arrow in arrows}) < 2:
        return None
    if stats["claimed"] < max(3, len(arrows)):
        return None

    if mode == "free":
        if stats["blocked_by_arrow"] != 0 or stats["blocked_by_ray"] != 0:
            return None
    elif mode == "ray_overlap":
        if stats["blocked_by_ray"] < 1 or stats["blocked_by_arrow"] != 0:
            return None
    elif mode == "arrow_block":
        if stats["blocked_by_arrow"] < 1 or stats["blocked_by_ray"] != 0:
            return None
    else:
        raise ValueError(f"unknown generation mode: {mode}")

    return grid


def _gen_free() -> np.ndarray | None:
    h = random.randint(8, 18)
    w = random.randint(8, 18)
    total = _choose_total_arrows(h, w, minimum=2)
    specs: list[Spec] = []
    if not _fill_random_specs(h, w, specs, total):
        return None
    return _accept_candidate(_random_orient(_build_grid(h, w, specs)), mode="free")


def _gen_ray_overlap() -> np.ndarray | None:
    h = random.randint(10, 16)
    w = random.randint(10, 16)
    total = _choose_total_arrows(h, w, minimum=2)
    colors = random.sample(PALETTE, k=2)

    cross_r = random.randint(3, h - 4)
    cross_c = random.randint(3, w - 4)
    down_head_r = random.randint(1, cross_r - 2)
    right_head_c = random.randint(1, cross_c - 2)

    specs: list[Spec] = [
        (colors[0], (down_head_r, cross_c), (1, 0)),
        (colors[1], (cross_r, right_head_c), (0, 1)),
    ]
    if not _specs_valid(h, w, specs):
        return None
    if not _fill_random_specs(h, w, specs, total):
        return None
    return _accept_candidate(_random_orient(_build_grid(h, w, specs)), mode="ray_overlap")


def _gen_arrow_block() -> np.ndarray | None:
    h = random.randint(10, 16)
    w = random.randint(10, 16)
    total = _choose_total_arrows(h, w, minimum=2)
    colors = random.sample(PALETTE, k=2)

    target_r = random.randint(4, h - 4)
    target_c = random.randint(3, w - 4)
    down_head_r = random.randint(1, target_r - 3)

    specs: list[Spec] = [
        (colors[0], (down_head_r, target_c), (1, 0)),
        (colors[1], (target_r, target_c), (0, 1)),
    ]
    if not _specs_valid(h, w, specs):
        return None
    if not _fill_random_specs(h, w, specs, total):
        return None
    return _accept_candidate(_random_orient(_build_grid(h, w, specs)), mode="arrow_block")


def generate_input() -> np.ndarray:
    generators = (_gen_free, _gen_ray_overlap, _gen_arrow_block)
    for _ in range(GEN_ATTEMPTS):
        grid = random.choice(generators)()
        if grid is not None:
            _ = main(grid)
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray puzzle input")
