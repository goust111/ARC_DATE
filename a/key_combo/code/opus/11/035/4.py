import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

SQUARE_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _control_candidates(r: int, c: int):
    return [
        (r - 1, c, "U"),
        (r - 1, c + 1, "U"),
        (r + 2, c, "D"),
        (r + 2, c + 1, "D"),
        (r, c - 1, "L"),
        (r + 1, c - 1, "L"),
        (r, c + 2, "R"),
        (r + 1, c + 2, "R"),
    ]


def _detect_squares(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    used = np.zeros((h, w), dtype=bool)
    squares: list[dict] = []

    for r in range(h - 1):
        for c in range(w - 1):
            if used[r : r + 2, c : c + 2].any():
                continue

            block = grid[r : r + 2, c : c + 2]
            if not np.isin(block, SQUARE_COLORS).all():
                continue

            ctrl_hits = []
            for rr, cc, direction in _control_candidates(r, c):
                if _in_bounds(grid.shape, rr, cc) and grid[rr, cc] == GREY:
                    ctrl_hits.append((rr, cc, direction))
            if len(ctrl_hits) != 1:
                continue

            rr, cc, direction = ctrl_hits[0]
            color_count = len({int(v) for v in block.flat})
            if not (2 <= color_count <= 4):
                continue

            squares.append(
                {
                    "top_left": (r, c),
                    "control": (rr, cc),
                    "direction": direction,
                    "length": color_count,
                }
            )
            used[r : r + 2, c : c + 2] = True

    return squares


def _draw_rays(
    grid: np.ndarray,
    squares: list[dict],
    *,
    override_length: int | None = None,
    return_masks: bool = False,
):
    out = np.array(grid, dtype=int, copy=True)
    ray_masks: list[np.ndarray] = []

    for sq in squares:
        rr, cc = sq["control"]
        dr, dc = DIRS[sq["direction"]]
        steps = sq["length"] if override_length is None else int(override_length)
        mask = np.zeros_like(out, dtype=bool)
        for k in range(steps):
            r = rr + dr * k
            c = cc + dc * k
            if _in_bounds(out.shape, r, c):
                out[r, c] = ORANGE
                mask[r, c] = True
        ray_masks.append(mask)

    if return_masks:
        return out, ray_masks
    return out


def _fill_enclosed_background(
    grid: np.ndarray,
    *,
    return_regions: bool = False,
):
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if out[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            region = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                region.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(out.shape, nr, nc) and out[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                for rr, cc in region:
                    out[rr, cc] = PINK
                regions.append(region)

    if return_regions:
        return out, regions
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count colors inside each 2x2 square, cast a fixed-length ray from its control pixel, and fill the ray-enclosed pockets.

    Input structure:
    - The background is black.
    - There are 3 or 4 isolated 2x2 multicolor squares using only blue, red, green, and yellow.
    - Each square has exactly one grey control pixel on one of its four sides, outside the square.

    Rule:
    - For each 2x2 square, count its distinct non-black colors to get K in {2,3,4}.
    - Use the grey pixel next to that square as the ray origin, and determine the unique direction pointing away from the square.
    - Recolor the control pixel and the next K-1 cells in that direction orange, clipping at the grid boundary.
    - Treat every non-black cell as a barrier; among the remaining black cells, fill every 4-connected region that does not touch the grid edge with pink.
    - Leave the original squares and the orange ray traces unchanged.

    Key insight:
    - The geometry cannot be resolved from the control pixels alone: the number of distinct colors inside each 2x2 square determines the exact ray length, and those lengths decide which background pockets become closed and therefore filled.
    """
    g = np.asarray(grid, dtype=int)
    squares = _detect_squares(g)
    ray_grid = _draw_rays(g, squares)
    return _fill_enclosed_background(ray_grid)


def _make_square(k: int) -> np.ndarray:
    palette = list(SQUARE_COLORS)
    random.shuffle(palette)

    if k == 4:
        vals = palette[:4]
    elif k == 3:
        vals = palette[:3] + [palette[0]]
        random.shuffle(vals)
    elif k == 2:
        vals = [palette[0], palette[0], palette[1], palette[1]]
        if random.random() < 0.5:
            vals[random.randrange(4)] = palette[random.randrange(2)]
        random.shuffle(vals)
    else:
        raise ValueError("square color count must be 2, 3, or 4")

    sprite = np.array(vals, dtype=int).reshape(2, 2)
    if len({int(v) for v in sprite.flat}) != k:
        sprite[0, 0] = palette[0]
        sprite[1, 1] = palette[1]
    assert len({int(v) for v in sprite.flat}) == k
    return sprite


def _apply_square(grid: np.ndarray, r: int, c: int, sprite: np.ndarray) -> None:
    grid[r : r + 2, c : c + 2] = sprite


def _place_core_pair(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    r = random.randint(2, h - 6)
    c = random.randint(2, w - 10)

    left_square = _make_square(4)
    right_square = _make_square(4)

    _apply_square(grid, r, c, left_square)
    _apply_square(grid, r + 1, c + 6, right_square)
    grid[r, c + 2] = GREY
    grid[r + 2, c + 5] = GREY

    return [
        {"top_left": (r, c), "control": (r, c + 2), "direction": "R", "length": 4},
        {"top_left": (r + 1, c + 6), "control": (r + 2, c + 5), "direction": "L", "length": 4},
    ]


def _input_neighbors_nonblack(grid: np.ndarray, r: int, c: int) -> int:
    total = 0
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] != BLACK:
            total += 1
    return total


def _top_left_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _place_extra_square(grid: np.ndarray, placed: list[dict], k: int) -> dict | None:
    h, w = grid.shape
    options = [
        ("U", (-1, 0), "U"),
        ("U", (-1, 1), "U"),
        ("D", (2, 0), "D"),
        ("D", (2, 1), "D"),
        ("L", (0, -1), "L"),
        ("L", (1, -1), "L"),
        ("R", (0, 2), "R"),
        ("R", (1, 2), "R"),
    ]

    for _ in range(300):
        r = random.randint(1, h - 4)
        c = random.randint(1, w - 4)
        if any(_top_left_distance((r, c), sq["top_left"]) < 6 for sq in placed):
            continue
        if np.any(grid[r : r + 2, c : c + 2] != BLACK):
            continue

        sprite = _make_square(k)
        trial_options = options[:]
        random.shuffle(trial_options)
        for _name, (dr0, dc0), direction in trial_options:
            rr = r + dr0
            cc = c + dc0
            if not _in_bounds(grid.shape, rr, cc) or grid[rr, cc] != BLACK:
                continue

            test = np.array(grid, copy=True)
            _apply_square(test, r, c, sprite)
            test[rr, cc] = GREY
            if _input_neighbors_nonblack(test, rr, cc) != 1:
                continue

            _apply_square(grid, r, c, sprite)
            grid[rr, cc] = GREY
            return {"top_left": (r, c), "control": (rr, cc), "direction": direction, "length": k}

    return None


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=int, copy=True)


def _regions_from_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            region = []
            while q:
                rr, cc = q.popleft()
                region.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == color and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append(region)
    return regions


def _region_touches_multiple_rays(region: list[tuple[int, int]], ray_masks: list[np.ndarray]) -> bool:
    touched = set()
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(ray_masks[0].shape, nr, nc):
                continue
            for idx, mask in enumerate(ray_masks):
                if mask[nr, nc]:
                    touched.add(idx)
    return len(touched) >= 2


def _validate_generated(inp: np.ndarray) -> bool:
    squares = _detect_squares(inp)
    if len(squares) not in (3, 4):
        return False
    if not any(sq["length"] != 4 for sq in squares):
        return False

    ray_grid, ray_masks = _draw_rays(inp, squares, return_masks=True)
    out, enclosed = _fill_enclosed_background(ray_grid, return_regions=True)
    if not enclosed:
        return False
    if not any(_region_touches_multiple_rays(region, ray_masks) for region in enclosed):
        return False
    if not np.array_equal(out, main(inp)):
        return False

    fixed4 = _fill_enclosed_background(_draw_rays(inp, squares, override_length=4))
    if np.array_equal(fixed4, out):
        return False

    no_fill = ray_grid
    if np.array_equal(no_fill, out):
        return False

    pink_regions = _regions_from_color(out, PINK)
    if not pink_regions:
        return False

    for sq in squares:
        rr, cc = sq["control"]
        if inp[rr, cc] != GREY:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        grid = np.zeros((h, w), dtype=int)

        placed = _place_core_pair(grid)
        extra_total = random.choice((1, 2))
        extra_lengths = [random.choice((2, 3))]
        if extra_total == 2:
            extra_lengths.append(random.choice((2, 3, 4)))

        ok = True
        for k in extra_lengths:
            sq = _place_extra_square(grid, placed, k)
            if sq is None:
                ok = False
                break
            placed.append(sq)
        if not ok:
            continue

        grid = _random_transform(grid)
        if _validate_generated(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle input")
