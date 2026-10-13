import random
from typing import Dict, List, Optional, Tuple

import numpy as np

BG_COLOR = 8
GREY = 5
YELLOW = 4

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _dominant_color(grid: np.ndarray) -> int:
    flat = grid.reshape(-1)
    counts = np.bincount(flat.astype(int), minlength=10)
    return int(np.argmax(counts))


def _arrow_pixels(center: Tuple[int, int], direction: Tuple[int, int]) -> List[Tuple[int, int]]:
    r, c = center
    dr, dc = direction
    # center + one stem cell + two lateral cells
    return [(r, c), (r + dr, c + dc), (r + dc, c + dr), (r - dc, c - dr)]


def _find_arrows(grid: np.ndarray, bg: int) -> Tuple[List[Dict[str, object]], Dict[Tuple[int, int], int]]:
    h, w = grid.shape
    arrows: List[Dict[str, object]] = []
    pixel_to_arrow: Dict[Tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, GREY, YELLOW):
                continue

            for d in DIRECTIONS.values():
                cells = _arrow_pixels((r, c), d)
                valid = True
                for rr, cc in cells:
                    if not _inside(h, w, rr, cc) or int(grid[rr, cc]) != color:
                        valid = False
                        break
                if not valid:
                    continue

                # avoid re-detecting same arrow via its own limb as center
                if any(pos in pixel_to_arrow for pos in cells):
                    continue

                idx = len(arrows)
                arrows.append(
                    {
                        "center": (r, c),
                        "direction": d,
                        "color": color,
                        "pixels": cells,
                    }
                )
                for pos in cells:
                    pixel_to_arrow[pos] = idx
                break

    return arrows, pixel_to_arrow


def _scan_next(
    grid: np.ndarray,
    pixel_to_arrow: Dict[Tuple[int, int], int],
    tip: Tuple[int, int],
    direction: Tuple[int, int],
) -> Tuple[Optional[int], Tuple[int, int]]:
    h, w = grid.shape
    dr, dc = direction
    stop_r, stop_c = tip
    r, c = tip[0] + dr, tip[1] + dc

    while _inside(h, w, r, c):
        cell = int(grid[r, c])
        if cell == GREY:
            return None, (stop_r, stop_c)

        idx = pixel_to_arrow.get((r, c))
        if idx is not None:
            return idx, (stop_r, stop_c)

        stop_r, stop_c = r, c
        r += dr
        c += dc

    return None, (stop_r, stop_c)


def _paint_ray(
    out: np.ndarray,
    bg: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    direction: Tuple[int, int],
    color: int,
) -> None:
    h, w = out.shape
    r, c = start
    er, ec = end
    dr, dc = direction

    while _inside(h, w, r, c):
        if out[r, c] == bg:
            out[r, c] = color
        if (r, c) == (er, ec):
            break
        r += dr
        c += dc


def _opposite(direction: Tuple[int, int]) -> Tuple[int, int]:
    dr, dc = direction
    return -dr, -dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the "arrow chain" puzzle: trace arrows in line-of-sight order and paint light rays.

    Input
    -----
    grid: np.ndarray
        2-D ARC board encoded with integer colors.

    Output
    ------
    np.ndarray
        Same shape as input with added light-ray pixels.

    Procedure
    ---------
    1) Detect all 4-pixel T-shaped arrows and their directions.
    2) Find the unique start arrow from the yellow tail marker.
    3) March through arrows by shooting rays in each arrow's direction and paint
       the path until the first hit obstacle/boundary/next arrow.
    """

    if not isinstance(grid, np.ndarray):
        raise TypeError("main expects a numpy ndarray")
    if grid.ndim != 2:
        raise ValueError("main expects a 2D grid")

    grid = grid.astype(int, copy=False)
    out = np.array(grid, dtype=int, copy=True)
    bg = _dominant_color(out)

    arrows, pixel_to_arrow = _find_arrows(out, bg)
    if not arrows:
        return out

    start_index: Optional[int] = None
    h, w = out.shape

    for i, arrow in enumerate(arrows):
        r, c = arrow["center"]
        dr, dc = arrow["direction"]
        tail = (r + _opposite((dr, dc))[0], c + _opposite((dr, dc))[1])
        if _inside(h, w, *tail) and out[tail] == YELLOW:
            start_index = i
            break

    if start_index is None:
        return out

    visited = set()
    current = start_index

    while current not in visited:
        visited.add(current)
        arrow = arrows[current]
        center = arrow["center"]
        direction = arrow["direction"]
        color = arrow["color"]

        tip = (center[0] + direction[0], center[1] + direction[1])
        if not _inside(h, w, *tip):
            break

        nxt, stop = _scan_next(out, pixel_to_arrow, tip, direction)
        _paint_ray(out, bg, tip, stop, direction, color)

        if nxt is None:
            break
        current = nxt

    return out


def _can_place_arrow(grid: np.ndarray, occupied: set, center: Tuple[int, int], direction: Tuple[int, int]) -> bool:
    h, w = grid.shape
    cells = _arrow_pixels(center, direction)
    for r, c in cells:
        if not _inside(h, w, r, c):
            return False
        if (r, c) in occupied:
            return False
        if int(grid[r, c]) != BG_COLOR:
            return False
    return True


def _place_arrow(
    grid: np.ndarray,
    occupied: set,
    center: Tuple[int, int],
    direction: Tuple[int, int],
    color: int,
) -> None:
    for rr, cc in _arrow_pixels(center, direction):
        grid[rr, cc] = color
        occupied.add((rr, cc))


def generate_input() -> np.ndarray:
    """
    Generate a valid random ARC-like input for this task.

    Returns
    -------
    np.ndarray
        Random grid with <=30 size, integer colors, one marked start arrow,
        and optional extra gray obstacles/arrows.
    """

    color_choices = [1, 2, 3, 6, 7, 9]
    dir_items = list(DIRECTIONS.items())

    for _ in range(80):
        h = random.randint(10, 30)
        w = random.randint(10, 30)
        grid = np.full((h, w), BG_COLOR, dtype=int)
        occupied = set()

        obstacle_count = random.randint(0, max(1, (h * w) // 18))
        for _ in range(obstacle_count):
            for _ in range(50):
                r = random.randint(0, h - 1)
                c = random.randint(0, w - 1)
                if (r, c) not in occupied:
                    grid[r, c] = GREY
                    occupied.add((r, c))
                    break

        chain_len = random.randint(1, 6)
        arrows = []
        # Start arrow
        start = None
        for _ in range(500):
            dname, dvec = random.choice(dir_items)
            color = random.choice(color_choices)
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            if not _can_place_arrow(grid, occupied, (r, c), dvec):
                continue

            tail = (r - dvec[0], c - dvec[1])
            if not _inside(h, w, *tail) or grid[tail] != BG_COLOR:
                continue

            _place_arrow(grid, occupied, (r, c), dvec, color)
            start = (r, c, dvec, color)
            arrows.append((r, c, dvec, color))
            grid[tail] = YELLOW
            occupied.add(tail)
            break

        if start is None:
            continue

        current = start
        # Extend chain by placing the next arrow along the current direction line.
        for _ in range(chain_len - 1):
            r, c, dvec, _ = current
            dr, dc = dvec
            placed = False
            for _ in range(120):
                dist = random.randint(2, 7)
                nr = r + dr * dist
                nc = c + dc * dist
                ndvec = random.choice([v for v in DIRECTIONS.values()])
                ncolor = random.choice(color_choices)
                if not _can_place_arrow(grid, occupied, (nr, nc), ndvec):
                    continue
                _place_arrow(grid, occupied, (nr, nc), ndvec, ncolor)
                current = (nr, nc, ndvec, ncolor)
                arrows.append(current)
                placed = True
                break
            if not placed:
                break

        # Optional extra disconnected arrows for diversity.
        extra = random.randint(0, 3)
        for _ in range(extra):
            for _ in range(80):
                dname, dvec = random.choice(dir_items)
                color = random.choice(color_choices)
                r = random.randint(1, h - 2)
                c = random.randint(1, w - 2)
                if _can_place_arrow(grid, occupied, (r, c), dvec):
                    _place_arrow(grid, occupied, (r, c), dvec, color)
                    break

        return grid

    # Fallback minimal guaranteed valid sample.
    grid = np.full((12, 12), BG_COLOR, dtype=int)
    _place_arrow(grid, {(1, 1)}, (4, 5), DIRECTIONS["right"], 1)
    grid[4, 4] = YELLOW
    return grid


if __name__ == "__main__":
    sample = generate_input()
    result = main(sample)
    print(sample.shape, np.sum(sample != result), "pixels changed")
