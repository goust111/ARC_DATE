import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
SEPARATOR = 9
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_RETRIES = 400


@dataclass(frozen=True)
class Square:
    r0: int
    c0: int
    size: int
    outer: int
    inner: int

    @property
    def r1(self) -> int:
        return self.r0 + self.size - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.size - 1

    @property
    def center(self) -> tuple[int, int]:
        d = self.size // 2
        return self.r0 + d, self.c0 + d


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_square(grid: np.ndarray, comp: list[tuple[int, int]]) -> Square | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 3 or h % 2 == 0:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(patch == BLACK):
        return None
    if len(comp) != h * w:
        return None
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    outer_vals = np.unique(patch[border_mask])
    inner_vals = np.unique(patch[~border_mask])
    if len(outer_vals) != 1 or len(inner_vals) != 1:
        return None
    outer = int(outer_vals[0])
    inner = int(inner_vals[0])
    if outer == inner:
        return None
    return Square(r0=r0, c0=c0, size=h, outer=outer, inner=inner)


def _detect_squares(grid: np.ndarray) -> list[Square]:
    comps = _components8(grid != BLACK)
    squares: list[Square] = []
    for comp in comps:
        sq = _extract_square(grid, comp)
        assert sq is not None, "every foreground component must be a solid two-color bordered square"
        squares.append(sq)
    squares.sort(key=lambda sq: (sq.center[0], sq.center[1], sq.size))
    assert squares, "expected at least one square"
    return squares


def _swap_square(out: np.ndarray, sq: Square) -> None:
    out[sq.r0 : sq.r1 + 1, sq.c0 : sq.c1 + 1] = sq.inner
    if sq.size > 2:
        out[sq.r0 + 1 : sq.r1, sq.c0 + 1 : sq.c1] = sq.outer


def _fill_background(out: np.ndarray, squares: list[Square]) -> tuple[np.ndarray, np.ndarray]:
    centers = np.array([sq.center for sq in squares], dtype=int)
    colors = np.array([sq.outer for sq in squares], dtype=int)
    diff_tie = np.zeros_like(out, dtype=bool)
    same_tie = np.zeros_like(out, dtype=bool)
    for r, c in np.argwhere(out == BLACK):
        distances = np.abs(centers[:, 0] - r) + np.abs(centers[:, 1] - c)
        best = int(distances.min())
        nearest_colors = np.unique(colors[distances == best])
        if len(nearest_colors) == 1:
            out[r, c] = int(nearest_colors[0])
            if int((distances == best).sum()) > 1:
                same_tie[r, c] = True
        else:
            out[r, c] = SEPARATOR
            diff_tie[r, c] = True
    return diff_tie, same_tie


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each bordered square's core/frame colors, then fill the black background by nearest square-center diffusion.

    Input structure:
    - A black background contains several separated odd-sized filled squares with a 1-pixel border and a uniform inner core.
    - Each square uses exactly two non-black colors: one on the border ring and one in the inner square.
    - The squares do not overlap, and each square's geometric center is a single grid cell.

    Rule:
    - For every bordered square, exchange the border color and the inner-core color while keeping the square geometry unchanged.
    - Use each square center as a Manhattan-distance seed, with diffusion color equal to the swapped inner color, i.e. the square's original border color.
    - Recolor every black background cell by the nearest seed's diffusion color; if multiple nearest seeds exist and their diffusion colors are not all the same, color that cell maroon (9).
    - Leave the swapped squares themselves unchanged after the background fill.

    Key insight:
    - The large visible core is not the fill source: after the core/frame swap, the new core color, which was originally the border color, becomes the seed color for the nearest-partition fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    squares = _detect_squares(g)
    out = g.copy()
    for sq in squares:
        _swap_square(out, sq)
    _fill_background(out, squares)
    return out


def _bbox_from_center(center: tuple[int, int], size: int) -> tuple[int, int, int, int]:
    r, c = center
    d = size // 2
    return r - d, c - d, r + d, c + d


def _boxes_far_enough(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 3,
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_square(grid: np.ndarray, sq: Square) -> None:
    grid[sq.r0 : sq.r1 + 1, sq.c0 : sq.c1 + 1] = sq.outer
    if sq.size > 2:
        grid[sq.r0 + 1 : sq.r1, sq.c0 + 1 : sq.c1] = sq.inner


def _semantic_stats(grid: np.ndarray) -> tuple[list[Square], np.ndarray, np.ndarray, np.ndarray]:
    squares = _detect_squares(grid)
    probe = grid.copy()
    for sq in squares:
        _swap_square(probe, sq)
    diff_tie, same_tie = _fill_background(probe, squares)
    return squares, probe, diff_tie, same_tie


def _sample_even(lo: int, hi: int) -> int | None:
    choices = [x for x in range(lo, hi + 1) if x % 2 == 0]
    if not choices:
        return None
    return random.choice(choices)


def _random_free_square(
    h: int,
    w: int,
    size: int,
    boxes: list[tuple[int, int, int, int]],
    attempts: int = 80,
) -> tuple[int, int] | None:
    d = size // 2
    for _ in range(attempts):
        r = random.randint(d, h - 1 - d)
        c = random.randint(d, w - 1 - d)
        box = _bbox_from_center((r, c), size)
        if all(_boxes_far_enough(box, other) for other in boxes):
            return r, c
    return None


def _build_layout() -> tuple[np.ndarray, bool]:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    num_squares = random.choice([3, 4])
    need_same_tie = random.random() < 0.55

    sizes = [random.choice([3, 5]) for _ in range(num_squares)]
    a_size, b_size = sizes[0], sizes[1]
    a_rad, b_rad = a_size // 2, b_size // 2
    min_dx = a_rad + b_rad + 4
    if min_dx % 2 == 1:
        min_dx += 1
    max_dx = (w - 1 - b_rad) - a_rad
    dx = _sample_even(min_dx, max_dx)
    if dx is None:
        raise ValueError("no horizontal spacing")

    c_a_hi = w - 1 - b_rad - dx
    if c_a_hi < a_rad:
        raise ValueError("no base column")

    if need_same_tie:
        c_size = sizes[2]
        c_rad = c_size // 2
        min_dy = a_rad + c_rad + 4
        if min_dy % 2 == 1:
            min_dy += 1
        max_dy = (h - 1 - c_rad) - a_rad
        dy = _sample_even(min_dy, max_dy)
        if dy is None:
            raise ValueError("no vertical spacing")
        r_a_hi = h - 1 - c_rad - dy
        if r_a_hi < a_rad:
            raise ValueError("no base row")
        r_a = random.randint(a_rad, r_a_hi)
    else:
        dy = None
        r_a = random.randint(a_rad, h - 1 - a_rad)

    c_a = random.randint(a_rad, c_a_hi)
    centers = [(r_a, c_a), (r_a, c_a + dx)]
    if need_same_tie:
        centers.append((r_a + dy, c_a))

    boxes = [_bbox_from_center(center, size) for center, size in zip(centers, sizes)]
    while len(centers) < num_squares:
        size = sizes[len(centers)]
        center = _random_free_square(h, w, size, boxes)
        if center is None:
            raise ValueError("failed to place extra square")
        centers.append(center)
        boxes.append(_bbox_from_center(center, size))

    colors = GEN_COLORS.copy()
    random.shuffle(colors)
    if need_same_tie:
        outer_a = colors.pop()
        outer_b = colors.pop()
        outer_colors = [outer_a, outer_b, outer_a]
        while len(outer_colors) < num_squares:
            outer_colors.append(colors.pop())
    else:
        outer_colors = [colors.pop() for _ in range(num_squares)]

    used_outer = set(outer_colors)
    inner_pool = [color for color in GEN_COLORS if color not in used_outer]
    random.shuffle(inner_pool)
    inner_colors = inner_pool[:num_squares]

    pairs = set(zip(outer_colors, inner_colors))
    assert len(pairs) == num_squares

    grid = np.zeros((h, w), dtype=int)
    for idx in range(num_squares):
        r, c = centers[idx]
        size = sizes[idx]
        d = size // 2
        sq = Square(r0=r - d, c0=c - d, size=size, outer=outer_colors[idx], inner=inner_colors[idx])
        _draw_square(grid, sq)

    if random.random() < 0.5:
        grid = np.flipud(grid)
    if random.random() < 0.5:
        grid = np.fliplr(grid)
    if random.random() < 0.5:
        grid = grid.T.copy()
    return grid, need_same_tie


def _valid_generated_grid(grid: np.ndarray, need_same_tie: bool) -> bool:
    try:
        squares, out, diff_tie, same_tie = _semantic_stats(grid)
    except AssertionError:
        return False

    if len(squares) < 3 or len(squares) > 4:
        return False
    if any(sq.size not in (3, 5) for sq in squares):
        return False
    if len({(sq.outer, sq.inner) for sq in squares}) != len(squares):
        return False
    if not diff_tie.any():
        return False
    if need_same_tie and not same_tie.any():
        return False
    if np.any(out == BLACK):
        return False
    if len(np.unique(out)) < 3:
        return False
    if len({sq.outer for sq in squares}) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, need_same_tie = _build_layout()
        except ValueError:
            continue
        if _valid_generated_grid(grid, need_same_tie):
            return grid
    raise RuntimeError("failed to generate a valid input")
