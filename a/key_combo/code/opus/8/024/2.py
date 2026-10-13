from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, bbox, components_mask
except ModuleNotFoundError:
    utils_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(utils_dir) not in sys.path:
        sys.path.append(str(utils_dir))
    from arc_utils import Color, bbox, components_mask


BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
GREY = int(Color.GREY)

GRID_SIZES = (13, 15, 17)
MASK_SHAPES = ((3, 4), (4, 3), (4, 5), (5, 4))
GEN_ATTEMPTS = 500


def _symmetry_coords(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    last = n - 1
    return tuple(
        sorted(
            {
                (r, c),
                (r, last - c),
                (last - r, c),
                (last - r, last - c),
                (c, r),
                (last - c, last - r),
                (c, last - r),
                (last - c, r),
            }
        )
    )


def _recover_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    n, m = g.shape
    assert n == m

    mask = g == GREEN
    mask_box = bbox(mask)
    assert mask_box is not None
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1
    r0, c0, r1, c1 = mask_box
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1])
    assert int(mask.sum()) == int((r1 - r0 + 1) * (c1 - c0 + 1))

    restored = g.copy()
    for r, c in np.argwhere(mask):
        visible = {
            int(g[rr, cc])
            for rr, cc in _symmetry_coords(int(r), int(c), n)
            if int(g[rr, cc]) != GREEN
        }
        assert visible
        assert len(visible) == 1
        restored[int(r), int(c)] = visible.pop()
    return restored


def _ray_output(base: np.ndarray, *, swap_mapping: bool = False) -> np.ndarray:
    g = np.asarray(base, dtype=int)
    out = g.copy()
    h, w = g.shape
    red_fill = np.zeros_like(g, dtype=bool)
    blue_fill = np.zeros_like(g, dtype=bool)

    red_dirs = ((0, -1), (0, 1)) if swap_mapping else ((-1, 0), (1, 0))
    blue_dirs = ((-1, 0), (1, 0)) if swap_mapping else ((0, -1), (0, 1))

    for r, c in np.argwhere(g == RED):
        r = int(r)
        c = int(c)
        for dr, dc in red_dirs:
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BLACK:
                red_fill[rr, cc] = True
                rr += dr
                cc += dc

    for r, c in np.argwhere(g == BLUE):
        r = int(r)
        c = int(c)
        for dr, dc in blue_dirs:
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == BLACK:
                blue_fill[rr, cc] = True
                rr += dr
                cc += dc

    black = g == BLACK
    out[black & red_fill & ~blue_fill] = RED
    out[black & blue_fill & ~red_fill] = BLUE
    out[black & red_fill & blue_fill] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the masked D4-symmetric pattern, uncover hidden colored seeds, and extend their rays.

    Input structure:
    - The input is a square black-background grid whose non-green content is symmetric under the
      rotations and reflections of the square.
    - Gray pixels form a geometric frame or line structure, and red/blue single pixels are ray seeds.
    - One solid green rectangle hides part of the symmetric pattern, including some gray structure
      and one or two seed pixels.

    Rule:
    - Reconstruct each green cell from its visible dihedral-symmetry equivalents, ignoring any
      equivalent cells that are also green.
    - Replace the green rectangle with those recovered colors to obtain the full pattern.
    - On the recovered grid, extend every red seed vertically and every blue seed horizontally
      through black cells only, stopping before any non-black cell.
    - If a red extension and a blue extension claim the same black cell, color that cell yellow;
      preserve all gray structure and the original seed pixels.

    Key insight:
    - The green rectangle is not merely damage to the gray scaffold: it also hides some seed colors,
      so symmetry recovery must happen first before the color-dependent ray directions can be drawn.
    """
    restored = _recover_grid(grid)
    return _ray_output(restored)


def _draw_square_frame(grid: np.ndarray, radius: int) -> None:
    mid = grid.shape[0] // 2
    r0 = mid - radius
    c0 = mid - radius
    r1 = mid + radius
    c1 = mid + radius
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _build_base_pattern(n: int) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    center = n // 2
    outer = random.randint(max(4, center - 3), center - 2)
    inner = random.randint(2, outer - 2)
    _draw_square_frame(grid, outer)
    _draw_square_frame(grid, inner)
    if outer - inner >= 4 and random.random() < 0.4:
        middle = random.randint(inner + 2, outer - 2)
        _draw_square_frame(grid, middle)
    return grid


def _orbit_cache(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = _symmetry_coords(r, c, n)
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


_ORBIT_CACHE = {n: _orbit_cache(n) for n in GRID_SIZES}


def _mask_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _mask_is_recoverable(mask_cells: set[tuple[int, int]], n: int) -> bool:
    for r, c in mask_cells:
        orbit = _symmetry_coords(r, c, n)
        if all((rr, cc) in mask_cells for rr, cc in orbit):
            return False
    return True


def _black_orbits(base: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    n = base.shape[0]
    return [
        orbit
        for orbit in _ORBIT_CACHE[n]
        if all(int(base[r, c]) == BLACK for r, c in orbit)
    ]


def _hidden_seed_changes_output(
    base: np.ndarray,
    orbit: tuple[tuple[int, int], ...],
    color: int,
    mask_cells: set[tuple[int, int]],
) -> bool:
    full = base.copy()
    for r, c in orbit:
        full[r, c] = color

    visible_only = full.copy()
    for r, c in orbit:
        if (r, c) in mask_cells:
            visible_only[r, c] = BLACK

    return not np.array_equal(_ray_output(full), _ray_output(visible_only))


def _seed_cells_in_mask(
    full: np.ndarray, r0: int, c0: int, h: int, w: int
) -> list[tuple[int, int]]:
    cells = []
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if int(full[r, c]) in (RED, BLUE):
                cells.append((r, c))
    return cells


def _validate_instance(
    inp: np.ndarray,
    full: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> bool:
    mask = inp == GREEN
    if int(mask.sum()) != h * w:
        return False
    if len(components_mask(mask, connectivity=4)) != 1:
        return False
    mask_box = bbox(mask)
    if mask_box != (r0, c0, r0 + h - 1, c0 + w - 1):
        return False

    hidden_seed_cells = _seed_cells_in_mask(full, r0, c0, h, w)
    if not (1 <= len(hidden_seed_cells) <= 2):
        return False

    expected = _ray_output(full)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    visible_only = full.copy()
    for r, c in hidden_seed_cells:
        visible_only[r, c] = BLACK
    if np.array_equal(expected, _ray_output(visible_only)):
        return False
    if np.array_equal(expected, _ray_output(full, swap_mapping=True)):
        return False
    if np.array_equal(expected, full):
        return False
    if not np.any(full == RED) or not np.any(full == BLUE):
        return False
    if not np.any(expected == YELLOW):
        return False
    return True


def _make_instance() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    base = _build_base_pattern(n)
    black_orbits = _black_orbits(base)
    if len(black_orbits) < 3:
        return None

    shapes = list(MASK_SHAPES)
    random.shuffle(shapes)
    for h, w in shapes:
        rects = [(r0, c0) for r0 in range(n - h + 1) for c0 in range(n - w + 1)]
        random.shuffle(rects)
        for r0, c0 in rects:
            mask_cells = _mask_cells(r0, c0, h, w)
            if not _mask_is_recoverable(mask_cells, n):
                continue
            if int(np.count_nonzero(base[r0 : r0 + h, c0 : c0 + w] == GREY)) < 2:
                continue

            hidden_candidates = []
            visible_candidates = []
            for orbit in black_orbits:
                hidden_count = sum((cell in mask_cells) for cell in orbit)
                if hidden_count == 0:
                    visible_candidates.append(orbit)
                    continue
                if hidden_count > 2:
                    continue
                colors = [
                    color
                    for color in (RED, BLUE)
                    if _hidden_seed_changes_output(base, orbit, color, mask_cells)
                ]
                if colors:
                    hidden_candidates.append((orbit, hidden_count, colors))

            if not hidden_candidates or len(visible_candidates) < 1:
                continue

            for _ in range(30):
                selected: list[tuple[tuple[tuple[int, int], ...], int]] = []
                hidden_pixels = 0

                one_cell_hidden = [
                    item for item in hidden_candidates if int(item[1]) == 1
                ]
                if len(one_cell_hidden) >= 2 and random.random() < 0.5:
                    first, second = random.sample(one_cell_hidden, 2)
                    if RED not in first[2] and RED not in second[2]:
                        continue
                    if BLUE not in first[2] and BLUE not in second[2]:
                        continue
                    first_color = RED if RED in first[2] else BLUE
                    second_color = BLUE if BLUE in second[2] else RED
                    if first_color == second_color:
                        continue
                    selected = [(first[0], first_color), (second[0], second_color)]
                    hidden_pixels = 2
                else:
                    orbit, hidden_count, colors = random.choice(hidden_candidates)
                    color = random.choice(colors)
                    selected = [(orbit, color)]
                    hidden_pixels = int(hidden_count)

                used = {orbit for orbit, _color in selected}
                extras = [orbit for orbit in visible_candidates if orbit not in used]
                if not extras:
                    continue

                present_colors = {color for _orbit, color in selected}
                if RED not in present_colors:
                    selected.append((extras.pop(), RED))
                if BLUE not in present_colors:
                    if not extras:
                        continue
                    selected.append((extras.pop(), BLUE))
                if extras and random.random() < 0.5:
                    selected.append((extras.pop(), random.choice((RED, BLUE))))

                full = base.copy()
                for orbit, color in selected:
                    for r, c in orbit:
                        full[r, c] = color

                inp = full.copy()
                inp[r0 : r0 + h, c0 : c0 + w] = GREEN
                if hidden_pixels not in (1, 2):
                    continue
                if _validate_instance(inp, full, r0, c0, h, w):
                    return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        instance = _make_instance()
        if instance is not None:
            return instance
    raise RuntimeError("failed to generate a valid symmetry-seed-ray input")
