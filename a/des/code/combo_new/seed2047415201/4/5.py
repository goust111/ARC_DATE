import random
from collections import defaultdict

import numpy as np


GREY = 5
MAROON = 9
FOREGROUND_COLORS = (0, 1, 2, 3, 4, 6, 7, 8)
MIN_PARTICIPANTS = 4
GEN_ATTEMPTS = 500


def _symmetry_centers(grid: np.ndarray) -> dict[tuple[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(grid != GREY):
        by_color[int(grid[r, c])].append((int(r), int(c)))

    center_pixels: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for pts in by_color.values():
        for i, (r0, c0) in enumerate(pts):
            for r1, c1 in pts[i + 1 :]:
                sr, sc = r0 + r1, c0 + c1
                if sr % 2 or sc % 2:
                    continue
                cr, cc = sr // 2, sc // 2
                if 0 <= cr < h and 0 <= cc < w:
                    center_pixels[(cr, cc)].add((r0, c0))
                    center_pixels[(cr, cc)].add((r1, c1))

    return {
        center: pixels
        for center, pixels in center_pixels.items()
        if len(pixels) >= MIN_PARTICIPANTS
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect hidden 180-degree symmetry centers, mark them, and erase asymmetric noise.

    Input structure:
    - The grid background is Grey.
    - Sparse colored foreground pixels form several hidden point-symmetric fragments.
    - A real symmetry center is a Grey cell in the input; extra isolated foreground pixels are noise.

    Rule:
    - For every grid cell, collect foreground pixels whose same-colored 180-degree counterpart exists across that cell.
    - If at least four foreground pixels participate around a cell, keep those pixels as one symmetric pattern.
    - Paint every detected center Maroon, and turn every foreground pixel that participates in no detected symmetry back to Grey.

    Key insight:
    - The center is not given by a visible marker; it is the common midpoint of multiple same-colored reflected pixel pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    centers = _symmetry_centers(g)
    participating: set[tuple[int, int]] = set()
    for pts in centers.values():
        participating.update(pts)

    out = np.full(g.shape, GREY, dtype=int)
    for r, c in participating:
        out[r, c] = int(g[r, c])
    for r, c in centers:
        out[r, c] = MAROON
    return out


# --------------- generator ---------------

HALF_PLANE_DIRS = (
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
    (1, 2),
    (2, 1),
    (2, -1),
)


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _canon_offset(dr: int, dc: int) -> tuple[int, int]:
    if dr < 0 or (dr == 0 and dc < 0):
        return -dr, -dc
    return dr, dc


def _offset_pool(h: int, w: int, center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    cr, cc = center
    offsets = []
    seen = set()
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if max(abs(dr), abs(dc)) < 2:
                continue
            off = _canon_offset(dr, dc)
            if off in seen:
                continue
            rr0, cc0 = cr + off[0], cc + off[1]
            rr1, cc1 = cr - off[0], cc - off[1]
            if _in_bounds(h, w, rr0, cc0) and _in_bounds(h, w, rr1, cc1):
                seen.add(off)
                offsets.append(off)
    return offsets


def _line_offsets(h: int, w: int, center: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    dirs = list(HALF_PLANE_DIRS)
    random.shuffle(dirs)
    for dr, dc in dirs:
        offsets = []
        for k in range(1, 8):
            off = (dr * k, dc * k)
            if max(abs(off[0]), abs(off[1])) < 2:
                continue
            cr, cc = center
            if _in_bounds(h, w, cr + off[0], cc + off[1]) and _in_bounds(h, w, cr - off[0], cc - off[1]):
                offsets.append(off)
        if len(offsets) >= count:
            return random.sample(offsets, count)
    return None


def _corner_offsets(h: int, w: int, center: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    for _ in range(40):
        a = random.randint(2, 6)
        b = random.randint(2, 6)
        candidates = [(a, 0), (0, b), (a, b), (a, -b), (a + 1, b), (a, b + 1)]
        random.shuffle(candidates)
        valid = []
        cr, cc = center
        for off in candidates:
            if _in_bounds(h, w, cr + off[0], cc + off[1]) and _in_bounds(h, w, cr - off[0], cc - off[1]):
                valid.append(off)
        if len(valid) >= count:
            return valid[:count]
    return None


def _choose_offsets(h: int, w: int, center: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    mode = random.choice(("cloud", "line", "corner"))
    if mode == "line":
        return _line_offsets(h, w, center, count)
    if mode == "corner":
        return _corner_offsets(h, w, center, count)

    pool = _offset_pool(h, w, center, random.randint(4, 8))
    if len(pool) < count:
        return None
    return random.sample(pool, count)


def _touches_other(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied and (rr, cc) not in cells:
                    return True
    return False


def _try_add_pattern(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    centers: set[tuple[int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    h, w = grid.shape
    for _ in range(80):
        center = (random.randint(2, h - 3), random.randint(2, w - 3))
        if center in occupied or center in centers:
            continue

        pair_count = random.randint(2, 5)
        offsets = _choose_offsets(h, w, center, pair_count)
        if offsets is None:
            continue

        colors = random.sample(FOREGROUND_COLORS, pair_count)
        cells: set[tuple[int, int]] = set()
        assignments = []
        cr, cc = center
        ok = True
        for (dr, dc), color in zip(offsets, colors):
            p0 = (cr + dr, cc + dc)
            p1 = (cr - dr, cc - dc)
            if p0 in occupied or p1 in occupied or p0 in centers or p1 in centers:
                ok = False
                break
            if p0 == center or p1 == center:
                ok = False
                break
            cells.add(p0)
            cells.add(p1)
            assignments.append((p0, p1, color))
        if not ok or len(cells) != pair_count * 2:
            continue
        if _touches_other(cells, occupied):
            continue

        for p0, p1, color in assignments:
            grid[p0] = color
            grid[p1] = color
        occupied.update(cells)
        centers.add(center)
        return center, cells
    return None


def _try_add_noise(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    centers: set[tuple[int, int]],
    count: int,
) -> bool:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _ in range(80):
            p = (random.randrange(h), random.randrange(w))
            if p in occupied or p in centers:
                continue
            if _touches_other({p}, occupied):
                continue
            grid[p] = random.choice(FOREGROUND_COLORS)
            occupied.add(p)
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_output(
    inp: np.ndarray,
    centers: set[tuple[int, int]],
    participants: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full(inp.shape, GREY, dtype=int)
    for r, c in participants:
        out[r, c] = int(inp[r, c])
    for r, c in centers:
        out[r, c] = MAROON
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(11, 28)
    w = random.randint(11, 28)
    grid = np.full((h, w), GREY, dtype=int)

    occupied: set[tuple[int, int]] = set()
    centers: set[tuple[int, int]] = set()
    participants: set[tuple[int, int]] = set()

    pattern_count = random.randint(1, 4)
    for _ in range(pattern_count):
        placed = _try_add_pattern(grid, occupied, centers)
        if placed is None:
            return None
        _, cells = placed
        participants.update(cells)

    noise_count = random.randint(1, 7)
    if not _try_add_noise(grid, occupied, centers, noise_count):
        return None

    expected = _expected_output(grid, centers, participants)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid symmetry-center input")
