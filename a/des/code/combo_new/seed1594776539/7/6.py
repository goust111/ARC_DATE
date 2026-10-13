from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)

LEGEND_SIDES = ("up", "right", "down", "left")
SIDE_TO_DIRECTION = {
    "up": (1, 0),
    "right": (0, -1),
    "down": (-1, 0),
    "left": (0, 1),
}

GEN_RETRIES = 400
BLOB_RETRIES = 80


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return bool(np.any(bg & ~seen))


def _fill_region(out: np.ndarray, cells: list[tuple[int, int]], direction: tuple[int, int], seed_color: int) -> None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    dr, dc = direction
    if dr == 1:
        base = min(rows)
        index = lambda r, c: r - base
    elif dr == -1:
        base = max(rows)
        index = lambda r, c: base - r
    elif dc == 1:
        base = min(cols)
        index = lambda r, c: c - base
    else:
        base = max(cols)
        index = lambda r, c: base - c

    for r, c in cells:
        out[r, c] = seed_color if index(r, c) % 2 == 0 else MAROON


def _corner_centers(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    centers = [(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)]
    out: list[tuple[int, int]] = []
    for center in centers:
        if center not in out:
            out.append(center)
    return out


def _find_legend(grid: np.ndarray) -> tuple[dict[int, tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[dict[int, tuple[int, int]], np.ndarray]] = []

    for r, c in _corner_centers(grid.shape):
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            continue
        if int(grid[r, c]) != BLACK:
            continue

        arms = {
            "up": (r - 1, c),
            "right": (r, c + 1),
            "down": (r + 1, c),
            "left": (r, c - 1),
        }
        values = {side: int(grid[rr, cc]) for side, (rr, cc) in arms.items()}
        if any(value in (BLACK, GREY) for value in values.values()):
            continue
        if len(set(values.values())) != 4:
            continue

        diagonals = [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
        if any(int(grid[rr, cc]) != BLACK for rr, cc in diagonals):
            continue

        mapping = {values[side]: SIDE_TO_DIRECTION[side] for side in LEGEND_SIDES}
        legend_mask = np.zeros((h, w), dtype=bool)
        legend_mask[r - 1 : r + 2, c - 1 : c + 2] = True
        candidates.append((mapping, legend_mask))

    assert len(candidates) == 1, "expected exactly one valid corner legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed cavity with seed-colored and maroon stripes according to the corner legend.

    Input structure:
    - Black background with several Grey wall loops that enclose 4-connected cavities.
    - Each cavity contains exactly one non-Black, non-Grey seed pixel.
    - One corner holds a 3x3 color-ring legend: four colored arm pixels around a Black center.

    Rule:
    - Read the legend as a color-to-direction code where each arm color points to the opposite fill direction.
    - Ignore the legend, find every non-boundary cavity enclosed by Grey walls, and read its unique seed color.
    - Fill the whole cavity with 1-pixel stripes perpendicular to the fill direction, starting from the cavity edge on that side and alternating seed color then Maroon.

    Key insight:
    - The legend colors do not name fill colors directly; each color encodes a direction via the opposite arm, and that direction anchors the stripe order across the enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _find_legend(g)
    passable = (g != GREY) & (~legend_mask)
    h, w = g.shape

    out = g.copy()
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        seeds = [(r, c) for r, c in comp if int(g[r, c]) not in (BLACK, GREY)]
        assert len(seeds) == 1, "each enclosed cavity must contain exactly one seed"
        seed_r, seed_c = seeds[0]
        seed_color = int(g[seed_r, seed_c])
        assert seed_color in mapping, "seed color must be present in the legend"
        _fill_region(out, comp, mapping[seed_color], seed_color)

    return out


def _random_blob(rng: random.Random) -> np.ndarray | None:
    for _ in range(BLOB_RETRIES):
        box_h = rng.randint(3, 6)
        box_w = rng.randint(3, 6)
        area_lo = max(4, (box_h * box_w) // 3)
        area_hi = min(box_h * box_w - 1, (2 * box_h * box_w) // 3 + 1)
        if area_lo > area_hi:
            continue
        target_area = rng.randint(area_lo, area_hi)

        seed = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {seed}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))

        if len(cells) != target_area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_mask(mask)

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if _has_hole(mask):
            continue

        area = int(mask.sum())
        if area == mask.shape[0] * mask.shape[1] and area > 4 and rng.random() < 0.8:
            continue
        return mask
    return None


def _make_region_patches(
    rng: random.Random, seed_color: int, direction: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    interior = _random_blob(rng)
    if interior is None:
        return None

    ih, iw = interior.shape
    cavity = np.zeros((ih + 2, iw + 2), dtype=bool)
    cavity[1 : ih + 1, 1 : iw + 1] = interior

    wall = np.zeros_like(cavity)
    cells = [(int(r), int(c)) for r, c in np.argwhere(cavity)]
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if not cavity[nr, nc]:
                wall[nr, nc] = True

    patch_in = np.full(cavity.shape, BLACK, dtype=int)
    patch_out = np.full(cavity.shape, BLACK, dtype=int)
    patch_in[wall] = GREY
    patch_out[wall] = GREY

    seed_r, seed_c = rng.choice(cells)
    patch_in[seed_r, seed_c] = seed_color
    _fill_region(patch_out, cells, direction, seed_color)
    return patch_in, patch_out


def _legend_center_for_corner(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, w - 2
    if corner == "bl":
        return h - 2, 1
    if corner == "br":
        return h - 2, w - 2
    raise ValueError(corner)


def _stamp_legend(
    grid: np.ndarray, expected: np.ndarray, occupied: np.ndarray, corner: str, side_to_color: dict[str, int]
) -> None:
    r, c = _legend_center_for_corner(grid.shape, corner)
    arms = {
        "up": (r - 1, c),
        "right": (r, c + 1),
        "down": (r + 1, c),
        "left": (r, c - 1),
    }
    for side, color in side_to_color.items():
        rr, cc = arms[side]
        grid[rr, cc] = color
        expected[rr, cc] = color

    r0 = max(0, r - 2)
    r1 = min(grid.shape[0], r + 3)
    c0 = max(0, c - 2)
    c1 = min(grid.shape[1], c + 3)
    occupied[r0:r1, c0:c1] = True


def _place_patch(
    rng: random.Random,
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
) -> bool:
    h, w = grid.shape
    ph, pw = patch_in.shape
    origins: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            origins.append((r0, c0))
    if not origins:
        return False

    r0, c0 = rng.choice(origins)
    grid[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(grid[r0 : r0 + ph, c0 : c0 + pw], patch_in)
    expected[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(expected[r0 : r0 + ph, c0 : c0 + pw], patch_out)
    occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(2, 4)
    min_side = 16 + 2 * region_count
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)

    grid = np.full((h, w), BLACK, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((h, w), dtype=bool)

    legend_colors = rng.sample(COLOR_POOL, 4)
    side_to_color = {side: color for side, color in zip(LEGEND_SIDES, legend_colors, strict=False)}
    corner = rng.choice(("tl", "tr", "bl", "br"))
    _stamp_legend(grid, expected, occupied, corner, side_to_color)

    chosen_sides = rng.sample(list(LEGEND_SIDES), region_count)
    rng.shuffle(chosen_sides)
    for side in chosen_sides:
        patch_pair = _make_region_patches(rng, side_to_color[side], SIDE_TO_DIRECTION[side])
        if patch_pair is None:
            return None
        patch_in, patch_out = patch_pair
        if not _place_patch(rng, grid, expected, occupied, patch_in, patch_out):
            return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
