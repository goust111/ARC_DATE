from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _source_aligned_quadrants(motif: np.ndarray) -> list[np.ndarray]:
    side = int(motif.shape[0])
    assert motif.shape == (side, side) and side % 2 == 0
    n = side // 2
    top_left = motif[:n, :n]
    top_right = motif[:n, n:]
    bottom_left = motif[n:, :n]
    bottom_right = motif[n:, n:]

    return [
        top_left.copy(),
        np.rot90(top_right, 1),
        np.rot90(bottom_right, 2),
        np.rot90(bottom_left, -1),
    ]


def _missing_only_from_aligned(aligned: list[np.ndarray], bg: int) -> np.ndarray:
    n = int(aligned[0].shape[0])
    out = np.full((n, n), int(bg), dtype=int)
    missing_per_quadrant = [0, 0, 0, 0]

    for r in range(n):
        for c in range(n):
            vals = [int(q[r, c]) for q in aligned]
            visible = [v for v in vals if v != int(bg)]
            if not visible:
                continue

            color = visible[0]
            assert all(v == color for v in visible), "rotated intact copies disagree"
            hidden_count = vals.count(int(bg))
            assert hidden_count in (0, 1), "missing residues must be complementary"
            if hidden_count == 1:
                out[r, c] = color
                missing_per_quadrant[vals.index(int(bg))] += 1

    assert all(n_missing > 0 for n_missing in missing_per_quadrant), (
        "each quadrant must contribute at least one missing cell"
    )
    assert np.any(out != int(bg)), "expected visible missing content"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct only the missing source-coordinate residues from four damaged rotated copies.

    Input structure:
    - A dominant background surrounds one even-sided square motif.
    - The motif is split into four equal quadrants: source orientation at upper-left,
      clockwise rotation at upper-right, 180-degree rotation at lower-right, and
      counterclockwise rotation at lower-left.
    - Each quadrant has some colored source cells erased to the background.

    Rule:
    - Crop the square motif and inverse-rotate each quadrant back into the source
      coordinate system.
    - For each source coordinate, the intact non-background copies must agree on
      one color; if exactly one quadrant is background there, that quadrant's hole
      is restored with the agreed color.
    - Return a source-tile-sized grid containing only those restored hole cells;
      all other coordinates are background.

    Key insight:
    - The answer is not the repaired full tile: the four complementary residue
      masks are first unrotated, then unioned into the hidden missing-only image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    box = _bbox(g != bg)
    assert box is not None, "expected a foreground rotational motif"
    r0, c0, r1, c1 = box
    motif = g[r0 : r1 + 1, c0 : c1 + 1]

    height, width = motif.shape
    assert height == width and height % 2 == 0, "motif must be an even-sided square"
    aligned = _source_aligned_quadrants(motif)
    return _missing_only_from_aligned(aligned, bg)


# --------------- generator ---------------

TILE_SIZE_RANGE = (4, 9)
MAX_ATTEMPTS = 500
SOURCE_ATTEMPTS = 120
BACKGROUND_COLORS = tuple(range(10))


def _neighbors4(r: int, c: int, n: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < n:
            out.append((rr, cc))
    return out


def _grow_hidden_shape(n: int, count: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    allowed = [(r, c) for r in range(n) for c in range(n) if (r, c) not in blocked]
    seed = random.choice(allowed)
    cells = {seed}
    while len(cells) < count:
        frontier = sorted(
            {
                nb
                for r, c in cells
                for nb in _neighbors4(r, c, n)
                if nb not in blocked and nb not in cells
            }
        )
        if not frontier:
            return []
        cells.add(random.choice(frontier))
    return list(cells)


def _paint_random_colors(
    source: np.ndarray,
    cells: list[tuple[int, int]],
    palette: list[int],
    *,
    force_two_colors: bool = False,
) -> None:
    colors = []
    if force_two_colors and len(cells) >= 2 and len(palette) >= 2:
        colors.extend(random.sample(palette, 2))
    while len(colors) < len(cells):
        colors.append(random.choice(palette))
    random.shuffle(colors)
    for (r, c), color in zip(cells, colors, strict=False):
        source[r, c] = int(color)


def _make_source_tile(
    n: int, bg: int
) -> tuple[np.ndarray, list[tuple[int, int]], list[np.ndarray]] | None:
    corners = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}
    usable = n * n - len(corners)
    min_hidden = 5
    max_hidden = min(14, usable - 2, usable // 2 + 2)
    if max_hidden < min_hidden:
        return None

    palette_size = random.randint(3, min(5, 9))
    palette = random.sample([c for c in range(10) if c != int(bg)], palette_size)

    for _ in range(SOURCE_ATTEMPTS):
        hidden_count = random.randint(min_hidden, max_hidden)
        hidden = _grow_hidden_shape(n, hidden_count, corners)
        if len(hidden) != hidden_count:
            continue

        source = np.full((n, n), int(bg), dtype=int)
        _paint_random_colors(source, sorted(corners), palette)
        _paint_random_colors(source, hidden, palette, force_two_colors=True)

        blocked = set(hidden) | corners
        extra_pool = [(r, c) for r in range(n) for c in range(n) if (r, c) not in blocked]
        min_extra = min(len(extra_pool), max(4, n))
        max_extra = min(len(extra_pool), max(min_extra, n * n // 3))
        if min_extra <= 0 or max_extra < min_extra:
            continue
        extras = random.sample(extra_pool, random.randint(min_extra, max_extra))
        _paint_random_colors(source, extras, palette)

        if any(np.array_equal(source, np.rot90(source, k)) for k in (1, 2, 3)):
            continue

        owners = list(range(4))
        owners.extend(random.randrange(4) for _ in range(len(hidden) - 4))
        random.shuffle(owners)
        masks = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        for (r, c), owner in zip(hidden, owners, strict=False):
            masks[owner][r, c] = True

        return source, hidden, masks
    return None


def _assemble_motif(source: np.ndarray, masks: list[np.ndarray], bg: int) -> np.ndarray:
    n = int(source.shape[0])
    motif = np.full((2 * n, 2 * n), int(bg), dtype=int)
    placements = (
        (0, 0, 0),
        (0, n, -1),
        (n, n, 2),
        (n, 0, 1),
    )
    for idx, (r0, c0, rotation) in enumerate(placements):
        quad = np.rot90(source, rotation).copy()
        missing = np.rot90(masks[idx], rotation)
        quad[missing] = int(bg)
        motif[r0 : r0 + n, c0 : c0 + n] = quad
    return motif


def _expected_output(
    source: np.ndarray, hidden: list[tuple[int, int]], bg: int
) -> np.ndarray:
    out = np.full(source.shape, int(bg), dtype=int)
    for r, c in hidden:
        out[r, c] = int(source[r, c])
    return out


def _place_motif(motif: np.ndarray, bg: int) -> np.ndarray:
    side = int(motif.shape[0])
    height = random.randint(side + 2, 30)
    width = random.randint(side + 2, 30)
    r0 = random.randint(1, height - side - 1)
    c0 = random.randint(1, width - side - 1)
    grid = np.full((height, width), int(bg), dtype=int)
    grid[r0 : r0 + side, c0 : c0 + side] = motif
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BACKGROUND_COLORS)
    n = random.randint(*TILE_SIZE_RANGE)
    made = _make_source_tile(n, bg)
    if made is None:
        return None
    source, hidden, masks = made
    motif = _assemble_motif(source, masks, bg)
    grid = _place_motif(motif, bg)
    expected = _expected_output(source, hidden, bg)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(expected, source):
        return None
    if len(set(map(int, expected[expected != int(bg)]))) < 2:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-hole reconstruction input")
