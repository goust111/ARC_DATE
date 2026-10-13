from __future__ import annotations

import numpy as np


BG = 0
CROSS = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8)
SPRITE_COLOR_CANDIDATES = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 600
SHAPE_ATTEMPTS = 240


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    mins = coords.min(axis=0)
    return (coords - mins).astype(int)


def _seed_from_centroid(coords: np.ndarray) -> tuple[tuple[int, int], bool]:
    centroid = coords.mean(axis=0)
    dist2 = np.sum((coords - centroid) ** 2, axis=1)
    best = float(dist2.min())
    tied = np.flatnonzero(np.abs(dist2 - best) <= 1e-9)
    tied_coords = coords[tied]
    order = np.lexsort((tied_coords[:, 1], tied_coords[:, 0]))
    seed = tuple(map(int, tied_coords[order[0]]))
    return seed, len(tied) == 1


def _chebyshev(seed: tuple[int, int], coord: tuple[int, int]) -> int:
    return max(abs(int(coord[0]) - int(seed[0])), abs(int(coord[1]) - int(seed[1])))


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    full_rows = [r for r in range(h) if np.all(grid[r, :] != BG)]
    full_cols = [c for c in range(w) if np.all(grid[:, c] != BG)]
    assert len(full_rows) == 1 and len(full_cols) == 1, "expected one separator cross"
    cr, cc = int(full_rows[0]), int(full_cols[0])
    assert 0 < cr < h - 1 and 0 < cc < w - 1
    assert int(grid[cr, cc]) == CROSS
    return cr, cc


def _read_arm(grid: np.ndarray, cr: int, cc: int, direction: str) -> list[int]:
    if direction == "up":
        coords = [(r, cc) for r in range(cr - 1, -1, -1)]
    elif direction == "down":
        coords = [(r, cc) for r in range(cr + 1, grid.shape[0])]
    elif direction == "left":
        coords = [(cr, c) for c in range(cc - 1, -1, -1)]
    elif direction == "right":
        coords = [(cr, c) for c in range(cc + 1, grid.shape[1])]
    else:
        raise ValueError(direction)
    palette = [int(grid[r, c]) for r, c in coords if int(grid[r, c]) != CROSS]
    assert 2 <= len(palette) <= 3
    assert len(set(palette)) == len(palette)
    return palette


def _extract_single_sprite(region: np.ndarray) -> tuple[np.ndarray, int]:
    colors = sorted(set(map(int, region[region != BG].tolist())))
    assert len(colors) == 1, "each quadrant must contain exactly one sprite color"
    color = int(colors[0])
    comps = _component_coords(region == color)
    assert len(comps) == 1, "each quadrant must contain one 4-connected sprite"
    return comps[0], color


def _recolor_component(
    out: np.ndarray,
    coords: np.ndarray,
    color: int,
    palette: list[int],
    offset: tuple[int, int],
) -> None:
    seed_local, _ = _seed_from_centroid(coords)
    max_d = 0
    for rr, cc in coords:
        d = _chebyshev(seed_local, (int(rr), int(cc)))
        max_d = max(max_d, d)
    assert max_d <= len(palette), "sprite exceeds available palette depth"

    or0, oc0 = offset
    for rr, cc in coords:
        d = _chebyshev(seed_local, (int(rr), int(cc)))
        out[or0 + int(rr), oc0 + int(cc)] = color if d == 0 else int(palette[d - 1])


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor quadrant sprites by Chebyshev layers using the separator cross arms as rotated palettes.

    Input structure:
    - A black background contains one full non-black row and one full non-black
      column that intersect as a grey cross separator.
    - The cross divides the grid into four quadrants, each containing exactly one
      single-color 4-connected sprite away from the separator.
    - Along each arm of the cross, the non-grey cells encountered from the center
      outward form a short color sequence of length 2 or 3.

    Rule:
    - Find the unique separator row and column and read four arm palettes by
      collecting non-grey colors from the center outward.
    - Apply the rotated palette mapping UL->left, UR->up, LL->down, LR->right.
    - For each quadrant sprite, choose the sprite pixel nearest its geometric
      centroid as the seed, keep that seed in the original sprite color, and
      recolor every other sprite pixel by Chebyshev distance layer using the
      corresponding palette entry.
    - Remove the separator cross and all background, so only the recolored sprites
      remain on black.

    Key insight:
    - The separator cross has a dual role: it partitions the scene into quadrants
      and simultaneously stores the distance palettes that recolor those quadrant
      sprites after a one-step rotational remapping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cr, cc = _find_cross(g)
    palettes = {
        "up": _read_arm(g, cr, cc, "up"),
        "down": _read_arm(g, cr, cc, "down"),
        "left": _read_arm(g, cr, cc, "left"),
        "right": _read_arm(g, cr, cc, "right"),
    }

    out = np.zeros_like(g)
    quadrants = [
        (g[:cr, :cc], palettes["left"], (0, 0)),
        (g[:cr, cc + 1 :], palettes["up"], (0, cc + 1)),
        (g[cr + 1 :, :cc], palettes["down"], (cr + 1, 0)),
        (g[cr + 1 :, cc + 1 :], palettes["right"], (cr + 1, cc + 1)),
    ]

    for region, palette, offset in quadrants:
        coords, color = _extract_single_sprite(region)
        seed_local, unique = _seed_from_centroid(coords)
        assert unique, "centroid-nearest sprite pixel must be unique"
        max_d = max(_chebyshev(seed_local, tuple(map(int, rc))) for rc in coords)
        assert max_d <= len(palette)
        _recolor_component(out, coords, color, palette, offset)
    return out


def _random_connected_shape(rng: np.random.Generator, max_depth: int) -> tuple[np.ndarray, int] | None:
    side = max_depth + 2
    for _ in range(SHAPE_ATTEMPTS):
        target_cells = int(rng.integers(3, 9))
        start = (int(rng.integers(0, side)), int(rng.integers(0, side)))
        cells = {start}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < side and 0 <= nc < side
        }
        while len(cells) < target_cells and frontier:
            front = sorted(frontier)
            weights = []
            center = np.array([side / 2.0, side / 2.0])
            for rr, cc in front:
                dist = float(np.sum((np.array([rr, cc]) - center) ** 2))
                weights.append(1.0 / (1.0 + dist))
            probs = np.array(weights, dtype=float)
            probs /= probs.sum()
            pick = front[int(rng.choice(len(front), p=probs))]
            frontier.remove(pick)
            cells.add(pick)
            for nr, nc in _neighbors4(*pick):
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        coords = np.array(sorted(cells), dtype=int)
        norm = _normalize_coords(coords)
        seed, unique = _seed_from_centroid(norm)
        if not unique:
            continue
        max_d = max(_chebyshev(seed, tuple(map(int, rc))) for rc in norm)
        if 1 <= max_d <= max_depth:
            return norm, max_d
    return None


def _place_shape(
    grid: np.ndarray,
    quadrant_origin: tuple[int, int],
    quadrant_shape: tuple[int, int],
    coords: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> None:
    qh, qw = quadrant_shape
    sh = int(coords[:, 0].max()) + 1
    sw = int(coords[:, 1].max()) + 1
    assert qh >= sh + 2 and qw >= sw + 2
    rr = int(rng.integers(1, qh - sh))
    cc = int(rng.integers(1, qw - sw))
    r0, c0 = quadrant_origin
    for dr, dc in coords:
        grid[r0 + rr + int(dr), c0 + cc + int(dc)] = color


def _make_candidate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(15, 21))
    w = int(rng.integers(15, 21))
    cr = int(rng.integers(7, h - 7))
    cc = int(rng.integers(7, w - 7))

    palette_pool = list(rng.choice(PALETTE_COLORS, size=int(rng.integers(3, 5)), replace=False))
    sprite_colors = [
        int(c)
        for c in rng.choice(
            [c for c in SPRITE_COLOR_CANDIDATES if c not in palette_pool],
            size=4,
            replace=False,
        )
    ]

    arm_specs = {}
    for name in ("up", "down", "left", "right"):
        length = int(rng.integers(2, 4))
        arm_specs[name] = [int(c) for c in rng.choice(palette_pool, size=length, replace=False)]

    grid = np.zeros((h, w), dtype=int)
    grid[cr, :] = CROSS
    grid[:, cc] = CROSS
    for idx, color in enumerate(arm_specs["up"], start=1):
        grid[cr - idx, cc] = color
    for idx, color in enumerate(arm_specs["down"], start=1):
        grid[cr + idx, cc] = color
    for idx, color in enumerate(arm_specs["left"], start=1):
        grid[cr, cc - idx] = color
    for idx, color in enumerate(arm_specs["right"], start=1):
        grid[cr, cc + idx] = color

    quadrants = [
        ((0, 0), (cr, cc), arm_specs["left"], sprite_colors[0]),
        ((0, cc + 1), (cr, w - cc - 1), arm_specs["up"], sprite_colors[1]),
        ((cr + 1, 0), (h - cr - 1, cc), arm_specs["down"], sprite_colors[2]),
        ((cr + 1, cc + 1), (h - cr - 1, w - cc - 1), arm_specs["right"], sprite_colors[3]),
    ]

    max_depths: list[int] = []
    for origin, qshape, palette, color in quadrants:
        shape = _random_connected_shape(rng, len(palette))
        if shape is None:
            return None
        coords, max_d = shape
        sh = int(coords[:, 0].max()) + 1
        sw = int(coords[:, 1].max()) + 1
        if qshape[0] < sh + 2 or qshape[1] < sw + 2:
            return None
        _place_shape(grid, origin, qshape, coords, color, rng)
        max_depths.append(max_d)

    if max(max_depths) < 2:
        return None

    full_rows = [r for r in range(h) if np.all(grid[r, :] != BG)]
    full_cols = [c for c in range(w) if np.all(grid[:, c] != BG)]
    if full_rows != [cr] or full_cols != [cc]:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    sprite_pixels = int(np.count_nonzero(np.delete(np.delete(grid, cr, axis=0), cc, axis=1)))
    if int(np.count_nonzero(out)) != sprite_pixels:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _make_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid separator-palette cross example")
