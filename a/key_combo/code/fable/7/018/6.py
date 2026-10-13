from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (n, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    height, width = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        cells = []
        while stack:
            rr, cc = stack.pop()
            cells.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _normal_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _candidate_solutions(grid: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int]]]:
    """Enumerate interpretations satisfying every geometric promise of the puzzle."""
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    rotated_grid = np.rot90(g, 2)
    found: list[tuple[np.ndarray, tuple[int, int]]] = []

    for color_a, color_b in combinations(colors, 2):
        sticker_data = []
        valid = True
        for color in (color_a, color_b):
            comps = _components(g == color)
            if len(comps) != 1 or len(comps[0]) < 3:
                valid = False
                break
            coords = comps[0]
            shape = _normal_mask(coords)
            # Rotation must be a real, observable operation rather than a no-op.
            if np.array_equal(shape, np.rot90(shape, 2)):
                valid = False
                break
            mr = g.shape[0] - 1 - coords[:, 0]
            mc = g.shape[1] - 1 - coords[:, 1]
            mirror_values = g[mr, mc]
            if np.any(mirror_values == bg) or np.any(np.isin(mirror_values, (color_a, color_b))):
                valid = False
                break
            sticker_data.append((color, coords, shape))
        if not valid:
            continue

        # Delete both foreign-colored groups and reveal the values supplied by
        # their point-symmetric counterparts.
        cleared = g.copy()
        for _, coords, _ in sticker_data:
            cleared[coords[:, 0], coords[:, 1]] = rotated_grid[coords[:, 0], coords[:, 1]]

        hole_mask = (cleared == bg) & (np.rot90(cleared, 2) != bg)
        holes = _components(hole_mask)
        if len(holes) != 2 or any(len(hole) < 3 for hole in holes):
            continue
        hole_shapes = [_normal_mask(hole) for hole in holes]

        matches: list[list[int]] = []
        for _, _, sticker_shape in sticker_data:
            turned = np.rot90(sticker_shape, 2)
            matches.append([i for i, hole_shape in enumerate(hole_shapes) if np.array_equal(turned, hole_shape)])
        if any(len(options) != 1 for options in matches) or matches[0][0] == matches[1][0]:
            continue
        # Reject any interpretation in which direct translation would also work.
        if any(np.array_equal(shape, hole_shapes[matches[i][0]]) for i, (_, _, shape) in enumerate(sticker_data)):
            continue

        # Filling the holes from their symmetric counterparts must recover a
        # genuinely point-symmetric underlying canvas.
        repaired_canvas = cleared.copy()
        for hole in holes:
            rr, cc = hole[:, 0], hole[:, 1]
            repaired_canvas[rr, cc] = cleared[g.shape[0] - 1 - rr, g.shape[1] - 1 - cc]
        if not np.array_equal(repaired_canvas, np.rot90(repaired_canvas, 2)):
            continue

        out = cleared.copy()
        for i, (color, _, _) in enumerate(sticker_data):
            hole = holes[matches[i][0]]
            out[hole[:, 0], hole[:, 1]] = color
        found.append((out, tuple(sorted((color_a, color_b)))))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore two rotated, misplaced stickers to their uniquely matching holes on a point-symmetric canvas.

    Input structure:
    - The same-size grid contains a multicolor canvas whose intact layer is symmetric under a 180-degree
      rotation about the grid center, two background-colored holes, and two connected foreign-color stickers.
    - The stickers have different colors and different asymmetric shapes; each is displayed after a half-turn.
    - Sticker sites, holes, and their point-symmetric evidence sites are mutually disjoint.

    Rule:
    - For each sticker, replace its misplaced cells by the colors at their 180-degree opposite positions.
    - Extract the two holes as the remaining background/non-background point-symmetry discrepancies.
    - Rotate each sticker mask by 180 degrees, match it to its unique hole, and paint that hole in its color.
    - Leave every other cell unchanged.

    Key insight:
    - Point symmetry is both an oracle for the hidden colors under a sticker and the clue that the visible
      sticker mask, rather than an untranslated copy of it, must be half-turned to fit its old hole.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    solutions = _candidate_solutions(g)
    assert len(solutions) == 1, "expected one uniquely determined pair of misplaced stickers"
    return solutions[0][0]


# ------------------------------ generator ------------------------------

_BG = 0
_BASE_PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)
_MAX_ATTEMPTS = 500


def _random_polyomino() -> np.ndarray:
    """Make a compact connected, tight-bbox, half-turn-asymmetric mask."""
    for _ in range(100):
        target = random.randint(4, 7)
        cells = {(0, 0)}
        while len(cells) < target:
            r, c = random.choice(tuple(cells))
            options = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            nr, nc = random.choice(options)
            trial = cells | {(nr, nc)}
            rs = [x for x, _ in trial]
            cs = [y for _, y in trial]
            if max(rs) - min(rs) < 4 and max(cs) - min(cs) < 4:
                cells = trial
        arr = np.asarray(list(cells), dtype=int)
        mask = _normal_mask(arr)
        if min(mask.shape) >= 2 and not np.array_equal(mask, np.rot90(mask, 2)):
            return mask
    raise RuntimeError("could not generate an asymmetric sticker")


def _bbox_cells(origin: tuple[int, int], mask: np.ndarray) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {(r0 + r, c0 + c) for r, c in np.argwhere(mask)}


def _far_enough(a: set[tuple[int, int]], b: set[tuple[int, int]], gap: int = 2) -> bool:
    return all(max(abs(r1 - r2), abs(c1 - c2)) > gap for r1, c1 in a for r2, c2 in b)


def _choose_sites(base: np.ndarray, masks: list[np.ndarray]):
    height, width = base.shape
    selected: list[set[tuple[int, int]]] = []
    origins: list[tuple[int, int]] = []
    for mask in masks:
        options = []
        mh, mw = mask.shape
        for r0 in range(1, height - mh):
            for c0 in range(1, width - mw):
                cells = _bbox_cells((r0, c0), mask)
                if any(base[r, c] == _BG for r, c in cells):
                    continue
                mirror = {(height - 1 - r, width - 1 - c) for r, c in cells}
                if not _far_enough(cells, mirror):
                    continue
                if any(not _far_enough(cells, old) or not _far_enough(mirror, old) for old in selected):
                    continue
                options.append(((r0, c0), cells, mirror))
        if not options:
            return None
        origin, cells, mirror = random.choice(options)
        origins.append(origin)
        # Reserve both the damaged site and its symmetry-oracle site.
        selected.extend((cells, mirror))
    return origins, selected


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(17, 25)
    width = random.randint(17, 25)
    base_colors = random.sample(_BASE_PALETTE, random.randint(2, 3))
    sticker_colors = random.sample([c for c in _BASE_PALETTE if c not in base_colors], 2)

    # Paint paired cells together, making the whole underlying canvas point-symmetric.
    base = np.full((height, width), _BG, dtype=int)
    for r in range(height):
        for c in range(width):
            mr, mc = height - 1 - r, width - 1 - c
            if (r, c) > (mr, mc):
                continue
            value = _BG if random.random() < 0.24 else random.choice(base_colors)
            base[r, c] = base[mr, mc] = value
    if any(np.count_nonzero(base == color) < 20 for color in base_colors):
        return None

    shape_a = _random_polyomino()
    shape_b = _random_polyomino()
    # Different holes give a unique rotation-based pairing in both directions.
    if any(
        np.array_equal(x, y)
        for x in (shape_a, np.rot90(shape_a, 2))
        for y in (shape_b, np.rot90(shape_b, 2))
    ):
        return None

    # Home masks are unrotated. Destination masks visibly show their half-turns.
    site_masks = [shape_a, shape_b, np.rot90(shape_a, 2), np.rot90(shape_b, 2)]
    chosen = _choose_sites(base, site_masks)
    if chosen is None:
        return None
    origins, _ = chosen
    home_origins, destination_origins = origins[:2], origins[2:]

    expected = base.copy()
    inp = base.copy()
    home_sets = []
    destination_sets = []
    for color, shape, home, destination in zip(sticker_colors, (shape_a, shape_b), home_origins, destination_origins):
        home_cells = _bbox_cells(home, shape)
        visible = np.rot90(shape, 2)
        destination_cells = _bbox_cells(destination, visible)
        home_sets.append(home_cells)
        destination_sets.append(destination_cells)
        for r, c in home_cells:
            expected[r, c] = color
            inp[r, c] = _BG
        for r, c in destination_cells:
            inp[r, c] = color

    # Semantic checks: both operations matter, the recovered underlay is nontrivial,
    # rotation is indispensable, and the complete interpretation is unique.
    if any(np.array_equal(shape, np.rot90(shape, 2)) for shape in (shape_a, shape_b)):
        return None
    covered_values = [{int(base[r, c]) for r, c in cells} for cells in destination_sets]
    if any(len(values) < 2 for values in covered_values):
        return None
    if any(all(base[r, c] == color for r, c in cells) for color, cells in zip(sticker_colors, home_sets)):
        return None
    solutions = _candidate_solutions(inp)
    if len(solutions) != 1 or not np.array_equal(solutions[0][0], expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid double-sticker puzzle after {_MAX_ATTEMPTS} attempts")
