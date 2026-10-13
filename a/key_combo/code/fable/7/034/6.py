"""ARC task 6: audit a color-keyed legend and crop the liar's object."""

from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    h, w = mask.shape
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        todo = [(sr, sc)]
        points: list[tuple[int, int]] = []
        while todo:
            r, c = todo.pop()
            points.append((r, c))
            for dr, dc in N4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(np.asarray(points, dtype=int))
    return result


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _read_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    """Locate the unique rectangular frame and decode (key color, claim) rows."""
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    for frame_color in set(map(int, grid.ravel())) - {BG}:
        for comp in _components(grid == frame_color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 6:
                continue
            border = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                      if r in (r0, r1) or c in (c0, c1)]
            if any(int(grid[r, c]) != frame_color for r, c in border):
                continue

            rows: list[tuple[int, int]] = []
            valid = True
            keys: set[int] = set()
            for r in range(r0 + 1, r1):
                key = int(grid[r, c0 + 1])
                if key in (BG, frame_color) or key in keys:
                    valid = False
                    break
                keys.add(key)
                # All even-offset separator cells, including the cell before the
                # right border, are visibly blank.
                if any(int(grid[r, c]) != BG for c in range(c0 + 2, c1, 2)):
                    valid = False
                    break
                claims = [int(grid[r, c]) for c in range(c0 + 3, c1, 2)]
                claim = 0
                while claim < len(claims) and claims[claim] == key:
                    claim += 1
                if claim == 0 or any(v != BG for v in claims[claim:]):
                    valid = False
                    break
                rows.append((key, claim))
            if valid and len(rows) >= 3:
                candidates.append(((r0, c0, r1, c1), rows))
    assert len(candidates) == 1, "the color-keyed rectangular legend must be unique"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the sole outside object whose color's legend claim is false.

    Input structure:
    A black grid contains one rectangular legend frame.  In each interior row,
    the left pixel is a distinct key color and same-colored dots at every other
    column state an object count.  Outside the frame are separated monochrome
    4-connected objects in those key colors.

    Rule:
    Remove the entire legend rectangle from consideration, count the outside
    4-connected objects of every key color, and compare each count with its row's
    dots.  Exactly one row disagrees and that color has exactly one object; return
    the subgrid inside that object's inclusive bounding box.

    Key insight:
    Having one object is deliberately not enough: truthful colors may also have
    one.  The unique claim-versus-count mismatch identifies which object to crop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and int(g.min()) >= 0
    legend_box, rows = _read_legend(g)
    r0, c0, r1, c1 = legend_box
    scene = g.copy()
    scene[r0:r1 + 1, c0:c1 + 1] = BG

    mismatches: list[tuple[int, list[np.ndarray]]] = []
    for color, claimed in rows:
        objects = _components(scene == color)
        if len(objects) != claimed:
            mismatches.append((color, objects))
    assert len(mismatches) == 1, "exactly one legend row must lie"
    _color, objects = mismatches[0]
    assert len(objects) == 1, "the lying color must have exactly one outside object"
    br0, bc0, br1, bc1 = _bbox(objects[0])
    return g[br0:br1 + 1, bc0:bc1 + 1].copy()


def _random_shape(rng: random.Random, target: bool = False) -> np.ndarray:
    """Make a small connected polyomino; target crops always expose background."""
    for _ in range(100):
        size = rng.randint(4, 8) if target else rng.randint(1, 6)
        cells = {(0, 0)}
        while len(cells) < size:
            r, c = rng.choice(tuple(cells))
            dr, dc = rng.choice(N4)
            cells.add((r + dr, c + dc))
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = {(r - min_r, c - min_c) for r, c in cells}
        hh = max(r for r, _ in norm) + 1
        ww = max(c for _, c in norm) + 1
        if hh > 4 or ww > 4:
            continue
        if target and (hh < 2 or ww < 2 or len(norm) == hh * ww):
            continue
        arr = np.zeros((hh, ww), dtype=bool)
        for r, c in norm:
            arr[r, c] = True
        return arr
    raise RuntimeError("could not form a suitable polyomino")


def _place_shape(grid: np.ndarray, blocked: np.ndarray, shape: np.ndarray,
                 color: int, rng: random.Random) -> np.ndarray | None:
    h, w = grid.shape
    sh, sw = shape.shape
    positions = [(r, c) for r in range(1, h - sh) for c in range(1, w - sw)]
    rng.shuffle(positions)
    for r0, c0 in positions:
        coords = np.argwhere(shape) + np.array([r0, c0])
        if any(blocked[int(r), int(c)] for r, c in coords):
            continue
        for r, c in coords:
            grid[int(r), int(c)] = color
        # One-cell Chebyshev moat prevents same- or different-color objects from touching.
        for r, c in coords:
            rr0, rr1 = max(0, int(r) - 1), min(h, int(r) + 2)
            cc0, cc1 = max(0, int(c) - 1), min(w, int(c) + 2)
            blocked[rr0:rr1, cc0:cc1] = True
        return coords
    return None


def _generate_once(rng: random.Random) -> np.ndarray | None:
    h, w = rng.randint(21, 30), rng.randint(21, 30)
    row_count = rng.randint(3, 5)
    colors = rng.sample(PALETTE, row_count)
    target_index = rng.randrange(row_count)
    lure_index = rng.choice([i for i in range(row_count) if i != target_index])

    actual = [rng.randint(2, 3) for _ in colors]
    actual[target_index] = 1
    actual[lure_index] = 1                 # a necessary anti-shortcut distractor
    claims = actual.copy()
    claims[target_index] = rng.choice((2, 3))
    max_claim = max(claims)
    fh, fw = row_count + 2, 2 * max_claim + 4

    # All four broad scene regions occur: the legend is not tied to top-left.
    anchors = (
        (rng.randint(1, 3), rng.randint(1, w - fw - 1)),
        (h - fh - rng.randint(1, 3), rng.randint(1, w - fw - 1)),
        (rng.randint(1, h - fh - 1), rng.randint(1, 3)),
        (rng.randint(1, h - fh - 1), w - fw - rng.randint(1, 3)),
    )
    lr, lc = rng.choice(anchors)
    grid = np.zeros((h, w), dtype=int)
    grid[lr, lc:lc + fw] = FRAME
    grid[lr + fh - 1, lc:lc + fw] = FRAME
    grid[lr:lr + fh, lc] = FRAME
    grid[lr:lr + fh, lc + fw - 1] = FRAME
    for i, (color, claim) in enumerate(zip(colors, claims)):
        r = lr + 1 + i
        grid[r, lc + 1] = color
        for k in range(claim):
            grid[r, lc + 3 + 2 * k] = color

    blocked = np.zeros((h, w), dtype=bool)
    blocked[max(0, lr - 1):min(h, lr + fh + 1),
            max(0, lc - 1):min(w, lc + fw + 1)] = True
    target_coords: np.ndarray | None = None
    for i, (color, count) in enumerate(zip(colors, actual)):
        for j in range(count):
            shape = _random_shape(rng, target=(i == target_index and j == 0))
            coords = _place_shape(grid, blocked, shape, color, rng)
            if coords is None:
                return None
            if i == target_index:
                target_coords = coords

    if target_coords is None:
        return None
    # Independent semantic guards: exactly one mismatch, at least two singleton
    # colors, and the answer is a genuine bbox crop rather than a solid swatch.
    counts = {color: len(_components((grid == color) & ~_box_mask(grid.shape, lr, lc, fh, fw)))
              for color in colors}
    if sum(counts[c] != claim for c, claim in zip(colors, claims)) != 1:
        return None
    if sum(counts[c] == 1 for c in colors) < 2 or counts[colors[target_index]] != 1:
        return None
    tr0, tc0, tr1, tc1 = _bbox(target_coords)
    expected = grid[tr0:tr1 + 1, tc0:tc1 + 1]
    if not np.any(expected == BG) or not np.array_equal(main(grid), expected):
        return None
    return grid


def _box_mask(shape: tuple[int, int], r: int, c: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[r:r + h, c:c + w] = True
    return mask


def generate_input() -> np.ndarray:
    """Generate a varied, fully audited instance of the legend-liar task."""
    rng = random.Random()
    for _ in range(300):
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-liar instance")
