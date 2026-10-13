import random
from collections import Counter

import numpy as np


BG = 0
DIVIDER = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
TRIOMINO_CELLS = {
    (0, 0): ((0, 1), (1, 0), (1, 1)),
    (0, 1): ((0, 0), (1, 0), (1, 1)),
    (1, 0): ((0, 0), (0, 1), (1, 1)),
    (1, 1): ((0, 0), (0, 1), (1, 0)),
}

SIZE_OPTIONS = (11, 13, 15, 17, 19, 21, 23, 25, 27, 29)
MAX_GROUPS = 6
MAX_GENERATION_ATTEMPTS = 400
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each orientation-outlier L-triomino with the color of its left-right mirror mate.

    Input structure
    - The background is black.
    - One full maroon row and one full maroon column divide the grid into four quadrants.
    - Each quadrant contains monochrome L-triominoes occupying 2x2 boxes with one missing corner.
    - Every triomino has mirror-position partners in the other three quadrants.

    Rule
    - Group the four triominoes that share the same mirrored 2x2 anchor across the divider cross.
    - In each group, compare the missing-corner orientations and find the unique orientation outlier.
    - Recolor only that outlier triomino with the color of the triomino reflected across the vertical divider from it.
    - Keep every triomino's shape, the divider lines, and all non-outlier triominoes unchanged.

    Key insight
    - The edited object is determined by orientation, not by color; the replacement color comes from the outlier's left-right mirror counterpart.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    divider_color, mid_r, mid_c = _find_dividers(g)
    objects = _extract_triominoes(g, divider_color, mid_r, mid_c)
    assert objects and len(objects) % 4 == 0, "expected mirrored triomino quartets"

    groups = {}
    for obj in objects:
        r0, c0, _, _ = obj["bbox"]
        key = _canonical_anchor(int(r0), int(c0), mid_r, mid_c)
        groups.setdefault(key, {})[obj["quad"]] = obj

    out = g.copy()
    for quartet in groups.values():
        assert set(quartet) == set(QUADRANTS), "each anchor must appear in all four quadrants"
        orientations = Counter(obj["orientation"] for obj in quartet.values())
        assert sorted(orientations.values()) == [1, 3], "each quartet must contain exactly one orientation outlier"

        odd_orientation = next(orientation for orientation, count in orientations.items() if count == 1)
        odd_obj = next(obj for obj in quartet.values() if obj["orientation"] == odd_orientation)
        qr, qc = odd_obj["quad"]
        mate = quartet[(qr, 1 - qc)]
        new_color = int(mate["color"])

        coords = odd_obj["coords"]
        out[coords[:, 0], coords[:, 1]] = new_color

    return out


def _find_dividers(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        r
        for r in range(h)
        if int(grid[r, 0]) != BG and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        c
        for c in range(w)
        if int(grid[0, c]) != BG and np.all(grid[:, c] == grid[0, c])
    ]

    assert len(row_candidates) == 1, "expected exactly one full divider row"
    assert len(col_candidates) == 1, "expected exactly one full divider column"

    mid_r = int(row_candidates[0])
    mid_c = int(col_candidates[0])
    divider_color = int(grid[mid_r, 0])
    assert int(grid[0, mid_c]) == divider_color, "divider row and column must share one color"
    return divider_color, mid_r, mid_c


def _extract_triominoes(
    grid: np.ndarray, divider_color: int, mid_r: int, mid_c: int
) -> list[dict[str, object]]:
    mask = (grid != BG) & (grid != divider_color)
    objects = []
    for coords in _components(mask):
        colors = {int(grid[r, c]) for r, c in coords.tolist()}
        assert len(colors) == 1, "each object must be monochrome"
        color = next(iter(colors))

        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        assert (r1 - r0, c1 - c0) == (1, 1), "each object must fill a 2x2 box"
        assert len(coords) == 3, "each object must contain exactly three cells"
        assert r1 < mid_r or r0 > mid_r, "objects must stay inside one horizontal half"
        assert c1 < mid_c or c0 > mid_c, "objects must stay inside one vertical half"

        quad = (0 if r0 < mid_r else 1, 0 if c0 < mid_c else 1)
        orientation = _missing_corner(coords, r0, c0)
        objects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "orientation": orientation,
                "quad": quad,
            }
        )

    return objects


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            groups.append(np.array(coords, dtype=int))

    return groups


def _missing_corner(coords: np.ndarray, r0: int, c0: int) -> tuple[int, int]:
    present = {(int(r - r0), int(c - c0)) for r, c in coords.tolist()}
    missing = list(set(MISSING_CORNERS) - present)
    assert len(missing) == 1, "expected a single missing corner"
    return missing[0]


def _canonical_anchor(r0: int, c0: int, mid_r: int, mid_c: int) -> tuple[int, int]:
    base_r = r0 if r0 < mid_r else 2 * mid_r - r0 - 1
    base_c = c0 if c0 < mid_c else 2 * mid_c - c0 - 1
    return base_r, base_c


def _paint_triomino(
    grid: np.ndarray, r0: int, c0: int, missing_corner: tuple[int, int], color: int
) -> None:
    for dr, dc in TRIOMINO_CELLS[missing_corner]:
        grid[r0 + dr, c0 + dc] = color


def _mirrored_anchors(r0: int, c0: int, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int]]:
    mirror_r = 2 * mid_r - r0 - 1
    mirror_c = 2 * mid_c - c0 - 1
    return {
        (0, 0): (r0, c0),
        (0, 1): (r0, mirror_c),
        (1, 0): (mirror_r, c0),
        (1, 1): (mirror_r, mirror_c),
    }


def _anchor_limit(half_h: int, half_w: int) -> int:
    pack_h = max(1, (half_h + 1) // 3)
    pack_w = max(1, (half_w + 1) // 3)
    return min(MAX_GROUPS, pack_h * pack_w)


def _sample_anchors(rng: random.Random, half_h: int, half_w: int) -> list[tuple[int, int]] | None:
    max_groups = _anchor_limit(half_h, half_w)
    target = rng.randint(1, max_groups)
    candidates = [(r, c) for r in range(half_h - 1) for c in range(half_w - 1)]
    rng.shuffle(candidates)

    anchors = []
    for r, c in candidates:
        if all(not (abs(r - rr) <= 2 and abs(c - cc) <= 2) for rr, cc in anchors):
            anchors.append((r, c))
            if len(anchors) == target:
                return anchors
    return None


def _sample_colors(rng: random.Random, odd_quad: tuple[int, int]) -> dict[tuple[int, int], int]:
    colors = {quad: rng.choice(PALETTE) for quad in QUADRANTS}
    mate_quad = (odd_quad[0], 1 - odd_quad[1])
    if colors[odd_quad] == colors[mate_quad]:
        colors[odd_quad] = rng.choice([color for color in PALETTE if color != colors[mate_quad]])
    return colors


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.choice(SIZE_OPTIONS)
    w = rng.choice(SIZE_OPTIONS)
    mid_r = h // 2
    mid_c = w // 2

    anchors = _sample_anchors(rng, mid_r, mid_c)
    if anchors is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[mid_r, :] = DIVIDER
    grid[:, mid_c] = DIVIDER

    for r0, c0 in anchors:
        common_orientation = rng.choice(MISSING_CORNERS)
        odd_orientation = rng.choice([corner for corner in MISSING_CORNERS if corner != common_orientation])
        odd_quad = rng.choice(QUADRANTS)
        colors = _sample_colors(rng, odd_quad)

        for quad, (rr, cc) in _mirrored_anchors(r0, c0, mid_r, mid_c).items():
            orientation = odd_orientation if quad == odd_quad else common_orientation
            _paint_triomino(grid, rr, cc, orientation, colors[quad])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-triomino input")
