import random

import numpy as np


BLACK = 0
GREY = 5
YELLOW = 4
ORANGE = 7
MAROON = 9

INPUT_COLORS = (1, 2, 3, 6, 8)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 120
GRID_MIN = 10
GRID_MAX = 20

TEMPLATES = {
    "single": np.array([[1]], dtype=bool),
    "line3": np.array([[1, 1, 1]], dtype=bool),
    "line4": np.array([[1, 1, 1, 1]], dtype=bool),
    "elbow3": np.array([[1, 0], [1, 1]], dtype=bool),
    "elbow4": np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    "zigzag4": np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    "square2": np.array([[1, 1], [1, 1]], dtype=bool),
    "tee4": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    "plus5": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
}

PATH_TEMPLATES = ("line3", "line4", "elbow3", "elbow4", "zigzag4")
JUNCTION_TEMPLATES = ("tee4", "plus5")
EXTRA_TEMPLATES = ("single", "line3", "line4", "elbow3", "elbow4", "zigzag4", "square2", "tee4", "plus5")


def _parity_grid(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    rows, cols = shape
    rr, cc = np.indices((rows, cols))
    return np.where((rr + cc) % 2 == 0, int(even_color), int(odd_color)).astype(int)


def _infer_checkerboard(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    colors = [int(color) for color in np.unique(g)]
    assert len(colors) >= 2, "expected at least two colors for the checkerboard"

    best_score = -1
    best_expected = None
    for even_color in colors:
        for odd_color in colors:
            if even_color == odd_color:
                continue
            expected = _parity_grid(g.shape, even_color, odd_color)
            score = int(np.count_nonzero(g == expected))
            if score > best_score:
                best_score = score
                best_expected = expected

    assert best_expected is not None, "failed to infer checkerboard colors"
    return best_expected


def _overlay_degrees(grid: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    overlay = g != expected
    height, width = g.shape
    degrees: dict[tuple[int, int], int] = {}

    for r, c in np.argwhere(overlay):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        degree = 0
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and bool(overlay[rr, cc]) and int(g[rr, cc]) == color:
                degree += 1
        degrees[(r, c)] = int(degree)
    return overlay, degrees


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the checkerboard background and recolor each overlay pixel by its same-color degree.

    Input structure:
    - The canvas is a grey/black checkerboard whose cells may be replaced by colored pixels.
    - Every colored cluster is monochrome, and its local topology is read with 4-neighbor adjacency.
    - Colored pixels can be isolated, lie on simple paths, or sit at junctions.

    Rule:
    - Infer the underlying checkerboard and restore it as the base of the output.
    - For each non-checkerboard input pixel, count how many orthogonal neighbors share its input color.
    - Degree 0 pixels disappear into the checkerboard, degree 1 pixels become yellow,
      degree 2 pixels become orange, and degree 3 or 4 pixels become maroon.

    Key insight: Input colors only define which neighbors count as belonging to the same local structure; the output color depends only on the pixel's same-color degree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    expected = _infer_checkerboard(g)
    _overlay, degrees = _overlay_degrees(g, expected)

    out = expected.copy()
    for (r, c), degree in degrees.items():
        if degree == 1:
            out[r, c] = YELLOW
        elif degree == 2:
            out[r, c] = ORANGE
        elif degree >= 3:
            out[r, c] = MAROON
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_mask(name: str, rng: random.Random) -> np.ndarray:
    mask = TEMPLATES[name].copy()
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, rng.randrange(4))
    return _trim_mask(mask)


def _place_mask(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> bool:
    height, width = grid.shape
    mask_h, mask_w = mask.shape
    if mask_h > height - 2 or mask_w > width - 2:
        return False

    cells_local = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    top_max = height - mask_h - 1
    left_max = width - mask_w - 1
    if top_max < 1 or left_max < 1:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(1, top_max)
        left = rng.randint(1, left_max)
        cells = [(top + r, left + c) for r, c in cells_local]
        if any((r, c) in forbidden for r, c in cells):
            continue

        for r, c in cells:
            grid[r, c] = int(color)
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        forbidden.add((rr, cc))
        return True
    return False


def _role_signature(grid: np.ndarray) -> set[int]:
    expected = _infer_checkerboard(grid)
    _overlay, degrees = _overlay_degrees(grid, expected)
    roles = set()
    for degree in degrees.values():
        if degree == 0:
            roles.add(0)
        elif degree == 1:
            roles.add(1)
        elif degree == 2:
            roles.add(2)
        else:
            roles.add(3)
    return roles


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)

    if rng.random() < 0.5:
        grid = _parity_grid((height, width), BLACK, GREY)
    else:
        grid = _parity_grid((height, width), GREY, BLACK)

    mandatory = [
        "single",
        rng.choice(PATH_TEMPLATES),
        rng.choice(JUNCTION_TEMPLATES),
    ]
    extra_count = rng.randint(1, 2)
    shapes = mandatory + [rng.choice(EXTRA_TEMPLATES) for _ in range(extra_count)]
    rng.shuffle(shapes)

    if len(shapes) > len(INPUT_COLORS):
        return None

    colors = list(INPUT_COLORS)
    rng.shuffle(colors)
    forbidden: set[tuple[int, int]] = set()

    for shape_name, color in zip(shapes, colors):
        mask = _random_mask(shape_name, rng)
        if not _place_mask(grid, mask, color, forbidden, rng):
            return None

    roles = _role_signature(grid)
    if roles != {0, 1, 2, 3}:
        return None

    out = main(grid)
    if not {YELLOW, ORANGE, MAROON}.issubset(set(map(int, np.unique(out)))):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
