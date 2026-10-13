import random

import numpy as np


BG = 0
LABEL_COLOR_POOL = tuple(range(1, 9))
CONTENT_ANCHOR_COLOR = 9
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 80

SHAPE_TEMPLATES = (
    np.array([[1]], dtype=int),
    np.array([[1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
)


def _permute_interior(inner: np.ndarray, col_labels: list[int], row_labels: list[int]) -> np.ndarray:
    row_pos = {color: idx for idx, color in enumerate(row_labels)}
    col_pos = {color: idx for idx, color in enumerate(col_labels)}

    col_dest = [row_pos[color] for color in col_labels]
    row_dest = [col_pos[color] for color in row_labels]

    after_cols = np.empty_like(inner)
    after_cols[:, col_dest] = inner

    out = np.empty_like(inner)
    out[row_dest, :] = after_cols
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the interior picture by matching the top-edge and left-edge color labels.

    Input structure:
    - The background is black.
    - The grid is square with shape (N+1) x (N+1), where the top-left corner is black.
    - The top row from column 1 onward contains N distinct column-label colors.
    - The left column from row 1 onward contains the same N colors in a different order as row labels.
    - The N x N interior contains the actual picture and may use any colors, including black.

    Rule:
    - Read the top labels left-to-right as C1..CN and the left labels top-to-bottom as R1..RN.
    - Move interior column j to the position where its label Cj appears in the row-label sequence.
    - Then move interior row i to the position where its label Ri appears in the column-label sequence.
    - Return only the transformed N x N interior, without the label row or label column.

    Key insight:
    - The colored edge pixels are not part of the picture; together they encode two matching permutations that reorder the interior along the two axes.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] < 5:
        raise ValueError("expected a square 2D grid of size at least 5x5")

    n = g.shape[0] - 1
    col_labels = [int(v) for v in g[0, 1:]]
    row_labels = [int(v) for v in g[1:, 0]]

    if len(set(col_labels)) != n or set(col_labels) != set(row_labels):
        raise ValueError("top labels and left labels must be distinct and use the same color set")

    inner = g[1:, 1:]
    return _permute_interior(inner, col_labels, row_labels)


def _sample_row_labels(rng: random.Random, col_labels: list[int]) -> list[int]:
    n = len(col_labels)
    min_moved = max(2, n // 2)
    for _ in range(60):
        row_labels = col_labels[:]
        rng.shuffle(row_labels)
        if sum(a != b for a, b in zip(col_labels, row_labels)) >= min_moved:
            return row_labels
    shift = rng.randint(1, n - 1)
    return col_labels[shift:] + col_labels[:shift]


def _content_palette(rng: random.Random, label_colors: list[int]) -> list[int]:
    extra = [color for color in LABEL_COLOR_POOL if color not in label_colors]
    rng.shuffle(extra)
    palette = [CONTENT_ANCHOR_COLOR] + extra
    if len(palette) < 3:
        fallback = label_colors[:]
        rng.shuffle(fallback)
        palette.extend(fallback)
    return palette


def _rotated_mask(rng: random.Random, base: np.ndarray) -> np.ndarray:
    return np.rot90(base, rng.randrange(4))


def _try_place_mask(
    rng: random.Random,
    pattern: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> list[tuple[int, int]] | None:
    n = pattern.shape[0]
    h, w = mask.shape
    coords_local = [tuple(map(int, pt)) for pt in np.argwhere(mask == 1)]

    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(0, n - h)
        left = rng.randint(0, n - w)
        coords = [(top + r, left + c) for r, c in coords_local]
        if any(blocked[r, c] for r, c in coords):
            continue

        for r, c in coords:
            pattern[r, c] = color

        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n:
                        blocked[rr, cc] = True
        return coords

    return None


def _generate_pattern(rng: random.Random, n: int, label_colors: list[int]) -> np.ndarray | None:
    pattern = np.full((n, n), BG, dtype=int)
    blocked = np.zeros((n, n), dtype=bool)
    palette = _content_palette(rng, label_colors)

    target_shapes = rng.randint(2, min(4, n + 1))
    placed: list[list[tuple[int, int]]] = []
    used_colors: list[int] = []
    rich_shape = False

    attempts = 0
    while len(placed) < target_shapes and attempts < target_shapes * 20:
        attempts += 1
        base = rng.choice(SHAPE_TEMPLATES)
        mask = _rotated_mask(rng, base)
        if mask.shape[0] > n or mask.shape[1] > n:
            continue

        if len(used_colors) < len(palette):
            color = palette[len(used_colors)]
        else:
            color = rng.choice(palette[: min(len(palette), target_shapes)])

        coords = _try_place_mask(rng, pattern, blocked, mask, color)
        if coords is None:
            continue

        placed.append(coords)
        used_colors.append(color)

        rows = {r for r, _ in coords}
        cols = {c for _, c in coords}
        if len(coords) >= 3 and len(rows) >= 2 and len(cols) >= 2:
            rich_shape = True

    non_black = int(np.count_nonzero(pattern))
    if len(placed) < 2 or non_black < max(4, n) or not rich_shape:
        return None
    return pattern


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(4, 8)
    col_labels = rng.sample(list(LABEL_COLOR_POOL), n)
    row_labels = _sample_row_labels(rng, col_labels)

    pattern = _generate_pattern(rng, n, col_labels)
    if pattern is None:
        return None

    transformed = _permute_interior(pattern, col_labels, row_labels)
    changed = int(np.count_nonzero(transformed != pattern))
    if changed < max(4, n // 2):
        return None

    grid = np.full((n + 1, n + 1), BG, dtype=int)
    grid[0, 1:] = np.array(col_labels, dtype=int)
    grid[1:, 0] = np.array(row_labels, dtype=int)
    grid[1:, 1:] = pattern
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
