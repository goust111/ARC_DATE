import numpy as np


BG = 0
SEP = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MAX_ATTEMPTS = 400


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _quadrants(grid: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, grid.shape)
    assert h % 2 == 1 and w % 2 == 1, "expected odd grid dimensions"
    mid_r, mid_c = h // 2, w // 2
    assert np.all(grid[mid_r, :] == SEP), "expected grey separator row"
    assert np.all(grid[:, mid_c] == SEP), "expected grey separator column"
    quads = [
        grid[:mid_r, :mid_c],
        grid[:mid_r, mid_c + 1 :],
        grid[mid_r + 1 :, :mid_c],
        grid[mid_r + 1 :, mid_c + 1 :],
    ]
    assert len({q.shape for q in quads}) == 1, "quadrants must have the same shape"
    return [q.copy() for q in quads]


def _unique_quadrant(quadrants: list[np.ndarray]) -> np.ndarray:
    counts: dict[bytes, int] = {}
    for q in quadrants:
        key = q.tobytes()
        counts[key] = counts.get(key, 0) + 1
    unique = [q for q in quadrants if counts[q.tobytes()] == 1]
    assert len(unique) == 1, "expected exactly one outlier quadrant"
    return unique[0].copy()


def _compose_ring(sprite: np.ndarray) -> np.ndarray:
    h, w = map(int, sprite.shape)
    out = np.full((h + 2 * w, h + 2 * w), BG, dtype=int)
    out[w : w + h, :w] = sprite
    out[w : w + h, w + h :] = np.fliplr(sprite)
    out[:w, w : w + h] = np.rot90(sprite, -1)
    out[w + h :, w : w + h] = np.rot90(sprite, 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the unique separated quadrant and build a four-way ring from it.

    Input structure:
    - The grid has black background with odd height and width.
    - The middle row and middle column are entirely grey, forming a separator cross.
    - The cross divides the canvas into four equal rectangular quadrants; three are identical and one is the outlier.

    Rule:
    - Extract the four full quadrant arrays and compare them cell by cell.
    - Select the quadrant pattern that appears exactly once and keep it as the sprite, including its internal black cells.
    - Create a black square of side `h + 2w` for sprite size `h x w`, then place the sprite on the left, its horizontal mirror on the right, its clockwise rotation on top, and its counterclockwise rotation on the bottom, leaving the center empty.

    Key insight:
    - The odd quadrant is only an intermediate clue; the final output is the fixed ring composition built from that uniquely occurring quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    unique = _unique_quadrant(_quadrants(g))
    return _compose_ring(unique)


def _variant_diff(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape != b.shape or not np.array_equal(a, b)


def _is_asymmetric(mask: np.ndarray) -> bool:
    if not mask.any() or mask.all():
        return False
    if not _variant_diff(mask, np.fliplr(mask)):
        return False
    if not _variant_diff(mask, np.flipud(mask)):
        return False
    for k in (1, 2, 3):
        if not _variant_diff(mask, np.rot90(mask, k)):
            return False
    return True


def _random_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    min_fg = max(2, int(np.ceil(area * 0.25)))
    max_fg = min(area - 2, int(np.floor(area * 0.65)))
    if min_fg > max_fg:
        min_fg, max_fg = 2, area - 2
    fg_count = int(np.random.randint(min_fg, max_fg + 1))
    mask = np.zeros((height, width), dtype=bool)
    cells = np.random.choice(area, size=fg_count, replace=False)
    mask.flat[cells] = True
    return mask


def _generate_asymmetric_mask(height: int, width: int) -> np.ndarray:
    for _ in range(120):
        mask = _random_mask(height, width)
        if _is_asymmetric(mask):
            return mask
    raise RuntimeError("failed to generate an asymmetric mask")


def _shape_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _make_unique_mask(majority_mask: np.ndarray) -> np.ndarray:
    h, w = map(int, majority_mask.shape)
    area = h * w
    min_diff = max(2, area // 6)
    max_diff = max(min_diff, min(area - 2, area // 2))
    for _ in range(160):
        candidate = _generate_asymmetric_mask(h, w)
        dist = _shape_distance(candidate, majority_mask)
        if min_diff <= dist <= max_diff:
            return candidate
    raise RuntimeError("failed to generate a distinct unique mask")


def _paint(mask: np.ndarray, color: int) -> np.ndarray:
    quad = np.zeros(mask.shape, dtype=int)
    quad[mask] = int(color)
    return quad


def _choose_sizes() -> tuple[int, int]:
    for _ in range(100):
        if np.random.rand() < 0.35:
            size = int(np.random.randint(3, 9))
            h, w = size, size
        else:
            h = int(np.random.randint(3, 9))
            w = int(np.random.randint(3, 12))
        if h + 2 * w <= 30:
            return h, w
    raise RuntimeError("failed to choose valid sprite dimensions")


def _gen_attempt() -> np.ndarray | None:
    h, w = _choose_sizes()
    majority_mask = _generate_asymmetric_mask(h, w)
    branch = np.random.choice(["color", "shape", "both"], p=[0.4, 0.4, 0.2])

    majority_color = int(np.random.choice(PALETTE))
    if branch == "color":
        unique_mask = majority_mask.copy()
        unique_color = int(np.random.choice(PALETTE[PALETTE != majority_color]))
    elif branch == "shape":
        unique_mask = _make_unique_mask(majority_mask)
        unique_color = majority_color
    else:
        unique_mask = _make_unique_mask(majority_mask)
        unique_color = int(np.random.choice(PALETTE[PALETTE != majority_color]))

    majority_quad = _paint(majority_mask, majority_color)
    unique_quad = _paint(unique_mask, unique_color)
    if np.array_equal(majority_quad, unique_quad):
        return None

    slots = [majority_quad.copy() for _ in range(4)]
    unique_index = int(np.random.randint(4))
    slots[unique_index] = unique_quad

    grid = np.full((2 * h + 1, 2 * w + 1), BG, dtype=int)
    grid[h, :] = SEP
    grid[:, w] = SEP
    grid[:h, :w] = slots[0]
    grid[:h, w + 1 :] = slots[1]
    grid[h + 1 :, :w] = slots[2]
    grid[h + 1 :, w + 1 :] = slots[3]

    quads = _quadrants(grid)
    key_counts: dict[bytes, int] = {}
    for q in quads:
        key = q.tobytes()
        key_counts[key] = key_counts.get(key, 0) + 1
    if sorted(key_counts.values()) != [1, 3]:
        return None

    detected = _unique_quadrant(quads)
    if not np.array_equal(detected, unique_quad):
        return None
    if not _is_asymmetric(detected != BG):
        return None

    out = main(grid)
    expected = _compose_ring(unique_quad)
    if not np.array_equal(out, expected):
        return None
    if out.shape != (h + 2 * w, h + 2 * w):
        return None

    center = out[w : w + h, w : w + h]
    if np.any(center != BG):
        return None
    if not np.array_equal(out[w : w + h, :w], unique_quad):
        return None
    if not np.array_equal(out[w : w + h, w + h :], np.fliplr(unique_quad)):
        return None
    if not np.array_equal(out[:w, w : w + h], np.rot90(unique_quad, -1)):
        return None
    if not np.array_equal(out[w + h :, w : w + h], np.rot90(unique_quad, 1)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
