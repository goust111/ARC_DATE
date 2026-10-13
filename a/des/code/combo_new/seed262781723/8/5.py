import random
from dataclasses import dataclass

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class SquareSprite:
    size: int
    center: tuple[int, int]
    main_color: int
    marker_color: int


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(coords)

    return components


def _paint_square(
    grid: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: int,
    marker_color: int | None = None,
) -> None:
    half = size // 2
    r, c = center
    r0 = r - half
    c0 = c - half
    r1 = r + half + 1
    c1 = c + half + 1
    assert 0 <= r0 < r1 <= grid.shape[0]
    assert 0 <= c0 < c1 <= grid.shape[1]
    grid[r0:r1, c0:c1] = color
    if marker_color is not None:
        grid[r, c] = marker_color


def _parse_sprite(grid: np.ndarray, coords: list[tuple[int, int]]) -> SquareSprite:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    assert height == width and height >= 3 and height % 2 == 1, "each object must be an odd square"
    center = (r0 + height // 2, c0 + width // 2)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]

    body_mask = np.ones((height, width), dtype=bool)
    body_mask[height // 2, width // 2] = False
    body = sub[body_mask]
    assert np.all(body != BG), "square body must be solid and non-background"
    body_colors = np.unique(body)
    assert body_colors.size == 1, "all non-center body cells must share one main color"
    main_color = int(body_colors[0])

    marker_color = int(sub[height // 2, width // 2])
    expected_mask = np.ones((height, width), dtype=bool)
    if marker_color == BG:
        expected_mask[height // 2, width // 2] = False
        assert np.count_nonzero(sub == BG) == 1, "root square may only use background at its center"
    else:
        assert marker_color != main_color, "marker color must differ from the square color"
        assert np.count_nonzero(sub == marker_color) == 1, "marker color must appear once"

    actual_mask = sub != BG
    assert np.array_equal(actual_mask, expected_mask), "square must fill its bbox except possibly the center"

    return SquareSprite(
        size=height,
        center=(int(center[0]), int(center[1])),
        main_color=main_color,
        marker_color=marker_color,
    )


def _extract_sprites(grid: np.ndarray) -> list[SquareSprite]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    components = _find_components(g != BG)
    assert components, "expected at least one square sprite"
    return [_parse_sprite(g, comp) for comp in components]


def _validate_hierarchy(sprites: list[SquareSprite]) -> SquareSprite:
    by_color = {sprite.main_color: sprite for sprite in sprites}
    assert len(by_color) == len(sprites), "sprite colors must be unique"

    roots = [sprite for sprite in sprites if sprite.marker_color == BG]
    assert len(roots) == 1, "expected exactly one outermost square"
    root = roots[0]
    assert root.size == max(sprite.size for sprite in sprites), "root must be the largest square"

    for sprite in sprites:
        if sprite is root:
            continue
        assert sprite.marker_color in by_color, "each marker must point to another square color"
        parent = by_color[sprite.marker_color]
        assert parent.size > sprite.size, "parent square must be larger than its child"

    for sprite in sprites:
        seen: set[int] = set()
        current = sprite
        while current.marker_color != BG:
            assert current.marker_color not in seen, "marker chain must be acyclic"
            seen.add(current.marker_color)
            current = by_color[current.marker_color]
        assert current is root, "all marker chains must terminate at the largest square"

    return root


def _render_nested(shape: tuple[int, int], sprites: list[SquareSprite], center: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for sprite in sorted(sprites, key=lambda item: item.size, reverse=True):
        _paint_square(out, center, sprite.size, sprite.main_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse scattered marked squares into one concentric stack centered on the marked outer square.

    Input structure:
    - The background is black.
    - Several disconnected odd-sized solid square sprites of distinct colors are scattered on the grid.
    - Each square has exactly one center marker pixel whose color matches another, larger square's body color.
    - The largest square uses a black center marker, indicating that it is the outermost layer.

    Rule:
    - Detect every square sprite and read its size, body color, and center marker color.
    - Treat each non-black marker as a pointer to the larger square with that body color, and treat the black-marked largest square as the root layer.
    - Clear the scattered arrangement and redraw all squares as solid filled squares, ordered from largest to smallest and centered on the root square's original center.
    - Later, smaller squares overwrite earlier, larger ones, producing concentric colored rings with no marker dots left in the output.

    Key insight:
    - The center marker is a visible parent pointer: once those pointers reveal the nesting hierarchy, all squares share the same final center and can be stacked by size.
    """
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g)
    root = _validate_hierarchy(sprites)
    return _render_nested(g.shape, sprites, root.center)


def _rect_is_clear(
    occupied: np.ndarray,
    top: int,
    left: int,
    size: int,
    padding: int = 1,
) -> bool:
    height, width = occupied.shape
    bottom = top + size
    right = left + size
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(height, bottom + padding)
    c1 = min(width, right + padding)
    return not bool(occupied[r0:r1, c0:c1].any())


def _sample_sizes(count: int) -> list[int]:
    root_size = random.choice([size for size in range(2 * count + 1, 12, 2)])
    smaller_pool = list(range(3, root_size, 2))
    smaller_sizes = random.sample(smaller_pool, count - 1)
    return [root_size] + sorted(smaller_sizes, reverse=True)


def _place_sprites(sizes: list[int], shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    height, width = shape
    occupied = np.zeros(shape, dtype=bool)
    placements: list[tuple[int, int]] = []

    for size in sizes:
        placed = False
        for _ in range(2000):
            top = random.randint(0, height - size)
            left = random.randint(0, width - size)
            if _rect_is_clear(occupied, top, left, size, padding=1):
                occupied[top : top + size, left : left + size] = True
                placements.append((top + size // 2, left + size // 2))
                placed = True
                break
        if not placed:
            return None

    return placements


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 5)
    sizes = _sample_sizes(count)
    min_side = max(sizes[0] + 4, 11 + 2 * count)
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    centers = _place_sprites(sizes, (height, width))
    if centers is None:
        return None

    colors = random.sample(list(range(1, 10)), count)
    sprites = [
        SquareSprite(
            size=size,
            center=center,
            main_color=colors[index],
            marker_color=BG if index == 0 else colors[index - 1],
        )
        for index, (size, center) in enumerate(zip(sizes, centers))
    ]

    grid = np.full((height, width), BG, dtype=int)
    for sprite in sprites:
        _paint_square(grid, sprite.center, sprite.size, sprite.main_color, sprite.marker_color)

    expected = _render_nested(grid.shape, sprites, sprites[0].center)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-square input")
